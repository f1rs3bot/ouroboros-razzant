"""Route-limit fallback policy: the old-versus-new decision matrix.

WHY THIS FILE EXISTS. A candidate once shipped a policy whose default silently
REMOVED a rotation that already worked: an empty content-policy refusal used to
classify as ``llm_empty_response``, which the gate did not exclude, so the round
walked the cross-model chain — and the candidate's new default stopped it. Every
test in that candidate passed, because every test checked the NEW behaviour and
none checked the OLD one. This file is the missing half: the pre-change decision
is frozen as EXPECTED_DEFAULT_DECISION and the policy is asserted to be a no-op
on a default install.

HOW THE TABLE WAS DERIVED. By reading the pre-change source, not the change:
``loop_transport.fallback_chain_allowed`` denied the chain for exactly
``{"context_overflow", "provider_outcome_unknown", "deadline_exhausted"}`` once
the exact-route, unresolved-transport-death and live-episode rules had passed —
so every OTHER kind rotated. That is what a reviewer must check this table
against (HEAD's ``loop_transport.py``), not against the implementation here.

COMPLETENESS. The error-kind vocabulary has no single producer, so the table is
additionally ratcheted by a source scan: every kind literal a producer can emit
through a classification call, a ``"kind":`` body-error field or an
``*event_type`` assignment must be a key here, so a NEW kind fails this test
instead of silently escaping the matrix. Disclosed residual: the scan matches
those producer shapes only; a kind assembled another way (a bare ``return
"x", True`` pair from the provider-unavailable rail's SOURCE vocabulary, a value
built at runtime) is outside its reach — that vocabulary is deliberately not
this one, and review covers what the scan cannot.
"""

from __future__ import annotations

import json
import pathlib
import re

import pytest

from ouroboros import loop_transport
from ouroboros.model_slots import (
    ROUTE_LIMIT_FALLBACK_ALLOW,
    ROUTE_LIMIT_FALLBACK_DENY,
    ROUTE_LIMIT_FALLBACK_KEY,
    ROUTE_LIMIT_ERROR_KINDS,
    get_limit_fallback_policy,
    limit_blocks_fallback,
)

# kind -> whether the cross-model chain was permitted BEFORE this change.
EXPECTED_DEFAULT_DECISION = {
    # Route limits — permitted before, and still permitted by the shipped default.
    "provider_transient": True,          # 408 / 429 / 5xx / overloaded
    "rate_limit": True,                  # HTTP-200 body error carrying a 429
    "subscription_window_exhausted": True,  # typed window refusal with reset_at
    # Not limits — their decisions must not move under either policy value.
    "provider_policy_refusal": True,     # content-policy refusal
    "llm_empty_response": True,
    "provider_incomplete_response": True,
    "provider_body_error": True,
    "provider_error": True,
    "auth_error": True,
    "bad_request": True,
    "quota_exhausted": True,
    "request_too_large": True,
    "remote_context_overflow": True,
    "local_context_overflow": True,
    "transport_unavailable": True,
    "model_outcome_unknown": True,
    # The three kinds the pre-change gate already excluded.
    "context_overflow": False,
    "provider_outcome_unknown": False,
    "deadline_exhausted": False,
}

_PRODUCER_SOURCES = ("ouroboros/loop_llm_call.py", "ouroboros/llm_openai_compatible.py")


class _Ctx:
    """The gate's only context dependency: the frozen-configured-actor flag."""

    exact_model_route = False


def _gate(kind: str, **kwargs) -> bool:
    return loop_transport.fallback_chain_allowed(_Ctx(), kind, None, {}, **kwargs)


@pytest.fixture
def shipped_default(monkeypatch):
    """The shipped posture: the key is unset, so the floor applies."""
    monkeypatch.delenv(ROUTE_LIMIT_FALLBACK_KEY, raising=False)
    return ROUTE_LIMIT_FALLBACK_ALLOW


@pytest.mark.parametrize("kind", sorted(EXPECTED_DEFAULT_DECISION))
def test_shipped_default_reproduces_the_pre_change_decision(kind, shipped_default):
    """Claim 1: on a default install the gate returns HEAD's value for every kind."""
    assert shipped_default == ROUTE_LIMIT_FALLBACK_ALLOW
    assert _gate(kind) is EXPECTED_DEFAULT_DECISION[kind]


@pytest.mark.parametrize("kind", sorted(EXPECTED_DEFAULT_DECISION))
def test_deny_flips_exactly_the_limit_kinds(kind, monkeypatch):
    """Claim 2: `deny` is scoped — one root cause per class, nothing else moves."""
    monkeypatch.setenv(ROUTE_LIMIT_FALLBACK_KEY, ROUTE_LIMIT_FALLBACK_DENY)
    expected = EXPECTED_DEFAULT_DECISION[kind] and kind not in ROUTE_LIMIT_ERROR_KINDS
    assert _gate(kind) is expected
    # The flip set is exactly the closed limit set, no matter how many kinds exist.
    assert (expected is not EXPECTED_DEFAULT_DECISION[kind]) == (kind in ROUTE_LIMIT_ERROR_KINDS)


def test_policy_can_only_deny_never_widen(monkeypatch):
    """One-directional: `deny` cannot resurrect a kind the gate already excluded."""
    monkeypatch.setenv(ROUTE_LIMIT_FALLBACK_KEY, ROUTE_LIMIT_FALLBACK_DENY)
    already_denied = [k for k, v in EXPECTED_DEFAULT_DECISION.items() if v is False]
    assert already_denied, "the matrix must retain the pre-existing exclusions"
    for kind in already_denied:
        assert _gate(kind) is False


@pytest.mark.parametrize("raw", ["", " ", "yes", "no", "off", "on", "WAIT", "rotate", "allow ", "DENY_FALLBACK"])
def test_unknown_policy_values_resolve_to_the_shipped_default(raw, monkeypatch):
    """A malformed value must never select a spend posture the owner did not choose."""
    monkeypatch.setenv(ROUTE_LIMIT_FALLBACK_KEY, raw)
    assert get_limit_fallback_policy() == ROUTE_LIMIT_FALLBACK_ALLOW
    assert limit_blocks_fallback("provider_transient") is False


@pytest.mark.parametrize("raw", ["deny", "DENY", " Deny "])
def test_deny_is_accepted_case_and_space_insensitively(raw, monkeypatch):
    monkeypatch.setenv(ROUTE_LIMIT_FALLBACK_KEY, raw)
    assert get_limit_fallback_policy() == ROUTE_LIMIT_FALLBACK_DENY


def test_limit_set_is_closed_and_matches_the_policy(monkeypatch):
    """The blocking predicate is the set membership AND the policy — nothing looser."""
    assert ROUTE_LIMIT_ERROR_KINDS == {
        "provider_transient", "rate_limit", "subscription_window_exhausted",
    }
    monkeypatch.setenv(ROUTE_LIMIT_FALLBACK_KEY, ROUTE_LIMIT_FALLBACK_DENY)
    for kind in EXPECTED_DEFAULT_DECISION:
        assert limit_blocks_fallback(kind) is (kind in ROUTE_LIMIT_ERROR_KINDS)
    # A kind nobody enumerated is never blocked: the policy is closed-world.
    assert limit_blocks_fallback("__new_unlisted_kind__") is False


def test_policy_denial_is_disclosed_but_silent_by_default(tmp_path, monkeypatch):
    """Claim 3: suppressing a spend decision is never silent — and never invents a row."""
    evidence = loop_transport.limit_denial_evidence(
        tmp_path, task_id="t1", model="m1", error_kind="provider_transient", round_idx=7,
    )
    assert loop_transport.fallback_chain_allowed(_Ctx(), "provider_transient", None, {}, evidence) is True
    assert not (tmp_path / "events.jsonl").exists(), "the default must emit nothing"

    monkeypatch.setenv(ROUTE_LIMIT_FALLBACK_KEY, ROUTE_LIMIT_FALLBACK_DENY)
    assert loop_transport.fallback_chain_allowed(_Ctx(), "provider_transient", None, {}, evidence) is False
    rows = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["type"] for row in rows] == ["route_limit_policy_denied"]
    assert rows[0]["error_kind"] == "provider_transient"
    assert rows[0]["model"] == "m1"
    assert rows[0]["round"] == 7
    assert rows[0]["policy"] == ROUTE_LIMIT_FALLBACK_DENY
    assert rows[0]["detail"], "the row must say what did NOT happen"


def test_a_pre_existing_rule_denial_is_not_reported_as_a_policy_denial(tmp_path, monkeypatch):
    """The deciding rule reports itself: an earlier rule's denial claims no policy cause."""
    monkeypatch.setenv(ROUTE_LIMIT_FALLBACK_KEY, ROUTE_LIMIT_FALLBACK_DENY)
    evidence = loop_transport.limit_denial_evidence(
        tmp_path, task_id="t1", model="m1", error_kind="provider_transient",
    )

    exact = _Ctx()
    exact.exact_model_route = True
    assert loop_transport.fallback_chain_allowed(exact, "provider_transient", None, {}, evidence) is False

    episode = loop_transport.TransportWaitEpisode(started_monotonic=0.0)
    assert loop_transport.fallback_chain_allowed(
        _Ctx(), "provider_transient", episode, {}, evidence) is False

    fenced = {loop_transport.TRANSPORT_DEATHS_KEY: {"count": 1}}
    assert loop_transport.fallback_chain_allowed(
        _Ctx(), "provider_transient", None, fenced, evidence) is False

    assert not (tmp_path / "events.jsonl").exists(), (
        "a denial owned by an earlier rule must not be attributed to the owner's policy"
    )


def test_kind_vocabulary_scan_is_complete():
    """The completeness ratchet: a new producer kind literal must enter the matrix."""
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    scanned: set[str] = set()
    for relative in _PRODUCER_SOURCES:
        text = (repo_root / relative).read_text(encoding="utf-8")
        # 1. LlmErrorClassification("kind", ...) and LlmErrorClassification(CONSTANT, ...)
        scanned.update(m.group(1) for m in re.finditer(r'LlmErrorClassification\(\s*"([a-z_]+)"', text))
        for match in re.finditer(r"LlmErrorClassification\(\s*([A-Z_][A-Z0-9_]*)", text):
            from ouroboros import loop_llm_call

            value = getattr(loop_llm_call, match.group(1), None)
            if isinstance(value, str):
                scanned.add(value)
        # 2. An HTTP-200 body error's typed kind.
        scanned.update(m.group(1) for m in re.finditer(r'"kind":\s*"([a-z_]+)"', text))
        # 3. Any *event_type assignment (local_context_overflow / remote_context_overflow).
        scanned.update(m.group(1) for m in re.finditer(r'\w*event_type\w*\s*=\s*"([a-z_]+)"', text))

    assert scanned, "the scan must actually read the producers"
    unlisted = sorted(scanned - set(EXPECTED_DEFAULT_DECISION))
    assert not unlisted, (
        f"producer kind literals missing from EXPECTED_DEFAULT_DECISION: {unlisted}. "
        "Add each to the matrix with its pre-change decision — an unlisted kind is a "
        "decision no test would notice changing."
    )
