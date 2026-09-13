"""Route-scoped serialized prompt-size bounds (codex-lb class fix).

The provider's typed rejection (param + `maximum length N`) teaches the
route_fp-scoped serialized bound once; every later task on the same route_fp
preempts the doomed oversized first call instead of re-paying classification.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from ouroboros.capability_evidence import (
    _PROMPT_SIZE_BOUNDS_TTL_SEC,
    get_prompt_size_violation,
    prompt_size_violation_details,
    record_prompt_size_violation,
)
from ouroboros.loop_model_call import (
    _prompt_size_gate,
    _serialized_system_bytes,
)


# ---------- capability_evidence namespace ----------

def test_record_and_lookup_violation(tmp_path):
    record_prompt_size_violation(tmp_path, "fp-a", "instructions", 1048576)
    assert get_prompt_size_violation(tmp_path, "fp-a") == {
        "param": "instructions", "max_bytes": 1048576,
    }


def test_lookup_missing_checkpoint_returns_none(tmp_path):
    assert get_prompt_size_violation(tmp_path, "fp-a") is None
    assert get_prompt_size_violation(tmp_path, "") is None


def test_lower_bound_wins_within_ttl(tmp_path):
    record_prompt_size_violation(tmp_path, "fp-a", "instructions", 1048576)
    record_prompt_size_violation(tmp_path, "fp-a", "instructions", 2097152)
    assert get_prompt_size_violation(tmp_path, "fp-a")["max_bytes"] == 1048576


def test_min_merge_respects_ttl(tmp_path, monkeypatch):
    record_prompt_size_violation(tmp_path, "fp-a", "instructions", 2097152)
    record_prompt_size_violation(tmp_path, "fp-a", "instructions", 1048576)
    assert get_prompt_size_violation(tmp_path, "fp-a")["max_bytes"] == 1048576


def test_expired_entry_is_not_returned(tmp_path, monkeypatch):
    from ouroboros import capability_evidence as ce
    record_prompt_size_violation(tmp_path, "fp-a", "instructions", 1048576)
    monkeypatch.setattr(ce, "_age_seconds", lambda ts: float(_PROMPT_SIZE_BOUNDS_TTL_SEC))
    assert get_prompt_size_violation(tmp_path, "fp-a") is None


def test_invalid_inputs_are_no_ops(tmp_path):
    record_prompt_size_violation(tmp_path, "", "instructions", 100)
    record_prompt_size_violation(tmp_path, "fp-a", "instructions", 0)
    record_prompt_size_violation(tmp_path, "fp-a", "instructions", "not-a-number")
    store = tmp_path / "state" / "capability_evidence.json"
    if store.exists():
        data = json.loads(store.read_text(encoding="utf-8"))
    else:
        data = {"prompt_size_bounds": {}}
    assert data.get("prompt_size_bounds", {} ) == {}


# ---------- rejection-body extraction ----------

def test_codex_lb_rejection_shape_parses():
    body = {
        "error": {
            "message": (
                "Invalid 'instructions': string too long. Expected a string "
                "with maximum length 1048576, but got a string with length "
                "1098283 instead."
            ),
            "type": "invalid_request_error",
            "code": "string_above_max_length",
            "param": "instructions",
        }
    }
    assert prompt_size_violation_details(body) == {
        "param": "instructions", "max_bytes": 1048576,
    }


def test_body_without_numeric_maximum_yields_none():
    body = {"error": {"message": "unparseable", "code": "something", "param": "instructions"}}
    assert prompt_size_violation_details(body) is None
    assert prompt_size_violation_details({}) is None


# ---------- measurement helper ----------

def test_system_serialized_bytes_matches_canonical_projection():
    from ouroboros.llm_attempt import _canonical_candidate_bytes
    messages = [
        {"role": "system", "content": "tier-0 core"},
        {"role": "system", "content": "second"},
        {"role": "user", "content": "ignored"},
    ]
    expected = len(_canonical_candidate_bytes({"system": ["tier-0 core", "second"]}))
    assert _serialized_system_bytes(messages) == expected


def test_no_system_message_is_unbounded():
    assert _serialized_system_bytes([{"role": "user", "content": "x"}]) == 0
    assert _serialized_system_bytes(None) == 0


# ---------- pre-dispatch gate ----------

class _StubPlan:
    def __init__(self, route_fp: str, low_parts):
        self.route_fp = route_fp
        self.low_parts = low_parts
        self.reprojected = False

    def reproject_transcript(self, messages, mode):
        if mode != "low":
            return messages
        self.reprojected = True
        return [
            {"role": "system", "content": self.low_parts},
            *[m for m in messages if m.get("role") != "system"],
        ]


def _ctx(plan, system_parts, mode="max"):
    return SimpleNamespace(
        context_fit_plan=plan,
        active_context_mode=mode,
        messages=[
            {"role": "system", "content": system_parts},
            {"role": "user", "content": "go"},
        ],
    )


@pytest.fixture(autouse=True)
def _canonical_root(tmp_path, monkeypatch):
    """Redirect capability_evidence's canonical observation root to tmp."""
    from ouroboros import capability_evidence as ce
    monkeypatch.setattr(ce, "canonical_evidence_root", lambda: tmp_path)

    def _stub_reproject(ctx):
        plan = ctx.context_fit_plan
        ctx.messages[:] = (
            plan.reproject_transcript(ctx.messages, "low")
            if plan is not None
            else ctx.messages
        )
        ctx.active_context_mode = "low"

    monkeypatch.setattr(
        "ouroboros.loop_model_call._reproject_actual_overflow_low",
        _stub_reproject,
    )
    return tmp_path


def test_no_recorded_bound_leaves_dispatch_free(tmp_path, _canonical_root):
    plan = _StubPlan("fp-x", "small")
    ctx = _ctx(plan, "huge tier0 section")
    assert _prompt_size_gate(ctx) is None


def test_reproject_to_low_when_under_bound_after(tmp_path, _canonical_root):
    from ouroboros.llm_attempt import _canonical_candidate_bytes
    huge = "x" * 1000
    bound = len(_canonical_candidate_bytes({"system": ["x" * 100]}))
    record_prompt_size_violation(tmp_path, plan_fp := "fp-x", "instructions", bound)
    plan = _StubPlan(plan_fp, "x")
    ctx = _ctx(plan, huge)
    assert _prompt_size_gate(ctx) is None
    assert plan.reprojected


def test_still_over_bound_after_reprojection_is_typed_skip(tmp_path, _canonical_root):
    from ouroboros.llm_attempt import _canonical_candidate_bytes
    bound = len(_canonical_candidate_bytes({"system": ["small"]}))
    huge = "x" * 1000
    record_prompt_size_violation(tmp_path, "fp-x", "instructions", bound)
    plan = _StubPlan("fp-x", huge)
    ctx = _ctx(plan, huge)
    assert _prompt_size_gate(ctx) == "prompt_size_bound_exceeded"


def test_non_recorded_other_route_unaffected(tmp_path, _canonical_root):
    record_prompt_size_violation(tmp_path, "fp-x", "instructions", 1)
    plan = _StubPlan("fp-other", [])
    ctx = _ctx(plan, "large content")
    assert _prompt_size_gate(ctx) is None


def test_already_low_mode_over_bound_is_typed_skip(tmp_path, _canonical_root):
    from ouroboros.llm_attempt import _canonical_candidate_bytes
    bound = len(_canonical_candidate_bytes({"system": ["x"]}))
    record_prompt_size_violation(tmp_path, "fp-x", "instructions", bound)
    plan = _StubPlan("fp-x", "xx")
    ctx = _ctx(plan, "xx", mode="low")
    assert _prompt_size_gate(ctx) == "prompt_size_bound_exceeded"


def test_serialized_size_equal_to_bound_passes(tmp_path, _canonical_root):
    """The provider's wording accepts length == maximum; the gate mirrors it."""
    from ouroboros.llm_attempt import _canonical_candidate_bytes
    bound = len(_canonical_candidate_bytes({"system": ["xx"]}))
    record_prompt_size_violation(tmp_path, "fp-x", "instructions", bound)
    plan = _StubPlan("fp-x", "xx")
    ctx = _ctx(plan, "xx")
    assert _prompt_size_gate(ctx) is None


def test_already_low_mode_under_bound_passes(tmp_path, _canonical_root):
    from ouroboros.llm_attempt import _canonical_candidate_bytes
    bound = len(_canonical_candidate_bytes({"system": ["x" * 100]}))
    record_prompt_size_violation(tmp_path, "fp-x", "instructions", bound)
    plan = _StubPlan("fp-x", "small")
    ctx = _ctx(plan, "small", mode="low")
    assert _prompt_size_gate(ctx) is None


def test_unprojectable_param_stays_unbounded(tmp_path, _canonical_root):
    from ouroboros.llm_attempt import _canonical_candidate_bytes
    bound = len(_canonical_candidate_bytes({"system": ["huge"]}))
    record_prompt_size_violation(tmp_path, "fp-x", "max_tokens", bound)
    plan = _StubPlan("fp-x", "huge")
    ctx = _ctx(plan, "huge")
    assert _prompt_size_gate(ctx) is None
