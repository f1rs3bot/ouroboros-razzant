"""Safety-lane fallback rotation (owner incident: one Light-route 429 stalled the
whole Safety Supervisor for ~1.5h; a 502 whose body carried ``cyber_policy`` killed
security-work tasks instead of rotating).

The walk: candidates are ``[light_model] + OUROBOROS_MODEL_FALLBACKS``; a genuine
throughput 429 keeps its ONE bounded retry per candidate, a structured
content-policy refusal rotates WITHOUT a same-model retry (it is permanent for
that request), quota/auth/permanent classes never rotate, and terminal shapes are
non-verdict — ``SAFETY_UNAVAILABLE``, never a false ``SAFETY_VIOLATION``.

Shares the stub-client idiom of tests/test_safety_rate_limit.py (repeated here
rather than imported across test modules).
"""

from __future__ import annotations

import json
import pathlib

import pytest

FALLBACK_MODEL = "openai/gpt-5.6-luna"


@pytest.fixture(autouse=True)
def _ensure_remote_key(monkeypatch):
    """A fake remote key keeps the LLM path active so ``_resolve_safety_routing``
    doesn't take the misconfigured-fail-open branch."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key-for-routing")
    monkeypatch.delenv("USE_LOCAL_LIGHT", raising=False)
    # The inherited process env may carry a non-full mode (the worker applies the
    # owner settings); these tests assert FULL-mode blocking/checking semantics.
    monkeypatch.setenv("OUROBOROS_SAFETY_MODE", "full")
    yield


@pytest.fixture(autouse=True)
def _cold_storm_latch(monkeypatch):
    import ouroboros.safety as safety

    monkeypatch.setattr(safety, "_SAFETY_STORM_UNTIL", 0.0)


@pytest.fixture
def _no_backoff(monkeypatch):
    import ouroboros.safety as safety

    monkeypatch.setattr(safety, "_safety_rate_limit_backoff", lambda ctx: None)


@pytest.fixture
def _fallback_chain(monkeypatch):
    """One fallback candidate: the owner's configured chain is the walk."""
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", FALLBACK_MODEL)


def _patch_llm_client(monkeypatch, stub) -> None:
    import ouroboros.safety as safety
    from ouroboros.llm import LLMClient

    factory = lambda: stub
    factory.supports_response_format = LLMClient.supports_response_format
    monkeypatch.setattr(safety, "LLMClient", factory)


class _PerModelLLMClient:
    """Script per MODEL: each entry is an Exception to raise or a ``(content, usage)``
    tuple to return. The last entry per model repeats. ``calls`` records the model of
    every physical attempt, in order."""

    def __init__(self, scripts: dict[str, list]):
        self.scripts = {model: list(steps) for model, steps in scripts.items()}
        self.calls: list[str] = []

    def chat(self, *, messages, model, use_local, **kwargs):
        self.calls.append(model)
        steps = self.scripts[model]
        step = steps[min(len([c for c in self.calls if c == model]) - 1, len(steps) - 1)]
        if isinstance(step, Exception):
            raise step
        content, usage = step
        return {"content": content}, usage


class _RateLimitError(Exception):
    status_code = 429


class _CyberPolicyError(Exception):
    """The production shape: codex-lb wraps the upstream OpenAI cyber filter as a
    502 whose body carries the structured refusal."""

    status_code = 502
    body = {
        "error": {
            "message": "This content was flagged for possible cybersecurity risk.",
            "type": "invalid_request",
            "code": "cyber_policy",
        }
    }


class _QuotaError(Exception):
    """Structured insufficient-quota that ALSO carries HTTP 429: PERMANENT must win
    over the status code and keep blocking, never rotate."""

    status_code = 429
    code = "insufficient_quota"


_SAFE = ('{"status":"SAFE","reason":"Scoped file read"}', {"prompt_tokens": 12, "completion_tokens": 3})


class _DriveCtx:
    def __init__(self, root):
        self.task_id = "t-safety"
        self.drive_root = str(root)
        self._logs = pathlib.Path(root) / "logs"
        self._logs.mkdir(parents=True, exist_ok=True)

    def drive_logs(self):
        return self._logs


def _read_events(ctx, event_type: str | None = None) -> list[dict]:
    path = ctx.drive_logs() / "events.jsonl"
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [r for r in rows if event_type is None or r.get("type") == event_type]


def _light_model() -> str:
    from ouroboros.config import get_light_model

    return get_light_model()


def test_rate_limit_rotates_to_fallback_and_serves(monkeypatch, tmp_path, _no_backoff, _fallback_chain):
    """First candidate 429s twice; the walk rotates to the owner's fallback, which
    answers SAFE. The guarded call is CHECKED, not blocked; the rotation is
    disclosed in a durable event."""
    from ouroboros.safety import check_safety

    light = _light_model()
    stub = _PerModelLLMClient({
        light: [_RateLimitError("Rate limit exceeded")],
        FALLBACK_MODEL: [_SAFE],
    })
    _patch_llm_client(monkeypatch, stub)
    ctx = _DriveCtx(tmp_path)

    ok, msg = check_safety("create_github_issue", {"title": "x"}, ctx=ctx)

    assert ok is True and msg == "", "a rotated-and-served check is a clean SAFE"
    assert stub.calls == [light, light, FALLBACK_MODEL], (
        "two attempts on the throttled route, one on the fallback, no more"
    )
    rows = _read_events(ctx, "safety_fallback_rotation")
    assert len(rows) == 1
    assert rows[0]["from_model"] == light and rows[0]["to_model"] == FALLBACK_MODEL
    assert rows[0]["rotation_class"] == "rate_limit"


def test_all_candidates_rate_limited_keeps_unavailable_terminal(monkeypatch, tmp_path, _no_backoff, _fallback_chain):
    """Every candidate throttles: the existing typed SAFETY_UNAVAILABLE non-verdict
    outcome, exactly as before the walk — never a false verdict."""
    from ouroboros.safety import check_safety

    light = _light_model()
    stub = _PerModelLLMClient({
        light: [_RateLimitError("Rate limit exceeded")],
        FALLBACK_MODEL: [_RateLimitError("Rate limit exceeded")],
    })
    _patch_llm_client(monkeypatch, stub)
    ctx = _DriveCtx(tmp_path)

    ok, msg = check_safety("create_github_issue", {"title": "x"}, ctx=ctx)

    assert ok is False
    assert msg.startswith("⚠️ SAFETY_UNAVAILABLE:"), "typed non-verdict, not a verdict"
    assert "SAFETY_VIOLATION" not in msg
    assert stub.calls == [light, light, FALLBACK_MODEL, FALLBACK_MODEL]
    assert len(_read_events(ctx, "safety_check_rate_limited")) == 1


def test_rate_limit_rotation_honors_owner_deny_policy(monkeypatch, tmp_path, _no_backoff, _fallback_chain):
    """OUROBOROS_ROUTE_LIMIT_FALLBACK=deny: a LIMIT is never answered by a metered
    substitution — the lane keeps its pre-walk single-route terminal, the denial
    is disclosed, and no fallback call is paid."""
    from ouroboros.safety import check_safety

    monkeypatch.setenv("OUROBOROS_ROUTE_LIMIT_FALLBACK", "deny")
    light = _light_model()
    stub = _PerModelLLMClient({
        light: [_RateLimitError("Rate limit exceeded")],
        FALLBACK_MODEL: [_SAFE],
    })
    _patch_llm_client(monkeypatch, stub)
    ctx = _DriveCtx(tmp_path)

    ok, msg = check_safety("create_github_issue", {"title": "x"}, ctx=ctx)

    assert ok is False and msg.startswith("⚠️ SAFETY_UNAVAILABLE:")
    assert stub.calls == [light, light], "the deny gate keeps the pre-walk single-route shape"
    assert _read_events(ctx, "safety_fallback_rotation") == []
    rows = _read_events(ctx, "route_limit_policy_denied")
    assert len(rows) == 1 and rows[0]["model"] == light


def test_deny_policy_does_not_cover_policy_refusal(monkeypatch, tmp_path, _no_backoff, _fallback_chain):
    """The deny set is LIMIT kinds only: a content-policy refusal still rotates —
    the owner denied metered substitution for limits, not for refusals."""
    from ouroboros.safety import check_safety

    monkeypatch.setenv("OUROBOROS_ROUTE_LIMIT_FALLBACK", "deny")
    light = _light_model()
    stub = _PerModelLLMClient({
        light: [_CyberPolicyError("flagged")],
        FALLBACK_MODEL: [_SAFE],
    })
    _patch_llm_client(monkeypatch, stub)
    ctx = _DriveCtx(tmp_path)

    ok, msg = check_safety("run_command", {"cmd": ["nuclei", "-u", "https://target"]}, ctx=ctx)

    assert ok is True and msg == ""
    assert stub.calls == [light, FALLBACK_MODEL]


def test_quota_refusal_never_rotates(monkeypatch, tmp_path, _no_backoff, _fallback_chain):
    """A structured insufficient-quota 429 is PERMANENT: it re-raises on the first
    candidate with the pre-walk semantics — no fallback call is paid."""
    from ouroboros.safety import check_safety

    light = _light_model()
    stub = _PerModelLLMClient({
        light: [_QuotaError("quota")],
        FALLBACK_MODEL: [_SAFE],
    })
    _patch_llm_client(monkeypatch, stub)
    ctx = _DriveCtx(tmp_path)

    ok, msg = check_safety("create_github_issue", {"title": "x"}, ctx=ctx)

    assert ok is False and "SAFETY_VIOLATION" in msg, "pre-walk permanent-error shape preserved"
    assert stub.calls == [light], "no same-model retry, no rotation"
    assert _read_events(ctx, "safety_fallback_rotation") == []


def test_cyber_policy_refusal_rotates_without_same_model_retry(monkeypatch, tmp_path, _no_backoff, _fallback_chain):
    """The owner's 502: a body-carried `cyber_policy` is permanent for THIS request —
    exactly one attempt on the refusing route, then the fallback serves the check."""
    from ouroboros.safety import check_safety

    light = _light_model()
    stub = _PerModelLLMClient({
        light: [_CyberPolicyError("flagged")],
        FALLBACK_MODEL: [_SAFE],
    })
    _patch_llm_client(monkeypatch, stub)
    ctx = _DriveCtx(tmp_path)

    ok, msg = check_safety("run_command", {"cmd": ["nuclei", "-u", "https://target"]}, ctx=ctx)

    assert ok is True and msg == ""
    assert stub.calls == [light, FALLBACK_MODEL], "no paid same-model retry of a permanent refusal"
    rows = _read_events(ctx, "safety_fallback_rotation")
    assert len(rows) == 1 and rows[0]["rotation_class"] == "policy_refusal"


def test_body_carried_policy_code_rotates_too(monkeypatch, tmp_path, _no_backoff, _fallback_chain):
    """The HTTP-200 sibling: an error body stamped on ``usage["provider_error"]``
    (the production shape that never raises) carrying `code: cyber_policy`
    rotates exactly like the raised shape — never a false unparseable verdict."""
    from ouroboros.safety import check_safety

    light = _light_model()
    body_refusal = ("", {
        "provider_error": {
            "code": "cyber_policy",
            "type": "invalid_request",
            "message": "This content was flagged for possible cybersecurity risk.",
            "kind": "provider_error",
        },
        "prompt_tokens": 12,
        "completion_tokens": 0,
    })
    stub = _PerModelLLMClient({
        light: [body_refusal],
        FALLBACK_MODEL: [_SAFE],
    })
    _patch_llm_client(monkeypatch, stub)
    ctx = _DriveCtx(tmp_path)

    ok, msg = check_safety("run_command", {"cmd": ["nuclei", "-u", "https://target"]}, ctx=ctx)

    assert ok is True and msg == ""
    assert stub.calls == [light, FALLBACK_MODEL]
    rows = _read_events(ctx, "safety_fallback_rotation")
    assert len(rows) == 1 and rows[0]["rotation_class"] == "policy_refusal"


def test_all_candidates_policy_refused_is_unavailable_not_violation(monkeypatch, tmp_path, _no_backoff, _fallback_chain):
    """When even the fallback refuses the safety prompt: an honest
    SAFETY_UNAVAILABLE infrastructure outcome with a retry contract and a durable
    audit row — never a false accusation against the guarded call."""
    from ouroboros.safety import check_safety

    light = _light_model()
    stub = _PerModelLLMClient({
        light: [_CyberPolicyError("flagged")],
        FALLBACK_MODEL: [_CyberPolicyError("flagged")],
    })
    _patch_llm_client(monkeypatch, stub)
    ctx = _DriveCtx(tmp_path)

    ok, msg = check_safety("run_command", {"cmd": ["nuclei", "-u", "https://target"]}, ctx=ctx)

    assert ok is False, "an unchecked guarded call must not execute in full mode"
    assert msg.startswith("⚠️ SAFETY_UNAVAILABLE:"), "infrastructure fact, not a verdict"
    assert "SAFETY_VIOLATION" not in msg and "NOT a verdict" in msg
    assert stub.calls == [light, FALLBACK_MODEL]
    rows = _read_events(ctx, "safety_check_policy_refused")
    assert len(rows) == 1 and rows[0]["action"] == "blocked_unchecked_policy_refused"


def test_classifier_reads_structured_policy_code_from_body():
    """Unit pin of the routing fact the walk relies on: a 502 whose body carries
    `cyber_policy` is a PERMANENT provider_policy_refusal, not a transient outage."""
    from ouroboros.loop_llm_call import PROVIDER_POLICY_REFUSAL, classify_llm_exception

    found = classify_llm_exception(_CyberPolicyError("flagged"), "")
    assert found.kind == PROVIDER_POLICY_REFUSAL
    assert found.retry_same_request is False


def test_classifier_reads_string_above_max_length_as_context_overflow():
    """codex-lb's `instructions` overflow is a context-window fact: the shrink
    ladder owns it, not a bad-request terminal."""
    from ouroboros.loop_llm_call import classify_llm_exception

    class _InstructionsTooLong(Exception):
        status_code = 400
        body = {
            "error": {
                "message": "Invalid 'instructions': string too long.",
                "type": "invalid_request_error",
                "code": "string_above_max_length",
                "param": "instructions",
            }
        }

    found = classify_llm_exception(_InstructionsTooLong("too long"), "")
    assert found.kind == "context_overflow"


def test_classifier_keeps_plain_invalid_request_as_bad_request():
    """The narrow policy set must not swallow a genuine bad request."""
    from ouroboros.loop_llm_call import classify_llm_exception

    class _PlainBadRequest(Exception):
        status_code = 400
        body = {"error": {"message": "unknown parameter", "type": "invalid_request", "code": "unknown_parameter"}}

    found = classify_llm_exception(_PlainBadRequest("bad"), "")
    assert found.kind == "bad_request"
