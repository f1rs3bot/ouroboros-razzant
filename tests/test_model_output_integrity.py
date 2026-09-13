from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace


RAW = "  <|close|>濟\n" + ("broken-token-stream " * 500)


def test_classifier_is_narrow_and_preserves_legitimate_text():
    from ouroboros.task_finalization import leaked_model_control_marker

    assert leaked_model_control_marker(RAW) == "<|close|>"
    for text in (
        "",
        "Обычный Unicode-ответ 濟",
        "The literal <|close|> can appear in tokenizer diagnostics.",
        "```text\n<|close|>\n```",
        "x" * 100_000,
        "<|endoftext|> is not an observed incident marker",
    ):
        assert leaked_model_control_marker(text) == ""


def test_candidate_quarantine_preserves_exact_hash_addressed_bytes(tmp_path):
    from ouroboros.task_finalization import (
        MODEL_OUTPUT_INTEGRITY_NOTICE,
        quarantine_model_output_candidate,
    )

    public, projection = quarantine_model_output_candidate(tmp_path, "task-1", RAW)

    assert public == MODEL_OUTPUT_INTEGRITY_NOTICE
    assert projection["terminal_origin"] == "host_notice"
    assert projection["execution_status"] == "failed"
    assert projection["reason_code"] == "model_output_integrity"
    receipt = projection["model_output_integrity"]
    data = RAW.encode("utf-8")
    assert receipt == {
        "marker": "<|close|>",
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
        "size_chars": len(RAW),
        "path": receipt["path"],
        "preserved": True,
    }
    path = Path(receipt["path"])
    assert path.read_bytes() == data
    assert receipt["sha256"] in path.name


def test_hash_addressed_salvage_is_immutable_and_legacy_path_is_unchanged(tmp_path):
    from ouroboros.observability import preserve_salvaged_output

    legacy = Path(preserve_salvaged_output(tmp_path, "task-1", "legacy"))
    first = Path(preserve_salvaged_output(
        tmp_path, "task-1", "first", identity="model-output-first",
    ))
    second = Path(preserve_salvaged_output(
        tmp_path, "task-1", "second", identity="model-output-second",
    ))

    assert legacy.name == "task-1.txt"
    assert legacy.read_text(encoding="utf-8") == "legacy"
    assert first != second
    assert first.read_text(encoding="utf-8") == "first"
    assert second.read_text(encoding="utf-8") == "second"


def test_handle_text_preserves_host_authorship():
    from ouroboros.loop import _handle_text_response

    for origin in ("host_notice", "host_salvage"):
        usage = {"terminal_origin": origin}
        text, returned, _trace = _handle_text_response(
            "Host-authored terminal text", {"reasoning_notes": []}, usage,
        )
        assert text == "Host-authored terminal text"
        assert returned["terminal_origin"] == origin


def test_ordinary_final_is_rejected_before_acceptance_spend(tmp_path, monkeypatch):
    import ouroboros.loop as loop
    from tests.test_delivery_forced_finalization import _forced_test_context

    _loop, registry, ctx, trace = _forced_test_context(tmp_path)
    registry._ctx.task_contract = {}
    canonical_root = tmp_path / "canonical"
    registry._ctx.task_metadata["budget_drive_root"] = str(canonical_root)
    registry._ctx.budget_drive_root = str(canonical_root)
    monkeypatch.setattr(
        loop, "_begin_task_acceptance_fence",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("fence called")),
    )

    result = loop._no_tool_final_answer(
        RAW, ctx, trace, registry, ctx.incoming_messages, ctx.owner_msg_seen,
        lambda _message: None,
    )

    assert result is not None
    text, usage, returned_trace = result
    assert "invalid generation" in text and RAW not in text
    assert usage["reason_code"] == "model_output_integrity"
    candidate = registry._ctx._delivery_candidate
    assert candidate.full_text == text
    assert candidate.model_text == RAW
    assert candidate.terminal_projection["reason_code"] == "model_output_integrity"
    assert Path(usage["model_output_integrity"]["path"]).is_relative_to(canonical_root)
    assert returned_trace["review_decision"] == {
        "eligibility": "not_eligible", "trigger": "model_output_integrity",
    }


def test_rejected_candidate_hold_then_clean_candidate_has_no_stale_terminal_state(tmp_path, monkeypatch):
    import ouroboros.loop as loop
    from tests.test_delivery_forced_finalization import _forced_test_context

    _loop, registry, ctx, trace = _forced_test_context(tmp_path)
    registry._ctx.task_contract = {}
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    rejected = loop._replace_model_delivery_candidate(
        registry, ctx, trace, RAW, control="candidate",
    )
    loop._hold_delivery_for_skill_action(registry, trace)

    result = loop._no_tool_final_answer(
        "Clean replacement after the action.", ctx, trace, registry,
        ctx.incoming_messages, ctx.owner_msg_seen, lambda _message: None,
    )

    assert result is not None
    text, usage, returned_trace = result
    clean = registry._ctx._delivery_candidate
    assert rejected.terminal_projection["reason_code"] == "model_output_integrity"
    assert clean is not rejected and clean.terminal_projection == {}
    assert text == "Clean replacement after the action."
    assert usage["terminal_origin"] == "model_final"
    assert "reason_code" not in usage and "model_output_integrity" not in usage
    assert returned_trace["review_decision"]["trigger"] != "model_output_integrity"

    # Even public-text equality cannot reuse rejected state: candidate identity
    # includes the model-authored bytes and projection, not only the rendered notice.
    notice = rejected.full_text
    same_public = loop._replace_model_delivery_candidate(
        registry, ctx, trace, notice, control="candidate",
    )
    assert same_public is not rejected
    assert same_public.model_text == notice and same_public.terminal_projection == {}


def test_rejected_candidate_supersedes_prior_acceptance_without_spending_again(tmp_path, monkeypatch):
    import ouroboros.loop as loop
    from tests.test_delivery_forced_finalization import _bind_host_pass, _forced_test_context

    _loop, registry, ctx, trace = _forced_test_context(tmp_path)
    registry._ctx.task_contract = {}
    accepted = loop._replace_delivery_candidate(
        registry, ctx, trace, "Previously accepted answer.", control="candidate",
    )
    prior = _bind_host_pass(loop, registry, trace, accepted)
    monkeypatch.setattr(
        loop, "_begin_task_acceptance_fence",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("fence called")),
    )

    result = loop._no_tool_final_answer(
        RAW, ctx, trace, registry, ctx.incoming_messages,
        ctx.owner_msg_seen, lambda _message: None,
    )

    rejected = registry._ctx._delivery_candidate
    assert prior["superseded_by_revision"] is True
    assert registry._ctx._task_acceptance_reviewed is False
    assert rejected.acceptance_binding["authoritative"] is False
    assert result is not None
    assert result[1]["reason_code"] == "model_output_integrity"
    assert result[2]["review_decision"] == {
        "eligibility": "not_eligible", "trigger": "model_output_integrity",
    }


def test_replace_control_cannot_reintroduce_leaked_model_output(tmp_path, monkeypatch):
    import ouroboros.loop as loop
    from tests.test_delivery_forced_finalization import _forced_test_context

    _loop, registry, ctx, trace = _forced_test_context(tmp_path)
    registry._ctx.task_contract = {}
    old = loop._replace_delivery_candidate(
        registry, ctx, trace, "Retained complete answer.", control="awaiting_control",
    )
    registry._ctx._delivery_control_required = True
    monkeypatch.setattr(
        loop, "_begin_task_acceptance_fence",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("fence called")),
    )
    control = json.dumps({"delivery_control": "replace", "full_answer": RAW})

    result = loop._no_tool_final_answer(
        control, ctx, trace, registry, ctx.incoming_messages, ctx.owner_msg_seen,
        lambda _message: None,
    )

    assert result is not None
    text, usage, returned_trace = result
    assert "invalid generation" in text and RAW not in text
    assert usage["reason_code"] == "model_output_integrity"
    assert returned_trace["review_decision"]["trigger"] == "model_output_integrity"
    assert registry._ctx._delivery_candidate is not old


def test_forced_model_final_quarantines_before_return(tmp_path, monkeypatch):
    import ouroboros.loop as loop
    from tests.test_delivery_forced_finalization import _forced_test_context

    _loop, registry, ctx, trace = _forced_test_context(tmp_path)
    monkeypatch.setattr(loop, "call_llm_with_retry", lambda *_a, **_k: ({"content": RAW}, 0.0))

    text, usage, returned_trace = loop._forced_final_answer(
        ctx, prompt="finalize", fallback_text="fallback", reason_code="round_limit",
    )

    assert "invalid generation" in text and RAW not in text
    assert usage["terminal_origin"] == "host_notice"
    assert usage["reason_code"] == "model_output_integrity"
    assert returned_trace["forced_finalization"]["candidate_sha256"] == hashlib.sha256(text.encode()).hexdigest()


def test_pipeline_keeps_raw_out_of_result_live_replay_and_presence(tmp_path, monkeypatch):
    import ouroboros.agent_task_pipeline as pipeline
    from ouroboros.task_results import load_task_result
    from supervisor.terminal_delivery import build_completed_result_event

    monkeypatch.setattr(pipeline, "_run_post_task_processing_async", lambda *_a, **_k: None)
    env = SimpleNamespace(drive_root=tmp_path, repo_dir=tmp_path)
    usage = {"terminal_origin": "model_final", "rounds": 1, "cost": 0.0}
    trace = {"tool_calls": [], "reasoning_notes": []}

    task = {"id": "corrupt", "type": "task", "chat_id": 7, "text": "do it"}
    pending = []
    pipeline.emit_task_results(
        env, None, None, pending, task, RAW, usage, trace,
        start_time=0.0, drive_logs=tmp_path / "logs",
    )
    sent = next(row for row in pending if row["type"] == "send_message")
    stored = load_task_result(tmp_path, "corrupt")
    replay = build_completed_result_event(tmp_path, task, "corrupt", stored)
    assert RAW not in sent["text"] and RAW not in stored["result"] and RAW not in replay["text"]
    assert sent["role"] == "system" and sent["terminal_origin"] == "host_notice"
    assert stored["status"] == "failed"
    assert Path(stored["model_output_integrity"]["path"]).read_text(encoding="utf-8") == RAW

    presence = {
        "id": "presence-corrupt", "type": "task", "chat_id": 7, "text": "do it",
        "_presence_turn": True,
    }
    presence_pending = []
    presence_ctx = SimpleNamespace(
        _presence_completion={"outcome": "message", "message": RAW},
        _swarm_handoff_attempt={}, _skip_post_task_synthesis=True,
    )
    pipeline.emit_task_results(
        env, None, None, presence_pending, presence, RAW,
        {"terminal_origin": "model_final", "rounds": 1, "cost": 0.0},
        {"tool_calls": [], "reasoning_notes": []},
        start_time=0.0, drive_logs=tmp_path / "logs", ctx=presence_ctx,
    )
    presence_event = next(row for row in presence_pending if row["type"] == "presence_result")
    assert RAW not in presence_event["text"] and "invalid generation" in presence_event["text"]


def test_pipeline_leaves_normal_model_final_byte_exact(tmp_path, monkeypatch):
    import ouroboros.agent_task_pipeline as pipeline
    from ouroboros.task_results import load_task_result

    monkeypatch.setattr(pipeline, "_run_post_task_processing_async", lambda *_a, **_k: None)
    env = SimpleNamespace(drive_root=tmp_path, repo_dir=tmp_path)
    task = {"id": "normal", "type": "task", "chat_id": 7, "text": "do it"}
    text = "Technical prose quotes <|close|> safely. 濟"
    pending = []
    pipeline.emit_task_results(
        env, None, None, pending, task, text,
        {"terminal_origin": "model_final", "rounds": 1, "cost": 0.0},
        {"tool_calls": [], "reasoning_notes": []},
        start_time=0.0, drive_logs=tmp_path / "logs",
    )
    sent = next(row for row in pending if row["type"] == "send_message")
    stored = load_task_result(tmp_path, "normal")
    assert sent["text"] == text and stored["result"] == text
    assert stored["terminal_origin"] == "model_final"
    assert "model_output_integrity" not in stored
