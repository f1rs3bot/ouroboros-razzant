from __future__ import annotations

import hashlib
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


def test_quarantine_preserves_exact_hash_addressed_bytes(tmp_path):
    from ouroboros.task_finalization import (
        MODEL_OUTPUT_INTEGRITY_NOTICE,
        quarantine_model_output,
        terminal_result_fields,
    )

    usage = {"terminal_origin": "model_final"}
    public = quarantine_model_output(tmp_path, "task-1", RAW, usage)

    assert public == MODEL_OUTPUT_INTEGRITY_NOTICE
    assert usage["terminal_origin"] == "host_notice"
    assert usage["execution_status"] == "failed"
    assert usage["reason_code"] == "model_output_integrity"
    receipt = usage["model_output_integrity"]
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
    assert terminal_result_fields(usage)["model_output_integrity"] == receipt


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


def test_ordinary_final_is_rejected_before_acceptance_spend(tmp_path, monkeypatch):
    import ouroboros.loop as loop
    from tests.test_delivery_forced_finalization import _forced_test_context

    _loop, registry, ctx, trace = _forced_test_context(tmp_path)
    registry._ctx.task_contract = {}
    canonical_root = tmp_path / "canonical"
    child_root = tmp_path / "child"
    registry._ctx.task_metadata["budget_drive_root"] = str(canonical_root)
    registry._ctx.budget_drive_root = str(canonical_root)
    ctx.drive_root = child_root

    result = loop._no_tool_final_answer(
        RAW, ctx, trace, registry, ctx.incoming_messages, ctx.owner_msg_seen,
        lambda _message: None,
    )

    assert result is not None
    text, usage, returned_trace = result
    assert "invalid generation" in text
    assert RAW not in text
    assert usage["reason_code"] == "model_output_integrity"
    assert registry._ctx._model_output_integrity_rejected is False
    candidate = registry._ctx._delivery_candidate
    assert candidate.full_text == text
    assert candidate.acceptance_binding["authoritative"] is False
    receipt_path = Path(usage["model_output_integrity"]["path"])
    assert receipt_path.is_relative_to(canonical_root)
    assert not receipt_path.is_relative_to(child_root)
    assert returned_trace["review_decision"] == {
        "eligibility": "not_eligible", "trigger": "model_output_integrity",
    }
    assert any(
        "leading leaked model control marker was withheld" in note
        for note in returned_trace["reasoning_notes"]
    )
    assert any("invalid generation" in note for note in returned_trace["reasoning_notes"])


def test_acceptance_refuses_rejected_model_output_without_fence_or_panel(tmp_path, monkeypatch):
    import ouroboros.loop as loop
    from tests.test_delivery_forced_finalization import _forced_test_context

    _loop, registry, ctx, trace = _forced_test_context(tmp_path)
    registry._ctx._model_output_integrity_rejected = True
    monkeypatch.setattr(loop, "_begin_task_acceptance_fence", lambda *_a: (_ for _ in ()).throw(AssertionError("fence called")))

    assert loop._run_task_acceptance_review_once(
        tools=registry, content="host notice", task_id=ctx.task_id,
        task_type=ctx.task_type, llm_trace=trace, drive_root=ctx.drive_root,
        messages=ctx.messages, emit_progress=lambda _message: None,
    ) is False
    assert trace["review_decision"] == {
        "eligibility": "not_eligible", "trigger": "model_output_integrity",
    }
    assert registry._ctx._model_output_integrity_rejected is False


def test_replace_control_cannot_reintroduce_leaked_model_output(tmp_path, monkeypatch):
    import json
    import ouroboros.loop as loop
    from tests.test_delivery_forced_finalization import _forced_test_context

    _loop, registry, ctx, trace = _forced_test_context(tmp_path)
    registry._ctx.task_contract = {}
    candidate = loop._replace_delivery_candidate(
        registry, ctx, trace, "Retained complete answer.", control="awaiting_control",
    )
    registry._ctx._delivery_control_required = True
    monkeypatch.setattr(
        loop, "_begin_task_acceptance_fence",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("fence called")),
    )
    control = json.dumps({
        "delivery_control": "replace",
        "full_answer": RAW,
    })

    result = loop._no_tool_final_answer(
        control, ctx, trace, registry, ctx.incoming_messages, ctx.owner_msg_seen,
        lambda _message: None,
    )

    assert result is not None
    text, usage, returned_trace = result
    assert "invalid generation" in text
    assert RAW not in text
    assert usage["reason_code"] == "model_output_integrity"
    assert returned_trace["review_decision"] == {
        "eligibility": "not_eligible", "trigger": "model_output_integrity",
    }
    assert registry._ctx._delivery_control_required is False
    assert registry._ctx._delivery_candidate.full_text == text
    assert candidate.full_text == "Retained complete answer."


def test_pipeline_defense_keeps_raw_out_of_result_live_delivery_and_replay(tmp_path, monkeypatch):
    import ouroboros.agent_task_pipeline as pipeline
    from ouroboros.task_results import load_task_result
    from supervisor.terminal_delivery import build_completed_result_event

    monkeypatch.setattr(pipeline, "_run_post_task_processing_async", lambda *_a, **_k: None)
    env = SimpleNamespace(drive_root=tmp_path, repo_dir=tmp_path)
    task = {"id": "forced-corrupt", "type": "task", "chat_id": 7, "text": "do it"}
    usage = {"terminal_origin": "model_final", "rounds": 1, "cost": 0.0}
    trace = {"tool_calls": [], "reasoning_notes": []}
    pending = []

    pipeline.emit_task_results(
        env, None, None, pending, task, RAW, usage, trace,
        start_time=0.0, drive_logs=tmp_path / "logs",
    )

    sent = next(row for row in pending if row["type"] == "send_message")
    stored = load_task_result(tmp_path, "forced-corrupt")
    assert RAW not in sent["text"]
    assert sent["role"] == "system"
    assert sent["terminal_origin"] == "host_notice"
    assert stored["result"] == sent["text"]
    assert RAW not in stored["result"]
    assert stored["status"] == "failed"
    receipt = stored["model_output_integrity"]
    assert Path(receipt["path"]).read_text(encoding="utf-8") == RAW
    replay = build_completed_result_event(tmp_path, task, "forced-corrupt", stored)
    assert replay["text"] == sent["text"]
    assert RAW not in replay["text"]


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
    assert sent["text"] == text
    assert stored["result"] == text
    assert stored["terminal_origin"] == "model_final"
    assert "model_output_integrity" not in stored
