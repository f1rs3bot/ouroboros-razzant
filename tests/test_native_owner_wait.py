"""Ordinary chat keeps the existing owner-wait capability without a pool slot."""

import queue
from types import SimpleNamespace

import pytest

from ouroboros.owner_mailbox import drain_owner_entries, write_owner_message
from ouroboros.model_wait import TaskModelWait
from ouroboros.owner_wait import direct_owner_wait, wait_after_tools
from ouroboros.task_results import load_task_result
from tests.test_owner_wait import context


def native_context(tmp_path):
    ctx = context(tmp_path)
    ctx.owner_wait_callback = direct_owner_wait
    ctx.event_queue = queue.Queue()
    ctx.model_wait_context = TaskModelWait(
        task={"id": ctx.task_id}, drive_root=tmp_path,
        event_queue=ctx.event_queue, worker_slot_held=False,
    )
    ctx.model_wait_context.tool_context = ctx
    return ctx


def test_native_wait_retains_source_and_leaves_answer_delivery_to_loop(tmp_path, monkeypatch):
    ctx = native_context(tmp_path)
    question = {"type": "send_quiz", "task_id": ctx.task_id, "quiz_id": "q1"}
    ctx.pending_events = [question]
    messages = [{"role": "tool", "tool_call_id": "saved", "content": "Saved form"}]
    waits = []

    def answer(_seconds):
        waits.append(load_task_result(tmp_path, ctx.task_id)["owner_wait"])
        assert ctx.event_queue.get_nowait() == question
        assert write_owner_message(tmp_path, "Continue with that form", ctx.task_id, msg_id="answer")

    monkeypatch.setattr("ouroboros.owner_wait.time.sleep", answer)
    wait_after_tools(ctx, messages, {}, {}, 4, [], set())
    assert len(waits) == 1 and waits[0]["state"] == "waiting"
    after = load_task_result(tmp_path, ctx.task_id)["owner_wait"]
    assert after == {**waits[0], "state": "resumed"}
    assert after["source_ref"] and "restart_transaction_id" not in after
    assert ctx.pending_events == [] and ctx._owner_wait_requested == ""
    assert ctx._loop_mailbox_seen_ids == set()
    assert messages == [{"role": "tool", "tool_call_id": "saved", "content": "Saved form"}]
    assert drain_owner_entries(tmp_path, ctx.task_id, set())[0]["text"] == "Continue with that form"


@pytest.mark.parametrize("reason", ["cancelled", "finalize_requested", "deadline", "absolute_ceiling"])
def test_native_wait_rejoins_existing_control_without_waiting_for_answer(tmp_path, monkeypatch, reason):
    ctx = native_context(tmp_path)
    monkeypatch.setattr(ctx.model_wait_context, "control_reason", lambda: reason)
    monkeypatch.setattr("ouroboros.owner_wait.time.sleep", lambda _: pytest.fail("control did not release wait"))
    wait_after_tools(ctx, [], {}, {}, 1, [], set())
    assert load_task_result(tmp_path, ctx.task_id)["owner_wait"]["state"] == "resumed"
    assert not drain_owner_entries(tmp_path, ctx.task_id, set())  # no fabricated owner answer


def test_native_agent_factory_binds_existing_wait_callback(tmp_path, monkeypatch):
    from supervisor import workers

    actor = SimpleNamespace()
    monkeypatch.setattr("ouroboros.agent.make_agent", lambda **_: actor)
    monkeypatch.setattr(workers, "REPO_DIR", tmp_path)
    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(workers, "get_event_q", queue.Queue)
    assert workers._get_chat_agent() is actor
    assert actor.owner_wait_callback is direct_owner_wait


def test_direct_root_can_request_wait_through_the_existing_quiz(tmp_path):
    from ouroboros.tools.core_artifacts import _escalate
    from ouroboros.tools.registry import ToolContext

    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="native",
                      is_direct_chat=True, current_chat_id=1, event_queue=queue.Queue())
    ctx.owner_wait_callback = direct_owner_wait
    result = _escalate(ctx, question="Continue?", options=[{"label": "Yes"}, {"label": "No"}],
                       wait_for_answer=True)
    assert result.startswith("OK:"), result
    event = ctx.event_queue.get_nowait()
    assert event["type"] == "send_quiz" and event["wait_for_answer"] is True
    assert ctx._owner_wait_requested == event["quiz_id"]
    assert load_task_result(tmp_path, "native")["owner_quiz"][event["quiz_id"]]["wait_for_answer"] is True


@pytest.mark.parametrize("required", [False, True])
def test_answer_frame_describes_only_work_that_continued(required):
    from ouroboros.gateway.task_decision import _quiz_answer_frame

    block = {"quiz_id": "q", "question": "Proceed?", "options": ["Yes", "No"],
             "assumption": "Keep the current layout", "wait_for_answer": required}
    frame = _quiz_answer_frame(block, 0, "Proceed with the saved form")
    assert "Proceed with the saved form" in frame
    assert ("You continued under the assumption" in frame) is not required


@pytest.mark.parametrize("cold", [False, True])
def test_expired_wait_clock_precedes_post_tool_budget(tmp_path, monkeypatch, cold):
    """A saved tail may have spent its budget, but an expired clock keeps its cause."""
    import json
    import time
    from ouroboros import loop, model_wait, owner_wait, task_pacing
    from tests.test_owner_wait_cold_loop import cold_registry
    from tests.test_loop_transport_wait import _loop_kwargs

    def no_network(*_args, **_kwargs):
        pytest.fail("synthetic wait must never contact a provider")
    monkeypatch.setattr("ouroboros.llm.LLMClient.chat", no_network)
    monkeypatch.setattr("ouroboros.llm.LLMClient.chat_async", no_network)
    monkeypatch.setattr("ouroboros.pricing._fetch_live_rows", no_network)
    monkeypatch.setenv("OUROBOROS_TASK_ABS_CEILING_SEC", "21600")
    registry = cold_registry(tmp_path, monkeypatch, task_pacing.CostCeiling(state="active", ceiling_usd=9.0))
    ctx = registry._ctx
    ctx.is_direct_chat, ctx.current_chat_id, ctx.event_queue = True, 1, queue.Queue()
    calls = []

    with model_wait.task_model_wait_scope(task={"id": ctx.task_id, "_is_direct_chat": True},
            drive_root=tmp_path, event_queue=ctx.event_queue, worker_slot_held=False) as controller:
        ctx.model_wait_context, controller.tool_context = controller, ctx

        def expire_and_wait(context, checkpoint):
            controller.started_monotonic = time.monotonic() - 21601
            assert controller.control_reason() == "absolute_ceiling"
            owner_wait.direct_owner_wait(context, checkpoint)
        ctx.owner_wait_callback = expire_and_wait
        if not cold:
            ctx.owner_wait_resume, ctx._owner_wait_requested = None, ""

        def dispatch(call, _disposition, **_kwargs):
            calls.append("ordinary_send")
            call.accumulated_usage["cost"] = 10.0
            return {"role": "assistant", "content": "", "tool_calls": [{
                "id": "ask", "type": "function", "function": {
                    "name": "escalate", "arguments": json.dumps({"question": "Continue?",
                        "options": [{"label": "Yes"}, {"label": "No"}], "wait_for_answer": True})}}]}, 0.0
        monkeypatch.setattr(loop, "_dispatch_round_model", dispatch)
        monkeypatch.setattr(loop, "_call_forced_model_once", lambda *_a, **_k: pytest.fail("expired clock entered a paid finalizer"))
        monkeypatch.setattr(loop, "_finish_tool_round_budget", lambda *_a, **_k: pytest.fail("expired clock entered budget tail"))
        _text, usage, trace = loop.run_llm_loop(**{
            **_loop_kwargs(tmp_path, registry, []), "event_queue": ctx.event_queue})

    assert calls == ([] if cold else ["ordinary_send"])
    assert usage["reason_code"] != "budget_exhausted"
    assert trace["forced_finalization"]["control_reason"] == "absolute_ceiling"
