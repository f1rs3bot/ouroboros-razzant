"""Bounded pacing for the packet review substrate's one transient retry."""

from types import SimpleNamespace
from unittest.mock import Mock

from ouroboros.review_substrate import ReviewRequest, ReviewSlot, run_review_request


def _request(task_id: str, *, surface: str = "multi_model_review") -> ReviewRequest:
    return ReviewRequest(
        surface=surface, goal="review", task_id=task_id, call_type=surface,
    )


def _slot() -> ReviewSlot:
    return ReviewSlot(slot_id="slot_a", model="same/model")


def test_retryable_packet_failure_waits_before_same_route_resend(tmp_path, monkeypatch):
    waits = []

    def _wait(seconds, wake_check):
        waits.append(seconds)
        assert wake_check() is False
        return False

    monkeypatch.setattr("ouroboros.loop_transport.interruptible_wait_sleep", _wait)
    llm = Mock()
    llm.chat.side_effect = [
        TimeoutError("transient timeout"),
        ({"content": '{"verdict":"PASS","findings":[]}'}, {}),
    ]

    result = run_review_request(
        _request("paced-retry"), slots=[_slot()], drive_root=tmp_path, llm=llm,
    )

    assert result.aggregate_signal == "PASS"
    assert llm.chat.call_count == 2
    assert waits == [4.0]
    assert llm.chat.call_args_list[0].kwargs == llm.chat.call_args_list[1].kwargs


def test_nonretryable_packet_failure_neither_waits_nor_resends(tmp_path, monkeypatch):
    waits = []
    monkeypatch.setattr(
        "ouroboros.loop_transport.interruptible_wait_sleep",
        lambda seconds, wake_check: waits.append(seconds) or False,
    )
    llm = Mock()
    llm.chat.side_effect = RuntimeError("AuthenticationError('401 invalid_api_key')")

    result = run_review_request(
        _request("terminal-failure"), slots=[_slot()], drive_root=tmp_path, llm=llm,
    )

    assert result.actors[0]["status"] == "error"
    assert llm.chat.call_count == 1
    assert waits == []


def test_empty_and_format_repair_resends_remain_immediate(tmp_path, monkeypatch):
    waits = []
    monkeypatch.setattr(
        "ouroboros.loop_transport.interruptible_wait_sleep",
        lambda seconds, wake_check: waits.append(seconds) or False,
    )

    empty_llm = Mock()
    empty_llm.chat.side_effect = [
        ({"content": ""}, {}),
        ({"content": '{"verdict":"PASS","findings":[]}'}, {}),
    ]
    empty = run_review_request(
        _request("empty-retry"), slots=[_slot()], drive_root=tmp_path, llm=empty_llm,
    )

    repair_llm = Mock()
    repair_llm.chat.side_effect = [
        ({"content": "malformed"}, {}),
        ({"content": "[]"}, {}),
    ]
    repaired = run_review_request(
        _request("format-repair", surface="task_acceptance"),
        slots=[_slot()], drive_root=tmp_path, llm=repair_llm,
    )

    assert empty.aggregate_signal == "PASS"
    assert repaired.actors[0]["status"] == "ok"
    assert empty_llm.chat.call_count == repair_llm.chat.call_count == 2
    assert waits == []


def test_cancellation_during_wait_prevents_second_physical_send(tmp_path, monkeypatch):
    cancelled = {"value": False}
    monkeypatch.setattr(
        "ouroboros.review_substrate.review_retry_cancelled",
        lambda _ctx: cancelled["value"],
    )

    def _wait(_seconds, wake_check):
        cancelled["value"] = True
        assert wake_check() is True
        return True

    monkeypatch.setattr("ouroboros.loop_transport.interruptible_wait_sleep", _wait)
    llm = Mock(side_effect=AssertionError("unused"))
    llm.chat.side_effect = TimeoutError("transient timeout")
    ctx = SimpleNamespace(task_id="cancelled-retry", pending_events=[])

    result = run_review_request(
        _request(ctx.task_id), slots=[_slot()], drive_root=tmp_path,
        llm=llm, usage_ctx=ctx,
    )

    assert llm.chat.call_count == 1
    assert result.actors[0]["status"] == "error"
    assert result.actors[0]["error"] == "transient timeout"
    assert result.actors[0]["usage"]["review_retry_stop_reason"] == "cancelled"


def test_logical_deadline_during_wait_prevents_second_physical_send(tmp_path, monkeypatch):
    clock = {"now": 0.0}
    monkeypatch.setattr("ouroboros.review_substrate.monotonic_now", lambda: clock["now"])

    def _wait(_seconds, wake_check):
        clock["now"] = 1e20
        assert wake_check() is True
        return True

    monkeypatch.setattr("ouroboros.loop_transport.interruptible_wait_sleep", _wait)
    llm = Mock()
    llm.chat.side_effect = TimeoutError("transient timeout")

    result = run_review_request(
        _request("deadline-retry"), slots=[_slot()], drive_root=tmp_path, llm=llm,
    )

    assert llm.chat.call_count == 1
    assert result.actors[0]["status"] == "error"
    assert result.actors[0]["error"] == "transient timeout"
    assert result.actors[0]["usage"]["review_retry_stop_reason"] == "deadline"


def test_logical_deadline_before_wait_prevents_sleep_and_second_send(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr("ouroboros.review_substrate.monotonic_now", lambda: 10.0)
    monkeypatch.setattr(
        "ouroboros.review_custody.monotonic_now", lambda _slot_id="": 10.0,
    )
    monkeypatch.setattr(
        "ouroboros.loop_transport.interruptible_wait_sleep",
        Mock(side_effect=AssertionError("deadline must stop before retry sleep")),
    )
    llm = Mock()
    llm.chat.side_effect = TimeoutError("transient timeout")

    result = run_review_request(
        _request("pre-wait-deadline"),
        slots=[ReviewSlot(slot_id="slot_a", model="same/model", timeout_sec=1.0)],
        drive_root=tmp_path,
        llm=llm,
    )

    assert llm.chat.call_count == 1
    assert result.actors[0]["status"] == "error"
    assert result.actors[0]["error"] == "transient timeout"
    assert result.actors[0]["usage"]["review_retry_stop_reason"] == "deadline"
