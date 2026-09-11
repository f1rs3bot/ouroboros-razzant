"""Ephemeral model waits use the real turn, producer, owner and decision seams."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
import queue
import threading
from types import SimpleNamespace

import pytest

from ouroboros import agent as agent_module, model_wait, owner_mailbox
from ouroboros.gateways.claudexor import ClaudexorUnavailable
from ouroboros.utils import append_jsonl
from supervisor import active_activity, workers
from supervisor.task_model_wait import handle_task_model_wait
from tests.test_llm_claudexor import MODEL, ledger, result, setup as gateway_fixture
from tests.test_model_wait import _action_for, _decision_clients, _refusal

setup = gateway_fixture


@pytest.fixture
def ephemeral_call(setup, monkeypatch):
    """Control context/model replies; run the real Agent, turn and terminal owners."""
    root, transport, client = setup
    registry = active_activity.DirectActivityRegistry()
    monkeypatch.setattr(active_activity, "_DIRECT_ACTIVITY_REGISTRY", registry)
    from supervisor import message_bus, queue as task_queue

    monkeypatch.setattr(workers, "DRIVE_ROOT", root)
    monkeypatch.setattr(task_queue, "DRIVE_ROOT", root)
    monkeypatch.setattr(task_queue, "RUNNING", {})
    monkeypatch.setattr(task_queue, "PENDING", [])
    busy = SimpleNamespace(_busy=True, _current_task_id="other-turn", _accepting_owner_messages=True,
                           _current_task_metadata={}, _current_task_text="Other work", _current_chat_id=1)
    registry.register("other-turn", 1, actor=busy)
    events, published, failures = queue.Queue(), [], []
    monkeypatch.setattr(workers, "get_event_q", lambda: events)
    monkeypatch.setattr(workers, "send_with_budget", lambda *a, **k: failures.append((a, k)))
    monkeypatch.setattr(message_bus, "get_bridge", lambda: SimpleNamespace(send_chat_action=lambda *a, **k: True))
    monkeypatch.setattr(agent_module.subagent_runtime, "apply_task_start_settings_or_disclose", lambda *a: None)
    entered, release = threading.Event(), threading.Event()
    transport.results = [_refusal(), result()]
    transport.dispatch = ["not_started", "response_received"]
    monkeypatch.setattr(client, "claudexor_model_sources", lambda: {
        "sources": [{"id": "codex", "credentialHarness": "fixture-harness"}]})

    def catalog(*args, **kwargs):
        entered.set()
        assert release.wait(30), "test did not release the catalog read"
        raise ClaudexorUnavailable("subscription_window_exhausted", "Still exhausted")

    monkeypatch.setattr(client, "claudexor_model_catalog", catalog)
    messages = [{"role": "user", "content": "Continue the same conversational turn"}]
    observed = {}

    monkeypatch.setattr(agent_module.OuroborosAgent, "_log_worker_boot_once", lambda *a: None)
    turn = agent_module.OuroborosAgent(agent_module.Env(root, root))
    turn.llm = client
    monkeypatch.setattr(turn, "_start_task_heartbeat_loop", lambda *a: None)

    def prepare(task, _refusal):
        observed.update(task=task, owner=model_wait.current_model_wait())
        ctx = turn.tools._ctx
        ctx.task_id, ctx.chat_id = task["id"], task["chat_id"]
        ctx.task_metadata = task.get("metadata") or {}
        ctx.task_attempt, ctx.is_ephemeral_turn = 1, True
        ctx.model_wait_context = observed["owner"]
        observed["owner"].tool_context = ctx
        return ctx, messages, {"budget_remaining": 100}

    def run_loop(**kwargs):
        observed["answer"] = client.chat(kwargs["messages"], MODEL, model_role="main")
        answer, usage = observed["answer"]
        return answer["content"], usage, {"tool_calls": [], "reasoning_notes": []}

    monkeypatch.setattr(turn, "_prepare_task_context", prepare)
    monkeypatch.setattr(agent_module, "run_llm_loop", run_loop)
    context = SimpleNamespace(RUNNING={}, DRIVE_ROOT=root, consciousness=None,
                              append_jsonl=append_jsonl, bridge=SimpleNamespace(push_log=published.append))

    @contextmanager
    def running(chat_id=7):
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(workers._run_chat_task, turn, chat_id, messages[0]["content"],
                                     task_metadata={"client_message_id": "owner-input"}, ephemeral=True)
            try:
                assert entered.wait(5), failures
                produced, initial_events = [], []
                while not events.empty():
                    event = events.get_nowait()
                    initial_events.append(event)
                    if event.get("type") == "task_model_wait":
                        produced.append(event)
                assert produced
                latest = produced[-1]
                handle_task_model_wait(latest, context)
                yield SimpleNamespace(root=root, transport=transport, client=client, registry=registry,
                    task=observed["task"], owner=observed["owner"], event=latest, published=published,
                    context=context, release=release, future=future, observed=observed, events=events,
                    initial_events=initial_events)
            finally:
                observed.get("owner") and observed["owner"].close()
                release.set()
                future.result(timeout=5)
    return running


@pytest.mark.serial
@pytest.mark.parametrize("first", ["web", "host"])
@pytest.mark.parametrize("action", ["retry", "switch"])
def test_ephemeral_producer_reaches_decision_and_resumes_same_call(ephemeral_call, first, action):
    with ephemeral_call() as flow, _decision_clients(flow.root) as clients:
        assert len(flow.published) == 1, "real ephemeral wait event was dropped"
        assert flow.published[0]["chat_id"] == 7
        assert workers.direct_chat_turn(flow.task["id"]) is None
        assert workers.direct_chat_turn()["id"] == "other-turn"
        body = _action_for(flow.event, action, **({"model": MODEL, "credential_profile_id": "replacement",
                           "use_local": False, "persist_role": False} if action == "switch" else {}))
        response = clients[first](body)
        assert response.status_code == 202, response.json()
        assert response.json()["saved"] is False and response.json()["applied"] is False
        replay = clients["host" if first == "web" else "web"](body)
        assert replay.status_code == 200 and replay.json()["duplicate"] is True
        assert not (flow.root / "task_results" / f"{flow.task['id']}.json").exists()
        mailbox = owner_mailbox._mailbox_path(flow.root, flow.task["id"])
        assert mailbox.exists()
        flow.release.set()
        flow.future.result(timeout=5)
        assert flow.observed["answer"][0] == result()["message"]
        assert len(flow.transport.operations) == 2
        assert flow.transport.uploads[0][0]["messages"] == flow.transport.uploads[1][0]["messages"]
        if action == "switch":
            assert flow.transport.uploads[-1][0]["account"] == {"mode": "pin", "profileId": "replacement"}
        assert [row["state"] for row in ledger(flow.root)] == [
            "reserved", "dispatched", "released", "reserved", "dispatched", "settled"]
        assert flow.owner.closed
        assert [row["activity_id"] for row in flow.registry.snapshot()] == ["other-turn"]
        assert not mailbox.exists()
        assert not (flow.root / "task_results" / f"{flow.task['id']}.json").exists()
        assert clients[first](body).status_code == 409
        handle_task_model_wait(flow.event, flow.context)
        assert len(flow.published) == 1  # Closed activity cannot revive its wait card.


@pytest.mark.serial
def test_ephemeral_wait_rejects_stale_attempt_revision_and_closed_owner(ephemeral_call):
    with ephemeral_call() as flow, _decision_clients(flow.root) as clients:
        assert len(flow.published) == 1
        body = _action_for(flow.event, "auto_continue", auto_continue=False)
        for field, value in (("task_attempt", 2), ("revision", 999)):
            invalid = {**flow.event, field: value}
            handle_task_model_wait(invalid, flow.context)
            assert len(flow.published) == 1
        stale = clients["web"]({**body, "revision": body["revision"] + 1})
        assert stale.status_code == 409 and stale.json()["reason_code"] == "stale_model_wait"
        row = flow.owner.waits[flow.event["wait_id"]]
        row["task_attempt"] = 2
        stale = clients["host"](body)
        assert stale.status_code == 409 and stale.json()["reason_code"] == "stale_model_wait"
        row["task_attempt"] = 1
        assert clients["host"](body).status_code == 202
        flow.owner._drain_controls()
        assert row["auto_continue"] is False
        assert clients["web"](body).json()["applied"] is True
        snapshot = flow.registry.get(flow.task["id"]).to_dict()
        assert snapshot["kind"] == "ephemeral_decision" and snapshot["model_waits"]
        assert "cancelable" not in snapshot and "model_wait_owner" not in snapshot
        before = deepcopy(snapshot["model_waits"])
        flow.owner.close()  # Closing the scope wins even while the activity entry remains.
        assert clients["web"](body).status_code == 409
        handle_task_model_wait(flow.event, flow.context)
        assert len(flow.published) == 1
        assert not owner_mailbox._mailbox_path(flow.root, flow.task["id"]).exists()
        assert before == {key: {k: v for k, v in row.items() if not k.startswith("_")}
                          for key, row in flow.owner.waits.items()}


@pytest.mark.serial
def test_ephemeral_close_waits_for_admitted_mailbox_write_then_cleans(ephemeral_call, monkeypatch):
    with ephemeral_call() as flow, _decision_clients(flow.root) as clients:
        entered, release, closing = threading.Event(), threading.Event(), threading.Event()
        write = owner_mailbox.write_owner_message

        def held_write(*args, **kwargs):
            entered.set()
            assert release.wait(5)
            return write(*args, **kwargs)

        def close():
            closing.set()
            flow.owner.close()

        monkeypatch.setattr(owner_mailbox, "write_owner_message", held_write)
        body = _action_for(flow.event, "retry")
        with ThreadPoolExecutor(max_workers=2) as executor:
            decision = executor.submit(clients["web"], body)
            try:
                assert entered.wait(5)
                stopped = executor.submit(close)
                assert closing.wait(5)
                assert not stopped.done()
            finally:
                release.set()
            assert decision.result(timeout=5).status_code == 202
            stopped.result(timeout=5)
        assert flow.owner.closed
        assert not owner_mailbox._mailbox_path(flow.root, flow.task["id"]).exists()
        assert clients["host"](body).status_code == 409


@pytest.mark.serial
def test_ephemeral_wait_keeps_cancel_and_registration_fences(ephemeral_call):
    from ouroboros.cancel_intents import request_cancel

    with ephemeral_call() as flow, _decision_clients(flow.root) as clients:
        task_id = flow.task["id"]
        assert flow.registry.ephemeral_model_wait(flow.root / "other", task_id) is None
        wrong_root = SimpleNamespace(**{**vars(flow.context), "DRIVE_ROOT": flow.root / "other"})
        handle_task_model_wait(flow.event, wrong_root)
        assert len(flow.published) == 1
        request_cancel(flow.root, task_id)
        refused = clients["web"](_action_for(flow.event, "retry"))
        assert refused.status_code == 409 and refused.json()["reason_code"] == "cancel_pending"
        assert not owner_mailbox._mailbox_path(flow.root, task_id).exists()
        flow.registry.unregister(task_id)
        assert flow.registry.ephemeral_model_wait(flow.root, task_id) is None
        handle_task_model_wait(flow.event, flow.context)
        assert len(flow.published) == 1
        assert clients["host"](_action_for(flow.event, "retry")).status_code == 409
