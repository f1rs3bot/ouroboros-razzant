"""A real ephemeral wait producer reaches the browser and its decision ingress."""

import json
from contextlib import ExitStack

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from ouroboros.gateway.history import make_chat_history_endpoint
from ouroboros.gateway.state import _chat_activities_snapshot_safe
from ouroboros.gateway.task_decision import api_decision_answer
from supervisor.events_worker_reports import _handle_log_event
from tests.test_ephemeral_model_wait import ephemeral_call as turn_fixture
from tests.test_llm_claudexor import setup as gateway_fixture
from tests.test_subscription_setup_browser import capture, subscription_ui as ui_fixture

setup = gateway_fixture
ephemeral_call = turn_fixture
subscription_ui = ui_fixture
pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]


def test_ephemeral_wait_hydrates_without_task_controls_and_browser_retry_resumes(
    subscription_ui, ephemeral_call,
):
    ui, sockets, responses = subscription_ui, [], []
    page = ui["page"]
    page.route_web_socket("**/ws", lambda socket: sockets.append(socket))
    with ephemeral_call(chat_id=1) as flow, ExitStack() as stack:
        task_id = flow.task["id"]
        app = Starlette(routes=[Route("/api/chat/history", make_chat_history_endpoint(flow.root)),
                               Route("/api/decisions", api_decision_answer, methods=["POST"])])
        app.state.drive_root = flow.root
        client = stack.enter_context(TestClient(app))

        def respond(route, body):
            route.fulfill(content_type="application/json", body=json.dumps(body))

        def history(route):
            response = client.get("/api/chat/history?chat_id=1")
            route.fulfill(status=response.status_code, content_type="application/json", body=response.content)

        def decide(route):
            response = client.post("/api/decisions", json=route.request.post_data_json)
            responses.append(response)
            route.fulfill(status=response.status_code, content_type="application/json", body=response.content)

        page.route("**/api/state", lambda route: respond(route, {"sha": "fixture", "supervisor_ready": True,
            "projects": [], "active_chat_activities": _chat_activities_snapshot_safe(flow.root)}))
        page.route("**/api/chat/history*", history)
        page.route("**/api/decisions", decide)
        page.goto(ui["url"] + "/")
        waiter = page.locator(f'[data-wait-id="{flow.event["wait_id"]}"]')
        waiter.wait_for()
        card = page.locator(f'.chat-live-card[data-task-id="{task_id}"]')
        capture(page, "ephemeral-real-wait-hydrated")
        assert card.locator("[data-turn-into-project]").count() == 0
        assert card.locator("[data-cancel-run]").count() == 0
        # The actual task_started event and wait were emitted by the Agent.
        # Replaying those transport envelopes must keep the same card/control form.
        for event in flow.initial_events:
            if event.get("type") == "log_event":
                before = len(flow.published)
                _handle_log_event(event, flow.context)
                for published in flow.published[before:]:
                    sockets[-1].send(json.dumps({"type": "log", "chat_id": 1, "data": published}))
        page.reload()
        waiter.wait_for()
        assert card.locator("[data-turn-into-project]").count() == 0
        waiter.locator("[data-wait-retry]").click()
        waiter.locator("[data-wait-notice]").filter(has_text="Request accepted").wait_for()
        assert responses[-1].status_code == 202
        flow.release.set()
        flow.future.result(timeout=5)
        terminal = next(event for event in list(flow.events.queue) if event.get("type") == "task_done")
        assert terminal["_ephemeral"] is True and terminal["ephemeral_decision"] is True
        sockets[-1].send(json.dumps({"type": "log", "chat_id": 1, "data": terminal}))
        waiter.wait_for(state="detached")
        page.wait_for_selector(f'.chat-live-card[data-task-id="{task_id}"][data-finished="1"]')
        assert len(flow.transport.operations) == 2
        assert [row["activity_id"] for row in flow.registry.snapshot()] == ["other-turn"]
        capture(page, "ephemeral-real-wait-completed")
