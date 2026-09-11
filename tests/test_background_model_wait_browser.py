"""The existing background card reuses subscription waiting across live cycles."""

import copy
import json

import pytest

from tests import test_subscription_setup_browser as ui_fixture

subscription_ui = ui_fixture.subscription_ui
capture = ui_fixture.capture

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]


def test_background_wait_snapshot_stays_in_its_own_chat(subscription_ui):
    ui, page = subscription_ui, subscription_ui["page"]
    row = {"wait_id": "bg-only-main", "task_attempt": 1, "revision": 1,
           "role": "consciousness", "model": "claudexor::codex=gpt-test",
           "source": "codex", "credential_profile_id": "personal",
           "credential_harness": "codex", "reason": "quota", "auto_continue": True,
           "state": "waiting", "worker_slot_held": False}
    snapshot = {"model_wait_owner_id": "cycle-a", "chat_id": 1,
                "model_waits": {row["wait_id"]: row}, "running": True}
    project = {"id": "project-two", "name": "Project two", "chat_id": 2, "lifecycle": "active"}

    def reply(route, value):
        route.fulfill(content_type="application/json", body=json.dumps(value))

    page.route("**/api/state", lambda route: reply(route, {
        "supervisor_ready": True, "projects": [project], "active_chat_activities": [],
        "bg_consciousness_enabled": True, "bg_consciousness_state": snapshot,
    }))
    page.route("**/api/projects", lambda route: reply(route, {"projects": [project]}))
    page.route("**/api/chat/history*", lambda route: reply(route, {"messages": []}))
    page.goto(ui["url"] + "/")
    page.wait_for_selector('[data-wait-id="bg-only-main"]')
    with page.expect_response("**/api/chat/history?chat_id=2"):
        page.locator(".nav-project-item").filter(has_text="Project two").click()
    assert page.locator('#project-panel-body [data-wait-id="bg-only-main"]').count() == 0
    assert page.locator('#page-chat [data-wait-id="bg-only-main"]').count() == 1
    capture(page, "background-wait-isolated-from-project")


def test_background_wait_reload_pause_and_new_cycle_use_one_card(subscription_ui):
    ui, page = subscription_ui, subscription_ui["page"]
    row = {"wait_id": "bg-wait-a", "task_attempt": 1, "revision": 1, "role": "consciousness",
           "model": "claudexor::codex=gpt-test", "source": "codex", "credential_profile_id": "personal",
           "credential_harness": "codex", "reason": "quota", "reset_at": "", "auto_continue": True,
           "state": "waiting", "worker_slot_held": False, "model_wait_owner_id": "cycle-a"}
    snapshot = {"model_wait_owner_id": "cycle-a", "chat_id": 1, "model_waits": {row["wait_id"]: row},
                "running": True, "paused": False, "detail": "Waiting for model access"}
    # test_background_wait_forwarding_and_reload_use_existing_owner pins that
    # active-owner history never terminal-stamps the old background progress.
    history_rows = [{"role": "system", "text": "Earlier background thought", "task_id": "bg-consciousness",
                     "is_progress": True, "ts": "2026-09-07T00:00:00Z"}]
    actions = []

    def reply(route, body):
        route.fulfill(content_type="application/json", body=json.dumps(body))

    page.route("**/api/state", lambda route: reply(route, {
        "supervisor_ready": True, "active_chat_activities": [], "projects": [],
        "bg_consciousness_enabled": True, "bg_consciousness_state": snapshot,
    }))
    page.route("**/api/chat/history*", lambda route: reply(route, {"messages": [*history_rows, {
        "role": "system", "text": "", "system_type": "task_model_wait", "task_id": "bg-consciousness",
        "is_progress": False, "model_wait_live": True, **snapshot,
    }]}))

    def decide(route):
        body = route.request.post_data_json
        actions.append(body)
        assert body["decision_id"] == f"model_wait:bg-consciousness:{row['wait_id']}"
        row.update(revision=row["revision"] + 1, auto_continue=body["auto_continue"],
                   applied_request_id=body["request_id"])
        reply(route, {"ok": True, "wait": row, "applied": True, "saved": False})

    page.route("**/api/decisions", decide)
    page.goto(ui["url"] + "/")
    page.wait_for_selector('[data-wait-id="bg-wait-a"]')
    card = page.locator('.chat-live-card').filter(has=page.locator('[data-wait-id="bg-wait-a"]'))
    assert "Background" in card.inner_text()
    assert "No worker slot is held" in card.inner_text()
    assert page.locator('.model-waits').count() == 1
    page.locator('[data-wait-auto]').uncheck()
    page.wait_for_function("() => !document.querySelector('[data-wait-auto]').disabled")
    assert actions and actions[0]["auto_continue"] is False
    page.reload()
    page.wait_for_selector('[data-wait-id="bg-wait-a"]')
    assert not page.locator('[data-wait-auto]').is_checked()
    capture(page, "background-wait-reloaded")

    snapshot["paused"] = True
    page.reload()
    page.wait_for_selector('[data-wait-id="bg-wait-a"]')
    page.wait_for_function("() => [...document.querySelectorAll('.chat-live-phase')].some(el => el.textContent.includes('Paused for foreground'))")
    capture(page, "background-wait-foreground-paused")

    history_rows.append({"role": "system", "text": "", "system_type": "task_model_wait",
                         "task_id": "bg-consciousness", "model_waits": {row["wait_id"]: copy.deepcopy(row)}})
    row.update(wait_id="bg-wait-b", revision=1, model_wait_owner_id="cycle-b", auto_continue=True)
    row.pop("applied_request_id", None)
    snapshot.update(model_wait_owner_id="cycle-b", paused=False, model_waits={row["wait_id"]: row})
    page.reload()
    page.wait_for_selector('[data-wait-id="bg-wait-b"]')
    assert page.locator('[data-wait-id="bg-wait-a"]').count() == 0
    assert page.locator('.model-waits').count() == 1 and page.locator('[data-wait-auto]').is_checked()
    page.set_viewport_size({"width": 390, "height": 844})
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    capture(page, "background-wait-new-cycle-narrow")


@pytest.mark.parametrize("profile", ["", "personal"])
def test_auth_wait_only_starts_login_for_a_known_profile(subscription_ui, profile):
    ui, page = subscription_ui, subscription_ui["page"]
    row = {"wait_id": "auth-wait", "task_attempt": 1, "revision": 1, "role": "consciousness",
           "model": "claudexor::codex=gpt-test", "source": "codex", "credential_profile_id": profile,
           "credential_harness": "codex", "reason": "auth", "reset_at": "", "auto_continue": True,
           "state": "waiting", "worker_slot_held": False, "model_wait_owner_id": "auth-cycle"}
    snapshot = {"model_wait_owner_id": "auth-cycle", "chat_id": 1, "model_waits": {row["wait_id"]: row},
                "running": True, "paused": False}
    logins = []

    def reply(route, value):
        route.fulfill(content_type="application/json", body=json.dumps(value))

    page.route("**/api/state", lambda route: reply(route, {"supervisor_ready": True, "projects": [],
               "active_chat_activities": [], "bg_consciousness_enabled": True, "bg_consciousness_state": snapshot}))
    page.route("**/api/chat/history*", lambda route: reply(route, {"messages": [{
        "role": "system", "text": "", "system_type": "task_model_wait", "task_id": "bg-consciousness",
        "is_progress": False, "model_wait_live": True, **snapshot,
    }]}))
    login_reply = {"job_id": "controlled", "job": {"state": "waiting_for_input", "phase": "awaiting_user"},
                   "sequence": 1, "deviceCode": {"flow": "chatgptDeviceCode", "verificationUrl": "https://auth.example/device", "userCode": "TEST-ONLY"}}

    def login(route):
        logins.append(route.request.post_data_json)
        reply(route, login_reply)

    page.route("**/api/claudexor/login", login)
    page.route("**/api/claudexor/login/controlled", lambda route: reply(route, login_reply))
    page.goto(ui["url"] + "/")
    page.wait_for_selector('[data-wait-id="auth-wait"]')
    page.locator('[data-wait-login]').click()
    page.wait_for_function("() => document.querySelector('[data-settings-tab=providers]')?.getAttribute('aria-selected') === 'true'")
    if profile:
        page.wait_for_selector('[data-login-card]')
        assert logins and logins[0]["profile_id"] == profile
    else:
        assert logins == [] and page.locator('[data-login-card]').count() == 0
