"""Acting role and exact account survive ordinary, prospective and forced sends."""

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from ouroboros import loop_forced_finalization as forced, task_pacing
from ouroboros.loop_model_call import _RoundModelCallContext, _adopt_fallback_route, _call_round_model
from ouroboros.llm import LLMClient
from ouroboros.loop_llm_call import call_llm_with_retry
from ouroboros.loop_round_limits import _RoundLimitContext
from ouroboros.model_slots import MODEL_ACCOUNTS_KEY, task_model_binding
from ouroboros.model_wait import task_model_wait_scope
from ouroboros.tools.registry import ToolRegistry
from tests.test_llm_claudexor import MODEL, setup as setup


@pytest.fixture
def acting(setup, monkeypatch):
    root, gateway, client = setup
    gateway.results *= 4
    gateway.dispatch *= 4
    monkeypatch.setenv(MODEL_ACCOUNTS_KEY, json.dumps({"main": "main-only", "fallback": ["fallback-only"]}))
    monkeypatch.setenv("OUROBOROS_IMAGE_INPUT_MODE", "inline")
    catalogs = []

    def catalog(source, profile=None, *, requested_model=None):
        catalogs.append((source, profile, requested_model))
        return {"source": source, "credentialProfileId": profile, "models": [
            {"id": requested_model, "inputModalities": ["text"] if profile == "main-only" else ["text", "image"]}]}

    monkeypatch.setattr(LLMClient, "claudexor_model_catalog", staticmethod(catalog))
    messages = [{"role": "system", "content": "Own SYSTEM"}, {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}]
    tools = ToolRegistry(repo_dir=root.parent, drive_root=root)
    tools._ctx.task_metadata = {"configured_subagent": {"selected_subagent_id": "visual-actor",
        "route": {"kind": "api_model", "target_id": MODEL, "credential_profile_id": "actor-only"}}}
    tools._ctx.active_model = MODEL
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    ctx = _RoundLimitContext(messages=deepcopy(messages), llm=client, active_model=MODEL,
        active_effort="high", max_retries=1, drive_logs=logs, task_id="task-one", round_idx=2,
        event_queue=None, accumulated_usage={}, task_type="task", active_use_local=False,
        max_rounds=2, drive_root=root, tools=tools, tool_schemas=[])
    return SimpleNamespace(root=root, gateway=gateway, client=client, messages=messages,
                           tools=tools, ctx=ctx, catalogs=catalogs)


def test_forced_call_keeps_configured_actor_account_and_image(acting):
    role, pin = task_model_binding({"task_metadata": acting.tools._ctx.task_metadata})
    call_llm_with_retry(acting.client, deepcopy(acting.messages), MODEL, [], "high", 1,
        acting.ctx.drive_logs, "task-one", 1, None, {}, model_role=role, model_account_override=pin)
    assert forced._call_forced_model_once(acting.ctx) == "Ответ 🐍"
    normal, final = [payload for payload, _ in acting.gateway.uploads]
    assert normal["account"] == {"mode": "pin", "profileId": "actor-only"}
    assert final["account"] == normal["account"]
    assert normal["messages"] == final["messages"] == acting.messages


def test_prospective_image_preparation_keeps_actor_pin(acting):
    _, prepared = task_pacing.prepared_wrapup_candidate(
        acting.ctx, deepcopy(acting.messages), allow_server_web_search=False)
    assert prepared == acting.messages
    assert acting.catalogs[-1][1] == "actor-only"


def test_subscription_prepared_candidate_admits_the_actual_forced_send(acting):
    acting.ctx.messages = [{"role": "user", "content": "Please finish"}]
    request, prepared = task_pacing.prepared_wrapup_candidate(
        acting.ctx, deepcopy(acting.ctx.messages), allow_server_web_search=False)
    assert request.provider == "claudexor"
    text = forced._call_forced_model_once(acting.ctx, initial_messages=prepared, admitted_request=request)
    assert text == "Ответ 🐍", acting.ctx.accumulated_usage
    assert len(acting.gateway.creates) == 1


@pytest.mark.parametrize("shape", ["mid_round_image", "late_system_notice"])
def test_subscription_prospective_and_send_share_transcript_normalization(acting, shape):
    if shape == "mid_round_image":
        assistant = {"role": "assistant", "content": "", "tool_calls": [
            {"id": "shot", "type": "function", "function": {"name": "browser", "arguments": "{}"}}],
            "nativeContinuation": {"format": "codex.responses.v1", "payload": [
                {"type": "reasoning", "encrypted_content": "opaque+==\r\n"}]}}
        messages = [{"role": "system", "content": "Own SYSTEM"},
            {"role": "user", "content": "Open the page"}, assistant,
            deepcopy(acting.messages[-1]),
            {"role": "tool", "tool_call_id": "shot", "content": "Screenshot attached"}]
    else:
        messages = [{"role": "system", "content": "Own SYSTEM"},
            {"role": "user", "content": "Start"}, {"role": "assistant", "content": "Started"},
            {"role": "system", "content": "Runtime notice"}]
    messages.append({"role": "user", "content": "Finish from the current evidence"})
    original = deepcopy(messages)
    normalized = LLMClient._normalize_system_message_placement(messages)
    assert normalized != original
    acting.ctx.messages = messages
    request, prepared = task_pacing.prepared_wrapup_candidate(
        acting.ctx, messages, allow_server_web_search=False)
    assert forced._call_forced_model_once(acting.ctx, initial_messages=prepared,
                                          admitted_request=request) == "Ответ 🐍"
    sent = acting.gateway.uploads[0][0]["messages"]
    assert sent == normalized and messages == original
    assert len(acting.gateway.creates) == 1
    if shape == "mid_round_image":
        assert sent[3]["role"] == "tool" and sent[3]["tool_call_id"] == "shot"
        assert sent[4]["content"][0]["type"] == "image_url"
        assert sent[2]["nativeContinuation"] == assistant["nativeContinuation"]


@pytest.mark.parametrize("pin", ["temporary-actor", ""])
def test_forced_and_prospective_preserve_task_actor_override_including_auto(acting, pin):
    task = {"id": "task-one", "_attempt": 1}
    with task_model_wait_scope(task=task, drive_root=acting.root,
                              event_queue=None, worker_slot_held=True) as wait:
        wait.overrides["subagent:visual-actor"] = {
            "model": MODEL, "use_local": False, "model_account_override": pin}
        request, prepared = task_pacing.prepared_wrapup_candidate(
            acting.ctx, deepcopy(acting.messages), allow_server_web_search=False)
        assert prepared == acting.messages
        assert forced._call_forced_model_once(acting.ctx, initial_messages=prepared,
                                              admitted_request=request) == "Ответ 🐍"
    payload = acting.gateway.uploads[0][0]
    assert payload["account"] == ({"mode": "pin", "profileId": pin} if pin else {"mode": "auto"})
    assert payload["messages"] == acting.messages
    assert acting.catalogs[-1][1] == (pin or None)


def test_adopted_fallback_plan_keeps_its_role_not_actor_or_main(acting):
    plan = SimpleNamespace(model_role="fallback:0", model_route={"credentialProfileId": "observed-not-a-pin"})
    _adopt_fallback_route(acting.ctx, acting.tools, MODEL, False, acting.ctx.messages,
        deepcopy(acting.messages), plan, "max", [], acting.ctx.accumulated_usage)
    assert acting.tools._ctx.context_fit_plan is plan
    request, prepared = task_pacing.prepared_wrapup_candidate(
        acting.ctx, deepcopy(acting.messages), allow_server_web_search=False)
    assert forced._call_forced_model_once(acting.ctx, initial_messages=prepared,
                                          admitted_request=request) == "Ответ 🐍"
    assert acting.gateway.uploads[0][0]["account"] == {"mode": "pin", "profileId": "fallback-only"}
    assert prepared == acting.messages and acting.catalogs[-1][1] == "fallback-only"


def test_unconfigured_task_still_binds_main(acting):
    acting.tools._ctx.task_metadata = {}
    acting.ctx.messages = [{"role": "user", "content": "finish"}]
    assert forced._call_forced_model_once(acting.ctx) == "Ответ 🐍"
    assert acting.gateway.uploads[0][0]["account"] == {"mode": "pin", "profileId": "main-only"}


def test_task_binding_priority_is_role_then_plan_then_frozen_actor(acting):
    task = {"task_metadata": acting.tools._ctx.task_metadata}
    plan = SimpleNamespace(model_role="fallback:0", model_route={"credentialProfileId": "observed-only"})
    assert task_model_binding(task) == ("subagent:visual-actor", "actor-only")
    assert task_model_binding(task, context_fit_plan=plan) == ("fallback:0", None)
    assert task_model_binding({**task, "model_role": "main"}, context_fit_plan=plan) == ("main", None)
    assert task_model_binding(task, overrides={"subagent:visual-actor": {
        "model_account_override": ""}}) == ("subagent:visual-actor", "")


@pytest.mark.parametrize("fallback", [False, True])
def test_browser_attachment_uses_live_binding_before_main_send(acting, fallback):
    from ouroboros.tools.browser import _inject_native_screenshot

    ctx = acting.tools._ctx
    ctx.messages = []
    ctx.task_metadata["configured_subagent"]["route"]["credential_profile_id"] = "main-only"
    with task_model_wait_scope(task={"id": "task-one", "_attempt": 1}, drive_root=acting.root,
                              event_queue=None, worker_slot_held=True) as wait:
        if fallback:
            ctx.context_fit_plan = SimpleNamespace(model_role="fallback:0")
        else:
            wait.overrides["subagent:visual-actor"] = {
                "model": MODEL, "use_local": False, "model_account_override": "temporary-actor"}
        _inject_native_screenshot(ctx, "QUFBQQ==")
    assert ctx.messages and ctx.messages[-1]["content"][-1]["type"] == "image_url"
    assert acting.catalogs[-1][1] == ("fallback-only" if fallback else "temporary-actor")


@pytest.mark.parametrize("fallback", [False, True])
def test_ordinary_round_uses_the_same_actor_or_active_plan_binding(acting, fallback):
    if fallback:
        acting.tools._ctx.context_fit_plan = SimpleNamespace(model_role="fallback:0",
            model_route=deepcopy(acting.gateway.results[0]["route"]))
    ctx = _RoundModelCallContext(llm=acting.client, messages=deepcopy(acting.messages),
        tools=acting.tools, context_fit_plan=None, active_model=MODEL, tool_schemas=[],
        active_effort="high", max_retries=1, drive_logs=acting.ctx.drive_logs, task_id="task-one",
        round_idx=1, event_queue=None, accumulated_usage={}, task_type="task",
        active_use_local=False, active_context_mode="max", drive_root=acting.root, attempt_cap=1)
    message, _, _ = _call_round_model(ctx)
    assert message["content"] == "Ответ 🐍"
    payload = acting.gateway.uploads[0][0]
    assert payload["account"] == {"mode": "pin", "profileId": "fallback-only" if fallback else "actor-only"}
    assert payload["messages"] == acting.messages
