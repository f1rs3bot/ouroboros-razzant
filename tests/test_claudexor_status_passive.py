"""Account status reads metadata; only launch preparation probes executables."""

from __future__ import annotations

import json

import httpx
import pytest
from starlette.applications import Starlette
from starlette.testclient import TestClient
from starlette.routing import Route

from ouroboros import claudexor_daemon as owned
from ouroboros import claudexor_runtime as runtime
from ouroboros.gateway.claudexor_accounts import api_claudexor_status
from ouroboros.gateway.models import api_model_catalog


@pytest.fixture
def metadata_engine(monkeypatch, tmp_path):
    """Real descriptor discovery and gateway HTTP, with no engine or vendor process."""
    from ouroboros.gateways import claudexor as wire

    config_dir = tmp_path / "claudexor"
    monkeypatch.setattr(owned, "owned_config_dir", lambda: config_dir)
    calls = []

    def forbidden(*_args, **_kwargs):
        pytest.fail("metadata must not start, install, reconcile or use the operator engine")

    monkeypatch.setattr(owned.OwnedClaudexorDaemon, "ensure_running", forbidden)
    monkeypatch.setattr(owned.OwnedClaudexorDaemon, "reconcile_rotation", forbidden)
    monkeypatch.setattr(runtime.ClaudexorRuntimeManager, "ensure", forbidden)
    monkeypatch.setattr(owned.subprocess, "Popen", forbidden)
    monkeypatch.setattr(wire, "discover_daemon", forbidden)
    state = {"reachable": True, "handshake_error": False}

    def respond(request):
        calls.append((request.method, request.url.path, dict(request.url.params)))
        assert request.headers["Authorization"] == "Bearer fixture-owned-token"
        if not state["reachable"]:
            raise httpx.ConnectError("owned engine is stopped", request=request)
        if request.url.path == "/v2/handshake":
            return httpx.Response(200, json={"compatible": not state["handshake_error"],
                "protocolMajor": wire.CLAUDEXOR_PROTOCOL_MAJOR,
                "engine": {"version": wire.CLAUDEXOR_MIN_VERSION}})
        if request.url.path == "/v2/model-sources":
            return httpx.Response(200, json={"sources": [{"id": "codex", "label": "Codex",
                "credentialHarness": "codex"}]})
        assert request.url.path == "/v2/model-sources/codex/models"
        return httpx.Response(200, json={"source": "codex", "credentialProfileId": "account-a",
            "accountFingerprint": "identity-a", "provenance": "fixture exact catalog",
            "observedAt": "2026-09-07T00:00:00Z", "models": [{"id": "exact-model",
                "contextWindow": 272000, "maxContextWindow": 872000}]})

    original_init = wire.ClaudexorGateway.__init__
    clients = []

    def initialize(gateway, endpoint):
        original_init(gateway, endpoint)
        gateway._client.close()
        gateway._client = httpx.Client(base_url="http://127.0.0.1:1",
            headers={"Authorization": f"Bearer {endpoint.token}"},
            transport=httpx.MockTransport(respond))
        clients.append(gateway._client)

    monkeypatch.setattr(wire.ClaudexorGateway, "__init__", initialize)

    def provision():
        descriptor = config_dir / "daemon" / "control-api.json"
        descriptor.parent.mkdir(parents=True)
        token = config_dir / "daemon" / "token"
        token.write_text("fixture-owned-token", encoding="utf-8")
        descriptor.write_text(json.dumps({"host": "127.0.0.1", "port": 1,
            "tokenPath": str(token)}), encoding="utf-8")

    return state, calls, clients, provision, config_dir


@pytest.mark.parametrize("provisioned", [False, True])
@pytest.mark.parametrize("query", ["", "?source_id=codex&credential_profile_id=account-a"])
def test_catalog_get_cold_or_stopped_never_starts_engine(metadata_engine, monkeypatch, query, provisioned):
    from ouroboros.gateway import models

    state, calls, clients, provision, config_dir = metadata_engine
    monkeypatch.setattr(models, "load_settings", lambda: {})
    if provisioned:
        provision()
        state["reachable"] = False
    before = sorted(str(path) for path in config_dir.rglob("*"))
    app = Starlette(routes=[Route("/api/model-catalog", api_model_catalog)])
    with TestClient(app) as client:
        payload = client.get("/api/model-catalog" + query).json()
    assert payload["items"] == [] and payload["model_sources"] == []
    if provisioned or query:
        assert payload["errors"][0]["code"] == (
            "daemon_unreachable" if provisioned else "daemon_not_discovered")
    else:
        assert payload["errors"] == [] and calls == []
    assert sorted(str(path) for path in config_dir.rglob("*")) == before
    assert all(client.is_closed for client in clients)


def test_warm_owned_metadata_preserves_models_profiles_and_closes(metadata_engine):
    from ouroboros.llm import LLMClient

    _, calls, clients, provision, _ = metadata_engine
    provision()
    app = Starlette(routes=[Route("/api/model-catalog", api_model_catalog)])
    with TestClient(app) as client:
        payload = client.get("/api/model-catalog?source_id=codex&credential_profile_id=account-a").json()
    assert payload["errors"] == []
    assert payload["items"][0]["value"] == "claudexor::codex=exact-model"
    assert payload["items"][0]["credential_profile_id"] == "account-a"
    assert LLMClient.claudexor_model_sources()["sources"][0]["id"] == "codex"
    catalog = LLMClient.claudexor_model_catalog("codex", "account-a", requested_model="exact-model")
    assert catalog["models"][0]["maxContextWindow"] == 872000
    assert calls[-1][2] == {"credentialProfileId": "account-a", "requestedModel": "exact-model"}
    assert sum(path == "/v2/handshake" for _, path, _ in calls) == 3
    assert len(clients) == 3 and all(client.is_closed for client in clients)


@pytest.mark.parametrize("operation", ["sources", "catalog", "capability"])
def test_cold_metadata_is_unknown_not_healthy(metadata_engine, tmp_path, operation):
    from ouroboros import capability_evidence as ce
    from ouroboros.gateways.claudexor import ClaudexorUnavailable
    from ouroboros.llm import LLMClient

    _, calls, clients, _, config_dir = metadata_engine
    if operation == "capability":
        evidence = ce.probe(tmp_path / "evidence", provider="claudexor",
            model="claudexor::codex=exact-model", allow_fetch=True, allow_generative=False)
        assert evidence.status == ce.STATUS_FAILED and evidence.window_tokens == 0
        assert not ce.confirms_at_least(evidence)
    else:
        with pytest.raises(ClaudexorUnavailable) as caught:
            if operation == "sources":
                LLMClient.claudexor_model_sources()
            else:
                LLMClient.claudexor_model_catalog("codex")
        assert caught.value.code == "daemon_not_discovered"
    assert not config_dir.exists() and not calls and not clients


def test_read_gateway_closes_after_failed_handshake(metadata_engine):
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    state, _, clients, provision, _ = metadata_engine
    provision()
    state["handshake_error"] = True
    with pytest.raises(ClaudexorUnavailable) as caught:
        owned.read_owned_gateway()
    assert caught.value.code == "protocol_incompatible"
    assert len(clients) == 1 and clients[0].is_closed


@pytest.mark.parametrize("state", ["not_provisioned", "stale"])
def test_status_get_never_prepares_or_probes_a_launch(monkeypatch, tmp_path, state):
    """Installed metadata remains visible without even the first Node probe."""
    manager = runtime.ClaudexorRuntimeManager()
    pin = manager.pin
    assert pin is not None
    metadata = {
        "version": pin.version, "build_sha": pin.build_sha,
        "node_version": pin.node_version, "archive_source": "cache",
    }
    monkeypatch.setattr(manager, "_managed_metadata", lambda: dict(metadata))
    monkeypatch.setattr(manager, "_install_in_progress", lambda: False)
    monkeypatch.setattr(runtime, "get_runtime_manager", lambda: manager)
    monkeypatch.setattr(runtime, "managed_runtime_root", lambda: tmp_path / "runtime")
    daemon = owned.OwnedClaudexorDaemon()
    monkeypatch.setattr(daemon, "_classify_liveness", lambda: (None, state, ""))
    monkeypatch.setattr(owned, "get_owned_daemon", lambda: daemon)
    monkeypatch.setattr(owned, "owned_config_dir", lambda: tmp_path / "claudexor")
    monkeypatch.setattr(owned, "verify_owned_home", lambda: "")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("passive status must not prepare a command or start a process")

    monkeypatch.setattr(manager, "resolve_command", forbidden)
    monkeypatch.setattr(manager, "ensure", forbidden)
    monkeypatch.setattr(daemon, "ensure_running", forbidden)
    monkeypatch.setattr(owned.subprocess, "Popen", forbidden)
    app = Starlette(routes=[Route("/api/claudexor/status", api_claudexor_status)])
    with TestClient(app) as client:
        for _ in range(2):
            response = client.get("/api/claudexor/status?include=models")
            assert response.status_code == 200
            payload = response.json()
            assert payload["daemon"]["state"] == state
            assert payload["daemon"]["runtime"]["state"] == "ready"
            assert payload["daemon"]["runtime"]["node_version"] == pin.node_version
            assert "binary" not in payload["daemon"]
            assert payload["reads"] == dict.fromkeys(("catalog", "accounts", "quota"), "not_read")


def test_launch_preparation_still_probes_the_exact_node(monkeypatch, tmp_path):
    manager = runtime.ClaudexorRuntimeManager()
    pin = manager.pin
    assert pin is not None
    monkeypatch.delenv("OUROBOROS_CLAUDEXOR_BIN", raising=False)
    monkeypatch.setattr(manager, "_managed_metadata", lambda: {"version": pin.version})
    monkeypatch.setattr(runtime, "managed_runtime_root", lambda: tmp_path / "runtime")
    calls = []

    def resolve_node(requested_pin):
        assert requested_pin is pin
        calls.append("node")
        return str(tmp_path / "node")

    monkeypatch.setattr(manager, "_resolve_node", resolve_node)
    monkeypatch.setattr(manager, "_probe", lambda command, requested_pin: calls.append("engine"))
    assert manager.ensure()[0] == str(tmp_path / "node")
    assert calls == ["node", "engine"]
