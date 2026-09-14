"""Regression coverage for causal work-item closure authority."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

from ouroboros.tools import plan_evidence, plan_spec, plan_work_items


def _spec(refs_marker: str) -> dict:
    raw = {"goal": "close exact absorbed items"}
    if refs_marker == "explicit":
        raw["work_item_refs"] = ["ibl-2", "ibl-1", "ibl-2"]
    elif refs_marker == "empty":
        raw["work_item_refs"] = []
    spec, errors = plan_spec.normalize_spec(raw)
    assert errors == []
    return spec


def test_work_item_refs_are_identity_and_ordered():
    omitted = _spec("omitted")
    empty = _spec("empty")
    refs = _spec("explicit")

    assert "work_item_refs" not in omitted
    assert empty["work_item_refs"] == []
    assert refs["work_item_refs"] == ["ibl-2", "ibl-1"]
    assert plan_spec.spec_hash(empty) != plan_spec.spec_hash(omitted)
    assert plan_spec.spec_hash(refs) != plan_spec.spec_hash(empty)


def test_work_item_refs_malformed_are_typed_errors():
    _spec_value, errors = plan_spec.normalize_spec({"goal": "g", "work_item_refs": ["ok", 1]})
    assert errors == ["work_item_refs[1]: must be a string"]
    _spec_value, errors = plan_spec.normalize_spec({"goal": "g", "work_item_refs": "ibl-1"})
    assert errors == ["work_item_refs: must be an array of strings"]


def test_plan_fingerprint_preserves_historical_json_wire():
    spec = {"goal": "goal", "work_item_refs": ["ibl-1"]}
    payload = {
        "goal": "goal", "plan": "plan", "spec": spec,
        "evidence_manifest_hash": "a" * 64, "constitutional": True,
    }
    expected = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    assert expected == "0b60598964435a945eb5a4f138e32566566b1f5c654e4969e73309c858a0524f"
    assert plan_spec.plan_fingerprint("goal", "plan", spec, "a" * 64, True) == expected


def test_plan_evidence_deny_paths_owns_runtime_boundaries(tmp_path, monkeypatch):
    from ouroboros import config as cfg

    data_root = tmp_path / "data"
    settings = data_root / "settings.json"
    monkeypatch.setattr(cfg, "DATA_DIR", data_root, raising=False)
    monkeypatch.setattr(cfg, "SETTINGS_PATH", settings, raising=False)
    denied = plan_evidence.evidence_deny_paths(SimpleNamespace(drive_root=data_root))
    assert str(data_root) in denied
    assert str(settings) in denied


def test_evolution_refs_validate_against_open_backlog(tmp_path):
    from ouroboros.improvement_backlog import append_backlog_items

    append_backlog_items(tmp_path, [{
        "id": "ibl-1", "summary": "one", "priority": "high",
        "kind": "bug", "category": "process", "source": "test",
    }])
    assert plan_work_items.validate_evolution_work_item_refs(tmp_path, ["ibl-1"]) is None
    assert "ibl-unknown" in plan_work_items.validate_evolution_work_item_refs(
        tmp_path, ["ibl-unknown"]
    )
    assert "not-a-ref" in plan_work_items.validate_evolution_work_item_refs(
        tmp_path, ["not-a-ref"]
    )


def test_current_exact_closed_wave_binding_and_replacement():
    state = {
        "current_attempt": {"fingerprint": "b" * 64},
        "waves": [
            {"request_fingerprint": "a" * 64, "closed": False, "work_item_refs": ["ibl-old"]},
            {"request_fingerprint": "b" * 64, "closed": True, "work_item_refs": []},
        ],
    }
    binding = plan_work_items.plan_work_item_binding(state)
    assert binding == {"plan_fingerprint": "b" * 64, "refs": []}

    state["current_attempt"] = {"fingerprint": "c" * 64}
    assert plan_work_items.plan_work_item_binding(state) is None

    state["current_attempt"] = {}
    assert plan_work_items.plan_work_item_binding(state) == {
        "plan_fingerprint": "b" * 64, "refs": []
    }


def test_transaction_binding_reader():
    tx = {"commit_intent": {"work_item_binding": {"plan_fingerprint": "f", "refs": ["ibl-1"]}}}
    assert plan_work_items.transaction_work_item_binding(tx) == {
        "plan_fingerprint": "f", "refs": ["ibl-1"]
    }
    assert plan_work_items.transaction_work_item_binding({"commit_intent": {}}) is None


def test_plan_task_refuses_unknown_evolution_refs_before_reviewers(tmp_path):
    from ouroboros.tools import plan_review

    ctx = SimpleNamespace(
        drive_root=tmp_path,
        task_metadata={"evolution_transaction": {"transaction_id": "tx"}},
    )
    result = plan_review._handle_plan_task(
        ctx,
        goal="close exact absorbed items",
        plan="bind first",
        spec={"goal": "close exact absorbed items", "work_item_refs": ["ibl-missing"]},
    )
    assert "ERROR: PLAN_SPEC_INVALID" in result
    assert "ibl-missing" in result


def test_exact_wave_artifact_preserves_work_item_refs(tmp_path):
    from ouroboros.task_results import write_task_result
    from ouroboros.tools.plan_review_artifacts import persist_wave, read_wave

    write_task_result(tmp_path, "task-1", {"status": "running"})
    fingerprint = "f" * 64
    wave = {
        "request_fingerprint": fingerprint,
        "cycle_index": 1,
        "aggregate": "GREEN",
        "closed": True,
        "spec": {"goal": "g", "work_item_refs": ["ibl-1"]},
        "findings": [],
        "dispositions": [],
    }
    ref = persist_wave(tmp_path, "task-1", wave)
    exact = read_wave(tmp_path, "task-1", ref)
    assert exact["spec"]["work_item_refs"] == ["ibl-1"]


def test_commit_intent_copies_only_current_closed_wave(tmp_path, monkeypatch):
    from ouroboros.tools import git_evolution

    captured = {}
    monkeypatch.setattr(
        "supervisor.evolution_lifecycle.update_evolution_transaction",
        lambda task_id, **updates: captured.update(updates) or True,
    )
    state = {
        "current_attempt": {"fingerprint": "f" * 64},
        "waves": [{
            "request_fingerprint": "f" * 64, "closed": True,
            "work_item_refs": ["ibl-1", "ibl-2"], "aggregate": "GREEN",
        }],
    }
    monkeypatch.setattr(
        "ouroboros.task_results.load_plan_review_state",
        lambda _root, _task_id: state,
    )
    ctx = SimpleNamespace(drive_path=tmp_path, task_id="task-1")
    git_evolution._record_evolution_commit_intent(
        ctx, {"task_id": "task-1", "transaction_id": "tx-1"},
        {"binding": {"tree_sha": "tree", "parents": ["parent"]}},
    )
    intent = captured["commit_intent"]
    assert intent["work_item_binding"] == {
        "plan_fingerprint": "f" * 64, "refs": ["ibl-1", "ibl-2"]
    }
    assert intent["tree_sha"] == "tree"

    captured.clear()
    state["current_attempt"] = {"fingerprint": "0" * 64}
    git_evolution._record_evolution_commit_intent(
        ctx, {"task_id": "task-1", "transaction_id": "tx-1"},
        {"binding": {"tree_sha": "tree", "parents": ["parent"]}},
    )
    assert "work_item_binding" not in captured["commit_intent"]
