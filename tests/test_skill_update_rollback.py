"""Hub update failures preserve the installed payload and its working lifecycle."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from ouroboros import extension_loader
from ouroboros.marketplace import install as installer, ouroboroshub as hub
from ouroboros.skill_loader import (
    SkillReviewState, load_skill, save_enabled, save_review_state, skill_state_dir,
)
from tests._extension_loader_shared import _write_ext_skill
from tests._extension_loader_shared import _clear_loader_state  # noqa: F401


async def _run_blocking(func, *args, **kwargs):
    kwargs.pop("log_label", None)
    return func(*args, **kwargs)


@pytest.fixture
def working_hub_skill(tmp_path, monkeypatch):
    drive = tmp_path / "data"
    bucket = drive / "skills" / "ouroboroshub"
    monkeypatch.setattr(hub, "get_ouroboroshub_skills_dir", lambda: bucket)
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(tmp_path / "empty"))
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
    target = _write_ext_skill(
        bucket, "demo", permissions=["tool"],
        plugin_body="def register(api):\n    api.register_tool('ping', lambda ctx: 'old', description='ping', schema={})\n",
    )
    (target / ".ouroboroshub.json").write_text(json.dumps({
        "schema_version": 1, "source": "ouroboroshub", "slug": "demo", "sanitized_name": "demo",
    }))
    loaded = load_skill(target, drive)
    save_review_state(drive, "demo", SkillReviewState(status="clean", content_hash=loaded.content_hash))
    save_enabled(drive, "demo", True)
    state = extension_loader.reconcile_extension("demo", drive, lambda: {})
    assert state["live_loaded"], state
    state_dir = skill_state_dir(drive, "demo")
    (state_dir / "grants.json").write_text(json.dumps({"content_hash": loaded.content_hash, "granted_keys": []}))
    (state_dir / "review_history.jsonl").write_text('{"old_review": true}\n')
    env = target / ".ouroboros_env"
    env.mkdir()
    (env / "old-environment.txt").write_text("restorable environment")
    originals = installer.snapshot_lifecycle_state(drive, "demo")

    def download(summary, raw_base, staging):
        del summary, raw_base
        _write_ext_skill(staging.parent, staging.name, permissions=["tool"],
                         plugin_body="def register(api):\n    raise RuntimeError('new plugin failed to load')\n")

    monkeypatch.setattr(hub, "_download_skill_files", download)
    monkeypatch.setattr(hub, "load_catalog", lambda: {
        "raw_base_url": "https://example.invalid", "skills": [{"slug": "demo", "version": "2.0.0"}],
    })
    return drive, target, originals


@pytest.mark.parametrize("failure", ["review", "deps", "load"])
def test_update_failure_restores_actual_live_extension(working_hub_skill, failure):
    drive, target, originals = working_hub_skill

    async def review(payload, name):
        current = load_skill(target, drive)
        status = "blockers" if failure == "review" else "clean"
        save_review_state(drive, name, SkillReviewState(status=status, content_hash=current.content_hash))
        state_dir = skill_state_dir(drive, name)
        (state_dir / "grants.json").write_text(json.dumps({"content_hash": current.content_hash, "granted_keys": []}))
        (state_dir / "accepted_rebuttals.json").write_text('{"new": true}')
        with (state_dir / "review_history.jsonl").open("a") as stream:
            stream.write('{"new_review": true}\n')
        deps = "failed" if failure == "deps" else "not_required"
        payload.update(review_status=status, deps_status=deps)
        return status, "dependency failure" if failure == "deps" else "", deps

    result = asyncio.run(hub.run_hub_update(
        "demo", drive_root=drive, progress=SimpleNamespace(set=lambda value: None),
        run_blocking=_run_blocking, apply_review_and_deps=review,
    ))
    assert result["ok"] is False
    assert result["rolled_back"] is True, result
    assert installer.snapshot_lifecycle_state(drive, "demo") == originals
    assert (target / ".ouroboros_env" / "old-environment.txt").read_text() == "restorable environment"
    assert extension_loader.is_extension_live("demo", drive)
    tool = extension_loader.get_tool(extension_loader.extension_surface_name("demo", "ping"))
    assert tool["handler"](SimpleNamespace()) == "old"
    assert "new_review" in (skill_state_dir(drive, "demo") / "review_history.jsonl").read_text()


def test_partial_state_restore_never_deletes_restored_payload(tmp_path, monkeypatch):
    drive = tmp_path / "data"
    target = drive / "skills" / "ouroboroshub" / "demo"
    target.mkdir(parents=True)
    (target / "payload.txt").write_text("old")
    snapshot = installer.snapshot_payload_state(drive, "demo", target)
    (target / "payload.txt").write_text("new")
    restore = installer.restore_lifecycle_state
    monkeypatch.setattr(installer, "restore_lifecycle_state", lambda *args: ["controlled state failure"])
    with pytest.raises(OSError, match="controlled state failure"):
        installer.restore_payload_state(snapshot)
    assert (target / "payload.txt").read_text() == "old"
    monkeypatch.setattr(installer, "restore_lifecycle_state", restore)
    installer.restore_payload_state(snapshot)
    assert (target / "payload.txt").read_text() == "old"


def test_missing_snapshot_refuses_before_removing_current_payload(tmp_path):
    drive = tmp_path / "data"
    target = drive / "skills" / "ouroboroshub" / "demo"
    target.mkdir(parents=True)
    (target / "payload.txt").write_text("keep")
    snapshot = installer.PayloadRollbackSnapshot(drive, "demo", target, tmp_path / "missing")
    with pytest.raises(OSError, match="rollback payload is missing"):
        installer.restore_payload_state(snapshot)
    assert (target / "payload.txt").read_text() == "keep"


def test_update_rollback_preserves_independent_disable(working_hub_skill):
    drive, target, originals = working_hub_skill

    async def review(payload, name):
        save_enabled(drive, name, False, actor="owner_ui")
        return "blockers", "controlled review refusal", "not_required"

    result = asyncio.run(hub.run_hub_update(
        "demo", drive_root=drive, progress=SimpleNamespace(set=lambda value: None),
        run_blocking=_run_blocking, apply_review_and_deps=review,
    ))
    assert result["rolled_back"] is True
    assert installer.snapshot_lifecycle_state(drive, "demo") == originals
    assert load_skill(target, drive).enabled is False
    assert extension_loader.is_extension_live("demo", drive) is False


def test_update_partial_restore_is_reported_once(working_hub_skill, monkeypatch):
    drive, target, _originals = working_hub_skill
    calls = []

    async def review(payload, name):
        return "blockers", "controlled review refusal", "not_required"

    def fail_state_restore(*args):
        calls.append(args)
        return ["controlled state failure"]

    monkeypatch.setattr(installer, "restore_lifecycle_state", fail_state_restore)
    result = asyncio.run(hub.run_hub_update(
        "demo", drive_root=drive, progress=SimpleNamespace(set=lambda value: None),
        run_blocking=_run_blocking, apply_review_and_deps=review,
    ))
    assert result["ok"] is False
    assert result["rolled_back"] is False
    assert "controlled state failure" in result["rollback_errors"][0]
    assert len(calls) == 1
    assert "'old'" in (target / "plugin.py").read_text()


def test_failed_unload_never_starts_replacement(working_hub_skill, monkeypatch):
    drive, target, originals = working_hub_skill
    previous = (target / "plugin.py").read_bytes()
    downloads = []

    def fail_unload(name):
        raise OSError("controlled unload failure")

    def unexpected_install(*args, **kwargs):
        downloads.append(args)
        raise AssertionError("replacement must not follow failed unload")

    with monkeypatch.context() as scoped:
        scoped.setattr(extension_loader, "unload_extension", fail_unload)
        scoped.setattr(hub, "install", unexpected_install)
        result = asyncio.run(hub.run_hub_update(
            "demo", drive_root=drive, progress=SimpleNamespace(set=lambda value: None),
            run_blocking=_run_blocking, apply_review_and_deps=None,
        ))
    assert result["ok"] is False and "controlled unload failure" in result["error"], result
    assert not downloads
    assert (target / "plugin.py").read_bytes() == previous
    assert installer.snapshot_lifecycle_state(drive, "demo") == originals
