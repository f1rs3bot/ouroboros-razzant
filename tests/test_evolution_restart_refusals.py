"""The evolution-restart claim verdict: every refusal is precise and durable.

Regression context (durable evidence, campaign b6db45d9 cycle #1): an
agent-requested restart was refused with "the live checkout no longer matches the
exact reviewed evolution commit" while HEAD in fact equalled the reviewed commit
— the cause was a CONCURRENT task's dirty worktree. The refusal wrote no durable
row, so the absorption boundary's real reason was unreconstructible and the cycle
sat at ``waiting_for_restart`` for ~10.5 hours until the owner pressed Restart
(whose own row, ``reset_unsynced_rescued_then_reset`` with dirty_count=16, names
the cause the agent path never reported).

These pins hold the two properties that failure broke: the reason is
DISTINGUISHED (dirty worktree vs moved HEAD vs unreadable state vs missing
receipt), and every refusal of this boundary leaves exactly ONE durable
``evolution_restart_refused`` row naming the surface that refused.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
from types import SimpleNamespace

import pytest

from tests._evolution_state_shared import _active_transaction


def _scratch_repo(tmp_path: pathlib.Path):
    repo = tmp_path / "repo"
    repo.mkdir()

    def run(*args: str) -> None:
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)

    run("init")
    run("config", "user.name", "Test")
    run("config", "user.email", "test@example.com")
    (repo / "file.txt").write_text("reviewed\n", encoding="utf-8")
    run("add", ".")
    run("commit", "-m", "reviewed")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True,
    ).stdout.strip()
    return repo, head


@pytest.fixture
def claim_env(tmp_path):
    """A scratch git repo + drive with one recorded reviewed evolution commit.

    ``git_ops`` is rebound for the test and restored afterwards: the verdict
    reads HEAD and the worktree facts through that module on purpose, so this is
    also the pin that the restarted tree and the classified tree are one root.
    """
    from supervisor import git_ops

    repo, head = _scratch_repo(tmp_path)
    drive = tmp_path / "drive"
    (drive / "logs").mkdir(parents=True, exist_ok=True)
    saved = {name: getattr(git_ops, name) for name in ("REPO_DIR", "DRIVE_ROOT", "REMOTE_URL")}
    git_ops.init(repo, drive, "")
    campaign, tx = _active_transaction(drive)
    yield SimpleNamespace(repo=repo, drive=drive, head=head, campaign=campaign, tx=tx)
    for name, value in saved.items():
        setattr(git_ops, name, value)


def _refusal_rows(env) -> list:
    path = env.drive / "logs" / "supervisor.jsonl"
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [row for row in rows if row.get("type") == "evolution_restart_refused"]


def _assess(env, claim):
    from ouroboros.server_restart import assess_evolution_restart_claim

    return assess_evolution_restart_claim(
        drive_root=env.drive, claim=claim, restart_reason="evolution restart",
    )


def _arm(env, sha):
    """Record the reviewed commit and return the claim a restart would carry.

    Authority binds the claim to the RECORDED sha, so the two mismatch shapes are
    distinguishable on purpose: recording ``head`` exercises the clean/dirty
    cases, recording a foreign sha exercises a genuinely moved checkout.
    """
    from supervisor import evolution_lifecycle

    recorded = evolution_lifecycle.record_evolution_commit(
        env.campaign["id"], env.tx["transaction_id"], env.tx["task_id"], commit_sha=sha,
    )
    assert recorded["ok"] is True, recorded
    return {
        "campaign_id": env.campaign["id"],
        "transaction_id": env.tx["transaction_id"],
        "task_id": env.tx["task_id"],
        "commit_sha": sha,
    }


def test_clean_matching_worktree_authorizes_the_restart(claim_env):
    verdict = _assess(claim_env, _arm(claim_env, claim_env.head))

    assert verdict["ok"] is True
    assert verdict["reason"] == ""
    assert _refusal_rows(claim_env) == []


def test_dirty_worktree_with_matching_head_names_dirt_not_a_moved_checkout(claim_env):
    """The exact misdiagnosis that cost ~10.5 hours: HEAD matched, the tree was dirty."""
    (claim_env.repo / "file.txt").write_text("another task's in-flight work\n", encoding="utf-8")

    verdict = _assess(claim_env, _arm(claim_env, claim_env.head))

    assert verdict["ok"] is False
    assert verdict["reason"] == "unsynced_tree"
    assert verdict["observed_head"] == claim_env.head
    assert verdict["dirty_count"] > 0
    assert verdict["dirty_preview"]
    # The wrong cause must never be reported again for a matching HEAD.
    assert "no longer matches" not in verdict["message"]
    assert "unsynced" in verdict["message"]

    rows = _refusal_rows(claim_env)
    assert len(rows) == 1
    assert rows[0]["reason"] == "unsynced_tree"
    assert rows[0]["source"] == "claim"
    assert rows[0]["dirty_count"] == verdict["dirty_count"]
    assert rows[0]["expected_sha"] == claim_env.head


def test_moved_head_is_refused_with_a_distinct_reason(claim_env):
    # Authority binds this foreign sha, while the live checkout stands at head.
    verdict = _assess(claim_env, _arm(claim_env, "2" * 40))

    assert verdict["ok"] is False
    assert verdict["reason"] == "head_mismatch"
    assert verdict["reason"] != "unsynced_tree"
    assert "no longer matches" in verdict["message"]
    assert verdict["observed_head"] == claim_env.head
    rows = _refusal_rows(claim_env)
    assert len(rows) == 1 and rows[0]["reason"] == "head_mismatch"


def test_absent_receipt_is_a_recorded_refusal(claim_env):
    """A receipt that vanished during the drain must refuse AND be recorded."""
    verdict = _assess(claim_env, {})

    assert verdict["ok"] is False
    assert verdict["reason"] == "receipt_missing"
    assert "receipt is missing" in verdict["message"]
    rows = _refusal_rows(claim_env)
    assert len(rows) == 1 and rows[0]["reason"] == "receipt_missing"


def test_authority_change_is_a_recorded_refusal(claim_env):
    from supervisor import state

    # Arm while the campaign is healthy: the authority is re-checked by the
    # verdict, and that is what this test exercises — not the record step.
    claim = _arm(claim_env, claim_env.head)
    live = state.load_state()
    live.update({"evolution_owner_stopped": True, "evolution_mode_enabled": False})
    state.save_state(live)

    verdict = _assess(claim_env, claim)

    assert verdict["ok"] is False
    assert verdict["reason"] == "authority_changed"
    assert "owner_stopped" in verdict["message"]
    rows = _refusal_rows(claim_env)
    assert len(rows) == 1 and rows[0]["reason"] == "authority_changed"


def test_unreadable_repository_state_fails_closed(claim_env, monkeypatch):
    from supervisor import git_ops

    monkeypatch.setattr(
        git_ops,
        "_collect_repo_sync_state",
        lambda: {
            "current_branch": "ouroboros",
            "dirty_lines": [],
            "unpushed_lines": [],
            "warnings": ["status_error:git status exited 128 without stderr"],
        },
    )

    verdict = _assess(claim_env, _arm(claim_env, claim_env.head))

    assert verdict["ok"] is False
    assert verdict["reason"] == "repo_state_unreadable"
    rows = _refusal_rows(claim_env)
    assert len(rows) == 1 and rows[0]["reason"] == "repo_state_unreadable"


def test_non_evolution_restart_refusal_is_not_recorded(tmp_path):
    """The universal 'every refusal is recorded' is scoped to the evolution path.

    The downstream not-ok branch is shared with ordinary restarts, so the guard
    (not a caller's discipline) is what keeps this row type the absorption
    boundary's own record.
    """
    from ouroboros.server_restart import record_restart_refusal

    drive = tmp_path / "drive"
    (drive / "logs").mkdir(parents=True, exist_ok=True)

    record_restart_refusal(drive, source="downstream", reason="checkout_gate_refused",
                           detail="ordinary restart", evolution_restart=False)

    assert not (drive / "logs" / "supervisor.jsonl").exists()


def test_downstream_gate_refusal_is_recorded_once(claim_env, monkeypatch):
    """The second refusal point is recorded too — an absorption never defers silently."""
    import server

    claim = _arm(claim_env, claim_env.head)
    marker = claim_env.drive / "state" / "pending_restart_verify.json"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        json.dumps({"reason": "evolution restart", "evolution_claim": claim}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        server, "_safe_restart_serialized",
        lambda *a, **k: (False, "Managed update merge is still being resolved; restart was deferred."),
    )
    messages = []
    ctx = SimpleNamespace(
        DRIVE_ROOT=claim_env.drive,
        load_state=lambda: {"owner_chat_id": 1},
        save_state=lambda state: None,
        safe_restart=lambda **kwargs: (True, "ok"),
        send_with_budget=lambda *a: messages.append(a),
    )

    server._perform_supervisor_restart(
        ctx, restart_reason="evolution restart", evolution_restart=True,
    )

    rows = _refusal_rows(claim_env)
    assert len(rows) == 1
    assert rows[0]["source"] == "downstream"
    assert rows[0]["reason"] == "checkout_gate_refused"
    assert "restart was deferred" in rows[0]["detail"]


def test_server_branch_refuses_a_dirty_claim_without_calling_safe_restart(claim_env, monkeypatch):
    import server

    (claim_env.repo / "file.txt").write_text("dirty\n", encoding="utf-8")
    marker = claim_env.drive / "state" / "pending_restart_verify.json"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        json.dumps({
            "reason": "evolution restart",
            "evolution_claim": _arm(claim_env, claim_env.head),
        }),
        encoding="utf-8",
    )
    restarted = []
    messages = []
    monkeypatch.setattr(server, "_safe_restart_serialized",
                        lambda *a, **k: restarted.append(k) or (True, "ok"))
    ctx = SimpleNamespace(
        DRIVE_ROOT=claim_env.drive,
        load_state=lambda: {"owner_chat_id": 1},
        safe_restart=lambda **k: (True, "ok"),
        send_with_budget=lambda *a: messages.append(a),
    )

    server._perform_supervisor_restart(
        ctx, restart_reason="evolution restart", evolution_restart=True,
    )

    assert restarted == []
    assert "unsynced" in messages[0][1]
    rows = _refusal_rows(claim_env)
    assert len(rows) == 1 and rows[0]["reason"] == "unsynced_tree"


def test_non_evolution_restart_still_proceeds_past_the_verdict(claim_env, monkeypatch):
    """evolution_restart=False must not consult the verdict at all."""
    import server

    calls = []
    monkeypatch.setattr(server, "_safe_restart_serialized",
                        lambda *a, **k: calls.append(k) or (True, "ok"))
    monkeypatch.setattr(server, "_request_restart_exit", lambda: calls.append("exit"))
    ctx = SimpleNamespace(
        DRIVE_ROOT=claim_env.drive,
        load_state=lambda: {},
        safe_restart=lambda **k: (True, "ok"),
        kill_workers=lambda **k: None,
        save_state=lambda state: None,
        persist_queue_snapshot=lambda **k: None,
    )

    server._perform_supervisor_restart(
        ctx, restart_reason="agent_requested_restart", evolution_restart=False,
    )

    assert calls  # the restart proceeded
    assert _refusal_rows(claim_env) == []


def test_refusal_reasons_are_the_documented_closed_set(claim_env):
    from ouroboros.server_restart import (
        EVOLUTION_RESTART_REFUSAL_REASONS,
        RESTART_GATE_REFUSAL_REASON,
    )

    assert EVOLUTION_RESTART_REFUSAL_REASONS == (
        "receipt_missing",
        "authority_changed",
        "head_mismatch",
        "repo_state_unreadable",
        "unsynced_tree",
    )
    assert RESTART_GATE_REFUSAL_REASON not in EVOLUTION_RESTART_REFUSAL_REASONS
