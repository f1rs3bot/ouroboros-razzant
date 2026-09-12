"""Restart-adjacent helpers the composition root calls at shutdown time.

The live-task census the restart drain consults, the teardown arguments that
finalize interrupted tasks with an honest reason, the managed-update guard on
preserving queued work, the checkout/update serialization gate, the owned-work
stop of the owner's manual Restart, the planned restart's engine-pin daemon stop,
the evolution-restart claim verdict with its durable refusal record, and the
event bus shutdown. The restart
transaction itself — the deferred drain record and the performer that raises
the exit signal — stays in ``server.py`` for now: the upstream delegation
train coupled it to the composition root through the planned-handoff
transaction id (see docs/v7next/LEDGER_CORRECTIONS.md, D11).
"""

from __future__ import annotations

import pathlib
import time
from typing import Any, Dict, Optional

from ouroboros.server_process import DATA_DIR, _owner_restart_requested, _restart_requested, log


def _owned_live_task_ids(ctx: Any) -> list:
    """Every id this generation's cancel intent can address: pooled tasks,
    in-process direct/ephemeral activities and running post-task synthesis."""
    from ouroboros.post_task_checkpoint import POST_TASK_SYNTHESIS_INFLIGHT, POST_TASK_SYNTHESIS_LOCK
    from supervisor.active_activity import get_direct_activity_registry

    task_ids = list(dict(ctx.RUNNING or {}))
    task_ids.extend(str(row.get("activity_id") or "") for row in get_direct_activity_registry().snapshot())
    root = str(pathlib.Path(DATA_DIR).resolve(strict=False))
    with POST_TASK_SYNTHESIS_LOCK:
        task_ids.extend(task_id for (path, task_id) in POST_TASK_SYNTHESIS_INFLIGHT if path == root)
    return [tid for tid in dict.fromkeys(task_ids) if tid]


def _stop_owned_work(ctx: Any) -> None:
    """The owner's manual Restart: stop what this generation owns, then let it re-exec.

    Runs AFTER the checkout gate and the durable no-resume flags, so nothing
    here can veto: an unconfirmed step is a critical diagnostic with custody
    retained, and the next generation's startup custody sweep reconciles the
    remainder (owner restart is a no-resume cause; nothing is adopted). In
    order: one durable cancel intent per owned live id, ``kill_workers`` with
    Panic's ``reconcile_delegate_custody=False``, delegated-run cancellation
    through the public owner-gone seam over the attach-only gateway, and the
    attested owned-daemon stop exactly as Panic makes it. Between the cancel
    intents and that stop nothing may call ``ensure_owned_gateway`` — it would
    start a dead daemon — which is what the two flags above guarantee.
    """
    from ouroboros.cancel_intents import request_cancel
    from ouroboros.claudexor_daemon import read_owned_gateway
    from ouroboros.delegate_custody import reconcile_orphaned_runs

    for task_id in _owned_live_task_ids(ctx):
        try:
            request_cancel(DATA_DIR, task_id, reason="Owner restart", source="owner_restart",
                           requested_by="owner", requested_stop_policy="immediate",
                           allow_settled_target=True)
        except Exception:
            log.warning("Owner restart: cancel intent for %s was not recorded", task_id, exc_info=True)
    try:
        confirmed = ctx.kill_workers(
            force=True, terminal_status="cancelled",
            result_reason="Owner restart stopped this task before process restart.",
            reconcile_delegate_custody=False, **_managed_update_pending_kwargs(),
        )
    except Exception:
        log.critical("Owner restart: worker shutdown raised; the restart proceeds and the next "
                     "generation reconciles the remainder", exc_info=True)
    else:
        if confirmed is False:
            log.critical("Owner restart: worker shutdown is unconfirmed; the restart proceeds and "
                         "the next generation reconciles the remainder")
    try:
        reconcile_orphaned_runs(DATA_DIR, running_task_ids=set(), gateway_factory=read_owned_gateway)
    except Exception:
        log.warning("Owner restart: delegated-run cancellation did not complete; custody retained",
                    exc_info=True)
    _stop_owned_daemon("Owner restart")


def _stop_owned_daemon(label: str) -> None:
    """The attested owned-daemon stop a restart makes; never a veto.

    An unconfirmed stop was already disclosed by ``stop_outcome`` (critical log
    plus the ``process_stop_unconfirmed`` supervisor row, custody retained); a
    raising stop gets the same row here, as Panic records it. Either way the
    restart proceeds and the next generation attaches to a still-live daemon.
    """
    from ouroboros.claudexor_daemon import CUSTODY_PURPOSE, get_owned_daemon
    from ouroboros.utils import append_jsonl, utc_now_iso

    try:
        outcome = get_owned_daemon().stop_outcome()
    except Exception as exc:
        log.critical("%s: owned Claudexor stop raised %s; custody is unconfirmed", label, type(exc).__name__)
        try:
            append_jsonl(pathlib.Path(DATA_DIR) / "logs" / "supervisor.jsonl", {
                "ts": utc_now_iso(), "type": "process_stop_unconfirmed",
                "purpose": CUSTODY_PURPOSE, "reason": f"stop raised {type(exc).__name__}",
            })
        except Exception:
            log.critical("%s: failed to record unconfirmed daemon stop", label)
    else:
        if outcome == "unconfirmed":  # stop_outcome already disclosed the remainder
            log.critical("%s: owned Claudexor stop unconfirmed; custody retained, the "
                         "restart proceeds and the next generation attaches to the live daemon", label)


def _stop_owned_daemon_for_new_pin() -> None:
    """A planned restart ends the owned daemon only when the landed checkout pins another engine.

    Runs in the server lifespan teardown of every requested restart (planned
    self-restart, managed update or rollback; the owner's manual Restart has
    already stopped the daemon, so nothing answers). It compares the pin the
    next generation selects — ``load_runtime_pin`` reads the checkout that
    already landed, not this process's cached pin — with the serving engine's
    handshake through the attach-only ``read_owned_gateway``. An unprovisioned
    home, an unpublished or unreadable pin and an unreachable, foreign or
    stopped daemon leave the existing handoff untouched: the next generation
    attaches. A different version or build SHA makes the same attested stop the
    manual Restart makes, so the next generation spawns the pinned engine; the
    delegated runs that daemon served end with it, and the next generation's
    startup sweep and resumed parents close them as absent (no invented spend).
    """
    from ouroboros.claudexor_daemon import owned_daemon_provisioned, read_owned_gateway
    from ouroboros.claudexor_runtime import load_runtime_pin

    if not owned_daemon_provisioned():
        return
    try:
        pin = load_runtime_pin()
        if pin is None:
            return
        with read_owned_gateway() as gateway:
            serving = (gateway.engine_version, gateway.engine_build_sha)
    except Exception as exc:
        log.info("Planned restart keeps the owned Claudexor daemon: the engine pin comparison is "
                 "unavailable (%s)", exc)
        return
    if serving == (pin.version, pin.build_sha):
        return
    log.warning("Planned restart stops the owned Claudexor daemon: engine %s (%s) is serving while the "
                "checkout pins %s (%s); the next generation starts the pinned engine and the runs "
                "in flight end with this one", serving[0] or "unknown", serving[1][:12] or "unknown",
                pin.version, pin.build_sha[:12])
    _stop_owned_daemon("Planned restart")


def _live_running_task_ids(ctx: Any) -> list:
    """Pooled tasks with a fresh heartbeat and registered native executions.

    Heartbeat staleness belongs to the generic supervisor queue, not to the
    planning-scout wait policy.  The latter intentionally waits until terminal
    state or its shared cutoff even when a scout heartbeat is stale.
    """
    from supervisor.queue import HEARTBEAT_STALE_SEC

    now = time.time()
    live = []
    for tid, meta in dict(ctx.RUNNING or {}).items():
        if not isinstance(meta, dict):
            continue
        try:
            hb = float(meta.get("last_heartbeat_at") or 0.0)
        except (TypeError, ValueError):
            hb = 0.0
        if hb and (now - hb) < HEARTBEAT_STALE_SEC:
            live.append(str(tid))
    from supervisor.active_activity import get_direct_activity_registry

    return list(dict.fromkeys(live + [
        row["activity_id"] for row in get_direct_activity_registry().snapshot()
    ]))


def _managed_update_pending_kwargs() -> dict:
    """Preserve queued work while a durable tx or its pre-tx quiesce owns restart."""
    try:
        from ouroboros.delegate_recovery import has_planned_restart_handoffs

        if (
            has_planned_restart_handoffs(DATA_DIR)
            and _restart_requested.is_set()
            and not _owner_restart_requested.is_set()
        ):
            return {"preserve_pending": True}
        from supervisor.update_merge import active_update_tx

        if active_update_tx():
            return {"preserve_pending": True}
        from supervisor.workers import repo_writer_admission_closed, worker_pool_admission_state

        gate = repo_writer_admission_closed()
        disabled = str(worker_pool_admission_state().get("disabled_reason") or "")
        if gate.startswith("managed_update:") or disabled == "managed_update":
            return {"preserve_pending": True}
        return {}
    except Exception:
        return {"preserve_pending": True}


def _safe_restart_serialized(safe_restart_fn, *, reason: str, unsynced_policy: str):
    """Serialize checkout/reset with update apply; only a landed update may restart."""
    from supervisor import git_ops
    from supervisor.update_merge import (
        acquire_update_lock,
        read_update_tx_strict,
        release_update_lock,
    )

    try:
        lock_fh = acquire_update_lock()
    except RuntimeError:
        return False, "Managed update is changing the checkout; restart was deferred."
    try:
        status, tx = read_update_tx_strict()
        if status == "corrupt":
            return False, "Managed update state is unreadable; restart was deferred."
        if status == "future":
            return False, "Managed update state was recorded by a newer version; restart was deferred."
        if status == "absent" and not git_ops._clear_update_intent():
            return False, (
                "An update intent marker with no update transaction could not be removed; "
                "restart was deferred rather than applying an orphaned update."
            )
        allowed_phases = {"pending_boot_smoke", "applying_replace"}
        if status == "valid" and str(tx.get("phase") or "") not in allowed_phases:
            return False, "Managed update merge is still being resolved; restart was deferred."
        return safe_restart_fn(reason=reason, unsynced_policy=unsynced_policy)
    finally:
        release_update_lock(lock_fh)


# Closed reason set for a refusal of the evolution absorption boundary's claim
# verdict. Every one is fail-closed: the reviewed commit is not proven live, so
# the restart must not proceed.
EVOLUTION_RESTART_REFUSAL_REASONS = (
    "receipt_missing",
    "authority_changed",
    "head_mismatch",
    "repo_state_unreadable",
    "unsynced_tree",
)
# The downstream checkout/update gate's own refusal, recorded under the same row
# type so a deferred absorption is never chat-only whichever surface refused it.
RESTART_GATE_REFUSAL_REASON = "checkout_gate_refused"


def record_restart_refusal(
    drive_root: Any,
    *,
    source: str,
    reason: str,
    detail: str = "",
    evolution_restart: bool = True,
    facts: Optional[Dict[str, Any]] = None,
) -> None:
    """One durable row per refused restart of the evolution absorption boundary.

    Guarded by ``evolution_restart``: an ordinary restart refusal records
    nothing, so this row type stays the evolution boundary's own record. Never
    raises — a refusal that cannot be recorded must still refuse.
    """
    if not evolution_restart:
        return
    try:
        from ouroboros.utils import append_jsonl, utc_now_iso

        facts = facts or {}
        append_jsonl(
            pathlib.Path(drive_root) / "logs" / "supervisor.jsonl",
            {
                "ts": utc_now_iso(),
                "type": "evolution_restart_refused",
                "source": source,
                "reason": reason,
                "detail": str(detail or ""),
                "restart_reason": str(facts.get("restart_reason") or ""),
                "expected_sha": str(facts.get("expected_sha") or ""),
                "observed_head": str(facts.get("observed_head") or ""),
                "dirty_count": int(facts.get("dirty_count") or 0),
                "dirty_preview": list(facts.get("dirty_preview") or []),
                "unpushed_count": int(facts.get("unpushed_count") or 0),
                "warnings": list(facts.get("warnings") or []),
            },
        )
    except Exception:
        log.warning("Failed to record an evolution-restart refusal", exc_info=True)


def _unsynced_bits(state: Dict[str, Any]) -> str:
    """Detail vocabulary for a proven-unsynced worktree.

    A DECLARED STRICT SUBSET of the reset admission gate's classifier: a merge
    in progress or an unreadable MERGE_HEAD is deliberately not re-derived here.
    The gate owns that classification downstream (its module is a protected
    release invariant this path may not edit) and ITS refusal is recorded too,
    so no refusal escapes the audit and no second spelling of "unsynced" is
    introduced.
    """
    bits = []
    if state.get("dirty_lines"):
        bits.append(f"dirty={len(state.get('dirty_lines') or [])}")
    if any(str(w).startswith("status_error:") for w in (state.get("warnings") or [])):
        bits.append("status_unreadable")
    return ", ".join(bits) or "unsynced"


def assess_evolution_restart_claim(
    *, drive_root: Any, claim: Any, restart_reason: str = "",
) -> Dict[str, Any]:
    """The one owner of the evolution-restart claim verdict.

    Consulted whenever a restart carries ``evolution_restart`` — INCLUDING when
    the claim is absent, which is itself a refusal: a restart whose exact receipt
    vanished during the drain must never proceed. Returns ``{"ok", "reason",
    "message"}`` plus the observed facts, and records exactly one durable
    ``evolution_restart_refused`` row (``source="claim"``) per refusal.

    HEAD and the worktree facts are both read through ``supervisor.git_ops``,
    i.e. against the same configured repo root the checkout/reset machinery
    moves, so the tree this verdict classifies IS the tree the restart would act
    on. A dirty worktree is refused rather than reset: resetting would discard
    another actor's uncommitted work, and the owner's own Restart is the
    consented path that rescues and resets it.
    """
    claim = claim if isinstance(claim, dict) else {}
    expected_sha = str(claim.get("commit_sha") or "")
    facts: Dict[str, Any] = {
        "expected_sha": expected_sha,
        "restart_reason": str(restart_reason or ""),
    }

    def _refuse(reason: str, message: str) -> Dict[str, Any]:
        record_restart_refusal(
            drive_root, source="claim", reason=reason, detail=message, facts=facts,
        )
        return {"ok": False, "reason": reason, "message": message, **facts}

    if not claim:
        return _refuse(
            "receipt_missing",
            "🧬 Restart cancelled: the exact evolution restart receipt is missing.",
        )
    from supervisor.evolution_lifecycle import check_evolution_authority

    authority = check_evolution_authority(
        str(claim.get("campaign_id") or ""),
        str(claim.get("transaction_id") or ""),
        str(claim.get("task_id") or ""),
        commit_sha=expected_sha,
    )
    if not authority.get("ok"):
        return _refuse(
            "authority_changed",
            "🧬 Restart cancelled: evolution authority changed "
            f"({authority.get('reason') or 'unknown'}).",
        )
    from supervisor import git_ops

    unsynced = git_ops._collect_repo_sync_state()
    facts.update({
        "dirty_count": len(unsynced.get("dirty_lines") or []),
        "dirty_preview": list(unsynced.get("dirty_lines") or [])[:20],
        "unpushed_count": len(unsynced.get("unpushed_lines") or []),
        "warnings": list(unsynced.get("warnings") or []),
    })
    if any(str(w).startswith("status_error:") for w in facts["warnings"]):
        return _refuse(
            "repo_state_unreadable",
            "🧬 Restart cancelled: the repository state could not be read, so the "
            "reviewed checkout could not be proven.",
        )
    rc, head_out, _head_err = git_ops.rescue_git_capture(["git", "rev-parse", "HEAD"])
    observed = str(head_out or "").strip() if rc == 0 else ""
    facts["observed_head"] = observed
    if not expected_sha or rc != 0 or observed != expected_sha:
        return _refuse(
            "head_mismatch",
            "🧬 Restart cancelled: the live checkout no longer matches the exact "
            f"reviewed evolution commit (HEAD {observed or 'unreadable'} != "
            f"{expected_sha or 'missing'}).",
        )
    if facts["dirty_count"]:
        preview = "; ".join(str(row) for row in facts["dirty_preview"])
        return _refuse(
            "unsynced_tree",
            "🧬 Restart cancelled: the worktree is unsynced "
            f"({_unsynced_bits(unsynced)}) and an agent-initiated restart does not "
            "reset another actor's uncommitted work"
            + (f": {preview}" if preview else "")
            + ". Press Restart to rescue and reset the tree (that path absorbs), "
            "or wait for the unsynced work to land and retry.",
        )
    return {"ok": True, "reason": "", "message": "", **facts}


def _shutdown_task_cleanup_args(restart_requested: bool) -> tuple[str, str]:
    """Return ``(terminal_status, result_reason)`` for tasks torn down by a
    graceful server shutdown.

    A graceful shutdown — a requested restart (exit 42) or an external
    stop/restart signal (SIGTERM/SIGINT) — is not a worker crash storm, so a
    still-running task is finalized as ``cancelled`` with an honest reason
    instead of the default crash-storm text the supervisor uses for real
    worker deaths.
    """
    if restart_requested:
        reason = (
            "Server restarted before this task finished; the task was "
            "interrupted by the restart, not a worker crash."
        )
    else:
        reason = (
            "Server shut down (external stop/restart signal) before this task "
            "finished; the task was interrupted, not a worker crash."
        )
    return "cancelled", reason


def _shutdown_supervisor_event_bus() -> None:
    try:
        from supervisor.workers import shutdown_event_q

        shutdown_event_q()
    except Exception:
        pass
