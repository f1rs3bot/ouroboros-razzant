#!/usr/bin/env python3
"""External OSWorld step-loop adapter backed by local Ouroboros.

Unlike ``run_installed_agent.py``, this runner does not install Ouroboros inside
the VM. It keeps the official OSWorld rhythm:

    observe VM -> ask Ouroboros for next action(s) -> env.step(action) -> repeat

Every action returned by Ouroboros is passed through ``env.step(...)`` and is
therefore visible in OSWorld's normal trajectory/action history. Screenshots are
saved under ``data/uploads`` so Ouroboros can inspect them with ``vlm_query``.

Alignment target: OSWorld 2.0 (``ALIGNED_UPSTREAM`` below). The runner mirrors
the official per-example artifact contract consumed by upstream
``show_result.py`` / ``lib_run_single.py``:

    <result_dir>/<action_space>/<observation_type>/<model>/<domain>/<example_id>/
        traj.jsonl          # official per-step rows (step_num, action, response,
                            # reward, done, info, screenshot_file, ...)
        step_<n>_<ts>.png   # post-action screenshot per step (official naming)
        result.txt          # final env.evaluate() score (scoring authority)
        result.json         # full dict result when evaluate() returns one

Not implemented (be honest when comparing to official 2.0 numbers): inline
checkpoint evaluations (``--checkpoint_eval_mode inline --checkpoint_steps
150,300``), multi-phase tasks, the human-in-the-loop user simulator
(``ASK_USER`` rows), ``recording.mp4``, and cloud providers (aws/azure/gcp).
"""

from __future__ import annotations

import argparse
import base64
import datetime as _dt
import json
import os
import re
import shutil
import subprocess
import sys
import time
import types
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from devtools.benchmarks.common.manifests import (
    BenchmarkAdmissionRefused,
    RuntimeAttestationRefused,
    admit_benchmark_run,
    finalize_run_manifest,
    runtime_attestation,
    write_json,
)
from devtools.benchmarks.common.model_slots import (
    runtime_actor_snapshot,
    single_model_subagents_setting,
)
from devtools.benchmarks.common.result_index import append_result_index, task_result_row
from devtools.benchmarks.common.run_roots import (
    assert_outside_repo,
    ensure_outside_repo,
    repo_root_from_devtools,
)
_REPO_ROOT = Path(__file__).resolve().parents[3]
_WORKSPACE_ROOT = _REPO_ROOT.parent

DEFAULT_OSWORLD_ROOT = os.environ.get("OSWORLD_ROOT", str(_WORKSPACE_ROOT / "OSWorld"))
DEFAULT_VM = os.environ.get("OSWORLD_VM", str(Path(DEFAULT_OSWORLD_ROOT) / "vmware_vm_data" / "Ubuntu0" / "Ubuntu0.vmx"))
DEFAULT_TASK = "evaluation_examples/examples/os/f9be0997-4b7c-45c5-b05c-4612b44a6118.json"
DEFAULT_REPO = str(_REPO_ROOT)
DEFAULT_DATA = os.environ.get("OUROBOROS_OSWORLD_DATA_DIR", str(_WORKSPACE_ROOT / "bench_runs" / "osworld_data"))
DEFAULT_SETTINGS = os.environ.get("OUROBOROS_SETTINGS_PATH", str(_WORKSPACE_ROOT / "data" / "settings.json"))
DEFAULT_OUROBOROS_BIN = os.environ.get("OUROBOROS_BIN", str(_REPO_ROOT / ".venv" / "bin" / "ouroboros"))
VMWARE_FUSION_PATHS = (
    "/Applications/VMware Fusion.app/Contents/Public",
    "/Applications/VMware Fusion.app/Contents/Library",
)
SPECIAL_ACTIONS = {"WAIT", "DONE", "FAIL"}

# The exact upstream this adapter is aligned against. Verified 2026-07-03 from
# primary sources (repo tree, run scripts, lib_run_single.py, desktop_env.py,
# show_result.py at this commit; paper arXiv:2606.29537):
# - Official launch scripts run with ``--max_steps 500`` and inline checkpoint
#   evaluations at 150/300 (scripts/bash/run_multienv_claude.sh); the bare
#   ``run.py`` argparse default is the legacy 15.
# - Evaluation is VM-state-only: ``DesktopEnv.evaluate()`` scores getters over
#   files/app/OS/browser state; the ONLY agent-message channel is the special
#   ``FAIL`` action for ``evaluator.func == "infeasible"`` tasks.
# - ``show_result.py`` consumes ``<result_dir>/<action_space>/<observation_type>/
#   <model>/<domain>/<example_id>/result.txt``.
ALIGNED_UPSTREAM = {
    "repo": "https://github.com/xlang-ai/OSWorld-V2",
    "commit": "c261cb57a699bd18db128787ca4e71b749141762",
    "commit_date": "2026-06-30",
    "paper": "arXiv:2606.29537 (OSWorld 2.0: Benchmarking Computer Use Agents on Long-Horizon Real-World Tasks)",
    "protocol_max_steps": 500,
    "protocol_checkpoint_steps": [150, 300],
    "legacy_repo": "https://github.com/xlang-ai/OSWorld",
}

# Providers this adapter can actually drive locally. Official OSWorld 2.0 also
# supports aws/azure/gcp/aliyun/volcengine, but this adapter has no cloud path.
SUPPORTED_PROVIDERS = ("vmware", "docker")


def osworld_checkout_info(osworld_root: Path) -> dict[str, Any]:
    """Describe an OSWorld checkout: variant (v1/v2), git commit, key modules.

    Variant markers verified against the upstream trees:
    ``evaluation_examples/test_v2.json`` exists only in OSWorld-V2;
    ``evaluation_examples/test_all.json`` only in classic OSWorld.
    """

    root = Path(osworld_root).expanduser().resolve(strict=False)
    info: dict[str, Any] = {
        "root": str(root),
        "exists": root.is_dir(),
        "variant": "unknown",
        "git_commit": "",
        "matches_aligned_commit": False,
        "has_desktop_env": (root / "desktop_env" / "desktop_env.py").is_file(),
        "aligned_upstream": dict(ALIGNED_UPSTREAM),
    }
    if (root / "evaluation_examples" / "test_v2.json").is_file():
        info["variant"] = "v2"
    elif (root / "evaluation_examples" / "test_all.json").is_file():
        info["variant"] = "v1"
    elif (root / "evaluation_examples").is_dir():
        info["variant"] = "examples_only"
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if proc.returncode == 0:
            info["git_commit"] = proc.stdout.strip()
    except Exception:
        pass
    info["matches_aligned_commit"] = bool(info["git_commit"]) and info["git_commit"] == ALIGNED_UPSTREAM["commit"]
    return info


def provider_preflight_failures(provider_name: str, path_to_vm: str) -> list[str]:
    """Fail loudly (with what is missing) when the VM provider cannot run here."""

    provider = str(provider_name or "").strip().lower()
    failures: list[str] = []
    if provider not in SUPPORTED_PROVIDERS:
        failures.append(
            f"provider '{provider}' is not supported by this adapter "
            f"(supported: {', '.join(SUPPORTED_PROVIDERS)}); official OSWorld 2.0 cloud "
            "providers (aws/azure/gcp) have no local adapter path"
        )
        return failures
    if provider == "vmware":
        vm_path = Path(path_to_vm).expanduser()
        if not vm_path.exists():
            failures.append(f"VM path not found: {vm_path}")
        _ensure_vmrun_on_path()
        if not any((Path(path) / "vmrun").exists() for path in VMWARE_FUSION_PATHS) and not shutil.which("vmrun"):
            failures.append("vmrun not found (checked VMware Fusion app paths and PATH)")
    elif provider == "docker":
        docker = shutil.which("docker")
        if not docker:
            failures.append("docker CLI not found on PATH (required by the docker provider)")
        else:
            try:
                proc = subprocess.run(
                    [docker, "info", "--format", "{{.ServerVersion}}"],
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                if proc.returncode != 0:
                    failures.append(
                        "docker daemon not reachable: "
                        + (proc.stderr or proc.stdout or "").strip()[:200]
                    )
            except Exception as exc:  # noqa: BLE001 - preflight diagnostics
                failures.append(f"docker daemon probe failed: {type(exc).__name__}: {exc}")
    return failures


def _persist_evaluation_result(result: Any, run_dir: Path) -> float:
    """Persist ``env.evaluate()`` output the way official lib_run_single.py does.

    Upstream (OSWorld-V2 ``_persist_evaluation_result``): the result may be a
    float (legacy) or a dict whose ``score`` field is the canonical float; dict
    results are additionally written to ``result.json``. ``result.txt`` is what
    official ``show_result.py`` scores.
    """

    if isinstance(result, dict):
        try:
            score = float(result.get("score", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        (run_dir / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
    else:
        score = float(result)
    (run_dir / "result.txt").write_text(f"{score}\n", encoding="utf-8")
    return score


@dataclass
class StepAgentConfig:
    ouroboros_bin: str
    ouroboros_url: str
    repo_dir: Path
    data_dir: Path
    settings_path: Path
    result_dir: Path
    task_id: str
    model: str
    timeout_sec: int
    max_obs_chars: int
    screenshot_check_only: bool
    disable_tools: str = "claude_code_edit"


@dataclass
class TaskRecordConfig:
    run_dir: Path
    result_root: Path
    repo_dir: Path
    settings_path: Path
    example_id: str
    domain: str
    reward: float | None
    steps: int
    status: str
    reason_code: str
    # The ADMITTED manifest (persisted by `admit_benchmark_run` before anything could
    # refuse). Required, and deliberately without a default: the records used to fall back
    # to REBUILDING it with `require_clean=False`, which wrote a manifest whose `seed_gate`
    # said the run was admissible on exactly the path where the gate had REFUSED it.
    base_manifest: dict[str, Any]
    error: str = ""
    extra: dict[str, Any] | None = None


@dataclass
class PreflightConfig:
    osworld_root: Path
    task_path: Path
    path_to_vm: str
    repo_dir: Path
    data_dir: Path
    settings_path: Path
    result_root: Path
    ouroboros_url: str
    model: str
    provider_name: str = "vmware"
    allow_scaffold_mismatch: bool = False


def _install_optional_dependency_stubs() -> None:
    """Avoid heavy optional evaluator imports when a selected task does not use them."""

    if "easyocr" not in sys.modules:
        easyocr = types.ModuleType("easyocr")

        class _UnavailableReader:
            def __init__(self, *_args: Any, **_kwargs: Any) -> None:
                raise RuntimeError("easyocr is not installed; OCR metrics unavailable")

        easyocr.Reader = _UnavailableReader  # type: ignore[attr-defined]
        sys.modules["easyocr"] = easyocr

    if "fastdtw" not in sys.modules:
        fastdtw_mod = types.ModuleType("fastdtw")

        def _fastdtw_unavailable(*_args: Any, **_kwargs: Any) -> tuple[float, list[Any]]:
            raise RuntimeError("fastdtw is not installed; audio metrics unavailable")

        fastdtw_mod.fastdtw = _fastdtw_unavailable  # type: ignore[attr-defined]
        sys.modules["fastdtw"] = fastdtw_mod


def _ensure_vmrun_on_path() -> None:
    path_parts = os.environ.get("PATH", "").split(os.pathsep)
    changed = False
    for candidate in VMWARE_FUSION_PATHS:
        if Path(candidate, "vmrun").exists() and candidate not in path_parts:
            path_parts.insert(0, candidate)
            changed = True
    if changed:
        os.environ["PATH"] = os.pathsep.join(path_parts)


def _safe_slug(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-._")
    return cleaned[:80] or uuid.uuid4().hex[:8]


def _http_json(url: str, timeout: float = 5.0) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    return json.loads(raw) if raw.strip().startswith("{") else {"raw": raw}


_DEFAULT_DESKTOP_PORT = 8765
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "0.0.0.0", "::1", "[::1]", ""})


def _is_default_desktop_server(url: str) -> bool:
    """True if ``url`` points at the LIVE desktop server's port on any loopback
    spelling. The guard keyed on the literal ``http://127.0.0.1:8765`` string, so
    ``localhost:8765`` / ``127.0.0.2:8765`` / ``[::1]:8765`` bypassed it and could
    still write into the live data root (adversarial review r1)."""
    from urllib.parse import urlparse

    try:
        parsed = urlparse(str(url or "").strip())
    except Exception:
        return False
    host = (parsed.hostname or "").strip().lower()
    port = parsed.port if parsed.port is not None else (443 if parsed.scheme == "https" else 80)
    is_loopback = host in _LOOPBACK_HOSTS or host.startswith("127.")
    return is_loopback and port == _DEFAULT_DESKTOP_PORT


# --------------------------------------------------------------------------- #
# Shared OSWorld launcher helpers (imported by run_cu_bridge_agent.py, which
# already reuses this module for the live-server guard and checkout probe).
# --------------------------------------------------------------------------- #

def amend_task_manifest(base_manifest: dict[str, Any], *, output_paths: dict[str, Any],
                        extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return the per-task manifest: the ONE early-built manifest plus late facts.

    The clean-seed/provenance gate lives in ``benchmark_run_manifest`` itself, so the
    manifest has to be built BEFORE the first paid step; the outcome-time facts
    (output paths, counters, evaluator status) are merged in here instead of building
    a second manifest after the money is spent.
    """
    manifest = dict(base_manifest)
    manifest["output_paths"] = {**(manifest.get("output_paths") or {}), **output_paths}
    manifest["extra"] = {**(manifest.get("extra") or {}), **(extra or {})}
    return manifest


def admit_step_loop_run(manifest_path: Path, *, result_root: Path, repo_dir: Path,
                        settings_path: Path, example_id: str,
                        require_clean: bool) -> dict[str, Any]:
    """ADMISSION seam for the step loop: build the manifest, PERSIST it, then enforce.

    Routes through the shared `admit_benchmark_run` (v6.75.0) rather than pairing the bare
    manifest builder with a later `write_json()`: the complete gate/refusal
    payload reaches disk BEFORE a refusal propagates, so a refused run is exactly as
    auditable as an admitted one. A refusal raises `BenchmarkAdmissionRefused` carrying
    that same manifest, which the caller records and finalizes.
    """
    return admit_benchmark_run(
        manifest_path,
        benchmark="osworld",
        run_root=result_root,
        repo_dir=repo_dir,
        requested_task_ids=[example_id],
        dataset="OSWorld",
        settings_path=settings_path,
        require_clean=require_clean,
        harness={
            "adapter": "external_step_loop",
            "official_actions": True,
            "memory_mode": "empty_per_ouroboros_step",
            "action_space": "pyautogui",
            "aligned_upstream": dict(ALIGNED_UPSTREAM),
        },
    )


def _teardown_partial_desktop_env(env: Any) -> None:
    """Best-effort teardown of a DesktopEnv whose ``__init__`` raised.

    ``env.close()`` is the official path (it calls
    ``provider.stop_emulator(path_to_vm)``); a construction that died before
    ``provider``/``path_to_vm`` were assigned cannot use it, so fall back to the
    provider directly. Never raises: cleanup must not mask the original failure.
    """
    try:
        env.close()
        return
    except Exception:
        pass
    provider = getattr(env, "provider", None)
    if provider is None:
        return
    try:
        provider.stop_emulator(getattr(env, "path_to_vm", None))
    except Exception:
        pass


def construct_desktop_env(desktop_env_cls: Any, *, attempts: int, deadline: float,
                          retry_sleep_sec: float = 5.0, **kwargs: Any) -> Any:
    """Construct ``DesktopEnv``, retrying a failed boot and tearing down each attempt.

    THE AUTHORISED BENEFIT IS THE RETRY (owner decision Q15=A). ``DesktopEnv.__init__``
    boots the VM/container inside ``_start_emulator()``, and the launchers used to retry
    only ``env.reset`` — so one transient boot failure (a lost
    ``/tmp/docker_port_allocation.lck`` race, a slow image load) burned the whole task.
    The constructor is now retried inside the startup window instead.

    Teardown of failed attempts is BELT-AND-BRACES, not a fix for measured debris: no run
    here has been shown to accumulate leaked containers. It is done because a raise inside
    ``__init__`` discards the half-built object, so whatever ``_start_emulator()`` had
    already started would be unreachable and therefore unstoppable. Constructing through
    ``__new__`` + explicit ``__init__`` keeps that partially-initialised instance
    reachable, which is the only way to close it at all.

    ``deadline`` is an absolute ``time.time()`` bound on STARTING a new attempt (an
    in-flight attempt is never cut short), so a run cannot spend its whole startup window
    respawning VMs.
    """
    last_err = ""
    for attempt in range(1, max(1, int(attempts)) + 1):
        if attempt > 1 and time.time() >= deadline:
            last_err = f"{last_err}; startup deadline reached, no further attempts"
            break
        env = desktop_env_cls.__new__(desktop_env_cls)
        try:
            env.__init__(**kwargs)
            return env
        except Exception as exc:  # noqa: BLE001 - every failed boot must be cleaned up
            last_err = f"attempt {attempt}: {type(exc).__name__}: {exc}"
            print(f"[osworld] DesktopEnv construction failed ({last_err}); tearing down", flush=True)
            _teardown_partial_desktop_env(env)
            time.sleep(max(0.0, float(retry_sleep_sec)))
    raise RuntimeError(f"DesktopEnv construction failed: {last_err}")


class ClaimDirNotConfined(ValueError):
    """The claim directory would put lock/marker files inside repo/ or the live data root."""


def confined_claims_dir(claims_dir: Path, *, repo_dir: Path) -> Path:
    """Resolve a claim directory, REFUSING one inside ANY checkout this run touches.

    The claim dir is operator-supplied (`--claim-dir`) and the helpers below create it and
    write `.lock`, `.scored` and `.scored_unconfirmed` into it, so an unchecked path mutates a
    repository or the live runtime data. Routed through the SAME boundary every benchmark
    output root uses (`assert_outside_repo`, which also covers `live_data_roots()`), in its
    PURE form so the refusal happens before anything is created.

    ``repo_dir`` is REQUIRED and is the checkout actually being executed (`--repo-dir`, the one
    the run manifest attests). Deriving the authority from this module's own location instead —
    which is all this helper used to do — confined the claim dir against the LAUNCHER's
    checkout, so `--repo-dir /other/bench-clone --claim-dir /other/bench-clone/.claims` wrote
    lock and marker state straight into the execution checkout: the very tree whose cleanliness
    the seed gate is about to attest. Both roots are checked, active checkout first; the static
    one is belt-and-braces, never the sole authority.
    """
    resolved = Path(claims_dir)
    for authority in (Path(repo_dir).expanduser(), repo_root_from_devtools()):
        try:
            resolved = assert_outside_repo(resolved, authority)
        except ValueError as exc:
            raise ClaimDirNotConfined(f"--claim-dir is not confined: {exc}") from exc
    return resolved


def task_claim_key(domain: str, example_id: str) -> str:
    """Filesystem-safe claim identity for one OSWorld task."""
    return f"{_safe_slug(str(domain))}__{_safe_slug(str(example_id))}"


def claim_stale_sec(task_timeout_sec: float, startup_timeout_sec: float, margin_sec: float) -> float:
    """Lock staleness bound: longer than every rail the legitimate holder can be inside.

    A holder spends TWO startup windows, not one: ``construct_desktop_env`` gets its
    own ``startup_timeout`` deadline and the reset-to-usable-screenshot loop then gets
    a fresh one (sharing a single window would let a slow boot eat the reset budget).
    Adding the task timeout, the bound is ``task_timeout + 2 * startup_timeout +
    margin``. A one-window bound could expire while a lane was still legitimately
    working, which is exactly how two lanes end up on one task.

    ``env.evaluate()`` runs after all of those and is UNBOUNDED — upstream getters may
    fetch over the network — so no formula can cover it. That residual is what
    ``margin`` is for: raise ``--claim-margin-sec`` for domains with slow evaluators
    instead of widening the formula with a term nothing enforces.
    """
    return (float(task_timeout_sec) + 2.0 * float(startup_timeout_sec)
            + max(0.0, float(margin_sec)))


def acquire_task_claim(claims_dir: Path, claim_key: str, *, stale_sec: float,
                       repo_dir: Path, metadata: str = "") -> tuple[int | None, str]:
    """Claim one task for this lane. Returns ``(lock_fd, reason)``.

    ``lock_fd is None`` means DO NOT run this task; ``reason`` is one of
    ``already_scored`` (another attempt produced an official score — the "first scored attempt
    wins" rule, enforced rather than merely documented), ``scored_unconfirmed`` (a score exists
    but its canonical marker could not be persisted; see ``mark_task_scored``) or ``in_flight``
    (another attempt holds the lock). Reuses the portable O_EXCL lockfile from
    ``ouroboros.platform_layer``; no daemon, no registry, no lease.

    The scored STATE is read from markers, never from the lock, so a refusal on it is
    STALENESS-INDEPENDENT: the lock is deliberately expirable (`stale_sec` reclaims a crashed
    holder's task) and a protection built on it would fail open the moment somebody waited long
    enough. `scored_unconfirmed` therefore refuses forever, until an operator clears it.

    The state is checked TWICE, and the second check is the load-bearing one. Checking it only
    BEFORE waiting for the lock is a live TOCTOU hole: two attempts both see no marker, the
    first wins the lock, scores, marks and releases, and the second then acquires the lock with
    the marker already on disk and would still be told ``claimed`` — rerunning a task that
    already has an official score, which is the exact corruption the rule forbids. So the state
    is re-read once the lock is HELD (nobody can be mid-transition then) and the lock we just
    took is released again if the answer changed.
    """
    from ouroboros.platform_layer import acquire_exclusive_file_lock, release_exclusive_file_lock

    claims_dir = confined_claims_dir(claims_dir, repo_dir=repo_dir)
    claims_dir.mkdir(parents=True, exist_ok=True)
    state = scored_claim_state(claims_dir, claim_key)
    if state:
        return None, state
    lock_path = claims_dir / f"{claim_key}.lock"
    fd = acquire_exclusive_file_lock(
        lock_path, timeout_sec=1.0, stale_sec=stale_sec,
        metadata=metadata or f"pid={os.getpid()} ts={time.time()}\n",
    )
    if fd is None:
        return None, "in_flight"
    state = scored_claim_state(claims_dir, claim_key)
    if state:
        # Scored by the previous holder while we were blocking on the lock. Give back the lock
        # we just took — keeping it would park a task nobody may run for the whole staleness
        # window — and step aside.
        release_exclusive_file_lock(lock_path, fd)
        return None, state
    return fd, "claimed"


UNCONFIRMED_SCORE_SUFFIX = ".scored_unconfirmed"


class ClaimMarkerNotDurable(RuntimeError):
    """The permanent ``<key>.scored`` marker could not be persisted.

    "First SCORED attempt wins" (owner Q14=A) is an AUTHORITY fixed before any numbers were
    read, not an optimisation: with no marker another attempt reruns a task that already has an
    official score, and the pre-registered dedup rule is violated in the direction that
    CORRUPTS results. So a marker-persistence failure is raised rather than swallowed.

    ``unconfirmed_marker`` is the durable record of the "scored but unmarked" state — the
    ``<key>.scored_unconfirmed`` path — or ``None`` when even THAT could not be written. The
    distinction is the whole recovery story: with the marker, the refusal is permanent and
    visible; without it, nothing on disk remembers that a score exists, so retaining the
    in-flight lock is all that is left and that lock EXPIRES. The caller must then refuse
    loudly rather than pretend the task is protected.
    """

    def __init__(self, message: str, *, unconfirmed_marker: Path | None = None) -> None:
        super().__init__(message)
        self.unconfirmed_marker = unconfirmed_marker


def record_unconfirmed_score(claims_dir: Path, claim_key: str, *, repo_dir: Path, reason: str,
                             payload: dict[str, Any] | None = None) -> Path | None:
    """Durably record "this task HAS an official score that no canonical marker names".

    Returns the marker path, or ``None`` when even THIS could not be written. Never raises: it
    is called on paths whose job is to decide WHICH refusal to make, including one that is
    already unwinding a ``KeyboardInterrupt``, and a second failure there must not replace the
    operator's interrupt with a disk error.

    ``<key>.scored_unconfirmed`` is the only STALENESS-INDEPENDENT protection available once the
    canonical marker is missing: `stale_sec` reclaims the in-flight lock BY DESIGN, so a
    lock-only protection fails open the moment somebody waits long enough. This marker never
    expires, so ``scored_claim_state`` refuses the task until an operator clears it.

    Idempotent in the direction that matters: an existing canonical ``.scored`` marker means the
    score IS properly recorded, so it is returned untouched and no unconfirmed state is created.
    """
    from ouroboros.utils import atomic_write_json

    try:
        claims_dir = confined_claims_dir(claims_dir, repo_dir=repo_dir)
        marker = claims_dir / f"{claim_key}.scored"
        if marker.is_file():
            return marker
        unconfirmed = claims_dir / f"{claim_key}{UNCONFIRMED_SCORE_SUFFIX}"
        claims_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            unconfirmed,
            {"claim_key": claim_key, "ts_unix": time.time(), "reason": reason,
             "canonical_marker": str(marker), **(payload or {})},
            trailing_newline=True,
            fsync=True,
        )
        return unconfirmed if unconfirmed.is_file() else None
    except BaseException:  # noqa: BLE001 - the caller ESCALATES a None, never a new exception
        return None


def mark_task_scored(claims_dir: Path, claim_key: str, *, repo_dir: Path,
                     payload: dict[str, Any] | None = None) -> Path:
    """Fail-CLOSED durable claim transition: this task HAS an official score.

    Called immediately after ``env.evaluate()`` and BEFORE the score is projected into any
    result artefact, which is what makes the rule survive a crash. The only orderings a
    process death can then produce are "marker, no result" (a later lane steps aside; the
    denominator shows the missing row) and "no marker, no result" (a later lane legitimately
    retries). "Result without marker" — the one ordering that makes a lane rerun an
    already-scored task — is unreachable.

    Written with ``fsync=True``: "durable" has to mean survived-the-power-cut, not
    reached-the-page-cache. Idempotent — an existing marker IS the first scored attempt and
    is never overwritten.

    If the canonical marker cannot be written, the "scored but unmarked" state is recorded
    DURABLY at ``<key>.scored_unconfirmed`` instead, and the refusal carries that path. Leaving
    only the in-flight lock behind was a protection with an expiry date: `stale_sec` makes that
    lock reclaimable BY DESIGN, so once enough time passed another attempt claimed a task that
    already had an official score — the same corruption, merely delayed. The marker never
    expires, so the refusal is permanent and an operator can see it.
    """
    from ouroboros.utils import atomic_write_json

    claims_dir = confined_claims_dir(claims_dir, repo_dir=repo_dir)
    marker = claims_dir / f"{claim_key}.scored"
    unconfirmed = claims_dir / f"{claim_key}{UNCONFIRMED_SCORE_SUFFIX}"
    try:
        claims_dir.mkdir(parents=True, exist_ok=True)
        if not marker.exists():
            atomic_write_json(
                marker,
                {"claim_key": claim_key, "ts_unix": time.time(), **(payload or {})},
                trailing_newline=True,
                fsync=True,
            )
        if not marker.is_file():
            raise OSError(f"scored marker is absent after a successful write: {marker}")
    except Exception as exc:  # noqa: BLE001 - re-raised as the typed fail-closed refusal
        # SECOND and LAST attempt, at a different path. Not a further layer of best-effort: it
        # decides WHICH refusal the caller must make. If it succeeds the task is permanently
        # refused by `scored_claim_state`; if it fails, nothing on disk remembers the score and
        # the caller has to say so loudly instead of promising a protection that expires.
        recorded = record_unconfirmed_score(
            claims_dir, claim_key, repo_dir=repo_dir, reason="scored_marker_write_failed",
            payload={"error": f"{type(exc).__name__}: {exc}", **(payload or {})},
        )
        if recorded is not None:
            raise ClaimMarkerNotDurable(
                f"could not persist the scored-claim marker {marker}: {type(exc).__name__}: "
                f"{exc}; recorded the scored-but-unmarked state at {recorded} instead, which "
                "refuses this task permanently (staleness cannot reclaim it)",
                unconfirmed_marker=recorded,
            ) from exc
        raise ClaimMarkerNotDurable(
            f"could not persist the scored-claim marker {marker} NOR the fallback "
            f"{unconfirmed}: {type(exc).__name__}: {exc}; the claim directory is unusable, so "
            "NOTHING on disk records that this task has an official score and the in-flight "
            "lock will expire — refuse loudly and do not continue against this claim dir",
            unconfirmed_marker=None,
        ) from exc
    return marker


def scored_claim_state(claims_dir: Path | None, claim_key: str) -> str:
    """READ-ONLY ownership question. ``""``, ``"already_scored"`` or ``"scored_unconfirmed"``.

    Deliberately pure (``exists()`` only — no mkdir, no lock, no write) so a launcher can ask it
    BEFORE admission and step aside leaving ZERO footprint. "First SCORED attempt wins" means a
    later attempt must not even write its own admission record into the winner's per-task run
    directory: the manifest path is shared between attempts, so a footprint there is a clobber.

    Neither answer involves the lock, so neither expires. ``scored_unconfirmed`` means a score
    exists but its canonical marker could not be persisted (``mark_task_scored``); it needs an
    operator, and until then the task stays refused rather than silently becoming claimable.
    """
    if claims_dir is None:
        return ""
    claims_dir = Path(claims_dir)
    if (claims_dir / f"{claim_key}.scored").exists():
        return "already_scored"
    if (claims_dir / f"{claim_key}{UNCONFIRMED_SCORE_SUFFIX}").exists():
        return "scored_unconfirmed"
    return ""


def task_already_scored(claims_dir: Path | None, claim_key: str) -> bool:
    """True when this task must not be run again — either scored state counts."""
    return bool(scored_claim_state(claims_dir, claim_key))


def release_task_claim(claims_dir: Path, claim_key: str, lock_fd: int | None, *,
                       scored: bool, repo_dir: Path,
                       payload: dict[str, Any] | None = None) -> None:
    """Release the in-flight lock; a SCORED attempt keeps its permanent marker.

    Only a scored attempt owns ``<key>.scored``. An unscored attempt (adapter error,
    preflight block, crashed lane) deliberately leaves the task claimable again so a later
    lane may retry it — "first SCORED attempt wins", not "first attempt wins".

    A scored claim is released ONLY once its marker is confirmed on disk. The marker is the
    entire mechanism that stops a rerun, so releasing the lock without it hands the task
    straight back to the next attempt; ``mark_task_scored`` raises ``ClaimMarkerNotDurable``
    instead (the write used to be wrapped in a bare ``except: pass``) and the release below
    never runs.
    """
    from ouroboros.platform_layer import release_exclusive_file_lock

    claims_dir = Path(claims_dir)
    if scored:
        mark_task_scored(claims_dir, claim_key, repo_dir=repo_dir, payload=payload)
    release_exclusive_file_lock(claims_dir / f"{claim_key}.lock", lock_fd)


def _preflight(config: PreflightConfig) -> dict[str, Any]:
    failures: list[str] = []
    details: dict[str, Any] = {}
    selected_model = str(config.model or "")
    checkout = osworld_checkout_info(config.osworld_root)
    details["osworld_checkout"] = checkout
    if not checkout["exists"]:
        failures.append(f"OSWorld checkout not found: {config.osworld_root}")
    if not (config.osworld_root / "evaluation_examples").exists():
        failures.append(f"OSWorld checkout shape not recognized: {config.osworld_root}")
    if checkout["exists"] and not checkout["has_desktop_env"]:
        failures.append(
            f"desktop_env package missing in OSWorld checkout (expected desktop_env/desktop_env.py): {config.osworld_root}"
        )
    if checkout["exists"] and not checkout["matches_aligned_commit"]:
        details["upstream_pin_warning"] = (
            f"checkout commit {checkout['git_commit'] or '<unknown>'} differs from the aligned "
            f"OSWorld 2.0 pin {ALIGNED_UPSTREAM['commit']} ({ALIGNED_UPSTREAM['repo']}); "
            "results are only comparable against the pinned protocol"
        )
    if not config.task_path.is_file():
        failures.append(f"task JSON not found: {config.task_path}")
    failures.extend(provider_preflight_failures(config.provider_name, config.path_to_vm))
    if not config.repo_dir.is_dir() or not (config.repo_dir / "VERSION").exists():
        failures.append(f"Ouroboros repo shape not recognized: {config.repo_dir}")
    if not config.settings_path.is_file():
        failures.append(f"settings.json not found: {config.settings_path}")
    else:
        try:
            settings = json.loads(config.settings_path.read_text(encoding="utf-8"))
            selected_model = str(config.model or settings.get("OUROBOROS_MODEL") or "")
            details["model"] = selected_model
            if not selected_model:
                failures.append(
                    "a measured model must be declared through --model or settings "
                    "OUROBOROS_MODEL"
                )
            else:
                try:
                    declared_actor = runtime_actor_snapshot(
                        settings,
                        expected_model=selected_model,
                    )
                except ValueError as exc:
                    failures.append(f"declared actor invalid in settings: {exc}")
                else:
                    details["declared_runtime_actor"] = declared_actor
                    details["available_subagents"] = declared_actor["available_subagents"]
                    failures.extend(
                        f"declared settings {mismatch}"
                        for mismatch in declared_actor["mismatches"]
                    )
            from ouroboros.provider_models import PROVIDER_ENV_KEYS, provider_for_model

            provider = provider_for_model(selected_model)
            env_key = PROVIDER_ENV_KEYS.get(provider, "OPENROUTER_API_KEY")
            # The provider key lives on the TARGET server (steps submit over
            # `ouroboros run --url`); the CLIENT settings/env cannot prove the
            # server has it, and /api/settings masks it. So a missing client-side
            # key is a WARNING, not a preflight pass/fail — the server scaffold
            # check below is the authoritative "is the executing server usable"
            # gate (adversarial review r1: client-side key check was misleading).
            if not str(os.environ.get(env_key) or settings.get(env_key) or "").strip():
                details["client_provider_key_absent"] = (
                    f"{env_key} not set client-side for provider {provider}; the TARGET server "
                    "must carry it (not verifiable here — /api/settings masks secrets)."
                )
        except Exception as exc:
            failures.append(f"settings.json unreadable: {type(exc).__name__}: {exc}")
    try:
        ensure_outside_repo(config.data_dir, config.repo_dir)
    except Exception as exc:
        failures.append(f"data dir must be outside repo/live data: {exc}")
    try:
        uploads = config.data_dir / "uploads" / "osworld" / "_preflight"
        uploads.mkdir(parents=True, exist_ok=True)
        probe = uploads / "write_probe.txt"
        probe.write_text("ok\n", encoding="utf-8")
        probe.unlink(missing_ok=True)
    except Exception as exc:
        failures.append(f"data/uploads not writable: {type(exc).__name__}: {exc}")
    try:
        state = _http_json(config.ouroboros_url.rstrip("/") + "/api/state", timeout=5)
        details["ouroboros_state"] = {
            "supervisor_ready": state.get("supervisor_ready"),
            "runtime_mode": state.get("runtime_mode"),
        }
        if not state.get("supervisor_ready"):
            failures.append("Ouroboros server reachable but supervisor_ready is false")
        # The adapter submits over the gateway (`ouroboros run --url`), so env
        # vars in the CLI subprocess can NOT configure the executing server.
        # The disclosed scaffold defaults are only real if the TARGET SERVER
        # already runs them — verify its effective settings and fail loudly on
        # drift (start the isolated server from osworld/settings_base.json).
        server_settings = _http_json(config.ouroboros_url.rstrip("/") + "/api/settings", timeout=5)
        expected = {
            "OUROBOROS_RUNTIME_MODE": "pro",
            "OUROBOROS_SAFETY_MODE": "light",
            "OUROBOROS_MAX_WORKERS": 4,
            # The scaffold's blocking review lane is only real if the TARGET server
            # runs it — CLI env cannot configure the executing server, so the
            # preflight must verify it (adversarial review r2 #6).
            "OUROBOROS_REVIEW_ENFORCEMENT": "blocking",
        }
        mismatches = []
        for key, want in expected.items():
            got = server_settings.get(key)
            if str(got).strip().lower() != str(want).strip().lower():
                mismatches.append(f"{key}: server={got!r} expected={want!r}")
        if selected_model:
            target_actor = runtime_actor_snapshot(
                server_settings,
                expected_model=selected_model,
            )
            details["target_runtime_actor"] = target_actor
            mismatches.extend(target_actor["mismatches"])
        details["server_scaffold_settings"] = {
            k: server_settings.get(k)
            for k in (
                "OUROBOROS_RUNTIME_MODE",
                "OUROBOROS_SAFETY_MODE",
                "OUROBOROS_MAX_WORKERS",
                "OUROBOROS_MODEL",
                "OUROBOROS_SUBAGENTS",
            )
        }
        if mismatches:
            message = (
                "target server settings do not match the disclosed OSWorld scaffold "
                "(render devtools/benchmarks/osworld/settings_base.json into an isolated "
                "server and point --ouroboros-url at it): " + "; ".join(mismatches)
            )
            if config.allow_scaffold_mismatch:
                details["scaffold_mismatch_allowed"] = mismatches
            else:
                failures.append(message)
    except Exception as exc:
        failures.append(f"Ouroboros server not reachable: {type(exc).__name__}: {exc}")
    # Owner Q9=A+B / Q10: this launcher attaches to a live server URL, so its own admission
    # path attests WHAT is running (the HTTP `runtime_version`) against the checkout it was
    # started from (local HEAD + VERSION) before a single paid step. The shared helper fails
    # CLOSED by raising; translate that into a typed preflight failure so the caller still
    # gets `blocked/preflight_failed` records instead of a bare traceback.
    #
    # A refusal CARRIES the record it built (`RuntimeAttestationRefused.attestation`): the exact
    # typed reason plus `runtime_version`, `repo_head` and `repo_version`. Keeping only the
    # message threw those away at the moment they matter most — the manifest then said
    # `runtime_attestation_failed` and nothing about WHICH runtime disagreed with WHICH commit.
    try:
        details["runtime_attestation"] = runtime_attestation(config.ouroboros_url, config.repo_dir)
    except RuntimeAttestationRefused as exc:
        details["runtime_attestation"] = dict(exc.attestation)
        failures.append(f"runtime attestation failed reason={exc.attestation.get('reason') or ''}: {exc}")
    except RuntimeError as exc:
        # No record to keep (the helper raised before building one).
        details["runtime_attestation"] = {"ok": False, "reason": "runtime_attestation_failed",
                                          "error": f"{type(exc).__name__}: {exc}"}
        failures.append(f"runtime attestation failed: {exc}")
    try:
        ensure_outside_repo(config.result_root, config.repo_dir)
    except Exception as exc:
        failures.append(str(exc))
    return {"ok": not failures, "failures": failures, "details": details}


def _bind_target_actor(manifest: dict[str, Any], preflight: dict[str, Any]) -> None:
    """Record the actor exposed by the target, including allowed scaffold drift."""
    target_actor = (preflight.get("details") or {}).get("target_runtime_actor") or {}
    manifest["available_subagents"] = (
        dict(target_actor.get("available_subagents") or {})
        if isinstance(target_actor, dict)
        else {}
    )
    manifest["harness"] = {
        **(manifest.get("harness") or {}),
        "target_runtime_actor": target_actor,
    }


def _json_from_text(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not match:
        return {}
    try:
        value = json.loads(match.group(0))
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        return {}


def _shell_action(command: str, cwd: str = "", timeout: int = 300) -> str:
    """Render a structured shell action as an OSWorld pyautogui/Python snippet.

    OSWorld records the resulting Python snippet as the official action and runs
    the command through a non-interactive bash. We deliberately do NOT fabricate
    ``~/.bash_history`` entries: writing the command into the history file to
    satisfy a terminal-task evaluator is hidden-verifier-knowledge / answer
    fitting (forbidden by the audit's methodology rules — the command's real
    execution path simply does not produce interactive history).
    """

    command = str(command or "").strip()
    cwd = str(cwd or "").strip()
    try:
        timeout = max(1, int(timeout))
    except Exception:
        timeout = 300
    encoded = base64.b64encode(command.encode("utf-8", errors="replace")).decode("ascii")
    return (
        "import base64, pathlib, subprocess, tempfile\n"
        f"cmd = base64.b64decode({encoded!r}).decode('utf-8', errors='replace')\n"
        f"cwd = {cwd!r} or None\n"
        f"timeout = {timeout!r}\n"
        "with tempfile.NamedTemporaryFile('w', suffix='.sh', delete=False) as script:\n"
        "    script.write('set -e\\n' + cmd + '\\n')\n"
        "    script_path = script.name\n"
        "try:\n"
        "    result = subprocess.run(['/bin/bash', script_path], cwd=cwd, text=True, capture_output=True, timeout=timeout)\n"
        "finally:\n"
        "    pathlib.Path(script_path).unlink(missing_ok=True)\n"
        "print(result.stdout)\n"
        "print(result.stderr)\n"
        "result.check_returncode()\n"
    )


def _click_action(x: Any, y: Any) -> str:
    return (
        "import pyautogui, time\n"
        f"pyautogui.click({int(float(x))}, {int(float(y))})\n"
        "time.sleep(0.5)\n"
    )


def _type_action(text: str, interval: float = 0.01) -> str:
    return (
        "import pyautogui, time\n"
        f"pyautogui.typewrite({str(text or '')!r}, interval={float(interval)!r})\n"
        "time.sleep(0.2)\n"
    )


def _hotkey_action(keys: Any) -> str:
    if isinstance(keys, str):
        key_list = [part.strip() for part in keys.split("+") if part.strip()]
    elif isinstance(keys, list):
        key_list = [str(part).strip() for part in keys if str(part).strip()]
    else:
        key_list = []
    return (
        "import pyautogui, time\n"
        f"pyautogui.hotkey(*{key_list!r})\n"
        "time.sleep(0.3)\n"
    )


def _wait_action(seconds: Any = 1.0) -> str:
    try:
        seconds = max(0.0, float(seconds))
    except Exception:
        seconds = 1.0
    return f"import time\ntime.sleep({seconds!r})\n"


def _initial_observation_with_retries(
    env: Any,
    example: dict[str, Any],
    *,
    startup_timeout_sec: int,
    reset_retries: int,
    wait_after_reset_sec: float,
    retry_sleep_sec: float,
    run_dir: Path,
) -> dict[str, Any]:
    """Reset OSWorld and wait for a usable first observation.

    VM reset, in-VM server readiness, screenshot capture, and accessibility-tree
    availability are startup concerns, not agent reasoning steps. Keep retrying
    them within a dedicated startup budget so transient VM/controller slowness does
    not become a task failure.
    """

    deadline = time.time() + max(1, int(startup_timeout_sec))
    attempts = max(1, int(reset_retries))
    errors: list[str] = []
    last_obs: dict[str, Any] = {}

    for attempt in range(1, attempts + 1):
        if time.time() >= deadline:
            break
        try:
            obs = env.reset(task_config=example)
            if wait_after_reset_sec > 0:
                time.sleep(wait_after_reset_sec)
            while time.time() < deadline:
                try:
                    obs = env._get_obs()
                    last_obs = obs if isinstance(obs, dict) else {}
                    screenshot = last_obs.get("screenshot")
                    if isinstance(screenshot, (bytes, bytearray)) and screenshot:
                        (run_dir / "startup_readiness.json").write_text(
                            json.dumps(
                                {
                                    "ok": True,
                                    "attempt": attempt,
                                    "has_screenshot": True,
                                    "has_accessibility_tree": bool(last_obs.get("accessibility_tree")),
                                },
                                ensure_ascii=False,
                                indent=2,
                            ),
                            encoding="utf-8",
                        )
                        return last_obs
                    errors.append(f"attempt {attempt}: observation missing screenshot")
                except Exception as exc:  # noqa: BLE001 - startup retry diagnostics
                    errors.append(f"attempt {attempt}: _get_obs {type(exc).__name__}: {exc}")
                time.sleep(max(0.1, retry_sleep_sec))
            break
        except Exception as exc:  # noqa: BLE001 - reset retry diagnostics
            errors.append(f"attempt {attempt}: reset {type(exc).__name__}: {exc}")
            time.sleep(max(0.1, retry_sleep_sec))

    (run_dir / "startup_readiness.json").write_text(
        json.dumps(
            {
                "ok": False,
                "errors": errors[-20:],
                "last_obs_keys": sorted(last_obs.keys()),
                "startup_timeout_sec": startup_timeout_sec,
                "reset_retries": reset_retries,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    raise RuntimeError(
        f"OSWorld startup did not produce a usable screenshot within {startup_timeout_sec}s; "
        f"last errors: {errors[-3:]}"
    )


def _normalize_structured_action(item: Any) -> str:
    """Convert a model action object to one OSWorld action string."""

    if isinstance(item, str):
        text = item.strip()
        return text.upper() if text.upper() in SPECIAL_ACTIONS else text
    if not isinstance(item, dict):
        return ""
    kind = str(item.get("type") or item.get("action") or "").strip().lower()
    if kind in {"done", "finish"}:
        return "DONE"
    if kind in {"fail", "infeasible"}:
        return "FAIL"
    if kind == "wait":
        if "seconds" in item:
            return _wait_action(item.get("seconds"))
        return "WAIT"
    if kind == "shell":
        return _shell_action(
            str(item.get("command") or item.get("cmd") or ""),
            cwd=str(item.get("cwd") or ""),
            timeout=int(item.get("timeout_sec") or item.get("timeout") or 300),
        )
    if kind == "click":
        return _click_action(item.get("x", 0), item.get("y", 0))
    if kind == "type":
        return _type_action(str(item.get("text") or ""), interval=float(item.get("interval") or 0.01))
    if kind == "hotkey":
        return _hotkey_action(item.get("keys") or item.get("key") or "")
    if kind in {"press", "key"}:
        return _hotkey_action([item.get("key") or item.get("keys") or ""])
    if kind == "python":
        return str(item.get("code") or "").strip()
    return ""


class OuroborosStepAgent:
    def __init__(
        self,
        config: StepAgentConfig | None = None,
        **kwargs: Any,
    ) -> None:
        if config is None:
            config = StepAgentConfig(**kwargs)
        self.ouroboros_bin = config.ouroboros_bin
        self.ouroboros_url = config.ouroboros_url
        self.repo_dir = config.repo_dir
        self.data_dir = config.data_dir
        self.settings_path = config.settings_path
        self.result_dir = config.result_dir
        self.model = config.model
        self.timeout_sec = config.timeout_sec
        self.max_obs_chars = config.max_obs_chars
        self.screenshot_check_only = config.screenshot_check_only
        self.disable_tools = config.disable_tools
        self.step_idx = 0
        self.history: list[dict[str, Any]] = []
        self.notes: list[str] = []
        self.final_answer = ""
        self.terminal_action = ""
        self.last_response = ""

    def reset(self) -> None:
        self.step_idx = 0
        self.history.clear()
        self.notes.clear()
        self.final_answer = ""
        self.terminal_action = ""
        self.last_response = ""

    def _save_screenshot(self, obs: dict[str, Any]) -> tuple[str, str]:
        screenshot = obs.get("screenshot")
        if not isinstance(screenshot, (bytes, bytearray)):
            return "", ""
        self.step_idx += 1
        name = f"step_{self.step_idx:03d}.png"
        local_path = self.result_dir / f"obs_{name}"
        local_path.write_bytes(bytes(screenshot))
        return str(local_path), str(local_path.name)

    @staticmethod
    def _prioritize_a11y(tree: str, budget: int) -> str:
        """Budget-bounded a11y view that PRIORITIZES interactive elements with
        coordinates instead of a blind head-slice (WS-9.6).

        A head-slice (the previous behavior) routinely cut the tree before the
        actionable widgets, so the agent never saw the controls it needed to
        click and resorted to blind/CLI moves. Here, when over budget, lines
        that name an interactive role AND/OR carry coordinates are kept first,
        then the rest in document order until the budget is spent.
        """
        if len(tree) <= budget:
            return tree
        lines = tree.splitlines()
        interactive = ("button", "menu", "entry", "text", "link", "check", "radio",
                       "tab", "combo", "field", "item", "toggle", "slider", "icon", "edit")
        coord_markers = ("coord", "position", "x=", "cp:", "screencoord", "bbox", "point")

        def score(line: str) -> int:
            low = line.lower()
            s = 0
            if any(k in low for k in interactive):
                s += 2
            if any(k in low for k in coord_markers):
                s += 2
            return s

        kept: list[tuple[int, str]] = []
        total = 0
        for _s, idx, line in sorted(((score(ln), i, ln) for i, ln in enumerate(lines)),
                                    key=lambda t: (-t[0], t[1])):
            if _s == 0 and total > 0:
                continue  # only spend budget on signal-bearing lines once we have some
            if total + len(line) + 1 > budget:
                continue
            kept.append((idx, line))
            total += len(line) + 1
        kept.sort()
        body = "\n".join(line for _i, line in kept)
        return body + "\n...[a11y prioritized: interactive/coordinate nodes kept, low-signal nodes dropped]"

    def _prompt(self, instruction: str, obs: dict[str, Any], screenshot_path: str, *, max_steps: int) -> str:
        a11y_tree = self._prioritize_a11y(str(obs.get("accessibility_tree") or ""), self.max_obs_chars)

        history_json = json.dumps(self.history[-12:], ensure_ascii=False, indent=2)
        notes_json = json.dumps(self.notes[-8:], ensure_ascii=False, indent=2)
        screenshot_instruction = (
            f'The current VM screenshot is attached to this Ouroboros run and also saved at "{screenshot_path}". '
            "Use the image directly when choosing GUI actions. If image input is unavailable, "
            "fall back to vlm_query(file_path=that path, prompt='Describe the Ubuntu desktop state and relevant controls')."
            if screenshot_path
            else "No screenshot bytes were available in this observation."
        )
        if self.screenshot_check_only:
            task_directive = (
                "This is a screenshot visibility smoke test. Use vlm_query on the "
                "screenshot path, then return WAIT with a short description of what "
                "you saw."
            )
        else:
            task_directive = (
                f"Choose the next OSWorld action(s). You are on step {self.step_idx} of at most {max_steps}. "
                "Prefer structured actions, not raw "
                "Python. Supported action objects: "
                '{"type":"shell","command":"...","cwd":"/home/user/Desktop"} (runs via non-interactive bash); '
                '{"type":"click","x":100,"y":200}; '
                '{"type":"type","text":"..."}; '
                '{"type":"hotkey","keys":["ctrl","l"]}; '
                '{"type":"wait","seconds":1}; '
                '{"type":"done"}; {"type":"fail"}. '
                'Use {"type":"python","code":"..."} only when no structured action fits. '
                "THE GRADER INSPECTS VM STATE ONLY. The OSWorld evaluator scores the virtual machine's "
                "state after your final step: files saved at the exact requested paths, in-application "
                "document state, the browser's ACTIVE TAB URL, and OS configuration. Text you write in "
                "chat is NEVER read by the evaluator. If the task asks a question, navigate the GUI until "
                "the answer is shown in the expected application/page and LEAVE the environment in that "
                "state (for example the browser tab open on the page that answers the question) before done. "
                "If the task edits a document or spreadsheet, SAVE the file to the exact expected path "
                "before done — an unsaved buffer or a chat answer scores zero. "
                "In app-named tasks, work in the named app first; if you edit files directly, reopen/verify in that app before done. "
                "Use done only after independently checking the evaluator-facing state. "
                "Use fail when demonstrably infeasible (missing hardware/resource, blocked permissions, feature absent); an out-of-app workaround is not success for an in-app task. "
                'When you return done or fail, ALSO set "final_answer" to your definitive short answer '
                "(for question-style tasks) or a one-line completion/infeasibility summary — it is recorded "
                "in the audit ledger, but it never replaces the required VM state. "
                "Do NOT claim a screenshot or VLM 'confirmed' / 'shows' anything unless you actually called vlm_query (or were given image input) THIS step; otherwise describe only what the accessibility tree and action history establish."
            )

        return f"""You are Ouroboros acting as an external OSWorld step-loop agent.
Return ONLY a JSON object, with no markdown and no prose outside JSON.

JSON schema:
{{"response": "short rationale", "notes": "optional cross-step note for yourself", "final_answer": "REQUIRED with done/fail: definitive short answer or completion summary", "actions": [{{"type": "shell", "command": "..."}}]}}

{task_directive}
{screenshot_instruction}

Task:
{instruction}

Recent official OSWorld action history:
{history_json}

Cross-step notes:
{notes_json}

Accessibility tree (may be empty/truncated):
{a11y_tree}
"""

    def predict(self, instruction: str, obs: dict[str, Any], *, max_steps: int) -> tuple[str, list[str], dict[str, Any]]:
        screenshot_path, local_screenshot = self._save_screenshot(obs)
        prompt = self._prompt(instruction, obs, screenshot_path, max_steps=max_steps)
        step = self.step_idx
        prompt_path = self.result_dir / f"prompt_step_{step:03d}.txt"
        prompt_path.write_text(prompt, encoding="utf-8")

        env = os.environ.copy()
        env.pop("OUROBOROS_MODEL_HEAVY", None)
        env.pop("USE_LOCAL_HEAVY", None)
        # NB: `ouroboros run --url` submits over the gateway, so these env vars
        # configure only the CLI subprocess, NOT the executing server — the
        # disclosed scaffold defaults are ENFORCED by the preflight check of the
        # target server's /api/settings (see _preflight). Kept here so any
        # CLI-local behavior matches the scaffold too.
        env.update({
            "OUROBOROS_REPO_DIR": str(self.repo_dir),
            "OUROBOROS_DATA_DIR": str(self.data_dir),
            "OUROBOROS_SETTINGS_PATH": str(self.settings_path),
            "OUROBOROS_RUNTIME_MODE": "pro",
            "OUROBOROS_MAX_WORKERS": "4",
            "OUROBOROS_SAFETY_MODE": "light",
            "OUROBOROS_REVIEW_ENFORCEMENT": "blocking",
            "PYTHONUNBUFFERED": "1",
        })
        if self.model:
            env.update({
                "OUROBOROS_MODEL": self.model,
                "OUROBOROS_MODEL_LIGHT": self.model,
                "OUROBOROS_MODEL_FALLBACKS": self.model,
                "OUROBOROS_SUBAGENTS": single_model_subagents_setting(self.model),
            })

        cmd = [
            self.ouroboros_bin,
            "run",
            "--url",
            self.ouroboros_url,
            "--memory-mode",
            "empty",
            "--quiet",
            *(["--disable-tools", self.disable_tools] if self.disable_tools else []),
            *([ "--attach", screenshot_path ] if screenshot_path else []),
            # E2BIG hygiene (C5): the per-step prompt (a11y tree + history) can be
            # huge; it already lives on disk above, so it travels as a file, never
            # as an argv tail.
            "--prompt-file",
            str(prompt_path),
        ]
        timed_out = False
        try:
            completed = subprocess.run(
                cmd,
                cwd=str(self.repo_dir),
                env=env,
                text=True,
                capture_output=True,
                timeout=self.timeout_sec,
            )
            returncode = completed.returncode
            stdout = completed.stdout or ""
            stderr = completed.stderr or ""
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            returncode = 124
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
            if isinstance(stdout, bytes):
                stdout = stdout.decode("utf-8", errors="replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", errors="replace")
            stderr = (stderr + "\n" if stderr else "") + (
                f"OSWorld adapter: Ouroboros step timed out after {self.timeout_sec}s"
            )
        (self.result_dir / f"ouroboros_step_{step:03d}.stdout.txt").write_text(stdout, encoding="utf-8")
        (self.result_dir / f"ouroboros_step_{step:03d}.stderr.txt").write_text(stderr, encoding="utf-8")

        payload = _json_from_text(stdout.strip())
        response = str(payload.get("response") or stdout.strip() or stderr.strip() or "")
        note = str(payload.get("notes") or "").strip()
        if note:
            self.notes.append(note[:1000])
        raw_actions = payload.get("actions")
        _known_kinds = {"done", "finish", "fail", "wait", "shell", "click", "type", "hotkey", "key", "python"}
        actions = []
        unknown_kinds: list[str] = []
        if isinstance(raw_actions, list):
            for item in raw_actions:
                translated = _normalize_structured_action(item)
                if translated.strip():
                    actions.append(translated)
                elif isinstance(item, dict):
                    k = str(item.get("type") or item.get("action") or "").strip().lower()
                    if k and k not in _known_kinds:
                        unknown_kinds.append(k)
        if unknown_kinds:
            # Feed unknown/dropped action types back to the model (was a silent
            # drop) so it stops re-emitting them and picks a supported action.
            self.notes.append(
                f"[adapter] dropped unsupported action type(s) {sorted(set(unknown_kinds))}; "
                "use only the supported action objects listed in the directive."
            )
        if returncode != 0:
            response = (
                f"Ouroboros step timed out after {self.timeout_sec}s: {response}"
                if timed_out
                else f"ouroboros exited {returncode}: {response}"
            )
            actions = actions or ["WAIT"]
        actions = [action.upper() if action.upper() in SPECIAL_ACTIONS else action for action in actions]
        actions = actions or ["WAIT"]
        if self.screenshot_check_only and "DONE" not in actions and "FAIL" not in actions:
            actions = ["WAIT"]

        # Terminal-message capture (the cu_bridge sample-60 defect: agents that
        # answered "chat-style" left final_answer empty and the run's own
        # objective ledger degraded to not_evaluated). When the agent ends the
        # episode, persist its explicit final_answer — falling back to the
        # terminal response text — so the audit trail always carries the
        # agent's answer even though official scoring stays VM-state-only.
        if response.strip():
            self.last_response = response.strip()
        if "DONE" in actions or "FAIL" in actions:
            self.terminal_action = "FAIL" if "FAIL" in actions else "DONE"
            explicit = str(payload.get("final_answer") or "").strip()
            self.final_answer = explicit or response.strip()

        debug = {
            "step": step,
            "returncode": returncode,
            "timed_out": timed_out,
            "screenshot_upload_path": screenshot_path,
            "screenshot_file": local_screenshot,
            "payload": payload,
            "normalized_actions": actions,
        }
        return response, actions, debug

    def record_action(self, *, action: str, response: str, reward: float, done: bool, info: dict[str, Any]) -> None:
        self.history.append({
            "action": action,
            "response": response,
            "reward": reward,
            "done": done,
            "info": info,
        })


def _write_task_records(config: TaskRecordConfig) -> dict[str, Any]:
    details = dict(config.extra or {})
    outcome = {
        "ok": config.status == "completed",
        "task_id": config.example_id,
        "domain": config.domain,
        "reward": config.reward,
        "steps": config.steps,
        "status": config.status,
        "reason_code": config.reason_code,
        "error": config.error,
        "result_dir": str(config.run_dir),
        **details,
    }
    write_json(config.run_dir / "task_outcome.json", outcome)
    # Amend the ADMITTED manifest IN PLACE so the finalization seam writes one dict that
    # carries both these late facts and the run's terminal outcome. Rebuilding a manifest
    # here (the old fallback, with `require_clean=False`) recorded a WAIVED gate on the very
    # path where the real gate had refused the run.
    config.base_manifest.update(amend_task_manifest(
        config.base_manifest,
        output_paths={
            "task_outcome": str(config.run_dir / "task_outcome.json"),
            "traj": str(config.run_dir / "traj.jsonl"),
            "task_run_manifest": str(config.run_dir / "task_run_manifest.json"),
        },
        extra=details,
    ))
    # The manifest itself is NOT published here. Every caller runs INSIDE an active
    # `finalize_run_manifest` over this very path, and the seam merges the terminal
    # outcome/exit_code/refusal into this same retained dict only when its context EXITS —
    # so a write here publishes a pre-merge record (for a refusal, the admission seam's
    # generic payload saying exit_code 1 while the process will exit 2) that a concurrent
    # reader can observe and an interruption makes durable. The seam writes this path on
    # every exit path already, so the write was pure duplication with a window attached.
    # Enforced for the whole family by launcher_audit Invariant C.
    append_result_index(
        config.result_root,
        task_result_row(
            benchmark="osworld",
            instance_id=config.example_id,
            status=config.status,
            reason_code=config.reason_code,
            official_eval_status="completed" if config.reward is not None else "not_run",
            output_paths={
                "task_outcome": str(config.run_dir / "task_outcome.json"),
                "task_run_manifest": str(config.run_dir / "task_run_manifest.json"),
                "traj": str(config.run_dir / "traj.jsonl"),
            },
            error=config.error,
            details={"domain": config.domain, "reward": config.reward, "steps": config.steps, **details},
        ),
    )
    return outcome


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--osworld-root", default=DEFAULT_OSWORLD_ROOT)
    parser.add_argument("--provider_name", default="vmware", help=f"VM provider; this adapter supports: {', '.join(SUPPORTED_PROVIDERS)}")
    parser.add_argument("--path_to_vm", default=DEFAULT_VM)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--result_dir", default="results/ouroboros_step_agent")
    parser.add_argument(
        "--observation_type",
        choices=["screenshot", "screenshot_a11y_tree"],
        default="screenshot_a11y_tree",
        help="official observation_type path segment; also controls require_a11y_tree",
    )
    parser.add_argument("--repo-dir", default=DEFAULT_REPO)
    parser.add_argument("--data-dir", default=DEFAULT_DATA)
    parser.add_argument("--settings-path", default=DEFAULT_SETTINGS)
    parser.add_argument("--ouroboros-bin", default=DEFAULT_OUROBOROS_BIN)
    parser.add_argument("--ouroboros-url", default="http://127.0.0.1:8765",
                        help="Ouroboros server URL. The default is the LIVE desktop server; a real bench "
                             "run must point at an isolated server (see --allow-live-server).")
    parser.add_argument("--allow-scaffold-mismatch", action="store_true",
                        help="explicit ablation opt-in: run even when the target server's effective "
                             "settings differ from the disclosed scaffold defaults (recorded in the "
                             "preflight details; the run is then NOT comparable to default-scaffold runs).")
    parser.add_argument("--allow-live-server", action="store_true",
                        help="explicit opt-in to run against the default desktop server URL "
                             "(http://127.0.0.1:8765). Without it, real runs refuse the default: every "
                             "step submits tasks/screenshots into whichever server answers there, and a "
                             "LIVE data/ root must never absorb bench writes. Start an isolated server "
                             "on another port instead.")
    parser.add_argument("--model", default="anthropic/claude-opus-4-7")
    # OSWorld 2.0 protocol budget (official launch scripts pass 500; the paper
    # reports 150/300/500-step curves). The old 15/50 conventions are legacy.
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument(
        "--disable-tools",
        default="claude_code_edit",
        help="comma-separated Ouroboros tools withheld per step (bench scaffold default: claude_code_edit)",
    )
    parser.add_argument("--step_timeout_sec", type=int, default=240)
    parser.add_argument("--sleep_after_execution", type=float, default=1.0)
    parser.add_argument("--wait_after_reset_sec", type=float, default=8.0)
    parser.add_argument("--startup_timeout_sec", type=int, default=600)
    parser.add_argument("--reset_retries", type=int, default=3)
    parser.add_argument("--startup_retry_sleep_sec", type=float, default=5.0)
    parser.add_argument("--max_obs_chars", type=int, default=12000)
    parser.add_argument("--screenshot-check-only", action="store_true")
    parser.add_argument("--show-vm", action="store_true")
    parser.add_argument("--allow-dirty-seed", action="store_true",
                        help="run even when this Ouroboros checkout is dirty or its git identity is "
                             "unreadable (default: fail closed before the VM boots). Recorded in the "
                             "manifest; a dirty seed makes the run's provenance irreproducible.")
    return parser


def main() -> int:
    # NOTHING but argument parsing and pure local derivation until `admit_step_loop_run`
    # below. `_ensure_vmrun_on_path()` probes the filesystem for `vmrun` and mutates $PATH,
    # `_install_optional_dependency_stubs()` mutates `sys.modules`, and the `sys.path` insert
    # is process state: all three moved into `_run_step_loop`, i.e. after the run is on disk.
    args = build_arg_parser().parse_args()

    osworld_root = Path(args.osworld_root).expanduser().resolve(strict=False)
    if _is_default_desktop_server(args.ouroboros_url) and not args.allow_live_server:
        raise SystemExit(
            "refusing the default desktop server port (8765 on a loopback host): bench steps would "
            "write tasks/screenshots into the LIVE Ouroboros data root. Point --ouroboros-url at an "
            "isolated server (fresh OUROBOROS_DATA_DIR, non-default port), or pass --allow-live-server "
            "for an explicit local-debug run."
        )
    task_path = Path(args.task).expanduser()
    if not task_path.is_absolute():
        task_path = osworld_root / task_path
    domain = task_path.parent.name
    example_id = task_path.stem
    result_root = Path(args.result_dir).expanduser()
    if not result_root.is_absolute():
        result_root = osworld_root / result_root
    # ASSERT, not ensure (see run_cu_bridge_agent): no directory before the manifest.
    result_root = assert_outside_repo(result_root, Path(args.repo_dir).expanduser().resolve(strict=False))
    # Official example dir layout consumed by upstream show_result.py:
    # <result_dir>/<action_space>/<observation_type>/<model>/<domain>/<example_id>
    run_dir = (
        result_root
        / "pyautogui"
        / args.observation_type
        / _safe_slug(args.model or "default")
        / domain
        / example_id
    )
    # NO mkdir here: ADMISSION IS THE OUTER BOUNDARY (v6.75.0), so nothing touches the
    # filesystem before the manifest is persisted. The atomic manifest write creates the tree.
    manifest_path = run_dir / "task_run_manifest.json"

    repo_dir = Path(args.repo_dir).expanduser().resolve(strict=False)
    data_dir = Path(args.data_dir).expanduser().resolve(strict=False)
    settings_path = Path(args.settings_path).expanduser().resolve(strict=False)
    # ADMISSION: build the manifest, WRITE it, then let the seed gate enforce — before the
    # preflight, the VM boot and every paid step. `_write_task_records` amends this same dict
    # and `finalize_run_manifest` records how the run ended on EVERY exit path.
    #
    # The gate fails CLOSED (owner Q19). Nothing has been spent at this point, so report it as
    # this launcher's own typed refusal — `blocked/seed_gate_failed` with a ledger row —
    # instead of a bare traceback, over the manifest `admit_benchmark_run` already persisted.
    try:
        base_manifest = admit_step_loop_run(
            manifest_path,
            result_root=result_root, repo_dir=repo_dir, settings_path=settings_path,
            example_id=example_id, require_clean=not args.allow_dirty_seed,
        )
    except BenchmarkAdmissionRefused as exc:
        error = f"{type(exc).__name__}: {exc}"
        refused = exc.manifest
        # exit_code 2 is what this launcher REALLY exits with for a blocked run; the seam's
        # generic refusal payload says 1, and a record that disagrees with the process status
        # is the exact class of bug the parity test exists to catch.
        with finalize_run_manifest(manifest_path, refused, outcome="refused", exit_code=2) as final:
            final["refusal"] = {**((refused.get("extra") or {}).get("refusal") or {}),
                                "exit_code": 2}
            outcome = _write_task_records(TaskRecordConfig(
                run_dir=run_dir,
                result_root=result_root,
                repo_dir=repo_dir,
                settings_path=settings_path,
                example_id=example_id,
                domain=domain,
                reward=None,
                steps=0,
                status="blocked",
                reason_code="seed_gate_failed",
                base_manifest=refused,
                error=error,
                extra={"allow_dirty_seed": bool(args.allow_dirty_seed),
                       "seed_gate_error": error},
            ))
        print(json.dumps(outcome, ensure_ascii=False, indent=2))
        return 2
    base_manifest["extra"] = {**(base_manifest.get("extra") or {}),
                              "allow_dirty_seed": bool(args.allow_dirty_seed)}
    with finalize_run_manifest(manifest_path, base_manifest) as final:
        return _run_step_loop(args, final, StepLoopPaths(
            run_dir=run_dir, result_root=result_root, repo_dir=repo_dir, data_dir=data_dir,
            settings_path=settings_path, osworld_root=osworld_root, task_path=task_path,
            domain=domain, example_id=example_id, base_manifest=base_manifest,
        ))


@dataclass
class StepLoopPaths:
    """Resolved paths + the ADMITTED manifest handed to the post-admission body."""

    run_dir: Path
    result_root: Path
    repo_dir: Path
    data_dir: Path
    settings_path: Path
    osworld_root: Path
    task_path: Path
    domain: str
    example_id: str
    base_manifest: dict[str, Any]


def _run_step_loop(args: argparse.Namespace, final: dict[str, Any],
                   paths: StepLoopPaths) -> int:
    """Everything AFTER admission: preflight, VM boot, the step loop, official evaluate.

    Split out of `main()` so the admission function's pre-admission statements stay trivially
    auditable (the seam meta-test walks them with `ast` and denies every filesystem/docker/
    subprocess/network call there). `final` is the finalization seam's mutable record.
    """
    run_dir, result_root = paths.run_dir, paths.result_root
    repo_dir, data_dir, settings_path = paths.repo_dir, paths.data_dir, paths.settings_path
    domain, example_id = paths.domain, paths.example_id
    base_manifest = paths.base_manifest
    osworld_root, task_path = paths.osworld_root, paths.task_path
    # Process/environment preparation belongs HERE, after the persisted admission boundary:
    # each of these probes the filesystem or mutates process state, so a refusal in any of
    # them must land on a run that already has a durable record.
    _ensure_vmrun_on_path()
    _install_optional_dependency_stubs()
    sys.path.insert(0, str(osworld_root))
    preflight = _preflight(PreflightConfig(
        osworld_root=osworld_root,
        task_path=task_path,
        path_to_vm=args.path_to_vm,
        repo_dir=repo_dir,
        data_dir=data_dir,
        settings_path=settings_path,
        result_root=result_root,
        ouroboros_url=args.ouroboros_url,
        model=args.model,
        provider_name=args.provider_name,
        allow_scaffold_mismatch=bool(args.allow_scaffold_mismatch),
    ))
    write_json(run_dir / "preflight.json", preflight)
    # Bind the run to the actor the TARGET server actually exposed. This remains
    # true for an explicit --allow-scaffold-mismatch ablation: the manifest records
    # the drifted actor, never the nicer local declaration.
    _bind_target_actor(base_manifest, preflight)
    # The attestation record travels with the manifest, not just the preflight file, so a
    # scored row can always be attributed to a runtime version + commit.
    base_manifest["extra"]["runtime_attestation"] = (
        (preflight.get("details") or {}).get("runtime_attestation") or {})
    if not preflight["ok"]:
        # The documented contract is that a launcher NAMES the exact attestation reason in its
        # typed refusal, not just that a preflight failed: `runtime_skew` and "the task JSON is
        # missing" are different operator actions. When the attestation is what refused, its
        # reason and stage are the refusal; otherwise the generic preflight one stands.
        attestation = base_manifest["extra"]["runtime_attestation"]
        attestation_reason = (str(attestation.get("reason") or "")
                              if isinstance(attestation, dict) and not attestation.get("ok", True)
                              else "")
        refusal = ({"stage": "runtime_attestation", "reason": attestation_reason, "exit_code": 2}
                   if attestation_reason
                   else {"stage": "preflight", "reason": "preflight_failed", "exit_code": 2})
        final.update({"outcome": "blocked", "exit_code": 2, "refusal": refusal})
        outcome = _write_task_records(TaskRecordConfig(
            run_dir=run_dir,
            result_root=result_root,
            repo_dir=repo_dir,
            settings_path=settings_path,
            example_id=example_id,
            domain=domain,
            reward=None,
            steps=0,
            status="blocked",
            reason_code="preflight_failed",
            base_manifest=base_manifest,
            error="; ".join(preflight["failures"]),
            extra={"preflight": preflight},
        ))
        print(json.dumps(outcome, ensure_ascii=False, indent=2))
        return 2
    example = json.loads(task_path.read_text(encoding="utf-8"))
    example_id = str(example.get("id") or task_path.stem)
    (run_dir / "task.json").write_text(json.dumps(example, ensure_ascii=False, indent=2), encoding="utf-8")
    from desktop_env.desktop_env import DesktopEnv

    env = None
    agent = OuroborosStepAgent(StepAgentConfig(
        ouroboros_bin=args.ouroboros_bin,
        ouroboros_url=args.ouroboros_url,
        repo_dir=repo_dir,
        data_dir=data_dir,
        settings_path=settings_path,
        result_dir=run_dir,
        task_id=example_id,
        model=args.model,
        timeout_sec=args.step_timeout_sec,
        max_obs_chars=args.max_obs_chars,
        screenshot_check_only=args.screenshot_check_only,
        disable_tools=args.disable_tools,
    ))

    try:
        # The constructor boots the VM, so a failed boot needs the SAME retry+cleanup
        # discipline the reset loop already has (see construct_desktop_env).
        env = construct_desktop_env(
            DesktopEnv,
            attempts=max(1, int(args.reset_retries)),
            deadline=time.time() + max(1, int(args.startup_timeout_sec)),
            retry_sleep_sec=max(0.1, float(args.startup_retry_sleep_sec)),
            provider_name=args.provider_name,
            path_to_vm=args.path_to_vm,
            action_space="pyautogui",
            screen_size=(1920, 1080),
            headless=not args.show_vm,
            os_type="Ubuntu",
            require_a11y_tree=args.observation_type == "screenshot_a11y_tree",
        )
        obs = _initial_observation_with_retries(
            env,
            example,
            startup_timeout_sec=args.startup_timeout_sec,
            reset_retries=args.reset_retries,
            wait_after_reset_sec=max(0.0, args.wait_after_reset_sec),
            retry_sleep_sec=max(0.1, args.startup_retry_sleep_sec),
            run_dir=run_dir,
        )
        agent.reset()
        instruction = str(example["instruction"])
        done = False
        step_idx = 0
        while not done and step_idx < args.max_steps:
            response, actions, debug = agent.predict(instruction, obs, max_steps=args.max_steps)
            (run_dir / f"debug_step_{step_idx + 1:03d}.json").write_text(
                json.dumps(debug, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            for action_index, action in enumerate(actions, start=1):
                ts = _dt.datetime.now().strftime("%Y%m%d@%H%M%S%f")
                obs, reward, done, info = env.step(action, args.sleep_after_execution)
                agent.record_action(
                    action=action,
                    response=response,
                    reward=float(reward),
                    done=bool(done),
                    info=dict(info or {}),
                )
                # Official convention (lib_run_single.py): save the post-action
                # screenshot for the last action of a step (or on done) under
                # step_<n>_<ts>.png and reference it from the traj row.
                screenshot_file = None
                if action_index == len(actions) or done:
                    shot = obs.get("screenshot") if isinstance(obs, dict) else None
                    if isinstance(shot, (bytes, bytearray)) and shot:
                        screenshot_file = f"step_{step_idx + 1}_{ts}.png"
                        (run_dir / screenshot_file).write_bytes(bytes(shot))
                with (run_dir / "traj.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "step_num": step_idx + 1,
                        "action_timestamp": ts,
                        "action": action,
                        "response": response,
                        "reward": reward,
                        "done": done,
                        "info": info,
                        "screenshot_file": screenshot_file,
                        "adapter_debug": debug,
                    }, ensure_ascii=False, default=str) + "\n")
                if done:
                    break
            step_idx += 1
            if args.screenshot_check_only:
                break

        reward = _persist_evaluation_result(env.evaluate(), run_dir)
        evaluator_cfg = example.get("evaluator") if isinstance(example, dict) else None
        evaluator_func = evaluator_cfg.get("func") if isinstance(evaluator_cfg, dict) else None
        outcome = _write_task_records(TaskRecordConfig(
            run_dir=run_dir,
            result_root=result_root,
            repo_dir=repo_dir,
            settings_path=settings_path,
            example_id=example_id,
            domain=domain,
            reward=reward,
            steps=step_idx,
            status="completed",
            reason_code="official_evaluate",
            base_manifest=base_manifest,
            extra={
                "screenshot_check_only": bool(args.screenshot_check_only),
                "final_answer": agent.final_answer or agent.last_response,
                "terminal_action": agent.terminal_action or "max_steps_exhausted",
                "infeasible_declared": agent.terminal_action == "FAIL",
                "evaluator_func": evaluator_func,
                "observation_type": args.observation_type,
                "max_steps": args.max_steps,
            },
        ))
        print(json.dumps(outcome, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:  # noqa: BLE001 - denominator-preserving adapter failure
        error = f"{type(exc).__name__}: {exc}"
        final.update({"outcome": "adapter_error", "exit_code": 1,
                      "error": {"type": type(exc).__name__, "message": str(exc)[:4000]}})
        outcome = _write_task_records(TaskRecordConfig(
            run_dir=run_dir,
            result_root=result_root,
            repo_dir=repo_dir,
            settings_path=settings_path,
            example_id=example_id,
            domain=domain,
            reward=None,
            steps=locals().get("step_idx", 0),
            status="adapter_error",
            reason_code=type(exc).__name__,
            base_manifest=base_manifest,
            error=error,
            extra={
                "final_answer": agent.final_answer or agent.last_response,
                "terminal_action": agent.terminal_action,
                "infeasible_declared": agent.terminal_action == "FAIL",
            },
        ))
        print(json.dumps(outcome, ensure_ascii=False, indent=2))
        return 1
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
