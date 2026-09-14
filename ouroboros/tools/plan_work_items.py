"""Work-item reference authority for reviewed evolution closure.

``plan_task`` stays domain-neutral: the field below is one bounded list of
reference strings. Only evolution interprets the ``ibl-*`` namespace, and the
per-cycle transaction is the closure authority — boot reconciliation never
reads mutable plan state. Absence and an explicit empty list are distinct:
omitted preserves the legacy post-task nomination; ``[]`` replaces it.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

def plan_fingerprint(goal: str, plan: str, spec: dict, manifest_hash: str, constitutional: bool) -> str:
    """Identity of one review request: goal, prose, canonical spec, evidence identity,
    the constitutional fact — never the exploration log (it changes no obligation)."""
    from hashlib import sha256
    import json

    payload = {"goal": goal, "plan": plan, "spec": spec, "evidence_manifest_hash": manifest_hash,
               "constitutional": bool(constitutional)}
    return sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()


WORK_ITEM_ID_LIMIT = 20


def evidence_deny_paths(ctx: Any) -> list[str]:
    """Plan-evidence deny paths: runtime data and the live settings file are a boundary."""
    from ouroboros import config

    out: list[str] = []
    for value in (getattr(config, "SETTINGS_PATH", ""), getattr(config, "DATA_DIR", "")):
        text = str(value or "").strip()
        if text:
            out.append(text)
    try:
        from ouroboros.tool_access import canonical_data_root

        drive = canonical_data_root(ctx)
        if drive:
            out.append(str(drive))
    except Exception:
        pass
    return out

_MISSING = object()


def _sanitize_ref(value: Any) -> str:
    text = str(value or "").strip()
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in text)[:80]


def normalize_work_item_refs(raw: Any) -> tuple[Optional[List[str]], Optional[str]]:
    """Normalize the optional field.

    Returns ``(None, None)`` when omitted (no modern binding), ``([], None)``
    for an explicit empty replacement, and deduplicated refs otherwise. Any
    malformed input is one typed error; a typo must not silently drop legacy
    fallback or suppress an item.
    """
    if raw is _MISSING:
        return None, None
    if not isinstance(raw, list):
        return None, "work_item_refs: must be an array of strings"
    refs: List[str] = []
    for index, item in enumerate(raw):
        if not isinstance(item, str):
            return None, f"work_item_refs[{index}]: must be a string"
        ref = _sanitize_ref(item)
        if not ref:
            return None, f"work_item_refs[{index}]: must be non-empty"
        if ref not in refs:
            refs.append(ref)
        if len(refs) > WORK_ITEM_ID_LIMIT:
            return None, f"work_item_refs: at most {WORK_ITEM_ID_LIMIT} references"
    return refs, None


def validate_evolution_work_item_refs(drive_root: Any, refs: List[str]) -> Optional[str]:
    """Evolution-only ``ibl-*`` validation against the current open backlog."""
    if not refs:
        return None
    bad_ns = [ref for ref in refs if not ref.startswith("ibl-")]
    if bad_ns:
        return "work_item_refs for evolution must use ibl-* ids: " + ", ".join(bad_ns)
    try:
        from ouroboros.improvement_backlog import load_backlog_items

        open_ids = {
            str(item.get("id") or "")
            for item in load_backlog_items(drive_root)
            if str(item.get("status") or "open").lower() != "done"
        }
    except Exception as exc:  # fail closed: unreadable source cannot certify
        return f"work_item_refs: open backlog unreadable: {exc}"
    unknown = [ref for ref in refs if ref not in open_ids]
    if unknown:
        return "work_item_refs does not name current open backlog ids: " + ", ".join(unknown)
    return None


def plan_work_item_binding(state: Any) -> Optional[Dict[str, Any]]:
    """Resolve the current exact closed wave's binding, if it exists.

    ``None`` means no modern binding (legacy fallback applies). A dict with an
    empty ``refs`` list is still a binding — the explicit replacement. Absent,
    compact, open, stale, compact-only, unreadable, or mismatched attempts
    never confer authority.
    """
    if not isinstance(state, dict):
        return None
    try:
        from ouroboros.task_results import current_plan_review_wave

        wave = current_plan_review_wave(state)
    except Exception:
        return None
    if not isinstance(wave, dict) or wave.get("compact") or not wave.get("closed"):
        return None
    attempt = state.get("current_attempt") if isinstance(state.get("current_attempt"), dict) else {}
    attempt_fp = str(attempt.get("fingerprint") or "")
    if attempt_fp and attempt_fp != str(wave.get("request_fingerprint") or ""):
        return None
    if "work_item_refs" not in wave:
        return None
    refs = list(wave.get("work_item_refs") or [])
    return {
        "plan_fingerprint": str(wave.get("request_fingerprint") or ""),
        "refs": refs,
    }


def transaction_work_item_binding(tx: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Read a copied commit-time binding from transaction intent."""
    if not isinstance(tx, dict):
        return None
    intent = tx.get("commit_intent") if isinstance(tx.get("commit_intent"), dict) else {}
    binding = intent.get("work_item_binding") if isinstance(intent.get("work_item_binding"), dict) else None
    if not isinstance(binding, dict):
        return None
    return {
        "plan_fingerprint": str(binding.get("plan_fingerprint") or ""),
        "refs": [str(ref) for ref in (binding.get("refs") or [])],
    }
