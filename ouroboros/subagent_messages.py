"""Compact, durable identity for owner-visible subagent messages."""

from __future__ import annotations

from typing import Any, Dict, Mapping


SUBAGENT_MESSAGE_FIELDS: tuple[str, ...] = (
    "subagent_event",
    "subagent_task_id",
    "root_task_id",
    "parent_task_id",
    "delegation_role",
    "subagent_role",
    "write_surface",
    "task_group_id",
    "model_lane",
    "effective_model_lane",
    "model",
    "executor_route",
)


def subagent_message_meta(
    record: Mapping[str, Any] | None,
    *,
    task_id: str = "",
    event: str = "",
) -> Dict[str, Any]:
    """Return the bounded lineage/execution facts that identify a child message.

    Task rows and task results keep some fields at the top level and some in
    ``metadata``. Reading both here gives producers, supervisor recovery, and
    history replay one projection without persisting the whole task record.
    """
    source = record if isinstance(record, Mapping) else {}
    nested = source.get("metadata")
    metadata = nested if isinstance(nested, Mapping) else {}
    raw_constraint = source.get("task_constraint") or metadata.get("task_constraint")
    constraint = raw_constraint if isinstance(raw_constraint, Mapping) else {}

    def first(*keys: str) -> str:
        for key in keys:
            for candidate in (source.get(key), metadata.get(key)):
                value = str(candidate or "").strip()
                if value:
                    return value
        return ""

    if first("delegation_role").lower() != "subagent":
        return {}
    child_id = str(task_id or first("subagent_task_id", "id", "task_id")).strip()
    meta: Dict[str, Any] = {
        "subagent_task_id": child_id,
        "root_task_id": first("root_task_id"),
        "parent_task_id": first("parent_task_id"),
        "delegation_role": "subagent",
        "subagent_role": first("subagent_role", "role"),
        "write_surface": first("write_surface") or str(constraint.get("surface") or "").strip(),
        "task_group_id": first("task_group_id"),
        "model_lane": first("requested_model_lane", "model_lane"),
        "effective_model_lane": first("effective_model_lane"),
        "model": first("model"),
        "executor_route": first("executor_route"),
    }
    if event:
        meta["subagent_event"] = str(event)
    return meta
