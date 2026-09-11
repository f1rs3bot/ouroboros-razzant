"""In-memory, process-local registry for active direct and ephemeral chat activities.

Tracks in-flight direct conversational turns and ephemeral decision turns so
the Gateway (/api/state) and WebSocket activity pipeline have authoritative,
thread-safe visibility into in-progress work without creating spurious queue records.
"""

from __future__ import annotations

import logging
import pathlib
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

log = logging.getLogger(__name__)


@dataclass
class DirectActivityEntry:
    activity_id: str
    chat_id: int
    project_id: str = ""
    client_message_id: str = ""
    kind: str = "direct_chat"  # "direct_chat" | "ephemeral_decision"
    phase: str = "thinking"
    started_at: float = field(default_factory=time.time)
    origin_message_ref: Dict[str, Any] = field(default_factory=dict)
    model_wait_owner: Any = field(default=None, repr=False, compare=False)

    actor: Any = field(default=None, repr=False, compare=False)

    def to_dict(self) -> Dict[str, Any]:
        row = {
            "activity_id": self.activity_id,
            "chat_id": self.chat_id,
            "project_id": self.project_id,
            "client_message_id": self.client_message_id,
            "kind": self.kind,
            "phase": self.phase,
            "started_at": self.started_at,
        }
        owner = self.model_wait_owner
        if owner is not None and not owner.closed:
            row.update(model_waits=owner.snapshot()["model_waits"], task_attempt=owner.attempt)
        return row


class DirectActivityRegistry:
    """Thread-safe registry for active direct-chat and ephemeral-decision turns."""

    def __init__(self) -> None:
        self._lock = threading.Condition()
        self._activities: Dict[str, DirectActivityEntry] = {}

    def register(
        self,
        activity_id: str,
        chat_id: int,
        *,
        client_message_id: str = "",
        project_id: str = "",
        kind: str = "direct_chat",
        phase: str = "thinking",
        origin_message_ref: Optional[Dict[str, Any]] = None,
        actor: Any = None,
    ) -> DirectActivityEntry:
        aid = str(activity_id or "").strip()
        if not aid:
            raise ValueError("activity_id cannot be empty")
        entry = DirectActivityEntry(
            activity_id=aid,
            chat_id=int(chat_id or 0),
            project_id=str(project_id or ""),
            client_message_id=str(client_message_id or ""),
            kind=str(kind or "direct_chat"),
            phase=str(phase or "thinking"),
            started_at=time.time(),
            origin_message_ref=dict(origin_message_ref or {}),
            actor=actor,
        )
        with self._lock:
            self._activities[aid] = entry
        log.debug("Registered direct activity: %s (chat_id=%s, kind=%s)", aid, chat_id, kind)
        return entry

    def unregister(self, activity_id: str) -> Optional[DirectActivityEntry]:
        aid = str(activity_id or "").strip()
        with self._lock:
            entry = self._activities.pop(aid, None)
            self._lock.notify_all()
        if entry:
            log.debug("Unregistered direct activity: %s (chat_id=%s)", aid, entry.chat_id)
        return entry

    def snapshot(self, chat_id: Optional[int] = None) -> List[Dict[str, Any]]:
        with self._lock:
            entries = list(self._activities.values())
        # Wait snapshots take the owner's lock; never nest it inside this lock.
        return [e.to_dict() for e in entries if chat_id is None or e.chat_id == int(chat_id)]

    @contextmanager
    def bind_model_wait(self, owner: Any) -> Iterator[None]:
        """Attach a turn's existing wait owner without minting another activity."""
        with self._lock:
            entry = self._activities.get(owner.task_id)
            if entry is not None and entry.kind == "ephemeral_decision":
                entry.model_wait_owner = owner
        try:
            yield
        finally:
            with self._lock:
                if entry is not None and entry.model_wait_owner is owner:
                    entry.model_wait_owner = None

    def ephemeral_model_wait(self, drive_root: Any, task_id: str) -> Any:
        """A live ephemeral call, not ordinary task/cancel/Project authority."""
        entry = self.get(task_id)
        owner = entry.model_wait_owner if entry is not None and entry.kind == "ephemeral_decision" else None
        if (owner is not None and not owner.closed and owner.task_id == str(task_id)
                and pathlib.Path(owner.canonical_root).resolve() == pathlib.Path(drive_root).resolve()):
            return owner
        return None

    def get(self, activity_id: str) -> Optional[DirectActivityEntry]:
        aid = str(activity_id or "").strip()
        with self._lock:
            return self._activities.get(aid)

    def actors(self) -> List[DirectActivityEntry]:
        """Private actor handles; never include execution objects in UI snapshots."""
        with self._lock:
            return list(self._activities.values())

    def wait_until_empty(self, timeout: float) -> List[str]:
        """Wait for whole executions, including preparation and post-task work."""
        with self._lock:
            self._lock.wait_for(lambda: not self._activities, timeout=max(0.0, timeout))
            return list(self._activities)

    def clear(self) -> None:
        """Clear registry — primarily for tests and process resets."""
        with self._lock:
            self._activities.clear()
            self._lock.notify_all()


# Global process-local singleton
_DIRECT_ACTIVITY_REGISTRY = DirectActivityRegistry()


def get_direct_activity_registry() -> DirectActivityRegistry:
    return _DIRECT_ACTIVITY_REGISTRY


@contextmanager
def track_direct_activity(
    activity_id: str,
    chat_id: int,
    *,
    client_message_id: str = "",
    project_id: str = "",
    kind: str = "direct_chat",
    phase: str = "thinking",
    origin_message_ref: Optional[Dict[str, Any]] = None,
) -> Iterator[DirectActivityEntry]:
    registry = get_direct_activity_registry()
    entry = registry.register(
        activity_id,
        chat_id,
        client_message_id=client_message_id,
        project_id=project_id,
        kind=kind,
        phase=phase,
        origin_message_ref=origin_message_ref,
    )
    try:
        yield entry
    finally:
        registry.unregister(activity_id)
