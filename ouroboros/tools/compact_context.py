"""LLM-requested tool-history compaction trigger."""

from __future__ import annotations

import logging
from typing import List

from ouroboros.tools.registry import ToolEntry

log = logging.getLogger(__name__)


def _compact_context(ctx, keep_last_n: int = 6, **kwargs) -> str:
    """Store a pending compaction request for the next loop round."""

    keep_last_n = max(2, min(keep_last_n, 20))

    ctx._pending_compaction = keep_last_n

    return (
        f"✅ Context reclaim scheduled: keeping the last {keep_last_n} completed tool units raw. "
        "After a useful selection is known, Ouroboros checkpoints the exact actor-visible "
        "transcript and replaces only fully covered older atomic units with active-context "
        "summaries carrying checkpoint/CAS provenance. The raw evidence remains retrievable "
        "from those references. This takes effect on the next round."
    )


def get_tools() -> List[ToolEntry]:
    return [
        ToolEntry(
            name="compact_context",
            schema={
                "name": "compact_context",
                "description": (
                    "Request complete-input context reclaim for old completed tool units. "
                    "Keeps recent N units raw. A selected older assistant tool call and all of its "
                    "contiguous matching results stay atomic; Ouroboros checkpoints their exact "
                    "actor-visible bytes before replacing them with summaries whose metadata points "
                    "to the checkpoint/CAS evidence. Active context becomes summarized; raw evidence "
                    "remains retrievable through the recorded provenance."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "keep_last_n": {
                            "type": "integer",
                            "description": "Number of recent completed atomic tool units to keep raw (default 6, range 2-20). Lower = more reclaim.",
                            "default": 6,
                        },
                    },
                    "required": [],
                },
            },
            handler=_compact_context,
            timeout_sec=15,
        ),
    ]
