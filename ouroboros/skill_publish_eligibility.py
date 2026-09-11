"""Passive skill→hub visibility and managed-task admission facts.

The passive skill index deliberately does not run Betterleaks, so it cannot claim
that the current bytes are publication-ready.  It does know whether the action is
visible and whether an ordinary managed task may start to repair/review/publish the
skill.  The selected-skill preflight owns the later scanner-backed readiness fact.

Imports ONLY the config-level review-status constants — no skill loading, no gateway
code — so it stays a thin, dependency-light predicate.
"""

from __future__ import annotations

from typing import Any, Dict

from ouroboros.skill_review_status import STATUS_CLEAN, STATUS_WARNINGS, normalize_skill_review_status

# Sources whose payload may be submitted to the hub (native without a marker is
# handled by skill_loader reclassification, not here).
PUBLISHABLE_SOURCES = ("external", "self_authored", "user_repo", "ouroboroshub", "clawhub")
# A no-blocker review — clean OR advisory-only warnings (warnings are disclosed in the
# PR body). This is the SSOT both the backend gate and the UI predicate use.
PUBLISHABLE_STATUSES = frozenset({STATUS_CLEAN, STATUS_WARNINGS})


def submit_hub_eligibility(
    *,
    source: str,
    review_status: str,
    review_profile: str = "",
    review_stale: bool = False,
    github_token_configured: bool = False,
) -> Dict[str, Any]:
    """Return passive publication visibility and task-admission facts.

    ``disabled`` is only the compatibility projection of
    ``not task_start_allowed``.  Repairable review states keep the ordinary task
    available; the authoritative tool still enforces a fresh publishable review
    before any outbound effect.
    """
    src = str(source or "native").lower()
    # Normalize the verdict the SAME way the backend publish gate does, so a raw verdict
    # (e.g. 'pass'/'advisory_pass') and the normalized form ('clean'/'warnings') agree.
    review_status = normalize_skill_review_status(review_status)
    if src not in PUBLISHABLE_SOURCES:
        return {
            "visible": False,
            "publication_ready": False,
            "task_start_allowed": False,
            "disabled": True,
            "state": "hard_block",
            "reason": "",
        }
    if not github_token_configured:
        return {
            "visible": True,
            "publication_ready": False,
            "task_start_allowed": False,
            "disabled": True,
            "state": "hard_block",
            "reason": "Configure GITHUB_TOKEN in Settings → Secrets",
        }
    if str(review_profile or "") == "owner_attested":
        # Owner-attested skills SKIPPED the LLM review; a public submission needs the
        # full tri-model review, so the hub refuses them.
        reason = "Owner-attested skills need a full LLM review before publication"
    elif review_stale:
        reason = "Skill needs a fresh review before publication"
    elif str(review_status or "") not in PUBLISHABLE_STATUSES:
        reason = "Skill needs review work before publication"
    else:
        # A passive inventory read has not scanned the selected immutable bytes.
        reason = "Open Publish to check the current bytes before publication"
    return {
        "visible": True,
        "publication_ready": False,
        "task_start_allowed": True,
        "disabled": False,
        "state": "needs_attention",
        "reason": reason,
    }
