"""Publication eligibility from durable records, never from response prose."""

import hashlib
import json
from typing import Any, Dict

from core.review_state import aggregate_reviews, content_hash, publish_allowed
from .models import CampaignWorkflow, ContentVersion, ReviewContext

REQUIRED_REVIEWS = ("fact_check", "compliance", "content_quality", "channel_validation")


def build_review_context(
    workflow: CampaignWorkflow, version: ContentVersion
) -> ReviewContext:
    brief = workflow.brief.model_dump(mode="json")
    evidence = [dict(item) for item in workflow.knowledge_snapshot.documents]
    refs = [f"brief:{key}" for key in sorted(brief)]
    refs.extend(
        str(item["document_id"]) for item in evidence if item.get("document_id")
    )
    payload = {
        "workflow_id": workflow.workflow_id,
        "version_id": version.version_id,
        "version": version.version,
        "channel": version.channel,
        "content": version.content,
        "content_hash": content_hash(version.content),
        "citation_ids": version.citation_ids,
        "unresolved_fields": version.unresolved_fields,
        "brief": brief,
        "brief_hash": version.brief_hash,
        "knowledge_snapshot_hash": version.knowledge_snapshot_hash,
        "evidence": evidence,
        "evidence_refs": list(dict.fromkeys(refs)),
        "rejected_constraints": version.rejected_constraints,
    }
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return ReviewContext(**payload, review_context_hash=digest)


def version_binding(version: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "version": version["version"],
        "content_hash": version["content_hash"],
        "review_context_hash": context["review_context_hash"],
    }


def delivery_eligibility(detail: Dict[str, Any]) -> Dict[str, Any]:
    """Return both known business decisions and independent completeness/binding flags."""
    workflow_data = detail.get("workflow", {})
    reasons, states, per_version = [], [], {}
    try:
        workflow = CampaignWorkflow.model_validate(workflow_data)
    except (ValueError, TypeError):
        return {
            "review_status": "incomplete",
            "review_complete": False,
            "approval_status": "pending",
            "publish_allowed": False,
            "bindings_valid": False,
            "reasons": ["invalid_workflow"],
            "versions": {},
        }
    versions = {item["version_id"]: item for item in detail.get("versions", [])}
    contexts = {item["version_id"]: item for item in detail.get("review_contexts", [])}
    if (
        set(workflow.latest_versions) != set(workflow.brief.channels)
        or not workflow.latest_versions
    ):
        reasons.append("missing_latest_channel")
    for channel in workflow.brief.channels:
        vid = workflow.latest_versions.get(channel)
        raw, context = versions.get(vid), contexts.get(vid)
        local, decisions, complete = [], [], True
        binding = None
        if raw is None or context is None:
            local.append("missing_version_or_context")
            complete = False
        else:
            try:
                version = ContentVersion.model_validate(raw)
                expected = build_review_context(workflow, version).model_dump(
                    mode="json"
                )
                if (
                    version.workflow_id != workflow.workflow_id
                    or version.channel != channel
                    or version.brief_hash != workflow.brief.snapshot_hash()
                    or version.knowledge_snapshot_hash
                    != workflow.knowledge_snapshot.snapshot_hash
                    or not version.content_hash
                    or version.content_hash != content_hash(version.content)
                    or context != expected
                ):
                    local.append("version_context_mismatch")
                    complete = False
                binding = version_binding(raw, expected)
            except (ValueError, TypeError, KeyError):
                local.append("invalid_version_context")
                complete = False
        selected = []
        for review_type in REQUIRED_REVIEWS:
            candidates = [
                r
                for r in detail.get("reviews", [])
                if r.get("version_id") == vid
                and r.get("review_type") == review_type
                and context
                and r.get("review_context_hash") == context.get("review_context_hash")
            ]
            review = max(
                candidates,
                key=lambda r: (
                    r.get("attempt", 1),
                    r.get("created_at", ""),
                    r.get("review_id", ""),
                ),
                default=None,
            )
            if review is None:
                complete = False
                local.append("missing_required_review")
                continue
            selected.append(review)
            decision = review.get("decision")
            decisions.append(decision)
            if (
                decision not in {"pass", "changes_required", "blocked"}
                or review.get("error_code")
                or not raw
                or not review.get("content_hash")
                or review["content_hash"] != raw.get("content_hash")
                or review.get("knowledge_snapshot_hash")
                != workflow.knowledge_snapshot.snapshot_hash
                or decision == "blocked"
                and not review.get("evidence")
                or decision == "pass"
                and any(
                    e.get("impact") in {"blocking_issue", "required_change"}
                    for e in review.get("evidence", [])
                )
            ):
                complete = False
                local.append("invalid_required_review")
        state = aggregate_reviews(decisions, complete=complete)
        approvals = [
            a for a in detail.get("approvals", []) if vid in a.get("version_ids", [])
        ]
        latest_approval = max(
            approvals,
            key=lambda a: (a.get("created_at", ""), a.get("committed_revision") or 0),
            default=None,
        )
        approval_valid = bool(
            latest_approval
            and binding
            and latest_approval.get("decision") == "approved"
            and latest_approval.get("version_bindings", {}).get(vid) == binding
            and latest_approval.get("committed_revision")
            and latest_approval["committed_revision"] <= workflow.revision
            and latest_approval.get("request_hash")
        )
        if not approval_valid:
            local.append("missing_valid_approval")
        if state["review_status"] != "pass":
            local.append("review_not_passed")
        if raw and raw.get("status") != "approved":
            local.append("version_not_approved")
        per_version[vid or channel] = {
            **state,
            "binding": binding,
            "approval_valid": approval_valid,
            "reasons": list(dict.fromkeys(local)),
            "reviews": selected,
            "approval": latest_approval,
        }
        reasons.extend(local)
        states.append(state)
    overall = aggregate_reviews(
        [s["review_status"] for s in states],
        complete=bool(states)
        and all(s["review_complete"] for s in states)
        and "missing_latest_channel" not in reasons,
    )
    approval_status = (
        "human_required"
        if overall["review_status"] == "blocked"
        else (
            "rejected"
            if any(
                v["approval"] and v["approval"].get("decision") == "rejected"
                for v in per_version.values()
            )
            else (
                "changes_requested"
                if any(
                    v["approval"]
                    and v["approval"].get("decision") == "changes_requested"
                    for v in per_version.values()
                )
                else (
                    "approved"
                    if per_version
                    and all(v["approval_valid"] for v in per_version.values())
                    else (
                        "partial"
                        if any(v["approval_valid"] for v in per_version.values())
                        else "pending"
                    )
                )
            )
        )
    )
    if workflow.status.value not in {"approved", "delivered"}:
        reasons.append("workflow_not_approved")
    bindings_valid = not reasons
    return {
        **overall,
        "approval_status": approval_status,
        "bindings_valid": bindings_valid,
        "publish_allowed": publish_allowed(
            **overall, approval_status=approval_status, bindings_valid=bindings_valid
        ),
        "reasons": list(dict.fromkeys(reasons)),
        "versions": per_version,
    }
