"""Deterministic review semantics; execution and publication are independent."""

import hashlib
from typing import Any, Dict, Iterable
from .compliance_policy import validate_review_result
from prometheus_client import Counter

REVIEW_VALIDATIONS = Counter(
    "contentops_review_validations_total",
    "Structured review validation outcomes, not model failures",
    ["outcome"],
)


class ReviewOutputError(ValueError):
    """Keep audit evidence when an internal format repair cannot be trusted."""

    def __init__(self, message: str, outputs):
        super().__init__(message)
        self.outputs = list(outputs)


def content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def structured_review(value: Any) -> Dict[str, Any]:
    try:
        result = _validate_structured_review(value)
    except ValueError:
        REVIEW_VALIDATIONS.labels("invalid").inc()
        raise
    REVIEW_VALIDATIONS.labels(result["decision"]).inc()
    return result


def _validate_structured_review(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict) or value.get("decision") not in {
        "pass",
        "changes_required",
        "blocked",
    }:
        raise ValueError("invalid structured review decision")
    decision = value["decision"]
    if "risk_items" in value:
        errors = validate_review_result(value)
        if errors:
            raise ValueError("; ".join(errors))
    # Compliance's versioned risk_items schema predates need_revision.
    risks = value.get("findings", value.get("risk_items"))
    if not isinstance(risks, list) or any(not isinstance(item, dict) for item in risks):
        raise ValueError("review evidence must be an array of objects")
    expected = decision != "pass"
    if "need_revision" in value:
        if (
            type(value["need_revision"]) is not bool
            or value["need_revision"] != expected
        ):
            raise ValueError("review decision and need_revision disagree")
    elif "risk_items" not in value:
        raise ValueError("missing need_revision")
    if not isinstance(value.get("summary"), str):
        raise ValueError("missing review summary")
    if decision == "blocked" and not risks:
        raise ValueError("blocked requires structured risk evidence")
    for risk in risks:
        if not any(
            isinstance(risk.get(key), str) and risk[key].strip()
            for key in ("reason", "matched_text", "suggestion")
        ):
            raise ValueError("empty structured risk evidence")
        refs = risk.get("evidence_refs", [])
        if not isinstance(refs, list) or any(not isinstance(ref, str) for ref in refs):
            raise ValueError("evidence_refs must be an array of strings")
    return {**value, "need_revision": expected, "findings": risks}


def aggregate_reviews(decisions: Iterable[str], *, complete: bool) -> Dict[str, Any]:
    decisions = list(decisions)
    complete = (
        complete
        and bool(decisions)
        and all(item in {"pass", "changes_required", "blocked"} for item in decisions)
    )
    # Keep a known block visible even when another required reviewer failed.
    status = (
        "blocked"
        if "blocked" in decisions
        else (
            "incomplete"
            if not complete
            else "changes_required" if "changes_required" in decisions else "pass"
        )
    )
    return {"review_status": status, "review_complete": complete}


def publish_allowed(
    *,
    review_status: str,
    review_complete: bool,
    approval_status: str,
    bindings_valid: bool,
) -> bool:
    """The single publication predicate; never accepts generated prose."""
    return (
        review_complete
        and review_status == "pass"
        and approval_status == "approved"
        and bindings_valid
    )
