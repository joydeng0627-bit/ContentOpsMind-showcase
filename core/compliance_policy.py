"""Shared compliance taxonomy and structured-result compatibility helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

POLICY_PATH = (
    Path(__file__).resolve().parent.parent / "skills" / "compliance" / "rules.json"
)


def load_policy(path: Path = POLICY_PATH) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    rules = data.get("rules")
    if not isinstance(rules, list) or not rules:
        raise ValueError("compliance policy requires a non-empty rules list")
    ids = [str(rule.get("id", "")) for rule in rules]
    if any(not rule_id for rule_id in ids) or len(ids) != len(set(ids)):
        raise ValueError("compliance policy rule ids must be non-empty and unique")
    return data


POLICY = load_policy()
RULES_BY_ID = {str(item["id"]): item for item in POLICY["rules"]}
RISK_LABELS = frozenset(RULES_BY_ID)
DECISIONS = frozenset(POLICY["decisions"])
SEVERITIES = frozenset(POLICY["severities"])


def normalize_risk_labels(values: Iterable[Any]) -> List[str]:
    return list(
        dict.fromkeys(
            str(value) if str(value) in RISK_LABELS else f"invalid_label:{value}"
            for value in (values or [])
        )
    )


def normalize_review_labels(values: Iterable[Any], text: str = "") -> List[str]:
    """Apply generic overlap relations declared by the versioned policy."""
    labels = normalize_risk_labels(values)
    config = POLICY.get("normalization", {})
    for active, suppressed in config.get("suppresses", {}).items():
        if active in labels:
            labels = [label for label in labels if label not in suppressed]
    for rule in config.get("cooccurrence", []):
        if (
            rule.get("when") in labels
            and any(marker in text for marker in rule.get("text_any", []))
            and any(marker in text for marker in rule.get("contradiction_any", []))
            and rule.get("add") not in labels
        ):
            labels.append(str(rule["add"]))
    return labels


def decision_for_risks(labels: Iterable[str], proposed: str = "") -> str:
    valid = [RULES_BY_ID[label] for label in labels if label in RULES_BY_ID]
    if not valid:
        return "pass"
    if any(rule["decision"] == "blocked" for rule in valid):
        return "blocked"
    return "changes_required"


def validate_review_result(value: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    if value.get("decision") not in DECISIONS:
        errors.append("invalid decision")
    if not isinstance(value.get("risk_items"), list):
        errors.append("risk_items must be an array")
        return errors
    for index, item in enumerate(value["risk_items"]):
        if not isinstance(item, dict):
            errors.append(f"risk_items[{index}] must be an object")
            continue
        if item.get("rule_id") not in RISK_LABELS:
            errors.append(f"risk_items[{index}].rule_id is invalid")
        if item.get("severity") not in SEVERITIES:
            errors.append(f"risk_items[{index}].severity is invalid")
        for key in ("matched_text", "reason", "suggestion"):
            if not isinstance(item.get(key), str):
                errors.append(f"risk_items[{index}].{key} must be a string")
        if not isinstance(item.get("evidence_refs"), list):
            errors.append(f"risk_items[{index}].evidence_refs must be an array")
    if not isinstance(value.get("missing_evidence"), list):
        errors.append("missing_evidence must be an array")
    if not isinstance(value.get("summary"), str):
        errors.append("summary must be a string")
    expected = decision_for_risks(
        [
            item.get("rule_id")
            for item in value.get("risk_items", [])
            if isinstance(item, dict)
        ]
    )
    if value.get("decision") != expected:
        errors.append(f"decision must follow policy matrix: {expected}")
    return errors


def add_legacy_review_fields(value: Dict[str, Any]) -> Dict[str, Any]:
    """Return structured output plus fields consumed by existing UI/evaluators."""
    result = dict(value)
    items = (
        result.get("risk_items") if isinstance(result.get("risk_items"), list) else []
    )
    result["risk_labels"] = list(
        dict.fromkeys(
            item.get("rule_id")
            for item in items
            if isinstance(item, dict) and item.get("rule_id")
        )
    )
    result["evidence"] = [
        {
            "matched_text": item.get("matched_text", ""),
            "reason": item.get("reason", ""),
            "evidence_refs": item.get("evidence_refs", []),
        }
        for item in items
        if isinstance(item, dict)
    ]
    return result
