"""Deterministic end-to-end regression evaluation for knowledge queries."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import pathlib
import statistics
import sys
import time
import unicodedata
import uuid
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

import httpx

_ROOT = pathlib.Path(__file__).parent.parent.resolve()
DEFAULT_CASES_PATH = _ROOT / "data" / "eval" / "knowledge_query_cases.json"
DEFAULT_OUTPUT_DIR = _ROOT / "data" / "eval"
ALLOWED_REQUEST_OVERRIDES = {"task_type", "channel", "tone"}


class CaseValidationError(ValueError):
    """Raised when a knowledge regression case has an invalid contract."""


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).lower()
    return " ".join(text.split())


def validate_cases(raw: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw, list) or not raw:
        raise CaseValidationError(
            "knowledge query cases must be a non-empty JSON array"
        )
    seen = set()
    cases: List[Dict[str, Any]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise CaseValidationError(f"case #{index + 1} must be an object")
        case_id = str(item.get("id") or "").strip()
        query = str(item.get("query") or "").strip()
        if not case_id or not query:
            raise CaseValidationError(f"case #{index + 1} requires id and query")
        if case_id in seen:
            raise CaseValidationError(f"duplicate case id: {case_id}")
        seen.add(case_id)
        overrides = item.get("request_overrides") or {}
        if not isinstance(overrides, dict):
            raise CaseValidationError(f"{case_id}.request_overrides must be an object")
        unsupported = set(overrides) - ALLOWED_REQUEST_OVERRIDES
        if unsupported:
            raise CaseValidationError(
                f"{case_id}.request_overrides contains unsupported keys: {sorted(unsupported)}"
            )
        for key in ("expected_all", "forbidden_any", "expected_source_urls", "setup"):
            if key in item and not isinstance(item[key], list):
                raise CaseValidationError(f"{case_id}.{key} must be an array")
        groups = item.get("expected_any_groups") or []
        if not isinstance(groups, list) or any(
            not isinstance(group, list) or not group for group in groups
        ):
            raise CaseValidationError(
                f"{case_id}.expected_any_groups must contain non-empty arrays"
            )
        policy = item.get("expected_memory_policy") or {}
        if not isinstance(policy, dict):
            raise CaseValidationError(
                f"{case_id}.expected_memory_policy must be an object"
            )
        cases.append(item)
    return cases


def load_cases(path: pathlib.Path = DEFAULT_CASES_PATH) -> List[Dict[str, Any]]:
    return validate_cases(json.loads(path.read_text(encoding="utf-8")))


def _contains(text: str, value: Any) -> bool:
    return normalize_text(value) in text


def score_case(
    case: Dict[str, Any], response: Dict[str, Any], trace: Dict[str, Any]
) -> Dict[str, Any]:
    answer = str(response.get("response") or "")
    normalized = normalize_text(answer)
    expected_all = [str(item) for item in case.get("expected_all") or []]
    groups = [
        [str(item) for item in group] for group in case.get("expected_any_groups") or []
    ]
    forbidden = [str(item) for item in case.get("forbidden_any") or []]
    missing_expected = [
        item for item in expected_all if not _contains(normalized, item)
    ]
    missing_groups = [
        group
        for group in groups
        if not any(_contains(normalized, item) for item in group)
    ]
    forbidden_hits = [item for item in forbidden if _contains(normalized, item)]

    expected_intent = case.get("expected_intent")
    expected_agent = case.get("expected_agent")
    expected_knowledge = case.get("expected_knowledge_used")
    intent_pass = expected_intent is None or response.get("intent") == expected_intent
    actual_agents = list(response.get("routed_agents") or [])
    if response.get("agent_type") and response.get("agent_type") not in actual_agents:
        actual_agents.append(response.get("agent_type"))
    agent_pass = expected_agent is None or expected_agent in actual_agents
    knowledge_pass = expected_knowledge is None or bool(
        response.get("knowledge_used")
    ) is bool(expected_knowledge)
    success_pass = response.get("success") is True

    docs = (
        ((trace.get("retrieval") or {}).get("docs") or [])
        if isinstance(trace, dict)
        else []
    )
    actual_sources = sorted(
        {str(doc.get("source_url")) for doc in docs if doc.get("source_url")}
    )
    expected_sources = [str(item) for item in case.get("expected_source_urls") or []]
    missing_sources = [item for item in expected_sources if item not in actual_sources]
    source_pass = not missing_sources

    actual_policy = (
        (trace.get("memory_policy") or {}) if isinstance(trace, dict) else {}
    )
    expected_policy = case.get("expected_memory_policy") or {}
    policy_mismatches = {
        key: {"expected": value, "actual": actual_policy.get(key)}
        for key, value in expected_policy.items()
        if actual_policy.get(key) != value
    }
    policy_pass = not policy_mismatches
    text_pass = not missing_expected and not missing_groups
    forbidden_pass = not forbidden_hits
    assertions = {
        "response_success": success_pass,
        "expected_content": text_pass,
        "forbidden_content": forbidden_pass,
        "intent": intent_pass,
        "agent": agent_pass,
        "knowledge_used": knowledge_pass,
        "source_urls": source_pass,
        "memory_policy": policy_pass,
    }
    return {
        "id": case["id"],
        "description": case.get("description", ""),
        "passed": all(assertions.values()),
        "infrastructure_error": None,
        "assertions": assertions,
        "missing_expected": missing_expected,
        "missing_expected_groups": missing_groups,
        "forbidden_hits": forbidden_hits,
        "missing_source_urls": missing_sources,
        "actual_source_urls": actual_sources,
        "memory_policy_mismatches": policy_mismatches,
        "intent": response.get("intent"),
        "routed_agents": response.get("routed_agents") or [],
        "knowledge_used": bool(response.get("knowledge_used")),
        "request_id": response.get("request_id"),
        "latency_ms": float(response.get("latency_ms") or 0.0),
        "response_preview": answer[:800],
    }


def infrastructure_failure(
    case: Dict[str, Any], reason: str, latency_ms: float = 0.0
) -> Dict[str, Any]:
    return {
        "id": case["id"],
        "description": case.get("description", ""),
        "passed": False,
        "infrastructure_error": reason,
        "assertions": {},
        "request_id": None,
        "latency_ms": round(latency_ms, 1),
        "response_preview": "",
    }


def aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    passed = sum(bool(row.get("passed")) for row in rows)
    latencies = sorted(
        float(row.get("latency_ms") or 0.0)
        for row in rows
        if not row.get("infrastructure_error")
    )

    def assertion_rate(
        name: str, eligible: Optional[Iterable[Dict[str, Any]]] = None
    ) -> float:
        selected = list(eligible if eligible is not None else rows)
        values = [
            bool(row.get("assertions", {}).get(name))
            for row in selected
            if name in row.get("assertions", {})
        ]
        return round(statistics.mean(values), 4) if values else 1.0

    p95 = 0.0
    if latencies:
        p95 = latencies[
            min(len(latencies) - 1, max(0, int(len(latencies) * 0.95 + 0.9999) - 1))
        ]
    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": round(passed / total, 4) if total else 0.0,
        "expected_coverage_rate": assertion_rate("expected_content"),
        "forbidden_violation_rate": round(1.0 - assertion_rate("forbidden_content"), 4),
        "intent_accuracy": assertion_rate("intent"),
        "source_match_rate": assertion_rate("source_urls"),
        "memory_policy_pass_rate": assertion_rate("memory_policy"),
        "infrastructure_failures": sum(
            bool(row.get("infrastructure_error")) for row in rows
        ),
        "avg_latency_ms": round(statistics.mean(latencies), 1) if latencies else 0.0,
        "p50_latency_ms": round(statistics.median(latencies), 1) if latencies else 0.0,
        "p95_latency_ms": round(p95, 1),
    }


async def _request_json(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    response = await client.request(method, url, json=payload)
    response.raise_for_status()
    value = response.json()
    if not isinstance(value, dict):
        raise ValueError("response JSON must be an object")
    return value


async def run_case(
    client: httpx.AsyncClient, base_url: str, case: Dict[str, Any]
) -> Dict[str, Any]:
    suffix = f"{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
    user_id = f"knowledge_eval_{case['id']}_{suffix}"
    conv_id = f"knowledge_eval_conv_{case['id']}_{suffix}"
    started = time.monotonic()
    try:
        for setup in case.get("setup") or []:
            setup_payload = {
                "message": str(setup),
                "user_id": user_id,
                "conv_id": conv_id,
            }
            setup_response = await _request_json(
                client, "POST", f"{base_url}/chat", setup_payload
            )
            if setup_response.get("success") is not True:
                return infrastructure_failure(
                    case,
                    f"setup request failed: {setup_response.get('error_code') or 'unknown_error'}",
                    (time.monotonic() - started) * 1000,
                )
        payload = {
            "message": case["query"],
            "user_id": user_id,
            "conv_id": conv_id,
            **(case.get("request_overrides") or {}),
        }
        response = await _request_json(client, "POST", f"{base_url}/chat", payload)
        request_id = str(response.get("request_id") or "").strip()
        if not request_id:
            return infrastructure_failure(
                case,
                "chat response missing request_id",
                (time.monotonic() - started) * 1000,
            )
        trace = await _request_json(client, "GET", f"{base_url}/trace/{request_id}")
        return score_case(case, response, trace)
    except (httpx.HTTPError, ValueError, json.JSONDecodeError) as ex:
        return infrastructure_failure(
            case, f"{type(ex).__name__}: {ex}", (time.monotonic() - started) * 1000
        )


async def run_evaluation(
    *,
    base_url: str,
    case_ids: Optional[List[str]] = None,
    fail_fast: bool = False,
    cases_path: pathlib.Path = DEFAULT_CASES_PATH,
    output_dir: pathlib.Path = DEFAULT_OUTPUT_DIR,
    write_report: bool = True,
    timeout_seconds: float = 120.0,
) -> Dict[str, Any]:
    cases = load_cases(cases_path)
    if case_ids:
        requested = set(case_ids)
        known = {case["id"] for case in cases}
        unknown = sorted(requested - known)
        if unknown:
            raise CaseValidationError(f"unknown case ids: {unknown}")
        cases = [case for case in cases if case["id"] in requested]
    rows: List[Dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout_seconds)) as client:
        for case in cases:
            row = await run_case(client, base_url.rstrip("/"), case)
            rows.append(row)
            if fail_fast and not row["passed"]:
                break
    report: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "base_url": base_url.rstrip("/"),
        "cases_path": str(cases_path),
        **aggregate(rows),
        "results": rows,
    }
    if write_report:
        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        report_path = output_dir / f"knowledge_query_report_{stamp}.json"
        latest_path = output_dir / "latest_knowledge_query_report.json"
        serialized = json.dumps(report, ensure_ascii=False, indent=2)
        report_path.write_text(serialized, encoding="utf-8")
        temp_path = latest_path.with_suffix(".json.tmp")
        temp_path.write_text(serialized, encoding="utf-8")
        temp_path.replace(latest_path)
        report["output_path"] = str(report_path)
        report["latest_path"] = str(latest_path)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run ContentOpsMind knowledge-query regressions"
    )
    parser.add_argument(
        "--case", action="append", dest="case_ids", help="Run one case ID; repeatable"
    )
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--base-url", default=os.getenv("CONTENTOPS_BASE_URL", "http://127.0.0.1:8010")
    )
    parser.add_argument("--cases-path", type=pathlib.Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        report = asyncio.run(
            run_evaluation(
                base_url=args.base_url,
                case_ids=args.case_ids,
                fail_fast=args.fail_fast,
                cases_path=args.cases_path,
                output_dir=args.output_dir,
                timeout_seconds=args.timeout,
            )
        )
    except (CaseValidationError, OSError, json.JSONDecodeError) as ex:
        print(f"Knowledge query evaluation failed: {ex}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {key: value for key, value in report.items() if key not in {"results"}},
            ensure_ascii=False,
            indent=2,
        )
    )
    for row in report["results"]:
        status = "PASS" if row["passed"] else "FAIL"
        reason = row.get("infrastructure_error") or ", ".join(
            name for name, passed in row.get("assertions", {}).items() if not passed
        )
        print(f"[{status}] {row['id']}{': ' + reason if reason else ''}")
    return 0 if report["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
