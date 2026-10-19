"""Lightweight request tracing and dashboard metrics for ContentOpsMind."""

from __future__ import annotations

import time
from collections import Counter, deque
from threading import Lock
from typing import Any, Dict, List, Optional


class TraceStore:
    """In-memory ring buffer for recent agent request traces."""

    def __init__(self, maxlen: int = 500):
        self._items = deque(maxlen=maxlen)
        self._by_id: Dict[str, Dict[str, Any]] = {}
        self._lock = Lock()

    def record(self, trace: Dict[str, Any]) -> None:
        request_id = str(trace.get("request_id", ""))
        if not request_id:
            return
        item = dict(trace)
        item.setdefault("recorded_at", time.time())
        with self._lock:
            if len(self._items) == self._items.maxlen and self._items:
                old = self._items[0]
                old_id = old.get("request_id")
                if old_id:
                    self._by_id.pop(str(old_id), None)
            self._items.append(item)
            self._by_id[request_id] = item

    def get(self, request_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            item = self._by_id.get(request_id)
            return dict(item) if item else None

    def recent(self, limit: int = 20) -> List[Dict[str, Any]]:
        limit = max(1, min(int(limit), 200))
        with self._lock:
            return [dict(item) for item in list(self._items)[-limit:]][::-1]

    def dashboard(self) -> Dict[str, Any]:
        with self._lock:
            items = list(self._items)
        total = len(items)
        if not total:
            return {
                "request_count": 0,
                "avg_latency_ms": 0.0,
                "failure_rate": 0.0,
                "rag_hit_rate": 0.0,
                "agent_call_counts": {},
                "agent_call_ratio": {},
                "avg_agent_latency_ms": {},
                "avg_synthesis_latency_ms": 0.0,
                "avg_retrieval_latency_ms": 0.0,
                "recent_errors": [],
                "tool_call_count": 0,
                "tool_call_counts": {},
                "tool_success_rate": 0.0,
                "tool_validation_failure_rate": 0.0,
                "avg_tool_latency_ms": {},
                "agent_tool_call_counts": {},
            }

        agent_counts: Counter[str] = Counter()
        agent_latency_sum: Counter[str] = Counter()
        agent_latency_count: Counter[str] = Counter()
        synthesis_latencies: List[float] = []
        retrieval_latencies: List[float] = []
        rag_used = 0
        failures = 0
        recent_errors: List[Dict[str, Any]] = []
        tool_counts: Counter[str] = Counter()
        tool_latency_sum: Counter[str] = Counter()
        tool_latency_count: Counter[str] = Counter()
        agent_tool_counts: Counter[str] = Counter()
        tool_success = 0
        tool_total = 0
        tool_validation_failures = 0

        for item in items:
            if item.get("status") != "success":
                failures += 1
                if item.get("error"):
                    recent_errors.append(
                        {
                            "request_id": item.get("request_id"),
                            "error": item.get("error"),
                        }
                    )
            retrieval = item.get("retrieval") or {}
            if retrieval.get("used"):
                rag_used += 1
            if retrieval.get("latency_ms") is not None:
                retrieval_latencies.append(float(retrieval.get("latency_ms") or 0.0))
            synthesis = item.get("synthesis") or {}
            if synthesis.get("latency_ms") is not None:
                synthesis_latencies.append(float(synthesis.get("latency_ms") or 0.0))
            for execution in item.get("agent_execution") or []:
                agent = str(execution.get("agent", "unknown"))
                agent_counts[agent] += 1
                agent_latency_sum[agent] += float(execution.get("latency_ms") or 0.0)
                agent_latency_count[agent] += 1
                for call in execution.get("tool_calls") or []:
                    tool = str(call.get("tool", "unknown"))
                    tool_total += 1
                    tool_counts[tool] += 1
                    agent_tool_counts[f"{agent}:{tool}"] += 1
                    tool_latency_sum[tool] += float(call.get("latency_ms") or 0.0)
                    tool_latency_count[tool] += 1
                    if call.get("success"):
                        tool_success += 1
                    if call.get("validation_error"):
                        tool_validation_failures += 1

        total_agent_calls = sum(agent_counts.values()) or 1
        avg_latency = (
            sum(float(item.get("latency_ms") or 0.0) for item in items) / total
        )
        return {
            "request_count": total,
            "avg_latency_ms": round(avg_latency, 1),
            "failure_rate": round(failures / total, 4),
            "rag_hit_rate": round(rag_used / total, 4),
            "agent_call_counts": dict(agent_counts),
            "agent_call_ratio": {
                agent: round(count / total_agent_calls, 4)
                for agent, count in agent_counts.items()
            },
            "avg_agent_latency_ms": {
                agent: round(
                    agent_latency_sum[agent] / max(agent_latency_count[agent], 1), 1
                )
                for agent in agent_counts
            },
            "avg_synthesis_latency_ms": (
                round(sum(synthesis_latencies) / len(synthesis_latencies), 1)
                if synthesis_latencies
                else 0.0
            ),
            "avg_retrieval_latency_ms": (
                round(sum(retrieval_latencies) / len(retrieval_latencies), 1)
                if retrieval_latencies
                else 0.0
            ),
            "recent_errors": recent_errors[-10:],
            "tool_call_count": tool_total,
            "tool_call_counts": dict(tool_counts),
            "tool_success_rate": (
                round(tool_success / tool_total, 4) if tool_total else 0.0
            ),
            "tool_validation_failure_rate": (
                round(tool_validation_failures / tool_total, 4) if tool_total else 0.0
            ),
            "avg_tool_latency_ms": {
                tool: round(
                    tool_latency_sum[tool] / max(tool_latency_count[tool], 1), 1
                )
                for tool in tool_counts
            },
            "agent_tool_call_counts": dict(agent_tool_counts),
        }
