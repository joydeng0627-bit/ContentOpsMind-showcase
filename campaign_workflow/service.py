from __future__ import annotations
from core.llm_gateway import llm_task, ensure_task_active

import asyncio
import hashlib
import json
import re
import random
import time
import os
import contextlib
import logging
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from types import SimpleNamespace
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from prometheus_client import Counter, Histogram
from redis.exceptions import RedisError

from agents.agent_orchestrator import AgentOrchestrator, OrchestratorResult, Request
from core.intent_recognizer import IntentCategory
from mcp.tool_manager import MCPToolManager
from core.compliance_policy import RULES_BY_ID
from core.review_state import content_hash, structured_review
from .eligibility import build_review_context, delivery_eligibility

from .models import (
    ApprovalDecision,
    ApprovalRecord,
    CampaignApprovalRequest,
    CampaignBriefUpdate,
    CampaignRevisionRequest,
    CampaignWorkflow,
    ChannelGenerationResult,
    ContentVersion,
    EventStatus,
    GenerationAttempt,
    GenerationAttemptState,
    GenerationConflict,
    GenerationStatus,
    KnowledgeSnapshot,
    ReviewContext,
    ReviewDecision,
    ReviewEnvelope,
    ReviewRemediation,
    ReviewResult,
    WorkflowExecutionEvent,
    WorkflowStage,
    WorkflowStatus,
    build_default_plan,
    ensure_status_transition,
    utc_now,
)
from .package_builder import build_content_package
from .business_validator import requires_revision, validate_generated_content
from .store import CampaignWorkflowStore, WorkflowConflictError, WorkflowLockLostError

logger = logging.getLogger(__name__)


class WorkflowExecutionError(RuntimeError):
    pass


class WorkflowBusyError(RuntimeError):
    pass


class ReviewContextMismatchError(WorkflowExecutionError):
    pass


WORKFLOW_RUNS = Counter(
    "campaign_workflow_runs_total", "Campaign workflow runs", ["status"]
)
WORKFLOW_LATENCY = Histogram(
    "campaign_workflow_latency_seconds",
    "Campaign workflow end-to-end latency",
    buckets=(1, 5, 10, 20, 30, 60, 120, 300),
)
WORKFLOW_STAGE_FAILURES = Counter(
    "campaign_workflow_stage_failures_total",
    "Campaign workflow stage failures",
    ["stage", "error_code"],
)
LOCK_ACQUIRE = Counter(
    "campaign_lock_acquire_total", "Workflow lock acquisitions", ["result"]
)
LOCK_RENEW = Counter("campaign_lock_renew_total", "Workflow lock renewals", ["result"])
LOCK_LOST = Counter("campaign_lock_lost_total", "Workflow lock losses", ["operation"])
LOCK_HOLD = Histogram(
    "campaign_lock_hold_seconds", "Workflow lock hold time", ["operation"]
)
LOCK_RENEW_COUNT = Histogram(
    "campaign_lock_renew_count",
    "Renewals per workflow lock",
    ["operation"],
    buckets=(0, 1, 2, 3, 5, 10, 20, 50),
)
LOCK_RELEASE = Counter(
    "campaign_lock_release_total", "Workflow lock releases", ["result"]
)


@dataclass
class WorkflowLockLease:
    workflow_id: str
    token: str
    operation: str
    ttl_seconds: float
    renew_interval_seconds: float
    execution_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    acquired_at: float = field(default_factory=time.time)
    renew_count: int = 0
    last_renewed_at: Optional[float] = None
    lost: bool = False
    lost_reason: Optional[str] = None
    watchdog_task: Optional[asyncio.Task] = None
    closed: bool = False

    @property
    def token_hint(self) -> str:
        return hashlib.sha256(self.token.encode("utf-8")).hexdigest()[:8]


_ACTIVE_LEASE: ContextVar[Optional[WorkflowLockLease]] = ContextVar(
    "campaign_active_lease", default=None
)


class CampaignWorkflowService:
    """Persisted business workflow that reuses the existing RAG and Agent runtime."""

    def __init__(
        self,
        store: CampaignWorkflowStore,
        orchestrator: AgentOrchestrator,
        tool_manager: MCPToolManager,
        *,
        model_timeout_seconds: Optional[float] = None,
        retrieval_timeout_seconds: Optional[float] = None,
        max_attempts: Optional[int] = None,
        lock_ttl_seconds: Optional[float] = None,
        lock_renew_interval_seconds: Optional[float] = None,
        approval_lock_ttl_seconds: Optional[float] = None,
    ):
        self.store = store
        self.orchestrator = orchestrator
        self.tool_manager = tool_manager
        self.model_timeout_seconds = (
            model_timeout_seconds
            if model_timeout_seconds is not None
            else float(os.getenv("CAMPAIGN_MODEL_TIMEOUT_SECONDS", "90"))
        )
        self.retrieval_timeout_seconds = (
            retrieval_timeout_seconds
            if retrieval_timeout_seconds is not None
            else float(os.getenv("CAMPAIGN_RETRIEVAL_TIMEOUT_SECONDS", "20"))
        )
        self.max_attempts = (
            max_attempts
            if max_attempts is not None
            else int(os.getenv("CAMPAIGN_MAX_ATTEMPTS", "1"))
        )
        self.retry_base_seconds = float(
            os.getenv("CAMPAIGN_RETRY_BASE_SECONDS", "0.25")
        )
        self.lock_ttl_seconds = (
            lock_ttl_seconds
            if lock_ttl_seconds is not None
            else float(os.getenv("CAMPAIGN_LOCK_TTL_SECONDS", "120"))
        )
        self.lock_renew_interval_seconds = (
            lock_renew_interval_seconds
            if lock_renew_interval_seconds is not None
            else float(os.getenv("CAMPAIGN_LOCK_RENEW_INTERVAL_SECONDS", "30"))
        )
        self.approval_lock_ttl_seconds = (
            approval_lock_ttl_seconds
            if approval_lock_ttl_seconds is not None
            else float(os.getenv("CAMPAIGN_APPROVAL_LOCK_TTL_SECONDS", "120"))
        )
        self.input_price_per_million = self._optional_nonnegative_float(
            "LLM_INPUT_PRICE_PER_MILLION"
        )
        self.output_price_per_million = self._optional_nonnegative_float(
            "LLM_OUTPUT_PRICE_PER_MILLION"
        )
        if self.lock_ttl_seconds <= 0 or self.lock_renew_interval_seconds <= 0:
            raise ValueError("Campaign lock TTL and renew interval must be positive")
        if self.lock_renew_interval_seconds >= self.lock_ttl_seconds / 2:
            raise ValueError(
                "CAMPAIGN_LOCK_RENEW_INTERVAL_SECONDS must be less than half the lock TTL"
            )
        if self.model_timeout_seconds >= self.lock_ttl_seconds:
            raise ValueError(
                "CAMPAIGN_MODEL_TIMEOUT_SECONDS must be less than CAMPAIGN_LOCK_TTL_SECONDS"
            )
        if self.approval_lock_ttl_seconds <= 0:
            raise ValueError("CAMPAIGN_APPROVAL_LOCK_TTL_SECONDS must be positive")

    @staticmethod
    def _optional_nonnegative_float(name: str) -> Optional[float]:
        raw = os.getenv(name, "").strip()
        if not raw:
            return None
        value = float(raw)
        if value < 0:
            raise ValueError(f"{name} must be non-negative")
        return value

    def _record_result_usage(
        self, event: WorkflowExecutionEvent, result: OrchestratorResult
    ) -> None:
        event.model_calls += int(getattr(result, "model_calls", 0) or 0)
        event.input_tokens += int(getattr(result, "input_tokens", 0) or 0)
        event.output_tokens += int(getattr(result, "output_tokens", 0) or 0)
        event.token_usage = event.input_tokens + event.output_tokens
        if (
            self.input_price_per_million is not None
            and self.output_price_per_million is not None
        ):
            event.cost_available = True
            event.estimated_cost_usd = (
                event.input_tokens * self.input_price_per_million
                + event.output_tokens * self.output_price_per_million
            ) / 1_000_000

    def _acquire_lease(
        self, workflow_id: str, operation: str, *, watchdog: bool
    ) -> WorkflowLockLease:
        ttl = self.lock_ttl_seconds if watchdog else self.approval_lock_ttl_seconds
        token = self.store.acquire_lock(workflow_id, ttl_seconds=max(1, int(ttl)))
        if not token:
            LOCK_ACQUIRE.labels(result="busy").inc()
            raise WorkflowBusyError("Workflow 正在执行，请勿重复提交")
        LOCK_ACQUIRE.labels(result="acquired").inc()
        lease = WorkflowLockLease(
            workflow_id=workflow_id,
            token=token,
            operation=operation,
            ttl_seconds=ttl,
            renew_interval_seconds=self.lock_renew_interval_seconds,
        )
        logger.info(
            "workflow_lock_acquired workflow_id=%s execution_id=%s operation=%s ttl=%s token_hint=%s",
            workflow_id,
            lease.execution_id,
            operation,
            ttl,
            lease.token_hint,
        )
        return lease

    async def _watch_lock(self, lease: WorkflowLockLease) -> None:
        while True:
            await asyncio.sleep(lease.renew_interval_seconds)
            try:
                renewed = self.store.renew_lock(
                    lease.workflow_id, lease.token, max(1, int(lease.ttl_seconds))
                )
            except (RedisError, ConnectionError, OSError) as ex:
                LOCK_RENEW.labels(result="error").inc()
                self._mark_lease_lost(lease, f"redis_error:{type(ex).__name__}")
                return
            if not renewed:
                LOCK_RENEW.labels(result="lost").inc()
                self._mark_lease_lost(lease, "ownership_lost")
                return
            lease.renew_count += 1
            lease.last_renewed_at = time.time()
            LOCK_RENEW.labels(result="renewed").inc()
            logger.info(
                "workflow_lock_renewed workflow_id=%s execution_id=%s renew_count=%s token_hint=%s",
                lease.workflow_id,
                lease.execution_id,
                lease.renew_count,
                lease.token_hint,
            )

    @staticmethod
    def _mark_lease_lost(lease: WorkflowLockLease, reason: str) -> None:
        if lease.lost:
            return
        lease.lost = True
        lease.lost_reason = reason
        LOCK_LOST.labels(operation=lease.operation).inc()
        logger.error(
            "workflow_lock_lost workflow_id=%s execution_id=%s operation=%s reason=%s token_hint=%s",
            lease.workflow_id,
            lease.execution_id,
            lease.operation,
            reason,
            lease.token_hint,
        )

    def _lease_token(self, workflow_id: str) -> Optional[str]:
        lease = _ACTIVE_LEASE.get()
        if lease is None or lease.workflow_id != workflow_id:
            return None
        if lease.lost:
            raise WorkflowLockLostError("执行租约已失效，当前任务已在持久化前终止")
        return lease.token

    def _assert_active_lease(self, workflow_id: str) -> None:
        ensure_task_active()
        lease = _ACTIVE_LEASE.get()
        if lease is None or lease.workflow_id != workflow_id:
            return
        if lease.lost or not self.store.owns_lock(workflow_id, lease.token):
            self._mark_lease_lost(
                lease, lease.lost_reason or "ownership_lost_at_stage_boundary"
            )
            raise WorkflowLockLostError("执行租约已失效，当前任务已在持久化前终止")

    async def _close_lease(self, lease: WorkflowLockLease) -> None:
        if lease.closed:
            return
        lease.closed = True
        if lease.watchdog_task is not None:
            lease.watchdog_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await lease.watchdog_task
        released = False
        try:
            released = self.store.release_lock(lease.workflow_id, lease.token)
        except Exception:
            logger.warning(
                "workflow_lock_release_error workflow_id=%s execution_id=%s",
                lease.workflow_id,
                lease.execution_id,
                exc_info=True,
            )
        LOCK_RELEASE.labels(result="released" if released else "not_owner").inc()
        LOCK_HOLD.labels(operation=lease.operation).observe(
            max(0.0, time.time() - lease.acquired_at)
        )
        LOCK_RENEW_COUNT.labels(operation=lease.operation).observe(lease.renew_count)
        outcome = "lost" if lease.lost else ("released" if released else "not_owner")
        audit = {
            "workflow_id": lease.workflow_id,
            "execution_id": lease.execution_id,
            "operation": lease.operation,
            "acquired_at": lease.acquired_at,
            "finished_at": time.time(),
            "ttl_seconds": lease.ttl_seconds,
            "renew_interval_seconds": lease.renew_interval_seconds,
            "renew_count": lease.renew_count,
            "last_renewed_at": lease.last_renewed_at,
            "outcome": outcome,
            "lost_reason": lease.lost_reason,
            "token_hint": lease.token_hint,
        }
        try:
            self.store.save_lock_audit(lease.workflow_id, lease.execution_id, audit)
        except Exception:
            logger.warning(
                "workflow_lock_audit_write_failed workflow_id=%s execution_id=%s",
                lease.workflow_id,
                lease.execution_id,
                exc_info=True,
            )
        logger.info(
            "workflow_lock_closed workflow_id=%s execution_id=%s operation=%s outcome=%s renew_count=%s",
            lease.workflow_id,
            lease.execution_id,
            lease.operation,
            outcome,
            lease.renew_count,
        )

    def _close_sync_lease(self, lease: WorkflowLockLease) -> None:
        if lease.closed:
            return
        lease.closed = True
        released = False
        try:
            released = self.store.release_lock(lease.workflow_id, lease.token)
        except Exception:
            logger.warning(
                "workflow_lock_release_error workflow_id=%s execution_id=%s",
                lease.workflow_id,
                lease.execution_id,
                exc_info=True,
            )
        LOCK_RELEASE.labels(result="released" if released else "not_owner").inc()
        LOCK_HOLD.labels(operation=lease.operation).observe(
            max(0.0, time.time() - lease.acquired_at)
        )
        LOCK_RENEW_COUNT.labels(operation=lease.operation).observe(0)
        outcome = "lost" if lease.lost else ("released" if released else "not_owner")
        try:
            self.store.save_lock_audit(
                lease.workflow_id,
                lease.execution_id,
                {
                    "workflow_id": lease.workflow_id,
                    "execution_id": lease.execution_id,
                    "operation": lease.operation,
                    "acquired_at": lease.acquired_at,
                    "finished_at": time.time(),
                    "ttl_seconds": lease.ttl_seconds,
                    "renew_interval_seconds": None,
                    "renew_count": 0,
                    "last_renewed_at": None,
                    "outcome": outcome,
                    "lost_reason": lease.lost_reason,
                    "token_hint": lease.token_hint,
                },
            )
        except Exception:
            logger.warning(
                "workflow_lock_audit_write_failed workflow_id=%s execution_id=%s",
                lease.workflow_id,
                lease.execution_id,
                exc_info=True,
            )

    @llm_task(None, lambda self, *args, **kwargs: self.model_timeout_seconds)
    async def _call_agent(
        self, request: Request, event: WorkflowExecutionEvent
    ) -> OrchestratorResult:
        last_error: Optional[BaseException] = None
        for attempt in range(1, self.max_attempts + 1):
            event.attempt = max(event.attempt, attempt)
            try:
                result = await asyncio.wait_for(
                    self.orchestrator.run(request), timeout=self.model_timeout_seconds
                )
                self._record_result_usage(event, result)
                if request.task_id:
                    self._assert_active_lease(request.task_id)
                if result.success or result.error_code not in {
                    "timeout",
                    "rate_limited",
                    "service_unavailable",
                    "llm_call_timeout",
                    "llm_service_error",
                    "llm_transport_error",
                    "llm_rate_limited",
                }:
                    return result
                last_error = WorkflowExecutionError(
                    result.error_code or "agent_execution_failed"
                )
            except (asyncio.TimeoutError, ConnectionError) as ex:
                last_error = ex
            if attempt < self.max_attempts:
                await asyncio.sleep(
                    random.uniform(
                        0, min(4, self.retry_base_seconds * (2 ** (attempt - 1)))
                    )
                )
        event.retryable = True
        raise WorkflowExecutionError(
            f"Agent 重试耗尽: {type(last_error).__name__ if last_error else 'unknown'}"
        )

    @llm_task("CAMPAIGN_TASK_TIMEOUT_SECONDS", 900, "CAMPAIGN_LLM_ATTEMPT_BUDGET", 256)
    async def run(self, workflow_id: str) -> Dict[str, Any]:
        run_started = time.monotonic()
        run_outcome = "failed"
        lease = self._acquire_lease(workflow_id, "run", watchdog=True)
        context_token = _ACTIVE_LEASE.set(lease)
        lease.watchdog_task = asyncio.create_task(self._watch_lock(lease))
        try:
            workflow = self.store.get(workflow_id)
            review_resume = workflow.status in {
                WorkflowStatus.REVIEWING,
                WorkflowStatus.REVIEW_INCOMPLETE,
            }
            resumable_failure = (
                workflow.status == WorkflowStatus.FAILED
                and workflow.current_stage
                in {
                    WorkflowStage.KNOWLEDGE_RETRIEVAL,
                    WorkflowStage.CONTENT_GENERATION,
                }
            )
            resumable_generation = workflow.status == WorkflowStatus.GENERATING
            if (
                workflow.status
                not in {WorkflowStatus.VALIDATED, WorkflowStatus.PLANNED}
                and not resumable_failure
                and not resumable_generation
                and not review_resume
            ):
                raise WorkflowConflictError(
                    f"当前状态不可执行: {workflow.status.value}"
                )
            if review_resume:
                if workflow.status == WorkflowStatus.REVIEW_INCOMPLETE:
                    workflow = self._save_status(
                        workflow,
                        WorkflowStatus.REVIEWING,
                        WorkflowStage.PARALLEL_REVIEW,
                        last_error=None,
                    )
                versions = self._latest_versions(workflow)
                if not versions:
                    raise WorkflowExecutionError("没有可恢复审核的内容版本")
            elif workflow.status == WorkflowStatus.FAILED:
                if workflow.current_stage == WorkflowStage.KNOWLEDGE_RETRIEVAL:
                    workflow = self._save_status(
                        workflow,
                        WorkflowStatus.PLANNED,
                        WorkflowStage.KNOWLEDGE_RETRIEVAL,
                        last_error=None,
                    )
                    workflow, knowledge_context = await self._retrieve(workflow)
                    workflow, versions = await self._generate(
                        workflow, knowledge_context
                    )
                else:
                    workflow = self._save_status(
                        workflow,
                        WorkflowStatus.GENERATING,
                        WorkflowStage.CONTENT_GENERATION,
                        last_error=None,
                    )
                    knowledge_context = self._knowledge_context(
                        workflow.knowledge_snapshot
                    )
                    workflow, versions = await self._generate(
                        workflow, knowledge_context
                    )
            elif resumable_generation:
                knowledge_context = self._knowledge_context(workflow.knowledge_snapshot)
                workflow, versions = await self._generate(workflow, knowledge_context)
            else:
                workflow = await self._plan(workflow)
                workflow, knowledge_context = await self._retrieve(workflow)
                workflow, versions = await self._generate(workflow, knowledge_context)
            if workflow.status == WorkflowStatus.NEEDS_CLARIFICATION:
                run_outcome = "needs_clarification"
                await self._close_lease(lease)
                return self.store.detail(workflow.workflow_id)
            expected_workflow_revision = workflow.revision
            try:
                contexts = [
                    self._frozen_review_context(workflow, version)
                    for version in versions
                ]
            except ReviewContextMismatchError as ex:
                workflow = self._mark_review_incomplete(
                    workflow, versions, expected_workflow_revision, str(ex)
                )
                run_outcome = "review_incomplete"
                return self.store.detail(workflow.workflow_id)
            await self._run_parallel_reviews(workflow, versions, contexts)
            workflow = self._aggregate_reviews(
                workflow,
                versions,
                contexts,
                expected_workflow_revision=expected_workflow_revision,
            )
            run_outcome = (
                "review_incomplete"
                if workflow.status == WorkflowStatus.REVIEW_INCOMPLETE
                else "success"
            )
            await self._close_lease(lease)
            return self.store.detail(workflow.workflow_id)
        except (WorkflowLockLostError, WorkflowConflictError):
            raise
        except Exception as ex:
            current = self.store.get(workflow_id)
            if current.status not in {WorkflowStatus.FAILED, WorkflowStatus.CANCELLED}:
                try:
                    current = self._save_status(
                        current,
                        WorkflowStatus.FAILED,
                        current.current_stage,
                        last_error=f"{type(ex).__name__}: {ex}",
                    )
                except Exception:
                    pass
            raise
        finally:
            _ACTIVE_LEASE.reset(context_token)
            await self._close_lease(lease)
            WORKFLOW_RUNS.labels(status=run_outcome).inc()
            WORKFLOW_LATENCY.observe(max(0.0, time.monotonic() - run_started))

    @llm_task("CAMPAIGN_TASK_TIMEOUT_SECONDS", 900, "CAMPAIGN_LLM_ATTEMPT_BUDGET", 256)
    async def revise(
        self, workflow_id: str, request: CampaignRevisionRequest
    ) -> Dict[str, Any]:
        lease = self._acquire_lease(workflow_id, "revise", watchdog=True)
        context_token = _ACTIVE_LEASE.set(lease)
        lease.watchdog_task = asyncio.create_task(self._watch_lock(lease))
        try:
            workflow = self.store.get(workflow_id)
            if workflow.revision != request.expected_workflow_revision:
                raise WorkflowConflictError("Workflow 已更新，请刷新后再提交修订")
            if workflow.status not in {
                WorkflowStatus.CHANGES_REQUESTED,
                WorkflowStatus.AWAITING_APPROVAL,
                WorkflowStatus.APPROVED,
            }:
                raise WorkflowConflictError(
                    f"当前状态不可修订: {workflow.status.value}"
                )
            if request.channel not in workflow.brief.channels:
                raise WorkflowConflictError(f"未知渠道: {request.channel}")
            if (
                workflow.latest_versions.get(request.channel)
                != request.parent_version_id
            ):
                raise WorkflowConflictError("只能修订该渠道的最新版本")

            versions = self.store.list_content_versions(workflow_id)
            parent = next(
                (
                    item
                    for item in versions
                    if item.version_id == request.parent_version_id
                ),
                None,
            )
            if parent is None:
                raise WorkflowConflictError("父版本不存在")

            if workflow.status in {
                WorkflowStatus.AWAITING_APPROVAL,
                WorkflowStatus.APPROVED,
            }:
                workflow = self._save_status(
                    workflow,
                    WorkflowStatus.CHANGES_REQUESTED,
                    WorkflowStage.CONTENT_GENERATION,
                    approval_status="changes_requested",
                )
            workflow = self._save_status(
                workflow, WorkflowStatus.GENERATING, WorkflowStage.CONTENT_GENERATION
            )
            knowledge_context = self._knowledge_context(workflow.knowledge_snapshot)
            new_version = await self._generate_revision(
                workflow, parent, request.instruction, knowledge_context
            )
            latest = dict(workflow.latest_versions)
            latest[request.channel] = new_version.version_id
            workflow = self._save_status(
                workflow,
                WorkflowStatus.REVIEWING,
                WorkflowStage.PARALLEL_REVIEW,
                latest_versions=latest,
                approval_status="pending",
            )
            expected_workflow_revision = workflow.revision
            latest_versions = self._latest_versions(workflow)
            try:
                contexts = [
                    self._frozen_review_context(workflow, version)
                    for version in latest_versions
                ]
            except ReviewContextMismatchError as ex:
                workflow = self._mark_review_incomplete(
                    workflow, latest_versions, expected_workflow_revision, str(ex)
                )
                return self.store.detail(workflow.workflow_id)
            await self._run_parallel_reviews(workflow, latest_versions, contexts)
            workflow = self._aggregate_reviews(
                workflow,
                latest_versions,
                contexts,
                expected_workflow_revision=expected_workflow_revision,
            )
            await self._close_lease(lease)
            return self.store.detail(workflow_id)
        except (WorkflowConflictError, WorkflowLockLostError):
            raise
        except Exception as ex:
            current = self.store.get(workflow_id)
            if current.status not in {WorkflowStatus.FAILED, WorkflowStatus.CANCELLED}:
                try:
                    self._save_status(
                        current,
                        WorkflowStatus.FAILED,
                        current.current_stage,
                        last_error=f"{type(ex).__name__}: {ex}",
                    )
                except Exception:
                    pass
            raise
        finally:
            _ACTIVE_LEASE.reset(context_token)
            await self._close_lease(lease)

    def update_brief(
        self, workflow_id: str, request: CampaignBriefUpdate
    ) -> Dict[str, Any]:
        lease = self._acquire_lease(workflow_id, "update_brief", watchdog=False)
        context_token = _ACTIVE_LEASE.set(lease)
        try:
            workflow = self.store.get(workflow_id)
            if workflow.revision != request.expected_workflow_revision:
                raise WorkflowConflictError("Workflow 已更新，请刷新后再修改 Brief")
            if workflow.status != WorkflowStatus.NEEDS_CLARIFICATION:
                raise WorkflowConflictError(
                    f"当前状态不可修改 Brief: {workflow.status.value}"
                )
            workflow = self._save_status(
                workflow,
                WorkflowStatus.VALIDATED,
                WorkflowStage.PLANNING,
                brief=request.brief,
                knowledge_snapshot=KnowledgeSnapshot(),
                latest_versions={},
                active_generation_attempt_id=None,
                approval_status="pending",
                last_error=None,
            )
            self._close_sync_lease(lease)
            return self.store.detail(workflow_id)
        finally:
            _ACTIVE_LEASE.reset(context_token)
            self._close_sync_lease(lease)

    def approve(
        self, workflow_id: str, request: CampaignApprovalRequest
    ) -> Dict[str, Any]:
        lease = self._acquire_lease(workflow_id, "approve", watchdog=False)
        context_token = _ACTIVE_LEASE.set(lease)
        try:
            snapshot = self.store.delivery_snapshot(workflow_id)
            workflow = self.store.get(workflow_id)
            request_payload = request.model_dump(mode="json")
            request_payload["version_ids"] = sorted(request.version_ids)
            request_hash = hashlib.sha256(
                json.dumps(
                    {"workflow_id": workflow_id, **request_payload},
                    ensure_ascii=False,
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest()
            approval_id = f"approval_{request_hash[:32]}"
            replay = next(
                (
                    item
                    for item in self.store.list_approvals(workflow_id)
                    if item.approval_id == approval_id
                ),
                None,
            )
            if replay is not None:
                if (
                    replay.request_hash != request_hash
                    or replay.committed_revision != workflow.revision
                ):
                    raise WorkflowConflictError(
                        "重复审批对应的工作流已变化，禁止复用旧审批资格"
                    )
                self.store.commit_delivery_snapshot(
                    workflow_id, snapshot, [], lease.token
                )
                return self._checked_delivery_detail(workflow_id, lease.token)
            if workflow.revision != request.expected_workflow_revision:
                raise WorkflowConflictError("Workflow 已更新，请刷新后再审批")
            if workflow.status != WorkflowStatus.AWAITING_APPROVAL:
                raise WorkflowConflictError(
                    f"当前状态不可审批: {workflow.status.value}"
                )

            versions = {
                item.version_id: item
                for item in self.store.list_content_versions(workflow_id)
            }
            latest_ids = set(workflow.latest_versions.values())
            requested_ids = set(request.version_ids)
            if not requested_ids.issubset(latest_ids):
                raise WorkflowConflictError("只能审批各渠道的最新版本")
            if any(version_id not in versions for version_id in requested_ids):
                raise WorkflowConflictError("审批版本不存在")

            eligibility = delivery_eligibility(self.store.detail(workflow_id))
            for version_id in requested_ids:
                state = eligibility["versions"].get(version_id, {})
                if (
                    not state.get("review_complete")
                    or state.get("review_status") != "pass"
                    or not state.get("binding")
                ):
                    raise WorkflowConflictError(
                        "审批资格校验失败: 必要审核未通过或版本绑定不一致"
                    )

            approval = ApprovalRecord(
                approval_id=approval_id,
                workflow_id=workflow_id,
                version_ids=request.version_ids,
                decision=request.decision,
                reviewer_id=request.reviewer_id,
                comment=request.comment,
                version_bindings={
                    vid: eligibility["versions"][vid]["binding"]
                    for vid in request.version_ids
                },
                request_hash=request_hash,
                committed_revision=workflow.revision + 1,
            )
            for version_id in request.version_ids:
                version = versions[version_id]
                version.status = {
                    ApprovalDecision.APPROVED: "approved",
                    ApprovalDecision.REJECTED: "rejected",
                    ApprovalDecision.CHANGES_REQUESTED: "changes_requested",
                }[request.decision]

            if request.decision == ApprovalDecision.APPROVED:
                all_approved = all(
                    versions[version_id].status == "approved"
                    and (
                        version_id in requested_ids
                        or eligibility["versions"]
                        .get(version_id, {})
                        .get("approval_valid")
                    )
                    for version_id in latest_ids
                )
                target = (
                    WorkflowStatus.APPROVED
                    if all_approved
                    else WorkflowStatus.AWAITING_APPROVAL
                )
                stage = (
                    WorkflowStage.DELIVERY
                    if all_approved
                    else WorkflowStage.HUMAN_APPROVAL
                )
                approval_status = "approved" if all_approved else "partial"
            else:
                target, stage = (
                    WorkflowStatus.CHANGES_REQUESTED,
                    WorkflowStage.CONTENT_GENERATION,
                )
                approval_status = request.decision.value
            ensure_status_transition(workflow.status, target)
            workflow = workflow.model_copy(
                update={
                    "status": target,
                    "current_stage": stage,
                    "approval_status": approval_status,
                    "updated_at": utc_now(),
                    "revision": workflow.revision + 1,
                }
            )
            self.store.commit_approval(
                workflow,
                [versions[vid] for vid in request.version_ids],
                approval,
                snapshot,
                lease.token,
            )
            return self._checked_delivery_detail(workflow_id, lease.token)
        finally:
            _ACTIVE_LEASE.reset(context_token)
            self._close_sync_lease(lease)

    def _checked_delivery_detail(self, workflow_id: str, token: str) -> Dict[str, Any]:
        snapshot = self.store.delivery_snapshot(workflow_id)
        detail = self.store.detail(workflow_id)
        self.store.commit_delivery_snapshot(workflow_id, snapshot, [], token)
        return detail

    def package(self, workflow_id: str) -> Dict[str, Any]:
        lease = self._acquire_lease(workflow_id, "package", watchdog=False)
        context_token = _ACTIVE_LEASE.set(lease)
        try:
            snapshot = self.store.delivery_snapshot(workflow_id)
            detail = self.store.detail(workflow_id)
            package = build_content_package(detail)
            key = f"campaign:workflow:{workflow_id}:package:{detail['workflow']['revision']}"
            existing = self.store.redis.get(key)
            snapshot["checks"][key] = existing
            if existing is not None:
                try:
                    saved = json.loads(existing)
                except (ValueError, TypeError) as ex:
                    raise WorkflowConflictError(
                        "历史交付包格式无效，禁止覆盖或交付"
                    ) from ex
                if not isinstance(saved, dict) or {
                    k: v for k, v in saved.items() if k != "generated_at"
                } != {k: v for k, v in package.items() if k != "generated_at"}:
                    raise WorkflowConflictError("历史交付包绑定不一致")
                package = saved
                writes = []
            else:
                writes = [
                    {
                        "op": "set",
                        "key": key,
                        "value": json.dumps(package, ensure_ascii=False),
                    }
                ]
            self.store.commit_delivery_snapshot(
                workflow_id, snapshot, writes, lease.token
            )
            return package
        finally:
            _ACTIVE_LEASE.reset(context_token)
            self._close_sync_lease(lease)

    async def _generate_revision(
        self,
        workflow: CampaignWorkflow,
        parent: ContentVersion,
        instruction: str,
        knowledge_context: str,
    ) -> ContentVersion:
        event = self._start_event(
            workflow, WorkflowStage.CONTENT_GENERATION, parent.channel
        )
        try:
            result = await self._call_agent(
                Request(
                    message=(
                        "根据人工反馈修订当前渠道草稿。保留 Brief 中的事实，不得新增价格、优惠、库存或期限。"
                        "只返回 JSON 对象，字段为 channel、content、citation_ids、unresolved_fields；content 只能是可交付正文。\n"
                        f"修订要求: {instruction}\n当前草稿:\n{parent.content}"
                    ),
                    user_id=workflow.user_id,
                    conv_id=workflow.conv_id or workflow.workflow_id,
                    task_id=workflow.workflow_id,
                    context=self._agent_context(
                        workflow, knowledge_context, parent.channel
                    ),
                    intent=IntentCategory.COPY_REWRITE,
                    secondary_intents=[],
                    trace_context={
                        "workflow": {
                            "workflow_id": workflow.workflow_id,
                            "stage": "revision",
                            "channel": parent.channel,
                        }
                    },
                ),
                event,
            )
            if not result.success:
                raise WorkflowExecutionError(result.error_code or "revision_failed")
            payload, result = await self._ensure_generation_payload(
                workflow, parent.channel, result, event, knowledge_context
            )
            channel_versions = [
                item
                for item in self.store.list_content_versions(workflow.workflow_id)
                if item.channel == parent.channel
            ]
            version = ContentVersion(
                workflow_id=workflow.workflow_id,
                channel=parent.channel,
                version=max((item.version for item in channel_versions), default=0) + 1,
                content=payload["content"],
                content_hash=content_hash(payload["content"]),
                citation_ids=payload["citation_ids"],
                unresolved_fields=payload["unresolved_fields"],
                rejected_constraints=parent.rejected_constraints,
                brief_hash=workflow.brief.snapshot_hash(),
                knowledge_snapshot_hash=workflow.knowledge_snapshot.snapshot_hash,
                generation_request_id=result.request_id,
                parent_version_id=parent.version_id,
            )
            self.store.save_content_version(
                version, lock_token=self._lease_token(workflow.workflow_id)
            )
            self._save_channel_validation(workflow, version)
            event.request_id = result.request_id
            self._finish_event(event, EventStatus.SUCCESS)
            return version
        except WorkflowLockLostError:
            raise
        except Exception:
            self._finish_event(event, EventStatus.FAILED, error_code="revision_failed")
            raise

    async def _plan(self, workflow: CampaignWorkflow) -> CampaignWorkflow:
        event = self._start_event(workflow, WorkflowStage.PLANNING)
        try:
            if workflow.status == WorkflowStatus.VALIDATED:
                workflow = self._save_status(
                    workflow,
                    WorkflowStatus.PLANNED,
                    WorkflowStage.KNOWLEDGE_RETRIEVAL,
                    plan=build_default_plan(workflow.brief.channels),
                )
            self._finish_event(event, EventStatus.SUCCESS)
            return workflow
        except WorkflowLockLostError:
            raise
        except Exception as ex:
            self._finish_event(event, EventStatus.FAILED, error_code="planning_failed")
            raise WorkflowExecutionError("任务规划失败") from ex

    async def _retrieve(
        self, workflow: CampaignWorkflow
    ) -> Tuple[CampaignWorkflow, str]:
        workflow = self._save_status(
            workflow, WorkflowStatus.RETRIEVING, WorkflowStage.KNOWLEDGE_RETRIEVAL
        )
        event = self._start_event(workflow, WorkflowStage.KNOWLEDGE_RETRIEVAL)
        try:
            query = self._retrieval_query(workflow)
            filters = {
                "corpus_id": os.getenv(
                    "CAMPAIGN_KNOWLEDGE_CORPUS_ID", "demo-production"
                ),
                "status": "current",
            }
            result = await asyncio.wait_for(
                self.tool_manager.search_with_rewrite(
                    "knowledge_search",
                    query,
                    top_k=5,
                    context={
                        "workflow_id": workflow.workflow_id,
                        "retrieval_filters": filters,
                        "min_score": float(
                            os.getenv("CAMPAIGN_KNOWLEDGE_MIN_SCORE", "0.08")
                        ),
                    },
                ),
                timeout=self.retrieval_timeout_seconds,
            )
            retrieval_usage = (getattr(result, "metadata", {}) or {}).get(
                "retrieval_usage", {}
            )
            self._record_result_usage(
                event,
                SimpleNamespace(
                    **{
                        key: int(retrieval_usage.get(key, 0) or 0)
                        for key in ("model_calls", "input_tokens", "output_tokens")
                    }
                ),
            )
            event.usage_complete = bool(retrieval_usage.get("usage_available", False))
            documents = []
            if result.success and isinstance(result.data, list):
                for item in self._select_campaign_documents(workflow, result.data):
                    if (
                        not isinstance(item, dict)
                        or not str(item.get("content", "")).strip()
                    ):
                        continue
                    if str(item.get("corpus_id", "")) != filters[
                        "corpus_id"
                    ] or "[BENCHMARK:" in str(item.get("title", "")):
                        logger.warning(
                            "campaign_retrieval_rejected_out_of_scope_document title=%s corpus=%s",
                            item.get("title"),
                            item.get("corpus_id"),
                        )
                        continue
                    documents.append(
                        {
                            "document_id": str(item.get("document_id", "")),
                            "corpus_id": str(item.get("corpus_id", "")),
                            "title": str(item.get("title", "")),
                            "content": str(item.get("content", ""))[:1200],
                            "score": item.get("score"),
                            "source_url": str(item.get("source_url", "")),
                            "published_at": str(item.get("published_at", "")),
                            "document_type": str(item.get("document_type", "")),
                            "version": str(item.get("version", "")),
                            "status": str(item.get("status", "")),
                            "effective_at": str(item.get("effective_at", "")),
                            "structured_data": (
                                item.get("structured_data", {})
                                if isinstance(item.get("structured_data", {}), dict)
                                else {}
                            ),
                        }
                    )
            payload = json.dumps(documents, ensure_ascii=False, sort_keys=True)
            snapshot = KnowledgeSnapshot(
                retrieved_at=utc_now(),
                queries=list((result.metadata or {}).get("queries", [query])),
                documents=documents,
                snapshot_hash=hashlib.sha256(payload.encode("utf-8")).hexdigest(),
            )
            knowledge_context = self._knowledge_context(snapshot)
            workflow = self._save_status(
                workflow,
                WorkflowStatus.GENERATING,
                WorkflowStage.CONTENT_GENERATION,
                knowledge_snapshot=snapshot,
            )
            self._finish_event(event, EventStatus.SUCCESS)
            return workflow, knowledge_context
        except WorkflowLockLostError:
            raise
        except Exception as ex:
            self._finish_event(event, EventStatus.FAILED, error_code="retrieval_failed")
            raise WorkflowExecutionError("知识检索失败") from ex

    async def _generate(
        self,
        workflow: CampaignWorkflow,
        knowledge_context: str,
    ) -> Tuple[CampaignWorkflow, List[ContentVersion]]:
        workflow, attempt = self._generation_attempt(workflow)

        async def generate_channel(channel: str) -> ChannelGenerationResult:
            event = self._start_event(
                workflow, WorkflowStage.CONTENT_GENERATION, channel
            )
            staged = self.store.get_generation_result(
                attempt.generation_attempt_id, channel
            )
            active_phase = "copywriting"
            try:
                if staged is not None:
                    self._validate_generation_result_context(workflow, attempt, staged)
                    if staged.status in {
                        GenerationStatus.READY_TO_COMMIT,
                        GenerationStatus.CLARIFICATION_REQUIRED,
                    }:
                        event.request_id = (
                            staged.style_request_id or staged.copywriting_request_id
                        )
                        self._finish_event(event, EventStatus.SKIPPED)
                        return staged

                if staged is None or not staged.draft_content:
                    copy_attempt = (
                        int(
                            (staged.phase_attempts if staged else {}).get(
                                "copywriting", 0
                            )
                        )
                        + 1
                    )
                    draft = await self._call_agent(
                        Request(
                            message=self._generation_message(workflow, channel),
                            user_id=workflow.user_id,
                            conv_id=workflow.conv_id or workflow.workflow_id,
                            task_id=workflow.workflow_id,
                            context=self._agent_context(
                                workflow, knowledge_context, channel
                            ),
                            intent=IntentCategory.COPY_GENERATION,
                            secondary_intents=[],
                            trace_context={
                                "workflow": {
                                    "workflow_id": workflow.workflow_id,
                                    "stage": "content_generation",
                                    "channel": channel,
                                    "generation_attempt_id": attempt.generation_attempt_id,
                                }
                            },
                        ),
                        event,
                    )
                    if not draft.success:
                        raise WorkflowExecutionError(
                            draft.error_code or "generation_failed"
                        )
                    envelope, draft = await self._ensure_copywriting_envelope(
                        workflow, channel, draft, event, knowledge_context
                    )
                    attempts = dict(staged.phase_attempts if staged else {})
                    attempts["copywriting"] = copy_attempt
                    status = (
                        GenerationStatus.CLARIFICATION_REQUIRED
                        if envelope["generation_status"] == "clarification_required"
                        else GenerationStatus.COPY_READY
                    )
                    staged = ChannelGenerationResult(
                        generation_result_id=(
                            staged.generation_result_id
                            if staged
                            else f"generation_result_{uuid.uuid4().hex[:16]}"
                        ),
                        generation_attempt_id=attempt.generation_attempt_id,
                        workflow_id=workflow.workflow_id,
                        channel=channel,
                        brief_hash=attempt.brief_hash,
                        knowledge_snapshot_hash=attempt.knowledge_snapshot_hash,
                        generation_context_hash=attempt.generation_context_hash,
                        channels=attempt.channels,
                        status=status,
                        result_revision=(staged.result_revision + 1 if staged else 1),
                        content_version_id=(
                            staged.content_version_id
                            if staged
                            else f"cv_{uuid.uuid4().hex[:16]}"
                        ),
                        draft_content=envelope.get("content", ""),
                        citation_ids=envelope.get("citation_ids", []),
                        unresolved_fields=envelope.get("unresolved_fields", []),
                        rejected_constraints=envelope.get("rejected_constraints", []),
                        conflicts=[
                            GenerationConflict.model_validate(item)
                            for item in envelope.get("conflicts", [])
                        ],
                        copywriting_request_id=draft.request_id,
                        phase_attempts=attempts,
                        created_at=(staged.created_at if staged else utc_now()),
                        updated_at=utc_now(),
                    )
                    self._assert_active_lease(workflow.workflow_id)
                    self.store.save_generation_result(
                        staged,
                        expected_revision=staged.result_revision - 1,
                        lock_token=self._lease_token(workflow.workflow_id),
                    )
                    if staged.status == GenerationStatus.CLARIFICATION_REQUIRED:
                        event.request_id = draft.request_id
                        self._finish_event(event, EventStatus.SUCCESS)
                        return staged

                if not staged.final_content:
                    active_phase = "style"
                    style_attempt = int(staged.phase_attempts.get("style", 0)) + 1
                    styled = await self._call_agent(
                        Request(
                            message=(
                                f"将以下草稿适配为{channel}渠道的最终成稿。不要增加事实、审核结论、优化建议或解释。"
                                "只返回 JSON 对象，字段为 channel、content、citation_ids、unresolved_fields。\n"
                                f"草稿:\n{staged.draft_content}"
                            ),
                            user_id=workflow.user_id,
                            conv_id=workflow.conv_id or workflow.workflow_id,
                            task_id=workflow.workflow_id,
                            context=self._agent_context(
                                workflow, knowledge_context, channel
                            ),
                            intent=IntentCategory.STYLE_ADAPTATION,
                            secondary_intents=[],
                            trace_context={
                                "workflow": {
                                    "workflow_id": workflow.workflow_id,
                                    "stage": "style_adaptation",
                                    "channel": channel,
                                    "generation_attempt_id": attempt.generation_attempt_id,
                                }
                            },
                        ),
                        event,
                    )
                    if not styled.success:
                        raise WorkflowExecutionError(
                            styled.error_code or "style_adaptation_failed"
                        )
                    payload, styled = await self._ensure_generation_format(
                        workflow, channel, styled, event, knowledge_context
                    )
                    attempts = dict(staged.phase_attempts)
                    attempts["style"] = style_attempt
                    staged = staged.model_copy(
                        update={
                            "status": GenerationStatus.STYLE_READY,
                            "result_revision": staged.result_revision + 1,
                            "final_content": payload["content"],
                            "citation_ids": payload["citation_ids"],
                            "unresolved_fields": payload["unresolved_fields"],
                            "style_request_id": styled.request_id,
                            "phase_attempts": attempts,
                            "error_code": None,
                            "updated_at": utc_now(),
                        }
                    )
                    self._assert_active_lease(workflow.workflow_id)
                    self.store.save_generation_result(
                        staged,
                        expected_revision=staged.result_revision - 1,
                        lock_token=self._lease_token(workflow.workflow_id),
                    )

                payload = {
                    "channel": channel,
                    "content": staged.final_content,
                    "citation_ids": staged.citation_ids,
                    "unresolved_fields": staged.unresolved_fields,
                }
                active_phase = "channel_validation"
                issues = validate_generated_content(
                    channel, staged.final_content, workflow.knowledge_snapshot.documents
                )
                final_request_id = (
                    staged.style_request_id or staged.copywriting_request_id
                )
                if requires_revision(issues):
                    active_phase = "validation_revision"
                    validation_attempt = (
                        int(staged.phase_attempts.get("validation_revision", 0)) + 1
                    )
                    revised = await self._call_agent(
                        Request(
                            message=(
                                "根据确定性校验结果修订正文，只修复列出的问题，不增加新事实。"
                                "只返回 channel、content、citation_ids、unresolved_fields 四个字段的 JSON 对象。\n"
                                f"校验问题: {json.dumps(issues, ensure_ascii=False)}\n原结果: {json.dumps(payload, ensure_ascii=False)}"
                            ),
                            user_id=workflow.user_id,
                            conv_id=workflow.conv_id or workflow.workflow_id,
                            task_id=workflow.workflow_id,
                            context=self._agent_context(
                                workflow, knowledge_context, channel
                            ),
                            intent=IntentCategory.COPY_REWRITE,
                            secondary_intents=[],
                            trace_context={
                                "workflow": {
                                    "workflow_id": workflow.workflow_id,
                                    "stage": "deterministic_validation_revision",
                                    "channel": channel,
                                    "generation_attempt_id": attempt.generation_attempt_id,
                                }
                            },
                        ),
                        event,
                    )
                    if not revised.success:
                        raise WorkflowExecutionError(
                            revised.error_code
                            or "deterministic_business_validation_failed"
                        )
                    payload, revised = await self._ensure_generation_format(
                        workflow, channel, revised, event, knowledge_context
                    )
                    issues = validate_generated_content(
                        channel,
                        payload["content"],
                        workflow.knowledge_snapshot.documents,
                    )
                    if requires_revision(issues):
                        labels = ",".join(
                            sorted({item.get("rule_id", "unknown") for item in issues})
                        )
                        raise WorkflowExecutionError(
                            "deterministic_business_validation_failed"
                            + (f":{labels}" if labels else ":invalid_payload")
                        )
                    attempts = dict(staged.phase_attempts)
                    attempts["validation_revision"] = validation_attempt
                    staged = staged.model_copy(
                        update={
                            "final_content": payload["content"],
                            "citation_ids": payload["citation_ids"],
                            "unresolved_fields": payload["unresolved_fields"],
                            "validation_revision_request_id": revised.request_id,
                            "phase_attempts": attempts,
                        }
                    )
                    final_request_id = revised.request_id

                validation = self._build_channel_validation(
                    workflow,
                    staged.content_version_id,
                    channel,
                    payload["content"],
                    issues,
                )
                staged = staged.model_copy(
                    update={
                        "status": GenerationStatus.READY_TO_COMMIT,
                        "result_revision": staged.result_revision + 1,
                        "final_content": payload["content"],
                        "citation_ids": payload["citation_ids"],
                        "unresolved_fields": payload["unresolved_fields"],
                        "channel_validation": validation.model_dump(mode="json"),
                        "error_code": None,
                        "updated_at": utc_now(),
                    }
                )
                self._assert_active_lease(workflow.workflow_id)
                self.store.save_generation_result(
                    staged,
                    expected_revision=staged.result_revision - 1,
                    lock_token=self._lease_token(workflow.workflow_id),
                )
                event.request_id = final_request_id
                self._finish_event(event, EventStatus.SUCCESS)
                return staged
            except WorkflowLockLostError:
                raise
            except Exception as ex:
                current = self.store.get_generation_result(
                    attempt.generation_attempt_id, channel
                )
                attempts = dict(current.phase_attempts if current else {})
                attempts[active_phase] = int(attempts.get(active_phase, 0)) + 1
                if current is not None:
                    failed = current.model_copy(
                        update={
                            "status": GenerationStatus.EXECUTION_FAILED,
                            "result_revision": current.result_revision + 1,
                            "error_code": str(ex)[:300],
                            "phase_attempts": attempts,
                            "updated_at": utc_now(),
                        }
                    )
                    expected_result_revision = current.result_revision
                else:
                    failed = ChannelGenerationResult(
                        generation_attempt_id=attempt.generation_attempt_id,
                        workflow_id=workflow.workflow_id,
                        channel=channel,
                        brief_hash=attempt.brief_hash,
                        knowledge_snapshot_hash=attempt.knowledge_snapshot_hash,
                        generation_context_hash=attempt.generation_context_hash,
                        channels=attempt.channels,
                        status=GenerationStatus.EXECUTION_FAILED,
                        phase_attempts=attempts,
                        error_code=str(ex)[:300],
                    )
                    expected_result_revision = 0
                self._assert_active_lease(workflow.workflow_id)
                self.store.save_generation_result(
                    failed,
                    expected_revision=expected_result_revision,
                    lock_token=self._lease_token(workflow.workflow_id),
                )
                self._finish_event(
                    event, EventStatus.FAILED, error_code="content_generation_failed"
                )
                raise

        generated = await asyncio.gather(
            *(generate_channel(channel) for channel in attempt.channels)
        )
        if any(
            item.status == GenerationStatus.CLARIFICATION_REQUIRED for item in generated
        ):
            attempt = attempt.model_copy(
                update={
                    "state": GenerationAttemptState.NEEDS_CLARIFICATION,
                    "updated_at": utc_now(),
                }
            )
            self._assert_active_lease(workflow.workflow_id)
            self.store.save_generation_attempt(
                attempt, lock_token=self._lease_token(workflow.workflow_id)
            )
            workflow = self._save_status(
                workflow,
                WorkflowStatus.NEEDS_CLARIFICATION,
                WorkflowStage.CONTENT_GENERATION,
                last_error=None,
            )
            return workflow, []
        if any(item.status != GenerationStatus.READY_TO_COMMIT for item in generated):
            raise WorkflowExecutionError("generation_results_incomplete")

        prior_versions = self.store.list_content_versions(workflow.workflow_id)
        next_numbers = {
            channel: max(
                (item.version for item in prior_versions if item.channel == channel),
                default=0,
            )
            + 1
            for channel in attempt.channels
        }
        versions = [
            ContentVersion(
                version_id=item.content_version_id,
                workflow_id=workflow.workflow_id,
                channel=item.channel,
                version=next_numbers[item.channel],
                content=item.final_content,
                content_hash=content_hash(item.final_content),
                citation_ids=item.citation_ids,
                unresolved_fields=item.unresolved_fields,
                rejected_constraints=item.rejected_constraints,
                status="pending_review",
                brief_hash=attempt.brief_hash,
                knowledge_snapshot_hash=attempt.knowledge_snapshot_hash,
                generation_request_id=(
                    item.validation_revision_request_id
                    or item.style_request_id
                    or item.copywriting_request_id
                ),
            )
            for item in generated
        ]
        validations = [
            ReviewResult.model_validate(item.channel_validation) for item in generated
        ]
        for version, validation in zip(versions, validations):
            context = self._build_review_context(workflow, version)
            validation.content_hash = version.content_hash
            validation.review_context_hash = context.review_context_hash
            validation.knowledge_snapshot_hash = version.knowledge_snapshot_hash
        expected = workflow.revision
        latest = {item.channel: item.version_id for item in versions}
        updated = workflow.model_copy(
            update={
                "status": WorkflowStatus.REVIEWING,
                "current_stage": WorkflowStage.PARALLEL_REVIEW,
                "latest_versions": latest,
                "approval_status": "pending",
                "last_error": None,
                "updated_at": utc_now(),
                "revision": expected + 1,
            }
        )
        ensure_status_transition(workflow.status, WorkflowStatus.REVIEWING)
        self._assert_active_lease(workflow.workflow_id)
        self.store.commit_generation_batch(
            updated,
            attempt,
            generated,
            versions,
            validations,
            expected_revision=expected,
            lock_token=self._lease_token(workflow.workflow_id),
        )
        return self.store.get(workflow.workflow_id), versions

    def _build_channel_validation(
        self,
        workflow: CampaignWorkflow,
        version_id: str,
        channel: str,
        content: str,
        issues: Optional[List[Dict[str, Any]]] = None,
    ) -> ReviewResult:
        issues = (
            issues
            if issues is not None
            else validate_generated_content(
                channel, content, workflow.knowledge_snapshot.documents
            )
        )
        impacts = {item.get("impact") for item in issues}
        if "blocking_issue" in impacts:
            decision, blocking = ReviewDecision.BLOCKED, True
        elif "required_change" in impacts:
            decision, blocking = ReviewDecision.CHANGES_REQUIRED, True
        else:
            decision, blocking = ReviewDecision.PASS, False
        evidence = [
            {
                "rule_id": item["rule_id"],
                "category": item["rule_id"],
                "matched_text": "",
                "source": "确定性渠道与业务规则校验",
                "severity": (
                    "high"
                    if item["impact"] == "blocking_issue"
                    else "medium" if item["impact"] == "required_change" else "low"
                ),
                "suggestion": item["message"],
                "impact": item["impact"],
            }
            for item in issues
        ]
        summary = (
            "渠道与业务硬约束校验通过。"
            if not issues
            else "；".join(item["message"] for item in issues)
        )
        return ReviewResult(
            version_id=version_id,
            content_hash=content_hash(content),
            review_type="channel_validation",
            decision=decision,
            blocking=blocking,
            summary=summary,
            agent="deterministic_validator",
            evidence=evidence,
        )

    def _save_channel_validation(
        self, workflow: CampaignWorkflow, version: ContentVersion
    ) -> None:
        validation = self._build_channel_validation(
            workflow, version.version_id, version.channel, version.content
        )
        context = self._frozen_review_context(workflow, version)
        validation.review_context_hash = context.review_context_hash
        validation.knowledge_snapshot_hash = version.knowledge_snapshot_hash
        self.store.save_review(
            workflow.workflow_id,
            validation,
            lock_token=self._lease_token(workflow.workflow_id),
        )

    def _recoverable_generated_versions(
        self, workflow: CampaignWorkflow
    ) -> List[ContentVersion]:
        brief_hash = workflow.brief.snapshot_hash()
        snapshot_hash = workflow.knowledge_snapshot.snapshot_hash
        candidates = [
            version
            for version in self.store.list_content_versions(workflow.workflow_id)
            if version.channel in workflow.brief.channels
            and version.brief_hash == brief_hash
            and version.knowledge_snapshot_hash == snapshot_hash
            and not version.parent_version_id
        ]
        latest_by_channel: Dict[str, ContentVersion] = {}
        for version in candidates:
            current = latest_by_channel.get(version.channel)
            if current is None or version.version > current.version:
                latest_by_channel[version.channel] = version
        return list(latest_by_channel.values())

    def _latest_versions(self, workflow: CampaignWorkflow) -> List[ContentVersion]:
        by_id = {
            version.version_id: version
            for version in self.store.list_content_versions(workflow.workflow_id)
        }
        return [
            by_id[workflow.latest_versions[channel]]
            for channel in workflow.brief.channels
            if workflow.latest_versions.get(channel) in by_id
        ]

    @staticmethod
    def _build_review_context(
        workflow: CampaignWorkflow, version: ContentVersion
    ) -> ReviewContext:
        snapshot_hash = workflow.knowledge_snapshot.snapshot_hash
        if version.knowledge_snapshot_hash != snapshot_hash:
            raise ReviewContextMismatchError(
                "ContentVersion 与冻结 ReviewContext 的知识快照不一致"
            )
        if version.brief_hash != workflow.brief.snapshot_hash():
            raise ReviewContextMismatchError("ContentVersion 与当前 Brief 不一致")
        if not version.content_hash or version.content_hash != content_hash(
            version.content
        ):
            raise ReviewContextMismatchError(
                "ContentVersion 缺少有效正文哈希，请创建新版本"
            )
        return build_review_context(workflow, version)

    def _frozen_review_context(
        self, workflow: CampaignWorkflow, version: ContentVersion
    ) -> ReviewContext:
        existing = self.store.get_review_context(
            workflow.workflow_id, version.version_id
        )
        if existing is not None:
            if (
                self._build_review_context(workflow, version).model_dump()
                != existing.model_dump()
            ):
                raise ReviewContextMismatchError(
                    "ContentVersion 与已冻结 ReviewContext 的知识快照不一致"
                )
            return existing
        context = self._build_review_context(workflow, version)
        return self.store.save_review_context(
            workflow.workflow_id,
            context,
            lock_token=self._lease_token(workflow.workflow_id),
        )

    @staticmethod
    def _review_specs() -> List[Tuple[str, WorkflowStage, IntentCategory, str]]:
        return [
            (
                "fact_check",
                WorkflowStage.FACT_CHECK,
                IntentCategory.RULE_CHECK,
                "rule_check",
            ),
            (
                "compliance",
                WorkflowStage.COMPLIANCE_REVIEW,
                IntentCategory.COMPLIANCE_CHECK,
                "compliance",
            ),
            (
                "content_quality",
                WorkflowStage.CONTENT_REVIEW,
                IntentCategory.CONTENT_REVIEW,
                "review",
            ),
        ]

    async def _run_parallel_reviews(
        self,
        workflow: CampaignWorkflow,
        versions: List[ContentVersion],
        contexts: List[ReviewContext],
    ) -> List[ReviewResult]:
        contexts_by_version = {item.version_id: item for item in contexts}
        tasks = []
        for version in versions:
            context = contexts_by_version[version.version_id]
            for review_type, stage, intent, agent in self._review_specs():
                latest = self.store.latest_review(
                    workflow.workflow_id,
                    version.version_id,
                    review_type,
                    context.review_context_hash,
                )
                if (
                    latest is not None
                    and latest.decision != ReviewDecision.EXECUTION_FAILED
                    and not latest.error_code
                ):
                    continue
                attempt = (latest.attempt + 1) if latest is not None else 1
                tasks.append(
                    self._run_review_agent(
                        workflow,
                        version,
                        context,
                        stage=stage,
                        intent=intent,
                        review_type=review_type,
                        agent=agent,
                        attempt=attempt,
                    )
                )
        if not tasks:
            return []
        outcomes = await asyncio.gather(*tasks, return_exceptions=True)
        lock_error = next(
            (item for item in outcomes if isinstance(item, WorkflowLockLostError)), None
        )
        if lock_error is not None:
            raise lock_error
        return [item for item in outcomes if isinstance(item, ReviewResult)]

    async def _run_review_agent(
        self,
        workflow: CampaignWorkflow,
        version: ContentVersion,
        review_context: ReviewContext,
        *,
        stage: WorkflowStage,
        intent: IntentCategory,
        review_type: str,
        agent: str,
        attempt: int,
    ) -> ReviewResult:
        event = self._start_event(workflow, stage, version.channel)
        event.attempt = attempt
        frozen_context = review_context.model_dump_json()
        raw_outputs = []
        try:
            result = await self._call_agent(
                Request(
                    message=self._review_message(review_context, review_type),
                    user_id=workflow.user_id,
                    conv_id=workflow.conv_id or workflow.workflow_id,
                    task_id=workflow.workflow_id,
                    context=f"[FROZEN_REVIEW_CONTEXT]\n{frozen_context}",
                    intent=intent,
                    secondary_intents=[],
                    trace_context={
                        "workflow": {
                            "workflow_id": workflow.workflow_id,
                            "stage": stage.value,
                            "channel": version.channel,
                            "review_context_hash": review_context.review_context_hash,
                            "knowledge_snapshot_hash": review_context.knowledge_snapshot_hash,
                            "expected_workflow_revision": workflow.revision,
                        },
                        "tool_mode": "frozen_only",
                    },
                ),
                event,
            )
            raw_outputs.append(str(result.response or ""))
            raw_outputs.extend(
                raw
                for record in getattr(result, "review_records", [])
                for raw in record.get("raw_outputs", [])
            )
            if (
                result.success
                and getattr(result.agent_type, "value", result.agent_type) != agent
            ):
                raise ValueError("必要审核不得由其他 Agent 降级替代")
            try:
                normalized = self._normalize_review_result(
                    result, review_type, frozen_context
                )
            except (ValueError, TypeError):
                result = await self._call_agent(
                    Request(
                        message=(
                            self._review_message(review_context, review_type)
                            + "\n\n上一次输出不符合审核信封约束。请修正且只返回合法 JSON：\n"
                            + str(result.response or "")[:4000]
                        ),
                        user_id=workflow.user_id,
                        conv_id=workflow.conv_id or workflow.workflow_id,
                        task_id=workflow.workflow_id,
                        context=f"[FROZEN_REVIEW_CONTEXT]\n{frozen_context}",
                        intent=intent,
                        secondary_intents=[],
                        trace_context={
                            "workflow": {
                                "workflow_id": workflow.workflow_id,
                                "stage": stage.value,
                                "channel": version.channel,
                                "review_context_hash": review_context.review_context_hash,
                                "knowledge_snapshot_hash": review_context.knowledge_snapshot_hash,
                                "expected_workflow_revision": workflow.revision,
                                "output_repair": True,
                            },
                            "tool_mode": "frozen_only",
                        },
                    ),
                    event,
                )
                raw_outputs.append(str(result.response or ""))
                raw_outputs.extend(
                    raw
                    for record in getattr(result, "review_records", [])
                    for raw in record.get("raw_outputs", [])
                )
                if (
                    result.success
                    and getattr(result.agent_type, "value", result.agent_type) != agent
                ):
                    raise ValueError("必要审核不得由其他 Agent 降级替代")
                normalized = self._normalize_review_result(
                    result, review_type, frozen_context
                )
                if normalized[0] != ReviewDecision.BLOCKED and any(
                    (self._json_object(raw) or {}).get("decision") == "blocked"
                    for raw in raw_outputs[:-1]
                ):
                    raise ValueError("格式修复不得消除已识别的阻断风险")
            decision, blocking, summary, evidence, need_revision, remediation = (
                normalized
            )
            review = ReviewResult(
                version_id=version.version_id,
                content_hash=version.content_hash,
                raw_outputs=raw_outputs,
                findings=(self._json_object(result.response) or {}).get("findings")
                or (self._json_object(result.response) or {}).get("risk_items")
                or [],
                review_type=review_type,
                decision=decision,
                blocking=blocking,
                summary=summary,
                agent=agent,
                request_id=result.request_id,
                evidence=evidence,
                review_context_hash=review_context.review_context_hash,
                knowledge_snapshot_hash=review_context.knowledge_snapshot_hash,
                attempt=attempt,
                error_code=result.error_code
                or (f"{review_type}_failed" if not result.success else None),
                need_revision=need_revision,
                remediation=remediation,
            )
            self.store.save_review(
                workflow.workflow_id,
                review,
                lock_token=self._lease_token(workflow.workflow_id),
            )
            event.request_id = result.request_id
            self._finish_event(
                event,
                EventStatus.SUCCESS if result.success else EventStatus.FAILED,
                error_code=result.error_code,
            )
            return review
        except WorkflowLockLostError:
            raise
        except Exception as ex:
            blocked_output = next(
                (
                    self._json_object(raw)
                    for raw in raw_outputs
                    if (self._json_object(raw) or {}).get("decision") == "blocked"
                ),
                None,
            )
            blocked_findings = (
                (blocked_output or {}).get("findings")
                or (blocked_output or {}).get("risk_items")
                or []
            )
            blocked_findings = (
                [item for item in blocked_findings if isinstance(item, dict)]
                if isinstance(blocked_findings, list)
                else []
            )
            retained_evidence = self._risk_evidence(
                json.dumps({"findings": blocked_findings}), review_type, frozen_context
            )
            review = ReviewResult(
                version_id=version.version_id,
                content_hash=version.content_hash,
                review_type=review_type,
                decision=(
                    ReviewDecision.BLOCKED
                    if blocked_output
                    else ReviewDecision.EXECUTION_FAILED
                ),
                blocking=bool(blocked_output),
                raw_outputs=raw_outputs,
                findings=blocked_findings,
                evidence=retained_evidence,
                summary="审核 Agent 执行失败。",
                agent=agent,
                review_context_hash=review_context.review_context_hash,
                knowledge_snapshot_hash=review_context.knowledge_snapshot_hash,
                attempt=attempt,
                error_code=f"{review_type}_failed",
                need_revision=True if blocked_output else None,
                remediation=(
                    ReviewRemediation.HUMAN_REQUIRED
                    if blocked_output
                    else ReviewRemediation.RETRY_REVIEW
                ),
            )
            self.store.save_review(
                workflow.workflow_id,
                review,
                lock_token=self._lease_token(workflow.workflow_id),
            )
            self._finish_event(
                event, EventStatus.FAILED, error_code=f"{review_type}_failed"
            )
            return review

    def _aggregate_reviews(
        self,
        workflow: CampaignWorkflow,
        versions: List[ContentVersion],
        contexts: List[ReviewContext],
        *,
        expected_workflow_revision: int,
    ) -> CampaignWorkflow:
        if workflow.revision != expected_workflow_revision:
            raise WorkflowConflictError("审核聚合使用了陈旧的 Workflow revision")
        qualification = delivery_eligibility(self.store.detail(workflow.workflow_id))
        any_incomplete = not qualification["review_complete"]
        any_content_change = False
        any_human_required = qualification["review_status"] == "blocked"
        for version in versions:
            state = qualification["versions"].get(version.version_id, {})
            if not state.get("review_complete"):
                version.status = "review_incomplete"
                any_incomplete = True
            elif state["review_status"] == "blocked":
                version.status = "changes_requested"
                any_content_change = True
                any_human_required = True
            elif state["review_status"] == "changes_required":
                version.status = "changes_requested"
                any_content_change = True
            else:
                version.status = "pending_approval"
        if any_incomplete:
            target = WorkflowStatus.REVIEW_INCOMPLETE
            stage = WorkflowStage.PARALLEL_REVIEW
            approval_status = "review_incomplete"
        elif any_content_change:
            target = WorkflowStatus.CHANGES_REQUESTED
            stage = WorkflowStage.PARALLEL_REVIEW
            approval_status = (
                "human_required" if any_human_required else "changes_requested"
            )
        else:
            target = WorkflowStatus.AWAITING_APPROVAL
            stage = WorkflowStage.HUMAN_APPROVAL
            approval_status = "pending"
        ensure_status_transition(workflow.status, target)
        updated = workflow.model_copy(
            update={
                "status": target,
                "current_stage": stage,
                "approval_status": approval_status,
                "last_error": None,
                "updated_at": utc_now(),
                "revision": expected_workflow_revision + 1,
            }
        )
        self._assert_active_lease(workflow.workflow_id)
        return self.store.save_review_aggregation(
            updated,
            versions,
            expected_revision=expected_workflow_revision,
            lock_token=self._lease_token(workflow.workflow_id),
        )

    def _mark_review_incomplete(
        self,
        workflow: CampaignWorkflow,
        versions: List[ContentVersion],
        expected_workflow_revision: int,
        error: str,
    ) -> CampaignWorkflow:
        for version in versions:
            version.status = "review_incomplete"
        ensure_status_transition(workflow.status, WorkflowStatus.REVIEW_INCOMPLETE)
        updated = workflow.model_copy(
            update={
                "status": WorkflowStatus.REVIEW_INCOMPLETE,
                "current_stage": WorkflowStage.PARALLEL_REVIEW,
                "approval_status": "review_incomplete",
                "last_error": error,
                "updated_at": utc_now(),
                "revision": expected_workflow_revision + 1,
            }
        )
        return self.store.save_review_aggregation(
            updated,
            versions,
            expected_revision=expected_workflow_revision,
            lock_token=self._lease_token(workflow.workflow_id),
        )

    def _save_status(
        self,
        workflow: CampaignWorkflow,
        status: WorkflowStatus,
        stage: WorkflowStage,
        **updates: Any,
    ) -> CampaignWorkflow:
        ensure_status_transition(workflow.status, status)
        expected = workflow.revision
        updated = workflow.model_copy(
            update={
                "status": status,
                "current_stage": stage,
                "updated_at": utc_now(),
                "revision": expected + 1,
                **updates,
            }
        )
        self._assert_active_lease(workflow.workflow_id)
        return self.store.save(
            updated,
            expected_revision=expected,
            lock_token=self._lease_token(workflow.workflow_id),
        )

    def _start_event(
        self,
        workflow: CampaignWorkflow,
        stage: WorkflowStage,
        channel: Optional[str] = None,
    ) -> WorkflowExecutionEvent:
        event = WorkflowExecutionEvent(
            usage_accounting_version=2,
            workflow_id=workflow.workflow_id,
            stage=stage,
            channel=channel,
            status=EventStatus.RUNNING,
            started_at=utc_now(),
        )
        return self.store.save_event(
            event, lock_token=self._lease_token(workflow.workflow_id)
        )

    def _finish_event(
        self,
        event: WorkflowExecutionEvent,
        status: EventStatus,
        error_code: Optional[str] = None,
    ) -> None:
        finished = utc_now()
        event.status = status
        event.finished_at = finished
        event.error_code = error_code
        if status == EventStatus.FAILED:
            WORKFLOW_STAGE_FAILURES.labels(
                stage=event.stage.value, error_code=error_code or "unknown"
            ).inc()
        if event.started_at:
            event.latency_ms = max(
                0.0, (finished - event.started_at).total_seconds() * 1000
            )
        self.store.save_event(event, lock_token=self._lease_token(event.workflow_id))

    @staticmethod
    def _review_decision(result: OrchestratorResult) -> Tuple[ReviewDecision, bool]:
        if not result.success:
            return ReviewDecision.EXECUTION_FAILED, True
        content = result.response or ""
        try:
            structured = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            structured = None
        if isinstance(structured, dict):
            decision = structured.get("decision")
            if decision == "blocked":
                return ReviewDecision.BLOCKED, True
            if decision == "changes_required":
                return ReviewDecision.CHANGES_REQUIRED, True
            if decision == "pass":
                return ReviewDecision.PASS, False
        requires_revision = bool(
            re.search(r"修订决策\s*[:：]\s*yes", content, re.IGNORECASE)
        )
        blocked = result.escalated or any(
            term in content for term in ("禁止发布", "高风险", "严重合规")
        )
        if blocked:
            return ReviewDecision.BLOCKED, True
        if requires_revision:
            return ReviewDecision.CHANGES_REQUIRED, True
        return ReviewDecision.PASS, False

    @staticmethod
    def _risk_evidence(
        content: str, review_type: str, knowledge_context: str
    ) -> List[Dict[str, Any]]:
        """Extract a conservative evidence record without inventing model claims."""
        text = content or ""
        try:
            structured = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            structured = None
        finding_items = []
        if isinstance(structured, dict):
            finding_items = (
                structured.get("findings") or structured.get("risk_items") or []
            )
        if isinstance(finding_items, list):
            result = []
            for item in finding_items:
                if not isinstance(item, dict):
                    continue
                raw_rule_id = str(item.get("rule_id") or review_type)
                rule_id = (
                    raw_rule_id
                    if review_type != "compliance" or raw_rule_id in RULES_BY_ID
                    else "unknown_rule"
                )
                configured_decision = str(
                    RULES_BY_ID.get(rule_id, {}).get("decision", "changes_required")
                )
                impact = str(
                    item.get("impact")
                    or (
                        "blocking_issue"
                        if configured_decision == "blocked"
                        else (
                            "non_blocking_suggestion"
                            if configured_decision == "pass"
                            else "required_change"
                        )
                    )
                )
                if impact not in {
                    "blocking_issue",
                    "required_change",
                    "non_blocking_suggestion",
                    "delivery_configuration",
                }:
                    impact = "required_change"
                result.append(
                    {
                        "rule_id": rule_id,
                        "category": rule_id,
                        "matched_text": str(item.get("matched_text") or "")[:300],
                        "source": (
                            ", ".join(
                                str(ref)
                                for ref in item.get("evidence_refs", [])
                                if isinstance(ref, str)
                            )
                            if isinstance(item.get("evidence_refs", []), list)
                            and item.get("evidence_refs")
                            else "Campaign Brief"
                        ),
                        "severity": str(item.get("severity") or "warning"),
                        "suggestion": str(item.get("suggestion") or "")[:500],
                        "impact": impact,
                        "reason": str(item.get("reason") or ""),
                        "evidence_refs": (
                            [
                                ref
                                for ref in item.get("evidence_refs", [])
                                if isinstance(ref, str)
                            ]
                            if isinstance(item.get("evidence_refs", []), list)
                            else []
                        ),
                    }
                )
            return result
        severity = (
            "high"
            if any(term in text for term in ("禁止发布", "高风险", "严重合规"))
            else "warning"
        )
        if re.search(r"修订决策\s*[:：]\s*no", text, re.IGNORECASE) or any(
            phrase in text for phrase in ("未发现阻塞问题", "未发现风险", "审核通过")
        ):
            return []
        if not any(
            term in text for term in ("风险", "问题", "修订决策: yes", "修订决策：yes")
        ):
            return []
        matched = next(
            (line.strip() for line in text.splitlines() if line.strip()), text[:160]
        )
        source = (
            "Campaign Brief / 冻结知识快照"
            if "[KNOWLEDGE_SNAPSHOT]" in knowledge_context
            else "Campaign Brief"
        )
        return [
            {
                "rule_id": "unknown_rule",
                "category": "unknown_rule",
                "matched_text": matched[:300],
                "source": source,
                "severity": severity,
                "suggestion": "按审核摘要修订并重新提交人工审批。",
                "impact": "required_change",
            }
        ]

    @classmethod
    def _normalize_review_result(
        cls,
        result: OrchestratorResult,
        review_type: str,
        knowledge_context: str,
    ) -> Tuple[
        ReviewDecision,
        bool,
        str,
        List[Dict[str, Any]],
        Optional[bool],
        ReviewRemediation,
    ]:
        structured = cls._json_object(result.response)
        evidence = cls._risk_evidence(result.response, review_type, knowledge_context)
        summary = str(
            (structured or {}).get("summary") or result.response or "未提供审核摘要"
        ).strip()
        if not result.success:
            return (
                (
                    ReviewDecision.BLOCKED
                    if (structured or {}).get("decision") == "blocked"
                    else ReviewDecision.EXECUTION_FAILED
                ),
                True,
                summary,
                evidence,
                True if (structured or {}).get("decision") == "blocked" else None,
                (
                    ReviewRemediation.HUMAN_REQUIRED
                    if (structured or {}).get("decision") == "blocked"
                    else ReviewRemediation.RETRY_REVIEW
                ),
            )
        raw_decision = str((structured or {}).get("decision") or "")
        structured = structured_review(structured)
        expected_scope = {
            "fact_check": "fact_rule",
            "compliance": "compliance",
            "content_quality": "content_quality",
        }[review_type]
        if structured.get("scope", expected_scope) != expected_scope:
            raise ValueError("review scope does not match requested reviewer")
        need_revision_value = structured["need_revision"]

        # Safe-degraded constraints are no longer active requirements. Content Review
        # must not reintroduce them through a completeness finding.
        if (
            review_type == "content_quality"
            and raw_decision != "blocked"
            and isinstance(structured, dict)
        ):
            try:
                frozen = json.loads(knowledge_context)
            except (json.JSONDecodeError, TypeError):
                frozen = {}
            rejected = (
                [
                    str(item).strip()
                    for item in (frozen.get("rejected_constraints") or [])
                    if str(item).strip()
                ]
                if isinstance(frozen, dict)
                else []
            )
            findings = structured.get("findings") or structured.get("risk_items") or []
            if rejected and isinstance(findings, list):

                def mentions_rejected(item: Any) -> bool:
                    if not isinstance(item, dict):
                        return False
                    text = " ".join(
                        str(item.get(key) or "")
                        for key in (
                            "matched_text",
                            "reason",
                            "suggestion",
                        )
                    )
                    return any(value in text for value in rejected)

                retained = [item for item in findings if not mentions_rejected(item)]
                removed = len(retained) != len(findings)
                if removed:
                    structured = dict(structured)
                    structured["findings"] = retained
                    evidence = [
                        item
                        for item in evidence
                        if not any(
                            value
                            in " ".join(
                                (
                                    str(item.get("matched_text") or ""),
                                    str(item.get("suggestion") or ""),
                                )
                            )
                            for value in rejected
                        )
                    ]
                    if not retained:
                        raw_decision = "pass"
                        need_revision_value = False
                        summary = (
                            "已排除生成阶段安全删除的约束；未发现其他内容质量问题。"
                        )

        # A RuleCheck result containing advisory findings only cannot block content.
        if (
            review_type == "fact_check"
            and raw_decision != "blocked"
            and evidence
            and all(
                item.get("impact") == "non_blocking_suggestion" for item in evidence
            )
        ):
            raw_decision = "pass"
            need_revision_value = False
        envelope = ReviewEnvelope(
            scope={
                "fact_check": "fact_rule",
                "compliance": "compliance",
                "content_quality": "content_quality",
            }[review_type],
            decision=ReviewDecision(raw_decision),
            need_revision=need_revision_value,
            summary=summary,
            findings=(structured or {}).get("findings")
            or (structured or {}).get("risk_items")
            or [],
        )
        remediation = {
            ReviewDecision.PASS: ReviewRemediation.NONE,
            ReviewDecision.CHANGES_REQUIRED: ReviewRemediation.REVISION_ALLOWED,
            ReviewDecision.BLOCKED: ReviewRemediation.HUMAN_REQUIRED,
        }[envelope.decision]
        return (
            envelope.decision,
            envelope.decision != ReviewDecision.PASS,
            envelope.summary,
            evidence,
            envelope.need_revision,
            remediation,
        )

    @staticmethod
    def _retrieval_query(workflow: CampaignWorkflow) -> str:
        brief = workflow.brief
        facts = " ".join(f"{key} {value}" for key, value in brief.facts.items())
        return " ".join(
            filter(
                None,
                [
                    brief.campaign_name,
                    brief.product,
                    brief.objective,
                    facts,
                    "活动规则 品牌规范 渠道规范 合规要求",
                ],
            )
        )

    @staticmethod
    def _knowledge_context(snapshot: KnowledgeSnapshot) -> str:
        if not snapshot.documents:
            return (
                "[KNOWLEDGE_SNAPSHOT]\n本次未检索到可用知识文档。不得因此补造活动事实。"
            )
        parts = ["[KNOWLEDGE_SNAPSHOT]"]
        for index, doc in enumerate(snapshot.documents, 1):
            parts.append(
                f"{index}. {doc.get('title', '')}\n"
                f"document_id: {doc.get('document_id') or '未提供'}\n"
                f"发布日期: {doc.get('published_at') or '未明确'}\n"
                f"来源: {doc.get('source_url') or '内部治理知识'}\n"
                f"内容: {doc.get('content', '')}"
            )
        return "\n\n".join(parts)

    @staticmethod
    def _select_campaign_documents(
        workflow: CampaignWorkflow, items: Any
    ) -> List[Dict[str, Any]]:
        candidates = [item for item in (items or []) if isinstance(item, dict)]
        selected: List[Dict[str, Any]] = []
        product_names = [
            value.strip()
            for value in re.split(r"[、,，；;和与]", workflow.brief.product)
            if value.strip()
        ]
        for product_name in product_names:
            match = next(
                (
                    item
                    for item in candidates
                    if item not in selected
                    and item.get("document_type") == "product_fact"
                    and product_name in str(item.get("title", ""))
                ),
                None,
            )
            if match is not None:
                selected.append(match)
        campaign_rule = next(
            (
                item
                for item in candidates
                if item not in selected and item.get("document_type") == "campaign_rule"
            ),
            None,
        )
        if campaign_rule is not None:
            selected.insert(0, campaign_rule)
        for item in candidates:
            if item not in selected:
                selected.append(item)
            if len(selected) >= 3:
                break
        return selected[:3]

    @staticmethod
    def _brief_text(workflow: CampaignWorkflow) -> str:
        brief = workflow.brief
        facts = (
            "\n".join(f"- {key}: {value}" for key, value in brief.facts.items())
            or "- 未提供额外事实"
        )
        constraints = (
            "\n".join(f"- {item}" for item in brief.constraints) or "- 无额外约束"
        )
        return (
            f"活动名称: {brief.campaign_name}\n目标: {brief.objective}\n商品: {brief.product}\n"
            f"受众: {brief.audience}\n渠道: {', '.join(brief.channels)}\n"
            f"明确事实:\n{facts}\n约束:\n{constraints}\n补充说明: {brief.brief_text or '无'}"
        )

    def _agent_context(
        self, workflow: CampaignWorkflow, knowledge_context: str, channel: str
    ) -> str:
        return (
            f"[CAMPAIGN_WORKFLOW id={workflow.workflow_id}]\n"
            f"{self._brief_text(workflow)}\n当前渠道: {channel}\n\n{knowledge_context}\n\n"
            "[WORKFLOW_BOUNDARY]\n只能使用当前 Campaign Brief 与本次冻结的知识快照。"
            "不得继承普通聊天历史、其他活动、用户长期偏好或未提供的价格、权益、期限和库存。"
        )

    @staticmethod
    def _generation_identity(
        workflow: CampaignWorkflow,
    ) -> Tuple[str, str, str, List[str]]:
        brief_hash = workflow.brief.snapshot_hash()
        snapshot_hash = workflow.knowledge_snapshot.snapshot_hash
        channels = sorted(workflow.brief.channels)
        payload = json.dumps(
            {
                "workflow_id": workflow.workflow_id,
                "brief_hash": brief_hash,
                "knowledge_snapshot_hash": snapshot_hash,
                "channels": channels,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return (
            brief_hash,
            snapshot_hash,
            hashlib.sha256(payload.encode("utf-8")).hexdigest(),
            channels,
        )

    def _generation_attempt(
        self, workflow: CampaignWorkflow
    ) -> Tuple[CampaignWorkflow, GenerationAttempt]:
        brief_hash, snapshot_hash, context_hash, channels = self._generation_identity(
            workflow
        )
        existing = (
            self.store.get_generation_attempt(workflow.active_generation_attempt_id)
            if workflow.active_generation_attempt_id
            else None
        )
        if existing is not None:
            matches = (
                existing.workflow_id == workflow.workflow_id
                and existing.brief_hash == brief_hash
                and existing.knowledge_snapshot_hash == snapshot_hash
                and existing.generation_context_hash == context_hash
                and existing.channels == channels
            )
            if matches:
                return workflow, existing
        attempt = GenerationAttempt(
            workflow_id=workflow.workflow_id,
            brief_hash=brief_hash,
            knowledge_snapshot_hash=snapshot_hash,
            generation_context_hash=context_hash,
            channels=channels,
        )
        self._assert_active_lease(workflow.workflow_id)
        self.store.save_generation_attempt(
            attempt, lock_token=self._lease_token(workflow.workflow_id)
        )
        workflow = self._save_status(
            workflow,
            WorkflowStatus.GENERATING,
            WorkflowStage.CONTENT_GENERATION,
            active_generation_attempt_id=attempt.generation_attempt_id,
        )
        return workflow, attempt

    @staticmethod
    def _validate_generation_result_context(
        workflow: CampaignWorkflow,
        attempt: GenerationAttempt,
        result: ChannelGenerationResult,
    ) -> None:
        if not (
            result.workflow_id == workflow.workflow_id == attempt.workflow_id
            and result.generation_attempt_id == attempt.generation_attempt_id
            and result.brief_hash == attempt.brief_hash
            and result.knowledge_snapshot_hash == attempt.knowledge_snapshot_hash
            and result.generation_context_hash == attempt.generation_context_hash
            and result.channels == attempt.channels
            and result.channel in attempt.channels
        ):
            raise WorkflowConflictError("生成暂存结果上下文不匹配，禁止复用")

    @staticmethod
    def _json_object(content: str) -> Optional[Dict[str, Any]]:
        text = str(content or "").strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        try:
            value = json.loads(text)
            return value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            start, end = text.find("{"), text.rfind("}")
            if start >= 0 and end > start:
                try:
                    value = json.loads(text[start : end + 1])
                    return value if isinstance(value, dict) else None
                except json.JSONDecodeError:
                    return None
            return None

    def _valid_generation_payload(
        self, workflow: CampaignWorkflow, channel: str, content: str
    ) -> Optional[Dict[str, Any]]:
        value = self._json_object(content)
        if value is None or str(value.get("channel", "")).strip() != channel:
            return None
        body = str(value.get("content", "")).strip()
        if (
            not body
            or not isinstance(value.get("citation_ids", []), list)
            or not isinstance(value.get("unresolved_fields", []), list)
        ):
            return None
        allowed = {
            str(item.get("document_id", "")).strip()
            for item in workflow.knowledge_snapshot.documents
            if str(item.get("document_id", "")).strip()
        }
        citations = list(
            dict.fromkeys(
                str(item).strip()
                for item in value.get("citation_ids", [])
                if str(item).strip()
            )
        )
        if any(item not in allowed for item in citations):
            return None
        return {
            "channel": channel,
            "content": body,
            "citation_ids": citations,
            "unresolved_fields": list(
                dict.fromkeys(
                    str(item).strip()
                    for item in value.get("unresolved_fields", [])
                    if str(item).strip()
                )
            ),
        }

    def _valid_copywriting_envelope(
        self, workflow: CampaignWorkflow, channel: str, content: str
    ) -> Optional[Dict[str, Any]]:
        value = self._json_object(content)
        if value is None or str(value.get("channel", "")).strip() != channel:
            return None
        generation_status = str(value.get("generation_status") or "").strip()
        if generation_status not in {"success", "clarification_required"}:
            return None
        conflicts = value.get("conflicts", [])
        rejected = value.get("rejected_constraints", [])
        unresolved = value.get("unresolved_fields", [])
        citations = value.get("citation_ids", [])
        if not all(
            isinstance(item, list)
            for item in (conflicts, rejected, unresolved, citations)
        ):
            return None
        allowed = {
            str(item.get("document_id", "")).strip()
            for item in workflow.knowledge_snapshot.documents
            if str(item.get("document_id", "")).strip()
        }
        citation_ids = list(
            dict.fromkeys(str(item).strip() for item in citations if str(item).strip())
        )
        if any(item not in allowed for item in citation_ids):
            return None
        body = str(value.get("content", "")).strip()
        normalized_conflicts: List[Dict[str, Any]] = []
        for item in conflicts:
            if not isinstance(item, dict):
                return None
            try:
                conflict = GenerationConflict.model_validate(item)
            except Exception:
                return None
            if not conflict.requested_value or not conflict.verified_value:
                return None
            if not conflict.brief_refs or not conflict.evidence_refs:
                return None
            normalized_conflicts.append(conflict.model_dump(mode="json"))
        if generation_status == "success" and (not body or normalized_conflicts):
            return None
        if generation_status == "clarification_required" and (
            body or not normalized_conflicts
        ):
            return None
        return {
            "generation_status": generation_status,
            "channel": channel,
            "content": body,
            "citation_ids": citation_ids,
            "unresolved_fields": list(
                dict.fromkeys(
                    str(item).strip() for item in unresolved if str(item).strip()
                )
            ),
            "rejected_constraints": list(
                dict.fromkeys(
                    str(item).strip() for item in rejected if str(item).strip()
                )
            ),
            "conflicts": normalized_conflicts,
        }

    async def _ensure_copywriting_envelope(
        self,
        workflow: CampaignWorkflow,
        channel: str,
        result: OrchestratorResult,
        event: WorkflowExecutionEvent,
        knowledge_context: str,
    ) -> Tuple[Dict[str, Any], OrchestratorResult]:
        envelope = self._valid_copywriting_envelope(workflow, channel, result.response)
        if envelope is not None:
            return envelope, result
        repaired = await self._call_agent(
            Request(
                message=(
                    "只修复下面结果的生成信封格式，不重新创作文案，也不要增加事实。"
                    f"channel 必须是 {channel}。generation_status 只能是 success 或 clarification_required。"
                    "success 必须提供可交付 content 且 conflicts 为空；clarification_required 必须让 content 为空，"
                    "并提供至少一项含 category、brief_refs、evidence_refs、requested_value、verified_value、suggestion 的 conflicts。"
                    "同时返回 citation_ids、unresolved_fields、rejected_constraints 数组。只返回 JSON。\n原结果:\n"
                    + result.response
                ),
                user_id=workflow.user_id,
                conv_id=workflow.conv_id or workflow.workflow_id,
                task_id=workflow.workflow_id,
                context=self._agent_context(workflow, knowledge_context, channel),
                intent=IntentCategory.COPY_REWRITE,
                secondary_intents=[],
                trace_context={
                    "workflow": {
                        "workflow_id": workflow.workflow_id,
                        "stage": "generation_envelope_repair",
                        "channel": channel,
                    }
                },
            ),
            event,
        )
        envelope = self._valid_copywriting_envelope(
            workflow, channel, repaired.response
        )
        if envelope is None:
            raise WorkflowExecutionError("generation_output_schema_invalid")
        return envelope, repaired

    async def _ensure_generation_format(
        self,
        workflow: CampaignWorkflow,
        channel: str,
        result: OrchestratorResult,
        event: WorkflowExecutionEvent,
        knowledge_context: str,
    ) -> Tuple[Dict[str, Any], OrchestratorResult]:
        payload = self._valid_generation_payload(workflow, channel, result.response)
        if payload is not None:
            return payload, result
        repaired = await self._call_agent(
            Request(
                message=(
                    "只修复下面结果的格式，不重新创作文案，也不要增加事实。"
                    f"channel 必须是 {channel}；content 只能包含最终可交付正文；"
                    "citation_ids 只能使用知识快照中的 document_id；unresolved_fields 为字符串数组。"
                    "只返回 JSON 对象。\n原结果:\n" + result.response
                ),
                user_id=workflow.user_id,
                conv_id=workflow.conv_id or workflow.workflow_id,
                task_id=workflow.workflow_id,
                context=self._agent_context(workflow, knowledge_context, channel),
                intent=IntentCategory.COPY_REWRITE,
                secondary_intents=[],
                trace_context={
                    "workflow": {
                        "workflow_id": workflow.workflow_id,
                        "stage": "generation_format_repair",
                        "channel": channel,
                    }
                },
            ),
            event,
        )
        payload = self._valid_generation_payload(workflow, channel, repaired.response)
        if payload is None:
            raise WorkflowExecutionError("generation_output_schema_invalid")
        return payload, repaired

    async def _ensure_generation_payload(
        self,
        workflow: CampaignWorkflow,
        channel: str,
        result: OrchestratorResult,
        event: WorkflowExecutionEvent,
        knowledge_context: str,
    ) -> Tuple[Dict[str, Any], OrchestratorResult]:
        payload = self._valid_generation_payload(workflow, channel, result.response)
        if payload is None:
            repaired = await self._call_agent(
                Request(
                    message=(
                        "只修复下面结果的格式，不重新创作文案，也不要增加事实。"
                        f"channel 必须是 {channel}；content 只能包含最终可交付正文；"
                        "citation_ids 只能使用知识快照中的 document_id；unresolved_fields 为字符串数组。"
                        "只返回 JSON 对象。\n原结果:\n" + result.response
                    ),
                    user_id=workflow.user_id,
                    conv_id=workflow.conv_id or workflow.workflow_id,
                    task_id=workflow.workflow_id,
                    context=self._agent_context(workflow, knowledge_context, channel),
                    intent=IntentCategory.COPY_REWRITE,
                    secondary_intents=[],
                    trace_context={
                        "workflow": {
                            "workflow_id": workflow.workflow_id,
                            "stage": "generation_format_repair",
                            "channel": channel,
                        }
                    },
                ),
                event,
            )
            payload = self._valid_generation_payload(
                workflow, channel, repaired.response
            )
            result = repaired
        if payload is None:
            raise WorkflowExecutionError("generation_output_schema_invalid")
        issues = validate_generated_content(
            channel, payload["content"], workflow.knowledge_snapshot.documents
        )
        if requires_revision(issues):
            revised = await self._call_agent(
                Request(
                    message=(
                        "根据确定性校验结果修订正文，只修复列出的问题，不增加新事实。"
                        "只返回 channel、content、citation_ids、unresolved_fields 四个字段的 JSON 对象。\n"
                        f"校验问题: {json.dumps(issues, ensure_ascii=False)}\n原结果: {json.dumps(payload, ensure_ascii=False)}"
                    ),
                    user_id=workflow.user_id,
                    conv_id=workflow.conv_id or workflow.workflow_id,
                    task_id=workflow.workflow_id,
                    context=self._agent_context(workflow, knowledge_context, channel),
                    intent=IntentCategory.COPY_REWRITE,
                    secondary_intents=[],
                    trace_context={
                        "workflow": {
                            "workflow_id": workflow.workflow_id,
                            "stage": "deterministic_validation_revision",
                            "channel": channel,
                        }
                    },
                ),
                event,
            )
            payload = self._valid_generation_payload(
                workflow, channel, revised.response
            )
            remaining_issues = (
                []
                if payload is None
                else validate_generated_content(
                    channel, payload["content"], workflow.knowledge_snapshot.documents
                )
            )
            if payload is None or requires_revision(remaining_issues):
                labels = ",".join(
                    sorted(
                        {item.get("rule_id", "unknown") for item in remaining_issues}
                    )
                )
                raise WorkflowExecutionError(
                    "deterministic_business_validation_failed"
                    + (f":{labels}" if labels else ":invalid_payload")
                )
            result = revised
        return payload, result

    def _generation_message(self, workflow: CampaignWorkflow, channel: str) -> str:
        return (
            f"为当前活动生成一份{channel}渠道内容草稿。"
            "只返回 JSON 对象，字段为 generation_status、channel、content、citation_ids、"
            "unresolved_fields、rejected_constraints、conflicts。generation_status 只能是 success 或 clarification_required。"
            "content 只能包含可直接交付的正文，不得包含面向运营人员的选稿理由、备选稿、审核结论、优化建议或内部说明；面向读者的推荐亮点属于正文，可自然呈现。"
            "金额、门槛、有效期、适用范围、资格、SKU 或定量产品事实与已确认依据直接冲突时，"
            "返回 clarification_required、空 content，并在 conflicts 中提供 category、brief_refs、evidence_refs、"
            "requested_value、verified_value、suggestion。无依据功效、最高级或夸张营销要求应删除并写入"
            "rejected_constraints，仍返回安全正文。正文非必需的缺失信息写入 unresolved_fields。"
            "citation_ids 只能填写本次知识快照中的 document_id。\n"
            f"{self._brief_text(workflow)}"
        )

    def _review_message(self, context: ReviewContext, review_type: str) -> str:
        task = {
            "fact_check": (
                "只核验正文实际声明或明确暗示的业务事实和活动规则。只有正文中的声明缺少冻结证据、"
                "与冻结证据冲突，或通过省略限定造成错误适用暗示时，才可判定 changes_required 或 blocked。"
                "正文没有提及的叠加、库存、名额、退款等非必需规则缺失，只能记录为 impact=non_blocking_suggestion "
                "的 advisory，不得单独改变 pass 决策。"
                "不要判断广告法、平台合规、品牌语气或文案质量。"
            ),
            "compliance": (
                "只审核广告和平台合规：禁用词、绝对化宣传、无依据功效、虚假承诺和披露风险。"
                "不要检查活动字段完整性、渠道文风或转化效果。"
            ),
            "content_quality": (
                "只审核内容质量：需求相关性、结构、品牌语气、可读性、CTA 和渠道表达。"
                "小红书种草审核应区分事实依据与创意表达：生活共鸣、情绪、联名兴趣和角色偏好可以作为推荐吸引点；"
                "资料只支持系列特点时，允许围绕整个活动展开，不要求逐款制造独立口味、功效或人群差异。"
                "不能仅因采用共同系列特点或角色偏好就判定推荐理由不足；"
                "若要求修改，必须指出具体空泛、重复或偏离需求的文字及影响。"
                "不得因语气热情、短句或适量表情直接要求克制，也不得把增加无依据差异作为修订建议。"
                "当渠道为小红书且需求为种草时，检查推荐主题、场景关联、可验证的推荐理由和扫读节奏；"
                "多款产品可分款介绍，单款可集中展开。"
                "不强制角色对话、指定互动句或固定分段数量，不因口语或适量表情直接要求改为公告体。"
                "仅有表情和互动句不能视为已满足种草需求。"
                "修改建议须指出具体不足，不得要求补写无证据的口味、体验或优惠。"
                "ReviewContext.rejected_constraints 是生成阶段已安全删除的无效约束，必须从需求完整性检查中排除；"
                "不得因为正文未包含这些约束而要求补回或判定 changes_required。只要求正文满足当前仍有效的约束。"
                "不要判断业务事实真伪、活动规则或法律合规。"
            ),
        }[review_type]
        scope = {
            "fact_check": "fact_rule",
            "compliance": "compliance",
            "content_quality": "content_quality",
        }[review_type]
        return (
            f"任务：{task}\n渠道：{context.channel}\n"
            "只能使用背景中的冻结 ReviewContext，不得重新检索或调用业务工具。"
            "不得因为正文未涉及库存、价格、活动或配置就假设其必须存在。"
            "只返回 JSON 对象，字段为 scope、decision、need_revision、summary、findings。"
            "decision 只能是 pass、changes_required、blocked；pass 对应 need_revision=false，"
            "其余两者对应 true。findings 每项包含 rule_id、severity、matched_text、"
            "evidence_refs、reason、suggestion、impact。"
            f"scope 必须为 {scope}。\n\n"
            "冻结 ReviewContext 由背景消息中的 FROZEN_REVIEW_CONTEXT 提供。"
        )
