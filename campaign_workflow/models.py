from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


class WorkflowStatus(str, Enum):
    DRAFT = "draft"
    VALIDATED = "validated"
    PLANNED = "planned"
    RETRIEVING = "retrieving"
    GENERATING = "generating"
    NEEDS_CLARIFICATION = "needs_clarification"
    REVIEWING = "reviewing"
    REVIEW_INCOMPLETE = "review_incomplete"
    FACT_CHECKING = "fact_checking"
    COMPLIANCE_REVIEWING = "compliance_reviewing"
    AWAITING_APPROVAL = "awaiting_approval"
    CHANGES_REQUESTED = "changes_requested"
    APPROVED = "approved"
    DELIVERED = "delivered"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WorkflowStage(str, Enum):
    BRIEF_VALIDATION = "brief_validation"
    PLANNING = "planning"
    KNOWLEDGE_RETRIEVAL = "knowledge_retrieval"
    CONTENT_GENERATION = "content_generation"
    PARALLEL_REVIEW = "parallel_review"
    FACT_CHECK = "fact_check"
    COMPLIANCE_REVIEW = "compliance_review"
    CONTENT_REVIEW = "content_review"
    HUMAN_APPROVAL = "human_approval"
    DELIVERY = "delivery"


class ApprovalDecision(str, Enum):
    APPROVED = "approved"
    REJECTED = "rejected"
    CHANGES_REQUESTED = "changes_requested"


class ReviewDecision(str, Enum):
    PASS = "pass"
    CHANGES_REQUIRED = "changes_required"
    BLOCKED = "blocked"
    EXECUTION_FAILED = "execution_failed"


class ReviewRemediation(str, Enum):
    NONE = "none"
    REVISION_ALLOWED = "revision_allowed"
    HUMAN_REQUIRED = "human_required"
    RETRY_REVIEW = "retry_review"


class EventStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class GenerationStatus(str, Enum):
    COPY_READY = "copy_ready"
    STYLE_READY = "style_ready"
    READY_TO_COMMIT = "ready_to_commit"
    CLARIFICATION_REQUIRED = "clarification_required"
    EXECUTION_FAILED = "execution_failed"


class GenerationAttemptState(str, Enum):
    ACTIVE = "active"
    NEEDS_CLARIFICATION = "needs_clarification"
    COMMITTED = "committed"


class CampaignModel(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=False)


class RiskEvidence(CampaignModel):
    rule_id: str = "general"
    category: str = "general"
    matched_text: str = ""
    source: str = ""
    severity: str = "info"
    suggestion: str = ""
    impact: str = "required_change"
    reason: str = ""
    evidence_refs: List[str] = Field(default_factory=list)


class CampaignBrief(CampaignModel):
    campaign_name: str = Field(min_length=1, max_length=200)
    objective: str = Field(min_length=1, max_length=1000)
    product: str = Field(min_length=1, max_length=500)
    audience: str = Field(min_length=1, max_length=500)
    channels: List[str] = Field(min_length=1, max_length=10)
    facts: Dict[str, str] = Field(default_factory=dict)
    constraints: List[str] = Field(default_factory=list, max_length=30)
    brief_text: str = Field(default="", max_length=5000)

    @field_validator("campaign_name", "objective", "product", "audience", "brief_text")
    @classmethod
    def strip_text(cls, value: str) -> str:
        return value.strip()

    @field_validator("channels")
    @classmethod
    def normalize_channels(cls, values: List[str]) -> List[str]:
        normalized = [str(value).strip() for value in values if str(value).strip()]
        unique = list(dict.fromkeys(normalized))
        if not unique:
            raise ValueError("至少需要一个有效渠道")
        return unique

    @field_validator("facts")
    @classmethod
    def normalize_facts(cls, values: Dict[str, str]) -> Dict[str, str]:
        return {
            str(key).strip(): str(value).strip()
            for key, value in values.items()
            if str(key).strip() and str(value).strip()
        }

    @field_validator("constraints")
    @classmethod
    def normalize_constraints(cls, values: List[str]) -> List[str]:
        return list(
            dict.fromkeys(str(value).strip() for value in values if str(value).strip())
        )

    def snapshot_hash(self) -> str:
        payload = json.dumps(
            self.model_dump(mode="json"), ensure_ascii=False, sort_keys=True
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class WorkflowPlanItem(CampaignModel):
    stage: WorkflowStage
    channel: Optional[str] = None
    order: int = Field(ge=1)


class KnowledgeSnapshot(CampaignModel):
    retrieved_at: Optional[datetime] = None
    queries: List[str] = Field(default_factory=list)
    documents: List[Dict[str, Any]] = Field(default_factory=list)
    snapshot_hash: str = ""


class ContentVersion(CampaignModel):
    version_id: str = Field(default_factory=lambda: new_id("cv"))
    workflow_id: str
    channel: str
    version: int = Field(ge=1)
    content: str
    content_hash: str = ""
    citation_ids: List[str] = Field(default_factory=list)
    unresolved_fields: List[str] = Field(default_factory=list)
    rejected_constraints: List[str] = Field(default_factory=list)
    status: str = "pending_approval"
    brief_hash: str
    knowledge_snapshot_hash: str = ""
    generation_request_id: str = ""
    parent_version_id: Optional[str] = None
    created_at: datetime = Field(default_factory=utc_now)


class GenerationConflict(CampaignModel):
    category: str = "business_fact_conflict"
    brief_refs: List[str] = Field(default_factory=list)
    evidence_refs: List[str] = Field(default_factory=list)
    requested_value: str = ""
    verified_value: str = ""
    suggestion: str = ""


class GenerationAttempt(CampaignModel):
    generation_attempt_id: str = Field(default_factory=lambda: new_id("generation"))
    workflow_id: str
    brief_hash: str
    knowledge_snapshot_hash: str
    generation_context_hash: str
    channels: List[str]
    state: GenerationAttemptState = GenerationAttemptState.ACTIVE
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @field_validator("channels")
    @classmethod
    def normalize_attempt_channels(cls, values: List[str]) -> List[str]:
        normalized = sorted(
            set(str(value).strip() for value in values if str(value).strip())
        )
        if not normalized:
            raise ValueError("GenerationAttempt 至少需要一个渠道")
        return normalized


class ChannelGenerationResult(CampaignModel):
    generation_result_id: str = Field(
        default_factory=lambda: new_id("generation_result")
    )
    generation_attempt_id: str
    workflow_id: str
    channel: str
    brief_hash: str
    knowledge_snapshot_hash: str
    generation_context_hash: str
    channels: List[str]
    status: GenerationStatus
    result_revision: int = Field(default=1, ge=1)
    content_version_id: str = Field(default_factory=lambda: new_id("cv"))
    draft_content: str = ""
    final_content: str = ""
    citation_ids: List[str] = Field(default_factory=list)
    unresolved_fields: List[str] = Field(default_factory=list)
    rejected_constraints: List[str] = Field(default_factory=list)
    conflicts: List[GenerationConflict] = Field(default_factory=list)
    copywriting_request_id: str = ""
    style_request_id: str = ""
    validation_revision_request_id: str = ""
    phase_attempts: Dict[str, int] = Field(default_factory=dict)
    error_code: Optional[str] = None
    channel_validation: Optional[Dict[str, Any]] = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @field_validator("channels")
    @classmethod
    def normalize_result_channels(cls, values: List[str]) -> List[str]:
        return sorted(set(str(value).strip() for value in values if str(value).strip()))


class ReviewContext(CampaignModel):
    workflow_id: str
    version_id: str
    channel: str
    content: str
    content_hash: str = ""
    version: Optional[int] = None
    citation_ids: List[str] = Field(default_factory=list)
    unresolved_fields: List[str] = Field(default_factory=list)
    brief: Dict[str, Any]
    brief_hash: str
    knowledge_snapshot_hash: str
    evidence: List[Dict[str, Any]] = Field(default_factory=list)
    evidence_refs: List[str] = Field(default_factory=list)
    rejected_constraints: List[str] = Field(default_factory=list)
    review_context_hash: str


class ReviewEnvelope(CampaignModel):
    scope: str
    decision: ReviewDecision
    need_revision: bool
    summary: str = ""
    findings: List[Dict[str, Any]] = Field(default_factory=list)

    @model_validator(mode="after")
    def decision_matches_revision_flag(self) -> "ReviewEnvelope":
        if self.decision == ReviewDecision.EXECUTION_FAILED:
            raise ValueError("execution_failed 只能由运行系统创建")
        expected = self.decision in {
            ReviewDecision.CHANGES_REQUIRED,
            ReviewDecision.BLOCKED,
        }
        if self.need_revision != expected:
            raise ValueError(
                f"decision={self.decision.value} requires need_revision={str(expected).lower()}"
            )
        return self


class ReviewResult(CampaignModel):
    review_id: str = Field(default_factory=lambda: new_id("review"))
    version_id: str
    content_hash: str = ""
    findings: List[Dict[str, Any]] = Field(default_factory=list)
    raw_outputs: List[str] = Field(default_factory=list)
    review_type: str
    decision: ReviewDecision
    blocking: bool = False
    summary: str = ""
    agent: str
    request_id: str = ""
    evidence: List[RiskEvidence] = Field(default_factory=list)
    review_context_hash: str = ""
    knowledge_snapshot_hash: str = ""
    attempt: int = Field(default=1, ge=1)
    error_code: Optional[str] = None
    need_revision: Optional[bool] = None
    remediation: ReviewRemediation = ReviewRemediation.NONE
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_new_decision_semantics(self) -> "ReviewResult":
        # Legacy records omitted need_revision, so None remains readable except
        # for system-created execution failures where None is the contract.
        if self.decision == ReviewDecision.EXECUTION_FAILED:
            if self.need_revision is not None:
                raise ValueError("execution_failed requires need_revision=null")
            return self
        expected = self.decision in {
            ReviewDecision.CHANGES_REQUIRED,
            ReviewDecision.BLOCKED,
        }
        if self.need_revision is not None and self.need_revision != expected:
            raise ValueError(
                f"decision={self.decision.value} requires need_revision={str(expected).lower()}"
            )
        return self


class ApprovalRecord(CampaignModel):
    approval_id: str = Field(default_factory=lambda: new_id("approval"))
    workflow_id: str
    version_ids: List[str] = Field(min_length=1)
    version_bindings: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    request_hash: str = ""
    committed_revision: Optional[int] = None
    decision: ApprovalDecision
    reviewer_id: str = Field(min_length=1, max_length=200)
    comment: str = Field(default="", max_length=2000)
    created_at: datetime = Field(default_factory=utc_now)


class WorkflowExecutionEvent(CampaignModel):
    event_id: str = Field(default_factory=lambda: new_id("event"))
    workflow_id: str
    stage: WorkflowStage
    channel: Optional[str] = None
    status: EventStatus = EventStatus.PENDING
    attempt: int = Field(default=1, ge=1)
    request_id: str = ""
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    latency_ms: float = Field(default=0.0, ge=0.0)
    error_code: Optional[str] = None
    retryable: bool = False
    model_calls: int = Field(default=0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    token_usage: int = Field(default=0, ge=0)
    estimated_cost_usd: float = Field(default=0.0, ge=0.0)
    cost_available: bool = False
    usage_accounting_version: int = 1
    usage_complete: bool = True


class CampaignWorkflow(CampaignModel):
    workflow_id: str = Field(default_factory=lambda: new_id("cw"))
    campaign_id: str = Field(default_factory=lambda: new_id("campaign"))
    user_id: str = Field(min_length=1, max_length=200)
    conv_id: Optional[str] = None
    status: WorkflowStatus = WorkflowStatus.DRAFT
    current_stage: WorkflowStage = WorkflowStage.BRIEF_VALIDATION
    brief: CampaignBrief
    plan: List[WorkflowPlanItem] = Field(default_factory=list)
    knowledge_snapshot: KnowledgeSnapshot = Field(default_factory=KnowledgeSnapshot)
    latest_versions: Dict[str, str] = Field(default_factory=dict)
    active_generation_attempt_id: Optional[str] = None
    approval_status: str = "pending"
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    revision: int = Field(default=1, ge=1)
    last_error: Optional[str] = None

    @model_validator(mode="after")
    def channels_match_latest_versions(self) -> "CampaignWorkflow":
        unknown = set(self.latest_versions) - set(self.brief.channels)
        if unknown:
            raise ValueError(f"latest_versions 包含未知渠道: {sorted(unknown)}")
        return self


class CampaignWorkflowCreate(CampaignModel):
    user_id: str = Field(default="anonymous", min_length=1, max_length=200)
    conv_id: Optional[str] = None
    campaign_id: Optional[str] = None
    idempotency_key: Optional[str] = Field(default=None, min_length=8, max_length=200)
    brief: CampaignBrief


class CampaignRevisionRequest(CampaignModel):
    channel: str = Field(min_length=1, max_length=100)
    instruction: str = Field(min_length=1, max_length=3000)
    parent_version_id: str = Field(min_length=1, max_length=100)
    expected_workflow_revision: int = Field(ge=1)

    @field_validator("channel", "instruction", "parent_version_id")
    @classmethod
    def strip_revision_text(cls, value: str) -> str:
        return value.strip()


class CampaignBriefUpdate(CampaignModel):
    brief: CampaignBrief
    expected_workflow_revision: int = Field(ge=1)


class CampaignApprovalRequest(CampaignModel):
    version_ids: List[str] = Field(min_length=1, max_length=10)
    decision: ApprovalDecision
    reviewer_id: str = Field(min_length=1, max_length=200)
    comment: str = Field(default="", max_length=2000)
    expected_workflow_revision: int = Field(ge=1)

    @field_validator("version_ids")
    @classmethod
    def unique_version_ids(cls, values: List[str]) -> List[str]:
        normalized = list(
            dict.fromkeys(str(value).strip() for value in values if str(value).strip())
        )
        if not normalized:
            raise ValueError("至少需要一个有效版本 ID")
        return normalized


ALLOWED_STATUS_TRANSITIONS: Dict[WorkflowStatus, set[WorkflowStatus]] = {
    WorkflowStatus.DRAFT: {
        WorkflowStatus.VALIDATED,
        WorkflowStatus.CANCELLED,
        WorkflowStatus.FAILED,
    },
    WorkflowStatus.VALIDATED: {
        WorkflowStatus.PLANNED,
        WorkflowStatus.CANCELLED,
        WorkflowStatus.FAILED,
    },
    WorkflowStatus.PLANNED: {
        WorkflowStatus.RETRIEVING,
        WorkflowStatus.CANCELLED,
        WorkflowStatus.FAILED,
    },
    WorkflowStatus.RETRIEVING: {
        WorkflowStatus.GENERATING,
        WorkflowStatus.FAILED,
        WorkflowStatus.CANCELLED,
    },
    WorkflowStatus.GENERATING: {
        WorkflowStatus.NEEDS_CLARIFICATION,
        WorkflowStatus.REVIEWING,
        WorkflowStatus.FACT_CHECKING,
        WorkflowStatus.FAILED,
        WorkflowStatus.CANCELLED,
    },
    WorkflowStatus.NEEDS_CLARIFICATION: {
        WorkflowStatus.VALIDATED,
        WorkflowStatus.CANCELLED,
    },
    WorkflowStatus.REVIEWING: {
        WorkflowStatus.REVIEW_INCOMPLETE,
        WorkflowStatus.AWAITING_APPROVAL,
        WorkflowStatus.CHANGES_REQUESTED,
        WorkflowStatus.FAILED,
        WorkflowStatus.CANCELLED,
    },
    WorkflowStatus.REVIEW_INCOMPLETE: {
        WorkflowStatus.REVIEWING,
        WorkflowStatus.FAILED,
        WorkflowStatus.CANCELLED,
    },
    WorkflowStatus.FACT_CHECKING: {
        WorkflowStatus.COMPLIANCE_REVIEWING,
        WorkflowStatus.FAILED,
        WorkflowStatus.CANCELLED,
    },
    WorkflowStatus.COMPLIANCE_REVIEWING: {
        WorkflowStatus.AWAITING_APPROVAL,
        WorkflowStatus.CHANGES_REQUESTED,
        WorkflowStatus.FAILED,
        WorkflowStatus.CANCELLED,
    },
    WorkflowStatus.AWAITING_APPROVAL: {
        WorkflowStatus.APPROVED,
        WorkflowStatus.CHANGES_REQUESTED,
        WorkflowStatus.CANCELLED,
    },
    WorkflowStatus.CHANGES_REQUESTED: {
        WorkflowStatus.GENERATING,
        WorkflowStatus.CANCELLED,
    },
    WorkflowStatus.APPROVED: {
        WorkflowStatus.DELIVERED,
        WorkflowStatus.CHANGES_REQUESTED,
    },
    WorkflowStatus.DELIVERED: set(),
    WorkflowStatus.FAILED: {
        WorkflowStatus.PLANNED,
        WorkflowStatus.RETRIEVING,
        WorkflowStatus.GENERATING,
        WorkflowStatus.CANCELLED,
    },
    WorkflowStatus.CANCELLED: set(),
}


def ensure_status_transition(current: WorkflowStatus, target: WorkflowStatus) -> None:
    if target == current:
        return
    if target not in ALLOWED_STATUS_TRANSITIONS[current]:
        raise ValueError(f"非法工作流状态转换: {current.value} -> {target.value}")


def build_default_plan(channels: List[str]) -> List[WorkflowPlanItem]:
    items = [
        WorkflowPlanItem(stage=WorkflowStage.BRIEF_VALIDATION, order=1),
        WorkflowPlanItem(stage=WorkflowStage.PLANNING, order=2),
        WorkflowPlanItem(stage=WorkflowStage.KNOWLEDGE_RETRIEVAL, order=3),
    ]
    order = 4
    for channel in channels:
        items.append(
            WorkflowPlanItem(
                stage=WorkflowStage.CONTENT_GENERATION, channel=channel, order=order
            )
        )
        order += 1
    for stage in (WorkflowStage.PARALLEL_REVIEW, WorkflowStage.HUMAN_APPROVAL):
        items.append(WorkflowPlanItem(stage=stage, order=order))
        order += 1
    return items
