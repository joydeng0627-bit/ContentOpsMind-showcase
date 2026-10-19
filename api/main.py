"""
ContentOpsMind 内容运营助手 — FastAPI 入口
所有核心组件在 lifespan 中初始化，通过环境变量配置。
"""

import asyncio
import logging
import os
import pathlib
import re
import sys
import time
import uuid
import hashlib
from contextlib import asynccontextmanager
from datetime import date
from typing import Any, Dict, List, Optional, Union

# 将项目根目录加入 sys.path，确保无论从哪里执行都能找到 agents/core/memory 等模块
# 这一行必须在所有项目内部 import 之前执行
_ROOT = str(pathlib.Path(__file__).parent.parent.resolve())
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Response, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from core.llm_gateway import llm_task, managed_lifespan, managed_entrypoint
from pydantic import BaseModel, Field
from campaign_workflow.models import (
    CampaignApprovalRequest,
    CampaignBriefUpdate,
    CampaignRevisionRequest,
    CampaignWorkflowCreate,
)

load_dotenv()

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

BANNER = r"""
    ʕ•ᴥ•ʔ  ʕ•ᴥ•ʔ  ʕ•ᴥ•ʔ
   ╔══════════════════════╗
   ║   ContentOpsMind  v1.0     ║
   ║   内容运营 Agent 系统    ║
   ╚══════════════════════╝
    ʕ•ᴥ•ʔ  ʕ•ᴥ•ʔ  ʕ•ᴥ•ʔ
"""

# ── 全局组件（lifespan 中初始化）─────────────────────────────────────────────
_orchestrator = None
_memory = None
_tool_manager = None
_monitor = None
_evaluator = None
_skill_manager = None
_trace_store = None
_intent_recognizer = None
_campaign_workflow_store = None
_campaign_workflow_service = None


def _anthropic_cfg() -> Dict[str, Any]:
    key = os.getenv("ANTHROPIC_API_KEY", "")
    if not key:
        raise RuntimeError("未设置 ANTHROPIC_API_KEY")
    cfg: Dict[str, Any] = {
        "api_key": key,
        "model": os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"),
    }
    base_url = os.getenv("ANTHROPIC_BASE_URL", "").strip()
    if base_url:
        cfg["base_url"] = base_url
    return cfg


@asynccontextmanager
@managed_lifespan
async def lifespan(app: FastAPI):
    global _orchestrator, _memory, _tool_manager, _monitor, _evaluator, _skill_manager, _trace_store, _intent_recognizer, _campaign_workflow_store, _campaign_workflow_service

    print(BANNER, flush=True)

    from agents.agent_orchestrator import AgentOrchestrator, Request
    from core.intent_recognizer import IntentRecognizer
    from evaluation.evaluator import EndToEndEvaluator
    from mcp.knowledge_base import KnowledgeBase
    from mcp.tool_manager import MCPToolManager, Tool
    from memory.conversation_memory import MemoryManager
    from monitor.performance_monitor import PerformanceMonitor
    from core.skill_loader import SkillManager
    from core.observability import TraceStore

    cfg = _anthropic_cfg()
    logger.info(f"模型: {cfg['model']}  base_url: {cfg.get('base_url', '(官方)')}")

    # 意图识别器（Orchestrator 内部也会创建，这里单独暴露给 Evaluator）
    recognizer = IntentRecognizer(
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
    )
    _intent_recognizer = recognizer

    # Skills：启动时从目录加载业务能力说明，并在 Agent 调用 LLM 时动态注入。
    skills_dir = os.getenv(
        "CONTENTOPSMIND_SKILLS_DIR", str(pathlib.Path(_ROOT) / "skills")
    )
    _skill_manager = SkillManager(
        root_dir=skills_dir,
        max_prompt_chars=int(
            os.getenv("CONTENTOPSMIND_SKILLS_MAX_PROMPT_CHARS", "5000")
        ),
    )
    _skill_manager.load()

    # Agent Trace / Observability
    _trace_store = TraceStore(maxlen=int(os.getenv("TRACE_MAXLEN", "500")))

    # Agent 编排器
    _orchestrator = AgentOrchestrator(
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
        skill_manager=_skill_manager,
        trace_store=_trace_store,
    )

    # 记忆管理器（Redis 工作记忆 + ChromaDB 情景记忆/用户画像）
    _memory = MemoryManager(
        redis_url=os.getenv("REDIS_URL", "redis://redis:6379/0"),
        chroma_host=os.getenv("CHROMA_HOST", "chromadb"),
        chroma_port=int(os.getenv("CHROMA_PORT", "8000")),
        chroma_path=os.getenv("CHROMA_PERSIST_DIRECTORY", "/app/data/chroma"),
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
    )

    # Campaign Workflow 使用独立 Redis keyspace，不复用 24 小时会话 TTL。
    from campaign_workflow.store import CampaignWorkflowStore

    _campaign_workflow_store = CampaignWorkflowStore(
        redis_url=os.getenv("REDIS_URL", "redis://redis:6379/0")
    )

    # MCP 工具管理器 + RAG 知识库（基于 ChromaDB 的真实检索）
    _tool_manager = MCPToolManager(
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
    )
    kb = KnowledgeBase(
        chroma_host=os.getenv("CHROMA_HOST", "chromadb"),
        chroma_port=int(os.getenv("CHROMA_PORT", "8000")),
        chroma_path=os.getenv(
            "CHROMA_PERSIST_DIRECTORY", "/app/data/content_ops_chroma"
        ),
        load_default_docs=False,
    )
    seed_dir = pathlib.Path(_ROOT) / "data" / "rag_docs"
    try:
        import json as _json

        docs_by_title: Dict[str, Dict[str, Any]] = {}
        source_files = sorted(seed_dir.glob("*.json"))
        for source_path in source_files:
            payload = _json.loads(source_path.read_text(encoding="utf-8"))
            if not isinstance(payload, list):
                raise ValueError(f"{source_path.name} 顶层必须是 JSON 数组")
            for raw_doc in payload:
                if not isinstance(raw_doc, dict):
                    raise ValueError(f"{source_path.name} 包含非对象文档")
                title = str(raw_doc.get("title", "")).strip()
                content = str(raw_doc.get("content", "")).strip()
                if not title or not content:
                    raise ValueError(
                        f"{source_path.name} 包含缺少 title/content 的文档"
                    )
                if title in docs_by_title:
                    logger.warning(
                        "知识库标题重复，后加载文档覆盖前文档: title=%s file=%s",
                        title,
                        source_path.name,
                    )
                doc = dict(raw_doc)
                doc["source_file"] = source_path.name
                docs_by_title[title] = doc
        docs = list(docs_by_title.values())
        sync_result = kb.sync_documents(docs)
        logger.info(
            "内容运营知识库同步完成: files=%s docs=%s result=%s",
            len(source_files),
            len(docs),
            sync_result,
        )
    except Exception as ex:
        logger.warning(f"内容运营知识库导入失败: {ex}")
    logger.info(f"知识库已加载: {kb.doc_count} 个文档片段")

    # 本地 Chroma 首次查询时才会下载/加载默认 ONNX embedding 模型。
    # 在服务对外就绪前完成一次预热，避免第一个真实用户请求只召回到
    # 后执行的写作规范 lane，却错过产品事实 lane。
    if os.getenv("KNOWLEDGE_WARMUP", "true").strip().lower() not in {
        "0",
        "false",
        "no",
    }:
        try:
            await asyncio.to_thread(
                kb.search,
                "内容发布审核规则",
                1,
                filters={"corpus_id": "default"},
                min_score=0.0,
            )
            logger.info("知识库 embedding 预热完成")
        except Exception as ex:
            logger.warning("知识库 embedding 预热失败: %s", ex)

    def knowledge_fallback(
        params: Dict[str, Any], context: Optional[Dict[str, Any]], error: str
    ):
        query = params.get("query", "")
        return [
            {
                "title": "知识库降级结果",
                "content": f"知识库暂时不可用，未能完成对“{query}”的语义检索。请稍后重试，或转人工客服确认。",
                "score": 0.0,
                "fallback": True,
                "error": error,
            }
        ]

    _tool_manager.register(
        Tool(
            name="knowledge_search",
            description="搜索知识库（基于 ChromaDB 向量检索）",
            handler=kb.search_handler,
            schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer"},
                    "filters": {"type": "object"},
                    "min_score": {"type": "number"},
                    "recall_mode": {"type": "string"},
                    "branch_k": {"type": "integer"},
                },
                "required": ["query"],
            },
            cache_ttl=300.0,
            supports_rerank=True,
            supports_rrf=True,
            fallback=knowledge_fallback,
        )
    )

    from campaign_workflow.service import CampaignWorkflowService

    _campaign_workflow_service = CampaignWorkflowService(
        store=_campaign_workflow_store,
        orchestrator=_orchestrator,
        tool_manager=_tool_manager,
        model_timeout_seconds=float(os.getenv("CAMPAIGN_MODEL_TIMEOUT_SECONDS", "90")),
        retrieval_timeout_seconds=float(
            os.getenv("CAMPAIGN_RETRIEVAL_TIMEOUT_SECONDS", "20")
        ),
        max_attempts=int(os.getenv("CAMPAIGN_MAX_ATTEMPTS", "2")),
    )

    # 性能监控（可选启动 Prometheus）
    prom_port = int(os.getenv("PROMETHEUS_PORT", "0")) or None
    _monitor = PerformanceMonitor(
        orchestrator=_orchestrator,
        tool_manager=_tool_manager,
        interval_s=float(os.getenv("MONITOR_INTERVAL", "10")),
        webhook_url=os.getenv("ALERT_WEBHOOK_URL") or None,
        prometheus_port=prom_port,
    )
    await _monitor.start()

    # 评测器
    _evaluator = EndToEndEvaluator(
        orchestrator=_orchestrator,
        recognizer=recognizer,
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
        baseline_path=os.getenv("EVAL_BASELINE_PATH", "/app/data/eval/baseline.json"),
    )

    logger.info("ContentOpsMind 已就绪")
    try:
        yield
    finally:
        try:
            await _monitor.stop()
        finally:
            await _tool_manager.close()
    logger.info("ContentOpsMind 已关闭")


# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(
    title="ContentOpsMind 内容运营助手",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


FRONTEND_DIR = pathlib.Path(_ROOT) / "web"
if FRONTEND_DIR.exists():
    app.mount(
        "/app", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend"
    )


# ── 请求/响应模型 ─────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    user_id: str = "anonymous"
    conv_id: Optional[str] = None
    task_id: Optional[str] = None
    task_type: Optional[str] = None
    channel: Optional[str] = None
    tone: Optional[str] = None
    knowledge_filters: Dict[str, str] = Field(default_factory=dict)
    knowledge_min_score: Optional[float] = None
    skip_knowledge_retrieval: bool = False


class ChatResponse(BaseModel):
    request_id: str
    conv_id: str
    response: str
    intent: str
    secondary_intents: List[str] = Field(default_factory=list)
    agent_type: str
    routed_agents: List[str] = Field(default_factory=list)
    escalated: bool
    latency_ms: float
    knowledge_used: bool = False
    evidence_status: str = "no_match"
    success: bool = True
    execution_status: str = "success"
    review_status: str = "not_reviewed"
    review_complete: bool = False
    approval_status: str = "pending"
    publish_allowed: bool = False
    draft_version: Optional[int] = None
    content_hash: Optional[str] = None
    draft_content: Optional[str] = None
    review_records: List[Dict[str, Any]] = Field(default_factory=list)
    error_code: Optional[str] = None
    task_id: str = ""
    context_inherited: bool = False
    citation_ids: List[str] = Field(default_factory=list)
    citations: List[Dict[str, Any]] = Field(default_factory=list)


class CampaignWorkflowListResponse(BaseModel):
    items: List[Dict[str, Any]] = Field(default_factory=list)
    total: int = 0


# ── 路由 ──────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    if _orchestrator is None:
        raise HTTPException(503, "服务未就绪")
    return {"status": "ok", "agents": _orchestrator.get_stats()}


@app.get("/config", tags=["System"])
async def public_config():
    """Return non-secret feature flags used by the static workbench."""
    return {
        "campaign_dataset_version": "campaign-golden-v1",
    }


@app.get("/skills", tags=["Skills"])
async def skills_summary():
    """查看当前已加载的 Skills，便于确认热加载结果和排查解析错误。"""
    if _skill_manager is None:
        raise HTTPException(503, "Skills 未初始化")
    return _skill_manager.summary()


@app.post("/campaign-workflows", status_code=201, tags=["Campaign Workflow"])
async def create_campaign_workflow(payload: CampaignWorkflowCreate):
    """Validate and persist a campaign Brief without invoking an Agent."""
    if _campaign_workflow_store is None:
        raise HTTPException(503, "Campaign Workflow 服务未就绪")
    from campaign_workflow.models import CampaignWorkflow, WorkflowStage, WorkflowStatus
    from campaign_workflow.store import WorkflowConflictError

    workflow = CampaignWorkflow(
        workflow_id=(
            "cw_"
            + hashlib.sha256(
                f"{payload.user_id}:{payload.idempotency_key}".encode("utf-8")
            ).hexdigest()[:16]
            if payload.idempotency_key
            else f"cw_{uuid.uuid4().hex[:16]}"
        ),
        campaign_id=payload.campaign_id or f"campaign_{uuid.uuid4().hex[:16]}",
        user_id=payload.user_id,
        conv_id=payload.conv_id,
        status=WorkflowStatus.VALIDATED,
        current_stage=WorkflowStage.PLANNING,
        brief=payload.brief,
    )
    try:
        return _campaign_workflow_store.create(workflow).model_dump(mode="json")
    except WorkflowConflictError as ex:
        if payload.idempotency_key:
            existing = _campaign_workflow_store.get(workflow.workflow_id)
            if (
                existing.user_id == payload.user_id
                and existing.brief.snapshot_hash() == payload.brief.snapshot_hash()
            ):
                return existing.model_dump(mode="json")
            raise HTTPException(409, "幂等键已用于不同的 Campaign Brief") from ex
        raise HTTPException(409, str(ex)) from ex


@app.get(
    "/campaign-workflows",
    response_model=CampaignWorkflowListResponse,
    tags=["Campaign Workflow"],
)
async def list_campaign_workflows(user_id: Optional[str] = None, limit: int = 50):
    if _campaign_workflow_store is None:
        raise HTTPException(503, "Campaign Workflow 服务未就绪")
    workflows = _campaign_workflow_store.list(user_id=user_id, limit=limit)
    items = [workflow.model_dump(mode="json") for workflow in workflows]
    return CampaignWorkflowListResponse(items=items, total=len(items))


@app.get("/campaign-workflows/{workflow_id}", tags=["Campaign Workflow"])
async def get_campaign_workflow(workflow_id: str):
    if _campaign_workflow_store is None:
        raise HTTPException(503, "Campaign Workflow 服务未就绪")
    from campaign_workflow.store import WorkflowNotFoundError

    try:
        return _campaign_workflow_store.detail(workflow_id)
    except WorkflowNotFoundError as ex:
        raise HTTPException(404, "Campaign Workflow 不存在") from ex


@app.post("/campaign-workflows/{workflow_id}/run", tags=["Campaign Workflow"])
async def run_campaign_workflow(workflow_id: str):
    """Execute the persisted workflow through RAG, generation and review checkpoints."""
    if _campaign_workflow_service is None:
        raise HTTPException(503, "Campaign Workflow 服务未就绪")
    from campaign_workflow.service import WorkflowBusyError, WorkflowExecutionError
    from campaign_workflow.store import (
        WorkflowConflictError,
        WorkflowLockLostError,
        WorkflowNotFoundError,
    )

    try:
        return await _campaign_workflow_service.run(workflow_id)
    except WorkflowNotFoundError as ex:
        raise HTTPException(404, "Campaign Workflow 不存在") from ex
    except WorkflowBusyError as ex:
        raise HTTPException(
            409, {"code": "workflow_busy", "message": "Workflow 正在执行"}
        ) from ex
    except WorkflowLockLostError as ex:
        raise HTTPException(
            409,
            {
                "code": "workflow_lock_lost",
                "message": "执行租约已失效，当前任务已在持久化前终止",
            },
        ) from ex
    except WorkflowConflictError as ex:
        raise HTTPException(
            409, {"code": "workflow_conflict", "message": str(ex)}
        ) from ex
    except WorkflowExecutionError as ex:
        raise HTTPException(
            422, {"code": "workflow_execution_failed", "message": str(ex)}
        ) from ex


@app.post("/campaign-workflows/{workflow_id}/revisions", tags=["Campaign Workflow"])
async def revise_campaign_workflow(workflow_id: str, payload: CampaignRevisionRequest):
    if _campaign_workflow_service is None:
        raise HTTPException(503, "Campaign Workflow 服务未就绪")
    from campaign_workflow.service import WorkflowBusyError, WorkflowExecutionError
    from campaign_workflow.store import (
        WorkflowConflictError,
        WorkflowLockLostError,
        WorkflowNotFoundError,
    )

    try:
        return await _campaign_workflow_service.revise(workflow_id, payload)
    except WorkflowNotFoundError as ex:
        raise HTTPException(404, "Campaign Workflow 不存在") from ex
    except WorkflowBusyError as ex:
        raise HTTPException(
            409, {"code": "workflow_busy", "message": "Workflow 正在执行"}
        ) from ex
    except WorkflowLockLostError as ex:
        raise HTTPException(
            409,
            {
                "code": "workflow_lock_lost",
                "message": "执行租约已失效，当前任务已在持久化前终止",
            },
        ) from ex
    except WorkflowConflictError as ex:
        raise HTTPException(
            409, {"code": "workflow_conflict", "message": str(ex)}
        ) from ex
    except WorkflowExecutionError as ex:
        raise HTTPException(
            422, {"code": "workflow_execution_failed", "message": str(ex)}
        ) from ex


@app.patch("/campaign-workflows/{workflow_id}/brief", tags=["Campaign Workflow"])
async def update_campaign_workflow_brief(
    workflow_id: str, payload: CampaignBriefUpdate
):
    if _campaign_workflow_service is None:
        raise HTTPException(503, "Campaign Workflow 服务未就绪")
    from campaign_workflow.service import WorkflowBusyError
    from campaign_workflow.store import (
        WorkflowConflictError,
        WorkflowLockLostError,
        WorkflowNotFoundError,
    )

    try:
        return _campaign_workflow_service.update_brief(workflow_id, payload)
    except WorkflowNotFoundError as ex:
        raise HTTPException(404, "Campaign Workflow 不存在") from ex
    except WorkflowBusyError as ex:
        raise HTTPException(
            409, {"code": "workflow_busy", "message": "Workflow 正在执行"}
        ) from ex
    except WorkflowLockLostError as ex:
        raise HTTPException(
            409,
            {"code": "workflow_lock_lost", "message": "执行租约已失效，拒绝修改 Brief"},
        ) from ex
    except WorkflowConflictError as ex:
        raise HTTPException(
            409, {"code": "workflow_conflict", "message": str(ex)}
        ) from ex


@app.post("/campaign-workflows/{workflow_id}/approvals", tags=["Campaign Workflow"])
async def approve_campaign_workflow(workflow_id: str, payload: CampaignApprovalRequest):
    if _campaign_workflow_service is None:
        raise HTTPException(503, "Campaign Workflow 服务未就绪")
    from campaign_workflow.service import WorkflowBusyError
    from campaign_workflow.store import (
        WorkflowConflictError,
        WorkflowLockLostError,
        WorkflowNotFoundError,
    )

    try:
        return _campaign_workflow_service.approve(workflow_id, payload)
    except WorkflowNotFoundError as ex:
        raise HTTPException(404, "Campaign Workflow 不存在") from ex
    except WorkflowBusyError as ex:
        raise HTTPException(
            409, {"code": "workflow_busy", "message": "Workflow 正在执行"}
        ) from ex
    except WorkflowLockLostError as ex:
        raise HTTPException(
            409,
            {
                "code": "workflow_lock_lost",
                "message": "执行租约已失效，当前任务已在持久化前终止",
            },
        ) from ex
    except WorkflowConflictError as ex:
        raise HTTPException(
            409, {"code": "workflow_conflict", "message": str(ex)}
        ) from ex


@app.get("/campaign-workflows/{workflow_id}/package", tags=["Campaign Workflow"])
async def get_campaign_content_package(workflow_id: str, format: str = "json"):
    if _campaign_workflow_service is None:
        raise HTTPException(503, "Campaign Workflow 服务未就绪")
    from campaign_workflow.package_builder import (
        PackageNotReadyError,
        render_package_markdown,
    )
    from campaign_workflow.store import (
        WorkflowNotFoundError,
        WorkflowConflictError,
        WorkflowLockLostError,
    )
    from campaign_workflow.service import WorkflowBusyError

    try:
        package = _campaign_workflow_service.package(workflow_id)
        if format.lower() == "markdown":
            return Response(
                content=render_package_markdown(package),
                media_type="text/markdown; charset=utf-8",
                headers={
                    "Content-Disposition": f'attachment; filename="{workflow_id}.md"'
                },
            )
        if format.lower() != "json":
            raise HTTPException(422, "format 仅支持 json 或 markdown")
        return package
    except WorkflowNotFoundError as ex:
        raise HTTPException(404, "Campaign Workflow 不存在") from ex
    except WorkflowBusyError as ex:
        raise HTTPException(
            409, {"code": "workflow_busy", "message": "Workflow 正在执行"}
        ) from ex
    except WorkflowLockLostError as ex:
        raise HTTPException(
            409, {"code": "workflow_lock_lost", "message": str(ex)}
        ) from ex
    except WorkflowConflictError as ex:
        raise HTTPException(
            409, {"code": "workflow_conflict", "message": str(ex)}
        ) from ex
    except PackageNotReadyError as ex:
        raise HTTPException(409, str(ex)) from ex


@app.post("/skills/reload", tags=["Skills"])
async def reload_skills():
    """运行时重新扫描 Skill 目录，不需要重启服务。"""
    if _skill_manager is None:
        raise HTTPException(503, "Skills 未初始化")
    _skill_manager.reload()
    if _orchestrator is not None:
        _orchestrator.set_skill_manager(_skill_manager)
    return _skill_manager.summary()


@app.post("/chat", response_model=ChatResponse)
@llm_task()
async def chat(req: ChatRequest):
    """
    主对话接口。完整流程：
      记忆读取 → 意图识别 → Agent 路由 → 执行 → 记忆写入
    """
    if _orchestrator is None or _memory is None:
        raise HTTPException(503, "服务未就绪")

    from agents.agent_orchestrator import Request as OrcReq
    from agents.agent_orchestrator import AgentType
    from core.context_scope import (
        build_missing_rewrite_response,
        decide_context_scope,
        has_source_text,
        infer_channel,
        infer_tone,
        is_rewrite_request,
        looks_like_source_text,
        memory_context_policy,
        preference_context,
        refers_to_active_artifact,
        unsupported_task_facts,
    )
    from core.intent_recognizer import IntentCategory
    from memory.conversation_memory import MsgRole

    conv_id = req.conv_id or str(uuid.uuid4())

    # 1. 读取记忆，但先做任务作用域判定；默认不继承旧任务。
    mem_ctx = await _memory.get_context(req.user_id, conv_id, query=req.message)
    recent_records = [
        {"role": m.role.value, "content": m.content, **(m.metadata or {})}
        for m in mem_ctx.recent_messages
    ]
    pending_task = await _memory.get_pending_task(req.user_id, conv_id)
    active_artifact = await _memory.get_active_artifact(req.user_id, conv_id)
    pending_resolved = bool(
        pending_task
        and pending_task.get("status") == "awaiting_input"
        and pending_task.get("missing_fields") == ["source_text"]
        and looks_like_source_text(req.message)
    )
    artifact_resolved = bool(
        not pending_task and active_artifact and refers_to_active_artifact(req.message)
    )
    scope = decide_context_scope(req.message, recent_records)
    previous_task_id = next(
        (
            str(item.get("task_id"))
            for item in reversed(recent_records)
            if item.get("task_id")
        ),
        "",
    )
    if pending_resolved:
        task_id = str(pending_task.get("task_id") or uuid.uuid4())
    elif artifact_resolved:
        task_id = str(active_artifact.get("task_id") or uuid.uuid4())
    else:
        if pending_task:
            await _memory.clear_pending_task(req.user_id, conv_id)
        task_id = (
            previous_task_id
            if scope.inherit_previous_task and previous_task_id
            else (
                req.task_id
                if req.task_id and req.task_id != previous_task_id
                else str(uuid.uuid4())
            )
        )
    memory_metadata = {
        "memory_type": "task_fact",
        "scope": "task",
        "conv_id": conv_id,
        "task_id": task_id,
        "ttl": 86400,
    }

    # 改写对象缺失时直接澄清，不进入 RAG、Agent 路由或旧任务上下文。
    if scope.needs_source_text and not pending_resolved:
        request_id = str(uuid.uuid4())[:8]
        # Slot precedence: explicit current message > current form fields > defaults.
        pending_channel = infer_channel(req.message) or req.channel or ""
        pending_tone = infer_tone(req.message) or req.tone or ""
        clarification = build_missing_rewrite_response(
            channel=pending_channel,
            tone=pending_tone,
            topic_conflict=scope.reason == "explicit_reference_but_topic_conflict",
        )
        await _memory.add_message(
            req.user_id, conv_id, MsgRole.USER, req.message, memory_metadata
        )
        await _memory.add_message(
            req.user_id,
            conv_id,
            MsgRole.ASSISTANT,
            clarification,
            memory_metadata,
        )
        await _memory.set_pending_task(
            req.user_id,
            conv_id,
            {
                "status": "awaiting_input",
                "intent": IntentCategory.COPY_REWRITE.value,
                "missing_fields": ["source_text"],
                "target_channel": pending_channel,
                "target_style": pending_tone,
                "tone": pending_tone,
                "channel": pending_channel,
                "task_id": task_id,
                "created_from_turn": request_id,
            },
        )
        return ChatResponse(
            request_id=request_id,
            conv_id=conv_id,
            response=clarification,
            intent=IntentCategory.COPY_REWRITE.value,
            agent_type=AgentType.GENERAL.value,
            routed_agents=[],
            escalated=False,
            latency_ms=0.0,
            knowledge_used=False,
            task_id=task_id,
            context_inherited=False,
        )

    # 2. 只有“明确指代 + 主题兼容”时，历史任务才进入意图识别和 Agent。
    inherited_records: List[Dict[str, Any]] = []
    if scope.inherit_previous_task and not pending_resolved and not artifact_resolved:
        if previous_task_id:
            inherited_records = [
                item
                for item in recent_records
                if item.get("task_id") == previous_task_id
            ][-4:]
        else:
            # Legacy messages have no task_id; inherit only the immediately
            # preceding user/assistant pair, never a broad history window.
            inherited_records = recent_records[-2:]
    history = [
        {"role": item["role"], "content": item["content"]} for item in inherited_records
    ] or None

    effective_message = req.message
    forced_intent = None
    forced_secondary: List[IntentCategory] = []
    if pending_resolved:
        pending_channel = str(pending_task.get("channel") or "")
        pending_tone = str(pending_task.get("tone") or "")
        effective_message = (
            "请改写以下用户提供的原文。只能使用原文中的产品事实，不得新增体验、场景、卖点或促销信息。\n"
            f"目标渠道/风格：{pending_channel or '按用户上一轮要求'}\n"
            f"目标语气：{pending_tone or '按用户上一轮要求'}\n"
            f"[SOURCE_TEXT]\n{req.message}"
        )
        forced_intent = IntentCategory.COPY_REWRITE
        forced_secondary = [IntentCategory.STYLE_ADAPTATION]
    elif artifact_resolved:
        artifact_source = str(active_artifact.get("source_text") or "")
        artifact_draft = str(active_artifact.get("current_draft") or "")
        effective_message = (
            "请根据用户反馈继续修改当前草稿，只调整表达风格和组织方式。"
            "不得新增事实来源中不存在的产品、优惠、期限、体验或活动信息。\n"
            f"[用户反馈]\n{req.message}\n\n"
            f"[当前草稿]\n{artifact_draft}\n\n"
            f"[事实来源]\n{artifact_source}"
        )
        forced_intent = IntentCategory.STYLE_ADAPTATION
        forced_secondary = [IntentCategory.COPY_REWRITE]

    if forced_intent is not None:
        selected_intent = forced_intent
        selected_secondary = forced_secondary
        selected_urgency = None
    else:
        if _intent_recognizer is None:
            raise HTTPException(503, "意图识别服务未就绪")
        recognized = await _intent_recognizer.recognize(
            effective_message, history=history
        )
        selected_intent = recognized.intent
        selected_secondary = recognized.secondary_intents
        selected_urgency = recognized.urgency

    rewrite_has_source = (
        pending_resolved
        or artifact_resolved
        or (is_rewrite_request(req.message) and has_source_text(req.message))
    )
    rewrite_without_retrieval = (
        rewrite_has_source
        or selected_intent == IntentCategory.COPY_REWRITE
        or is_rewrite_request(req.message)
    )
    if (
        req.skip_knowledge_retrieval
        or rewrite_without_retrieval
        or selected_intent == IntentCategory.GREETING
    ):
        knowledge_text, knowledge_used = "", False
        retrieval_trace = {
            "used": False,
            "queries": [],
            "docs": [],
            "latency_ms": 0.0,
            "reranked": False,
            "skipped_reason": (
                "explicit_context_is_authoritative"
                if req.skip_knowledge_retrieval
                else (
                    "rewrite_does_not_require_knowledge"
                    if rewrite_without_retrieval
                    else "greeting_does_not_require_knowledge"
                )
            ),
        }
    else:
        knowledge_text, knowledge_used, retrieval_trace = (
            await _build_knowledge_context(
                req.message,
                filters=req.knowledge_filters,
                min_score=req.knowledge_min_score,
            )
        )

    channel = (
        str(pending_task.get("channel") or "")
        if pending_resolved
        else (
            str(active_artifact.get("channel") or "")
            if artifact_resolved
            else (infer_channel(req.message) or req.channel or "")
        )
    )
    tone = (
        str(pending_task.get("tone") or "")
        if pending_resolved
        else (
            str(active_artifact.get("tone") or "")
            if artifact_resolved
            else (infer_tone(req.message) or req.tone or "")
        )
    )
    context_policy = memory_context_policy(selected_intent)
    context_parts: List[str] = []
    preference_text = (
        preference_context(mem_ctx.user_profile, channel)
        if context_policy["allow_long_term_preference"]
        else ""
    )
    if preference_text:
        context_parts.append(preference_text)

    current_fields = {
        "task_type": (
            "copy_rewrite"
            if rewrite_has_source or is_rewrite_request(req.message)
            else (req.task_type or "")
        ),
        "channel": channel if context_policy["allow_current_task_fields"] else "",
        "tone": tone if context_policy["allow_current_task_fields"] else "",
        "source_text": (
            req.message
            if pending_resolved
            else (
                str(active_artifact.get("source_text") or "")
                if artifact_resolved
                else ""
            )
        ),
    }
    field_text = (
        "\n".join(f"{key}: {value}" for key, value in current_fields.items() if value)
        if context_policy["allow_current_task_fields"]
        else ""
    )
    if field_text:
        context_parts.append(f"[CURRENT_TASK_FIELDS task_id={task_id}]\n{field_text}")
    scoped_task_records = (
        inherited_records if context_policy["allow_task_history"] else []
    )
    if scoped_task_records:
        task_context = "\n".join(
            f"{item['role']}: {item['content']}" for item in scoped_task_records
        )
        context_parts.append(f"[TASK_CONTEXT task_id={task_id}]\n{task_context}")
    if knowledge_text:
        context_parts.append(knowledge_text)
    elif retrieval_trace.get("attempted") or retrieval_trace.get("queries"):
        context_parts.append(
            "[KNOWLEDGE_GAP]\n"
            "已执行知识检索，但没有找到足以支撑该具体产品或活动事实的可靠材料。"
            "不得用常识猜测配方、成分、价格、活动门槛或有效期；请明确说明无法确认，并请用户补充官方资料。"
        )
    context_parts.append(
        "[PRODUCT_EVIDENCE_POLICY]\n营养事实必须保留杯型、冷热和标准配方限定，包括标题、摘要和正文。"
        "标题如无法容纳完整限定则不要写营养数字。禁止由奶的清甜推断不加糖、无添加糖、低糖或糖含量；"
        "禁止由营养成分推断运动后恢复、增肌、减重等效果。未检索到某条规则不等于官网从未提到，"
        "只能说明当前资料不足以确认现行状态，不得断言公告没有提及。"
        "公告的产品说明不证明当前在售、价格或活动有效；季节限定产品只能表述为公告曾推出。"
        "没有价格或促销依据时省略相关内容。缺少目标产品事实时只询问官方介绍链接或产品资料，"
        "不要要求补齐非必要的价格、活动、人群和品牌话术，不解释内部知识库内容。"
    )
    context_parts.append(
        "[CONTEXT_BOUNDARY]\n"
        "只能使用当前用户输入、当前 task_id 的显式上下文、当前任务允许使用的明确偏好和知识库规范。"
        "其他历史任务中的价格、优惠、截止时间、渠道和商品范围均不可继承。"
        "不得向用户复述或解释内部记忆、用户画像、任务字段、上下文边界、路由、Agent、"
        "检索或偏好注入过程，也不得根据与问题无关的设置追加营销任务建议。"
    )
    full_context = "\n\n".join(part for part in context_parts if part)
    memory_policy = {
        "intent": selected_intent.value,
        "long_term_preference_used": bool(preference_text),
        "current_task_fields_used": bool(field_text),
        "task_history_used": bool(scoped_task_records),
        "episodic_memory_used": False,
        "business_knowledge_used": bool(knowledge_used),
        "excluded_sources": [
            name
            for name, excluded in (
                (
                    "long_term_preference",
                    not context_policy["allow_long_term_preference"],
                ),
                (
                    "current_task_fields",
                    not context_policy["allow_current_task_fields"],
                ),
                ("task_history", not context_policy["allow_task_history"]),
                ("episodic_memory", True),
            )
            if excluded
        ],
    }
    retrieval_trace["scope"] = {
        "task_id": task_id,
        "context_inherited": scope.inherit_previous_task
        or pending_resolved
        or artifact_resolved,
        "inherited_from_pending_request": pending_resolved,
        "resolved_target": (
            "active_artifact"
            if artifact_resolved
            else ("pending_source_text" if pending_resolved else "none")
        ),
        "artifact_id": (
            str(active_artifact.get("artifact_id") or "") if artifact_resolved else ""
        ),
        "reason": (
            "active_artifact_revision"
            if artifact_resolved
            else ("pending_source_text_resolved" if pending_resolved else scope.reason)
        ),
        "current_topics": sorted(scope.current_topics),
        "previous_topics": sorted(scope.previous_topics),
        "channel": channel if context_policy["allow_current_task_fields"] else "",
        "tone": tone if context_policy["allow_current_task_fields"] else "",
        "source_text_from_current_turn": pending_resolved,
        "active_artifact_used": artifact_resolved,
    }

    orch_req = OrcReq(
        message=effective_message,
        user_id=req.user_id,
        conv_id=conv_id,
        task_id=task_id,
        context=full_context,
        history=history if context_policy["allow_task_history"] else None,
        intent=selected_intent,
        secondary_intents=selected_secondary,
        urgency=selected_urgency,
        trace_context={
            "retrieval": retrieval_trace,
            "memory_policy": memory_policy,
        },
    )

    # 3. 执行
    result = await _orchestrator.run(orch_req)
    if (
        selected_intent == IntentCategory.COPY_GENERATION
        and retrieval_trace.get("evidence_status") == "product_supported"
    ):
        from core.product_grounding import qualify_pro_publication

        result.response = qualify_pro_publication(
            result.response, retrieval_trace.get("docs", [])
        )

    if pending_resolved:
        await _memory.clear_pending_task(req.user_id, conv_id)

    blocked_facts: List[str] = []
    if artifact_resolved:
        blocked_facts = unsupported_task_facts(
            result.response,
            [
                str(active_artifact.get("source_text") or ""),
                str(active_artifact.get("current_draft") or ""),
                req.message,
            ],
        )
        if blocked_facts:
            result.response = (
                "本次修改检测到不属于当前咖啡杯编辑链的历史任务事实，已阻止输出："
                + "、".join(blocked_facts)
                + "。当前版本保持不变，请重新提交风格调整要求。"
            )

    artifact_intents = {
        IntentCategory.COPY_GENERATION,
        IntentCategory.COPY_REWRITE,
        IntentCategory.STYLE_ADAPTATION,
    }
    if result.intent in artifact_intents and not blocked_facts:
        source_text = (
            str(active_artifact.get("source_text") or "")
            if artifact_resolved
            else req.message
        )
        artifact_id = (
            str(active_artifact.get("artifact_id") or "")
            if artifact_resolved
            else f"copy_{time.strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:4]}"
        )
        version = int(active_artifact.get("version", 0)) + 1 if artifact_resolved else 1
        await _memory.set_active_artifact(
            req.user_id,
            conv_id,
            {
                "artifact_id": artifact_id,
                "task_id": task_id,
                "artifact_type": "social_copy" if channel == "小红书" else "copy",
                "source_text": source_text,
                "current_draft": result.response,
                "channel": channel,
                "tone": tone,
                "version": version,
                "source_turn_id": result.request_id,
            },
        )

    # 4. 写入记忆
    await _memory.add_message(
        req.user_id, conv_id, MsgRole.USER, req.message, memory_metadata
    )
    await _memory.add_message(
        req.user_id, conv_id, MsgRole.ASSISTANT, result.response, memory_metadata
    )

    # 5. 异步更新用户画像（不阻塞响应）
    asyncio.create_task(
        _memory.update_profile(req.user_id, req.message, result.request_id)
    )

    citations = list(retrieval_trace.get("docs", []))
    citations.extend(_inline_document_citations(req.message))
    citations = list({item["citation_id"]: item for item in citations}.values())
    return ChatResponse(
        request_id=result.request_id,
        conv_id=conv_id,
        response=result.response,
        intent=result.intent.value if result.intent else "other",
        secondary_intents=[intent.value for intent in result.secondary_intents],
        agent_type=result.agent_type.value,
        routed_agents=[agent.value for agent in result.routed_agents],
        escalated=result.escalated,
        latency_ms=round(result.latency_ms, 1),
        knowledge_used=knowledge_used,
        evidence_status=retrieval_trace.get("evidence_status", "no_match"),
        success=result.success,
        execution_status=result.execution_status,
        review_status=result.review_status,
        review_complete=result.review_complete,
        approval_status=result.approval_status,
        publish_allowed=result.publish_allowed,
        draft_version=result.draft_version,
        content_hash=result.content_hash,
        draft_content=result.draft_content,
        review_records=result.review_records,
        error_code=result.error_code,
        task_id=task_id,
        context_inherited=scope.inherit_previous_task
        or pending_resolved
        or artifact_resolved,
        citation_ids=[
            str(item["citation_id"]) for item in citations if item.get("citation_id")
        ],
        citations=citations,
    )


def _stable_citation_id(item: Dict[str, Any]) -> str:
    """Return a stable, user-visible source id independent of retrieval rank."""
    explicit = str(item.get("document_id", "")).strip()
    if explicit:
        return explicit
    title = str(item.get("title", ""))
    benchmark = re.search(r"\[BENCHMARK:([^\]]+)\]", title)
    if benchmark:
        return benchmark.group(1)
    identity = "|".join(
        (
            str(item.get("source_url", "")).strip(),
            title.strip(),
            str(item.get("version", "")).strip(),
        )
    )
    return "doc_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]


def _inline_document_citations(message: str) -> List[Dict[str, Any]]:
    """Extract explicitly supplied document ids from the current request."""
    return [
        {
            "citation_id": document_id.strip(),
            "title": f"Inline document {document_id.strip()}",
            "source": "request_inline_document",
        }
        for document_id in dict.fromkeys(
            re.findall(r"\[DOC\s+id=([^\]\s]+)\]", str(message or ""), flags=re.I)
        )
        if document_id.strip()
    ]


async def _build_knowledge_context(
    message: str,
    top_k: int = 3,
    filters: Optional[Dict[str, str]] = None,
    min_score: Optional[float] = None,
) -> tuple[str, bool, Dict[str, Any]]:
    """
    为 /chat 主链路构建 RAG 知识上下文。返回文本、是否命中、以及 trace 元数据。
    """
    trace: Dict[str, Any] = {
        "used": False,
        "queries": [],
        "docs": [],
        "latency_ms": 0.0,
        "reranked": False,
    }
    if _tool_manager is None:
        return "", False, trace
    if not _should_use_knowledge(message):
        return "", False, trace
    try:
        rag_t0 = asyncio.get_running_loop().time()
        retrieval_context = {
            "retrieval_filters": dict(filters or {}),
            "min_score": (
                float(min_score)
                if min_score is not None
                else float(os.getenv("KNOWLEDGE_MIN_RELEVANCE_SCORE", "0.12"))
            ),
        }
        result = await _tool_manager.search_with_rewrite(
            "knowledge_search", message, top_k=top_k, context=retrieval_context
        )
        trace["latency_ms"] = round(
            (asyncio.get_running_loop().time() - rag_t0) * 1000, 1
        )
        trace["queries"] = list((result.metadata or {}).get("queries", [message]))
        trace["reranked"] = bool(result.reranked)
        trace["metadata"] = result.metadata or {}
        trace["evidence_status"] = (result.metadata or {}).get(
            "evidence_status", "no_match"
        )
        if not result.success or not isinstance(result.data, list) or not result.data:
            trace["error"] = result.error
            return "", False, trace

        minimum_score = retrieval_context["min_score"]
        relevant_items = [
            item
            for item in result.data
            if isinstance(item, dict)
            and float(item.get("score", 0.0) or 0.0) >= minimum_score
            and str(item.get("status", "current") or "current") != "expired"
        ]
        if not relevant_items:
            trace["skipped_reason"] = "low_relevance"
            trace["attempted"] = True
            return "", False, trace

        parts = ["[知识库检索结果]"]
        docs_for_trace = []
        used = False
        for i, item in enumerate(relevant_items[:5], start=1):
            if not isinstance(item, dict):
                continue
            title = str(item.get("title", "未命名文档"))
            content = str(item.get("content", "")).strip()
            score = item.get("score", "")
            source_url = str(item.get("source_url", "")).strip()
            published_at = str(item.get("published_at", "")).strip()
            document_type = str(item.get("document_type", "")).strip()
            if not content:
                continue
            used = True
            trace_doc = {
                "citation_id": _stable_citation_id(item),
                "title": title,
                "score": score,
                "version": str(item.get("version", "")),
            }
            if source_url:
                trace_doc["source_url"] = source_url
            if published_at:
                trace_doc["published_at"] = published_at
            if document_type:
                trace_doc["document_type"] = document_type
            docs_for_trace.append(trace_doc)
            source_lines = []
            if published_at:
                source_lines.append(f"   发布日期: {published_at}")
            if document_type:
                source_lines.append(f"   内容类型: {document_type}")
            if source_url:
                source_lines.append(f"   官方来源: {source_url}")
            source_block = "\n".join(source_lines)
            if source_block:
                source_block = f"\n{source_block}"
            parts.append(
                f"{i}. 标题: {title}\n   相关度: {score}"
                f"{source_block}\n   内容: {content[:1200]}"
            )

        if not used:
            return "", False, trace
        trace["used"] = True
        trace["docs"] = docs_for_trace
        parts.append(
            "请优先依据以上知识库内容回答；官网新闻和活动事实必须结合发布日期判断时效，"
            "不得把历史活动描述为当前仍有效。如果知识库内容不足，再结合通用客服能力说明。"
        )
        return "\n".join(parts), True, trace
    except Exception as ex:
        logger.warning(f"构建知识库上下文失败: {ex}")
        trace["error"] = str(ex)
        return "", False, trace


def _should_use_knowledge(message: str) -> bool:
    """Only retrieve knowledge for content-operation tasks."""
    msg = (message or "").strip().lower()
    if not msg:
        return False
    greetings = {"你好", "您好", "在吗", "hi", "hello", "hey"}
    if msg in greetings:
        return False
    business_keywords = [
        "文案",
        "活动",
        "优惠券",
        "满减",
        "会员日",
        "双十一",
        "新品",
        "合规",
        "违规",
        "广告法",
        "极限词",
        "禁用词",
        "品牌",
        "调性",
        "风格",
        "小红书",
        "公众号",
        "短信",
        "push",
        "渠道",
        "标题",
        "发布",
        "审核",
        "规则",
        "门槛",
        "到手价",
        "copy",
        "campaign",
        "compliance",
        "brand",
        "sms",
        "push",
    ]
    return any(keyword in msg for keyword in business_keywords) or len(msg) >= 8


@app.get("/trace/recent")
async def trace_recent(limit: int = 20):
    """查看最近请求的 Agent Trace。"""
    if _trace_store is None:
        raise HTTPException(503, "TraceStore 未初始化")
    return {"items": _trace_store.recent(limit)}


@app.get("/trace/{request_id}")
async def trace_detail(request_id: str):
    """按 request_id 查看单次请求的完整 Trace。"""
    if _trace_store is None:
        raise HTTPException(503, "TraceStore 未初始化")
    item = _trace_store.get(request_id)
    if item is None:
        raise HTTPException(404, "Trace 不存在或已过期")
    return item


@app.get("/dashboard")
async def dashboard():
    """Agent/RAG/Latency 观测面板摘要。"""
    if _trace_store is None:
        raise HTTPException(503, "TraceStore 未初始化")
    return _trace_store.dashboard()


@app.get("/monitor")
async def monitor_summary():
    """实时监控摘要：Agent 成功率、工具统计、告警、优化建议。"""
    if _monitor is None:
        raise HTTPException(503, "服务未就绪")
    return _monitor.summary()


@app.get("/metrics")
async def prometheus_metrics():
    """Prometheus 指标入口。"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/search")
@llm_task()
async def search(query: str, top_k: int = 5):
    """
    演示检索优化链路：查询改写 → 并行召回 → 重排 → Top-K。
    """
    if _tool_manager is None:
        raise HTTPException(503, "服务未就绪")
    result = await _tool_manager.search_with_rewrite(
        "knowledge_search", query, top_k=top_k
    )
    return {
        "query": query,
        "results": result.data,
        "reranked": result.reranked,
        "retrieval": {
            key: result.metadata.get(key)
            for key in (
                "fusion_strategy",
                "fusion_applied",
                "fusion_k",
                "branch_k",
                "rerank_candidate_k",
                "rankings",
            )
        },
    }


class DocInput(BaseModel):
    """单篇文档输入。"""

    title: str
    content: str
    document_id: str = ""
    corpus_id: str = "default"
    brand: str = ""
    campaign: str = ""
    status: str = "current"
    version: str = "1.0"
    published_at: str = ""
    effective_at: str = ""
    document_type: str = "knowledge"
    category: str = "general"
    tags: List[str] = Field(default_factory=list)


class BatchDocInput(BaseModel):
    """批量文档导入请求体。"""

    documents: List[DocInput]


class KnowledgeSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=3, ge=1, le=20)
    filters: Dict[str, str] = Field(default_factory=dict)
    min_score: float = Field(default=0.0, ge=0.0, le=1.0)


class EvalIntentInput(BaseModel):
    """意图识别评测用例。"""

    message: str
    expected_intent: Union[str, List[str]]
    context: Optional[Dict[str, Any]] = None


class EvalDialogInput(BaseModel):
    """对话质量评测用例。question 单轮，turns 多轮。"""

    question: Optional[str] = None
    turns: Optional[List[str]] = None
    user_id: Optional[str] = None
    conv_id: Optional[str] = None


class EvalRunInput(BaseModel):
    """评测请求。为空时使用内置默认用例。"""

    intent_cases: Optional[List[EvalIntentInput]] = None
    dialog_cases: Optional[List[EvalDialogInput]] = None


class KnowledgeEvalInput(BaseModel):
    """确定性知识问答回归请求。"""

    case_ids: Optional[List[str]] = None
    fail_fast: bool = False


@app.post("/knowledge/add", tags=["知识库"])
async def add_knowledge(body: BatchDocInput):
    """
    批量导入文档到知识库。

    文档会自动切片（每片 500 字）并存入 ChromaDB，ChromaDB 内置 Embedding 模型自动向量化。

    示例请求体：
    ```json
    {
      "documents": [
        {"title": "退款政策", "content": "用户在购买后 7 天内可以申请无理由退款..."},
        {"title": "配送说明", "content": "标准配送 3-5 个工作日..."}
      ]
    }
    ```
    """
    tool = _tool_manager._tools.get("knowledge_search") if _tool_manager else None
    if tool is None:
        raise HTTPException(503, "知识库未初始化")
    kb = tool.handler.__self__
    count = kb.add_documents([d.model_dump() for d in body.documents])
    return {
        "message": f"成功导入 {count} 个文档片段",
        "added_chunks": count,
        "total_chunks": kb.doc_count,
    }


@app.post("/knowledge/search", tags=["知识库"])
@llm_task()
async def search_knowledge(body: KnowledgeSearchRequest):
    """Expose the production retriever for reproducible RAG baseline evaluation."""
    tool = _tool_manager._tools.get("knowledge_search") if _tool_manager else None
    if tool is None:
        raise HTTPException(503, "知识库未初始化")
    kb = tool.handler.__self__
    return {
        "query": body.query,
        "filters": body.filters,
        "results": kb.search(
            body.query,
            top_k=body.top_k,
            filters=(
                body.filters
                if body.filters.get("corpus_id")
                else dict(body.filters, _scope="real", _target_query=body.query)
            ),
            min_score=body.min_score,
        ),
    }


@app.post("/knowledge/upload", tags=["知识库"])
async def upload_knowledge(file: UploadFile = File(...)):
    """
    上传文件导入知识库。

    支持格式：
    - `.txt` / `.md`：整个文件作为一篇文档，文件名作为标题
    - `.json`：JSON 数组格式 `[{"title": "...", "content": "..."}, ...]`

    文件大小限制：10MB
    """
    tool = _tool_manager._tools.get("knowledge_search") if _tool_manager else None
    if tool is None:
        raise HTTPException(503, "知识库未初始化")
    kb = tool.handler.__self__

    content = await file.read()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(413, "文件大小超过 10MB 限制")

    text = content.decode("utf-8", errors="ignore")
    filename = file.filename or "unknown"

    if filename.endswith(".json"):
        import json as _json

        try:
            docs = _json.loads(text)
            if not isinstance(docs, list):
                raise HTTPException(
                    400, "JSON 文件应为数组格式: [{title, content}, ...]"
                )
        except _json.JSONDecodeError as e:
            raise HTTPException(400, f"JSON 解析失败: {e}")
    else:
        # txt / md：整个文件作为一篇文档
        title = filename.rsplit(".", 1)[0] if "." in filename else filename
        docs = [{"title": title, "content": text}]

    count = kb.add_documents(docs)
    return {
        "message": f"文件 {filename} 导入成功",
        "added_chunks": count,
        "total_chunks": kb.doc_count,
    }


@app.get("/knowledge/stats", tags=["知识库"])
async def knowledge_stats():
    """查看知识库统计信息（文档片段总数）。"""
    tool = _tool_manager._tools.get("knowledge_search") if _tool_manager else None
    if tool is None:
        raise HTTPException(503, "知识库未初始化")
    kb = tool.handler.__self__
    return {"total_chunks": kb.doc_count}


@app.post("/eval/run")
async def run_eval(body: Optional[EvalRunInput] = None):
    """运行内置评测用例，返回评测报告。"""
    if _evaluator is None:
        raise HTTPException(503, "服务未就绪")
    from evaluation.evaluator import (
        DEFAULT_DIALOG_CASES,
        DEFAULT_INTENT_CASES,
        IntentTestCase,
    )

    if body and body.intent_cases is not None:
        intent_cases = [
            IntentTestCase(
                message=c.message,
                expected_intent=c.expected_intent,
                context=c.context,
            )
            for c in body.intent_cases
        ]
    else:
        intent_cases = DEFAULT_INTENT_CASES

    if body and body.dialog_cases is not None:
        dialog_cases = [c.model_dump(exclude_none=True) for c in body.dialog_cases]
    else:
        dialog_cases = DEFAULT_DIALOG_CASES

    report = await _evaluator.run(
        intent_cases=intent_cases,
        dialog_cases=dialog_cases,
    )
    return {
        "pass_rate": report.pass_rate,
        "total": report.total,
        "passed": report.passed,
        "avg_scores": report.avg_scores,
        "regressions": report.regressions,
        "recommendations": report.recommendations,
        "results": [
            {
                "test_id": r.test_id,
                "passed": r.passed,
                "scores": r.scores,
                "detail": r.detail,
                "metadata": r.metadata,
            }
            for r in report.results
        ],
    }


@app.post("/eval/knowledge", tags=["Evaluation"])
async def run_knowledge_eval(body: Optional[KnowledgeEvalInput] = None):
    """运行真实 /chat + /trace 知识问答确定性回归。"""
    from evaluation.knowledge_query_eval import CaseValidationError, run_evaluation

    request = body or KnowledgeEvalInput()
    try:
        return await run_evaluation(
            base_url=os.getenv("KNOWLEDGE_EVAL_BASE_URL", "http://127.0.0.1:8000"),
            case_ids=request.case_ids,
            fail_fast=request.fail_fast,
            cases_path=pathlib.Path(_ROOT)
            / "data"
            / "eval"
            / "knowledge_query_cases.json",
            output_dir=pathlib.Path(_ROOT) / "data" / "eval",
        )
    except CaseValidationError as ex:
        raise HTTPException(400, str(ex)) from ex


# ── 交互式 CLI ────────────────────────────────────────────────────────────────
@managed_entrypoint
async def _cli():
    print(BANNER)
    print("ContentOpsMind CLI — 输入 quit 退出\n")

    from agents.agent_orchestrator import AgentOrchestrator, Request
    from core.context_scope import MISSING_REWRITE_SOURCE_RESPONSE, decide_context_scope
    from memory.conversation_memory import MemoryManager, MsgRole
    from core.skill_loader import SkillManager
    from core.observability import TraceStore

    cfg = _anthropic_cfg()
    skill_manager = SkillManager(
        root_dir=os.getenv(
            "CONTENTOPSMIND_SKILLS_DIR", str(pathlib.Path(_ROOT) / "skills")
        ),
        max_prompt_chars=int(
            os.getenv("CONTENTOPSMIND_SKILLS_MAX_PROMPT_CHARS", "5000")
        ),
    )
    skill_manager.load()
    orch = AgentOrchestrator(
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
        skill_manager=skill_manager,
    )
    mem = MemoryManager(
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
        chroma_host=os.getenv("CHROMA_HOST", "localhost"),
        chroma_port=int(os.getenv("CHROMA_PORT", "8000")),
        chroma_path=os.getenv("CHROMA_PERSIST_DIRECTORY", "/tmp/chroma"),
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
    )

    user_id, conv_id = "cli_user", str(uuid.uuid4())

    while True:
        try:
            msg = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见 ʕ•ᴥ•ʔ")
            break
        if not msg or msg.lower() in ("quit", "exit", "退出"):
            print("再见 ʕ•ᴥ•ʔ")
            break

        ctx = await mem.get_context(user_id, conv_id, query=msg)
        records = [
            {"role": m.role.value, "content": m.content, **(m.metadata or {})}
            for m in ctx.recent_messages
        ]
        scope = decide_context_scope(msg, records)
        previous_task_id = next(
            (
                str(item.get("task_id"))
                for item in reversed(records)
                if item.get("task_id")
            ),
            "",
        )
        task_id = (
            previous_task_id
            if scope.inherit_previous_task and previous_task_id
            else str(uuid.uuid4())
        )
        metadata = {
            "memory_type": "task_fact",
            "scope": "task",
            "conv_id": conv_id,
            "task_id": task_id,
            "ttl": 86400,
        }
        if scope.needs_source_text:
            await mem.add_message(user_id, conv_id, MsgRole.USER, msg, metadata)
            await mem.add_message(
                user_id,
                conv_id,
                MsgRole.ASSISTANT,
                MISSING_REWRITE_SOURCE_RESPONSE,
                metadata,
            )
            print(f"\nContentOpsMind [general]: {MISSING_REWRITE_SOURCE_RESPONSE}\n")
            continue

        if scope.inherit_previous_task and previous_task_id:
            inherited = [
                item for item in records if item.get("task_id") == previous_task_id
            ][-4:]
        elif scope.inherit_previous_task:
            inherited = records[-2:]
        else:
            inherited = []
        history = [
            {"role": item["role"], "content": item["content"]} for item in inherited
        ] or None
        # The CLI does not have structured channel/task controls, so it avoids
        # injecting global preferences without an intent-scoped assembly pass.
        context_parts = []
        if inherited:
            context_parts.append(
                "[TASK_CONTEXT]\n"
                + "\n".join(f"{item['role']}: {item['content']}" for item in inherited)
            )
        context_parts.append(
            "[CONTEXT_BOUNDARY]\n旧任务中的价格、优惠、截止时间、渠道和商品范围不得自动继承。"
        )
        req = Request(
            message=msg,
            user_id=user_id,
            conv_id=conv_id,
            task_id=task_id,
            context="\n\n".join(part for part in context_parts if part),
            history=history,
        )
        result = await orch.run(req)

        await mem.add_message(user_id, conv_id, MsgRole.USER, msg, metadata)
        await mem.add_message(
            user_id, conv_id, MsgRole.ASSISTANT, result.response, metadata
        )

        print(f"\nContentOpsMind [{result.agent_type.value}]: {result.response}\n")


if __name__ == "__main__":
    if "--cli" in sys.argv:
        asyncio.run(_cli())
    else:
        uvicorn.run(
            "api.main:app",
            host=os.getenv("API_HOST", "0.0.0.0"),
            port=int(os.getenv("API_PORT", "8000")),
            reload=os.getenv("APP_ENV") == "development",
        )
