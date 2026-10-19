"""ContentOpsMind multi-agent orchestration.

The orchestrator routes content-operation requests to specialized agents,
runs compound tasks in parallel, and synthesizes agent outputs into a
product-facing final answer.
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from contextvars import ContextVar
from typing import Any, Dict, List, Optional, Union

from anthropic import AsyncAnthropic
from core.llm_gateway import shared_client, gateway_call, llm_task, LLMError

from core.intent_recognizer import IntentCategory, IntentRecognizer, UrgencyLevel
from core.llm_response import EmptyModelResponseError, extract_model_text
from core.compliance_policy import add_legacy_review_fields, validate_review_result
from core.skill_loader import SKILL_OUTPUT_REPAIRS, SKILL_SCHEMA_FAILURES
from core.review_state import (
    ReviewOutputError,
    aggregate_reviews,
    content_hash,
    publish_allowed,
    structured_review,
)
from core.tool_calling import BusinessToolRegistry, ToolCallingManager, ToolExecution

logger = logging.getLogger(__name__)


class AgentType(Enum):
    COPYWRITING = "copywriting"
    RULE_CHECK = "rule_check"
    COMPLIANCE = "compliance"
    STYLE = "style"
    REVIEW = "review"
    GENERAL = "general"


@dataclass
class AgentStats:
    total: int = 0
    success: int = 0
    total_ms: float = 0.0
    monitor_penalty: float = 0.0
    in_flight: int = 0

    @property
    def success_rate(self) -> float:
        return self.success / self.total if self.total else 1.0

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.total if self.total else 0.0

    def routing_score(self) -> float:
        """
        路由评分：（成功率*0.7 + 延迟评分*0.3）*（1-monitor_penalty)
        """
        latency_score = 1.0 / (1.0 + self.avg_ms / 1000)
        return (self.success_rate * 0.7 + latency_score * 0.3) * max(
            0.0, 1.0 - self.monitor_penalty
        )


@dataclass
class AgentResponse:
    agent_type: AgentType
    content: str
    success: bool
    confidence: float = 1.0
    latency_ms: float = 0.0
    escalate: bool = False
    requested_agent_type: Optional[AgentType] = None
    fallback_used: bool = False
    degraded_reason: Optional[str] = None
    error_code: Optional[str] = None
    tool_calls: List[ToolExecution] = field(default_factory=list)
    model_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    skills: List[Dict[str, Any]] = field(default_factory=list)
    review_outputs: List[str] = field(default_factory=list)


@dataclass
class Request:
    message: str
    user_id: str
    conv_id: str
    task_id: str = ""
    context: str = ""
    history: Optional[List[Dict[str, str]]] = None
    intent: Optional[Union[IntentCategory, str]] = None
    secondary_intents: List[Union[IntentCategory, str]] = field(default_factory=list)
    urgency: Optional[UrgencyLevel] = None
    trace_context: Dict[str, Any] = field(default_factory=dict)
    request_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])


@dataclass
class OrchestratorResult:
    request_id: str
    response: str
    agent_type: AgentType
    intent: Optional[IntentCategory]
    secondary_intents: List[IntentCategory] = field(default_factory=list)
    routed_agents: List[AgentType] = field(default_factory=list)
    escalated: bool = False
    latency_ms: float = 0.0
    success: bool = True
    error_code: Optional[str] = None
    trace: Dict[str, Any] = field(default_factory=dict)
    model_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    execution_status: str = ""
    review_status: str = "not_reviewed"
    review_complete: bool = False
    approval_status: str = "pending"
    publish_allowed: bool = False
    draft_version: Optional[int] = None
    content_hash: Optional[str] = None
    review_records: List[Dict[str, Any]] = field(default_factory=list)
    draft_content: Optional[str] = None

    def __post_init__(self) -> None:
        """Derive usage once from the trace without counting an invocation twice."""
        self.execution_status = self.execution_status or (
            self.trace.get("status")
            if self.trace.get("status") in {"success", "partial_success", "failed"}
            else "success" if self.success else "failed"
        )
        # Ordinary agent calls have no durable approval binding.
        self.publish_allowed = publish_allowed(
            review_status=self.review_status,
            review_complete=self.review_complete,
            approval_status=self.approval_status,
            bindings_valid=False,
        )
        executions = self.trace.get("agent_execution", []) if self.trace else []
        synthesis = self.trace.get("synthesis", {}) if self.trace else {}
        self.model_calls = sum(
            int(item.get("model_calls", 0) or 0) for item in executions
        )
        self.input_tokens = sum(
            int(item.get("input_tokens", 0) or 0) for item in executions
        )
        self.output_tokens = sum(
            int(item.get("output_tokens", 0) or 0) for item in executions
        )
        if synthesis.get("used"):
            self.model_calls += int(synthesis.get("model_calls", 0) or 0)
            self.input_tokens += int(synthesis.get("input_tokens", 0) or 0)
            self.output_tokens += int(synthesis.get("output_tokens", 0) or 0)


class BaseAgent:
    """
    Agent基类
    """

    agent_type: AgentType
    system_prompt: str
    allowed_tools: List[str] = []

    def __init__(
        self,
        client: AsyncAnthropic,
        model: str,
        skill_manager: Optional[Any] = None,
        tool_registry: Optional[BusinessToolRegistry] = None,
    ):
        self._client = client
        self._model = model
        self._skill_manager = skill_manager
        self._tool_registry = tool_registry
        self._tool_caller = (
            ToolCallingManager(client, model, tool_registry)
            if tool_registry is not None
            else None
        )
        self._invocation_state = ContextVar(
            f"agent_invocation_{id(self)}", default=None
        )
        self._last_tool_calls: List[ToolExecution] = []
        self._usage = {"model_calls": 0, "input_tokens": 0, "output_tokens": 0}
        self._last_skills: List[Dict[str, Any]] = []
        self.stats = AgentStats()

    def _request_state(self):
        state = self._invocation_state.get()
        if state is None:
            state = {
                "usage": {"model_calls": 0, "input_tokens": 0, "output_tokens": 0},
                "tool_calls": [],
                "skills": [],
            }
            self._invocation_state.set(state)
        return state

    @property
    def _usage(self):
        return self._request_state()["usage"]

    @_usage.setter
    def _usage(self, value):
        self._request_state()["usage"] = value

    @property
    def _last_tool_calls(self):
        return self._request_state()["tool_calls"]

    @_last_tool_calls.setter
    def _last_tool_calls(self, value):
        self._request_state()["tool_calls"] = value

    @property
    def _last_skills(self):
        return self._request_state()["skills"]

    @_last_skills.setter
    def _last_skills(self, value):
        self._request_state()["skills"] = value

    @llm_task()
    async def handle(self, req: Request) -> AgentResponse:
        invocation_token = self._invocation_state.set(None)
        t0 = time.monotonic()
        # `total` means completed attempts. Counting a request before it has
        # finished makes concurrent fan-out look like a burst of failures to
        # the performance monitor (success / started), which can incorrectly
        # remove a healthy singleton Agent from routing while it is working.
        self.stats.in_flight += 1
        try:
            content = await self._call_llm(req)
            if any(
                item.get("skill_id") == "compliance_review"
                for item in self._last_skills
            ):
                content = await self._ensure_compliance_contract(content)
            ms = (time.monotonic() - t0) * 1000
            self.stats.total += 1
            self.stats.success += 1
            self.stats.total_ms += ms
            return AgentResponse(
                agent_type=self.agent_type,
                content=content,
                success=True,
                latency_ms=ms,
                escalate=self._needs_escalation(content),
                requested_agent_type=self.agent_type,
                fallback_used=False,
                tool_calls=list(self._last_tool_calls),
                skills=list(self._last_skills),
                **self._usage,
            )
        except Exception as ex:
            ms = (time.monotonic() - t0) * 1000
            self.stats.total += 1
            self.stats.total_ms += ms
            error_code = (
                ex.code
                if isinstance(ex, LLMError)
                else (
                    "empty_model_response"
                    if isinstance(ex, EmptyModelResponseError)
                    else "agent_execution_failed"
                )
            )
            logger.exception(
                "%s failed error_code=%s",
                self.agent_type.value,
                error_code,
            )
            return AgentResponse(
                agent_type=self.agent_type,
                content=(
                    next(
                        (
                            raw
                            for raw in ex.outputs
                            if (self._json_object(raw) or {}).get("decision")
                            == "blocked"
                        ),
                        "处理请求时出现异常，请稍后重试或交给人工运营负责人复核。",
                    )
                    if isinstance(ex, ReviewOutputError)
                    else "处理请求时出现异常，请稍后重试或交给人工运营负责人复核。"
                ),
                review_outputs=ex.outputs if isinstance(ex, ReviewOutputError) else [],
                escalate=isinstance(ex, ReviewOutputError)
                and any(
                    (self._json_object(raw) or {}).get("decision") == "blocked"
                    for raw in ex.outputs
                ),
                success=False,
                latency_ms=ms,
                requested_agent_type=self.agent_type,
                fallback_used=False,
                degraded_reason=f"{error_code}: {ex}",
                error_code=error_code,
                skills=list(self._last_skills),
                **self._usage,
            )
        finally:
            self.stats.in_flight = max(0, self.stats.in_flight - 1)
            self._invocation_state.reset(invocation_token)

    async def _call_llm(self, req: Request) -> str:
        self._last_tool_calls = []
        self._usage = {"model_calls": 0, "input_tokens": 0, "output_tokens": 0}
        messages = self._build_messages(req)
        system_prompt = self._build_system_prompt(req)
        frozen_review = req.trace_context.get("tool_mode") == "frozen_only"
        if self._tool_caller is not None and self.allowed_tools and not frozen_review:
            tool_system = f"""{system_prompt}

[可用业务工具]
你可以在确实需要结构化业务事实、活动规则确认、商品主数据验证、
合规确定性检查或渠道长度校验时调用工具。

必须遵守：
- 普通风格改写、基于知识库的文案生成或日常对话，不要为了调用工具而调用工具。
- 不要臆造 campaign_id、coupon_id、product_id，也不要使用与用户请求无关的
  内置示例冒充真实业务对象。
- 只有用户明确提供业务 ID，或系统已可靠解析出唯一 ID 时，才调用依赖该 ID 的工具。
- 外部官方知识文档不要求具有内部 product_id。
  当知识库已经提供足够、可信的产品事实时，可以直接基于知识库生成内容。
- 工具未找到内部商品记录，不代表商品不存在，也不代表知识库信息不匹配。
- 若缺少的信息会影响事实准确性，应明确列出需要确认的业务事实；
  不得暴露内部 ID、工具调用过程、函数名、参数名或系统实现。
- 工具结果优先于模型常识，但工具结果不得覆盖更高优先级的人工治理规则。
- 最终答复应直接面向用户，不解释内部检索、路由或降级过程。
- 工具返回的商品与用户目标商品不一致时，必须忽略该工具结果，不得将
  “商品不匹配”“与目标商品无关”等内部判断展示给用户。
- 内部商品库未匹配到 product_id，不代表目标商品不存在，也不代表知识库
  内容无效。
- 当官方知识库已经提供足够事实时，直接基于官方知识生成，不说明内部商品
  匹配、工具降级或检索切换过程。
- 用户侧答复不得出现“该商品”“内置示例”“商品ID不匹配”等无上下文表达。
"""
            try:
                tool_result = await self._tool_caller.run(
                    agent=self.agent_type.value,
                    system=tool_system,
                    messages=messages,
                    allowed_tools=self.allowed_tools,
                    max_tokens=1200,
                    temperature=0,
                )

                self._last_tool_calls = tool_result.tool_calls
                self._usage["model_calls"] += tool_result.model_calls
                self._usage["input_tokens"] += tool_result.input_tokens
                self._usage["output_tokens"] += tool_result.output_tokens

                has_tool_calls = bool(tool_result.tool_calls)

                if has_tool_calls and tool_result.final_text:
                    enriched_messages = [
                        *messages,
                        {
                            "role": "system",
                            "content": (
                                "[已核验业务事实]\n"
                                f"{tool_result.final_text}\n\n"
                                "只使用与用户目标对象明确一致的事实。"
                                "若存在不匹配、无结果或内部示例，请忽略。"
                                "不得向用户暴露工具调用、业务ID、匹配失败、"
                                "检索切换或降级过程。"
                            ),
                        },
                    ]

                    return await self._call_plain_llm(
                        system_prompt,
                        enriched_messages,
                    )

                # 没有实际调用工具，直接走正式生成流程
                return await self._call_plain_llm(
                    system_prompt,
                    messages,
                )

            except LLMError:
                raise
            except Exception as ex:
                logger.warning(
                    "tool calling failed for %s, falling back to plain LLM: %s",
                    self.agent_type.value,
                    ex,
                )
                self._last_tool_calls = []

            return await self._call_plain_llm(system_prompt, messages)

        # Agents without business tools (for example GeneralAgent) still need
        # to execute the normal model generation path.
        return await self._call_plain_llm(system_prompt, messages)

    def _build_messages(self, req: Request) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = []
        if req.context:
            messages.append(
                {
                    "role": "user",
                    "content": f"[可用背景/知识库]\n{self._clean(req.context)}",
                }
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": "已读取背景信息，会优先遵守其中的规则。",
                }
            )
        messages.append({"role": "user", "content": self._clean(req.message)})
        return messages

    async def _call_plain_llm(
        self, system_prompt: str, messages: List[Dict[str, Any]]
    ) -> str:
        resp = await gateway_call(
            self._client,
            model=self._model,
            max_tokens=1200,
            temperature=0,
            system=system_prompt,
            messages=messages,
        )
        usage = getattr(resp, "usage", None)
        self._usage["model_calls"] += 1
        self._usage["input_tokens"] += int(getattr(usage, "input_tokens", 0))
        self._usage["output_tokens"] += int(getattr(usage, "output_tokens", 0))
        return extract_model_text(resp)

    async def _ensure_compliance_contract(self, content: str) -> str:
        """Validate the policy output and allow one generic format-repair call."""
        outputs = [content]
        parsed = self._json_object(content)
        original_blocked = parsed is not None and parsed.get("decision") == "blocked"
        errors = (
            validate_review_result(parsed)
            if parsed is not None
            else ["response is not a JSON object"]
        )
        repair_count = 0
        if errors:
            SKILL_SCHEMA_FAILURES.labels("compliance_review").inc()
            repair_prompt = (
                "将下面的审核结果仅转换为合规 Skill 要求的 JSON。不得增加原文没有的风险或证据。"
                "必须包含 decision、risk_items、missing_evidence、summary；风险项包含 rule_id、severity、"
                "matched_text、evidence_refs、reason、suggestion。\n校验错误："
                + "; ".join(errors)
                + "\n原结果：\n"
                + content
            )
            try:
                repaired = await self._call_plain_llm(
                    "你是结构化输出修复器，只修复格式，不重新审核。",
                    [{"role": "user", "content": repair_prompt}],
                )
            except Exception as ex:
                self._record_skill_validation(False, 1, ["format repair failed"])
                raise ReviewOutputError(
                    "compliance_output_repair_failed", outputs
                ) from ex
            SKILL_OUTPUT_REPAIRS.labels("compliance_review").inc()
            repair_count = 1
            outputs.append(repaired)
            parsed = self._json_object(repaired)
            errors = (
                validate_review_result(parsed)
                if parsed is not None
                else ["repair is not a JSON object"]
            )
            if original_blocked and (parsed or {}).get("decision") != "blocked":
                errors.append("format repair cannot remove an explicit block")
        if errors:
            self._record_skill_validation(False, repair_count, errors)
            raise ReviewOutputError(
                "compliance_output_schema_invalid: " + "; ".join(errors), outputs
            )
        self._record_skill_validation(True, repair_count, [])
        return json.dumps(add_legacy_review_fields(parsed), ensure_ascii=False)

    def _record_skill_validation(
        self, valid: bool, repair_count: int, errors: List[str]
    ) -> None:
        for item in self._last_skills:
            if item.get("skill_id") == "compliance_review":
                item["schema_valid"] = valid
                item["output_repair_count"] = repair_count
                item["schema_errors"] = list(errors)

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

    def _build_system_prompt(self, req: Request) -> str:
        prompt = (
            f"{self.system_prompt}\n\n"
            "[上下文保密边界]\n"
            "内部记忆、用户画像、任务字段、上下文标签、路由、Agent、工具和检索过程"
            "只用于生成当前答案，不得向用户复述、解释或列举。"
            "不得依据与当前问题无关的偏好或任务事实追加营销建议、追问或任务引导。"
        )
        self._last_skills = []
        if self._skill_manager is not None:
            intent = (
                req.intent.value
                if isinstance(req.intent, Enum)
                else str(req.intent or "")
            )
            selections = self._skill_manager.resolve(
                req.message, self.agent_type.value, intent
            )
            self._last_skills = [selection.to_trace() for selection in selections]
            if selections:
                skill_prompt = "\n\n".join(
                    selection.prompt_block for selection in selections
                )
                prompt = f"{prompt}\n\n[版本化业务 Skills]\n{skill_prompt}"
        return prompt

    @staticmethod
    def _clean(value: Any) -> str:
        return str(value or "").encode("utf-8", errors="ignore").decode("utf-8")

    @staticmethod
    def _needs_escalation(content: str) -> bool:
        safe_content = content if isinstance(content, str) else ""
        parsed = BaseAgent._json_object(safe_content)
        if parsed and parsed.get("decision") == "blocked":
            return True
        return any(
            keyword in safe_content
            for keyword in ["人工复核", "法务复核", "禁止发布", "高风险"]
        )


class CopywritingAgent(BaseAgent):
    agent_type = AgentType.COPYWRITING
    allowed_tools = [
        "get_campaign_rule",
        "get_coupon_policy",
        "query_product_info",
        "estimate_message_length",
    ]

    system_prompt = """
    你是内容运营文案生成 Agent。职责是基于用户提供的信息、活动规则、品牌规范和目标渠道生成可发布草稿。
    
    要求:
    
    【事实约束】
    - 不编造用户未提供的活动权益、价格、库存、有效期、适用范围和活动状态。
    - 活动时间、库存数量、优惠限制、适用条件必须来自用户输入或知识库。
    - 不主动添加未经确认的信息，缺少关键约束时，用“需确认”提示，不自行补全。
    - 不编造用户未提供的活动权益、价格、库存、有效期和适用范围。
    - 没有产品名称、核心卖点或待改写原文时，不得生成成稿，结论必须为“需补充”。
    
    【内容生成】
    - 用户只要求生成文案时，优先输出可直接使用的文案，不展开完整规则审计报告。
    - 短信/Push 优先给 1 条主推文案，可附 1 条备选；不要输出 3 个以上版本。
    - 根据目标渠道调整表达风格，但不要改变用户提供的业务事实。
    
    【发布提醒】
    - 文案后最多给 2-3 条发布提醒。
    - 只保留强相关信息，例如：
      有效期确认、适用范围确认、落地页规则确认。
    只针对当前生成文案中实际出现的信息提供提醒。
    禁止引用：
    - 当前文案未使用的知识库信息
    - 历史活动
    - 旧优惠规则
    - 未写入正文的产品限制
    如果正文没有涉及活动、优惠、价格、有效期，
    不要生成对应提醒。
    发布提醒不是知识库摘要。
    - 不因轻微优化建议直接判定不可发布。
    
    【结论】
    - 结论只用：
      “可发布 / 需补充 / 不建议发布”
      三者之一。
    
    【合规协同】
    - 只有用户明确要求合规检查或规则检查时，才展开详细审核项。
    - 当合规 Agent 作为辅助 Agent 参与时，只保留必要风险提醒，不影响文案生成主任务。
    """


class RuleCheckAgent(BaseAgent):
    agent_type = AgentType.RULE_CHECK
    allowed_tools = ["get_campaign_rule", "get_coupon_policy", "query_product_info"]
    system_prompt = """你是业务事实与活动规则核验 Agent。
只核验正文实际声明或明确暗示的产品事实与活动规则能否追溯到当前输入和当前任务证据。
只有正文声明缺少证据、与证据冲突，或省略限定造成错误适用暗示时，才可要求修改或阻断。
正文没有提及的叠加、库存、名额、退款等非必需规则缺失只能作为 advisory，不得单独改变通过结论。
不要判断广告法、平台合规、品牌语气、内容结构或转化效果，也不要重写文案。
审核任务要求 JSON 时，只返回 scope=fact_rule、decision、need_revision、summary、findings。"""


class ComplianceAgent(BaseAgent):
    agent_type = AgentType.COMPLIANCE
    allowed_tools = ["check_banned_words"]

    system_prompt = """你是广告与平台合规审核 Agent。只审核当前营销草稿，不生成整篇营销文案。
当前输入和当前 task_id 的证据优先于历史记忆；不得把旧任务事实当作当前事实。
只检查禁用词、绝对化宣传、无依据功效、虚假承诺和必要披露风险。
不要检查活动字段是否完整，不评价渠道文风、品牌语气或转化效果。
必须遵循已加载合规 Skill 的规则枚举、严重度矩阵和 JSON 输出契约。
最终只输出 JSON，不使用自由生成的风险标签，也不得虚构证据。"""


class StyleAgent(BaseAgent):
    agent_type = AgentType.STYLE
    system_prompt = """你是品牌风格与渠道适配 Agent。
职责是根据目标渠道调整内容结构和表达方式。

渠道要求：

【小红书】
- 输出符合小红书内容结构：
  1. 标题（吸引点击但不过度夸张）
  2. 场景化正文
  3. emoji 分段
  4. 用户互动引导
  5. 相关话题标签（3-8个）
- 强调真实体验、使用场景和用户共鸣。
- 避免虚假种草、绝对化承诺和医疗功效表达。

【微信公众号】
- 强调完整结构、品牌信息和事实说明。

【短信/Push】
- 简洁明确，突出核心信息。

保持品牌调性一致，不为了吸引点击牺牲准确性。
只能基于已有事实改写，不补充无来源卖点。"""


class ReviewAgent(BaseAgent):
    agent_type = AgentType.REVIEW

    system_prompt = """你是内容质量与渠道表达审核 Agent。
只检查内容与需求的相关性、结构、品牌语气、可读性、CTA 和渠道表达质量。
输入中的 rejected_constraints 是生成阶段已安全删除的无效约束，必须从需求完整性检查中排除；不得要求正文补回这些约束，也不得因其缺失判定需要修改。
只要求正文满足当前仍有效的约束。
不要判断产品或活动事实真假，不检查活动规则，也不要给出广告法或平台合规结论。
不得复用旧任务内容。审核任务要求 JSON 时，只返回 scope=content_quality、decision、need_revision、summary、findings。"""


class GeneralAgent(BaseAgent):
    agent_type = AgentType.GENERAL
    system_prompt = """你是 ContentOpsMind 内容运营助手。帮助运营人员完成活动规则理解、文案生成、渠道适配、合规检查和内容评审。
如果用户只是寒暄，简短回应；如果需求不完整，引导补充活动目标、商品、渠道、受众和限制条件。"""


class AgentOrchestrator:
    MIN_ROUTING_SCORE = 0.25
    GENERAL_FALLBACK_ALLOWED = {AgentType.COPYWRITING}

    _INTENT_ROUTING: Dict[IntentCategory, AgentType] = {
        IntentCategory.COPY_GENERATION: AgentType.COPYWRITING,
        IntentCategory.COPY_REWRITE: AgentType.COPYWRITING,
        IntentCategory.CAMPAIGN_PLANNING: AgentType.COPYWRITING,
        IntentCategory.COMPLIANCE_CHECK: AgentType.COMPLIANCE,
        IntentCategory.RULE_CHECK: AgentType.RULE_CHECK,
        IntentCategory.STYLE_ADAPTATION: AgentType.STYLE,
        IntentCategory.CONTENT_REVIEW: AgentType.REVIEW,
        IntentCategory.KNOWLEDGE_QUERY: AgentType.GENERAL,
    }

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        model: str = "claude-3-5-sonnet-20241022",
        skill_manager: Optional[Any] = None,
        trace_store: Optional[Any] = None,
    ):
        kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = shared_client(AsyncAnthropic, **kwargs)
        self._model = model
        self._intent_recognizer = IntentRecognizer(
            api_key=api_key, base_url=base_url, model=model
        )
        self._skill_manager = skill_manager
        self._trace_store = trace_store
        self._tool_registry = BusinessToolRegistry()
        self._pool: Dict[AgentType, List[BaseAgent]] = {
            AgentType.COPYWRITING: [
                CopywritingAgent(
                    self._client, model, skill_manager, self._tool_registry
                )
            ],
            AgentType.RULE_CHECK: [
                RuleCheckAgent(self._client, model, skill_manager, self._tool_registry)
            ],
            AgentType.COMPLIANCE: [
                ComplianceAgent(self._client, model, skill_manager, self._tool_registry)
            ],
            AgentType.STYLE: [
                StyleAgent(self._client, model, skill_manager, self._tool_registry)
            ],
            AgentType.REVIEW: [
                ReviewAgent(self._client, model, skill_manager, self._tool_registry)
            ],
            AgentType.GENERAL: [
                GeneralAgent(self._client, model, skill_manager, self._tool_registry)
            ],
        }

    def set_skill_manager(self, skill_manager: Optional[Any]) -> None:
        self._skill_manager = skill_manager
        for agents in self._pool.values():
            for agent in agents:
                agent._skill_manager = skill_manager

    @llm_task()
    async def run(self, req: Request) -> OrchestratorResult:
        t0 = time.monotonic()
        trace = self._new_trace(req)
        try:
            if req.intent is None:
                intent_result = await self._intent_recognizer.recognize(
                    req.message, history=req.history
                )
                req.intent = intent_result.intent
                req.secondary_intents = intent_result.secondary_intents
                req.urgency = intent_result.urgency

            req.intent = self._normalize_intent(req.intent)
            req.secondary_intents = self._filter_secondary_intents(
                req.message,
                self._normalize_secondary_intents(req.secondary_intents),
            )
            if self._should_add_style_intent(req):
                req.secondary_intents = list(
                    dict.fromkeys(
                        [*req.secondary_intents, IntentCategory.STYLE_ADAPTATION]
                    )
                )

            targets = self._collaboration_targets(req)
            trace["intent"] = {
                "primary": req.intent.value if req.intent else "other",
                "secondary": [intent.value for intent in req.secondary_intents],
            }
            trace["routing"] = [agent.value for agent in targets]
            trace["retrieval"] = req.trace_context.get("retrieval", {})

            if len(targets) > 1:
                if self._requires_hybrid_dag(req, targets):
                    return await self.run_hybrid_dag(req, targets, t0, trace)
                return await self.run_parallel(req, targets, t0, trace)

            agent_type = targets[0] if targets else self._route(req.intent)
            response = await self._execute(req, agent_type)
            trace["agent_execution"].append(self._agent_trace(response))
            trace["status"] = "success" if response.success else "failed"
            trace["latency_ms"] = round((time.monotonic() - t0) * 1000, 1)

            result = OrchestratorResult(
                request_id=req.request_id,
                response=response.content,
                agent_type=response.agent_type,
                intent=req.intent,
                secondary_intents=req.secondary_intents,
                routed_agents=[response.agent_type],
                escalated=response.escalate or req.urgency == UrgencyLevel.CRITICAL,
                latency_ms=trace["latency_ms"],
                success=response.success,
                error_code=response.error_code,
                trace=trace,
                **self._unbound_review_state([response], [agent_type], trace),
            )
            self._record_trace(trace)
            return result
        except Exception as ex:
            trace["status"] = "failed"
            trace["error"] = str(ex)
            trace["latency_ms"] = round((time.monotonic() - t0) * 1000, 1)
            self._record_trace(trace)
            raise

    @llm_task()
    async def run_parallel(
        self,
        req: Request,
        agent_types: List[AgentType],
        started_at: Optional[float] = None,
        trace: Optional[Dict[str, Any]] = None,
    ) -> OrchestratorResult:
        t0 = started_at or time.monotonic()
        trace = trace or self._new_trace(req)
        trace["orchestration"] = {
            "mode": "parallel_fanout",
            "stages": [
                {
                    "name": "parallel_analysis",
                    "agents": [agent.value for agent in agent_types],
                },
                {"name": "synthesis", "agents": ["synthesizer"]},
            ],
        }
        responses = await asyncio.gather(
            *(self._execute(req, agent_type) for agent_type in agent_types),
            return_exceptions=True,
        )
        successful = [
            r for r in responses if isinstance(r, AgentResponse) and r.success
        ]
        trace["agent_execution"] = [
            self._agent_trace(r) for r in responses if isinstance(r, AgentResponse)
        ]
        trace["agent_errors"] = self._agent_errors_from_results(responses)
        escalated = any(response.escalate for response in successful)
        synthesis_t0 = time.monotonic()
        if successful:
            combined, synthesis_usage = await self._synthesize_response(req, successful)
            trace["synthesis"] = {
                "used": True,
                "latency_ms": round((time.monotonic() - synthesis_t0) * 1000, 1),
                **synthesis_usage,
            }
        else:
            combined = "所有 Agent 均处理失败，请稍后重试。"
            trace["synthesis"] = {"used": False, "latency_ms": 0.0}

        if successful and trace["agent_errors"]:
            trace["status"] = "partial_success"
        else:
            trace["status"] = "success" if successful else "failed"
        trace["latency_ms"] = round((time.monotonic() - t0) * 1000, 1)
        if not successful:
            trace["error"] = "all agents failed"
        result = OrchestratorResult(
            request_id=req.request_id,
            response=combined,
            agent_type=agent_types[0],
            intent=req.intent,
            secondary_intents=req.secondary_intents,
            routed_agents=agent_types,
            escalated=escalated,
            latency_ms=trace["latency_ms"],
            success=bool(successful),
            error_code=None if successful else "agent_execution_failed",
            trace=trace,
            **self._unbound_review_state(responses, agent_types, trace),
        )
        self._record_trace(trace)
        return result

    @llm_task()
    async def run_hybrid_dag(
        self,
        req: Request,
        agent_types: List[AgentType],
        started_at: Optional[float] = None,
        trace: Optional[Dict[str, Any]] = None,
    ) -> OrchestratorResult:
        """Run dependent workflows as producer -> optional style -> review -> optional revision."""
        t0 = started_at or time.monotonic()
        trace = trace or self._new_trace(req)
        producer_agent = (
            AgentType.COPYWRITING
            if AgentType.COPYWRITING in agent_types
            else AgentType.STYLE
        )
        producer_phase = (
            "draft_generation"
            if producer_agent == AgentType.COPYWRITING
            else "style_adaptation"
        )
        style_requested = (
            producer_agent == AgentType.COPYWRITING and AgentType.STYLE in agent_types
        )
        review_agent_types = [
            agent
            for agent in agent_types
            if agent not in (AgentType.COPYWRITING, AgentType.STYLE)
        ]
        trace["orchestration"] = {
            "mode": "hybrid_dag",
            "producer_agent": producer_agent.value,
            "stages": [
                {"name": producer_phase, "agents": [producer_agent.value]},
                {
                    "name": "style_adaptation",
                    "agents": [AgentType.STYLE.value],
                    "conditional": style_requested,
                },
                {
                    "name": "parallel_review",
                    "agents": [agent.value for agent in review_agent_types],
                },
                {
                    "name": "revision",
                    "agents": [AgentType.COPYWRITING.value],
                    "conditional": True,
                },
                {"name": "synthesis", "agents": ["synthesizer"]},
            ],
        }

        producer_response = await self._execute(req, producer_agent)
        trace["agent_execution"].append(
            self._agent_trace(producer_response, phase=producer_phase)
        )
        if not producer_response.success:
            trace["status"] = "failed"
            trace["error"] = f"{producer_phase}_failed"
            trace["agent_errors"] = self._agent_errors_from_results([producer_response])
            trace["latency_ms"] = round((time.monotonic() - t0) * 1000, 1)
            self._record_trace(trace)
            return OrchestratorResult(
                request_id=req.request_id,
                response=producer_response.content,
                agent_type=producer_agent,
                intent=req.intent,
                secondary_intents=req.secondary_intents,
                routed_agents=agent_types,
                escalated=True,
                latency_ms=trace["latency_ms"],
                success=False,
                error_code=producer_response.error_code or "agent_execution_failed",
                trace=trace,
            )

        working_draft = producer_response
        agent_errors: List[Dict[str, Any]] = []
        if style_requested:
            style_req = self._build_style_request(req, producer_response)
            style_response = await self._execute(style_req, AgentType.STYLE)
            trace["agent_execution"].append(
                self._agent_trace(style_response, phase="style_adaptation")
            )
            if style_response.success:
                working_draft = style_response
            else:
                agent_errors.extend(self._agent_errors_from_results([style_response]))

        review_successful: List[AgentResponse] = []
        review_incomplete = False
        if review_agent_types:
            review_req = self._build_review_request(req, working_draft)
            review_results = await self._execute_review_batch(
                review_req, review_agent_types, trace, phase="parallel_review"
            )
            review_successful = [
                r for r in review_results if isinstance(r, AgentResponse) and r.success
            ]
            review_incomplete = len(review_successful) != len(review_agent_types)
            agent_errors.extend(self._agent_errors_from_results(review_results))
        trace["agent_errors"] = agent_errors

        final_draft = working_draft
        draft_version = 1
        review_records = self._bound_review_records(
            review_results if review_agent_types else [],
            working_draft,
            1,
            review_agent_types,
        )
        revision_used = False
        review_decisions = [item["decision"] for item in review_records]
        auto_revision_allowed = (
            not review_incomplete
            and all(item["valid"] for item in review_records)
            and "blocked" not in review_decisions
            and "execution_failed" not in review_decisions
            and "changes_required" in review_decisions
        )
        if review_successful and auto_revision_allowed:
            revision_req = self._build_revision_request(
                req, working_draft, review_successful
            )
            revision = await self._execute(revision_req, AgentType.COPYWRITING)
            trace["agent_execution"].append(
                self._agent_trace(revision, phase="revision")
            )
            revision_used = revision.success
            if revision.success:
                final_draft = revision
                draft_version = 2
                post_review_req = self._build_review_request(req, final_draft)
                post_review_results = await self._execute_review_batch(
                    post_review_req,
                    review_agent_types,
                    trace,
                    phase="post_revision_review",
                )
                post_successful = [
                    item
                    for item in post_review_results
                    if isinstance(item, AgentResponse) and item.success
                ]
                review_incomplete = len(post_successful) != len(review_agent_types)
                review_successful = post_successful
                review_records.extend(
                    self._bound_review_records(
                        post_review_results, final_draft, 2, review_agent_types
                    )
                )
                agent_errors.extend(
                    self._agent_errors_from_results(post_review_results)
                )
                trace["agent_errors"] = agent_errors
            else:
                agent_errors.extend(self._agent_errors_from_results([revision]))
                trace["agent_errors"] = agent_errors

        synthesis_t0 = time.monotonic()
        final_records = [
            item for item in review_records if item["draft_version"] == draft_version
        ]
        review_state = (
            aggregate_reviews(
                [item["decision"] for item in final_records],
                complete=bool(review_agent_types)
                and len(final_records) == len(review_agent_types)
                and all(item["valid"] for item in final_records),
            )
            if review_agent_types
            else {"review_status": "not_reviewed", "review_complete": False}
        )
        review_incomplete = (
            bool(review_agent_types) and not review_state["review_complete"]
        )
        approval_status = (
            "human_required"
            if review_state["review_status"] == "blocked"
            else "pending"
        )
        trace["review_state"] = {
            **review_state,
            "approval_status": approval_status,
            "publish_allowed": False,
            "draft_version": draft_version,
            "content_hash": content_hash(final_draft.content),
            "records": review_records,
        }
        synthesis_inputs = [final_draft]
        synthesis_inputs.extend(review_successful)
        combined, synthesis_usage = await self._synthesize_response(
            req, synthesis_inputs
        )
        review_label = {
            "pass": "本轮审核通过",
            "blocked": "已阻断",
            "changes_required": "需要修改",
            "incomplete": "未完成",
            "not_reviewed": "未审核",
        }[review_state["review_status"]]
        approval_label = (
            "需要人工处理" if approval_status == "human_required" else "待人工审批"
        )
        combined += f"\n\n审核状态：{review_label}；审批状态：{approval_label}；发布资格：未满足。"
        if review_incomplete:
            combined = (
                f"{combined}\n\n必要审核尚未全部完成：存在缺失、执行失败或无效结果。"
                "当前结果不能标记为可发布，请重试失败的审核。"
            )
        trace["synthesis"] = {
            "used": True,
            "latency_ms": round((time.monotonic() - synthesis_t0) * 1000, 1),
            "revision_used": revision_used,
            **synthesis_usage,
        }
        trace["status"] = "partial_success" if agent_errors else "success"
        trace["latency_ms"] = round((time.monotonic() - t0) * 1000, 1)
        escalated = (
            any(response.escalate for response in synthesis_inputs)
            or bool(agent_errors)
            or review_state["review_status"] == "blocked"
            or review_incomplete
        )
        result = OrchestratorResult(
            request_id=req.request_id,
            response=combined,
            agent_type=producer_agent,
            intent=req.intent,
            secondary_intents=req.secondary_intents,
            routed_agents=agent_types,
            escalated=escalated,
            latency_ms=trace["latency_ms"],
            success=True,
            error_code=None,
            trace=trace,
            **review_state,
            approval_status=approval_status,
            draft_version=draft_version,
            content_hash=content_hash(final_draft.content),
            draft_content=final_draft.content,
            review_records=review_records,
        )
        self._record_trace(trace)
        return result

    @staticmethod
    def _bound_review_records(
        results: List[Any],
        draft: Optional[AgentResponse],
        version: Optional[int],
        expected_agents: Optional[List[AgentType]] = None,
    ) -> List[Dict[str, Any]]:
        records = []
        for index, result in enumerate(results):
            value = (
                BaseAgent._json_object(result.content)
                if isinstance(result, AgentResponse)
                else None
            )
            prior_outputs = (
                result.review_outputs if isinstance(result, AgentResponse) else []
            )
            prior_block = next(
                (
                    BaseAgent._json_object(raw)
                    for raw in prior_outputs
                    if (BaseAgent._json_object(raw) or {}).get("decision") == "blocked"
                ),
                None,
            )
            known_block = prior_block or (
                value
                if isinstance(value, dict) and value.get("decision") == "blocked"
                else None
            )
            decision, valid, error = (
                "blocked" if known_block else "execution_failed",
                False,
                None,
            )
            if isinstance(result, AgentResponse) and result.success:
                try:
                    value = structured_review(value)
                    if prior_block and value["decision"] != "blocked":
                        raise ValueError(
                            "retry cannot erase a previously identified block"
                        )
                    if (
                        result.fallback_used
                        or expected_agents
                        and result.agent_type != expected_agents[index]
                    ):
                        raise ValueError("required reviewer was replaced by a fallback")
                    expected_scope = {
                        AgentType.COMPLIANCE: "compliance",
                        AgentType.RULE_CHECK: "fact_rule",
                        AgentType.REVIEW: "content_quality",
                    }.get(result.agent_type)
                    if "scope" in value and value["scope"] != expected_scope:
                        raise ValueError(
                            "review scope does not match requested reviewer"
                        )
                    decision, valid = value["decision"], True
                except ValueError as ex:
                    error = str(ex)
                    # A malformed blocking result never becomes a pass.
                    if isinstance(value, dict) and value.get("decision") == "blocked":
                        decision = "blocked"
                    if prior_block:
                        value = prior_block
            records.append(
                {
                    "agent": (
                        result.agent_type.value
                        if isinstance(result, AgentResponse)
                        else "unknown"
                    ),
                    "draft_version": version,
                    "content_hash": content_hash(draft.content) if draft else None,
                    "decision": decision,
                    "valid": valid,
                    "error": error,
                    "result": value,
                    "draft_content": draft.content if draft else None,
                    "raw_output": (
                        result.content if isinstance(result, AgentResponse) else None
                    ),
                    "raw_outputs": prior_outputs,
                }
            )
        return records

    @classmethod
    def _unbound_review_state(
        cls, results: List[Any], agents: List[AgentType], trace: Dict[str, Any]
    ) -> Dict[str, Any]:
        reviewers = {AgentType.COMPLIANCE, AgentType.RULE_CHECK, AgentType.REVIEW}
        slots = [
            (result, agent)
            for result, agent in zip(results, agents)
            if agent in reviewers
        ]
        if not slots:
            return {}
        records = cls._bound_review_records(
            [r for r, _ in slots], None, None, [a for _, a in slots]
        )
        state = aggregate_reviews(
            [r["decision"] for r in records], complete=all(r["valid"] for r in records)
        )
        approval_status = (
            "human_required" if state["review_status"] == "blocked" else "pending"
        )
        trace["review_state"] = {
            **state,
            "approval_status": approval_status,
            "publish_allowed": False,
            "records": records,
        }
        return {**state, "approval_status": approval_status, "review_records": records}

    async def _execute_review_batch(
        self,
        request: Request,
        agent_types: List[AgentType],
        trace: Dict[str, Any],
        *,
        phase: str,
    ) -> List[Any]:
        """Run independent reviewers concurrently and retry only failed slots once."""
        results: List[Any] = list(
            await asyncio.gather(
                *(self._execute(request, agent_type) for agent_type in agent_types),
                return_exceptions=True,
            )
        )
        trace["agent_execution"].extend(
            self._agent_trace(item, phase=phase)
            for item in results
            if isinstance(item, AgentResponse)
        )
        failed_indexes = [
            index
            for index, item in enumerate(results)
            if not isinstance(item, AgentResponse) or not item.success
        ]
        if failed_indexes:
            retries = await asyncio.gather(
                *(
                    self._execute(request, agent_types[index])
                    for index in failed_indexes
                ),
                return_exceptions=True,
            )
            for index, retry in zip(failed_indexes, retries):
                original = results[index]
                if isinstance(original, AgentResponse) and isinstance(
                    retry, AgentResponse
                ):
                    retry.review_outputs = [
                        *original.review_outputs,
                        original.content,
                        *retry.review_outputs,
                    ]
                if isinstance(retry, AgentResponse):
                    trace["agent_execution"].append(
                        self._agent_trace(retry, phase=f"{phase}_retry")
                    )
                results[index] = retry
        return results

    def _new_trace(self, req: Request) -> Dict[str, Any]:
        return {
            "request_id": req.request_id,
            "conv_id": req.conv_id,
            "task_id": req.task_id,
            "user_id": req.user_id,
            "message_preview": req.message[:120],
            "started_at": time.time(),
            "intent": {},
            "routing": [],
            "retrieval": req.trace_context.get("retrieval", {}),
            "memory_policy": req.trace_context.get("memory_policy", {}),
            "agent_execution": [],
            "synthesis": {"used": False, "latency_ms": 0.0},
            "orchestration": {"mode": "single", "stages": []},
            "status": "running",
        }

    @staticmethod
    def _agent_trace(response: AgentResponse, phase: str = "execute") -> Dict[str, Any]:
        return {
            "agent": response.agent_type.value,
            "requested_agent": (
                response.requested_agent_type or response.agent_type
            ).value,
            "actual_agent": response.agent_type.value,
            "fallback_used": response.fallback_used,
            "degraded_reason": response.degraded_reason,
            "error_code": response.error_code,
            "model_calls": response.model_calls,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "phase": phase,
            "success": response.success,
            "latency_ms": round(response.latency_ms, 1),
            "escalate": response.escalate,
            "tool_calls": [call.to_trace() for call in response.tool_calls],
            "skills": list(response.skills),
        }

    @staticmethod
    def _agent_errors_from_results(results: List[Any]) -> List[Dict[str, Any]]:
        errors: List[Dict[str, Any]] = []
        for result in results:
            if isinstance(result, AgentResponse):
                if result.success:
                    continue
                requested_agent = result.requested_agent_type or result.agent_type
                errors.append(
                    {
                        "agent": requested_agent.value,
                        "actual_agent": result.agent_type.value,
                        "fallback_used": result.fallback_used,
                        "reason": result.degraded_reason or result.content,
                    }
                )
            elif isinstance(result, Exception):
                errors.append({"agent": "unknown", "reason": str(result)})
        return errors

    def _record_trace(self, trace: Dict[str, Any]) -> None:
        if self._trace_store is not None:
            self._trace_store.record(trace)

    def _filter_secondary_intents(
        self, message: str, secondary_intents: List[IntentCategory]
    ) -> List[IntentCategory]:
        return list(dict.fromkeys(secondary_intents))

    @staticmethod
    def _normalize_intent(
        intent: Optional[Union[IntentCategory, str]],
    ) -> Optional[IntentCategory]:
        if intent is None or isinstance(intent, IntentCategory):
            return intent
        try:
            return IntentCategory(str(intent))
        except ValueError:
            return IntentCategory.OTHER

    @classmethod
    def _normalize_secondary_intents(
        cls,
        intents: Optional[List[Union[IntentCategory, str]]],
    ) -> List[IntentCategory]:
        normalized: List[IntentCategory] = []
        for intent in intents or []:
            parsed = cls._normalize_intent(intent)
            if parsed not in (None, IntentCategory.OTHER):
                normalized.append(parsed)
        return list(dict.fromkeys(normalized))

    async def _synthesize_response(
        self, req: Request, responses: List[AgentResponse]
    ) -> tuple[str, Dict[str, int]]:
        """Merge parallel agent outputs with an intent-specific product template."""
        source_payload = [
            {
                "agent": response.agent_type.value,
                "latency_ms": round(response.latency_ms, 1),
                "content": response.content[:2500],
            }
            for response in responses
        ]
        template_name, template_rules = self._synthesis_template(req)
        prompt = f"""你是内容运营助手的最终回复编辑器。请根据任务类型选择合适结构，而不是固定输出同一种格式。
发布资格只能由服务端结合版本、审核和持久化人工审批确定。当前普通助手输出没有持久化审批资格。
不要把执行成功或本轮审核通过表述为已获准发布；审核通过仍须人工审批。不得覆盖结构化审核中的阻断或未完成结论。

用户原始需求:
{req.message}

任务类型:
{template_name}

Agent 输出 JSON:
{json.dumps(source_payload, ensure_ascii=False, indent=2)}

输出模板要求:
{template_rules}

通用要求:

- 不要提及 Agent、模型、工具、融合、内部流程、系统实现或答案来源。
- 不要展示内部推理过程。

【内容保真约束】
- 你的职责是整理和融合已有 Agent 输出，而不是重新创作文案。
- 不得新增 Agent 输出中不存在的事实信息。
- 不得补充未经用户输入或知识库确认的：
  - 活动日期
  - 库存数量
  - 优惠力度
  - 适用范围
  - 产品性能
  - 功效描述

【营销表达约束】
- 可以优化语言感染力和渠道风格。
- 但不得改变用户提供的活动状态。
- “今晚24点截止”只能表达为截止时间，不应扩展为：
  “最后机会”
  “最后冲刺”
  “即将售罄”
  “错过再等一年”
  等未经确认的紧迫性表达。

【任务边界】
- 如果用户只是要生成文案，审核内容要短，只保留必要提醒，不要喧宾夺主。
- 如果用户只是要合规检查，不要主动生成完整营销文案，只给必要修改建议。
- 如果用户只是要规则检查，不要主动生成营销内容。

【输出要求】
- 保留 Agent 输出中的核心信息。
- 只进行结构整理、去重和表达优化。
- 输出必须能够被运营人员直接使用。
- 如果存在必须确认的关键业务信息，不要输出“可发布”，应输出“需补充”。
- 不得新增 Agent 输出和当前任务来源中不存在的产品体验、生活场景、卖点或促销事实。
- 改写任务没有可解析的 source_text 时，只能要求补充输入，不得生成草稿。"""
        try:
            resp = await gateway_call(
                self._client,
                model=self._model,
                max_tokens=900,
                temperature=0,
                system="你负责把多路处理结果整理成符合当前任务类型的产品级回复。隐藏所有内部实现信息。",
                messages=[{"role": "user", "content": prompt}],
            )
            usage = getattr(resp, "usage", None)
            return self._clean_product_response(extract_model_text(resp)), {
                "model_calls": 1,
                "input_tokens": int(getattr(usage, "input_tokens", 0)),
                "output_tokens": int(getattr(usage, "output_tokens", 0)),
            }
        except Exception as ex:
            logger.warning(
                "response synthesis failed, using deterministic fallback: %s", ex
            )
            return self._fallback_synthesis(req, responses), {
                "model_calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
            }

    def _synthesis_template(self, req: Request) -> tuple[str, str]:
        """Choose a user-facing synthesis template by primary intent."""
        intent = req.intent
        if intent in (
            IntentCategory.COPY_GENERATION,
            IntentCategory.COPY_REWRITE,
            IntentCategory.STYLE_ADAPTATION,
        ):
            extra_rules = ""

            if IntentCategory.COMPLIANCE_CHECK in req.secondary_intents:
                extra_rules += """
            4. 合规提醒
               - 只总结关键风险点。
               - 不输出完整合规审核报告。
               - 不要将低风险表达直接判定为禁止发布。
            """

            if IntentCategory.STYLE_ADAPTATION in req.secondary_intents:
                extra_rules += """
            5. 风格适配说明
               - 简短体现已符合目标渠道风格。
               - 不解释内部处理过程。
               
            6. 渠道格式要求
               - 如果目标渠道为小红书：
                 - 推荐文案必须包含标题、正文、结尾互动。
                 - 正文结束后增加3-8个相关话题标签。
                 - 标签格式使用 #xxx。
               - 如果目标渠道为微信公众号：
                 - 不强制添加话题标签。
            """

            return (
                "copy_task",
                f"""结构:

            【推荐文案】
               - 直接给最终可用文案。
               - 如果是短信/Push，尽量只给 1 条主推版本，可附 1 条备选。

            【简短优化建议】
               - 最多 2-3 条。
               - 只检查最终输出文案中实际出现的信息。
               - 不得因为知识库存在某个历史活动、优惠、规则，就主动添加风险提醒。
               - 如果文案未提及某活动，则不要提示该活动。
               - 只有文案中出现价格、优惠、活动、日期、门店范围等信息时，
                 才检查这些信息是否需要确认。

            【结论】
               - 用“可发布 / 需补充 / 不建议发布”之一。

            {extra_rules}

            不要输出长篇表格；
            不要展示 Agent、模型、多模块协作等内部信息；
            不要展开完整合规审计报告，除非用户明确要求检查合规。
            """,
            )
        if intent == IntentCategory.COMPLIANCE_CHECK:
            return (
                "compliance_check",
                """结构:
1. 审核结果
   - 给出风险等级: 低 / 中 / 高。
   - 给出是否可发布。
2. 问题列表
   - 列出命中的违规词、夸大表达或遗漏风险。
   - 没有风险时明确说明未发现明显风险。
3. 修改建议
   - 给替代表达或局部改写建议。
不要主动生成完整营销文案，除非用户明确要求。""",
            )
        if intent == IntentCategory.RULE_CHECK:
            return (
                "rule_check",
                """结构:
1. 检查清单
   - 活动时间、门槛、优惠金额、适用范围、叠加限制、库存/名额、退款规则。
2. 缺失项/冲突项
   - 明确哪些信息缺失或互相冲突。
3. 建议补充
   - 给可直接放进活动页的规则补充句。
不要主动生成营销文案。""",
            )
        if intent == IntentCategory.CONTENT_REVIEW:
            return (
                "content_review",
                """结构:
1. 总体评价
   - 简短判断当前内容是否可用。
2. 评分/问题
   - 从相关性、完整性、品牌一致性、合规性、转化清晰度给结论。
3. 优化建议
   - 给可执行修改方向。
不要重写全文，除非用户要求。""",
            )
        if intent == IntentCategory.CAMPAIGN_PLANNING:
            return (
                "campaign_planning",
                """结构:
1. 活动思路
2. 内容节奏/渠道安排
3. 关键规则与风险提醒
4. 下一步执行清单
重点给方案，不要写成单条文案。""",
            )
        return (
            "general_content_ops",
            """结构:
1. 直接回答用户问题。
2. 如涉及规则或风险，补充简短注意事项。
3. 给下一步建议。""",
        )

    @staticmethod
    def _clean_product_response(text: str) -> str:
        """Remove implementation-flavored phrases from the final user-facing answer."""
        banned_fragments = [
            "融合多 Agent 输出",
            "多 Agent 输出",
            "多Agent输出",
            "Agent 输出",
            "已为您融合",
            "面向运营人员的最终回复",
            "内部标签",
            "内部推理",
            "答案来源",
            "流水线",
            "系统实现",
            "多个专家",
            "专家意见",
            "协同结果",
            "处理结果",
            "多个模块",
        ]
        lines = []
        for line in (text or "").splitlines():
            stripped = line.strip()
            if any(fragment in stripped for fragment in banned_fragments):
                continue
            lines.append(line)
        cleaned = "\n".join(lines).strip()
        while cleaned.startswith("---"):
            cleaned = cleaned[3:].strip()
        return cleaned or text

    @staticmethod
    def _fallback_synthesis(
        req: Request,
        responses: List[AgentResponse],
    ) -> str:
        by_type = {
            (response.requested_agent_type or response.agent_type): response.content
            for response in responses
        }

        parts: List[str] = []

        if AgentType.COPYWRITING in by_type or AgentType.STYLE in by_type:
            draft = by_type.get(AgentType.STYLE) or by_type.get(
                AgentType.COPYWRITING, ""
            )

            parts.append("推荐文案\n" + draft.strip())

        checks = []

        if AgentType.RULE_CHECK in by_type:
            checks.append("规则检查\n" + by_type[AgentType.RULE_CHECK].strip())

        if AgentType.COMPLIANCE in by_type:
            checks.append("合规检查\n" + by_type[AgentType.COMPLIANCE].strip())

        if AgentType.REVIEW in by_type:
            checks.append("内容评审\n" + by_type[AgentType.REVIEW].strip())

        if checks:
            parts.append("### 发布前检查\n" + "\n\n".join(checks))

        return "\n\n".join(parts) if parts else responses[0].content

    @staticmethod
    def _should_add_style_intent(req: Request) -> bool:
        # Campaign stages own style adaptation and require structured agent outputs.
        if (req.trace_context or {}).get("workflow"):
            return False
        generative_intents = {
            IntentCategory.COPY_GENERATION,
            IntentCategory.COPY_REWRITE,
            IntentCategory.CAMPAIGN_PLANNING,
        }
        if req.intent not in generative_intents:
            return False
        if IntentCategory.STYLE_ADAPTATION in req.secondary_intents:
            return False
        message = req.message or ""
        style_keywords = [
            "小红书风格",
            "公众号风格",
            "短信风格",
            "push风格",
            "风格",
            "语气",
            "调性",
            "口吻",
            "改成小红书",
            "克制一点",
            "种草",
        ]
        return any(keyword in message for keyword in style_keywords)

    def _requires_hybrid_dag(self, req: Request, agent_types: List[AgentType]) -> bool:
        """

        :param req: 当前用户请求，里面至少包含req.intent；
        :param agent_types: 本次请求路由到的Agent列表；
        :return: True:需要使用混合DAG； False：不需要特殊依赖调度，可以采用普通并行或其他执行方式

        函数本质上在判断两类依赖关系：
        1.文案生成Agent是否需要先产出内容，再交给其他Agent处理；
        2.风格Agent是否需要先产出改写结果，再交给审核Agent审核。
        """
        # 定义生成类意图
        generative_intents = {
            IntentCategory.COPY_GENERATION,
            IntentCategory.COPY_REWRITE,
            IntentCategory.CAMPAIGN_PLANNING,
        }
        # 定义审核类Agent
        review_agents = {
            AgentType.COMPLIANCE,
            AgentType.RULE_CHECK,
            AgentType.REVIEW,
        }
        # 定义依赖文案的Agent(集合并集运算符|，将审核Agent和风格Agent合并）
        copy_dependent_agents = review_agents | {AgentType.STYLE}
        # 判断“文案生成后续依赖”，三个条件必须全部满足
        copy_produces_review_target = (
            # 条件一：请求属于生成类意图
            req.intent in generative_intents
            # 条件二：路由结果里包含文案Agent
            and AgentType.COPYWRITING in agent_types
            # 条件三：还存在依赖文案的其他Agent
            and any(
                agent in copy_dependent_agents
                for agent in agent_types
                if agent != AgentType.COPYWRITING
            )
        )
        # 判断“风格改写后审核”
        style_produces_review_target = (
            # 条件一：主意图是风格适配
            req.intent == IntentCategory.STYLE_ADAPTATION
            # 条件二：实际调用了风格Agent
            and AgentType.STYLE in agent_types
            # 条件三：至少存在一个审核Agent
            and any(agent in review_agents for agent in agent_types)
        )
        return copy_produces_review_target or style_produces_review_target

    @staticmethod
    def _build_style_request(req: Request, draft: AgentResponse) -> Request:
        return Request(
            message=(
                "请基于原始需求对以下草稿做风格适配或渠道改写。只输出改写后的文案和必要提醒，不要做合规审计。\n\n"
                f"原始用户需求：\n{req.message}\n\n"
                f"待改写草稿：\n{draft.content}"
            ),
            user_id=req.user_id,
            conv_id=req.conv_id,
            task_id=req.task_id,
            context=req.context,
            history=req.history,
            intent=req.intent,
            secondary_intents=req.secondary_intents,
            urgency=req.urgency,
            trace_context=req.trace_context,
            request_id=req.request_id,
        )

    @staticmethod
    def _build_review_request(req: Request, draft: AgentResponse) -> Request:
        """
        原始生成请求+已生成草稿 --> 面向审核 Agent 的专用请求
        """
        context = (
            f"{req.context}\n\n[原始用户需求]\n{req.message}\n\n[待审核草稿]\n{draft.content}"
            if req.context
            else f"[原始用户需求]\n{req.message}\n\n[待审核草稿]\n{draft.content}"
        )
        return Request(
            message=(
                "请审核以下生成草稿。不要重新生成营销文案，只检查草稿是否符合用户原始需求、活动规则、品牌边界和合规要求。\n\n"
                "只返回符合当前审核角色契约的 JSON：decision(pass/changes_required/blocked)、summary、findings 或 risk_items(结构化风险证据数组)。"
                "使用 findings 时同时返回 need_revision(bool)；合规审核遵循 Skill 的 risk_items、missing_evidence 契约。blocked 必须包含风险证据。\n"
                f"原始用户需求：\n{req.message}\n\n"
                f"待审核草稿：\n{draft.content}"
            ),
            user_id=req.user_id,
            conv_id=req.conv_id,
            task_id=req.task_id,
            context=context,
            history=req.history,
            intent=req.intent,
            secondary_intents=req.secondary_intents,
            urgency=req.urgency,
            trace_context=req.trace_context,
            request_id=req.request_id,
        )

    @staticmethod
    def _build_revision_request(
        req: Request, draft: AgentResponse, reviews: List[AgentResponse]
    ) -> Request:
        review_text = "\n\n".join(
            f"[{response.agent_type.value}]\n{response.content}" for response in reviews
        )
        return Request(
            message=(
                "请根据审核意见修订当前草稿。\n\n"
                f"只修改审核意见明确指出的问题，保留原草稿已经形成的渠道风格、品牌语气、结构和其他正确内容。\n\n"
                f"不要因为局部问题重写整篇文案。\n\n"
                f"也不要新增未经用户输入、知识库或审核意见确认的事实信息。\n\n"
                f"只输出修订后的最终版本和必要提醒。\n\n"
                f"原始用户需求：\n{req.message}\n\n"
                f"当前待修订草稿：\n{draft.content}\n\n"
                f"审核意见：\n{review_text}"
            ),
            user_id=req.user_id,
            conv_id=req.conv_id,
            task_id=req.task_id,
            context=req.context,
            history=req.history,
            intent=req.intent,
            secondary_intents=req.secondary_intents,
            urgency=req.urgency,
            trace_context=req.trace_context,
            request_id=req.request_id,
        )

    @staticmethod
    def _review_decision_value(response: AgentResponse) -> str:
        """Normalize a reviewer response to the shared review decision semantics."""
        if not response.success:
            return "execution_failed"

        content = response.content or ""
        structured = BaseAgent._json_object(content)
        if structured:
            decision = str(structured.get("decision") or "").strip().lower()
            if decision in {"pass", "changes_required", "blocked"}:
                return decision

        positive_markers = [
            "修订决策: yes",
            "修订决策：yes",
            "need_revision: true",
            '"need_revision": true',
            "需要修订: yes",
            "需要修订：yes",
        ]

        negative_markers = [
            "修订决策: no",
            "修订决策：no",
            "need_revision: false",
            '"need_revision": false',
            "需要修订: no",
            "需要修订：no",
        ]

        blocked_terms = [
            "高风险",
            "禁止发布",
            "不建议发布",
            "不可发布",
            "严重违规",
            "人工介入",
        ]
        changes_terms = ["必须修改", "必须补充", "规则冲突", "存在冲突", "关键缺失"]
        normalized = content.lower().replace(" ", "")
        if any(term in content for term in blocked_terms):
            return "blocked"
        if any(
            marker.lower().replace(" ", "") in normalized for marker in positive_markers
        ):
            return "changes_required"
        if any(
            marker.lower().replace(" ", "") in normalized for marker in negative_markers
        ):
            return "pass"
        if any(term in content for term in changes_terms):
            return "changes_required"
        return "pass"

    @classmethod
    def _needs_revision(cls, responses: List[AgentResponse]) -> bool:
        """Only completed, non-blocked ``changes_required`` reviews may revise."""
        decisions = [cls._review_decision_value(response) for response in responses]
        return (
            "changes_required" in decisions
            and "blocked" not in decisions
            and "execution_failed" not in decisions
        )

    def _route(self, intent: Optional[Union[IntentCategory, str]]) -> AgentType:
        normalized = self._normalize_intent(intent)
        if normalized and normalized in self._INTENT_ROUTING:
            return self._INTENT_ROUTING[normalized]
        return AgentType.GENERAL

    def _collaboration_targets(self, req: Request) -> List[AgentType]:
        targets: List[AgentType] = []

        primary_agent = self._route(req.intent)
        if primary_agent:
            targets.append(primary_agent)

        for intent in self._normalize_secondary_intents(req.secondary_intents):
            mapped = self._INTENT_ROUTING.get(intent)
            if mapped is not None:
                targets.append(mapped)

        if req.intent in {
            IntentCategory.COPY_GENERATION,
            IntentCategory.COPY_REWRITE,
            IntentCategory.STYLE_ADAPTATION,
        } and not req.trace_context.get("workflow"):
            targets.extend(
                [
                    AgentType.RULE_CHECK,
                    AgentType.COMPLIANCE,
                    AgentType.REVIEW,
                ]
            )

        if not targets:
            targets.append(AgentType.GENERAL)

        deduped = list(dict.fromkeys(targets))
        return [target for target in deduped if self._pool.get(target)]

    def _best_agent(
        self, agent_type: AgentType, enforce_threshold: bool = True
    ) -> Optional[BaseAgent]:
        agents = self._pool.get(agent_type, [])
        if not agents:
            return None
        best = max(agents, key=lambda agent: agent.stats.routing_score())
        score = best.stats.routing_score()
        if (
            enforce_threshold
            and best.stats.total > 0
            and score < self.MIN_ROUTING_SCORE
        ):
            logger.warning(
                "agent %s degraded by monitor: routing_score %.3f < %.3f",
                agent_type.value,
                score,
                self.MIN_ROUTING_SCORE,
            )
            return None
        return best

    @staticmethod
    def _degraded_response(agent_type: AgentType, reason: str) -> AgentResponse:
        messages = {
            AgentType.COMPLIANCE: "合规审核节点暂不可用，不能将当前内容视为已完成合规审核。请稍后重试或交给人工复核。",
            AgentType.RULE_CHECK: "规则校验节点暂不可用，不能确认活动规则完整性。请稍后重试或交给人工复核。",
            AgentType.STYLE: "风格适配节点暂不可用，已保留原始草稿。",
            AgentType.REVIEW: "内容评审节点暂不可用，请稍后重试或交给人工复核。",
            AgentType.COPYWRITING: "文案生成节点暂不可用，请稍后重试。",
        }
        return AgentResponse(
            agent_type=agent_type,
            content=messages.get(agent_type, "当前节点暂不可用，请稍后重试。"),
            success=False,
            requested_agent_type=agent_type,
            fallback_used=False,
            degraded_reason=reason,
        )

    async def _execute(self, req: Request, agent_type: AgentType) -> AgentResponse:
        agent = self._best_agent(agent_type)
        if agent is None:
            if (
                agent_type in self.GENERAL_FALLBACK_ALLOWED
                and agent_type != AgentType.GENERAL
            ):
                fallback = self._best_agent(AgentType.GENERAL)
                if fallback:
                    response = await fallback.handle(req)
                    response.requested_agent_type = agent_type
                    response.fallback_used = True
                    response.degraded_reason = (
                        "requested_agent_degraded_general_fallback"
                    )
                    return response
            return self._degraded_response(
                agent_type, "routing_score_below_threshold_or_unavailable"
            )

        response = await agent.handle(req)
        response.requested_agent_type = agent_type
        if response.success:
            return response

        if (
            agent_type in self.GENERAL_FALLBACK_ALLOWED
            and agent_type != AgentType.GENERAL
        ):
            fallback = self._best_agent(AgentType.GENERAL)
            if fallback:
                fallback_response = await fallback.handle(req)
                fallback_response.requested_agent_type = agent_type
                fallback_response.fallback_used = True
                fallback_response.degraded_reason = (
                    f"{agent_type.value}_failed_general_fallback"
                )
                return fallback_response

        return response

    def get_stats(self) -> Dict[str, Any]:
        result = {}
        for agent_type, agents in self._pool.items():
            for index, agent in enumerate(agents):
                result[f"{agent_type.value}_{index}"] = {
                    "total": agent.stats.total,
                    "success_rate": round(agent.stats.success_rate, 3),
                    "avg_ms": round(agent.stats.avg_ms, 1),
                    "monitor_penalty": round(agent.stats.monitor_penalty, 3),
                    "routing_score": round(agent.stats.routing_score(), 3),
                    "in_flight": agent.stats.in_flight,
                }
        return result

    def update_routing_penalties(self, penalties: Dict[str, float]) -> None:
        for agent_type, agents in self._pool.items():
            for index, agent in enumerate(agents):
                key = f"{agent_type.value}_{index}"
                agent.stats.monitor_penalty = min(
                    max(float(penalties.get(key, 0.0)), 0.0), 0.9
                )
