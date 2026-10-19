"""OpenAI-compatible tool calling layer for ContentOpsMind agents.

The project keeps tool definitions in the OpenAI compatible
``type=function`` shape, then converts them to Anthropic Messages tools at
runtime. Business tools are deterministic local functions so the full
selection -> validation -> dispatch -> result injection loop can be tested
without external business systems.
"""

from __future__ import annotations

import asyncio
import json
import pathlib
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from anthropic import AsyncAnthropic
from core.llm_gateway import gateway_call, llm_task
from core.llm_response import extract_model_text


@dataclass
class ToolExecution:
    agent: str
    tool: str
    arguments: Dict[str, Any]
    success: bool
    latency_ms: float
    result_preview: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    cached: bool = False
    validation_error: Optional[str] = None

    def to_trace(self) -> Dict[str, Any]:
        payload = {
            "agent": self.agent,
            "tool": self.tool,
            "arguments": _summarize(self.arguments),
            "success": self.success,
            "latency_ms": round(self.latency_ms, 1),
            "result_preview": _summarize(self.result_preview),
            "cached": self.cached,
        }
        if self.error:
            payload["error"] = self.error
        if self.validation_error:
            payload["validation_error"] = self.validation_error
        return payload


@dataclass
class ToolCallingResult:
    final_text: str
    tool_calls: List[ToolExecution] = field(default_factory=list)
    used: bool = False
    rounds: int = 0
    fallback_reason: Optional[str] = None
    model_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


class BusinessToolRegistry:
    """Deterministic mock business tools with OpenAI-compatible schemas."""

    def __init__(self, data_path: Optional[str] = None):
        root = pathlib.Path(__file__).parent.parent
        self._data_path = (
            pathlib.Path(data_path)
            if data_path
            else root / "data" / "tools" / "content_ops_tools.json"
        )
        self._data = self._load_data()
        self._schemas = self._build_schemas()

    def _load_data(self) -> Dict[str, Any]:
        try:
            return json.loads(self._data_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {
                "campaign_rules": {},
                "coupon_policies": {},
                "products": {},
                "banned_words": [],
                "channel_limits": {},
            }

    def _build_schemas(self) -> Dict[str, Dict[str, Any]]:
        return {
            "get_campaign_rule": {
                "type": "function",
                "function": {
                    "name": "get_campaign_rule",
                    "description": "查询内容运营活动规则，包括门槛、优惠、有效期、适用范围、叠加限制和退款说明。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "campaign_id": {
                                "type": "string",
                                "description": "活动ID，例如 double11_399_80 或 member_day_199_30。未知时可根据用户上下文选择最匹配活动。",
                            }
                        },
                        "required": ["campaign_id"],
                    },
                },
            },
            "get_coupon_policy": {
                "type": "function",
                "function": {
                    "name": "get_coupon_policy",
                    "description": "查询优惠券策略，包括门槛、面额、有效期、适用类目、排除标签和状态。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "coupon_id": {
                                "type": "string",
                                "description": "优惠券ID，例如 CP39980 或 CP19930。",
                            }
                        },
                        "required": ["coupon_id"],
                    },
                },
            },
            "query_product_info": {
                "type": "function",
                "function": {
                    "name": "query_product_info",
                    "description": "查询商品结构化信息，包括名称、价格、库存、卖点、限制性功效词和适用人群。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "product_id": {
                                "type": "string",
                                "description": "商品ID，例如 P_SKIN_SERUM_01 或 P_HOME_CLEAN_02。",
                            }
                        },
                        "required": ["product_id"],
                    },
                },
            },
            "check_banned_words": {
                "type": "function",
                "function": {
                    "name": "check_banned_words",
                    "description": "检查文案是否命中广告合规禁用词或高风险表达，并返回替代表达建议。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "需要检查的中文营销文案。",
                            }
                        },
                        "required": ["text"],
                    },
                },
            },
            "estimate_message_length": {
                "type": "function",
                "function": {
                    "name": "estimate_message_length",
                    "description": "估算短信、Push、小红书、公众号或活动页文案长度是否超过渠道限制。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "需要估算长度的文案。",
                            },
                            "channel": {
                                "type": "string",
                                "description": "渠道名称，如 短信、Push、小红书、公众号、活动页。",
                            },
                        },
                        "required": ["text", "channel"],
                    },
                },
            },
        }

    def openai_tools(
        self, allowed_tools: Optional[Iterable[str]] = None
    ) -> List[Dict[str, Any]]:
        allowed = set(allowed_tools or self._schemas.keys())
        return [schema for name, schema in self._schemas.items() if name in allowed]

    def anthropic_tools(
        self, allowed_tools: Optional[Iterable[str]] = None
    ) -> List[Dict[str, Any]]:
        tools = []
        for schema in self.openai_tools(allowed_tools):
            fn = schema["function"]
            tools.append(
                {
                    "name": fn["name"],
                    "description": fn.get("description", ""),
                    "input_schema": fn.get(
                        "parameters", {"type": "object", "properties": {}}
                    ),
                }
            )
        return tools

    def validate(
        self, name: str, arguments: Dict[str, Any], allowed_tools: Sequence[str]
    ) -> Optional[str]:
        if name not in allowed_tools:
            return f"tool_not_allowed:{name}"
        schema = self._schemas.get(name)
        if not schema:
            return f"unknown_tool:{name}"
        params = schema["function"].get("parameters", {})
        required = params.get("required", [])
        properties = params.get("properties", {})
        for key in required:
            if key not in arguments or arguments.get(key) in (None, ""):
                return f"missing_required_argument:{key}"
        for key, spec in properties.items():
            if key not in arguments or arguments.get(key) is None:
                continue
            if spec.get("type") == "string" and not isinstance(arguments.get(key), str):
                return f"invalid_argument_type:{key}:expected_string"
        return None

    async def execute(
        self, name: str, arguments: Dict[str, Any], timeout_s: float = 2.0
    ) -> Dict[str, Any]:
        return await asyncio.wait_for(
            asyncio.to_thread(self._execute_sync, name, arguments), timeout=timeout_s
        )

    def _execute_sync(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        if name == "get_campaign_rule":
            campaign_id = self._normalize_campaign_id(arguments.get("campaign_id"))
            item = self._data.get("campaign_rules", {}).get(campaign_id)
            if not item:
                return {
                    "success": False,
                    "error": "campaign_not_found",
                    "campaign_id": campaign_id,
                }
            return {"success": True, "data": item}
        if name == "get_coupon_policy":
            coupon_id = str(arguments.get("coupon_id", "")).strip().upper()
            item = self._data.get("coupon_policies", {}).get(coupon_id)
            if not item:
                return {
                    "success": False,
                    "error": "coupon_not_found",
                    "coupon_id": coupon_id,
                }
            return {"success": True, "data": item}
        if name == "query_product_info":
            product_id = str(arguments.get("product_id", "")).strip().upper()
            item = self._data.get("products", {}).get(product_id)
            if not item:
                return {
                    "success": False,
                    "error": "product_not_found",
                    "product_id": product_id,
                }
            return {"success": True, "data": item}
        if name == "check_banned_words":
            text = str(arguments.get("text", ""))
            hits = []
            for rule in self._data.get("banned_words", []):
                word = str(rule.get("word", ""))
                if word and word in text:
                    hits.append(rule)
            risk_order = {"low": 1, "medium": 2, "high": 3}
            risk_level = "low"
            if hits:
                risk_level = max(
                    (hit.get("risk_level", "low") for hit in hits),
                    key=lambda value: risk_order.get(value, 0),
                )
            return {
                "success": True,
                "data": {
                    "risk_level": risk_level,
                    "hit_count": len(hits),
                    "hits": hits,
                },
            }
        if name == "estimate_message_length":
            text = str(arguments.get("text", ""))
            channel = str(arguments.get("channel", "短信"))
            limit = self._data.get("channel_limits", {}).get(channel) or self._data.get(
                "channel_limits", {}
            ).get("短信", {})
            max_chars = int(limit.get("max_chars", 70))
            link_chars = int(limit.get("link_placeholder_chars", 0))
            effective_length = len(text) + link_chars
            return {
                "success": True,
                "data": {
                    "channel": channel,
                    "text_length": len(text),
                    "effective_length": effective_length,
                    "max_chars": max_chars,
                    "over_limit": effective_length > max_chars,
                    "remaining_chars": max_chars - effective_length,
                },
            }
        return {"success": False, "error": "unknown_tool", "tool": name}

    @staticmethod
    def _normalize_campaign_id(value: Any) -> str:
        raw = str(value or "").strip().lower()
        aliases = {
            "double11": "double11_399_80",
            "双十一": "double11_399_80",
            "双11": "double11_399_80",
            "399": "double11_399_80",
            "member_day": "member_day_199_30",
            "会员日": "member_day_199_30",
            "199": "member_day_199_30",
        }
        if raw in aliases:
            return aliases[raw]
        for key, target in aliases.items():
            if key in raw:
                return target
        return raw or "double11_399_80"


class ToolCallingManager:
    """Runs bounded tool-calling loops and returns final answer plus trace."""

    def __init__(
        self,
        client: AsyncAnthropic,
        model: str,
        registry: BusinessToolRegistry,
        max_rounds: int = 3,
        timeout_s: float = 2.0,
    ):
        self._client = client
        self._model = model
        self._registry = registry
        self._max_rounds = max(1, max_rounds)
        self._timeout_s = timeout_s

    @llm_task()
    async def run(
        self,
        *,
        agent: str,
        system: str,
        messages: List[Dict[str, Any]],
        allowed_tools: Sequence[str],
        max_tokens: int = 1200,
        temperature: float = 0.25,
    ) -> ToolCallingResult:
        if not allowed_tools:
            return ToolCallingResult(final_text="", fallback_reason="no_allowed_tools")

        tools = self._registry.anthropic_tools(allowed_tools)
        tool_calls: List[ToolExecution] = []
        model_calls = input_tokens = output_tokens = 0
        seen: Dict[Tuple[str, str], Dict[str, Any]] = {}
        working_messages: List[Dict[str, Any]] = list(messages)

        for round_index in range(1, self._max_rounds + 1):
            resp = await gateway_call(
                self._client,
                model=self._model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                messages=working_messages,
                tools=tools,
            )
            model_calls += 1
            usage = getattr(resp, "usage", None)
            input_tokens += int(getattr(usage, "input_tokens", 0))
            output_tokens += int(getattr(usage, "output_tokens", 0))
            blocks = [_block_to_dict(block) for block in resp.content]
            uses = [block for block in blocks if block.get("type") == "tool_use"]
            if not uses:
                return ToolCallingResult(
                    final_text=_text_from_blocks(blocks),
                    tool_calls=tool_calls,
                    used=bool(tool_calls),
                    rounds=round_index,
                    model_calls=model_calls,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )

            working_messages.append({"role": "assistant", "content": blocks})
            result_blocks = []
            for use in uses:
                name = str(use.get("name", ""))
                args = use.get("input") or {}
                if not isinstance(args, dict):
                    args = {}
                started = time.monotonic()
                validation_error = self._registry.validate(name, args, allowed_tools)
                key = (name, json.dumps(args, ensure_ascii=False, sort_keys=True))
                if validation_error:
                    result = {"success": False, "error": validation_error}
                    execution = ToolExecution(
                        agent=agent,
                        tool=name,
                        arguments=args,
                        success=False,
                        latency_ms=(time.monotonic() - started) * 1000,
                        result_preview=result,
                        validation_error=validation_error,
                    )
                elif key in seen:
                    result = seen[key]
                    execution = ToolExecution(
                        agent=agent,
                        tool=name,
                        arguments=args,
                        success=bool(result.get("success")),
                        latency_ms=(time.monotonic() - started) * 1000,
                        result_preview=result,
                        cached=True,
                    )
                else:
                    try:
                        result = await self._registry.execute(
                            name, args, timeout_s=self._timeout_s
                        )
                        seen[key] = result
                        execution = ToolExecution(
                            agent=agent,
                            tool=name,
                            arguments=args,
                            success=bool(result.get("success")),
                            latency_ms=(time.monotonic() - started) * 1000,
                            result_preview=result,
                        )
                    except asyncio.TimeoutError:
                        result = {"success": False, "error": "tool_timeout"}
                        execution = ToolExecution(
                            agent=agent,
                            tool=name,
                            arguments=args,
                            success=False,
                            latency_ms=(time.monotonic() - started) * 1000,
                            result_preview=result,
                            error="tool_timeout",
                        )
                    except Exception as ex:
                        result = {"success": False, "error": str(ex)}
                        execution = ToolExecution(
                            agent=agent,
                            tool=name,
                            arguments=args,
                            success=False,
                            latency_ms=(time.monotonic() - started) * 1000,
                            result_preview=result,
                            error=str(ex),
                        )
                tool_calls.append(execution)
                result_blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": use.get("id"),
                        "content": json.dumps(result, ensure_ascii=False),
                        "is_error": not execution.success,
                    }
                )
            working_messages.append({"role": "user", "content": result_blocks})

        final_resp = await gateway_call(
            self._client,
            model=self._model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=working_messages
            + [
                {
                    "role": "user",
                    "content": "工具调用轮次已达到上限。请基于已经获得的工具结果给出最终答复；如果信息不足，请明确说明需要人工确认。",
                }
            ],
        )
        final_blocks = [_block_to_dict(block) for block in final_resp.content]
        model_calls += 1
        usage = getattr(final_resp, "usage", None)
        input_tokens += int(getattr(usage, "input_tokens", 0))
        output_tokens += int(getattr(usage, "output_tokens", 0))
        return ToolCallingResult(
            final_text=_text_from_blocks(final_blocks),
            tool_calls=tool_calls,
            used=bool(tool_calls),
            rounds=self._max_rounds,
            fallback_reason="max_tool_rounds_reached",
            model_calls=model_calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )


def _block_to_dict(block: Any) -> Dict[str, Any]:
    if isinstance(block, dict):
        return block
    if hasattr(block, "model_dump"):
        return block.model_dump(exclude_none=True)
    if hasattr(block, "dict"):
        return block.dict(exclude_none=True)
    payload = {"type": getattr(block, "type", "text")}
    if hasattr(block, "text"):
        payload["text"] = getattr(block, "text")
    if hasattr(block, "name"):
        payload["name"] = getattr(block, "name")
    if hasattr(block, "id"):
        payload["id"] = getattr(block, "id")
    if hasattr(block, "input"):
        payload["input"] = getattr(block, "input")
    return payload


def _text_from_blocks(blocks: Sequence[Dict[str, Any]]) -> str:
    return extract_model_text(blocks)


def _summarize(value: Any, max_text: int = 240) -> Any:
    if isinstance(value, dict):
        result: Dict[str, Any] = {}
        for key, item in list(value.items())[:8]:
            result[key] = _summarize(item, max_text=max_text)
        return result
    if isinstance(value, list):
        return [_summarize(item, max_text=max_text) for item in value[:5]]
    if isinstance(value, str):
        return value if len(value) <= max_text else value[:max_text] + "..."
    return value
