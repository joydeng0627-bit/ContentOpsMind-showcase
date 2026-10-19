"""
ContentOpsMind end-to-end intent recognizer.

It keeps EchoMind's three-signal design: LLM few-shot understanding,
lightweight embedding similarity, and keyword pattern matching. The intent
schema is migrated from customer service to content operations.
"""

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from anthropic import AsyncAnthropic
from core.llm_gateway import shared_client, gateway_call, llm_task
from core.llm_response import extract_model_text

logger = logging.getLogger(__name__)


class IntentCategory(Enum):
    COPY_GENERATION = "copy_generation"
    COPY_REWRITE = "copy_rewrite"
    CAMPAIGN_PLANNING = "campaign_planning"
    COMPLIANCE_CHECK = "compliance_check"
    RULE_CHECK = "rule_check"
    STYLE_ADAPTATION = "style_adaptation"
    CONTENT_REVIEW = "content_review"
    KNOWLEDGE_QUERY = "knowledge_query"
    GREETING = "greeting"
    OTHER = "other"


class UrgencyLevel(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class IntentResult:
    intent: IntentCategory
    confidence: float
    urgency: UrgencyLevel
    entities: Dict[str, List[str]]
    reasoning: str
    latency_ms: float
    secondary_intents: List[IntentCategory] = field(default_factory=list)


_TEMPLATES: Dict[IntentCategory, List[str]] = {
    IntentCategory.COPY_GENERATION: [
        "帮我写一条双十一优惠券活动文案",
        "生成一版小红书种草文案",
        "写三条短信促销文案",
    ],
    IntentCategory.COPY_REWRITE: [
        "把这段文案改得更高级一点",
        "帮我压缩到 50 字以内",
        "把口吻改得更年轻化",
    ],
    IntentCategory.CAMPAIGN_PLANNING: [
        "帮我设计一个会员日活动方案",
        "双十一预热内容节奏怎么排",
        "给新品上市做一套运营计划",
    ],
    IntentCategory.COMPLIANCE_CHECK: [
        "检查这段文案有没有夸大宣传",
        "有没有违规词或绝对化用语",
        "帮我做广告合规审查",
    ],
    IntentCategory.RULE_CHECK: [
        "这段活动说明有没有漏掉优惠券限制",
        "核对活动规则是否完整",
        "检查满减门槛有没有写清楚",
    ],
    IntentCategory.STYLE_ADAPTATION: [
        "改成我们品牌的温暖专业风格",
        "适配公众号和短信两个渠道",
        "按照品牌调性润色",
    ],
    IntentCategory.CONTENT_REVIEW: [
        "帮我评审这篇推文",
        "看看这版文案哪里可以优化",
        "给这段运营内容打分并提出建议",
    ],
    IntentCategory.KNOWLEDGE_QUERY: [
        "品牌禁用词有哪些",
        "优惠券活动规则是什么",
        "短信文案字数限制是多少",
    ],
    IntentCategory.GREETING: ["你好", "hello", "在吗"],
}

_URGENCY_KEYWORDS = {
    UrgencyLevel.CRITICAL: ["马上发", "今天上线", "紧急", "urgent", "asap"],
    UrgencyLevel.HIGH: ["今天", "今晚", "尽快", "上线前", "发布前"],
    UrgencyLevel.MEDIUM: ["这周", "明天", "快点"],
}


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


class IntentRecognizer:
    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        model: str = "claude-3-5-sonnet-20241022",
        confidence_threshold: float = 0.45,
    ):
        kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = shared_client(AsyncAnthropic, **kwargs)
        self.model = model
        self.threshold = confidence_threshold
        self._embedding_enabled = not bool(base_url)
        self._tpl_embeddings: Dict[IntentCategory, List[List[float]]] = {}
        self._cache: Dict[str, IntentResult] = {}
        self.cache_hits = 0
        self.cache_misses = 0

    @llm_task()
    async def recognize(
        self, message: str, history: Optional[List[Dict[str, str]]] = None
    ) -> IntentResult:
        key = self._cache_key(message)
        if key in self._cache:
            self.cache_hits += 1
            return self._cache[key]
        self.cache_misses += 1
        t0 = time.monotonic()

        llm_task = asyncio.create_task(self._llm_recognize(message, history))
        emb_task = (
            asyncio.create_task(self._embedding_recognize(message))
            if self._embedding_enabled
            else None
        )
        pat = self._pattern_recognize(message)
        if emb_task:
            llm, emb = await asyncio.gather(llm_task, emb_task)
        else:
            llm = await llm_task
            emb = {"intent": IntentCategory.OTHER, "confidence": 0.0}

        intent, secondary = self._vote(llm, emb, pat)
        secondary = self._filter_secondary_intents(message, intent, secondary)
        result = IntentResult(
            intent=intent,
            confidence=float(llm.get("confidence", 0.0)),
            urgency=self._urgency(message, intent),
            entities=await self._extract_entities(message),
            reasoning=str(llm.get("reasoning", "")),
            latency_ms=(time.monotonic() - t0) * 1000,
            secondary_intents=secondary,
        )
        if len(self._cache) >= 1000:
            for k in list(self._cache)[:500]:
                del self._cache[k]
        self._cache[key] = result
        return result

    def learn(self, message: str, correct: IntentCategory) -> None:
        values = _TEMPLATES.setdefault(correct, [])
        if message not in values:
            values.append(message)
            self._tpl_embeddings.pop(correct, None)

    async def _llm_recognize(
        self, message: str, history: Optional[List[Dict[str, str]]]
    ) -> Dict[str, Any]:
        message = self._clean_text(message)
        examples = "\n".join(
            f'消息: "{tpls[0]}" -> 意图: {cat.value}'
            for cat, tpls in _TEMPLATES.items()
            if tpls
        )
        ctx = ""
        if history:
            ctx = "\n最近对话:\n" + "\n".join(
                f"{self._clean_text(m.get('role', 'user'))}: {self._clean_text(m.get('content', ''))}"
                for m in history[-4:]
            )
        prompt = f"""你是内容运营助手的意图识别器。请判断用户最主要的内容运营任务，并返回 JSON。

可选意图: {', '.join(item.value for item in IntentCategory)}

示例:
{examples}
{ctx}
用户消息: "{message}"

返回格式，仅 JSON:
{{"intent":"<主意图>","secondary_intents":["<次意图>"],"confidence":0.0,"reasoning":"一句话说明"}}

规则:
- 生成新文案优先 copy_generation。
- 对已有文案改写、缩短、换语气优先 copy_rewrite。
- 活动节奏、栏目规划、渠道排期优先 campaign_planning。
- 违规词、极限词、夸大宣传、广告法、平台审核优先 compliance_check。
- 优惠券门槛、适用范围、库存、时间、限制条件核对优先 rule_check。
- 品牌调性、渠道格式、小红书/公众号/短信/Push 适配优先 style_adaptation。
- 评审已有内容并给优化建议优先 content_review。
- 查询规则、规范、禁用词、历史模板优先 knowledge_query。
- 复合任务要在 secondary_intents 中保留需要协作的意图，例如生成文案并检查合规。"""
        try:
            resp = await gateway_call(
                self.client,
                model=self.model,
                max_tokens=256,
                temperature=0.1,
                messages=[{"role": "user", "content": self._clean_text(prompt)}],
            )
            raw = extract_model_text(resp)
            s, e = raw.find("{"), raw.rfind("}") + 1
            data = json.loads(raw[s:e])
            data["intent"] = self._parse_intent(data.get("intent"))
            data["secondary_intents"] = self._parse_intent_list(
                data.get("secondary_intents", [])
            )
            return data
        except Exception as ex:
            logger.warning("LLM intent recognition failed: %s", ex)
            return {
                "intent": IntentCategory.OTHER,
                "secondary_intents": [],
                "confidence": 0.0,
                "reasoning": "LLM failed",
                "failed": True,
            }

    async def _embedding_recognize(self, message: str) -> Dict[str, Any]:
        await self._load_template_embeddings()
        msg_vec = await self._embed_text(message)
        best, best_score = IntentCategory.OTHER, 0.0
        for cat, vecs in self._tpl_embeddings.items():
            score = max(_cosine(msg_vec, vec) for vec in vecs)
            if score > best_score:
                best, best_score = cat, score
        return {"intent": best, "confidence": best_score}

    def _pattern_recognize(self, message: str) -> Dict[str, Any]:
        msg = message.lower()
        patterns = {
            IntentCategory.COMPLIANCE_CHECK: [
                "合规",
                "违规",
                "广告法",
                "极限词",
                "禁用词",
                "夸大",
                "审核",
                "检查",
                "风险",
            ],
            IntentCategory.RULE_CHECK: [
                "规则",
                "门槛",
                "限制",
                "优惠券",
                "满减",
                "库存",
                "有效期",
            ],
            IntentCategory.COPY_GENERATION: [
                "写",
                "生成",
                "产出",
                "文案",
                "标题",
                "短信",
                "push",
            ],
            IntentCategory.COPY_REWRITE: [
                "改写",
                "润色",
                "缩短",
                "扩写",
                "优化这段",
                "换个说法",
            ],
            IntentCategory.CAMPAIGN_PLANNING: [
                "活动方案",
                "排期",
                "节奏",
                "运营计划",
                "预热",
                "复盘",
            ],
            IntentCategory.STYLE_ADAPTATION: [
                "风格",
                "调性",
                "小红书",
                "公众号",
                "短信",
                "渠道",
                "语气",
            ],
            IntentCategory.CONTENT_REVIEW: [
                "评审",
                "打分",
                "看看",
                "哪里可以优化",
                "review",
            ],
            IntentCategory.KNOWLEDGE_QUERY: [
                "有哪些",
                "是什么",
                "规范",
                "模板",
                "案例",
            ],
            IntentCategory.GREETING: ["你好", "hello", "hi", "在吗"],
        }
        best, best_score = IntentCategory.OTHER, 0.0
        for cat, kws in patterns.items():
            hits = sum(1 for kw in kws if kw in msg)
            if hits:
                score = hits / len(kws)
                if score > best_score:
                    best, best_score = cat, score
        return {"intent": best, "confidence": best_score}

    def _vote(
        self, llm: Dict[str, Any], emb: Dict[str, Any], pat: Dict[str, Any]
    ) -> Tuple[IntentCategory, List[IntentCategory]]:
        if llm.get("failed"):
            for source in (pat, emb):
                if (
                    source.get("intent") != IntentCategory.OTHER
                    and source.get("confidence", 0) > 0
                ):
                    return source["intent"], []
            return IntentCategory.OTHER, []
        weights = (
            [(llm, 0.7), (emb, 0.2), (pat, 0.1)]
            if self._embedding_enabled
            else [(llm, 0.85), (pat, 0.15)]
        )
        scores: Dict[IntentCategory, float] = {}
        for source, weight in weights:
            cat = source.get("intent", IntentCategory.OTHER)
            conf = float(source.get("confidence", 0.0) or 0.0)
            scores[cat] = scores.get(cat, 0.0) + weight * conf
        best = max(scores, key=scores.get)
        if scores[best] < self.threshold:
            best = (
                pat.get("intent", IntentCategory.OTHER)
                if pat.get("confidence", 0) > 0
                else IntentCategory.OTHER
            )
        secondary: List[IntentCategory] = []
        for cat in self._parse_intent_list(llm.get("secondary_intents", [])):
            if cat not in (best, IntentCategory.OTHER):
                secondary.append(cat)
        for source in (pat, emb):
            cat = source.get("intent", IntentCategory.OTHER)
            if (
                cat not in (best, IntentCategory.OTHER)
                and source.get("confidence", 0.0) >= 0.5
            ):
                secondary.append(cat)
        return best, list(dict.fromkeys(secondary))

    def _filter_secondary_intents(
        self,
        message: str,
        primary: IntentCategory,
        secondary: List[IntentCategory],
    ) -> List[IntentCategory]:
        """Keep secondary intents explicit enough to avoid over-routing."""
        msg = message.lower()

        def has_any(words: List[str]) -> bool:
            return any(word in msg for word in words)

        rule_trigger = has_any(
            [
                "检查规则",
                "核对规则",
                "规则是否",
                "有没有遗漏",
                "是否完整",
                "缺失",
                "冲突",
                "校验",
            ]
        ) or (
            has_any(["检查", "核对", "校验"])
            and has_any(
                [
                    "规则",
                    "门槛",
                    "限制",
                    "活动页",
                    "优惠券",
                    "满减",
                    "叠加",
                    "库存",
                    "有效期",
                ]
            )
        )
        compliance_trigger = has_any(
            [
                "合规",
                "违规",
                "广告法",
                "极限词",
                "禁用词",
                "夸大",
                "风险",
            ]
        ) or (has_any(["审核", "检查"]) and has_any(["广告法", "违规", "风险", "极限词", "禁用词"]))
        review_trigger = has_any(
            ["评审", "打分", "优化建议", "哪里可以优化", "整体评价"]
        )
        style_trigger = has_any(
            ["风格", "调性", "小红书", "公众号", "渠道", "品牌", "适配", "改成", "语气"]
        )

        filtered: List[IntentCategory] = []
        for intent in secondary:
            if intent in (primary, IntentCategory.OTHER):
                continue
            if intent == IntentCategory.RULE_CHECK and not rule_trigger:
                continue
            if intent == IntentCategory.COMPLIANCE_CHECK and not compliance_trigger:
                continue
            if intent == IntentCategory.CONTENT_REVIEW:
                if primary == IntentCategory.COMPLIANCE_CHECK and compliance_trigger:
                    continue
                if not review_trigger:
                    continue
            if intent == IntentCategory.STYLE_ADAPTATION and not style_trigger:
                continue
            if intent in (
                IntentCategory.COPY_GENERATION,
                IntentCategory.COPY_REWRITE,
                IntentCategory.CAMPAIGN_PLANNING,
            ) and primary in (
                IntentCategory.COMPLIANCE_CHECK,
                IntentCategory.RULE_CHECK,
                IntentCategory.CONTENT_REVIEW,
            ):
                continue
            filtered.append(intent)
        return list(dict.fromkeys(filtered))

    async def _extract_entities(self, message: str) -> Dict[str, List[str]]:
        prompt = f"""从内容运营需求中抽取实体，返回 JSON，字段值均为数组。
消息: "{self._clean_text(message)}"
格式: {{"campaign":[],"product":[],"channel":[],"audience":[],"discount":[],"deadline":[],"risk_terms":[]}}"""
        try:
            resp = await gateway_call(
                self.client,
                model=self.model,
                max_tokens=256,
                temperature=0.0,
                messages=[{"role": "user", "content": self._clean_text(prompt)}],
            )
            raw = extract_model_text(resp)
            s, e = raw.find("{"), raw.rfind("}") + 1
            return json.loads(raw[s:e])
        except Exception:
            return {
                "campaign": [],
                "product": [],
                "channel": [],
                "audience": [],
                "discount": [],
                "deadline": [],
                "risk_terms": [],
            }

    async def _load_template_embeddings(self) -> None:
        missing = [cat for cat in _TEMPLATES if cat not in self._tpl_embeddings]
        for cat in missing:
            self._tpl_embeddings[cat] = [
                await self._embed_text(text) for text in _TEMPLATES[cat]
            ]

    async def _embed_text(self, text: str) -> List[float]:
        return self._local_embedding(text)

    @staticmethod
    def _local_embedding(text: str, dims: int = 256) -> List[float]:
        normalized = text.lower().strip()
        vec = [0.0] * dims
        tokens = set()
        for n in (1, 2, 3):
            if len(normalized) >= n:
                tokens.update(
                    normalized[i : i + n] for i in range(len(normalized) - n + 1)
                )
        if not tokens:
            tokens.add(normalized)
        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % dims
            vec[idx] += 1.0 if digest[4] % 2 == 0 else -1.0
        return vec

    def _urgency(self, message: str, intent: IntentCategory) -> UrgencyLevel:
        msg = message.lower()
        for level, kws in _URGENCY_KEYWORDS.items():
            if any(kw in msg for kw in kws):
                return level
        if intent in (IntentCategory.COMPLIANCE_CHECK, IntentCategory.RULE_CHECK):
            return UrgencyLevel.MEDIUM
        return UrgencyLevel.LOW

    @staticmethod
    def _parse_intent(value: Any) -> IntentCategory:
        try:
            return (
                value
                if isinstance(value, IntentCategory)
                else IntentCategory(str(value))
            )
        except ValueError:
            return IntentCategory.OTHER

    @classmethod
    def _parse_intent_list(cls, values: Any) -> List[IntentCategory]:
        if not isinstance(values, list):
            values = [values] if values else []
        return [
            cat
            for cat in (cls._parse_intent(v) for v in values)
            if cat != IntentCategory.OTHER
        ]

    def _cache_key(self, message: str) -> str:
        return self._clean_text(message)[:200]

    @staticmethod
    def _clean_text(value: Any) -> str:
        if value is None:
            return ""
        if not isinstance(value, str):
            value = str(value)
        return value.encode("utf-8", errors="ignore").decode("utf-8")

    @property
    def cache_stats(self) -> Dict[str, Any]:
        total = self.cache_hits + self.cache_misses
        return {
            "size": len(self._cache),
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "hit_rate": self.cache_hits / total if total else 0.0,
        }
