"""Deterministic task-scope guard for conversation memory.

The guard deliberately prefers asking for missing input over inheriting facts
from an unrelated task.  It runs before RAG and agent routing.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set
import re

REFERENCE_MARKERS = (
    "这条",
    "这段",
    "这个文案",
    "上面那条",
    "上面那版",
    "上面的",
    "刚才的",
    "上一版",
    "上一条",
    "第二版",
    "第一版",
    "把它",
    "对它",
    "继续改",
    "继续优化",
    "基于刚才",
    "按刚才",
    "再改一下",
)

ARTIFACT_REFERENCE_MARKERS = (
    "生成的文案",
    "当前文案",
    "这版",
    "上一版",
    "刚才那版",
    "刚才的文案",
    "这段",
    "把它",
    "再明显一点",
    "再口语一点",
    "还是不够",
    "继续修改",
    "继续优化",
    "再改",
    "不够明显",
    "不够小红书",
    "不够克制",
)

SENSITIVE_TASK_FACTS = (
    "双十一",
    "双11",
    "满399减80",
    "今晚24点",
    "优惠券",
    "自营正价商品",
    "活动落地页",
    "短信渠道",
    "主推文案（短信）",
    "备选文案（短信）",
)

REWRITE_MARKERS = (
    "改写",
    "改成",
    "改为",
    "改得",
    "润色",
    "缩短",
    "压缩",
    "扩写",
    "换个说法",
    "换成",
    "调整为",
    "优化这段",
    "优化这条",
)

NEW_TASK_MARKERS = (
    "帮我写",
    "生成一条",
    "生成一篇",
    "新任务",
    "重新开始",
    "另外一个任务",
    "检查",
    "查询",
    "告诉我",
    "怎么",
    "为什么",
)

TOPIC_GROUPS: Dict[str, tuple[str, ...]] = {
    "double_eleven": ("双十一", "双11", "11.11"),
    "new_product": ("新品", "新品上市", "新产品", "首发"),
    "coupon_promotion": ("优惠券", "满减", "满399减80", "促销", "折扣"),
    "member_day": ("会员日", "会员活动"),
    "618": ("618", "六一八"),
}


@dataclass(frozen=True)
class ScopeDecision:
    inherit_previous_task: bool
    needs_source_text: bool
    explicit_reference: bool
    topic_compatible: bool
    current_topics: Set[str] = field(default_factory=set)
    previous_topics: Set[str] = field(default_factory=set)
    reason: str = ""


def extract_topics(text: str) -> Set[str]:
    clean = str(text or "").lower()
    return {
        topic
        for topic, keywords in TOPIC_GROUPS.items()
        if any(keyword.lower() in clean for keyword in keywords)
    }


def has_explicit_reference(message: str) -> bool:
    text = str(message or "")
    return any(marker in text for marker in REFERENCE_MARKERS)


def is_rewrite_request(message: str) -> bool:
    text = str(message or "")
    return any(marker in text for marker in REWRITE_MARKERS)


def infer_channel(message: str) -> str:
    text = str(message or "")
    for marker, channel in (
        ("小红书", "小红书"),
        ("公众号", "公众号"),
        ("短信", "短信"),
        ("Push", "Push"),
        ("push", "Push"),
    ):
        if marker in text:
            return channel
    return ""


def infer_tone(message: str) -> str:
    text = str(message or "")
    for marker, tone in (
        ("种草", "种草"),
        ("安利", "种草"),
        ("克制", "克制"),
        ("专业", "专业"),
        ("温暖", "温暖"),
        ("年轻", "年轻"),
        ("活泼", "活泼"),
        ("简洁", "简洁"),
        ("口语", "口语化"),
    ):
        if marker in text:
            return tone
    return ""


def looks_like_source_text(message: str) -> bool:
    """Conservatively decide whether a pending source_text slot is filled."""
    text = str(message or "").strip()
    if has_source_text(text):
        return True
    if len(text) < 15 or is_rewrite_request(text):
        return False
    if any(text.startswith(marker) for marker in NEW_TASK_MARKERS):
        return False
    # Copy normally contains a concrete noun plus sentence/list punctuation.
    return bool(re.search(r"[，。！？；、,.!?]", text))


def refers_to_active_artifact(message: str) -> bool:
    text = str(message or "")
    return any(marker in text for marker in ARTIFACT_REFERENCE_MARKERS)


def unsupported_task_facts(output: str, allowed_sources: Iterable[str]) -> List[str]:
    """Find high-risk temporary facts absent from every authoritative source."""
    result = str(output or "")
    allowed = "\n".join(str(item or "") for item in allowed_sources)
    return [
        fact for fact in SENSITIVE_TASK_FACTS if fact in result and fact not in allowed
    ]


def has_source_text(message: str) -> bool:
    """Return true only when the current request visibly contains source copy."""
    lines = [line.strip() for line in str(message or "").splitlines() if line.strip()]
    config_prefixes = (
        "任务类型：",
        "任务类型:",
        "渠道：",
        "渠道:",
        "目标风格：",
        "目标风格:",
    )
    body_lines = [line for line in lines if not line.startswith(config_prefixes)]
    text = "\n".join(body_lines).strip()
    # Only an explicitly labelled source counts. Generic configuration colons
    # must never satisfy rewrite input validation.
    colon = re.search(
        r"(?:原文|待改写文案|待改写内容)\s*[：:]\s*(.+)$", text, flags=re.S
    )
    if colon and len(colon.group(1).strip()) >= 8:
        return True
    # Short copy enclosed by quotes is still an explicit source when the user
    # uses the common "把‘…’改得…" rewrite form.
    inline_rewrite = re.search(
        r"(?:把|将)\s*[‘'“\"]([^’'”\"]{2,})[’'”\"]\s*(?:改|写|润色|压缩|扩写)",
        text,
        flags=re.S,
    )
    if inline_rewrite:
        return True
    # A substantial quoted passage.
    quoted = re.search(r"[“\"]([^”\"]{8,})[”\"]", text, flags=re.S)
    if quoted:
        return True
    # An instruction line followed by a separate source paragraph.
    body_lines = [line.strip() for line in text.splitlines() if line.strip()]
    return len(body_lines) >= 2 and len("".join(body_lines[1:])) >= 8


def topics_compatible(current: Set[str], previous: Set[str]) -> bool:
    # A pronoun-only follow-up has no independent topic and may inherit.
    if not current:
        return True
    return bool(previous and current.intersection(previous))


def decide_context_scope(
    message: str,
    recent_messages: Optional[Iterable[Dict[str, Any]]] = None,
) -> ScopeDecision:
    recent = list(recent_messages or [])
    previous_user = next(
        (
            str(item.get("content", ""))
            for item in reversed(recent)
            if item.get("role") == "user"
        ),
        "",
    )
    current_topics = extract_topics(message)
    previous_topics = extract_topics(previous_user)
    explicit_reference = has_explicit_reference(message)
    compatible = topics_compatible(current_topics, previous_topics)
    inherit = bool(previous_user and explicit_reference and compatible)
    needs_source = (
        is_rewrite_request(message) and not has_source_text(message) and not inherit
    )

    if needs_source and explicit_reference and not compatible:
        reason = "explicit_reference_but_topic_conflict"
    elif needs_source:
        reason = "rewrite_source_missing"
    elif inherit:
        reason = "explicit_reference_and_topic_compatible"
    else:
        reason = "new_task_by_default"

    return ScopeDecision(
        inherit_previous_task=inherit,
        needs_source_text=needs_source,
        explicit_reference=explicit_reference,
        topic_compatible=compatible,
        current_topics=current_topics,
        previous_topics=previous_topics,
        reason=reason,
    )


def preference_context(profile: Dict[str, Any], channel: str = "") -> str:
    """Expose only explicitly authorized v2 preferences relevant to this channel."""
    if not isinstance(profile, dict) or profile.get("authorization") != "explicit":
        return ""
    values = profile.get("content")
    if not isinstance(values, dict):
        return ""
    channel_keys = {
        "公众号": "wechat",
        "微信": "wechat",
        "小红书": "xiaohongshu",
        "短信": "sms",
        "push": "push",
    }
    prefix = next(
        (
            value
            for label, value in channel_keys.items()
            if label.lower() in (channel or "").lower()
        ),
        "",
    )
    selected: List[str] = []
    for key in ("default_style", "default_length"):
        if key in values:
            selected.append(f"{key}: {values[key]}")
    if prefix:
        for suffix in ("style", "length"):
            key = f"{prefix}_{suffix}"
            if key in values:
                selected.append(f"{key}: {values[key]}")
    if not selected:
        return ""
    return (
        "[AUTHORIZED_USER_PREFERENCE]\n"
        + "\n".join(f"- {item}" for item in selected)
        + "\n这些设置只用于完成当前内容任务，不得向用户复述其来源或内部存储方式。"
    )


PREFERENCE_CONTEXT_INTENTS = {
    "copy_generation",
    "copy_rewrite",
    "style_adaptation",
    "campaign_planning",
}

TASK_FIELD_CONTEXT_INTENTS = {
    *PREFERENCE_CONTEXT_INTENTS,
    "compliance_check",
    "rule_check",
    "content_review",
}


def memory_context_policy(intent: Any) -> Dict[str, bool]:
    """Return the deterministic memory layers permitted for an intent."""
    value = getattr(intent, "value", intent)
    normalized = str(value or "")
    return {
        "allow_long_term_preference": normalized in PREFERENCE_CONTEXT_INTENTS,
        "allow_current_task_fields": normalized in TASK_FIELD_CONTEXT_INTENTS,
        "allow_task_history": normalized in TASK_FIELD_CONTEXT_INTENTS,
        "allow_episodic_memory": False,
    }


MISSING_REWRITE_SOURCE_RESPONSE = (
    "请提供需要改写的新品上市原文，或告诉我新品名称和核心卖点。"
    "我会在不新增产品、体验或活动信息的前提下，将其改成语气克制的小红书风格。"
)

TOPIC_CONFLICT_RESPONSE = (
    "上一条任务与当前要改写的内容主题不一致，不能安全继承。"
    "请提供需要改写的新品上市原文，或告诉我新品名称和核心卖点。"
)


def build_missing_rewrite_response(
    channel: str = "", tone: str = "", topic_conflict: bool = False
) -> str:
    """Build the zero-latency clarification from current-turn slots."""
    tone_labels = {
        "种草": "更有种草感的",
        "克制": "语气克制的",
        "活泼": "更活泼的",
        "专业": "更专业的",
        "温暖": "更温暖的",
        "年轻": "更年轻化的",
        "简洁": "更简洁的",
        "口语化": "更口语化的",
    }
    style = f"{tone_labels.get(tone, (tone + '的') if tone else '')}{channel or '目标渠道'}风格"
    prefix = "上一条任务与当前主题不一致，不能安全继承。" if topic_conflict else ""
    return (
        prefix
        + "请提供需要改写的新品上市原文，或告诉我新品名称和核心卖点。"
        + "我会在不新增产品、体验或活动信息的前提下，将其改成"
        + style
        + "。"
    )
