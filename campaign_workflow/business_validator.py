"""Deterministic checks for claims that should not depend on LLM judgement."""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List


def _structured_documents(
    documents: Iterable[Dict[str, Any]], kind: str
) -> List[Dict[str, Any]]:
    return [
        item
        for item in documents
        if isinstance(item.get("structured_data"), dict)
        and item["structured_data"].get("kind") == kind
    ]


def validate_generated_content(
    channel: str,
    content: str,
    documents: Iterable[Dict[str, Any]],
) -> List[Dict[str, str]]:
    issues: List[Dict[str, str]] = []
    docs = list(documents)
    promotions = _structured_documents(docs, "promotion")
    products = _structured_documents(docs, "product")
    policies = _structured_documents(docs, "channel_policy")

    allowed_tiers = {
        (int(tier["threshold_cny"]), int(tier["discount_cny"]))
        for doc in promotions
        for tier in doc["structured_data"].get("tiers", [])
        if "threshold_cny" in tier and "discount_cny" in tier
    }
    claimed_tiers = {
        (int(threshold), int(discount))
        for threshold, discount in re.findall(
            r"满\s*(\d+)\s*(?:元)?\s*减\s*(\d+)", content
        )
    }
    for tier in (claimed_tiers - allowed_tiers) if promotions else set():
        issues.append(
            {
                "rule_id": "promotion_tier_mismatch",
                "impact": "blocking_issue",
                "message": f"文案中的满 {tier[0]} 减 {tier[1]} 不属于当前有效活动档位",
            }
        )
    for threshold, discount in allowed_tiers:
        shorthand = list(
            re.finditer(rf"{threshold}\s*(?:元)?\s*减\s*{discount}", content)
        )
        actionable = False
        for match in shorthand:
            # “满 199 减 30”允许在“满”与数字间带空格；固定宽度负向后顾
            # 无法正确识别这种写法，因此显式检查匹配前缀。
            prefix = content[max(0, match.start() - 4) : match.start()]
            if re.search(r"满\s*$", prefix):
                continue
            context = content[max(0, match.start() - 18) : match.end() + 18]
            if not any(
                term in context
                for term in ("未达", "没达到", "没有达到", "不满足", "不可用", "不适用")
            ):
                actionable = True
                break
        if actionable:
            issues.append(
                {
                    "rule_id": "missing_threshold",
                    "impact": "required_change",
                    "message": f"优惠 {threshold} 减 {discount} 缺少“满”字门槛",
                }
            )

    mentioned_prices = []
    for doc in products:
        title = str(doc.get("title", "")).removeprefix("产品资料：")
        data = doc["structured_data"]
        if title and title in content and data.get("price_cny") is not None:
            mentioned_prices.append(int(data["price_cny"]))
    if len(mentioned_prices) >= 2 and claimed_tiers:
        subtotal = sum(mentioned_prices)
        for threshold, _ in claimed_tiers:
            positions = [
                match.start() for match in re.finditer(rf"满?\s*{threshold}", content)
            ]
            contexts = [content[max(0, pos - 45) : pos + 55] for pos in positions]
            positive_claim = any(
                any(
                    term in context
                    for term in ("达到", "满足", "可用", "适用", "推荐", "选择")
                )
                and any(
                    term in context
                    for term in ("两件", "合计", "一起", "组合", "配齐", "凑单")
                )
                and not any(
                    term in context
                    for term in (
                        "未达到",
                        "未达",
                        "没达到",
                        "没有达到",
                        "不满足",
                        "不可用",
                        "不适用",
                    )
                )
                for context in contexts
            )
            if positive_claim and subtotal < threshold:
                issues.append(
                    {
                        "rule_id": "ineligible_promotion_recommendation",
                        "impact": "blocking_issue",
                        "message": f"文案提及商品合计 {subtotal} 元，不满足 {threshold} 元门槛",
                    }
                )

    for doc in promotions:
        end = str(doc["structured_data"].get("ends_at", ""))
        if not end:
            continue
        expected = re.search(r"-(\d{2})-(\d{2})T(\d{2}):(\d{2})", end)
        if not expected:
            continue
        month, day, hour, minute = expected.groups()
        date_pattern = r"(\d{1,2})\s*月\s*(\d{1,2})\s*日(?:\s*(\d{1,2})[:：](\d{2}))?"
        dates = re.findall(rf"(?:截至|截止到|有效期至|至)\s*{date_pattern}", content)
        dates.extend(re.findall(rf"{date_pattern}\s*(?:前)?\s*(?:截止|结束)", content))
        for value in dates:
            normalized = tuple(part.zfill(2) if part else part for part in value)
            if normalized[0:2] != (month, day) or (
                hour and normalized[2:4] != (hour, minute)
            ):
                issues.append(
                    {
                        "rule_id": "incorrect_end_date",
                        "impact": "required_change",
                        "message": f"活动截止时间必须完整写为 {int(month)} 月 {int(day)} 日 {hour}:{minute}",
                    }
                )
        compact_dates = re.findall(
            r"(\d{1,2})[./](\d{1,2})(?:\s*(\d{1,2})[:：](\d{2}))?\s*(?:前)?\s*(?:截止|结束|止)",
            content,
        )
        for value in compact_dates:
            normalized = tuple(part.zfill(2) if part else part for part in value)
            if normalized[0:2] != (month, day) or (
                hour and normalized[2:4] != (hour, minute)
            ):
                issues.append(
                    {
                        "rule_id": "incorrect_end_date",
                        "impact": "required_change",
                        "message": f"截止时间必须完整写为 {int(month)} 月 {int(day)} 日 {hour}:{minute}",
                    }
                )

    normalized_channel = channel.lower()
    if promotions and claimed_tiers:
        scope = str(promotions[0]["structured_data"].get("scope", ""))
        required_scope_terms = [
            term for term in ("官方商城", "自营正价") if term in scope
        ]
        missing_scope = [term for term in required_scope_terms if term not in content]
        # 短信优先容纳品牌、优惠、截止时间、短链和退订提示；完整适用范围
        # 可以由活动页承载。没有活动页入口时仍要求正文披露，避免静默丢失限定。
        has_landing_page = bool(
            re.search(r"https?://|[a-zA-Z0-9]{4,}\.(?:cn|com)|活动页", content)
        )
        scope_may_live_on_landing_page = (
            normalized_channel in {"短信", "sms"} and has_landing_page
        )
        if missing_scope and not scope_may_live_on_landing_page:
            issues.append(
                {
                    "rule_id": "missing_scope_qualifier",
                    "impact": "required_change",
                    "message": "缺少活动适用范围：" + "、".join(missing_scope),
                }
            )

    policy = next(
        (
            item["structured_data"].get(channel.lower())
            for item in policies
            if item["structured_data"].get(channel.lower())
        ),
        None,
    )
    if normalized_channel in {"短信", "sms"}:
        policy = policy or next(
            (
                item["structured_data"].get("sms")
                for item in policies
                if item["structured_data"].get("sms")
            ),
            {},
        )
        max_chars = int((policy or {}).get("recommended_max_chars", 100))
        suffix = str((policy or {}).get("required_suffix", "退订回T"))
        if len(content) > max_chars:
            issues.append(
                {
                    "rule_id": "channel_length_exceeded",
                    "impact": "required_change",
                    "message": f"短信共 {len(content)} 字符，超过建议上限 {max_chars}",
                }
            )
        if suffix and suffix not in content:
            issues.append(
                {
                    "rule_id": "missing_unsubscribe",
                    "impact": "required_change",
                    "message": f"短信缺少退订提示“{suffix}”",
                }
            )
        if not re.search(r"https?://|[a-zA-Z0-9]{4,}\.(?:cn|com)|活动页", content):
            issues.append(
                {
                    "rule_id": "missing_delivery_link",
                    "impact": "delivery_configuration",
                    "message": "发送前需要绑定活动页短链",
                }
            )
    elif normalized_channel == "push":
        policy = policy or next(
            (
                item["structured_data"].get("push")
                for item in policies
                if item["structured_data"].get("push")
            ),
            {},
        )
        title_match = re.search(r"(?:标题[:：])([^\n]+)", content)
        body_match = re.search(r"(?:正文[:：])([\s\S]+)", content)
        if title_match and len(title_match.group(1).strip()) > int(
            (policy or {}).get("title_max_chars", 24)
        ):
            issues.append(
                {
                    "rule_id": "channel_length_exceeded",
                    "impact": "required_change",
                    "message": "Push 标题超过渠道上限",
                }
            )
        if body_match and len(body_match.group(1).strip()) > int(
            (policy or {}).get("body_max_chars", 80)
        ):
            issues.append(
                {
                    "rule_id": "channel_length_exceeded",
                    "impact": "required_change",
                    "message": "Push 正文超过渠道上限",
                }
            )
    return issues


def requires_revision(issues: Iterable[Dict[str, str]]) -> bool:
    return any(
        item.get("impact") in {"blocking_issue", "required_change"} for item in issues
    )
