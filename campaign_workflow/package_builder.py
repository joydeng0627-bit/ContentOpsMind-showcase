from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List
from .eligibility import delivery_eligibility


class PackageNotReadyError(RuntimeError):
    def __init__(self, message: str, reasons=None):
        super().__init__(message)
        self.reasons = reasons or []


def build_content_package(detail: Dict[str, Any]) -> Dict[str, Any]:
    eligibility = delivery_eligibility(detail)
    if not eligibility["publish_allowed"]:
        raise PackageNotReadyError(
            "交付资格校验未通过: " + ", ".join(eligibility["reasons"]),
            eligibility["reasons"],
        )
    workflow = detail["workflow"]
    if workflow["status"] not in {"approved", "delivered"}:
        raise PackageNotReadyError("Workflow 尚未完成全部最新版本审批")

    versions_by_id = {item["version_id"]: item for item in detail.get("versions", [])}
    approved_versions: List[Dict[str, Any]] = []
    for channel, version_id in workflow.get("latest_versions", {}).items():
        version = versions_by_id.get(version_id)
        if not version or version.get("status") != "approved":
            raise PackageNotReadyError(f"渠道 {channel} 的最新版本尚未批准")
        approved_versions.append(version)

    latest_ids = {item["version_id"] for item in approved_versions}
    reviews = [
        review
        for state in eligibility["versions"].values()
        for review in state["reviews"]
    ]
    approvals_by_id = {
        state["approval"].get(
            "approval_id", state["approval"].get("request_hash")
        ): state["approval"]
        for state in eligibility["versions"].values()
    }
    approvals = list(approvals_by_id.values())
    snapshot = workflow.get("knowledge_snapshot", {})
    return {
        "package_id": f"pkg_{workflow['workflow_id']}_{workflow['revision']}",
        "workflow_id": workflow["workflow_id"],
        "campaign_id": workflow["campaign_id"],
        "workflow_revision": workflow["revision"],
        "brief": workflow["brief"],
        "knowledge_snapshot": {
            "snapshot_hash": snapshot.get("snapshot_hash", ""),
            "retrieved_at": snapshot.get("retrieved_at"),
            "queries": snapshot.get("queries", []),
            "sources": [
                {
                    "document_id": item.get("document_id", ""),
                    "title": item.get("title", ""),
                    "source_url": item.get("source_url", ""),
                    "published_at": item.get("published_at", ""),
                    "document_type": item.get("document_type", ""),
                    "version": item.get("version", ""),
                    "status": item.get("status", ""),
                }
                for item in snapshot.get("documents", [])
            ],
        },
        "contents": sorted(approved_versions, key=lambda item: item["channel"]),
        "reviews": reviews,
        "review_contexts": [
            item
            for item in detail.get("review_contexts", [])
            if item.get("version_id") in latest_ids
        ],
        "approvals": approvals,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "approved",
        "publish_allowed": True,
        "review_status": eligibility["review_status"],
        "review_complete": eligibility["review_complete"],
        "version_bindings": {
            vid: state["binding"] for vid, state in eligibility["versions"].items()
        },
    }


def render_package_markdown(package: Dict[str, Any]) -> str:
    brief = package["brief"]
    lines = [
        f"# {brief['campaign_name']} 内容交付包",
        "",
        f"- Workflow: `{package['workflow_id']}`",
        f"- Campaign: `{package['campaign_id']}`",
        f"- 版本快照: `{package['workflow_revision']}`",
        f"- 生成时间: {package['generated_at']}",
        "",
        "## 活动 Brief",
        "",
        f"- 目标：{brief['objective']}",
        f"- 商品：{brief['product']}",
        f"- 受众：{brief['audience']}",
        f"- 渠道：{', '.join(brief['channels'])}",
    ]
    for content in package["contents"]:
        lines.extend(
            [
                "",
                f"## {content['channel']} · Version {content['version']}",
                "",
                content["content"],
            ]
        )
    lines.extend(["", "## 审批记录", ""])
    for approval in package["approvals"]:
        lines.append(
            f"- {approval['reviewer_id']} · {approval['decision']} · {approval['created_at']}"
            + (f" · {approval['comment']}" if approval.get("comment") else "")
        )
    lines.extend(["", "## 知识来源", ""])
    for source in package["knowledge_snapshot"]["sources"]:
        label = source.get("title") or "未命名来源"
        url = source.get("source_url") or "无外部链接"
        lines.append(f"- {label}：{url}")
    return "\n".join(lines).strip() + "\n"
