"""Redis working memory and ChromaDB episodic memory and user profiles.

Conversation summaries use the configured language model; ChromaDB handles embeddings.
"""

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

import chromadb
import redis
from anthropic import AsyncAnthropic
from core.llm_gateway import shared_client, gateway_call, llm_task
from core.llm_response import extract_model_text

logger = logging.getLogger(__name__)


class MsgRole(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass
class Message:
    role: MsgRole
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryContext:
    """传给 Agent 的完整上下文。"""

    recent_messages: List[Message]  # 工作记忆：最近对话
    relevant_history: List[str]  # 情景记忆：语义相关的历史片段
    user_profile: Dict[str, Any]  # 用户画像：偏好、常用实体
    summary: str  # 当前会话摘要（压缩后）

    @staticmethod
    def _clean(text: str) -> str:
        """移除 Unicode 代理字符，防止编码错误。"""
        return text.encode("utf-8", errors="ignore").decode("utf-8")

    def to_prompt_text(self) -> str:
        """Format conversation-local memory only.

        Long-term preferences are assembled separately after intent gating.
        Episodic history is deliberately excluded from Agent prompts.
        """
        parts = []
        if self.summary:
            parts.append(f"[会话摘要]\n{self._clean(self.summary)}")
        if self.recent_messages:
            parts.append("[最近对话]")
            for m in self.recent_messages:
                parts.append(f"{m.role.value}: {self._clean(m.content)}")
        return "\n\n".join(parts)


class MemoryManager:
    """
    三级记忆管理器。

    工作记忆存 Redis（TTL 24h），情景记忆和用户画像存 ChromaDB（持久化）。
    """

    WORKING_MAX = 20  # 工作记忆最大条数，超过则触发压缩
    COMPRESS_AT = 15  # 达到此条数时压缩，保留摘要 + 最近 5 条
    HISTORY_TOP_K = 5  # 情景记忆检索返回条数

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        chroma_host: str = "localhost",
        chroma_port: int = 8000,
        chroma_path: str = "./data/chroma",
        api_key: str = "",
        base_url: Optional[str] = None,
        model: str = "claude-3-5-sonnet-20241022",
    ):
        kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = shared_client(AsyncAnthropic, **kwargs)
        self._model = model

        self._redis = redis.from_url(redis_url, decode_responses=True)

        # ChromaDB：优先连接独立服务（docker compose 模式），连不上则降级为本地嵌入式
        try:
            # HttpClient 默认也会初始化 ChromaDB telemetry；显式关闭避免 posthog 兼容性错误日志。
            chroma = chromadb.HttpClient(
                host=chroma_host,
                port=chroma_port,
                settings=chromadb.Settings(anonymized_telemetry=False),
            )
            chroma.heartbeat()  # 测试连接
            logger.info(f"ChromaDB 已连接: {chroma_host}:{chroma_port}")
        except Exception:
            logger.info(f"ChromaDB 服务不可用，使用本地嵌入式模式: {chroma_path}")
            chroma = chromadb.PersistentClient(
                path=chroma_path,
                settings=chromadb.Settings(anonymized_telemetry=False),
            )

        # 情景记忆：存储历史对话片段
        self._episodic = chroma.get_or_create_collection("episodic")
        # 用户画像：存储提炼出的偏好和实体
        self._profile = chroma.get_or_create_collection("user_profile")

    # ── 写入 ──────────────────────────────────────────────────────────────────

    @llm_task()
    async def add_message(
        self,
        user_id: str,
        conv_id: str,
        role: MsgRole,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """将一条消息写入工作记忆，超阈值时自动压缩。"""
        user_id = self._safe_text(user_id)
        conv_id = self._safe_text(conv_id)
        clean_metadata = {
            self._safe_text(k): self._safe_metadata_value(v)
            for k, v in (metadata or {}).items()
        }
        msg = Message(
            role=role, content=self._safe_text(content), metadata=clean_metadata
        )
        key = self._wm_key(user_id, conv_id)

        # 追加到 Redis 列表（左推，最新在前）
        self._redis.lpush(
            key,
            json.dumps(
                {
                    "role": msg.role.value,
                    "content": msg.content,
                    "ts": msg.timestamp.isoformat(),
                    "metadata": msg.metadata,
                }
            ),
        )
        self._redis.expire(key, 86400)  # 24h TTL

        # 超过压缩阈值时触发压缩
        if self._redis.llen(key) >= self.COMPRESS_AT:
            await self._compress(user_id, conv_id)

    async def set_pending_task(
        self, user_id: str, conv_id: str, task: Dict[str, Any]
    ) -> None:
        """Persist a task that is waiting for one or more user-provided slots."""
        key = self._pending_task_key(self._safe_text(user_id), self._safe_text(conv_id))
        clean = self._safe_metadata_value(task)
        self._redis.setex(key, 1800, json.dumps(clean, ensure_ascii=False))

    async def get_pending_task(
        self, user_id: str, conv_id: str
    ) -> Optional[Dict[str, Any]]:
        key = self._pending_task_key(self._safe_text(user_id), self._safe_text(conv_id))
        raw = self._redis.get(key)
        if not raw:
            return None
        try:
            value = json.loads(raw)
            return value if isinstance(value, dict) else None
        except (TypeError, json.JSONDecodeError):
            self._redis.delete(key)
            return None

    async def clear_pending_task(self, user_id: str, conv_id: str) -> None:
        self._redis.delete(
            self._pending_task_key(self._safe_text(user_id), self._safe_text(conv_id))
        )

    async def set_active_artifact(
        self, user_id: str, conv_id: str, artifact: Dict[str, Any]
    ) -> None:
        """Store the latest editable output for deterministic follow-up edits."""
        key = self._active_artifact_key(
            self._safe_text(user_id), self._safe_text(conv_id)
        )
        clean = self._safe_metadata_value(artifact)
        self._redis.setex(key, 86400, json.dumps(clean, ensure_ascii=False))

    async def get_active_artifact(
        self, user_id: str, conv_id: str
    ) -> Optional[Dict[str, Any]]:
        key = self._active_artifact_key(
            self._safe_text(user_id), self._safe_text(conv_id)
        )
        raw = self._redis.get(key)
        if not raw:
            return None
        try:
            value = json.loads(raw)
            return value if isinstance(value, dict) else None
        except (TypeError, json.JSONDecodeError):
            self._redis.delete(key)
            return None

    async def clear_active_artifact(self, user_id: str, conv_id: str) -> None:
        self._redis.delete(
            self._active_artifact_key(
                self._safe_text(user_id), self._safe_text(conv_id)
            )
        )

    async def update_profile(
        self, user_id: str, message: str, request_id: str = ""
    ) -> bool:
        """Apply only explicitly authorized, durable preferences from this user turn."""
        user_id = self._safe_text(user_id)
        message = self._safe_text(message).strip()
        change = self._parse_preference_change(message)
        if change is None:
            return False
        try:
            current = await self._get_profile(user_id)
            content = (
                dict(current.get("content") or {}) if isinstance(current, dict) else {}
            )
            action, values = change
            if action == "clear":
                if values:
                    for key in values:
                        content.pop(key, None)
                else:
                    content.clear()
            else:
                content.update(values)

            doc_id = self._profile_id(user_id)
            if not content:
                self._profile.delete(ids=[doc_id])
                logger.info("用户明确偏好已清空: %s", user_id)
                return True

            now = datetime.now().isoformat()
            profile_data = {
                "version": 2,
                "type": "user_preference",
                "content": content,
                "authorization": "explicit",
                "updated_at": now,
            }
            self._profile.upsert(
                ids=[doc_id],
                documents=[json.dumps(profile_data, ensure_ascii=False)],
                metadatas=[
                    {
                        "user_id": user_id,
                        "memory_type": "user_preference",
                        "scope": "global",
                        "schema_version": 2,
                        "authorization": "explicit",
                        "request_id": self._safe_text(request_id),
                        "ts": now,
                    }
                ],
            )
            logger.info("用户明确偏好已更新: %s keys=%s", user_id, sorted(values))
            return True
        except Exception as ex:
            logger.warning("更新用户明确偏好失败: %s", ex)
            return False

    # ── 读取 ──────────────────────────────────────────────────────────────────

    async def get_context(
        self, user_id: str, conv_id: str, query: str = ""
    ) -> MemoryContext:
        """
        构建完整的记忆上下文。

        query 用于从情景记忆中检索语义相关的历史片段。
        """
        # 1. 工作记忆（当前会话最近消息）
        user_id = self._safe_text(user_id)
        conv_id = self._safe_text(conv_id)
        query = self._safe_text(query)

        recent = await self._get_working_memory(user_id, conv_id)

        # 2. 情景记忆（跨会话语义检索）
        # Task facts never cross conversation boundaries. Legacy episodic
        # records remain available for audit but are not injected by default.
        history: List[str] = []

        # 3. 用户画像
        profile = await self._get_profile(user_id)

        # 4. 会话摘要（如果已压缩过）
        summary = self._redis.get(self._summary_key(user_id, conv_id)) or ""

        return MemoryContext(
            recent_messages=recent,
            relevant_history=history,
            user_profile=profile,
            summary=summary,
        )

    # ── 压缩（防止 context 爆炸）─────────────────────────────────────────────

    async def _compress(self, user_id: str, conv_id: str) -> None:
        """
        工作记忆压缩：
          1. 用 LLM 对旧消息生成摘要
          2. 摘要存 Redis（覆盖旧摘要）
          3. 旧消息存入情景记忆（ChromaDB）供跨会话检索
          4. 工作记忆只保留最近 5 条
        """
        messages = await self._get_working_memory(user_id, conv_id)
        if len(messages) < self.COMPRESS_AT:
            return

        to_compress = messages[:-5]  # 保留最近 5 条
        keep = messages[-5:]

        # LLM 摘要
        text = self._safe_text(
            "\n".join(f"{m.role.value}: {m.content}" for m in to_compress)
        )
        prompt = self._safe_text(f"用 2-3 句话总结以下对话的关键信息：\n{text}")
        try:
            resp = await gateway_call(
                self._client,
                model=self._model,
                max_tokens=256,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            summary = self._safe_text(extract_model_text(resp)).strip()
        except Exception:
            summary = f"对话包含 {len(to_compress)} 条消息（摘要生成失败）"

        # 存摘要到 Redis
        skey = self._summary_key(user_id, conv_id)
        old_summary = self._redis.get(skey) or ""
        new_summary = self._safe_text(f"{old_summary}\n{summary}").strip()
        self._redis.setex(skey, 86400, new_summary)

        # 重置工作记忆为最近 5 条
        key = self._wm_key(user_id, conv_id)
        self._redis.delete(key)
        for m in reversed(keep):
            self._redis.lpush(
                key,
                json.dumps(
                    {
                        "role": m.role.value,
                        "content": m.content,
                        "ts": m.timestamp.isoformat(),
                        "metadata": m.metadata,
                    }
                ),
            )
        self._redis.expire(key, 86400)
        logger.info(f"工作记忆压缩完成: {user_id}/{conv_id}，摘要 {len(summary)} 字")

    # ── 内部辅助 ──────────────────────────────────────────────────────────────

    async def _get_working_memory(self, user_id: str, conv_id: str) -> List[Message]:
        key = self._wm_key(user_id, conv_id)
        raws = self._redis.lrange(key, 0, self.WORKING_MAX - 1)
        msgs = []
        for raw in reversed(raws):  # Redis lpush 最新在前，reversed 还原时序
            d = json.loads(raw)
            msgs.append(
                Message(
                    role=MsgRole(d["role"]),
                    content=d["content"],
                    timestamp=datetime.fromisoformat(d["ts"]),
                    metadata=d.get("metadata", {}),
                )
            )
        return msgs

    async def _search_episodic(self, user_id: str, query: str) -> List[str]:
        """语义检索情景记忆。ChromaDB 内置 embedding，不依赖外部 API。"""
        query_text = self._safe_text(query).strip()
        if not query_text:
            return []
        try:
            # 直接传 query_texts，ChromaDB 内置模型自动生成向量做匹配
            results = self._episodic.query(
                query_texts=[query_text],
                n_results=self.HISTORY_TOP_K,
                where={"user_id": self._safe_text(user_id)},
            )
            docs = results["documents"][0] if results["documents"] else []
            return [
                self._safe_text(doc)
                for doc in docs
                if isinstance(doc, str) and doc.strip()
            ]
        except Exception as ex:
            logger.warning(f"情景记忆检索失败: {ex}")
            return []

    async def _store_episodic(
        self, user_id: str, conv_id: str, text: str, summary: str
    ) -> None:
        """将压缩后的对话片段存入情景记忆。ChromaDB 内置 embedding，不依赖外部 API。"""
        try:
            user_id = self._safe_text(user_id)
            conv_id = self._safe_text(conv_id)
            text = self._safe_text(text)
            summary = self._safe_text(summary)
            doc_id = hashlib.md5(
                f"{user_id}{conv_id}{time.time()}".encode()
            ).hexdigest()
            # 直接传 documents，ChromaDB 内置模型自动生成 embedding
            self._episodic.add(
                ids=[doc_id],
                documents=[summary],
                metadatas=[
                    {
                        "user_id": user_id,
                        "conv_id": conv_id,
                        "memory_type": "episodic_memory",
                        "scope": "conversation",
                        "ts": datetime.now().isoformat(),
                        "full_text": self._safe_text(text[:500]),
                    }
                ],
            )
        except Exception as ex:
            logger.warning(f"存储情景记忆失败: {ex}")

    async def _get_profile(self, user_id: str) -> Dict[str, Any]:
        """Get the single canonical, explicitly authorized profile document."""
        try:
            results = self._profile.get(ids=[self._profile_id(user_id)])
            if results["documents"]:
                value = json.loads(results["documents"][0])
                if (
                    isinstance(value, dict)
                    and value.get("version") == 2
                    and value.get("authorization") == "explicit"
                ):
                    return value
        except Exception:
            pass
        return {}

    @staticmethod
    def _profile_id(user_id: str) -> str:
        return "user_pref_v2_" + hashlib.sha256(user_id.encode("utf-8")).hexdigest()

    @classmethod
    def _parse_preference_change(
        cls,
        message: str,
    ) -> Optional[tuple[str, Dict[str, Any]]]:
        text = (message or "").strip()
        lowered = text.lower()
        channels = {
            "微信公众号": "wechat",
            "公众号": "wechat",
            "微信": "wechat",
            "小红书": "xiaohongshu",
            "短信": "sms",
            "push": "push",
        }
        channel = next(
            (value for label, value in channels.items() if label.lower() in lowered), ""
        )
        forget = any(word in text for word in ("忘记", "取消", "删除", "清除")) and any(
            word in text for word in ("偏好", "默认", "设置", "记住")
        )
        if forget:
            if not channel:
                return "clear", {}
            keys = {f"{channel}_style": "", f"{channel}_length": ""}
            if "字" in text or "长度" in text:
                keys = {f"{channel}_length": ""}
            elif any(
                style in text
                for style in (
                    "风格",
                    "语气",
                    "克制",
                    "专业",
                    "活泼",
                    "温暖",
                    "年轻",
                    "简洁",
                )
            ):
                keys = {f"{channel}_style": ""}
            return "clear", keys

        authorized = any(
            marker in text
            for marker in (
                "请记住",
                "记住以后",
                "以后默认",
                "今后默认",
                "以后都",
                "今后都",
                "我偏好",
                "我的偏好",
                "设为默认",
            )
        )
        if not authorized:
            return None

        values: Dict[str, Any] = {}
        style = next(
            (
                item
                for item in ("克制", "专业", "活泼", "温暖", "年轻化", "简洁")
                if item in text
            ),
            "",
        )
        length_match = re.search(r"(\d{1,4})\s*字(?:以内|以下|上限)?", text)
        if channel and style:
            values[f"{channel}_style"] = style
        elif style:
            values["default_style"] = style
        if length_match:
            length = int(length_match.group(1))
            if 1 <= length <= 5000:
                values[f"{channel}_length" if channel else "default_length"] = length
        if not values:
            return None
        return "set", values

    @staticmethod
    def _wm_key(user_id: str, conv_id: str) -> str:
        return f"wm:{user_id}:{conv_id}"

    @staticmethod
    def _summary_key(user_id: str, conv_id: str) -> str:
        return f"summary:{user_id}:{conv_id}"

    @staticmethod
    def _pending_task_key(user_id: str, conv_id: str) -> str:
        return f"pending_task:{user_id}:{conv_id}"

    @staticmethod
    def _active_artifact_key(user_id: str, conv_id: str) -> str:
        return f"active_artifact:{user_id}:{conv_id}"

    @staticmethod
    def _safe_text(value: Any) -> str:
        """转成 ChromaDB 可接受的普通 UTF-8 字符串。"""
        if value is None:
            return ""
        if not isinstance(value, str):
            value = str(value)
        return value.encode("utf-8", errors="ignore").decode("utf-8")

    @classmethod
    def _safe_metadata_value(cls, value: Any) -> Any:
        """递归清洗 metadata，避免 Redis/ChromaDB 后续读写遇到非法 UTF-8。"""
        if isinstance(value, str):
            return cls._safe_text(value)
        if isinstance(value, dict):
            return {
                cls._safe_text(k): cls._safe_metadata_value(v) for k, v in value.items()
            }
        if isinstance(value, list):
            return [cls._safe_metadata_value(v) for v in value]
        return value
