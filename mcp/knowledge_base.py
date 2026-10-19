"""ChromaDB ingestion and governed keyword/vector retrieval.

Knowledge documents and conversation memory use separate collections.
"""

import hashlib
import json
import logging
import re
from typing import Any, Dict, List, Optional

import chromadb
from opencc import OpenCC
from core.retrieval_policy import eligible

logger = logging.getLogger(__name__)
_SEARCH_CONVERTER = OpenCC("t2s")


class KnowledgeBase:
    """
    基于 ChromaDB 的 RAG 知识库。

    ChromaDB 内置了 Embedding 模型（all-MiniLM-L6-v2），
    调用 add() 时自动生成向量，query() 时自动做语义匹配。
    不需要额外调用 Anthropic Embeddings API。
    """

    COLLECTION_NAME = "content_ops_knowledge_base"

    def __init__(
        self,
        chroma_host: str = "localhost",
        chroma_port: int = 8000,
        chroma_path: str = "./data/chroma",
        load_default_docs: bool = True,
    ):
        # 优先连接独立 ChromaDB 服务（服务端内置 embedding 模型，客户端无需下载）
        self._use_server = False
        try:
            # HttpClient 默认也会初始化 ChromaDB telemetry；显式关闭避免 posthog 兼容性错误日志。
            self._client = chromadb.HttpClient(
                host=chroma_host,
                port=chroma_port,
                settings=chromadb.Settings(anonymized_telemetry=False),
            )
            self._client.heartbeat()
            self._use_server = True
            logger.info(f"知识库 ChromaDB 已连接: {chroma_host}:{chroma_port}")
        except Exception:
            logger.info(f"知识库 ChromaDB 服务不可用，使用本地模式: {chroma_path}")
            self._client = chromadb.PersistentClient(
                path=chroma_path,
                settings=chromadb.Settings(anonymized_telemetry=False),
            )

        # 使用服务端时不传 embedding_function，让服务端处理
        # 本地模式时也不传，使用 ChromaDB 默认的（会触发模型下载）
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"description": "ContentOpsMind RAG 知识库"},
        )

        # 如果知识库为空，导入默认文档
        if load_default_docs and self._collection.count() == 0:
            self._load_default_docs()

    # ── 文档管理 ──────────────────────────────────────────────────────────────

    def add_documents(self, documents: List[Dict[str, str]]) -> int:
        """
        批量导入文档到知识库。

        documents 格式: [{"title": "...", "content": "..."}, ...]
        长文档会自动切片（每片 500 字）。
        """
        ids, docs, metas = [], [], []

        # `document_id` is the public stable identity. Chroma chunk ids include
        # content, so an upsert alone would leave stale chunks behind whenever
        # a document is revised. Remove only matching stable ids before adding
        # their replacement chunks; unrelated corpora remain untouched.
        replacement_ids = set()
        for document in documents:
            source = str(document.get("source", "curated_content_ops"))
            default_corpus = (
                "demo-production" if source == "demo_brand_business" else "default"
            )
            stable_id = (
                str(document.get("document_id", "")).strip()
                or hashlib.sha256(
                    f"{default_corpus}|{document.get('title', '')}".encode("utf-8")
                ).hexdigest()[:20]
            )
            replacement_ids.add(stable_id)
        if replacement_ids:
            existing = self._collection.get(include=["metadatas"])
            stale_chunk_ids = [
                chunk_id
                for chunk_id, metadata in zip(
                    existing.get("ids", []) or [],
                    existing.get("metadatas", []) or [],
                )
                if str((metadata or {}).get("document_id", "")).strip()
                in replacement_ids
            ]
            if stale_chunk_ids:
                self._collection.delete(ids=stale_chunk_ids)

        for doc in documents:
            title = doc.get("title", "")
            content = doc.get("content", "")
            source = str(doc.get("source", "curated_content_ops"))
            default_corpus = (
                "demo-production" if source == "demo_brand_business" else "default"
            )
            default_status = (
                "current" if source == "demo_brand_business" else "approved"
            )
            requested_status = str(doc.get("status", default_status))
            if (
                source == "demo_brand_business"
                and str(doc.get("document_type", "")) == "historical_copy"
            ):
                normalized_status = "historical"
            else:
                normalized_status = (
                    "current"
                    if source == "demo_brand_business"
                    and requested_status == "approved"
                    else requested_status
                )
            stable_document_id = (
                str(doc.get("document_id", "")).strip()
                or hashlib.sha256(
                    f"{default_corpus}|{title}".encode("utf-8")
                ).hexdigest()[:20]
            )
            chunks = doc.get("chunks") or self._chunk_text(content, chunk_size=500)

            for i, chunk in enumerate(chunks):
                doc_id = hashlib.md5(f"{title}_{i}_{chunk[:50]}".encode()).hexdigest()
                ids.append(doc_id)
                docs.append(chunk)
                metadata = {
                    "document_id": stable_document_id,
                    "corpus_id": str(doc.get("corpus_id", default_corpus)),
                    "title": title,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "category": str(doc.get("category", "general")),
                    "document_type": str(doc.get("document_type", "knowledge")),
                    "channel": str(doc.get("channel", "all")),
                    "version": str(doc.get("version", "1.0")),
                    "source": source,
                    "source_file": str(doc.get("source_file", "")),
                    "brand": str(doc.get("brand", "")),
                    "product_id": str(doc.get("product_id", "")),
                    "campaign_id": str(doc.get("campaign_id", "")),
                    "campaign": str(doc.get("campaign", doc.get("campaign_id", ""))),
                    "status": normalized_status,
                    "audience": str(doc.get("audience", "all")),
                    "published_at": str(doc.get("published_at", "")),
                    "effective_at": str(doc.get("effective_at", "")),
                    "tags": ",".join(str(tag) for tag in doc.get("tags", [])),
                    "source_url": str(doc.get("source_url", "")),
                    "content_hash": str(doc.get("content_hash", "")),
                    "crawled_at": str(doc.get("crawled_at", "")),
                    "parser_version": str(doc.get("parser_version", "")),
                    "location": str(doc.get("location", "")),
                    "structured_data_json": json.dumps(
                        doc.get("structured_data", {}),
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                }
                metas.append(metadata)

        if ids:
            # ChromaDB 会自动生成 Embedding
            self._collection.upsert(ids=ids, documents=docs, metadatas=metas)
            logger.info(f"知识库导入 {len(ids)} 个文档片段")

        return len(ids)

    def corpus_chunk_ids(self, corpus_id: str) -> List[str]:
        result = self._collection.get(where={"corpus_id": {"$eq": str(corpus_id)}})
        return list(result.get("ids", []) or [])

    def replace_corpus(self, corpus_id: str, documents: List[Dict[str, Any]]) -> int:
        """Replace one explicit corpus without touching evaluation or other tenant data."""
        normalized = [dict(item, corpus_id=corpus_id) for item in documents]
        existing_ids = self.corpus_chunk_ids(corpus_id)
        if existing_ids:
            self._collection.delete(ids=existing_ids)
        return self.add_documents(normalized)

    def get_source_document_state(self, source_url: str) -> Dict[str, Any] | None:
        """Return stored chunk ids and source metadata for one canonical URL."""
        if not source_url:
            return None
        result = self._collection.get(
            where={"source_url": source_url},
            include=["metadatas"],
        )
        ids = list(result.get("ids", []) or [])
        if not ids:
            return None
        metadatas = list(result.get("metadatas", []) or [])
        metadata = dict(metadatas[0] or {}) if metadatas else {}
        return {
            "document_id": metadata.get("document_id", ""),
            "source_url": source_url,
            "content_hash": str(metadata.get("content_hash", "")),
            "chunk_ids": ids,
            "chunk_count": len(ids),
            "published_at": str(metadata.get("published_at", "")),
            "crawled_at": str(metadata.get("crawled_at", "")),
        }

    def delete_source_document(self, source_url: str) -> int:
        """Delete only chunks originating from one canonical source URL."""
        state = self.get_source_document_state(source_url)
        if not state:
            return 0
        ids = list(state["chunk_ids"])
        self._collection.delete(ids=ids)
        logger.info("知识来源已删除: source_url=%s chunks=%s", source_url, len(ids))
        return len(ids)

    def upsert_source_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Incrementally replace one approved external source document.

        Existing curated documents and unrelated external sources are untouched.
        """
        source_url = str(document.get("source_url", "")).strip()
        content_hash = str(document.get("content_hash", "")).strip()
        content = str(document.get("content", "")).strip()
        status = str(document.get("status", "approved"))
        if not source_url or not content_hash or not content:
            raise ValueError("source_url、content_hash 和 content 为必填字段")
        if status != "approved":
            raise ValueError(f"文档状态不可入库: {status}")

        previous = self.get_source_document_state(source_url)
        if previous and previous.get("content_hash") == content_hash:
            return {
                "changed": False,
                "source_url": source_url,
                "content_hash": content_hash,
                "added_chunks": 0,
                "deleted_chunks": 0,
                "chunk_ids": list(previous.get("chunk_ids", [])),
            }

        deleted = self.delete_source_document(source_url) if previous else 0
        chunks = self._chunk_text(content, chunk_size=500)
        ids, docs, metas = [], [], []
        for index, chunk in enumerate(chunks):
            chunk_id = hashlib.sha256(
                f"{source_url}|{content_hash}|{index}".encode("utf-8")
            ).hexdigest()
            ids.append(chunk_id)
            docs.append(chunk)
            metas.append(
                {
                    "title": str(document.get("title", "")),
                    "document_id": str(document.get("document_id", "")),
                    "corpus_id": str(document.get("corpus_id", "default")),
                    "chunk_index": index,
                    "total_chunks": len(chunks),
                    "category": str(document.get("category", "general")),
                    "document_type": str(document.get("document_type", "knowledge")),
                    "channel": str(document.get("channel", "all")),
                    "version": str(document.get("version", "1.0")),
                    "source": str(document.get("source", "external")),
                    "source_file": str(document.get("source_file", source_url)),
                    "brand": str(document.get("brand", "")),
                    "product_id": str(document.get("product_id", "")),
                    "campaign_id": str(document.get("campaign_id", "")),
                    "campaign": str(
                        document.get("campaign", document.get("campaign_id", ""))
                    ),
                    "status": status,
                    "audience": str(document.get("audience", "all")),
                    "published_at": str(document.get("published_at", "")),
                    "tags": ",".join(str(tag) for tag in document.get("tags", [])),
                    "source_url": source_url,
                    "content_hash": content_hash,
                    "crawled_at": str(document.get("crawled_at", "")),
                    "parser_version": str(document.get("parser_version", "")),
                    "location": str(document.get("location", "")),
                    "structured_data_json": json.dumps(
                        document.get("structured_data", {}),
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                }
            )
        if ids:
            self._collection.upsert(ids=ids, documents=docs, metadatas=metas)
        logger.info(
            "知识来源已同步: source_url=%s deleted=%s added=%s",
            source_url,
            deleted,
            len(ids),
        )
        return {
            "changed": True,
            "source_url": source_url,
            "content_hash": content_hash,
            "added_chunks": len(ids),
            "deleted_chunks": deleted,
            "chunk_ids": ids,
        }

    def sync_documents(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Refresh curated corpus without deleting source-scoped external documents."""
        canonical = json.dumps(documents, ensure_ascii=False, sort_keys=True)
        corpus_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
        metadata = self._collection.metadata or {}
        if metadata.get("corpus_hash") == corpus_hash:
            return {
                "changed": False,
                "corpus_hash": corpus_hash,
                "chunk_count": self.doc_count,
            }

        existing = self._collection.get(include=["metadatas"])
        curated_ids = []
        for doc_id, item_metadata in zip(
            existing.get("ids", []) or [],
            existing.get("metadatas", []) or [],
        ):
            if not str((item_metadata or {}).get("source_url", "")).strip():
                curated_ids.append(doc_id)
        if curated_ids:
            self._collection.delete(ids=curated_ids)
        count = self.add_documents(documents)
        collection_metadata = dict(metadata)
        collection_metadata.update(
            {
                "description": "ContentOpsMind versioned content operations knowledge base",
                "corpus_hash": corpus_hash,
            }
        )
        self._collection.modify(metadata=collection_metadata)
        logger.info("知识库语料已同步: hash=%s chunks=%s", corpus_hash, count)
        return {"changed": True, "corpus_hash": corpus_hash, "chunk_count": count}

    def search(
        self,
        query: str,
        top_k: int = 5,
        *,
        filters: Optional[Dict[str, str]] = None,
        min_score: float = 0.0,
        parameter_field: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        混合检索：向量语义分 + 中文标题/标签/正文关键词分。

        小规模品牌知识库中，产品名、活动名和渠道名属于强约束实体，
        不能仅依赖通用多语言 Embedding。这里先做向量召回，同时读取
        当前集合做轻量词法匹配，再统一排序。
        """
        collection_count = self._collection.count()
        if collection_count <= 0:
            return []
        semantic_k = min(collection_count, max(top_k * 4, 20))
        normalized_filters = {
            key: str(value).strip()
            for key, value in (filters or {}).items()
            if str(value).strip()
            and key
            in {
                "corpus_id",
                "brand",
                "campaign",
                "status",
                "document_type",
                "product_id",
                "version",
            }
        }
        where = self._where_clause(normalized_filters)
        results = self._collection.query(
            query_texts=[query],
            n_results=semantic_k,
            **({"where": where} if where else {}),
        )

        candidates: Dict[str, Dict[str, Any]] = {}
        if results["documents"] and results["documents"][0]:
            for doc_id, doc, meta, dist in zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                candidates[doc_id] = self._result_item(
                    doc,
                    meta,
                    vector_score=1.0 - dist,
                )

        # 词法检索覆盖全量元数据，确保精确产品名、活动名不会被向量召回漏掉。
        all_docs = self._collection.get(
            include=["documents", "metadatas"],
            **({"where": where} if where else {}),
        )
        for doc_id, doc, meta in zip(
            all_docs.get("ids", []),
            all_docs.get("documents", []),
            all_docs.get("metadatas", []),
        ):
            if doc_id not in candidates:
                candidates[doc_id] = self._result_item(doc, meta, vector_score=0.0)

        candidates = {
            k: v for k, v in candidates.items() if eligible(query, v, filters or {})
        }
        for item in candidates.values():
            lexical_score = self._lexical_score(query, item)
            vector_score = max(0.0, float(item["vector_score"]))
            # 精确业务实体优先，向量语义分负责补充同义表达。
            item["lexical_score"] = round(lexical_score, 4)
            item["score"] = round(0.8 * lexical_score + 0.2 * vector_score, 4)

        ranked = sorted(
            candidates.values(),
            key=lambda item: (
                item["score"],
                item["lexical_score"],
                item["vector_score"],
            ),
            reverse=True,
        )
        # A document may have multiple chunks or historical versions. Keep one
        # authoritative result per stable document id, preferring current and
        # effective-dated material before score.
        authoritative = self._prefer_versions(ranked, filters or {})
        deduped = []
        for item in authoritative:
            if float(item.get("score", 0.0)) < min_score:
                continue
            deduped.append(item)
            if len(deduped) >= top_k:
                break
        return deduped

    @staticmethod
    def _prefer_versions(ranked, filters):
        versions: Dict[str, List[Dict[str, Any]]] = {}
        for item in ranked:
            stable_id = str(
                item.get("document_id") or f"{item.get('title')}|{item.get('version')}"
            )
            if filters.get("_scope") == "real":
                stable_id += "|" + str(item.get("chunk", 0))
            versions.setdefault(stable_id, []).append(item)
        authoritative = [
            max(
                items,
                key=lambda item: (
                    str(item.get("status", "")) == "current",
                    str(item.get("effective_at", "")),
                    float(item.get("score", 0.0)),
                ),
            )
            for items in versions.values()
        ]
        authoritative.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
        return authoritative

    def search_branches(self, query, top_k=20, *, filters=None, min_score=0.0):
        """Return lexical/vector lists after scope, metadata and version governance."""
        if not 1 <= top_k <= 100:
            raise ValueError("branch top_k must be between 1 and 100")
        filters = filters or {}
        where = self._where_clause(
            {
                key: str(value).strip()
                for key, value in filters.items()
                if str(value).strip()
                and key
                in {
                    "corpus_id",
                    "brand",
                    "campaign",
                    "status",
                    "document_type",
                    "product_id",
                    "version",
                }
            }
        )
        kwargs = {"where": where} if where else {}
        count = self._collection.count()
        if not count:
            return {"lexical": [], "vector": []}
        results = self._collection.query(
            query_texts=[query], n_results=min(count, top_k), **kwargs
        )
        candidates, vector_order = {}, {}
        for rank, (doc_id, text, meta, distance) in enumerate(
            zip(
                results.get("ids", [[]])[0],
                results.get("documents", [[]])[0],
                results.get("metadatas", [[]])[0],
                results.get("distances", [[]])[0],
            ),
            1,
        ):
            candidates[doc_id] = self._result_item(
                text, meta, vector_score=1.0 - distance
            )
            vector_order[doc_id] = rank
        all_docs = self._collection.get(include=["documents", "metadatas"], **kwargs)
        for doc_id, text, meta in zip(
            all_docs.get("ids", []),
            all_docs.get("documents", []),
            all_docs.get("metadatas", []),
        ):
            if doc_id not in candidates:
                candidates[doc_id] = self._result_item(text, meta, vector_score=0.0)
        from core.rank_fusion import item_key

        eligible_rows = []
        for doc_id, item in candidates.items():
            if not eligible(query, item, filters):
                continue
            lexical = self._lexical_score(query, item)
            item["lexical_score"] = round(lexical, 4)
            # Preserve the existing relevance threshold scale; RRF scores use a different scale.
            item["score"] = round(
                0.8 * lexical + 0.2 * max(0.0, float(item["vector_score"])), 4
            )
            item["recall_vector_rank"] = vector_order.get(doc_id)
            eligible_rows.append(item)
        # Version choice precedes both branch truncation and fusion.
        governed = self._prefer_versions(eligible_rows, filters)
        governed = [item for item in governed if item["score"] >= min_score]
        lexical = sorted(
            (item for item in governed if item["lexical_score"] > 0),
            key=lambda item: (-item["lexical_score"], item_key(item)),
        )[:top_k]
        vector = sorted(
            (item for item in governed if item["recall_vector_rank"] is not None),
            key=lambda item: item["recall_vector_rank"],
        )[:top_k]
        return {"lexical": lexical, "vector": vector}

    @staticmethod
    def _where_clause(filters: Dict[str, str]) -> Optional[Dict[str, Any]]:
        clauses = [{key: {"$eq": value}} for key, value in filters.items()]
        if not clauses:
            return None
        return clauses[0] if len(clauses) == 1 else {"$and": clauses}

    @staticmethod
    def _result_item(
        document: str,
        metadata: Dict[str, Any],
        vector_score: float,
    ) -> Dict[str, Any]:
        try:
            structured_data = json.loads(
                str(metadata.get("structured_data_json", "{}"))
            )
        except json.JSONDecodeError:
            structured_data = {}
        return {
            "document_id": metadata.get("document_id", ""),
            "corpus_id": metadata.get("corpus_id", "default"),
            "title": metadata.get("title", ""),
            "content": document,
            "score": round(vector_score, 4),
            "vector_score": round(vector_score, 4),
            "lexical_score": 0.0,
            "chunk": metadata.get("chunk_index", 0),
            "category": metadata.get("category", "general"),
            "document_type": metadata.get("document_type", "knowledge"),
            "channel": metadata.get("channel", "all"),
            "version": metadata.get("version", "1.0"),
            "source": metadata.get("source", "curated_content_ops"),
            "source_file": metadata.get("source_file", ""),
            "brand": metadata.get("brand", ""),
            "product_id": metadata.get("product_id", ""),
            "campaign_id": metadata.get("campaign_id", ""),
            "campaign": metadata.get("campaign", metadata.get("campaign_id", "")),
            "status": metadata.get("status", "approved"),
            "audience": metadata.get("audience", "all"),
            "published_at": metadata.get("published_at", ""),
            "effective_at": metadata.get("effective_at", ""),
            "source_url": metadata.get("source_url", ""),
            "content_hash": metadata.get("content_hash", ""),
            "crawled_at": metadata.get("crawled_at", ""),
            "parser_version": metadata.get("parser_version", ""),
            "location": metadata.get("location", ""),
            "tags": [tag for tag in str(metadata.get("tags", "")).split(",") if tag],
            "structured_data": (
                structured_data if isinstance(structured_data, dict) else {}
            ),
        }

    @classmethod
    def _lexical_score(cls, query: str, item: Dict[str, Any]) -> float:
        terms = cls._query_terms(query)
        if not terms:
            return 0.0
        normalized_query = cls._normalize_search_text(query)
        title = cls._normalize_search_text(item.get("title", ""))
        tag_values = [
            cls._normalize_search_text(tag)
            for tag in item.get("tags", [])
            if cls._normalize_search_text(tag)
        ]
        tags = cls._normalize_search_text(" ".join(tag_values))
        content = cls._normalize_search_text(item.get("content", ""))
        metadata = cls._normalize_search_text(
            " ".join(
                [
                    str(item.get("brand", "")),
                    str(item.get("product_id", "")),
                    str(item.get("campaign_id", "")),
                    str(item.get("document_type", "")),
                    str(item.get("channel", "")),
                ]
            )
        )
        total_weight = sum(len(term) ** 2 for term in terms)
        matched_weight = 0.0
        for term in terms:
            weight = len(term) ** 2
            if term in title:
                matched_weight += weight
            elif term in tags or term in metadata:
                matched_weight += weight * 0.8
            elif term in content:
                matched_weight += weight * 0.45
        ngram_score = matched_weight / total_weight if total_weight else 0.0
        longest_title_match = cls._longest_common_substring(normalized_query, title)
        title_score = min(1.0, longest_title_match / 6.0)
        tag_score = max(
            (
                min(1.0, len(tag) / 4.0)
                for tag in tag_values
                if len(tag) >= 2 and tag in normalized_query
            ),
            default=0.0,
        )
        type_score = cls._document_type_boost(normalized_query, item)
        freshness_score = 0.0
        if any(
            marker in normalized_query for marker in ("现在", "当前", "最新", "现行")
        ):
            current_text = title + content
            if any(marker in current_text for marker in ("现行", "最新版", "当前有效")):
                freshness_score = 0.25
        # Versioned, effective-dated rules are more authoritative than undated
        # examples when both are lexically relevant.
        versioned_score = 0.18 if str(item.get("effective_at", "")).strip() else 0.0
        return min(
            1.0,
            0.2 * ngram_score
            + 0.25 * title_score
            + 0.35 * tag_score
            + type_score
            + freshness_score
            + versioned_score,
        )

    @staticmethod
    def _longest_common_substring(left: str, right: str) -> int:
        if not left or not right:
            return 0
        previous = [0] * (len(right) + 1)
        longest = 0
        for left_char in left:
            current = [0]
            for index, right_char in enumerate(right, start=1):
                value = previous[index - 1] + 1 if left_char == right_char else 0
                current.append(value)
                longest = max(longest, value)
            previous = current
        return longest

    @staticmethod
    def _document_type_boost(query: str, item: Dict[str, Any]) -> float:
        document_type = str(item.get("document_type", ""))
        category = str(item.get("category", ""))
        rules = [
            (("规格", "参数", "产品资料", "卖点"), {"product_fact"}, {"product"}, 0.25),
            (("品牌语气", "品牌规范", "调性"), {"brand_guide"}, {"brand"}, 0.25),
            (
                ("活动规则", "活动限制", "优惠规则"),
                {"campaign_rule"},
                {"campaign"},
                0.25,
            ),
            (("优秀文案", "历史文案", "案例"), {"historical_copy"}, {"example"}, 0.3),
            (
                ("已发布", "发布了", "商城", "官方介绍"),
                {"published_content"},
                {"published"},
                0.3,
            ),
            (("客服", "怎么回答", "标准问答"), {"faq"}, {"customer_service"}, 0.3),
        ]
        for markers, types, categories, boost in rules:
            if any(marker in query for marker in markers):
                if document_type in types or category in categories:
                    return boost
        return 0.0

    @staticmethod
    def _normalize_search_text(value: Any) -> str:
        return re.sub(r"\s+", "", _SEARCH_CONVERTER.convert(str(value)).lower())

    @classmethod
    def _query_terms(cls, query: str) -> List[str]:
        normalized = cls._normalize_search_text(query)
        segments = re.findall(r"[a-z0-9][a-z0-9._-]*|[\u4e00-\u9fff]+", normalized)
        terms = set()
        for segment in segments:
            if re.fullmatch(r"[\u4e00-\u9fff]+", segment):
                max_size = min(6, len(segment))
                for size in range(2, max_size + 1):
                    terms.update(
                        segment[index : index + size]
                        for index in range(len(segment) - size + 1)
                    )
            elif len(segment) >= 2:
                terms.add(segment)
        return sorted(terms, key=lambda term: (-len(term), term))

    @property
    def doc_count(self) -> int:
        return self._collection.count()

    # ── MCP 工具 handler ─────────────────────────────────────────────────────

    async def search_handler(self, params: Dict[str, Any], context: Any) -> Any:
        """
        作为 MCP 工具的 handler 注册。

        MCPToolManager.register(Tool(
            name="knowledge_search",
            handler=kb.search_handler,
            ...
        ))
        """
        query = params.get("query", "")
        top_k = params.get("top_k", 5)
        filters = params.get("filters") or {}
        min_score = float(params.get("min_score", 0.0))
        if params.get("recall_mode") == "branches":
            import asyncio

            return await asyncio.to_thread(
                self.search_branches,
                query,
                int(params.get("branch_k", 20)),
                filters=filters,
                min_score=min_score,
            )
        import asyncio

        return await asyncio.to_thread(
            self.search, query, top_k=top_k, filters=filters, min_score=min_score
        )

    # ── 内部方法 ──────────────────────────────────────────────────────────────

    def _chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """将长文本按 chunk_size 切片，保留语义完整性（按句号/换行切分）。"""
        if len(text) <= chunk_size:
            return [text] if text.strip() else []

        chunks = []
        current = ""
        # 按句子切分
        sentences = text.replace("\n", "。").split("。")
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            if len(current) + len(sent) + 1 > chunk_size:
                if current:
                    chunks.append(current)
                current = sent
            else:
                current = f"{current}。{sent}" if current else sent

        if current:
            chunks.append(current)

        return chunks

    def _load_default_docs(self) -> None:
        """导入默认知识库文档（客服场景常见问题）。"""
        default_docs = [
            {
                "title": "退款政策",
                "content": (
                    "退款政策说明。"
                    "用户在购买后 7 天内可以申请无理由退款。"
                    "退款申请提交后，系统会在 1-3 个工作日内审核。"
                    "审核通过后，款项将在 5-7 个工作日内退回原支付账户。"
                    "如果商品已发货，需要先完成退货流程才能退款。"
                    "退货运费由用户承担，除非是商品质量问题。"
                    "超过 7 天但未超过 30 天的订单，需要提供商品质量问题的证据才能退款。"
                ),
            },
            {
                "title": "订单查询",
                "content": (
                    "订单查询指南。"
                    "用户可以通过订单号查询订单状态。"
                    "订单状态包括：待支付、已支付、已发货、运输中、已签收、已完成。"
                    "如果订单显示已发货但超过 7 天未收到，可以联系客服申请查件。"
                    "物流信息通常在发货后 24 小时内更新。"
                    "如果订单显示异常，请提供订单号联系客服处理。"
                ),
            },
            {
                "title": "账户安全",
                "content": (
                    "账户安全说明。"
                    "建议用户定期修改密码，密码长度至少 8 位，包含字母和数字。"
                    "如果忘记密码，可以通过绑定的手机号或邮箱重置。"
                    "发现账户异常登录时，系统会自动锁定账户并发送通知。"
                    "用户可以在安全设置中开启两步验证，提高账户安全性。"
                    "不要将密码分享给他人，客服人员不会索要用户密码。"
                ),
            },
            {
                "title": "技术故障排查",
                "content": (
                    "常见技术问题排查。"
                    "应用崩溃：请尝试清除缓存后重启应用，如果问题持续请更新到最新版本。"
                    "登录失败 401 错误：表示认证失败，请检查用户名密码是否正确，或尝试重置密码。"
                    "页面加载慢：检查网络连接，尝试切换 WiFi 或移动数据。"
                    "支付失败：确认银行卡余额充足，检查是否开启了网上支付功能。"
                    "500 服务器错误：这是服务端问题，请稍后重试，如果持续出现请联系技术支持。"
                ),
            },
            {
                "title": "会员与积分",
                "content": (
                    "会员积分规则。"
                    "每消费 1 元累积 1 积分。"
                    "积分可以在下次购物时抵扣，100 积分 = 1 元。"
                    "会员等级分为：普通会员、银卡会员（累计消费 1000 元）、金卡会员（累计消费 5000 元）。"
                    "银卡会员享受 95 折优惠，金卡会员享受 9 折优惠。"
                    "积分有效期为 1 年，过期自动清零。"
                    "生日当月消费可获得双倍积分。"
                ),
            },
            {
                "title": "配送说明",
                "content": (
                    "配送服务说明。"
                    "标准配送：3-5 个工作日送达，免运费（订单满 99 元）。"
                    "加急配送：1-2 个工作日送达，运费 15 元。"
                    "同城配送：当日达或次日达，运费 10 元。"
                    "偏远地区可能需要额外 2-3 天。"
                    "配送时间为每天 9:00-18:00，节假日可能延迟。"
                    "如果需要修改收货地址，请在发货前联系客服。"
                ),
            },
        ]
        self.add_documents(default_docs)
        logger.info(f"已导入默认知识库: {len(default_docs)} 篇文档")
