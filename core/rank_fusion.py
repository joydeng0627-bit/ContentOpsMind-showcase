"""Deterministic reciprocal-rank fusion for governed retrieval lists."""

import hashlib
import json


def item_key(item):
    stable = item.get("document_id")
    if stable:
        identity = [
            item.get("corpus_id", ""),
            stable,
            item.get("chunk", 0),
            item.get("version", ""),
        ]
    else:
        identity = [
            item.get("corpus_id", ""),
            item.get("title", ""),
            item.get("content", ""),
        ]
    return hashlib.sha256(
        json.dumps(identity, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def reciprocal_rank_fusion(rankings, k=60):
    if type(k) is not int or k < 1:
        raise ValueError("RRF k must be a positive integer")
    scores, items = {}, {}
    for ranking in rankings:
        seen = set()
        for rank, item in enumerate(ranking, 1):
            key = item_key(item)
            if key in seen:
                continue
            seen.add(key)
            items.setdefault(key, item)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
    return [
        dict(items[key], rrf_score=scores[key])
        for key in sorted(scores, key=lambda key: (-scores[key], key))
    ]
