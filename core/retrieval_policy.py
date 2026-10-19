import re

GUIDANCE_TYPES = {"style_guide", "template", "brand_guide", "channel_guide"}
GUIDANCE_CATEGORIES = {"style", "template", "brand", "channel"}
HISTORICAL = ("历史", "当时", "公告中", "2026年4月", "2026-04", "曾经")
KNOWN_BRANDS = ("星巴克", "Starbucks", "瑞幸", "喜茶", "奈雪", "澄光")


def target(query):
    brand = next((b for b in KNOWN_BRANDS if b.lower() in query.lower()), "")
    if brand.lower() == "starbucks":
        brand = "星巴克"
    if not brand and any(
        w in query.lower().replace(" ", "") for w in ("高蛋白拿铁pro",)
    ):
        brand = "星巴克"
    product = (
        "starbucks-pro"
        if brand == "星巴克"
        and any(w in query.lower() for w in ("高蛋白", "拿铁pro", "拿铁 pro"))
        else ""
    )
    return brand, product


def is_guidance(item):
    return (
        item.get("document_type") in GUIDANCE_TYPES
        or item.get("category") in GUIDANCE_CATEGORIES
    )


def eligible(query, item, filters):
    if filters.get("_scope") != "real":
        return True
    guidance = is_guidance(item)
    if item.get("document_type") in {"historical_copy", "example"}:
        return False
    if guidance:
        brand, _product = target(filters.get("_target_query", query))
        item_brand = str(item.get("brand", ""))
        # Real-scope conversations may use brand-neutral writing guidance, but
        # must not even surface another brand's rules in the candidate audit.
        if item_brand and (not brand or brand.lower() not in item_brand.lower()):
            return False
        return filters.get("_lane") != "business"
    if filters.get("_lane") == "guidance":
        return False
    if not str(item.get("corpus_id", "")).startswith("real-"):
        return False
    original = filters.get("_target_query", query)
    if item.get("status") in {"historical", "expired"} and not any(
        w in original for w in HISTORICAL
    ):
        return False
    brand, product = target(original)
    if brand and brand.lower() not in str(item.get("brand", "")).lower():
        return False
    if product and item.get("product_id") != product:
        return False
    # No open-ended brand/product inference: fail closed for known-brand requests
    # outside the single reviewed product family supported by this release.
    if not product:
        return False
    return True


def evidence_status(items):
    if any(i.get("document_type") == "product_fact" for i in items):
        return "product_supported"
    return "guidance_only" if any(is_guidance(i) for i in items) else "no_match"
