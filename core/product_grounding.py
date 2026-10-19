"""Narrow output safeguards for the reviewed PRO publication, not a legal approval."""

import re


def qualify_pro_publication(text, citations):
    if not any(
        "starbucks-high-protein-latte-pro" in str(i.get("source_url", ""))
        for i in citations
    ):
        return text
    lines = []
    for line in text.splitlines():
        if line.lstrip().startswith("> 说明：标题"):
            continue
        if re.search(r"喝过的人|亲测|奶香先到|咖啡随后", line):
            line = "官方发布资料描述的风味为清甜奶香、丝滑绵密、醇厚柔和。"
        if "三分之一" in line and "NRV" not in line:
            line += "（蛋白质日需量参考GB 28050营养素参考值NRV）"

        if re.search(r"20\s*(?:克|g)", line, re.I) and not (
            "标准配方" in line and re.search(r"大杯.*热", line)
        ):
            line += "（按标准配方计，大杯热饮约含20克蛋白质）"
        # Correct a model-generated self-audit claiming numeric titles are absent.
        line = line.replace(
            "标题中未写入营养数字，如需在标题体现，请同步补全杯型与冷热限定。",
            "标题与正文中的营养数字均应保留标准配方、大杯热饮限定。",
        )
        lines.append(line)
    return "资料时间说明：以下为基于星巴克中国2026年4月7日公告的产品介绍草稿，不证明当前在售或优惠有效。\n\n" + "\n".join(
        lines
    )
