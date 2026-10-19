---
skill_id: compliance_review
name: 广告合规审核
version: 1.0.0
priority: 100
description: 基于冻结风险枚举和证据的营销内容审核规范
keywords: 合规,违规,广告法,审核,风险,禁用词,夸大,功效,最低价,排名
intents: compliance_check
agents: compliance
resources: rules.json,output_schema.json,examples.json
output_schema: output_schema.json
enabled: true
---

# 广告合规审核

## 能力边界

只审核用户提供的营销草稿，不主动生成整篇文案，不把一般表达优化扩大为发布风险。动态商品事实、活动规则和证明材料只能来自当前输入或当前任务的知识证据。

## 审核步骤

1. 从草稿中提取可核验主张及其原文片段。
2. 对需要依据的主张核对当前任务证据；不得依据模型常识推定其真实。
3. 使用 `rules.json` 中的固定规则 ID、严重度和决策矩阵分类。
4. 主观体验表达与客观宣传承诺必须结合语境区分，例如“我最喜欢的一点”本身不是最高级宣传。
5. 输出必须符合 `output_schema.json`，每个风险项必须定位原文、说明依据并给出局部修改建议。

## 决策原则

- 没有风险项时为 `pass`。
- 存在可通过删改修复的中低风险时为 `changes_required`。
- 命中规则矩阵中的阻断风险时为 `blocked`。
- 证据不足时记录在 `missing_evidence`，不得虚构证据引用。
