# ContentOpsMind Skills

这些 Skills 会在 Agent 调用 LLM 时按 Agent 类型、意图和关键词动态装配，用来约束业务边界。
目录中的 `SKILL.md` 是能力入口；规则、输出 Schema 和示例属于显式声明的资源，不会被当成独立 Skill。

- brand_style: 品牌语气与风格
- campaign_rules: 活动规则校验
- compliance: 广告合规与禁用词
- channel_templates: 小红书/公众号/短信/Push 渠道规范

合规 Skill 是版本化规则包：`SKILL.md` 描述审核步骤，`rules.json` 保存唯一风险枚举与决策矩阵，
`output_schema.json` 约束结构化结果，`examples.json` 保存少量人工确认的边界案例。

通过 `GET /skills` 查看版本和资源校验状态，通过 `POST /skills/reload` 原子热更新；若新配置校验失败，
服务继续使用上一份有效快照。
