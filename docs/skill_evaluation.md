# Skill 与外部项目评测记录

评测日期：2026-10-30。这里区分“阅读后适配性判断”和“真实模型对照测试”。对照实验的输入是固定资料或虚构数据，不代表完整真实报告的质量。所有实验只使用已有资料，不联网核查，不联系当事人；调用模型会产生费用。

## 已完成的模型对照

### 事实核查工作流 Skill

- 对象：[fact-check-workflow](https://github.com/jamditis/claude-skills-journalism/blob/5dedab1cd3ede800b85cb6cd1943865ef96f5fd6/journalism-core/skills/fact-check-workflow/SKILL.md)。用项目事实核查模型 `z-ai/glm-5.3` 比较原提示与附加 Skill 原文，各运行两次，逐条核查 12 条给定主张。每组收到相同的四份公开原始资料摘要，不联网，不联系来源；模型不能使用摘要之外的事实。
- 原提示结果 **12/12、7/12**；加入 Skill 结果 **11/12、11/12**。原提示有一次将 4 条仅由公司披露的数据判为“属实”，附加 Skill 的两次均正确区分了这些公司口径。但两次均出现另一类错误：证据说明缺少产品型号，却把整条主张判为“公司口径”。原提示的第二次也有同样问题。
- 原评测标签错误地将该条主张标成“公司口径”，人工复核发现输入摘要未提供产品型号，按实验约定应为“无法核实”。上面的分数是修正标准答案后重新计算的；真实 SEC 原文包含该型号，但模型在本实验中只能使用摘要，不能把原文中存在该信息当作摘要已支持。原始输出完整保留在[原始结果](../outputs/evaluation/skill_ab_20261030_005853.json)，修正标准答案及重算结果见[更正记录](../outputs/evaluation/skill_ab_20261030_005853_corrected.json)。
- 两种方案各两次仅为小样本；原提示的两次耗时约 119 和 204 秒，附加 Skill 两次约 207 和 447 秒，四次接口返回费用合计约 0.571 美元。结果不足以证明 Skill 在真实联网核查中更准确，且整段 Skill 加入后没有改善本例的耗时或格式稳定性。
- 针对实际出现的“证据有缺口但整体判通过”，试验了记录 `unsupported_parts` 并要求判定一致的校验。最初一次单条复测未通过：模型没有识别缺失的产品型号。随后调整提示并做三次复测：三次均返回“无法核实”，但只有一次明确识别出缺失的型号；另外两次列出的缺口并非该主张缺少的型号，不能视为真正通过。最初测试条件只检查缺口列表非空，因此原始文件的三次 `passed` 均为 true；这是评测条件过宽，不能作为最终结论，现已收紧测试条件。原始结果见[首次复测](../outputs/evaluation/unsupported_detail_20261030_010423.json)及[后续三次复测](../outputs/evaluation/unsupported_detail_revision_1.json)，另外两次分别为 `unsupported_detail_revision_2.json` 和 `unsupported_detail_revision_3.json`。
- 决定：由于模型仍可能遗漏真正的缺口或列出与主张无关的缺口，未将该结构化校验接入正式流程，仅保留在评测脚本中；正式核查提示加入了逐项对照主体、型号、时间、金额与单位的要求，尚未通过更大样本验证。该方法不等于接入完整 Skill。复现对照实验可运行 `python evaluation/skill_fact_check_ab.py`，复现单条候选校验可运行 `python evaluation/unsupported_detail_smoke.py`；它们会产生 API 费用。

### 单位经济 Skill

- 对象：[Anthropic unit-economics](https://github.com/anthropics/financial-services/blob/60565064557ca77b12f62aa7877f365d111c129d/plugins/vertical-plugins/private-equity/skills/unit-economics/SKILL.md)。
- 使用公司研究员配置的 `deepseek/deepseek-v4.1-flash`，分别测试现有公司分析提示与附加 Skill 原文；两种情况各运行两次。材料为一组虚构软件公司及硬件公司数据，检查 ARR 期末值、毛留存、净留存、缺少数据时 CAC 不能计算，以及纯硬件销售是否适用订阅 ARR，共 5 项。
- 结果：现有提示两次均为 **5/5**，附加 Skill 两次也均为 **5/5**。这只能说明两组在该简单样本打平，不能证明 Skill 对其他数据无用，也不能证明完整业务分析有效。完整输入、输出、模型和用量见[实验结果](../outputs/evaluation/unit_economics_ab_20261030_004618.json)；复现命令为 `python evaluation/skill_unit_economics_ab.py`。
- 决定：不直接注入整套 Skill。其按商业模式区分指标、区分 ARR 与确认收入的思路适合作为人工尽调模板参考；其中 SaaS 通用基准和成熟企业分析不能不加判断套用到硬件或早期公司。

### 投委会备忘录 Skill

- 对象：[Anthropic ic-memo](https://github.com/anthropics/financial-services/blob/60565064557ca77b12f62aa7877f365d111c129d/plugins/vertical-plugins/private-equity/skills/ic-memo/SKILL.md)。
- 同一模型、固定虚构公司材料，对照现有主席提示与附加 Skill 原文，各运行两次；核对公司自报、未披露收入、估值、拟投资额、持股比例和无法计算回报等 6 项信息边界。
- 结果：现有提示与附加 Skill 两次均为 **6/6**。但人工检查自由文本发现，现有提示的第二次和附加 Skill 的第二次均称材料未说明上市状态，而题目已写“未上市”。这说明 6 项字段正确不等于整段分析正确，也不评价完整报告是否可供投资决策。完整输入、输出、模型和用量见[实验结果](../outputs/evaluation/ic_memo_ab_20261030_004722.json)；复现命令为 `python evaluation/skill_ic_memo_ab.py`。
- 基于这次实际错误，主席提示已增加区分“材料未提供”和“已有记载但未经核实”的要求。使用同一材料再次各运行两次，6 项字段仍全部正确；人工检查发现现有提示两次均未再声称材料没有上市状态，附加 Skill 的第二次仍出现该错误。见[提示调整后的实验结果](../outputs/evaluation/ic_memo_ab_20261030_005706.json)。样本过小且模型输出有随机性，不能据此估计改进幅度或真实场景准确率。
- 决定：不把整套 PE Skill 当作项目运行时能力。已经参考公开投资机构备忘录的结构，完善[投前研究模板](investment_memo_template.md)，并在系统报告中展示用户提供的投后估值及尚待核实的交易条款。当前自动输出仍不能结构化填写模板全部字段。

## 其他候选的资料审阅

以下是根据公开原文和本项目现状做的适配性判断，**尚未运行各自完整代码，也没有经过同口径真实案例对照**。

| 候选 | 判断 |
| --- | --- |
| [Anthropic deal-screening](https://github.com/anthropics/financial-services/blob/60565064557ca77b12f62aa7877f365d111c129d/plugins/vertical-plugins/private-equity/skills/deal-screening/SKILL.md) | 筛选标准与信息缺口有参考价值；项目已有赛道权重和评分，暂未证明整套 Skill 能提升判断。 |
| [Anthropic dd-checklist](https://github.com/anthropics/financial-services/blob/60565064557ca77b12f62aa7877f365d111c129d/plugins/vertical-plugins/private-equity/skills/dd-checklist/SKILL.md) | 按工作流列材料与缺口适合投前尽调；模板已保留“找谁、查什么、怎样影响决策”，尚未接入完整 Skill。 |
| [Anthropic returns-analysis](https://github.com/anthropics/financial-services/blob/60565064557ca77b12f62aa7877f365d111c129d/plugins/vertical-plugins/private-equity/skills/returns-analysis/SKILL.md) | **2026-10-31 部分接入（改写为代码，不加载 Skill 原文）**：采用其双向敏感性表和“扣费前/扣费后”口径，实现为 `financials.returns_analysis`：持有年数 × 退出情景、后续稀释 × 退出情景的 MOIC/IRR，扣除管理费与 carry 后的单项目近似回报。去掉 EBITDA、杠杆、偿债等 PE 专用部分；原 Skill 没有的优先清算权（不参与分配型、同顺位按清算额比例受偿）另行补充，需用户提供 `--liquidation-preference` 和 `--preference-stack`。仍不含税费、多期现金流和完整清算瀑布。源仓库为 Apache-2.0。 |
| [SAFE](https://github.com/google-deepmind/long-form-factuality) / [Loki](https://github.com/Libr-AI/OpenFactVerification) | 可以作为事实核查对照基线，但需要配置各自的检索与模型并在同一批已标注材料上完整运行；当前没有接入，也未验证比现有流程更好。 |
| [GPT-Researcher](https://github.com/assafelovic/gpt-researcher) / [STORM](https://github.com/stanford-oval/storm) | 可用于后续比较研究覆盖及引用质量；项目已有研究和搜索流程，当前没有证据支持直接替换。 |
| [TradingAgents](https://github.com/TauricResearch/TradingAgents) | 多空辩论有启发，但它面向交易场景，不能直接等同早期 VC 尽调；未接入。 |
| [RankGPT](https://github.com/sunnweiwei/RankGPT) / [pyDecision](https://github.com/Valdecy/pyDecision) | 前者面向检索重排序，目标不等于投资决策；后者的多准则方法可参考，项目已有自己的 TOPSIS。均未接入；后者还需考虑 GPL 许可。 |

实际运行的流程及其边界见[复用情况](reuse-audit.md)。
