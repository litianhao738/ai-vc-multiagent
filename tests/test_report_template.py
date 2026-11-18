import copy
import json
import unittest
from pathlib import Path

from vcagents import CompanyInput, ExpandedSectorPipeline as SectorPipeline, SectorRequest, Settings
from vcagents.brief import render_brief
from vcagents.report import render_memo

ROOT = Path(__file__).resolve().parent.parent


def load_example(name: str) -> dict:
    return json.loads((ROOT / "examples" / name).read_text(encoding="utf-8"))


class ReportTemplateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        companies = [CompanyInput("甲", post_money_musd=50, investment_musd=5, round_size_musd=10,
                                  financial_data=load_example("financial_inputs.example.json")),
                     CompanyInput("乙"), CompanyInput("丙")]
        cls.base = SectorPipeline(Settings(mock=True)).run(
            SectorRequest("aigc-content", companies, market_data=load_example("market_inputs.example.json")))

    def with_top(self, name="甲"):
        result = copy.deepcopy(self.base)
        result.decision.update(top_pick=name, recommended=[name], action="立项")
        pipeline = SectorPipeline(Settings(mock=True))
        pipeline.companies = {"甲": CompanyInput("甲", post_money_musd=50, investment_musd=5)}
        result.returns = pipeline._returns(result.decision)["scenarios"]
        return result

    def test_blank_fields_after_cleanup_use_neutral_text_not_undisclosed(self):
        result = copy.deepcopy(self.base)
        result.analyses["甲"]["team_assessment"] = "待核实"
        result.fact_sheets["甲"].update(listing_status="待核实", listing_evidence="待核实")
        report = render_memo(result)
        self.assertIn("**团队**\n现有资料未覆盖", report)
        self.assertIn("上市状态：未确认 · 本轮投后估值 50 百万美元（交易假设），隐含 ARR 倍数 41.67 倍", report)
        self.assertNotIn("（）", report)
        self.assertNotIn("上市状态：未披露", report)

    def test_risks_precede_exits_and_turns_stay_in_the_audit(self):
        from vcagents.report import render_audit
        report = render_memo(self.base)
        self.assertLess(report.index("### 主要风险与应对"), report.index("### 退出情景与回报"))
        self.assertNotIn("### 多空辩论", report)
        audit = render_audit(self.base)
        self.assertIn("### 多空辩论", audit)
        for turn in self.base.debate:
            self.assertIn(turn["name"], audit)
        exit_part = report.split("### 退出情景与回报", 1)[1].split("## 七、", 1)[0]
        self.assertNotIn("####", exit_part)

    def test_metric_list_has_no_empty_missing_clause_and_follows_sector(self):
        report = render_memo(self.base)
        self.assertNotIn("无法计算：。", report)
        self.assertNotIn("无法计算。", report)
        result = SectorPipeline(Settings(mock=True)).run(SectorRequest("embodied-ai", [
            CompanyInput("甲", financial_data=load_example("hardware_inputs.example.json")), CompanyInput("乙"), CompanyInput("丙")]))
        report = render_memo(result)
        for text in ("交付验收率", "80.0%", "量产良率", "88.0%", "单台毛利率", "30.0%", "回款率", "75.0%"):
            self.assertIn(text, report)
        for saas in ("净收入留存率 NRR", "Burn multiple", "LTV / CAC"):
            self.assertNotIn(saas, report.split("## 七、", 1)[0])

    def test_dilution_default_is_labeled_and_user_setting_is_distinguished(self):
        self.assertIn("后续持股稀释：50%（研究假设", render_memo(self.base))
        result = SectorPipeline(Settings(mock=True, exit_dilution=0.36)).run(
            SectorRequest("aigc-content", [CompanyInput(n) for n in ("甲", "乙", "丙")]))
        self.assertIn("后续持股稀释：36%（交易假设）", render_memo(result))
        self.assertEqual(result.return_summary["dilution_source"], "用户设定")

    def test_deal_terms_and_market_size_reach_the_report(self):
        report = render_memo(self.with_top())
        opening = report.split("## 二、", 1)[0]
        for text in ("本轮投后估值：50 百万美元", "拟投资金额：5 百万美元", "拟持股比例：10.0%", "本轮融资额：10 百万美元"):
            self.assertNotIn(text, opening)
            self.assertIn(text, report.split("### 交易条件与估值", 1)[1])
        for text in ("| 投前估值 | 40 百万美元 |", "| 隐含 ARR 倍数 | 41.67 倍 |", "| 本轮后现金跑道 | 120 月 |"):
            self.assertIn(text, report)
        self.assertNotIn("未提供拟投资金额", report)
        market = report.split("## 二、", 1)[1].split("## 三、", 1)[0]
        for text in ("| 目标客户数 | 1000 家 |", "| TAM（自下而上） | 10 百万美元 |", "| TAM（自上而下） | 12 百万美元 |",
                     "| SOM | 1 百万美元 |", "之比为 0.83"):
            self.assertIn(text, market)

    def test_user_source_notes_are_not_treated_as_model_claims(self):
        result = copy.deepcopy(self.base)
        result.financial_metrics["甲"]["sources"]["cash_musd"] = "BP 第 3 页，未经独立核实"
        self.assertIn("BP 第 3 页，未经独立核实", render_memo(result))

    def test_no_suitable_target_is_a_valid_decision(self):
        result = copy.deepcopy(self.base)
        result.decision.update(action="不投", recommended=[])
        report = render_memo(result)
        self.assertIn("**本次建议：不投**", report)
        self.assertIn("**重点跟进**：暂无合适标的", report)
        self.assertIn(f"## 六、重点分析对象：{result.decision['top_pick']}（本次不建议投资）", report)

    def test_scoring_details_are_in_appendix(self):
        report = render_memo(self.base)
        body, appendix = report.split("## 七、", 1)
        for text in ("**TOPSIS 交叉检查**", "**评分维度与权重**", "| 风险可控度 | 10% |", "**排名敏感性**"):
            self.assertNotIn(text, body)
            self.assertIn(text, appendix)
        self.assertIn("排名反映赛道内的相对位置", body)

    def test_inconsistent_deal_inputs_fail_before_agent_calls(self):
        cases = [(CompanyInput("甲", investment_musd=20, round_size_musd=10), "不能超过本轮融资额"),
                 (CompanyInput("甲", post_money_musd=10, round_size_musd=10), "必须小于投后估值"),
                 (CompanyInput("甲", investment_musd=-1), "必须大于零")]
        for company, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    SectorPipeline(Settings(mock=True)).run(
                        SectorRequest("aigc-content", [company, CompanyInput("乙"), CompanyInput("丙")]))
        with self.assertRaisesRegex(ValueError, "--financial-data"):
            SectorPipeline(Settings(mock=True)).run(SectorRequest(
                "aigc-content", [CompanyInput(n) for n in ("甲", "乙", "丙")],
                market_data=load_example("financial_inputs.example.json")))


class SingleCompanyReportTest(unittest.TestCase):
    def test_brief_orders_a_single_company_thesis_and_preserves_archived_inputs(self):
        result = SectorPipeline(Settings(mock=True)).run(
            SectorRequest("embodied-ai", [CompanyInput("岱川机器人", post_money_musd=80)]))
        subtitle = "减速器标定的复用能力与维护成本"
        bridges = {
            "customer": "采购方关注产线停机时间。",
            "technology": "客户要求不停线，控制参数能否迁移决定改线工时。",
            "economics": "节省的工时能否计入服务费，决定公司能捕获多少收益。",
            "valuation": "已签服务合同可用于估值基准，增量利润仍需按情景测算。",
            "risks": "估值假设最依赖续签，需要核对同批合同。",
        }
        result.decision.update(report_subtitle=subtitle, section_openings=bridges)
        brief = render_brief(result)
        self.assertTrue(brief.startswith(f"# 岱川机器人 投资报告\n\n{subtitle}\n\n"))
        chapters = [part.split("\n", 1) for part in brief.split("\n## ")[1:]]
        self.assertEqual([heading for heading, _ in chapters], [
            "一、投资摘要", "二、公司概况与客户需求", "三、核心技术与竞争壁垒", "四、商业模式与经营质量",
            "五、估值与回报情景", "六、关键风险与尽调优先级", "七、资料来源与测算口径"])
        for (_, body), lead in zip(chapters[1:6], bridges.values()):
            self.assertTrue(body.strip().startswith(lead))
        for i, content in ((0, "核心判断"), (2, "投资逻辑与证据"), (3, "财务与单位经济"),
                           (4, "本次建议"), (4, "悲观"), (5, "主要风险与应对"), (5, "尽调计划"), (6, "方法与假设")):
            self.assertIn(content, chapters[i][1])
        self.assertNotIn("本次建议", chapters[0][1])
        self.assertIn(result.analyses["岱川机器人"]["summary"], chapters[1][1])
        self.assertIn("**长期竞争力**", chapters[6][1])
        result.decision.pop("section_openings")
        result.decision.pop("report_subtitle")
        legacy = render_brief(result)
        self.assertNotIn(subtitle, legacy)
        self.assertIn("## 七、资料来源与测算口径", legacy)
        self.assertIn(result.decision["thesis"], legacy)
        result.analyses.clear()
        result.ranking.clear()
        self.assertIn(result.fact_sheets["岱川机器人"]["one_liner"], render_brief(result))

    def test_single_company_report_compares_with_named_rivals_and_answers_six_questions(self):
        from vcagents import CompanyInput, ExpandedSectorPipeline as SectorPipeline, SectorRequest, Settings
        from vcagents.report import render_memo

        request = SectorRequest("aigc-content", [CompanyInput("Liblib")])
        request.companies[0].post_money_musd, request.companies[0].investment_musd = 2000, 10
        pipeline = SectorPipeline(Settings(mock=True, holding_years=5))
        result = pipeline.run(request)
        result.comparison["per_company"] = [
            {"company": "Liblib", "position": "目标公司", "edge": "收入规模", "gap": "集团口径"},
            {"company": "PixVerse", "position": "对手", "edge": "用户多", "gap": "变现弱"}]
        memo = render_memo(result)
        self.assertEqual(memo.splitlines()[0], "# Liblib 投资报告")
        self.assertEqual(render_brief(result).splitlines()[0], "# Liblib 投资报告")
        self.assertIn("## 三、竞争对比：为什么是这家", memo)
        self.assertIn("| PixVerse | 对手 | 用户多 | 变现弱 |", memo)
        self.assertNotIn("排序不确定性", memo)
        for question in ("建议是什么、投多少", "为什么是这家、为什么是现在", "最可能让我们亏钱的是什么",
                         "最好和最差的情况", "哪些是事实、哪些是推断", "接下来要核实什么"):
            self.assertIn(f"| {question} |", memo)
        self.assertIn("拟持股比例 0.5%", memo)

    def test_two_named_companies_get_a_comparison_report(self):
        request = SectorRequest("aigc-content", [CompanyInput("甲"), CompanyInput("乙")], max_companies=2)
        result = SectorPipeline(Settings(mock=True)).run(request)
        memo = render_memo(result)
        self.assertEqual(result.companies, ["甲", "乙"])
        self.assertIn("研究对象：甲、乙", memo)
        self.assertIn("## 三、公司比较与排序", memo)


if __name__ == "__main__":
    unittest.main()
