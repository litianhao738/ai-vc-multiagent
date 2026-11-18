import copy
import tempfile
import unittest

from vcagents import CompanyInput, SectorPipeline, SectorRequest, Settings, save_result
from vcagents.financials import calculate_deal, calculate_metrics, calculate_returns
from vcagents.report import render_memo
from vcagents.report_checks import publication_issues


class PublicationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = SectorPipeline(Settings(mock=True, holding_years=5)).run(
            SectorRequest("aigc-content", [CompanyInput("Higgsfield", post_money_musd=5400, investment_musd=10)]))

    def test_old_conclusions_scores_and_citations_cannot_be_published(self):
        result = copy.deepcopy(self.base)
        result.decision["action"] = "继续尽调"
        result.sector_brief["summary"] = "**投资结论：建议投资。综合加权得分 8.30 / 10。** 收入10亿美元[E-4]。"
        result.analyses["Higgsfield"]["summary"] = "经营增长，建议投资，交割前复核。"
        issues = publication_issues(result, render_memo(result))
        for text in ("E-4", "本次建议", "8.30", "研究摘要"):
            self.assertTrue(any(text in issue for issue in issues), issues)
        with tempfile.TemporaryDirectory() as root:
            folder = save_result(result, root)
            self.assertFalse((folder / "report.md").exists())
            self.assertIn("不能作为正式报告", (folder / "research-draft.md").read_text(encoding="utf-8"))
            self.assertIn("E-4", (folder / "audit.md").read_text(encoding="utf-8"))

    def test_disclosed_nrr_cannot_be_called_undisclosed(self):
        result = copy.deepcopy(self.base)
        result.sector_brief["evidence"] = [{"evidence_id": "E-S-1", "claim": "Higgsfield公司披露NRR为300%",
            "excerpt": "net revenue retention of 300 percent", "source_type": "公司口径", "as_of": "2026-10",
            "source_ref": "[S-1]", "source_linked": True, "excerpt_verified": True}]
        result.sector_brief["sources"] = ["[S-1] https://example.org/nrr"]
        result.decision["headline"] = "毛利和留存都没有披露。"
        self.assertTrue(any("净收入留存" in issue for issue in publication_issues(result, render_memo(result))))
        result.decision["headline"] = "据公司披露NRR为300%[E-S-1]；没有具体毛利率和付费客户队列。"
        self.assertFalse(any("已有净收入留存披露" in issue for issue in publication_issues(result, render_memo(result))))

    def test_baseline_loss_does_not_support_a_safety_margin(self):
        result = copy.deepcopy(self.base)
        result.returns = calculate_returns(5400, .5, [{"scenario": "基线", "exit_valuation_musd": 6000,
                                                     "holding_years": 5}])["scenarios"]
        result.decision["valuation_view"] = "进入价格有安全边际。"
        self.assertTrue(any("安全边际" in issue for issue in publication_issues(result, render_memo(result))))
        result.decision["valuation_view"] = "基线不足以回收本金，不能证明价格存在安全边际。"
        self.assertFalse(any("安全边际" in issue for issue in publication_issues(result, render_memo(result))))

    def test_run_rate_is_not_relabelled_as_recurring_revenue(self):
        metrics = calculate_metrics({"annualized_revenue_musd": 700})
        deal = calculate_deal(5400, 10, 400, {"annualized_revenue_musd": 700})
        self.assertIsNone(metrics["arr_musd"])
        self.assertIsNone(deal["implied_arr_multiple"])
        self.assertAlmostEqual(deal["implied_run_rate_multiple"], 5400 / 700)
        self.assertIn("拟持股比例：0.185%", render_memo(self.base))

    def test_editor_receives_actual_returns_and_cannot_change_the_exit_plan(self):
        pipeline = SectorPipeline(Settings(mock=True, holding_years=5))
        original = pipeline._run
        tasks = {}

        def run(role, task, toolbox, agent_id, name, schema=None):
            tasks[agent_id] = task
            output = original(role, task, toolbox, agent_id, name, schema)
            if agent_id == "chair":
                for scenario, value in zip(output["exit_scenarios"], (21000, 6000, 900)):
                    scenario["exit_valuation_musd"] = value
            if agent_id == "chair_editor":
                self.assertNotIn("exit_scenarios", schema["properties"]["decision"]["properties"])
                self.assertIn('"multiple": 0.5556', task)
                self.assertIn('"holding_years": 5', task)
                output["decision"]["action"] = "继续尽调"
                output["decision"]["valuation_view"] = "基线无法回收本金，当前不支持投资。"
            return output

        pipeline._run = run
        result = pipeline.run(SectorRequest("aigc-content", [CompanyInput("Higgsfield", post_money_musd=5400)]))
        self.assertEqual([s["exit_valuation_musd"] for s in result.decision["exit_scenarios"]], [21000, 6000, 900])
        self.assertEqual(result.returns[1]["multiple"], .5556)
        self.assertEqual(result.decision["action"], "继续尽调")
        self.assertIn("当前不支持投资", tasks["fact_checker_decision"])

