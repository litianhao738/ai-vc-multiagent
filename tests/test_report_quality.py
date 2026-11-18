import time
import unittest
from unittest.mock import Mock, patch

from vcagents import CompanyInput, DocumentStore, ExpandedSectorPipeline as SectorPipeline, SectorRequest, Settings
from vcagents import schemas
from vcagents.agent import Event
from vcagents.llm import ToolCall, Turn
from vcagents.pipeline import AnalysisResult, _namespace_sources
from vcagents.report import render_memo
from vcagents.roles import CHAIR, FACT_CHECKER
from vcagents.sectors import load_sector
from vcagents.tools import Toolbox


class ReportQualityTest(unittest.TestCase):
    def test_user_inputs_reach_calculations_and_report_without_model_arithmetic(self):
        data = {"currency": "USD", "period": "2025", "previous_period": "2024",
                "inputs": {"revenue_current_musd": 1.5, "revenue_previous_musd": 1, "direct_cost_musd": 0.6},
                "sources": {"revenue_current_musd": "用户 BP 第3页", "revenue_previous_musd": "用户 BP 第3页", "direct_cost_musd": "用户 BP 第4页"}}
        companies = [CompanyInput(n, post_money_musd=50, financial_data=data) for n in ("甲", "乙", "丙")]
        pipeline = SectorPipeline(Settings(mock=True, exit_dilution=0.36, holding_years=6,
                                           scenario_probabilities={"乐观": 0.2, "基线": 0.5, "悲观": 0.3}))
        original = pipeline._run

        def run(role, *args, **kwargs):
            output = original(role, *args, **kwargs)
            if role == CHAIR and "exit_scenarios" in output:
                for scenario, value in zip(output["exit_scenarios"], (1500, 400, 0)):
                    scenario["exit_valuation_musd"] = value
                    scenario["valuation_basis"] = "固定测试假设"
            return output

        pipeline._run = run
        result = pipeline.run(SectorRequest("aigc-content", companies))
        self.assertEqual(result.return_summary["expected_multiple"], 6.4)
        self.assertEqual([r["multiple"] for r in result.returns], [19.2, 5.12, 0])
        self.assertEqual(result.returns[-1]["annualized_irr"], -1)
        self.assertAlmostEqual(result.financial_metrics["甲"]["metrics"]["gross_margin"], 0.6)
        report = render_memo(result)
        for text in ("6.4 倍", "60.0%", "-100.0%", "用户 BP 第3页", "简化年化 IRR"):
            self.assertIn(text, report)
        self.assertFalse(result.pairwise)
        self.assertEqual(len(result.debate), 2)
        result.fact_check["claims"] = [{"claim": "未核实主张", "source_agent": "research", "original_source": "",
                                       "evidence_ids": [], "verdict": "无法核实", "evidence": "缺原文",
                                       "correction": "未经证实的更正不能作为正确说法"}]
        self.assertNotIn("未经证实的更正不能作为正确说法", pipeline._chair_task(result))
        self.assertNotIn("未经证实的更正不能作为正确说法", render_memo(result))

    def test_bad_user_probabilities_fail_before_agent_calls(self):
        events: list[Event] = []
        pipeline = SectorPipeline(Settings(mock=True, scenario_probabilities={"乐观": 0.5}), emit=events.append)
        with self.assertRaisesRegex(ValueError, "情景概率"):
            pipeline.run(SectorRequest("aigc-content", [CompanyInput(n) for n in ("甲", "乙", "丙")]))
        self.assertFalse(events)

    def test_namespace_does_not_rewrite_matched_original_excerpt(self):
        data = {"sources": ["[1] https://example.com"], "evidence": [{"source_ref": "[1]", "excerpt": "表[1]显示39亿美元", "excerpt_verified": True}]}
        result = _namespace_sources(data, "C1")
        self.assertEqual(result["evidence"][0]["excerpt"], "表[1]显示39亿美元")
        self.assertTrue(result["evidence"][0]["excerpt_verified"])

    def test_matched_full_url_remains_linked_after_namespacing(self):
        toolbox = Toolbox()
        with patch("vcagents.tools.fetch_url", return_value="标题：公告\nURL：https://example.com/a\n\nrevenue 39 million"):
            toolbox.execute(ToolCall("1", "fetch_url", {"url": "https://example.com/a"}))
        data = {"sources": ["[1] https://example.com/a"],
                "evidence": [{"source_ref": "https://example.com/a", "excerpt": "revenue 39 million", "source_type": "未知"}]}
        result = _namespace_sources(toolbox.verify_evidence(data), "C1")
        self.assertTrue(result["evidence"][0]["excerpt_verified"])
        self.assertTrue(result["evidence"][0]["source_linked"])

    def test_pipeline_rejects_repeatedly_unread_verification_quote(self):
        pipeline = SectorPipeline(Settings(openrouter_api_key="test"))
        pipeline.sector, pipeline.companies = load_sector("aigc-content"), {}
        fact = {"subject": "公司", "fact_type": "收入或ARR", "statement": "收入390", "quantities": [], "entities": [],
                "date": {"text": "", "basis": "未给出"}, "quote": {"source": "W1", "from": 1, "to": 1},
                "date_quote": {"source": "", "from": 0, "to": 0}}
        claim = {"target": "收入390", "source_agent": "research", "original_source": "来源", "evidence_ids": [],
                 "fact": fact, "support": "完全支持", "verdict": "属实", "evidence": "模型称已证实", "correction": ""}
        session = Mock()
        session.send.return_value = Turn(tool_calls=[ToolCall("1", "submit_result", {"claims": [claim], "summary": "核查"})])
        session.input_tokens, session.output_tokens, session.cost_usd = 1, 1, 0
        llm = Mock()
        llm.session.return_value = session
        pipeline.router = Mock()
        pipeline.router.for_role.return_value = llm
        pipeline.last_result = AnalysisResult("aigc-content", "AIGC")
        data = pipeline._run(FACT_CHECKER, "测试", Toolbox(), "fact_checker", "核查",
                             schemas.fact_check_schema(["research"], []))
        self.assertEqual(session.send.call_count, 3)
        self.assertEqual(data["claims"][0]["verdict"], "无法核实")  # 没读到原文的主张不能算属实
        self.assertEqual(pipeline.last_result.unresolved_checks[0]["problem"], "内容检查未通过")

    def test_failed_checks_are_not_presented_as_checked_report(self):
        pipeline = SectorPipeline(Settings(openrouter_api_key="test"))
        result = AnalysisResult("aigc-content", "AIGC", status="completed", errors={"fact_checker": "失败"})
        pipeline._finish_result(result, time.monotonic())
        self.assertEqual(result.report_status, "核查或研究未完成草稿")


if __name__ == "__main__":
    unittest.main()
