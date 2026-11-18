import copy
import unittest

from vcagents import CompanyInput, SectorPipeline, SectorRequest, Settings
from vcagents.pipeline import AnalysisResult
from vcagents.report import format_run_cost, render_audit, render_memo


class ReportOutputTest(unittest.TestCase):
    def test_partial_cost_counts_returned_receipts_without_double_counting_search(self):
        result = AnalysisResult("sector", "sector", costs_usd={"research": 1.0, "audit": None},
                                search_usage={"calls": 2, "cost_usd": 0.2},
                                run_control={"reported_cost_usd": 1.7})
        self.assertEqual(format_run_cost(result), "已返回费用合计 $1.7000，1 项费用未知")
        result.costs_usd = {"research": None}
        result.search_usage = {"calls": 1, "cost_usd": None}
        result.run_control = {"reported_cost_usd": 0.0}
        self.assertEqual(format_run_cost(result), "接口未返回费用，2 项费用未知")

    @classmethod
    def setUpClass(cls):
        cls.base = SectorPipeline(Settings(mock=True)).run(
            SectorRequest("aigc-content", [CompanyInput(name) for name in ("甲", "乙", "丙")]))

    def test_formal_report_omits_rejected_fact_and_keeps_correction_and_company_disclosure(self):
        result = copy.deepcopy(self.base)
        rejected = {"claim": "甲公司2025年收入1.2亿美元", "verdict": "无法核实", "source_agent": "company_researcher:甲",
                    "original_source": "原始BP", "evidence_ids": [], "evidence": "缺少原始披露", "correction": ""}
        result.repairs = [{"source_agent": "company_researcher:甲", "claims": [rejected], "before": {}, "after": {}}]
        result.check_history = [{"stage": "修正前", "result": {"claims": [rejected], "summary": "原主张无法核实"}}]
        result.fact_check = {"claims": [], "summary": "修正完成"}
        result.report_status = "存在待核实关键事实的研究稿"
        result.decision["thesis"] = "甲公司2025年收入1.2亿美元。未披露客户留存数据。产品按月订阅。"
        result.analyses["甲"]["summary"] = (
            "甲公司2025年收入1.2亿美元。修正后，据公司披露2025年收入0.12亿美元。"
            "公司收入99亿美元，待核实。定价版本见 https://example.com/v1.2/pricing。")
        result.fact_sheets["甲"]["sources"] = ["[C1-1] 公司披露 https://example.com/report", "[C1-2] 未读取来源"]
        result.fact_sheets["甲"]["evidence"] = [
            {"evidence_id": "E-C1-1", "claim": "据公司披露2025年收入0.12亿美元", "source_type": "公司口径",
             "as_of": "2025", "source_ref": "[C1-1]", "excerpt": "2025年收入0.12亿美元",
             "source_linked": True, "excerpt_verified": True},
            {"evidence_id": "E-C1-2", "claim": "收入99亿美元", "source_type": "未知", "as_of": "2025",
             "source_ref": "[C1-2]", "excerpt": "收入99亿美元", "source_linked": True, "excerpt_verified": False},
        ]
        report = render_memo(result)
        for phrase in ("无法核实", "待核实", rejected["claim"], "收入99亿美元", "未读取来源", "运行指标", "模型原判"):
            self.assertNotIn(phrase, report)
        for phrase in ("据公司披露2025年收入0.12亿美元", "未披露客户留存数据", "公司口径", "https://example.com/report",
                       "https://example.com/v1.2/pricing", "产品按月订阅"):
            self.assertIn(phrase, report)
        audit = render_audit(result)
        self.assertIn(rejected["claim"], audit)
        self.assertIn("无法核实", audit)
        self.assertIn("运行指标", audit)
        self.assertEqual(result.decision["thesis"], "甲公司2025年收入1.2亿美元。未披露客户留存数据。产品按月订阅。")

    def test_audit_accepts_partial_result_and_lists_snapshot_metadata_without_body(self):
        result = AnalysisResult("aigc-content", "AIGC内容", status="failed", errors={"pipeline": "HTTP 402"})
        result.source_reads = {"researcher": [{"kind": "web", "urls": ["https://example.com"],
                                               "snapshot_id": "s123", "content_sha256": "hash123",
                                               "text": "原文正文不应在审计摘要重复"}]}
        audit = render_audit(result)
        self.assertIn("HTTP 402", audit)
        self.assertIn("s123", audit)
        self.assertIn("hash123", audit)
        self.assertIn("text_chars", audit)
        self.assertNotIn("原文正文不应在审计摘要重复", audit)

    def test_new_primary_evidence_can_resolve_an_earlier_missing_source(self):
        result = copy.deepcopy(self.base)
        claim = "据公司披露2025年收入0.12亿美元"
        result.decision["thesis"] = claim + "。"
        result.check_history = [{"result": {"claims": [{"claim": claim, "verdict": "无法核实"}]}}]
        result.fact_check = {"claims": [{"claim": claim, "verdict": "公司口径", "excerpt_verified": True}]}
        self.assertIn(claim, render_memo(result))

    def test_unread_source_cannot_leave_a_fact_or_dangling_reference_in_report(self):
        result = copy.deepcopy(self.base)
        result.fact_sheets["甲"]["sources"] = ["[C1-9] 新闻 https://source.example/read", "[C1-8] https://source.example/unread"]
        result.fact_sheets["甲"]["evidence"] = []
        result.analyses["甲"]["summary"] = "已读来源里的业务描述[C1-9]。缺少原文的收入999亿美元[C1-8]。"
        result.source_reads = {"researcher": [{"kind": "web", "urls": ["https://source.example/read"], "text": "业务描述"}]}
        report = render_memo(result)
        self.assertIn("已读来源里的业务描述[C1-9]", report)
        self.assertIn("https://source.example/read", report)
        self.assertNotIn("999亿美元", report)
        self.assertNotIn("[C1-8]", report)


if __name__ == "__main__":
    unittest.main()
