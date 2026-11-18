"""Writing-equivalence review must not become a general audit or a new source of facts."""
from copy import deepcopy
import json
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from vcagents import Settings, grounding, SectorPipeline, CompanyInput, SectorRequest, save_result
from vcagents.governor import RunCancelled
from vcagents.llm import LLMRouter, ToolCall, Turn
from vcagents.rejection_review import RejectionReviewer
from vcagents.report_checks import memo_materials
from vcagents.report_review import apply_report_reviews
from vcagents.tools import Toolbox
from vcagents.pipeline import AnalysisResult
from vcagents.roles import COMPANY_RESEARCHER
from vcagents.schemas import FACT_SHEET
from vcagents.sectors import load_sector


def reviewer(decisions=(True,), **settings):
    session = Mock(input_tokens=30, output_tokens=20, cost_usd=0.001)
    session.send.return_value = Turn(tool_calls=[ToolCall("r", "review_rejection", {"decisions": [
        {"issue_id": i, "unnecessary": value, "reason": "仅写法不同" if value else "存在实质差异"}
        for i, value in enumerate(decisions)]})])
    router = Mock()
    router.for_role.return_value.session.return_value = session
    return RejectionReviewer(Settings(**settings), router), session


def box_with_review(review, source_text="Alpha paid-user share is 0.203."):
    box = Toolbox(rejection_review=review.filter)
    source = box.registry.register_web("https://example.com/a", "Alpha", source_text)
    source.render()
    return box


def evidence(statement="付费用户比例为20.3%", quantity="20.3%"):
    return {"subject": "Alpha", "fact_type": "用户或客户", "statement": statement,
        "quantities": [{"metric": "付费用户比例", "text": quantity}], "entities": [],
        "date": {"text": "", "basis": "未给出"}, "quote": {"source": "W1", "from": 1, "to": 1},
        "date_quote": {"source": "", "from": 0, "to": 0}, "source_ref": "[1]", "source_type": "独立来源"}


def verify(box, item):
    return box.verify_evidence({"sources": ["[1] Alpha - https://example.com/a"], "evidence": [item]})["evidence"][0]


class RejectionReviewTest(unittest.TestCase):
    def test_only_confirmed_items_are_waived_without_rewriting_and_cached(self):
        review, session = reviewer((True, False))
        issues = ["标点不同", "金额不同"]
        context = {"source_excerpt": "USD 400M", "submitted": "4亿美元"}
        before = deepcopy(context)
        self.assertEqual(review.filter(issues, context), ["金额不同"])
        self.assertEqual(review.filter(issues, context), ["金额不同"])
        self.assertEqual(context, before)
        self.assertEqual(issues, ["标点不同", "金额不同"])
        self.assertEqual(session.send.call_count, 1)
        self.assertEqual(review.records[0]["waived"], ["标点不同"])
        system = review.router.for_role.return_value.session.call_args.args[0]
        self.assertIn("只有一个任务", system)
        self.assertIn("不要重新研究", system)


    def test_report_check_requires_current_quote_and_a_recomputed_table_value(self):
        row = "| Alpha 年度收入 | 1 |"
        source = "单位：百万美元\n" + row
        claim = "Alpha 年度收入100万美元[E-S-1]。"
        review, session = reviewer()
        session.send.return_value = Turn(tool_calls=[ToolCall("r", "review_amount", {
            "verdict": "equivalent", "basis": row, "reason": "表头明确以百万美元为单位"})])

        def check(excerpt, statement=claim, invoke=None):
            index = grounding.evidence_index({"evidence": [{"evidence_id": "E-S-1", "excerpt": excerpt,
                                                            "excerpt_verified": True}]}, {})
            report = grounding.check_material("chair", statement, index)
            return apply_report_reviews(report, review.records, invoke)

        baseline = check(source)
        self.assertTrue(grounding.errors_for_submission(baseline))
        self.assertFalse(grounding.errors_for_submission(baseline, defer_amount_review=True))
        self.assertFalse(check(source, invoke=review.review_amount)["findings"])
        self.assertFalse(check(source)["findings"])  # read-only export reuses the current proof
        self.assertTrue(check(source.replace("百万美元", "百万人民币"))["findings"])
        self.assertTrue(check(source.replace(" | 1 |", " | 2 |"))["findings"])
        self.assertTrue(check(source, claim.replace("100万", "101万"))["findings"])
        self.assertEqual(session.send.call_count, 1)

    def test_equal_numbers_with_currency_or_period_differences_reach_the_right_check(self):
        review, session = reviewer()
        session.send.return_value = Turn(tool_calls=[ToolCall("r", "review_amount", {
            "verdict": "insufficient", "basis": "", "reason": "缺少货币代码或适用期间"})])
        claim = "Alpha 2025年年度收入100万美元 [E-S-1]。"

        def check(excerpt):
            index = grounding.evidence_index({"evidence": [{"evidence_id": "E-S-1", "excerpt": excerpt,
                                                            "excerpt_verified": True}]}, {})
            return apply_report_reviews(grounding.check_material("chair", claim, index), review.records, review.review_amount)

        self.assertFalse(check("Alpha 2025年年度收入为1,000,000美元。")["findings"])
        self.assertIn("币种不同", check("Alpha 2025年年度收入100万人民币。")["findings"][0]["problem"])
        self.assertIn("指标不同", check("Alpha 2025年年化收入100万美元。")["findings"][0]["problem"])
        self.assertIn("期间不同", check("Alpha 2024年年度收入100万美元。")["findings"][0]["problem"])
        self.assertIn("信息不足", check("Alpha revenue was $1 million.")["findings"][0]["problem"])
        self.assertEqual(session.send.call_count, 1)

    def test_invalid_report_review_cannot_waive_a_table_value(self):
        for verdict, basis in (("equivalent", "来源没写的行"), ("different", "| 年度收入 | 1 |"), ("insufficient", "")):
            with self.subTest(verdict=verdict, basis=basis):
                review, session = reviewer()
                session.send.return_value = Turn(tool_calls=[ToolCall("r", "review_amount", {
                    "verdict": verdict, "basis": basis, "reason": "复核说明"})])
                index = grounding.evidence_index({"evidence": [{"evidence_id": "E-S-1",
                    "excerpt": "单位：百万美元\n| 年度收入 | 1 |", "excerpt_verified": True}]}, {})
                report = grounding.check_material("chair", "年度收入100万美元 [E-S-1]。", index)
                checked = apply_report_reviews(report, review.records, review.review_amount)
                self.assertTrue(grounding.errors_for_submission(checked))
                self.assertFalse(review.records[0]["waived"])

    def test_pipeline_and_export_reuse_local_report_review(self):
        pipeline = SectorPipeline(Settings(mock=True))
        result = pipeline.run(SectorRequest("aigc-content", [CompanyInput(n) for n in ("Alpha", "Beta", "Gamma")]))
        result.report_status = "待人工复核的研究报告"
        pipeline.settings.mock = False  # Main agents have finished in demo mode. Stub only the real review entry point.
        row = "| Alpha 年度收入 | 1 |"
        excerpt = "单位：百万美元\n" + row
        result.sector_brief["evidence"].append({"evidence_id": "E-S-999", "source_ref": "[S-999]",
            "claim": "Alpha 年度收入100万美元", "excerpt": excerpt, "excerpt_verified": True, "source_linked": True,
            "source_type": "独立来源", "as_of": "", "quote_location": "W99:s1-s2"})
        result.sector_brief["sources"].append("[S-999] Alpha 年报 https://example.com/alpha")
        result.comparison["summary"] = "Alpha 年度收入100万美元 [E-S-999]。"
        review, session = reviewer()
        session.send.return_value = Turn(tool_calls=[ToolCall("r", "review_amount", {
            "verdict": "equivalent", "basis": row, "reason": "表头与对应单元格支持相同金额"})])
        pipeline.rejection_reviewer = review
        result.rejection_reviews = review.records
        audit_hints = pipeline._evidence_priorities(result, {"chair": memo_materials(result)})
        self.assertIn(row, audit_hints)
        self.assertFalse(pipeline._publication_preflight(result))
        self.assertFalse(pipeline._publication_preflight(result))
        with tempfile.TemporaryDirectory() as root:
            folder = save_result(result, root)
            self.assertTrue((folder / "report.md").exists())
            self.assertIn("成稿金额", (folder / "audit.md").read_text(encoding="utf-8"))
            result.sector_brief["evidence"][-1]["excerpt"] = excerpt.replace(" | 1 |", " | 2 |")
            revised = save_result(result, root)
            self.assertTrue((revised / "research-draft.md").exists())
        self.assertEqual(session.send.call_count, 1)

    def test_changed_source_is_not_approved_from_an_old_cache_entry(self):
        review, session = reviewer()
        review.filter(["写法不同"], {"source_excerpt": "USD 400M"})
        review.filter(["写法不同"], {"source_excerpt": "USD 500M"})
        self.assertEqual(session.send.call_count, 2)

    def test_malformed_missing_duplicate_or_string_boolean_decisions_keep_rejection(self):
        cases = [None, {"decisions": []},
                 {"decisions": [{"issue_id": 0, "unnecessary": "false", "reason": "不等值"}]},
                 {"decisions": [{"issue_id": True, "unnecessary": True, "reason": "排版"}]},
                 {"decisions": [{"issue_id": 0, "unnecessary": True, "reason": ""}]}]
        for data in cases:
            with self.subTest(data=data):
                review, session = reviewer()
                session.send.return_value = Turn(tool_calls=[ToolCall("r", "review_rejection", data)])
                self.assertEqual(review.filter(["问题"], {}), ["问题"])
                self.assertFalse(review.records[0]["waived"])
        review, session = reviewer((True, True))
        session.send.return_value.tool_calls[0].input["decisions"][1]["issue_id"] = 0
        self.assertEqual(review.filter(["a", "b"], {}), ["a", "b"])

    def test_truncation_and_api_failure_keep_original_result_without_retries(self):
        for result in (Turn(truncated=True), ConnectionError("offline")):
            with self.subTest(result=result):
                review, session = reviewer()
                if isinstance(result, Exception):
                    session.send.side_effect = result
                else:
                    session.send.return_value = result
                self.assertEqual(review.filter(["a"], {}), ["a"])
                self.assertEqual(review.filter(["b"], {}), ["b"])
                self.assertEqual(session.send.call_count, 1)

    def test_budget_disabled_mock_and_oversized_inputs_do_not_trigger_more_requests(self):
        for options in ({"mock": True}, {"rejection_review_enabled": False}, {"rejection_review_max_calls": 0}):
            review, session = reviewer(**options)
            self.assertEqual(review.filter(["a"], {}), ["a"])
            session.send.assert_not_called()
        review, session = reviewer(rejection_review_max_calls=1)
        review.filter(["a"], {})
        self.assertEqual(review.filter(["b"], {}), ["b"])
        self.assertEqual(session.send.call_count, 1)
        review.reset()
        self.assertEqual(review.filter(["a"], {"source_excerpt": "x" * 13000}), ["a"])
        self.assertFalse(review.records)

    def test_run_cancellation_is_not_swallowed(self):
        review, session = reviewer()
        session.send.side_effect = RunCancelled("cancelled")
        with self.assertRaises(RunCancelled):
            review.filter(["a"], {})

    def test_reviewer_has_short_timeout_bounded_output_and_no_sdk_retries(self):
        settings = Settings(openrouter_api_key="test")
        llm = LLMRouter(settings).for_role("rejection_review")
        self.addCleanup(llm.client.close)
        self.assertEqual(llm.model, "google/gemini-3.1-flash-lite")
        self.assertEqual(llm.request_options["extra_body"]["reasoning"], {"effort": "minimal"})
        self.assertEqual(llm.request_options["tool_choice"], "required")
        self.assertEqual(llm.timeout, 45)
        self.assertEqual(llm.retries, 0)
        self.assertEqual(llm.max_tokens, 1200)


class CitationWritingReviewTest(unittest.TestCase):
    def test_equivalent_percentage_is_accepted_and_remains_usable_downstream(self):
        review, session = reviewer((True, True))
        box = box_with_review(review)
        item = evidence()
        original = deepcopy(item)
        checked = verify(box, item)
        self.assertTrue(checked["excerpt_verified"], checked)
        self.assertEqual(item, original)
        self.assertEqual(checked["excerpt"], "Alpha paid-user share is 0.203.")
        self.assertIn(20.3, checked["equivalent_values"])
        self.assertEqual(len(checked["writing_review"]["waived"]), 2)
        checked["evidence_id"] = "E-C1-1"
        index = grounding.evidence_index({}, {"Alpha": {"evidence": [checked]}})
        report = grounding.check_material("analysis", {"text": "Alpha付费用户比例为20.3%[E-C1-1]。"}, index)
        self.assertFalse(grounding.errors_for_submission(report))
        verify(Toolbox(registry=box.registry, rejection_review=review.filter), item)
        self.assertEqual(session.send.call_count, 1)

    def test_actual_numeric_difference_is_still_rejected(self):
        review, _ = reviewer((False, False))
        checked = verify(box_with_review(review, "Alpha paid-user share is 20.3%."), evidence("付费用户比例为203%", "203%"))
        self.assertFalse(checked["excerpt_verified"])
        self.assertFalse(checked["equivalent_values"])

    def test_invalid_or_unread_quote_and_wrong_source_never_reach_model(self):
        for mutation, read in (({"quote": {"source": "W9", "from": 1, "to": 1}}, True),
                               ({"source_ref": "[9]"}, True), ({}, False)):
            review, session = reviewer((True, True))
            box = box_with_review(review)
            if not read:
                box.registry.get("W1").shown.clear()
            checked = verify(box, {**evidence(), **mutation})
            self.assertFalse(checked["excerpt_verified"])
            session.send.assert_not_called()

    def test_source_date_and_length_limits_cannot_be_waived(self):
        review, session = reviewer()
        box = box_with_review(review, "Alpha launched a product.")
        item = evidence("长" * 161, "")
        item["date"] = {"text": "2031年", "basis": "摘录"}
        self.assertFalse(verify(box, item)["excerpt_verified"])
        session.send.assert_not_called()

    def test_equivalence_receipts_cannot_be_faked_in_a_submission(self):
        box = box_with_review(SimpleNamespace(filter=None), "Alpha paid-user share is 20.3%.")
        item = {**evidence(), "writing_review": {"waived": ["fake"]}, "equivalent_values": [999999]}
        checked = verify(box, item)
        self.assertTrue(checked["excerpt_verified"])
        self.assertEqual(checked["writing_review"], {})
        self.assertEqual(checked["equivalent_values"], [])

    def test_legacy_evidence_also_clears_forged_equivalence_receipts(self):
        box = Toolbox(shared_reads=[{"kind": "web", "urls": ["https://example.com/a"], "text": "原文20.3%"}])
        item = {"source_ref": "[1]", "excerpt": "原文20.3%", "source_type": "独立来源",
                "writing_review": {"waived": ["fake"]}, "equivalent_values": [999999]}
        checked = verify(box, item)
        self.assertTrue(checked["excerpt_verified"])
        self.assertEqual(checked["writing_review"], {})
        self.assertFalse(checked["equivalent_values"])

    def test_native_research_submission_reuses_review_during_postprocessing(self):
        pipeline = SectorPipeline(Settings(mock=True))
        pipeline.settings.mock = False
        pipeline.sector = load_sector("aigc-content")
        pipeline.companies = {"Alpha": CompanyInput("Alpha")}
        pipeline.last_result = AnalysisResult("aigc-content", "AIGC")
        box = pipeline._research_toolbox()
        source = box.registry.register_web("https://example.com/a", "Alpha", "Alpha paid-user share is 0.203.")
        source.render()
        sheet = {"company_name": "Alpha", "one_liner": "产品", "stage": "未知", "listing_status": "未上市",
                 "listing_evidence": "资料", "fact_sheet": "付费用户比例20.3%", "information_gaps": [],
                 "sources": ["[1] Alpha - https://example.com/a"], "evidence": [evidence()]}
        main_session = Mock(input_tokens=20, output_tokens=20, cost_usd=0.001)
        main_session.send.return_value = Turn(tool_calls=[ToolCall("s", "submit_result", sheet)])
        main_model = Mock(label="scripted")
        main_model.session.return_value = main_session
        review, review_session = reviewer((True, True))
        review_model = review.router.for_role.return_value
        pipeline.router.for_role = lambda role: review_model if role == "rejection_review" else main_model
        checked = pipeline._run(COMPANY_RESEARCHER, "test", box, "research_candidates", "research", FACT_SHEET)
        self.assertTrue(checked["evidence"][0]["excerpt_verified"])
        self.assertIn(20.3, checked["evidence"][0]["equivalent_values"])
        self.assertEqual(main_session.send.call_count, 1)
        self.assertEqual(review_session.send.call_count, 1)
        self.assertEqual(list(pipeline.agents), ["research"])

    def test_fact_checker_keeps_verdict_when_only_percentage_notation_differs(self):
        review, _ = reviewer((True, True, True))
        box = box_with_review(review)
        fact = evidence()
        claim = {"target": "Alpha付费用户比例为20.3%", "source_agent": "research", "original_source": "",
                 "evidence_ids": [], "fact": fact, "support": "完全支持", "verdict": "属实", "evidence": "", "correction": ""}
        checked = box.verify_fact_check({"claims": [claim], "summary": ""})["claims"][0]
        self.assertEqual(checked["verdict"], "属实")
        self.assertFalse(checked["unmatched_numbers"])

    def test_fact_checker_can_cite_the_original_cell_and_table_unit(self):
        review, session = reviewer()
        box = box_with_review(review, "| 指标（百万美元） | 数值 |\n| Alpha 年度收入 | 1 |")
        fact = evidence(statement="年度收入为 1", quantity="1")
        fact["quote"]["to"] = 2
        claim = {"target": "Alpha 年度收入100万美元。", "source_agent": "chair", "original_source": "",
                 "evidence_ids": [], "fact": fact, "support": "完全支持", "verdict": "属实", "evidence": "", "correction": ""}
        self.assertEqual(box.citation_errors("fact_checker", {"claims": [claim]}), [])
        self.assertEqual(box.verify_fact_check({"claims": [claim], "summary": ""})["claims"][0]["verdict"], "属实")
        session.send.assert_not_called()


if __name__ == "__main__":
    unittest.main()
