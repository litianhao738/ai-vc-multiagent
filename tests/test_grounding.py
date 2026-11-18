import tempfile
import unittest

from vcagents import CompanyInput, ExpandedSectorPipeline as SectorPipeline, SectorRequest, Settings, save_result
from vcagents.grounding import check_material, errors_for_submission, evidence_index
from vcagents.quantities import unmatched
from vcagents.roles import COMPANY_ANALYST, FACT_CHECKER

BRIEF = {"evidence": [
    {"evidence_id": "E-S-1", "claim": "Higgsfield 估值 54 亿美元", "excerpt": "at a $5.4 billion valuation",
     "source_ref": "[S-1]", "excerpt_verified": True},
    {"evidence_id": "E-S-2", "claim": "Higgsfield 年化收入 7 亿美元", "excerpt": "reached $700 million in annualized revenue",
     "source_ref": "[S-2]", "excerpt_verified": True},
    {"evidence_id": "E-S-3", "claim": "SimilarWeb 月访问 255.4K", "excerpt": "", "source_ref": "[S-3]", "excerpt_verified": False},
]}
SHEETS = {"甲": {"evidence": [{"evidence_id": "E-C1-1", "claim": "甲完成超千万美元天使轮", "excerpt": "完成超千万美元天使轮",
                               "source_ref": "[C1-2]", "excerpt_verified": True}]}}


def check(text):
    return check_material("chair", {"headline": text}, evidence_index(BRIEF, SHEETS))


class EvidenceFirstTest(unittest.TestCase):
    def test_chinese_amount_scale_is_applied_once_in_audit_and_report_checks(self):
        for excerpt in ("收入100万美元。", "收入100 万美元。", "收入一百万美元。"):
            with self.subTest(excerpt=excerpt):
                index = evidence_index({"evidence": [{"evidence_id": "E-S-9", "excerpt": excerpt,
                                                     "excerpt_verified": True}]}, {})
                for amount in ("100万美元", "1000000美元", "1,000,000美元", "USD 1 million"):
                    self.assertEqual(unmatched(amount, excerpt), [])
                    self.assertFalse(check_material("chair", f"收入{amount} [E-S-9]。", index)["findings"])
                self.assertEqual(unmatched("1万美元", excerpt), ["1万"])
                report = check_material("chair", "收入1万美元 [E-S-9]。", index)
                self.assertTrue(errors_for_submission(report))

        self.assertEqual(unmatched("1万美元", "收入100万美元。支出1万美元。"), [])

    def test_cited_numbers_must_appear_in_verified_evidence(self):
        ok = check("Higgsfield 投后估值 54 亿美元（E-S-1）。")
        self.assertEqual((ok["numeric_sentences"], ok["supported"], ok["findings"]), (1, 1, []))
        self.assertEqual(check("甲完成 1000 万美元天使轮 [C1-2]。")["findings"], [])
        wrong = check("Higgsfield 投后估值 64 亿美元 [E-S-1]。")["findings"]
        self.assertEqual((wrong[0]["numbers"], wrong[0]["problem"]), (["64 亿"], "数字不在所引证据 E-S-1 中"))

    def test_uncited_fact_numbers_are_flagged_but_explicit_scenarios_are_not(self):
        self.assertEqual(check("Max 起价 $100/月。")["findings"][0]["problem"], "未标注证据编号")
        self.assertEqual(check("公司成立于 2025 年，排名第 1。")["numeric_sentences"], 0)
        self.assertEqual(check("假设上游涨价 30%，贡献毛利转负。")["findings"], [])
        self.assertTrue(check("Pexo售价20美元。")["findings"])
        self.assertTrue(check("根据Sacra估算，Pexo年收入为999亿美元。")["findings"])
        self.assertTrue(check("预计行业市场规模为999亿美元。")["findings"])
        self.assertTrue(check("自动路由10+个模型[E-C1-1]。")["findings"])

    def test_derived_number_with_formula_is_accepted(self):
        self.assertEqual(check("对应约 7.7 倍（54 亿 ÷ 7 亿）[E-S-1][E-S-2]。")["findings"], [])
        self.assertIn("不符", check("对应约 9.9 倍（54 亿 ÷ 7 亿）[E-S-1][E-S-2]。")["findings"][0]["problem"])

    def test_unverified_or_unknown_evidence_is_reported(self):
        self.assertEqual(check("月访问 255.4K [E-S-3]。")["findings"][0]["problem"], "所引证据未通过原文核对")
        self.assertIn("不存在", check("ARR 500 万美元 [E-C9-9]。")["findings"][0]["problem"])

    def test_claim_itself_cannot_supply_numbers_missing_from_original_quote(self):
        index = evidence_index({"evidence": [{"evidence_id": "E-S-9", "source_ref": "[S-9]",
            "claim": "收入999亿美元", "excerpt": "Revenue was $1 million.", "excerpt_verified": True}]}, {})
        report = check_material("chair", {"headline": "收入999亿美元[E-S-9]。"}, index)
        self.assertEqual(report["findings"][0]["numbers"], ["999亿"])

    def test_submission_errors_explain_how_to_fix(self):
        errors = errors_for_submission(check("Max 起价 $100/月。"))
        self.assertIn("$100", errors[0])
        self.assertIn("标注支持该数字的证据编号", errors[-1])
        self.assertEqual(errors_for_submission(check("没有数字。")), [])

    def test_pipeline_prioritizes_unsupported_sentences_and_reports_coverage(self):
        pipeline = SectorPipeline(Settings(mock=True))
        original_run = pipeline._run
        tasks = {}

        def run(role, task, toolbox, agent_id, name, schema=None):
            output = original_run(role, task, toolbox, agent_id, name, schema)
            if role == COMPANY_ANALYST:
                output["operating_assessment"] = "据公司披露 ARR 达到 500 万美元。"
            if agent_id == "chair_editor":
                output["companies"]["甲"]["operating_assessment"] = "据公司披露 ARR 达到 500 万美元。"
            if role == FACT_CHECKER:
                tasks[agent_id] = task
            return output

        pipeline._run = run
        result = pipeline.run(SectorRequest("aigc-content", [CompanyInput("甲"), CompanyInput("乙"), CompanyInput("丙")]))
        self.assertIn("请优先从中挑选核查", tasks["fact_checker:analysis"])
        self.assertIn("ARR 达到 500 万美元", tasks["fact_checker:analysis"])
        self.assertNotIn("请优先从中挑选核查", tasks["fact_checker"])
        self.assertTrue(any(f["numbers"] == ["500 万"] for c in result.evidence_checks.values() for f in c["findings"]))
        with tempfile.TemporaryDirectory() as tmp:
            folder = save_result(result, tmp)
            self.assertFalse((folder / "report.md").exists())
            self.assertTrue(result.report_checks)
            self.assertIn("不能作为正式报告", (folder / "research-draft.md").read_text(encoding="utf-8"))
            self.assertIn("证据核对未通过的数字句", (folder / "audit.md").read_text(encoding="utf-8"))

    def test_submission_check_never_accepts_an_unchanged_error(self):
        pipeline = SectorPipeline(Settings(mock=True))
        pipeline.run(SectorRequest("aigc-content", [CompanyInput("甲"), CompanyInput("乙"), CompanyInput("丙")]))
        check_once = pipeline._evidence_first_check("chair")
        data = {"headline": "Max 起价 $100/月。"}
        errors = check_once(data)
        self.assertTrue(errors)
        self.assertEqual(check_once(data), errors)

    def test_explicit_assumption_still_requires_correct_arithmetic(self):
        self.assertTrue(check("假设市场规模1亿美元（100家×100美元）。")["findings"])
        self.assertFalse(check("假设市场规模1万美元（100家×100美元）。")["findings"])
        self.assertFalse(check("假设可服务市场32.4亿美元（3240亿美元×10%×10%）。")["findings"])
        self.assertTrue(check("假设可服务市场324亿美元（3240亿美元×10%×10%）。")["findings"])

    def test_arithmetic_checks_display_rounding_percentages_and_precedence(self):
        cases = ["假设市场规模约19.3亿美元（482万人×400美元）。",
                 "假设毛利约16亿美元（32.4亿美元×50%）。",
                 "假设渗透率20%（20家÷100家）。",
                 "假设情景估值22亿美元（2亿美元+5亿美元×4）。"]
        for text in cases:
            self.assertFalse(check(text)["findings"], (text, check(text)))
        self.assertTrue(check("假设渗透率30%（20家÷100家）。")["findings"])
        self.assertTrue(check("假设倍数3倍（10÷0）。")["findings"])

    def test_exit_assumption_formula_is_checked_even_without_external_fact_check(self):
        data = {"exit_scenarios": [{"valuation_basis": "假设10亿美元×5倍=100亿美元"}]}
        report = check_material("chair", data, {})
        self.assertTrue(report["findings"])
        data["exit_scenarios"][0]["valuation_basis"] = "假设10亿美元×5倍=50亿美元"
        self.assertFalse(check_material("chair", data, {})["findings"])

    def test_pipeline_weights_and_other_input_fields_do_not_validate_a_price(self):
        pipeline = SectorPipeline(Settings(mock=True))
        pipeline.run(SectorRequest("aigc-content", [CompanyInput(n) for n in ("甲", "乙", "丙")]))
        pipeline.last_result.market_metrics = {"inputs": {"serviceable_share": 0.2}}
        self.assertTrue(pipeline._evidence_report("chair", {"headline": "Pexo售价20美元。"})["findings"])
        self.assertFalse(pipeline._evidence_report("chair", {"dd_checklist": ["核实复购率50%的分母"]})["findings"])

    def test_uncited_fact_cannot_borrow_a_matching_user_or_scenario_value(self):
        index = {"U-1": {"values": {5_000_000.0}, "verified": True}}
        claim = "据公司披露 ARR 达到 500 万美元。"
        report = check_material("chair", {"headline": claim}, index)
        self.assertEqual(report["findings"][0]["severity"], "error")
        self.assertTrue(errors_for_submission(report))
        cited = check_material("chair", {"headline": "据输入表，收入为500万美元[U-1]。"}, index)
        self.assertFalse(cited["findings"])


class UserInputCitationTest(unittest.TestCase):
    def test_input_amount_must_belong_to_the_claimed_company_and_field(self):
        from vcagents import AnalysisResult, MainAgentPipeline

        pipeline = MainAgentPipeline(Settings(mock=True, memory_enabled=False))
        pipeline.last_result = AnalysisResult("aigc-content", "示例", companies=["甲公司", "乙公司"],
            deal_metrics={"甲公司": {"post_money_musd": 100, "pre_money_musd": 90, "investment_musd": 10}},
            financial_metrics={"乙公司": {"period": "2025年", "inputs": {"revenue_current_musd": 1}}})
        for statement in ("甲公司的投后估值为1亿美元", "甲公司的投前估值为9000万美元",
                          "乙公司2025年度收入为100万美元", "乙公司的本期收入为1,000,000美元",
                          "甲公司投后估值1亿美元，乙公司本期收入100万美元",
                          "对甲公司继续尽调，暂不按1亿美元投后估值投1,000万美元"):
            with self.subTest(statement=statement):
                report = pipeline._evidence_report("chair", statement + " [U-1]。")
                self.assertEqual((report["supported"], report["findings"]), (1, []))
        for statement in ("乙公司的年度收入为1亿美元", "甲公司的年度收入为1亿美元", "乙公司的投后估值为1亿美元",
                          "甲公司的投前估值为1亿美元", "甲公司投后估值1亿美元，乙公司本期收入1亿美元",
                          "乙公司2024年度收入为100万美元", "乙公司本期收入为100万人民币", "乙公司本期收入为1美元"):
            with self.subTest(statement=statement):
                report = pipeline._evidence_report("chair", statement + " [U-1]。")
                self.assertEqual(report["supported"], 0)
                self.assertTrue(errors_for_submission(report))

        omitted = pipeline._evidence_report("analysis", {"analyses": {"甲公司": {"cost": "本期收入100万美元 [U-1]。"}}})
        self.assertTrue(errors_for_submission(omitted))
        ambiguous = pipeline._evidence_report("chair", "表中金额为1亿美元 [U-1]。")
        self.assertEqual((ambiguous["supported"], ambiguous["findings"][0]["severity"]), (0, "note"))
        pipeline.last_result.fact_sheets = {"乙公司": {"evidence": [{"evidence_id": "E-C2-1",
            "excerpt": "乙公司年度收入为1亿美元。", "excerpt_verified": True}]}}
        both = pipeline._evidence_report("chair", "乙公司年度收入为1亿美元 [E-C2-1][U-1]。")
        self.assertEqual((both["supported"], both["findings"]), (1, []))

    def test_single_company_does_not_supply_an_unrecognized_company_or_an_annual_total_from_a_quarter(self):
        from vcagents import AnalysisResult, MainAgentPipeline

        pipeline = MainAgentPipeline(Settings(mock=True, memory_enabled=False))
        pipeline.last_result = AnalysisResult("aigc-content", "示例", companies=["Alpha"],
            deal_metrics={"Alpha": {"post_money_musd": 100}},
            financial_metrics={"Alpha": {"period": "2025年第一季度", "inputs": {"revenue_current_musd": 2}}})
        for claim in ("Alpha投后估值1亿美元", "本轮投后估值为100,000,000美元", "Alpha 2025年第一季度本期收入200万美元"):
            self.assertEqual(pipeline._evidence_report("chair", claim + " [U-1]。")["findings"], [])
        for claim in ("乙公司投后估值1亿美元", "Beta 的投后估值100,000,000美元", "Alpha 2025年度收入200万美元"):
            self.assertEqual(pipeline._evidence_report("chair", claim + " [U-1]。")["supported"], 0)

    def test_scope_is_checked_in_fact_audit_and_again_at_publication(self):
        from vcagents import MainAgentPipeline
        from vcagents.report_checks import publication_findings

        pipeline = MainAgentPipeline(Settings(mock=True, memory_enabled=False))
        result = pipeline.run(SectorRequest("aigc-content", [
            CompanyInput("甲公司", post_money_musd=100), CompanyInput("乙公司"), CompanyInput("丙公司")]))
        bad = "乙公司的年度收入为1亿美元 [U-1]。"
        result.decision["valuation_view"] = bad
        findings = publication_findings(result, "")
        self.assertTrue(any(row["kind"] == "grounding" and row["target"] == bad for row in findings))

        box = pipeline._factcheck_toolbox()
        source = box.registry.register_web("https://example.org/research", "公司资料", "乙公司提供产品。")
        source.render()
        fact = {"subject": "乙公司", "fact_type": "产品", "statement": "乙公司提供产品", "quantities": [],
                "entities": [], "date": {"text": "", "basis": "未给出"}, "quote": {"source": source.id, "from": 1, "to": 1},
                "date_quote": {"source": "", "from": 0, "to": 0}}
        claim = {"source_agent": "chair", "target": bad, "fact": fact, "support": "完全支持", "verdict": "属实",
                 "evidence_ids": [], "original_source": "", "evidence": "", "correction": ""}
        box.rejection_review = lambda issues, context: []
        checked = box.verify_fact_check({"claims": [claim], "summary": ""})["claims"][0]
        self.assertEqual((checked["verdict"], checked["unmatched_numbers"]), ("无法核实", ["1亿"]))
        claim["target"] = "甲公司投后估值为100,000,000美元 [U-1]。"
        self.assertEqual(box.verify_fact_check({"claims": [claim], "summary": ""})["claims"][0]["unmatched_numbers"], [])

    def test_explicit_scenario_can_cite_its_base_input_but_not_an_unknown_source(self):
        index = {"U-1": {"verified": True, "values": {7e8}}}
        text = "假设门槛：若收入低于7000万美元或毛利低于40%，则不投[U-1]。"
        self.assertFalse(check_material("chair", {"headline": text}, index)["findings"])
        bad = text.replace("[U-1]", "[E-C1-999]")
        self.assertTrue(check_material("chair", {"headline": bad}, index)["findings"])

    def test_user_inputs_and_code_results_are_citable_as_u1(self):
        from vcagents.grounding import USER_KEY, user_values

        index = evidence_index(BRIEF, SHEETS)
        index[USER_KEY] = {"values": user_values({"Liblib": {"post_money_musd": 2000, "investment_musd": 10,
                                                             "ownership": 0.005, "implied_arr_multiple": 6.6667}}),
                           "verified": True}
        text = "拟按投后 20 亿美元投 1000 万美元（约 0.5%，约 6.7 倍 ARR）[U-1]。"
        report = check_material("chair", {"headline": text}, index)
        self.assertEqual((report["supported"], report["findings"]), (1, []))
        self.assertTrue(check_material("chair", {"headline": "拟投 2000 万美元 [U-1]。"}, index)["findings"])

    def test_fact_check_accepts_company_alias_and_user_numbers(self):
        from unittest.mock import patch
        from vcagents.llm import ToolCall
        from vcagents.tools import Toolbox

        box = Toolbox(known_values={2e9, 6.6667}, aliases=("Liblib", "演语科技（Liblib）"))
        with patch("vcagents.tools.fetch_url", return_value="标题：报道\nURL：https://n.example/a\n\n演语科技完成近3亿美元B+轮融资。"):
            box.execute(ToolCall("1", "fetch_url", {"url": "https://n.example/a"}))
        fact = {"subject": "Liblib", "fact_type": "融资", "statement": "完成近 3 亿美元 B+ 轮融资",
                "quantities": [{"metric": "融资额", "text": "近3亿美元"}], "entities": ["Liblib"],
                "date": {"text": "", "basis": "未给出"}, "quote": {"source": "W1", "from": 1, "to": 1},
                "date_quote": {"source": "", "from": 0, "to": 0}}
        claim = {"target": "Liblib 完成近 3 亿美元 B+ 轮融资，对应约 6.7 倍[U-1]", "source_agent": "chair", "original_source": "",
                 "evidence_ids": [], "fact": fact, "support": "完全支持", "verdict": "属实", "evidence": "", "correction": ""}
        checked = box.verify_fact_check({"claims": [claim], "summary": ""})["claims"][0]
        self.assertEqual((checked["verdict"], checked["field_issues"], checked["unmatched_numbers"]), ("属实", [], []))


if __name__ == "__main__":
    unittest.main()
