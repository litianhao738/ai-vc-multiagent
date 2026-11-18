from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pypdf import PdfReader

from vcagents import CompanyInput, SectorPipeline, SectorRequest, Settings, save_result
from vcagents.brief import render_brief
from vcagents.config import REPORT_PAGE_CAP
from vcagents.pdf_report import build_report_pdf, ReportPageLimitError
from vcagents.report import render_memo


class ReportBudgetTest(unittest.TestCase):
    def setUp(self):
        report_index = patch("vcagents.pipeline.index_export")
        report_index.start()
        self.addCleanup(report_index.stop)

    @classmethod
    def setUpClass(cls):
        cls.result = SectorPipeline(Settings(mock=True)).run(
            SectorRequest("aigc-content", [CompanyInput(n) for n in ("Alpha", "Beta", "Gamma", "Delta", "Epsilon")]))

    def test_five_company_report_has_real_pdf_with_at_most_ten_pages(self):
        with tempfile.TemporaryDirectory() as root:
            r = deepcopy(self.result)
            folder = save_result(r, root)
            pdf = folder / "report.pdf"
            reader = PdfReader(pdf)
            self.assertLessEqual(len(reader.pages), 10)
            self.assertEqual(len(reader.pages), r.report_pages)
            self.assertEqual(reader.metadata.title, f"{r.sector_name} 投资报告")
            self.assertEqual((folder / "report.md").read_text(encoding="utf-8").splitlines()[0], f"# {r.sector_name} 投资报告")
            self.assertIn("核心判断", reader.pages[0].extract_text())
            self.assertIn("悲观", "\n".join(p.extract_text() for p in reader.pages))

    def test_consecutive_exports_use_the_current_research_subject_in_every_title(self):
        pipeline = SectorPipeline(Settings(mock=True, memory_enabled=False))
        requests = [
            ("aigc-content", ["Higgsfield"]),
            ("embodied-ai", ["岚桥机器人"]),
            ("embodied-ai", ["岚桥机器人", "星禾自动化"]),
        ]
        with tempfile.TemporaryDirectory() as root:
            for sector, companies in requests:
                with self.subTest(companies=companies):
                    result = pipeline.run(SectorRequest(sector, [CompanyInput(name) for name in companies]))
                    subject = companies[0] if len(companies) == 1 else result.sector_name
                    expected = f"{subject} 投资报告"
                    self.assertEqual(render_memo(result).splitlines()[0], f"# {expected}")
                    folder = save_result(result, root)
                    self.assertEqual((folder / "report.md").read_text(encoding="utf-8").splitlines()[0], f"# {expected}")
                    reader = PdfReader(folder / "report.pdf")
                    self.assertEqual(reader.metadata.title, expected)
                    self.assertIn(expected, reader.pages[0].extract_text().splitlines())

    def test_archived_subject_names_survive_filtering_of_unverified_claims(self):
        r = deepcopy(self.result)
        r.decision.pop("technical_moat")
        for subject in (r.sector_name, r.decision["top_pick"]):
            r.fact_check["claims"].append({"claim": subject, "verdict": "无法核实"})
        for render in (render_brief, render_memo):
            memo = render(r)
            self.assertEqual(memo.splitlines()[0], f"# {r.sector_name} 投资报告")
            self.assertIn("研究对象：" + "、".join(r.companies), memo)

    def test_overlength_editor_result_is_compacted_before_final_fact_check(self):
        pipeline = SectorPipeline(Settings(mock=True, report_max_pages=10))
        original = pipeline._run
        calls = []

        def run(role, task, toolbox, agent_id, name, schema=None):
            output = original(role, task, toolbox, agent_id, name, schema)
            if agent_id == "chair_editor":
                calls.append("editor")
                if calls.count("editor") == 1:
                    output["decision"]["thesis"] = "重复背景必须压缩，重大风险必须保留。" * 1800
                else:
                    self.assertIn("篇幅预检没有通过", task)
            if agent_id == "fact_checker_decision":
                calls.append("final_check")
                self.assertNotIn("重复背景必须压缩", task)
            return output

        pipeline._run = run
        r = pipeline.run(SectorRequest("aigc-content", [CompanyInput("Alpha")]))
        self.assertEqual(calls[:3], ["editor", "editor", "final_check"])
        self.assertEqual(len(r.memo_reviews), 2)
        self.assertFalse(r.errors)

    def test_page_cap_is_enforced_independently_of_prompt_compliance(self):
        oversized = "# 报告\n\n" + "\n\n".join(["必须完整呈现下行风险与原文证据，不得截断。" * 25] * 70)
        with self.assertRaises(ReportPageLimitError):
            build_report_pdf(oversized, 10)
        self.assertEqual(build_report_pdf("短报告", REPORT_PAGE_CAP)[1], 1)
        with self.assertRaises(ValueError):
            build_report_pdf("短报告", REPORT_PAGE_CAP + 1)

    def test_default_is_ten_pages_and_long_reports_can_opt_in_to_twenty_five(self):
        self.assertEqual(REPORT_PAGE_CAP, 25)
        self.assertEqual(Settings().report_max_pages, 10)
        self.assertEqual(self.result.report_max_pages, 10)
        for value in (0, 26, True, 12.5):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "页数上限"):
                SectorPipeline(Settings(mock=True, report_max_pages=value)).run(
                    SectorRequest("aigc-content", [CompanyInput("Alpha")]))

    def test_prompts_and_word_budgets_follow_the_configured_cap(self):
        from vcagents.roles import CHAIR, system_prompt
        from vcagents.sectors import load_sector
        sector = load_sector("aigc-content")
        self.assertIn("最多10页", system_prompt(CHAIR, sector, "", "2026-11-17"))
        self.assertIn("最多12页", system_prompt(CHAIR, sector, "", "2026-11-17", 12))
        self.assertEqual(SectorPipeline(Settings(mock=True))._char_budget(3500), 3500)
        self.assertEqual(SectorPipeline(Settings(mock=True, report_max_pages=25))._char_budget(3500), 8800)
        self.assertEqual(SectorPipeline(Settings(mock=True, report_max_pages=10))._char_budget(2200), 2200)

    def test_repeat_exports_cannot_leave_an_old_public_report_in_a_failed_export(self):
        r = deepcopy(self.result)
        with tempfile.TemporaryDirectory() as root:
            first = save_result(r, root)
            r.report_status = "核查或研究未完成草稿"
            second = save_result(r, root)
            self.assertNotEqual(first, second)
            self.assertTrue((first / "report.pdf").exists())
            self.assertFalse((second / "report.pdf").exists())
            self.assertTrue((second / "research-draft.pdf").exists())

    def test_overflow_blocks_public_report_and_keeps_full_research_record(self):
        r = SectorPipeline(Settings(mock=True, memory_enabled=False)).run(
            SectorRequest("embodied-ai", [CompanyInput("青石机械")]))
        r.report_max_pages = 10
        r.decision["thesis"] = "必须完整呈现下行风险与原文证据。" * 2000
        with tempfile.TemporaryDirectory() as root:
            folder = save_result(r, root)
            self.assertFalse((folder / "report.pdf").exists())
            self.assertFalse((folder / "report.md").exists())
            self.assertIn("页数", r.report_status)
            draft = (folder / "research-draft.md").read_text(encoding="utf-8")
            self.assertEqual(draft.splitlines()[0], "# 青石机械 投资报告")
            self.assertIn("不能作为正式报告", draft)
            reader = PdfReader(folder / "research-draft.pdf")
            self.assertEqual(reader.metadata.title, "青石机械 投资报告")
            self.assertLessEqual(len(reader.pages), 10)
            saved = json.loads((folder / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["decision"]["thesis"], r.decision["thesis"])

    def test_references_include_used_evidence_and_its_source_only(self):
        r = deepcopy(self.result)
        e = {"evidence_id": "E-C1-1", "claim": "Alpha历史收入100万美元", "source_ref": "[C1-1]",
             "source_type": "公司口径", "as_of": "2025", "excerpt": "Revenue $1 million.",
             "source_linked": True, "excerpt_verified": True}
        r.fact_sheets["Alpha"]["evidence"] = [e]
        r.fact_sheets["Alpha"]["sources"] = ["[C1-1] Alpha - https://alpha.example/revenue", "[C1-2] 未使用 - https://alpha.example/unused"]
        r.analyses["Alpha"]["operating_assessment"] = "历史收入100万美元 [E-C1-1]。"
        brief = render_brief(r)
        self.assertIn("- E-C1-1 ·", brief)
        self.assertIn("- [C1-1]", brief)
        self.assertNotIn("https://alpha.example/unused", brief)


if __name__ == "__main__":
    unittest.main()
