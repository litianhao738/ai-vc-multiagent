import tempfile
import unittest
from unittest.mock import Mock

from vcagents import CompanyInput, ExpandedSectorPipeline as SectorPipeline, SectorRequest, Settings, save_result
from vcagents.roles import CHAIR, COMPANY_ANALYST, COMPANY_RESEARCHER, FACT_CHECKER


class CorrectionFlowTest(unittest.TestCase):
    def test_explicit_empty_financial_upload_is_rejected_before_research(self):
        pipeline = SectorPipeline(Settings(mock=True))
        pipeline._run = Mock()
        companies = [CompanyInput("甲", financial_data={}), CompanyInput("乙"), CompanyInput("丙")]
        with self.assertRaises(ValueError):
            pipeline.run(SectorRequest("aigc-content", companies))
        pipeline._run.assert_not_called()

    def test_incomplete_research_is_saved_as_draft_with_audit(self):
        result = SectorPipeline(Settings(mock=True)).run(
            SectorRequest("aigc-content", [CompanyInput(n) for n in ("甲", "乙", "丙")]))
        result.report_status = "核查或研究未完成草稿"
        result.errors = {"fact_checker": "接口失败"}
        with tempfile.TemporaryDirectory() as directory:
            folder = save_result(result, directory)
            self.assertFalse((folder / "report.md").exists())
            self.assertTrue((folder / "research-draft.md").exists())
            self.assertIn("接口失败", (folder / "audit.md").read_text(encoding="utf-8"))

    def test_unsupported_research_is_sent_back_and_deleted(self):
        pipeline = SectorPipeline(Settings(mock=True))
        original = pipeline._run
        calls = []

        def run(role, task, toolbox, agent_id, name, schema=None):
            calls.append((agent_id, task))
            if role == FACT_CHECKER:
                if agent_id == "fact_checker":
                    return {"claims": [{"claim": "不存在的融资事实", "source_agent": "company_researcher:甲",
                                        "original_source": "", "verdict": "无法核实", "evidence": "没有原文", "correction": ""}],
                            "summary": "需要删除"}
                result = {"claims": [], "summary": "已删除"}
                if agent_id == "fact_checker_recheck":
                    result["repair_checks"] = [{"claim_id": "R-1", "status": "已删除", "reason": "当前事实表和分析已删除"}]
                return result
            return original(role, task, toolbox, agent_id, name, schema)

        pipeline._run = run
        result = pipeline.run(SectorRequest("aigc-content", [CompanyInput(n) for n in ("甲", "乙", "丙")]))
        self.assertIn("repair:company_researcher:甲", [call[0] for call in calls])
        self.assertIn("reanalysis:甲", [call[0] for call in calls])
        self.assertIn("fact_checker_recheck", [call[0] for call in calls])
        self.assertEqual(result.repairs[0]["claims"][0]["verdict"], "无法核实")

    def test_chair_correction_reaches_fact_sheet_analysis_and_export(self):
        old = "甲公司2025年收入为10百万美元"
        corrected = "据公司披露，甲公司2025年收入为1百万美元"
        pipeline = SectorPipeline(Settings(mock=True))
        original = pipeline._run
        calls = []

        def claim(text, source, verdict):
            return {"claim": text, "source_agent": source, "original_source": "[C1-1]",
                    "evidence_ids": ["E-C1-1"], "verdict": verdict, "evidence": "原文收入1百万美元",
                    "correction": corrected if verdict == "不实" else ""}

        def run(role, task, toolbox, agent_id, name, schema=None):
            calls.append(agent_id)
            if role == FACT_CHECKER:
                if agent_id == "fact_checker":
                    return {"claims": [], "summary": "第一轮未覆盖该事实"}
                if agent_id == "fact_checker_decision":
                    return {"claims": [claim(old, "chair", "不实")], "summary": "收入错误"}
                result = {"claims": [claim(corrected, "chair" if "decision" in agent_id else "company_researcher:甲", "公司口径")],
                          "summary": "已纠正"}
                if "reasoning_checks" in schema["properties"]:
                    result["reasoning_checks"] = original(role, task, toolbox, agent_id, name, schema)["reasoning_checks"]
                ids = schema["properties"]["repair_checks"]["items"]["properties"]["claim_id"]["enum"]
                result["repair_checks"] = [{"claim_id": key, "status": "已修正", "reason": "已按原文改为1百万美元"} for key in ids]
                return result
            output = original(role, task, toolbox, agent_id, name, schema)
            if agent_id == "chair_editor":
                evidence_id = pipeline.last_result.fact_sheets["甲"]["evidence"][0]["evidence_id"]
                output["companies"]["甲"]["summary"] = corrected + f"[{evidence_id}]。"
            revised = agent_id.startswith(("repair:", "reanalysis:"))
            if role == COMPANY_RESEARCHER and agent_id.endswith(":甲"):
                output["fact_sheet"] = corrected if revised else old
                output["evidence"] = [{"claim": corrected if revised else old, "source_ref": "[1]",
                                       "excerpt": "2025年收入为1百万美元", "source_type": "公司口径", "as_of": "2025",
                                       "excerpt_verified": True}]
            if role == COMPANY_ANALYST and agent_id.endswith(":甲"):
                output["summary"] = corrected if revised else old
            if role == CHAIR:
                output["investment_thesis"] = corrected if revised else old
            return output

        pipeline._run = run
        result = pipeline.run(SectorRequest("aigc-content", [CompanyInput(n) for n in ("甲", "乙", "丙")]))
        self.assertIn("repair:company_researcher:甲", calls)
        self.assertIn("reanalysis:甲", calls)
        self.assertIn(corrected, result.fact_sheets["甲"]["fact_sheet"])
        self.assertIn(corrected, result.analyses["甲"]["summary"])
        with tempfile.TemporaryDirectory() as directory:
            folder = save_result(result, directory)
            report = (folder / "report.md").read_text(encoding="utf-8")
            audit = (folder / "audit.md").read_text(encoding="utf-8")
        self.assertIn(corrected, report)
        self.assertNotIn(old, report)
        self.assertIn(old, audit)


if __name__ == "__main__":
    unittest.main()
