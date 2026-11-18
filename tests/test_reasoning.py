import copy
import random
import tempfile
import unittest

from vcagents import CompanyInput, SectorPipeline, SectorRequest, Settings, save_result
from vcagents import schemas
from vcagents.llm import _fake
from vcagents.reasoning import review_issues, review_targets
from vcagents.report import render_memo
from vcagents.report_checks import publication_issues


class ReasoningReviewTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = SectorPipeline(Settings(mock=True)).run(
            SectorRequest("aigc-content", [CompanyInput("甲")]))

    def test_every_core_judgment_needs_its_own_review(self):
        schema = schemas.decision_check_schema(self.base.decision, [])
        data = _fake(schema, random.Random(1))
        self.assertEqual(schemas.validate_submission("fact_checker", schema, data), [])
        data["reasoning_checks"][-1]["target_id"] = "T1"
        self.assertTrue(any("不能重复或遗漏" in e for e in schemas.validate_submission("fact_checker", schema, data)))

    def test_pass_requires_cited_basis_and_conditional_inference_requires_conditions(self):
        schema = schemas.decision_check_schema(self.base.decision, [])
        data = _fake(schema, random.Random(1))
        data["reasoning_checks"][0].update(verdict="条件性推断", evidence_ids=[], conditions=[])
        errors = schemas.validate_submission("fact_checker", schema, data)
        self.assertTrue(any("依据编号" in e for e in errors))
        self.assertTrue(any("明确的条件" in e for e in errors))

    def test_publication_rejects_missing_or_stale_semantic_review(self):
        result = copy.deepcopy(self.base)
        self.assertFalse(review_issues(result.decision, result.decision_fact_check))
        result.decision["top_pick_thesis"][0]["judgment"] = "融资额较大，证明资本效率领先。"
        self.assertEqual(review_issues(result.decision, result.decision_fact_check)[0]["target_id"], "T1")
        self.assertTrue(any("核心投资判断 T1" in i for i in publication_issues(result, render_memo(result))))
        result.decision_fact_check.pop("reasoning_checks")
        self.assertEqual(set(i["target_id"] for i in review_issues(result.decision, result.decision_fact_check)),
                         set(review_targets(result.decision)))

    def test_unlocated_draft_prose_is_not_an_analyst_fact_source(self):
        pipeline = SectorPipeline(Settings(mock=True))
        pipeline.run(SectorRequest("aigc-content", [CompanyInput("甲")]))
        sheet, brief = copy.deepcopy(pipeline.last_result.fact_sheets["甲"]), copy.deepcopy(pipeline.last_result.sector_brief)
        sheet["fact_sheet"], brief["summary"] = "杜撰的经营优势", "杜撰的市场结论"
        sheet["evidence"] = [
            {"evidence_id": "E-C1-1", "excerpt": "公司自述收入增长", "excerpt_verified": True},
            {"evidence_id": "E-C1-2", "excerpt": "杜撰的成本优势", "excerpt_verified": False}]
        task = pipeline._analyst_task("甲", sheet, brief)
        for text in ("杜撰的经营优势", "杜撰的市场结论", "杜撰的成本优势"):
            self.assertNotIn(text, task)
        self.assertIn("公司自述收入增长", task)

    def _run_reasoning_correction(self, persists=False):
        pipeline = SectorPipeline(Settings(mock=True))
        original = pipeline._run
        calls, drafts = [], []

        def run(role, task, toolbox, agent_id, name, schema=None):
            calls.append(agent_id)
            output = original(role, task, toolbox, agent_id, name, schema)
            if agent_id == "chair_editor":
                drafts.append(copy.deepcopy(output))
                output["decision"]["top_pick_thesis"][0]["judgment"] = (
                    "融资规模较大，证明资本效率领先。" if len(drafts) == 1 or persists
                    else "资本效率需结合现金消耗和单位经济评估。")
            if agent_id.startswith("fact_checker_decision") or agent_id == "quality_final_audit":
                if agent_id == "fact_checker_decision" or persists:
                    output["reasoning_checks"][0].update(
                        verdict="依据不足", reason="融资额不能单独证明资本效率。", correction="删除资本效率领先的判断。")
            return output

        pipeline._run = run
        return pipeline.run(SectorRequest("aigc-content", [CompanyInput("甲")])), calls

    def test_unsupported_inference_is_repaired_without_redoing_research_or_exit_assumptions(self):
        result, calls = self._run_reasoning_correction()
        self.assertIn("fact_checker_decision_recheck", calls)
        self.assertEqual(calls.count("chair_editor"), 2)
        self.assertFalse(any(c.startswith(("repair:", "reanalysis:")) for c in calls))
        self.assertIn("需结合现金消耗", result.decision["top_pick_thesis"][0]["judgment"])
        self.assertFalse(review_issues(result.decision, result.decision_fact_check))
        repair = result.repairs[-1]
        self.assertEqual(repair["reasoning"][0]["target_id"], "T1")
        self.assertEqual(repair["before"]["exit_scenarios"], repair["after"]["exit_scenarios"])
        self.assertEqual([s["weighted"] for s in result.ranking],
                         [s["weighted"] for s in self.base.ranking])

    def test_persistent_unsupported_inference_is_not_exported(self):
        result, calls = self._run_reasoning_correction(persists=True)
        self.assertEqual(calls.count("fact_checker_decision_recheck"), 1)
        self.assertTrue(any(u["claim_id"] == "T1" for u in result.unresolved_checks))
        with tempfile.TemporaryDirectory() as root:
            folder = save_result(result, root)
            self.assertFalse((folder / "report.md").exists())
            self.assertIn("不能作为正式报告", (folder / "research-draft.md").read_text(encoding="utf-8"))
            self.assertIn("融资额不能单独证明资本效率", (folder / "audit.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
