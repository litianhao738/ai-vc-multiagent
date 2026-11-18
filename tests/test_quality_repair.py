"""End-to-end publication failures must return to the responsible existing agent."""
from copy import deepcopy
import tempfile
import unittest

from vcagents import CompanyInput, SectorPipeline, SectorRequest, Settings, save_result
from vcagents.brief import render_brief
from vcagents.report_checks import publication_findings
from vcagents.reasoning import review_targets


BAD = "据公司披露，收入达到987654321万美元。"


def request():
    return SectorRequest("aigc-content", [CompanyInput(n) for n in ("Alpha", "Beta", "Gamma")])


class QualityDiagnosisTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = SectorPipeline(Settings(mock=True)).run(request())

    def test_unsupported_company_narrative_routes_to_analysis(self):
        result = deepcopy(self.base)
        result.fact_sheets["Beta"]["fact_sheet"] = BAD  # Another company's copy is not Alpha's research provenance.
        result.analyses["Alpha"]["operating_assessment"] = BAD
        findings = publication_findings(result, render_brief(result))
        found = next(item for item in findings if item["target"] == BAD)
        self.assertEqual((found["owner"], found["location"]), ("analysis", "analyses.Alpha.operating_assessment"))

    def test_fact_repeated_from_research_routes_back_to_research(self):
        result = deepcopy(self.base)
        result.fact_sheets["Alpha"]["fact_sheet"] = BAD
        result.fact_sheets["Beta"]["fact_sheet"] = BAD
        result.analyses["Beta"]["operating_assessment"] = BAD
        findings = publication_findings(result, render_brief(result))
        found = next(item for item in findings if item["target"] == BAD)
        self.assertEqual((found["owner"], found["location"]), ("research", "company_researcher:Beta"))

    def test_missing_review_is_an_audit_task_instead_of_a_rewrite(self):
        result = deepcopy(self.base)
        result.decision_fact_check["reasoning_checks"] = []
        findings = publication_findings(result, render_brief(result))
        self.assertTrue(findings)
        self.assertTrue(all(item["owner"] == "audit" for item in findings))


class PublicationRepairTest(unittest.TestCase):
    def test_lead_repairs_memo_and_auditor_rechecks_without_new_research(self):
        pipeline = SectorPipeline(Settings(mock=True))
        original = pipeline._run
        phases = []

        def run(role, task, toolbox, agent_id, name, schema=None):
            phases.append(agent_id)
            output = original(role, task, toolbox, agent_id, name, schema)
            if agent_id == "chair_editor" and not pipeline._quality_active:
                output["decision"]["thesis"] = BAD
            return output

        pipeline._run = run
        result = pipeline.run(request())
        self.assertEqual(len(result.quality_repairs), 1)
        self.assertEqual(result.quality_repairs[0]["actions"], ["lead", "audit_final"])
        self.assertEqual(phases.count("research_bundle"), 1)
        self.assertEqual(phases.count("analysis_scores"), 1)
        self.assertEqual(len(result.main_agents), 5)
        self.assertFalse(result.quality_issues)
        self.assertFalse(result.errors)
        self.assertNotIn(BAD, result.decision["thesis"])
        self.assertEqual(result.decision_fact_check["reviewed_targets"], review_targets(result.decision))

    def test_analysis_repair_targets_the_affected_company_and_recomputes_downstream(self):
        pipeline = SectorPipeline(Settings(mock=True))
        original = pipeline._run
        phases, preserved = [], {}

        def run(role, task, toolbox, agent_id, name, schema=None):
            phases.append(agent_id)
            if agent_id == "quality_analysis":
                self.assertEqual(set(schema["properties"]["analyses"]["properties"]), {"Alpha"})
                preserved.update({d: s["score"] for d, s in pipeline.last_result.analyses["Beta"]["scores"].items()})
            output = original(role, task, toolbox, agent_id, name, schema)
            if agent_id == "chair_editor" and not pipeline._quality_active:
                output["companies"]["Alpha"]["operating_assessment"] = BAD
            return output

        pipeline._run = run
        result = pipeline.run(request())
        self.assertEqual(phases.count("quality_analysis"), 1)
        self.assertIn("quality_decision", phases)
        self.assertEqual(phases.count("research_bundle"), 1)
        self.assertEqual(preserved, {d: s["score"] for d, s in result.analyses["Beta"]["scores"].items()})
        self.assertFalse(result.quality_issues)
        self.assertEqual(len(pipeline.agents), 5)

    def test_publication_source_problem_reuses_research_and_invalidates_downstream(self):
        pipeline = SectorPipeline(Settings(mock=True))
        original = pipeline._run
        phases = []

        def run(role, task, toolbox, agent_id, name, schema=None):
            phases.append(agent_id)
            output = original(role, task, toolbox, agent_id, name, schema)
            if agent_id == "research_bundle" and not pipeline._quality_active:
                output["companies"]["Alpha"]["fact_sheet"] = BAD
            if agent_id == "chair_editor" and not pipeline._quality_active:
                output["companies"]["Alpha"]["operating_assessment"] = BAD
            return output

        pipeline._run = run
        result = pipeline.run(request())
        self.assertEqual(phases.count("research_bundle"), 2)
        self.assertEqual(phases.count("analysis_scores"), 2)
        self.assertEqual(result.material_versions["research"]["version"], 2)
        self.assertIn("quality_decision", phases)
        self.assertFalse(result.quality_issues)
        self.assertFalse(result.unresolved_checks)
        self.assertEqual(len(pipeline.agents), 5)

    def test_missing_final_audit_retries_audit_without_rewriting_valid_material(self):
        pipeline = SectorPipeline(Settings(mock=True))
        original = pipeline._run
        phases = []

        def run(role, task, toolbox, agent_id, name, schema=None):
            phases.append(agent_id)
            if agent_id == "fact_checker_decision":
                raise RuntimeError("一次模拟审计接口失败")
            return original(role, task, toolbox, agent_id, name, schema)

        pipeline._run = run
        result = pipeline.run(request())
        self.assertEqual(phases.count("chair_editor"), 1)
        self.assertEqual(phases.count("research_bundle"), 1)
        self.assertIn("quality_final_audit", phases)
        self.assertFalse(result.quality_issues)
        self.assertFalse(result.errors)

    def test_transient_repair_audit_failure_retries_audit_within_budget(self):
        pipeline = SectorPipeline(Settings(mock=True))
        original = pipeline._run
        phases = []

        def run(role, task, toolbox, agent_id, name, schema=None):
            phases.append(agent_id)
            if agent_id == "quality_final_audit" and phases.count(agent_id) == 1:
                raise RuntimeError("模拟修复后审计超时")
            output = original(role, task, toolbox, agent_id, name, schema)
            if agent_id == "chair_editor" and not pipeline._quality_active:
                output["decision"]["thesis"] = BAD
            return output

        pipeline._run = run
        result = pipeline.run(request())
        self.assertEqual(phases.count("quality_final_audit"), 2)
        self.assertEqual(phases.count("chair_editor"), 2)
        self.assertEqual(result.quality_repairs[0]["status"], "failed")
        self.assertFalse(result.quality_issues)
        self.assertFalse(result.errors)

    def test_disabling_automatic_repairs_keeps_the_failed_draft(self):
        pipeline = SectorPipeline(Settings(mock=True, publication_repair_rounds=0))
        original = pipeline._run

        def run(role, task, toolbox, agent_id, name, schema=None):
            output = original(role, task, toolbox, agent_id, name, schema)
            if agent_id == "chair_editor":
                output["decision"]["thesis"] = BAD
            return output

        pipeline._run = run
        result = pipeline.run(request())
        self.assertFalse(result.quality_repairs)
        self.assertTrue(result.quality_issues)
        self.assertNotEqual(result.report_status, "演示稿")

    def test_persistent_problem_stops_at_limit_and_exports_a_draft(self):
        pipeline = SectorPipeline(Settings(mock=True, publication_repair_rounds=2))
        original = pipeline._run

        def run(role, task, toolbox, agent_id, name, schema=None):
            output = original(role, task, toolbox, agent_id, name, schema)
            if agent_id == "chair_editor":
                output["decision"]["thesis"] = BAD
            return output

        pipeline._run = run
        result = pipeline.run(request())
        self.assertEqual(len(result.quality_repairs), 2)
        self.assertTrue(result.quality_issues)
        self.assertNotEqual(result.report_status, "演示稿")
        self.assertEqual(len(result.main_agents), 5)
        with tempfile.TemporaryDirectory() as root:
            folder = save_result(result, root)
            self.assertFalse((folder / "report.md").exists())
            self.assertTrue((folder / "research-draft.md").exists())
            self.assertIn("发布退回与定向修复", (folder / "audit.md").read_text(encoding="utf-8"))

    def test_new_failure_from_repair_audit_is_retained(self):
        pipeline = SectorPipeline(Settings(mock=True))
        original = pipeline._run
        new_problem = "修订稿新增的经营结论缺少依据"

        def run(role, task, toolbox, agent_id, name, schema=None):
            output = original(role, task, toolbox, agent_id, name, schema)
            if agent_id == "chair_editor" and not pipeline._quality_active:
                output["decision"]["thesis"] = BAD
            if agent_id == "quality_final_audit":
                claim = deepcopy(output["claims"][0])
                claim.update(target=new_problem, claim=new_problem, verdict="无法核实", source_agent="chair")
                output["claims"] = [claim]
            return output

        pipeline._run = run
        result = pipeline.run(request())
        self.assertTrue(any(item["target"] == new_problem for item in result.quality_issues))
        self.assertNotEqual(result.report_status, "演示稿")

    def test_unconfirmed_repair_cannot_pass_even_when_text_changes(self):
        pipeline = SectorPipeline(Settings(mock=True))
        original = pipeline._run

        def run(role, task, toolbox, agent_id, name, schema=None):
            output = original(role, task, toolbox, agent_id, name, schema)
            if agent_id == "chair_editor" and not pipeline._quality_active:
                output["decision"]["thesis"] = BAD
            if agent_id == "quality_final_audit":
                for row in output["repair_checks"]:
                    row.update(status="无法判断", reason="修复尚无依据")
            return output

        pipeline._run = run
        result = pipeline.run(request())
        self.assertTrue(result.unresolved_checks)
        self.assertTrue(result.quality_issues)
        self.assertNotEqual(result.report_status, "演示稿")

    def test_overlength_report_returns_to_lead_then_is_reaudited(self):
        pipeline = SectorPipeline(Settings(mock=True, report_max_pages=10))
        original = pipeline._run
        phases = []

        def run(role, task, toolbox, agent_id, name, schema=None):
            phases.append(agent_id)
            if agent_id == "quality_final_audit":
                self.assertNotIn("repair_checks", schema["properties"], "页数由渲染器验证，不让模型猜测实际页数")
            output = original(role, task, toolbox, agent_id, name, schema)
            if agent_id == "chair_editor" and not pipeline._quality_active:
                output["decision"]["thesis"] = "必须保留重大风险与下行情景。" * 2000
            return output

        pipeline._run = run
        result = pipeline.run(request())
        self.assertEqual(phases.count("chair_editor"), 3)
        self.assertIn("quality_final_audit", phases)
        self.assertLessEqual(result.report_pages, 10)
        self.assertFalse(result.quality_issues)
        self.assertFalse(any(row["stage"] == "篇幅预检" for row in result.unresolved_checks))

    def test_invalid_repair_budget_is_rejected_before_agents_run(self):
        for value in (-1, 4, True, 1.5):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "修复轮数"):
                SectorPipeline(Settings(mock=True, publication_repair_rounds=value)).run(request())


if __name__ == "__main__":
    unittest.main()
