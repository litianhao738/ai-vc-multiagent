"""Question-driven additions retain the five-agent pipeline and publication gates."""
from copy import deepcopy
from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import unittest

from vcagents import CompanyInput, SectorPipeline, SectorRequest, Settings, save_result
from vcagents.handoffs import HandoffError
from vcagents.inquiry import (initial_state, checkpoint_errors, apply_checkpoint, invalidate_answers,
                             replay_method_case)
from vcagents.memory import ResearchMemory
from vcagents.pipeline import AnalysisResult, PipelineError


def request():
    return SectorRequest("aigc-content", [CompanyInput(name) for name in ("Alpha", "Beta", "Gamma")])


def checkpoint(state, action="finish", *, scope="Alpha", new_question=False):
    qid = state["questions"][0]["question_id"]
    return {"next_action": action, "reason": "补查用户成本有助于区分不同解释" if action == "research" else "保留缺口并成稿",
            "working_hypothesis": "需要验证任务成本与工作流差异", "question_updates": [],
            "follow_up_tasks": [{"question_id": "" if new_question else qid,
                "question": "相同任务下，差异能否持续？" if new_question else "", "scope": scope,
                "method": "读取同口径产品定价和功能原文，对照任务成本；不冒充实测",
                "expected_impact": "影响对产品优势持续性的判断"}] if action == "research" else []}


def scripted_pipeline(*, rounds=1, repeat=False, followup_failure=False, downstream_failure=False):
    pipeline = SectorPipeline(Settings(mock=True, research_refinement_rounds=rounds))
    original = pipeline._run
    observations = {"phases": [], "scores": [], "native": {}}

    def run(role, task, toolbox, agent_id, name, schema=None):
        observations["phases"].append(agent_id)
        if agent_id.startswith("research_followup_") and followup_failure:
            raise TimeoutError("补查接口超时")
        if agent_id == "analysis_scores" and observations["scores"] and downstream_failure:
            raise ValueError("补查后的分析失败")
        data = original(role, task, toolbox, agent_id, name, schema)
        for key, agent in pipeline.agents.items():
            observations["native"].setdefault(key, set()).add(id(agent.scope_agent))
        if agent_id == "analysis_scores":
            observations["scores"].append(deepcopy(data["analyses"]))
        if agent_id.startswith("lead_inquiry_"):
            number = int(agent_id.rsplit("_", 1)[-1])
            return checkpoint(pipeline.last_result.research_state, "research" if number == 0 or repeat else "finish")
        if agent_id.startswith("research_followup_"):
            data["companies"]["Alpha"]["one_liner"] += "补查后的产品观察"
        if agent_id == "audit_inquiry":
            return {"claims": [], "summary": "演示数据不验证方法价值", "question_checks": [
                {"question_id": qid, "verdict": "unresolved", "reason": "缺真实证据", "evidence_ids": []}
                for qid in pipeline._inquiry_audit_questions], "method_checks": [
                {"method_id": mid, "verdict": "not_demonstrated", "reason": "只有演示内容", "evidence_ids": [],
                 "limitations": "需要真实原文与实际研究结果"} for mid in pipeline._inquiry_audit_methods]}
        return data

    pipeline._run = run
    return pipeline, observations


class InquiryStateTest(unittest.TestCase):
    def test_questions_have_stable_ids_and_user_selection_is_recorded_honestly(self):
        state = initial_state({"research_questions": ["成本如何？", "成本如何？", "为何持续？"]}, True)
        self.assertEqual([q["question_id"] for q in state["questions"]], ["Q1", "Q2"])
        self.assertEqual(state["selection_basis"], "用户指定研究对象")
        self.assertIn("推荐", initial_state({}, False)["selection_basis"])

    def test_new_questions_and_hypothesis_updates_do_not_rewrite_initial_agenda(self):
        plan = {"research_questions": ["成本如何？"], "working_hypothesis": "原假设"}
        state = initial_state(plan, True)
        data = checkpoint(state, "research", new_question=True)
        self.assertFalse(checkpoint_errors(data, state, set(), {"sector", "Alpha"}))
        tasks = apply_checkpoint(state, data, 1)
        self.assertEqual(tasks[0]["question_id"], "Q2")
        self.assertEqual(tasks[0]["task_id"], "T1-1")
        self.assertEqual(plan["working_hypothesis"], "原假设")
        self.assertEqual(state["initial_hypothesis"], "原假设")

    def test_answered_without_current_evidence_or_unknown_task_is_rejected(self):
        state = initial_state({}, True)
        data = checkpoint(state, "research", scope="Other")
        data["question_updates"] = [{"question_id": "Q1", "status": "answered", "answer": "结论",
            "supporting_evidence_ids": ["old-id"], "contradicting_evidence_ids": []}]
        self.assertTrue(checkpoint_errors(data, state, set(), {"Alpha", "sector"}))

    def test_alternative_explanation_can_stay_open_without_counterevidence(self):
        state = initial_state({"research_questions": ["客户是否主动续约？"]}, True)
        data = checkpoint(state)
        update = {"question_id": "Q1", "status": "open", "answer": "",
                  "supporting_evidence_ids": [], "contradicting_evidence_ids": [],
                  "alternative_explanations": ["如果合同还未到期，继续使用也可能由合同约束解释。"],
                  "missing_information": ["到期后客户的选择"], "decision_impact": "影响留存解释",
                  "next_step": "核对有退出权的客户是否自愿续约。"}
        data["question_updates"] = [update]
        self.assertEqual(checkpoint_errors(data, state, set(), {"Alpha", "sector"}), [])
        apply_checkpoint(state, data, 1)
        question = state["questions"][0]
        self.assertEqual(question["alternative_explanations"], update["alternative_explanations"])
        self.assertEqual(question["contradicting_evidence_ids"], [])

        update.update(status="answered", answer="客户受合同约束。")
        self.assertTrue(checkpoint_errors(data, state, set(), {"Alpha", "sector"}))

    def test_material_revision_invalidates_previous_answers_and_methods(self):
        state = initial_state({}, True)
        state["research_sha256"] = "old"
        state["questions"][0].update(status="answered", audit_verdict="supported")
        methods = [{"status": "reviewed"}]
        invalidate_answers(state, methods, "new")
        self.assertEqual(state["questions"][0]["status"], "open")
        self.assertEqual(state["questions"][0]["audit_verdict"], "stale")
        self.assertEqual(methods[0]["status"], "stale")


class InquiryPipelineTest(unittest.TestCase):
    def test_followup_reuses_five_agents_and_preserves_unrelated_scores_and_report_framework(self):
        pipeline, observations = scripted_pipeline()
        result = pipeline.run(request())
        self.assertEqual(len(pipeline.agents), 5)
        self.assertTrue(all(len(ids) == 1 for ids in observations["native"].values()))
        self.assertEqual(len(observations["scores"]), 2)
        self.assertEqual(set(observations["scores"][1]), {"Alpha"})
        self.assertEqual({key: score["score"] for key, score in observations["scores"][0]["Beta"]["scores"].items()},
                         {key: score["score"] for key, score in result.analyses["Beta"]["scores"].items()})
        self.assertEqual(result.material_versions["research"]["version"], 2)
        self.assertEqual(pipeline._repair_count, 0, "研究深化不消耗原事实纠错次数")
        self.assertFalse(result.errors)
        self.assertEqual(len([r for r in result.research_iterations if r["kind"] == "followup"]), 1)
        self.assertEqual(result.research_methods[0]["validation"]["verdict"], "not_demonstrated")
        self.assertTrue(result.fact_check["claims"], "方法复核不能覆盖原研究事实核查")
        with tempfile.TemporaryDirectory() as root:
            folder = save_result(result, root)
            memo = next(folder.glob("*.md"))
            report = folder / ("report.md" if (folder / "report.md").exists() else "research-draft.md")
            text = report.read_text(encoding="utf-8")
            self.assertIn("## 一、投资价值与技术护城河", text)
            self.assertIn("## 七、尽调计划与附录", text)
            self.assertLessEqual(result.report_pages, 10)
            self.assertIn("问题驱动研究", (folder / "audit.md").read_text(encoding="utf-8"))

    def test_round_cap_and_duplicate_task_prevent_unbounded_research(self):
        for rounds in (1, 2):
            with self.subTest(rounds=rounds):
                pipeline, seen = scripted_pipeline(rounds=rounds, repeat=True)
                result = pipeline.run(request())
                self.assertEqual(sum(phase.startswith("research_followup_") for phase in seen["phases"]), 1)
                self.assertTrue("上限" in result.research_state["stop_reason"] or "重复" in result.research_state["stop_reason"])

    def test_refinement_can_be_disabled_to_run_the_original_path(self):
        pipeline, seen = scripted_pipeline(rounds=0)
        result = pipeline.run(request())
        self.assertFalse(any(phase.startswith("lead_inquiry_") for phase in seen["phases"]))
        self.assertEqual(pipeline.agents["research"].reply_count, 1)
        self.assertEqual(pipeline.agents["analysis"].reply_count, 4)
        self.assertFalse(result.research_iterations)

    def test_followup_failure_before_commit_keeps_existing_materials_and_report(self):
        pipeline, _ = scripted_pipeline(followup_failure=True)
        result = pipeline.run(request())
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.material_versions["research"]["version"], 1)
        self.assertTrue(result.decision)
        self.assertFalse(result.errors)
        self.assertEqual(next(r for r in result.research_iterations if r["kind"] == "followup")["status"], "failed")

    def test_downstream_failure_after_new_research_cannot_publish_old_decision(self):
        pipeline, _ = scripted_pipeline(downstream_failure=True)
        with self.assertRaises(PipelineError):
            pipeline.run(request())
        self.assertEqual(pipeline.last_result.status, "failed")
        self.assertFalse(pipeline.last_result.decision)
        self.assertEqual(pipeline.last_result.material_versions["research"]["version"], 2)

    def test_scope_and_pin_checks_are_not_bypassed_by_optional_checkpoint(self):
        pipeline = SectorPipeline(Settings(mock=True))
        original = pipeline._run

        def run(role, task, box, phase, name, schema=None):
            data = original(role, task, box, phase, name, schema)
            if phase.startswith("lead_inquiry_"):
                return checkpoint(pipeline.last_result.research_state, "research", scope="Other")
            return data

        pipeline._run = run
        result = pipeline.run(request())
        self.assertFalse(any(row["kind"] == "followup" for row in result.research_iterations))
        self.assertEqual(result.research_iterations[0]["status"], "unavailable")
        self.assertEqual(result.material_versions["research"]["version"], 1)

    def test_invalid_refinement_setting_is_rejected_before_work(self):
        for value in (-1, 3, True, 1.5):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "补查轮数"):
                SectorPipeline(Settings(mock=True, research_refinement_rounds=value)).run(request())

    def test_stale_materials_cannot_be_swallowed_as_an_optional_checkpoint_failure(self):
        pipeline = SectorPipeline(Settings(mock=True))
        original = pipeline._run

        def run(role, task, box, phase, name, schema=None):
            if phase.startswith("lead_inquiry_"):
                raise HandoffError("材料交接版本失效")
            return original(role, task, box, phase, name, schema)

        pipeline._run = run
        with self.assertRaises(HandoffError):
            pipeline.run(request())
        self.assertEqual(pipeline.last_result.status, "failed")

    def test_checkpoint_timeout_falls_back_to_original_report_flow(self):
        pipeline = SectorPipeline(Settings(mock=True))
        original = pipeline._run

        def run(role, task, box, phase, name, schema=None):
            if phase.startswith("lead_inquiry_"):
                raise TimeoutError("附加研究决策超时")
            return original(role, task, box, phase, name, schema)

        pipeline._run = run
        result = pipeline.run(request())
        self.assertEqual(result.status, "completed")
        self.assertTrue(result.decision)
        self.assertFalse(result.errors)
        self.assertIn("未完成", result.research_state["stop_reason"])


def method_case():
    result = AnalysisResult("aigc-content", "AIGC", status="completed", report_status="待人工复核的研究报告",
                            models={"lead": "scripted"})
    evidence = {"evidence_id": "E-C1-1", "claim": "Alpha月费10美元", "fact_type": "其他", "as_of": "2026-11-17",
        "quantities": [{"metric": "定价", "text": "$10"}], "source_type": "公司口径", "excerpt": "Alpha costs $10.",
        "source_linked": True, "excerpt_verified": True,
        "source_snapshots": [{"snapshot_id": "snap", "content_sha256": "digest", "url": "https://alpha.example/pricing"}]}
    result.fact_sheets = {"Alpha": {"evidence": [evidence]}}
    result.fact_check = {"claims": [{"evidence_ids": ["E-C1-1"], "verdict": "公司口径", "support": "完全支持",
                                     "excerpt_verified": True}]}
    result.material_versions = {"research": {"sha256": "current"}}
    result.research_state = initial_state({}, True)
    result.research_state["questions"][0].update(status="answered", audit_verdict="supported", answer="已核对报价")
    task = {"task_id": "T1-1", "method": "读取同日定价原文，按统一计费单位比较"}
    result.research_iterations = [{"kind": "followup", "status": "completed", "tasks": [task]}]
    result.handoffs = [{"phase": "research_followup_1", "status": "accepted"}]
    method = {"method_id": "M1-1", "task_id": "T1-1", "question_id": "Q1", "phase": "research_followup_1",
              "scope": "Alpha", "question": "产品报价差异是什么？", "procedure": task["method"],
              "status": "reviewed", "research_sha256": "current", "validation": {"verdict": "useful",
                  "reason": "实际定价原文缩小了计费口径差异", "evidence_ids": ["E-C1-1"],
                  "limitations": "只能比较公开报价，不能推断内部交付成本"}}
    result.research_methods = [method]
    return result


class MethodCaseReplayTest(unittest.TestCase):
    def test_subsequent_report_records_only_selected_and_executed_historical_methods(self):
        root = self.enterContext(tempfile.TemporaryDirectory())
        memory = ResearchMemory(Path(root) / "memory.sqlite3")
        memory.remember_result(method_case())
        original = memory.recall_methods("aigc-content")[0]
        for outcome in ("useful", "no_change", "inconclusive", "failed", "private", "unselected", "stale"):
            result = method_case()
            result.research_state["method_memory"] = [original] if outcome != "unselected" else []
            task = result.research_iterations[0]
            task.update(round=1, research_seconds=10, after={"research": {"sha256": "current"}})
            task["tasks"][0]["origin_method_id"] = original["method_id"]
            method = result.research_methods[0]
            method["origin_method_id"] = original["method_id"]
            if outcome in ("no_change", "failed"):
                task["status"] = outcome
                result.research_methods = []
            elif outcome == "inconclusive":
                method["validation"]["verdict"] = "not_demonstrated"
            elif outcome == "private":
                result.memory_config["private_inputs"] = True
            elif outcome == "stale":
                method["research_sha256"] = "old"
            counts = memory.remember_result(result)
            self.assertEqual(counts.get("method_reuse", 0), int(outcome in ("useful", "inconclusive", "no_change")))
            self.assertFalse(memory.remember_result(result).get("method_reuse"))
        self.assertEqual(memory.recall_methods("aigc-content")[0]["reuse"],
                         {"trials": 3, "useful": 1, "no_change": 1, "inconclusive": 1})
        self.assertEqual(len(memory.recall_methods("aigc-content")), 1)
        self.assertEqual(memory.learning.reuse_history("aigc-content")[0]["batch_seconds"], 10)

    def test_public_independently_reviewed_case_can_be_replayed_and_remembered_as_method_only(self):
        result = method_case()
        self.assertTrue(replay_method_case(result, result.research_methods[0])["accepted"])
        with tempfile.TemporaryDirectory() as root:
            memory = ResearchMemory(Path(root) / "memory.sqlite3")
            self.assertEqual(memory.remember_result(result)["methods"], 1)
            methods = memory.recall_methods("aigc-content")
            self.assertEqual(len(methods), 1)
            self.assertEqual(methods[0]["validated_cases"], 1)
            self.assertNotIn("E-C1-1", str(methods))
            self.assertIn("不作为公司事实", methods[0]["usage"])
            self.assertFalse(memory.recall_methods("embodied-ai"))
            self.assertEqual(memory.remember_result(result)["methods"], 0)

    def test_failed_private_stale_unexecuted_or_self_approved_methods_are_not_promoted(self):
        for scenario in ("draft", "private", "stale", "not_executed", "not_supported", "no_snapshot", "model_only"):
            with self.subTest(scenario=scenario):
                result = method_case()
                method = result.research_methods[0]
                if scenario == "draft":
                    result.report_checks = ["仍有发布问题"]
                elif scenario == "private":
                    result.source_reads = {"research": [{"kind": "document"}]}
                elif scenario == "stale":
                    result.material_versions["research"]["sha256"] = "new"
                elif scenario == "not_executed":
                    result.research_iterations = []
                elif scenario == "not_supported":
                    result.fact_check = {}
                elif scenario == "no_snapshot":
                    result.fact_sheets["Alpha"]["evidence"][0]["source_snapshots"] = []
                else:
                    method["status"] = "candidate"
                self.assertFalse(replay_method_case(result, method)["accepted"])

    def test_expired_method_is_not_retrieved(self):
        with tempfile.TemporaryDirectory() as root:
            memory = ResearchMemory(Path(root) / "memory.sqlite3")
            memory.remember_result(method_case())
            with closing(memory._connect()) as db, db:
                db.execute("UPDATE research_method_memory SET expires_at='2000-01-01'")
            self.assertFalse(memory.recall_methods("aigc-content"))


if __name__ == "__main__":
    unittest.main()
