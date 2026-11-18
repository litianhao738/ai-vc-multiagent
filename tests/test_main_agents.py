"""The default workflow owns exactly five real, persistent AgentScope agents."""
from copy import deepcopy
from dataclasses import replace
from io import BytesIO
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from agentscope.agent import Agent as ScopeAgent
from pypdf import PdfReader

from vcagents import CompanyInput, DocumentStore, MainAgentPipeline, SectorPipeline, SectorRequest, Settings, save_result
from vcagents.handoffs import MaterialBoard, HandoffError
from vcagents import schemas
from vcagents.llm import LLMRouter
from vcagents.main_roles import MAIN_ROLES
from vcagents.pipeline import _namespace_sources
from vcagents.roles import FACT_CHECKER
from vcagents.tools import normalize_fact_check
from vcagents.reasoning import review_targets


def request(count=3):
    return SectorRequest("aigc-content", [CompanyInput(name) for name in ("Alpha", "Beta", "Gamma", "Delta", "Epsilon")[:count]])


class MaterialHandoffTest(unittest.TestCase):
    def test_consumer_cannot_mutate_shared_facts(self):
        board = MaterialBoard()
        source = {"claim": {"amount": 100}, "evidence_ids": ["E-C1-1"]}
        ref = board.publish("research", source)
        source["claim"]["amount"] = 900
        snapshot = board.read(ref)
        snapshot["claim"]["amount"] = 800
        self.assertEqual(board.read(ref)["claim"]["amount"], 100)

    def test_stale_versions_wrong_hash_and_conflicting_writer_are_rejected(self):
        board = MaterialBoard()
        first = board.publish("research", {"claim": "old"})
        with self.assertRaises(HandoffError):
            board.read(replace(first, sha256="wrong"))
        second = board.publish("research", {"claim": "corrected"}, expected_version=first.version)
        with self.assertRaises(HandoffError):
            board.read(first)
        with self.assertRaises(HandoffError):
            board.publish("research", {"claim": "stale writer"}, expected_version=first.version)
        self.assertEqual(board.read(second)["claim"], "corrected")


class FiveMainAgentsTest(unittest.TestCase):
    def test_deepseek_research_override_preserves_the_other_core_routes(self):
        vendors = {"lead": "google", "research": "moonshotai", "analysis": "anthropic",
                   "bear": "openai", "audit": "z-ai"}
        self.assertEqual(set(MAIN_ROLES), set(vendors))
        defaults = Settings().role_models
        self.assertEqual({role: defaults[role].split("/")[0] for role in MAIN_ROLES}, vendors)
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "models.yaml"
            config.write_text("research: deepseek/deepseek-v4.1-flash-1017\n", encoding="utf-8")
            settings = Settings.load(config, openrouter_api_key="test")
        router = LLMRouter(settings)
        for role in MAIN_ROLES:
            with self.subTest(role=role):
                client = router.for_role(role)
                self.addCleanup(client.client.close)
                expected = "deepseek/deepseek-v4.1-flash-1017" if role == "research" else defaults[role]
                self.assertEqual(client.model, expected)

    def test_bull_can_defend_or_revise_with_a_conclusion_and_a_reason(self):
        position = {"thesis": "同一批到期客户的续费率上升。", "arguments": [],
                    "rebuttals": [], "would_change_mind": "复核到期合同及续费口径。"}
        self.assertEqual(schemas.validate(schemas.DEBATE_POSITION, position), [])
        opening = schemas.comparison_position_schema(["Alpha"])["properties"]["bull"]
        self.assertTrue(schemas.validate(opening, {**position, "thesis": " "}))
        for thesis, replies, valid in (
            (position["thesis"], ["新增客户不在本期到期客户的分母中。"], True),
            ("目前只能确认总使用量增长。", ["若统计含新增客户，总量无法说明老客留存。"], True),
            (position["thesis"], [], False), (position["thesis"], [" "], False), ("", ["同意这个质疑。"], False),
        ):
            with self.subTest(thesis=thesis, replies=replies):
                response = {**position, "thesis": thesis, "rebuttals": replies}
                self.assertEqual(schemas.validate(schemas.BULL_RESPONSE, response) == [], valid)

    def test_five_speeches_use_two_contexts_and_respond_to_prior_turns(self):
        pipeline = SectorPipeline(Settings(mock=True, debate_rounds=2))
        original = pipeline._run
        phases = ["analysis_compare", "bear_r1", "bull_r2", "bear_r2", "bull_close"]
        speakers = []
        leads = []
        checkpoints = []

        def run(role, task, toolbox, agent_id, name, schema=None):
            if agent_id in phases:
                self.assertEqual(agent_id, phases[len(speakers)])
                for prior in phases[:len(speakers)]:
                    self.assertIn("speech-" + prior, task)
                speakers.append(pipeline._group(role))
                if agent_id in ("bear_r1", "bull_close"):
                    self.assertIn("detail-from-opening", task)
                elif agent_id in ("bull_r2", "bear_r2"):
                    self.assertNotIn("detail-from-opening", task)
            if agent_id == "chair":
                leads.append(task)
            if agent_id == "lead_inquiry_0":
                checkpoints.append(task)
            if agent_id in ("bear_r1", "bear_r2"):
                self.assertFalse(role.tools)
            if agent_id == "bull_r2":
                self.assertEqual(schema, schemas.BULL_RESPONSE)
            if agent_id == "bull_close":
                self.assertEqual(schema["properties"]["bull"], schemas.BULL_RESPONSE)
            output = original(role, task, toolbox, agent_id, name, schema)
            if agent_id in ("bear_r1", "bear_r2"):
                self.assertTrue(schemas.validate(schema, {**output, "target": "invented"}))
                output["target"] = "thesis"
            if agent_id in phases:
                if agent_id in ("analysis_compare", "bull_close"):
                    output["comparison"]["summary"] = "comparison-" + agent_id
                    if agent_id == "analysis_compare":
                        output["comparison"]["per_company"][0]["edge"] = "detail-from-opening"
                    position = output["bull"]
                else:
                    position = output
                position["thesis"] = "speech-" + agent_id
            return output

        pipeline._run = run
        result = pipeline.run(request())
        self.assertEqual(speakers, ["analysis", "bear", "analysis", "bear", "analysis"])
        self.assertIsNot(pipeline.agents["analysis"].scope_agent, pipeline.agents["bear"].scope_agent)
        self.assertEqual(pipeline.agents["bear"].reply_count, 2)
        self.assertEqual([d["round"] for d in result.debate], [1, 1, 2, 2, 2])
        self.assertTrue(all("speech-" + phase in leads[0] for phase in phases))
        for task in (checkpoints[0], leads[0]):
            self.assertIn("comparison-bull_close", task)
            self.assertNotIn("comparison-analysis_compare", task)
        analysis = pipeline.board.read(pipeline.board.refs("analysis")[0])
        self.assertEqual(analysis["comparison"]["summary"], "comparison-bull_close")
        self.assertEqual([p["addressed_claim"] for p in analysis["perspectives"] if p["side"] == "bear"],
                         ["speech-analysis_compare", "speech-bull_r2"])
        materials = pipeline._check_materials(result)
        self.assertTrue({"bull_r1", "bear_r1", "bull_r2", "bear_r2", "bull_close"} <= materials.keys())

    def test_failed_rebuttal_or_closing_cannot_publish_a_partial_case(self):
        for failed_phase in ("bear_r2", "bull_close"):
            with self.subTest(phase=failed_phase):
                pipeline = SectorPipeline(Settings(mock=True))
                original = pipeline._run

                def run(role, task, toolbox, agent_id, name, schema=None):
                    if agent_id == failed_phase:
                        raise RuntimeError(failed_phase)
                    return original(role, task, toolbox, agent_id, name, schema)

                pipeline._run = run
                with self.assertRaisesRegex(RuntimeError, failed_phase):
                    pipeline.run(request())
                self.assertFalse(pipeline.last_result.debate)
                self.assertFalse(pipeline.last_result.comparison)
                self.assertFalse(pipeline.last_result.decision)

    def test_public_default_is_five_main_agents_for_one_to_five_named_companies(self):
        self.assertIs(SectorPipeline, MainAgentPipeline)
        for count in range(1, 6):
            with self.subTest(companies=count):
                pipeline = SectorPipeline(Settings(mock=True))
                result = pipeline.run(request(count))
                self.assertEqual(set(result.main_agents), {"lead", "research", "analysis", "bear", "audit"})
                self.assertEqual(set(result.usage), set(MAIN_ROLES))
                self.assertEqual(len(pipeline.agents), 5)
                self.assertEqual(len(result.companies), count)
                self.assertTrue(all(isinstance(a.scope_agent, ScopeAgent) and a.persistent for a in pipeline.agents.values()))
                self.assertEqual(set(result.analyses), set(result.companies))
                self.assertFalse(result.errors)
                self.assertFalse(any("lane" in key or "company_analyst:" in key for key in result.usage))

    def test_all_phases_reuse_the_same_five_native_agent_objects(self):
        pipeline = SectorPipeline(Settings(mock=True))
        native_ids = {}
        original = pipeline._run

        def run(role, task, toolbox, agent_id, name, schema=None):
            output = original(role, task, toolbox, agent_id, name, schema)
            for group, agent in pipeline.agents.items():
                native_ids.setdefault(group, set()).add(id(agent.scope_agent))
            return output

        pipeline._run = run
        result = pipeline.run(request())
        self.assertTrue(all(len(identities) == 1 for identities in native_ids.values()))
        self.assertGreaterEqual(pipeline.agents["lead"].reply_count, 3)
        self.assertEqual(pipeline.agents["research"].reply_count, 1)
        self.assertEqual(pipeline.agents["analysis"].reply_count, 4)
        self.assertGreaterEqual(pipeline.agents["audit"].reply_count, 2)
        self.assertGreaterEqual(len(result.handoffs), 8)
        self.assertEqual(len(result.main_agents), 5)

    def test_analysis_and_audit_can_overlap_without_shared_agent_state(self):
        rendezvous = threading.Barrier(2, timeout=5)
        pipeline = SectorPipeline(Settings(mock=True, max_workers=4))
        original = pipeline._run

        def run(role, task, toolbox, agent_id, name, schema=None):
            if agent_id in ("analysis_scores", "audit_research"):
                rendezvous.wait()
            return original(role, task, toolbox, agent_id, name, schema)

        pipeline._run = run
        self.assertFalse(pipeline.run(request()).errors)

    def test_candidate_discovery_reuses_the_research_agent_for_the_requested_count(self):
        for count in range(1, 6):
            with self.subTest(companies=count):
                pipeline = SectorPipeline(Settings(mock=True))
                result = pipeline.run(SectorRequest("embodied-ai", max_companies=count))
                self.assertEqual(len(result.main_agents), 5)
                self.assertEqual(pipeline.agents["research"].reply_count, 2)
                self.assertEqual(len(result.companies), count)
                self.assertEqual(list(result.fact_sheets), result.companies)

    def test_material_revision_during_analysis_is_not_silently_accepted(self):
        pipeline = SectorPipeline(Settings(mock=True))
        original = pipeline._run

        def run(role, task, toolbox, agent_id, name, schema=None):
            output = original(role, task, toolbox, agent_id, name, schema)
            if agent_id == "analysis_scores":
                pipeline.board.publish("research", {"changed": "new version"})
            return output

        pipeline._run = run
        with self.assertRaises(HandoffError):
            pipeline.run(request())
        self.assertEqual(pipeline.last_result.status, "failed")

    def test_evidence_and_pdf_export_contracts_are_kept(self):
        docs = DocumentStore()
        docs.add_text("BP.txt", "Alpha 团队、收入与融资均为虚构演示材料。")
        pipeline = SectorPipeline(Settings(mock=True), {"Alpha": docs})
        result = pipeline.run(request(5))
        self.assertTrue(result.source_reads.get("research"))
        self.assertTrue(all(h["status"] == "accepted" for h in result.handoffs))
        self.assertEqual(result.material_versions["research"]["version"], 1)
        self.assertIn("final_audit", result.material_versions)
        self.assertEqual(result.decision_fact_check["reviewed_targets"], review_targets(result.decision))
        with tempfile.TemporaryDirectory() as root:
            folder = save_result(result, root)
            name = "report.pdf" if (folder / "report.pdf").exists() else "research-draft.pdf"
            self.assertLessEqual(len(PdfReader(folder / name).pages), 10)
            self.assertIn("5 个主Agent", (folder / "audit.md").read_text(encoding="utf-8"))
            self.assertIn("材料交接", (folder / "audit.md").read_text(encoding="utf-8"))

    def test_old_extra_agent_features_require_explicit_expanded_mode(self):
        with self.assertRaisesRegex(ValueError, "expanded"):
            SectorPipeline(Settings(mock=True, pairwise_checks=True)).run(request())

    def test_research_correction_reuses_agents_and_invalidates_old_materials(self):
        pipeline = SectorPipeline(Settings(mock=True))
        original = pipeline._run
        first = True

        def run(role, task, toolbox, agent_id, name, schema=None):
            nonlocal first
            output = original(role, task, toolbox, agent_id, name, schema)
            if agent_id == "audit_research" and first:
                first = False
                claim = deepcopy(output["claims"][0])
                claim.update(claim="错误历史融资", verdict="不实", source_agent="company_researcher:Alpha",
                             original_source="", correction="删除错误主张", evidence="原文没有支持")
                output["claims"] = [claim]
            return output

        pipeline._run = run
        result = pipeline.run(request())
        self.assertEqual(len(pipeline.agents), 5)
        self.assertEqual(pipeline.agents["research"].reply_count, 2)
        self.assertEqual(pipeline.agents["analysis"].reply_count, 8)
        self.assertEqual(result.material_versions["research"]["version"], 2)
        self.assertTrue(result.repairs)
        self.assertFalse(result.errors)
        self.assertFalse(result.unresolved_checks)

    def test_repair_confirmation_cannot_replace_a_fresh_research_audit(self):
        for verdict in ("不实", "无法核实", None):
            with self.subTest(verdict=verdict):
                pipeline = SectorPipeline(Settings(mock=True))
                original_run, original_finish = pipeline._run, pipeline._finish_result
                research_audits = 0
                new_claim = "补查后新增的经营主张缺少原文支持"

                def run(role, task, toolbox, agent_id, name, schema=None):
                    nonlocal research_audits
                    output = original_run(role, task, toolbox, agent_id, name, schema)
                    if agent_id == "audit_research":
                        research_audits += 1
                        text = "应删除的旧融资主张" if research_audits == 1 else new_claim
                        claim = deepcopy(output["claims"][0])
                        claim.update(target=text, claim=text, source_agent="company_researcher:Alpha",
                                     verdict=verdict, original_source="", evidence="原文不支持", correction="删除")
                        output["claims"] = [] if research_audits > 1 and verdict is None else [claim]
                        if research_audits == 1:
                            claim["verdict"] = "不实"
                    elif agent_id == "audit_repair":
                        if verdict is not None:
                            output["claims"] = []
                        for check in output["repair_checks"]:
                            check.update(status="已删除", reason="旧问题已删除")
                    elif agent_id == "quality_research_audit" and verdict is None:
                        output["claims"] = []  # Missing coverage persists even after the new explicit retry.
                    return output

                def finish(result, started):
                    # Exercise production publication status with offline model responses.
                    with patch.object(pipeline.settings, "mock", False):
                        original_finish(result, started)

                pipeline._run, pipeline._finish_result = run, finish
                result = pipeline.run(request(1))
                self.assertEqual(research_audits, 2)
                self.assertEqual(len(result.main_agents), 5)
                if verdict is not None:
                    self.assertIn(new_claim, [c["claim"] for c in result.fact_check["claims"]])
                    self.assertTrue(any(c["claim"] == new_claim for c in result.unresolved_checks))
                    self.assertNotIn("R-1", [c["claim_id"] for c in result.fact_check["claims"]])
                else:
                    self.assertIn("audit_research", result.errors)
                self.assertNotIn(result.report_status, ("演示稿", "待人工复核的研究报告"))
                audit_ref = pipeline.board.refs("research_audit")[0]
                self.assertEqual(pipeline.board.read(audit_ref), result.fact_check)


if __name__ == "__main__":
    unittest.main()
