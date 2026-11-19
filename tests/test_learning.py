from contextlib import closing
from datetime import date, timedelta
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import Mock, patch

from vcagents import CompanyInput, SectorPipeline, SectorRequest, Settings
from vcagents.learning import editor_version
from vcagents.learning_replay import audit_summary, generate_trial, replay_audit
from vcagents.llm import ToolCall, Turn
from vcagents.memory import ResearchMemory
from vcagents.pipeline import AnalysisResult, save_result
from vcagents.sectors import load_sector


class LearningTest(unittest.TestCase):
    def setUp(self):
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.settings = Settings(memory_path=str(root / "memory.sqlite3"), knowledge_path=str(root / "sources.sqlite3"))
        self.memory = ResearchMemory(self.settings.memory_path)
        self.journal = self.memory.learning
        self.sector = load_sector("aigc-content")
        self.version = editor_version(self.settings.role_models["lead"])

    def propose(self, rule="只在有同条件对照时描述实测差异。", parent=None):
        return self.journal.propose(self.sector.key, self.version, "Alpha", "原判断", "意见", rule, parent)

    def vote_candidate(self, rid, company, sparse):
        self.journal.save_trial(rid, company, company + " 公开摘录", sparse, "基线", "候选")
        trial = next(t for t in self.journal.trials(rid) if t["company"] == company)
        self.assertNotIn("candidate_side", trial)
        return self.journal.vote(rid, trial["company_key"], "A" if trial["text_a"] == "候选" else "B", True)

    def test_existing_two_table_db_upgrades_in_place(self):
        path = Path(self.settings.memory_path).with_name("old.sqlite3")
        with closing(sqlite3.connect(path)) as db, db:
            db.execute("""CREATE TABLE research_method_memory (id TEXT PRIMARY KEY, sector TEXT, procedure TEXT,
                question TEXT, limitations TEXT, cases TEXT, recorded_at TEXT, expires_at TEXT)""")
            db.execute("INSERT INTO research_method_memory VALUES ('old','ai','旧方法','原问题','局限','[]','2020','2099')")
        self.assertEqual(ResearchMemory(path).recall_methods("ai")[0]["procedure"], "旧方法")
        self.assertEqual(len(ResearchMemory(path).learning.audit_pairs()), 2)

    def test_feedback_deduplicates_and_unvoted_text_cannot_reach_editor(self):
        self.assertEqual(self.propose(""), self.propose(""))
        self.assertEqual(self.propose(), self.propose())
        self.assertFalse(self.journal.active_rule(self.sector.key, self.version)["rule"])
        with self.assertRaises(ValueError):
            self.journal.prepare_trial(self.propose(), " Ａｌｐｈａ ", "新材料")

    def test_blind_promotion_is_scoped_pinned_and_reversible(self):
        rid = self.propose()
        stale = self.propose("另一版准则")
        self.assertFalse(self.vote_candidate(rid, "Bravo", False))
        with self.assertRaises(ValueError):
            self.journal.save_trial(rid, "别名", "Bravo 公开摘录", True, "旧", "新")
        self.assertFalse(self.vote_candidate(rid, "Charlie", False))
        self.assertTrue(self.vote_candidate(rid, "Delta", True))
        with self.assertRaises(ValueError):
            self.journal.prepare_trial(stale, "新公司", "材料")
        for sector, version in (("other", self.version), (self.sector.key, "other")):
            self.assertIsNone(self.journal.active_rule(sector, version)["id"])

        # Use the actual planning and editing template paths, without a live model call.
        pipeline = SectorPipeline(self.settings)
        result = AnalysisResult(self.sector.key, self.sector.name)
        pipeline.last_result, pipeline.sector = result, self.sector
        with patch.object(pipeline, "_run", return_value={}):
            pipeline._plan_research(SectorRequest(self.sector.key, [CompanyInput("Delta")]), result)
        self.assertEqual(result.research_state["editing_rule_id"], rid)
        active = self.journal.active_rule(self.sector.key, self.version)["rule"]
        self.assertIn(active, pipeline._prompt("lead_editing"))
        self.assertNotIn(active, pipeline._prompt("lead_planning"))
        next_id = self.propose("另一条完整编辑准则", rid)
        self.vote_candidate(next_id, "Zulu", True)
        self.assertTrue(self.vote_candidate(next_id, "Echo", False))
        self.journal.rollback(next_id)
        self.assertEqual(self.journal.active_rule(self.sector.key, self.version)["id"], rid)
        self.journal.rollback(rid)
        self.assertIsNone(self.journal.active_rule(self.sector.key, self.version)["id"])
        self.assertIn(active, pipeline._prompt("lead_editing"))  # in-flight run is pinned

    def test_a_tie_or_unsupported_judgment_requires_revision(self):
        rid = self.propose()
        self.journal.save_trial(rid, "Beta", "冻结材料", True, "同一句", "同一句")
        with self.assertRaises(ValueError):
            self.journal.vote(rid, "beta", "A", True)
        self.assertFalse(self.journal.vote(rid, "beta", "平局", False))
        self.assertEqual(self.journal.rules(self.sector.key, self.version)[0]["status"], "needs_revision")
        with self.assertRaises(ValueError):
            self.journal.vote(rid, "beta", "A", True)

    def test_two_homepages_use_identical_frozen_material_and_failed_pair_can_retry(self):
        rid = self.propose()
        material = "冻结的产品文档 2025-01-01"
        def variants(*turns):
            model = Mock()
            model.session.return_value.send.side_effect = turns
            return model
        failed = variants(Turn(text="第一页"), Turn(truncated=True))
        with patch("vcagents.learning_replay._router") as router:
            router.return_value.for_role.return_value = failed
            with self.assertRaises(ValueError):
                generate_trial(self.journal, rid, self.sector, "Beta", material, True, self.settings)
        self.assertEqual(self.journal.trials(rid), [])
        model = variants(Turn(text="首页一"), Turn(tool_calls=[ToolCall("t", "submit_text", {"text": "首页二"})]))
        with patch("vcagents.learning_replay._router") as router:
            router.return_value.for_role.return_value = model
            generate_trial(self.journal, rid, self.sector, "Beta", material, True, self.settings)
        calls = model.session.call_args_list
        self.assertEqual(calls[0].args[0], calls[1].args[0])
        self.assertTrue(all(material in c.args[1] for c in calls))
        self.assertNotIn("经盲评采用", calls[0].args[1])
        self.assertIn("经盲评采用", calls[1].args[1])
        self.assertEqual(self.journal.trials(rid)[0]["material"], material)

    def test_audit_replay_hides_labels_and_counts_both_error_directions(self):
        self.journal.add_audit_pair("年份", "2024年发布", "发布于2024年", "发布于2025年")
        def session(system, prompt, tools, effort):
            data = json.loads(prompt)
            decisions = []
            for item in data["comparisons"]:
                self.assertEqual(set(item), {"issue_id", "submitted", "source_excerpt"})
                # Deliberately wrong on both members of the custom pair.
                right = any(p["equivalent"] == item["submitted"] for p in self.journal.audit_pairs())
                decisions.append({"issue_id": item["issue_id"], "unnecessary": right ^ ("发布" in item["submitted"]),
                                  "reason": "脚本化复核"})
            turn = Turn(tool_calls=[ToolCall("r", "review_rejection", {"decisions": decisions})])
            return Mock(send=lambda: turn, input_tokens=0, output_tokens=0, cost_usd=0)
        with patch("vcagents.learning_replay._router") as router:
            router.return_value.for_role.return_value.session.side_effect = session
            results = replay_audit(self.journal, self.settings)
            self.assertEqual(audit_summary(results), {"正确": 4, "误退": 1, "漏判": 1, "未评分": 0})
            router.return_value.for_role.return_value.session.side_effect = TimeoutError("超时")
            self.assertEqual(audit_summary(replay_audit(self.journal, self.settings))["未评分"], 6)
        history = self.journal.audit_replays()
        self.assertEqual(len(history), 2)
        self.assertEqual(history[1]["results"], results)
        with self.assertRaises(ValueError):
            replay_audit(self.journal, Settings(mock=True))

    def test_thesis_snapshot_and_observations_are_separate_from_current_evidence(self):
        started = date.today() - timedelta(days=40)
        result = AnalysisResult(self.sector.key, self.sector.name, status="completed", finished_at=started.isoformat(),
                                report_status="待人工复核的研究报告", models={"lead": "test"})
        result.decision = {"top_pick": "Beta", "headline": "原论点", "dissent": "另一种因果解释", "technical_moat": {
            "evidence": "当时观察", "replication": "替代解释", "would_fail_if": "同任务比较", "status": "形成中"}}
        self.assertEqual(self.memory.remember_result(result)["theses"], 1)
        card = self.journal.cards(self.sector.key)[0]
        cid = card["id"]
        self.assertEqual(card["original"]["alternative"], "另一种因果解释")
        self.assertEqual(card["review_on"], (started + timedelta(days=90)).isoformat())
        result.decision["headline"] = "事后改写"
        self.assertEqual(self.memory.remember_result(result)["theses"], 0)
        old = (started - timedelta(days=1)).isoformat()
        with self.assertRaises(ValueError):
            self.journal.observe(cid, old, "仍无数据", "无")
        with self.assertRaises(ValueError):
            self.journal.observe(cid, date.today().isoformat(), "被削弱", "新比较", "")
        for _ in range(2):
            self.journal.observe(cid, date.today().isoformat(), "被削弱", "新比较", "https://beta.example/study")
        self.journal.observe(cid, started.isoformat(), "仍无数据", "等待资料")
        self.assertEqual(len(self.journal.observations(cid)), 2)
        own = self.journal.thesis_leads(self.sector.key, ["BETA"])[0]
        self.assertEqual((own["原判断"], own["随访"]), ("原论点", "被削弱"))
        peer = self.journal.thesis_leads(self.sector.key, ["Other"])[0]
        self.assertNotIn("新比较", str(peer))
        self.assertNotIn("https://", str(peer))
        self.assertEqual(self.journal.thesis_leads("unrelated", ["Beta"]), [])
        self.assertEqual(self.memory.recall(self.sector.key, kinds=("fact",)), [])
        for kind in ("private", "draft", "demo"):
            result.run_id = kind
            result.memory_config = {"private_inputs": kind == "private"}
            result.quality_issues = ["x"] if kind == "draft" else []
            result.models["lead"] = "mock" if kind == "demo" else "test"
            self.assertEqual(self.memory.remember_result(result).get("theses", 0), 0)
        self.assertEqual(len(self.journal.cards(self.sector.key)), 1)

    def test_failed_archive_cannot_teach_the_next_run(self):
        result = AnalysisResult(self.sector.key, self.sector.name,
                                memory_config={"enabled": True, "path": self.settings.memory_path, "ttl_days": 30})
        folder = Path(self.settings.memory_path).parent / "exports"
        with (patch("vcagents.brief.render_brief", return_value="# 投资报告"),
              patch("vcagents.report_checks.publication_issues", return_value=[]),
              patch("vcagents.report_checks.publication_warnings", return_value=[]),
              patch("vcagents.pdf_report.build_report_pdf", return_value=(b"PDF", 1)),
              patch("pathlib.Path.write_bytes", side_effect=OSError("disk full")),
              patch.object(ResearchMemory, "remember_result") as remember):
            with self.assertRaises(OSError):
                save_result(result, folder)
            remember.assert_not_called()
