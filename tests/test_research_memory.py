import json
from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import unittest

from vcagents.memory import ResearchMemory
from vcagents.pipeline import AnalysisResult
from vcagents.tools import Toolbox
from vcagents.llm import ToolCall
from vcagents.roles import COMPANY_RESEARCHER


def result_with_evidence():
    r = AnalysisResult("aigc-content", "AIGC", report_status="待人工复核的研究报告")
    e = {"evidence_id": "E-C1-1", "claim": "Alpha 的历史收入为100万美元", "fact_type": "收入或ARR",
         "as_of": "2025-12-31", "quantities": [{"metric": "收入", "text": "$1 million"}],
         "excerpt": "Alpha revenue was $1 million.", "source_linked": True, "excerpt_verified": True,
         "source_type": "公司口径", "source_snapshots": [{"snapshot_id": "snap", "content_sha256": "hash",
             "url": "https://alpha.example/revenue", "fetched_at": "2026-11-14"}]}
    r.fact_sheets = {"Alpha": {"evidence": [e]}}
    r.fact_check = {"claims": [{"evidence_ids": ["E-C1-1"], "verdict": "公司口径", "support": "完全支持",
                               "excerpt_verified": True}], "summary": ""}
    return r


class ResearchMemoryTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "memory.sqlite3"
        self.memory = ResearchMemory(self.path)
        self.addCleanup(self.temp.cleanup)

    def test_memory_persists_but_never_returns_old_evidence_ids_as_current_proof(self):
        self.assertEqual(self.memory.remember_result(result_with_evidence())["facts"], 1)
        items = ResearchMemory(self.path).recall("aigc-content", "Alpha")
        self.assertEqual(len(items), 1)
        self.assertTrue(items[0]["requires_source_read"])
        self.assertEqual(items[0]["payload"]["verdict"], "公司口径")
        self.assertNotIn("E-C1-1", json.dumps(items))
        self.assertFalse(self.memory.recall("embodied-ai", "Alpha"))
        self.assertFalse(self.memory.recall("aigc-content", "Beta"))

    def test_quote_match_alone_cannot_be_promoted_into_fact_memory(self):
        r = result_with_evidence()
        r.fact_check = {}
        self.assertEqual(self.memory.remember_result(r)["facts"], 0)
        self.assertFalse(self.memory.recall("aigc-content", "Alpha"))

    def test_private_uploads_and_failed_reports_do_not_promote_facts(self):
        r = result_with_evidence()
        r.fact_sheets["Alpha"]["evidence"][0]["source_snapshots"] = [{"document": "BP.pdf", "page": 1}]
        self.assertEqual(self.memory.remember_result(r)["facts"], 0)
        r = result_with_evidence()
        r.report_checks = ["判断与回报冲突"]
        self.assertEqual(self.memory.remember_result(r)["facts"], 0)

    def test_conflicts_and_expired_memories_remain_explicit(self):
        self.memory.remember_result(result_with_evidence())
        r = result_with_evidence()
        r.fact_sheets["Alpha"]["evidence"][0]["claim"] = "Alpha 的同期间收入为200万美元"
        self.memory.remember_result(r)
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("UPDATE research_memory SET expires_at='2000-01-01T00:00:00+00:00'")
        items = self.memory.recall("aigc-content", "Alpha")
        self.assertEqual(len(items), 2)
        self.assertTrue(all(item["conflicted"] and item["stale"] for item in items))

    def test_rejected_claim_is_retired_and_returned_as_correction(self):
        r = result_with_evidence()
        self.memory.remember_result(r)
        claim = r.fact_sheets["Alpha"]["evidence"][0]["claim"]
        r.fact_check = {"claims": [{"claim": claim, "source_agent": "company_researcher:Alpha", "verdict": "不实", "evidence": "口径错误"}]}
        self.memory.remember_result(r)
        items = self.memory.recall("aigc-content", "Alpha")
        self.assertEqual([item["kind"] for item in items], ["correction"])

    def test_successful_repair_still_retires_the_original_false_memory(self):
        r = result_with_evidence()
        self.memory.remember_result(r)
        old = r.fact_sheets["Alpha"]["evidence"][0]["claim"]
        r.fact_check = {"claims": []}
        r.repairs = [{"source_agent": "company_researcher:Alpha", "claims": [
            {"claim": old, "source_agent": "company_researcher:Alpha", "verdict": "不实"}]}]
        self.memory.remember_result(r)
        items = self.memory.recall("aigc-content", "Alpha")
        self.assertEqual([item["kind"] for item in items], ["correction"])

    def test_bp_derived_hypotheses_are_not_saved_in_shared_memory(self):
        r = result_with_evidence()
        r.source_reads = {"company_researcher:Alpha": [{"kind": "document", "name": "BP.txt"}]}
        r.decision = {"top_pick": "Alpha", "headline": "私有财务判断", "action": "继续尽调"}
        counts = self.memory.remember_result(r)
        self.assertEqual(counts["hypotheses"], 0)
        self.assertNotIn("私有财务判断", json.dumps(self.memory.recall("aigc-content", "Alpha"), ensure_ascii=False))

    def test_native_memory_tool_cannot_cross_its_company_scope(self):
        self.memory.remember_result(result_with_evidence())
        tool = Toolbox(memory=self.memory, memory_sector="aigc-content", memory_entity="Beta")
        self.assertIn("search_memory", [s.name for s in tool.specs(COMPANY_RESEARCHER.tools)])
        out = tool.execute(ToolCall("memory", "search_memory", {"query": "Alpha"}))
        self.assertEqual(json.loads(out.content), [])
        self.assertNotIn("search_memory", [s.name for s in Toolbox().specs(COMPANY_RESEARCHER.tools)])


if __name__ == "__main__":
    unittest.main()
