from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx

from vcagents.knowledge import KnowledgeStore
from vcagents.llm import ToolCall
from vcagents.tools import MAX_PAGE_CHARS, Toolbox


def research(source, excerpt):
    return {"sources": [f"[1] {source}"], "evidence": [{"claim": "测试事实", "source_ref": "[1]",
            "excerpt": excerpt, "source_type": "独立来源", "as_of": "2026-10-30"}]}


def fact_check(source, claim, excerpt, verdict="属实"):
    return {"claims": [{"claim": claim, "source_agent": "research", "original_source": "[1]",
            "evidence_ids": [], "verdict": verdict, "evidence": "模型核查解释", "correction": "",
            "verification_source": source, "verification_excerpt": excerpt}], "summary": "固定资料核查"}


class SourceToolsTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name, "knowledge.sqlite3")
        self.store = KnowledgeStore(self.path)

    def read_archived_text(self, text, url="https://source.example/report"):
        metadata = self.store.archive(url, "公开报告", text)
        box = Toolbox(knowledge=self.store)
        result = box.execute(ToolCall("read", "read_source", {"snapshot_id": metadata["snapshot_id"], "chunk_id": 0}))
        self.assertFalse(result.is_error)
        return box, metadata

    def test_fetch_archives_full_text_but_initial_read_does_not_verify_the_hidden_tail(self):
        requested = "https://redirect.example/report"
        final_url = "https://source.example/report"
        tail = "独特尾段：The company disclosed USD 1 million."
        html = "<html><head><title>融资报告</title></head><body><p>开头正文 " + "a" * (MAX_PAGE_CHARS + 500) + "</p><p>" + tail + "</p></body></html>"
        response = httpx.Response(200, text=html, headers={"content-type": "text/html"},
                                  request=httpx.Request("GET", final_url))
        box = Toolbox(knowledge=self.store)
        with patch("vcagents.tools._check_public_url") as check, patch("vcagents.tools.httpx.get", return_value=response) as get:
            result = box.execute(ToolCall("fetch", "fetch_url", {"url": requested}))
        self.assertFalse(result.is_error)
        check.assert_called_once_with(requested)
        get.assert_called_once()
        self.assertIn("快照：", result.content)
        self.assertIn("SHA256：", result.content)
        self.assertNotIn(tail, result.content)
        initial = box.verify_evidence(research(final_url, tail))["evidence"][0]
        self.assertFalse(initial["excerpt_verified"])
        self.assertEqual(initial["source_snapshots"], [])

        found = box.execute(ToolCall("search", "search_knowledge", {"query": "独特尾段"}))
        self.assertFalse(found.is_error)
        hits = json.loads(found.content)
        self.assertTrue(any(tail in hit["excerpt"] for hit in hits))
        grounded = box.verify_evidence(research(final_url, tail))["evidence"][0]
        self.assertTrue(grounded["excerpt_verified"])
        self.assertTrue(grounded["source_snapshots"])
        for record in grounded["source_snapshots"]:
            original = self.store.read(record["snapshot_id"])
            self.assertEqual(record["url"], final_url)
            self.assertEqual(record["content_sha256"], hashlib.sha256(original["text"].encode("utf-8")).hexdigest())
            self.assertIn(tail, original["text"][record["start_offset"]:record["end_offset"]])
            self.assertGreaterEqual(record["end_offset"], MAX_PAGE_CHARS)
        wrong = box.verify_evidence(research("https://fabricated.example/report", tail))["evidence"][0]
        self.assertFalse(wrong["excerpt_verified"])
        self.assertEqual(wrong["source_snapshots"], [])

    def test_new_toolbox_can_search_and_read_an_old_snapshot_without_network(self):
        url = "https://source.example/history"
        text = "较早的公开材料。\n\n唯一历史段落：收入 USD 3 million。"
        old_time = datetime(2020, 2, 3, 4, 5, 6, tzinfo=timezone.utc)
        with patch("vcagents.knowledge.datetime") as clock:
            clock.now.return_value = old_time
            snapshot = self.store.archive(url, "历史资料", text)
        self.assertEqual(snapshot["fetched_at"], old_time.isoformat(timespec="seconds"))

        with patch("vcagents.tools.httpx.get", side_effect=AssertionError("断网不应访问 HTTP")) as get, \
                patch("vcagents.tools._check_public_url", side_effect=AssertionError("断网不应解析网址")) as check:
            reopened = KnowledgeStore(self.path)
            box = Toolbox(knowledge=reopened)
            search = box.execute(ToolCall("search", "search_knowledge", {"query": "唯一历史段落"}))
            self.assertFalse(search.is_error)
            hit = json.loads(search.content)[0]
            self.assertEqual(hit["fetched_at"], snapshot["fetched_at"])
            self.assertEqual(hit["content_sha256"], snapshot["content_sha256"])

            unread_box = Toolbox(knowledge=reopened)
            quote = "唯一历史段落：收入 USD 3 million。"
            self.assertFalse(unread_box.verify_evidence(research(url, quote))["evidence"][0]["excerpt_verified"])
            read = unread_box.execute(ToolCall("read", "read_source", {"snapshot_id": hit["snapshot_id"], "chunk_id": hit["chunk_id"]}))
            self.assertFalse(read.is_error)
            record = json.loads(read.content)
            self.assertEqual(record["fetched_at"], snapshot["fetched_at"])
            self.assertEqual(record["text"], hit["excerpt"])
            grounded = unread_box.verify_evidence(research(url, quote))["evidence"][0]
            self.assertTrue(grounded["excerpt_verified"])
            self.assertEqual(grounded["source_snapshots"][0]["snapshot_id"], snapshot["snapshot_id"])
            self.assertEqual(grounded["source_snapshots"][0]["fetched_at"], snapshot["fetched_at"])
            get.assert_not_called()
            check.assert_not_called()

    def test_matching_quote_does_not_verify_a_different_claim_number(self):
        quote = "The company raised USD 1 million."
        box, snapshot = self.read_archived_text(quote)
        claim = box.verify_fact_check(fact_check(snapshot["url"], "The company raised USD 10 million.", quote))["claims"][0]
        self.assertTrue(claim["excerpt_verified"])
        self.assertEqual(claim["verification_status"], "matched")
        self.assertEqual(claim["verdict"], "无法核实")
        self.assertEqual(claim["model_verdict"], "属实")
        self.assertEqual(claim["unmatched_numbers"], ["10 million"])
        self.assertEqual(claim["source_snapshots"][0]["content_sha256"], snapshot["content_sha256"])

    def test_matching_numbers_with_thousands_separators_and_decimal_format_pass(self):
        quote = "The company raised USD 1,000 million."
        box, snapshot = self.read_archived_text(quote)
        for amount in ("1000", "1,000", "1000.00", "１，０００．０"):
            with self.subTest(amount=amount):
                item = box.verify_fact_check(fact_check(snapshot["url"], f"The company raised USD {amount} million.", quote))["claims"][0]
                self.assertTrue(item["excerpt_verified"])
                self.assertEqual(item["verdict"], "属实")
                self.assertEqual(item["unmatched_numbers"], [])

    def test_numeric_matcher_compares_values_across_unit_scales(self):
        quote = "The company raised USD 1,000,000."
        box, snapshot = self.read_archived_text(quote)
        item = box.verify_fact_check(fact_check(snapshot["url"], "The company raised USD 1 million.", quote))["claims"][0]
        self.assertTrue(item["excerpt_verified"])
        self.assertEqual(item["verdict"], "属实")
        wrong = box.verify_fact_check(fact_check(snapshot["url"], "The company raised USD 10 million.", quote))["claims"][0]
        self.assertEqual(wrong["unmatched_numbers"], ["10 million"])

    def test_false_verdict_with_a_real_contradicting_quote_is_not_changed_by_number_mismatch(self):
        quote = "The company raised USD 1 million."
        box, snapshot = self.read_archived_text(quote)
        item = box.verify_fact_check(fact_check(snapshot["url"], "The company raised USD 10 million.", quote, "不实"))["claims"][0]
        self.assertTrue(item["excerpt_verified"])
        self.assertEqual(item["model_verdict"], "不实")
        self.assertEqual(item["verdict"], "不实")
        self.assertEqual(item["unmatched_numbers"], [])

    def test_unsupported_correction_is_kept_only_as_model_history(self):
        quote = "The company raised USD 1 million."
        box, snapshot = self.read_archived_text(quote)
        data = fact_check(snapshot["url"], "The company raised USD 10 million.", quote, "不实")
        data["claims"][0]["correction"] = "The company raised USD 999 million."
        item = box.verify_fact_check(data)["claims"][0]
        self.assertEqual(item["verdict"], "不实")
        self.assertEqual(item["correction"], "")
        self.assertIn("999", item["model_correction"])

    def test_read_source_rejects_invalid_arguments_as_tool_results(self):
        snapshot = self.store.archive("https://source.example/a", "原文", "原始资料")
        box = Toolbox(knowledge=self.store)
        invalid = [
            {"snapshot_id": snapshot["snapshot_id"]},
            {"snapshot_id": snapshot["snapshot_id"], "chunk_id": -1},
            {"snapshot_id": snapshot["snapshot_id"], "chunk_id": True},
            {"snapshot_id": snapshot["snapshot_id"], "chunk_id": "0"},
            {"snapshot_id": snapshot["snapshot_id"], "chunk_id": 0.5},
            {"snapshot_id": "", "chunk_id": 0},
            {"snapshot_id": "does-not-exist", "chunk_id": 0},
            {"snapshot_id": snapshot["snapshot_id"], "chunk_id": 999},
        ]
        for args in invalid:
            with self.subTest(args=args):
                result = box.execute(ToolCall("read", "read_source", args))
                self.assertTrue(result.is_error)
                self.assertTrue(result.content)
        disabled = Toolbox().execute(ToolCall("read", "read_source", {"snapshot_id": snapshot["snapshot_id"], "chunk_id": 0}))
        self.assertTrue(disabled.is_error)
        self.assertIn("未启用", disabled.content)


if __name__ == "__main__":
    unittest.main()
