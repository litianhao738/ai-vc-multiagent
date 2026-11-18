from datetime import datetime, timezone
import hashlib
from pathlib import Path
import tempfile
import unittest

from vcagents.knowledge import KnowledgeStore, MAX_CHUNK_CHARS, MAX_SEARCH_RESULTS


class KnowledgeStoreTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name, "knowledge", "sources.sqlite3")
        self.store = KnowledgeStore(self.path)

    def test_archive_preserves_raw_text_hash_and_utc_timestamp(self):
        text = "  第一段原文。\r\n\r\n第二段含 3.5 亿美元。\n"
        metadata = self.store.archive("https://example.com/a", "来源标题", text)
        self.assertEqual(metadata["content_sha256"], hashlib.sha256(text.encode("utf-8")).hexdigest())
        self.assertEqual(datetime.fromisoformat(metadata["fetched_at"]).utcoffset(), timezone.utc.utcoffset(None))
        record = self.store.read(metadata["snapshot_id"])
        self.assertEqual(record["text"], text)
        self.assertEqual(record["start_offset"], 0)
        self.assertEqual(record["end_offset"], len(text))
        self.assertIsNone(record["chunk_id"])
        self.assertEqual(record["title"], "来源标题")

    def test_same_url_and_content_is_idempotent_across_reopening(self):
        first = self.store.archive("https://example.com/a", "首次标题", "可检索的融资原文")
        reopened = KnowledgeStore(self.path)
        repeated = reopened.archive("https://example.com/a", "后来的标题", "可检索的融资原文")
        self.assertEqual(first, repeated)
        hits = reopened.search("融资")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["snapshot_id"], first["snapshot_id"])

    def test_changed_content_keeps_both_versions(self):
        old = self.store.archive("https://example.com/a", "财务报告", "收入为 100，旧版本。")
        new = self.store.archive("https://example.com/a", "财务报告", "收入为 200，新版本。")
        self.assertNotEqual(old["snapshot_id"], new["snapshot_id"])
        self.assertEqual(self.store.read(old["snapshot_id"])["text"], "收入为 100，旧版本。")
        self.assertEqual(self.store.read(new["snapshot_id"])["text"], "收入为 200，新版本。")
        self.assertEqual({hit["snapshot_id"] for hit in self.store.search("收入")},
                         {old["snapshot_id"], new["snapshot_id"]})

    def test_snapshot_identity_includes_the_source_url(self):
        first = self.store.archive("https://example.com/a", "A", "相同原文")
        second = self.store.archive("https://example.com/b", "B", "相同原文")
        self.assertNotEqual(first["snapshot_id"], second["snapshot_id"])
        self.assertEqual(first["content_sha256"], second["content_sha256"])

    def test_snapshot_id_is_stable_in_another_database(self):
        first = self.store.archive("https://example.com/a", "标题", "原文")
        other = KnowledgeStore(Path(self.directory.name, "another.sqlite3"))
        second = other.archive("https://example.com/a", "标题", "原文")
        self.assertEqual(first["snapshot_id"], second["snapshot_id"])

    def test_chunks_are_exact_character_slices_with_paragraph_boundaries(self):
        text = "  第一段收入。 \r\n\r\n第二段\n仍在第二段。\n\n" + "长" * (MAX_CHUNK_CHARS + 17)
        metadata = self.store.archive("https://example.com/a", "标题", text)
        chunks = [self.store.read(metadata["snapshot_id"], i) for i in range(4)]
        self.assertEqual(chunks[0]["text"], "第一段收入。")
        self.assertEqual(chunks[1]["text"], "第二段\n仍在第二段。")
        self.assertEqual(len(chunks[2]["text"]), MAX_CHUNK_CHARS)
        self.assertEqual(len(chunks[3]["text"]), 17)
        for chunk in chunks:
            self.assertEqual(chunk["text"], text[chunk["start_offset"]:chunk["end_offset"]])
        self.assertLess(chunks[0]["end_offset"], chunks[1]["start_offset"])
        self.assertEqual(chunks[2]["end_offset"], chunks[3]["start_offset"])
        self.assertIsNone(self.store.read(metadata["snapshot_id"], 4))

    def test_chinese_bigrams_find_words_inside_unsegmented_text(self):
        expected = self.store.archive("https://example.com/cash", "经营报告", "公司现金跑道为十八个月。")
        self.store.archive("https://example.com/robots", "产品报告", "机器人完成工厂搬运。")
        hits = self.store.search("现金跑道")
        self.assertEqual(hits[0]["snapshot_id"], expected["snapshot_id"])
        self.assertEqual(hits[0]["excerpt"], "公司现金跑道为十八个月。")
        self.assertEqual(hits[0]["content_sha256"], expected["content_sha256"])
        self.assertEqual(hits[0]["fetched_at"], expected["fetched_at"])

    def test_single_chinese_character_and_title_are_searchable(self):
        metadata = self.store.archive("https://example.com/cloud", "云计算公司报告", "经营数据在此。")
        self.assertEqual(self.store.search("云")[0]["snapshot_id"], metadata["snapshot_id"])
        self.assertEqual(self.store.search("计算")[0]["snapshot_id"], metadata["snapshot_id"])

    def test_english_and_chinese_queries_use_the_same_index(self):
        expected = self.store.archive("https://example.com/arr", "财务", "ARR revenue 增长与毛利率变化。")
        self.assertEqual(self.store.search("arr 毛利")[0]["snapshot_id"], expected["snapshot_id"])
        self.assertEqual(self.store.search("REVENUE")[0]["snapshot_id"], expected["snapshot_id"])

    def test_search_is_ranked_by_bm25_and_returns_raw_chunk_offsets(self):
        expected = self.store.archive("https://example.com/a", "经营", "现金跑道现金跑道现金跑道。")
        self.store.archive("https://example.com/b", "经营", "现金流与融资情况。")
        hits = self.store.search("现金跑道")
        self.assertEqual(hits[0]["snapshot_id"], expected["snapshot_id"])
        self.assertEqual([hit["bm25_score"] for hit in hits], sorted(hit["bm25_score"] for hit in hits))
        for hit in hits:
            read = self.store.read(hit["snapshot_id"], hit["chunk_id"])
            self.assertEqual(read["text"], hit["excerpt"])
            self.assertEqual(read["start_offset"], hit["start_offset"])
            self.assertEqual(read["end_offset"], hit["end_offset"])

    def test_limit_is_respected_and_bounded(self):
        for i in range(MAX_SEARCH_RESULTS + 3):
            self.store.archive(f"https://example.com/{i}", f"来源{i}", "机器人公开资料。")
        self.assertEqual(len(self.store.search("机器人", limit=1)), 1)
        self.assertEqual(len(self.store.search("机器人", limit=10000)), MAX_SEARCH_RESULTS)
        self.assertEqual(self.store.search("机器人", limit=0), [])

    def test_query_punctuation_is_not_executed_as_fts_syntax(self):
        self.store.archive("https://example.com/a", "资料", "机器人收入。")
        self.assertTrue(self.store.search('机器人 OR "收入" * ('))
        self.assertEqual(self.store.search('"*() :'), [])
        self.assertEqual(self.store.search(""), [])

    def test_only_explicit_archives_are_searchable_and_missing_reads_are_none(self):
        self.assertEqual(self.store.search("融资"), [])
        self.assertIsNone(self.store.read("not-a-snapshot"))
        item = self.store.archive("https://example.com/a", "来源", "没有融资消息。")
        self.assertIsNone(self.store.read(item["snapshot_id"], 99))

    def test_empty_text_can_be_retained_without_fake_chunks(self):
        item = self.store.archive("https://example.com/a", "", "")
        self.assertEqual(item["title"], "https://example.com/a")
        self.assertEqual(self.store.read(item["snapshot_id"])["text"], "")
        self.assertIsNone(self.store.read(item["snapshot_id"], 0))
        self.assertEqual(self.store.search("example"), [])

    def test_archive_requires_a_traceable_url(self):
        with self.assertRaisesRegex(ValueError, "URL"):
            self.store.archive(" ", "标题", "原文")


if __name__ == "__main__":
    unittest.main()
