import unittest

from vcagents.pipeline import _namespace_sources


def material(rows):
    return {"sources": [f"[{i}] {url}" for i, (url, _) in enumerate(rows, 1)],
            "evidence": [{"source_ref": f"[{i}]", "claim": claim, "excerpt": claim}
                         for i, (_, claim) in enumerate(rows, 1)]}


class SourceRevisionTest(unittest.TestCase):
    def test_namespacing_updates_inline_evidence_references(self):
        data = material([("https://example.com/a", "收入1亿")])
        data["summary"] = "据公司披露收入1亿[E-1]，来源[1]。"
        after = _namespace_sources(data, "S")
        self.assertEqual(after["summary"], "据公司披露收入1亿[E-S-1]，来源[S-1]。")
        revised = material([("https://example.com/a", "据公司披露收入1亿")])
        revised["summary"] = "收入1亿[E-1]。"
        again = _namespace_sources(revised, "S", after)
        self.assertEqual(again["summary"], "收入1亿[E-S-2]。")

    def test_reordering_sources_keeps_ids_and_changed_facts_get_new_ids(self):
        before = _namespace_sources(material([("https://example.com/a", "收入1亿"), ("https://example.com/b", "用户100万")]), "S")
        after = _namespace_sources(material([("https://example.com/b", "用户100万"), ("https://example.com/c", "新事实")]), "S", before)
        self.assertTrue(after["sources"][0].startswith("[S-2]"))
        self.assertTrue(after["sources"][1].startswith("[S-3]"))
        self.assertEqual([e["evidence_id"] for e in after["evidence"]], ["E-S-2", "E-S-3"])
        self.assertEqual(after["evidence"][0]["source_ref"], "[S-2]")
        again = _namespace_sources(material([("https://example.com/d", "另一个事实")]), "S", after)
        self.assertEqual(again["evidence"][0]["evidence_id"], "E-S-4")
        self.assertTrue(again["sources"][0].startswith("[S-4]"))
