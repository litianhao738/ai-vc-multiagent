import unittest
from unittest.mock import Mock, patch

from vcagents.llm import ToolCall
from vcagents.tools import DocumentStore, Toolbox, ToolError


def research(sources, ref, excerpt, source_type="独立来源"):
    return {"sources": sources, "evidence": [{"claim": "测试主张", "source_ref": ref,
            "excerpt": excerpt, "source_type": source_type, "as_of": "2026-10-30"}]}


def fact_check(source, excerpt, verdict="属实"):
    return {"claims": [{"claim": "测试主张", "source_agent": "research", "original_source": "[1]",
            "evidence_ids": [], "verdict": verdict, "evidence": "模型原始核查解释", "correction": "",
            "verification_source": source, "verification_excerpt": excerpt}], "summary": "核查摘要"}


class EvidenceGroundingTest(unittest.TestCase):
    def fetch(self, toolbox, url, body, final_url=None):
        output = f"标题：来源\nURL：{final_url or url}\n\n{body}"
        with patch("vcagents.tools.fetch_url", return_value=output):
            result = toolbox.execute(ToolCall("fetch", "fetch_url", {"url": url}))
        self.assertFalse(result.is_error)

    def test_literal_excerpt_matches_actual_read_and_keeps_input_unchanged(self):
        box = Toolbox()
        self.fetch(box, "https://news.example/a", "收入 100\n亿元，尚未盈利。")
        data = research(["[S1] 财务报道 - https://news.example/a"], "[S1]", "收入 100 亿元")
        result = box.verify_evidence(data)["evidence"][0]
        self.assertTrue(result["excerpt_verified"])
        self.assertEqual(result["excerpt_status"], "matched")
        self.assertNotIn("excerpt_verified", data["evidence"][0])

    def test_requested_and_final_urls_refer_to_the_same_read(self):
        box = Toolbox()
        self.fetch(box, "https://redirect.example/a", "实际正文", "https://news.example/final")
        data = research(["[1] https://news.example/final"], "https://redirect.example/a", "实际正文")
        self.assertTrue(box.verify_evidence(data)["evidence"][0]["excerpt_verified"])

    def test_quote_from_another_read_source_does_not_match(self):
        box = Toolbox()
        self.fetch(box, "https://news.example/a", "收入 100 亿元")
        self.fetch(box, "https://news.example/b", "收入 200 亿元")
        data = research(["[1] https://news.example/a", "[2] https://news.example/b"], "[2]", "收入 100 亿元")
        item = box.verify_evidence(data)["evidence"][0]
        self.assertFalse(item["excerpt_verified"])
        self.assertEqual(item["excerpt_status"], "excerpt_not_found")

    def test_changed_number_does_not_match(self):
        box = Toolbox()
        self.fetch(box, "https://news.example/a", "收入 100 亿元")
        item = box.verify_evidence(research(["[1] https://news.example/a"], "[1]", "收入 1000 亿元"))["evidence"][0]
        self.assertFalse(item["excerpt_verified"])

    def test_unread_source_and_search_summary_are_not_original_text(self):
        search = Mock()
        search.search.return_value = "收入 100 亿元。来源：https://news.example/a"
        box = Toolbox(search=search)
        box.execute(ToolCall("search", "web_search", {"query": "收入"}))
        item = box.verify_evidence(research(["[1] https://news.example/a"], "[1]", "收入 100 亿元"))["evidence"][0]
        self.assertFalse(item["excerpt_verified"])
        self.assertEqual(item["excerpt_status"], "source_not_read")

    def test_text_beyond_the_returned_body_is_not_verified(self):
        box = Toolbox()
        self.fetch(box, "https://news.example/a", "实际返回的前半段\n…（已截断）")
        item = box.verify_evidence(research(["[1] https://news.example/a"], "[1]", "未返回的后半段"))["evidence"][0]
        self.assertFalse(item["excerpt_verified"])
        metadata = box.verify_evidence(research(["[1] https://news.example/a"], "[1]", "URL：https://news.example/a"))["evidence"][0]
        self.assertFalse(metadata["excerpt_verified"])

    def test_web_reads_do_not_verify_cited_page_numbers(self):
        box = Toolbox()
        self.fetch(box, "https://news.example/a.pdf", "实际返回的文档正文")
        for page_ref in ("第20页", "page 20", "Pages 20-21", "p. 20"):
            with self.subTest(page_reference=page_ref):
                item = box.verify_evidence(research([f"[1] https://news.example/a.pdf {page_ref}"],
                                                   "[1]", "实际返回的文档正文"))["evidence"][0]
                self.assertFalse(item["excerpt_verified"])
                self.assertEqual(item["excerpt_status"], "page_not_verified")
        document_level = box.verify_evidence(research(["[1] https://news.example/a.pdf"],
                                                     "[1]", "实际返回的文档正文"))["evidence"][0]
        self.assertTrue(document_level["excerpt_verified"])

    def test_page_reference_in_evidence_ref_also_requires_page_level_read(self):
        box = Toolbox()
        self.fetch(box, "https://news.example/a.pdf", "文档正文")
        item = box.verify_evidence(research(["[1] https://news.example/a.pdf"],
                                           "[1] 第3页", "文档正文"))["evidence"][0]
        self.assertFalse(item["excerpt_verified"])
        self.assertEqual(item["excerpt_status"], "page_not_verified")

    def test_excerpts_are_not_assembled_from_separate_sources(self):
        box = Toolbox()
        self.fetch(box, "https://news.example/a", "前半句")
        self.fetch(box, "https://news.example/b", "后半句")
        item = box.verify_evidence(research(["[1] https://news.example/a", "[2] https://news.example/b"],
                                           "[1][2]", "前半句 后半句"))["evidence"][0]
        self.assertFalse(item["excerpt_verified"])

    def test_failed_fetch_is_not_recorded(self):
        box = Toolbox()
        with patch("vcagents.tools.fetch_url", side_effect=ToolError("抓取失败")):
            result = box.execute(ToolCall("fetch", "fetch_url", {"url": "https://news.example/a"}))
        self.assertTrue(result.is_error)
        item = box.verify_evidence(research(["[1] https://news.example/a"], "[1]", "正文"))["evidence"][0]
        self.assertEqual(item["excerpt_status"], "source_not_read")

    def test_label_and_explicit_url_cannot_point_to_different_sources(self):
        box = Toolbox()
        self.fetch(box, "https://news.example/a", "原文")
        item = box.verify_evidence(research(["[1] https://news.example/a", "[2] https://news.example/b"],
                                           "[1] https://news.example/b", "原文"))["evidence"][0]
        self.assertFalse(item["excerpt_verified"])
        self.assertEqual(item["excerpt_status"], "source_not_found")

    def test_duplicate_label_does_not_select_an_arbitrary_source(self):
        box = Toolbox()
        self.fetch(box, "https://news.example/a", "原文")
        item = box.verify_evidence(research(["[1] https://news.example/a", "[1] https://news.example/b"],
                                           "[1]", "原文"))["evidence"][0]
        self.assertFalse(item["excerpt_verified"])

    def test_missing_excerpt_has_explicit_status(self):
        item = Toolbox().verify_evidence(research(["[1] https://news.example/a"], "[1]", ""))["evidence"][0]
        self.assertFalse(item["excerpt_verified"])
        self.assertEqual(item["excerpt_status"], "missing_excerpt")

    def test_document_page_reference_only_matches_that_page(self):
        docs = DocumentStore()
        docs.docs["BP.txt"] = ["第一页收入 100", "第二页收入 200"]
        box = Toolbox(docs)
        box.execute(ToolCall("read", "read_document", {"name": "BP.txt", "start_page": 1, "end_page": 2}))
        data = research(["[1] 《BP.txt》第 1 页", "[2] 《BP.txt》第 2 页"], "[1]", "第二页收入 200")
        self.assertFalse(box.verify_evidence(data)["evidence"][0]["excerpt_verified"])
        data["evidence"][0]["source_ref"] = "《BP.txt》第 2 页"
        self.assertTrue(box.verify_evidence(data)["evidence"][0]["excerpt_verified"])

    def test_explicit_page_in_evidence_ref_cannot_override_source_page(self):
        docs = DocumentStore()
        docs.docs["BP.txt"] = ["第一页", "第二页"]
        box = Toolbox(docs)
        box.execute(ToolCall("read", "read_document", {"name": "BP.txt", "start_page": 1, "end_page": 2}))
        item = box.verify_evidence(research(["[1] 《BP.txt》第 1 页"], "[1] 第 2 页", "第二页"))["evidence"][0]
        self.assertFalse(item["excerpt_verified"])

    def test_document_excerpts_do_not_cross_page_boundaries(self):
        docs = DocumentStore()
        docs.docs["BP.txt"] = ["第一页末尾", "第二页开头"]
        box = Toolbox(docs)
        box.execute(ToolCall("read", "read_document", {"name": "BP.txt", "start_page": 1, "end_page": 2}))
        item = box.verify_evidence(research(["[1] 《BP.txt》第1-2页"], "[1]", "第一页末尾 第二页开头"))["evidence"][0]
        self.assertFalse(item["excerpt_verified"])

    def test_search_hits_from_different_files_remain_separate(self):
        docs = DocumentStore()
        docs.docs["A.txt"] = ["营收 100 来自甲"]
        docs.docs["B.txt"] = ["营收 200 来自乙"]
        box = Toolbox(docs)
        box.execute(ToolCall("search", "search_documents", {"query": "营收"}))
        sources = ["[1] 《A.txt》第1页", "[2] 《B.txt》第1页"]
        self.assertFalse(box.verify_evidence(research(sources, "[2]", "营收 100 来自甲"))["evidence"][0]["excerpt_verified"])
        self.assertTrue(box.verify_evidence(research(sources, "[1]", "营收 100 来自甲"))["evidence"][0]["excerpt_verified"])

    def test_search_only_verifies_the_returned_snippet(self):
        docs = DocumentStore()
        docs.docs["BP.txt"] = ["关键词" + "很长的正文" * 300 + "未返回内容"]
        box = Toolbox(docs)
        box.execute(ToolCall("search", "search_documents", {"query": "关键词"}))
        item = box.verify_evidence(research(["[1] 《BP.txt》第1页"], "[1]", "未返回内容"))["evidence"][0]
        self.assertFalse(item["excerpt_verified"])

    def test_document_name_substrings_are_not_the_same_file(self):
        docs = DocumentStore()
        docs.docs["BP.txt"] = ["甲的原文"]
        docs.docs["otherBP.txt"] = ["乙的原文"]
        box = Toolbox(docs)
        box.execute(ToolCall("read", "read_document", {"name": "BP.txt", "start_page": 1, "end_page": 1}))
        item = box.verify_evidence(research(["[1] otherBP.txt page 1"], "BP.txt page 1", "甲的原文"))["evidence"][0]
        self.assertFalse(item["excerpt_verified"])

    def test_company_website_source_is_not_independent(self):
        box = Toolbox(company_website="https://www.company.example")
        self.fetch(box, "https://company.example/news", "公司宣布融资")
        item = box.verify_evidence(research(["[1] https://company.example/news"], "[1]", "公司宣布融资"))["evidence"][0]
        self.assertTrue(item["excerpt_verified"])
        self.assertEqual(item["source_type"], "公司口径")
        self.assertTrue(item["source_type_note"])

    def test_company_domain_rule_does_not_guess_other_or_sec_domains(self):
        box = Toolbox(company_website="company.example")
        data = research(["[1] https://company.example.evil.test/news", "[2] https://www.sec.gov/filing"], "[2]", "披露")
        self.fetch(box, "https://www.sec.gov/filing", "披露")
        self.assertEqual(box.verify_evidence(data)["evidence"][0]["source_type"], "独立来源")
        data["evidence"][0]["source_ref"] = "[1]"
        self.assertEqual(box.verify_evidence(data)["evidence"][0]["source_type"], "独立来源")

    def test_fact_check_keeps_a_verdict_with_an_actual_same_source_quote(self):
        box = Toolbox()
        self.fetch(box, "https://news.example/a", "收入 100 亿元")
        data = fact_check("https://news.example/a", "收入 100 亿元")
        result = box.verify_fact_check(data)
        claim = result["claims"][0]
        self.assertEqual(claim["verdict"], "属实")
        self.assertEqual(claim["model_verdict"], "属实")
        self.assertTrue(claim["excerpt_verified"])
        self.assertEqual(claim["verification_status"], "matched")
        self.assertEqual(result["summary"], data["summary"])
        self.assertNotIn("model_verdict", data["claims"][0])

    def test_fact_check_downgrades_a_changed_quote_without_erasing_model_evidence(self):
        box = Toolbox()
        self.fetch(box, "https://news.example/a", "收入 100 亿元")
        claim = box.verify_fact_check(fact_check("https://news.example/a", "收入 1000 亿元"))["claims"][0]
        self.assertEqual(claim["verdict"], "无法核实")
        self.assertEqual(claim["model_verdict"], "属实")
        self.assertEqual(claim["evidence"], "模型原始核查解释")
        self.assertEqual(claim["verification_status"], "excerpt_not_found")

    def test_fact_check_downgrades_an_unread_source(self):
        claim = Toolbox().verify_fact_check(fact_check("https://news.example/a", "未读过的原文"))["claims"][0]
        self.assertEqual(claim["verdict"], "无法核实")
        self.assertEqual(claim["verification_status"], "source_not_read")

    def test_fact_check_with_a_malformed_source_is_unresolved(self):
        claim = Toolbox().verify_fact_check(fact_check("https://[invalid", "伪造来源中的文字"),
                                            ["https://[invalid"])["claims"][0]
        self.assertEqual(claim["verdict"], "无法核实")
        self.assertFalse(claim["excerpt_verified"])

    def test_fact_check_cannot_use_another_sources_quote(self):
        box = Toolbox()
        self.fetch(box, "https://news.example/a", "甲公司的收入")
        self.fetch(box, "https://news.example/b", "乙公司的收入")
        claim = box.verify_fact_check(fact_check("https://news.example/b", "甲公司的收入"))["claims"][0]
        self.assertEqual(claim["verdict"], "无法核实")
        self.assertFalse(claim["excerpt_verified"])

    def test_fact_check_downgrades_web_quote_with_an_unverified_page(self):
        box = Toolbox()
        self.fetch(box, "https://news.example/a.pdf", "第一页的数据")
        claim = box.verify_fact_check(fact_check("https://news.example/a.pdf page 20", "第一页的数据"))["claims"][0]
        self.assertEqual(claim["verdict"], "无法核实")
        self.assertEqual(claim["model_verdict"], "属实")
        self.assertEqual(claim["verification_status"], "page_not_verified")

    def test_unproven_negative_claim_is_not_left_as_false(self):
        claim = Toolbox().verify_fact_check(fact_check("https://news.example/a", "", "不实"))["claims"][0]
        self.assertEqual(claim["verdict"], "无法核实")
        self.assertEqual(claim["model_verdict"], "不实")
        self.assertEqual(claim["verification_status"], "missing_excerpt")

    def test_verified_negative_claim_is_preserved(self):
        box = Toolbox()
        self.fetch(box, "https://company.example/a", "公司原文明确反驳该说法")
        claim = box.verify_fact_check(fact_check("https://company.example/a", "明确反驳该说法", "不实"),
                                      ["https://company.example"])["claims"][0]
        self.assertEqual(claim["verdict"], "不实")
        self.assertTrue(claim["excerpt_verified"])

    def test_fact_check_caps_same_company_domain_to_company_reported(self):
        box = Toolbox()
        self.fetch(box, "https://company.example/a", "公司披露数据")
        claim = box.verify_fact_check(fact_check("https://company.example/a", "公司披露数据"),
                                      ["", "https://www.company.example"])["claims"][0]
        self.assertEqual(claim["model_verdict"], "属实")
        self.assertEqual(claim["verdict"], "公司口径")
        self.assertTrue(claim["excerpt_verified"])

    def test_fact_check_does_not_infer_sec_is_or_is_not_company_owned(self):
        box = Toolbox()
        self.fetch(box, "https://www.sec.gov/a", "核查原文")
        claim = box.verify_fact_check(fact_check("https://www.sec.gov/a", "核查原文"),
                                      ["https://company.example"])["claims"][0]
        self.assertEqual(claim["verdict"], "属实")
        self.assertEqual(claim["model_verdict"], "属实")

    def test_fact_check_document_quote_must_match_the_cited_page(self):
        docs = DocumentStore()
        docs.docs["BP.txt"] = ["第一页", "第二页"]
        box = Toolbox(docs)
        box.execute(ToolCall("read", "read_document", {"name": "BP.txt", "start_page": 1, "end_page": 2}))
        first = box.verify_fact_check(fact_check("《BP.txt》第1页", "第二页", "公司口径"))["claims"][0]
        self.assertEqual(first["verdict"], "无法核实")
        second = box.verify_fact_check(fact_check("《BP.txt》第2页", "第二页", "公司口径"))["claims"][0]
        self.assertEqual(second["verdict"], "公司口径")

    def test_fact_check_never_upgrades_an_unresolved_claim(self):
        box = Toolbox()
        self.fetch(box, "https://news.example/a", "原文存在")
        claim = box.verify_fact_check(fact_check("https://news.example/a", "原文存在", "无法核实"))["claims"][0]
        self.assertEqual(claim["verdict"], "无法核实")
        self.assertTrue(claim["excerpt_verified"])

    # 排版差异与本轮共享读取
    def test_spaces_inserted_around_chinese_by_page_markup_still_match(self):
        box = Toolbox()
        self.fetch(box, "https://news.example/a", "8月底从老东家离职后， 公司成立仅半个月 ，就已经拿下了投资。")
        claim = box.verify_fact_check(fact_check("https://news.example/a", "离职后，公司成立仅半个月，就已经拿下了投资"))["claims"][0]
        self.assertEqual(claim["verdict"], "属实")

    def test_curly_and_straight_quotes_are_the_same_text(self):
        box = Toolbox()
        self.fetch(box, "https://news.example/a", "“你们公司的护城河会是什么？”\n\n廖谦沉默了片刻")
        claim = box.verify_fact_check(fact_check("https://news.example/a", '"你们公司的护城河会是什么？" 廖谦沉默了片刻'))["claims"][0]
        self.assertEqual(claim["verdict"], "属实")

    def test_typography_rules_do_not_accept_changed_wording(self):
        box = Toolbox()
        self.fetch(box, "https://news.example/a", "母公司完成超千万美元天使轮，达泰资本领投")
        claim = box.verify_fact_check(fact_check("https://news.example/a", "母公司已完成超千万美元天使轮融资"))["claims"][0]
        self.assertEqual(claim["verdict"], "无法核实")

    def test_english_word_spacing_is_still_significant(self):
        box = Toolbox()
        self.fetch(box, "https://news.example/a", "revenue grew to 100 million")
        claim = box.verify_fact_check(fact_check("https://news.example/a", "revenuegrew to 100 million"))["claims"][0]
        self.assertEqual(claim["verdict"], "无法核实")

    def test_same_source_read_by_another_role_in_this_run_verifies_quote(self):
        researcher = Toolbox()
        self.fetch(researcher, "https://news.example/a", "收入 100 亿元，尚未盈利。")
        checker = Toolbox(shared_reads=researcher.read_records())
        claim = checker.verify_fact_check(fact_check("https://news.example/a", "收入 100 亿元"))["claims"][0]
        self.assertEqual(claim["verdict"], "属实")
        self.assertEqual(claim["verification_status"], "matched")

    def test_shared_reads_do_not_lend_text_to_a_different_source(self):
        researcher = Toolbox()
        self.fetch(researcher, "https://news.example/a", "收入 100 亿元")
        checker = Toolbox(shared_reads=researcher.read_records())
        claim = checker.verify_fact_check(fact_check("https://news.example/b", "收入 100 亿元"))["claims"][0]
        self.assertEqual(claim["verdict"], "无法核实")
        self.assertEqual(claim["verification_status"], "source_not_read")

    def test_shared_reads_still_require_the_literal_excerpt(self):
        researcher = Toolbox()
        self.fetch(researcher, "https://news.example/a", "收入 100 亿元")
        checker = Toolbox(shared_reads=researcher.read_records())
        claim = checker.verify_fact_check(fact_check("https://news.example/a", "收入 1000 亿元"))["claims"][0]
        self.assertEqual(claim["verdict"], "无法核实")


if __name__ == "__main__":
    unittest.main()
