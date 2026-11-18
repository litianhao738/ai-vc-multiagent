import unittest
from unittest.mock import Mock, patch

from vcagents.agent import Agent, AgentError
from vcagents.citations import SourceRegistry, segment_spans
from vcagents.llm import ToolCall, Turn
from vcagents.roles import COMPANY_RESEARCHER
from vcagents.schemas import FACT_SHEET
from vcagents.tools import DocumentStore, Toolbox

PAGE = ("视频 Agent Pexo 的母公司「极致上下文」完成超千万美元天使轮，达泰资本、松禾创投领投。"
        "自 7 月正式上线以来，Pexo 已覆盖全球 80 多个国家和地区。\n"
        "Aug. 17, 2026 /PRNewswire/ -- Higgsfield raised $400 million. The round was led by DST Global.\n"
        "The global market was valued at USD 14.8 billion in 2024.\n第六句。第七句。")


def fact(statement, source="W1", first=1, last=1, quantities=(), entities=(), date=("", "未给出"), date_quote=None):
    return {"subject": "主体", "fact_type": "融资", "statement": statement,
            "quantities": [{"metric": "金额", "text": q} for q in quantities], "entities": list(entities),
            "date": {"text": date[0], "basis": date[1]}, "quote": {"source": source, "from": first, "to": last},
            "date_quote": date_quote or {"source": "", "from": 0, "to": 0}}


def evidence(statement, ref="[1]", **kwargs):
    return {**fact(statement, **kwargs), "source_ref": ref, "source_type": "独立来源"}


def check(target, verdict="属实", support="完全支持", **kwargs):
    return {"target": target, "source_agent": "chair", "original_source": "", "evidence_ids": [],
            "fact": fact(target, **kwargs), "support": support, "verdict": verdict, "evidence": "理由", "correction": ""}


class CitationTest(unittest.TestCase):
    def test_unrelated_input_value_cannot_validate_an_external_fact(self):
        box = Toolbox(known_values={20})
        self.fetch(box, body="Alpha charges $10 per month.")
        result = box.verify_fact_check({"claims": [check("Alpha每月收费20美元[E-C1-9]")], "summary": ""})["claims"][0]
        self.assertEqual(result["verdict"], "无法核实")
        self.assertEqual(result["unmatched_numbers"], ["20"])

    def fetch(self, box, url="https://news.example/a", body=PAGE):
        with patch("vcagents.tools.fetch_url", return_value=f"标题：报道\nURL：{url}\n\n{body}"):
            result = box.execute(ToolCall("1", "fetch_url", {"url": url}))
        self.assertFalse(result.is_error)
        return result.content

    def research(self, box, *items):
        return box.verify_evidence({"sources": ["[1] 报道 - https://news.example/a"], "evidence": list(items)})["evidence"]

    def test_segments_split_chinese_and_english_sentences_and_cap_length(self):
        spans = segment_spans(PAGE)
        sentences = [PAGE[a:b] for a, b in spans]
        self.assertEqual(sentences[0], "视频 Agent Pexo 的母公司「极致上下文」完成超千万美元天使轮，达泰资本、松禾创投领投。")
        self.assertIn("Higgsfield raised $400 million.", sentences[2])
        self.assertEqual(sentences[3], "The round was led by DST Global.")
        long = "很长，" * 200
        self.assertTrue(all(b - a <= 200 for a, b in segment_spans(long)))
        self.assertEqual(segment_spans(PAGE), segment_spans(PAGE))

    def test_tool_output_is_numbered_and_registry_is_shared_across_toolboxes(self):
        registry = SourceRegistry()
        content = self.fetch(Toolbox(registry=registry))
        self.assertIn("来源编号：W1", content)
        self.assertIn("[W1:s1] 视频 Agent Pexo", content)
        self.assertIn("来源编号：W1", self.fetch(Toolbox(registry=registry)))
        self.assertIn("来源编号：W2", self.fetch(Toolbox(registry=registry), "https://news.example/b", "另一篇。"))

    def test_pointer_evidence_takes_excerpt_from_source_and_composes_claim(self):
        box = Toolbox()
        self.fetch(box)
        item = self.research(box, evidence("完成 1000 万美元天使轮", quantities=["超千万美元"], entities=["达泰资本"]))[0]
        self.assertTrue(item["excerpt_verified"])
        self.assertEqual(item["excerpt_status"], "matched")
        self.assertTrue(item["excerpt"].startswith("视频 Agent Pexo 的母公司"))
        self.assertEqual(item["claim"], "主体完成 1000 万美元天使轮（达泰资本）")
        self.assertEqual(item["quote_location"], "W1:s1")

    def test_pointer_evidence_rejects_unsupported_fields(self):
        box = Toolbox()
        self.fetch(box)
        cases = {
            "fields_unsupported": evidence("完成融资", quantities=["一亿美元"]),
            "source_mismatch": evidence("完成融资", ref="[2]"),
            "quote_too_long": evidence("完成融资", first=1, last=6),
            "source_not_found": evidence("完成融资", source="W9"),
        }
        for status, item in cases.items():
            with self.subTest(status=status):
                self.assertEqual(self.research(box, item)[0]["excerpt_status"], status)
        added_year = self.research(box, evidence("2031 年 7 月正式上线", first=2, last=2))[0]
        self.assertFalse(added_year["excerpt_verified"])
        self.assertTrue(any("2031" in issue for issue in added_year["field_issues"]))

    def test_unread_sentences_cannot_be_cited(self):
        registry = SourceRegistry()
        registry.register_web("https://news.example/a", "报道", PAGE)
        item = self.research(Toolbox(registry=registry), evidence("完成融资"))[0]
        self.assertEqual(item["excerpt_status"], "source_not_read")

    def test_inferred_year_needs_date_quote_and_is_labelled(self):
        box = Toolbox()
        self.fetch(box)
        item = self.research(box, evidence("Pexo 正式上线", first=2, last=2, date=("2026 年 7 月", "来源其他位置"),
                                           date_quote={"source": "W1", "from": 3, "to": 3}))[0]
        self.assertTrue(item["excerpt_verified"], item["field_issues"])
        self.assertEqual(item["as_of"], "2026 年 7 月（推断）")
        self.assertIn("日期据来源其他位置推断", item["claim"])

    def test_fact_check_pointer_path(self):
        box = Toolbox()
        self.fetch(box)
        ok = box.verify_fact_check({"claims": [check("全球市场 2024 年为 148 亿美元", first=5, last=5)], "summary": ""})["claims"][0]
        self.assertEqual(ok["verdict"], "属实")
        self.assertEqual(ok["verification_source"], "https://news.example/a")
        self.assertIn("USD 14.8 billion", ok["verification_excerpt"])
        partial = box.verify_fact_check({"claims": [check("完成天使轮", support="部分支持")], "summary": ""})["claims"][0]
        self.assertEqual((partial["verdict"], partial["verification_status"]), ("无法核实", "partial_support"))
        extra = box.verify_fact_check({"claims": [check("2025 年完成超千万美元天使轮")], "summary": ""})["claims"][0]
        self.assertEqual((extra["verdict"], extra["unmatched_numbers"]), ("无法核实", ["2025"]))

    def test_correct_quote_does_not_validate_a_smaller_unscaled_amount(self):
        box = Toolbox()
        self.fetch(box, body="Alpha revenue was $1 million.")
        result = box.verify_fact_check({"claims": [check("Alpha收入为1美元", quantities=["$1 million"])], "summary": ""})["claims"][0]
        self.assertEqual(result["verdict"], "无法核实")
        self.assertEqual(result["unmatched_numbers"], ["1"])

    def test_submission_check_keeps_rejecting_unfixed_citations(self):
        box = Toolbox(materials={"chair": {"headline": "Pexo 完成超千万美元天使轮。"}})
        self.fetch(box)
        bad = {"claims": [check("不在材料里的句子", first=1, last=1), check("完成超千万美元天使轮", source="W1", first=9, last=9)],
               "summary": ""}
        errors = box.citation_errors("fact_checker", bad)
        self.assertTrue(any("target 必须从材料 chair 中摘录原句" in e for e in errors))
        self.assertTrue(any("句号范围无效" in e for e in errors))
        clean = Toolbox(materials={"chair": {"headline": "Pexo 完成超千万美元天使轮。"}})
        self.fetch(clean)
        self.assertEqual(clean.citation_errors("fact_checker", {"claims": [check("完成超千万美元天使轮")], "summary": ""}), [])
        self.assertEqual(box.citation_errors("fact_checker", bad), errors)
        self.assertEqual(clean.citation_errors("fact_checker", {"claims": [check("Pexo完成超千万美元的天使轮")], "summary": ""}), [])

    def test_uploaded_document_pages_are_citable(self):
        docs = DocumentStore()
        docs.add_text("bp.txt", "公司 2026 年 9 月 ARR 为 120 万美元。毛利率 55%。")
        box = Toolbox(docs)
        content = box.execute(ToolCall("1", "read_document", {"name": "bp.txt", "start_page": 1, "end_page": 1})).content
        self.assertIn("【第 1 页】来源编号：D1", content)
        item = box.verify_evidence({"sources": ["[1] 《bp.txt》第 1 页"], "evidence": [
            evidence("2026 年 9 月 ARR 为 120 万美元", source="D1", date=("2026 年 9 月", "摘录"))]})["evidence"][0]
        self.assertTrue(item["excerpt_verified"], item["field_issues"])

    def test_agent_round_trip_reads_page_fixes_citation_and_submits(self):
        sheet = {"company_name": "Pexo", "one_liner": "视频 Agent", "stage": "天使轮", "listing_status": "未上市",
                 "listing_evidence": "无", "fact_sheet": "完成天使轮 [1]", "information_gaps": [],
                 "sources": ["[1] 报道 - https://news.example/a"]}
        wrong = {**sheet, "evidence": [evidence("完成超千万美元天使轮", first=6, last=6, quantities=["超千万美元"])]}
        right = {**sheet, "evidence": [evidence("完成超千万美元天使轮", first=1, last=1, quantities=["超千万美元"])]}
        session = Mock()
        session.send.side_effect = [
            Turn(tool_calls=[ToolCall("1", "fetch_url", {"url": "https://news.example/a"})]),
            Turn(tool_calls=[ToolCall("2", "submit_result", wrong)]),
            Turn(tool_calls=[ToolCall("3", "submit_result", right)]),
        ]
        session.input_tokens = session.output_tokens = 0
        session.cost_usd = 0
        llm = Mock()
        llm.session.return_value = session
        box = Toolbox()
        agent = Agent(COMPANY_RESEARCHER, llm, box, lambda event: None, system="", schema=FACT_SHEET,
                      extra_check=lambda data: box.citation_errors("company_researcher", data))
        with patch("vcagents.tools.fetch_url", return_value=f"标题：报道\nURL：https://news.example/a\n\n{PAGE}"):
            data = agent.run("研究 Pexo")
        returned = session.add_tool_results.call_args_list[1].args[0][0].content
        self.assertIn("超千万美元", returned)
        self.assertIn("在引用原文及相邻句中找不到", returned)
        item = box.verify_evidence(data)["evidence"][0]
        self.assertTrue(item["excerpt_verified"])
        self.assertIn("达泰资本", item["excerpt"])


class FactCheckFormatToleranceTest(unittest.TestCase):
    def test_stringified_claims_and_missing_date_fields_are_normalized(self):
        import json
        from vcagents import schemas

        schema = schemas.fact_check_schema(["chair"], [])
        claim = check("完成超千万美元天使轮")
        del claim["fact"]["date_quote"], claim["fact"]["date"]
        data = schemas.normalize(schema, {"claims": [json.dumps(claim, ensure_ascii=False)], "summary": ""})
        self.assertEqual(schemas.validate(schema, data), [])
        self.assertEqual(data["claims"][0]["fact"]["date"], {"text": "", "basis": "未给出"})


class CitationStepBudgetTest(unittest.TestCase):
    def test_rejected_citations_keep_the_work_and_record_the_problem(self):
        sheet = {"company_name": "Pexo", "one_liner": "视频 Agent", "stage": "天使轮", "listing_status": "未上市",
                 "listing_evidence": "无", "fact_sheet": "完成天使轮 [1]", "information_gaps": [],
                 "sources": ["[1] 报道 - https://news.example/a"], "evidence": [evidence("完成天使轮", source="W9")]}
        session = Mock()
        session.send.side_effect = ([Turn(tool_calls=[ToolCall(str(i), "web_search", {"query": "x"})]) for i in range(3)]
                                    + [Turn(tool_calls=[ToolCall("s", "submit_result", sheet)])] * 10)
        session.input_tokens = session.output_tokens = 0
        session.cost_usd = 0
        llm = Mock()
        llm.session.return_value = session
        search = Mock()
        search.search.return_value = "摘要"
        box = Toolbox(search=search)
        agent = Agent(COMPANY_RESEARCHER, llm, box, lambda event: None, system="", schema=FACT_SHEET, max_steps=4,
                      extra_check=lambda data: ["引用无效"])
        data = agent.run("研究 Pexo")
        self.assertEqual(data["company_name"], "Pexo")
        self.assertEqual(agent.unresolved_issues, ["引用无效"])
        self.assertEqual(session.send.call_count, 6)


if __name__ == "__main__":
    unittest.main()
