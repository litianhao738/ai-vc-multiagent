"""来自真实运行的反例：写法差异不能再被退回，真正的事实和数字错误仍要拦住。"""
import copy
import tempfile
import unittest
from unittest.mock import Mock, patch

import httpx
import openai

from vcagents import CompanyInput, SectorPipeline, SectorRequest, Settings, grounding, save_result, schemas
from vcagents.agent import Agent
from vcagents.citations import SourceRegistry, fact_issues, resolve
from vcagents.llm import LLMError, OpenAICompatLLM, ToolCall, Turn
from vcagents.quantities import unmatched
from vcagents.report import render_memo
from vcagents.report_checks import publication_issues, publication_warnings
from vcagents.roles import COMPANY_RESEARCHER
from vcagents.schemas import FACT_SHEET

PAGE = ("Higgsfield raised a $400M Series B at a $5.4B valuation, led by DST Global.\n"
        "Users grew 超过 3000% and 20.3 % of them pay.\n"
        "Revenue under contract has grown tenfold since June, 18 months after launch.\n"
        "第四句。第五句。第六句。第七句。第八句。\n"
        "Competitor Acme reached $999 million.\n")


def quote(first=1, last=None, page=PAGE, url="https://news.example/2026/10/26/a"):
    registry = SourceRegistry()
    source = registry.register_web(url, "报道", page)
    source.render()
    return resolve(registry, {"source": source.id, "from": first, "to": last or first})


def fact(statement="完成融资", quantities=(), entities=(), date=""):
    return {"subject": "Higgsfield", "statement": statement, "entities": list(entities),
            "quantities": [{"metric": "金额", "text": q} for q in quantities],
            "date": {"text": date, "basis": "摘录" if date else "未给出"}}


class FieldToleranceTest(unittest.TestCase):
    def test_writing_differences_are_not_errors(self):
        self.assertEqual(fact_issues(fact(quantities=["$400 million", "$5.4 billion"]), quote(1), None), [])
        self.assertEqual(fact_issues(fact(quantities=["超过3000%", "20.3%"]), quote(2), None), [])
        self.assertEqual(fact_issues(fact(quantities=["Eighteen months"]), quote(3), None), [])
        self.assertEqual(fact_issues(fact(entities=["dst global", "D.S.T Global", "ＤＳＴ　Global"]), quote(1), None), [])

    def test_adjacent_sentence_counts_and_excerpt_includes_it(self):
        self.assertEqual(fact_issues(fact(quantities=["$400M"]), quote(2), None), [])

    def test_name_elsewhere_in_the_source_is_enough(self):
        self.assertEqual(fact_issues(fact(entities=["Acme"]), quote(1), None), [])

    def test_date_may_come_from_the_rest_of_the_source_or_its_url(self):
        self.assertEqual(fact_issues(fact(date="2026-10-26"), quote(1), None), [])
        self.assertEqual(fact_issues(fact("2026 年完成融资"), quote(1), None), [])

    def test_multiples_written_in_words_match(self):
        self.assertEqual(unmatched("6月以来增长10倍", "grown tenfold since June"), [])
        self.assertEqual(unmatched("ARR 翻到 2 倍", "ARR doubled"), [])

    def test_real_errors_are_still_rejected(self):
        self.assertTrue(fact_issues(fact(quantities=["$500 million"]), quote(1), None))
        self.assertTrue(fact_issues(fact("融资 9 亿美元"), quote(1), None))
        self.assertTrue(fact_issues(fact(entities=["红杉资本"]), quote(1), None))
        self.assertTrue(fact_issues(fact(date="2019-03-01"), quote(1), None))
        self.assertTrue(fact_issues(fact(quantities=["$999 million"]), quote(1), None), "太远的数字不算引用附近")

    def test_decimal_points_and_signs_are_not_ignored_as_formatting(self):
        source = quote(page="The reported margin was 20.3%.")
        self.assertEqual(fact_issues(fact(quantities=["２０．３％"]), source, None), [])
        for value in ("203%", "2.03%", "-20.3%", "−20.3%", "－２０．３％"):
            with self.subTest(value=value):
                self.assertTrue(fact_issues(fact(quantities=[value]), source, None))

    def test_equivalent_minus_styles_preserve_negative_values(self):
        source = quote(page="The reported margin was −20.3%.")
        for value in ("-20.3%", "−20.3%", "－２０．３％"):
            with self.subTest(value=value):
                self.assertEqual(fact_issues(fact(quantities=[value]), source, None), [])
        self.assertTrue(fact_issues(fact(quantities=["20.3%"]), source, None))

    def test_statement_punctuation_and_length_follow_the_relaxed_contract(self):
        statement = "公司披露融资用途为产品研发；" + "相关表述属于同一事实的背景说明，" * 5
        self.assertGreater(len(statement), 60)
        self.assertLessEqual(len(statement), 160)
        self.assertEqual(fact_issues(fact(statement), quote(page=statement), None), [])
        self.assertTrue(fact_issues(fact("长" * 161), quote(), None))


class SchemaToleranceTest(unittest.TestCase):
    def test_enum_ignores_case_and_punctuation_and_unknown_category_falls_back(self):
        schema = {"type": "string", "enum": ["收入或ARR", "其他"]}
        self.assertEqual(schemas.normalize(schema, " 收入或arr "), "收入或ARR")
        self.assertEqual(schemas.normalize(schema, "估值倍数"), "其他")
        strict = {"type": "string", "enum": ["属实", "不实"]}
        self.assertEqual(schemas.normalize(strict, "也许"), "也许")

    def test_entity_object_and_fenced_json_are_accepted(self):
        self.assertEqual(schemas.normalize({"type": "string"}, {"name": "腾讯"}), "腾讯")
        schema = {"type": "object", "properties": {"a": {"type": "string"}}}
        self.assertEqual(schemas.normalize(schema, '```json\n{"a": "1"}\n```'), {"a": "1"})


class GroundingToleranceTest(unittest.TestCase):
    index = {"E-C1-1": {"values": {3e8}, "verified": True}, "E-C1-2": {"values": {2e8}, "verified": True}}

    def check(self, text):
        return grounding.check_material("x", {"text": text}, self.index)

    def assert_clean(self, text):
        report = self.check(text)
        self.assertEqual(report["findings"], [], text)

    def test_citation_writing_differences(self):
        self.assert_clean("收入 3 亿美元[e-c1-1]。")
        self.assert_clean("收入 3 亿美元[E－C1－1]。")
        self.assert_clean("收入 3 亿美元（E-C1-1）。")

    def test_citation_after_the_full_stop_or_later_in_the_paragraph(self):
        self.assert_clean("收入 3 亿美元。[E-C1-1] 下一句没有数字。")
        self.assert_clean("收入 3 亿美元。增长很快。[E-C1-1]")

    def test_table_row_assumption(self):
        self.assert_clean("| 假设：按 20% 稀释 | 约 16 亿美元 |")

    def test_assumption_labelled_inside_the_sentence(self):
        self.assert_clean("倍数超过 15 倍（该门槛为本报告假设）。")
        self.assertTrue(grounding.errors_for_submission(self.check("机构估算收入 15 亿美元。")))

    def test_known_number_with_wrong_or_missing_citation_is_only_a_note(self):
        for text in ("收入 2 亿美元[E-C1-1]。", "收入 2 亿美元。"):
            report = self.check(text)
            self.assertEqual([f["severity"] for f in report["findings"]], ["note"], text)
            self.assertEqual(grounding.errors_for_submission(report), [], text)

    def test_invented_number_is_still_rejected(self):
        for text in ("收入 9 亿美元[E-C1-1]。", "收入 9 亿美元。"):
            report = self.check(text)
            self.assertTrue(grounding.errors_for_submission(report), text)

    def test_arithmetic(self):
        self.assert_clean("假设约 7.7 倍（54 亿 ÷ 7 亿）。")
        self.assert_clean("假设约 7.7 倍（5400 ÷ 701）。")
        self.assertTrue(grounding.errors_for_submission(self.check("假设约 9 倍（54 亿 ÷ 7 亿）。")))


class AgentRoundsTest(unittest.TestCase):
    def test_keeps_the_work_after_two_corrections_and_records_the_problem(self):
        sheet = {"company_name": "Pexo", "one_liner": "视频", "stage": "天使轮", "listing_status": "未上市",
                 "listing_evidence": "无", "fact_sheet": "完成", "information_gaps": [], "sources": [], "evidence": []}
        session = Mock()
        session.send.return_value = Turn(tool_calls=[ToolCall("s", "submit_result", sheet)])
        session.input_tokens = session.output_tokens = 0
        session.cost_usd = 0
        llm = Mock()
        llm.session.return_value = session
        agent = Agent(COMPANY_RESEARCHER, llm, None, lambda event: None, system="", schema=FACT_SHEET,
                      extra_check=lambda data: ["引用无效"])
        data = agent.run("研究")
        self.assertEqual(data["company_name"], "Pexo")
        self.assertEqual(agent.unresolved_issues, ["引用无效"])
        self.assertEqual(session.send.call_count, 3)


class TimeoutMessageTest(unittest.TestCase):
    def send(self, error):
        llm = OpenAICompatLLM("k", "m", "http://localhost")
        llm.client = Mock()
        llm.client.chat.completions.create.side_effect = error
        return llm.session("s", "p", []).send()

    def test_timeout_is_not_reported_as_a_network_failure(self):
        request = httpx.Request("POST", "http://localhost")
        with self.assertRaisesRegex(LLMError, "超时"):
            self.send(openai.APITimeoutError(request=request))
        with self.assertRaisesRegex(LLMError, "无法连接"):
            self.send(openai.APIConnectionError(request=request))

    def test_long_output_roles_have_enough_room(self):
        self.assertEqual(OpenAICompatLLM("k", "m", "http://localhost").client.timeout, 600)


class ReportCheckTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = SectorPipeline(Settings(mock=True, holding_years=5)).run(
            SectorRequest("aigc-content", [CompanyInput("Higgsfield", post_money_musd=5400, investment_musd=10)]))

    def issues(self, **decision):
        result = copy.deepcopy(self.base)
        result.decision.update(decision)
        result.returns = [{"scenario": "乐观", "multiple": 1.9}, {"scenario": "中性", "multiple": .56},
                          {"scenario": "悲观", "multiple": .08}]
        return publication_issues(result, render_memo(result))

    def value_flagged(self, text):
        return any("低估或存在安全边际" in issue for issue in self.issues(valuation_view=text))

    def test_value_claims_with_synonyms_and_other_scenario_names(self):
        for text in ("价格不贵，具有安全边际。", "目前明显低估。", "估值便宜。"):
            self.assertTrue(self.value_flagged(text), text)

    def test_negated_or_conditional_value_claims_are_fine(self):
        for text in ("谈不上有安全边际。", "现有依据不足以认定显著低估。", "如果收入可持续，才有安全边际。"):
            self.assertFalse(self.value_flagged(text), text)

    def memo_issues(self, sentence):
        result = copy.deepcopy(self.base)
        result.sector_brief["evidence"] = [{"evidence_id": "E-S-1", "claim": "Higgsfield公司称毛利为正", "source_type": "公司口径",
                                            "excerpt": "positive gross margin and net revenue retention of 300 percent",
                                            "as_of": "2026-10", "source_ref": "[S-1]", "source_linked": True, "excerpt_verified": True}]
        result.sector_brief["sources"] = ["[S-1] https://example.org/a"]
        result.decision["headline"] = sentence
        return publication_issues(result, render_memo(result))

    def test_disclosure_contradiction_only_when_really_contradictory(self):
        self.assertTrue(any("已有毛利披露" in i for i in self.memo_issues("毛利尚未披露。")))
        self.assertTrue(any("已有净收入留存披露" in i for i in self.memo_issues("NRR 没有披露。")))
        for sentence in ("毛利率数字没有披露。", "NRR 的计算口径未披露。"):
            self.assertFalse(any("披露，正文仍称" in i for i in self.memo_issues(sentence)), sentence)

    def score_flagged(self, sentence):
        return any("加权分" in i for i in self.memo_issues(sentence))

    def test_score_conflict_with_other_wordings(self):
        for sentence in ("综合得分 8.30 分。", "综合评分为 8.30。", "加权总分 8.30 / 10。"):
            self.assertTrue(self.score_flagged(sentence), sentence)
        self.assertFalse(self.score_flagged("综合得分范围是 1.0–9.9 分。"))

    def test_internal_wording_warns_but_does_not_block(self):
        result = copy.deepcopy(self.base)
        result.decision["headline"] = "依据底稿和用户提供的数据。"
        memo = render_memo(result)
        self.assertFalse(any("内部用语" in i or "程序用语" in i for i in publication_issues(result, memo)))
        self.assertTrue(any("内部用语" in w for w in publication_warnings(result, memo)))

    def test_blocked_report_keeps_the_full_draft(self):
        result = copy.deepcopy(self.base)
        result.decision["valuation_view"] = "价格不贵，具有安全边际。"
        result.returns = [{"scenario": "基线", "multiple": .56}]
        with tempfile.TemporaryDirectory() as root:
            folder = save_result(result, root)
            self.assertFalse((folder / "report.md").exists())
            draft = (folder / "research-draft.md").read_text(encoding="utf-8")
        self.assertIn("不能作为正式报告", draft)
        self.assertIn("## 一、投资摘要", draft)


if __name__ == "__main__":
    unittest.main()
