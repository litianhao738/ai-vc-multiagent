"""不调用任何模型的测试：python -m unittest discover tests"""
import tempfile
import unittest
import json
import random
from types import SimpleNamespace
from pathlib import Path

from backtest import evaluate
from evaluation.fact_checker_smoke import benchmark_cases, summarize
from evaluation.skill_fact_check_ab import CASES as SKILL_FACT_CASES, summarize as summarize_skill_facts
from evaluation.unsupported_detail_smoke import passed_case
from vcagents import CompanyInput, DocumentStore, ExpandedSectorPipeline as SectorPipeline, SectorRequest, Settings, save_partial_result, save_result
from vcagents import schemas
from vcagents.agent import Agent, AgentError
from vcagents.llm import LLM, ChatSession, LLMRouter, OpenAICompatSession, ToolCall, ToolSpec, Turn, _fake, _parse_call
from vcagents.pipeline import PipelineError, _namespace_sources
from vcagents.roles import BULL, COMPANY_RESEARCHER, FACT_CHECKER
from vcagents.report import _multiple, render_memo
from vcagents.scoring import exit_multiple, rank_companies, ranking_sensitivity, score_ranges, topsis, weighted_score
from vcagents.sectors import list_sectors, load_sector
from vcagents.tools import Toolbox


def real_status(pipeline, result):
    """mock 运行固定为演示稿；按真实模式规则重新判定报告状态。"""
    pipeline.settings.mock = False
    pipeline._finish_result(result, 0)
    return result.report_status


class ScriptedSession(ChatSession):
    """按脚本依次返回预设的回复，并记录智能体发回来的内容。"""

    def __init__(self, turns):
        super().__init__()
        self.turns = list(turns)
        self.received = []

    def send(self):
        return self.turns.pop(0)

    def add_tool_results(self, results, note=""):
        self.received.append(("tool", results, note))

    def add_user(self, text):
        self.received.append(("user", text))


class ScriptedLLM(LLM):
    def __init__(self, turns):
        self.last = ScriptedSession(turns)

    def session(self, system, prompt, tools, effort="medium"):
        return self.last


VALID_POSITION = {"thesis": "论点", "arguments": ["a"], "rebuttals": [], "would_change_mind": "x"}
REPORT_HEADINGS = ("## 一、投资价值与技术护城河", "## 二、赛道和市场", "## 三、公司比较与排序", "## 四、各公司研究摘要",
                   "## 五、共同风险", "## 六、", "## 七、尽调计划与附录", "### 方法与假设", "### 主要风险与应对")


class SectorPipelineTest(unittest.TestCase):
    def test_mock_pipeline_end_to_end(self):
        docs = DocumentStore()
        docs.add_text("bp.txt", "公司 2026 年 ARR 达到 100 万美元，完成天使轮融资。")
        events = []
        companies = [CompanyInput("甲公司"), CompanyInput("乙公司"), CompanyInput("丙公司")]
        result = SectorPipeline(Settings(mock=True, pairwise_checks=True, debate_rounds=2), {"甲公司": docs}, emit=events.append).run(
            SectorRequest("aigc-content", companies)
        )

        self.assertEqual(set(result.analyses), {"甲公司", "乙公司", "丙公司"})
        self.assertEqual([row["rank"] for row in result.ranking], [1, 2, 3])
        self.assertEqual(len(result.debate), 4)  # 默认 2 轮 × 多空双方
        self.assertEqual(set(result.usage), set(result.agent_seconds))
        self.assertEqual(len(result.pairwise), 3)
        self.assertTrue(all(p["forward"] and p["reverse"] for p in result.pairwise))
        self.assertTrue(result.decision_fact_check["claims"])
        self.assertIsNotNone(result.fact_check_pass_rate)
        self.assertIn(result.decision["verdict"], schemas.SECTOR_VERDICTS)
        self.assertIn(result.decision["top_pick"], result.analyses)
        self.assertTrue(set(result.decision["recommended"]).issubset(result.analyses))
        self.assertTrue(0 <= len(result.decision["recommended"]) <= 2)
        self.assertIn(result.decision["action"], schemas.ACTIONS)
        self.assertEqual(len(result.decision["recommended"]), len(set(result.decision["recommended"])))
        self.assertEqual(len(result.returns), len(result.decision["exit_scenarios"]))
        self.assertFalse(result.errors)
        self.assertTrue(any(e.kind == "log" and "检索资料" in e.message for e in events))

        with tempfile.TemporaryDirectory() as tmp:
            report = (save_result(result, tmp) / "report.md").read_text(encoding="utf-8")
        for heading in REPORT_HEADINGS:
            self.assertIn(heading, report)
        opening = report.split("## 二、赛道和市场", 1)[0]
        self.assertNotIn("本次建议", opening)
        self.assertIn(f"本次建议：{result.decision['action']}", report.split("### 交易条件与估值", 1)[1])
        self.assertIn(result.decision["headline"], opening)
        self.assertIn(result.decision["technical_moat"]["evidence"], opening)
        deep = report.split("## 六、", 1)[1].split("## 七、", 1)[0]
        for item in result.decision["top_pick_thesis"]:
            self.assertIn(item["judgment"], deep)
        for item in result.decision["top_pick_risks"][:2]:
            self.assertIn(item, deep)
        for item in result.decision["dd_checklist"][:3]:
            self.assertIn(item, report.split("## 七、", 1)[1])
        self.assertNotIn("TOPSIS |", report.split("## 七、", 1)[0])  # 评分方法细节只在附录
        self.assertIn("本轮投后估值：待补充", report)
        result.deal_metrics[result.decision["top_pick"]] = {"post_money_musd": 123}
        self.assertIn("本轮投后估值：123 百万美元", render_memo(result))
        result.errors["fact_checker"] = "核查未完成"
        self.assertIn("待复核研究稿", render_memo(result).split("## 一、投资价值与技术护城河", 1)[0])

    def test_sector_researcher_recommends_companies_when_none_given(self):
        for count in range(1, 6):
            with self.subTest(companies=count):
                result = SectorPipeline(Settings(mock=True)).run(SectorRequest("embodied-ai", max_companies=count))
                self.assertEqual(len(result.companies), count)
                self.assertEqual(result.companies, list(result.fact_sheets))
                self.assertFalse(result.pairwise)

    def test_shortlist_keeps_first_distinct_names_and_flags_too_few(self):
        schema = schemas.sector_brief_schema(2)
        wanted = schema["properties"]["candidates"]
        supplied = {"candidates": [" 甲 ", "", "甲", "乙", "丙"]}
        selected = schemas.normalize(schema, supplied)["candidates"]
        self.assertEqual(selected, ["甲", "乙"])
        self.assertFalse(schemas.validate(wanted, selected))
        too_few = schemas.normalize(schema, {"candidates": ["甲", " 甲 "]})["candidates"]
        self.assertTrue(schemas.validate(wanted, too_few))

    def test_report_renders_results_from_an_older_running_job(self):
        companies = [CompanyInput("甲"), CompanyInput("乙"), CompanyInput("丙")]
        result = SectorPipeline(Settings(mock=True)).run(SectorRequest("aigc-content", companies))
        del result.sensitivity
        del result.score_ranges
        del result.agent_seconds
        for sheet in result.fact_sheets.values():
            sheet.pop("listing_status")
            sheet.pop("listing_evidence")
        for row in result.returns:
            row.pop("dilution")
        report = render_memo(result)
        self.assertIn("上市状态：未记录", report)
        self.assertIn("后续稀释比例未记录", report)

    def test_duplicate_company_names_rejected(self):
        with self.assertRaises(ValueError):
            SectorPipeline(Settings(mock=True)).run(SectorRequest("aigc-content", [CompanyInput("A"), CompanyInput("A")]))

    def test_rejects_company_count_instead_of_truncating(self):
        companies = [CompanyInput(str(i)) for i in range(4)]
        with self.assertRaisesRegex(ValueError, "数量"):
            SectorPipeline(Settings(mock=True)).run(SectorRequest("aigc-content", companies, max_companies=3))
        for count in (0, 6, 2.5, True):
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, "公司数量"):
                SectorPipeline(Settings(mock=True)).run(SectorRequest("aigc-content", max_companies=count))

    def test_listed_company_is_excluded_from_candidate_ranking(self):
        pipeline = SectorPipeline(Settings(mock=True))
        original_run = pipeline._run

        def run(role, task, toolbox, agent_id, name, schema=None):
            output = original_run(role, task, toolbox, agent_id, name, schema)
            if agent_id.startswith("company_researcher:甲:") or agent_id == "company_researcher:甲":
                output["listing_status"] = "已上市"
                output["listing_evidence"] = "交易所公告"
            return output

        pipeline._run = run
        result = pipeline.run(SectorRequest("aigc-content", [CompanyInput("甲"), CompanyInput("乙"), CompanyInput("丙")]))
        self.assertEqual(set(result.analyses), {"乙", "丙"})
        self.assertNotIn("甲", result.decision["recommended"])
        self.assertEqual(result.fact_sheets["甲"]["listing_status"], "已上市")

    def test_repairs_false_claim_and_rechecks(self):
        pipeline = SectorPipeline(Settings(mock=True))
        original_run = pipeline._run
        checks = []

        def run(role, task, toolbox, agent_id, name, schema=None):
            if agent_id == "repair:company_researcher:甲公司":
                self.assertEqual(toolbox.company_host, "example.com")
            if role == FACT_CHECKER:
                checks.append(agent_id)
                claims = []
                if len(checks) == 1:
                    claims = [{"claim": "错误数字", "source_agent": "company_researcher:甲公司",
                               "original_source": "", "verdict": "不实", "evidence": "证据", "correction": "正确数字"}]
                result = {"claims": claims, "summary": "已检查"}
                if agent_id == "fact_checker_recheck":
                    result["repair_checks"] = [{"claim_id": "R-1", "status": "已修正", "reason": "已更正"}]
                return result
            return original_run(role, task, toolbox, agent_id, name, schema)

        pipeline._run = run
        companies = [CompanyInput("甲公司", "https://www.example.com"), CompanyInput("乙公司"), CompanyInput("丙公司")]
        result = pipeline.run(SectorRequest("aigc-content", companies))
        self.assertEqual(result.repairs[0]["source_agent"], "company_researcher:甲公司")
        self.assertIn("before", result.repairs[0])
        self.assertIn("after", result.repairs[0])
        self.assertIn("fact_checker_recheck", checks)
        self.assertIn("fact_checker_decision", checks)
        self.assertEqual(len(result.ranking), 3)

    def test_source_numbers_are_unique_per_material(self):
        material = {"fact_sheet": "收入 100 [1]，融资 [2]", "sources": ["[1] 官网", "[2] 新闻"],
                    "evidence": [{"claim": "收入 100", "source_ref": "[1]", "excerpt": "100",
                                  "source_type": "公司口径", "as_of": "2025"},
                                 {"claim": "融资", "source_ref": "[3]", "excerpt": "", "source_type": "未知", "as_of": "待核实"}]}
        renamed = _namespace_sources(material, "C1")
        self.assertEqual(renamed["fact_sheet"], "收入 100 [C1-1]，融资 [C1-2]")
        self.assertEqual(renamed["sources"], ["[C1-1] 官网", "[C1-2] 新闻"])
        self.assertEqual(renamed["evidence"][0]["evidence_id"], "E-C1-1")
        self.assertTrue(renamed["evidence"][0]["source_linked"])
        self.assertFalse(renamed["evidence"][1]["source_linked"])

    def test_named_source_labels_match_evidence_and_are_stable_on_reprocessing(self):
        material = {"fact_sheet": "收入 [S1]，融资 [S2]", "sources": ["[S1] 官网", "[S2] 新闻"],
                    "evidence": [{"claim": "收入", "source_ref": "[S1]", "excerpt": "",
                                  "source_type": "公司口径", "as_of": "待核实"}]}
        renamed = _namespace_sources(material, "C1")
        self.assertEqual(renamed["fact_sheet"], "收入 [C1-1]，融资 [C1-2]")
        self.assertEqual(renamed["evidence"][0]["source_ref"], "[C1-1]")
        self.assertTrue(renamed["evidence"][0]["source_linked"])
        self.assertEqual(_namespace_sources(renamed, "C1"), renamed)

    def test_researcher_must_explicitly_submit_evidence_even_if_empty(self):
        data = _fake(schemas.FACT_SHEET, random.Random(1))
        data.pop("evidence")
        normalized = schemas.normalize(schemas.FACT_SHEET, data)
        self.assertTrue(any("evidence" in error for error in schemas.validate(schemas.FACT_SHEET, normalized)))
        data["evidence"] = []
        self.assertFalse(schemas.validate(schemas.FACT_SHEET, schemas.normalize(schemas.FACT_SHEET, data)))

    def test_unresolved_false_claim_stops_report(self):
        pipeline = SectorPipeline(Settings(mock=True))
        original_run = pipeline._run

        def run(role, task, toolbox, agent_id, name, schema=None):
            if agent_id.startswith("fact_checker_decision"):
                return original_run(role, task, toolbox, agent_id, name, schema)
            if role == FACT_CHECKER:
                result = {"claims": [{"claim": "错误", "source_agent": "company_researcher:甲公司",
                                    "original_source": "", "verdict": "不实", "evidence": "证据", "correction": "更正"}],
                        "summary": "不实"}
                if agent_id == "fact_checker_recheck":
                    result["repair_checks"] = [{"claim_id": "R-1", "status": "已修正", "reason": "已更正"}]
                return result
            return original_run(role, task, toolbox, agent_id, name, schema)

        pipeline._run = run
        companies = [CompanyInput("甲公司"), CompanyInput("乙公司"), CompanyInput("丙公司")]
        result = pipeline.run(SectorRequest("aigc-content", companies))
        self.assertEqual(result.status, "completed")
        self.assertEqual(real_status(pipeline, result), "核查或研究未完成草稿")
        self.assertTrue(any(u["problem"] == "不实" for u in result.unresolved_checks))
        with tempfile.TemporaryDirectory() as tmp:
            folder = save_result(result, tmp)
            self.assertFalse((folder / "report.md").exists())
            draft = (folder / "research-draft.md").read_text(encoding="utf-8")
            self.assertNotIn("### 未解决的核查事项", draft)
            self.assertIn("当前为研究草稿，不能作为正式报告", draft)
            self.assertIn("未解决的核查事项", (folder / "audit.md").read_text(encoding="utf-8"))

    def test_checked_source_coverage_is_limited_to_checked_claims(self):
        pipeline = SectorPipeline(Settings(mock=True))
        original_run = pipeline._run

        def run(role, task, toolbox, agent_id, name, schema=None):
            if role == FACT_CHECKER:
                return {"claims": [
                    {"claim": "a", "source_agent": "sector_researcher", "original_source": "[S-1]",
                     "verdict": "属实", "evidence": "证据", "correction": ""},
                    {"claim": "b", "source_agent": "sector_researcher", "original_source": "",
                     "verdict": "公司口径", "evidence": "BP", "correction": ""}], "summary": "抽查"}
            return original_run(role, task, toolbox, agent_id, name, schema)

        pipeline._run = run
        result = pipeline.run(SectorRequest("aigc-content", [CompanyInput("甲"), CompanyInput("乙"), CompanyInput("丙")]))
        self.assertEqual(result.checked_source_coverage, 0.5)
        self.assertEqual(result.fact_check_pass_rate, 0.5)

    def test_company_reported_claim_is_not_independently_verified(self):
        pipeline = SectorPipeline(Settings(mock=True))
        original_run = pipeline._run

        def run(role, task, toolbox, agent_id, name, schema=None):
            if role == FACT_CHECKER:
                return {"claims": [{"claim": "收入", "source_agent": "sector_researcher", "original_source": "[S-1]",
                                    "verdict": "公司口径", "evidence": "BP", "correction": ""}], "summary": "公司口径"}
            return original_run(role, task, toolbox, agent_id, name, schema)

        pipeline._run = run
        result = pipeline.run(SectorRequest("aigc-content", [CompanyInput("甲"), CompanyInput("乙"), CompanyInput("丙")]))
        self.assertEqual(result.fact_check_pass_rate, 0.0)

    def test_repair_must_be_confirmed_for_each_false_claim(self):
        pipeline = SectorPipeline(Settings(mock=True))
        original_run = pipeline._run

        def run(role, task, toolbox, agent_id, name, schema=None):
            if agent_id.startswith("fact_checker_decision"):
                return original_run(role, task, toolbox, agent_id, name, schema)
            if role == FACT_CHECKER:
                if agent_id == "fact_checker":
                    return {"claims": [{"claim": "收入", "source_agent": "company_researcher:甲",
                                        "original_source": "", "verdict": "不实", "evidence": "证据", "correction": ""}],
                            "summary": "存在不实"}
                return {"claims": [], "summary": "未能确认", "repair_checks": [
                    {"claim_id": "R-1", "status": "无法判断", "reason": "内容有歧义"}]}
            return original_run(role, task, toolbox, agent_id, name, schema)

        pipeline._run = run
        result = pipeline.run(SectorRequest("aigc-content", [CompanyInput("甲"), CompanyInput("乙"), CompanyInput("丙")]))
        self.assertEqual([(u["claim_id"], u["problem"]) for u in result.unresolved_checks], [("R-1", "无法判断")])
        self.assertEqual(real_status(pipeline, result), "核查或研究未完成草稿")

    def test_unverified_claim_left_after_repair_yields_research_draft(self):
        pipeline = SectorPipeline(Settings(mock=True))
        original_run = pipeline._run

        def run(role, task, toolbox, agent_id, name, schema=None):
            if agent_id.startswith("fact_checker_decision"):
                return original_run(role, task, toolbox, agent_id, name, schema)
            if role == FACT_CHECKER:
                claim = {"claim": "内测于 3 月上线", "source_agent": "company_researcher:甲", "original_source": "",
                         "verdict": "无法核实", "evidence": "原文未覆盖", "correction": ""}
                if agent_id == "fact_checker":
                    return {"claims": [claim], "summary": "缺少原文"}
                if agent_id == "fact_checker_recheck":
                    return {"claims": [claim], "summary": "仍缺原文",
                            "repair_checks": [{"claim_id": "R-1", "status": "仍存在", "reason": "底稿仍保留"}]}
                return {"claims": [], "summary": "无"}
            return original_run(role, task, toolbox, agent_id, name, schema)

        pipeline._run = run
        result = pipeline.run(SectorRequest("aigc-content", [CompanyInput("甲"), CompanyInput("乙"), CompanyInput("丙")]))
        self.assertEqual(result.status, "completed")
        self.assertEqual({u["problem"] for u in result.unresolved_checks}, {"仍存在", "无法核实"})
        self.assertNotIn("不实", {u["verdict"] for u in result.unresolved_checks})
        result.decision_fact_check = {"claims": [{"claim": "x", "verdict": "属实"}]}
        self.assertEqual(real_status(pipeline, result), "关键事实整理中的研究稿")

    def test_fact_check_runs_in_two_batches(self):
        pipeline = SectorPipeline(Settings(mock=True))
        original_run = pipeline._run
        seen = {}

        def run(role, task, toolbox, agent_id, name, schema=None):
            if role == FACT_CHECKER and agent_id in ("fact_checker", "fact_checker:analysis"):
                seen[agent_id] = schema["properties"]["claims"]["items"]["properties"]["source_agent"]["enum"]
                self.assertEqual(schema["properties"]["claims"]["maxItems"], 8)
            return original_run(role, task, toolbox, agent_id, name, schema)

        pipeline._run = run
        pipeline.run(SectorRequest("aigc-content", [CompanyInput("甲"), CompanyInput("乙"), CompanyInput("丙")]))
        self.assertTrue(all(k == "sector_researcher" or k.startswith("company_researcher:") for k in seen["fact_checker"]))
        self.assertIn("comparator", seen["fact_checker:analysis"])
        self.assertFalse(set(seen["fact_checker"]) & set(seen["fact_checker:analysis"]))

    def _run_with_failures(self, failing_ids):
        pipeline = SectorPipeline(Settings(mock=True))
        original_run = pipeline._run
        captured = {}

        def run(role, task, toolbox, agent_id, name, schema=None):
            if agent_id == "fact_checker":
                return {"claims": [{"claim": "收入", "source_agent": "company_researcher:甲", "original_source": "",
                                    "verdict": "无法核实", "evidence": "缺原文", "correction": ""}], "summary": "缺原文"}
            if agent_id in failing_ids:
                raise AgentError("接口错误 402")
            output = original_run(role, task, toolbox, agent_id, name, schema)
            if agent_id.startswith("company_analyst:") or agent_id == "comparator":
                captured[agent_id] = output
            return output

        pipeline._run = run
        try:
            result = pipeline.run(SectorRequest("aigc-content", [CompanyInput("甲"), CompanyInput("乙"), CompanyInput("丙")]))
        except PipelineError:
            result = pipeline.last_result
        return pipeline, result, captured

    def test_failed_reanalysis_invalidates_old_analysis_and_stops_publication(self):
        pipeline, result, captured = self._run_with_failures({"reanalysis:乙"})
        self.assertEqual(result.status, "failed")
        self.assertNotIn("乙", result.analyses)
        self.assertIsNot(result.analyses["甲"], captured["company_analyst:甲"])
        self.assertIn("reanalysis:乙", result.errors)
        self.assertFalse(result.ranking)
        self.assertFalse(result.decision)
        with tempfile.TemporaryDirectory() as tmp:
            folder = save_partial_result(result, tmp)
            self.assertFalse((folder / "report.md").exists())
            self.assertTrue((folder / "partial-result.json").exists())
        archived = next(item for item in result.repairs if item["source_agent"] == "下游分析失效")
        self.assertEqual(archived["before"]["analyses"]["乙"], captured["company_analyst:乙"])

    def test_failed_repair_and_recomparison_keep_original_materials(self):
        pipeline, result, captured = self._run_with_failures({"repair:company_researcher:甲", "recomparison"})
        self.assertEqual(result.status, "completed")
        self.assertIn("repair:company_researcher:甲", result.errors)
        self.assertIn("recomparison", result.errors)
        self.assertIsNot(result.comparison, captured["comparator"])
        self.assertFalse(result.memo_reviews[0]["before"]["comparison"])
        self.assertTrue(any(r.get("failed") for r in result.repairs))

    def test_failed_reanalysis_of_every_company_flags_each_one(self):
        pipeline, result, captured = self._run_with_failures({"reanalysis:甲", "reanalysis:乙", "reanalysis:丙"})
        self.assertEqual(result.status, "failed")
        self.assertFalse(result.analyses)
        self.assertFalse(result.decision)
        self.assertFalse(result.ranking)

    def test_chair_repair_failure_does_not_emit_original_decision(self):
        pipeline = SectorPipeline(Settings(mock=True))
        original_run = pipeline._run

        def run(role, task, toolbox, agent_id, name, schema=None):
            if agent_id == "fact_checker":
                return {"claims": [], "summary": "核查完成"}
            if agent_id == "fact_checker_decision":
                return {"claims": [{"claim": "错误数字", "source_agent": "chair", "original_source": "",
                                    "verdict": "不实", "evidence": "证据", "correction": ""}], "summary": "不实"}
            if agent_id == "repair:chair":
                raise AgentError("修正失败")
            return original_run(role, task, toolbox, agent_id, name, schema)

        pipeline._run = run
        with self.assertRaisesRegex(AgentError, "修正失败"):
            pipeline.run(SectorRequest("aigc-content", [CompanyInput("甲"), CompanyInput("乙"), CompanyInput("丙")]))

    def test_backtest_counts_known_outcomes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "result.json").write_text(json.dumps({"decision": {"top_pick": "甲"}}), encoding="utf-8")
            cases = [{"result": "result.json", "as_of": "2022-12-31", "outcomes": {"甲": "融资"}},
                     {"result": "result.json", "as_of": "2022-12-31", "outcomes": {}}]
            report = evaluate(cases, root)
            self.assertEqual((report["known_cases"], report["positive_cases"], report["positive_rate"]), (1, 1, 1.0))


class AgentLoopTest(unittest.TestCase):
    def make_agent(self, turns):
        llm = ScriptedLLM(turns)
        return Agent(BULL, llm, None, lambda e: None, system="测试", max_steps=5), llm

    def test_nudges_when_model_answers_in_plain_text(self):
        agent, llm = self.make_agent([
            Turn(text="我觉得值得投。"),
            Turn(tool_calls=[ToolCall("1", "submit_result", VALID_POSITION)]),
        ])
        self.assertEqual(agent.run("任务")["thesis"], "论点")
        self.assertEqual(llm.last.received[0][0], "user")

    def test_rejects_invalid_submission_then_accepts_fix(self):
        agent, llm = self.make_agent([
            Turn(tool_calls=[ToolCall("1", "submit_result", {"thesis": "缺字段"})]),
            Turn(tool_calls=[ToolCall("2", "submit_result", VALID_POSITION)]),
        ])
        agent.run("任务")
        _, results, _ = llm.last.received[0]
        self.assertTrue(results[0].is_error)
        self.assertIn("缺少字段", results[0].content)

    def test_parses_json_written_as_text(self):
        agent, _ = self.make_agent([Turn(text='结果如下：{"thesis": "t", "arguments": "单条", "would_change_mind": "x"}')])
        out = agent.run("任务")
        self.assertEqual(out["arguments"], ["单条"])  # 字符串被修正为数组
        self.assertEqual(out["rebuttals"], [])  # 缺失的数组字段补为空

    def test_gives_up_after_repeated_plain_text(self):
        agent, _ = self.make_agent([Turn(text="不提交")] * 3)
        with self.assertRaises(AgentError):
            agent.run("任务")

    def test_stops_after_repeated_truncated_outputs(self):
        agent, _ = self.make_agent([Turn(truncated=True)] * 3)
        with self.assertRaisesRegex(AgentError, "连续截断 3 次"):
            agent.run("任务")

    def test_structured_submission_requests_strict_output(self):
        session = OpenAICompatSession(SimpleNamespace(), "系统", "任务", [
            ToolSpec("submit_result", "提交", schemas.DEBATE_POSITION, strict=True),
            ToolSpec("web_search", "搜索", {"type": "object"}),
        ])
        self.assertTrue(session.tools[0]["function"]["strict"])
        self.assertNotIn("strict", session.tools[1]["function"])

    def test_invalid_json_retry_reports_position_without_raw_content(self):
        call = _parse_call("1", "submit_result", '{"secret": "value",}')
        self.assertIsNone(call.input)
        self.assertIn("位置", call.error)
        self.assertNotIn("secret", call.error)
        agent, llm = self.make_agent([Turn(tool_calls=[call]), Turn(tool_calls=[ToolCall("2", "submit_result", VALID_POSITION)])])
        agent.run("任务")
        self.assertIn("位置", llm.last.received[0][1][0].content)

    def test_chair_decision_requires_consistent_pick_and_all_scenarios(self):
        schema = schemas.decision_schema(["甲", "乙", "丙"])
        data = _fake(schema, random.Random(1))
        data["top_pick"] = "甲"
        data["recommended"] = ["乙"]
        self.assertIn("首选公司必须在推荐名单中", schemas.validate_submission("chair", schema, data))
        data["recommended"] = ["甲"]
        data["action"] = "附条件投资"
        data["action_conditions"] = [" "]
        self.assertIn("附条件投资必须列出投资条件", schemas.validate_submission("chair", schema, data))
        data["action"], data["recommended"] = "立项", []
        self.assertIn("没有推荐公司时，本次建议只能是不投", schemas.validate_submission("chair", schema, data))
        data["action"] = "不投"
        for scenario, value in zip(data["exit_scenarios"], (100, 300, None)):
            scenario["exit_valuation_musd"], scenario["valuation_basis"] = value, "测试假设"
        self.assertEqual(["退出股权价值必须满足乐观 ≥ 基线 ≥ 悲观"], schemas.validate_submission("chair", schema, data))
        data["exit_scenarios"][1]["exit_valuation_musd"] = None
        self.assertEqual([], schemas.validate_submission("chair", schema, data))  # 没有推荐公司、未知估值都允许
        data["exit_scenarios"][0]["scenario"] = data["exit_scenarios"][1]["scenario"]
        self.assertIn("退出情景必须各包含一次乐观、基线和悲观", schemas.validate_submission("chair", schema, data))

    def test_empty_evidence_is_reviewed_once_without_forcing_fabrication(self):
        data = _fake(schemas.FACT_SHEET, random.Random(1))
        data["sources"] = ["[1] example source"]
        data["evidence"] = []
        llm = ScriptedLLM([Turn(tool_calls=[ToolCall("1", "submit_result", data)]),
                           Turn(tool_calls=[ToolCall("2", "submit_result", data)])])
        agent = Agent(COMPANY_RESEARCHER, llm, None, lambda e: None, system="测试", max_steps=5)
        result = agent.run("任务")
        self.assertEqual(result["evidence"], [])
        self.assertEqual(agent.evidence_reviews, 1)
        self.assertEqual(agent.format_failures, 0)
        self.assertIn("重新检查原文", llm.last.received[0][1][0].content)


class ScoringTest(unittest.TestCase):
    def setUp(self):
        self.sector = load_sector("aigc-content")
        self.keys = [d.key for d in self.sector.dimensions]

    def analysis(self, score):
        return {"scores": {k: {"score": score, "reason": ""} for k in self.keys}}

    def test_weighted_score_uses_sector_weights(self):
        self.assertEqual(weighted_score({k: 7 for k in self.keys}, self.sector), 7.0)

    def test_analyst_scores_are_validated_before_ranking(self):
        schema = schemas.analyst_schema(self.sector)["properties"]["scores"]["properties"][self.keys[0]]["properties"]["score"]
        for invalid in (0, 11, True):
            with self.subTest(score=invalid):
                self.assertTrue(schemas.validate(schema, schemas.normalize(schema, invalid)))
        for valid in (1, 10):
            with self.subTest(score=valid):
                self.assertFalse(schemas.validate(schema, schemas.normalize(schema, valid)))

    def test_scoring_evidence_ids_are_limited_to_company_records(self):
        schema = schemas.analyst_schema(self.sector, ["E-C1-1"])["properties"]["scores"]["properties"][self.keys[0]]["properties"]["evidence_ids"]
        self.assertFalse(schemas.validate(schema, ["E-C1-1"]))
        self.assertTrue(schemas.validate(schema, ["E-C2-1"]))

    def test_fact_check_evidence_ids_are_limited_to_collected_records(self):
        schema = schemas.fact_check_schema(["company_researcher:甲"], ["E-C1-1"])
        refs = schema["properties"]["claims"]["items"]["properties"]["evidence_ids"]
        self.assertFalse(schemas.validate(refs, ["E-C1-1"]))
        self.assertTrue(schemas.validate(refs, ["E-C2-1"]))

    def test_topsis_puts_dominant_option_first(self):
        self.assertEqual(topsis([[9, 9], [5, 5], [1, 1]], [0.5, 0.5]), [1.0, 0.5, 0.0])

    def test_sensitivity_detects_first_place_change(self):
        result = ranking_sensitivity({"甲": self.analysis(7), "乙": self.analysis(7)}, self.sector)
        self.assertGreater(result["scenarios"], 0)
        self.assertGreater(result["changes"], 0)
        self.assertTrue(all(item["top_pick"] != result["original_top"] for item in result["examples"]))

    def test_score_range_reflects_low_confidence_and_missing_sources(self):
        analysis = {"scores": {k: {"score": 7, "confidence": "高", "evidence_ids": ["E1"]} for k in self.keys}}
        sheets = {"甲": {"evidence": [{"evidence_id": "E1", "source_linked": True, "source_type": "独立来源", "excerpt_verified": True}]}}
        self.assertEqual(score_ranges({"甲": analysis}, sheets, self.sector)["甲"]["min"], 7.0)
        analysis["scores"][self.keys[0]]["confidence"] = "低"
        item = score_ranges({"甲": analysis}, sheets, self.sector)["甲"]
        self.assertLess(item["min"], 7.0)
        self.assertGreater(item["max"], 7.0)
        sheets["甲"]["evidence"][0]["source_type"] = "公司口径"
        item = score_ranges({"甲": analysis}, sheets, self.sector)["甲"]
        self.assertEqual((item["min"], item["max"]), (1.0, 10.0))
        sheets["甲"]["evidence"][0]["source_type"] = "独立来源"
        sheets["甲"]["evidence"][0]["source_linked"] = False
        item = score_ranges({"甲": analysis}, sheets, self.sector)["甲"]
        self.assertEqual((item["min"], item["max"]), (1.0, 10.0))

    def test_ranking_orders_by_weighted_score(self):
        ranking = rank_companies({"低": self.analysis(4), "高": self.analysis(8)}, self.sector)
        self.assertEqual([r["company"] for r in ranking], ["高", "低"])
        self.assertFalse(ranking[0]["topsis_disagrees"])

    def test_exit_multiple(self):
        self.assertEqual(exit_multiple(770, 50, 0.5), 7.7)
        self.assertEqual(exit_multiple(0, 50, 0.5), 0)
        self.assertIsNone(exit_multiple(None, 50, 0.5))
        self.assertIsNone(exit_multiple(770, 0, 0.5))  # 没有本轮估值时不计算

    def test_missing_exit_valuation_is_not_misreported_as_missing_post_money(self):
        ret = {"multiple": None, "post_money_musd": 50, "exit_valuation_musd": None}
        self.assertIn("退出估值", _multiple(ret))
        self.assertNotIn("本轮估值", _multiple(ret))


class ConfigTest(unittest.TestCase):
    def test_bundled_sectors_load(self):
        for key in list_sectors():
            sector = load_sector(key)
            self.assertAlmostEqual(sum(d.weight for d in sector.dimensions), 1.0, places=2)

    def test_sector_weights_must_sum_to_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "bad.yaml").write_text(
                "key: bad\nname: 坏配置\ndimensions:\n  - {key: a, name: A, weight: 0.5, rubric: x}\n", encoding="utf-8"
            )
            with self.assertRaises(ValueError):
                load_sector("bad", Path(tmp))

    def test_models_file_overrides_and_rejects_unknown_roles(self):
        with tempfile.TemporaryDirectory() as tmp:
            good = Path(tmp, "good.yaml")
            good.write_text("chair: anthropic:claude-opus-5-5\n", encoding="utf-8")
            self.assertEqual(Settings.load(good).role_models["chair"], "anthropic:claude-opus-5-5")
            bad = Path(tmp, "bad.yaml")
            bad.write_text("ceo: some/model\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                Settings.load(bad)

    def test_roles_sharing_model_keep_role_specific_output_limits(self):
        settings = Settings(openrouter_api_key="test")
        settings.role_models["company_analyst"] = settings.role_models["sector_researcher"]
        router = LLMRouter(settings)
        self.assertEqual(router.for_role("sector_researcher").max_tokens, 32000)
        self.assertEqual(router.for_role("company_analyst").max_tokens, 32000)
        self.assertEqual(router.for_role("comparator").max_tokens, 16000)
        self.assertEqual(router.for_role("fact_checker").max_tokens, 32000)


class ToolsTest(unittest.TestCase):
    def test_chinese_document_search(self):
        docs = DocumentStore()
        docs.add_text("bp.txt", "我们的融资历史：2025 年完成天使轮。")
        self.assertIn("第 1 页", docs.search("融资历史"))
        self.assertIn("没有找到", docs.search("毛利率"))

    def test_fact_checker_can_read_each_companys_uploaded_bp(self):
        a, b = DocumentStore(), DocumentStore()
        a.add_text("BP.txt", "甲公司的材料")
        b.add_text("BP.txt", "乙公司的材料")
        toolbox = SectorPipeline(Settings(mock=True), {"甲": a, "乙": b})._factcheck_toolbox()
        self.assertIn("read_document", [tool.name for tool in toolbox.specs(FACT_CHECKER.tools)])
        self.assertIn("甲公司的材料", toolbox.docs.read("甲/BP.txt", 1, 1))
        self.assertIn("乙公司的材料", toolbox.docs.read("乙/BP.txt", 1, 1))

    def test_fetch_url_blocks_local_addresses(self):
        result = Toolbox().execute(ToolCall("1", "fetch_url", {"url": "http://127.0.0.1:8080/admin"}))
        self.assertTrue(result.is_error)

    def test_web_search_without_key_reports_error(self):
        result = Toolbox().execute(ToolCall("1", "web_search", {"query": "Pexo 融资"}))
        self.assertTrue(result.is_error)

    def test_fact_check_benchmark_rejects_duplicate_or_wrong_source(self):
        _, claims, expected = benchmark_cases()
        rows = [{**claim, "verdict": "公司口径"} for claim in claims]
        self.assertTrue(summarize(rows, expected)["passed"])
        duplicate = rows[:-1] + [rows[0]]
        self.assertFalse(summarize(duplicate, expected)["passed"])
        changed = [dict(row) for row in rows]
        changed[0]["claim"] = changed[0]["claim"].replace("2020", "2021")
        self.assertFalse(summarize(changed, expected)["passed"])
        rows[0]["source_agent"] = "company_researcher:Bravo"
        self.assertFalse(summarize(rows, expected)["passed"])

    def test_skill_fact_benchmark_counts_missing_and_duplicate_claims(self):
        rows = [{"claim": f"{key}: {claim}", "verdict": verdict, "source_agent": "research"}
                for key, claim, verdict in SKILL_FACT_CASES]
        self.assertEqual(summarize_skill_facts(rows)["correct"], len(SKILL_FACT_CASES))
        self.assertTrue(summarize_skill_facts(rows)["complete"])
        changed = rows[:-1] + [rows[0]]
        self.assertFalse(summarize_skill_facts(changed)["complete"])
        self.assertLess(summarize_skill_facts(changed)["correct"], len(SKILL_FACT_CASES))

    def test_partial_evidence_benchmark_requires_the_actual_missing_detail(self):
        text = "G1: 订单金额和 Digit 5 型号"
        row = {"claim": text, "verdict": "无法核实", "unsupported_parts": ["Digit 5 型号无来源"]}
        self.assertTrue(passed_case([row], text))
        row["unsupported_parts"] = ["订单合同条件未披露"]
        self.assertFalse(passed_case([row], text))


if __name__ == "__main__":
    unittest.main()
