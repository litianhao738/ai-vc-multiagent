"""Run-wide governance: deadline, cancellation, retry budget, request reuse, early comparison, audit coverage."""
import threading
import time
import unittest
from unittest.mock import Mock, patch

import httpx
import openai

from vcagents import CompanyInput, SectorPipeline, SectorRequest, Settings, schemas
from vcagents.governor import BudgetExceeded, RunCancelled, RunGovernor, RunTimeout
from vcagents.llm import LLMError, OpenAICompatLLM
from vcagents.quality import audit_plan, coverage_gaps
from vcagents.scheduler import DAGScheduler, TaskNode
from vcagents.tools import Toolbox
from vcagents.llm import ToolCall


def request(count=3):
    return SectorRequest("aigc-content", [CompanyInput(n) for n in ("Alpha", "Beta", "Gamma", "Delta", "Epsilon")[:count]])


def rate_limited():
    response = httpx.Response(429, request=httpx.Request("POST", "http://localhost"))
    return openai.RateLimitError("rate limited", response=response, body=None)


def armed(**kwargs):
    governor = RunGovernor(**kwargs)
    governor.backoff_base = 0.001
    governor.start()
    return governor


class GovernorTest(unittest.TestCase):
    def test_retries_are_drawn_from_one_run_wide_budget(self):
        governor = armed(retry_budget=3)
        attempts = []

        def flaky(_timeout):
            attempts.append(1)
            raise ConnectionError("down")

        for _ in range(2):  # Two requests, each allowed 4 retries locally, share 3 in total.
            with self.assertRaises(ConnectionError):
                governor.call("p", flaky, retries=4, timeout=10, retryable=lambda e: True)
        self.assertEqual(len(attempts), 5)  # 1+3 for the first, then 1 with no budget left
        self.assertEqual(governor.stats()["retries_left"], 0)

    def test_request_timeout_never_outlives_the_run(self):
        governor = armed(timeout_seconds=30)
        seen = []
        governor.call("p", lambda limit: seen.append(limit), retries=0, timeout=600)
        self.assertLessEqual(seen[0], 30)

    def test_deadline_cancel_and_cost_ceiling_stop_new_requests(self):
        clock = [0.0]
        governor = RunGovernor(timeout_seconds=10, clock=lambda: clock[0])
        governor.start()
        governor.check()
        clock[0] = 11
        with self.assertRaises(RunTimeout):
            governor.call("p", lambda limit: "never", retries=0, timeout=5)

        governor = armed()
        governor.cancel("用户停止")
        with self.assertRaisesRegex(RunCancelled, "用户停止"):
            governor.check()

        governor = armed(cost_budget_usd=1.0)
        governor.add_cost(0.6)
        governor.check()
        governor.add_cost(0.5)
        with self.assertRaises(BudgetExceeded):
            governor.check()

    def test_rate_limit_pauses_the_whole_provider_queue(self):
        governor = armed(retry_budget=5)
        calls = []

        def once_limited(_timeout):
            calls.append(time.monotonic())
            if len(calls) == 1:
                raise rate_limited()
            return "ok"

        with patch.object(governor, "max_retry_after", 0.2):
            exc = rate_limited()
            exc.response.headers["retry-after"] = "0.2"
            started = time.monotonic()
            result = governor.call("p", once_limited, retries=2, timeout=5,
                                   retryable=lambda e: getattr(e, "status_code", 0) == 429,
                                   retry_after=lambda e: 0.2)
        self.assertEqual(result, "ok")
        self.assertGreaterEqual(time.monotonic() - started, 0.18)
        # A different request to the same provider also waits for the cooldown window.
        self.assertGreater(governor._gate("p").cooldown_until, 0)

    def test_provider_rpm_spaces_requests(self):
        governor = armed(provider_rpm={"p": 600})  # one request every 0.1 s
        started = time.monotonic()
        for _ in range(3):
            governor.call("p", lambda _t: None, retries=0, timeout=5)
        self.assertGreaterEqual(time.monotonic() - started, 0.18)
        self.assertGreaterEqual(governor.stats()["rate_limit_waits"], 2)

    def test_identical_concurrent_requests_execute_once(self):
        governor = armed()
        gate, executed = threading.Event(), []

        def produce():
            executed.append(1)
            gate.wait(2)
            return "page"

        results = []
        threads = [threading.Thread(target=lambda: results.append(governor.shared(("k",), produce))) for _ in range(4)]
        for thread in threads:
            thread.start()
        time.sleep(0.1)
        gate.set()
        for thread in threads:
            thread.join(3)
        self.assertEqual(results, ["page"] * 4)
        self.assertEqual(len(executed), 1)
        self.assertEqual(governor.stats()["reused_requests"], 3)

    def test_failed_request_is_not_cached(self):
        governor = armed()
        attempts = []

        def produce():
            attempts.append(1)
            if len(attempts) == 1:
                raise ValueError("transient")
            return "ok"

        with self.assertRaises(ValueError):
            governor.shared(("k",), produce)
        self.assertEqual(governor.shared(("k",), produce), "ok")


class GovernedClientTest(unittest.TestCase):
    def test_model_requests_retry_within_the_run_budget_and_disable_sdk_retries(self):
        llm = OpenAICompatLLM("k", "qwen/model", "http://localhost")
        llm.governor = armed(retry_budget=1)
        llm.client = Mock()
        create = llm.client.with_options.return_value.chat.completions.create
        create.side_effect = rate_limited()
        with self.assertRaisesRegex(LLMError, "限流"):
            llm.session("s", "p", []).send()
        self.assertEqual(create.call_count, 2)  # one attempt + the only retry in the budget
        self.assertEqual(llm.client.with_options.call_args.kwargs["max_retries"], 0)
        self.assertEqual(llm.provider, "qwen")

    def test_same_search_and_page_are_fetched_once_but_recorded_per_agent(self):
        governor = armed()
        search = Mock()
        search.search.return_value = "结果"
        first = Toolbox(search=search, governor=governor)
        second = Toolbox(search=search, governor=governor)
        page = "标题：报道\nURL：https://n.example/a\n\nAlpha 收入 100 万美元。"
        with patch("vcagents.tools.fetch_url", return_value=page) as fetch:
            for box in (first, second):
                box.execute(ToolCall("1", "web_search", {"query": "Alpha  融资"}))
                box.execute(ToolCall("2", "web_search", {"query": "alpha 融资"}))
                box.execute(ToolCall("3", "fetch_url", {"url": "https://n.example/a"}))
        self.assertEqual(search.search.call_count, 1)
        self.assertEqual(fetch.call_count, 1)
        self.assertTrue(first.read_records() and second.read_records())

    def test_stopped_run_is_not_reported_to_the_model_as_a_tool_error(self):
        governor = armed()
        governor.cancel("停止")
        with self.assertRaises(RunCancelled):
            Toolbox(governor=governor).execute(ToolCall("1", "search_documents", {"query": "x"}))


class SchedulerTest(unittest.TestCase):
    def test_critical_failure_cancels_tasks_that_have_not_started(self):
        governor = armed()
        ran = []

        def fail():
            raise ValueError("评分失败")

        outcomes = DAGScheduler(1, governor).run([
            TaskNode("score", fail, critical=True),
            TaskNode("audit", lambda: ran.append("audit"))])
        self.assertIn("评分失败", outcomes["score"].error)
        self.assertEqual(ran, [])
        self.assertIsInstance(outcomes["audit"].exception, RunCancelled)

    def test_run_stop_is_raised_instead_of_returned_as_a_task_error(self):
        governor = armed()

        def stop():
            governor.cancel("用户停止")
            raise RunCancelled("用户停止")

        with self.assertRaises(RunCancelled):
            DAGScheduler(2, governor).run([TaskNode("a", stop), TaskNode("b", lambda: None, ("a",))])


class PipelineControlTest(unittest.TestCase):
    def test_comparison_starts_while_the_research_audit_is_still_running(self):
        compare_started = threading.Event()
        pipeline = SectorPipeline(Settings(mock=True))
        original = pipeline._run
        observed = []

        def run(role, task, toolbox, agent_id, name, schema=None):
            if agent_id == "analysis_compare":
                compare_started.set()
            if agent_id == "audit_research":
                observed.append(compare_started.wait(5))
            return original(role, task, toolbox, agent_id, name, schema)

        pipeline._run = run
        result = pipeline.run(request())
        self.assertEqual(observed, [True], "比较应在研究审计结束前开始")
        self.assertFalse(result.errors)
        self.assertEqual(pipeline.agents["analysis"].reply_count, 4)
        self.assertEqual(len(result.main_agents), 5)
        compare = next(h for h in result.handoffs if h["phase"] == "analysis_compare")
        self.assertNotIn("research_audit", [ref["name"] for ref in compare["inputs"]])
        tasks = {t["task"]: t for t in result.task_trace}
        self.assertLess(tasks["analysis_compare"]["start"],
                        tasks["audit_research"]["start"] + tasks["audit_research"]["seconds"])

    def test_scoring_failure_stops_the_parallel_audit(self):
        pipeline = SectorPipeline(Settings(mock=True, max_workers=1))
        original = pipeline._run
        phases = []

        def run(role, task, toolbox, agent_id, name, schema=None):
            phases.append(agent_id)
            if agent_id == "analysis_scores":
                raise RuntimeError("评分接口失败")
            return original(role, task, toolbox, agent_id, name, schema)

        pipeline._run = run
        with self.assertRaisesRegex(Exception, "评分"):
            pipeline.run(request())
        self.assertNotIn("audit_research", phases)

    def test_total_deadline_stops_the_run_and_keeps_partial_results(self):
        pipeline = SectorPipeline(Settings(mock=True, run_timeout_seconds=0.5))
        original = pipeline._run

        def run(role, task, toolbox, agent_id, name, schema=None):
            if agent_id == "research_bundle":
                time.sleep(0.7)
            return original(role, task, toolbox, agent_id, name, schema)

        pipeline._run = run
        with self.assertRaises(RunTimeout):
            pipeline.run(request())
        result = pipeline.last_result
        self.assertEqual(result.status, "failed")
        self.assertTrue(result.research_plan)
        self.assertIn("RunTimeout", result.run_control["stop_reason"])
        self.assertFalse(result.analyses)

    def test_cancel_from_another_thread_stops_later_phases(self):
        pipeline = SectorPipeline(Settings(mock=True))
        original = pipeline._run
        phases = []

        def run(role, task, toolbox, agent_id, name, schema=None):
            phases.append(agent_id)
            output = original(role, task, toolbox, agent_id, name, schema)
            if agent_id == "research_bundle":
                threading.Thread(target=pipeline.cancel, args=("页面已停止",)).start()
                time.sleep(0.1)
            return output

        pipeline._run = run
        with self.assertRaisesRegex(RunCancelled, "页面已停止"):
            pipeline.run(request())
        self.assertNotIn("chair_editor", phases)

    def test_each_run_gets_fresh_budgets(self):
        pipeline = SectorPipeline(Settings(mock=True))
        pipeline.cancel("上一轮被停止")
        self.assertFalse(pipeline.run(request()).errors)

    def test_invalid_limits_are_rejected_before_agents_run(self):
        for field, value in (("run_timeout_seconds", -1), ("retry_budget", -1), ("retry_budget", True),
                             ("cost_budget_usd", 0), ("provider_concurrency", 0), ("provider_rpm", {"x": 0}),
                             ("audit_claims_per_company", 5)):
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                SectorPipeline(Settings(mock=True, **{field: value})).run(request())

    def test_audit_ledger_records_run_control_and_coverage(self):
        from vcagents.report import render_audit
        result = SectorPipeline(Settings(mock=True)).run(request())
        self.assertIn("requests", result.run_control)
        self.assertEqual(set(result.audit_coverage["required"]),
                         {"sector_researcher", *(f"company_researcher:{n}" for n in result.companies)})
        audit = render_audit(result)
        self.assertIn("运行管控", audit)
        self.assertIn("研究核查覆盖", audit)


class AuditCoverageTest(unittest.TestCase):
    @staticmethod
    def sheet(*types):
        return {"evidence": [{"evidence_id": f"E-{i}", "fact_type": t, "statement": t} for i, t in enumerate(types, 1)]}

    def test_every_company_gets_its_highest_impact_evidence(self):
        plan = audit_plan({"sector_researcher": self.sheet("其他", "市场规模"),
                           "company_researcher:Alpha": self.sheet("产品发布", "融资", "团队履历", "估值"),
                           "company_researcher:Beta": self.sheet("产品发布"),
                           "company_researcher:Gamma": {"evidence": []}}, per_company=2)
        self.assertEqual([t["evidence_id"] for t in plan["sector_researcher"]["targets"]], ["E-2"])
        self.assertEqual([t["evidence_id"] for t in plan["company_researcher:Alpha"]["targets"]], ["E-4", "E-2"])
        self.assertEqual(plan["company_researcher:Beta"]["min_claims"], 1)
        self.assertEqual(plan["company_researcher:Gamma"], {"min_claims": 1, "targets": []})

    def test_claims_for_another_company_do_not_count(self):
        plan = audit_plan({"company_researcher:Alpha": self.sheet("融资", "估值"),
                           "company_researcher:Beta": self.sheet("融资")}, per_company=2)
        claims = [{"source_agent": "company_researcher:Alpha", "evidence_ids": ["E-1"]},
                  {"source_agent": "company_researcher:Alpha", "evidence_ids": ["E-2"]},
                  {"source_agent": "company_researcher:Alpha", "evidence_ids": ["E-1"]}]
        gaps = coverage_gaps(plan, claims)
        self.assertEqual([g["material"] for g in gaps], ["company_researcher:Beta"])
        self.assertEqual(gaps[0]["missing_evidence"], ["E-1"])

    def test_five_companies_raise_the_old_six_claim_ceiling(self):
        schema = schemas.fact_check_schema(["a"], None, min_claims=11)
        self.assertEqual(schema["properties"]["claims"]["minItems"], 11)
        self.assertGreaterEqual(schema["properties"]["claims"]["maxItems"], 11)
        self.assertEqual(schemas.fact_check_schema(["a"])["properties"]["claims"]["maxItems"], 6)

    def test_real_mode_coverage_gap_blocks_publication_and_routes_to_audit(self):
        pipeline = SectorPipeline(Settings(mock=True))
        result = pipeline.run(request())
        plan = pipeline._audit_plan
        result.fact_check["claims"] = [c for c in result.fact_check["claims"]
                                       if c["source_agent"] != "company_researcher:Beta"]
        with patch.object(pipeline.settings, "mock", False):
            pipeline._record_coverage(result, plan)
        self.assertIn("company_researcher:Beta", result.errors["audit_coverage"])
        from vcagents.quality import owner_for
        self.assertEqual(owner_for("audit_coverage"), "audit")

    def test_submission_with_missing_coverage_is_sent_back_to_the_auditor(self):
        pipeline = SectorPipeline(Settings(mock=True))
        pipeline.run(request())
        materials = pipeline._research_materials(pipeline.last_result)
        box = Toolbox()
        data = {"claims": [], "summary": "未覆盖"}
        schema = {"type": "object", "properties": {}, "required": []}
        with patch.object(pipeline.settings, "mock", False), patch.object(box, "citation_errors", return_value=[]):
            problems = pipeline._extra_check("audit", "audit_research", schema, box, data)
        self.assertTrue(any("还需核查" in p for p in problems))
        self.assertEqual(len(materials), 4)


if __name__ == "__main__":
    unittest.main()
