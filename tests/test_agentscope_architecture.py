"""Offline contract tests for the real AgentScope runtime and parallel DAG."""
import json
import asyncio
from copy import deepcopy
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

from agentscope.agent import Agent as ScopeAgent
from openai.types.chat import ChatCompletion

from vcagents import CompanyInput, ExpandedSectorPipeline as SectorPipeline, SectorRequest, Settings
from vcagents.llm import ChatSession, LLM, ClaudeLLM, ClaudeSession, OpenAICompatLLM, OpenAICompatSession, ToolCall, ToolSpec, Turn
from vcagents.roles import Role
from vcagents.runtime import AgentScopeAgent, SessionModel
from agentscope.message import AssistantMsg, SystemMsg, UserMsg
from vcagents.scheduler import DAGScheduler, TaskNode
from vcagents.tools import DocumentStore, Toolbox
from vcagents.citations import SourceRegistry


class SequenceSession(ChatSession):
    def __init__(self, turns):
        super().__init__()
        self.turns = list(turns)
        self.received = []

    def send(self):
        return self.turns.pop(0)

    def add_tool_results(self, results, note=""):
        self.received.extend(results)

    def add_user(self, text):
        pass


class SequenceModel(LLM):
    def __init__(self, turns):
        self.last = SequenceSession(turns)

    def session(self, system, prompt, tools, effort="medium"):
        self.specs = tools
        return self.last


SMALL_SCHEMA = {"type": "object", "properties": {"answer": {"type": "string"}},
                "required": ["answer"], "additionalProperties": False}


class AgentScopeRuntimeTest(unittest.TestCase):
    def test_openrouter_replays_signed_tool_turns_in_their_original_reply(self):
        llm = OpenAICompatLLM("test", "google/gemini-4-carbon", "https://example.com")
        self.addCleanup(llm.client.close)
        docs = DocumentStore()
        docs.add_text("BP.txt", "原始资料")
        agent = AgentScopeAgent(Role("research", "研究", "", SMALL_SCHEMA, ("read_document",)), llm,
                                Toolbox(docs), lambda e: None, system="读资料", persistent=True)
        details = [{"type": "reasoning.encrypted", "data": "opaque-signature", "id": "read", "index": 0}]
        read = {"name": "read_document", "arguments": '{"name":"BP.txt","start_page":1,"end_page":1}'}
        submit = {"name": "GenerateStructuredOutput", "arguments": '{"answer":"ok"}'}
        replies = iter([
            ([("read", read), ("parallel", read)], {"reasoning_details": details, "reasoning": "unused fallback"}),
            ([("done", submit)], {"reasoning": "first answer"}),
            ([("read", read)], {"reasoning_content": "second read"}),
            ([("done", submit)], {}),
        ])
        history = []

        def complete(**request):
            history.append(deepcopy([m for m in request["messages"] if m.get("tool_calls")]))
            calls, reasoning = next(replies)
            message = {"role": "assistant", "tool_calls": [{"type": "function", "id": key, "function": fn}
                                                          for key, fn in calls], **reasoning}
            return ChatCompletion.model_validate({"id": "test", "created": 0, "model": llm.model,
                "object": "chat.completion", "choices": [{"index": 0, "finish_reason": "tool_calls", "message": message}]})

        with patch.object(llm.client.chat.completions, "create", side_effect=complete):
            self.assertEqual(agent.run("初读"), {"answer": "ok"})
            self.assertEqual(agent.run("复核"), {"answer": "ok"})
        self.assertEqual([call["id"] for call in history[1][0]["tool_calls"]], ["read", "parallel"])
        self.assertEqual(history[1][0]["reasoning_details"], details)
        self.assertNotIn("reasoning", history[1][0])
        self.assertEqual(history[3][0]["reasoning_details"], details)
        self.assertEqual(history[3][1]["reasoning"], "first answer")
        self.assertEqual(history[3][2]["reasoning_content"], "second read")

    def test_plain_json_compaction_keeps_the_framework_helper_tool_name(self):
        llm = SequenceModel([Turn(text='{"answer":"summary"}')])
        owner = AgentScopeAgent(Role("bull", "test", "", SMALL_SCHEMA), llm, Toolbox(), lambda e: None, system="test")
        model = SessionModel(owner)
        response = asyncio.run(model._call_api("test", [UserMsg(name="user", content="summarize")],
            [{"type": "function", "function": {"name": "generate_structured_output", "parameters": SMALL_SCHEMA}}]))
        self.assertEqual(response.content[0].name, "generate_structured_output")

    def test_signed_anthropic_reasoning_survives_provider_translation(self):
        llm = ClaudeLLM("test", "test-model", server_search=False)
        self.addCleanup(llm.client.close)
        agent = AgentScopeAgent(Role("chair", "test", "", SMALL_SCHEMA), llm, Toolbox(), lambda e: None, system="test")
        model = SessionModel(agent)
        response = SimpleNamespace(stop_reason="tool_use", usage=SimpleNamespace(input_tokens=1, output_tokens=1,
            cache_read_input_tokens=0, cache_creation_input_tokens=0), content=[
                SimpleNamespace(type="thinking", thinking="opaque provider reasoning", signature="signed-block"),
                SimpleNamespace(type="tool_use", id="done", name="GenerateStructuredOutput", input={"answer": "ok"})])

        async def exercise():
            turn = await model._call_api("test", [SystemMsg(name="system", content="rules"), UserMsg(name="user", content="test")],
                [{"type": "function", "function": {"name": "GenerateStructuredOutput", "parameters": SMALL_SCHEMA}}])
            return await model.formatter.format([AssistantMsg(name="chair", content=turn.content)])

        with patch.object(ClaudeSession, "_create", return_value=response):
            messages = asyncio.run(exercise())
        self.assertEqual(messages[0]["content"][0]["signature"], "signed-block")

    def test_provider_receives_current_native_context_after_compaction(self):
        llm = OpenAICompatLLM("test", "test-model", "https://example.com")
        self.addCleanup(llm.client.close)
        agent = AgentScopeAgent(Role("bull", "test", "", SMALL_SCHEMA), llm, Toolbox(), lambda e: None, system="test")
        model = SessionModel(agent)
        tools = [{"type": "function", "function": {"name": "GenerateStructuredOutput", "parameters": SMALL_SCHEMA}}]

        async def exercise():
            await model._call_api("test", [SystemMsg(name="system", content="rules"), UserMsg(name="user", content="OLD_CONTEXT")], tools)
            await model._call_api("test", [SystemMsg(name="system", content="rules"), UserMsg(name="user", content="COMPACTED_CONTEXT")], tools)

        with patch.object(OpenAICompatSession, "send", return_value=Turn(tool_calls=[
                ToolCall("done", "GenerateStructuredOutput", {"answer": "ok"})])):
            asyncio.run(exercise())
        serialized = json.dumps(model.session.messages)
        self.assertIn("COMPACTED_CONTEXT", serialized)
        self.assertNotIn("OLD_CONTEXT", serialized)
    def test_same_bp_filename_from_two_companies_cannot_share_a_quote(self):
        registry = SourceRegistry()
        alpha, beta = DocumentStore(), DocumentStore()
        alpha.add_text("BP.txt", "Alpha 收入100万美元。")
        beta.add_text("BP.txt", "Beta 收入200万美元。")
        a = Toolbox(alpha, registry=registry, document_scope="Alpha")
        b = Toolbox(beta, registry=registry, document_scope="Beta")
        a.execute(ToolCall("a", "read_document", {"name": "BP.txt", "start_page": 1, "end_page": 1}))
        output = b.execute(ToolCall("b", "read_document", {"name": "BP.txt", "start_page": 1, "end_page": 1})).content
        self.assertIn("D2:s1", output)
        self.assertIn("Beta 收入200万美元", output)
        self.assertNotIn("Alpha 收入", output)
        self.assertFalse(a._entry_is_source("《BP.txt》第1页", registry.get("D2")))
        self.assertTrue(b._entry_is_source("《BP.txt》第1页", registry.get("D2")))
        combined = DocumentStore()
        combined.docs = {"Alpha/BP.txt": alpha.docs["BP.txt"], "Beta/BP.txt": beta.docs["BP.txt"]}
        reread = Toolbox(combined, registry=registry).execute(ToolCall("c", "read_document", {
            "name": "Beta/BP.txt", "start_page": 1, "end_page": 1})).content
        self.assertIn("D2:s1", reread, "核查员读取公司限定名称时应命中同一来源")
    def test_native_loop_reads_tool_and_corrects_an_unsupported_submission(self):
        docs = DocumentStore()
        docs.add_text("BP.txt", "收入必须核对原文。")
        llm = SequenceModel([
            Turn(tool_calls=[ToolCall("read", "read_document", {"name": "BP.txt", "start_page": 1, "end_page": 1})]),
            Turn(tool_calls=[ToolCall("bad", "submit_result", {"answer": "无依据"})]),
            Turn(tool_calls=[ToolCall("fixed", "submit_result", {"answer": "有依据"})]),
        ])
        role = Role("bull", "审阅", "", SMALL_SCHEMA, ("read_document",))
        agent = AgentScopeAgent(role, llm, Toolbox(docs), lambda e: None, system="核对资料",
                                extra_check=lambda data: ["先改正无依据的结论"] if data["answer"] == "无依据" else [])
        self.assertEqual(agent.run("核对"), {"answer": "有依据"})
        self.assertIsInstance(agent.scope_agent, ScopeAgent)
        self.assertTrue(any("D1:s1" in r.content for r in llm.last.received if r.call_id == "read"))
        self.assertTrue(any(r.is_error and "先改正" in r.content for r in llm.last.received))
        self.assertEqual(sum(r.call_id == "read" for r in llm.last.received), 1)

    def test_agent_scope_executes_independent_tools_concurrently(self):
        rendezvous = threading.Barrier(2, timeout=5)

        class ConcurrentTools(Toolbox):
            def specs(self, names):
                return [ToolSpec("web_search", "离线并行工具", {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]})]

            def execute(self, call):
                from vcagents.llm import ToolResult
                rendezvous.wait()
                return ToolResult(call.id, "完成 " + call.input["query"])

        llm = SequenceModel([
            Turn(tool_calls=[ToolCall("a", "web_search", {"query": "A"}), ToolCall("b", "web_search", {"query": "B"})]),
            Turn(tool_calls=[ToolCall("done", "submit_result", {"answer": "完成"})]),
        ])
        agent = AgentScopeAgent(Role("bull", "并行", "", SMALL_SCHEMA, ("web_search",)), llm, ConcurrentTools(),
                                lambda e: None, system="并行检索")
        self.assertEqual(agent.run("并行"), {"answer": "完成"})
        self.assertEqual({r.content for r in llm.last.received}, {"完成 A", "完成 B"})

    def test_research_and_analysis_overlap_across_companies(self):
        ready = threading.Event()
        pipeline = SectorPipeline(Settings(mock=True, max_workers=12))
        original = pipeline._run
        observed = []

        def run(role, task, toolbox, agent_id, name, schema=None):
            if agent_id == "company_researcher:Beta:lane3":
                observed.append(ready.wait(5))
            if agent_id == "company_analyst:Alpha":
                ready.set()
            return original(role, task, toolbox, agent_id, name, schema)

        pipeline._run = run
        result = pipeline.run(SectorRequest("aigc-content", [CompanyInput(n) for n in ("Alpha", "Beta", "Gamma")],
                                            core_idea="验证客户付费是否能支持持续价值捕获"))
        self.assertEqual(observed, [True], "Alpha 的分析应在 Beta 的研究结束前开始")
        self.assertFalse(result.errors)
        self.assertEqual(len([t for t in result.task_trace if t["task"].startswith("company_researcher:")]), 9)
        self.assertIn("working_hypothesis", result.research_plan)
        for trace in result.task_trace:
            if trace["task"].startswith("company_analyst:"):
                self.assertIn("sector_researcher", trace["depends_on"])

    def test_scheduler_blocks_only_dependents_of_a_failed_task(self):
        called = []

        def fail():
            raise ValueError("缺少原文")

        scheduler = DAGScheduler(2)
        result = scheduler.run([TaskNode("bad", fail), TaskNode("dependent", lambda: called.append("dependent"), ("bad",)),
                                TaskNode("independent", lambda: called.append("independent"))])
        self.assertEqual(called, ["independent"])
        self.assertIn("缺少原文", result["bad"].error)
        self.assertIn("依赖失败", result["dependent"].error)

    def test_scheduler_rejects_cycles_and_duplicate_ids(self):
        scheduler = DAGScheduler(2)
        with self.assertRaisesRegex(ValueError, "成环"):
            scheduler.run([TaskNode("a", lambda: None, ("b",)), TaskNode("b", lambda: None, ("a",))])
        with self.assertRaisesRegex(ValueError, "重复"):
            scheduler.run([TaskNode("a", lambda: None), TaskNode("a", lambda: None)])

    def test_parallel_limit_is_enforced(self):
        lock = threading.Lock()
        rendezvous = threading.Barrier(2, timeout=5)
        active = peak = 0

        def task():
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            rendezvous.wait()
            with lock:
                active -= 1

        outcomes = DAGScheduler(2).run([TaskNode(str(i), task) for i in range(8)])
        self.assertTrue(all(not value.error for value in outcomes.values()))
        self.assertEqual(peak, 2)


if __name__ == "__main__":
    unittest.main()
