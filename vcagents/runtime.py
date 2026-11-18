"""AgentScope 2.x runtime, retaining the existing provider and evidence contracts.

AgentScope owns the reasoning/acting loop, concurrent tool execution, structured
output and AgentState. The bridge only translates the project's model sessions;
it also lets offline scripted models exercise the real framework.
"""
from __future__ import annotations

import asyncio
import json
import time

from agentscope.agent import Agent as ScopeAgent, ContextConfig, InjectionConfig, ReActConfig
from agentscope.formatter import OpenAIChatFormatter, AnthropicChatFormatter
from agentscope.message import TextBlock, ThinkingBlock, ToolCallBlock, ToolResultState, UserMsg
from agentscope.middleware import MiddlewareBase
from agentscope.model import ChatModelBase, ChatResponse, ChatUsage, FinishedReason
from agentscope.permission import PermissionBehavior, PermissionDecision
from agentscope.tool import FunctionTool, Toolkit, ToolChunk, ToolResponse

from . import schemas
from .agent import Agent, AgentError, _json_from_text
from .governor import RunStopped
from .llm import MockLLM, ClaudeLLM, ClaudeSession, OpenAICompatSession, ToolCall, ToolResult, ToolSpec
from .tools import describe_call

STRUCTURED_TOOL = "GenerateStructuredOutput"


class OpenRouterChatFormatter(OpenAIChatFormatter):
    model_id: str

    async def format(self, msgs):
        formatted = []
        for msg in msgs:
            # AgentScope groups several tool turns into one assistant message.
            # Call IDs can repeat in another reply, so match inside this message.
            by_call = {block.tool_call_id: block.tool_reasoning for block in msg.get_content_blocks("thinking")
                       if getattr(block, "openrouter_model", None) == self.model_id}
            for message in await super().format([msg]):
                if calls := message.get("tool_calls"):
                    message.update(by_call.get(calls[0]["id"], {}))
                formatted.append(message)
        return formatted


class SessionModel(ChatModelBase):
    """Translate AgentScope messages into one isolated provider session."""

    def __init__(self, owner: "AgentScopeAgent"):
        label = getattr(owner.llm, "label", "vc-model")
        super().__init__(credential=None, model=label if isinstance(label, str) else type(owner.llm).__name__,
                         parameters=self.Parameters(), stream=False, max_retries=0, context_size=128000)
        self.owner = owner
        self.formatter = (AnthropicChatFormatter() if isinstance(owner.llm, ClaudeLLM)
                          else OpenRouterChatFormatter(model_id=self.model))
        self.session = None
        self.seen: set[str] = set()
        self.seen_results: set[tuple] = set()
        self.call_ids: dict[str, str] = {}
        self.rounds = 0
        self.truncations = 0

    async def count_tokens(self, messages, tools=None, **kwargs):
        # Conservative estimate for automatic context compaction, not billing.
        return sum(len(json.dumps(m.model_dump(mode="json"), ensure_ascii=False)) for m in messages)

    async def _call_api(self, model_name, messages, tools=None, tool_choice=None, **kwargs):
        specs = []
        structured_name = STRUCTURED_TOOL
        for spec in tools or []:
            fn = spec.get("function", spec)
            name = fn["name"]
            if name in (STRUCTURED_TOOL, "generate_structured_output"):
                structured_name = name
            if isinstance(self.owner.llm, MockLLM) and name in (STRUCTURED_TOOL, "generate_structured_output"):
                name = "submit_result"
            specs.append(ToolSpec(name, fn.get("description", ""), fn.get("parameters", fn.get("input_schema", {})),
                                  strict=fn["name"] in (STRUCTURED_TOOL, "generate_structured_output")))
        if self.session is None:
            system = "\n".join(m.get_text_content() or "" for m in messages if m.role == "system")
            prompt = "\n".join(m.get_text_content() or "" for m in messages if m.role == "user")
            self.session = self.owner.llm.session(system, prompt, specs, self.owner.role.effort)
            self.seen.update(m.id for m in messages)
        elif not isinstance(self.session, (OpenAICompatSession, ClaudeSession)):
            for msg in messages:
                results = []
                # AgentScope appends blocks to one assistant Msg for the entire
                # reply. A seen message ID does not imply its new blocks are read.
                for position, block in enumerate(msg.content):
                    marker = (msg.id, position, block.id)
                    if block.type == "tool_result" and marker not in self.seen_results:
                        self.seen_results.add(marker)
                        content = block.output if isinstance(block.output, str) else "\n".join(
                            b.text for b in block.output if b.type == "text")
                        results.append(ToolResult(self.call_ids.get(block.id, block.id), content,
                                                  block.state != ToolResultState.SUCCESS))
                if results:
                    self.session.add_tool_results(results)
                elif msg.id not in self.seen and msg.role == "user" and msg.get_text_content():
                    self.session.add_user(msg.get_text_content())
                self.seen.add(msg.id)
        if isinstance(self.owner.llm, MockLLM):
            # Context compaction uses a different structured tool/schema on the
            # same model. The offline provider must track that temporary schema
            # and restore the task schema when normal reasoning resumes.
            self.session.tools = {spec.name: spec for spec in specs}
        # AgentScope state is authoritative. Rebuild real provider histories on
        # every call so context compaction, repaired messages and tool offload
        # cannot be undone by an older, independent provider-side history.
        if isinstance(self.session, OpenAICompatSession):
            self.session.messages = await self.formatter.format(messages)
            self.session.tools = [{"type": "function", "function": {
                "name": spec.name, "description": spec.description, "parameters": spec.schema,
                **({"strict": True} if spec.strict else {})}} for spec in specs]
        elif isinstance(self.session, ClaudeSession):
            formatted = await self.formatter.format(messages)
            if formatted and formatted[0]["role"] == "system":
                self.session.system = formatted[0]["content"]
                formatted = formatted[1:]
            self.session.messages = formatted
            self.session.tools = [self.session._tool_param(spec) for spec in specs]
        before = (self.session.input_tokens, self.session.output_tokens)
        before_messages = len(self.session.messages) if isinstance(self.session, ClaudeSession) else 0
        started = time.monotonic()
        governor = getattr(self.owner.llm, "governor", None)
        if governor is not None:
            governor.check()  # 总时限、取消或费用上限：不再发起新的模型请求
        self.owner._log("log", "等待 AgentScope 模型响应")
        turn = await asyncio.to_thread(self.session.send)
        self.rounds += 1
        for note in turn.notes:
            self.owner._log("log", note)
        self.truncations = self.truncations + 1 if turn.truncated else 0
        if self.truncations >= self.owner.max_truncations:
            raise AgentError("模型连续三次截断，停止该角色")
        content = []
        if isinstance(self.session, ClaudeSession):
            # Preserve provider-signed thinking blocks across tool turns; the
            # native formatter can then send them back without losing signatures.
            for message in self.session.messages[before_messages:]:
                if message.get("role") != "assistant":
                    continue
                for block in message.get("content", []):
                    if getattr(block, "type", "") == "thinking":
                        content.append(ThinkingBlock(thinking=block.thinking, signature=block.signature))
                    elif getattr(block, "type", "") == "redacted_thinking":
                        content.append(ThinkingBlock(thinking="", redacted_thinking_data=block.data))
        calls = turn.tool_calls
        if not calls and (data := _json_from_text(turn.text)) is not None:
            calls = [ToolCall(f"json-{len(self.seen)}", structured_name, data)]
        for index, call in enumerate(calls):
            name = structured_name if call.name == "submit_result" else call.name
            data = call.input
            if name == STRUCTURED_TOOL and data is not None:
                data = schemas.normalize(self.owner.schema, data)
                if schemas.validate(self.owner.schema, data):
                    self.owner.format_failures += 1
            call_id = call.id if call.id not in self.call_ids else f"{call.id}-round{self.rounds}"
            self.call_ids[call_id] = call.id
            if index == 0 and turn.tool_reasoning:
                content.append(ThinkingBlock(thinking="", openrouter_model=self.model, tool_call_id=call_id,
                                             tool_reasoning=turn.tool_reasoning))
            content.append(ToolCallBlock(id=call_id, name=name, input=json.dumps(data, ensure_ascii=False)))
        if turn.text and not calls:
            content.append(TextBlock(text=turn.text))
        return ChatResponse(content=content, is_last=True,
                            finished_reason=FinishedReason.COMPLETED,
                            usage=ChatUsage(input_tokens=self.session.input_tokens - before[0],
                                            output_tokens=self.session.output_tokens - before[1],
                                            time=time.monotonic() - started))


class SubmissionGuard(MiddlewareBase):
    """Keep semantic checks inside the native structured-output tool lifecycle."""

    def __init__(self, owner: "AgentScopeAgent"):
        self.owner = owner

    async def on_acting(self, agent, input_kwargs, next_handler):
        call = input_kwargs["tool_call"]
        if call.name == STRUCTURED_TOOL:
            data = json.loads(call.input)
            errors = self.owner._submission_errors(self.owner.schema, data)
            if errors:
                self.owner.format_failures += 1
                self.owner._log("log", "提交需要修正：" + "；".join(errors[:3]))
                # A middleware must emit the completed ToolResponse, otherwise
                # AgentScope retries the same action without consulting the model.
                yield ToolResponse(content=[TextBlock(text="请修正后重新提交：" + "；".join(errors))],
                                   state=ToolResultState.ERROR)
                return
        async for item in next_handler(**input_kwargs):
            yield item


class AgentScopeAgent(Agent):
    """Compatibility-facing runner; all production pipeline roles use AgentScope."""

    def __init__(self, *args, persistent: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.persistent = persistent
        self.scope_agent = None
        self.reply_count = 0

    def _loop(self, task: str) -> dict:
        return asyncio.run(self._reply(task))

    async def _reply(self, task: str) -> dict:
        self.input_tokens = self.output_tokens = 0
        self.cost_usd = None
        model = SessionModel(self)
        tools = []
        for spec in self.toolbox.specs(self.role.tools):
            async def execute(_name=spec.name, **arguments):
                call = ToolCall(f"tool-{time.monotonic_ns()}", _name, arguments)
                self._log("log", describe_call(call))
                result = await asyncio.to_thread(self.toolbox.execute, call)
                return ToolChunk(content=[TextBlock(text=result.content)],
                                 state=ToolResultState.ERROR if result.is_error else ToolResultState.SUCCESS)
            tools.append(FunctionTool(execute, name=spec.name, description=spec.description,
                                      input_schema=spec.schema, is_concurrency_safe=True, is_read_only=True,
                                      permission=PermissionDecision(behavior=PermissionBehavior.ALLOW,
                                          message="本工具仅执行本次研究授权的来源检索与读取")))
        if not self.persistent or self.scope_agent is None:
            self.scope_agent = ScopeAgent(
                name=self.id, system_prompt=self.system.replace("submit_result", STRUCTURED_TOOL), model=model,
                toolkit=Toolkit(tools=tools), middlewares=[SubmissionGuard(self)],
                react_config=ReActConfig(max_iters=self.max_steps + self.max_bonus_steps, structured_output_grace_iters=1),
                context_config=ContextConfig(tool_result_limit=50000),
                injection_config=InjectionConfig(timezone="Asia/Shanghai"),
            )
        else:
            self.scope_agent.model = model
            self.scope_agent.react_config = ReActConfig(max_iters=self.max_steps + self.max_bonus_steps, structured_output_grace_iters=1)
        # A new reply is a new bounded task inside the same AgentState.
        self._content_failures = self._bonus_steps = self._missing_evidence = 0
        self._last_valid = None
        self.unresolved_issues = []
        self.reply_count += 1
        try:
            msg = await self.scope_agent.reply(UserMsg(name="user", content=task), structured_schema=self.schema)
            if msg.structured_output is not None:
                return msg.structured_output
            if self._last_valid is not None:
                self.unresolved_issues = self.unresolved_issues or ["步数用完，内容检查未完成"]
                return self._last_valid
            detail = str(msg.error) if msg.error else str(msg.finished_reason)
            raise AgentError(f"AgentScope 未提交合格的结构化结果：{detail}")
        except ExceptionGroup as exc:
            def causes(group):
                for error in group.exceptions:
                    if isinstance(error, ExceptionGroup):
                        yield from causes(error)
                    else:
                        yield error
            errors = list(causes(exc))
            for error in errors:
                if isinstance(error, RunStopped):
                    raise error from exc
            raise AgentError("AgentScope 执行失败：" + "；".join(f"{type(e).__name__}: {e}" for e in errors)) from exc
        finally:
            if model.session is not None:
                self.input_tokens = model.session.input_tokens
                self.output_tokens = model.session.output_tokens
                self.cost_usd = model.session.cost_usd
