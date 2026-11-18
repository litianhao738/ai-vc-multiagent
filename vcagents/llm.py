"""大模型接入层：把 OpenRouter（OpenAI 兼容接口）和 Anthropic 直连统一成同一个会话接口，智能体只依赖这里的抽象。"""
from __future__ import annotations

import json
import random
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from .config import OPENROUTER_BASE_URL, Settings


@dataclass
class ToolSpec:
    name: str
    description: str
    schema: dict
    strict: bool = False


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict | None  # None 表示模型给出的参数不是合法 JSON
    error: str = ""


@dataclass
class ToolResult:
    call_id: str
    content: str
    is_error: bool = False


@dataclass
class Turn:
    """模型的一轮回复。"""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)  # 服务端工具（如 Claude 内置搜索）的执行记录
    truncated: bool = False
    tool_reasoning: dict = field(default_factory=dict)  # 随本次工具调用原样回传给同一模型


class LLMError(RuntimeError):
    """模型调用失败（鉴权、限流、拒答等），信息可以直接展示给用户。"""


class ChatSession(ABC):
    """一个智能体的一次对话。智能体循环只调用这三个方法。"""

    def __init__(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost_usd: float | None = None
        self.cost_complete = True

    @abstractmethod
    def send(self) -> Turn: ...

    @abstractmethod
    def add_tool_results(self, results: list[ToolResult], note: str = "") -> None: ...

    @abstractmethod
    def add_user(self, text: str) -> None: ...


class LLM(ABC):
    label = ""
    provider = ""
    governor = None  # RunGovernor；为空时沿用 SDK 自带的重试与超时

    @abstractmethod
    def session(self, system: str, prompt: str, tools: list[ToolSpec], effort: str = "medium") -> ChatSession: ...


TRUNCATED_HINT = "（你上一次的输出过长被截断了，请精简内容后重新输出。）"


def governed_request(owner, sdk, send, *, retries: int, timeout: float):
    """Send through the run governor when one is attached.

    `owner` has `.client`, `.provider` and `.governor`; `send(client)` performs the
    request. Governed requests disable SDK retries so that every retry is counted
    against the run-wide budget and no attempt outlives the run deadline.
    """
    governor = owner.governor
    if governor is None:
        return send(owner.client)

    def retryable(exc):
        status = getattr(exc, "status_code", None)
        return isinstance(exc, sdk.APIConnectionError) or status in (408, 409, 429) or (status or 0) >= 500

    def retry_after(exc):
        headers = getattr(getattr(exc, "response", None), "headers", None) or {}
        try:
            seconds = float(headers.get("retry-after"))
        except (TypeError, ValueError):
            return None
        return seconds if seconds > 0 else None

    return governor.call(owner.provider, lambda limit: send(owner.client.with_options(max_retries=0, timeout=limit)),
                         retries=retries, timeout=timeout, retryable=retryable, retry_after=retry_after)


def _vendor(model: str) -> str:
    """OpenRouter 模型名的前缀就是上游供应商，例如 anthropic/claude-… → anthropic。"""
    return model.split("/", 1)[0] if "/" in model else "openai-compatible"


# ---------------------------------------------------------------- Claude

CLAUDE_RETRIES = 4
CLAUDE_TIMEOUT = 600


class ClaudeLLM(LLM):
    provider = "anthropic-direct"

    def __init__(self, api_key: str, model: str, base_url: str = "", server_search: bool = True):
        import anthropic

        self.anthropic = anthropic
        # api_key 为空时，SDK 会从环境变量或 `ant auth login` 的登录信息里读取凭证
        self.client = anthropic.Anthropic(api_key=api_key or None, base_url=base_url or None, max_retries=CLAUDE_RETRIES)
        self.model = model
        self.server_search = server_search
        self.label = f"Claude · {model}"

    def session(self, system, prompt, tools, effort="medium"):
        return ClaudeSession(self, system, prompt, tools, effort)


class ClaudeSession(ChatSession):
    max_pause_continuations = 5

    def __init__(self, llm: ClaudeLLM, system: str, prompt: str, tools: list[ToolSpec], effort: str):
        super().__init__()
        self.llm = llm
        self.system = system
        self.effort = effort
        self.messages: list[dict] = [{"role": "user", "content": prompt}]
        self.tools = [self._tool_param(t) for t in tools]

    def _tool_param(self, tool: ToolSpec) -> dict:
        # 开启内置搜索时，用 Claude 服务端的联网搜索/抓取替换本地实现
        if self.llm.server_search and tool.name == "web_search":
            return {"type": "web_search_20260209", "name": "web_search", "max_uses": 8}
        if self.llm.server_search and tool.name == "fetch_url":
            return {"type": "web_fetch_20260209", "name": "web_fetch", "max_uses": 5}
        param = {"name": tool.name, "description": tool.description, "input_schema": tool.schema}
        if tool.strict:
            param["strict"] = True
        return param

    def _create(self):
        a = self.llm.anthropic

        def send(client):
            return client.beta.messages.create(
                model=self.llm.model,
                max_tokens=16000,
                system=self.system,
                messages=self.messages,
                tools=self.tools,
                thinking={"type": "adaptive"},
                output_config={"effort": self.effort},
                cache_control={"type": "ephemeral"},
                # 安全分类器误拒时，由服务端自动换用推荐的备用模型重跑
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )

        try:
            return governed_request(self.llm, a, send, retries=CLAUDE_RETRIES, timeout=CLAUDE_TIMEOUT)
        except a.AuthenticationError:
            raise LLMError("Claude API Key 无效") from None
        except a.PermissionDeniedError as e:
            raise LLMError(f"Claude API 无权限：{e.message}") from None
        except a.NotFoundError:
            raise LLMError(f"模型不存在：{self.llm.model}") from None
        except a.RateLimitError:
            raise LLMError("Claude API 限流，请稍后重试或调低并发数") from None
        except a.BadRequestError as e:
            raise LLMError(f"Claude API 请求参数错误：{e.message}") from None
        except a.APIStatusError as e:
            raise LLMError(f"Claude API 错误 {e.status_code}：{e.message}") from None
        except a.APIConnectionError:
            raise LLMError("无法连接 Claude API，请检查网络或代理") from None

    def send(self) -> Turn:
        turn = Turn()
        for _ in range(self.max_pause_continuations + 1):
            resp = self._create()
            usage = resp.usage
            self.input_tokens += (usage.input_tokens or 0) + (usage.cache_read_input_tokens or 0) + (
                usage.cache_creation_input_tokens or 0
            )
            self.output_tokens += usage.output_tokens or 0

            if resp.stop_reason == "refusal":
                raise LLMError("模型基于安全策略拒绝了该请求")
            if resp.stop_reason == "max_tokens":
                # 被截断的回复里可能有不完整的工具调用，丢弃这次输出，要求精简后重来
                self.add_user(TRUNCATED_HINT)
                turn.truncated = True
                return turn

            self.messages.append({"role": "assistant", "content": resp.content})
            for block in resp.content:
                if block.type == "text":
                    turn.text += block.text
                elif block.type == "tool_use":
                    turn.tool_calls.append(
                        ToolCall(block.id, block.name, block.input if isinstance(block.input, dict) else None)
                    )
                elif block.type == "server_tool_use":
                    turn.notes.append(_describe_server_tool(block.name, block.input or {}))
            # pause_turn：服务端工具跑到单次上限，把已有内容发回去让模型接着做
            if resp.stop_reason != "pause_turn":
                return turn
        return turn

    def add_tool_results(self, results, note=""):
        content: list[dict] = [
            {"type": "tool_result", "tool_use_id": r.call_id, "content": r.content, "is_error": r.is_error}
            for r in results
        ]
        if note:
            content.append({"type": "text", "text": note})
        self.messages.append({"role": "user", "content": content})

    def add_user(self, text):
        self.messages.append({"role": "user", "content": text})


def _describe_server_tool(name: str, args: dict) -> str:
    if name == "web_search":
        return f"联网搜索：{args.get('query', '')}"
    if name == "web_fetch":
        return f"读取网页：{args.get('url', '')}"
    return name


# ---------------------------------------------------------------- OpenAI 兼容（OpenRouter）

REQUEST_TIMEOUT = 600  # 非流式长输出：真实运行中分析师一次响应约 220 秒，120 秒上限必然超时
REQUEST_RETRIES = 2


class OpenAICompatLLM(LLM):
    def __init__(self, api_key: str, model: str, base_url: str, headers: dict | None = None, max_tokens: int = 8000,
                 *, timeout: float = REQUEST_TIMEOUT, retries: int = REQUEST_RETRIES, request_options: dict | None = None):
        import openai

        self.openai = openai
        self.client = openai.OpenAI(
            api_key=api_key, base_url=base_url or None, default_headers=headers, max_retries=retries,
            timeout=timeout
        )
        self.model = model
        self.max_tokens = max_tokens
        self.timeout, self.retries = timeout, retries
        self.request_options = dict(request_options or {})
        self.label = model
        self.provider = _vendor(model)

    def session(self, system, prompt, tools, effort="medium"):
        return OpenAICompatSession(self, system, prompt, tools)


class OpenAICompatSession(ChatSession):
    def __init__(self, llm: OpenAICompatLLM, system: str, prompt: str, tools: list[ToolSpec]):
        super().__init__()
        self.llm = llm
        self.messages: list[dict] = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
        self.tools = [
            {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.schema,
                                                **({"strict": True} if t.strict else {})}}
            for t in tools
        ]

    def send(self) -> Turn:
        o = self.llm.openai
        try:
            resp = governed_request(self.llm, o, lambda client: client.chat.completions.create(
                model=self.llm.model, messages=self.messages, tools=self.tools, max_tokens=self.llm.max_tokens,
                **self.llm.request_options
            ), retries=self.llm.retries, timeout=self.llm.timeout)
        except o.AuthenticationError:
            raise LLMError("API Key 无效") from None
        except o.NotFoundError:
            raise LLMError(f"模型不存在：{self.llm.model}") from None
        except o.RateLimitError:
            raise LLMError("接口限流或余额不足，请稍后重试") from None
        except o.APIStatusError as e:
            raise LLMError(f"接口错误 {e.status_code}：{e.message}") from None
        except o.APITimeoutError:  # 是 APIConnectionError 的子类，必须先捕获
            raise LLMError(f"模型响应超时（超过 {self.llm.timeout:g} 秒），网络未必有问题；可减少输出或换更快的模型") from None
        except o.APIConnectionError as e:
            raise LLMError(f"无法连接模型接口，请检查 Base URL、代理和网络（{e.__cause__ or e}）") from None

        if resp.usage:
            self.input_tokens += resp.usage.prompt_tokens or 0
            self.output_tokens += resp.usage.completion_tokens or 0
            cost = (getattr(resp.usage, "model_extra", None) or {}).get("cost")
            if isinstance(cost, (int, float)) and self.llm.governor is not None:
                self.llm.governor.add_cost(cost)  # 费用上限按每次响应累计，不等阶段结束
            if isinstance(cost, (int, float)) and self.cost_complete:
                self.cost_usd = (self.cost_usd or 0.0) + cost
            else:
                self.cost_complete = False
                self.cost_usd = None
        else:
            self.cost_complete = False
            self.cost_usd = None
        choice = resp.choices[0]
        msg = choice.message
        if choice.finish_reason == "length":
            self.add_user(TRUNCATED_HINT)
            return Turn(truncated=True)

        assistant: dict = {"role": "assistant", "content": msg.content or ""}
        calls = []
        tool_reasoning = {}
        if msg.tool_calls:
            # OpenRouter can require a signed reasoning_details array on the next
            # tool turn. Plain-text reasoning is used only when no array is sent.
            for key in ("reasoning_details", "reasoning", "reasoning_content"):
                value = getattr(msg, key, None)
                if value and isinstance(value, (list, str)):
                    tool_reasoning[key] = value
                    break
            assistant.update(tool_reasoning)
            assistant["tool_calls"] = [
                {"id": c.id, "type": "function", "function": {"name": c.function.name, "arguments": c.function.arguments}}
                for c in msg.tool_calls
            ]
            calls = [_parse_call(c.id, c.function.name, c.function.arguments) for c in msg.tool_calls]
        self.messages.append(assistant)
        return Turn(text=msg.content or "", tool_calls=calls, tool_reasoning=tool_reasoning)

    def add_tool_results(self, results, note=""):
        for r in results:
            content = f"错误：{r.content}" if r.is_error else r.content
            self.messages.append({"role": "tool", "tool_call_id": r.call_id, "content": content})
        if note:
            self.add_user(note)

    def add_user(self, text):
        self.messages.append({"role": "user", "content": text})


def _parse_call(call_id: str, name: str, raw: str | None) -> ToolCall:
    try:
        value = json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        return ToolCall(call_id, name, None, f"JSON 格式错误：{exc.msg}（位置 {exc.pos}）")
    if not isinstance(value, dict):
        return ToolCall(call_id, name, None, "工具参数必须是 JSON 对象")
    return ToolCall(call_id, name, value)


# ---------------------------------------------------------------- 演示模式

class MockLLM(LLM):
    """不调用任何模型：先查一次上传资料，再按输出 schema 生成占位结果。用于跑通流程和测试。"""

    label = "演示模式（无模型调用）"

    def session(self, system, prompt, tools, effort="medium"):
        return MockSession(system, prompt, tools)


class MockSession(ChatSession):
    def __init__(self, system: str, prompt: str, tools: list[ToolSpec]):
        super().__init__()
        self.tools = {t.name: t for t in tools}
        self.rng = random.Random(system + prompt[:200])  # 同样的输入得到同样的结果
        self.steps = 0

    def send(self) -> Turn:
        self.steps += 1
        if self.steps == 1 and "search_documents" in self.tools:
            return Turn(tool_calls=[ToolCall("mock-1", "search_documents", {"query": "收入 融资 团队"})])
        submit = self.tools["submit_result"]
        data = _fake(submit.schema, self.rng)
        if "top_pick" in data and "recommended" in data and "exit_scenarios" in data:
            data["recommended"] = [data["top_pick"]]
        if "decision" in data:
            data["decision"]["recommended"] = [data["decision"]["top_pick"]]
        return Turn(tool_calls=[ToolCall(f"mock-{self.steps}", submit.name, data)])

    def add_tool_results(self, results, note=""):
        pass

    def add_user(self, text):
        pass


def _fake(schema: dict, rng: random.Random, key: str = ""):
    if "enum" in schema:
        if key == "next_action" and "finish" in schema["enum"]:
            return "finish"  # 演示默认沿原流程成稿；补查由脚本用例显式触发
        if key == "listing_status":
            return "未上市"
        if key == "verdict" and "属实" in schema["enum"]:
            return "属实"
        if key == "status" and "已修正" in schema["enum"]:
            return "已修正"
        if key == "verdict" and "有依据" in schema["enum"]:
            return "有依据"
        return rng.choice(schema["enum"])
    kind = schema.get("type")
    if isinstance(kind, list):
        kind = next(t for t in kind if t != "null")
    if kind == "object":
        return {k: _fake(v, rng, k) for k, v in schema["properties"].items()}
    if kind == "array":
        items = schema.get("items", {})
        if key in ("question_updates", "follow_up_tasks"):
            return []
        if key == "reasoning_checks":
            targets = items["properties"]["target_id"]["enum"]
            return [{**_fake(items, rng, key), "target_id": target} for target in targets]
        if key == "repair_checks":
            ids = items["properties"]["claim_id"]["enum"]
            return [{**_fake(items, rng, key), "claim_id": claim_id} for claim_id in ids]
        if key == "long_term_edge":
            return [{**_fake(items, rng, key), "dimension": d} for d in items["properties"]["dimension"]["enum"]]
        if key == "question_answers":
            ids = items["properties"]["question_id"].get("enum", [])[:schema.get("maxItems", 0)]
            return [{**_fake(items, rng, key), "question_id": qid} for qid in ids]
        if key == "exit_scenarios":
            scenarios = items["properties"]["scenario"]["enum"]
            rows = [{**_fake(items, rng, key), "scenario": scenario} for scenario in scenarios]
            values = sorted((row["exit_valuation_musd"] for row in rows), reverse=True)  # 乐观 ≥ 基线 ≥ 悲观
            return [{**row, "exit_valuation_musd": value} for row, value in zip(rows, values)]
        count = schema.get("minItems", 0) if key == "candidates" else rng.randint(2, 4)
        if "maxItems" in schema:
            count = min(count, schema["maxItems"])
        count = max(count, schema.get("minItems", 0))
        if schema.get("uniqueItems") and "enum" in items:
            return rng.sample(items["enum"], min(count, len(items["enum"])))
        if items.get("type") == "object" or "enum" in items:
            return [_fake(items, rng, key) for _ in range(count)]
        return [f"（演示）{key} 第 {i} 条" for i in range(1, count + 1)]
    if kind == "integer":
        return rng.randint(4, 9)
    if kind == "number":
        return float(rng.randint(4, 9))
    return f"（演示数据）“{key}”字段的占位内容。接入真实模型后，这里是基于检索资料的分析。"


# ---------------------------------------------------------------- 按角色分配模型

class LLMRouter:
    """按角色返回配置好的模型客户端。"""

    def __init__(self, settings: Settings, governor=None):
        self.settings = settings
        self.governor = governor
        self._clients: dict[tuple[str, str], LLM] = {}
        self._lock = threading.Lock()

    def for_role(self, role_key: str) -> LLM:
        if self.settings.mock:
            llm = MockLLM()
            llm.governor = self.governor
            return llm
        spec = self.settings.role_models[role_key]
        key = (spec, role_key)
        with self._lock:
            if key not in self._clients:
                self._clients[key] = self._build(spec, role_key)
                self._clients[key].governor = self.governor
            return self._clients[key]

    def _build(self, spec: str, role_key: str) -> LLM:
        if role_key == "rejection_review":
            if not self.settings.openrouter_api_key:
                raise LLMError("未设置 OPENROUTER_API_KEY")
            reasoning = ({"effort": "minimal"} if spec.split(":", 1)[0] == "google/gemini-3.1-flash-lite"
                         else {"enabled": False})
            return OpenAICompatLLM(self.settings.openrouter_api_key, spec, OPENROUTER_BASE_URL,
                headers={"X-Title": "AI VC Multi-Agent"}, max_tokens=1200, timeout=45, retries=0,
                request_options={"extra_body": {"reasoning": reasoning}, "tool_choice": "required"})
        if spec.startswith("anthropic:"):
            # 直连 Anthropic；这里的角色不需要联网，所以不开内置搜索
            return ClaudeLLM(self.settings.anthropic_api_key, spec.split(":", 1)[1], server_search=False)
        if not self.settings.openrouter_api_key:
            raise LLMError("未设置 OPENROUTER_API_KEY")
        return OpenAICompatLLM(
            self.settings.openrouter_api_key,
            spec,
            OPENROUTER_BASE_URL,
            headers={"X-Title": "AI VC Multi-Agent"},
            # 研究员的结构化证据较长，推理模型的思考内容也计入输出上限
            max_tokens={"fact_checker": 32000, "chair": 16000, "sector_researcher": 32000, "company_researcher": 32000,
                        "company_analyst": 32000, "research": 32000, "analysis": 32000, "audit": 32000}.get(role_key, 16000),
        )
