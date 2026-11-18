"""单个智能体的工具调用循环：调用模型 → 执行工具 → 回传结果，直到模型用 submit_result 提交结构化结果。"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable

from . import schemas
from .llm import LLM, ToolResult, ToolSpec
from .roles import Role
from .tools import Toolbox, describe_call

SUBMIT = "submit_result"


@dataclass
class Event:
    kind: str  # stage | start | log | done | error
    agent: str  # 智能体实例 id，如 team、bull_r2；阶段事件为空
    name: str  # 展示用名称
    message: str = ""


Emit = Callable[[Event], None]


class AgentError(RuntimeError):
    pass


class Agent:
    max_nudges = 2
    max_truncations = 3
    max_bonus_steps = 3  # 引用被退回时额外给的修正步数
    max_content_rounds = 2  # 内容检查最多退回两次，之后保留结果并记入未解决清单

    def __init__(self, role: Role, llm: LLM, toolbox: Toolbox | None, emit: Emit, *, system: str,
                 schema: dict | None = None, agent_id: str = "", name: str = "", max_steps: int = 14,
                 extra_check: Callable[[dict], list[str]] | None = None):
        self.role = role
        self.llm = llm
        self.toolbox = toolbox or Toolbox()
        self.emit = emit
        self.system = system
        self.schema = schema or role.output_schema
        self.id = agent_id or role.key
        self.name = name or role.name
        self.max_steps = max_steps
        self.input_tokens = 0
        self.output_tokens = 0
        self.format_failures = 0
        self.cost_usd: float | None = None
        self._evidence_reviewed = False
        self.evidence_reviews = 0
        self.extra_check = extra_check
        self._content_failures = 0
        self._last_valid: dict | None = None
        self.unresolved_issues: list[str] = []
        self._bonus_steps = 0
        self._missing_evidence = 0

    def _log(self, kind: str, message: str) -> None:
        self.emit(Event(kind, self.id, self.name, message))

    def run(self, task: str) -> dict:
        self._log("start", "开始工作")
        try:
            result = self._loop(task)
        except Exception as e:
            self._log("error", str(e))
            raise
        self._log("done", "已提交结果")
        return result

    def _submission_errors(self, schema: dict, data: dict) -> list[str]:
        errors = schemas.validate_submission(self.role.key, schema, data)
        if (not errors and self.role.key in ("sector_researcher", "company_researcher")
                and data["sources"] and not data["evidence"] and not self._evidence_reviewed):
            self._evidence_reviewed = True
            self.evidence_reviews += 1
            errors.append("已有来源但 evidence 为空，请重新检查原文并填写能确认的事实；"
                          "若没有可确认的事实，说明缺口后可再次提交空列表")
        if not errors and self.extra_check is not None:
            errors = self.extra_check(data)
            if errors:
                self._content_failures += 1
                self._last_valid = data
                if self._content_failures > self.max_content_rounds:
                    # 真实运行里约一半角色会被退回两次以上；再退回只增加费用。保留结果，问题记入清单，报告不能作为正式版
                    self.unresolved_issues = errors
                    self._log("log", "内容检查修正后仍有问题，保留结果并记入未解决清单：" + "；".join(errors[:3]))
                    return []
        if errors and self._bonus_steps < self.max_bonus_steps:
            self._bonus_steps += 1  # 修正提交不占用检索步数
        return errors

    def _loop(self, task: str) -> dict:
        schema = self.schema
        submit = ToolSpec(SUBMIT, "提交最终结构化结果；检查失败时按提示修正后重新提交。", schema, strict=True)
        session = self.llm.session(self.system, task, self.toolbox.specs(self.role.tools) + [submit], self.role.effort)
        nudges = 0
        truncations = 0
        step = 0
        try:
            while step < self.max_steps + self._bonus_steps:
                step += 1
                limit = self.max_steps + self._bonus_steps
                self._log("log", f"等待模型响应（第 {step}/{limit} 步）")
                turn = session.send()
                for note in turn.notes:
                    self._log("log", note)
                if turn.truncated:
                    truncations += 1
                    self._log("log", "输出过长被截断，要求精简后重试")
                    if truncations >= self.max_truncations:
                        raise AgentError(f"模型输出连续截断 {truncations} 次，无法完成结构化提交")
                    continue
                truncations = 0

                if not turn.tool_calls:
                    # 部分模型不擅长调用工具，会直接用文本输出 JSON，这里兜底解析
                    reviews_before = self.evidence_reviews
                    parsed = _json_from_text(turn.text)
                    if parsed is not None:
                        parsed = schemas.normalize(schema, parsed)
                        errors = self._submission_errors(schema, parsed)
                        if not errors:
                            return parsed
                    else:
                        errors = []
                    if self.evidence_reviews == reviews_before:
                        self.format_failures += 1
                    if nudges >= self.max_nudges:
                        raise AgentError("模型没有按要求提交结构化结果")
                    nudges += 1
                    detail = "；".join(errors[:3])
                    session.add_user(f"请调用 {SUBMIT} 工具提交最终结果，不要用普通文本回答。"
                                     + (f"请处理：{detail}" if detail else ""))
                    continue

                results: list[ToolResult] = []
                for call in turn.tool_calls:
                    if call.name != SUBMIT:
                        continue
                    reviews_before = self.evidence_reviews
                    data = schemas.normalize(schema, call.input) if call.input is not None else None
                    if data is None and self._bonus_steps < self.max_bonus_steps:
                        self._bonus_steps += 1
                    errors = [call.error or "参数不是合法 JSON"] if data is None else self._submission_errors(schema, data)
                    if errors == ["缺少字段 evidence"]:
                        self._missing_evidence += 1
                        if self._missing_evidence >= 3:  # 反复漏交证据：按“没有可核对的证据”接收，不让整个角色失败
                            self._log("log", "连续 3 次未提交 evidence，按无可核对证据接收；底稿中的事实均视为未核对")
                            data = {**data, "evidence": []}
                            errors = self._submission_errors(schema, data)
                    if not errors:
                        return data
                    if self.evidence_reviews == reviews_before:
                        self.format_failures += 1
                    self._log("log", f"提交需要修正：{'；'.join(errors[:3])}")
                    results.append(ToolResult(call.id, "提交需要修正：" + "；".join(errors) + "。请修正后重新提交。", True))

                others = [c for c in turn.tool_calls if c.name != SUBMIT]
                for call in others:
                    self._log("log", describe_call(call))
                results += self.toolbox.execute_many(others)

                note = ""
                if step >= self.max_steps + self._bonus_steps - 2:
                    note = f"已接近步数上限，请基于现有信息立即调用 {SUBMIT} 提交结果。"
                session.add_tool_results(results, note)
            if self._last_valid is not None:
                self.unresolved_issues = self.unresolved_issues or ["步数用完时内容问题仍未修正"]
                self._log("log", "步数用完，保留最后一次格式合格的提交并记入未解决清单")
                return self._last_valid
            raise AgentError(f"超过最大步数（{self.max_steps}）仍未提交结果")
        finally:
            self.input_tokens = session.input_tokens
            self.output_tokens = session.output_tokens
            self.cost_usd = session.cost_usd


def _json_from_text(text: str) -> dict | None:
    match = re.search(r"\{.*\}", text or "", re.S)
    if not match:
        return None
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None
