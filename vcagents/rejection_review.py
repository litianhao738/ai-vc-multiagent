"""Single-turn review of unnecessary writing-related rejections, without rewriting data."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import threading
import time

from .config import ROOT
from .governor import RunStopped
from .llm import ToolSpec
from .report_review import review_result


class RejectionReviewer:
    def __init__(self, settings, router, emit=None):
        self.settings, self.router = settings, router
        self.emit = emit or (lambda message: None)
        self._lock = threading.RLock()
        self.reset()

    def reset(self):
        with self._lock:
            self.records = []
            self._cache = {}
            self._unavailable = False

    def filter(self, issues: list[str], context: dict) -> list[str]:
        if not issues or len(issues) > 10:
            return list(issues)
        payload = {"issues": [{"issue_id": i, "message": message} for i, message in enumerate(issues)], **context}
        schema = {"type": "object", "additionalProperties": False, "required": ["decisions"],
            "properties": {"decisions": {"type": "array", "minItems": len(issues), "maxItems": len(issues),
                "items": {"type": "object", "additionalProperties": False,
                    "required": ["issue_id", "unnecessary", "reason"], "properties": {
                        "issue_id": {"type": "integer", "enum": list(range(len(issues)))},
                        "unnecessary": {"type": "boolean"}, "reason": {"type": "string"}}}}}}

        def interpret(data):
            rows = data.get("decisions") if isinstance(data, dict) else None
            if (not isinstance(rows, list) or len(rows) != len(issues)
                    or any(not isinstance(row, dict) or type(row.get("issue_id")) is not int
                           or type(row.get("unnecessary")) is not bool
                           or not isinstance(row.get("reason"), str) or not row["reason"].strip() for row in rows)
                    or sorted(row["issue_id"] for row in rows) != list(range(len(issues)))):
                raise ValueError("复核缺项、重复或类型不符，保留原退回")
            waived = {row["issue_id"] for row in rows if row["unnecessary"]}
            return rows, [issue for i, issue in enumerate(issues) if i in waived]

        checked = self._ask("rejection_review", payload, context, issues, interpret,
                            ToolSpec("review_rejection", "提交退回必要性判断", schema, strict=True),
                            "复核退回是否仅由标点、写法或等值单位换算引起")
        if checked is None:
            return list(issues)
        waived = {row["issue_id"] for row in checked["decisions"] if row["unnecessary"]}
        return [issue for i, issue in enumerate(issues) if i not in waived]

    def review_amount(self, context: dict) -> dict | None:
        """Check one claim against one verified quote with the same bounded model session."""
        schema = {"type": "object", "additionalProperties": False,
                  "required": ["verdict", "basis", "reason"], "properties": {
                      "verdict": {"type": "string", "enum": ["equivalent", "different", "insufficient"]},
                      "basis": {"type": "string"}, "reason": {"type": "string"}}}
        issue = context["issue"]

        def interpret(data):
            checked = review_result(context, data)
            waive = checked["verdict"] == "equivalent" and context["kind"] == "table_unit"
            return [checked], [issue] if waive else []

        record = self._ask("report_amount", context, context, [issue], interpret,
                           ToolSpec("review_amount", "提交成稿局部数值复核", schema, strict=True),
                           "复核一条成稿金额的单位与原文口径")
        return deepcopy(record["decisions"][0]) if record else None

    def _ask(self, scope, payload, context, issues, interpret, tool, progress) -> dict | None:
        if self.settings.mock or not self.settings.rejection_review_enabled:
            return None
        prompt = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        # Do not silently omit context and then let a model approve an incomplete comparison.
        if len(prompt) > 12000:
            return None
        key = hashlib.sha256((scope + "\n" + prompt).encode("utf-8")).hexdigest()
        with self._lock:
            if key in self._cache:
                return self._cache[key]
            if self._unavailable or len(self.records) >= self.settings.rejection_review_max_calls:
                return None
            record = {"scope": scope, "model": self.settings.role_models["rejection_review"], "issues": list(issues),
                      "context": deepcopy(context), "decisions": [], "waived": [], "status": "unavailable",
                      "usage": [0, 0], "cost_usd": None}
            session, started = None, time.monotonic()
            try:
                model = self.router.for_role("rejection_review")
                system = (ROOT / "prompts" / f"{scope}.md").read_text(encoding="utf-8")
                session = model.session(system, prompt, [tool], "low")
                self.emit(progress)
                turn = session.send()
                if turn.truncated:
                    raise ValueError("复核输出截断，保留原退回")
                if turn.tool_calls:
                    if len(turn.tool_calls) != 1 or turn.tool_calls[0].name != tool.name:
                        raise ValueError("复核未按约定提交，保留原退回")
                    data = turn.tool_calls[0].input
                else:
                    data = json.loads(turn.text.strip())
                rows, waived = interpret(data)
                record.update(status="reviewed", decisions=rows, waived=waived)
            except RunStopped:
                raise
            except Exception as exc:
                self._unavailable = True
                record["error"] = f"{type(exc).__name__}: {exc}"
                self.emit("退回复核暂不可用，沿用原检查结果")
            finally:
                if session is not None:
                    record["usage"] = [session.input_tokens, session.output_tokens]
                    record["cost_usd"] = session.cost_usd
                record["seconds"] = round(time.monotonic() - started, 3)
                self.records.append(record)
            self._cache[key] = record if record["status"] == "reviewed" else None
            if record["waived"]:
                self.emit(f"确认 {len(record['waived'])} 项仅为写法差异，撤销这些退回")
            return self._cache[key]
