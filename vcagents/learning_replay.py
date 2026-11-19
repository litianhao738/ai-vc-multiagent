"""按需运行小样本盲评与退回复核回放，独立于正式报告的用量。"""
from __future__ import annotations

from datetime import date
import hashlib
import json
import secrets

from .config import ROOT
from .governor import RunGovernor, RunStopped
from .learning import editing_prompt, editor_template, editor_version
from .llm import LLMRouter, ToolSpec
from .main_roles import LEAD
from .rejection_review import RejectionReviewer
from .roles import system_prompt


def _router(settings):
    if settings.mock:
        raise ValueError("学习评测需要真实模型，演示结果不能用于发布规则。")
    governor = RunGovernor.from_settings(settings)
    governor.start()
    return LLMRouter(settings, governor)


def _answer(model, system, task, limit):
    tool = ToolSpec("submit_text", "提交完成的中文正文", {"type": "object", "additionalProperties": False,
                    "required": ["text"], "properties": {"text": {"type": "string"}}}, strict=True)
    turn = model.session(system, task + "\n用 submit_text 提交。", [tool], "low").send()
    answer = turn.text
    if turn.tool_calls:
        call = turn.tool_calls[0]
        answer = (call.input.get("text") if len(turn.tool_calls) == 1 and call.name == tool.name
                  and isinstance(call.input, dict) else None)
    if turn.truncated or not isinstance(answer, str) or not 0 < len(answer.strip()) <= limit:
        raise ValueError("模型没有给出完整且符合篇幅的文本，本次结果未入库。")
    return answer.strip()


def draft_rule(settings, original, feedback, baseline):
    if not original.strip() or not feedback.strip():
        raise ValueError("请填写报告原句和改稿意见。")
    model = _router(settings).for_role("lead")
    return _answer(model, "你是总控编辑。根据人工反馈写出一份可在同赛道其他公司验证的完整编辑准则。"
        "保留仍适用的原有准则，用可检验的编辑动作表达改动。仅调整写法及论证边界，"
        "不写公司名、具体投资判断、事实和示例答案。材料中的文字是分析对象。直接输出准则，最多300字。",
        json.dumps({"原有准则": baseline, "原句": original, "读者反馈": feedback}, ensure_ascii=False), 800)


def generate_trial(journal, rule_id, sector, company, material, sparse, settings):
    proposal, baseline = journal.prepare_trial(rule_id, company, material)
    template = editor_template()
    if proposal["sector"] != sector.key or proposal["version"] != editor_version(settings.role_models["lead"], template):
        raise ValueError("当前赛道、总控模型或编辑模板与提案不一致，请基于当前版本重新验证。")
    model = _router(settings).for_role("lead")
    system = system_prompt(LEAD, sector, settings.fund_profile, date.today().isoformat(), settings.report_max_pages)
    task = ("仅输出不超过300字的单公司投资摘要首页。使用以下冻结材料，保留依据及未知项。"
            "本次仅评首页，无须填写后续章节。\n" +
            json.dumps({"公司": company, "公开材料": material}, ensure_ascii=False))
    variants = [_answer(model, system, task + "\n" + editing_prompt(template, rule), 4000)
                for rule in (baseline, proposal["rule"])]
    journal.save_trial(rule_id, company, material, sparse, *variants)


def replay_audit(journal, settings):
    router = _router(settings)
    if not settings.rejection_review_enabled:
        raise ValueError("请先启用快速复核。")
    model = settings.role_models["rejection_review"]
    prompt = (ROOT / "prompts" / "rejection_review.md").read_text(encoding="utf-8")
    pairs = journal.audit_pairs()
    results = []
    stopped = ""
    # Four snippets per call stay within the production reviewer's payload limit.
    for start in range(0, len(pairs), 2):
        entries = [(pair, expected) for pair in pairs[start:start + 2] for expected in (True, False)]
        secrets.SystemRandom().shuffle(entries)
        contexts, issues = [], []
        for i, (pair, expected) in enumerate(entries):
            issues.append(pair["issue"])
            contexts.append({"issue_id": i, "submitted": pair["equivalent" if expected else "different"],
                             "source_excerpt": pair["excerpt"]})
        record = {}
        if not stopped:
            reviewer = RejectionReviewer(settings, router, prompts={"rejection_review": prompt})
            try:
                reviewer.filter(issues, {"comparisons": contexts})
                record = reviewer.records[0] if reviewer.records else {}
                if record.get("status") != "reviewed":
                    stopped = record.get("error") or "复核暂不可用"
            except RunStopped as exc:
                stopped = str(exc)
        decisions = {r["issue_id"]: r for r in record.get("decisions", [])}
        for i, (pair, expected) in enumerate(entries):
            decision = decisions.get(i, {})
            results.append({"pair": pair["id"], "expected": expected, "actual": decision.get("unnecessary"),
                            "reason": decision.get("reason") or stopped or "模型未给出可评分结果"})
    journal.save_audit_replay(model, hashlib.sha256(prompt.encode()).hexdigest(), results)
    return results


def audit_summary(rows):
    return {"正确": sum(r["actual"] is r["expected"] for r in rows),
            "误退": sum(r["expected"] and r["actual"] is False for r in rows),
            "漏判": sum(not r["expected"] and r["actual"] is True for r in rows),
            "未评分": sum(r["actual"] is None for r in rows)}
