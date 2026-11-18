"""成稿一致性检查：以结构化结果为准核对正文。

只有影响结论可信度的问题才阻止发布（publication_issues）；用语、排版类问题只提醒（publication_warnings）。
判断先看结构化字段，再看正文；否定和条件按所在分句判断，不靠整句有没有“不”字。
"""
from __future__ import annotations

import re
import hashlib
import json

from . import grounding
from .reasoning import review_issues
from .corrections import affected_materials
from .report_review import apply_report_reviews


_REFERENCE = re.compile(r"\[((?:E-)?(?:C\d+|S)-\d+|E-\d+|U-1)\]")
_CLAUSE = re.compile(r"[^。；;！？!?\n]+")
_NEGATED = re.compile(r"(?:不|未|没有|缺乏|难以|谈不上|并非|无法|尚未|能否|是否|暂不|可否)")
_CONDITIONAL = re.compile(r"(?:若|如果|假设|除非|只有|只要|倘若)")
_INVEST = re.compile(r"(?:建议|推荐|值得|应当|应该)\s*(?:投资|买入|入股)(?!金额|额|比例|人|者|机构|方)|recommend(?:ed|s)?\s+invest", re.I)
_NO_INVEST = re.compile(r"(?:不建议|不推荐|不值得|不应)\s*(?:投资|买入)|建议不投|决定不投")
_VALUE_CLAIM = re.compile(r"安全边际|显著低估|明显低估|严重低估|估值便宜|价格便宜|性价比(?:极)?高|明显折价|显著折价|"
                          r"margin of safety|undervalued", re.I)
_SCORE = re.compile(r"(?:综合|加权|总)[^\d\n。；]{0,10}?分[^\d\n。；]{0,8}?(\d+(?:\.\d+)?)|"
                    r"(?:综合|加权|总)[^\d\n。；]{0,12}?(\d+(?:\.\d+)?)\s*(?:/\s*10|分)")
_RANGE_WORDS = re.compile(r"范围|区间|假设|情景|变化")
_UNDISCLOSED = r"[^。；\n]{0,18}?(?:未披露|没有披露|未公开|未见披露|尚未披露)"
_PARTIAL = re.compile(r"具体|数字|数值|明细|计算口径|口径|独立|审计|核验|第三方|分拆|队列")
_METRICS = {  # 指标名 → (证据里怎么写, 正文里怎么写)
    "净收入留存": (r"NRR|net revenue retention|净收入留存", r"NRR|净收入留存率?|留存"),
    "毛利": (r"gross margin|毛利", r"毛利(?!率)"),
    "年化收入": (r"annualized revenue|\bARR\b|年化收入|运行率", r"\bARR\b|年化收入"),
    "估值": (r"valuation|估值", r"估值"),
}
_INTERNAL = re.compile(r"audit\.md|--[a-z][a-z-]+|代码(?:计算|测算|排序)|用户(?:提供|输入)|底稿|原文匹配")
_CENTRAL_NAMES = ("基线", "基准", "中性")


def memo_materials(result) -> dict:
    return {"decision": result.decision,
            "sector": {k: result.sector_brief[k] for k in ("summary", "market_size", "why_now", "landscape", "benchmarks")},
            "analyses": result.analyses, "comparison": result.comparison, "debate": result.debate}


def _asserted(pattern: re.Pattern, text: str) -> list[str]:
    """正文确实在主张的句子：否定只看同一逗号分句（“不贵，具有安全边际”不算否定），条件看整句前文。"""
    out = []
    for clause in _CLAUSE.findall(text or ""):
        for match in pattern.finditer(clause):
            before = clause[:match.start()]
            phrase = re.split(r"[，,、:：]", before)[-1][-14:]
            if not _NEGATED.search(phrase) and not _CONDITIONAL.search(before):
                out.append(clause.strip())
                break
    return out


def _body_without_debate(memo: str) -> str:
    """多空双方可以持不同观点，保留意见同理，不算报告自身的结论。"""
    body = memo.split("### 关键事实与引用", 1)[0]
    if "### 多空辩论" in body:
        before, debate = body.split("### 多空辩论", 1)
        body = before + debate.split("### 退出情景与回报", 1)[-1]
    return body.split("### 保留意见", 1)[0] + "\n" + (body.split("### 方法与假设", 1)[-1] if "### 方法与假设" in body else "")


def _central_multiple(returns: list[dict]) -> float | None:
    """中心情景的回报：有基线类情景就用它，否则取中位数，不依赖情景叫什么。"""
    known = [row for row in returns if row.get("multiple") is not None]
    for row in known:
        if any(name in str(row.get("scenario", "")) for name in _CENTRAL_NAMES):
            return row["multiple"]
    if not known:
        return None
    ordered = sorted(row["multiple"] for row in known)
    return ordered[len(ordered) // 2]


def _conclusion_issues(result, body: str) -> list[str]:
    action = result.decision["action"]
    issues = []
    conflict = _INVEST if action not in ("投资", "附条件投资") else _NO_INVEST
    for declared in re.findall(r"(?:本次建议|投资结论)[：:]\s*\**([^\n。；]+)", body):
        if _asserted(conflict, declared):
            issues.append(f"正文投资结论与本次建议“{action}”冲突")
    if _asserted(conflict, result.sector_brief.get("summary", "")):
        issues.append(f"赛道摘要的结论与本次建议“{action}”冲突")
    for name, analysis in result.analyses.items():
        if _asserted(conflict, analysis.get("summary", "")):
            issues.append(f"{name}研究摘要仍包含与最终建议“{action}”不一致的结论")
    return issues


def _score_issues(result, body: str) -> list[str]:
    expected = {row["weighted"] for row in result.ranking}
    issues = []
    for clause in _CLAUSE.findall(body):
        if _RANGE_WORDS.search(clause):
            continue
        for match in _SCORE.finditer(clause):
            score = match.group(1) or match.group(2)
            if expected and all(abs(float(score) - value) > 0.05 for value in expected):
                issues.append(f"正文加权分 {score} 与评分表不一致")
    return issues


def _return_issues(result) -> list[str]:
    central = _central_multiple(result.returns)
    if central is None or central >= 1:
        return []
    decision = result.decision
    texts = [decision.get("headline", ""), decision.get("valuation_view", "")]
    texts += [item.get("judgment", "") for item in decision.get("top_pick_thesis", [])]
    if any(_asserted(_VALUE_CLAIM, text) for text in texts):
        return [f"中心情景回报只有 {central:.2f} 倍，估值判断却声称价格低估或存在安全边际"]
    return []


def _disclosure_issues(result, body: str) -> list[str]:
    """证据里明明有这个指标的披露，正文却说没有披露。说“没有具体数字、未经核验”不算矛盾。"""
    records = [e for sheet in (result.sector_brief, *result.fact_sheets.values()) for e in sheet.get("evidence", [])
               if e.get("excerpt_verified") and any(name.lower() in e.get("claim", "").lower() for name in result.analyses)]
    issues = []
    for label, (in_evidence, in_body) in _METRICS.items():
        if not any(re.search(in_evidence, e.get("excerpt", ""), re.I) for e in records):
            continue
        for clause in _CLAUSE.findall(body):
            match = re.search(rf"(?:{in_body}){_UNDISCLOSED}", clause, re.I)
            if match and not _PARTIAL.search(match.group()):
                issues.append(f"已有{label}披露，正文仍称没有披露")
                break
    return issues


def quality_finding(kind: str, owner: str, message: str, *, location="memo", target="", **detail) -> dict:
    identity = json.dumps([kind, owner, location, target, message], ensure_ascii=False)
    return {"issue_id": "Q-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12],
            "kind": kind, "owner": owner, "location": location, "target": target,
            "message": message, **detail}


def publication_findings(result, memo: str, *, review_amount=None) -> list[dict]:
    """Return located, owner-routed failures; keep human messages for export compatibility."""
    from .report import _public_view

    reviewed = result.decision_fact_check.get("reviewed_targets", {})
    checked_targets = {row["target_id"] for row in result.decision_fact_check.get("reasoning_checks", [])}
    issues = [quality_finding("reasoning", "audit" if (reviewed.get(item["target_id"]) != item["target"]
                                                     or item["target_id"] not in checked_targets) else "lead",
                             f"核心投资判断 {item['target_id']} 未通过复核：{item['reason']}",
                             location="decision." + item["target_id"], target=item["target"])
              for item in review_issues(result.decision, result.decision_fact_check)]
    body = memo.split("### 关键事实与引用", 1)[0]
    definitions = set(re.findall(r"^- (E-[A-Za-z0-9_-]+) ·", memo, re.M))
    definitions.update(re.findall(r"^- \[((?:C\d+|S)-\d+)\]", memo, re.M))
    if "### 交易与测算依据 [U-1]" in memo:
        definitions.add("U-1")
    missing = sorted(set(_REFERENCE.findall(body)) - definitions)
    if missing:
        issues.append(quality_finding("citation", "lead", "正文引用在附录中没有定义：" + "、".join(missing),
                                      target="、".join(missing)))

    stated = _body_without_debate(memo)
    for kind, messages in (("conclusion", _conclusion_issues(result, stated)),
                           ("score", _score_issues(result, stated)), ("valuation", _return_issues(result)),
                           ("disclosure", _disclosure_issues(result, stated))):
        issues += [quality_finding(kind, "lead", message) for message in messages]

    public = _public_view(result)
    index = grounding.evidence_index(public.sector_brief, public.fact_sheets)
    index[grounding.USER_KEY] = grounding.user_entry(result)
    checked = grounding.check_material("成稿", memo_materials(public), index)
    # Cleaning a public view must not erase evidence failures and turn a failed
    # run into an apparently publishable report. Audit the original submission.
    raw_index = grounding.evidence_index(result.sector_brief, result.fact_sheets)
    raw_index[grounding.USER_KEY] = index[grounding.USER_KEY]
    raw_checked = grounding.check_material("提交材料", memo_materials(result), raw_index)
    for key in ("findings", "scope_cases"):
        entries = checked.setdefault(key, [])
        seen = {(f["sentence"], f["problem"]) for f in entries}
        entries.extend(f for f in raw_checked.get(key, []) if (f["sentence"], f["problem"]) not in seen)
    checked = apply_report_reviews(checked, result.rejection_reviews, review_amount)
    result.evidence_checks = {"成稿": checked}
    research = {"sector_researcher": result.sector_brief,
                **{f"company_researcher:{name}": sheet for name, sheet in result.fact_sheets.items()}}
    for item in checked["findings"]:
        if item.get("severity", "error") != "error":
            continue
        upstream = affected_materials(research, [{"claim": item["sentence"]}])
        field = item["field"]
        for name in result.analyses:
            if field.startswith(f"analyses.{name}."):
                upstream = {key: value for key, value in upstream.items()
                            if key in ("sector_researcher", f"company_researcher:{name}")}
                break
        owner = ("research" if upstream else "analysis" if field.startswith(("analyses.", "comparison.", "debate[")) else "lead")
        issues.append(quality_finding("grounding", owner,
            f"事实数字缺少对应证据：{item['sentence'][:150]}（{item['problem']}）",
            location=next(iter(upstream), field), target=item["sentence"], problem=item["problem"]))
    return list({item["issue_id"]: item for item in issues}.values())


def publication_issues(result, memo: str) -> list[str]:
    """Export is a read-only safety gate; repair runs in the pipeline before export."""
    return list(dict.fromkeys(item["message"] for item in publication_findings(result, memo)))


def publication_warnings(result, memo: str) -> list[str]:
    """不影响结论、可以出稿后顺手改的问题。"""
    warnings = []
    body = memo.split("### 关键事实与引用", 1)[0]
    terms = sorted(set(_INTERNAL.findall(body)))
    if terms:
        warnings.append("正文含内部用语：" + "、".join(terms))
    checked = (getattr(result, "evidence_checks", None) or {}).get("成稿") or {}
    notes = [f for f in checked.get("findings", []) if f.get("severity") == "note" and not f.get("amount_review")]
    if notes:
        warnings.append(f"{len(notes)} 句数字能在证据库找到，但引用编号不对应或缺失，建议复核")
    if any(f.get("amount_review", {}).get("kind") == "scope" for f in checked.get("findings", [])):
        warnings.append("有数值相同但口径需要对照原文的主张，见内部审计记录")
    return warnings
