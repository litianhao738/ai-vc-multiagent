"""Decision-focused public report; long research and execution detail stay in audit.

No model call or factual rewrite occurs here. Statements are selected as whole
fields, and every reference actually used by the report is included.
"""
from __future__ import annotations

import re

from .report import (_public_view, _report_title, _decision_opening, _trade_decision, _bullets, _text, _score_table, _listing,
                     _market_section, _metrics_brief, _deal_table, _deal_summary, _competitor_table,
                     _risk_table, _thesis_items, _thesis_detail, _returns_section)
from .financials import INPUT_LABELS, MARKET_INPUT_LABELS
from .schemas import LONG_TERM_DIMENSIONS

REFERENCE = re.compile(r"\[((?:E-)?(?:C\d+|S)-\d+|U-1)\]")
_QUESTION_STATUS = {"open": "仍待回答", "answered": "已有回答", "blocked": "现有渠道无法回答"}
_COST_LABELS = (("rnd_investment", "研发投入"), ("delivery_cost", "交付成本"),
                ("user_cost", "用户总成本"), ("acquisition_cost", "获客成本"))


def _refs(ids, known: set[str]) -> str:
    """Structured evidence IDs become citations only if the evidence survived public filtering."""
    return "".join(f"[{eid}]" for eid in dict.fromkeys(ids or []) if eid in known)


def _selection(r, d) -> str:
    rationale = (r.research_state or {}).get("selection_rationale", "")
    parts = [f"委托方给出的理由为“{rationale}”。" if rationale else "",
             _text(d.get("selection_view"), "")]
    text = " ".join(p for p in parts if p)
    return f"**研究对象与选择依据**：{text}" if text else ""


def _question_block(r, d) -> str:
    """Keep decision-relevant questions and their evidence alongside the deeper analysis."""
    questions = {q["question_id"]: q for q in (r.research_state or {}).get("questions", [])
                 if _text(q.get("question"), "")}
    rows = [row for row in d.get("question_answers", [])
            if row.get("question_id") in questions and _text(row.get("answer"), "")][:3]
    if not rows:
        return ""
    blocks = [f"**{questions[row['question_id']]['question']}**\n"
              f"{row['answer']}（把握：{row.get('confidence') or '—'}）\n"
              f"- 依据：{_text(row.get('evidence'))}\n- 什么会改变这个回答：{_text(row.get('would_change_if'))}"
              for row in rows]
    answered = {row["question_id"] for row in rows}
    rest = [q for qid, q in questions.items() if qid not in answered]
    if rest:
        blocks.append("其余问题：" + "；".join(
            f"{q['question']}（{_QUESTION_STATUS.get(q.get('status'), '仍待回答')}）" for q in rest[:4])
            + ("等" if len(rest) > 4 else "") + "。")
    return "### 关键问题与回答\n" + "\n\n".join(blocks)


def _standout_list(r, known: set[str]) -> str:
    """Each company's single most distinctive point, side by side; lists wrap better than tables in the PDF."""
    rows = []
    for row in r.ranking:
        standout = r.analyses[row["company"]].get("standout") or {}
        if _text(standout.get("point"), ""):
            rows.append(f"**{row['company']}**：{standout['point']}——{_text(standout.get('mechanism'), '')}"
                        + _refs(standout.get("evidence_ids"), known))
    return "**各公司最突出的一点及其胜出机制**\n" + _bullets(rows) if rows else ""


def _competitor_differences(a: dict, known: set[str]) -> str:
    diffs = [row for row in a.get("competitor_diffs", []) if _text(row.get("difference"), "")][:3]
    if not diffs:
        return ""
    return "**与竞对的差异**\n" + _bullets([
        f"对比{_text(row.get('competitor'), '竞对')}（{_text(row.get('axis'), '综合')}）：{row['difference']}"
        + _refs(row.get("evidence_ids"), known) for row in diffs])


def _cost_structure(a: dict, known: set[str]) -> str:
    cost = a.get("cost_structure") or {}
    if not any(_text(cost.get(key), "") for key, _ in _COST_LABELS):
        return ""
    return ("**投入与成本结构**" + _refs(cost.get("evidence_ids"), known) + "\n"
            + _bullets([f"{label}：{_text(cost.get(key), '未披露')}" for key, label in _COST_LABELS]))


def _long_term_view(a: dict, known: set[str]) -> str:
    edge = {row.get("dimension"): row for row in a.get("long_term_edge", [])}
    rows = [edge[name] for name in LONG_TERM_DIMENSIONS if name in edge]
    if not rows:
        return ""
    return "**长期竞争力**\n" + _bullets([
        f"{row['dimension']}（{_text(row.get('judgment'), '未知')}）：{_text(row.get('reason'), '')}"
        + _refs(row.get("evidence_ids"), known) for row in rows])


def _insight_section(a: dict, known: set[str]) -> str:
    parts = [_competitor_differences(a, known), _cost_structure(a, known), _long_term_view(a, known)]
    details = "\n\n".join(part for part in parts if part)
    return f"### 竞争差异、成本与长期竞争力\n{details}" if details else ""


def _gap_list(d) -> str:
    rows = [g for g in d.get("gap_impacts", []) if _text(g.get("missing"), "")][:3]
    if not rows:
        return ""
    return "### 信息缺口及其影响\n" + _bullets([
        f"{g['missing']}：影响{_text(g.get('affects'), '—')}；拿到后{_text(g.get('if_obtained'), '—')}" for g in rows])


def _trade_terms(r, d, top: str, include_shortlist: bool = False) -> list[str]:
    deal = r.deal_metrics.get(top)
    return ["### 交易条件与估值", _trade_decision(d),
            f"**重点跟进**　{'、'.join(d['recommended']) or '暂无合适标的'}" if include_shortlist else "",
            f"**交易摘要（{top}）**　{_deal_summary(deal)}", _deal_table(deal),
            "**估值判断**：" + _text(d.get("valuation_view")),
            "**资金用途与里程碑**：" + _text(d.get("use_of_funds_milestones")),
            "**预测基础与缺口**：" + _text(d.get("forecast_assessment"))]


def _single_company_sections(r, thesis: list[dict], known: set[str], sector) -> list[str]:
    name = r.companies[0]
    d, b = r.decision, r.sector_brief
    a, sheet = r.analyses.get(name, {}), r.fact_sheets.get(name, {})
    leads = d.get("section_openings") or {}
    standout = a.get("standout") or {}
    mechanism = _text(standout.get("mechanism"), "")
    company_intro = dict.fromkeys((_text(leads.get("customer"), ""), _text(a.get("one_liner"), sheet.get("one_liner", "")),
                                   _text(a.get("summary"), "")))
    concerns = a.get("concerns", [])[:2]
    return [
        "## 二、公司概况与客户需求", *company_intro,
        _text(a.get("product_business")), "**团队与执行能力**　" + _text(a.get("team_assessment")),
        f"**融资与上市状态**　{sheet.get('stage', '未知')} · {_listing(sheet)}", _selection(r, d),
        "### 目标市场与采购需求", b["summary"], b["market_size"], _market_section(r.market_metrics),
        "**需求变化**\n" + _bullets(b["why_now"][:3]),
        "## 三、核心技术与竞争壁垒", _text(leads.get("technology"), ""),
        _text(d.get("core_answer"), "") if d.get("core_answer") != d.get("headline") else "",
        mechanism + _refs(standout.get("evidence_ids"), known) if mechanism else "",
        d["thesis"], "### 投资逻辑与证据\n" + _thesis_detail(thesis),
        "### 竞争格局与替代方案", b["landscape"], "**参照公司**\n" + _bullets(b["benchmarks"][:3]),
        r.comparison.get("summary", ""), _competitor_table(r) if r.comparison.get("per_company") else "",
        _competitor_differences(a, known), r.comparison.get("ranking_view", ""),
        "## 四、商业模式与经营质量", _text(leads.get("economics"), ""),
        "**经营事实与口径**　" + _text(a.get("operating_assessment")),
        _cost_structure(a, known), "### 财务与单位经济\n" + _metrics_brief(r.financial_metrics.get(name), sector),
        "## 五、估值与回报情景", _text(leads.get("valuation"), ""),
        *_trade_terms(r, d, name), _returns_section(r, d),
        "## 六、关键风险与尽调优先级", _text(leads.get("risks"), ""),
        "### 主要风险与应对\n" + _bullets(d["top_pick_risks"][:3]),
        "**公司经营与执行风险**\n" + _bullets(concerns) if concerns else "",
        "### 行业风险\n" + _risk_table(d["sector_risks"][:3]),
        _question_block(r, d), _gap_list(d), "### 尽调计划\n" + _bullets(d["dd_checklist"][:5]),
        "### 下一步\n" + _bullets(d["next_steps"][:3]), "### 保留意见\n" + _text(d["dissent"], "无"),
        "## 七、资料来源与测算口径",
    ]


def _multi_company_sections(r, thesis: list[dict], known: set[str], sector) -> list[str]:
    d, b, top = r.decision, r.sector_brief, r.decision["top_pick"]
    out = [
        "## 二、赛道和市场", b["summary"],
        "### 市场空间\n" + b["market_size"], _market_section(r.market_metrics),
        "**为什么是现在**\n" + _bullets(b["why_now"][:3]),
        "**行业格局**\n" + b["landscape"],
        "**参照公司**\n" + _bullets(b["benchmarks"][:3]),
        "## 三、公司比较与排序",
        _selection(r, d),
        r.comparison.get("summary", ""), _standout_list(r, known), _score_table(r, sector),
        r.comparison.get("ranking_view", ""),
        "> 排名反映赛道内的相对位置。投资建议及其条件见第六章。",
    ]
    if r.sensitivity:
        out.append(f"排序敏感性：单项评分上下变动一分，共 {r.sensitivity['scenarios']} 种情景，"
                   f"其中 {r.sensitivity['changes']} 种改变第一名；这是情景分析，不是成功概率。")
    out.append("## 四、各公司研究摘要")
    for row in r.ranking:
        name = row["company"]
        a, sheet = r.analyses[name], r.fact_sheets[name]
        out += [f"### {name} · 加权分 {row['weighted']} · 置信度{a['confidence']}",
                a["one_liner"], a["summary"],
                f"**产品与团队**：{_text(a.get('product_business'))} {_text(a.get('team_assessment'))}",
                f"**经营事实与口径**：{_text(a.get('operating_assessment'))}",
                f"**融资与上市状态**：{sheet.get('stage', '未知')}；{_listing(sheet)}",
                "**主要担忧**\n" + _bullets(a["concerns"][:2])]
    out += ["## 五、共同风险", _risk_table(d["sector_risks"][:3])]
    out += [f"## 六、重点分析对象：{top}",
            _text(d.get("core_answer"), "") if d.get("core_answer") != d.get("headline") else "",
            d["thesis"], "### 核心投资逻辑\n" + _thesis_detail(thesis),
            _insight_section(r.analyses.get(top, {}), known),
            "### 财务与单位经济\n" + _metrics_brief(r.financial_metrics.get(top), sector),
            *_trade_terms(r, d, top, include_shortlist=True),
            "### 主要风险与应对\n" + _bullets(d["top_pick_risks"][:3]),
            _question_block(r, d)]
    # Keep code-calculated economics and assumptions intact, including downside.
    out.append(_returns_section(r, d))
    out += ["## 七、尽调计划与附录", "### 尽调计划\n" + _bullets(d["dd_checklist"][:5]),
            "### 下一步\n" + _bullets(d["next_steps"][:3]),
            _gap_list(d),
            "### 保留意见\n" + _text(d["dissent"], "无")]
    return out


def render_brief(original) -> str:
    r = _public_view(original)
    d, b = r.decision, r.sector_brief
    sector = r.sector_config()
    single = len(r.companies) == 1
    thesis = _thesis_items(d)[:3]
    known = {e["evidence_id"] for sheet in (b, *r.fact_sheets.values()) for e in sheet.get("evidence", [])}
    # The H1 is the request's research subject. Flatten an optional prose subtitle into a single paragraph.
    subtitle = " ".join(_text(d.get("report_subtitle"), "").split()).strip("#* >") if single else ""
    if subtitle.rstrip("》").endswith("投资报告"):
        subtitle = ""
    metadata = f"> 研究对象：{'、'.join(r.companies)} · 赛道：{r.sector_name} · 报告状态：{r.report_status} · 赛道判断：{d['verdict']}"
    date = f"研究日期 {r.started_at[:10]} · " if r.started_at else ""
    out = [
        _report_title(r), subtitle,
        f"> {date}资料来源与测算口径见第七章。" if single else "",
        "> 演示模式：模拟数据，不代表真实研究结果。" if r.models.get("lead", r.models.get("chair")) == "mock" else "",
        "> 当前为研究草稿，不能作为正式报告。" if original.report_checks or original.errors or original.unresolved_checks else "",
        *_decision_opening(r, thesis, heading="投资摘要" if single else "投资价值与技术护城河"), metadata,
    ]
    if single:
        out.extend(_single_company_sections(r, thesis, known, sector))
    else:
        out.extend(_multi_company_sections(r, thesis, known, sector))
    out.append("### 方法与假设")
    if single:
        out.append(_long_term_view(r.analyses.get(r.companies[0], {}), known))
    if single and r.ranking:
        row = r.ranking[0]
        name = r.companies[0]
        confidence = _text(r.analyses.get(name, {}).get("confidence"), "未记录")
        out += [f"**{name}的综合评价**　加权分 {row['weighted']}，整体置信度{confidence}。",
                "| 评分维度 | 权重 | 得分 |\n| --- | --- | --- |\n" + "\n".join(
                    f"| {dim.name} | {dim.weight:.0%} | {row['scores'][dim.key]} |" for dim in sector.dimensions),
                "加权分为各维度权重与得分的乘积之和。"]
    elif not single:
        out += ["维度得分按赛道标准评估；加权分为各维度权重与得分的乘积之和，TOPSIS用于交叉检查。",
                "维度权重：" + "、".join(f"{dim.name} {dim.weight:.0%}" for dim in sector.dimensions) + "。"]
    out.append("回报倍数 = 退出股权价值 ×（1 - 后续稀释）÷ 本轮投后估值。"
               "简化年化IRR假设期初投入、期末一次退出；情景权重属于主观假设。"
               "历史原文与公司披露按各自时点解释，引用存在不代表事实已被独立证实。")
    if r.deal_metrics or r.financial_metrics or r.market_metrics:
        out.append("### 交易与测算依据 [U-1]\n金额统一为百万美元，比例按小数；以下来源说明未经独立核验。")
        for name, data in r.financial_metrics.items():
            out.append(f"**{name}**：期间 {data['period']}；输入与来源：" + "；".join(
                f"{INPUT_LABELS.get(key, key)}={value}（{data['sources'].get(key, '来源未提供')}）" for key, value in data["inputs"].items()))
        if r.market_metrics:
            data = r.market_metrics
            out.append(f"**市场测算**：期间 {data['period']}；" + "；".join(
                f"{MARKET_INPUT_LABELS.get(key, (key, ''))[0]}={value}（{data['sources'].get(key, '来源未提供')}）" for key, value in data["inputs"].items()))
    body = "\n\n".join(s for s in out if s)
    used = set(REFERENCE.findall(body))
    all_evidence = [("赛道", e) for e in b["evidence"]]
    all_evidence += [(name, e) for name, sheet in r.fact_sheets.items() for e in sheet["evidence"]]
    evidence_lines = []
    for name, e in all_evidence:
        if e["evidence_id"] not in used:
            continue
        evidence_lines.append(f"- {e['evidence_id']} · {name}：{e['claim']}；{e['source_type']}；"
                              f"时点：{e.get('as_of') or '未记录'}；{e['source_ref']}；{e.get('quote_location', '')}")
        used.update(REFERENCE.findall(e["source_ref"]))
    sources = [source for sheet in (b, *r.fact_sheets.values()) for source in sheet["sources"]
               if used.intersection(REFERENCE.findall(source))]
    if evidence_lines or sources:
        out += ["### 关键事实与引用", *evidence_lines, "### 信息来源", *["- " + s for s in dict.fromkeys(sources)]]
    return "\n\n".join(s for s in out if s) + "\n"
