"""把分析结果渲染成 7 部分的 Markdown 投资研究报告，结构与 docs/investment_memo_template.md 一致。"""
from __future__ import annotations

import json
import re
from copy import deepcopy

from .pipeline import AnalysisResult
from .sectors import Sector
from .financials import DEAL_LABELS, INPUT_LABELS, MARKET_INPUT_LABELS, MARKET_METRIC_LABELS, METRIC_LABELS

GAP = "现有资料未覆盖"  # 清理内部状态句后留空的字段用中性说明，不改写为“未披露”


def _bullets(items: list[str]) -> str:
    return "\n".join(f"- {x}" for x in items) or "- —"


def _text(value, fallback: str = GAP) -> str:
    return value if isinstance(value, str) and value.strip() else fallback


def _cell(value) -> str:
    return _text(value, "—").replace("\n", " ").replace("|", "\\|")


def _report_title(r: AnalysisResult) -> str:
    subject = r.companies[0] if len(r.companies) == 1 else r.sector_name
    return f"# {subject} 投资报告"


def _opens_with_trade_instruction(line: str, company: str) -> bool:
    point = line.strip().removeprefix(company).lstrip(" ，,：:")
    return point.startswith(("本次建议", "建议投资", "建议不投", "建议立项", "建议继续尽调",
                             "立项尽调", "继续尽调", "暂缓报价", "报价前", "待报价"))


def _decision_opening(r: AnalysisResult, thesis: list[dict], *, heading: str = "投资价值与技术护城河") -> list[str]:
    """Keep the company's capability, its evidence and its strongest challenge together."""
    d = r.decision
    lead = thesis[0] if thesis else {}
    analysis = r.analyses.get(d["top_pick"], {})
    standout = analysis.get("standout") or {}
    moat = d.get("technical_moat") or {}
    # Older headlines could contain an entire price recommendation. Preserve them with the trade terms.
    candidates = [d.get("headline")] if moat else []
    candidates.extend([standout.get("point"), lead.get("judgment")])
    headline = "现有材料尚未识别出有依据的独特能力。"
    for candidate in candidates:
        line = _text(candidate, "").strip()
        point = line.removeprefix(d["top_pick"]).strip(" ，,：:。.!！")
        action = d.get("action") or ""
        if (point and point not in {action, "建议" + action, "值得关注", "有潜力", "需进一步研究"}
                and not line.endswith(("？", "?")) and not _opens_with_trade_instruction(line, d["top_pick"])):
            headline = line
            break
    if moat:
        basis = _text(moat.get("evidence"), "")
        status = _text(moat.get("status")) if basis else "尚未证实"
        fields = [("技术护城河", f"{status}。{_text(moat.get('mechanism'))}"),
                  ("关键依据", basis or "现有材料缺少可核验的技术事实。"),
                  ("对手要复制什么", moat.get("replication")),
                  ("什么会推翻判断", moat.get("would_fail_if"))]
    else:
        tech = next((row for row in analysis.get("long_term_edge", []) if row.get("dimension") == "技术"), {})
        fields = [("技术观察", tech.get("reason")), ("关键依据", lead.get("evidence")),
                  ("技术护城河", "现有报告未单独核实技术的可复制性与持续性。")]
    return [f"## 一、{heading}", f"### 核心判断　{headline}",
            *(f"**{label}**　{value}" for label, value in fields if _text(value, "") and value != headline)]


def _trade_decision(d: dict) -> str:
    parts = [f"**本次建议：{_text(d.get('action'), '未记录')}**"]
    headline = _text(d.get("headline"), "")
    if headline and (not d.get("technical_moat") or _opens_with_trade_instruction(headline, d["top_pick"])):
        parts.append(headline)
    conditions = [c for c in d.get("action_conditions", []) if _text(c, "")]
    if conditions:
        parts.append("**推进与调整条件**\n" + _bullets(conditions))
    return "\n\n".join(parts)


def render_memo(r: AnalysisResult) -> str:
    r = _public_view(r)
    sector = r.sector_config()
    d, b = r.decision, r.sector_brief
    top, recommended = d["top_pick"], d["recommended"]
    deals = getattr(r, "deal_metrics", {})
    financials = getattr(r, "financial_metrics", {})
    thesis = _thesis_items(d)
    unresolved = getattr(r, "unresolved_checks", [])
    single = len(r.ranking) == 1  # 单公司投资报告：不做多公司排序，改为与具名对手对比
    out = [
        _report_title(r),
        "> 演示模式：以下内容为模拟数据，不代表真实研究结果。" if r.models.get("lead", r.models.get("chair")) == "mock" else "",
        "> 当前为待复核研究稿。" if r.errors or unresolved else "",
        *_decision_opening(r, thesis),
        f"> 研究对象：{'、'.join(r.companies)} · 赛道：{r.sector_name} · 生成时间：{r.finished_at}",
        f"> 报告状态：{r.report_status or '研究报告，供人工复核'} · 赛道判断：{d['verdict']}",

        # 二、赛道和市场
        "## 二、赛道和市场",
        b["summary"],
        f"### 市场空间\n{b['market_size']}",
        _market_section(getattr(r, "market_metrics", {})),
        f"### 为什么是现在\n{_bullets(b['why_now'])}",
        f"### 行业格局\n{b['landscape']}",
        f"### 参照公司\n{_bullets(b['benchmarks'])}",

        # 三、公司比较与排序（单公司报告为竞争对比）
        "## 三、竞争对比：为什么是这家" if single else "## 三、公司比较与排序",
    ]
    if r.comparison:
        out.append(r.comparison["summary"])
    if single:
        out += [_competitor_table(r), "**评分**（维度权重与评分标准见第七章“方法与假设”）\n\n" + _score_table(r, sector)]
        if r.comparison:
            out.append(f"**相对对手最站得住与最站不住的一点**：{r.comparison['ranking_view']}")
    else:
        out += [_comparison_table(r), _score_table(r, sector),
                "**排序依据**：加权分 = Σ 维度权重 × 维度得分；维度权重、评分标准与 TOPSIS 交叉检查见第七章“方法与假设”。"]
    sensitivity = getattr(r, "sensitivity", {})  # 已运行中的旧任务可能在代码更新后才开始渲染报告。
    notes = []
    if sensitivity:
        notes.append(f"单个维度评分 ±1 的 {sensitivity['scenarios']} 种情景中，有 {sensitivity['changes']} 种改变第一名")
    if r.ranking and r.ranking[0]["topsis_disagrees"]:
        notes.append("加权分与 TOPSIS 给出的第一名不一致，排序需要重点讨论")
    if notes and not single:
        out.append("**排序不确定性**：" + "；".join(notes) + "。详见第七章。")
    if r.comparison and not single:
        out.append(f"**对排序的意见**：{r.comparison['ranking_view']}")
    if not single:
        out.append("> 排名反映赛道内的相对位置。投资建议及其条件见第六章。")

    # 四、各公司研究摘要
    out.append("## 四、公司研究摘要" if single else "## 四、各公司研究摘要")
    for row in r.ranking:
        name = row["company"]
        a, sheet = r.analyses[name], r.fact_sheets[name]
        scores = "、".join(f"{dim.name} {row['scores'][dim.key]}（置信度{a['scores'][dim.key]['confidence']}）"
                           for dim in sector.dimensions)
        reasons = [f"{dim.name}：{a['scores'][dim.key]['reason']}（证据："
                   + ("、".join(a['scores'][dim.key]['evidence_ids']) or "未提供") + "）"
                   for dim in sector.dimensions]
        metrics = ("**经营测算**：见第六章“财务与单位经济”。" if name == top and name in financials
                   else _metrics_brief(financials.get(name), sector))
        out.append(
            (f"### {name}" if single else f"### {row['rank']}. {name}")
            + f"（加权分 {row['weighted']}，整体置信度{a['confidence']}）\n"
            f"{a['one_liner']}\n\n{a['summary']}\n\n"
            f"**产品与商业模式**\n{_text(a.get('product_business'))}\n\n"
            f"**团队**\n{_text(a.get('team_assessment'))}\n\n"
            f"**经营事实与口径**\n{_text(a.get('operating_assessment'))}\n\n{metrics}\n\n"
            f"**融资与上市状态**\n阶段：{_text(sheet.get('stage'), '未记录')} · 上市状态：{_listing(sheet)}"
            + _valuation_brief(deals.get(name)) + "\n\n"
            f"**AI 原生程度**：{_text(a['ai_native'])}\n\n"
            f"**亮点**\n{_bullets(a['strengths'])}\n\n**担忧**\n{_bullets(a['concerns'])}\n\n"
            f"**尽调问题**\n{_bullets(a['open_questions'])}\n\n"
            f"**评分依据**（{scores}）\n{_bullets(reasons)}"
        )
    excluded = [(n, sheet) for n, sheet in r.fact_sheets.items() if sheet.get("listing_status") == "已上市"]
    if excluded:
        out.append("### 已上市，未纳入投资候选排序\n" + _bullets([
            f"{name}：{_text(sheet['listing_evidence'], '依据未记录')}" for name, sheet in excluded
        ]))

    # 五、共同风险
    out += ["## 五、赛道风险" if single else "## 五、共同风险", _risk_table(d["sector_risks"])]

    # 六、首选公司深入分析
    invest = top in recommended and d.get("action") != "不投"
    out += [
        (f"## 六、深入分析：{top}" if single else f"## 六、首选公司深入分析：{top}") if invest
        else f"## 六、重点分析对象：{top}（本次不建议投资）",
        "### 投资逻辑\n" + _thesis_detail(thesis), d["thesis"],
        "### 财务与单位经济\n" + (_metrics_detail(financials[top], sector) if top in financials
                                 else f"缺少 {top} 可用于测算的经营数据；已披露的经营事实见第四章。"),
        f"### 未来 3–5 年预测\n{_text(d.get('forecast_assessment'), '缺少基础数据与逐年假设，尚不能形成可靠预测。')}",
        "### 交易条件与估值", _trade_decision(d),
        "" if single else f"**重点跟进**：{'、'.join(recommended) or '暂无合适标的'}",
        f"**交易摘要（{top}）**　{_deal_summary(deals.get(top))}",
        _at_a_glance(r, d, top, deals.get(top), thesis), _deal_table(deals.get(top))
        + f"\n\n**估值判断**：{_text(d.get('valuation_view'))}"
        + f"\n\n**资金用途与里程碑**：{_text(d.get('use_of_funds_milestones'))}"
        + "\n\n本次研究材料尚缺优先清算权、董事会席位、反稀释等具体条款。",
        f"### 主要风险与应对\n{_bullets(d['top_pick_risks'])}",
    ]
    out.append(_returns_section(r, d))

    # 七、尽调计划与附录
    out += [
        "## 七、尽调计划与附录",
        f"### 尽调计划\n{_bullets(d['dd_checklist'])}",
        f"### 下一步\n{_bullets(d['next_steps'])}",
        f"### 保留意见\n{_text(d['dissent'], '无')}",
        _methods(r, sector),
    ]
    out += _input_tables(r)
    evidence = [("赛道研究", e) for e in b["evidence"]]
    evidence += [(name, e) for name, sheet in r.fact_sheets.items() for e in sheet["evidence"]]
    if evidence:
        out.append("### 关键事实与引用\n公司披露与独立来源分别标注；测算假设单独列示。\n\n"
                   + _bullets([f"{e['evidence_id']} · {name}：{e['claim']}；{e['source_type']}；"
                               f"时间：{e['as_of'] or '未记录'}；来源：{e['source_ref']}；原文：{e['excerpt']}"
                               + (f"（{e['quote_location']}）" if e.get("quote_location") else "")
                               for name, e in evidence]))
    references = [e["source_ref"] for _, e in evidence]
    references += re.findall(r"\[(?:C\d+|S)-\d+\]", "\n".join(out))
    sources = [("赛道研究", b["sources"])] + [(name, sheet["sources"]) for name, sheet in r.fact_sheets.items()]
    source_blocks = []
    for name, items in sources:
        used = [source for source in items if any(_reference_matches(ref, source) for ref in references)]
        if used:
            source_blocks.append(f"**{name}**\n" + _bullets(used))
    if source_blocks:
        out.append("### 信息来源\n" + "\n\n".join(source_blocks))
    return "\n\n".join(part for part in out if part) + "\n"


# ---------------------------------------------------------------- 报告各节

def _unresolved_section(items: list[dict]) -> str:
    if not items:
        return ""
    rows = [f"| {_cell(u['claim'])} | {_cell(u['source_agent'])} | {_cell(u['problem'])} | {_cell(u['reason'])} |" for u in items]
    return ("### 未解决的核查事项\n以下主张经一轮补查和复查后仍未找到可逐字核对的原文，或修正未获确认；"
            "不能作为投资判断依据，需人工查证后决定保留、更正或删除。\n\n"
            "| 主张 | 所在材料 | 复查结果 | 说明 |\n| --- | --- | --- | --- |\n" + "\n".join(rows))


def _deal_summary(deal: dict | None) -> str:
    deal = deal or {}
    parts = [f"{DEAL_LABELS[key]['label']}：{_fmt(deal.get(key), DEAL_LABELS[key]['unit'], '待补充')}"
             for key in ("post_money_musd", "investment_musd", "ownership", "round_size_musd")]
    return "；".join(parts) + "。拟持股比例按拟投资金额与投后估值计算；具体优先权条款尚缺资料。"


def _valuation_brief(deal: dict | None) -> str:
    if not deal or deal.get("post_money_musd") is None:
        return ""
    text = f" · 本轮投后估值 {_fmt(deal['post_money_musd'], '百万美元')}（交易假设）"
    if deal.get("implied_arr_multiple") is not None:
        text += f"，隐含 ARR 倍数 {_fmt(deal['implied_arr_multiple'], '倍')}"
    if deal.get("implied_run_rate_multiple") is not None:
        text += f"，年化收入运行率倍数 {_fmt(deal['implied_run_rate_multiple'], '倍')}"
    return text


def _deal_table(deal: dict | None) -> str:
    deal = deal or {}
    rows = [f"| {item['label']} | {_fmt(deal.get(key), item['unit'], '待补充' if item['basis'] == '用户提供' else '无法计算')} | {item['basis'].replace('代码计算：', '测算：').replace('用户提供', '交易假设')} |"
            for key, item in DEAL_LABELS.items()]
    return "| 项目 | 数值 | 依据 |\n| --- | --- | --- |\n" + "\n".join(rows)


def _market_section(market: dict) -> str:
    if not market:
        return ""
    period, inputs, sources, metrics = market["period"], market["inputs"], market["sources"], market["metrics"]
    rows = [f"| {MARKET_INPUT_LABELS[key][0]} | {_fmt(value, MARKET_INPUT_LABELS[key][1])} | {_cell(period)} | {_cell(sources[key])} |"
            for key, value in inputs.items()]
    rows += [f"| {item['label']} | {_fmt(metrics[key], item['unit'])} | {_cell(period)} | 测算：{item['formula']} |"
             for key, item in MARKET_METRIC_LABELS.items() if metrics.get(key) is not None]
    text = ("**市场测算依据**\n\n| 测算输入或结果 | 数值与单位 | 期间 | 来源或假设依据 |\n"
            "| --- | --- | --- | --- |\n" + "\n".join(rows))
    missing = [item["label"] for key, item in MARKET_METRIC_LABELS.items() if metrics.get(key) is None]
    if missing:
        text += "\n\n缺少输入、无法计算：" + "、".join(missing) + "。"
    bottom_up, top_down = metrics.get("tam_musd"), metrics.get("tam_top_down_musd")
    if bottom_up and top_down:
        text += f"\n\n自下而上与自上而下 TAM 之比为 {bottom_up / top_down:.2f}，两种口径差异需人工核对。"
    return text + "\n\n不同统计口径、地区和时点不能直接混用；比例为研究假设。"


def _at_a_glance(r: AnalysisResult, d: dict, top: str, deal: dict | None, thesis: list[dict]) -> str:
    """投委会速览：读完应能回答的 6 个问题，全部取自正文已有内容和代码计算结果。"""
    deal = deal or {}
    terms = "；".join(f"{DEAL_LABELS[k]['label']} {_fmt(deal.get(k), DEAL_LABELS[k]['unit'], '待补充')}"
                      for k in ("post_money_musd", "investment_musd", "ownership"))
    returns = {row["scenario"]: row for row in getattr(r, "returns", [])}
    summary = getattr(r, "return_summary", {})
    outcome = "；".join(
        f"{name}：MOIC {_multiple(row)}" + (f"，本基金所得 {_fmt(row['proceeds_musd'], '百万美元')}" if row.get("proceeds_musd") is not None else "")
        for name, row in ((n, returns[n]) for n in ("乐观", "悲观") if n in returns))
    if outcome:
        years = next((row["holding_years"] for row in returns.values() if row.get("holding_years")), None)
        outcome += (f"（后续稀释 {_optional_pct(summary.get('dilution'))}，研究假设"
                    + (f"；持有 {_fmt(years, '年')}" if years else "") + "；假设与公式见第六章和第七章，可改假设重算）")
    facts = "经营数字按公司披露或独立来源标注；交易条件、市场渗透和退出情景中的假设另行说明。引文可定位不代表公司披露已经独立验证"
    rows = [
        ("建议是什么、投多少", f"{d.get('action', '未记录')}（{top}）；{terms}"),
        ("为什么是这家、为什么是现在", "；".join(t["judgment"] for t in thesis[:2]) or GAP),
        ("最可能让我们亏钱的是什么", "；".join(d["top_pick_risks"][:2]) or GAP),
        ("最好和最差的情况", outcome or "缺少本轮估值或退出估值，无法计算"),
        ("哪些是事实、哪些是推断", facts),
        ("接下来要核实什么", "；".join(d["dd_checklist"][:3]) or GAP),
    ]
    return ("**投委会速览**\n\n| 问题 | 回答 |\n| --- | --- |\n"
            + "\n".join(f"| {q} | {_cell(a)} |" for q, a in rows))


def _competitor_table(r: AnalysisResult) -> str:
    rows = [f"| {_cell(c['company'])} | {_cell(c['position'])} | {_cell(c['edge'])} | {_cell(c['gap'])} |"
            for c in (r.comparison or {}).get("per_company", [])]
    if not rows:
        return "对比分析未完成，竞争对手见第二章“参照公司”。"
    return "| 公司 | 在赛道中的位置 | 优势 | 短板 |\n| --- | --- | --- | --- |\n" + "\n".join(rows)


def _comparison_table(r: AnalysisResult) -> str:
    views = {c["company"]: c for c in (r.comparison or {}).get("per_company", [])}
    rows = []
    for row in r.ranking:
        name = row["company"]
        a, view = r.analyses[name], views.get(name, {})
        edge = view.get("edge") or (a["strengths"][0] if a["strengths"] else "")
        gap = view.get("gap") or (a["concerns"][0] if a["concerns"] else "")
        rows.append(f"| {name} | {_cell(a['one_liner'])} | {_cell(edge)} | {_cell(gap)} | {row['weighted']}（置信度{a['confidence']}） |")
    return "| 公司 | 核心产品及客户 | 优势和依据 | 主要短板 | 加权分与置信度 |\n| --- | --- | --- | --- | --- |\n" + "\n".join(rows)


def _score_table(r: AnalysisResult, sector: Sector) -> str:
    head = "| 排名 | 公司 | 加权分 | " + " | ".join(d.name for d in sector.dimensions) + " |"
    sep = "|" + " --- |" * (3 + len(sector.dimensions))
    rows = [f"| {row['rank']} | {row['company']} | {row['weighted']} | "
            + " | ".join(str(row["scores"][d.key]) for d in sector.dimensions) + " |" for row in r.ranking]
    return "\n".join([head, sep, *rows])


def _listing(sheet: dict) -> str:
    if "listing_status" not in sheet:
        return "未记录"
    status = sheet["listing_status"] or "未确认"  # “待核实”在对外版本中被清理为空
    evidence = sheet.get("listing_evidence", "")
    return f"{status}（{evidence}）" if evidence else status


def _metric_split(financial: dict, sector: Sector) -> tuple[list[str], list[str]]:
    """展示赛道适用指标和用户已提供输入的其他指标；“无法计算”只列赛道适用指标。"""
    metrics = financial["metrics"]
    order = list(sector.metrics) + [key for key in metrics if key not in sector.metrics]
    shown = [key for key in order if metrics.get(key) is not None]
    missing = [key for key in sector.metrics if metrics.get(key) is None]
    return shown, missing


def _metrics_brief(financial: dict | None, sector: Sector) -> str:
    if not financial:
        return "**经营测算**：未提供具备统一单位、期间和来源说明的结构化输入，暂不计算。"
    shown, missing = _metric_split(financial, sector)
    text = (f"**经营测算**：统计期间 {financial['period']}；前期 {financial.get('previous_period') or '未提供'}。"
            "原始数据及来源见第七章。")
    if shown:
        text += "\n\n| 指标 | 计算值 |\n| --- | --- |\n" + "\n".join(
            f"| {METRIC_LABELS[key]['label']} | {_fmt(financial['metrics'][key], METRIC_LABELS[key]['unit'])} |" for key in shown)
    if missing:
        text += "\n\n本赛道适用但缺少输入、无法计算：" + "、".join(METRIC_LABELS[key]["label"] for key in missing) + "。"
    return text


def _metrics_detail(financial: dict, sector: Sector) -> str:
    shown, missing = _metric_split(financial, sector)
    inputs, sources = financial["inputs"], financial["sources"]
    period = financial["period"] + (f"；前期 {financial['previous_period']}" if financial.get("previous_period") else "")
    rows = []
    for key in shown:
        used = [k for k in METRIC_LABELS[key]["inputs"] if k in inputs]
        notes = list(dict.fromkeys(sources[k] for k in used))
        rows.append(f"| {METRIC_LABELS[key]['label']} | {_fmt(financial['metrics'][key], METRIC_LABELS[key]['unit'])} | "
                    + "；".join(f"{INPUT_LABELS[k]} {_fmt(inputs[k], '')}" for k in used)
                    + f" | {_cell('；'.join(notes))} |")
    text = (f"统计期间：{period}。数据来源及统计口径随表列示。\n\n"
            "| 指标 | 测算结果 | 同期原始数据 | 来源说明 |\n| --- | --- | --- | --- |\n" + "\n".join(rows)) if rows else "已提供的输入不足以计算任何指标。"
    if missing:
        text += "\n\n本赛道适用但缺少输入、无法计算：" + "、".join(METRIC_LABELS[key]["label"] for key in missing) + "。"
    return text


def _risk_table(risks: list) -> str:
    rows = []
    for item in risks:
        if isinstance(item, str):  # 旧版结果
            item = {"risk": item}
        if _text(item.get("risk"), ""):
            rows.append(f"| {_cell(item.get('risk'))} | {_cell(item.get('impact'))} | {_cell(item.get('evidence_gap'))} | {_cell(item.get('response'))} |")
    if not rows:
        return "- —"
    return "| 风险 | 可能影响 | 目前依据及信息缺口 | 建议核实或应对 |\n| --- | --- | --- | --- |\n" + "\n".join(rows)


def _thesis_items(decision: dict) -> list[dict]:
    items = []
    for item in decision.get("top_pick_thesis", []):
        if isinstance(item, str):  # 旧版结果
            item = {"judgment": item}
        if _text(item.get("judgment"), ""):
            items.append(item)
    return items


def _thesis_detail(items: list[dict]) -> str:
    return "\n\n".join(
        f"**{i}. {t['judgment']}**\n- 证据：{_text(t.get('evidence'))}\n"
        f"- 为什么构成投资机会：{_text(t.get('why_opportunity'))}\n- 什么情况会推翻：{_text(t.get('disproof'))}"
        for i, t in enumerate(items, 1)) or GAP


def _returns_section(r: AnalysisResult, d: dict) -> str:
    summary = getattr(r, "return_summary", {}) or {}
    first = r.returns[0] if r.returns else {}
    years = first.get("holding_years")
    if "dilution" in first:
        source = summary.get("dilution_source", "来源未记录")
        dilution = f"{first['dilution']:.0%}（{'研究假设' if source == '系统默认' else '交易假设'}）"
    else:
        dilution = "后续稀释比例未记录"
    investment = first.get("investment_musd")
    table = ("| 情景 | 假设退出股权价值（百万美元） | 情景权重假设 | MOIC | 简化年化 IRR | 本基金退出所得（百万美元） |\n"
             "| --- | --- | --- | --- | --- | --- |" + "".join(
                 f"\n| {s['scenario']} | {s['exit_valuation_musd'] if s['exit_valuation_musd'] is not None else '无法估计'} | "
                 f"{_optional_pct(ret.get('probability'))} | {_multiple(ret)} | {_optional_pct(ret.get('annualized_irr'), '无法计算')} | "
                 + (_fmt(ret.get("proceeds_musd"), "", "无法计算") if investment else "未提供拟投资金额") + " |"
                 for s, ret in zip(d["exit_scenarios"], r.returns)))
    details = "\n\n".join(f"**{s['scenario']}情景**：{s['condition']}；退出方式：{s['exit_path']}；"
                          f"估值依据与假设：{_text(s.get('valuation_basis'), '未记录')}" for s in d["exit_scenarios"])
    expected = ("情景权重下的期望回报倍数："
                + (f"{summary['expected_multiple']} 倍。" if summary.get("expected_multiple") is not None
                   else "无法计算，需完整的用户情景权重、本轮投后估值及各情景退出估值。"))
    analysis = _returns_analysis_section(getattr(r, "returns_analysis", {}) or {})
    return (f"### 退出情景与回报\n持有年数：{f'{years:g} 年（研究假设）' if years else '未提供，不计算简化年化 IRR'}；"
            f"后续持股稀释：{dilution}。\n\n{table}\n\n{details}\n\n{expected}"
            + (f"\n\n{analysis}" if analysis else "")
            + "\n\n> 退出估值为公司 100% 股权价值的情景假设；未计税费，公式与适用条件见第七章。"
            + ("" if (getattr(r, "returns_analysis", {}) or {}).get("preference") else "未提供优先清算条款，回报未计优先清算权。"))


def _returns_analysis_section(a: dict) -> str:
    """敏感性、扣费后回报与优先清算权（方法改写自 Anthropic returns-analysis Skill，Apache-2.0）。"""
    if not a:
        return ""
    names = list(a["by_years"])
    head = "| | " + " | ".join(names) + " |\n|" + " --- |" * (len(names) + 1)
    years = "\n".join(
        f"| 持有 {y:g} 年 | " + " | ".join(
            f"{a['by_years'][n][i]['multiple']} 倍 / {_optional_pct(a['by_years'][n][i]['irr'], '—')}" for n in names) + " |"
        for i, y in enumerate(a["years_grid"]))
    dilution = "\n".join(
        f"| 稀释 {d:.0%} | " + " | ".join(f"{a['by_dilution'][n][i]['multiple']} 倍" for n in names) + " |"
        for i, d in enumerate(a["dilution_grid"]))
    parts = ["**敏感性：持有年数 × 退出情景**（MOIC / 简化年化 IRR）\n\n" + head + "\n" + years,
             "**敏感性：后续稀释 × 退出情景**（MOIC）\n\n" + head + "\n" + dilution]
    if a.get("net"):
        fees = a["fees"]
        parts.append(
            f"**扣除管理费和 carry 后**（单项目近似：年管理费 {fees['management_fee']:.0%} × 持有年数，"
            f"carry {fees['carry']:.0%} 按扣除本金与管理费后的利润计提；不等于基金层面的 LP 净回报）\n\n"
            "| 情景 | 扣费前所得 | 管理费 | carry | 扣费后所得 | 扣费后倍数 |\n| --- | --- | --- | --- | --- | --- |\n"
            + "\n".join(f"| {x['scenario']} | {_fmt(x['gross_proceeds_musd'], '百万美元')} | {_fmt(x['management_fee_musd'], '百万美元')} | "
                        f"{_fmt(x['carry_musd'], '百万美元')} | {_fmt(x['net_proceeds_musd'], '百万美元')} | {x['net_multiple']} 倍 |"
                        for x in a["net"]))
    pref = a.get("preference")
    if pref:
        parts.append(
            f"**优先清算权**（{pref['multiple']:g} 倍不参与分配型，与其他优先股同顺位，同顺位清算额合计 "
            f"{_fmt(pref['stack_musd'], '百万美元')}；按普通股分配与按优先权受偿取较高者）\n\n"
            "| 情景 | 按普通股分配 | 按优先权受偿 | 本基金所得 | 倍数 | 是否转换 |\n| --- | --- | --- | --- | --- | --- |\n"
            + "\n".join(f"| {x['scenario']} | {_fmt(x['as_common_musd'], '百万美元')} | {_fmt(x['as_preferred_musd'], '百万美元')} | "
                        f"{_fmt(x['proceeds_musd'], '百万美元')} | {x['multiple']} 倍 | {'转换为普通股' if x['converts'] else '行使优先权'} |"
                        for x in pref["rows"]))
    return "\n\n".join(parts)


def _methods(r: AnalysisResult, sector: Sector) -> str:
    parts = ["### 方法与假设", "**评分维度与权重**\n\n| 维度 | 权重 | 评分标准（1–10 分） |\n| --- | --- | --- |\n"
             + "\n".join(f"| {d.name} | {d.weight:.0%} | {_cell(d.rubric)} |" for d in sector.dimensions)]
    if len(r.ranking) > 1:
        parts.append("**TOPSIS 交叉检查**：计算每家公司到各维度最优、最差理想点的相对距离（0–1，越高越好），用来检查加权分排序。\n\n"
                     "| 公司 | 加权分 | TOPSIS |\n| --- | --- | --- |\n"
                     + "\n".join(f"| {row['company']} | {row['weighted']} | {row['topsis']} |" for row in r.ranking)
                     + ("\n\n加权分和 TOPSIS 给出的第一名不一致。" if r.ranking[0]["topsis_disagrees"] else ""))
    s = getattr(r, "sensitivity", {})
    if s and len(r.ranking) > 1:
        parts.append(f"**排名敏感性**：逐次将一家公司的一个维度加或减 1 分，共 {s['scenarios']} 种情景；"
                     f"其中 {s['changes']} 种改变第一名。该结果依赖现有评分，不代表投资成功概率。"
                     + ("\n\n" + _bullets([f"{e['company']}的{e['dimension']}分数 {e['delta']:+d}，第一名变为 {e['top_pick']}"
                                           for e in s["examples"]]) if s["examples"] else ""))
    ranges = getattr(r, "score_ranges", {})
    if ranges:
        intro = ("**信息缺口下的评分假设范围**：将低置信度或没有关联独立来源的维度按 1–10 分变化，其他维度保持原分数。"
                 "这是情景范围，不是统计置信区间或投资成功概率；来源性质由模型识别，来源关联也不等于事实已核实。")
        if all(len(item["uncertain_dimensions"]) == len(sector.dimensions) for item in ranges.values()):
            parts.append(intro + "\n\n所有公司的全部维度都缺少已匹配原文的独立来源或置信度低，假设范围均为 1–10 分，"
                         "不具区分度；当前排序主要依赖模型对公司口径资料的判断，需补充独立来源。")
        else:
            parts.append(intro + "\n\n" + _bullets([f"{name}：{item['min']}–{item['max']} 分；信息缺口涉及维度："
                                                    + ("、".join(item['uncertain_dimensions']) or "无") for name, item in ranges.items()]))
    if r.pairwise:
        parts.append("**两两比较与顺序影响**\n" + "\n".join(
            f"- {' / '.join(p['companies'])}：正序 {p['forward']['winner'] if p['forward'] else '未完成'}；"
            f"反序 {p['reverse']['winner'] if p['reverse'] else '未完成'}"
            + ("（顺序影响结果）" if p['order_disagrees'] else "")
            + (f"。正序依据：{p['forward']['reason']}" if p['forward'] else "")
            + (f"；反序依据：{p['reverse']['reason']}" if p['reverse'] else "")
            for p in r.pairwise))
    parts.append("**回报与交易测算口径**\n" + _bullets([
        "退出估值为退出时公司 100% 股权价值；由企业价值推算时须扣除净债务或加回净现金。",
        "MOIC = 退出股权价值 × (1 − 后续稀释) ÷ 本轮投后估值；本基金退出所得 = MOIC × 拟投资金额。",
        "简化年化 IRR = MOIC^(1/持有年数) − 1，仅适用于期初投入、期末一次退出且无中间现金流；未计优先清算权、费用或税费。"
        "下行并购情景中，优先清算权可能使实际回收高于按持股比例计算的结果。",
        "期望倍数 = Σ 用户权重 × MOIC；权重是用户主观假设，不是经验证的成功概率，也不计算概率加权 IRR。",
        "拟持股比例 = 拟投资金额 ÷ 投后估值；投前估值 = 投后估值 − 本轮融资额；隐含 ARR 倍数 = 投后估值 ÷ (MRR × 12)。",
    ]))
    parts.append("**经营指标口径**\n" + _bullets([
        "本赛道采用的经营指标：" + "、".join(METRIC_LABELS[key]["label"] for key in sector.metrics) + "。",
        "MRR × 12 只是一致经常性收入口径下的年化值，不等于经审计的全年确认收入；LTV 为稳定流失率假设下的简化估算。",
        "SaaS 指标不直接套用硬件销售，行业基准不是跨赛道的硬性投资门槛。",
    ]))
    return "\n\n".join(parts)


def _input_tables(r: AnalysisResult) -> list[str]:
    tables = []
    if r.deal_metrics or r.financial_metrics or r.market_metrics:
        tables.append("### 交易与测算依据 [U-1]\n交易条件及测算假设来自本次研究输入，来源说明随表列示。")
    market = getattr(r, "market_metrics", {})
    if market:
        tables.append(f"### 赛道市场测算原始输入\n期间：{market['period']}，币种 USD，金额单位百万美元；比例使用小数。\n\n"
                      "| 输入项 | 数值 | 来源说明 |\n| --- | --- | --- |\n"
                      + "\n".join(f"| {MARKET_INPUT_LABELS[key][0]} | {value} | {_cell(market['sources'][key])} |"
                                  for key, value in market["inputs"].items()))
    for name, financial in getattr(r, "financial_metrics", {}).items():
        tables.append(f"### {name}测算原始输入\n期间：{financial['period']}；前期：{financial.get('previous_period') or '未提供'}；"
                      "币种 USD，金额单位百万美元；比例使用小数。\n\n"
                      "| 输入项 | 数值 | 来源说明 |\n| --- | --- | --- |\n"
                      + "\n".join(f"| {INPUT_LABELS.get(key, key)} | {value} | {_cell(financial['sources'][key])} |"
                                  for key, value in financial["inputs"].items()))
    return tables


def format_run_cost(r: AnalysisResult) -> str:
    known = [value for value in r.costs_usd.values() if value is not None]
    missing = sum(value is None for value in r.costs_usd.values())
    search_cost = r.search_usage.get("cost_usd")
    if search_cost is not None:
        known.append(search_cost)
    elif r.search_usage.get("calls"):
        missing += 1
    # The governor also counts individual receipts from roles whose totals are incomplete.
    total = sum(known)
    if missing:
        control = getattr(r, "run_control", None) or {}
        total = max(total, control.get("reported_cost_usd") or 0.0)
    amount = f"已返回费用合计 ${total:.4f}" if known or total else "接口未返回费用"
    return f"{amount}，{missing} 项费用未知" if missing else amount


def render_audit(r: AnalysisResult) -> str:
    b = r.sector_brief
    rate = "未完成" if r.fact_check_pass_rate is None else f"{r.fact_check_pass_rate:.0%}"
    coverage = "未完成" if r.checked_source_coverage is None else f"{r.checked_source_coverage:.0%}"
    out = [f"# 内部研究审计：{r.sector_name}", f"报告状态：{r.report_status}",
           "本文件保留被删除或更正的事实、核查状态及运行信息，不属于正式投资报告。"
           "实际读取原文、修正前后材料与结构化结果见同目录 result.json（中断时为 partial-result.json）。"]
    if getattr(r, "main_agents", None):
        out.append(f"### 主Agent数量\n本次使用 {len(r.main_agents)} 个主Agent：" + "、".join(r.main_agents)
                   + "。各阶段复用原上下文；阶段调用次数不等于Agent数量。")
    if getattr(r, "handoffs", None):
        out.append("### 材料交接\n| 接收主Agent | 阶段 | 状态 | 输入材料版本 |\n| --- | --- | --- | --- |\n"
                   + "\n".join(f"| {h['agent']} | {h['phase']} | {h['status']} | "
                               + "、".join(f"{ref['name']} v{ref['version']} ({ref['sha256'][:8]})" for ref in h['inputs'])
                               + " |" for h in r.handoffs))
    if getattr(r, "research_plan", None):
        out.append("### 主 Agent 的研究议程\n```json\n" + json.dumps(r.research_plan, ensure_ascii=False, indent=2) + "\n```")
    if getattr(r, "research_state", None):
        from .inquiry import process_digest
        state = r.research_state
        steps = process_digest(r)
        out.append("### 问题驱动研究\n" + state.get("selection_basis", "")
                   + f"\n\n研究过程：初始问题 {steps['initial_questions']} 个，完成 {steps['followups_completed']} 轮定向补查。"
                   + "\n\n当前待证伪假设：" + state.get("working_hypothesis", "")
                   + "\n\n停止追加研究的原因：" + (state.get("stop_reason") or "尚未结束")
                   + "\n\n| 问题 | 状态 | 独立复核 | 判断影响与信息缺口 |\n| --- | --- | --- | --- |\n"
                   + "\n".join(f"| {q['question_id']}：{_cell(q['question'])} | {q['status']} | "
                       f"{q.get('audit_verdict', 'not_reviewed')} | "
                       + _cell(q.get("decision_impact", "") + "；" + "；".join(q.get("missing_information", []))) + " |"
                       for q in state.get("questions", [])))
        out.append("完整补查任务、材料版本、判断变化及方法复核保存在 result.json 的 research_iterations 和 research_methods；不改变投资报告原有七部分结构。")
    if r.debate:
        out.append("### 多空辩论\n" + "\n\n".join(
            f"#### {item['name']} · 第 {item['round']} 轮\n{item['thesis']}\n\n{_bullets(item['arguments'])}"
            + (f"\n\n反驳：\n{_bullets(item['rebuttals'])}" if item.get("rebuttals") else "")
            + f"\n\n改变观点所需证据：{item['would_change_mind']}" for item in r.debate))
    if getattr(r, "research_iterations", None):
        out.append("### 追加研究执行记录\n| 阶段 | 轮次 | 状态 | 排除的证据（审计判为不实或无法核实） |\n| --- | --- | --- | --- |\n"
                   + "\n".join(f"| {row['kind']} | {row.get('round', '-')} | {row['status']} | "
                               f"{_cell('、'.join(row.get('excluded_evidence_ids', [])))} |" for row in r.research_iterations))
    if getattr(r, "prompt_versions", None):
        out.append("### 提示词版本\n比较不同报告前先核对版本；版本不同的结果不能直接归因于同一提示词。\n\n| 提示词 | 内容哈希 |\n| --- | --- |\n"
                   + "\n".join(f"| {name} | {digest} |" for name, digest in sorted(r.prompt_versions.items())))
    if getattr(r, "rejection_reviews", None):
        out.append("### 退回必要性复核\n记录引用写法与成稿金额的局部复核。详细原文、判断和费用见 result.json。\n\n"
                   "| 范围 | 模型 | 状态 | 撤销项数 | 输入/输出 token | 费用（美元） |\n| --- | --- | --- | --- | --- | --- |\n"
                   + "\n".join(f"| {'成稿金额' if row.get('scope') == 'report_amount' else '引用字段'} | "
                       f"{row['model']} | {row['status']} | {len(row['waived'])} | "
                       f"{row['usage'][0]}/{row['usage'][1]} | "
                       + (f"{row['cost_usd']:.6f}" if row['cost_usd'] is not None else "未知") + " |"
                       for row in r.rejection_reviews))
    if getattr(r, "task_trace", None):
        out.append("### 并行任务与依赖\n开始时间相对所在并行批次的起点；开始时间加耗时有重叠的任务是同时执行的。\n\n"
                   "| 任务 | 前置任务 | 状态 | 开始（秒） | 耗时（秒） |\n| --- | --- | --- | --- | --- |\n"
                   + "\n".join(f"| {t['task']} | {', '.join(t['depends_on']) or '-'} | {t['status']} | "
                               f"{t.get('start', '未记录')} | {t['seconds']} |" for t in r.task_trace))
    coverage = getattr(r, "audit_coverage", None)
    if coverage:
        gaps = {gap["material"]: gap for gap in coverage.get("gaps", [])}
        out.append("### 研究核查覆盖\n必查证据由代码按事实类型选出（估值、融资、收入、客户优先），不由核查模型自行抽样；"
                   + ("覆盖不足时不能作为正式报告。" if coverage.get("enforced") else "演示模式只记录清单，不作为发布门槛。")
                   + "\n\n| 材料 | 要求条数 | 已核查条数 | 必查证据 | 缺口 |\n| --- | --- | --- | --- | --- |\n"
                   + "\n".join(
                       f"| {name} | {need['min_claims']} | {coverage.get('checked', {}).get(name, 0)} | "
                       f"{'、'.join(need['evidence_ids']) or '底稿没有结构化证据'} | "
                       + ("；".join(filter(None, (
                           f"少 {gaps[name]['missing_claims']} 条" if gaps[name]["missing_claims"] else "",
                           "未核查 " + "、".join(gaps[name]["missing_evidence"]) if gaps[name]["missing_evidence"] else "")))
                          if name in gaps else "无") + " |"
                       for name, need in coverage.get("required", {}).items()))
    if getattr(r, "memory_stats", None):
        out.append("### 跨任务研究记忆\n" + json.dumps(r.memory_stats, ensure_ascii=False))
    if getattr(r, "report_pages", None):
        out.append(f"### 报告篇幅\n实际 PDF {r.report_pages} 页，上限 {r.report_max_pages} 页（含来源与附录）。")
    if getattr(r, "quality_repairs", None):
        out.append("### 发布退回与定向修复\n| 轮次 | 责任主Agent及复查 | 状态 | 问题 |\n| --- | --- | --- | --- |\n"
            + "\n".join(f"| {row['round']} | {'、'.join(row['actions']) or '修复额度已用完'} | {row['status']} | "
                + "；".join(f"{item['issue_id']} [{item['owner']}] {item['location']}：{item['message']}" for item in row['issues'])
                + " |" for row in r.quality_repairs)
            + "\n\n每轮的问题定位、材料版本、已关闭问题及独立复查记录保存在 result.json。")
    claims = r.fact_check.get("claims") or []
    if claims:
        out.append(
            f"### 关键主张抽查（独立证据支持率 {rate}，原文来源标注率 {coverage}）\n"
            "独立证据支持率以全部抽查主张为分母，仅“属实”计入分子；公司口径单独列示。"
            "这两项只覆盖模型选出的主张；来源标注由核查员识别，不代表来源真实有效或报告全部事实的准确率。"
            f"\n\n{r.fact_check['summary']}\n\n| 结论 | 主张 | 出处 | 原文来源与证据 ID | 证据或更正 |\n| --- | --- | --- | --- | --- |"
            + "".join(
                f"\n| {c['verdict']} | {c['claim']} | {c['source_agent']} | {c['original_source'] or '未标注'}"
                f"；{', '.join(c.get('evidence_ids', [])) or '无证据 ID'} | {c['correction'] if c['verdict'] == '不实' and c['correction'] else c['evidence']}"
                + (f"；核查来源：{c['verification_source']}；摘录：{c['verification_excerpt']}；匹配：{c.get('verification_status', '未执行')}"
                   if c.get("verification_source") else "")
                + (f"；模型原判：{c['model_verdict']}" if c.get("model_verdict", c["verdict"]) != c["verdict"] else "")
                + " |" for c in claims
            )
        )
    if r.errors:
        out.append("### 未完成的智能体\n" + _bullets([f"{k}：{v}" for k, v in r.errors.items()]))
    if r.report_checks:
        out.append("### 成稿一致性检查\n" + _bullets(r.report_checks))
    if r.report_warnings:
        out.append("### 成稿提醒（不阻止发布）\n" + _bullets(r.report_warnings))
    if r.memo_reviews:
        out.append("### 正文复核\n已按实际回报统一正文；复核前后材料与固定退出情景见 result.json 的 memo_reviews。")
    if r.decision_fact_check:
        out.append("### 主席结论核查\n" + _bullets([
            f"[{c['verdict']}] {c['claim']}：{c['evidence']}" for c in r.decision_fact_check["claims"]
        ]))
        reviews = r.decision_fact_check.get("reasoning_checks", [])
        if reviews:
            out.append("### 核心投资推论复核\n" + _bullets([
                f"{row['target_id']} [{row['verdict']}] {row['reason']}；依据：{'、'.join(row['evidence_ids'])}"
                for row in reviews]))
    if r.repairs:
        out.append("### 已退回修正\n" + _bullets([
            f"{item['source_agent']}：" + "；".join(
                f"{claim.get('claim_id', '')} {claim['claim']}（核查建议：{claim['correction'] or '删除或待核实'}）"
                for claim in item['claims']) for item in r.repairs
        ]) + "\n\n修正前后的完整材料保存在 result.json 的 repairs 字段。")
    followups = [check for history in r.check_history for check in history['result'].get('repair_checks', [])]
    if followups:
        out.append("### 不实主张复查\n" + _bullets([
            f"{check['claim_id']}：{check['status']}；{check['reason']}" for check in followups
        ]))
    out.append(_unresolved_section(getattr(r, "unresolved_checks", [])))
    findings = [f for check in getattr(r, "evidence_checks", {}).values() for f in check["findings"]]
    if findings:
        out.append("### 证据核对未通过的数字句\n" + _bullets([
            f"[{f['material']}] {f['sentence']}（{f['problem']}：{'、'.join(f['numbers'])}）" for f in findings]))
    out += _input_tables(r)
    evidence = [("赛道研究", e) for e in b.get("evidence", [])]
    evidence += [(name, e) for name, sheet in r.fact_sheets.items() for e in sheet.get("evidence", [])]
    if evidence:
        out.append("### 结构化证据\n来源编号匹配只表示存在对应来源条目，摘录和主张仍须核查。\n\n" + _bullets([
            # 研究阶段中断时证据尚未编号，部分结果也要能渲染
            f"{e.get('evidence_id', '未编号')} · {name}：{e['claim']}；{e['source_type']}；时间：{e['as_of']}；"
            f"来源：{e['source_ref'] or '未提供'}（{'已关联' if e.get('source_linked') else '未关联'}）"
            + (f"；摘录：{e['excerpt']}" if e['excerpt'] else "；未提供原文摘录")
            + (f"（{e['quote_location']}）" if e.get("quote_location") else "")
            + (f"；字段问题：{'；'.join(e['field_issues'])}" if e.get("field_issues") else "")
            + f"；同源原文匹配：{'通过' if e.get('excerpt_verified') else '未通过或未执行'}"
            for name, e in evidence
        ]))
    sources = ["**赛道研究**\n" + _bullets(b.get("sources", []))]
    sources += [f"**{name}**\n" + _bullets(sheet.get("sources", [])) for name, sheet in r.fact_sheets.items()]
    out.append("### 信息来源\n" + "\n\n".join(sources))
    out.append(f"### 运行指标\n- 耗时：{r.elapsed_seconds} 秒\n- 格式失败次数：{sum(r.format_failures.values())}"
               f"\n- 搜索次数：{r.search_usage.get('calls', 0)}\n- 费用：{format_run_cost(r)}")
    control = getattr(r, "run_control", None)
    if control:
        limit = control.get("timeout_seconds")
        budget = control.get("retry_budget")
        cap = control.get("cost_budget_usd")
        out.append("### 运行管控\n" + _bullets([
            "任务总时限：" + (f"{limit / 60:g} 分钟" if limit else "未设置"),
            f"接口重试：已用 {control.get('retries', 0)} 次，" + ("未设总预算" if budget is None else f"总预算 {budget} 次"),
            f"外部请求：{control.get('requests', 0)} 次；复用本轮相同的搜索或网页读取 {control.get('reused_requests', 0)} 次；"
            f"按供应商配额等待 {control.get('rate_limit_waits', 0)} 次",
            "费用上限：" + (f"{cap:.2f} 美元（只统计接口返回了费用的请求，已计 {control.get('reported_cost_usd', 0):.4f} 美元）"
                        if cap else "未设置"),
            *([f"提前停止：{control['stop_reason']}"] if control.get("stop_reason") else []),
        ]))
    if r.usage:
        agent_seconds = getattr(r, "agent_seconds", {})
        out.append("### 角色用量与耗时\n各角色可能并行运行，耗时不能直接相加。\n\n"
                   "| 角色 | 输入 token | 输出 token | 耗时（秒） | 费用（美元） |\n| --- | --- | --- | --- | --- |"
                   + "".join(f"\n| {agent} | {tokens[0]} | {tokens[1]} | {agent_seconds.get(agent, '未记录')} | "
                             + (f"{r.costs_usd[agent]:.4f}" if r.costs_usd.get(agent) is not None else "未知") + " |"
                             for agent, tokens in r.usage.items()))
    checks = {"研究核查": r.fact_check, "主席核查": r.decision_fact_check, "复查历史": r.check_history}
    out.append("### 原始核查记录\n```json\n" + json.dumps(checks, ensure_ascii=False, indent=2) + "\n```")
    reads = [{"agent": agent, **{key: value for key, value in record.items() if key != "text"},
              "text_chars": len(record.get("text", ""))}
             for agent, records in r.source_reads.items() for record in records]
    if reads:
        out.append("### 读取来源台账\n```json\n" + json.dumps(reads, ensure_ascii=False, indent=2) + "\n```")
    return "\n\n".join(out) + "\n"


_INTERNAL_STATUS = re.compile(r"无法核实|待核实|待核验|待核查|未核实|未独立核实|未经核实|未能核实")
# 仅去除模型叙述中的轮次引导语；不删除其后的观点、否定词或引用。
_ROUND_LEAD = re.compile(r"(?m)(^|(?<=[。！？!?；;]))([ \t]*(?:#{1,6}[ \t]+)?(?:\*\*)?)"
                         r"(?:经过|经|在|通过)?(?:第?[一二三四五两数多\d]+轮)(?:的)?"
                         r"(?:反证|交锋|辩论|反驳|质疑)(?:之?后|结束后)[ \t]*[，、,:：]?[ \t]*")
_DEBATE_ATTRIBUTION = re.compile(
    r"(?m)(^|(?<=[。！？!?；;]))([ \t]*)(?:[正反]方认为[ \t]*[，,:：]?|"
    r"(?:第[一二三四五\d]+轮(?:质疑|反驳)|[正反]方观点|起始判断)[ \t]*[：:])[ \t]*")
# 用户输入与代码计算结果不是模型叙述，不按模型主张清理。
_USER_DATA = {"companies", "sector_name", "financial_metrics", "market_metrics", "deal_metrics", "returns", "return_summary"}


def _public_view(original: AnalysisResult) -> AnalysisResult:
    """只清理已知问题的原样残留；改写后的语义依赖上游修正与复查。"""
    company_names = set(original.companies)
    checks = [original.fact_check, original.decision_fact_check]
    resolved = {c["claim"].strip().rstrip("。；;.!?？！") for check in checks for c in check.get("claims", [])
                if c["verdict"] in ("属实", "公司口径") and c.get("excerpt_verified")}
    checks += [item.get("result", {}) for item in original.check_history]
    checks += [{"claims": item["claims"]} for item in original.repairs]
    rejected = [c["claim"].strip().rstrip("。；;.!?？！") for check in checks for c in check.get("claims", [])
                if c["verdict"] in ("不实", "无法核实") and c["claim"].strip()
                and c["claim"].strip().rstrip("。；;.!?？！") not in resolved]

    def clean(value, *, prose=False):
        if isinstance(value, str):
            if not prose and value in company_names:
                return value  # Identifiers also link the decision and ranking to their company records.
            sentences = re.split(r"(?<=[。！？!?；;\n])", value)
            kept = "".join(sentence for sentence in sentences if not _INTERNAL_STATUS.search(sentence)
                           and not any(claim and claim in sentence for claim in rejected)).strip()
            if prose:
                kept = _ROUND_LEAD.sub(r"\1\2", kept)
                kept = _DEBATE_ATTRIBUTION.sub(r"\1\2一种待检验的解释是，", kept)
            return kept.strip()
        if isinstance(value, list):
            return [item for value_item in value if (item := clean(value_item, prose=prose)) != ""]
        if isinstance(value, dict):
            return {key: clean(item, prose=prose and key not in (
                "sources", "evidence_ids", "excerpt", "company", "company_name", "top_pick", "recommended", "scenario"))
                    for key, item in value.items()}
        return value

    r = deepcopy(original)
    for key, value in vars(r).items():
        if key not in _USER_DATA and key != "unresolved_checks":
            setattr(r, key, clean(value, prose=key in ("decision", "analyses", "comparison", "sector_brief")))
    for raw, sheet in zip([original.sector_brief, *original.fact_sheets.values()],
                          [r.sector_brief, *r.fact_sheets.values()]):
        sheet["evidence"] = [clean(e) for e in raw.get("evidence", [])
                             if not _INTERNAL_STATUS.search(json.dumps(e, ensure_ascii=False))
                             and clean(e["claim"]) and e["source_ref"] and e["excerpt"]
                             and e.get("source_linked") and e.get("excerpt_verified")]
    evidence_ids = {e["evidence_id"] for sheet in (r.sector_brief, *r.fact_sheets.values()) for e in sheet["evidence"]}
    for analysis in r.analyses.values():
        for score in analysis["scores"].values():
            score["evidence_ids"] = [key for key in score["evidence_ids"] if key in evidence_ids]
    reads = [record for records in original.source_reads.values() for record in records]
    read_urls = {url for record in reads for url in record.get("urls", [])}
    read_urls.update(record["url"] for record in reads if record.get("url"))
    read_names = {record["name"] for record in reads if record.get("name")}
    allowed_labels, all_labels = set(), set()
    for sheet in (r.sector_brief, *r.fact_sheets.values()):
        accepted = []
        for source in sheet["sources"]:
            labels = set(re.findall(r"\[(?:C\d+|S)-\d+\]", source))
            all_labels.update(labels)
            has_evidence = any(_reference_matches(e["source_ref"], source) for e in sheet["evidence"])
            if has_evidence or any(url in source for url in read_urls) or any(name in source for name in read_names):
                accepted.append(source)
                allowed_labels.update(labels)
        sheet["sources"] = accepted
    rejected_labels = all_labels - allowed_labels
    if rejected_labels:
        def remove_unread_citations(value):
            if isinstance(value, str):
                return "".join(sentence for sentence in re.split(r"(?<=[。！？!?；;\n])", value)
                               if not rejected_labels.intersection(re.findall(r"\[(?:C\d+|S)-\d+\]", sentence))).strip()
            if isinstance(value, list):
                return [item for entry in value if (item := remove_unread_citations(entry)) != ""]
            if isinstance(value, dict):
                return {key: remove_unread_citations(entry) for key, entry in value.items()}
            return value
        for key in ("sector_brief", "fact_sheets", "analyses", "comparison", "debate", "decision"):
            setattr(r, key, remove_unread_citations(getattr(r, key)))
    return r


def _reference_matches(reference: str, source: str) -> bool:
    labels = set(re.findall(r"\[[A-Za-z0-9_-]+\]", reference))
    if labels.intersection(re.findall(r"\[[A-Za-z0-9_-]+\]", source)):
        return True
    return bool(reference) and reference in source


def _multiple(ret: dict) -> str:
    if ret["multiple"] is not None:
        return f"{ret['multiple']} 倍"
    missing = []
    if not ret["post_money_musd"]:
        missing.append("本轮估值")
    if ret["exit_valuation_musd"] is None:
        missing.append("退出估值")
    return "待补充：" + "、".join(missing)


def _optional_pct(value: float | None, missing: str = "未提供") -> str:
    return missing if value is None else f"{value:.1%}"


def _fmt(value: float | None, unit: str, missing: str = "无法计算") -> str:
    if value is None:
        return missing
    if unit == "比例":
        if 0 < abs(value) < 0.01:
            return f"{value * 100:.3g}%"
        return f"{value:.1%}"
    number = f"{value:,.0f}" if abs(value) >= 10000 else f"{value:.4g}"
    return f"{number} {unit}".strip()
