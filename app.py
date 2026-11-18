"""网页入口：基础研究优先，交易与回报数值均可留空。"""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Empty, Queue
import json
import math

import streamlit as st
from dotenv import load_dotenv

from vcagents import CompanyInput, DocumentStore, SectorPipeline, SectorRequest, Settings, save_partial_result, save_result
from vcagents.config import DEFAULT_REPORT_MAX_PAGES, MAX_COMPANIES, REPORT_PAGE_CAP
from vcagents.financials import parse_financial_data, parse_market_data
from vcagents.project_db import ProjectDatabase
from vcagents.report import format_run_cost, render_audit


def _migrate_optional_numbers():
    """Old zero values meant 'not supplied'; preserve that meaning on refresh."""
    if st.session_state.get("optional_numbers_version") != 1:
        for index in range(MAX_COMPANIES):
            for prefix in ("valuation", "investment", "round"):
                key = f"{prefix}_{index}"
                if st.session_state.get(key) == 0:
                    st.session_state[key] = None
        st.session_state.optional_numbers_version = 1


def _company_card(index):
    with st.container(border=True, height="stretch"):
        st.markdown(f"**公司 {index + 1}**")
        name = st.text_input("公司名称", key=f"name_{index}", placeholder="填写公司名称")
        website = st.text_input("官网（可选）", key=f"website_{index}", placeholder="https://…")
        files = st.file_uploader("BP 或其他资料（可选）", type=["pdf", "txt", "md"],
                                 accept_multiple_files=True, key=f"docs_{index}",
                                 help="可上传BP、产品介绍或其他研究资料；不上传也可以研究公开信息。")
        with st.expander("交易与经营数据（可选）", icon=":material/tune:"):
            st.caption("全部可留空。金额单位：百万美元，例如 50 表示 5000 万美元。填写的金额应大于 0。")
            valuation = st.number_input("本轮投后估值（可选）", min_value=0.0, value=None, step=1.0,
                                        placeholder="留空则不测算估值回报", key=f"valuation_{index}",
                                        help="融资完成后公司的整体估值，单位为百万美元。")
            investment = st.number_input("我方拟投资金额（可选）", min_value=0.0, value=None, step=0.1,
                                         placeholder="留空则不测算我方持股与所得", key=f"investment_{index}")
            round_size = st.number_input("本轮融资总额（可选）", min_value=0.0, value=None, step=1.0,
                                         placeholder="留空则跳过相关测算", key=f"round_{index}",
                                         help="本轮所有投资人的合计融资金额，单位为百万美元。")
            financial_file = st.file_uploader("经营数据 JSON（可选）", type=["json"], key=f"financial_{index}",
                                              help="用于有明确期间和来源的经营指标测算；不提供时按公开资料进行研究。")
    return (CompanyInput(name.strip(), website.strip(), post_money_musd=valuation,
                         investment_musd=investment, round_size_musd=round_size), files, financial_file)


def _optional_calculations():
    with st.expander("补充回报与市场测算（可选）", icon=":material/calculate:"):
        st.caption("没有这些数值也可以生成研究报告。未知数据请留空，不需要填 0 或猜测数值。")
        years_col, dilution_col = st.columns(2)
        years = years_col.number_input("假设持有年数（可选）", min_value=0.0, value=None, step=1.0,
                                        placeholder="留空则不计算年化回报", key="holding_years")
        dilution = dilution_col.number_input("后续持股稀释比例（可选）", min_value=0.0, max_value=0.99,
                                              value=None, step=0.05, placeholder="例如 0.3 表示稀释 30%", key="exit_dilution",
                                              help="0 表示不稀释。留空时沿用 50% 的系统研究假设，并在报告中明确标注。")
        st.caption("稀释比例留空时，报告会明确标注系统的 50% 研究假设；该假设不代表已确认的交易条款。")
        st.markdown("**情景权重（可选）**")
        st.caption("三项完整且合计为 100% 时才计算期望倍数；不完整时跳过该测算，研究照常进行。")
        weights = {}
        for col, scenario in zip(st.columns(3), ("乐观", "基线", "悲观")):
            weights[scenario] = col.number_input(f"{scenario}权重（%，可选）", min_value=0.0, max_value=100.0,
                                                 value=None, step=5.0, placeholder="可留空", key=f"weight_{scenario}")
        market_file = st.file_uploader("市场规模输入 JSON（可选）", type=["json"], key="market",
                                       help="仅用于补充有来源和口径的市场测算，缺少输入时不编造数值。")
    return years, dilution, weights, market_file


def _read_json(upload, label):
    try:
        return json.loads(upload.getvalue().decode("utf-8-sig"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError(f"{label}必须为有效的 UTF-8 JSON 文件。") from exc


def _submit(sector, entries, focus, core_idea, count, mock, memory_enabled, years, dilution, weights, market_file,
            selection_rationale="", report_max_pages=DEFAULT_REPORT_MAX_PAGES, cost_budget_usd=None):
    companies = [entry[0] for entry in entries]
    names = [company.name for company in companies]
    if any(not name for name in names):
        st.error("请填写每家公司的名称；也可以切换为“由研究员推荐”。其他资料和数值均可留空。")
        return
    if len(set(names)) != len(names):
        st.error("公司名称不能重复，请检查后再生成。")
        return
    for company in companies:
        for label, value in (("投后估值", company.post_money_musd), ("拟投资金额", company.investment_musd),
                             ("本轮融资总额", company.round_size_musd)):
            if value is not None and value <= 0:
                st.error(f"{company.name}的{label}填写时应大于 0；不提供时请清空该数值。")
                return
    if years is not None and years <= 0:
        st.error("持有年数填写时应大于 0；不计算年化回报时请留空。")
        return
    probabilities = {}
    if any(value is not None for value in weights.values()):
        if all(value is not None for value in weights.values()) and math.isclose(sum(weights.values()), 100, abs_tol=1e-8):
            probabilities = {name: value / 100 for name, value in weights.items()}
        else:
            st.info("情景权重未完整填写或合计不为 100%，本次跳过期望倍数测算；其他研究不受影响。")
    docs, market_data = {}, None
    try:
        if market_file:
            market_data = parse_market_data(_read_json(market_file, "市场规模输入"))
        for company, files, financial_file in entries:
            if financial_file:
                company.financial_data = parse_financial_data(_read_json(financial_file, f"{company.name}的经营数据"))
            if files:
                store = DocumentStore()
                for upload in files:
                    store.add_file(upload.name, upload.getvalue())
                docs[company.name] = store
    except Exception as exc:
        st.error(f"资料读取失败：{exc}")
        return
    for key in ("report", "result_json", "partial_json", "audit", "report_pdf", "result_summary"):
        st.session_state.pop(key, None)
    run_analysis(sector, companies, docs, focus, count, mock, dilution, years, probabilities,
                 market_data=market_data, core_idea=core_idea, memory_enabled=memory_enabled,
                 selection_rationale=selection_rationale, report_max_pages=report_max_pages,
                 cost_budget_usd=cost_budget_usd)


def _result_summary(result):
    return {"companies": result.companies, "status": result.report_status, "pages": result.report_pages,
            "elapsed_seconds": result.elapsed_seconds, "cost": format_run_cost(result)}


def _results():
    if "report" not in st.session_state and "partial_json" not in st.session_state:
        return
    st.subheader("研究结果", icon=":material/description:")
    summary = st.session_state.get("result_summary", {})
    if summary:
        details = ["、".join(summary["companies"]), summary["status"]]
        if summary["pages"]:
            details.append(f"PDF {summary['pages']} 页")
        if "elapsed_seconds" in summary:
            minutes, seconds = divmod(round(summary["elapsed_seconds"]), 60)
            details.extend([f"研究耗时 {minutes} 分 {seconds} 秒", summary["cost"]])
        st.caption("　·　".join(filter(None, details)))
    with st.container(horizontal=True):
        if "report_pdf" in st.session_state:
            st.download_button("下载 PDF", st.session_state.report_pdf,
                               file_name=st.session_state.report_filename.replace(".md", ".pdf"), mime="application/pdf",
                               icon=":material/download:", type="primary")
        if "report" in st.session_state:
            st.download_button("Markdown", st.session_state.report, file_name=st.session_state.report_filename, mime="text/markdown")
            st.download_button("结构化结果", st.session_state.result_json, file_name="result.json", mime="application/json")
        if "partial_json" in st.session_state:
            st.download_button("下载未完成的运行记录", st.session_state.partial_json, file_name="partial-result.json", mime="application/json")
        if "audit" in st.session_state:
            st.download_button("证据与修正台账", st.session_state.audit, file_name="audit.md", mime="text/markdown")
    if "report" in st.session_state:
        with st.expander("展开报告预览", expanded=True, icon=":material/article:"):
            st.markdown(st.session_state.report)


def _report_history(database, sectors):
    history = st.expander("历史报告", icon=":material/history:", key="history_open", on_change="rerun")
    if not history.open:
        return
    with history:
        sector = st.selectbox("按赛道筛选", sectors.keys(), format_func=sectors.get, index=None,
                              placeholder="全部赛道", key="history_sector")
        records = database.reports(sector)
        if not records:
            st.caption("这个赛道还没有归档记录。" if sector else "生成的报告会出现在这里。")
            return
        by_path = {row["record_path"]: row for row in records}
        titles = {}
        for key, row in by_path.items():
            subject = "、".join(row["companies"]) or (Path(key).stem if row["kind"] == "pdf" else row["sector_name"])
            date = row["saved_at"][:19].replace("T", " ")
            status = "未完成" if row["kind"] == "partial" else row["status"]
            titles[key] = f"{date} · {subject} · {status}"
        counts = Counter(titles.values())
        titles = {key: title if counts[title] == 1 else f"{title} · {key}" for key, title in titles.items()}
        selected = st.selectbox("选择归档记录", by_path.keys(), format_func=titles.get, index=None,
                                placeholder=f"共 {len(records)} 条，可搜索公司名或日期", key="history_report")
        if selected not in by_path:
            return
        row = by_path[selected]
        st.caption(row["sector_name"] + (f" · {row['pages']} 页" if row["pages"] else ""))
        try:
            documents = [database.document_path(value, selected) for value in row["files"]]
        except ValueError:
            st.info("原文件路径已变化，请更新报告索引后再打开。")
            return
        available = sorted((path for path in documents if path.is_file()),
                           key=lambda path: (path.suffix.lower() != ".pdf", path.name))
        if not available:
            st.info("原文件当前不可用。可在恢复文件后重新打开这条记录。")
            return
        files_by_name = {path.name: path for path in available}
        filename = st.selectbox("归档文件", list(files_by_name), key="history_file")
        doc = files_by_name.get(filename)
        if doc is None:
            return
        try:
            content = doc.read_bytes()
        except OSError:
            st.info("暂时无法读取所选文件，请检查文件是否已移动。")
            return
        mime = {".pdf": "application/pdf", ".md": "text/markdown", ".json": "application/json"}
        st.download_button("下载原文件", content, file_name=doc.name, mime=mime[doc.suffix.lower()],
                           key="history_download", icon=":material/download:")


def _page():
    st.title("AI 赛道投研", icon=":material/query_stats:")
    with st.container(horizontal=True, vertical_alignment="center"):
        mock = st.toggle("演示模式", value=True, key="demo_mode", help="开启时不调用模型和联网搜索，生成模拟内容。")
        if mock:
            st.badge("模拟数据", color="orange", icon=":material/science:")
        else:
            st.badge("真实研究", color="blue", icon=":material/search:")
        with st.popover("研究设置", icon=":material/settings:"):
            page_limit = st.selectbox("报告页数上限", range(1, REPORT_PAGE_CAP + 1),
                                      index=DEFAULT_REPORT_MAX_PAGES - 1, format_func=lambda n: f"{n} 页",
                                      key="report_page_limit", help="按实际 PDF 页数检查，包含来源与附录。")
            budget = None
            if not mock:
                budget = st.number_input("费用停止阈值（美元，可选）", min_value=0.01, value=None, step=1.0,
                                         key="run_cost_limit", placeholder="留空则不限",
                                         help="已回传费用达到阈值后停止发出新请求。在途请求仍可能计费。")
            memory_enabled = st.toggle("复用历史来源与纠错线索", value=True, key="memory_enabled")
            st.caption("历史线索需重新读取原文后才能作为本轮证据；上传资料不自动进入公共记忆。")
            st.caption("总控、研究、投资分析、审计四个模块共用五个 Agent。分析模块的正反双方交锋两轮，关键事实由审计核查。")
    if mock:
        st.caption(f"演示使用模拟资料，API 费用为零。本次报告含来源与附录最多 {page_limit} 页。")
    else:
        st.caption(f"本次报告含来源与附录最多 {page_limit} 页。模型与搜索按量计费，可在研究设置中设置费用停止阈值。")
    st.subheader("研究对象")
    sector_col, mode_col, count_col = st.columns([1.1, 1.5, 0.7], vertical_alignment="bottom")
    database = ProjectDatabase()
    sectors = database.sector_names()
    entered = st.session_state.get("sector")
    if entered is not None and entered not in sectors:
        try:
            profile = database.resolve_sector(entered)
            sectors[profile.key] = profile.name
            st.session_state.sector = profile.key
        except ValueError as exc:
            sector_col.error(str(exc))
            st.session_state.sector = None
    sector = sector_col.selectbox("研究赛道（可直接输入）", sectors.keys(),
                                  format_func=lambda value: sectors.get(value, value),
                                  accept_new_options=True, key="sector",
                                  help="选择已有赛道，或输入新名称后按回车保存。新赛道从通用投研评分标准开始。")
    mode = mode_col.segmented_control("公司来源", ["手动指定", "由研究员推荐"], default="手动指定", required=True,
                                      key="company_mode", width="stretch")
    count = count_col.number_input(f"研究公司数量（1–{MAX_COMPANIES} 家）", min_value=1, max_value=MAX_COMPANIES,
                                   value=3, step=1, required=True, key="company_count")
    with st.form("request", border=False, enter_to_submit=False):
        entries = []
        if mode == "手动指定":
            st.caption("只需填写公司名称。官网和资料可选，所有数值都可以留空。")
            for start in range(0, count, 3):
                indices = range(start, min(start + 3, count))
                for col, index in zip(st.columns(len(indices)), indices):
                    with col:
                        entries.append(_company_card(index))
        else:
            st.info(f"研究员将根据所选赛道推荐 {count} 家候选公司，无需先填写公司或交易金额。")
        st.caption("系统自动确定研究主线，研究公司突出点、竞对差异、投入成本、商业化、长期竞争力与主要风险。")
        with st.expander("补充要求（可选）", expanded=False, icon=":material/tune:"):
            st.caption("无需填写研究提纲。仅在有额外背景、专项问题或待验证观点时补充；留空由系统自主规划。")
            focus_col, idea_col = st.columns(2)
            focus = focus_col.text_area("额外关注的问题（可选）", key="focus", height=100,
                                        placeholder="例如：请特别核查对某个上游平台的依赖",
                                        help="在系统默认研究之外补充专项要求；无需列出常规分析维度。")
            core_idea = idea_col.text_area("已有观点或假设（可选）", key="core_idea", height=100,
                                           placeholder="例如：我怀疑差异主要来自工作流，希望检验这一点",
                                           help="作为待检验命题，同时寻找支持与反证；留空时由总控自行提出假设。")
            selection_rationale = ""
            if mode == "手动指定":
                selection_rationale = st.text_area(
                    "选取这些公司的背景（可选）", key="selection_rationale", height=80,
                    placeholder="例如：来自本次待评估项目清单，或此前客户访谈的线索",
                    help="仅补充你已知的选取背景。系统仍独立分析，不把背景说明当作已核实事实。")
        years, dilution, weights, market_file = _optional_calculations()
        submitted = st.form_submit_button("开始生成研究报告", key="generate", type="primary", width="stretch",
                                          icon=":material/search:", disabled=not sector)
        st.caption("不确定的数值可留空。关键事实与投资判断仍需人工尽调。")
    if submitted:
        _submit(sector, entries, focus, core_idea, count, mock, memory_enabled, years, dilution, weights, market_file,
                selection_rationale, report_max_pages=page_limit, cost_budget_usd=budget)
    _results()
    _report_history(database, sectors)


def main():
    load_dotenv()
    st.set_page_config(page_title="AI 赛道投研", page_icon=":material/query_stats:", layout="wide")
    _migrate_optional_numbers()
    with st.container(horizontal_alignment="center"):
        with st.container(width=1240):
            _page()


def run_analysis(sector, companies, docs, focus, count, mock, dilution, holding_years=None, probabilities=None, pairwise=False,
                 market_data=None, core_idea="", max_workers=4, research_lanes=1, memory_enabled=True,
                 selection_rationale="", report_max_pages=DEFAULT_REPORT_MAX_PAGES, cost_budget_usd=None):
    events = Queue()
    status = st.status("正在准备研究任务…", expanded=False)
    with status:
        log = st.empty()
    messages, pipeline = [], None
    try:
        settings = Settings.load(mock=mock, exit_dilution=dilution, holding_years=holding_years,
                                 scenario_probabilities=probabilities or {}, pairwise_checks=pairwise,
                                 max_workers=max_workers, research_lanes=research_lanes, memory_enabled=memory_enabled,
                                 report_max_pages=report_max_pages, cost_budget_usd=cost_budget_usd)
        request = SectorRequest(sector, companies, focus, count, market_data, core_idea,
                                selection_rationale=selection_rationale)
        pipeline = SectorPipeline(settings, docs, emit=events.put)
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(pipeline.run, request)
            try:
                while not future.done() or not events.empty():
                    try:
                        event = events.get(timeout=0.25)
                    except Empty:
                        continue
                    if event.kind == "stage":
                        status.update(label=event.message)
                    else:
                        messages.append(f"{event.name}：{event.message}")
                        log.text("\n".join(messages[-12:]))
            finally:
                # 页面被停止或刷新时，后台研究不能继续调用模型产生费用
                if not future.done():
                    pipeline.cancel("页面已停止本次研究")
            result = future.result()
        folder = save_result(result)
    except Exception as exc:
        status.update(label="研究未完成", state="error", expanded=True)
        if pipeline and pipeline.last_result:
            folder = save_partial_result(pipeline.last_result)
            st.session_state.partial_json = json.dumps(pipeline.last_result.to_dict(), ensure_ascii=False, indent=2)
            st.session_state.audit = render_audit(pipeline.last_result)
            st.session_state.result_summary = _result_summary(pipeline.last_result)
            st.error(f"生成失败：{exc}。已保留本次研究记录：{folder}")
        else:
            st.error(f"无法开始研究：{exc}")
        return
    st.session_state.report_filename = "report.md" if (folder / "report.md").exists() else "research-draft.md"
    st.session_state.report = (folder / st.session_state.report_filename).read_text(encoding="utf-8")
    st.session_state.report_pdf = (folder / st.session_state.report_filename.replace(".md", ".pdf")).read_bytes()
    st.session_state.audit = render_audit(result)
    st.session_state.result_json = json.dumps(result.to_dict(), ensure_ascii=False, indent=2)
    st.session_state.result_summary = _result_summary(result)
    status.update(label="研究流程已完成", state="complete", expanded=False)
    if result.errors or result.report_status not in ("演示稿", "待人工复核的研究报告"):
        st.warning("已保留研究草稿；尚有问题需要补充或复核，详见“证据与修正台账”。")
    else:
        st.success("报告已生成，可下载 PDF，或展开预览查看。")


if __name__ == "__main__":
    main()
