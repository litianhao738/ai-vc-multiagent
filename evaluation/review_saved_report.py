"""用当前成稿流程回放已保存的 Higgsfield 研究，不重新采集或冒充完整研究重跑。"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import re
import sqlite3
import time

from dotenv import load_dotenv

from vcagents import CompanyInput, SectorPipeline, Settings, save_partial_result, save_result
from vcagents.financials import calculate_deal, calculate_metrics, parse_financial_data
from vcagents.pipeline import AnalysisResult, _namespace_sources
from vcagents.tools import Toolbox


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    load_dotenv()
    result = AnalysisResult(**json.loads(args.result.read_text(encoding="utf-8")))
    settings = Settings.load(holding_years=result.returns[0]["holding_years"],
                             exit_dilution=result.return_summary["dilution"], max_agent_steps=4)
    pipeline = SectorPipeline(settings, emit=lambda e: print(f"{e.kind} {e.agent}: {e.message}", flush=True))
    pipeline.sector = result.sector_config()
    pipeline.last_result = result
    pipeline._source_reads = deepcopy(result.source_reads)
    result.sector_brief = _namespace_sources(result.sector_brief, "S", result.sector_brief)
    add_current_model_evidence(result, pipeline)
    financial = parse_financial_data(json.loads(Path("inputs/aigc_20261030/higgsfield.json").read_text(encoding="utf-8")))
    result.financial_metrics["Higgsfield"] = {**financial, "metrics": calculate_metrics(financial["inputs"])}
    result.deal_metrics["Higgsfield"] = calculate_deal(5400, 10, 400, financial["inputs"])
    pipeline._financial_metrics = result.financial_metrics
    pipeline.companies = {"Higgsfield": CompanyInput("Higgsfield", "https://higgsfield.ai", post_money_musd=5400,
                                                     investment_musd=10, round_size_musd=400, financial_data=financial)}
    original_errors = deepcopy(result.errors)
    original_started_at = result.started_at
    result.started_at = datetime.now().isoformat(timespec="seconds")
    started = time.monotonic()
    result.errors = {"research_replay": "本次仅用已归档来源复核正文，未重新完成公司分析、对比与全流程事实核查。"}
    result.report_status = "历史证据回放校订稿"
    result.decision_fact_check = {"claims": [], "summary": "本次未重新执行自动事实核查"}
    result.memo_reviews = []
    result.usage = {}
    result.costs_usd = {}
    result.agent_seconds = {}
    result.format_failures = {}
    result.search_usage = {}
    result.run_control = {}
    output_root = args.result.parent / "editorial-replay"
    try:
        pipeline._review_memo(result)
        result.source_reads = pipeline._source_reads
        result.usage, result.costs_usd = pipeline._usage, pipeline._costs
        result.agent_seconds, result.format_failures = pipeline._agent_seconds, pipeline._format_failures
        result.finished_at = datetime.now().isoformat(timespec="seconds")
        result.elapsed_seconds = round(time.monotonic() - started, 2)
        folder = save_result(result, output_root)
        (folder / "replay-context.json").write_text(json.dumps({"input": str(args.result.resolve()),
            "original_errors": original_errors, "original_started_at": original_started_at,
            "scope": "一次正文复核，不是完整真实运行验收"}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"folder": str(folder), "status": result.report_status,
                          "publication_issues": result.report_checks}, ensure_ascii=False), flush=True)
    except Exception as exc:
        result.errors["chair_editor"] = str(exc)
        result.status, result.report_status = "failed", "正文复核未完成"
        result.usage, result.costs_usd = pipeline._usage, pipeline._costs
        folder = save_partial_result(result, output_root)
        print(f"正文复核失败：{exc}；已保存 {folder}", flush=True)
        raise


def add_current_model_evidence(result, pipeline):
    url = "https://news.qq.com/rain/a/20261004A0487U00"
    database = Path(pipeline.settings.knowledge_path).resolve().as_posix()
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
        row = connection.execute("SELECT snapshot_id FROM snapshots WHERE url=? ORDER BY fetched_at DESC LIMIT 1", (url,)).fetchone()
    source = pipeline.knowledge.read(row[0])
    match = re.search(r"Higgsfield把35个以上的AI模型.*?平台根据任务自动选择模型。", source["text"], re.S)
    read = {**source, "kind": "snapshot", "text": match.group(), "start_offset": match.start(), "end_offset": match.end()}
    sheet = result.fact_sheets["Higgsfield"]
    previous = deepcopy(sheet)
    source_label = "[latest-models] 2026-10-04 Higgsfield创始人访谈整理 - " + url
    evidence = Toolbox(shared_reads=[read]).verify_evidence({"sources": [source_label], "evidence": [{
        "claim": "据2026年10月4日媒体访谈整理，Higgsfield聚合35个以上AI模型。",
        "subject": "Higgsfield", "source_ref": "[latest-models]", "source_type": "其他",
        "as_of": "2026-10-04（报道日期）", "excerpt": match.group(),
    }]})["evidence"][0]
    if not evidence["excerpt_verified"]:
        raise ValueError("新增模型数量证据没有匹配原文")
    sheet["sources"].append(source_label)
    sheet["evidence"].append(evidence)
    result.fact_sheets["Higgsfield"] = _namespace_sources(sheet, "C1", previous)
    pipeline._source_reads["editorial_evidence"] = [read]


if __name__ == "__main__":
    main()
