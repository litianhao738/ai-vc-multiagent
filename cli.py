"""命令行入口。

示例：
    python cli.py --sector aigc-content --company "Pexo|https://pexo.ai" --company HeyGen --company Higgsfield \
        --post-money Pexo=50 --investment Pexo=5 --round-size Pexo=10 --market-data examples/market_inputs.example.json
    python cli.py --sector embodied-ai                     # 不指定公司，由赛道研究员推荐
    python cli.py --sector aigc-content --company 测试公司 --mock   # 演示模式，不调用模型
    python cli.py --list-sectors
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from vcagents import CompanyInput, DocumentStore, SectorPipeline, ExpandedSectorPipeline, SectorRequest, Settings, save_partial_result, save_result
from vcagents.agent import Event
from vcagents.config import DEFAULT_REPORT_MAX_PAGES, MAX_COMPANIES, REPORT_PAGE_CAP
from vcagents.project_db import ProjectDatabase


def print_event(e: Event) -> None:
    if e.kind == "stage":
        print(f"\n=== {e.message} ===", flush=True)
    else:
        mark = {"start": "开始", "log": "进度", "done": "完成", "error": "失败"}[e.kind]
        print(f"{mark} [{e.name}] {e.message}", flush=True)


def parse_pairs(items: list[str], flag: str) -> list[tuple[str, str]]:
    pairs = []
    for item in items:
        if "=" not in item:
            raise SystemExit(f"{flag} 的格式是 公司名=值，收到：{item}")
        name, value = item.split("=", 1)
        pairs.append((name.strip(), value.strip()))
    return pairs


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="AI 赛道投研多智能体平台")
    parser.add_argument("--sector", help="输入赛道名称，或使用 --list-sectors 中的配置标识")
    parser.add_argument("--company", action="append", default=[], help="公司，格式“名称”或“名称|官网”，可多次使用")
    parser.add_argument("--doc", action="append", default=[], help="上传资料，格式“公司名=文件路径”，可多次使用")
    parser.add_argument("--post-money", action="append", default=[], help="本轮投后估值，格式“公司名=百万美元”")
    parser.add_argument("--investment", action="append", default=[], help="拟投资金额，格式“公司名=百万美元”")
    parser.add_argument("--round-size", action="append", default=[], help="本轮融资总额，格式“公司名=百万美元”")
    parser.add_argument("--financial-data", action="append", default=[], help="公司经营输入 JSON，格式“公司名=文件路径”")
    parser.add_argument("--market-data", help="赛道级市场规模输入 JSON 文件路径")
    parser.add_argument("--exit-dilution", type=float, help="投资到退出期间的持股稀释比例；不填时用系统默认 0.5 并在报告中标注")
    parser.add_argument("--holding-years", type=float, help="用户假设的持有年数；未提供时不计算简化 IRR")
    parser.add_argument("--scenario-probability", action="append", default=[], help="用户主观情景权重，如“乐观=0.2”；三项须合计为 1")
    parser.add_argument("--management-fee", type=float, default=0.02, help="年管理费率，用于扣费后回报测算，默认 0.02")
    parser.add_argument("--carry", type=float, default=0.2, help="超额收益分成比例，默认 0.2")
    parser.add_argument("--liquidation-preference", type=float, help="优先清算倍数（不参与分配型），如 1；不填时不测算优先清算权")
    parser.add_argument("--preference-stack", action="append", default=[],
                        help="与本基金同顺位的优先股清算额合计（含本基金），格式“公司名=百万美元”")
    parser.add_argument("--pairwise-checks", action="store_true", help="可选：额外进行交换顺序的两两比较")
    parser.add_argument("--focus", default="", help="希望重点关注的问题")
    parser.add_argument("--core-idea", default="", help="希望主 Agent 检验的核心思想；不是预设结论")
    parser.add_argument("--selection-rationale", default="",
                        help="为什么选这些公司；作为待检验命题，报告会说明证据支持到什么程度")
    parser.add_argument("--agent-layout", choices=["core", "expanded"], default="core", help="默认 core：四个模块、五个 Agent，正反交锋两轮；expanded为旧细分模式")
    parser.add_argument("--max-workers", type=int, default=4, help="并行任务额度，默认 4。core 模式使用五个常驻 Agent")
    parser.add_argument("--research-lanes", type=int, default=3, help="仅expanded模式：每家公司研究分工数，1-3")
    parser.add_argument("--run-timeout-minutes", type=float, default=90,
                        help="整份任务的总时限（分钟），默认 90；到时停止并保存已完成的部分，0 表示不限")
    parser.add_argument("--retry-budget", type=int, default=12, help="整份任务可用的接口重试总次数，默认 12")
    parser.add_argument("--max-cost-usd", type=float, help="费用硬上限（美元）；只统计接口返回了费用的请求，不填则不限")
    parser.add_argument("--provider-concurrency", type=int, default=8, help="同一供应商同时在途的请求数，默认 8")
    parser.add_argument("--provider-rpm", action="append", default=[],
                        help="供应商每分钟请求数，格式“供应商=次数”，如 perplexity=60；供应商是模型名的前缀")
    parser.add_argument("--audit-claims-per-company", type=int, default=2,
                        help="研究核查时每家公司至少核查的关键主张数，1-4，默认 2")
    parser.add_argument("--research-refinement-rounds", type=int, choices=[0, 1, 2], default=1,
                        help="原流程成稿前的问题驱动补查上限，默认1；0关闭；不增加Agent")
    parser.add_argument("--no-memory", action="store_true", help="禁用跨任务研究记忆")
    parser.add_argument("--memory-path", default=None, help="研究记忆 SQLite 路径，可按项目隔离")
    parser.add_argument("--report-max-pages", type=int, default=DEFAULT_REPORT_MAX_PAGES,
                        help=f"报告含来源的PDF页数上限，1-{REPORT_PAGE_CAP}，默认{DEFAULT_REPORT_MAX_PAGES}")
    parser.add_argument("--max-companies", type=int, choices=range(1, MAX_COMPANIES + 1), default=MAX_COMPANIES,
                        help="研究员推荐的公司数量，手动指定时为数量上限")
    parser.add_argument("--debate-rounds", type=int, default=None, help="core 固定两轮，expanded 默认一轮")
    parser.add_argument("--models", default=None, help="模型配置文件，默认是项目根目录的 models.yaml")
    parser.add_argument("--mock", action="store_true", help="演示模式：不调用任何模型和搜索")
    parser.add_argument("--out", type=Path, help="报告目录，默认使用项目内的 outputs/")
    parser.add_argument("--list-sectors", action="store_true", help="显示内置和已保存的自定义赛道")
    parser.add_argument("--index-reports", action="store_true", help="重建报告索引，可用 --out 指定归档目录")
    args = parser.parse_args()

    if args.list_sectors:
        for key, name in ProjectDatabase().sector_names().items():
            print(f"{key}\t{name}")
        return 0
    if args.index_reports:
        try:
            count, errors = ProjectDatabase().rebuild_report_index(args.out)
        except ValueError as exc:
            parser.error(str(exc))
        print(f"已索引 {count} 条归档记录。")
        for error in errors:
            print(error, file=sys.stderr)
        return int(bool(errors))
    if not args.sector:
        parser.error("请用 --sector 输入赛道名称，或用 --list-sectors 查看已有赛道")

    companies = []
    for item in args.company:
        name, _, website = item.partition("|")
        companies.append(CompanyInput(name.strip(), website.strip()))
    by_name = {c.name: c for c in companies}
    for flag, items, attr in (("--post-money", args.post_money, "post_money_musd"),
                              ("--investment", args.investment, "investment_musd"),
                              ("--round-size", args.round_size, "round_size_musd"),
                              ("--preference-stack", args.preference_stack, "preference_stack_musd")):
        for name, value in parse_pairs(items, flag):
            if name not in by_name:
                raise SystemExit(f"{flag} 里的公司 {name} 不在 --company 列表中")
            setattr(by_name[name], attr, float(value))

    docs: dict[str, DocumentStore] = {}
    for name, path in parse_pairs(args.financial_data, "--financial-data"):
        if name not in by_name:
            raise SystemExit(f"--financial-data 里的公司 {name} 不在 --company 列表中")
        by_name[name].financial_data = json.loads(Path(path).read_text(encoding="utf-8"))
    for name, path in parse_pairs(args.doc, "--doc"):
        if name not in by_name:
            raise SystemExit(f"--doc 里的公司 {name} 不在 --company 列表中")
        docs.setdefault(name, DocumentStore()).add_file(Path(path).name, Path(path).read_bytes())

    probabilities = {name: float(value) for name, value in parse_pairs(args.scenario_probability, "--scenario-probability")}
    try:
        rpm = {name: int(value) for name, value in parse_pairs(args.provider_rpm, "--provider-rpm")}
    except ValueError:
        raise SystemExit("--provider-rpm 的次数必须是整数，例如 perplexity=60") from None
    settings = Settings.load(args.models, mock=args.mock or None, debate_rounds=args.debate_rounds,
                             exit_dilution=args.exit_dilution, holding_years=args.holding_years,
                             scenario_probabilities=probabilities, pairwise_checks=args.pairwise_checks,
                             management_fee=args.management_fee, carry=args.carry,
                             liquidation_preference=args.liquidation_preference,
                             max_workers=args.max_workers, research_lanes=args.research_lanes,
                             memory_enabled=not args.no_memory, memory_path=args.memory_path,
                             report_max_pages=args.report_max_pages,
                             run_timeout_seconds=args.run_timeout_minutes * 60, retry_budget=args.retry_budget,
                             cost_budget_usd=args.max_cost_usd, provider_concurrency=args.provider_concurrency,
                             provider_rpm=rpm, audit_claims_per_company=args.audit_claims_per_company,
                             research_refinement_rounds=args.research_refinement_rounds)
    market = json.loads(Path(args.market_data).read_text(encoding="utf-8")) if args.market_data else None
    request = SectorRequest(args.sector, companies, args.focus, args.max_companies, market, args.core_idea,
                            selection_rationale=args.selection_rationale)
    pipeline_class = SectorPipeline if args.agent_layout == "core" else ExpandedSectorPipeline
    pipeline = pipeline_class(settings, docs, emit=print_event)
    try:
        result = pipeline.run(request)
    except (Exception, KeyboardInterrupt) as exc:
        pipeline.cancel("运行被中断")  # 并行任务不再发起新的模型与工具调用
        print("已中断。" if isinstance(exc, KeyboardInterrupt) else f"生成失败：{exc}", file=sys.stderr)
        if pipeline.last_result:
            folder = save_partial_result(pipeline.last_result, args.out)
            print(f"部分结果已保存到：{folder}", file=sys.stderr)
        return 1
    folder = save_result(result, args.out)

    d = result.decision
    print(f"\n本次建议：{d['action']} · 赛道：{d['verdict']} · 推荐：{'、'.join(d['recommended']) or '暂无'}")
    if result.errors:
        print("未完成的智能体：" + "；".join(f"{k}: {v}" for k, v in result.errors.items()))
    print(f"报告已保存到：{folder}")
    print(f"PDF报告：{result.report_pages} 页；状态：{result.report_status}")
    if result.main_agents:
        print(f"主Agent：{len(result.main_agents)} 个；角色：{'、'.join(result.main_agents)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
