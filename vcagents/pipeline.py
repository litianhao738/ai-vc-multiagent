"""编排流程：研究（赛道 + 各公司并行）→ 分析与对比 → 多空辩论和事实核查 → 投委会决策。"""
from __future__ import annotations

import json
import math
import re
import threading
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from itertools import combinations
from pathlib import Path
from uuid import uuid4

from . import schemas
from .agent import Emit, Event
from .runtime import AgentScopeAgent as Agent
from .scheduler import DAGScheduler, TaskNode
from .governor import RunGovernor, RunStopped
from .memory import ResearchMemory
from .config import DEFAULT_EXIT_DILUTION, DEFAULT_REPORT_MAX_PAGES, MAX_COMPANIES, REPORT_PAGE_CAP, ROOT, Settings
from .llm import LLMRouter
from .rejection_review import RejectionReviewer
from .report_review import apply_report_reviews
from .financials import (
    calculate_deal, calculate_market_metrics, calculate_metrics, calculate_returns, parse_financial_data, parse_market_data,
    returns_analysis,
)
from .knowledge import KnowledgeStore
from .project_db import ProjectDatabase, index_export
from .corrections import affected_materials
from .report_checks import memo_materials
from .reasoning import review_issues, review_targets
from .roles import (
    BEAR, BULL, CHAIR, COMPANY_ANALYST, COMPANY_RESEARCHER, COMPARATOR, FACT_CHECKER, SECTOR_RESEARCHER, Role,
    system_prompt,
)
from .scoring import rank_companies, ranking_sensitivity, score_ranges
from .sectors import Sector, load_sector, sector_from_dict
from .citations import SourceRegistry
from . import grounding
from .tools import DocumentStore, SonarSearch, Toolbox, normalize_fact_check


@dataclass
class CompanyInput:
    name: str
    website: str = ""
    urls: list[str] = field(default_factory=list)
    notes: str = ""  # 用户补充的信息，例如内部访谈纪要
    post_money_musd: float | None = None  # 本轮投后估值（百万美元），有就用来算回报倍数
    investment_musd: float | None = None  # 拟投资金额（百万美元），用来算持股比例和退出所得
    round_size_musd: float | None = None  # 本轮融资总额（百万美元），用来算投前估值和本轮后跑道
    financial_data: dict | None = None
    preference_stack_musd: float | None = None  # 与本基金同顺位的优先股清算额合计（含本基金），用来测算优先清算权


@dataclass
class SectorRequest:
    sector: str  # 自由输入的名称、内置配置 key 或已保存的赛道 key
    companies: list[CompanyInput] = field(default_factory=list)  # 为空时由赛道研究员推荐
    focus: str = ""
    max_companies: int = MAX_COMPANIES  # 推荐模式为目标数量，手动模式为上限
    market_data: dict | None = None  # 赛道级市场规模输入
    core_idea: str = ""  # 用户希望验证的核心思想；不能作为已成立事实
    selection_rationale: str = ""  # 用户为什么选这些公司；作为待检验命题，系统不替用户编造


@dataclass
class AnalysisResult:
    sector_key: str
    sector_name: str
    status: str = "running"
    companies: list[str] = field(default_factory=list)
    models: dict[str, str] = field(default_factory=dict)
    started_at: str = ""
    finished_at: str = ""
    sector_brief: dict = field(default_factory=dict)
    fact_sheets: dict[str, dict] = field(default_factory=dict)
    analyses: dict[str, dict] = field(default_factory=dict)
    ranking: list[dict] = field(default_factory=list)
    sensitivity: dict = field(default_factory=dict)
    score_ranges: dict = field(default_factory=dict)
    comparison: dict = field(default_factory=dict)
    pairwise: list[dict] = field(default_factory=list)
    debate: list[dict] = field(default_factory=list)  # [{round, side, name, ...立场}]
    fact_check: dict = field(default_factory=dict)
    fact_check_pass_rate: float | None = None
    checked_source_coverage: float | None = None
    repairs: list[dict] = field(default_factory=list)
    check_history: list[dict] = field(default_factory=list)
    unresolved_checks: list[dict] = field(default_factory=list)  # 一轮修正后仍未解决的核查问题
    source_index: dict[str, dict] = field(default_factory=dict)  # 来源编号（W3、D1）→ URL、快照与切句版本
    evidence_checks: dict[str, dict] = field(default_factory=dict)  # 材料 → 正文事实性数字与所引证据的核对结果
    returns_analysis: dict = field(default_factory=dict)  # 回报敏感性、扣费后回报与优先清算权（代码计算）
    memo_reviews: list[dict] = field(default_factory=list)
    report_checks: list[str] = field(default_factory=list)
    report_warnings: list[str] = field(default_factory=list)
    decision_fact_check: dict = field(default_factory=dict)
    decision: dict = field(default_factory=dict)
    returns: list[dict] = field(default_factory=list)  # 代码按退出情景算出的回报倍数
    return_summary: dict = field(default_factory=dict)
    financial_metrics: dict[str, dict] = field(default_factory=dict)
    market_metrics: dict = field(default_factory=dict)
    deal_metrics: dict[str, dict] = field(default_factory=dict)  # 公司 → 用户交易输入及代码计算结果
    source_reads: dict[str, list[dict]] = field(default_factory=dict)
    report_status: str = "研究稿"
    errors: dict[str, str] = field(default_factory=dict)  # 失败的智能体 → 原因
    usage: dict[str, list[int]] = field(default_factory=dict)  # 智能体 → [输入 token, 输出 token]
    format_failures: dict[str, int] = field(default_factory=dict)
    agent_seconds: dict[str, float] = field(default_factory=dict)
    elapsed_seconds: float = 0
    costs_usd: dict[str, float | None] = field(default_factory=dict)
    search_usage: dict = field(default_factory=dict)
    run_id: str = field(default_factory=lambda: uuid4().hex)
    architecture: str = "agentscope-2.0.11"
    research_plan: dict = field(default_factory=dict)
    task_trace: list[dict] = field(default_factory=list)
    memory_config: dict = field(default_factory=dict)
    memory_stats: dict = field(default_factory=dict)
    report_max_pages: int = DEFAULT_REPORT_MAX_PAGES
    report_pages: int | None = None
    main_agents: list[str] = field(default_factory=list)
    handoffs: list[dict] = field(default_factory=list)
    material_versions: dict = field(default_factory=dict)
    quality_issues: list[dict] = field(default_factory=list)
    quality_checks: list[dict] = field(default_factory=list)
    quality_repairs: list[dict] = field(default_factory=list)
    run_control: dict = field(default_factory=dict)  # 总时限、重试预算、请求复用与限流等待的实际用量
    audit_coverage: dict = field(default_factory=dict)  # 研究核查的必查清单与覆盖缺口
    rejection_reviews: list[dict] = field(default_factory=list)  # 仅写法等价性复核及撤销退回的记录
    research_state: dict = field(default_factory=dict)  # 问题、假设、证据与判断变化；不替代原研究议程
    research_iterations: list[dict] = field(default_factory=list)  # 总控补查决策及实际执行记录
    research_methods: list[dict] = field(default_factory=list)  # 实际执行的方法候选及独立评价
    prompt_versions: dict[str, str] = field(default_factory=dict)  # 本次实际使用的提示词 → 内容哈希
    sector_profile: dict = field(default_factory=dict)  # 本次实际采用的评分配置，支持移出项目后重现报告

    def sector_config(self) -> Sector:
        return sector_from_dict(self.sector_profile) if self.sector_profile else load_sector(self.sector_key)

    def to_dict(self) -> dict:
        return asdict(self)


class PipelineError(RuntimeError):
    pass


Job = tuple[Role, str, Toolbox | None, str, str, dict | None]  # 角色、任务、工具箱、实例 id、显示名、输出格式
RESEARCH_TRACKS = ("产品、技术与创始团队", "商业化、经营指标、融资与上市状态", "竞争格局、客户反馈与重大风险")


class SectorPipeline:
    def __init__(self, settings: Settings, docs: dict[str, DocumentStore] | None = None, emit: Emit | None = None):
        self.settings = settings
        self.governor = RunGovernor.from_settings(settings)
        self.router = LLMRouter(settings, self.governor)
        self.docs = docs or {}  # 公司名 → 该公司的上传资料
        self.fact_docs = DocumentStore()
        for company, store in self.docs.items():
            for name, pages in store.docs.items():
                self.fact_docs.docs[f"{company}/{name}"] = pages
        self.emit = emit or (lambda event: None)
        self.rejection_reviewer = RejectionReviewer(settings, self.router,
            lambda message: self.emit(Event("log", "rejection_review", "退回必要性复核", message)))
        self.today = date.today().isoformat()
        self.search = None
        if not settings.mock and settings.openrouter_api_key:
            self.search = SonarSearch(settings.openrouter_api_key, settings.role_models["search"], governor=self.governor)
        self._lock = threading.Lock()
        self._network_slots = threading.BoundedSemaphore(max(1, settings.max_workers * 2))
        self._usage: dict[str, list[int]] = {}
        self._format_failures: dict[str, int] = {}
        self._costs: dict[str, float | None] = {}
        self._agent_seconds: dict[str, float] = {}
        self.repair_notes = ""
        self.last_result: AnalysisResult | None = None
        self._financial_metrics: dict[str, dict] = {}
        self._source_reads: dict[str, list[dict]] = {}
        self.registry = SourceRegistry()
        self.knowledge = None if settings.mock else KnowledgeStore(settings.knowledge_path)
        self.memory = ResearchMemory(settings.memory_path, settings.memory_ttl_days) if settings.memory_enabled and not settings.mock else None
        self._streamed_analysis: set[str] = set()

    # ------------------------------------------------------------ 主流程

    def cancel(self, reason: str = "任务已取消") -> None:
        """可从其他线程调用：在途请求结束后不再发起新的模型与工具调用。"""
        self.governor.cancel(reason)

    def run(self, request: SectorRequest) -> AnalysisResult:
        started = time.monotonic()
        self.registry = SourceRegistry()
        self._validate_request(request)
        self.sector = ProjectDatabase().resolve_sector(request.sector)
        self.governor.configure(self.settings)
        self.governor.start()
        self.rejection_reviewer.reset()
        result = AnalysisResult(
            self.sector.key, self.sector.name,
            sector_profile=asdict(self.sector),
            models={k: ("mock" if self.settings.mock else v) for k, v in self.settings.role_models.items()},
            started_at=_now(),
            report_max_pages=self.settings.report_max_pages,
            memory_config={"enabled": bool(self.memory), "path": self.settings.memory_path,
                           "ttl_days": self.settings.memory_ttl_days},
        )
        self.last_result = result
        result.rejection_reviews = self.rejection_reviewer.records
        result.financial_metrics = {c.name: {**c.financial_data, "metrics": calculate_metrics(c.financial_data["inputs"])}
                                    for c in request.companies if c.financial_data}
        self._financial_metrics = result.financial_metrics
        if request.market_data:
            result.market_metrics = {**request.market_data, "metrics": calculate_market_metrics(request.market_data["inputs"])}
        result.deal_metrics = {
            c.name: calculate_deal(c.post_money_musd, c.investment_musd, c.round_size_musd,
                                   c.financial_data["inputs"] if c.financial_data else None)
            for c in request.companies if c.post_money_musd or c.investment_musd or c.round_size_musd
        }
        try:
            self._plan_research(request, result)
            self._research(request, result)
            self._analyze(result)
            self._compare(result)
            if self.settings.pairwise_checks:
                self._compare_pairs(result)
            self._debate(result)
            self._check_research(result)
            self._stage("决策：投委会主席做出结论")
            result.decision = self._run(*self._job(CHAIR, self._chair_task(result), None,
                                                   schema=self._decision_schema(result)))
            self._review_memo(result)
            self._fit_report_budget(result)
            self._check_decision(result)
            result.evidence_checks = {}
            self._evidence_priorities(result, {"chair": memo_materials(result)})
            result.status = "completed"
        except Exception as exc:
            result.status = "failed"
            result.errors["pipeline"] = f"{type(exc).__name__}: {exc}"
            raise
        except BaseException:  # Ctrl+C 等：并行线程在下一次请求前停下
            self.governor.cancel("运行被中断")
            result.status = "failed"
            result.errors["pipeline"] = "运行被中断"
            raise
        finally:
            self._finish_result(result, started)
        self._stage("分析完成")
        return result

    def _finish_result(self, result: AnalysisResult, started: float) -> None:
        claims = result.fact_check.get("claims", [])
        result.fact_check_pass_rate = round(sum(c["verdict"] == "属实" for c in claims) / len(claims), 3) if claims else None
        result.checked_source_coverage = round(sum(bool(c["original_source"].strip()) for c in claims) / len(claims), 3) if claims else None
        if result.status == "failed":
            result.report_status = "未完成"
        elif self.settings.mock:
            result.report_status = "演示稿"
        elif (result.errors or not result.fact_check.get("claims") or not result.decision_fact_check.get("claims")
              or any(u["verdict"] == "不实" or u["problem"] == "不实" for u in result.unresolved_checks)):
            result.report_status = "核查或研究未完成草稿"
        elif result.unresolved_checks or any(c["verdict"] == "无法核实" for checks in (result.fact_check, result.decision_fact_check)
                                             for c in checks["claims"]):
            result.report_status = "关键事实整理中的研究稿"
        else:
            result.report_status = "待人工复核的研究报告"
        reviews = self.rejection_reviewer.records
        if reviews:
            self._usage["rejection_review"] = [sum(row["usage"][i] for row in reviews) for i in (0, 1)]
            self._costs["rejection_review"] = (sum(row["cost_usd"] for row in reviews)
                if all(row["cost_usd"] is not None for row in reviews) else None)
            self._agent_seconds["rejection_review"] = round(sum(row["seconds"] for row in reviews), 3)
        result.usage = self._usage
        result.format_failures = self._format_failures
        result.agent_seconds = self._agent_seconds
        result.source_reads = self._source_reads
        result.source_index = self.registry.index()
        result.costs_usd = {k: 0.0 for k in self._costs} if self.settings.mock else self._costs
        if self.search:
            result.search_usage = {"calls": self.search.calls, "input_tokens": self.search.input_tokens,
                                   "output_tokens": self.search.output_tokens, "cost_usd": self.search.cost_usd}
        result.run_control = self.governor.stats()
        result.finished_at = _now()
        result.elapsed_seconds = round(time.monotonic() - started, 2)

    def _research(self, request: SectorRequest, result: AnalysisResult) -> None:
        companies = request.companies[:]
        nodes, shards = [], {}
        self._streamed_analysis = set()

        def sector_research():
            schema = schemas.sector_brief_schema(0 if companies else request.max_companies)
            data = self._run(*self._job(SECTOR_RESEARCHER, self._sector_task(request, companies),
                                        self._web_toolbox(), schema=schema))
            with self._lock:
                result.sector_brief = _namespace_sources(data, "S")
            return data

        if not companies:
            self._stage("研究：赛道研究员先推荐候选公司")
            sector_research()
            companies = [CompanyInput(name) for name in result.sector_brief["candidates"]]
        else:
            nodes.append(TaskNode("sector_researcher", sector_research))
        self.companies = {c.name: c for c in companies}
        result.companies = list(self.companies)
        tracks = RESEARCH_TRACKS[:self.settings.research_lanes]
        self._stage(f"并行研究：每家公司 {len(tracks)} 个研究分工；材料齐备后立即分析，并发上限 {self.settings.max_workers}")
        for index, company in enumerate(companies, 1):
            lane_ids = []
            for lane, topic in enumerate(tracks, 1):
                key = f"company_researcher:{company.name}" + (f":lane{lane}" if len(tracks) > 1 else "")
                lane_ids.append(key)

                def research(c=company, k=key, scope=topic):
                    job = self._company_job(c)
                    task = job[1] + (f"\n\n并行研究分工：{scope}。专注此范围，其他范围留给同事；"
                                     "每项事实要可定位，不以同事的摘要作为证据。" if len(tracks) > 1 else "")
                    data = self._run(job[0], task, job[2], k, f"公司研究 · {c.name} · {scope}", job[5])
                    with self._lock:
                        shards[k] = data
                    return data
                nodes.append(TaskNode(key, research))

            def merge(c=company, keys=tuple(lane_ids), prefix=f"C{index}"):
                available = [shards[k] for k in keys if k in shards]
                if not available:
                    raise PipelineError(f"{c.name} 的研究分工全部失败")
                data = _namespace_sources(_merge_company_shards(c.name, available), prefix)
                with self._lock:
                    result.fact_sheets[c.name] = data
                return data
            merge_id = f"facts:{company.name}"
            nodes.append(TaskNode(merge_id, merge, tuple(lane_ids), allow_failed_dependencies=True))

            def analyze(c=company):
                sheet = result.fact_sheets[c.name]
                self._streamed_analysis.add(c.name)
                if sheet["listing_status"] == "已上市":
                    return None
                self._stage(f"开始分析 {c.name}：该公司材料与赛道研究已齐备")
                data = self._run(*self._job(COMPANY_ANALYST, self._analyst_task(c.name, sheet, result.sector_brief), None,
                                           f"company_analyst:{c.name}", f"公司分析师 · {c.name}",
                                           schemas.analyst_schema(self.sector, [e["evidence_id"] for e in sheet["evidence"]])))
                with self._lock:
                    result.analyses[c.name] = data
                return data
            deps = (merge_id, "sector_researcher") if request.companies else (merge_id,)
            nodes.append(TaskNode(f"company_analyst:{company.name}", analyze, deps))
        scheduler = DAGScheduler(self.settings.max_workers, self.governor)
        try:
            outcomes = scheduler.run(nodes)
        finally:
            result.task_trace.extend(scheduler.trace)
        result.errors.update({key: out.error for key, out in outcomes.items() if out.error})
        if not result.sector_brief:
            raise PipelineError("赛道研究失败：" + result.errors.get("sector_researcher", "未返回结果"))
        result.fact_sheets = {c.name: result.fact_sheets[c.name] for c in companies if c.name in result.fact_sheets}
        result.analyses = {c.name: result.analyses[c.name] for c in companies if c.name in result.analyses}
        if not result.fact_sheets:
            raise PipelineError("所有公司研究员都失败了")

    def _plan_research(self, request: SectorRequest, result: AnalysisResult) -> None:
        self._stage("主 Agent：确定核心研究命题、优先因素与证伪问题")
        scope = "、".join(c.name for c in request.companies) or f"由研究员推荐 {request.max_companies} 家"
        task = ("先制定研究议程，不提前给出投资结论。主线是：价值由谁捕获、凭什么持续、何种价格有回报。"
                "优先回答为什么是现在、为什么是这家、什么证据会推翻判断。未知内容写为待验证假设。"
                "把用户核心思想当作待检验命题；有相反证据必须修改立场，不安排单向寻找支持的任务。"
                f"最终含引用最多{self.settings.report_max_pages}页，首屏一句话建议、最多3条投资逻辑；背景只保留改变决策的信息。\n"
                f"赛道：{self.sector.name}\n公司：{scope}\n"
                f"重点问题：{request.focus}\n用户核心思想：{request.core_idea or '由你提出待验证的研究命题'}"
                f"\n用户选择这些公司的理由：{request.selection_rationale or '用户未提供，不要代为编造'}"
                "\n研究问题要尖锐、可被证据回答，优先：这家公司最突出的一点是什么、凭什么在竞争中胜出；"
                "与最接近竞对的最大差异；研发、交付、获客投入与成本；长期竞争力取决于什么。"
                "用户给出选择理由时，把它拆成可被证伪的问题。")
        if self.memory:
            task += "\n历史纠错与线索（不作为事实依据）：" + json.dumps(
                self.memory.recall(self.sector.key, query=request.focus, limit=4, kinds=("correction",)), ensure_ascii=False)
        result.research_plan = self._run(*self._job(CHAIR, task, None, "lead_planner", "主 Agent · 研究议程", schemas.RESEARCH_PLAN))

    def _analyze(self, result: AnalysisResult) -> None:
        eligible = {n: sheet for n, sheet in result.fact_sheets.items() if sheet["listing_status"] != "已上市"}
        if not eligible:
            raise PipelineError("公司研究结果均已上市，没有可供分析的未上市候选公司")
        self._stage(f"分析：{len(eligible)} 位公司分析师并行打分")
        jobs = [
            self._job(COMPANY_ANALYST, self._analyst_task(n, sheet, result.sector_brief), None,
                      f"company_analyst:{n}", f"公司分析师 · {n}",
                      schemas.analyst_schema(self.sector, [e["evidence_id"] for e in sheet["evidence"]]))
            for n, sheet in eligible.items() if n not in self._streamed_analysis
        ]
        outputs, errors = self._run_parallel(jobs)
        result.errors.update(errors)
        result.analyses.update({n: outputs[f"company_analyst:{n}"] for n in eligible if f"company_analyst:{n}" in outputs})
        if not result.analyses:
            raise PipelineError("所有公司分析师都失败了")
        result.ranking = rank_companies(result.analyses, self.sector)
        result.sensitivity = ranking_sensitivity(result.analyses, self.sector)
        result.score_ranges = score_ranges(result.analyses, result.fact_sheets, self.sector)

    def _compare(self, result: AnalysisResult) -> None:
        self._stage("对比：对比分析师复核排序")
        try:
            result.comparison = self._run(*self._job(COMPARATOR, self._comparator_task(result), None))
        except RunStopped:
            raise
        except Exception as e:  # 对比失败不影响后续，主席仍能看到代码排序
            result.errors["comparator"] = str(e)

    def _check_research(self, result: AnalysisResult) -> None:
        self._stage("核查：分两批检查研究、分析、对比和辩论中的关键主张")
        all_materials = self._check_materials(result)
        materials = list(all_materials)
        research = [k for k in materials if k == "sector_researcher" or k.startswith("company_researcher:")]
        batches = {"fact_checker": research, "fact_checker:analysis": [k for k in materials if k not in research]}
        jobs = [self._job(FACT_CHECKER, self._fact_check_task(result, keys),
                          self._factcheck_toolbox({k: all_materials[k] for k in keys}), agent_id,
                          "事实核查员 · " + ("研究材料" if agent_id == "fact_checker" else "分析与辩论"),
                          schemas.fact_check_schema(keys, self._evidence_ids(result)))
                for agent_id, keys in batches.items() if keys]
        outputs, errors = self._run_parallel(jobs)
        result.errors.update(errors)
        if outputs:
            parts = [outputs[agent_id] for agent_id in batches if agent_id in outputs]
            result.fact_check = {"claims": [c for part in parts for c in part["claims"]],
                                 "summary": "\n".join(part["summary"] for part in parts)}
        if result.fact_check:
            for i, claim in enumerate(result.fact_check["claims"], 1):
                claim["claim_id"] = f"R-{i}"
            result.check_history.append({"stage": "研究与辩论", "result": deepcopy(result.fact_check)})
        if any(c["verdict"] in ("不实", "无法核实") for c in result.fact_check.get("claims", [])):
            self._repair(result)
        claims = result.fact_check.get("claims") or []
        if claims:
            result.fact_check_pass_rate = round(sum(c["verdict"] == "属实" for c in claims) / len(claims), 3)
            result.checked_source_coverage = round(sum(bool(c["original_source"].strip()) for c in claims) / len(claims), 3)
        elif result.fact_check:
            result.errors["fact_checker"] = "核查员没有返回可核查主张，抽查结果未完成"

    def _validate_request(self, request: SectorRequest) -> None:
        if not 1 <= self.settings.max_workers <= 32:
            raise ValueError("并发 Agent 数量须在 1 到 32 之间")
        if not 1 <= self.settings.research_lanes <= 3:
            raise ValueError("公司研究分工数量须在 1 到 3 之间")
        if (isinstance(self.settings.report_max_pages, bool) or not isinstance(self.settings.report_max_pages, int)
                or not 1 <= self.settings.report_max_pages <= REPORT_PAGE_CAP):
            raise ValueError(f"报告页数上限须为 1 到 {REPORT_PAGE_CAP} 之间的整数")
        limits = self.settings
        if limits.run_timeout_seconds is not None and (not math.isfinite(limits.run_timeout_seconds) or limits.run_timeout_seconds < 0):
            raise ValueError("任务总时限须为不小于零的有限秒数；0 或留空表示不限")
        if limits.retry_budget is not None and (isinstance(limits.retry_budget, bool) or not isinstance(limits.retry_budget, int)
                                                or limits.retry_budget < 0):
            raise ValueError("重试预算须为不小于零的整数")
        if limits.cost_budget_usd is not None and (not math.isfinite(limits.cost_budget_usd) or limits.cost_budget_usd <= 0):
            raise ValueError("费用上限须为大于零的有限数值")
        if isinstance(limits.provider_concurrency, bool) or not isinstance(limits.provider_concurrency, int) or not 1 <= limits.provider_concurrency <= 32:
            raise ValueError("单个供应商的并发请求数须在 1 到 32 之间")
        if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in limits.provider_rpm.values()):
            raise ValueError("供应商每分钟请求数须为正整数")
        if (isinstance(limits.audit_claims_per_company, bool) or not isinstance(limits.audit_claims_per_company, int)
                or not 1 <= limits.audit_claims_per_company <= 4):
            raise ValueError("每家公司的核查主张数须为 1 到 4 的整数")
        if (not isinstance(request.max_companies, int) or isinstance(request.max_companies, bool)
                or not 1 <= request.max_companies <= MAX_COMPANIES):
            raise ValueError(f"公司数量须为 1 到 {MAX_COMPANIES} 之间的整数")
        if len(request.companies) > request.max_companies:
            raise ValueError(f"指定公司数量超过本次上限 {request.max_companies} 家")
        names = [c.name.strip() for c in request.companies]
        if any(not n for n in names) or len(set(names)) != len(names):
            raise ValueError("公司名称不能为空或重复")
        if any(c.post_money_musd is not None and (not math.isfinite(c.post_money_musd) or c.post_money_musd <= 0) for c in request.companies):
            raise ValueError("投后估值必须大于零")
        dilution = self.settings.exit_dilution
        if dilution is not None and (not math.isfinite(dilution) or not 0 <= dilution < 1):
            raise ValueError("后续稀释比例必须在 0 到 1 之间（不含 1）")
        for c in request.companies:
            for label, value in (("拟投资金额", c.investment_musd), ("本轮融资额", c.round_size_musd)):
                if value is not None and (not math.isfinite(value) or value <= 0):
                    raise ValueError(f"{c.name} 的{label}必须大于零")
            if c.investment_musd and c.round_size_musd and c.investment_musd > c.round_size_musd:
                raise ValueError(f"{c.name} 的拟投资金额不能超过本轮融资额")
            if c.post_money_musd and c.round_size_musd and c.round_size_musd >= c.post_money_musd:
                raise ValueError(f"{c.name} 的本轮融资额必须小于投后估值")
            if c.post_money_musd and c.investment_musd and c.investment_musd > c.post_money_musd:
                raise ValueError(f"{c.name} 的拟投资金额不能超过投后估值")
        if request.market_data is not None:
            request.market_data = parse_market_data(request.market_data)
        if self.settings.holding_years is not None and (not math.isfinite(self.settings.holding_years) or self.settings.holding_years <= 0):
            raise ValueError("持有年数必须是大于零的有限数值")
        probabilities = self.settings.scenario_probabilities
        if probabilities and (set(probabilities) != set(schemas.SCENARIOS)
                              or any(not math.isfinite(v) or not 0 <= v <= 1 for v in probabilities.values())
                              or not math.isclose(sum(probabilities.values()), 1, abs_tol=1e-8)):
            raise ValueError("情景概率须完整提供乐观、基线、悲观三项，取值在 0 到 1 之间且合计为 1")
        for company in request.companies:
            if company.financial_data is not None:
                company.financial_data = parse_financial_data(company.financial_data)

    def _compare_pairs(self, result: AnalysisResult) -> None:
        self._stage("对比：交换顺序两两比较公司")
        jobs = []
        pairs = list(combinations(result.analyses, 2))
        for a, b in pairs:
            schema = schemas._obj({
                "winner": schemas._enum([a, b, "平局", "证据不足"], "优先跟进的公司"),
                "reason": schemas._str("比较依据和信息缺口"),
            })
            for first, second in ((a, b), (b, a)):
                task = (f"比较下列两家公司，结合赛道评分标准，判断哪家应优先跟进。"
                        f"\n第一家 {first}：{json.dumps(result.analyses[first], ensure_ascii=False)}"
                        f"\n第二家 {second}：{json.dumps(result.analyses[second], ensure_ascii=False)}")
                jobs.append(self._job(COMPARATOR, task, None, f"pair:{first}:{second}",
                                      f"两两比较 · {first} / {second}", schema))
        outputs, errors = self._run_parallel(jobs)
        result.errors.update(errors)
        for a, b in pairs:
            forward = outputs.get(f"pair:{a}:{b}")
            reverse = outputs.get(f"pair:{b}:{a}")
            result.pairwise.append({"companies": [a, b], "forward": forward, "reverse": reverse,
                                    "order_disagrees": bool(forward and reverse and forward["winner"] != reverse["winner"])})

    def _debate(self, result: AnalysisResult) -> None:
        self._stage("辩论：多空双方交换观点")
        for rnd in range(1, (1 if self.settings.debate_rounds is None else self.settings.debate_rounds) + 1):
            jobs = [
                self._job(role, self._debate_task(role, opponent, rnd, result), None,
                          f"{role.key}_r{rnd}", f"{role.name} · 第 {rnd} 轮")
                for role, opponent in ((BULL, BEAR), (BEAR, BULL))
            ]
            outputs, errors = self._run_parallel(jobs)
            result.errors.update(errors)
            for role in (BULL, BEAR):
                if (out := outputs.get(f"{role.key}_r{rnd}")) is not None:
                    result.debate.append({"round": rnd, "side": role.key, "name": role.name, **out})

    def _repair(self, result: AnalysisResult) -> None:
        claims = [c for c in result.fact_check["claims"] if c["verdict"] in ("不实", "无法核实")]
        repair_start = len(result.repairs)
        self._stage("修正：补查原文，纠正或删除缺少支持的主张，并重算相关分析")
        self.repair_notes = json.dumps(claims, ensure_ascii=False)
        materials = self._check_materials(result)
        for source in dict.fromkeys(c["source_agent"] for c in claims):
            if source not in materials:
                raise PipelineError(f"核查发现不实主张，但无法定位来源材料：{source}")
            if source.startswith("pair:"):
                result.repairs.append({"source_agent": source, "claims": [c for c in claims if c["source_agent"] == source],
                                       "before": deepcopy(materials[source])})
                continue
            original = deepcopy(materials[source])
            relevant = [c for c in claims if c["source_agent"] == source]
            task = ("请按原始证据修正以下已提交结果中的问题主张。先补查原文，纠正主体、对象、期间、金额和口径。"
                    "没有原文支持的事实及依赖它的推论必须删除；只有公司披露的写明据公司披露。"
                    "不要编造替代值，不要用无法核实、待核实等标签保留原事实；允许客观说明未披露。"
                    f"\n原结果：{json.dumps(original, ensure_ascii=False)}"
                    f"\n核查结果：{json.dumps(relevant, ensure_ascii=False)}")
            if source == "sector_researcher":
                role, toolbox, schema = SECTOR_RESEARCHER, self._web_toolbox(), schemas.sector_brief_schema(0)
            elif source.startswith("company_researcher:"):
                name = source.split(":", 1)[1]
                role, toolbox, schema = COMPANY_RESEARCHER, Toolbox(self.docs.get(name), self.search, company_website=self.companies[name].website,
                    knowledge=self.knowledge, registry=self.registry, memory=self.memory, memory_sector=self.sector.key,
                    memory_entity=name, network_slots=self._network_slots, document_scope=name, governor=self.governor), None
            elif source.startswith("company_analyst:"):
                name = source.split(":", 1)[1]
                role, toolbox = COMPANY_ANALYST, None
                schema = schemas.analyst_schema(self.sector, [e["evidence_id"] for e in result.fact_sheets[name]["evidence"]])
            elif source == "comparator":
                role, toolbox, schema = COMPARATOR, None, None
            else:
                role = BULL if source.startswith("bull_") else BEAR
                toolbox, schema = None, None
            try:
                revised = self._run(*self._job(role, task, toolbox, f"repair:{source}", f"修正 · {source}", schema))
            except RunStopped:
                raise
            except Exception as e:  # 补查失败时保留原材料，复查会把未修正的主张记入未解决清单
                result.errors[f"repair:{source}"] = str(e)
                result.repairs.append({"source_agent": source, "claims": relevant, "before": original, "failed": str(e)})
                continue
            if source == "sector_researcher":
                revised["candidates"] = original.get("candidates", [])  # 事实补查沿用本轮已选的研究对象
                result.sector_brief = _namespace_sources(revised, "S", original)
            elif source.startswith("company_researcher:"):
                index = list(result.fact_sheets).index(name) + 1
                result.fact_sheets[name] = _namespace_sources(revised, f"C{index}", original)
            elif source.startswith("company_analyst:"):
                result.analyses[source.split(":", 1)[1]] = revised
            elif source == "comparator":
                result.comparison = revised
            else:
                for item in result.debate:
                    if f"{item['side']}_r{item['round']}" == source:
                        item.update(revised)
            result.repairs.append({"source_agent": source, "claims": relevant, "before": original})

        if any(c["source_agent"] == "sector_researcher" or c["source_agent"].startswith("company_researcher:") for c in claims):
            result.repairs.append({"source_agent": "下游分析失效", "claims": [],
                                   "before": deepcopy(memo_materials(result))})
            result.analyses = {}
            result.ranking = []
            result.sensitivity = {}
            result.score_ranges = {}
            result.comparison = {}
            result.debate = []
            result.decision = {}
            result.returns = []
            result.return_summary = {}
            jobs = [self._job(COMPANY_ANALYST, self._analyst_task(n, sheet, result.sector_brief), None,
                              f"reanalysis:{n}", f"重新分析 · {n}",
                              schemas.analyst_schema(self.sector, [e["evidence_id"] for e in sheet["evidence"]]))
                    for n, sheet in result.fact_sheets.items() if sheet["listing_status"] != "已上市"]
            if not jobs:
                raise PipelineError("修正上市状态后没有可分析的未上市候选公司")
            outputs, errors = self._run_parallel(jobs)
            result.errors.update(errors)
            result.analyses = {n: outputs[f"reanalysis:{n}"] for n in result.fact_sheets if f"reanalysis:{n}" in outputs}
            if len(result.analyses) != len(jobs):
                raise PipelineError(f"修正后重新分析失败：{errors}")
        result.ranking = rank_companies(result.analyses, self.sector)
        result.sensitivity = ranking_sensitivity(result.analyses, self.sector)
        result.score_ranges = score_ranges(result.analyses, result.fact_sheets, self.sector)
        result.comparison = {}
        try:
            result.comparison = self._run(*self._job(COMPARATOR, self._comparator_task(result), None, "recomparison"))
        except RunStopped:
            raise
        except Exception as e:
            result.errors["recomparison"] = str(e)
        result.pairwise = []
        if self.settings.pairwise_checks:
            self._compare_pairs(result)
        result.debate = []
        self._debate(result)
        revised_materials = self._check_materials(result)
        for repair in result.repairs[repair_start:]:
            repair["after"] = deepcopy(revised_materials.get(repair["source_agent"]))
        ids = [c["claim_id"] for c in claims]
        task = (self._fact_check_task(result) + "\n\n请逐条复查以下先前问题主张及其依赖推论是否在当前材料中已修正或删除；"
                "repair_checks 必须覆盖每个 claim_id，无法确认时填写无法判断：\n" + json.dumps(claims, ensure_ascii=False))
        try:
            result.fact_check = self._run(*self._job(FACT_CHECKER, task, self._factcheck_toolbox(revised_materials), "fact_checker_recheck",
                                                      schema=schemas.repair_check_schema(ids, list(revised_materials), self._evidence_ids(result))))
        except RunStopped:
            raise
        except Exception as e:  # 复查失败不丢弃已完成的研究，按未复查记入研究稿
            result.errors["fact_checker_recheck"] = str(e)
            result.fact_check = {"claims": [], "summary": "", "repair_checks": []}
        result.check_history.append({"stage": "修正后复查", "result": deepcopy(result.fact_check)})
        self._record_unresolved(result, "研究与辩论修正后复查", claims, result.fact_check)

    def _record_unresolved(self, result: AnalysisResult, stage: str, previous: list[dict], check: dict) -> None:
        """一轮修正后仍未解决的问题不再重试，记入清单，报告降级为研究稿。"""
        checks = {c["claim_id"]: c for c in check.get("repair_checks", [])}
        for claim in previous:
            item = checks.get(claim["claim_id"])
            if item is None or item["status"] not in ("已修正", "已删除"):
                result.unresolved_checks.append({
                    "stage": stage, "claim_id": claim["claim_id"], "claim": claim["claim"],
                    "source_agent": claim["source_agent"], "verdict": claim["verdict"],
                    "problem": item["status"] if item else "未复查", "reason": item["reason"] if item else "",
                })
        for claim in check.get("claims", []):
            if claim["verdict"] in ("不实", "无法核实"):
                result.unresolved_checks.append({
                    "stage": stage, "claim_id": claim.get("claim_id", ""), "claim": claim["claim"],
                    "source_agent": claim["source_agent"], "verdict": claim["verdict"],
                    "problem": claim["verdict"], "reason": claim["evidence"],
                })

    def _review_memo(self, result: AnalysisResult, correction: str = "") -> None:
        self._stage("成稿复核：按实际回报、当前证据和既定评分统一各章")
        calculated = self._returns(result.decision)
        result.returns = calculated["scenarios"]
        result.return_summary = {"expected_multiple": calculated["expected_multiple"], **self._dilution()}
        result.returns_analysis = self._returns_analysis(result.decision)
        before = deepcopy(memo_materials(result))
        task = ("你负责本报告的最终正文复核。统一改写给定各章，不直接拼接不同阶段的结论。\n"
                "本次建议只由最终 decision.action 决定；研究员和分析师的早期投资建议、手算评分应删除。"
                "评分数值以给定排序表为准，只能重写理由。重点公司及退出情景已锁定，不能更改。\n"
                "逐项阅读实际 MOIC/IRR：基线不足1倍时不能无解释地声称有安全边际或回报有吸引力；"
                "可以调整投资建议，不能为配合建议抬高退出假设。未设基金收益门槛时不要编造门槛。\n"
                "同一估值倍数的采用或剔除须在全文一致；不同收入定义、估值时点和币种不得混算。"
                "同口径市场测算集中在第二章；假设不是独立佐证，不能称互相验证。\n"
                "资料中已披露、公司自述、输入表缺项、独立验证缺失是不同情况。原文披露NRR或正毛利，"
                "不能写没有披露留存或毛利；正毛利不代表披露了具体毛利率。\n"
                "同一产品指标核对所有来源日期；历史数字注明当时日期，不写成当前状态。"
                "全球最大、基座非自研、资本效率更高等结论必须有直接依据，融资或收入数字不能证明产品和成本优势。"
                "引用只能用下列当前证据编号或来源编号；内部运行过程、代码、命令行、audit.md不写进正文。\n"
                "保留影响核心判断的多空分歧，标明各自假设及区分它们的观察，将其纳入同一套投资建议。\n\n"
                f"篇幅：最终报告含引用最多{self.settings.report_max_pages}页，全部输出正文控制在{self._char_budget(3500)}字内。"
                "首页判断写公司哪项能力值得持续投入。technical_moat 给出技术机制、事实、最强替代及技术失效条件。交易动作归入估值章节。"
                "当输出要求 section_openings 时，五段应各用一至两句既有事实或有条件的推论，串起客户、技术、经营、估值及风险，并给实际研究主题写一个简短副标题。"
                "研究假设被反证时明确改变立场；背景段落只保留会改变决策的内容，同一事实只讲一次。"
                "市场测算最多400字；赛道摘要不超过100字；"
                "行业格局不超过250字；参照公司最多5条；每家经营摘要不超过150字；"
                "对比保留目标公司和最相关的3家对手。正文把实质异议写成替代解释、风险和验证条件，轮次及发言记录留在底稿。"
                "核心问题回答不超过120字，每个关键问题回答不超过100字；各公司最突出的一点不超过60字、机制不超过80字；"
                "重点公司的竞对差异、成本结构和长期竞争力每条不超过80字。观点只写一次，其他章节引用即可。\n\n"
                f"# 当前可引用证据（同一事实的旧编号不再使用）\n{self._evidence_context(result)}\n\n"
                f"# 既定评分\n{ranking_table(result, self.sector)}\n\n"
                f"# 实际回报与持有期\n{json.dumps(calculated, ensure_ascii=False)}\n\n"
                f"# 交易及经营测算依据 [U-1]\n{json.dumps({'deal': result.deal_metrics, 'financial': result.financial_metrics, 'market': result.market_metrics}, ensure_ascii=False)}\n\n"
                "下列正文包含待修订的意见。核对其中的事实前提，其他 Agent 重复过某项主张也不构成事实依据。\n"
                f"# 待统一正文\n{json.dumps(before, ensure_ascii=False)}\n\n{correction}")
        review = self._run(*self._job(CHAIR, task, None, "chair_editor", "主席 · 正文复核",
                                     schemas.memo_review_schema(self.sector, list(result.analyses), result.decision["top_pick"], len(result.debate),
                                                                self._evidence_ids(result), self._question_ids(result),
                                                                single_company=len(result.companies) == 1)))
        result.decision = {**review["decision"], "top_pick": result.decision["top_pick"],
                           "exit_scenarios": result.decision["exit_scenarios"]}
        result.sector_brief.update(review["sector"])
        for name, narrative in review["companies"].items():
            reasons = narrative.pop("score_reasons")
            result.analyses[name].update(narrative)
            for dimension, reason in reasons.items():
                score = result.analyses[name]["scores"][dimension]
                score["reason"], score["evidence_ids"] = reason["reason"], reason["evidence_ids"]
        result.comparison = review["comparison"]
        for original, revised in zip(result.debate, review["debate"]):
            original.update(revised)
        result.memo_reviews.append({"before": before, "returns": deepcopy(calculated),
                                    "after": deepcopy(memo_materials(result))})

    def _fit_report_budget(self, result: AnalysisResult) -> None:
        """One layout-driven revision, before the final evidence/reasoning review."""
        from .brief import render_brief
        from .pdf_report import build_report_pdf, ReportPageLimitError

        def pages():
            return build_report_pdf(render_brief(result), result.report_max_pages)[1]

        try:
            count = pages()
            self._stage(f"篇幅预检：当前 {count} 页，上限 {result.report_max_pages} 页")
            return
        except ReportPageLimitError as exc:
            problem = str(exc)
        self._stage("篇幅修订：主 Agent 压缩重复叙述，随后重新核查事实与投资判断")
        self._review_memo(result,
            "篇幅预检没有通过：" + problem + f"。本轮将叙述正文压缩至{self._char_budget(2200)}字以内，删去重复背景与次要事实；"
            "保留最终建议、关键支持及反证、下行情景、交易口径和所用来源。"
            "压缩整句，不截断句子，不隐藏重大风险，不改变评分、重点公司或退出情景。"
            "未使用的证据无需逐条复述。排版信息和压缩过程不要写入投资报告。")
        try:
            count = pages()
            self._stage(f"篇幅修订完成：{count} 页，进入最终核查")
        except ReportPageLimitError as exc:
            result.unresolved_checks.append({"stage": "篇幅预检", "claim_id": "", "claim": "报告篇幅仍超过预算",
                "source_agent": "chair_editor", "verdict": "", "problem": "篇幅修订未通过", "reason": str(exc)})

    def _evidence_context(self, result: AnalysisResult) -> str:
        return evidence_context(result.sector_brief, *result.fact_sheets.values())

    def _char_budget(self, at_ten_pages: int) -> int:
        """Narrative word budgets were calibrated for a 10-page report; scale them with the page cap."""
        return round(at_ten_pages * self.settings.report_max_pages / 10 / 100) * 100

    @staticmethod
    def _question_ids(result: AnalysisResult) -> list[str]:
        return [q["question_id"] for q in (result.research_state or {}).get("questions", [])]

    def _decision_schema(self, result: AnalysisResult) -> dict:
        return schemas.decision_schema(list(result.analyses), self._question_ids(result))

    def _reasoning_context(self, result: AnalysisResult) -> str:
        return ("\n# 必须逐项复核的投资判断\n"
                "reasoning_checks 必须覆盖以下每个编号。这是推论审查，不是寻找相同数字。"
                "逐项检查主体、指标、期间、比较基准及因果链。M1 核对首页主张、技术归属、技术依据与复刻难度。"
                "报价、续费和单位成本只能说明交易或商业兑现，不能单独证明技术壁垒。"
                "原文真实但不能推出结论时判依据不足；与原文矛盾判与证据冲突。"
                "条件性推断只适用于正文已明确写出条件的情景；条件尚未满足不能写成现有优势。"
                "负面判断也须有依据，缺少披露不等于经营差。\n"
                + json.dumps(review_targets(result.decision), ensure_ascii=False)
                + "\n# 交易输入与确定性测算 [U-1]\n"
                + json.dumps({"deal": result.deal_metrics, "financial": result.financial_metrics,
                              "market": result.market_metrics, "returns": result.returns,
                              "return_summary": result.return_summary}, ensure_ascii=False))

    def _check_decision(self, result: AnalysisResult) -> None:
        self._stage("核查：检查事实与核心投资推论")
        material = memo_materials(result)
        task = ("请核查以下最终正文中的关键事实，source_agent 必须写 chair：\n" + json.dumps(material, ensure_ascii=False)
                + self._evidence_priorities(result, {"chair": material})
                + "\n# 当前证据与来源（含已读原文与引用位置，可直接对照）\n" + self._evidence_context(result)
                + self._reasoning_context(result))
        if self.fact_docs:
            task += "\n\n可核查的上传资料：\n" + self.fact_docs.overview()
        try:
            result.decision_fact_check = self._run(*self._job(FACT_CHECKER, task, self._factcheck_toolbox({"chair": material}), "fact_checker_decision",
                                                               schema=schemas.decision_check_schema(result.decision, self._evidence_ids(result, verified_only=True))))
        except RunStopped:
            raise
        except Exception as e:
            result.errors["fact_checker_decision"] = str(e)
            return
        result.decision_fact_check["reviewed_targets"] = review_targets(result.decision)
        for i, claim in enumerate(result.decision_fact_check["claims"], 1):
            claim["claim_id"] = f"D-{i}"
        result.check_history.append({"stage": "主席结论", "result": deepcopy(result.decision_fact_check)})
        bad = [c for c in result.decision_fact_check["claims"] if c["verdict"] in ("不实", "无法核实")]
        reasoning_bad = review_issues(result.decision, result.decision_fact_check)
        if not bad and not reasoning_bad:
            return
        original = deepcopy(result.decision)
        # 只有原文明确矛盾，或根本找不到原文的主张才回溯修正上游材料；
        # 原文已对上、只是数字或写法没核对上的，只让主席改写对应句子，不触发整轮修正
        upstream = [c for c in bad if c["verdict"] == "不实" or not c.get("excerpt_verified", False)]
        affected = affected_materials(self._check_materials(result), upstream)
        if affected:
            propagated = [claim for claims in affected.values() for claim in claims]
            for i, claim in enumerate(propagated, 1):
                claim["claim_id"] = f"D-upstream-{i}"
            research_checks = deepcopy(result.fact_check)
            result.fact_check = {"claims": propagated, "summary": "主席核查问题回传到上游材料"}
            self._repair(result)
            result.check_history.append({"stage": "主席问题传播前的研究核查", "result": research_checks})
            current = result.fact_check.get("claims", [])
            if current:
                result.fact_check_pass_rate = round(sum(c["verdict"] == "属实" for c in current) / len(current), 3)
                result.checked_source_coverage = round(sum(bool(c["original_source"].strip()) for c in current) / len(current), 3)
        correction = ("以下事实缺少支持或存在错误，请根据已读原文更正；不能证实的事实及依赖推论删除。"
                      "不得保留无法核实、待核实标签，不得编造替代值；公司自述注明据公司披露。\n"
                      + json.dumps({"facts": bad, "reasoning": reasoning_bad}, ensure_ascii=False))
        if bad:
            result.decision = self._run(*self._job(CHAIR, self._chair_task(result) + "\n" + correction, None,
                                                   "repair:chair", schema=self._decision_schema(result)))
        self._review_memo(result, correction)
        result.repairs.append({"source_agent": "chair", "claims": bad, "reasoning": reasoning_bad,
                              "before": original, "after": deepcopy(result.decision)})
        ids = [c["claim_id"] for c in bad]
        revised_memo = memo_materials(result)
        task = ("请核查修正后的主席结论，source_agent 必须写 chair。"
                "逐一检查先前不实主张在当前结论中是否已修正或删除，repair_checks 必须覆盖每个 claim_id：\n"
                + json.dumps({"previous_claims": bad, "report": revised_memo}, ensure_ascii=False)
                + self._evidence_priorities(result, {"chair": revised_memo})
                + "\n# 当前证据与来源\n" + self._evidence_context(result) + self._reasoning_context(result))
        if self.fact_docs:
            task += "\n\n可核查的上传资料：\n" + self.fact_docs.overview()
        try:
            result.decision_fact_check = self._run(*self._job(FACT_CHECKER, task, self._factcheck_toolbox({"chair": memo_materials(result)}),
                                                              "fact_checker_decision_recheck", schema=schemas.decision_check_schema(result.decision, self._evidence_ids(result, verified_only=True), ids)))
            result.decision_fact_check["reviewed_targets"] = review_targets(result.decision)
        except RunStopped:
            raise
        except Exception as e:
            result.errors["fact_checker_decision_recheck"] = str(e)
            result.decision_fact_check = {"claims": [], "summary": "", "repair_checks": []}
        result.check_history.append({"stage": "主席结论修正后复查", "result": deepcopy(result.decision_fact_check)})
        self._record_unresolved(result, "主席结论修正后复查", bad, result.decision_fact_check)
        for issue in review_issues(result.decision, result.decision_fact_check):
            result.unresolved_checks.append({"stage": "核心推论复查", "claim_id": issue["target_id"],
                "claim": issue["target"], "source_agent": "chair", "verdict": issue["verdict"],
                "problem": "投资判断未通过复核", "reason": issue["reason"]})

    # ------------------------------------------------------------ 运行智能体

    def _job(self, role: Role, task: str, toolbox: Toolbox | None, agent_id: str = "", name: str = "",
             schema: dict | None = None) -> Job:
        return role, task, toolbox, agent_id or role.key, name or role.name, schema

    def _company_job(self, c: CompanyInput) -> Job:
        toolbox = Toolbox(self.docs.get(c.name), self.search, company_website=c.website, knowledge=self.knowledge,
                          registry=self.registry, memory=self.memory, memory_sector=self.sector.key, memory_entity=c.name,
                          network_slots=self._network_slots, document_scope=c.name, governor=self.governor)
        return self._job(COMPANY_RESEARCHER, self._company_task(c), toolbox,
                         f"company_researcher:{c.name}", f"公司研究员 · {c.name}")

    def _web_toolbox(self) -> Toolbox:
        return Toolbox(None, self.search, knowledge=self.knowledge, registry=self.registry,
                       memory=self.memory, memory_sector=getattr(getattr(self, "sector", None), "key", ""), network_slots=self._network_slots,
                       governor=self.governor)

    def _factcheck_toolbox(self, materials: dict | None = None) -> Toolbox:
        with self._lock:
            shared = [read for reads in self._source_reads.values() for read in reads if read["kind"] != "document"]
        aliases = {c.name for c in self.companies.values()} if hasattr(self, "companies") else set()
        if self.last_result:
            aliases |= {sheet.get("company_name", "") for sheet in self.last_result.fact_sheets.values()}
        inputs = grounding.user_entry(self.last_result) if self.last_result else {}
        return Toolbox(self.fact_docs, self.search, knowledge=self.knowledge, shared_reads=shared,
                       registry=self.registry, materials=materials, known_values=inputs.get("values"), aliases=aliases,
                       input_amounts=inputs.get("amounts"),
                       memory=self.memory, memory_sector=getattr(getattr(self, "sector", None), "key", ""), network_slots=self._network_slots,
                       governor=self.governor, rejection_review=self.rejection_reviewer.filter)

    def _run(self, role: Role, task: str, toolbox: Toolbox | None, agent_id: str, name: str,
             schema: dict | None = None) -> dict:
        self.governor.check()
        if toolbox is not None:
            toolbox.rejection_review = self.rejection_reviewer.filter
        started = time.monotonic()
        if agent_id != "lead_planner" and self.last_result and self.last_result.research_plan:
            task += ("\n\n# 主 Agent 的研究议程（待证伪，不是事实来源）\n"
                     + json.dumps(self.last_result.research_plan, ensure_ascii=False)
                     + f"\n只写与你分工有关的发现。保留已发现的相反事实，分析者也可提出有明确前提的逻辑质疑。最终报告最多{self.settings.report_max_pages}页，勿重复其他角色内容。")
        if self.repair_notes and role != FACT_CHECKER:
            task += "\n\n以下核查结论指出的错误不能作为论据：\n" + self.repair_notes
        extra_check = None
        if self.settings.mock:
            pass
        elif toolbox is not None and role in (SECTOR_RESEARCHER, COMPANY_RESEARCHER, FACT_CHECKER):
            extra_check = lambda data: toolbox.citation_errors(role.key, data)
        elif role in (COMPANY_ANALYST, COMPARATOR, BULL, BEAR, CHAIR) and self.last_result is not None and agent_id != "lead_planner":
            extra_check = self._evidence_first_check(agent_id)
        agent = Agent(
            role, self.router.for_role(role.key), toolbox, self.emit,
            system=system_prompt(role, self.sector, self.settings.fund_profile, self.today, self.settings.report_max_pages),
            schema=schema, agent_id=agent_id, name=name,
            max_steps=(min(4, self.settings.max_agent_steps) if agent_id == "chair_editor" else
                       max(self.settings.max_agent_steps, self.settings.role_max_steps.get(role.key, 0))),
            extra_check=extra_check,
        )
        try:
            data = agent.run(task)
            if agent.unresolved_issues and self.last_result is not None:
                with self._lock:
                    self.last_result.unresolved_checks.append({
                        "stage": "内容检查", "claim_id": "", "claim": f"{name} 提交后仍有内容问题", "source_agent": agent.id,
                        "verdict": "", "problem": "内容检查未通过", "reason": "；".join(agent.unresolved_issues[:5])})
            if role in (SECTOR_RESEARCHER, COMPANY_RESEARCHER):
                data = toolbox.verify_evidence(data)
            elif role == FACT_CHECKER and not self.settings.mock:
                data = toolbox.verify_fact_check(data, [c.website for c in self.companies.values() if c.website])
            elif role == FACT_CHECKER:
                data = normalize_fact_check(data)
            return data
        except Exception:
            self.governor.check()  # 停止信号优先于它在框架内部引起的次生错误
            raise
        finally:
            with self._lock:
                if toolbox is not None:
                    self._source_reads[agent.id] = toolbox.read_records()
                previous = self._usage.get(agent.id, [0, 0])
                self._usage[agent.id] = [previous[0] + agent.input_tokens, previous[1] + agent.output_tokens]
                self._format_failures[agent.id] = self._format_failures.get(agent.id, 0) + agent.format_failures
                self._agent_seconds[agent.id] = round(self._agent_seconds.get(agent.id, 0) + time.monotonic() - started, 2)
                if agent.id not in self._costs:
                    self._costs[agent.id] = agent.cost_usd
                elif self._costs[agent.id] is not None and agent.cost_usd is not None:
                    self._costs[agent.id] += agent.cost_usd
                else:
                    self._costs[agent.id] = None

    # ------------------------------------------------------------ 证据先行

    def _evidence_first_check(self, agent_id: str):
        """每次提交都检查，修正次数由 Agent 统一限制。"""
        def check(data: dict) -> list[str]:
            return grounding.errors_for_submission(self._evidence_report(agent_id, data))
        return check

    def _evidence_report(self, name: str, data) -> dict:
        r = self.last_result
        with self._lock:
            index = grounding.evidence_index(deepcopy(r.sector_brief), deepcopy(r.fact_sheets))
        index[grounding.USER_KEY] = grounding.user_entry(r)
        return grounding.check_material(name, data, index)

    def _evidence_priorities(self, result: AnalysisResult, materials: dict) -> str:
        """定位数字与引用问题，语义支持仍由核查员判断。"""
        reports = {name: self._evidence_report(name, data) for name, data in materials.items()
                   if name != "sector_researcher" and not name.startswith("company_researcher:")}
        if "chair" in reports:
            reports["chair"] = apply_report_reviews(reports["chair"], result.rejection_reviews,
                                                    self.rejection_reviewer.review_amount)
        result.evidence_checks.update(reports)
        findings = [f for report in reports.values() for f in report["findings"]]
        amounts = [item for report in reports.values() for item in report.get("reviewed_amounts", [])]
        supported = sum(report["supported"] for report in reports.values())
        if not findings and not supported:
            return ""
        lines = [f"- [{f['material']}] {f['sentence'][:120]}（{f['problem']}：{'、'.join(f['numbers'])}）" for f in findings[:20]]
        more = f"\n另有 {len(findings) - 20} 句同类问题。" if len(findings) > 20 else ""
        table_units = ("\n下列金额已按明确的表格单位复算。审计仍须核对原行的主体、指标与期间，引用范围包含单位和数值行。"
                       "结构化事实中的原文数值请保留原写法。\n" + json.dumps(amounts, ensure_ascii=False)) if amounts else ""
        return ("\n\n代码已初步核对正文中的金额、比例、倍数和模型数量：" + f"{supported} 句数值与引用原文匹配。"
                "数值匹配不证明主体、指标、时间、口径或推论一致，仍须核查关键主张；也要检查没有数字的产品、团队和条款结论。"
                + ("以下句子有证据、数值或口径疑点，请优先从中挑选核查：\n" + "\n".join(lines) + more if lines else "")
                + table_units)

    def _run_parallel(self, jobs: list[Job]) -> tuple[dict[str, dict], dict[str, str]]:
        if not jobs:
            return {}, {}
        scheduler = DAGScheduler(self.settings.max_workers, self.governor)
        try:
            outcomes = scheduler.run([TaskNode(job[3], lambda j=job: self._run(*j)) for job in jobs])
        finally:
            if self.last_result:
                self.last_result.task_trace.extend(scheduler.trace)
        return ({key: out.value for key, out in outcomes.items() if not out.error},
                {key: out.error for key, out in outcomes.items() if out.error})

    def _stage(self, message: str) -> None:
        self.emit(Event("stage", "", "", message))

    def _dilution(self) -> dict:
        if self.settings.exit_dilution is None:
            return {"dilution": DEFAULT_EXIT_DILUTION, "dilution_source": "系统默认"}
        return {"dilution": self.settings.exit_dilution, "dilution_source": "用户设定"}

    def _returns_analysis(self, decision: dict) -> dict:
        company = self.companies.get(decision["top_pick"])
        if company is None:
            return {}
        return returns_analysis(company.post_money_musd, company.investment_musd, self._dilution()["dilution"],
                                decision["exit_scenarios"], self.settings.holding_years,
                                self.settings.management_fee, self.settings.carry,
                                self.settings.liquidation_preference, company.preference_stack_musd)

    def _returns(self, decision: dict) -> dict:
        company = self.companies.get(decision["top_pick"])
        post_money = company.post_money_musd if company else None
        investment = company.investment_musd if company else None
        scenarios = [{**s, "probability": self.settings.scenario_probabilities.get(s["scenario"]),
                      "holding_years": self.settings.holding_years} for s in decision["exit_scenarios"]]
        return calculate_returns(post_money, self._dilution()["dilution"], scenarios, investment)

    # ------------------------------------------------------------ 各角色的任务输入

    def _company_block(self, c: CompanyInput) -> str:
        lines = [f"公司名称：{c.name}"]
        if c.website:
            lines.append(f"官网：{c.website}")
        if c.urls:
            lines.append("参考链接：\n" + "\n".join(f"- {u}" for u in c.urls))
        if c.notes:
            lines.append(f"用户补充信息：\n{c.notes}")
        return "\n".join(lines)

    def _sector_task(self, request: SectorRequest, companies: list[CompanyInput]) -> str:
        parts = [f"# 任务\n研究赛道「{self.sector.name}」。"]
        if request.focus:
            parts.append(f"希望重点关注：{request.focus}")
        if companies:
            parts.append(f"用户要分析的公司：{'、'.join(c.name for c in companies)}。candidates 给空数组。")
        else:
            parts.append("请按与赛道的相关性和可核实的选取依据，在 candidates 中推荐恰好 "
                         f"{request.max_companies} 家不同的早期未上市公司。每家只写一个公司名，勿编造标的凑数。")
        if request.market_data:
            parts.append("用户提供的赛道级市场测算输入（来源说明未经独立核验；market_size 应说明与其口径是否一致，不改写输入）：\n"
                         + json.dumps(request.market_data, ensure_ascii=False))
        return "\n\n".join(parts)

    def _company_task(self, c: CompanyInput) -> str:
        docs = self.docs.get(c.name)
        overview = f"\n\n## 上传资料\n{docs.overview()}" if docs else "\n\n（没有上传资料）"
        if c.financial_data:
            overview += ("\n\n## 用户提供的测算输入（未独立核验）\n" + json.dumps(c.financial_data, ensure_ascii=False)
                         + "\n如与实际资料冲突，记录差异及来源，不替用户改写原始输入。")
        return f"# 目标公司\n{self._company_block(c)}{overview}\n\n请整理这家公司的事实底稿。"

    def _analyst_task(self, name: str, sheet: dict, brief: dict) -> str:
        return (
            f"# 目标公司\n{self._company_block(self.companies[name])}\n上市状态：{sheet['listing_status']}；依据：{sheet['listing_evidence']}\n\n"
            "# 可用事实：公司与赛道原文证据\n" + evidence_context(brief, sheet) + "\n\n"
            "事实判断仅依据原文 excerpt 的主体、指标、期间和限定条件。statement/claim 是研究员摘要，"
            "不能扩展原文含义；公司自述须注明据公司披露。未定位的底稿文字不作为事实输入。\n\n"
            "# 用户提供的结构化经营输入及代码计算结果（来源说明未经独立核验）\n"
            + json.dumps(self._financial_metrics.get(name, {}), ensure_ascii=False) + "\n\n"
            f"信息缺口：{'；'.join(sheet['information_gaps']) or '无'}\n\n"
            "请按赛道评分标准给这家公司打分，并写一页纸结论。"
        )

    def _comparator_task(self, r: AnalysisResult) -> str:
        parts = [
            "# 可用事实：已定位原文\n" + self._evidence_context(r),
            "以下摘要和分析均为待评估的观点，事实只能以原文摘录为依据。",
            f"# 赛道背景\n{r.sector_brief['summary']}\n\n{r.sector_brief['landscape']}\n\n参照公司：\n"
            + "\n".join(f"- {b}" for b in r.sector_brief["benchmarks"]),
            f"# 代码给出的排序\n{ranking_table(r, self.sector)}",
            f"# 排名敏感性\n{json.dumps(r.sensitivity, ensure_ascii=False)}\n\n"
            f"# 信息缺口假设范围\n{json.dumps(r.score_ranges, ensure_ascii=False)}",
            _analysis_digest(r),
        ]
        if len(r.analyses) == 1:
            name = next(iter(r.analyses))
            return "\n\n".join(parts + [
                f"这是 {name} 的单公司投资报告，不做多公司排序。请把 {name} 与赛道研究中的主要竞争对手和参照公司逐项对比："
                "产品与客户、收入规模与增速、估值倍数、壁垒。per_company 第一条写目标公司，其后每个具名对手一条，"
                "edge 和 gap 写相对目标公司的优势与短板，数字标注来源编号。summary 说明与最接近的竞对相比，"
                "在最影响判断的一点上目标公司领先、相当、落后，还是现有材料不足以判断。"
                "ranking_view 写目标公司相对对手最站得住和最站不住的一点。"])
        if r.ranking and r.ranking[0]["topsis_disagrees"]:
            parts.append("注意：加权分和 TOPSIS 给出的第一名不一致，请重点判断。")
        return "\n\n".join(parts + ["请复核这个排序。"])

    def _debate_task(self, role: Role, opponent: Role, rnd: int, r: AnalysisResult) -> str:
        parts = [
            "# 可用事实：已定位原文\n" + self._evidence_context(r),
            "以下摘要和分析均为待评估的观点，不得因多个角色重复而当成独立佐证。",
            f"# 赛道：{self.sector.name}\n{r.sector_brief['summary']}",
            f"# 排序\n{ranking_table(r, self.sector)}\n\n代码排序的首选公司：{r.ranking[0]['company']}",
            f"# 排名敏感性与信息缺口假设范围\n{json.dumps({'sensitivity': r.sensitivity, 'score_ranges': r.score_ranges}, ensure_ascii=False)}",
        ]
        if r.comparison:
            parts.append(f"# 对比分析师的意见\n{r.comparison['summary']}\n\n{r.comparison['ranking_view']}")
        parts.append(_analysis_digest(r))
        if rnd > 1:
            previous = [d for d in r.debate if d["side"] == opponent.key and d["round"] == rnd - 1]
            if previous:
                parts.append(f"# {opponent.name}上一轮的观点\n{json.dumps(previous[0], ensure_ascii=False, indent=1)}")
            parts.append(f"这是第 {rnd} 轮，请在 rebuttals 中逐条回应{opponent.name}，并完善你的论点。")
        else:
            parts.append("这是第 1 轮，请陈述你的立场。")
        return "\n\n".join(parts)

    def _check_materials(self, r: AnalysisResult) -> dict:
        materials = {"sector_researcher": r.sector_brief, "comparator": r.comparison}
        materials.update({f"company_researcher:{n}": v for n, v in r.fact_sheets.items()})
        materials.update({f"company_analyst:{n}": v for n, v in r.analyses.items()})
        materials.update({d.get("phase") or f"{d['side']}_r{d['round']}": d for d in r.debate})
        materials.update({f"pair:{p['companies'][0]}:{p['companies'][1]}": p for p in r.pairwise})
        return materials

    def _evidence_ids(self, r: AnalysisResult, *, verified_only: bool = False) -> list[str]:
        records = r.sector_brief["evidence"] + [e for sheet in r.fact_sheets.values() for e in sheet["evidence"]]
        return [e["evidence_id"] for e in records if not verified_only or e.get("excerpt_verified")]

    def _fact_check_task(self, r: AnalysisResult, keys: list[str] | None = None) -> str:
        materials = self._check_materials(r)
        if keys is not None:
            materials = {k: v for k, v in materials.items() if k in keys}
        task = "请核查以下材料中影响结论的关键主张，写明具体材料和来源：\n" + json.dumps(materials, ensure_ascii=False)
        task += self._evidence_priorities(r, materials)
        if self.fact_docs:
            task += "\n\n可核查的上传资料：\n" + self.fact_docs.overview()
        return task

    def _chair_task(self, r: AnalysisResult) -> str:
        debate = "\n\n".join(
            f"## {d['name']} · 第 {d['round']} 轮\n"
            + json.dumps({k: d[k] for k in ("addressed_claim", "thesis", "arguments", "rebuttals", "would_change_mind") if k in d},
                         ensure_ascii=False, indent=1)
            for d in r.debate
        )
        checks = "\n".join(
            f"- [{c['verdict']}] {c['claim']}；来源：{c['original_source']}；核查依据：{c['evidence']}"
            + (f"；原文核查来源：{c['verification_source']}；原文摘录：{c['verification_excerpt']}；匹配状态：{c.get('verification_status', '未执行')}"
               if c.get("verification_source") else "")
            + (f"（核查建议更正：{c['correction']}）" if c["verdict"] == "不实" and c["correction"] else "")
            for c in r.fact_check.get("claims") or []
        )
        deals = {n: v for n, v in r.deal_metrics.items() if n in self.companies}
        missing = "、".join(k for k in r.errors)
        return "\n\n".join([
            "# 事实依据：已定位原文\n" + self._evidence_context(r),
            "以下研究摘要和多空意见是待评估的观点，不是独立证据。事实只从原文摘录中取；"
            "推论逐项写明前提，原文未支持的结论不能因多个角色重复而视为成立。",
            f"# 赛道研究\n{r.sector_brief['summary']}\n\n{r.sector_brief['market_size']}\n\n为什么是现在：\n"
            + "\n".join(f"- {w}" for w in r.sector_brief["why_now"]) + f"\n\n{r.sector_brief['landscape']}",
            f"# 排序（代码计算）\n{ranking_table(r, self.sector)}",
            f"# 排名敏感性（每次只将一家公司一个维度的分数加或减 1 分）\n{json.dumps(r.sensitivity, ensure_ascii=False)}",
            f"# 证据不足维度的分数假设范围（非概率）\n{json.dumps(r.score_ranges, ensure_ascii=False)}",
            f"# 对比分析师的意见\n{json.dumps(r.comparison, ensure_ascii=False, indent=1) if r.comparison else '（对比未完成）'}",
            f"# 交换顺序的两两比较\n{json.dumps(r.pairwise, ensure_ascii=False)}",
            _analysis_digest(r, detailed=True),
            f"# 多空辩论\n{debate or '（辩论未产出结果）'}",
            f"# 事实核查（抽查主张独立证据支持率 {_pct(r.fact_check_pass_rate)}）\n{checks or '（核查未完成）'}",
            "# 用户交易输入与代码计算结果（估值、拟投资金额、本轮融资额、持股比例、隐含 ARR 倍数、本轮后跑道；正文引用标注 [U-1]）\n"
            + (json.dumps(deals, ensure_ascii=False) if deals else "用户没有提供本轮估值或交易条款"),
            f"# 用户经营输入与代码计算结果（不得改写输入口径）\n{json.dumps(r.financial_metrics, ensure_ascii=False)}",
            f"# 赛道级市场测算（用户输入，代码计算）\n{json.dumps(r.market_metrics, ensure_ascii=False) if r.market_metrics else '用户没有提供结构化市场输入'}",
            "# 退出测算假设\n" + json.dumps({"holding_years": self.settings.holding_years, **self._dilution()}, ensure_ascii=False),
            "用户未提供情景概率或持有年数时不要补造；未来收入、利润、现金预测缺少基础数据或逐年假设时说明无法预测。"
            "排名第一不等于值得投资；没有合适标的时 action 给不投、recommended 给空数组。"
            "core_answer 直接回答研究议程的核心问题；gap_impacts 只写会改变判断的信息缺口。",
            f"未完成的智能体：{missing}" if missing else "",
            (f"这是 {next(iter(r.analyses))} 的单公司投资报告：top_pick 写这家公司，recommended 只能为空或只含这家；"
             "说明目标公司相对最接近的具名对手处于什么位置、这如何影响建议。证据不足时可以暂缓判断。") if len(r.analyses) == 1 else "",
            "请做出最终结论。",
        ])


# ---------------------------------------------------------------- 共用的文本片段

def _merge_company_shards(name: str, shards: list[dict]) -> dict:
    """Merge parallel research without another model inventing bridging facts."""
    if len(shards) == 1:
        return {**shards[0], "company_name": name}
    namespaced = [_namespace_sources(s, f"L{i}") for i, s in enumerate(shards, 1)]
    statuses = {s.get("listing_status", "待核实") for s in shards}
    # A conflict remains explicit; missing observations do not negate a disclosure.
    known = statuses - {"待核实"}
    listing = next(iter(known)) if len(known) == 1 else "待核实"
    gaps = list(dict.fromkeys(g for s in namespaced for g in s.get("information_gaps", [])))
    if len(known) > 1:
        gaps.append("并行研究的上市状态互相冲突，须读取监管或交易所原文复核")
    return {
        "company_name": name, "one_liner": shards[0].get("one_liner", ""),
        "stage": next((s["stage"] for s in shards if s.get("stage") not in (None, "", "未知")), "未知"),
        "listing_status": listing,
        "listing_evidence": "\n".join(dict.fromkeys(s.get("listing_evidence", "") for s in namespaced)),
        "fact_sheet": "\n\n".join(s.get("fact_sheet", "") for s in namespaced),
        "information_gaps": gaps,
        "sources": [source for s in namespaced for source in s["sources"]],
        "evidence": [e for s in namespaced for e in s["evidence"]],
    }


def evidence_context(*sheets: dict) -> str:
    keys = ("evidence_id", "subject", "fact_type", "statement", "excerpt", "as_of", "source_type",
            "source_ref", "quantities", "entities", "date", "quote", "date_quote", "quote_location")
    return json.dumps({"evidence": [{k: e[k] for k in keys if k in e} for sheet in sheets
                                    for e in sheet["evidence"] if e.get("excerpt_verified")],
                       "sources": [source for sheet in sheets for source in sheet["sources"]]}, ensure_ascii=False)

def ranking_table(r: AnalysisResult, sector: Sector) -> str:
    head = "| 排名 | 公司 | 加权分 | TOPSIS | " + " | ".join(d.name for d in sector.dimensions) + " |"
    sep = "|" + " --- |" * (4 + len(sector.dimensions))
    rows = [
        f"| {row['rank']} | {row['company']} | {row['weighted']} | {row['topsis']} | "
        + " | ".join(str(row["scores"][d.key]) for d in sector.dimensions) + " |"
        for row in r.ranking
    ]
    return "\n".join([head, sep, *rows])


def _analysis_digest(r: AnalysisResult, detailed: bool = False) -> str:
    blocks = []
    for row in r.ranking:
        a = r.analyses[row["company"]]
        reasons = "\n".join(f"- {k}：{v['score']} 分，置信度{v['confidence']}，{v['reason']}；证据：{', '.join(v['evidence_ids']) or '无'}" for k, v in a["scores"].items())
        block = (
            f"## {row['company']}（置信度{a['confidence']}，上市状态{r.fact_sheets[row['company']]['listing_status']}）\n{a['one_liner']}\n{a['summary']}\n"
            f"打分理由：\n{reasons}\n亮点：{'；'.join(a['strengths'])}\n担忧：{'；'.join(a['concerns'])}"
            f"\n团队与分工：{a.get('team_assessment', '未结构化记录')}"
            f"\n产品与商业模式：{a.get('product_business', '未结构化记录')}"
            f"\n经营事实：{a.get('operating_assessment', '未结构化记录')}"
        )
        if row["company"] in r.financial_metrics:
            block += "\n基于用户输入的代码测算（输入未独立核实）：" + json.dumps(r.financial_metrics[row["company"]]["metrics"], ensure_ascii=False)
        if detailed:
            block += f"\nAI 原生程度：{a['ai_native']}\n待核实：{'；'.join(a['open_questions'])}"
        blocks.append(block)
    return "# 各公司分析\n" + "\n\n".join(blocks)


def _namespace_sources(data: dict, prefix: str, previous: dict | None = None) -> dict:
    previous = previous or {}
    old_sources = {_source_identity(s): re.search(r"\[([^]]+)\]", s).group(1) for s in previous.get("sources", [])}
    next_source = previous.get("_next_source_id", 1 + max([int(s.rsplit("-", 1)[1]) for s in old_sources.values()] or [0]))
    sources = data["sources"]
    mapping, source_names = {}, []
    for i, source in enumerate(sources, 1):
        source_id = old_sources.get(_source_identity(source))
        if source_id is None:
            source_id = f"{prefix}-{next_source}"
            next_source += 1
        source_names.append(source_id)
        leading = re.match(r"\s*((?:\[[A-Za-z0-9_-]+\]\s*)+)", source)
        labels = re.findall(r"\[([A-Za-z0-9_-]+)\]", leading.group(1)) if leading else [str(i)]
        for label in labels:
            mapping[label] = source_id

    def replace(value):
        if isinstance(value, str):
            return re.sub(r"\[([A-Za-z0-9_-]+)\]", lambda m: f"[{mapping.get(m.group(1), m.group(1))}]", value)
        if isinstance(value, list):
            return [replace(item) for item in value]
        if isinstance(value, dict):
            return {key: item if key == "excerpt" else replace(item) for key, item in value.items()}
        return value

    updated = replace(data)
    updated["sources"] = [f"[{source_id}] " + re.sub(r"^\s*(?:\[[A-Za-z0-9_-]+\]\s*)+", "", s)
                          for source_id, s in zip(source_names, updated["sources"])]
    source_ids = {match.group(1) for s in updated["sources"] if (match := re.match(r"\s*\[([^]]+)\]", s))}
    old_evidence = {_evidence_identity(e): e["evidence_id"] for e in previous.get("evidence", [])}
    next_evidence = previous.get("_next_evidence_id", 1 + max([int(e.rsplit("-", 1)[1]) for e in old_evidence.values()] or [0]))
    for i, item in enumerate(updated["evidence"], 1):
        local_id = item.get("evidence_id", f"E-{i}")
        evidence_id = old_evidence.get(_evidence_identity(item))
        if evidence_id is None:
            evidence_id = f"E-{prefix}-{next_evidence}"
            next_evidence += 1
        item["evidence_id"] = evidence_id
        mapping[local_id] = evidence_id
        mapping[f"E-{i}"] = evidence_id
        refs = re.findall(r"\[([^]]+)\]", item["source_ref"])
        item["source_linked"] = (bool(refs) and all(ref in source_ids for ref in refs)) or bool(item.get("excerpt_verified", False))
    updated["_next_source_id"], updated["_next_evidence_id"] = next_source, next_evidence
    return replace(updated)


def _source_identity(source: str) -> str:
    url = re.search(r"https?://[^\s]+", source)
    return url.group(0) if url else re.sub(r"^\s*(?:\[[^]]+\]\s*)+", "", source).strip()


def _evidence_identity(evidence: dict) -> tuple[str, ...]:
    return tuple(re.sub(r"\s+", " ", evidence.get(key, "")).strip() for key in ("claim", "source_ref", "excerpt"))


def _pct(value: float | None) -> str:
    return "未知" if value is None else f"{value:.0%}"


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def save_result(result: AnalysisResult, root: str | Path | None = None) -> Path:
    from .report import _report_title, render_audit
    from .brief import render_brief
    from .report_checks import publication_issues, publication_warnings
    from .pdf_report import build_report_pdf, ReportPageLimitError

    folder = _output_folder(result, root)
    memo = render_brief(result)
    result.report_checks = publication_issues(result, memo)
    result.report_checks = list(dict.fromkeys(result.report_checks + [item["message"] for item in result.quality_issues]))
    result.report_warnings = publication_warnings(result, memo)
    if result.report_checks:  # 不出正式报告，但保留完整研究稿和问题清单，已花的研究费用不白费
        result.report_status = "报告一致性检查未通过"
        memo = render_brief(result)
    try:
        pdf, result.report_pages = build_report_pdf(memo, result.report_max_pages)
    except ReportPageLimitError as exc:
        result.report_checks.append(str(exc))
        result.report_status = "报告页数检查未通过"
        # Never publish an overlength report or silently delete investment risks.
        memo = (f"{_report_title(result)}\n\n> 研究稿状态：页数检查未通过，不能作为正式报告。\n\n"
                f"{exc}\n\n研究材料与问题记录已保存，需要压缩正文并重新核查后才能出稿。\n")
        pdf, result.report_pages = build_report_pdf(memo, result.report_max_pages)
    if result.memory_config.get("enabled"):
        try:
            memory = ResearchMemory(result.memory_config["path"], result.memory_config["ttl_days"])
            result.memory_stats = memory.remember_result(result)
        except Exception as exc:
            result.memory_stats = {"error": str(exc)}
            result.report_warnings.append("研究记忆写入失败，本次材料仍保存在结果文件中")
    filename = "report.md" if result.report_status in ("演示稿", "待人工复核的研究报告") else "research-draft.md"
    (folder / filename).write_text(memo, encoding="utf-8")
    (folder / filename.replace(".md", ".pdf")).write_bytes(pdf)
    (folder / "audit.md").write_text(render_audit(result), encoding="utf-8")
    (folder / "result.json").write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    index_export(folder / "result.json")
    return folder


def save_partial_result(result: AnalysisResult, root: str | Path | None = None) -> Path:
    from .report import render_audit

    folder = _output_folder(result, root)
    (folder / "partial-result.json").write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    (folder / "audit.md").write_text(render_audit(result), encoding="utf-8")
    index_export(folder / "partial-result.json")
    return folder


def _output_folder(result: AnalysisResult, root: str | Path | None) -> Path:
    safe = re.sub(r'[\\/:*?"<>|\s]+', "_", result.sector_key).strip("_") or "sector"
    tag = re.sub(r"[^a-zA-Z0-9]", "", result.run_id)[:8] or uuid4().hex[:8]
    stem = f"{safe}_{datetime.now():%Y%m%d_%H%M%S}_{tag}"
    base = Path(root) if root is not None else ROOT / "outputs"
    for version in range(1000):
        folder = base / (stem + (f"_{version}" if version else ""))
        try:
            folder.mkdir(parents=True, exist_ok=False)
            return folder
        except FileExistsError:
            continue
    raise PipelineError("无法创建独立的报告输出目录")
