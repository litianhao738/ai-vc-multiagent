"""Five persistent agents in four modules with versioned material handoffs."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import threading
import time

from . import schemas, grounding, inquiry
from .config import ROOT
from .handoffs import MaterialBoard, HandoffError
from .main_roles import CHALLENGER, MAIN_ROLES
from .learning import editing_prompt, editor_template, editor_version
from .pipeline import SectorPipeline as ExpandedPipeline, CompanyInput, PipelineError, _namespace_sources
from .roles import CHAIR, SECTOR_RESEARCHER, COMPANY_RESEARCHER, COMPANY_ANALYST, COMPARATOR, FACT_CHECKER, system_prompt
from .runtime import AgentScopeAgent
from .scoring import rank_companies, ranking_sensitivity, score_ranges
from .tools import Toolbox, normalize_fact_check
from .quality import diagnose, audit_plan, coverage_gaps, coverage_messages
from .governor import RunStopped
from .scheduler import DAGScheduler, TaskNode
from .report_checks import quality_finding, memo_materials
from .reasoning import review_targets, review_issues


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


class MainAgentPipeline(ExpandedPipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.board = MaterialBoard()
        self.agents: dict[str, AgentScopeAgent] = {}
        self._core_locks = {key: threading.RLock() for key in MAIN_ROLES}
        self._phase_sequence = 0
        self._repair_count = 0
        self._quality_active = False
        self._audit_plan: dict = {}

    def run(self, request):
        self.board = MaterialBoard()
        self.agents = {}
        self._repair_count = self._phase_sequence = 0
        self._quality_active = False
        self._usage, self._costs, self._format_failures, self._agent_seconds, self._source_reads = {}, {}, {}, {}, {}
        self.request = request
        return super().run(request)

    def _validate_request(self, request):
        super()._validate_request(request)
        if (isinstance(self.settings.publication_repair_rounds, bool)
                or not isinstance(self.settings.publication_repair_rounds, int)
                or not 0 <= self.settings.publication_repair_rounds <= 3):
            raise ValueError("发布定向修复轮数须为 0 到 3 的整数")
        if (isinstance(self.settings.research_refinement_rounds, bool)
                or not isinstance(self.settings.research_refinement_rounds, int)
                or not 0 <= self.settings.research_refinement_rounds <= 2):
            raise ValueError("问题驱动补查轮数须为 0 到 2 的整数")
        if self.settings.pairwise_checks or self.settings.debate_rounds not in (None, 2):
            raise ValueError("core 模式固定两轮交锋。自定义轮数或两两比较请选择 expanded")

    def _plan_research(self, request, result):
        result.architecture = "agentscope-five-agents-two-round-debate"
        result.main_agents = list(MAIN_ROLES)
        result.models = {key: "mock" if self.settings.mock else self.settings.role_models[key]
                         for key in (*MAIN_ROLES, "search")}
        self._method_memory = []
        self._editor_template = editor_template()
        self._editor_rule = {"id": None, "rule": ""}
        if self.memory:
            try:
                self._method_memory = self.memory.recall_methods(self.sector.key, query=request.focus)
                version = editor_version(result.models["lead"], self._editor_template)
                self._editor_rule = self.memory.learning.active_rule(self.sector.key, version)
            except Exception as exc:
                self._stage("方法记忆读取未完成，沿原研究议程继续：" + str(exc))
        # Which prompt text produced which report: required before any prompt change can be evaluated.
        result.prompt_versions = {f"role:{key}": _digest(role.brief) for key, role in MAIN_ROLES.items()}
        super()._plan_research(request, result)
        result.research_state = inquiry.initial_state(result.research_plan, bool(request.companies),
                                                      request.selection_rationale)
        result.research_state["method_memory"] = self._method_memory
        if self._editor_rule["id"] is not None:
            result.research_state["editing_rule_id"] = self._editor_rule["id"]
        self.board.publish("plan", result.research_plan)
        self.board.publish("inquiry", result.research_state)

    def _prompt(self, name):
        if name == "lead_editing":
            text = editing_prompt(self._editor_template, self._editor_rule["rule"])
        else:
            text = (ROOT / "prompts" / f"{name}.md").read_text(encoding="utf-8")
        self.last_result.prompt_versions[f"prompts/{name}.md"] = _digest(text)
        return text

    def _research_toolbox(self):
        return Toolbox(self.fact_docs, self.search, knowledge=self.knowledge, registry=self.registry,
                       memory=self.memory, memory_sector=self.sector.key, network_slots=self._network_slots,
                       governor=self.governor, rejection_review=self.rejection_reviewer.filter)

    def _research(self, request, result):
        companies = list(request.companies)
        if not companies:
            self._stage("资料研究主Agent：确定赛道与候选公司")
            sector = self._run(SECTOR_RESEARCHER, self._sector_task(request, []), self._research_toolbox(),
                               "research_candidates", "资料研究主Agent", schemas.sector_brief_schema(request.max_companies))
            companies = [CompanyInput(name) for name in sector["candidates"]]
        self.companies = {company.name: company for company in companies}
        result.companies = list(self.companies)
        self._thesis_leads = []
        if self.memory:
            try:
                self._thesis_leads = self.memory.learning.thesis_leads(self.sector.key, result.companies)
                if self._thesis_leads:
                    result.research_state["thesis_card_ids"] = [lead["card_id"] for lead in self._thesis_leads]
                    self.board.publish("inquiry", result.research_state)
            except Exception as exc:
                self._stage("随访线索暂不可用，继续使用本轮材料：" + str(exc))
        self._research_all(result)

    def _research_all(self, result, correction=""):
        self._stage("资料研究主Agent：在同一上下文研究赛道和全部公司，独立读取可并行")
        task = ("提交一份完整研究包：sector 为赛道事实，companies 按以下公司名分别提交底稿。"
                "围绕议程中可能改变判断的问题，优先读取能区分不同解释的一手资料，例如定价页、产品更新或用户反馈。"
                "如实记录支持或削弱当前假设的事实。只找到单侧材料也可提交，无需为每题寻找持相反观点的引文。"
                "产品团队、经营融资、竞争风险是覆盖底线，不是写作目标；公司名和资料前缀不可混用。"
                "并行调用相互独立的搜索或读取工具，不创建子Agent。"
                "每家公司保留最影响投资判断的事实，把仍待核实的关键事实列入信息缺口。\n"
                + json.dumps({"companies": [{"name": c.name, "website": c.website, "urls": c.urls, "notes": c.notes}
                                           for c in self.companies.values()],
                              "research_plan": result.research_plan, "focus": self.request.focus}, ensure_ascii=False))
        if self._thesis_leads:
            task += ("\n历史论点与人工随访，仅用于设计本轮的核查问题。核实当前公司并重读原文后才可引用。"
                     + json.dumps(self._thesis_leads, ensure_ascii=False))
        if self.fact_docs:
            task += "\n上传资料使用完整公司前缀读取：\n" + self.fact_docs.overview()
        if correction:
            task += ("\n当前研究包需更正：" + correction
                     + "\n只补查问题涉及的公司或赛道事实，沿用未受影响的已核验材料，不重复检索其他公司。"
                       "完整研究包用于统一版本交接；没有原文支持的主张与依赖说法必须删除。")
        output = self._run(COMPANY_RESEARCHER, task, self._research_toolbox(),
                           "research_bundle", "资料研究主Agent", schemas.research_bundle_schema(list(self.companies)))
        previous_sector, previous_sheets = result.sector_brief, result.fact_sheets
        result.sector_brief = _namespace_sources(output["sector"], "S", previous_sector)
        result.fact_sheets = {name: _namespace_sources(output["companies"][name], f"C{i}", previous_sheets.get(name))
                              for i, name in enumerate(self.companies, 1)}
        self._publish_research(result)

    def _publish_research(self, result):
        ref = self.board.publish("research", {"sector": result.sector_brief, "companies": result.fact_sheets})
        if result.research_state:
            inquiry.invalidate_answers(result.research_state, result.research_methods, ref.sha256)
            self.board.publish("inquiry", result.research_state)

    def _research_audit(self, result, lead_in):
        """One definition of a research audit: materials, the code-chosen must-check list, task and schema."""
        materials = self._research_materials(result)
        plan = self._audit_plan = audit_plan(materials, self.settings.audit_claims_per_company)
        task = (lead_in + "覆盖要求由代码给出，不是抽样建议：required 中每份材料的 targets 逐条核查，"
                "并在对应主张的 evidence_ids 填写该证据编号；每份材料的核查条数不少于 min_claims。"
                "清单之外再按对投资结论的影响补充，优先数字、融资、估值与上市状态。"
                "不同公司的原文不可互借。source_agent 写下列材料键。\n"
                + json.dumps({"required": plan, "materials": materials}, ensure_ascii=False))
        schema = schemas.fact_check_schema(list(materials), self._evidence_ids(result),
                                           sum(need["min_claims"] for need in plan.values()))
        return materials, plan, task, schema

    def _record_coverage(self, result, plan):
        """Coverage is a publication gate. Demo data has no real sources, so it is recorded but not enforced."""
        enforced = not self.settings.mock
        claims = result.fact_check.get("claims", [])
        gaps = coverage_gaps(plan, claims) if enforced and claims else []
        result.audit_coverage = {
            "enforced": enforced, "claims_per_company": self.settings.audit_claims_per_company,
            "required": {name: {"min_claims": need["min_claims"],
                                "evidence_ids": [t["evidence_id"] for t in need["targets"]]} for name, need in plan.items()},
            "checked": {name: sum(c.get("source_agent") == name for c in claims) for name in plan}, "gaps": gaps}
        if gaps:
            result.errors["audit_coverage"] = "研究事实核查覆盖不足：" + "；".join(coverage_messages(gaps))
        else:
            result.errors.pop("audit_coverage", None)

    def _analyze(self, result, only_companies=None):
        eligible = {name: sheet for name, sheet in result.fact_sheets.items() if sheet["listing_status"] != "已上市"}
        if not eligible:
            raise PipelineError("没有可供分析的未上市候选公司")
        self._stage("投资分析与独立审计并行：同读当前版本研究包；评分完成后立即比较，不等待审计结束")
        research_refs = self.board.refs("research")
        materials, plan, check_task, check_schema = self._research_audit(result, "核查当前版本研究包。")
        scoring = eligible if only_companies is None else {
            name: sheet for name, sheet in eligible.items() if name in only_companies or name not in result.analyses}
        if not scoring:
            scoring = eligible
        score_task = ("按统一赛道标准评估下列全部公司，逐维度打分并绑定本轮证据ID。"
                      "只提交分析与评分，排序由代码完成。不要预设投资建议。"
                      "每家公司另写观点字段。standout 选最影响判断的一项差异，说明它如何影响竞争。无法确认差异时直说。"
                      "competitor_diffs 与最接近的竞对比。"
                      "cost_structure 区分研发投入、交付成本、用户总成本和获客成本；long_term_edge 六个维度各一条。"
                      "这些是你的判断，不是数据罗列；证据不足就写未知或推断，可引用赛道或任一公司的当前证据编号。\n"
                      + json.dumps({"sector": result.sector_brief, "companies": scoring,
                                    "financial": result.financial_metrics, "market": result.market_metrics,
                                    "deal": result.deal_metrics}, ensure_ascii=False))
        score_schema = schemas.analysis_bundle_schema(self.sector, scoring)
        audit_box = self._factcheck_toolbox(materials)
        scored = {}

        def score():
            scored.update(self._run(COMPANY_ANALYST, score_task, None, "analysis_scores", "投资分析主Agent", score_schema))

        def compare():
            # The comparison reads the research package and the code ranking, never the
            # audit still in flight; it therefore overlaps the audit instead of waiting.
            self.board.validate(research_refs)
            result.analyses = ({name: analysis for name, analysis in result.analyses.items()
                               if name in eligible and name not in scoring} if only_companies is not None else {})
            result.analyses.update(scored["analyses"])
            result.ranking = rank_companies(result.analyses, self.sector)
            result.sensitivity = ranking_sensitivity(result.analyses, self.sector)
            result.score_ranges = score_ranges(result.analyses, result.fact_sheets, self.sector)
            self.board.publish("analysis", {"analyses": result.analyses, "ranking": result.ranking,
                                            "sensitivity": result.sensitivity, "score_ranges": result.score_ranges})
            self._run_compare(result)

        def audit():
            return self._run(FACT_CHECKER, check_task, audit_box, "audit_research", "独立审计主Agent", check_schema)

        # Scoring and comparison are critical: if either fails the run is void, so the
        # audit running beside them is cancelled instead of being paid for and discarded.
        scheduler = DAGScheduler(self.settings.max_workers, self.governor)
        try:
            outcomes = scheduler.run([TaskNode("analysis_scores", score, critical=True),
                                      TaskNode("analysis_compare", compare, ("analysis_scores",), critical=True),
                                      TaskNode("audit_research", audit)])
        finally:
            result.task_trace.extend(scheduler.trace)
        self.board.validate(research_refs)
        if outcomes["analysis_scores"].error:
            raise PipelineError("投资分析主Agent未完成评分：" + outcomes["analysis_scores"].error)
        if outcomes["analysis_compare"].error:
            raise outcomes["analysis_compare"].exception or PipelineError(outcomes["analysis_compare"].error)
        if outcomes["audit_research"].error:
            result.errors["audit_research"] = outcomes["audit_research"].error
        result.fact_check = outcomes["audit_research"].value or {"claims": [], "summary": "研究核查未完成"}
        if result.fact_check.get("claims"):
            result.errors.pop("audit_research", None)
        research_version = research_refs[0].version
        claim_prefix = "R" if research_version == 1 else f"R-v{research_version}"
        for index, claim in enumerate(result.fact_check["claims"], 1):
            claim["claim_id"] = f"{claim_prefix}-{index}"
        self._record_coverage(result, plan)
        result.check_history.append({"stage": "研究主Agent材料核查", "result": deepcopy(result.fact_check)})
        self.board.publish("research_audit", result.fact_check)

    def _compare(self, result):
        # Normally done inside _analyze, overlapping the research audit.
        if not result.comparison:
            self._run_compare(result)

    def _run_compare(self, result):
        self._stage("两轮投资辩论：正方提出、反方质疑、正方答辩、反方复核、正方收束")
        pins = self.board.refs("research", "analysis", "inquiry")
        opening_task = (self._comparator_task(result) + "\n提交公司比较 comparison 和正方初始主张 bull。"
                        "先在 thesis 明确回答当前核心问题。在 arguments 区分现有事实直接支持的部分与下一步依赖的假设。"
                        "围绕与最接近竞对的一项差异给出相对判断。缺少同口径资料时指出缺口，最多三条理由，说明推翻条件。")
        companies = list(result.analyses)
        opening = self._run(COMPARATOR, opening_task,
                            None, "analysis_compare", "投资分析主Agent", schemas.comparison_position_schema(companies))
        facts = self._evidence_context(result)
        debate = [{"side": "bull", "round": 1, "phase": "bull_r1", "name": "正方初始主张", **opening["bull"]}]

        comparison_summary = {key: opening["comparison"][key] for key in ("summary", "ranking_view")}

        def turn_task(instruction, comparison):
            self.board.validate(pins)
            return ("当前可引用的事实材料\n" + facts + "\n以下为开场比较及截至当前的发言。\n"
                    + json.dumps({"opening_comparison": comparison, "debate": debate},
                                 ensure_ascii=False, separators=(",", ":"))
                    + "\n" + instruction + " 始终回答同一个核心问题，允许更换解释这个问题的机制。最多三条理由，区分事实与假设。")

        turns = (
            (CHALLENGER, "bear", 1, "bear_r1", "反方首次质疑", "检验初始结论中最影响判断的一步。有实质漏洞时解释原因和影响，能成立时说明依据。"),
            (COMPARATOR, "bull", 2, "bull_r2", "正方回应质疑", "先写当前结论，用事实或逻辑回应异议。若另一个机制更能解释现象，说明为何改判、有什么依据、哪一步还只是设想。"),
            (CHALLENGER, "bear", 2, "bear_r2", "反方再次质疑", "检验正方的最新结论和答辩，说明原异议是否已经解决。只追问仍会改变判断的具体缺口。"),
        )
        for role, side, rnd, phase, label, instruction in turns:
            # Each side receives the full comparison on first use. Both keep the
            # verbatim exchanges and the current evidence throughout the debate.
            task = turn_task(instruction, opening["comparison"] if phase == "bear_r1" else comparison_summary)
            if side == "bear":
                previous = debate[-1]
                targets = {"thesis": previous["thesis"]}
                for field in ("arguments", "rebuttals"):
                    targets.update({f"{field}.{i}": statement for i, statement in enumerate(previous[field], 1)
                                    if statement.strip()})
                task += ("\ntarget 指向上一条正方发言的原话。列表项从 1 开始编号，可选编号为 "
                         + "、".join(targets) + "。")
                reply = self._run(role, task, None, phase, label, schemas.challenger_response_schema(targets))
                reply["addressed_claim"] = targets[reply["target"]]
            else:
                reply = self._run(role, task, None, phase, label, schemas.BULL_RESPONSE)
            debate.append({"side": side, "round": rnd, "phase": phase, "name": label, **reply})
        closing_task = turn_task("回应第二次质疑，在 bull.thesis 给出当前证据能支持的一句话结论。"
                                 "在 rebuttals 对照开场的一项具体判断，指出经质疑后保留、收窄或撤回了哪一步，"
                                 "用对应证据或明确的逻辑说明理由。判断成立时可以维持原判。"
                                 "明确结论适用的对象和范围。仍无法判断的部分，交代已有事实和缺失的关键前提。"
                                 "在 would_change_mind 指出最值得验证的观察。将结论对相对位置的影响同步到完整 comparison，"
                                 "包括 summary、ranking_view 及受影响公司的比较。", opening["comparison"])
        closing = self._run(COMPARATOR, closing_task, None, "bull_close", "正方最终主张",
                            schemas.comparison_position_schema(companies, responding=True))
        debate.append({"side": "bull", "round": 2, "phase": "bull_close", "name": "正方最终主张", **closing["bull"]})
        self.board.validate(pins)
        self.board.publish("analysis", {"analyses": result.analyses, "ranking": result.ranking,
                                        "comparison": closing["comparison"], "perspectives": debate},
                           expected_version=next(ref.version for ref in pins if ref.name == "analysis"))
        result.comparison, result.debate = closing["comparison"], debate

    def _debate(self, result):
        pass  # Comparison includes the complete five-turn exchange.

    @staticmethod
    def _research_materials(result):
        # These are material identifiers, retained for saved-result compatibility.
        return {"sector_researcher": result.sector_brief,
                **{f"company_researcher:{name}": sheet for name, sheet in result.fact_sheets.items()}}

    def _check_research(self, result):
        self._check_research_facts(result)
        self._deepen_research(result)

    def _check_research_facts(self, result):
        bad = [claim for claim in result.fact_check.get("claims", []) if claim["verdict"] in ("不实", "无法核实")]
        if bad:
            self._repair(result)
        elif not result.fact_check.get("claims"):
            result.errors["audit_research"] = "独立审计主Agent没有完成关键研究事实核查"

    def _inquiry_materials(self, result):
        return {"evidence": self._evidence_context(result),
                "analysis": {name: {key: data.get(key) for key in
                    ("summary", "product_business", "operating_assessment", "concerns", "confidence", *schemas.INSIGHT_KEYS)}
                    for name, data in result.analyses.items()},
                "comparison": result.comparison, "perspectives": result.debate,
                "fact_check": [{key: claim.get(key) for key in ("claim", "verdict", "evidence", "evidence_ids")}
                               for claim in result.fact_check.get("claims", [])]}

    def _deepen_research(self, result):
        limit = self.settings.research_refinement_rounds
        state = result.research_state
        if not state:
            return
        if limit == 0:
            state["stop_reason"] = "本次关闭追加补查，沿原研究与成稿流程完成"
            self.board.publish("inquiry", state)
            return
        seen = set()
        for round_number in range(limit + 1):
            self.governor.check()
            # Only a missing audit blocks the question loop. Rejected or unverifiable
            # facts are excluded from answers; the publication gate still handles them.
            if not result.fact_check.get("claims"):
                state["stop_reason"] = "研究核查未完成，交原纠错与发布流程处理"
                break
            remaining = self.governor.remaining()
            if remaining is not None and remaining < 300:
                state["stop_reason"] = "剩余总时限较短，保留缺口并进入原成稿流程"
                break
            self._stage("总控：根据当前证据判断关键问题是否值得继续补查")
            usable = inquiry.usable_evidence(result)
            excluded = sorted(inquiry.flagged_evidence(result))
            schema = schemas.research_checkpoint_schema([q["question_id"] for q in state["questions"]],
                list(usable), list(self.companies), [m["method_id"] for m in self._method_memory])
            record = {"kind": "checkpoint", "round": round_number, "status": "running",
                      "before": deepcopy(state), "materials": self.board.manifest(), "excluded_evidence_ids": excluded}
            result.research_iterations.append(record)
            try:
                task = (self._prompt("research_checkpoint")
                        + "\n本轮材料：" + json.dumps({"remaining_rounds": limit - round_number,
                            "excluded_evidence": {"ids": excluded, "reason": "独立审计判为不实或无法核实，不能作为回答依据；可作为补查对象"},
                            **self._inquiry_materials(result)}, ensure_ascii=False))
                data = self._run(CHAIR, task, None, f"lead_inquiry_{round_number}", "总控：研究补查决策", schema)
                errors = schemas.validate(schema, data)
                if not errors:
                    errors = inquiry.checkpoint_errors(data, state, set(usable), {"sector", *self.companies})
                if errors:
                    raise ValueError("；".join(errors))
                tasks = inquiry.apply_checkpoint(state, data, round_number + 1)
                record.update(status="accepted", decision=deepcopy(data), tasks=deepcopy(tasks), after=deepcopy(state))
                self.board.publish("inquiry", state)
            except (RunStopped, HandoffError):
                record["status"] = "stopped"
                raise
            except Exception as exc:
                record.update(status="unavailable", error=str(exc))
                state["stop_reason"] = "补查决策未完成，保留原有研究并进入原成稿流程"
                break
            if data["next_action"] == "finish":
                state["stop_reason"] = data["reason"] or "总控决定基于当前证据成稿，保留信息缺口"
                break
            if round_number >= limit:
                record["not_executed"] = "补查轮数上限"
                state["stop_reason"] = "已达追加补查上限，未执行的研究任务作为信息缺口保留"
                break
            signatures = {(task["scope"], " ".join(task["question"].casefold().split()),
                           " ".join(task["method"].casefold().split())) for task in tasks}
            if seen.intersection(signatures):
                record["not_executed"] = "重复任务"
                state["stop_reason"] = "总控重复提出相同补查，停止追加并保留缺口"
                break
            seen.update(signatures)
            if not self._execute_research_followup(result, tasks, round_number + 1):
                break
        self._evaluate_inquiry(result)
        self.board.publish("inquiry", state)

    def _execute_research_followup(self, result, tasks, round_number):
        scopes = {task["scope"] for task in tasks}
        names = [name for name in self.companies if name in scopes]
        include_sector = "sector" in scopes
        phase = f"research_followup_{round_number}"
        record = {"kind": "followup", "round": round_number, "phase": phase, "tasks": deepcopy(tasks),
                  "status": "running", "before": self.board.manifest()}
        result.research_iterations.append(record)
        committed = False
        try:
            previous = {"companies": {name: result.fact_sheets[name] for name in names}}
            if include_sector:
                previous["sector"] = result.sector_brief
            task = ("围绕总控给出的具体问题追加研究。只返回涉及范围的完整底稿，保留其中未受影响的事实、来源与引用。"
                    "其他公司的材料由代码原样保留，不要返回或重查。原文读取可并行；无法获取的信息写入缺口。"
                    "区分产品标价、用户总成本、公司交付成本、研发投入和获客成本；没有测试工具或人工测试记录，"
                    "不得把检索资料称为产品实测。不要为了补齐答案编造数字。\n"
                    + json.dumps({"tasks": tasks, "current_materials": previous}, ensure_ascii=False))
            started = time.monotonic()
            output = self._run(COMPANY_RESEARCHER, task, self._research_toolbox(), phase, "研究主Agent：问题定向补查",
                               schemas.research_followup_schema(names, include_sector))
            record["research_seconds"] = round(time.monotonic() - started, 2)
            if set(output["companies"]) != set(names) or ("sector" in output) != include_sector:
                raise ValueError("补查输出范围与授权任务不一致")
            sheets = {name: _namespace_sources(output["companies"][name], f"C{list(self.companies).index(name) + 1}",
                                              result.fact_sheets[name]) for name in names}
            sector = _namespace_sources(output["sector"], "S", result.sector_brief) if include_sector else result.sector_brief
            if sector == result.sector_brief and all(sheets[name] == result.fact_sheets[name] for name in names):
                record.update(status="no_change", after=self.board.manifest())
                result.research_state["stop_reason"] = "补查未改变材料，保留未解决问题并进入原成稿流程"
                return False
            result.sector_brief = sector
            result.fact_sheets.update(sheets)
            self._publish_research(result)
            committed = True
            # Research changes invalidate all comparisons and decisions. Company-only
            # followups retain unrelated scores; a changed sector re-evaluates everyone.
            affected = None if include_sector else set(names)
            result.analyses = {name: analysis for name, analysis in result.analyses.items()
                               if affected is not None and name not in affected}
            result.ranking, result.comparison, result.debate, result.decision = [], {}, [], {}
            result.returns, result.return_summary, result.returns_analysis = [], {}, {}
            result.fact_check = {"claims": [], "summary": "补查后的研究包待重新核查"}
            self._analyze(result, affected)
            self._check_research_facts(result)
            record.update(status="completed", after=self.board.manifest())
            research_hash = self.board.refs("research")[0].sha256
            for i, task in enumerate(tasks, 1):
                result.research_methods.append({"method_id": f"M{round_number}-{i}", "task_id": task["task_id"],
                    "question_id": task["question_id"], "question": task["question"], "scope": task["scope"],
                    "procedure": task["method"], "expected_impact": task["expected_impact"], "phase": phase,
                    "origin_method_id": task.get("origin_method_id", ""),
                    "status": "candidate", "research_sha256": research_hash, "validation": {}})
            return True
        except (RunStopped, HandoffError):
            record["status"] = "stopped"
            raise
        except Exception as exc:
            record.update(status="failed", error=str(exc), after=self.board.manifest())
            if committed:
                raise PipelineError("补查改变研究材料后，下游重新分析或核查未完成：" + str(exc)) from exc
            result.research_state["stop_reason"] = "追加补查未完成，保留原研究材料与缺口"
            return False

    def _evaluate_inquiry(self, result):
        methods = [method for method in result.research_methods if method["status"] == "candidate"]
        if not methods:
            return
        self._inquiry_audit_methods = [method["method_id"] for method in methods]
        self._inquiry_audit_questions = list(dict.fromkeys(method["question_id"] for method in methods))
        record = {"kind": "method_evaluation", "status": "running", "materials": self.board.manifest()}
        result.research_iterations.append(record)
        try:
            read_keys = ("kind", "url", "urls", "name", "page", "snapshot_id", "content_sha256")
            task = (self._prompt("inquiry_audit")
                    + "\n本轮记录：" + json.dumps({"methods": methods,
                        "executed": [row for row in result.research_iterations if row["kind"] == "followup"],
                        "read_records": [{key: read[key] for key in read_keys if key in read}
                                         for read in self._source_reads.get("research", [])],
                        **self._inquiry_materials(result)}, ensure_ascii=False))
            schema = schemas.inquiry_audit_schema(self._inquiry_audit_questions, self._inquiry_audit_methods,
                                                  list(inquiry.usable_evidence(result)))
            checked = self._run(FACT_CHECKER, task, self._factcheck_toolbox(self._research_materials(result)),
                                "audit_inquiry", "审计：研究问题与方法复核", schema)
            errors = schemas.validate(schema, checked)
            if not errors:
                errors = inquiry.audit_errors(checked, self._inquiry_audit_questions, self._inquiry_audit_methods,
                                               set(inquiry.usable_evidence(result)))
            if errors:
                raise ValueError("；".join(errors))
            by_id = {q["question_id"]: q for q in result.research_state["questions"]}
            for check in checked["question_checks"]:
                question = by_id[check["question_id"]]
                verdict = check["verdict"] if question["answer"].strip() else "unresolved"
                question.update(audit_verdict=verdict, audit_reason=check["reason"])
                if verdict == "supported":
                    question["status"] = "answered"
                elif question["status"] == "answered":
                    question["status"] = "open"
            checks = {row["method_id"]: row for row in checked["method_checks"]}
            for method in methods:
                method.update(status="reviewed", validation=deepcopy(checks[method["method_id"]]))
            record.update(status="reviewed", result=deepcopy(checked))
        except (RunStopped, HandoffError):
            record["status"] = "stopped"
            raise
        except Exception as exc:
            record.update(status="unavailable", error=str(exc))

    def _reasoning_context(self, result):
        text = (ROOT / "prompts" / "moat_review.md").read_text(encoding="utf-8")
        result.prompt_versions["prompts/moat_review.md"] = _digest(text)
        return (super()._reasoning_context(result) + "\n# 投委会成稿审计要求\n" + text
                + "\n用户持有期（年，null表示未提供，不得代填）：" + json.dumps(self.settings.holding_years))

    def _chair_task(self, result):
        return (super()._chair_task(result)
                + "\n# 各公司当前结构化分析观点\n"
                  "以下内容用于选择和检验研究焦点，属于待核验意见；事实与持续性判断仍须回到当前原文证据，不能直接照抄。\n"
                + json.dumps({name: {key: data[key] for key in schemas.INSIGHT_KEYS if key in data}
                              for name, data in result.analyses.items()}, ensure_ascii=False)
                + "\n首页给出值得持续投入的一项能力及客户价值，再给技术归属、已发生的证据、竞品复刻难度和技术失效条件。交易动作及其条件放在估值章节。"
                  "core_answer 直接回答研究议程的核心问题；question_answers 只选最影响判断的问题（编号来自当前研究状态），"
                  "每条写结论、已有事实依据和会改变它的观察。有事实反证时列出编号。"
                  "反方 Agent 的逻辑质疑若影响决策，写明它限制了哪一步推论，并把验证条件带入建议或尽调。"
                  "selection_view 对照研究状态中的 selection_rationale 说明证据支持到什么程度。"
                  "gap_impacts 写最影响判断的信息缺口、它影响哪个判断、拿到后可能怎样变化。"
                  "各公司分析中的 standout、competitor_diffs、cost_structure、long_term_edge 是分析主Agent的观点，"
                  "采用前回到原文证据。区分用户指定研究与系统筛选，不事后编造筛选理由。"
                  "问题回答、方法复盘和其他 Agent 意见中引用的事实都须回到本轮原文查证。"
                  "完整补查和运行过程另存内部审计台账；投资正文围绕有依据的核心发现自然展开。")

    def _repair(self, result):
        previous = [claim for claim in result.fact_check.get("claims", []) if claim["verdict"] in ("不实", "无法核实")]
        if self._repair_count >= 1:
            for claim in previous:
                result.unresolved_checks.append({"stage": "主Agent纠错", "claim_id": claim.get("claim_id", ""),
                    "claim": claim["claim"], "source_agent": claim["source_agent"], "verdict": claim["verdict"],
                    "problem": "修正后仍存在或未完成复查", "reason": claim.get("evidence", "")})
            return
        self._repair_count += 1
        before = deepcopy(self._check_materials(result))
        self._research_all(result, json.dumps(previous, ensure_ascii=False))
        # Invalidate all downstream opinions when the research version changes.
        result.analyses, result.ranking, result.comparison, result.debate, result.decision = {}, [], {}, [], {}
        result.returns, result.return_summary, result.returns_analysis = [], {}, {}
        self._analyze(result)
        current_audit = deepcopy(result.fact_check)
        if not current_audit.get("claims"):
            result.errors.setdefault("audit_research", "独立审计主Agent未完成修正后研究材料的关键事实核查")
        materials = self._check_materials(result)
        task = ("逐项复查以下问题事实及依赖推论在当前版本是否已修正或删除；repair_checks 覆盖全部 claim_id。\n"
                "原问题的处理状态不能替代当前研究包核查，新增问题仍须保留。\n"
                + json.dumps({"previous": previous, "current": materials, "current_audit": current_audit}, ensure_ascii=False))
        try:
            checked = self._run(FACT_CHECKER, task, self._factcheck_toolbox(materials), "audit_repair", "独立审计主Agent",
                                schemas.repair_check_schema([c["claim_id"] for c in previous], list(materials), self._evidence_ids(result)))
        except RunStopped:
            raise
        except Exception as exc:
            result.errors["audit_repair"] = str(exc)
            checked = {"claims": [], "summary": "修正复查未完成", "repair_checks": []}
        # Both audits examine the same current material, but the repair check
        # only closes old issues. It cannot erase new findings or missing coverage.
        checked = {**checked,
                   "claims": current_audit.get("claims", []) + checked.get("claims", []),
                   "summary": "\n".join(filter(None, (current_audit.get("summary"), checked.get("summary"))))}
        result.fact_check = checked
        result.check_history.append({"stage": "主Agent修正后复查", "result": deepcopy(checked)})
        result.repairs.append({"source_agent": "research", "claims": previous, "before": before,
                               "after": deepcopy(materials)})
        self._record_unresolved(result, "主Agent修正后复查", previous, checked)
        self.board.publish("research_audit", checked)

    @staticmethod
    def _group(role):
        if role.key in ("chair", "lead"):
            return "lead"
        if role.key in ("sector_researcher", "company_researcher", "research"):
            return "research"
        if role.key == "bear":
            return "bear"
        if role.key in ("company_analyst", "comparator", "bull", "analysis"):
            return "analysis"
        if role.key in ("fact_checker", "audit"):
            return "audit"
        raise PipelineError(f"未定义的主Agent分工：{role.key}")

    def _input_refs(self, group, phase):
        if phase == "lead_planner":
            return []
        names = ["plan", "inquiry"]
        if group != "research" or phase == "research_bundle" or phase.startswith("research_followup_"):
            names.append("research")
        if group == "lead" or phase == "audit_repair":
            names += ["analysis", "research_audit"]
        if phase in ("analysis_compare", "bear_r1", "bull_r2", "bear_r2", "bull_close"):
            names.append("analysis")  # 比较与研究审计重叠执行，不读取尚未完成的审计
        if group == "audit" and phase.startswith("fact_checker_decision"):
            names += ["analysis", "research_audit", "memo"]
        if phase == "audit_inquiry":
            names += ["analysis", "research_audit"]
        if phase == "chair_editor":
            names += ["decision", "memo"]
        if self._quality_active:
            names.append("quality")
            if phase == "quality_analysis":
                names.append("analysis")
            if phase == "quality_final_audit":
                names += ["analysis", "research_audit", "memo"]
        return self.board.refs(*dict.fromkeys(names))

    def _extra_check(self, group, phase, schema, toolbox, data):
        if group == "lead":
            errors = schemas.validate_submission("chair", schema, data)
        elif group == "audit":
            errors = schemas.validate_submission("fact_checker", schema, data)
        else:
            errors = schemas.validate(schema, data)
        if errors or self.settings.mock:
            return errors
        if phase.startswith("lead_inquiry_"):
            return inquiry.checkpoint_errors(data, self.last_result.research_state,
                set(inquiry.usable_evidence(self.last_result)), {"sector", *self.companies})
        if phase == "audit_inquiry":
            return inquiry.audit_errors(data, self._inquiry_audit_questions, self._inquiry_audit_methods,
                                        set(inquiry.usable_evidence(self.last_result)))
        if group == "research":
            sheets = [data] if "evidence" in data else ([data["sector"]] if "sector" in data else []) + list(data["companies"].values())
            return [problem for sheet in sheets for problem in toolbox.citation_errors("company_researcher", sheet)]
        if group == "audit":
            problems = toolbox.citation_errors("fact_checker", data)
            if phase in ("audit_research", "quality_research_audit"):
                problems += coverage_messages(coverage_gaps(self._audit_plan, data.get("claims", [])))
            return problems
        if phase in ("analysis_scores", "quality_analysis"):
            known = set(self._evidence_ids(self.last_result))
            return (schemas.insight_errors(data["analyses"], known)
                    + grounding.errors_for_submission(self._evidence_report(group, data),
                                                      defer_amount_review=self.settings.rejection_review_enabled))
        if phase != "lead_planner":
            return grounding.errors_for_submission(self._evidence_report(group, data),
                                                   defer_amount_review=self.settings.rejection_review_enabled)
        return []

    def _verify_research(self, data):
        if "companies" not in data:
            return self._research_toolbox().verify_evidence(data)
        verified = {"companies": {}}
        if "sector" in data:
            verified["sector"] = self._research_toolbox().verify_evidence(data["sector"])
        for name, sheet in data["companies"].items():
            box = Toolbox(self.fact_docs, self.search, company_website=self.companies[name].website,
                          registry=self.registry, knowledge=self.knowledge, aliases=(name,),
                          rejection_review=self.rejection_reviewer.filter)
            verified["companies"][name] = box.verify_evidence(sheet)
        return verified

    def _run(self, role, task, toolbox, agent_id, name, schema=None):
        group = self._group(role)
        core_role = MAIN_ROLES[group]
        schema = schema or role.output_schema
        box = toolbox or Toolbox()
        box.rejection_review = self.rejection_reviewer.filter
        self.governor.check()
        with self._core_locks[group]:
            refs = self._input_refs(group, agent_id)
            self.board.validate(refs)
            header = {"phase": agent_id, "materials": [ref.to_dict() for ref in refs],
                      "rule": "以下交接清单为唯一当前版本；旧版观点不得替代本轮证据。不得创建子Agent。"}
            task = json.dumps(header, ensure_ascii=False) + "\n\n" + task
            if agent_id == "lead_planner" and getattr(self, "_method_memory", None):
                task += ("\n历史案例验证的方法（仅作研究路径线索，不是事实）："
                         + json.dumps(self._method_memory, ensure_ascii=False))
            if self.last_result.research_plan and agent_id != "lead_planner":
                task += "\n研究议程（待证伪）：" + json.dumps(self.last_result.research_plan, ensure_ascii=False)
                if self.last_result.research_state:
                    task += ("\n当前研究状态（问题回答仍需原文支持；当前假设优先于初始议程，方法记忆不是事实）："
                             + json.dumps(self.last_result.research_state, ensure_ascii=False))
            if group == "lead" and not agent_id.startswith("lead_inquiry_"):
                stage_prompt = ("lead_planning" if agent_id == "lead_planner" else
                                "lead_editing" if agent_id == "chair_editor" else "lead_decision")
                task += "\n本次阶段范围与输出边界：\n" + self._prompt(stage_prompt)
            if group == "lead":
                task += "\n用户持有期（年，null表示未提供，不得代填）：" + json.dumps(self.settings.holding_years)
            if group not in self.agents:
                self.agents[group] = AgentScopeAgent(core_role, self.router.for_role(group), box, self.emit,
                    system=system_prompt(core_role, self.sector, self.settings.fund_profile, self.today,
                                         self.settings.report_max_pages),
                    schema=schema, agent_id=group, name=core_role.name, persistent=True,
                    max_steps=max(self.settings.max_agent_steps, self.settings.role_max_steps.get(group, 0)))
            agent = self.agents[group]
            agent.schema, agent.toolbox = schema, box
            agent.extra_check = lambda data: self._extra_check(group, agent_id, schema, box, data)
            started, before_format = time.monotonic(), agent.format_failures
            try:
                output = agent.run(task)
                self.board.validate(refs)
                if agent.unresolved_issues and not (agent_id.startswith("lead_inquiry_") or agent_id == "audit_inquiry"):
                    self.last_result.unresolved_checks.append({"stage": agent_id, "claim_id": "", "claim": "提交仍有内容问题",
                        "source_agent": group, "verdict": "", "problem": "内容检查未通过", "reason": "；".join(agent.unresolved_issues)})
                if group == "research":
                    output = self._verify_research(output)
                elif group == "audit":
                    output = (normalize_fact_check(output) if self.settings.mock else
                              box.verify_fact_check(output, [c.website for c in self.companies.values() if c.website]))
                if group == "lead" and agent_id != "lead_planner" and not agent_id.startswith("lead_inquiry_"):
                    self.board.publish("memo" if "decision" in output else "decision", output)
                self.last_result.handoffs.append({"agent": group, "phase": agent_id, "status": "accepted",
                    "inputs": [ref.to_dict() for ref in refs]})
                return output
            except Exception:
                self.last_result.handoffs.append({"agent": group, "phase": agent_id, "status": "failed",
                    "inputs": [ref.to_dict() for ref in refs]})
                self.governor.check()  # 停止信号优先于它在框架内部引起的次生错误
                raise
            finally:
                with self._lock:
                    old = self._usage.get(group, [0, 0])
                    self._usage[group] = [old[0] + agent.input_tokens, old[1] + agent.output_tokens]
                    self._format_failures[group] = self._format_failures.get(group, 0) + agent.format_failures - before_format
                    self._agent_seconds[group] = round(self._agent_seconds.get(group, 0) + time.monotonic() - started, 2)
                    if toolbox:
                        records = self._source_reads.setdefault(group, [])
                        records.extend(record for record in box.read_records() if record not in records)
                    if group not in self._costs:
                        self._costs[group] = agent.cost_usd
                    elif self._costs[group] is not None and agent.cost_usd is not None:
                        self._costs[group] += agent.cost_usd
                    else:
                        self._costs[group] = None

    def _review_memo(self, result, correction=""):
        super()._review_memo(result, correction)
        self.board.publish("memo", {"decision": result.decision, "sector": result.sector_brief,
                                    "analyses": result.analyses, "comparison": result.comparison, "perspectives": result.debate})

    def _check_decision(self, result):
        super()._check_decision(result)
        self._repair_publication(result)

    def _publication_preflight(self, result):
        from .brief import render_brief
        from .pdf_report import build_report_pdf, ReportPageLimitError
        preview = deepcopy(result)
        preview.report_status = "演示稿" if self.settings.mock else "待人工复核的研究报告"
        memo = render_brief(preview)
        page_issue = None
        try:
            _, result.report_pages = build_report_pdf(memo, result.report_max_pages)
            # Measured pagination supersedes an earlier failed length estimate.
            closed = [u for u in result.unresolved_checks if u["stage"] == "篇幅预检"]
            if closed:
                result.check_history.append({"stage": "篇幅重新测量通过", "result": {"claims": [], "resolved": deepcopy(closed)}})
                result.unresolved_checks = [u for u in result.unresolved_checks if u not in closed]
        except ReportPageLimitError as exc:
            result.report_pages = None
            page_issue = quality_finding("length", "lead", str(exc), location="memo")
        issues = diagnose(result, memo, review_amount=self.rejection_reviewer.review_amount)
        if page_issue:
            issues.append(page_issue)
        result.quality_issues = issues
        result.report_checks = list(dict.fromkeys(item["message"] for item in issues))
        self.board.publish("quality", {"issues": issues, "pages": result.report_pages})
        result.quality_checks.append({"issues": deepcopy(issues), "pages": result.report_pages,
                                      "materials": self.board.manifest()})
        return issues

    def _repair_publication(self, result):
        for attempt in range(self.settings.publication_repair_rounds + 1):
            issues = self._publication_preflight(result)
            if not issues or attempt == self.settings.publication_repair_rounds:
                return
            owners = {issue["owner"] for issue in issues}
            blocked = "research" in owners and self._repair_count >= 1
            if blocked:
                owners.remove("research")
            record = {"round": attempt + 1, "issues": deepcopy(issues), "actions": [],
                      "before": self.board.manifest(), "status": "running"}
            result.quality_repairs.append(record)
            if not owners:
                record.update(status="repair_limit", after=self.board.manifest())
                return
            self._stage(f"发布检查退回：第 {attempt + 1} 轮定向修复，责任主Agent：{'、'.join(sorted(owners))}")
            correction = ("发布检查未通过，请只处理定位到的问题及其依赖叙述。没有来源的事实主张补查原文或删除。"
                          "不得编造替代数据或缺失的用户输入。不要把修复过程写进报告。\n"
                          + json.dumps(issues, ensure_ascii=False))
            self._quality_active = True
            repair_owner = "lead"
            try:
                upstream = False
                if "research" in owners:
                    repair_owner = "research"
                    problems = [item for item in issues if item["owner"] == "research"]
                    result.check_history.append({"stage": "发布问题回传前的研究核查", "result": deepcopy(result.fact_check)})
                    result.fact_check = {"claims": [{"claim_id": item["issue_id"], "claim": item["target"],
                        "source_agent": item["location"], "verdict": "无法核实", "original_source": "",
                        "evidence": item["message"], "correction": "补查原文或删除主张及依赖说法"} for item in problems],
                        "summary": "发布问题回传研究主Agent"}
                    self._repair(result)
                    record["actions"].append("research")
                    upstream = True
                if "analysis" in owners and not upstream:
                    repair_owner = "analysis"
                    self._repair_publication_analysis(result, issues, correction)
                    record["actions"].append("analysis")
                    upstream = True
                if upstream:
                    repair_owner = "lead"
                    result.decision = self._run(CHAIR, self._chair_task(result) + "\n" + correction, None,
                        "quality_decision", "总控与决策主Agent", self._decision_schema(result))
                if upstream or "lead" in owners:
                    repair_owner = "lead"
                    if any(item["kind"] == "length" for item in issues):
                        correction += (f"\n篇幅预检没有通过；正文压缩至{self._char_budget(2200)}字以内，"
                                       "保留重大风险、反证、下行情景和所用来源。")
                    self._review_memo(result, correction)
                    record["actions"].append("lead")
                if any(item["location"] in ("research_audit", "audit_research", "audit_repair", "audit_coverage")
                       and item["owner"] == "audit" for item in issues):
                    repair_owner = "audit"
                    self._retry_publication_research_audit(result)
                    record["actions"].append("audit_research")
                repair_owner = "audit"
                self._audit_publication_repair(result, issues, attempt + 1)
                record["actions"].append("audit_final")
                for owner in ("research", "analysis", "lead", "audit"):
                    if owner in record["actions"] or owner == "audit":
                        result.errors.pop("quality_repair:" + owner, None)
                record.update(status="rechecked", after=self.board.manifest())
            except RunStopped as exc:
                record.update(status="stopped", error=str(exc), after=self.board.manifest())
                raise
            except Exception as exc:
                record.update(status="failed", error=str(exc), after=self.board.manifest())
                result.errors["quality_repair:" + repair_owner] = str(exc)
                # A partially changed upstream must never fall through as a valid memo.
                if not result.decision:
                    raise
                continue
            finally:
                self._quality_active = False

    def _repair_publication_analysis(self, result, issues, correction):
        names = [name for name in result.analyses if any(
            item["location"].startswith(f"analyses.{name}.") or item["location"] == f"company_analyst:{name}"
            for item in issues if item["owner"] == "analysis")]
        if not names and any(item["owner"] == "analysis" and item["kind"] == "stage_error" for item in issues):
            names = list(result.analyses)
        if names:
            sheets = {name: result.fact_sheets[name] for name in names}
            task = correction + "\n只修正以下公司的分析，保留统一评分口径，逐项绑定当前证据：\n" + json.dumps({
                "research": sheets, "sector": result.sector_brief, "previous": {n: result.analyses[n] for n in names},
                "financial": result.financial_metrics, "market": result.market_metrics, "deal": result.deal_metrics}, ensure_ascii=False)
            data = self._run(COMPANY_ANALYST, task, None, "quality_analysis", "投资分析主Agent",
                             schemas.analysis_bundle_schema(self.sector, sheets))
            result.analyses.update(data["analyses"])
        result.ranking = rank_companies(result.analyses, self.sector)
        result.sensitivity = ranking_sensitivity(result.analyses, self.sector)
        result.score_ranges = score_ranges(result.analyses, result.fact_sheets, self.sector)
        self.board.publish("analysis", {"analyses": result.analyses, "ranking": result.ranking})
        self._run_compare(result)  # 分析已改，旧比较失效，必须重做

    def _retry_publication_research_audit(self, result):
        materials, plan, task, schema = self._research_audit(result, "重新核查当前版本研究包的关键事实，不重做研究。")
        checked = self._run(FACT_CHECKER, task, self._factcheck_toolbox(materials),
                            "quality_research_audit", "独立审计主Agent", schema)
        result.fact_check = checked
        for index, claim in enumerate(checked.get("claims", []), 1):
            claim["claim_id"] = f"Q-research-{len(result.quality_repairs)}-{index}"
        self._record_coverage(result, plan)
        if checked.get("claims"):
            result.errors.pop("audit_research", None)
            result.errors.pop("audit_repair", None)
        result.check_history.append({"stage": "发布退回：研究核查重试", "result": deepcopy(checked)})
        self.board.publish("research_audit", checked)

    def _audit_publication_repair(self, result, issues, attempt):
        materials = {**self._research_materials(result), "chair": memo_materials(result)}
        # Actual pagination is verified by the renderer, not by a model's claim.
        audit_issues = [item for item in issues if item["kind"] != "length"
                        and not (item["kind"] == "unresolved" and item["unresolved"]["stage"] == "篇幅预检")]
        schema = schemas.decision_check_schema(result.decision, self._evidence_ids(result, verified_only=True),
                                               [item["issue_id"] for item in audit_issues])
        schema["properties"]["claims"]["items"]["properties"]["source_agent"]["enum"] = list(materials)
        task = ("独立复核修复后的当前成稿与核心推论。逐项复查下列发布问题，repair_checks覆盖每个issue_id；"
                "repair_checks 的 claim_id 填对应问题的 issue_id。"
                "有修改不等于已解决，不能确认则写无法判断；缺少原文不得编造。\n"
                + json.dumps({"previous_issues": audit_issues, "current": materials}, ensure_ascii=False)
                + self._evidence_priorities(result, {"chair": materials["chair"]})
                + "\n当前证据：\n" + self._evidence_context(result) + self._reasoning_context(result))
        checked = self._run(FACT_CHECKER, task, self._factcheck_toolbox(materials),
                            "quality_final_audit", "独立审计主Agent", schema)
        checked["reviewed_targets"] = review_targets(result.decision)
        result.decision_fact_check = checked
        result.check_history.append({"stage": f"发布定向修复第{attempt}轮复查", "result": deepcopy(checked)})
        if checked.get("claims") and not review_issues(result.decision, checked):
            for key in ("fact_checker_decision", "fact_checker_decision_recheck"):
                result.errors.pop(key, None)
        confirmed = {item["claim_id"] for item in checked.get("repair_checks", []) if item["status"] in ("已修正", "已删除")}
        bad = [c["claim"] for check in (result.fact_check, checked) for c in check.get("claims", [])
               if c["verdict"] in ("不实", "无法核实")]
        bad_reasoning = {item["target_id"] for item in review_issues(result.decision, checked)}
        resolved = [item["unresolved"] for item in issues if item["kind"] == "unresolved"
                    and item["issue_id"] in confirmed and item["target"] not in bad
                    and item["unresolved"]["claim_id"] not in bad_reasoning
                    and not (item["owner"] == "research" and self._repair_count >= 1
                             and (not result.fact_check.get("claims")
                                  or any(c["verdict"] in ("不实", "无法核实") for c in result.fact_check.get("claims", []))))]
        result.unresolved_checks = [item for item in result.unresolved_checks if item not in resolved]
        result.quality_repairs[-1]["resolved"] = deepcopy(resolved)
        statuses = {item["claim_id"]: item for item in checked.get("repair_checks", [])}
        for item in audit_issues:
            if item["issue_id"] in confirmed or item["kind"] == "unresolved":
                continue
            row = statuses.get(item["issue_id"], {})
            result.unresolved_checks.append({"stage": "发布定向修复", "claim_id": item["issue_id"],
                "claim": item["target"] or item["message"], "source_agent": item["owner"], "verdict": "无法核实",
                "problem": row.get("status", "未复查"), "reason": row.get("reason", "修复未得到独立确认")})
        self.board.publish("final_audit", checked)

    def _finish_result(self, result, started):
        super()._finish_result(result, started)
        if result.status != "failed" and result.quality_issues:
            result.report_status = "定向修复后仍未通过的研究草稿"
        if result.decision_fact_check:
            self.board.publish("final_audit", result.decision_fact_check)
        result.architecture = "agentscope-five-agents-two-round-debate"
        result.main_agents = list(self.agents)
        result.material_versions = self.board.manifest()
