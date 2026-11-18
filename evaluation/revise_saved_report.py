"""离线落实本次报告复核意见，复用模型检查点，不重新调用模型或检索网络。"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vcagents import schemas
from vcagents.financials import calculate_returns
from vcagents.grounding import check_material, evidence_index
from vcagents.pipeline import AnalysisResult
from vcagents.report import render_audit, render_memo
from vcagents.scoring import rank_companies, ranking_sensitivity, score_ranges
from vcagents.tools import Toolbox


def main():
    folder = Path(sys.argv[1])
    original = json.loads((folder.parent / "partial-result.json").read_text(encoding="utf-8"))
    saved = json.loads((folder / "checkpoint.json").read_text(encoding="utf-8"))
    stage = saved.pop("stage")
    result = AnalysisResult(**saved)
    generated = deepcopy(result.analyses)
    restore_nemo_facts(result, original)
    restore_market_estimates(result, original)
    revise_analyses(result)
    sector = result.sector_config()
    result.ranking = rank_companies(result.analyses, sector)
    result.sensitivity = ranking_sensitivity(result.analyses, sector)
    result.score_ranges = score_ranges(result.analyses, result.fact_sheets, sector)
    revise_conclusion(result)
    errors = schemas.validate_submission("chair", schemas.decision_schema(result.companies), result.decision)
    if errors:
        raise ValueError("；".join(errors))
    calculated = calculate_returns(None, 0.5, result.decision["exit_scenarios"])
    result.returns = calculated["scenarios"]
    result.return_summary = {"expected_multiple": None, "dilution": 0.5, "dilution_source": "系统默认"}
    result.status, result.report_status = "reviewed_draft", "证据复核修订稿（自动核查未完成）"
    result.errors = {"automatic_review": "自动核查反复检索，在形成有效提交前主动结束；未执行自动主席复核"}
    result.fact_check, result.decision_fact_check = {}, {}
    result.fact_check_pass_rate = result.checked_source_coverage = None
    result.check_history, result.unresolved_checks = [], []
    result.finished_at = datetime.now().isoformat(timespec="seconds")
    result.elapsed_seconds = round((datetime.now() - datetime.fromisoformat(result.started_at)).total_seconds(), 2)
    result.usage, result.costs_usd, result.agent_seconds, result.search_usage = {}, {}, {}, {}
    result.run_control = {}
    index = evidence_index(result.sector_brief, result.fact_sheets)
    result.evidence_checks = {f"company_analyst:{name}": check_material(f"company_analyst:{name}", analysis, index)
                              for name, analysis in result.analyses.items()}
    result.evidence_checks["chair"] = check_material("chair", result.decision, index)
    review = {"mode": "存档编辑复核；不是完整自动运行通过记录", "original": str(folder.parent.resolve()),
              "automatic_stage_reused": stage, "automatic_review": result.errors,
              "billing": "中断前的调用用量未完整归集，不将原运行费用当成本次费用，也不将缺失费用记为零",
              "score_rule": "商业化、竞争壁垒和风险缺少可比底稿时中性5分、低置信度；市场同赛道统一7分，不因公开流量加分；Vattention当前产品成熟度不由3月资金用途推断",
              "analysis_before_editor_review": generated, "analysis_after_editor_review": result.analyses,
              "checks": {"neutral_traction": all(a["scores"]["traction"]["score"] == 5 and a["scores"]["traction"]["confidence"] == "低" for a in result.analyses.values()),
                         "restored_nemo_facts": ["E-C2-4", "E-C2-5", "E-C2-6", "E-C2-18"],
                         "statement_findings": [f for check in result.evidence_checks.values() for f in check["findings"]]}}
    report = render_memo(result)
    for phrase in ("无法核实", "待核实", "现金跑道短", "10+个模型", "10+ 个模型", "credits 月末失效", "付款不可退款", "36个权重情景"):
        if phrase in report:
            raise ValueError("正文仍包含本次应删除或更正的表述：" + phrase)
    (folder / "result.json").write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    (folder / "editor-review.json").write_text(json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8")
    (folder / "research-revised.md").write_text(report, encoding="utf-8")
    audit = render_audit(result) + "\n\n## 本轮编辑复核\n\n" + json.dumps({k: v for k, v in review.items() if not k.startswith("analysis_")}, ensure_ascii=False, indent=2)
    (folder / "audit.md").write_text(audit, encoding="utf-8")
    print(json.dumps({"report": str((folder / "research-revised.md").resolve()), "ranking": result.ranking,
                      "numeric_findings": len(review["checks"]["statement_findings"])}, ensure_ascii=False))


def restore_nemo_facts(result, original):
    name = result.companies[1]
    originals = {e["evidence_id"]: e for e in original["fact_sheets"][name]["evidence"]}
    changes = {
        "E-C2-4": ("据报道，李政锦曾任TikTok Creative产品Leader，后深耕多模态领域。", "李政锦曾任TikTok Creative产品Leader，后深耕多模态领域；"),
        "E-C2-5": ("据报道，刘婧妍有北大和剑桥教育背景，负责过摩拜海外及TikTok全球创意运营。", "刘婧妍（北大+剑桥毕业）负责过摩拜海外及TikTok全球创意运营；"),
        "E-C2-6": ("据报道，张星辰此前任源码资本董事总经理。", "张星辰是前源码资本董事总经理，补齐商业布局短板。"),
        "E-C2-18": ("据公司文档，Seedance 2.5向Starter及以上套餐开放。", "Seedance 2.5 is available for Starter+"),
    }
    sheet = result.fact_sheets[name]
    before = deepcopy(sheet)
    edits = [{**originals[key], "claim": claim, "excerpt": quote} for key, (claim, quote) in changes.items()]
    reads = [read for records in original["source_reads"].values() for read in records if read["kind"] != "document"]
    checked = Toolbox(shared_reads=reads).verify_evidence({"sources": sheet["sources"], "evidence": edits})["evidence"]
    for item in checked:
        if not item["excerpt_verified"]:
            raise ValueError("修订摘录没有定位到已读原文：" + item["evidence_id"])
        item["source_linked"] = True
    sheet["evidence"] = [e for e in sheet["evidence"] if e["evidence_id"] not in changes] + checked
    sheet["fact_sheet"] = "\n".join(f"- {e['claim']} [{e['evidence_id']}]" for e in sheet["evidence"])
    result.repairs.append({"source_agent": "company_researcher:" + name, "claims": [], "before": before, "after": deepcopy(sheet),
                           "editor_reason": "按存档原文修正履历摘录，删除无依据的持股比例；Seedance仅保留文档支持的版本与套餐表述"})


def revise_analyses(r):
    pexo, nemo, vattention = r.companies
    for a in r.analyses.values():
        a["scores"].update({
            "market": {"score": 7, "confidence": "低", "evidence_ids": [], "reason": "按共同赛道机会统一设定研究评分；未取得各公司同口径可服务客户与客单价，不以公开流量差异加分。"},
            "competition": {"score": 5, "confidence": "低", "evidence_ids": [], "reason": "缺少客户迁移、留存及同任务横评，尚不足以比较壁垒强弱，采用中性排序假设。"},
            "traction": {"score": 5, "confidence": "低", "evidence_ids": [], "reason": "三家均缺少可比收入质量、付费队列留存和贡献毛利底稿；中性占位，不代表实际业绩。"},
            "risk": {"score": 5, "confidence": "低", "evidence_ids": [], "reason": "共有供应、版权与出海经营风险，但缺少条款、现金与消耗底稿，采用中性占位；不由融资金额推断现金跑道。"},
        })
        a["confidence"] = "低"
        a["open_questions"] = ["取得融资主体、股权与核心成员任职及全职到岗证明。", "取得付费队列、实际收入、模型账单、现金余额和净烧钱底稿。", "在相同素材与任务下测试成片质量、返工、耗时和全成本。"]
    a = r.analyses[pexo]
    a.update(one_liner="面向内容创作的对话式视频Agent，公开材料介绍了成片编辑与局部修改交互。",
        summary="团队公开履历与产品方向匹配，局部修改交互有具体材料；经营信号主要来自公司自述。优先安排产品和收入质量尽调，当前资料不支持投资批准。",
        product_business="据公司材料，Mark to Fix允许圈选画面元素并用自然语言说明修改意图[E-C1-15]。这说明交互设计，不能替代真实任务质量和单位成本测试；不保留模型个数或未经原文支持的收费条款。",
        operating_assessment="据所读团队访谈，单周收入年化运行率超过百万美元[E-C1-6]。据公司披露，注册用户近百万、付费用户复购率超过50%[E-C1-7]。后续重点对齐收入构成、复购分母及观察窗口，并取得队列留存和贡献毛利底稿。",
        ai_native="公开材料所描述的生成、编排与修改流程依赖模型能力；是否形成可持续工作流价值，需要同任务测试和付费留存验证。",
        strengths=["创始人报道履历涉及视频生成产品和海外创作工具[E-C1-3]。", "局部修改有明确交互描述[E-C1-15]，可据此设计体验测试。"],
        concerns=["收入运行率、复购和成本均缺少可独立核对的经营底稿，需统一口径。", "与既有创作工具相比的质量、成本及迁移壁垒尚需测试。"])
    a["scores"]["product"].update(score=7, confidence="低", reason="局部修改交互有具体公开描述，但未做同任务体验；该研究评分不表示已证明产品质量领先。", evidence_ids=["E-C1-15"])
    a = r.analyses[nemo]
    a.update(one_liner="面向剪辑与内容生产的视频Agent，公开材料涉及短剧粗剪及生成模型接入。",
        summary="团队公开报道能支持产品、增长及资本侧经验，不能再称创始人履历完全缺失。历史媒体实测与公司文档可作为测试线索；收入、留存和成本底稿仍需补齐。",
        team_assessment="据公开报道，李政锦曾任TikTok Creative产品Leader[E-C2-4]，刘婧妍负责过摩拜海外及TikTok全球创意运营[E-C2-5]，张星辰此前任源码资本董事总经理[E-C2-6]。这些是报道口径；不保留缺少原文支持的持股数字，任职证明与实际分工需尽调。",
        product_business="历史媒体实测讨论了剪辑和短剧粗剪[E-C2-20][E-C2-21]；公司文档介绍Seedance接入[E-C2-18]。历史评测不代表当前版本效果，接入某一模型也不证明只能依赖该供应商。",
        operating_assessment="本次资料未取得可比收入、付费留存、贡献毛利和现金底稿。融资报道提供资本背景[E-C2-8][E-C2-9]，不能替代经营表现或现金跑道测算。",
        ai_native="材料显示产品利用模型执行视频任务；是否降低交付成本、能否形成编辑工作流粘性，需以实际使用与付费数据验证。",
        strengths=["报道履历提供了产品与增长经验的具体核对线索[E-C2-4][E-C2-5]。", "已有历史实测，可据同类任务验证当前版本[E-C2-20]。"],
        concerns=["缺少当前版本的可重复测试和可比经营底稿。", "需要确认模型商务安排、备选方案与融资主体；不能据此直接认定存在致命风险。"])
    a["scores"]["team"].update(score=7, confidence="中", reason="补回报道支持的三位成员经历，保留任职与实际分工尽调要求。", evidence_ids=["E-C2-4", "E-C2-5", "E-C2-6"])
    a["scores"]["product"].update(score=5, confidence="低", reason="历史实测提供线索，尚缺当前版本同任务测试，采用中性成熟度假设。", evidence_ids=["E-C2-20", "E-C2-18"])
    a = r.analyses[vattention]
    a.update(one_liner="定位专业视频后期与人机协同创作的AI Agent项目。",
        summary="团队报道与视频后期方向有匹配线索，但现有材料不足以判断当前产品成熟度和经营表现。产品化资金用途仅代表披露当时的计划，不能推出截至研究日仍未形成产品。",
        team_assessment="公开报道介绍钟超的视频后期创作及产品经验[E-C3-6]，并介绍未具名CTO的大模型研发经历[E-C3-8]和团队能力组合[E-C3-9]。应取得实名、前雇主任职和全职到岗证明，不把背景介绍等同于商业化成绩。",
        product_business="材料介绍专业后期人机协同定位与“指数剪辑”框架[E-C3-1][E-C3-12]。这属于产品理念，尚无材料证明效率按指数函数增长。资金用途中的引擎产品化安排[E-C3-18]也不能作为当前完成状态的证据。",
        operating_assessment="所读资料没有覆盖可比收入、留存、毛利、现金余额与月消耗。种子轮报道[E-C3-2]不能单独证明现金充裕或跑道短。",
        ai_native="方向涉及模型对素材、上下文和编辑意图的理解；实际加速效果、控制能力及端到端交付仍需可操作版本与对照任务验证。",
        strengths=["报道中的创作与研发能力组合与后期方向相关[E-C3-6][E-C3-8][E-C3-9]。"],
        concerns=["当前版本与真实使用结果资料不足，不能判断效果或成熟度。", "需补齐经营、现金和团队到岗底稿；这些缺口不直接等于业务差。"])
    a["scores"]["team"].update(score=7, confidence="中", reason="保留报道所支持的创作及研发能力判断，实名和任职仍待尽调材料。", evidence_ids=["E-C3-6", "E-C3-8", "E-C3-9"])
    a["scores"]["product"].update(score=5, confidence="低", reason="不能由历史融资用途或招聘推断当前产品未完成；缺少同任务测试，中性占位。", evidence_ids=[])


def revise_conclusion(r):
    pexo, nemo, vattention = r.companies
    r.comparison = {"summary": "Pexo优先安排尽调，NemoVideo与Vattention按同档研究；当前材料不支持稳定的业务优劣或价格吸引力排序。",
        "per_company": [{"company": n, "position": a["one_liner"], "edge": "；".join(a["strengths"]), "gap": "；".join(a["concerns"])} for n, a in r.analyses.items()],
        "ranking_view": "编辑复核已同步修正团队信息、产品时点及缺失数据处理。Pexo的相对领先主要来自团队履历和更具体的交互材料；NemoVideo与Vattention同分，展示顺序不表示业务差异。商业化等维度采用低置信度占位，排序只决定尽调排期。评分扰动不是权重测试，也不产生投资成功概率。",
        "top_pick": pexo}
    r.debate = [{"round": 1, "side": "bull", "name": "看多意见（编辑复核归纳）", "thesis": "支持优先验证Pexo的创作与修改流程，尚不支持投资批准。",
        "arguments": ["公开报道中的产品经历与方向匹配[E-C1-3]。", "局部修改交互提供了可实际测试的产品命题[E-C1-15]。"], "rebuttals": [], "would_change_mind": "若同任务测试没有交付优势，或收入质量与贡献毛利不成立，则取消优先级。"},
        {"round": 1, "side": "bear", "name": "看空意见（编辑复核归纳）", "thesis": "保留三家公司继续尽调选择，避免把宣传功能或资料丰富程度等同于经营优势。",
        "arguments": ["公司自述的收入运行率和复购不足以证明可持续收入[E-C1-6][E-C1-7]。", "历史评测和资金用途不足以判定NemoVideo、Vattention当前产品效果[E-C2-20][E-C3-18]。"], "rebuttals": [], "would_change_mind": "可比队列数据、成本底稿和同任务测试证明稳定价值，并且交易价格合理。"}]
    r.decision = {"verdict": "值得关注", "action": "继续尽调", "action_conditions": ["取得真实收入、付费队列和模型成本底稿。", "取得本轮条款、估值、现金与消耗数据后，再讨论投资批准。", "完成同素材、同任务的产品测试，并允许据结果调整尽调优先级。"],
        "headline": "Pexo优先尽调，NemoVideo与Vattention同档跟进；当前资料支持研究优先级判断，尚不足以批准投资。",
        "recommended": [pexo, nemo], "top_pick": pexo,
        "thesis": "首选对象仅用于安排尽调资源。Pexo的团队报道与交互设计有较具体线索[E-C1-3][E-C1-15]；收入运行率、用户及成本仍属公司披露[E-C1-6][E-C1-7][E-C1-8]，需取得原始底稿。NemoVideo的团队履历已从存档报道补回，Vattention的产品成熟度也不再由历史资金用途反推。三家的商业化均采用中性占位，当前不作稳定业绩排名，也不作价格吸引力判断。",
        "sector_risks": [{"risk": "供应与交付成本", "impact": "模型价格、服务可用性与返工可能影响单位经济", "evidence_gap": "缺少合同、账单及同任务全成本数据", "response": "核对模型合同、账单与备选方案，计算合格交付成本"},
            {"risk": "可持续付费需求", "impact": "短期运行率可能不能持续", "evidence_gap": "缺少队列留存及收入构成", "response": "取得支付与退款底稿，区分订阅、加购和一次性收入"},
            {"risk": "内容权利与出海经营", "impact": "素材授权和主体安排可能影响业务推进", "evidence_gap": "缺少授权、投诉处理和数据流材料", "response": "核查素材授权、服务条款及经营主体，不以资料缺失认定已有违规"}],
        "top_pick_thesis": [{"judgment": "团队经历提供优先核对线索", "evidence": "公开报道介绍视频生成与海外创作工具经历[E-C1-3]。", "why_opportunity": "相关产品经验可能缩短产品化试错，但仍需核对实际贡献。", "disproof": "任职或业绩不能得到佐证，核心团队没有全职到岗。"},
            {"judgment": "局部修改值得设计真实任务测试", "evidence": "公司材料介绍Mark to Fix交互[E-C1-15]。", "why_opportunity": "若减少返工和人工接管，可能改善交付价值。", "disproof": "测试中质量、修改轮次或全成本没有优势。"},
            {"judgment": "收入信号需转化为可核对的经营底稿", "evidence": "公司披露单周收入年化运行率及用户、复购信号[E-C1-6][E-C1-7]。", "why_opportunity": "若付费队列和贡献毛利成立，才有进一步研究扩张的依据。", "disproof": "收入主要为一次性购买，留存或贡献毛利不能成立。"}],
        "top_pick_risks": ["收入口径和队列质量尚缺原始底稿；取得按期收入、退款及留存后再作判断。", "公司披露Token成本在经营支出中占比较高[E-C1-8]；需核对成本分母与贡献毛利，不能直接推定亏损。", "交付价值与迁移壁垒需同任务对照测试。"],
        "valuation_view": "本次没有本轮投后估值、拟投资额及可比收入底稿，不作估值合理性或隐含倍数判断。头部公司融资案例也不能直接替代同阶段可比。",
        "use_of_funds_milestones": "需向公司取得按期资金使用、团队与产品计划、现金消耗和经营里程碑；不由融资金额反推跑道。",
        "forecast_assessment": "补齐基期收入、付费队列、模型及人工成本、现金余额和增长假设后，才能建立逐年情景预测。当前只列驱动因素与验证路径。",
        "exit_scenarios": [{"scenario": s, "condition": c, "valuation_basis": "缺少收入、盈利、净现金及同阶段交易价格输入，暂不提供退出股权价值数值。", "exit_path": p, "exit_valuation_musd": None} for s, c, p in [
            ("乐观", "假设留存与贡献毛利成立，并形成可重复交付与客户粘性。", "战略并购或后续上市"),
            ("基线", "假设形成可持续细分业务，但增长或议价能力有限。", "并购或老股转让"),
            ("悲观", "假设需求、单位经济或后续融资不能成立。", "低价出售或清算")]],
        "dd_checklist": ["向创始人及财务负责人索取支付后台、退款与收入明细，统一收入定义。", "向产品和技术负责人索取可操作版本及账单，做同素材任务测试。", "向公司索取成员实名、任职与全职证明，核对团队分工。", "向财务负责人索取现金余额、月消耗和资金使用计划，按代码测算跑道。", "取得本轮估值、融资额和主要条款，才测算持股与退出回报。"],
        "next_steps": ["先完成三家的可比材料清单，再按同任务测试安排产品访谈。", "收入质量、产品效果与交易条件补齐后再更新评分和投资建议。"],
        "dissent": "当前分数包含较多低置信度研究假设，资料补齐可能改变顺序。NemoVideo与Vattention同分，不应把展示名次解释为真实业务差距。"}


def restore_market_estimates(r, original):
    b = r.sector_brief
    market = deepcopy(next(e for e in original["sector_brief"]["evidence"] if e["evidence_id"] == "E-S-19"))
    market["claim"] = "Research and Markets估计2025年AI内容生成市场接近48.1亿美元，预测2030年166.3亿美元、2035年547.1亿美元。"
    market["as_of"] = "2025基期；2030/2035预测"
    gvr = next(c for c in original["decision_fact_check"]["claims"] if c["claim_id"] == "D-7")
    b["sources"].append("[S-13] Grand View Research：Generative AI In Content Creation Market - " + gvr["verification_source"])
    second = {"evidence_id": "E-S-25", "source_ref": "[S-13]", "source_type": "独立来源",
              "claim": "Grand View Research估计2024年生成式AI内容创作市场为148亿美元，预测2026年260亿美元、2030年801亿美元。",
              "excerpt": gvr["verification_excerpt"], "as_of": "2024基期；2026/2030预测"}
    reads = [read for records in original["source_reads"].values() for read in records if read["kind"] != "document"]
    checked = Toolbox(shared_reads=reads).verify_evidence({"sources": b["sources"], "evidence": [market, second]})["evidence"]
    for e in checked:
        if not e["excerpt_verified"]:
            raise ValueError("机构市场估计原文未匹配：" + e["evidence_id"])
        e["source_linked"] = True
    b["evidence"] = [e for e in b["evidence"] if e["evidence_id"] not in ("E-S-19", "E-S-25")] + checked
    b["market_size"] = ("两组机构估计均有存档原文，保留其各自基期与预测年份：\n\n"
        "| 机构与范围 | 基期估计 | 预测 | 来源 |\n| --- | --- | --- | --- |\n"
        "| Research and Markets：AI内容生成 | 2025年48.1亿美元 | 2030年166.3亿美元；2035年547.1亿美元 | [E-S-19] |\n"
        "| Grand View Research：生成式AI内容创作 | 2024年148亿美元 | 2026年260亿美元；2030年801亿美元 | [E-S-25] |\n\n"
        "目前资料不足以对齐两份报告的统计范围与收入定义，列为两组外部估计。目标公司的可服务市场仍需客户数、客单价和去重底稿；"
        "不把不同日期的ARR或收入运行率汇总为实际年度市场收入。")


if __name__ == "__main__":
    main()
