"""各智能体的结构化输出格式（JSON Schema），以及宽松的类型修正和校验。"""
from __future__ import annotations

from copy import deepcopy

from .quantities import loose_text
import json
import re
import math

from .sectors import Sector
from .reasoning import VERDICTS, review_targets
from .citations import MAX_FACT_STATEMENT_CHARS


def _obj(props: dict) -> dict:
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def _str(desc: str) -> dict:
    return {"type": "string", "description": desc}


def _list(desc: str) -> dict:
    return {"type": "array", "items": {"type": "string"}, "description": desc}


def _int(desc: str) -> dict:
    return {"type": "integer", "description": desc}


def _enum(options: list[str], desc: str) -> dict:
    return {"type": "string", "enum": options, "description": desc}


def _rows(item: dict, desc: str) -> dict:
    return {"type": "array", "items": item, "description": desc}


SOURCES = _list("引用的来源，格式“[n] 标题 - URL”；来自上传资料的写“[n] 《文件名》第 x 页”")


def _quote(desc: str) -> dict:
    return {**_obj({
        "source": _str("工具返回的来源编号，例如 W3 或 D1；没有时写空字符串"),
        "from": _int("起始句号，即 [W3:s5] 中的 5"),
        "to": _int("结束句号；与起始句号最多相差 4，即最多连续 5 句"),
    }), "description": desc}


FACT_TYPES = ["融资", "估值", "收入或ARR", "用户或客户", "产品发布", "团队履历", "合作或订单", "市场规模", "其他"]
# 结构化事实：摘录由代码按引用位置取出，模型不抄写原文；主张句由代码拼出
FACT_FIELDS = {
    "subject": _str("事实主体，例如公司或市场名称"),
    "fact_type": _enum(FACT_TYPES, "事实类型，一条只写一种"),
    "statement": _str(f"一条围绕一个事实，最多 {MAX_FACT_STATEMENT_CHARS} 字；允许分号等常见标点。数字可换算成中文写法，但须与原文数值相等，不补原文没有的数字或年份"),
    "quantities": {**_rows(_obj({
        "metric": _str("数量含义，例如融资额、ARR、用户数"),
        "text": _str("按原文写法原样照抄，例如 USD 14.8 billion、超千万美元"),
    }), "事实中的关键数量，最多 3 个"), "maxItems": 3},
    "entities": _list("投资方、客户、合作方等名称，按原文写法照抄；没有时为空"),
    "date": _obj({
        "text": _str("事实日期，按原文写法，例如 7 月、2026-08-17；原文没写时为空字符串"),
        "basis": _enum(["摘录", "来源其他位置", "未给出"], "日期出处：引用的句子里、来源其他位置（须填 date_quote），或未给出"),
    }),
    "quote": _quote("支持该事实的原文位置：来源编号加连续 1–5 句的句号"),
    "date_quote": _quote("仅当 date.basis 为“来源其他位置”时填写写有日期的那句；否则 source 写空字符串、句号写 0"),
}
EVIDENCE = _rows(_obj({
    **FACT_FIELDS,
    "source_ref": _str("对应 sources 的来源编号，例如 [1]，须与 quote 引用的是同一来源"),
    "source_type": _enum(["公司口径", "独立来源", "其他", "未知"], "说明来源性质，不代表核查结论"),
}), "重要事实逐条记录，每条一个事实并给出引用位置；没有可引用原文的事实不要写入，列入信息缺口")

SECTOR_BRIEF = _obj({
    "summary": _str("一段话：这个赛道值不值得关注、为什么"),
    "market_size": _str("用 Markdown 说明市场规模。按有来源的输入选择自上而下或自下而上估算，注明市场范围、统计期间、来源和假设。"
                        "仅一种方法有依据时只给该结果。两种都有依据时可对照口径和差异。均缺乏依据时写明缺少哪些输入，保留总量未知。"),
    "why_now": _list("为什么是现在：2-4 条"),
    "landscape": _str("Markdown 赛道地图：玩家分哪几类，每类的代表公司和特点"),
    "benchmarks": _list("参照公司及其关键数据（收入、估值、融资），每条注明来源"),
    "candidates": _list("值得研究的早期、可投公司名单（只写公司名）；用户已指定公司时给空数组"),
    "sources": SOURCES,
    "evidence": EVIDENCE,
})

FACT_SHEET = _obj({
    "company_name": _str("公司名称（中英文）"),
    "one_liner": _str("一句话说明公司做什么"),
    "stage": _str("已知最新融资阶段，不知道就写“未知”"),
    "listing_status": _enum(["未上市", "已上市", "待核实"], "公司目前上市状态，不能确定时写待核实"),
    "listing_evidence": _str("上市状态的依据与来源；没有可靠依据时写明缺少哪类交易所或监管披露"),
    "fact_sheet": _str(
        "Markdown 事实底稿，分为：公司概况、创始团队、产品与技术、商业化与关键数据、融资历史、竞争对手、近期动态。"
        "关键事实后用 [n] 标注来源"
    ),
    "information_gaps": _list("重要但没查到、或无法核实的信息"),
    "sources": SOURCES,
    "evidence": EVIDENCE,
})


def analyst_schema(sector: Sector, evidence_ids: list[str] | None = None) -> dict:
    """公司分析师的输出格式随赛道变化：每个评分维度一项。"""
    def dimension(d) -> dict:
        ids = _list("支持评分的证据 ID；证据不足时为空")
        if evidence_ids is not None:
            if evidence_ids:
                ids["items"]["enum"] = evidence_ids
            else:
                ids["maxItems"] = 0
        return _obj({
            "score": {**_int(f"{d.name}评分，1-10 的整数，按赛道评分标准打分"), "minimum": 1, "maximum": 10},
            "reason": _str("打分理由，引用事实底稿中的证据"),
            "confidence": _enum(["高", "中", "低"], "对该维度评分的置信度"),
            "evidence_ids": ids,
        })

    return _obj({
        "one_liner": _str("一句话说明公司做什么"),
        "scores": _obj({d.key: dimension(d) for d in sector.dimensions}),
        "confidence": _enum(["高", "中", "低"], "对整体评分的把握，信息越少越低"),
        "ai_native": _str("AI 原生程度：产品是否以大模型为核心，去掉大模型还剩多少价值"),
        "summary": _str("一页纸结论，100 字以内"),
        "team_assessment": _str("团队背景、分工和方向匹配，引用资料；缺少资料时写明缺少哪类材料，不写核查状态标签，150 字以内"),
        "product_business": _str("产品、目标客户和收费模式，区分已落地与计划，150 字以内"),
        "operating_assessment": _str("收入、留存、客户、交付或现金等已披露经营事实，注明期间、口径和来源；未知不补造，200 字以内"),
        "strengths": _list("亮点"),
        "concerns": _list("担忧和问题"),
        "open_questions": _list("尽调时需要向公司核实的问题"),
        **insight_properties(),
    })


# 结构化观点：要求出现在报告中的分析判断必须有字段，否则只会被写进泛泛的段落。
LONG_TERM_DIMENSIONS = ["行业", "竞对", "技术", "商业模式", "收入", "成本"]
INSIGHT_KEYS = ("standout", "competitor_diffs", "cost_structure", "long_term_edge")


def insight_properties() -> dict:
    ids = _list("支撑该判断的当前证据编号（赛道或任一公司的 E- 编号均可）；没有证据时为空数组")
    return {
        "standout": _obj({
            "point": _str("这家公司相对最接近竞对最突出的一点，只选一个主点，一句话，最多60字；没有明显突出点就直说，不罗列多项优点。当前产品差异不自动等于持久护城河"),
            "mechanism": _str("这一点凭什么让它在竞争中胜出或难以持续：能力→用户收益→付费/成本→竞争的传导链，最多120字；"
                              "收入或融资规模只是结果，不能当作原因；未能证明持续性时，只称当前差异或待验证优势"),
            "evidence_ids": ids,
        }),
        "competitor_diffs": {**_rows(_obj({
            "competitor": _str("对比对象，优先选最接近的直接竞对或替代方案（含模型厂商自营产品）"),
            "axis": _str("比较维度，例如模型来源、目标客户、定价、交付成本、分发渠道"),
            "difference": _str("具体差异及其对投资判断的含义，最多100字；没有证据的差异写明是推断"),
            "evidence_ids": ids,
        }), "与竞对的最大差异，1-3 条，按重要性排序"), "minItems": 1, "maxItems": 3},
        "cost_structure": _obj({
            "rnd_investment": _str("研发投入：自研模型、算力、团队规模等证据；未知写“未披露”，最多80字"),
            "delivery_cost": _str("交付成本：每单位产出的推理、算力或上游模型费用；报价不等于公司成本；未知写“未披露”，最多80字"),
            "user_cost": _str("用户总成本：标价、套餐、实际使用花费，区分标价与实际成本；最多80字"),
            "acquisition_cost": _str("获客方式与成本；融资额不能证明获客效率；未知写“未披露”，最多80字"),
            "evidence_ids": ids,
        }),
        "long_term_edge": {**_rows(_obj({
            "dimension": _enum(LONG_TERM_DIMENSIONS, "长期竞争力维度，六个维度各一条"),
            "judgment": _enum(["加分", "减分", "中性", "未知"], "该维度对长期竞争力的影响；证据不足选未知"),
            "reason": _str("围绕同一主优势检验该维度，最多80字；未知时写缺口。说明竞争者为何难复制、资源归谁、价值能否由公司获取，不能把六个维度各写成一个独立护城河"),
            "evidence_ids": ids,
            "would_change_if": _str("会改变判断的可观察证据或失效信号；不编造监测阈值、持续年限或永久有效承诺"),
        }), "长期竞争力：行业、竞对、技术、商业模式、收入、成本各一条"), "minItems": 6, "maxItems": 6},
    }


def insight_errors(analyses: dict, evidence_ids: set[str]) -> list[str]:
    """Code checks that a schema cannot express: one row per dimension and known evidence only."""
    errors = []
    for name, analysis in analyses.items():
        rows = analysis.get("long_term_edge", [])
        if sorted(row.get("dimension") for row in rows) != sorted(LONG_TERM_DIMENSIONS):
            errors.append(f"{name} 的 long_term_edge 必须覆盖{'、'.join(LONG_TERM_DIMENSIONS)}各一次")
        cited = set(analysis.get("standout", {}).get("evidence_ids", []))
        cited |= set(analysis.get("cost_structure", {}).get("evidence_ids", []))
        cited |= {eid for key in ("competitor_diffs", "long_term_edge")
                  for row in analysis.get(key, []) for eid in row.get("evidence_ids", [])}
        unknown = sorted(cited - evidence_ids)
        if unknown:
            errors.append(f"{name} 的观点字段引用了不存在的证据编号：{'、'.join(unknown[:6])}")
    return errors


COMPARISON = _obj({
    "summary": _str("围绕最关键差异概括相对位置。可判断领先、相当、落后，也可说明证据尚不足"),
    "per_company": _rows(_obj({
        "company": _str("公司名"),
        "position": _str("在赛道中的位置"),
        "edge": _str("相对其他公司的主要优势"),
        "gap": _str("主要短板"),
    }), "每家公司一条"),
    "ranking_view": _str("是否同意代码给出的排序；不同意时说明理由和你的排序"),
    "top_pick": _str("单公司填目标公司，多公司填当前相对首选的研究对象。投资动作由总控决定"),
})

DEBATE_POSITION = _obj({
    "thesis": _str("核心论点，一段话"),
    "arguments": _list("按本阶段上限保留主要理由。事实前提引用本轮来源，未验证的前提写成假设"),
    "rebuttals": _list("针对具体论点指出推理缺口、替代解释或条件性反例，可由分析者提出。没有具体回应时可为空"),
    "would_change_mind": _str("哪项可观察的结果能区分解释，什么结果会改变本方立场"),
})

# A response follows an actual challenge. The opening statement need not rebut anyone.
BULL_RESPONSE = deepcopy(DEBATE_POSITION)
BULL_RESPONSE["properties"].update({
    "thesis": {**_str("就具体对象给出有范围的当前判断。证据不足时点明能确认什么、哪一步仍未知"), "pattern": r"\S"},
    "rebuttals": {**_list("回应对方的具体推理，给出维持或修订的依据。对方同意时说明共识成立的理由"),
                  "minItems": 1, "items": {"type": "string", "pattern": r"\S"}},
})


def challenger_response_schema(targets: dict[str, str]) -> dict:
    properties = deepcopy(DEBATE_POSITION["properties"])
    properties["target"] = _enum(list(targets), "从正方最新发言中选择本轮具体检验的一项")
    properties["rebuttals"] = {**_list("说明该项推导的具体缺口及其影响，或说明同意该项推导的理由"),
                               "minItems": 1, "items": {"type": "string", "pattern": r"\S"}}
    return _obj(properties)

CLAIM_VERDICTS = ["属实", "公司口径", "不实", "无法核实"]

FACT_CHECK = _obj({
    "claims": {**_rows(_obj({
        "target": _str("被核查材料中的原句或其中一段，原样复制；一条只核查一个事实，可只截取能被引用证明的那部分"),
        "source_agent": _str("主张所在材料的准确键名，必须与输入材料键名一致"),
        "original_source": _str("原材料中该主张已有的来源编号、网址或资料页码；未标注时写空字符串"),
        "evidence_ids": _list("与原主张相关的结构化证据 ID；未关联时为空"),
        "fact": _obj(FACT_FIELDS),
        "support": _enum(["完全支持", "部分支持", "不支持", "未找到原文"], "引用原文对 target 的支持程度"),
        "verdict": _enum(CLAIM_VERDICTS, "核查结论；只有完全支持时可判属实或公司口径"),
        "evidence": _str("判定理由，100 字以内"),
        "correction": _str("不实时给出正确说法，数字须出现在引用原文中；其他情况写空字符串"),
    }), "逐条核查的关键主张，优先核查对结论影响最大的数字和事实"), "maxItems": 6},
    "summary": _str("一段话：整体可信度如何，有哪些需要修正的地方"),
})


def fact_check_schema(material_ids: list[str], evidence_ids: list[str] | None = None,
                      min_claims: int | None = None) -> dict:
    """min_claims 由代码按材料覆盖要求算出；给出时放宽默认的六条上限，留出补充核查的余量。"""
    schema = deepcopy(FACT_CHECK)
    if min_claims is not None:
        schema["properties"]["claims"].update({
            "minItems": min_claims, "maxItems": max(6, min_claims + 3),
            "description": "逐条核查的关键主张：必查清单逐条覆盖，每份材料达到要求的条数，其余优先核查对结论影响最大的数字和事实"})
    properties = schema["properties"]["claims"]["items"]["properties"]
    properties["source_agent"]["enum"] = material_ids
    if evidence_ids is not None:
        if evidence_ids:
            properties["evidence_ids"]["items"]["enum"] = evidence_ids
        else:
            properties["evidence_ids"]["maxItems"] = 0
    return schema


def repair_check_schema(claim_ids: list[str], material_ids: list[str], evidence_ids: list[str]) -> dict:
    schema = fact_check_schema(material_ids, evidence_ids)
    schema["properties"]["repair_checks"] = _rows(_obj({
        "claim_id": _enum(claim_ids, "原问题主张的编号"),
        "status": _enum(["已修正", "已删除", "仍存在", "无法判断"], "当前材料中的处理状态"),
        "reason": _str("说明当前材料中对应内容及判定依据"),
    }), "对每条原问题主张及其依赖推论逐一复查")
    schema["properties"]["repair_checks"].update({"minItems": len(claim_ids), "maxItems": len(claim_ids)})
    schema["required"].append("repair_checks")
    return schema


def decision_check_schema(decision: dict, evidence_ids: list[str], claim_ids: list[str] | None = None) -> dict:
    schema = (repair_check_schema(claim_ids, ["chair"], evidence_ids) if claim_ids
              else fact_check_schema(["chair"], evidence_ids))
    targets = list(review_targets(decision))
    schema["properties"]["reasoning_checks"] = {
        **_rows(_obj({
            "target_id": _enum(targets, "必须逐一覆盖输入的关键判断编号"),
            "verdict": _enum(VERDICTS, "判断能否由证据及明确的推理步骤推出；条件性推断须已在正文写明条件"),
            "evidence_ids": {**_list("实际支持此判断的原文证据，或测算依据 U-1"),
                             "items": _enum([*evidence_ids, "U-1"], "当前证据编号")},
            "conditions": _list("条件性推断依赖的假设；必须已出现在被复核正文中，不得替正文补造"),
            "reason": _str("说明证据到结论的推理是否成立，检查主体、指标、期间、因果和缺失前提"),
            "correction": _str("依据不足或冲突时给出删改建议，不得编造替代事实"),
        }), "核心投资理由、估值判断及最终建议逐项复核，不占事实抽查的六条额度"),
        "minItems": len(targets), "maxItems": len(targets),
    }
    schema["required"].append("reasoning_checks")
    return schema

RESEARCH_PLAN = _obj({
    "core_question": _str("本次研究必须回答的一个核心决策问题，最多80字"),
    "working_hypothesis": _str("待证伪的核心思想；这是研究假设，不是预设投资结论，最多120字"),
    "priority_factors": {**_list("按重要性排列最多3个决策因素，每个最多60字"), "minItems": 1, "maxItems": 3},
    "disproof_tests": {**_list("最多3个可能推翻假设的证据检验；没有依据不编数字门槛"), "minItems": 1, "maxItems": 3},
    "research_questions": {**_list("给并行研究员的具体问题，最多6个"), "maxItems": 6},
    "report_focus": _str("报告应突出哪些决策信息，哪些背景可以压缩，最多120字"),
})


def sector_brief_schema(candidate_count: int) -> dict:
    sector = deepcopy(SECTOR_BRIEF)
    sector["properties"]["candidates"].update(
        minItems=candidate_count, maxItems=candidate_count,
        uniqueItems=True, items={"type": "string", "pattern": r"\S"},
    )
    return sector


def research_bundle_schema(companies: list[str]) -> dict:
    sheets = {}
    for name in companies:
        sheet = deepcopy(FACT_SHEET)
        sheet["properties"]["company_name"]["enum"] = [name]
        sheets[name] = sheet
    return _obj({"sector": sector_brief_schema(0), "companies": _obj(sheets)})


def research_followup_schema(companies: list[str], include_sector: bool) -> dict:
    schema = research_bundle_schema(companies)
    if not include_sector:
        schema["properties"].pop("sector")
        schema["required"].remove("sector")
    return schema


def _evidence_refs(ids: list[str]) -> dict:
    items = {"type": "string", **({"enum": ids} if ids else {})}
    return {"type": "array", "items": items, "maxItems": min(6, len(ids)), "uniqueItems": True}


def research_checkpoint_schema(question_ids: list[str], evidence_ids: list[str], companies: list[str]) -> dict:
    return _obj({
        "next_action": _enum(["finish", "research"], "成稿或补查；只在补查可能改变判断时选research"),
        "reason": _str("本轮研究决策的简短理由，说明补查价值或剩余局限"),
        "working_hypothesis": _str("结合当前证据与推理缺口修订的待证伪假设。未验证的解释保留条件，没有变化时留空"),
        "question_updates": {**_rows(_obj({
            "question_id": _enum(question_ids, "已有研究问题编号"),
            "status": _enum(["open", "answered", "blocked"], "尚未解决、已有证据回答、当前渠道无法解决"),
            "answer": _str("有依据的简短回答；未知留空，不把解释当成事实"),
            "alternative_explanations": {**_list("可由分析者提出的待验证解释，写明成立条件。没有则留空"), "maxItems": 3},
            "supporting_evidence_ids": _evidence_refs(evidence_ids),
            "contradicting_evidence_ids": _evidence_refs(evidence_ids),
            "decision_impact": _str("这个问题怎样影响投资判断，未知可留空"),
            "missing_information": _list("仍缺的信息及其限制，允许空数组"),
            "next_step": _str("下一步验证方法；不需继续或无可用渠道时可留空"),
        }), "只更新本轮涉及的问题，未更新的保留原状态"), "maxItems": len(question_ids)},
        "follow_up_tasks": {**_rows(_obj({
            "question_id": _enum([*question_ids, ""], "已有问题编号；新提出的问题写空字符串"),
            "question": _str("新问题的具体表述；沿用已有问题时可留空"),
            "scope": _enum(["sector", *companies], "补查范围：赛道或一家已存在的公司"),
            "method": _str("如何找原文或验证，说明如何区分不同解释；不能把搜索当作产品实测"),
            "expected_impact": _str("补齐后可能怎样改变判断；不编造量化门槛"),
        }), "最值得补查的任务；finish时为空，最多2项"), "maxItems": 2},
    })


def inquiry_audit_schema(question_ids: list[str], method_ids: list[str], evidence_ids: list[str]) -> dict:
    return _obj({
        "claims": {"type": "array", "items": {"type": "string"}, "maxItems": 0},
        "summary": _str("研究问题和方法复核摘要；原事实核查单独保留，不在此覆盖"),
        "question_checks": {**_rows(_obj({
            "question_id": _enum(question_ids, "待复核的问题编号"),
            "verdict": _enum(["supported", "conditional", "unresolved"], "回答有依据、只支持有条件判断、仍未解决"),
            "reason": _str("证据、逻辑或替代解释方面的理由"),
            "evidence_ids": _evidence_refs(evidence_ids),
        }), "逐项复核本轮问题"), "minItems": len(question_ids), "maxItems": len(question_ids)},
        "method_checks": {**_rows(_obj({
            "method_id": _enum(method_ids, "实际执行的方法候选编号"),
            "verdict": _enum(["not_demonstrated", "useful"], "本案例未证明有效，或实际执行且有助于回答问题"),
            "reason": _str("必须对照本轮读取记录与证据说明，不凭方法名称认定有效"),
            "evidence_ids": _evidence_refs(evidence_ids),
            "limitations": _str("方法适用条件和仍不能支持的判断；不可泛化为公司事实"),
        }), "逐项评价实际方法，未执行、无原文或未缩小问题时选not_demonstrated"),
            "minItems": len(method_ids), "maxItems": len(method_ids)},
    })


def analysis_bundle_schema(sector: Sector, sheets: dict[str, dict]) -> dict:
    return _obj({"analyses": _obj({name: analyst_schema(sector, [e["evidence_id"] for e in sheet["evidence"]])
                                   for name, sheet in sheets.items()})})


def comparison_position_schema(companies: list[str], *, responding: bool = False) -> dict:
    comparison = deepcopy(COMPARISON)
    comparison["properties"]["top_pick"]["enum"] = companies
    position = deepcopy(BULL_RESPONSE if responding else DEBATE_POSITION)
    if not responding:
        position["properties"]["thesis"] = {**_str("开场给出一项具体、可被反方检验的判断"), "pattern": r"\S"}
    return _obj({"comparison": comparison, "bull": position})


opening_case_schema = comparison_position_schema

SECTOR_VERDICTS = ["值得重点布局", "值得关注", "暂不布局"]
ACTIONS = ["立项", "继续尽调", "投资", "附条件投资", "不投"]
SCENARIOS = ["乐观", "基线", "悲观"]

DECISION = _obj({
    "verdict": _enum(SECTOR_VERDICTS, "赛道结论"),
    "action": _enum(ACTIONS, "本次建议的动作；排名第一不等于值得投资，没有合适标的时选不投"),
    "action_conditions": _list("推进本次建议前必须满足的条件，或出现后会改变建议的触发因素；附条件投资时必填"),
    "headline": _str("首页一句话点名公司，指出最值得持续投入的一项具体能力，它为哪类客户解决了什么难题。找不到时直接说明证据支持到哪里。交易动作、报价条件、内部过程放在各自字段。"),
    "technical_moat": _obj({
        "status": _enum(["有持续性证据", "形成中", "尚未证实"], "技术壁垒的证据强度。公开功能、单次成功或付费记录单独均不足以证明持久性。"),
        "mechanism": _str("写清公司控制的技术或诀窍、具体作用环节，以及什么能随使用或迭代积累。关键能力属于供应商或客户时明说。"),
        "evidence": _str("最能支撑或限制上述机制的已发生技术事实，附来源编号，写明公司披露、独立比较或未知。采购、续费与成本假设不能充作技术实证。"),
        "replication": _str("对照最强可行替代，指出对手已有能力、要额外重建什么及其已知难度。没有复刻成本或比较证据时如实说明。"),
        "would_fail_if": _str("哪项可观察的同任务技术对照结果能推翻难复制的判断。写清对手、控制条件和判别结果，标明尚待验证之处。"),
    }),
    "core_answer": _str("用最多120字解释该能力如何创造企业价值、在哪些条件下可能消失。与 headline、technical_moat 和首条投资逻辑对齐。"),
    "selection_view": _str("研究对象的选择依据：用户给了选择理由时，说明当前证据支持到什么程度；未给理由时直说“用户未说明选择理由”；"
                           "系统推荐时说明筛选依据。不事后编造理由，最多120字"),
    "question_answers": _rows(_obj({
        "question_id": _str("研究问题编号"),
        "answer": _str("对该问题的当前判断，最多100字；不是事实罗列，而是你的结论"),
        "evidence": _str("已查到的关键事实及本轮证据编号，含已发现的相反事实，最多100字"),
        "confidence": _enum(["高", "中", "低"], "对这个回答的把握"),
        "would_change_if": _str("什么新证据会改变这个回答"),
    }), "最影响投资判断的关键研究问题及回答，最多3个；只能引用当前研究状态中的问题编号"),
    "gap_impacts": {**_rows(_obj({
        "missing": _str("缺失的信息或数据"),
        "affects": _str("它影响了哪个判断（问题编号或结论）及影响方向"),
        "if_obtained": _str("拿到后结论可能怎样变化；不编造数值门槛"),
    }), "影响最大的信息缺口，最多3条，按影响排序"), "maxItems": 3},
    "recommended": _list("推荐重点跟进的 0-2 家公司名；没有合适标的时给空数组，此时 action 为不投"),
    "thesis": _str("Markdown：推荐或不推荐的理由，100-200 字；不重复 core_answer、关键问题回答和核心投资逻辑"),
    "sector_risks": _rows(_obj({
        "risk": _str("风险"),
        "impact": _str("可能影响"),
        "evidence_gap": _str("目前依据及信息缺口"),
        "response": _str("建议核实或应对"),
    }), "整个赛道共同面临的风险"),
    "top_pick": _str("重点分析对象。即使当前交易条件不成立，仍给出最值得深入分析的公司。"),
    "top_pick_thesis": _rows(_obj({
        "judgment": _str("一句话关键判断；首条是唯一主逻辑，回答最突出的一点及是否足以形成持续竞争优势；后续条目只补充该主逻辑的支持条件或重要限制"),
        "evidence": _str("最能支撑或限制此判断的一项已知事实及来源编号，必要时交代时点、样本或测算假设。"),
        "why_opportunity": _str("说明事实如何经用户收益、持续付费或成本优势转化为企业价值。首条区分产品亮点、正在形成的护城河与有持续性依据的护城河，交代难复制机制及成立条件；不能以有护城河代替估值判断"),
        "disproof": _str("出现什么可观察证据会推翻判断，尤其是竞品复制、上游替代、用户迁移、单位经济或价值归属变化；不承诺永久有效"),
    }), "一条主逻辑及最多两条支持条件或限制，不写三个无关投资故事；主逻辑允许为护城河尚未证实或没有明显优势"),
    "top_pick_risks": _list("重点分析对象的主要风险，每条附缓释办法"),
    "valuation_view": _str("本轮估值是否合理：引用代码计算的隐含倍数、可比融资案例和关键假设；缺少估值或收入数据时说明缺少什么，不编造可比倍数"),
    "use_of_funds_milestones": _str("本轮资金用途及预计可支撑到的经营里程碑和时间，引用 BP 或材料；未提供时写明缺少哪些材料"),
    "forecast_assessment": _str("未来 3–5 年预测所需的基础数据与假设；缺少基础营收、成本、现金或逐年假设时明确无法可靠预测，不编造逐年数字"),
    "exit_scenarios": _rows(_obj({
        "scenario": _enum(SCENARIOS, "情景"),
        "condition": _str("这个情景成立的条件"),
        "valuation_basis": _str("解释退出股权价值的依据与假设来源；由企业价值推算时写明净现金或净债务调整；无法估计时说明缺少哪些输入，不能把假设写成事实"),
        "exit_path": _str("退出方式：上市、并购或老股转让"),
        "exit_valuation_musd": {"type": ["number", "null"], "description": "退出时公司 100% 股权价值（不是企业价值 EV），单位百万美元；无法估计时写 null，0 仅表示该情景全损", "minimum": 0},
    }), "乐观、基线、悲观三种情景，退出股权价值应满足乐观 ≥ 基线 ≥ 悲观"),
    "dd_checklist": _list("投前必须完成的尽调事项，具体到找谁、核实什么"),
    "next_steps": _list("下一步行动"),
    "dissent": _str("对最终投资主张最有力的替代解释、适用条件及投资影响。直接写业务内容，保留假设语气，省略发言人和轮次。"),
})


def decision_schema(companies: list[str], question_ids: list[str] | None = None) -> dict:
    schema = deepcopy(DECISION)
    schema["properties"]["top_pick"]["enum"] = companies
    schema["properties"]["recommended"]["items"]["enum"] = companies
    schema["properties"]["recommended"].update({"minItems": 0, "maxItems": 2, "uniqueItems": True})
    schema["properties"]["top_pick_thesis"].update({"minItems": 1, "maxItems": 3})
    schema["properties"]["exit_scenarios"].update({"minItems": 3, "maxItems": 3})
    answers = schema["properties"]["question_answers"]
    if question_ids:
        answers["items"]["properties"]["question_id"] = _enum(list(question_ids), "当前研究状态中的问题编号")
        answers.update({"minItems": 1, "maxItems": min(3, len(question_ids))})
    else:  # 旧流程没有问题状态：不允许凭空编号
        answers["maxItems"] = 0
    return schema


def memo_review_schema(sector: Sector, companies: list[str], top_pick: str, debate_count: int, evidence_ids: list[str],
                       question_ids: list[str] | None = None, *, single_company: bool | None = None) -> dict:
    decision = decision_schema(companies, question_ids)["properties"]
    decision = {k: v for k, v in decision.items() if k != "exit_scenarios"}
    decision["top_pick"] = _enum([top_pick], "已锁定的重点分析对象")
    if single_company is None:
        single_company = len(companies) == 1
    if single_company:
        decision["report_subtitle"] = _str("简短的名词短语，概括本次研究的产品、能力与价值问题。主标题由程序从本次公司名生成，副标题省略公司名、报告类型和评级。")
        decision["section_openings"] = _obj({
            "customer": _str("公司服务哪类客户，其具体任务为何需要首页所述能力。"),
            "technology": _str("承接上述客户任务，写清公司控制的技术环节、可行替代和已知证据边界。"),
            "economics": _str("承接技术判断，解释客户收益如何经定价、使用或续费转为公司收入，扣除交付成本后如何影响利润。标明已观察到哪一步。"),
            "valuation": _str("承接经营质量，指出哪些收入、成本或交易输入可用于基准情景，哪些尚待验证的收益只在假设情景中讨论。"),
            "risks": _str("承接估值判断，点出最可能改变技术、盈利或估值结论的未知，以及能区分解释的查证方法。"),
        })
    # 报告只展示每家公司的最突出一点；完整观点字段只对重点分析对象展开，编辑也只改这些。
    narrative = {k: v for k, v in analyst_schema(sector)["properties"].items()
                 if k != "scores" and k not in INSIGHT_KEYS[1:]}
    cited_ids = {"type": "array", "items": _enum(evidence_ids, "当前证据编号")} if evidence_ids else {
        "type": "array", "items": {"type": "string"}, "maxItems": 0}
    narrative["score_reasons"] = _obj({d.key: _obj({
        "reason": _str("使用当前证据解释既定评分，不重新评分"), "evidence_ids": cited_ids,
    }) for d in sector.dimensions})
    focus = {**narrative, **{k: v for k, v in insight_properties().items() if k in INSIGHT_KEYS[1:]}}
    return _obj({
        "decision": _obj(decision),
        "sector": _obj({k: SECTOR_BRIEF["properties"][k] for k in
                        ("summary", "market_size", "why_now", "landscape", "benchmarks")}),
        "companies": _obj({name: _obj(focus if name == top_pick else narrative) for name in companies}),
        "comparison": COMPARISON,
        "debate": {**_rows(DEBATE_POSITION, "按原顺序保留多空观点，修正事实和口径；分歧写明是假设或意见"),
                   "minItems": debate_count, "maxItems": debate_count},
    })


# 语义上可以为空的结构化字段；模型漏填时补成“未使用”，避免整条提交被退回
_OPTIONAL_DEFAULTS = {
    "date_quote": {"source": "", "from": 0, "to": 0},
    "date": {"text": "", "basis": "未给出"},
}


def _closest_enum(options: list, value):
    """枚举值只差大小写、空格或标点时视为同一个；分类字段写了清单外的值归入“其他”。"""
    if not isinstance(value, str) or value in options:
        return value
    key = loose_text(value)
    for option in options:
        if isinstance(option, str) and loose_text(option) == key:
            return option
    return "其他" if "其他" in options else value


def normalize(schema: dict, value):
    """修正常见类型偏差；证据数组必须由研究员显式提交，允许为空。"""
    kind = schema.get("type")
    if "enum" in schema:
        return _closest_enum(schema["enum"], value)
    if kind == "string" and isinstance(value, dict):  # 真实反例：entities 写成 {"name": "腾讯"}
        for key in ("name", "text", "value", "title"):
            if isinstance(value.get(key), str):
                return value[key]
    if kind == "object" and isinstance(value, str):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", value.strip())
        if not raw.startswith("{"):
            return value
        try:  # 部分模型把数组里的对象写成 JSON 字符串
            value = json.loads(raw)
        except ValueError:
            return value
    if kind == "object" and isinstance(value, dict):
        out = dict(value)
        for key, sub in schema.get("properties", {}).items():
            if key in out:
                out[key] = normalize(sub, out[key])
                if key == "candidates" and isinstance(out[key], list):
                    names = dict.fromkeys(name.strip() for name in out[key] if isinstance(name, str) and name.strip())
                    out[key] = list(names)[:sub.get("maxItems")]
            elif sub.get("type") == "array" and key != "evidence":
                out[key] = []
            elif key in _OPTIONAL_DEFAULTS:
                out[key] = deepcopy(_OPTIONAL_DEFAULTS[key])
        return out
    if kind == "array":
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            return [normalize(schema.get("items", {}), v) for v in value]
    if kind == "integer" and not isinstance(value, bool):
        try:
            return int(round(float(value)))
        except (TypeError, ValueError):
            return value
    if kind == "string" and isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    return value


def validate(schema: dict, value, path: str = "结果") -> list[str]:
    """返回错误列表，空列表表示通过。只覆盖本项目 schema 用到的关键字。"""
    if "enum" in schema and value not in schema["enum"]:
        return [f"{path} 必须是 {schema['enum']} 之一"]
    kind = schema.get("type")
    if isinstance(kind, list):
        if value is None and "null" in kind:
            return []
        return validate({**schema, "type": next(t for t in kind if t != "null")}, value, path)
    if kind == "object":
        if not isinstance(value, dict):
            return [f"{path} 应为对象"]
        errors = [f"缺少字段 {k}" for k in schema.get("required", []) if k not in value]
        for key, sub in schema.get("properties", {}).items():
            if key in value:
                errors += validate(sub, value[key], key)
        return errors
    if kind == "array":
        if not isinstance(value, list):
            return [f"{path} 应为数组"]
        errors = [e for i, v in enumerate(value) for e in validate(schema.get("items", {}), v, f"{path}[{i}]")]
        if len(value) < schema.get("minItems", 0):
            errors.append(f"{path} 数量不足")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errors.append(f"{path} 数量过多")
        if schema.get("uniqueItems") and any(v in value[:i] for i, v in enumerate(value)):
            errors.append(f"{path} 包含重复值")
        return errors
    if kind == "string" and not isinstance(value, str):
        return [f"{path} 应为字符串"]
    if kind == "string" and "pattern" in schema and not re.search(schema["pattern"], value):
        return [f"{path} 需要填写内容"]
    if kind == "integer" and (isinstance(value, bool) or not isinstance(value, int)):
        return [f"{path} 应为整数"]
    if kind == "number" and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)):
        return [f"{path} 应为有限数值"]
    if kind in ("integer", "number"):
        if "minimum" in schema and value < schema["minimum"]:
            return [f"{path} 不能小于 {schema['minimum']}"]
        if "maximum" in schema and value > schema["maximum"]:
            return [f"{path} 不能大于 {schema['maximum']}"]
    return []


def validate_submission(role: str, schema: dict, value) -> list[str]:
    errors = validate(schema, value)
    if errors:
        return errors
    if role == "fact_checker" and "reasoning_checks" in schema["properties"]:
        expected = schema["properties"]["reasoning_checks"]["items"]["properties"]["target_id"]["enum"]
        if sorted(row["target_id"] for row in value["reasoning_checks"]) != sorted(expected):
            errors.append("reasoning_checks 必须逐一覆盖每个关键判断编号，不能重复或遗漏")
        for row in value["reasoning_checks"]:
            if row["verdict"] in ("有依据", "条件性推断") and not row["evidence_ids"]:
                errors.append(f"{row['target_id']} 通过推论复核必须列出依据编号")
            if row["verdict"] == "条件性推断" and not any(c.strip() for c in row["conditions"]):
                errors.append(f"{row['target_id']} 的条件性推断必须指出正文已明确的条件")
        return errors
    if role != "chair":
        return errors
    if "decision" in schema["properties"]:
        return validate_submission(role, schema["properties"]["decision"], value["decision"])
    if "recommended" not in schema["properties"]:
        return errors  # 主 Agent 的研究议程还不是投资决策
    if value["recommended"] and value["top_pick"] not in value["recommended"]:
        errors.append("首选公司必须在推荐名单中")
    if not value["recommended"] and value["action"] != "不投":
        errors.append("没有推荐公司时，本次建议只能是不投")
    if value["action"] == "附条件投资" and not any(c.strip() for c in value["action_conditions"]):
        errors.append("附条件投资必须列出投资条件")
    answered = [row["question_id"] for row in value.get("question_answers", [])]
    if len(answered) != len(set(answered)):
        errors.append("question_answers 每个问题编号只能回答一次")
    if "exit_scenarios" not in schema["properties"]:
        return errors
    if sorted(s["scenario"] for s in value["exit_scenarios"]) != sorted(SCENARIOS):
        errors.append("退出情景必须各包含一次乐观、基线和悲观")
        return errors
    if any(s["exit_valuation_musd"] is not None and not s["valuation_basis"].strip() for s in value["exit_scenarios"]):
        errors.append("有退出估值的情景必须说明估值依据与假设")
    values = {s["scenario"]: s["exit_valuation_musd"] for s in value["exit_scenarios"]}
    known = [(name, values[name]) for name in SCENARIOS if values[name] is not None]
    if any(high < low for (_, high), (_, low) in zip(known, known[1:])):
        errors.append("退出股权价值必须满足乐观 ≥ 基线 ≥ 悲观")
    return errors
