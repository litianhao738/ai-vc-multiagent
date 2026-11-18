"""用固定虚构资料检查真实模型是否识别“引文正确、推论错误”，会产生 API 费用。"""
from __future__ import annotations

import json
import random
import time
from dataclasses import replace
from datetime import date, datetime

from dotenv import load_dotenv

from vcagents.agent import Agent
from vcagents.config import ROOT, Settings
from vcagents.llm import LLMRouter, _fake
from vcagents.pipeline import AnalysisResult, SectorPipeline
from vcagents.reasoning import review_targets
from vcagents.roles import FACT_CHECKER, system_prompt
from vcagents.schemas import decision_check_schema, decision_schema
from vcagents.sectors import load_sector


def main() -> int:
    load_dotenv(ROOT / ".env")
    settings = Settings.load()
    pipeline = SectorPipeline(Settings(mock=True))
    pipeline.sector = load_sector("aigc-content")
    decision = _fake(decision_schema(["Alpha"]), random.Random(1))
    decision["top_pick_thesis"] = [
        {"judgment": "Alpha融资额高，因此资本效率领先。", "evidence": "融资2亿美元[E-C1-1]",
         "why_opportunity": "资本效率形成优势。", "disproof": "现金消耗增加。"},
        {"judgment": "客户留存率达到300%，说明客户不会流失。", "evidence": "公司披露NRR为300%[E-C1-2]",
         "why_opportunity": "客户稳定。", "disproof": "客户留存下降。"},
        {"judgment": "据公司披露，当前月收入年化为1亿美元；这不等于已确认的订阅ARR。",
         "evidence": "公司披露的当月收入年化口径[E-C1-3]", "why_opportunity": "提供经营规模线索，仍需核对收入构成。",
         "disproof": "收入记录与公司披露不符。"}]
    decision.update(action="投资", headline="本次投资能够确保回收本金。", action_conditions=[],
                    valuation_view="假设价格下调且复核后的基线MOIC达到3倍，才满足本次假设门槛；当前0.56倍不满足该条件[U-1]。")
    result = AnalysisResult("aigc-content", "虚构资料实验", decision=decision,
                            returns=[{"scenario": "基线", "multiple": .56}],
                            deal_metrics={"Alpha": {"post_money_musd": 5400}},
                            return_summary={"scope": "3倍为本实验明确设定的假设门槛"})
    evidence = [
        {"evidence_id": "E-C1-1", "source_type": "公司口径", "excerpt": "Alpha宣布融资2亿美元；未披露现金消耗或资本回报率。"},
        {"evidence_id": "E-C1-2", "source_type": "公司口径", "excerpt": "Alpha披露净收入留存率NRR为300%；未披露客户数量留存率。"},
        {"evidence_id": "E-C1-3", "source_type": "公司口径", "excerpt": "Alpha披露当月收入乘12为1亿美元，未披露其中订阅和一次性收入的占比。"}]
    task = ("固定虚构资料实验，仅用给定摘录，不联网，不补充事实。"
            "claims 提交空数组，本次只测试 reasoning_checks。\n"
            + json.dumps({"evidence": evidence}, ensure_ascii=False) + pipeline._reasoning_context(result))
    schema = decision_check_schema(decision, [e["evidence_id"] for e in evidence])
    schema["properties"]["claims"]["maxItems"] = 0
    llm = LLMRouter(settings).for_role("fact_checker")
    llm.client = llm.client.with_options(max_retries=0, timeout=60)
    llm.max_tokens = 6000
    agent = Agent(replace(FACT_CHECKER, tools=()), llm, None,
                  lambda event: print(event.message, flush=True),
                  system=system_prompt(FACT_CHECKER, pipeline.sector, settings.fund_profile, date.today().isoformat()),
                  schema=schema, max_steps=2)
    started = time.monotonic()
    expected = {"T1": ["依据不足"], "T2": ["依据不足", "与证据冲突"], "T3": ["有依据"],
                "V1": ["条件性推断", "有依据"], "A1": ["依据不足", "与证据冲突"]}
    try:
        output, error = agent.run(task), ""
        verdicts = {row["target_id"]: row["verdict"] for row in output["reasoning_checks"]}
        cases = {key: verdicts.get(key) in values for key, values in expected.items()}
    except Exception as exc:
        output, cases, error = {}, {}, f"{type(exc).__name__}: {exc}"
    passed = bool(cases) and all(cases.values())
    artifact = {"model": settings.role_models["fact_checker"], "task": task, "targets": review_targets(decision),
                "result": output, "expected": expected, "cases": cases, "passed": passed, "error": error,
                "cost_usd": agent.cost_usd, "seconds": round(time.monotonic() - started, 2),
                "scope": "固定虚构摘录的五条推论实验，不能代表真实投资报告准确率。"}
    path = ROOT / "outputs" / "evaluation" / f"reasoning_{datetime.now():%Y%m%d_%H%M%S}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"path": str(path), "passed": passed, "cases": cases,
                      "error": error, "cost_usd": agent.cost_usd}, ensure_ascii=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
