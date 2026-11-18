"""比较现有公司分析提示与单位经济 Skill 在固定虚构数据上的表现。"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import date, datetime
from pathlib import Path

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vcagents.agent import Agent
from vcagents.config import Settings
from vcagents.llm import LLMRouter
from vcagents.roles import COMPANY_ANALYST, system_prompt
from vcagents.schemas import _enum, _int, _obj, _str
from vcagents.sectors import load_sector


SKILL_SHA = "60565064557ca77b12f62aa7877f365d111c129d"
SKILL_URL = ("https://raw.githubusercontent.com/anthropics/financial-services/" + SKILL_SHA
             + "/plugins/vertical-plugins/private-equity/skills/unit-economics/SKILL.md")
SCHEMA = _obj({
    "ending_arr_thousand": _int("甲公司期末 ARR，单位千美元"),
    "gross_retention_pct": _int("甲公司收入毛留存率，整数百分比"),
    "net_retention_pct": _int("甲公司收入净留存率，整数百分比"),
    "cac_status": _enum(["可计算", "无法计算"], "甲公司获客成本能否根据材料计算"),
    "hardware_arr_status": _enum(["适用", "不适用", "无法判断"], "乙公司是否应使用订阅 ARR 指标"),
    "reason": _str("说明计算、口径和缺失数据"),
})
EXPECTED = {"ending_arr_thousand": 1500, "gross_retention_pct": 80, "net_retention_pct": 110,
            "cac_status": "无法计算", "hardware_arr_status": "不适用"}
TASK = ("只根据下列虚构数据填写结构化结果，不联网，不假设缺失数据。"
        "甲是订阅软件公司：期初 ARR 1000 千美元，本期新增 ARR 400 千美元，存量客户扩张 ARR 300 千美元，"
        "收缩 ARR 100 千美元，流失 ARR 100 千美元；没有获客费用或新客户数。"
        "计算期末 ARR、毛收入留存率、净收入留存率，并判断能否计算 CAC。"
        "乙仅一次性销售机器人硬件，没有订阅或经常性收入；判断订阅 ARR 是否适用。")


def run_variant(name: str, system: str, router: LLMRouter) -> dict:
    agent = Agent(COMPANY_ANALYST, router.for_role("company_researcher"), None,
                  lambda e: print(f"{name}: {e.message}", flush=True), system=system,
                  schema=SCHEMA, max_steps=6)
    started = time.monotonic()
    try:
        result = agent.run(TASK)
        fields = {key: {"expected": value, "actual": result[key], "correct": result[key] == value}
                  for key, value in EXPECTED.items()}
        summary, error = {"correct": sum(v["correct"] for v in fields.values()), "total": len(fields), "fields": fields}, ""
    except Exception as exc:
        result, summary, error = {}, {}, f"{type(exc).__name__}: {exc}"
    return {"name": name, "seconds": round(time.monotonic() - started, 2), "cost_usd": agent.cost_usd,
            "input_tokens": agent.input_tokens, "output_tokens": agent.output_tokens,
            "format_failures": agent.format_failures, "summary": summary, "error": error, "result": result}


def main() -> int:
    parser = argparse.ArgumentParser(description="单位经济 Skill 对照实验；真实 API 调用会产生费用")
    parser.add_argument("--out", help="结果 JSON；默认保存到 outputs/evaluation")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    response = httpx.get(SKILL_URL, timeout=30, follow_redirects=True)
    response.raise_for_status()
    skill = response.text
    settings = Settings.load()
    system = system_prompt(COMPANY_ANALYST, load_sector("aigc-content"), settings.fund_profile, date.today().isoformat())
    candidate = (system + "\n\n以下单位经济工作流仅供分析方法参考；本实验的材料、工具和输出规则优先，"
                 "不使用资料以外的基准或假设：\n<参考资料>\n" + skill + "\n</参考资料>")
    router = LLMRouter(settings)
    runs = [run_variant(name, prompt, router) for name, prompt in
            (("baseline_1", system), ("skill_1", candidate), ("skill_2", candidate), ("baseline_2", system))]
    output = {"skill_url": SKILL_URL, "skill_sha256": hashlib.sha256(skill.encode("utf-8")).hexdigest(),
              "model": settings.role_models["company_researcher"], "task": TASK, "runs": runs,
              "scope": "单组虚构数据，五个字段，两组各两次；只测指定计算与适用性，不代表完整投资分析表现。"}
    path = Path(args.out) if args.out else ROOT / "outputs" / "evaluation" / f"unit_economics_ab_{datetime.now():%Y%m%d_%H%M%S}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(path), "runs": [{k: run[k] for k in ("name", "seconds", "cost_usd", "summary", "error")}
                                                   for run in runs]}, ensure_ascii=False))
    return 0 if all(not run["error"] for run in runs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
