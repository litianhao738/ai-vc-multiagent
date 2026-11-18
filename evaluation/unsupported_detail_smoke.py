"""针对真实评测中出现的部分证据误判，检查核查员与提交校验。"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, datetime
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.skill_fact_check_ab import CASES, SOURCES
from vcagents.agent import Agent
from vcagents.config import Settings
from vcagents.llm import LLMRouter
from vcagents.roles import FACT_CHECKER, system_prompt
from vcagents.schemas import _list, fact_check_schema
from vcagents.sectors import load_sector


class CandidateAgent(Agent):
    def _submission_errors(self, schema: dict, data: dict) -> list[str]:
        errors = super()._submission_errors(schema, data)
        if not errors:
            for i, claim in enumerate(data["claims"], 1):
                if claim["unsupported_parts"] and claim["verdict"] in ("属实", "公司口径"):
                    errors.append(f"第 {i} 条主张仍有未支持的细节，不能将整条判为{claim['verdict']}")
        return errors


def passed_case(claims: list[dict], claim: str) -> bool:
    return (len(claims) == 1 and claims[0]["claim"].strip() == claim and claims[0]["verdict"] == "无法核实"
            and any("digit5" in part.lower().replace(" ", "") for part in claims[0]["unsupported_parts"]))


def main() -> int:
    parser = argparse.ArgumentParser(description="验证未支持细节；真实模型调用会产生费用")
    parser.add_argument("--out", help="结果 JSON；默认保存到 outputs/evaluation")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    settings = Settings.load()
    claim = next(f"{key}: {text}" for key, text, _ in CASES if key == "G1")
    task = ("固定资料核查实验：只能使用下面的资料摘要，不联网，不引入摘要之外的事实；"
            "请原样复制主张，source_agent 写 research。\n"
            + json.dumps({"资料": {"Agility": SOURCES["Agility"]}, "主张": [claim]}, ensure_ascii=False))
    schema = fact_check_schema(["research"], [])
    item = schema["properties"]["claims"]["items"]
    item["properties"]["unsupported_parts"] = _list("主张中尚无证据支持的具体细节；全部得到支持时为空数组")
    item["required"].append("unsupported_parts")
    system = (system_prompt(FACT_CHECKER, load_sector("embodied-ai"), settings.fund_profile, date.today().isoformat())
              + "\n将主张中未获支持的具体细节写入 unsupported_parts；存在缺口时不能将整条判为属实或公司口径。")
    agent = CandidateAgent(FACT_CHECKER, LLMRouter(settings).for_role("fact_checker"), None,
                  lambda event: print(event.message, flush=True),
                  system=system, schema=schema, max_steps=6)
    started = time.monotonic()
    try:
        result = agent.run(task)
        claims = result["claims"]
        passed = passed_case(claims, claim)
        error = ""
    except Exception as exc:
        result, passed, error = {}, False, f"{type(exc).__name__}: {exc}"
    output = {"model": settings.role_models["fact_checker"], "task": task, "result": result, "passed": passed,
              "error": error, "seconds": round(time.monotonic() - started, 2),
              "cost_usd": agent.cost_usd, "format_failures": agent.format_failures,
              "scope": "单条固定摘要，不联网，只验证缺失细节的识别与提交；不代表真实核查准确率。"}
    path = Path(args.out) if args.out else ROOT / "outputs" / "evaluation" / f"unsupported_detail_{datetime.now():%Y%m%d_%H%M%S}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(path), "passed": passed, "seconds": output["seconds"],
                      "cost_usd": output["cost_usd"], "error": error}, ensure_ascii=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
