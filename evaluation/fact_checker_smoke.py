"""使用 15 条虚构 BP 主张检查真实核查模型的基本分类能力。"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

from vcagents.agent import Agent
from vcagents.config import Settings
from vcagents.llm import LLMRouter
from vcagents.roles import FACT_CHECKER, system_prompt
from vcagents.schemas import fact_check_schema
from vcagents.sectors import load_sector
from vcagents.tools import DocumentStore, Toolbox


def benchmark_cases() -> tuple[DocumentStore, list[dict], dict[str, dict]]:
    docs, claims, expected = DocumentStore(), [], {}
    for i, name in enumerate(("Alpha", "Bravo", "Charlie")):
        facts = [("founded", 2020 + i), ("revenue_million", 10 + i), ("employees", 20 + i),
                 ("customers", 30 + i), ("funding_million", 40 + i)]
        docs.add_text(name + "/BP.txt", "Fictional company " + name + " BP. Self-reported facts: "
                      + ", ".join(f"{key}={value}" for key, value in facts) + ".")
        for j, (key, value) in enumerate(facts, 1):
            claim_id = chr(65 + i) + str(j)
            expected[claim_id] = {"claim": f"{claim_id}: {name} company reports {key}={value}",
                                  "source_agent": "company_researcher:" + name}
            claims.append({**expected[claim_id], "source_ref": name + "/BP.txt page 1"})
    return docs, claims, expected


def summarize(claims: list[dict], expected: dict[str, dict]) -> dict:
    ids = [match.group() for claim in claims if (match := re.search(r"\b[A-C][1-5]\b", claim["claim"]))]
    correct_source = sum(claim["source_agent"] == expected.get(match.group(), {}).get("source_agent")
                         for claim in claims if (match := re.search(r"\b[A-C][1-5]\b", claim["claim"])))
    correct_claim = sum(claim["claim"].strip() == expected.get(match.group(), {}).get("claim")
                        for claim in claims if (match := re.search(r"\b[A-C][1-5]\b", claim["claim"])))
    verdicts = dict(Counter(claim["verdict"] for claim in claims))
    passed = (len(claims) == len(expected) and set(ids) == set(expected) and len(set(ids)) == len(ids)
              and correct_source == len(expected) and correct_claim == len(expected)
              and verdicts == {"公司口径": len(expected)})
    return {"passed": passed, "claims": len(claims), "unique_ids": len(set(ids)),
            "correct_sources": correct_source, "correct_claims": correct_claim, "verdicts": verdicts}


def main() -> int:
    parser = argparse.ArgumentParser(description="运行 15 条虚构 BP 的真实模型核查测试，会产生 API 费用")
    parser.add_argument("--models", help="模型配置文件")
    parser.add_argument("--out", help="结果 JSON 路径；默认保存到 outputs/evaluation")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    settings = Settings.load(args.models)
    docs, claims, expected = benchmark_cases()
    task = ("Fictional BP-only benchmark. Check all 15 short claims against uploaded documents; do not use web. "
            "The only source is each company own BP, so distinguish company-reported from independent evidence. "
            "Copy each original claim text including its ID A1 through C5 verbatim, and use the exact source_agent. "
            "Documents: Alpha/BP.txt, Bravo/BP.txt, Charlie/BP.txt. Claims: " + json.dumps(claims))
    toolbox = Toolbox(docs, None)
    agent = Agent(
        FACT_CHECKER, LLMRouter(settings).for_role("fact_checker"), toolbox,
        lambda event: print(f"{event.kind}: {event.message}", flush=True),
        system=system_prompt(FACT_CHECKER, load_sector("embodied-ai"), settings.fund_profile, date.today().isoformat()),
        schema=fact_check_schema(list(dict.fromkeys(item["source_agent"] for item in expected.values())), []), max_steps=8,
    )
    started = time.monotonic()
    try:
        result = agent.run(task)
        result = toolbox.verify_fact_check(result)
        summary = summarize(result["claims"], expected)
        error = ""
    except Exception as exc:
        result, summary, error = {}, {"passed": False}, f"{type(exc).__name__}: {exc}"
    output = {
        "model": settings.role_models["fact_checker"], "summary": summary, "error": error,
        "seconds": round(time.monotonic() - started, 2), "input_tokens": agent.input_tokens,
        "output_tokens": agent.output_tokens, "cost_usd": agent.cost_usd, "result": result,
        "scope": "虚构、同质、仅公司 BP、提示明确；不代表真实资料的准确率或联网核查效果。",
    }
    path = Path(args.out) if args.out else ROOT / "outputs" / "evaluation" / f"factcheck_{datetime.now():%Y%m%d_%H%M%S}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summary, "seconds": output["seconds"], "cost_usd": output["cost_usd"],
                      "output": str(path)}, ensure_ascii=False))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
