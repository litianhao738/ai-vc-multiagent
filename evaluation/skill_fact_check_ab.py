"""用固定来源摘要对照现有核查提示和外部 Skill；真实模型调用会产生费用。"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
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
from vcagents.roles import FACT_CHECKER, system_prompt
from vcagents.schemas import fact_check_schema
from vcagents.sectors import load_sector


SKILL_SHA = "5dedab1cd3ede800b85cb6cd1943865ef96f5fd6"
SKILL_URL = ("https://raw.githubusercontent.com/jamditis/claude-skills-journalism/" + SKILL_SHA
             + "/journalism-core/skills/fact-check-workflow/SKILL.md")

SOURCES = {
    "SSE": {"type": "交易所公告", "url": "https://big5.sse.com.cn/site/cht/www.sse.com.cn/disclosure/announcement/listing/ipo/c/c_20260818_10829204.shtml",
            "text": "上交所公告称，宇树科技 A 股于 2026 年 8 月 19 日起在科创板上市交易。"},
    "Figure": {"type": "公司公告", "url": "https://www.figure.ai/news/series-c",
               "text": "Figure 2025 年 9 月 16 日公告：C 轮承诺资本超过 10 亿美元，投后估值 390 亿美元。"},
    "Apptronik": {"type": "公司公告", "url": "https://apptronik.com/news-collection/apptronik-closes-over-935-million-series-a",
                  "text": "Apptronik 2026 年 2 月 11 日宣布 5.2 亿美元 A-X 延长轮，A 轮总额超过 9.35 亿美元。"},
    "Agility": {"type": "公司新闻稿，收录于 SEC 文件", "url": "https://www.sec.gov/Archives/edgar/data/2074973/000121390026112499/ea0309659-425_churchill11.htm",
                "text": "Agility 新闻稿披露超过 3 亿美元多年订单的潜在价值，受合同条件约束，不代表当期收入。"},
}

# 标签针对上面的固定来源摘要，而不是对来源以外所有事实的最终判断。
CASES = [
    ("U1", "宇树科技 A 股于 2026 年 8 月 19 日起在上交所科创板上市交易。", "属实"),
    ("U2", "截至 2026 年 10 月 29 日，宇树科技尚未上市。", "不实"),
    ("F1", "Figure 表示其 C 轮承诺资本超过 10 亿美元。", "公司口径"),
    ("F2", "Figure 表示其 C 轮承诺资本恰好为 10 亿美元。", "不实"),
    ("F3", "Figure 公告披露的 C 轮投后估值为 390 亿美元。", "公司口径"),
    ("F4", "Figure 的 C 轮资金已经全部到账。", "无法核实"),
    ("A1", "Apptronik 表示其 A-X 延长轮融资为 5.2 亿美元。", "公司口径"),
    ("A2", "Apptronik 表示其 A 轮累计超过 9.35 亿美元。", "公司口径"),
    ("A3", "Apptronik 说其 A-X 延长轮融资为 5.2 亿人民币。", "不实"),
    ("G1", "Agility 表示其 Digit 5 多年订单潜在价值超过 3 亿美元。", "无法核实"),
    ("G2", "Agility 披露的超过 3 亿美元多年订单是当期收入。", "不实"),
    ("G3", "Agility 上述订单已经全部交付。", "无法核实"),
]


def summarize(claims: list[dict]) -> dict:
    expected = {key: (f"{key}: {claim}", verdict) for key, claim, verdict in CASES}
    by_id: dict[str, list[dict]] = {}
    for claim in claims:
        match = re.match(r"^([UFAG]\d+):", claim["claim"].strip())
        if match:
            by_id.setdefault(match.group(1), []).append(claim)
    rows = []
    for key, (text, verdict) in expected.items():
        found = by_id.get(key, [])
        item = found[0] if len(found) == 1 else {}
        exact = item.get("claim", "").strip() == text and item.get("source_agent") == "research"
        rows.append({"id": key, "expected": verdict, "actual": item.get("verdict"),
                     "exact_claim_and_source": exact, "correct": exact and item.get("verdict") == verdict})
    return {"correct": sum(row["correct"] for row in rows), "total": len(rows),
            "submitted": len(claims), "complete": len(claims) == len(expected) and all(row["exact_claim_and_source"] for row in rows),
            "rows": rows}


def run_variant(name: str, system: str, task: str, router: LLMRouter) -> dict:
    agent = Agent(FACT_CHECKER, router.for_role("fact_checker"), None,
                  lambda e: print(f"{name}: {e.message}", flush=True), system=system,
                  schema=fact_check_schema(["research"], []), max_steps=6)
    started = time.monotonic()
    try:
        result = agent.run(task)
        summary, error = summarize(result["claims"]), ""
    except Exception as exc:
        result, summary, error = {}, {}, f"{type(exc).__name__}: {exc}"
    return {"name": name, "seconds": round(time.monotonic() - started, 2), "cost_usd": agent.cost_usd,
            "input_tokens": agent.input_tokens, "output_tokens": agent.output_tokens,
            "format_failures": agent.format_failures, "summary": summary, "error": error, "result": result}


def main() -> int:
    parser = argparse.ArgumentParser(description="事实核查 Skill 对照实验；真实 API 调用会产生费用")
    parser.add_argument("--out", help="结果 JSON；默认保存到 outputs/evaluation")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    response = httpx.get(SKILL_URL, timeout=30, follow_redirects=True)
    response.raise_for_status()
    skill = response.text
    settings = Settings.load()
    system = system_prompt(FACT_CHECKER, load_sector("embodied-ai"), settings.fund_profile, date.today().isoformat())
    task = ("固定资料核查实验：只能使用给出的资料摘要，不联网，不联系任何人，不引入资料以外事实。"
            "资料摘要对应的原始链接仅供人工复查。逐条核查全部主张，原样复制编号和文字，"
            "source_agent 必须是 research。根据资料支持、矛盾或不足来判断，不能只因材料没有提及就判不实。\n"
            + json.dumps({"资料": SOURCES, "主张": [f"{key}: {claim}" for key, claim, _ in CASES]}, ensure_ascii=False))
    candidate = (system + "\n\n以下外部核查工作流仅供分析方法参考；本实验的范围、工具和输出规则优先，"
                 "不执行联系他人或发布等操作：\n<参考资料>\n" + skill + "\n</参考资料>")
    router = LLMRouter(settings)
    runs = [run_variant(name, prompt, task, router) for name, prompt in
            (("baseline_1", system), ("skill_1", candidate), ("skill_2", candidate), ("baseline_2", system))]
    output = {"skill_url": SKILL_URL, "skill_sha256": hashlib.sha256(skill.encode("utf-8")).hexdigest(),
              "model": settings.role_models["fact_checker"], "sources": SOURCES,
              "cases": [{"id": k, "claim": c, "expected": e} for k, c, e in CASES], "runs": runs,
              "scope": "12 条固定来源摘要，禁联网；两组各两次，不能推断完整真实报告的准确率或评估完整 Skill 工作流。"}
    path = Path(args.out) if args.out else ROOT / "outputs" / "evaluation" / f"skill_ab_{datetime.now():%Y%m%d_%H%M%S}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(path), "runs": [{k: run[k] for k in ("name", "seconds", "cost_usd", "summary", "error")}
                                                   for run in runs]}, ensure_ascii=False))
    return 0 if all(not run["error"] for run in runs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
