"""比较现有主席提示与 IC memo Skill 是否保留缺失交易数据的边界。"""
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
from vcagents.roles import CHAIR, Role, system_prompt
from vcagents.schemas import _enum, _list, _obj, _str
from vcagents.sectors import load_sector


SKILL_SHA = "60565064557ca77b12f62aa7877f365d111c129d"
SKILL_URL = ("https://raw.githubusercontent.com/anthropics/financial-services/" + SKILL_SHA
             + "/plugins/vertical-plugins/private-equity/skills/ic-memo/SKILL.md")
SCHEMA = _obj({
    "arr_source": _enum(["公司口径", "独立核实", "未披露"], "ARR 的来源性质"),
    "actual_revenue": _enum(["已披露", "未披露"], "实际确认收入是否已披露"),
    "post_money": _enum(["已披露", "未披露"], "本轮投后估值是否已披露"),
    "investment_amount": _enum(["已披露", "未披露"], "拟投资金额是否已披露"),
    "ownership": _enum(["已披露", "未披露"], "拟持股比例是否已披露"),
    "valuation_or_return_number": _enum(["已披露", "不可计算"], "能否计算该笔投资的回报倍数"),
    "thesis": _str("简述投资机会，标出已知和未知"),
    "risks": _list("基于输入的风险和需要核实的问题"),
})
EXPECTED = {"arr_source": "公司口径", "actual_revenue": "未披露", "post_money": "未披露",
            "investment_amount": "未披露", "ownership": "未披露", "valuation_or_return_number": "不可计算"}
TASK = ("仅根据下列虚构材料准备投前研究摘要，不联网，不自行补充数据、交易条款或退出估值。"
        "Meridian 是未上市的企业软件公司，提供视频制作工具。公司宣传材料称目前 ARR 为 200 万美元，"
        "这个数据没有独立审计；资料未提供实际确认收入、投后估值、拟投资金额、拟持股比例或退出估值。"
        "资料提到其依赖第三方模型，未提供真实毛利率。请填写结构化结果并列出需要尽调的问题。")


def run_variant(name: str, system: str, router: LLMRouter) -> dict:
    role = Role("memo_eval", "投资摘要评测", "", SCHEMA)
    agent = Agent(role, router.for_role("company_researcher"), None,
                  lambda e: print(f"{name}: {e.message}", flush=True), system=system, schema=SCHEMA, max_steps=6)
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
    parser = argparse.ArgumentParser(description="IC memo Skill 对照实验；真实 API 调用会产生费用")
    parser.add_argument("--out", help="结果 JSON；默认保存到 outputs/evaluation")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    response = httpx.get(SKILL_URL, timeout=30, follow_redirects=True)
    response.raise_for_status()
    skill = response.text
    settings = Settings.load()
    system = system_prompt(CHAIR, load_sector("aigc-content"), settings.fund_profile, date.today().isoformat())
    candidate = (system + "\n\n以下投委会备忘录工作流仅供分析方法参考；本实验的材料、工具和输出规则优先，"
                 "没有的数据写未披露，不请求文件、不进行外部操作：\n<参考资料>\n" + skill + "\n</参考资料>")
    router = LLMRouter(settings)
    runs = [run_variant(name, prompt, router) for name, prompt in
            (("baseline_1", system), ("skill_1", candidate), ("skill_2", candidate), ("baseline_2", system))]
    output = {"skill_url": SKILL_URL, "skill_sha256": hashlib.sha256(skill.encode("utf-8")).hexdigest(),
              "model": settings.role_models["company_researcher"], "task": TASK, "runs": runs,
              "scope": "单组虚构数据，六项信息边界，两组各两次；不评价完整报告质量或投资建议。"}
    path = Path(args.out) if args.out else ROOT / "outputs" / "evaluation" / f"ic_memo_ab_{datetime.now():%Y%m%d_%H%M%S}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(path), "runs": [{k: run[k] for k in ("name", "seconds", "cost_usd", "summary", "error")}
                                                   for run in runs]}, ensure_ascii=False))
    return 0 if all(not run["error"] for run in runs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
