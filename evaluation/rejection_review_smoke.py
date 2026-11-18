"""一次小批量真实调用，只验证退回必要性 prompt；不运行四主Agent研究。"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

from dotenv import load_dotenv

from vcagents.config import Settings
from vcagents.governor import RunGovernor
from vcagents.llm import LLMRouter
from vcagents.rejection_review import RejectionReviewer


# The expected answers stay local and are never sent to the model.
CASES = [
    ("标点或全半角不一致", "Alpha付费用户比例２０．３％；", "Alpha付费用户比例20.3%。", True),
    ("数量写法不一致", "Alpha融资4亿美元", "Alpha raised $400 million.", True),
    ("数量写法不一致", "Alpha估值10亿美元", "Alpha was valued at USD 1 billion.", True),
    ("百分数与小数写法不一致", "Alpha付费用户比例20.3%", "Alpha paid-user share is 0.203.", True),
    ("数量写法不一致", "Alpha付费用户比例203%", "Alpha paid-user share is 20.3%.", False),
    ("数量写法不一致", "Alpha利润率-20.3%", "Alpha profit margin is 20.3%.", False),
    ("金额写法不一致", "Alpha融资4亿人民币", "Alpha raised USD 400 million.", False),
    ("金额写法不一致", "Alpha的2025年ARR为4亿美元", "Alpha raised USD 400 million in 2026.", False),
    ("缺少有效原文", "Alpha融资4亿美元", "", False),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="发起一次真实 API 调用（复用 .env 中的 OpenRouter Key）")
    parser.add_argument("--model", help="临时覆盖复核模型")
    args = parser.parse_args()
    if not args.live:
        print("添加 --live 可真实验证九个小样例；不会运行整份研究。")
        return 0
    load_dotenv()
    settings = Settings.load()
    if args.model:
        settings.role_models["rejection_review"] = args.model
    governor = RunGovernor(timeout_seconds=60, retry_budget=0, cost_budget_usd=0.05)
    governor.start()
    reviewer = RejectionReviewer(settings, LLMRouter(settings, governor))
    issues = [f"案例{i}：{row[0]}" for i, row in enumerate(CASES)]
    context = {"comparisons": [{"issue_id": i, "submitted": row[1], "source_excerpt": row[2]}
                               for i, row in enumerate(CASES)]}
    reviewer.filter(issues, context)
    record = reviewer.records[0] if reviewer.records else {"status": "not_run", "decisions": []}
    decisions = {row["issue_id"]: row["unnecessary"] for row in record["decisions"]}
    results = [{"case": i, "expected_unnecessary": row[3], "actual_unnecessary": decisions.get(i),
                "passed": decisions.get(i) is row[3]} for i, row in enumerate(CASES)]
    passed = sum(row["passed"] for row in results)
    folder = Path("outputs/validation") / datetime.now().strftime("rejection-review_%Y%m%d_%H%M%S_%f")
    folder.mkdir(parents=True)
    (folder / "result.json").write_text(json.dumps({"scope": __doc__, "record": record, "cases": results},
                                                  ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": passed, "total": len(CASES), "model": record.get("model"),
                      "status": record["status"], "usage": record.get("usage"), "cost_usd": record.get("cost_usd"),
                      "seconds": record.get("seconds"), "error": record.get("error"), "folder": str(folder)}, ensure_ascii=False))
    return 0 if passed == len(CASES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
