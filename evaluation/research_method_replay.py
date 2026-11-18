"""离线回放已保存的研究方法案例；不调用模型、不改写提示词或研究结果。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from vcagents.inquiry import replay_method_case
from vcagents.pipeline import AnalysisResult


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path, help="一次运行的result.json或partial-result.json")
    args = parser.parse_args()
    result = AnalysisResult(**json.loads(args.result.read_text(encoding="utf-8")))
    evaluations = [replay_method_case(result, method) for method in result.research_methods]
    print(json.dumps({"run_id": result.run_id, "methods": evaluations,
                      "notice": "通过仅表示该案例执行和独立复核记录完整，不代表方法普遍有效或投资结论已获人工确认"},
                     ensure_ascii=False, indent=2))
    return 0 if all(row["accepted"] for row in evaluations) else 1


if __name__ == "__main__":
    raise SystemExit(main())
