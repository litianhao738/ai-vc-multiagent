"""汇总历史报告与用户提供的后续结果，检查首选公司的后续表现。"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="汇总历史投研报告的回测结果")
    parser.add_argument("cases", help="回测案例 JSON 文件")
    parser.add_argument("--out", default="backtest-result.json")
    args = parser.parse_args()
    path = Path(args.cases)
    cases = json.loads(path.read_text(encoding="utf-8"))
    results = evaluate(cases, path.parent)
    Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"回测结果已保存到 {args.out}")


def evaluate(cases: list[dict], directory: Path) -> dict:
    rows = []
    for case in cases:
        result = json.loads((directory / case["result"]).read_text(encoding="utf-8"))
        top_pick = result["decision"]["top_pick"]
        outcome = case["outcomes"].get(top_pick, "未知")
        if outcome not in ("融资", "收购", "倒闭", "未知"):
            raise ValueError(f"{top_pick} 的后续结果必须是融资、收购、倒闭或未知")
        rows.append({"result": case["result"], "as_of": case["as_of"], "top_pick": top_pick, "outcome": outcome})
    known = [r for r in rows if r["outcome"] != "未知"]
    positive = sum(r["outcome"] in ("融资", "收购") for r in known)
    return {"cases": rows, "known_cases": len(known), "positive_cases": positive,
            "positive_rate": round(positive / len(known), 3) if known else None,
            "note": "融资或收购只是后续事件，并不等同投资收益；须人工确认研究资料没有包含事后信息。"}


if __name__ == "__main__":
    main()
