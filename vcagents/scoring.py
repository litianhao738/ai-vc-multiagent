"""评分、排序和回报测算。数字都在这里由代码计算，模型只负责打分和写理由。"""
from __future__ import annotations

import math
from copy import deepcopy

from .sectors import Sector


def weighted_score(scores: dict[str, int], sector: Sector) -> float:
    return round(sum(d.weight * scores[d.key] for d in sector.dimensions), 2)


def topsis(matrix: list[list[float]], weights: list[float]) -> list[float]:
    """TOPSIS 多准则决策：离理想方案越近、离最差方案越远，得分越高（0-1）。所有维度都是越高越好。"""
    cols = range(len(weights))
    norms = [math.sqrt(sum(row[j] ** 2 for row in matrix)) or 1.0 for j in cols]
    weighted = [[weights[j] * row[j] / norms[j] for j in cols] for row in matrix]
    best = [max(row[j] for row in weighted) for j in cols]
    worst = [min(row[j] for row in weighted) for j in cols]
    out = []
    for row in weighted:
        d_best = math.dist(row, best)
        d_worst = math.dist(row, worst)
        out.append(round(d_worst / (d_best + d_worst), 3) if d_best + d_worst else 0.5)
    return out


def rank_companies(analyses: dict[str, dict], sector: Sector) -> list[dict]:
    """按加权分排序，并用 TOPSIS 交叉检查；两种方法的第一名不一致时标记出来，交给对比分析师判断。"""
    names = list(analyses)
    keys = [d.key for d in sector.dimensions]
    matrix = [[float(analyses[n]["scores"][k]["score"]) for k in keys] for n in names]
    closeness = topsis(matrix, [d.weight for d in sector.dimensions]) if names else []
    rows = [
        {
            "company": n,
            "weighted": weighted_score({k: analyses[n]["scores"][k]["score"] for k in keys}, sector),
            "topsis": closeness[i],
            "scores": {k: analyses[n]["scores"][k]["score"] for k in keys},
        }
        for i, n in enumerate(names)
    ]
    rows.sort(key=lambda r: (-r["weighted"], -r["topsis"], r["company"]))
    for i, row in enumerate(rows, 1):
        row["rank"] = i
    by_topsis = max(rows, key=lambda r: r["topsis"])["company"] if rows else None
    for row in rows:
        row["topsis_disagrees"] = row["rank"] == 1 and by_topsis != row["company"]
    return rows


def ranking_sensitivity(analyses: dict[str, dict], sector: Sector) -> dict:
    """逐个维度改变一家公司的分数一分，观察第一名是否变化。"""
    original = rank_companies(analyses, sector)[0]["company"]
    changes = []
    scenarios = 0
    for name, analysis in analyses.items():
        for dimension in sector.dimensions:
            score = analysis["scores"][dimension.key]["score"]
            for delta in (-1, 1):
                if not 1 <= score + delta <= 10:
                    continue
                variant = deepcopy(analyses)
                variant[name]["scores"][dimension.key]["score"] = score + delta
                new_top = rank_companies(variant, sector)[0]["company"]
                scenarios += 1
                if new_top != original:
                    changes.append({"company": name, "dimension": dimension.name, "delta": delta, "top_pick": new_top})
    return {"original_top": original, "scenarios": scenarios, "changes": len(changes), "examples": changes[:5]}


def score_ranges(analyses: dict[str, dict], fact_sheets: dict[str, dict], sector: Sector) -> dict:
    """对低置信度或缺少关联独立来源的维度，用 1–10 分展示假设范围。"""
    ranges = {}
    for name, analysis in analyses.items():
        linked = {e["evidence_id"] for e in fact_sheets[name]["evidence"]
                  if e["source_linked"] and e.get("excerpt_verified", False) and e["source_type"] == "独立来源"}
        lower, upper, uncertain = 0.0, 0.0, []
        for dim in sector.dimensions:
            item = analysis["scores"][dim.key]
            if item["confidence"] == "低" or not linked.intersection(item["evidence_ids"]):
                lower += dim.weight
                upper += 10 * dim.weight
                uncertain.append(dim.name)
            else:
                lower += dim.weight * item["score"]
                upper += dim.weight * item["score"]
        ranges[name] = {"min": round(lower, 2), "max": round(upper, 2), "uncertain_dimensions": uncertain}
    return ranges


def exit_multiple(exit_valuation: float | None, post_money: float, dilution: float) -> float | None:
    """回报倍数 = 退出估值 × (1 − 后续稀释) ÷ 本轮投后估值。"""
    if exit_valuation is None or post_money <= 0:
        return None
    return round(exit_valuation * (1 - dilution) / post_money, 2)
