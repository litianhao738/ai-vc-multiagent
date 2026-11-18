"""将投资判断绑定到具体正文，检查关键推论的复核覆盖及结果。"""
from __future__ import annotations

import json


VERDICTS = ["有依据", "条件性推断", "依据不足", "与证据冲突"]


def review_targets(decision: dict) -> dict[str, str]:
    targets = {f"T{i}": json.dumps(item, ensure_ascii=False, sort_keys=True)
               for i, item in enumerate(decision["top_pick_thesis"], 1)}
    targets["V1"] = decision["valuation_view"]
    if decision.get("technical_moat"):
        targets["M1"] = json.dumps({k: decision[k] for k in ("headline", "technical_moat")},
                                   ensure_ascii=False, sort_keys=True)
        action_fields = ("action", "action_conditions")
    else:
        # The archive's original A1 review covered the old, action-led headline as well.
        action_fields = ("action", "headline", "action_conditions")
    targets["A1"] = json.dumps({k: decision[k] for k in action_fields}, ensure_ascii=False, sort_keys=True)
    if decision.get("section_openings"):
        narrative = {"section_openings": decision["section_openings"], "report_subtitle": decision.get("report_subtitle", "")}
        targets["N1"] = json.dumps(narrative, ensure_ascii=False, sort_keys=True)
    if (decision.get("core_answer") or "").strip():
        targets["C1"] = decision["core_answer"]
    for row in decision.get("question_answers", []):
        if row.get("question_id") and (row.get("answer") or "").strip():
            targets[row["question_id"]] = json.dumps({k: row.get(k, "") for k in ("answer", "evidence")},
                                                     ensure_ascii=False, sort_keys=True)
    return targets


def review_issues(decision: dict, check: dict) -> list[dict]:
    targets = review_targets(decision)
    rows = {row["target_id"]: row for row in check.get("reasoning_checks", [])}
    reviewed = check.get("reviewed_targets", {})
    issues = []
    for key, text in targets.items():
        row = rows.get(key)
        if key not in reviewed or reviewed[key] != text or row is None:
            issues.append({"target_id": key, "target": text, "verdict": "依据不足",
                           "reason": "当前版本的关键判断尚未完成逐项复核", "correction": "复核当前判断"})
        elif row["verdict"] in ("依据不足", "与证据冲突"):
            issues.append({**row, "target": text})
    return issues
