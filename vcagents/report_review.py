"""Pass only citation-scoped amount questions to the writing reviewer."""
from __future__ import annotations

import re

from .citations import normal_text
from .quantities import claim_quantities, fact_quantities, same_value, table_unit_label


_CURRENCIES = {
    "USD": r"美元|美金|(?<![A-Za-z])(?:USD|U\.?S\.?\s+dollars?|US\$)(?![A-Za-z])",
    "CNY": r"人民币|(?<![A-Za-z])(?:CNY|RMB)(?![A-Za-z])",
    "HKD": r"港元|港币|(?<![A-Za-z])(?:HKD|HK\$)(?![A-Za-z])",
    "EUR": r"欧元|€|(?<![A-Za-z])EUR(?![A-Za-z])",
    "GBP": r"英镑|£|(?<![A-Za-z])GBP(?![A-Za-z])",
    "JPY": r"日元|(?<![A-Za-z])JPY(?![A-Za-z])",
}
_METRICS = {
    "ARR": r"年度经常性收入|annual recurring revenue|\bARR\b",
    "年化收入": r"年化(?:收入|营收)|annualized revenue",
    "年度收入": r"年度(?:收入|营收)|财年(?:收入|营收)|实际(?:收入|营收)|revenue for (?:the |fiscal )?year",
    "融资额": r"融资(?:额|金额|总额)|amount raised",
    "估值": r"估值|\bvaluation\b",
}
_YEARS = re.compile(r"(?:\bFY\s*((?:19|20)\d{2})\b|(?<!\d)((?:19|20)\d{2})\s*年)", re.I)
_SCALE = re.compile(r"万亿|百万|千万|万|亿|\b(thousand|million|billion)s?\b", re.I)
_CELL = re.compile(r"[+＋\-−－]?[0-9０-９]+(?:[,，][0-9０-９]{3})*(?:[.．][0-9０-９]+)?")


def _labels(text, patterns):
    return {name for name, pattern in patterns.items() if re.search(pattern, text, re.I)}


def currency_labels(text: str) -> set[str]:
    return _labels(text, _CURRENCIES)


def _years(text):
    return {fiscal or calendar for fiscal, calendar in _YEARS.findall(text)}


def amount_scope(claim: str, excerpt: str, numeric_match: bool) -> tuple[list[str], list[str]]:
    """Flag explicit conflicts and incomplete scope without inferring the unit of a bare dollar sign."""
    currencies = currency_labels(claim)
    if len(currencies) != 1:
        return [], []
    pairs = (
        ("币种", currencies, currency_labels(excerpt)),
        ("指标", _labels(claim, _METRICS), _labels(excerpt, _METRICS)),
        ("期间", _years(claim), _years(excerpt)),
    )
    conflicts, unclear = [], []
    for name, stated, cited in pairs:
        if len(stated) == len(cited) == 1 and stated.isdisjoint(cited):
            conflicts.append(f"{name}不同（正文 {'、'.join(stated)}，原文 {'、'.join(cited)}）")
        elif numeric_match and len(stated) == 1 and stated != cited:
            unclear.append(name)
    return conflicts, unclear


def table_amount_options(claim: str, excerpt: str) -> list[dict]:
    """Recompute a single stated amount from an explicit currency/scale and a bare value cell."""
    amounts = fact_quantities(claim)
    currencies = _labels(claim, _CURRENCIES)
    if len(amounts) != 1 or len(currencies) != 1 or amount_scope(claim, excerpt, False)[0]:
        return []
    options = []
    header = ""
    scale = None
    for line in excerpt.splitlines():
        if label := table_unit_label(line):
            unit = label[label.lastindex]
            scale_tokens = list(_SCALE.finditer(unit))
            scale = None
            if len(scale_tokens) == 1 and _labels(unit, _CURRENCIES) == currencies:
                token = scale_tokens[0]
                scale = claim_quantities(f"1 {token[1] or token[0]}")[0].value
                header = line.strip()
            if not _labels(line, _METRICS) or not re.search(r"[|\t]", line):
                continue
        if scale is None:
            continue
        cells = re.split(r"[|\t]", line) if re.search(r"[|\t]", line) else re.split(r"[:：]", line, maxsplit=1)
        if len(cells) < 2:
            continue
        for cell in cells:
            value = cell.strip()
            if _CELL.fullmatch(value) and same_value(claim_quantities(value)[0].value * scale, amounts[0].value):
                options.append({"unit_header": header, "row": line.strip(), "cell": value})
    return options


def review_result(context: dict, decision) -> dict:
    """An equivalent verdict must point back to an original line, and a table value must be code-checked."""
    if (not isinstance(decision, dict) or decision.get("verdict") not in ("equivalent", "different", "insufficient")
            or not isinstance(decision.get("reason"), str) or not decision["reason"].strip()
            or not isinstance(decision.get("basis"), str)):
        raise ValueError("成稿复核的结论、理由或依据格式有误")
    basis = normal_text(decision["basis"])
    excerpt = normal_text(context["evidence"]["excerpt"])
    if decision["verdict"] != "insufficient" and (not basis or basis not in excerpt):
        raise ValueError("成稿复核的依据须摘自所引原文")
    if decision["verdict"] == "equivalent" and context["kind"] == "table_unit":
        row = next((option["row"] for option in context["options"] if basis == normal_text(option["row"])), None)
        if row is None:
            raise ValueError("等值结论须选用程序已复算的原文表格行")
        return {**decision, "basis": row}
    return decision


def apply_report_reviews(report: dict, records: list[dict], reviewer=None) -> dict:
    """Apply current-source decisions. Export passes no reviewer and stays read-only."""
    findings, reviewed = [], []
    accepted = 0
    ordered = sorted(report["findings"], key=lambda finding: finding.get("amount_review", {}).get("kind") != "table_unit")
    ordered.extend(report.get("scope_cases", []))
    for finding in ordered:
        context = finding.get("amount_review")
        decision = None
        if context:
            matching = next((row for row in reversed(records) if row.get("scope") == "report_amount"
                             and row.get("context") == context), None)
            if matching and matching.get("status") == "reviewed":
                try:
                    decision = review_result(context, matching["decisions"][0])
                except (IndexError, KeyError, TypeError, ValueError):
                    pass
            elif reviewer is not None:
                decision = reviewer(context)
        if decision and decision["verdict"] == "equivalent" and context["kind"] == "table_unit":
            accepted += 1
            reviewed.append({"claim": context["claim"], "evidence_id": context["evidence"]["id"],
                             "basis": decision["basis"], "reason": decision["reason"]})
            continue
        if decision:
            label = {"equivalent": "写法等价", "different": "不等价", "insufficient": "信息不足"}[decision["verdict"]]
            finding = {**finding, "problem": f"{finding['problem']}。辅助复核为{label}，{decision['reason']}"}
        findings.append(finding)
    return {**report, "supported": report["supported"] + accepted, "findings": findings, "reviewed_amounts": reviewed}
