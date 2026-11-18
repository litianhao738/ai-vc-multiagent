"""证据先行：分析、对比、辩论和主席正文中的事实性数字，须能在所引证据中找到等值写法。

检查带量级、币种、百分号、倍数或模型数量的数字；年份、评分、序号由事实核查员对照原文。
引用写作 [E-C1-3]（证据编号）或 [C1-6]（来源编号）。推算出的数字须写出算式，例如“约 7.7 倍（54 亿 ÷ 7 亿）”。
检查结果不直接判定真伪：未标注或数字不符的句子交给事实核查员优先核查。
"""
from __future__ import annotations

import operator
import re

from .quantities import claim_quantities, display_matches as _display_matches, fact_quantities, same_value, source_values
from .report_review import amount_scope, table_amount_options
from .input_amounts import InputAmounts

# 证据编号 E-C1-3 / E-S-2，来源编号 C1-6 / S-12，或用户输入与代码计算结果 U-1；带不带方括号、圆括号都可以
_DASH = r"[-‐‑‒–—－]"
_CITE_ID = rf"E{_DASH}(?:C\d+|S){_DASH}\d+|(?:C\d+|S){_DASH}\d+|U{_DASH}1"
_CITE = re.compile(rf"(?<![A-Za-z0-9-])({_CITE_ID})(?!\d)", re.I)
_LEADING_CITES = re.compile(rf"^\s*(?:[\[(（【]?(?:{_CITE_ID})[\])）】]?[\s、,，]*)+", re.I)
USER_KEY = "U-1"
_SENTENCE = re.compile(r"[^。！？!?；;\n]+[。！？!?；;]?")
# 真实反例：表格行“| 假设：仅将……”也是假设句
_ASSUMPTION = re.compile(r"^[\s|*\-•#>\d.、)）(（]*(?:本报告假设|本次假设|情景假设|假设|若|如果)"
                         r"|本报告假设|本次假设|情景假设|假设门槛|均为假设|为本报告")  # 也接受句中明确标注，如“超过 15 倍（该门槛为本报告假设）”
# 不支持减号：与“2,000-10,000”这类区间和日期写法冲突
_OPS = {"÷": operator.truediv, "/": operator.truediv, "×": operator.mul, "*": operator.mul, "+": operator.add}
_OPERAND = (r"(?:US\$|\$|USD\s*)?\d+(?:,\d{3})*(?:\.\d+)?\s*"
            r"(?:万亿|千亿|百亿|十亿|亿|千万|百万|十万|万|billion|million|[BMK])?\s*[%％]?\s*"
            r"(?:美元|人民币|港元|欧元|元|倍|家客户|名用户|家|人|个)?")
_FORMULA = re.compile(rf"{_OPERAND}(?:\s*[÷/×*+]\s*{_OPERAND})+", re.I)
_RESULT_BEFORE = re.compile(rf"(?P<amount>{_OPERAND})\s*(?:[（(]|[=≈])\s*$", re.I)
_RESULT_AFTER = re.compile(rf"\s*[）)]?\s*[=≈]\s*(?P<amount>{_OPERAND})", re.I)
SKIP_KEYS = {"exit_scenarios", "evidence_ids", "open_questions", "dd_checklist", "next_steps",
             "action_conditions", "would_change_mind", "would_change_if", "if_obtained"}  # 情景、行动、改变条件和尽调问题不是事实断言


def evidence_index(sector_brief: dict, fact_sheets: dict[str, dict]) -> dict[str, dict]:
    """引用键 → {"values": 已核对证据中的数值, "verified": 是否有通过原文核对的证据}。"""
    index: dict[str, dict] = {}
    records = list(sector_brief.get("evidence", [])) + [e for sheet in fact_sheets.values() for e in sheet.get("evidence", [])]
    for record in records:
        excerpt = record.get("excerpt", "")
        values = source_values(excerpt)
        verified = bool(record.get("excerpt_verified"))
        # These aliases are stamped by Toolbox only after a writing-equivalence
        # review; the actual source excerpt is retained unchanged for audit.
        if verified and record.get("writing_review", {}).get("waived"):
            values.update(v for v in record.get("equivalent_values", []) if isinstance(v, (int, float)))
        keys = [record.get("evidence_id", "")] + re.findall(r"\[([^]]+)\]", record.get("source_ref", ""))
        for key in filter(None, keys):
            entry = index.setdefault(key, {"values": set(), "verified": False, "quotes": []})
            if verified:
                entry["values"] |= values
                entry["verified"] = True
                if excerpt:
                    entry["quotes"].append({"id": record.get("evidence_id", ""), "reference": record.get("source_ref", ""),
                                            "location": record.get("quote_location", ""), "excerpt": excerpt})
    return index


def check_material(name: str, data, index: dict[str, dict]) -> dict:
    """逐句检查一份材料；返回句子数、有支持的句子数和问题清单。"""
    findings, scope_cases, numeric, supported = [], [], 0, 0
    inputs = index.get(USER_KEY, {}).get("amounts")
    # Inputs and calculated scenarios support a statement only when U-1 is
    # cited. A coincidentally equal exit value cannot corroborate a revenue fact.
    corpus = set().union(*(entry["values"] for key, entry in index.items() if key != USER_KEY))
    for path, text in _strings(data, skip_keys=False):
        for sentence in _SENTENCE.findall(text):
            for formula, exact, shown in _calculations(_CITE.sub("", sentence)):
                if exact is None or (shown and not _display_matches(shown, exact) and not _nearly(shown, exact)):
                    problem = (f"算式 {formula.group().strip()} 的除数为零" if exact is None
                               else f"算式 {formula.group().strip()} 与所写结果 {shown.strip()} 不符")
                    findings.append(_finding(name, path, sentence.strip(), [shown.strip()] if shown else [], problem))
    for path, text in _strings(data):
        sentences = list(_SENTENCE.finditer(text))
        paragraph = _cite_keys(text)
        nearby = set().union(*(index[c]["values"] for c in paragraph if c != USER_KEY and c in index and index[c]["verified"]))
        company = inputs.material_company(name, path) if inputs else ""
        for i, match in enumerate(sentences):
            sentence = match.group(0).strip()
            plain = _CITE.sub("", sentence)
            quantities = fact_quantities(plain)
            if not quantities:
                continue
            numeric += 1
            cites = _cite_keys(sentence)
            if i + 1 < len(sentences):  # 引用常写在句号后面，被切到下一句开头
                leading = _LEADING_CITES.match(sentences[i + 1].group(0))
                if leading:
                    cites = list(dict.fromkeys(cites + _cite_keys(leading.group(0))))
            unknown = [c for c in cites if c not in index]
            if _ASSUMPTION.search(plain) and not unknown:
                numeric -= 1
                continue
            formula_values = _derived_values(plain)
            derived = formula_values
            if USER_KEY in cites and USER_KEY in index:  # 代码计算结果在正文中按显示精度四舍五入
                derived = formula_values | index[USER_KEY]["values"]
            unverified = [c for c in cites if c in index and not index[c]["verified"]]
            values = set().union(*(index[c]["values"] for c in cites if c in index)) if cites else set()
            missing = [q for q in quantities if not _known(q, values, derived)]
            if unknown:
                findings.append(_finding(name, path, sentence, [q.text for q in missing], f"引用的编号 {'、'.join(unknown)} 不存在"))
                continue
            if cites and unverified and len(unverified) == len(cites):
                findings.append(_finding(name, path, sentence, [q.text for q in quantities], "所引证据未通过原文核对"))
                continue
            input_issues = []
            if inputs and USER_KEY in cites:
                independent = set().union(*(index[c]["values"] for c in cites if c != USER_KEY and index[c]["verified"]))
                for q in quantities:
                    if _known(q, independent, formula_values):
                        continue
                    scope = inputs.check(plain, q, quantities, company)
                    if scope is None:
                        continue
                    if q in missing:
                        missing.remove(q)
                    if scope[0] != "supported":
                        input_issues.append(_finding(name, path, sentence, [q.text], scope[1], scope[0]))
                findings.extend(input_issues)
            if len(quantities) == 1 and USER_KEY not in cites and not _FORMULA.search(plain):
                issue = _amount_finding(name, path, sentence, plain, quantities[0], cites, index, not missing)
                if issue:
                    if issue["severity"] == "note":
                        scope_cases.append(issue)
                        supported += 1  # The value matches. The contextual review remains open.
                    else:
                        findings.append(issue)
                    continue
            # 同段其他引用已经覆盖这个数字，不算错；证据库里有但编号不对，只提醒；哪里都没有才退回
            still = [q for q in missing if not _known(q, nearby, derived)]
            if not still:
                if not input_issues:
                    supported += 1
                continue
            invented = [q.text for q in still if not _known(q, corpus, derived)]
            problem = f"数字不在所引证据 {'、'.join(cites)} 中" if cites else "未标注证据编号"
            findings.append(_finding(name, path, sentence, invented or [q.text for q in still], problem,
                                     "error" if invented else "note"))
    report = {"material": name, "numeric_sentences": numeric, "supported": supported, "findings": findings}
    if scope_cases:
        report["scope_cases"] = scope_cases
    return report


def _amount_finding(name, path, sentence, plain, quantity, cites, index, numeric_match):
    # Make a distinct, bounded comparison only when the citation identifies one read excerpt.
    quotes = {(q["id"], q["location"], q["excerpt"]): q for cite in cites for q in index[cite].get("quotes", [])}
    if len(quotes) != 1:
        return None
    evidence = next(iter(quotes.values()))
    conflicts, unclear = amount_scope(plain, evidence["excerpt"], numeric_match)
    if conflicts:
        return _finding(name, path, sentence, [quantity.text], "、".join(conflicts))
    options = [] if numeric_match else table_amount_options(plain, evidence["excerpt"])
    if options:
        kind, problem, severity = "table_unit", "原文表格的单位需与对应单元格核对", "error"
    elif unclear:
        kind, problem, severity = "scope", f"数值匹配，仍需对照原文核对{'、'.join(unclear)}", "note"
    else:
        return None
    finding = _finding(name, path, sentence, [quantity.text], problem, severity)
    finding["amount_review"] = {"kind": kind, "claim": sentence, "issue": problem, "evidence": evidence, "options": options}
    return finding


def user_values(*payloads) -> set[float]:
    """用户提供的交易、经营、市场数据及代码计算结果中的数值，供引用 [U-1] 时核对。
    金额字段以百万美元保存，正文常写成“20 亿美元”，因此 *_musd 同时接受换算后的数值；比例同时接受小数与百分数。"""
    values: set[float] = set()

    def walk(value, key: str = ""):
        if isinstance(value, bool):
            return
        if isinstance(value, (int, float)):
            values.update({float(value), float(value) * 100})
            if key.endswith("_musd"):
                values.add(float(value) * 1e6)
        elif isinstance(value, dict):
            for k, item in value.items():
                walk(item, str(k))
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item, key)

    for payload in payloads:
        walk(payload)
    return values


def user_entry(result) -> dict:
    return {"verified": True, "amounts": InputAmounts(result), "values": user_values(
        result.deal_metrics, result.financial_metrics, result.market_metrics, result.returns)}


def errors_for_submission(report: dict, limit: int = 8, *, defer_amount_review: bool = False) -> list[str]:
    items = [f for f in report["findings"] if f.get("severity", "error") == "error"
             and not (defer_amount_review and f.get("amount_review", {}).get("kind") == "table_unit")]
    errors = [f"“{f['sentence'][:60]}”中的 {'、'.join(f['numbers']) or '数字'}：{f['problem']}" for f in items[:limit]]
    if len(items) > limit:
        errors.append(f"另有 {len(items) - limit} 句同类问题")
    if errors:
        errors.append("请在句末标注支持该数字的证据编号（如 [E-C1-3]）或来源编号（如 [C1-6]）；用户提供的交易与经营数据"
                      "及代码计算结果标注 [U-1]；推算值写出算式，例如“约 7.7 倍（54 亿 ÷ 7 亿）”；情景假设写明“假设”；"
                      "没有证据的数字请删除")
    return errors


def _finding(name: str, path: str, sentence: str, numbers: list[str], problem: str, severity: str = "error") -> dict:
    return {"material": name, "field": path, "sentence": sentence, "numbers": numbers, "problem": problem, "severity": severity}


def _cite_keys(text: str) -> list[str]:
    """引用编号的规范写法：大小写和破折号样式不影响匹配。"""
    return list(dict.fromkeys(re.sub(_DASH, "-", key).upper() for key in _CITE.findall(text)))


def _nearly(text: str, exact: float) -> bool:
    """算式结果差 0.5% 以内不算算错，例如四舍五入后再写“约”。"""
    value = claim_quantities(text)[0].value
    if "%" in text or "％" in text:
        exact *= 100
    return abs(value - exact) <= 0.005 * abs(exact)


def _known(quantity, values: set[float], derived: set[float]) -> bool:
    return any(same_value(quantity.value, v) for v in values) or any(_display_matches(quantity.text, v) for v in derived)


def _rounds_to(shown: float, exact: float) -> bool:
    return _display_matches(f"{shown:.10f}".rstrip("0").rstrip("."), exact)


def _derived_values(sentence: str) -> set[float]:
    results = set()
    for _, exact, shown in _calculations(sentence):
        if exact is not None:
            results.add(exact)
            if shown and _display_matches(shown, exact):
                results.add(claim_quantities(shown)[0].value)
    return results


def _calculations(sentence: str):
    """只解析正文中写出的四则算式；百分数作为比例，乘除先于加法。"""
    for match in _FORMULA.finditer(sentence):
        parts = re.split(r"([÷/×*+])", match.group())
        values = [claim_quantities(term)[0].value / (100 if re.search(r"[%％]", term) else 1)
                  for term in parts[::2]]
        ops = parts[1::2]
        terms = [values[0]]
        exact = None
        for op, value in zip(ops, values[1:]):
            if op in "÷/" and value == 0:
                break
            if op == "+":
                terms.append(value)
            else:
                terms[-1] = _OPS[op](terms[-1], value)
        else:
            exact = sum(terms)
        before = _RESULT_BEFORE.search(sentence[:match.start()])
        after = _RESULT_AFTER.match(sentence[match.end():])
        shown = after.group("amount") if after else before.group("amount") if before else ""
        yield match, exact, shown


def _strings(value, path: str = "", *, skip_keys: bool = True):
    if isinstance(value, str):
        yield path, value
    elif isinstance(value, list):
        for i, item in enumerate(value):
            yield from _strings(item, f"{path}[{i}]", skip_keys=skip_keys)
    elif isinstance(value, dict):
        for key, item in value.items():
            if not skip_keys or key not in SKIP_KEYS:
                yield from _strings(item, f"{path}.{key}" if path else key, skip_keys=skip_keys)
