"""核对 [U-1] 所指的公司和金额字段。保留原字段归属，金额统一换算为美元。"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .financials import DEAL_LABELS, INPUT_LABELS, MARKET_INPUT_LABELS, MARKET_METRIC_LABELS, METRIC_LABELS
from .quantities import Quantity, display_matches, same_value
from .report_review import currency_labels


_NAMES = {**INPUT_LABELS, **{k: v[0] for k, v in MARKET_INPUT_LABELS.items()},
          **{k: v["label"] for k, v in (DEAL_LABELS | METRIC_LABELS | MARKET_METRIC_LABELS).items()},
          "exit_valuation_musd": "退出估值", "proceeds_musd": "退出所得"}
_MARKET = set(MARKET_INPUT_LABELS) | set(MARKET_METRIC_LABELS)
_ALIASES = {
    "post_money_musd": ("投后估值", "投后", "post-money valuation", "post-money"),
    "pre_money_musd": ("投前", "pre-money valuation", "pre-money"),
    "round_size_musd": ("本轮融资总额", "本轮融资", "融资额", "amount raised"),
    "investment_musd": ("拟投资额", "投资金额", "拟出资", "拟投资", "拟投"),
    "revenue_current_musd": ("本期营收", "本年收入"),
    "revenue_previous_musd": ("上期收入", "上年收入", "前期营收"),
    "arr_musd": ("ARR", "年度经常性收入", "年化经常性收入", "annual recurring revenue"),
    "annualized_revenue_musd": ("年化收入", "年化营收", "annualized revenue"),
    "mrr_musd": ("月度经常性收入", "monthly recurring revenue"),
    "tam_musd": ("TAM",), "sam_musd": ("SAM",), "som_musd": ("SOM",),
}
_FIELDS = {label.casefold(): {key} for key, label in _NAMES.items() if key.endswith("_musd")}
for _key, _aliases in _ALIASES.items():
    for _alias in _aliases:
        _FIELDS[_alias.casefold()] = {_key}
for _alias in ("收入", "营收", "年度收入", "年度营收", "年收入", "实际收入", "确认收入", "营业收入", "revenue"):
    _FIELDS[_alias] = {"revenue_current_musd", "revenue_previous_musd"}
_FIELDS["估值"] = _FIELDS["valuation"] = {"pre_money_musd", "post_money_musd"}
_FIELDS["tam"] = {"tam_musd", "tam_top_down_musd"}


def _pattern(names):
    # Use ASCII word boundaries for Latin names while allowing Chinese text directly beside them.
    parts = []
    for name in sorted(names, key=len, reverse=True):
        left = r"(?<![a-z0-9])" if re.match("[a-z0-9]", name, re.I) else ""
        right = r"(?![a-z0-9])" if re.search("[a-z0-9]$", name, re.I) else ""
        parts.append(left + re.escape(name) + right)
    return re.compile("|".join(parts) or r"(?!)", re.I)


_FIELD_PATTERN = _pattern(_FIELDS)
_BREAK = re.compile(r"[。；;！？!?\n]|[,，](?!\d{3}(?:\D|$))")
_OTHER_UNIT = re.compile(r"\s*(?:[%％倍×]|[xX]\b|名|人(?!民币)|家|户|台|个|种|年|月)")
_CURRENCY = (r"美元|美金|人民币|港元|港币|欧元|英镑|日元|元|U\.?S\.?\s+dollars?|US\$|HK\$|"
             r"(?:USD|CNY|RMB|HKD|EUR|GBP|JPY|AUD|CAD|SGD)(?![a-z])|[€£$¥￥]")
_BEFORE = re.compile(r"(" + _CURRENCY + r")\s*$", re.I)
_AFTER = re.compile(r"\s*(" + _CURRENCY + r")", re.I)
_YEAR = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")
_CLAIM_YEAR = re.compile(r"(?<!\d)((?:19|20)\d{2})(?=\s*(?:年|Q[1-4]))|\bFY\s*((?:19|20)\d{2})", re.I)
_ANNUAL_REVENUE = re.compile(r"(?:年度|全年|财年|(?<!化)年)(?:收入|营收)|annual revenue", re.I)
_PART_YEAR = re.compile(r"Q[1-4]|第?[一二三四1-4]季度|[上下]半年|单月|月度", re.I)
# Only inherit a company from the enclosing material for a claim without its own named subject.
_IMPLICIT = re.compile(r"(?:\s|[（(]|本轮|本次|目前|当前|本公司|该公司|目标公司|公司|我们|拟按|按|以|的)*")
_APPROX = re.compile(r"(?:约|近|大致|≈|约合|about|approximately)\s*(?:USD|US\$|\$)?\s*$", re.I)
_FUND_VERB = re.compile(r"(?:投(?:资|入)?|出资)\s*(?:约|近)?\s*(?:USD|US\$|\$)?\s*$", re.I)
_JOINT = re.compile(r"(?:均为|分别|合计|之和|总和)")
_NAMED_SUBJECT = re.compile(r"(?:[一-鿿]+公司|[a-z][a-z0-9 .&-]*)\s*的?\s*$", re.I)


def _field_context(left: str, right: str) -> tuple[set[str], str, bool]:
    before_number = list(_FIELD_PATTERN.finditer(left))
    label = before_number[-1] if before_number else _FIELD_PATTERN.search(right)
    subject = left[:before_number[0].start()] if before_number else left
    if len(before_number) > 1 and _JOINT.search(left):
        return set().union(*(_FIELDS[m[0].casefold()] for m in before_number)), subject, True
    if _FUND_VERB.search(left):
        return {"investment_musd"}, subject, False
    if before_number and re.search(r"折算|换算", left[label.end():]):
        return set(), subject, False  # The label describes the operand, not the converted amount.
    return _FIELDS[label[0].casefold()] if label else set(), subject, False


@dataclass(frozen=True)
class _Amount:
    company: str
    field: str
    dollars: float
    period: str
    scenario: str


class InputAmounts:
    def __init__(self, result):
        deal, financial = result.deal_metrics, result.financial_metrics
        companies = set(result.companies) | set(deal) | set(financial) | set(result.fact_sheets)
        self.companies = sorted(companies, key=len, reverse=True)
        self.aliases: dict[str, set[str]] = {}
        for company in self.companies:
            sheet_name = result.fact_sheets.get(company, {}).get("company_name") or company
            for name in (company, sheet_name):
                for alias in [name, *re.split(r"[（）()]", name)]:
                    if alias.strip():
                        self.aliases.setdefault(alias.strip().casefold(), set()).add(company)
        self.company_pattern = _pattern(self.aliases)
        self.rows: list[_Amount] = []
        for company, fields in deal.items():
            self._add(company, fields)
        for company, record in financial.items():
            self._add(company, record.get("inputs", {}), record)
            self._add(company, record.get("metrics", {}), record)
        for section in ("inputs", "metrics"):
            self._add("", result.market_metrics.get(section, {}), result.market_metrics)
        top_pick = (result.decision or {}).get("top_pick", "")
        for row in result.returns:
            self._add(top_pick, row, scenario=row.get("scenario", ""))

    def _add(self, company, fields, record=None, scenario=""):
        for field, value in fields.items():
            if field.endswith("_musd") and type(value) in (int, float):
                period_key = "previous_period" if field == "revenue_previous_musd" else "period"
                period = (record or {}).get(period_key) or ""
                self.rows.append(_Amount(company, field, value * 1e6, period, scenario))

    def material_company(self, material, path):
        for name in self.companies:
            if material in (f"company_analyst:{name}", f"company_researcher:{name}") or any(
                    path.startswith(prefix + name + ".") for prefix in ("analyses.", "companies.")):
                return name
        return ""

    def _owners(self, text):
        return {company for match in self.company_pattern.finditer(text)
                for company in self.aliases[match[0].casefold()]}

    def _claim_companies(self, clause, sentence, subject, default_company):
        owners = self._owners(clause)
        implicit = bool(_IMPLICIT.fullmatch(subject))
        if not owners and (implicit or not _NAMED_SUBJECT.search(subject)):
            owners = self._owners(sentence)
        if not owners and implicit:
            if default_company:
                return {default_company}
            if len(self.companies) == 1:
                return set(self.companies)
        return owners

    def check(self, text: str, quantity: Quantity, quantities: list[Quantity],
              default_company: str) -> tuple[str, str] | None:
        """Return (severity, reason) for a monetary claim, or None for a different kind of quantity."""
        prefix, suffix = text[:quantity.start], text[quantity.end:]
        if _OTHER_UNIT.match(suffix):
            return None
        # A unit or label for the next amount must not be applied to the current one.
        previous = max((q.end for q in quantities if q.end <= quantity.start), default=0)
        following = min((q.start for q in quantities if q.start >= quantity.end), default=len(text))
        left = _BREAK.split(text[previous:quantity.start])[-1]
        right = _BREAK.split(text[quantity.end:following])[0]
        units = [m[1] for m in (_BEFORE.search(left), _AFTER.match(right)) if m]
        currencies = currency_labels(" ".join(units)) | {u.upper() for u in units if u.upper() in {"AUD", "CAD", "SGD"}}
        fields, subject, joint = _field_context(left, right)
        if not (units or (fields and quantity.scaled)):
            return None
        if currencies and "USD" not in currencies:
            return "error", "该金额的币种与 [U-1] 中的美元口径不同"
        clause = _BREAK.split(prefix)[-1] + quantity.text + _BREAK.split(suffix)[0]
        owners = {""} if fields and fields <= _MARKET else self._claim_companies(clause, text, subject, default_company)
        years = {year or fiscal for year, fiscal in _CLAIM_YEAR.findall(clause)}
        scenarios = {s for s in ("乐观", "基线", "悲观") if s in clause}
        annual_revenue = bool(_ANNUAL_REVENUE.search(clause))
        candidates = [row for row in self.rows
                      if (not owners or row.company in owners) and (not fields or row.field in fields)
                      and (not years or not _YEAR.search(row.period) or years.intersection(_YEAR.findall(row.period)))
                      and not (annual_revenue and row.field in ("revenue_current_musd", "revenue_previous_musd")
                               and len(_PART_YEAR.findall(row.period)) == 1)
                      and (not scenarios or not row.scenario or row.scenario in scenarios)]
        approximate = bool(_APPROX.search(left))
        matching = [row for row in candidates if same_value(quantity.value, row.dollars)
                    or ((approximate or abs(quantity.value - row.dollars) <= abs(row.dollars) * 0.005)
                        and display_matches(quantity.text, row.dollars))]
        if not matching:
            return "error", "[U-1] 中没有与该公司、指标和金额相符的数据"
        if years and not any(_YEAR.search(row.period) for row in matching):
            return "note", "[U-1] 中的金额相符，仍需确认正文所指的期间"
        if (not fields or len(owners) != 1 or joint
                or len(currencies) > 1 or (units and not currencies)):
            return "note", "[U-1] 中有等值金额，仍需按上下文确认公司、指标或币种"
        return "supported", ""
