"""根据已确认单位和期间的数值计算投研指标，不补充缺失数据。"""
from __future__ import annotations

import math


INPUT_KEYS = frozenset({
    "mrr_musd", "annualized_revenue_musd", "revenue_current_musd", "revenue_previous_musd", "direct_cost_musd",
    "arr_begin_musd", "arr_expansion_musd", "arr_contraction_musd", "arr_churn_musd",
    "sales_marketing_cost_musd", "new_customers", "monthly_arpu_musd", "monthly_customer_churn_rate",
    "cash_musd", "monthly_net_burn_musd", "net_burn_musd", "net_new_arr_musd", "profit_margin",
    "units_delivered", "units_accepted", "units_produced", "units_passed_qc",
    "avg_selling_price_musd", "unit_cost_musd", "cash_collected_musd", "after_sales_cost_musd",
})

# 市场规模属于赛道，单独用一份赛道级输入，避免各公司填出互相矛盾的 TAM。
MARKET_INPUT_KEYS = frozenset({
    "target_customers", "annual_spend_per_customer_musd", "serviceable_share", "obtainable_share",
    "related_market_musd", "addressable_share", "market_initial_musd", "market_final_musd", "market_years",
})

INPUT_LABELS = {
    "mrr_musd": "MRR", "annualized_revenue_musd": "年化收入运行率", "revenue_current_musd": "本期收入", "revenue_previous_musd": "前期收入",
    "direct_cost_musd": "直接成本", "arr_begin_musd": "期初 ARR", "arr_expansion_musd": "增购 ARR",
    "arr_contraction_musd": "降级 ARR", "arr_churn_musd": "流失 ARR", "sales_marketing_cost_musd": "销售与市场费用",
    "new_customers": "新增客户数", "monthly_arpu_musd": "月 ARPU", "monthly_customer_churn_rate": "月客户流失率",
    "cash_musd": "期末现金", "monthly_net_burn_musd": "月净烧钱", "net_burn_musd": "期间净烧钱",
    "net_new_arr_musd": "净新增 ARR", "profit_margin": "利润率", "units_delivered": "交付台数",
    "units_accepted": "验收台数", "units_produced": "生产台数", "units_passed_qc": "质检合格台数",
    "avg_selling_price_musd": "单台平均售价", "unit_cost_musd": "单台成本", "cash_collected_musd": "期间回款",
    "after_sales_cost_musd": "售后成本",
}

_COUNT_KEYS = {"new_customers", "target_customers", "units_delivered", "units_accepted", "units_produced", "units_passed_qc"}
_SHARE_KEYS = {"serviceable_share", "obtainable_share", "addressable_share", "monthly_customer_churn_rate"}
_SIGNED_KEYS = {"monthly_net_burn_musd", "net_burn_musd", "net_new_arr_musd"}

METRIC_LABELS = {
    "arr_musd": {"label": "年化经常性收入 ARR", "unit": "百万美元", "inputs": ("mrr_musd",)},
    "annualized_revenue_musd": {"label": "年化收入运行率", "unit": "百万美元", "inputs": ("annualized_revenue_musd",)},
    "growth_rate": {"label": "收入增长率", "unit": "比例", "inputs": ("revenue_current_musd", "revenue_previous_musd")},
    "nrr": {"label": "净收入留存率 NRR", "unit": "比例",
            "inputs": ("arr_begin_musd", "arr_expansion_musd", "arr_contraction_musd", "arr_churn_musd")},
    "gross_margin": {"label": "毛利率", "unit": "比例", "inputs": ("revenue_current_musd", "direct_cost_musd")},
    "cac_musd": {"label": "获客成本 CAC", "unit": "百万美元", "inputs": ("sales_marketing_cost_musd", "new_customers")},
    "ltv_musd": {"label": "简化客户终身毛利 LTV", "unit": "百万美元",
                 "inputs": ("monthly_arpu_musd", "revenue_current_musd", "direct_cost_musd", "monthly_customer_churn_rate")},
    "ltv_cac": {"label": "LTV / CAC", "unit": "倍",
                "inputs": ("monthly_arpu_musd", "monthly_customer_churn_rate", "sales_marketing_cost_musd", "new_customers")},
    "cac_payback_months": {"label": "CAC 回收期", "unit": "月",
                           "inputs": ("sales_marketing_cost_musd", "new_customers", "monthly_arpu_musd", "revenue_current_musd", "direct_cost_musd")},
    "runway_months": {"label": "现金跑道", "unit": "月", "inputs": ("cash_musd", "monthly_net_burn_musd")},
    "burn_multiple": {"label": "Burn multiple", "unit": "倍", "inputs": ("net_burn_musd", "net_new_arr_musd")},
    "rule_of_40": {"label": "Rule of 40", "unit": "比例", "inputs": ("revenue_current_musd", "revenue_previous_musd", "profit_margin")},
    "unit_gross_margin": {"label": "单台毛利率", "unit": "比例", "inputs": ("avg_selling_price_musd", "unit_cost_musd")},
    "acceptance_rate": {"label": "交付验收率", "unit": "比例", "inputs": ("units_accepted", "units_delivered")},
    "production_yield": {"label": "量产良率", "unit": "比例", "inputs": ("units_passed_qc", "units_produced")},
    "collection_rate": {"label": "回款率", "unit": "比例", "inputs": ("cash_collected_musd", "revenue_current_musd")},
    "after_sales_cost_ratio": {"label": "售后成本占收入比", "unit": "比例", "inputs": ("after_sales_cost_musd", "revenue_current_musd")},
}

MARKET_INPUT_LABELS = {
    "target_customers": ("目标客户数", "家"),
    "annual_spend_per_customer_musd": ("每客户年付费", "百万美元"),
    "serviceable_share": ("可服务比例", "比例"),
    "obtainable_share": ("可获得份额", "比例"),
    "related_market_musd": ("相关总市场", "百万美元"),
    "addressable_share": ("可被替代比例", "比例"),
    "market_initial_musd": ("市场规模初值", "百万美元"),
    "market_final_musd": ("市场规模终值", "百万美元"),
    "market_years": ("初值到终值的年数", "年"),
}

MARKET_METRIC_LABELS = {
    "tam_musd": {"label": "TAM（自下而上）", "unit": "百万美元", "formula": "目标客户数 × 每客户年付费"},
    "tam_top_down_musd": {"label": "TAM（自上而下）", "unit": "百万美元", "formula": "相关总市场 × 可被替代比例"},
    "sam_musd": {"label": "SAM", "unit": "百万美元", "formula": "自下而上 TAM × 可服务比例"},
    "som_musd": {"label": "SOM", "unit": "百万美元", "formula": "SAM × 可获得份额"},
    "market_cagr": {"label": "市场 CAGR", "unit": "比例", "formula": "(终值 ÷ 初值)^(1 ÷ 年数) − 1"},
}

DEAL_LABELS = {
    "post_money_musd": {"label": "本轮投后估值", "unit": "百万美元", "basis": "用户提供"},
    "round_size_musd": {"label": "本轮融资额", "unit": "百万美元", "basis": "用户提供"},
    "pre_money_musd": {"label": "投前估值", "unit": "百万美元", "basis": "代码计算：投后估值 − 本轮融资额"},
    "investment_musd": {"label": "拟投资金额", "unit": "百万美元", "basis": "用户提供"},
    "ownership": {"label": "拟持股比例", "unit": "比例", "basis": "代码计算：拟投资金额 ÷ 投后估值"},
    "implied_arr_multiple": {"label": "隐含 ARR 倍数", "unit": "倍", "basis": "代码计算：投后估值 ÷ ARR（MRR × 12）"},
    "implied_run_rate_multiple": {"label": "年化收入运行率倍数", "unit": "倍", "basis": "测算：投后估值 ÷ 年化收入运行率"},
    "post_round_runway_months": {"label": "本轮后现金跑道", "unit": "月",
                                 "basis": "代码计算：(期末现金 + 本轮融资额) ÷ 月净烧钱，假设本轮资金未计入期末现金"},
}


def parse_financial_data(data: dict) -> dict:
    """公司级经营输入：收入、成本、留存、获客、现金与硬件交付。"""
    return _parse(data, INPUT_KEYS, "经营数据", allow_previous=True)


def parse_market_data(data: dict) -> dict:
    """赛道级市场输入：自下而上与自上而下的市场规模及增速。"""
    return _parse(data, MARKET_INPUT_KEYS, "市场数据", allow_previous=False)


def _parse(data: dict, keys: frozenset, label: str, allow_previous: bool) -> dict:
    if not isinstance(data, dict):
        raise ValueError(f"{label}应为 JSON 对象")
    if isinstance(data.get("inputs"), dict):
        misplaced = set(data["inputs"]) & (MARKET_INPUT_KEYS if keys is INPUT_KEYS else INPUT_KEYS)
        if misplaced:
            where = "赛道级市场数据（--market-data）" if keys is INPUT_KEYS else "公司经营数据（--financial-data）"
            raise ValueError("以下指标应放在" + where + "：" + "、".join(sorted(misplaced)))
    allowed ={"currency", "period", "inputs", "sources"} | ({"previous_period"} if allow_previous else set())
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"{label}包含未知字段：" + "、".join(sorted(map(str, unknown))))
    if data.get("currency") != "USD":
        raise ValueError("currency 必须为 USD，金额统一使用百万美元")
    period = data.get("period")
    if not isinstance(period, str) or not period.strip():
        raise ValueError("period 必须为非空统计期间说明")
    previous_period = data.get("previous_period")
    if previous_period is not None and (not isinstance(previous_period, str) or not previous_period.strip()):
        raise ValueError("previous_period 必须为非空对照期间说明")
    inputs, sources = data.get("inputs"), data.get("sources")
    if not isinstance(inputs, dict) or not isinstance(sources, dict):
        raise ValueError("inputs 和 sources 必须为对象")
    unknown = set(inputs) - keys
    if unknown:
        raise ValueError("inputs 包含未知指标：" + "、".join(sorted(map(str, unknown))))
    if "revenue_previous_musd" in inputs and previous_period is None:
        raise ValueError("提供 revenue_previous_musd 时必须填写 previous_period")
    if set(sources) != set(inputs):
        raise ValueError("sources 必须为每个 inputs 指标提供一条对应的来源说明，不能多列或遗漏")
    normalized, normalized_sources = {}, {}
    for key, value in inputs.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"inputs.{key} 必须为有限数字；未知值请省略")
        try:
            number = float(value)
        except (OverflowError, ValueError):
            raise ValueError(f"inputs.{key} 必须为可计算的有限数字") from None
        if not math.isfinite(number):
            raise ValueError(f"inputs.{key} 必须为有限数字")
        if key in _COUNT_KEYS:
            if number < 0 or not number.is_integer():
                raise ValueError(f"inputs.{key} 必须为非负整数")
            normalized[key] = int(number)
        elif key in _SHARE_KEYS:
            if not 0 <= number <= 1:
                raise ValueError(f"inputs.{key} 必须在 0 到 1 之间")
            normalized[key] = number
        elif key == "profit_margin":
            if number > 1:
                raise ValueError("inputs.profit_margin 不能超过 1，比例使用小数")
            normalized[key] = number
        elif key == "market_years":
            if number <= 0:
                raise ValueError("inputs.market_years 必须大于 0")
            normalized[key] = number
        else:
            if key not in _SIGNED_KEYS and number < 0:
                raise ValueError(f"inputs.{key} 不能为负数")
            normalized[key] = number
        source = sources[key]
        if not isinstance(source, str) or not source.strip():
            raise ValueError(f"sources.{key} 必须为非空来源说明")
        normalized_sources[key] = source.strip()
    arr_keys = ("arr_begin_musd", "arr_expansion_musd", "arr_contraction_musd", "arr_churn_musd")
    if all(key in normalized for key in arr_keys):
        beginning, expansion, contraction, churn = (normalized[key] for key in arr_keys)
        if beginning + expansion - contraction - churn < -1e-12:
            raise ValueError("ARR 变动后的存量收入不能为负，请核对期间与输入数值")
    for part, whole in (("units_accepted", "units_delivered"), ("units_passed_qc", "units_produced")):
        if part in normalized and whole in normalized and normalized[part] > normalized[whole]:
            raise ValueError(f"inputs.{part} 不能大于 {whole}")
    result = {"currency": "USD", "period": period.strip(), "inputs": normalized, "sources": normalized_sources}
    if allow_previous:
        result["previous_period"] = previous_period.strip() if previous_period is not None else None
    return result


def calculate_returns(post_money_musd: float | None, dilution: float, scenarios: list[dict],
                      investment_musd: float | None = None) -> dict:
    rows = []
    raw_multiples = []
    for scenario in scenarios:
        exit_value = scenario["exit_valuation_musd"]
        probability = scenario.get("probability")
        years = scenario.get("holding_years")
        multiple = _divide(_multiply(exit_value, 1 - dilution), post_money_musd)
        annualized = multiple ** (1 / years) - 1 if multiple is not None and years is not None and years > 0 else None
        proceeds = _multiply(multiple, investment_musd)
        raw_multiples.append(multiple)
        rows.append({
            "scenario": scenario["scenario"], "exit_valuation_musd": exit_value,
            "probability": probability, "holding_years": years, "post_money_musd": post_money_musd,
            "dilution": dilution, "multiple": round(multiple, 4) if multiple is not None else None,
            "annualized_irr": round(annualized, 6) if annualized is not None else None,
            "investment_musd": investment_musd, "proceeds_musd": round(proceeds, 4) if proceeds is not None else None,
        })
    expected = None
    if rows and all(row["probability"] is not None and multiple is not None for row, multiple in zip(rows, raw_multiples)):
        expected = round(sum(row["probability"] * multiple for row, multiple in zip(rows, raw_multiples)), 4)
    return {"scenarios": rows, "expected_multiple": expected}


SENSITIVITY_YEARS = (3, 5, 7)
SENSITIVITY_DILUTIONS = (0.3, 0.5, 0.7)


def returns_analysis(post_money_musd: float | None, investment_musd: float | None, dilution: float,
                     scenarios: list[dict], holding_years: float | None, management_fee: float = 0.02,
                     carry: float = 0.2, preference_multiple: float | None = None,
                     preference_stack_musd: float | None = None) -> dict:
    """VC 回报的敏感性、扣费后回报与优先清算权，按 Anthropic returns-analysis Skill（Apache-2.0）的
    双向敏感性表和“扣费前/扣费后”口径改写；去掉 PE 专用的 EBITDA、杠杆和偿债部分，补充原 Skill 没有的优先清算权。

    - 敏感性：每个退出情景分别在不同持有年数、不同后续稀释下的 MOIC 与简化 IRR。
    - 扣费后：管理费按拟投资金额 × 年费率 × 持有年数近似；carry 按扣除本金与管理费后的利润计提。
      这是单个项目的近似，不等于基金层面的 LP 净回报。
    - 优先清算权：1 倍等不参与分配型，与其他优先股同顺位按清算额比例受偿；可以选择转换为普通股按比例分配，取两者较高者。
      未提供优先股总清算额时不建模。
    """
    valid = [s for s in scenarios if s.get("exit_valuation_musd") is not None]
    if not post_money_musd or not valid:
        return {}
    ownership = (investment_musd / post_money_musd) if investment_musd else None

    def moic(exit_value: float, dil: float) -> float:
        return exit_value * (1 - dil) / post_money_musd

    def irr(multiple: float, years: float) -> float | None:
        return multiple ** (1 / years) - 1 if years and years > 0 else None

    years_grid = sorted({*SENSITIVITY_YEARS, *([holding_years] if holding_years else [])})
    dilution_grid = sorted({*SENSITIVITY_DILUTIONS, dilution})
    out = {
        "years_grid": years_grid, "dilution_grid": dilution_grid,
        "by_years": {s["scenario"]: [{"years": y, "multiple": round(moic(s["exit_valuation_musd"], dilution), 4),
                                      "irr": _round(irr(moic(s["exit_valuation_musd"], dilution), y), 6)} for y in years_grid]
                     for s in valid},
        "by_dilution": {s["scenario"]: [{"dilution": d, "multiple": round(moic(s["exit_valuation_musd"], d), 4)}
                                        for d in dilution_grid] for s in valid},
        "fees": {"management_fee": management_fee, "carry": carry},
        "net": [], "preference": None,
    }
    if investment_musd and holding_years:
        fees = investment_musd * management_fee * holding_years
        for s in valid:
            gross = moic(s["exit_valuation_musd"], dilution) * investment_musd
            carried = carry * max(0.0, gross - investment_musd - fees)
            out["net"].append({"scenario": s["scenario"], "gross_proceeds_musd": round(gross, 4),
                               "management_fee_musd": round(fees, 4), "carry_musd": round(carried, 4),
                               "net_proceeds_musd": round(gross - carried, 4),
                               "net_multiple": round((gross - carried) / (investment_musd + fees), 4)})
    if investment_musd and preference_multiple and preference_stack_musd:
        claim = investment_musd * preference_multiple
        stack = max(preference_stack_musd, claim)
        rows = []
        for s in valid:
            value = s["exit_valuation_musd"]
            as_common = ownership * (1 - dilution) * value
            as_preferred = min(claim, value * claim / stack)
            proceeds = max(as_common, as_preferred)
            rows.append({"scenario": s["scenario"], "as_common_musd": round(as_common, 4),
                         "as_preferred_musd": round(as_preferred, 4), "proceeds_musd": round(proceeds, 4),
                         "multiple": round(proceeds / investment_musd, 4),
                         "converts": as_common >= as_preferred})
        out["preference"] = {"multiple": preference_multiple, "stack_musd": preference_stack_musd, "rows": rows}
    return out


def _round(value: float | None, digits: int) -> float | None:
    return round(value, digits) if value is not None else None


def calculate_deal(post_money_musd: float | None, investment_musd: float | None, round_size_musd: float | None,
                   inputs: dict | None = None) -> dict:
    inputs = inputs or {}
    pre_money = post_money_musd - round_size_musd if post_money_musd is not None and round_size_musd is not None else None
    arr = _multiply(inputs.get("mrr_musd"), 12)
    cash = inputs.get("cash_musd")
    return {
        "post_money_musd": post_money_musd, "round_size_musd": round_size_musd, "pre_money_musd": pre_money,
        "investment_musd": investment_musd, "ownership": _divide(investment_musd, post_money_musd),
        "implied_arr_multiple": _divide(post_money_musd, arr),
        "implied_run_rate_multiple": _divide(post_money_musd, inputs.get("annualized_revenue_musd")),
        "post_round_runway_months": _divide(cash + round_size_musd if cash is not None and round_size_musd is not None else None,
                                            inputs.get("monthly_net_burn_musd")),
    }


def calculate_metrics(inputs: dict[str, float | None]) -> dict[str, float | None]:
    revenue = inputs.get("revenue_current_musd")
    previous_revenue = inputs.get("revenue_previous_musd")
    growth_ratio = _divide(revenue, previous_revenue)
    growth = growth_ratio - 1 if growth_ratio is not None else None
    cost = inputs.get("direct_cost_musd")
    gross_profit = revenue - cost if revenue is not None and cost is not None else None
    gross_margin = _divide(gross_profit, revenue)
    arr_values = [inputs.get(key) for key in ("arr_begin_musd", "arr_expansion_musd", "arr_contraction_musd", "arr_churn_musd")]
    retained_arr = max(0.0, arr_values[0] + arr_values[1] - arr_values[2] - arr_values[3]) if all(v is not None for v in arr_values) else None
    nrr = _divide(retained_arr, inputs.get("arr_begin_musd"))
    cac = _divide(inputs.get("sales_marketing_cost_musd"), inputs.get("new_customers"))
    monthly_gross_profit = _multiply(inputs.get("monthly_arpu_musd"), gross_margin)
    ltv = _divide(monthly_gross_profit, inputs.get("monthly_customer_churn_rate"))
    price, unit_cost = inputs.get("avg_selling_price_musd"), inputs.get("unit_cost_musd")
    profit_margin = inputs.get("profit_margin")
    return {
        "arr_musd": _multiply(inputs.get("mrr_musd"), 12),
        "annualized_revenue_musd": inputs.get("annualized_revenue_musd"),
        "growth_rate": growth, "nrr": nrr, "gross_margin": gross_margin,
        "cac_musd": cac, "ltv_musd": ltv, "ltv_cac": _divide(ltv, cac),
        "cac_payback_months": _divide(cac, monthly_gross_profit),
        "runway_months": _divide(inputs.get("cash_musd"), inputs.get("monthly_net_burn_musd")),
        "burn_multiple": _divide(inputs.get("net_burn_musd"), inputs.get("net_new_arr_musd")),
        "rule_of_40": growth + profit_margin if growth is not None and profit_margin is not None else None,
        "unit_gross_margin": _divide(price - unit_cost if price is not None and unit_cost is not None else None, price),
        "acceptance_rate": _divide(inputs.get("units_accepted"), inputs.get("units_delivered")),
        "production_yield": _divide(inputs.get("units_passed_qc"), inputs.get("units_produced")),
        "collection_rate": _divide(inputs.get("cash_collected_musd"), revenue),
        "after_sales_cost_ratio": _divide(inputs.get("after_sales_cost_musd"), revenue),
    }


def calculate_market_metrics(inputs: dict[str, float | None]) -> dict[str, float | None]:
    tam = _multiply(inputs.get("target_customers"), inputs.get("annual_spend_per_customer_musd"))
    sam = _multiply(tam, inputs.get("serviceable_share"))
    market_ratio = _divide(inputs.get("market_final_musd"), inputs.get("market_initial_musd"))
    market_years = inputs.get("market_years")
    return {
        "tam_musd": tam,
        "tam_top_down_musd": _multiply(inputs.get("related_market_musd"), inputs.get("addressable_share")),
        "sam_musd": sam, "som_musd": _multiply(sam, inputs.get("obtainable_share")),
        "market_cagr": market_ratio ** (1 / market_years) - 1 if market_ratio is not None and market_years is not None and market_years > 0 else None,
    }


def _multiply(left: float | None, right: float | None) -> float | None:
    return left * right if left is not None and right is not None else None


def _divide(numerator: float | None, denominator: float | None) -> float | None:
    return numerator / denominator if numerator is not None and denominator is not None and denominator > 0 else None
