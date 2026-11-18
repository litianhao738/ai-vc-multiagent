import unittest

from vcagents.financials import (
    INPUT_KEYS, MARKET_METRIC_LABELS, METRIC_LABELS, calculate_deal, calculate_market_metrics, calculate_metrics,
    calculate_returns, parse_financial_data, parse_market_data,
)


class FinancialDataParsingTest(unittest.TestCase):
    def payload(self, inputs):
        return {"currency": "USD", "period": " 2026 年全年，月度指标采用同期间平均月值 ",
                "inputs": inputs, "sources": {key: " 用户提供的 BP 第 3 页，未经独立核实 " for key in inputs}}

    def test_normalizes_sources_and_preserves_only_supplied_inputs(self):
        data = self.payload({"revenue_current_musd": 2, "revenue_previous_musd": 1, "new_customers": 100.0})
        data["previous_period"] = " 2025 年全年 "
        result = parse_financial_data(data)
        self.assertEqual(result["period"], "2026 年全年，月度指标采用同期间平均月值")
        self.assertEqual(result["previous_period"], "2025 年全年")
        self.assertEqual(result["inputs"], {"revenue_current_musd": 2.0, "revenue_previous_musd": 1.0, "new_customers": 100})
        self.assertTrue(all(source == "用户提供的 BP 第 3 页，未经独立核实" for source in result["sources"].values()))
        self.assertIsNone(parse_financial_data(self.payload({"cash_musd": 1}))["previous_period"])
        self.assertEqual(set(METRIC_LABELS), set(calculate_metrics({})))
        self.assertIn("cash_musd", INPUT_KEYS)

    def test_rejects_wrong_currency_unknown_fields_and_missing_periods(self):
        mutations = [("currency", "CNY"), ("peroid", "2026"), ("period", " "), ("previous_period", "")]
        for key, value in mutations:
            with self.subTest(field=key):
                data = self.payload({"cash_musd": 1})
                data[key] = value
                with self.assertRaises(ValueError):
                    parse_financial_data(data)
        with self.assertRaisesRegex(ValueError, "previous_period"):
            parse_financial_data(self.payload({"revenue_previous_musd": 1}))

    def test_requires_known_inputs_and_exact_nonempty_source_mapping(self):
        with self.assertRaisesRegex(ValueError, "未知指标"):
            parse_financial_data(self.payload({"reveneu_current_musd": 1}))
        for sources in ({}, {"cash_musd": ""}, {"cash_musd": "来源", "mrr_musd": "多余来源"}):
            with self.subTest(sources=sources):
                data = self.payload({"cash_musd": 1})
                data["sources"] = sources
                with self.assertRaises(ValueError):
                    parse_financial_data(data)

    def test_rejects_missing_nonnumeric_and_nonfinite_values(self):
        for value in (None, True, False, "1", float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    parse_financial_data(self.payload({"cash_musd": value}))

    def test_count_share_margin_and_hardware_domains(self):
        invalid = [("new_customers", -1), ("units_delivered", 1.5), ("monthly_customer_churn_rate", 1.01),
                   ("profit_margin", 1.1), ("cash_musd", -1), ("unit_cost_musd", -0.1)]
        for key, value in invalid:
            with self.subTest(metric=key, value=value):
                with self.assertRaises(ValueError):
                    parse_financial_data(self.payload({key: value}))
        valid = {"new_customers": 0, "monthly_customer_churn_rate": 1, "profit_margin": -2,
                 "net_burn_musd": -1, "monthly_net_burn_musd": -0.1, "net_new_arr_musd": -0.2,
                 "units_delivered": 10, "units_accepted": 10}
        self.assertEqual(parse_financial_data(self.payload(valid))["inputs"], valid)
        with self.assertRaisesRegex(ValueError, "units_accepted"):
            parse_financial_data(self.payload({"units_delivered": 5, "units_accepted": 6}))

    def test_market_inputs_belong_to_sector_level_data(self):
        with self.assertRaisesRegex(ValueError, "--market-data"):
            parse_financial_data(self.payload({"target_customers": 12}))
        with self.assertRaisesRegex(ValueError, "--financial-data"):
            parse_market_data(self.payload({"cash_musd": 1}))
        with self.assertRaisesRegex(ValueError, "previous_period"):
            parse_market_data({**self.payload({"target_customers": 12}), "previous_period": "2024"})
        invalid = [("target_customers", 1.5), ("serviceable_share", 1.1), ("obtainable_share", -0.1),
                   ("addressable_share", 2), ("market_years", 0), ("market_years", -1), ("related_market_musd", -1)]
        for key, value in invalid:
            with self.subTest(metric=key, value=value):
                with self.assertRaises(ValueError):
                    parse_market_data(self.payload({key: value}))
        valid = {"target_customers": 12, "serviceable_share": 0, "obtainable_share": 1, "market_years": 0.5}
        self.assertEqual(parse_market_data(self.payload(valid))["inputs"], valid)

    def test_rejects_inconsistent_arr_without_rejecting_zero_after_decimal_subtraction(self):
        invalid = {"arr_begin_musd": 1, "arr_expansion_musd": 0, "arr_contraction_musd": 2, "arr_churn_musd": 0}
        with self.assertRaisesRegex(ValueError, "ARR"):
            parse_financial_data(self.payload(invalid))
        zero = {"arr_begin_musd": 1.2, "arr_expansion_musd": 0, "arr_contraction_musd": 0.4, "arr_churn_musd": 0.8}
        normalized = parse_financial_data(self.payload(zero))
        self.assertEqual(calculate_metrics(normalized["inputs"])["nrr"], 0)


class FinancialReturnsTest(unittest.TestCase):
    def test_proceeds_need_investment_amount(self):
        scenarios = [{"scenario": "基线", "exit_valuation_musd": 400}]
        self.assertEqual(calculate_returns(50, 0.36, scenarios, investment_musd=5)["scenarios"][0]["proceeds_musd"], 25.6)
        self.assertIsNone(calculate_returns(50, 0.36, scenarios)["scenarios"][0]["proceeds_musd"])

    def test_deal_terms_are_computed_only_from_supplied_inputs(self):
        deal = calculate_deal(50, 5, 10, {"mrr_musd": 0.1, "cash_musd": 2, "monthly_net_burn_musd": 0.4})
        self.assertAlmostEqual(deal["ownership"], 0.1)
        self.assertEqual(deal["pre_money_musd"], 40)
        self.assertAlmostEqual(deal["implied_arr_multiple"], 50 / 1.2)
        self.assertAlmostEqual(deal["post_round_runway_months"], 30)
        empty = calculate_deal(50, None, None)
        self.assertEqual([empty[k] for k in ("ownership", "pre_money_musd", "implied_arr_multiple", "post_round_runway_months")],
                         [None] * 4)

    def test_reference_example_and_total_loss(self):
        scenarios = [
            {"scenario": "乐观", "exit_valuation_musd": 1500, "probability": 0.2, "holding_years": 6},
            {"scenario": "基线", "exit_valuation_musd": 400, "probability": 0.5, "holding_years": 6},
            {"scenario": "悲观", "exit_valuation_musd": 0, "probability": 0.3, "holding_years": 6},
        ]
        result = calculate_returns(50, 0.36, scenarios)
        self.assertEqual([row["multiple"] for row in result["scenarios"]], [19.2, 5.12, 0])
        self.assertEqual(result["expected_multiple"], 6.4)
        self.assertAlmostEqual(result["scenarios"][0]["annualized_irr"], 19.2 ** (1 / 6) - 1, places=6)
        self.assertEqual(result["scenarios"][2]["annualized_irr"], -1)

    def test_missing_assumptions_are_not_inferred(self):
        result = calculate_returns(50, 0.5, [{"scenario": "基线", "exit_valuation_musd": 200}])
        self.assertEqual(result["scenarios"][0]["multiple"], 2)
        self.assertIsNone(result["scenarios"][0]["annualized_irr"])
        self.assertIsNone(result["expected_multiple"])
        unknown_exit = calculate_returns(50, 0.5, [{"scenario": "未知", "exit_valuation_musd": None, "probability": 1, "holding_years": 5}])
        self.assertIsNone(unknown_exit["scenarios"][0]["multiple"])
        self.assertIsNone(unknown_exit["expected_multiple"])
        unknown_price = calculate_returns(None, 0.5, [{"scenario": "基线", "exit_valuation_musd": 200}])
        self.assertIsNone(unknown_price["scenarios"][0]["multiple"])


class FinancialMetricsTest(unittest.TestCase):
    def test_metrics_use_consistent_supplied_units(self):
        inputs = {
            "annualized_revenue_musd": 700,
            "mrr_musd": 0.1, "revenue_current_musd": 1.5, "revenue_previous_musd": 1,
            "direct_cost_musd": 0.6, "arr_begin_musd": 1, "arr_expansion_musd": 0.3,
            "arr_contraction_musd": 0.1, "arr_churn_musd": 0.1,
            "sales_marketing_cost_musd": 0.3, "new_customers": 100,
            "monthly_arpu_musd": 0.001, "monthly_customer_churn_rate": 0.02,
            "cash_musd": 2, "monthly_net_burn_musd": 0.1,
            "net_burn_musd": 0.2, "net_new_arr_musd": 0.4, "profit_margin": -0.1,
            "units_delivered": 100, "units_accepted": 80, "units_produced": 125, "units_passed_qc": 100,
            "avg_selling_price_musd": 0.05, "unit_cost_musd": 0.03, "cash_collected_musd": 1.2,
            "after_sales_cost_musd": 0.15,
        }
        result = calculate_metrics(inputs)
        expected = {"arr_musd": 1.2, "annualized_revenue_musd": 700, "growth_rate": 0.5, "nrr": 1.1, "gross_margin": 0.6,
                    "cac_musd": 0.003, "ltv_musd": 0.03, "ltv_cac": 10,
                    "cac_payback_months": 5, "runway_months": 20, "burn_multiple": 0.5,
                    "rule_of_40": 0.4, "unit_gross_margin": 0.4, "acceptance_rate": 0.8,
                    "production_yield": 0.8, "collection_rate": 0.8, "after_sales_cost_ratio": 0.1}
        for key, value in expected.items():
            with self.subTest(metric=key):
                self.assertAlmostEqual(result[key], value)
        self.assertEqual(set(expected), set(METRIC_LABELS))

    def test_market_size_bottom_up_top_down_and_cagr(self):
        result = calculate_market_metrics({
            "target_customers": 1000, "annual_spend_per_customer_musd": 0.01, "serviceable_share": 0.5,
            "obtainable_share": 0.2, "related_market_musd": 40, "addressable_share": 0.3,
            "market_initial_musd": 100, "market_final_musd": 121, "market_years": 2})
        expected = {"tam_musd": 10, "tam_top_down_musd": 12, "sam_musd": 5, "som_musd": 1, "market_cagr": 0.1}
        for key, value in expected.items():
            with self.subTest(metric=key):
                self.assertAlmostEqual(result[key], value)
        self.assertEqual(set(MARKET_METRIC_LABELS), set(result))

    def test_missing_inputs_are_not_treated_as_zero(self):
        self.assertTrue(all(value is None for value in calculate_metrics({}).values()))
        result = calculate_metrics({"arr_begin_musd": 1, "arr_expansion_musd": 0.3})
        self.assertIsNone(result["nrr"])
        result = calculate_market_metrics({"target_customers": 0, "annual_spend_per_customer_musd": 1})
        self.assertEqual(result["tam_musd"], 0)
        self.assertIsNone(result["sam_musd"])
        self.assertTrue(all(value is None for value in calculate_market_metrics({}).values()))

    def test_undefined_denominators_are_not_reported_as_metrics(self):
        result = calculate_metrics({"revenue_current_musd": 1, "revenue_previous_musd": 0,
                                    "cash_musd": 1, "monthly_net_burn_musd": 0,
                                    "net_burn_musd": 1, "net_new_arr_musd": -1})
        self.assertIsNone(result["growth_rate"])
        self.assertIsNone(result["runway_months"])
        self.assertIsNone(result["burn_multiple"])


class ReturnsAnalysisTest(unittest.TestCase):
    SCENARIOS = [{"scenario": "乐观", "exit_valuation_musd": 21000}, {"scenario": "基线", "exit_valuation_musd": 6000},
                 {"scenario": "悲观", "exit_valuation_musd": 900}]

    def analyse(self, **kwargs):
        from vcagents.financials import returns_analysis
        return returns_analysis(5400, 10, 0.5, self.SCENARIOS, 5, **kwargs)

    def test_two_way_sensitivity_grids(self):
        a = self.analyse()
        self.assertEqual(a["years_grid"], [3, 5, 7])
        base5 = a["by_years"]["基线"][1]
        self.assertAlmostEqual(base5["multiple"], 6000 * 0.5 / 5400, places=4)
        self.assertAlmostEqual(base5["irr"], (6000 * 0.5 / 5400) ** (1 / 5) - 1, places=5)
        self.assertEqual([round(x["multiple"], 4) for x in a["by_dilution"]["乐观"]],
                         [round(21000 * (1 - d) / 5400, 4) for d in (0.3, 0.5, 0.7)])

    def test_net_of_fees_and_carry(self):
        net = {x["scenario"]: x for x in self.analyse()["net"]}
        gross = 21000 * 0.5 / 5400 * 10
        fees = 10 * 0.02 * 5
        self.assertAlmostEqual(net["乐观"]["carry_musd"], 0.2 * (gross - 10 - fees), places=3)
        self.assertAlmostEqual(net["乐观"]["net_multiple"], (gross - 0.2 * (gross - 10 - fees)) / (10 + fees), places=3)
        self.assertEqual(net["悲观"]["carry_musd"], 0)

    def test_liquidation_preference_protects_downside_and_converts_upside(self):
        rows = {x["scenario"]: x for x in self.analyse(preference_multiple=1, preference_stack_musd=600)["preference"]["rows"]}
        self.assertFalse(rows["悲观"]["converts"])
        self.assertAlmostEqual(rows["悲观"]["proceeds_musd"], 10)  # 9 亿美元足以偿付 6 亿优先股总额，本基金拿回 1 倍
        self.assertTrue(rows["乐观"]["converts"])
        tight = {x["scenario"]: x for x in self.analyse(preference_multiple=1, preference_stack_musd=1800)["preference"]["rows"]}
        self.assertAlmostEqual(tight["悲观"]["proceeds_musd"], 900 * 10 / 1800, places=4)  # 不足偿付时按清算额比例受偿

    def test_no_preference_without_stack_and_no_analysis_without_valuation(self):
        from vcagents.financials import returns_analysis
        self.assertIsNone(self.analyse(preference_multiple=1)["preference"])
        self.assertEqual(returns_analysis(None, 10, 0.5, self.SCENARIOS, 5), {})


if __name__ == "__main__":
    unittest.main()
