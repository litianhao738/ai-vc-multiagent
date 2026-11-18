import unittest

from vcagents.quantities import claim_quantities, fact_quantities, source_values, unmatched


class QuantityNormalizationTest(unittest.TestCase):
    def test_reference_numbers_are_not_business_quantities(self):
        self.assertEqual(unmatched("GVR 2026 年预计 9.464 亿美元[E-28]",
                                   "The market will reach USD 946.4 million in 2026."), [])
        self.assertEqual(unmatched("融资4亿美元[E-C1-1][S-12][U-1]", "raised $400 million"), [])
        self.assertEqual(unmatched("融资40亿美元[E-C1-1]", "raised $400 million"), ["40亿"])

    def test_real_product_model_count_wordings_are_checked(self):
        self.assertEqual([q.value for q in fact_quantities("接入12个第三方视频模型[E-C1-20]")], [12])
        self.assertEqual([q.value for q in fact_quantities("聚合35个以上的AI模型")], [35])

    def test_signed_returns_preserve_their_sign_without_changing_date_parts(self):
        self.assertEqual([q.value for q in claim_quantities("IRR -11.1%")], [-11.1])
        self.assertEqual([q.value for q in claim_quantities("2026-10-30")], [2026, 10, 30])
        self.assertEqual(unmatched("增长50%", "growth was -50%"), ["50"])

    def test_chinese_scale_matches_english_scale(self):
        self.assertEqual(unmatched("2024 年为 148 亿美元，2030 年 801 亿美元",
                                   "valued at USD 14.8 billion in 2024 ... USD 80.1 billion by 2030"), [])
        self.assertEqual(unmatched("超 3,200 万用户、近 100 万付费订阅用户",
                                   "more than 32 million users and nearly 1 million paying subscribers"), [])
        self.assertEqual(unmatched("3.15 亿美元 E 轮，投后估值 53 亿美元", "a $315M Series E at a $5.3B valuation"), [])

    def test_wrong_scale_is_still_rejected(self):
        self.assertEqual(unmatched("收入 148 亿美元", "revenue of $148 million"), ["148 亿"])
        self.assertEqual(unmatched("融资 4 亿美元", "raised $40 million"), ["4 亿"])
        self.assertEqual(unmatched("收入1美元", "Revenue was $1 million."), ["1"])
        self.assertEqual(unmatched("收入1美元", "Revenue was one million dollars."), ["1"])
        self.assertEqual(unmatched("收入1b美元", "Revenue was $1 million."), ["1b"])
        self.assertEqual(unmatched("收入1b美元", "Revenue was $1 billion."), [])

    def test_month_names_and_number_words(self):
        self.assertEqual(unmatched("2026 年 6 月 ARR 突破 2 亿美元，8 个月翻倍",
                                   "June 25, 2026 — surpassed $200 million in ARR, doubling in eight months"), [])
        self.assertEqual(unmatched("前 4 个月突破 5 亿美元", "in the first four months of 2026, surpassed $500 million"), [])
        self.assertEqual(unmatched("twenty-five", ""), [])
        self.assertIn(25.0, source_values("twenty-five customers"))

    def test_chinese_numerals_in_source(self):
        self.assertEqual(unmatched("完成 1000 万美元天使轮", "完成超千万美元天使轮"), [])
        self.assertEqual(unmatched("ARR 突破 100 万美元，上线 3 个月", "上线三个月，ARR 突破百万美元"), [])
        self.assertIn(3.5e8, source_values("三亿五千万"))

    def test_names_and_round_letters_are_not_scaled(self):
        self.assertEqual([q.value for q in claim_quantities("2026 B轮")], [2026])
        self.assertEqual(claim_quantities("B2B 与 Veo3"), [])
        self.assertEqual([q.value for q in claim_quantities("$5.3B")], [5.3e9])

    def test_chinese_comma_between_years_is_not_a_thousands_separator(self):
        self.assertEqual([q.value for q in claim_quantities("2026，2025年底")], [2026, 2025])
        self.assertEqual([q.value for q in claim_quantities("3,200 万")], [3.2e7])

    def test_year_absent_from_source_is_reported(self):
        self.assertEqual(unmatched("2026 年 7 月上线", "自 7 月正式上线以来"), ["2026"])


if __name__ == "__main__":
    unittest.main()
