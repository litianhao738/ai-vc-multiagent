from copy import deepcopy
import unittest

from vcagents.corrections import affected_materials


class AffectedMaterialsTest(unittest.TestCase):
    def claim(self, text="2025年收入100万美元。", ids=None):
        return {"claim": text, "source_agent": "chair", "evidence_ids": ids or [],
                "claim_id": "D-1", "verdict": "不实", "correction": "正确值为10万美元。",
                "verification_source": "https://example.com/report", "verification_excerpt": "收入10万美元。"}

    def test_matches_full_statement_in_nested_text_ignoring_only_whitespace(self):
        claim = self.claim()
        materials = {"company_analyst:甲": {"scores": {"market": {"reason": "依据：2025 年收入 100 万美元。增长很快。"}}}}
        result = affected_materials(materials, [claim])
        self.assertEqual(result["company_analyst:甲"][0], {**claim, "source_agent": "company_analyst:甲"})

    def test_matches_exact_id_in_evidence_records_without_paraphrase_guessing(self):
        claim = self.claim(ids=["E-C1-1"])
        materials = {"company_researcher:甲": {"fact_sheet": "年度收入有所增加。",
                                             "evidence": [{"evidence_id": "E-C1-1", "claim": "收入情况"}]},
                     "company_researcher:乙": {"evidence": [{"evidence_id": "E-C1-10"}]},
                     "company_analyst:甲": {"scores": {"market": {"evidence_ids": ["E-C1-1"]}}}}
        self.assertEqual(set(affected_materials(materials, [claim])), {"company_researcher:甲"})

    def test_shared_source_url_does_not_imply_shared_fact(self):
        claim = self.claim()
        materials = {"sector_researcher": {"sources": [claim["verification_source"]], "summary": "行业收入增加。"}}
        self.assertEqual(affected_materials(materials, [claim]), {})

    def test_excludes_chair_and_deduplicates_same_material_same_statement(self):
        claim = self.claim()
        duplicate = {**claim, "claim_id": "D-2", "claim": "2025 年收入 100 万美元。"}
        materials = {"chair": {"thesis": claim["claim"]}, "sector_researcher": [claim["claim"], claim["claim"]]}
        result = affected_materials(materials, [claim, duplicate])
        self.assertEqual(list(result), ["sector_researcher"])
        self.assertEqual(len(result["sector_researcher"]), 1)

    def test_keeps_input_and_nested_evidence_unchanged(self):
        claim = self.claim(ids=["E-C1-1"])
        original = deepcopy(claim)
        materials = {"company_researcher:甲": {"evidence": [{"evidence_id": "E-C1-1"}]}}
        result = affected_materials(materials, [claim])
        result["company_researcher:甲"][0]["evidence_ids"].append("new")
        self.assertEqual(claim, original)

    def test_does_not_match_numeric_prefix_or_empty_statement(self):
        materials = {"research": {"fact_sheet": "收入1000，客户20。"}}
        self.assertEqual(affected_materials(materials, [self.claim("收入100")]), {})
        self.assertEqual(affected_materials(materials, [self.claim("")]), {})
        self.assertEqual(affected_materials(materials, [self.claim("收入1000")])["research"][0]["source_agent"], "research")

    def test_does_not_join_separate_text_leaves_into_one_claim(self):
        materials = {"research": {"part1": "2025年收入", "part2": "100万美元。"}}
        self.assertEqual(affected_materials(materials, [self.claim()]), {})

    def test_numeric_boundary_distinguishes_decimal_continuation_from_sentence_end(self):
        self.assertEqual(affected_materials({"research": "Revenue100.5"}, [self.claim("Revenue100")]), {})
        self.assertEqual(affected_materials({"research": "Revenue100,000"}, [self.claim("Revenue100")]), {})
        result = affected_materials({"research": "Revenue100. End."}, [self.claim("Revenue100")])
        self.assertIn("research", result)


if __name__ == "__main__":
    unittest.main()
