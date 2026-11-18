"""Question-driven loop, structured viewpoints and the report built around them."""
from copy import deepcopy
import json
import random
import unittest

from vcagents import CompanyInput, SectorPipeline, SectorRequest, Settings, schemas
from vcagents.brief import render_brief
from vcagents.inquiry import initial_state, process_digest, usable_evidence
from vcagents.llm import _fake
from vcagents.pipeline import AnalysisResult
from vcagents.reasoning import review_issues, review_targets
from vcagents.report import render_audit, render_memo
from vcagents.report_checks import publication_findings
from vcagents.sectors import load_sector


def request(**extra):
    return SectorRequest("aigc-content", [CompanyInput(name) for name in ("Alpha", "Beta", "Gamma")], **extra)


def evidence(eid, claim="Alpha月费10美元"):
    return {"evidence_id": eid, "claim": claim, "statement": claim, "fact_type": "其他", "as_of": "2026-11",
            "source_ref": "[C1-1]", "source_type": "公司口径", "excerpt": "Alpha costs $10.", "quantities": [],
            "source_linked": True, "excerpt_verified": True, "quote_location": "W1:s1"}


class GateTest(unittest.TestCase):
    def test_unverifiable_claims_are_excluded_instead_of_stopping_the_question_loop(self):
        pipeline = SectorPipeline(Settings(mock=True))
        original = pipeline._run
        phases = []

        def run(role, task, toolbox, agent_id, name, schema=None):
            phases.append(agent_id)
            data = original(role, task, toolbox, agent_id, name, schema)
            if agent_id == "audit_research":
                flagged = deepcopy(data["claims"][0])
                eid = pipeline.last_result.fact_sheets["Alpha"]["evidence"][0]["evidence_id"]
                flagged.update(verdict="无法核实", evidence_ids=[eid], source_agent="company_researcher:Alpha")
                data["claims"].append(flagged)
            return data

        pipeline._run = run
        result = pipeline.run(request())
        self.assertIn("lead_inquiry_0", phases, "早期公司常有无法核实项，不能因此跳过问题驱动补查")
        checkpoint = next(row for row in result.research_iterations if row["kind"] == "checkpoint")
        self.assertTrue(checkpoint["excluded_evidence_ids"])
        self.assertNotIn("研究核查仍有问题", result.research_state["stop_reason"])
        self.assertTrue(any(issue["kind"] == "fact" for issue in result.quality_issues), "发布门槛仍单独处理无法核实项")

    def test_missing_audit_still_stops_the_loop(self):
        pipeline = SectorPipeline(Settings(mock=True))
        original = pipeline._run

        def run(role, task, toolbox, agent_id, name, schema=None):
            data = original(role, task, toolbox, agent_id, name, schema)
            if agent_id == "audit_research":
                data["claims"] = []
            return data

        pipeline._run = run
        result = pipeline.run(request())
        self.assertIn("研究核查未完成", result.research_state["stop_reason"])

    def test_usable_evidence_drops_only_audit_rejected_ids(self):
        result = AnalysisResult("aigc-content", "AIGC")
        result.sector_brief = {"evidence": []}
        result.fact_sheets = {"Alpha": {"evidence": [evidence("E-C1-1"), evidence("E-C1-2", "Alpha 有100万用户")]}}
        result.fact_check = {"claims": [{"verdict": "无法核实", "evidence_ids": ["E-C1-2"]},
                                        {"verdict": "属实", "evidence_ids": ["E-C1-1"]}]}
        self.assertEqual(set(usable_evidence(result)), {"E-C1-1"})


class SchemaTest(unittest.TestCase):
    def setUp(self):
        self.sector = load_sector("aigc-content")

    def _analysis(self):
        data = _fake(schemas.analyst_schema(self.sector), random.Random(3))
        for row in [data["standout"], data["cost_structure"], *data["competitor_diffs"], *data["long_term_edge"]]:
            row["evidence_ids"] = []
        data["standout"]["evidence_ids"] = ["E-C1-1"]
        return data

    def test_long_term_edge_must_cover_each_dimension_and_cite_known_evidence(self):
        data = self._analysis()
        self.assertEqual(schemas.insight_errors({"Alpha": data}, {"E-C1-1"}), [])
        data["long_term_edge"][1]["dimension"] = data["long_term_edge"][0]["dimension"]
        data["cost_structure"]["evidence_ids"] = ["E-C9-9"]
        errors = schemas.insight_errors({"Alpha": data}, {"E-C1-1"})
        self.assertTrue(any("各一次" in e for e in errors))
        self.assertTrue(any("E-C9-9" in e for e in errors))

    def test_question_answers_are_limited_to_current_questions_and_unique(self):
        schema = schemas.decision_schema(["Alpha"], ["Q1", "Q2"])
        answers = schema["properties"]["question_answers"]
        self.assertEqual(answers["items"]["properties"]["question_id"]["enum"], ["Q1", "Q2"])
        self.assertEqual(schemas.decision_schema(["Alpha"])["properties"]["question_answers"]["maxItems"], 0)
        data = _fake(schema, random.Random(1))
        data["recommended"] = [data["top_pick"]]
        self.assertEqual(schemas.validate_submission("chair", schema, data), [])
        data["question_answers"] = [data["question_answers"][0]] * 2
        self.assertTrue(any("只能回答一次" in e for e in schemas.validate_submission("chair", schema, data)))

    def test_editor_revises_full_viewpoints_only_for_the_focus_company(self):
        schema = schemas.memo_review_schema(self.sector, ["Alpha", "Beta"], "Alpha", 2, [], ["Q1"])
        companies = schema["properties"]["companies"]["properties"]
        self.assertTrue(set(schemas.INSIGHT_KEYS) <= set(companies["Alpha"]["properties"]))
        self.assertIn("standout", companies["Beta"]["properties"])
        self.assertFalse({"competitor_diffs", "cost_structure", "long_term_edge"} & set(companies["Beta"]["properties"]))

    def test_editor_and_audit_follow_the_scope_of_the_requested_report(self):
        schema = schemas.memo_review_schema(self.sector, ["Alpha"], "Alpha", 2, [])
        decision = _fake(schema["properties"]["decision"], random.Random(2))
        self.assertEqual(set(decision["section_openings"]), {"customer", "technology", "economics", "valuation", "risks"})
        self.assertIn("report_subtitle", decision)
        targets = review_targets(decision)
        check = {"reviewed_targets": targets, "reasoning_checks": [{"target_id": key, "verdict": "有依据"} for key in targets]}
        decision["section_openings"]["economics"] = "新增一项尚未核实的收入归因。"
        self.assertEqual([issue["target_id"] for issue in review_issues(decision, check)], ["N1"])
        # A comparative report may have only one remaining eligible company.
        group = schemas.memo_review_schema(self.sector, ["Alpha"], "Alpha", 2, [], single_company=False)
        self.assertNotIn("section_openings", group["properties"]["decision"]["properties"])

    def test_core_and_question_answers_need_independent_reasoning_review(self):
        decision = _fake(schemas.decision_schema(["Alpha"], ["Q1", "Q2"]), random.Random(2))
        targets = review_targets(decision)
        self.assertIn("C1", targets)
        self.assertTrue({"Q1", "Q2"} <= set(targets))
        check = {"reviewed_targets": targets, "reasoning_checks": [
            {"target_id": key, "verdict": "有依据"} for key in targets]}
        self.assertFalse(review_issues(decision, check))
        self.assertIn("M1", targets)
        decision["technical_moat"]["replication"] += "复制时间缺少对照。"
        self.assertEqual([row["target_id"] for row in review_issues(decision, check)], ["M1"])
        check["reviewed_targets"] = review_targets(decision)
        decision["question_answers"][0]["answer"] = "改写后的结论未经复核"
        self.assertEqual([i["target_id"] for i in review_issues(decision, check)],
                         [decision["question_answers"][0]["question_id"]])


class PipelineRecordTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline = SectorPipeline(Settings(mock=True))
        cls.result = cls.pipeline.run(request(selection_rationale="收入规模领先且不自研模型"))

    def test_selection_rationale_and_prompt_versions_are_recorded(self):
        r = self.result
        self.assertEqual(r.research_state["selection_rationale"], "收入规模领先且不自研模型")
        self.assertTrue({"role:lead", "role:research", "role:analysis", "role:audit",
                         "prompts/research_checkpoint.md"} <= set(r.prompt_versions))
        self.assertTrue(all(len(v) == 12 for v in r.prompt_versions.values()))

    def test_every_company_has_structured_viewpoints(self):
        for name, analysis in self.result.analyses.items():
            with self.subTest(company=name):
                self.assertTrue(set(schemas.INSIGHT_KEYS) <= set(analysis))
                self.assertEqual(sorted(row["dimension"] for row in analysis["long_term_edge"]),
                                 sorted(schemas.LONG_TERM_DIMENSIONS))

    def test_checkpoint_receives_the_actual_standout_comparison_costs_and_durability(self):
        materials = self.pipeline._inquiry_materials(self.result)
        for name, analysis in self.result.analyses.items():
            with self.subTest(company=name):
                received = materials["analysis"][name]
                self.assertEqual(received["standout"], analysis["standout"])
                self.assertEqual(received["competitor_diffs"], analysis["competitor_diffs"])
                self.assertEqual(received["cost_structure"], analysis["cost_structure"])
                self.assertEqual(received["long_term_edge"], analysis["long_term_edge"])

    def test_bear_objection_and_its_verification_condition_reach_the_lead(self):
        result = deepcopy(self.result)
        objection = "客户在合同期内继续使用，也可能只是还没到续约日。"
        observation = "比较合同到期后可自由选择时的自愿续约情况。"
        bear = next(row for row in result.debate if row["side"] == "bear")
        bear.update(rebuttals=[objection], would_change_mind=observation)

        perspectives = self.pipeline._inquiry_materials(result)["perspectives"]
        received = next(row for row in perspectives if row["side"] == "bear")
        self.assertEqual(received["rebuttals"], [objection])
        self.assertEqual(received["would_change_mind"], observation)
        task = self.pipeline._chair_task(result)
        self.assertIn(objection, task)
        self.assertIn(observation, task)

    def test_initial_decision_receives_current_viewpoints_as_unverified_opinions(self):
        task = self.pipeline._chair_task(self.result)
        section = task.split("# 各公司当前结构化分析观点\n", 1)[1]
        note, payload = section.split("\n", 1)
        self.assertIn("待核验意见", note)
        received, _ = json.JSONDecoder().raw_decode(payload)
        self.assertEqual(set(received), set(self.result.analyses))
        for name, analysis in self.result.analyses.items():
            with self.subTest(company=name):
                self.assertEqual(received[name]["standout"], analysis["standout"])
                self.assertEqual(received[name]["competitor_diffs"], analysis["competitor_diffs"])
                self.assertEqual(received[name]["cost_structure"], analysis["cost_structure"])
                self.assertEqual(received[name]["long_term_edge"], analysis["long_term_edge"])


class ReportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = SectorPipeline(Settings(mock=True)).run(request(selection_rationale="收入规模领先且不自研模型"))

    def _populated(self):
        r = deepcopy(self.base)
        top = r.decision["top_pick"]
        index = list(r.fact_sheets).index(top) + 1
        r.fact_sheets[top]["evidence"] = [{**evidence(f"E-C{index}-1"), "source_ref": f"[C{index}-1]"}]
        r.fact_sheets[top]["sources"] = [f"[C{index}-1] 定价页 - https://alpha.example/pricing"]
        r.research_state["core_question"] = "聚合路线能否守住毛利？"
        r.research_state["questions"][0]["question"] = "上游涨价时毛利能否守住？"
        r.decision["core_answer"] = "有条件成立：取决于上游议价。"
        r.decision["question_answers"] = [{"question_id": "Q1", "answer": "短期可以，长期取决于自有模型。",
                                           "evidence": "定价页显示月费固定。", "confidence": "中",
                                           "would_change_if": "上游提价且公司未调价"}]
        a = r.analyses[top]
        a["standout"] = {"point": "工作流把多模型变成一次性交付", "mechanism": "降低创作者切换成本，从而提升付费",
                         "evidence_ids": [f"E-C{index}-1", "E-C9-9"]}
        a["long_term_edge"] = list(reversed(a["long_term_edge"]))
        r.decision["gap_impacts"] = [{"missing": "分产品毛利", "affects": "Q1：毛利能否守住", "if_obtained": "可判断定价权"}]
        return r, top, index

    def test_report_opens_with_investable_capability_and_tests_of_defensibility(self):
        r, top, index = self._populated()
        r.decision["headline"] = "经过两轮反证后，镜头标定能随生成引擎迁移。"
        r.decision["technical_moat"] = {
            "status": "形成中", "mechanism": "镜头设置随首帧传递。",
            "evidence": f"公司披露曾迁移一次预设。[E-C{index}-1]",
            "replication": "对手需要重建镜头标定。", "would_fail_if": "同底模直接生成也能达到相同控制精度。",
        }
        r.decision["top_pick_thesis"][0]["evidence"] = f"客户已经支付月费。[E-C{index}-1]"
        r.decision["action_conditions"] = ["核对同批客户的续费账单后再报价。"]
        memo = render_brief(r)
        opening = memo.split("## 二、赛道和市场", 1)[0]
        for public in (opening, render_memo(r).split("## 二、", 1)[0]):
            order = [public.index(text) for text in ("镜头标定能随生成引擎迁移", "形成中", "公司披露曾迁移一次预设", "对手需要重建", "相同控制精度", "研究对象")]
            self.assertEqual(order, sorted(order))
            self.assertNotIn("两轮反证后", public)
            for term in ("本次建议", "客户已经支付月费", "再报价"):
                self.assertNotIn(term, public)
        self.assertNotIn("### 关键问题与回答", opening)
        comparison = memo.split("## 三、", 1)[1].split("## 四、", 1)[0]
        self.assertIn("收入规模领先且不自研模型", comparison)
        self.assertIn("各公司最突出的一点及其胜出机制", comparison)
        self.assertIn(f"**{top}**：工作流把多模型变成一次性交付——降低创作者切换成本", comparison)
        deep = memo.split("## 六、", 1)[1].split("## 七、", 1)[0]
        for text in ("本次建议", "核对同批客户的续费账单后再报价", "有条件成立：取决于上游议价", "### 关键问题与回答", "上游涨价时毛利能否守住？", "什么会改变这个回答"):
            self.assertIn(text, deep)
        self.assertIn("### 竞争差异、成本与长期竞争力", deep)
        self.assertNotIn("工作流把多模型变成一次性交付", deep, "最突出的一点只在第三章出现一次")
        lines = deep.split("**长期竞争力**", 1)[1].strip().splitlines()
        rows = [line[2:].split("（", 1)[0] for line in lines[:6]]
        self.assertEqual(rows, schemas.LONG_TERM_DIMENSIONS, "按固定维度顺序展示")
        appendix = memo.split("## 七、", 1)[1]
        self.assertIn("### 信息缺口及其影响", appendix)
        self.assertNotIn("### 研究过程", memo)
        self.assertNotIn("### 多空辩论", memo)
        self.assertIn("研究过程", render_audit(r))
        self.assertIn("两轮反证后", r.decision["headline"])
        for heading in ("## 一、投资价值与技术护城河", "## 三、公司比较与排序", "## 五、共同风险", "## 七、尽调计划与附录"):
            self.assertIn(heading, memo)
        r.decision["headline"] = "报价前要看到同批客户减少返工并续费。"
        for render in (render_brief, render_memo):
            public = render(r)
            self.assertNotIn("报价前要看到", public.split("## 二、", 1)[0])
            self.assertIn(r.decision["headline"], public.split("### 交易条件与估值", 1)[1])

    def test_viewpoint_citations_use_only_public_evidence(self):
        r, top, index = self._populated()
        memo = render_brief(r)
        self.assertIn(f"[E-C{index}-1]", memo)
        self.assertIn(f"- E-C{index}-1 ·", memo)
        self.assertNotIn("E-C9-9", memo)
        self.assertFalse([f for f in publication_findings(r, memo) if f["kind"] == "citation"])

    def test_archived_question_headline_falls_back_without_rewriting_business_terms_or_sources(self):
        r, top, index = self._populated()
        r.decision.pop("technical_moat")
        r.decision["headline"] = "能否维持续费？"
        r.decision["core_answer"] = "第二轮融资后，续费仍缺独立证据。"
        r.decision["action_conditions"] = []
        r.decision["top_pick_thesis"][0]["disproof"] = "若能复验跨期续费，可进入报价。"
        r.decision["dissent"] = "反方认为，若竞品降价，品牌可能迁移。"
        raw_quote = "两轮反证后，原文仍保留这个表述。"
        r.fact_sheets[top]["evidence"][0]["excerpt"] = raw_quote
        r.decision["top_pick_thesis"][0]["evidence"] = f"已见付费。[E-C{index}-1]"
        for render in (render_brief, render_memo):
            opening = render(r).split("## 二、", 1)[0]
            self.assertIn("核心判断　工作流把多模型变成一次性交付", opening)
            self.assertIn("未单独核实技术的可复制性与持续性", opening)
            self.assertNotIn("进入报价", opening)
            self.assertNotIn("能否维持续费？", opening)
            memo = render(r)
            self.assertIn("若竞品降价，品牌可能迁移。", memo)
            self.assertNotIn("反方认为", memo)
        r.decision["headline"] = top + "值得关注。"
        self.assertIn("第二轮融资后，续费仍缺独立证据。", render_brief(r).split("## 六、", 1)[1])
        self.assertIn(raw_quote, render_memo(r).split("### 关键事实与引用", 1)[1])

        r.decision.update(action="继续尽调", headline="建议继续尽调，暂缓按本轮报价投资。主要依据是跨期客户付款。仍须证明正毛利。")
        for render in (render_brief, render_memo):
            front, rest = render(r).split("## 二、", 1)
            self.assertNotIn("主要依据是跨期客户付款", front)
            self.assertIn("主要依据是跨期客户付款。仍须证明正毛利。", rest)
        r.decision["headline"] = "建议继续尽调。但若授权到期且无法续展，应终止。"
        self.assertIn(r.decision["headline"], render_brief(r).split("### 交易条件与估值", 1)[1])

    def test_process_digest_counts_recorded_steps_only(self):
        state = initial_state({"research_questions": ["甲？", "乙？"], "working_hypothesis": "原假设"}, True)
        state["questions"].append({**state["questions"][0], "question_id": "Q3", "added": 1, "audit_verdict": "supported"})
        state["working_hypothesis"] = "新假设"
        r = AnalysisResult("aigc-content", "AIGC", research_state=state, research_iterations=[
            {"kind": "checkpoint", "status": "accepted"},
            {"kind": "followup", "status": "completed", "tasks": [{"question_id": "Q3", "method": "读定价页"}]},
            {"kind": "followup", "status": "failed", "tasks": [{"question_id": "Q1", "method": "未执行"}]}])
        digest = process_digest(r)
        self.assertEqual((digest["initial_questions"], digest["added_questions"]), (2, 1))
        self.assertEqual((digest["checkpoints"], digest["followups_completed"]), (1, 1))
        self.assertEqual([t["method"] for t in digest["followup_tasks"]], ["读定价页"])
        self.assertTrue(digest["hypothesis_changed"])

    def test_old_results_without_new_fields_still_render(self):
        r = deepcopy(self.base)
        for key in ("core_answer", "technical_moat", "selection_view", "question_answers", "gap_impacts"):
            r.decision.pop(key)
        for analysis in r.analyses.values():
            for key in schemas.INSIGHT_KEYS:
                analysis.pop(key)
        r.research_state = {}
        memo = render_brief(r)
        self.assertIn("## 一、投资价值与技术护城河", memo)
        self.assertNotIn("### 关键问题与回答", memo)
        self.assertNotIn("### 研究过程", memo)


if __name__ == "__main__":
    unittest.main()
