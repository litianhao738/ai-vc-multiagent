"""The web form must work with company names and no numeric inputs."""
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch
from io import BytesIO

from pypdf import PdfReader
from streamlit.testing.v1 import AppTest

import vcagents
from vcagents.project_db import ProjectDatabase


APP = str(Path(__file__).resolve().parents[1] / "app.py")


def make_app():
    return AppTest.from_file(APP, default_timeout=60).run()


def fill_names(app):
    for index, name in enumerate(("Alpha", "Beta", "Gamma")):
        app.text_input(key=f"name_{index}").set_value(name)


class OptionalInputAppTest(unittest.TestCase):
    def setUp(self):
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.enterContext(patch("vcagents.project_db.ROOT", root))
        self.enterContext(patch("vcagents.pipeline.ROOT", root))

    def test_memory_panel_can_save_feedback_and_reopen_without_model_calls(self):
        app = AppTest.from_file(APP, default_timeout=60)
        app.session_state["learning_open"] = True
        app.run()
        self.assertFalse(app.exception)
        next(w for w in app.text_input if w.label == "反馈来自哪家公司").set_value("Alpha")
        next(w for w in app.text_area if w.label == "报告原句").set_value("收入规模足以证明技术壁垒。")
        next(w for w in app.text_area if w.label == "你希望怎样改，为什么").set_value("区分商业和技术证据。")
        next(w for w in app.button if w.label == "保存改稿反馈").click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any("待提炼" in str(w.options) for w in app.selectbox))
        self.assertTrue(next(w for w in app.button if "由总控提炼" in w.label).disabled)
        next(w for w in app.text_area if w.label == "拟验证的完整准则").set_value("点出能直接观察到的技术机制。")
        next(w for w in app.button if w.label == "保存为候选准则").click().run()
        self.assertTrue(any("待盲评" in str(w.options) for w in app.selectbox))
        self.assertFalse(app.exception)
        app.get_by_key("learning_tab").set_value("审计回放").run()
        self.assertTrue(next(w for w in app.button if "回放当前审计" in w.label).disabled)
        self.assertFalse(app.exception)

    def test_typed_sector_can_generate_and_reopen_its_report_in_a_new_session(self):
        name = "工业视觉 / 质量检测"
        app = AppTest.from_file(APP, default_timeout=60)
        app.session_state["sector"] = name
        app.session_state["company_count"] = 1
        app.run()
        field = app.selectbox(key="sector")
        self.assertTrue(field.proto.accept_new_options)
        self.assertIn(name, field.options)
        app.text_input(key="name_0").set_value("测试公司")
        app.button(key="generate").click().run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        result = json.loads(app.session_state["result_json"])
        self.assertEqual(result["sector_name"], name)

        record, = ProjectDatabase().reports(result["sector_key"])
        history = AppTest.from_file(APP, default_timeout=60)
        history.session_state["history_open"] = True
        history.session_state["history_sector"] = result["sector_key"]
        history.run()
        self.assertIn(name, history.selectbox(key="sector").options)
        choices = history.selectbox(key="history_report")
        self.assertEqual(len(choices.options), 1)
        self.assertIn("测试公司", choices.options[0])
        choices.set_value(record["record_path"]).run()
        self.assertFalse(history.exception)
        self.assertTrue(history.selectbox(key="history_file").value.endswith(".pdf"))
        self.assertEqual(history.get("download_button")[0].label, "下载原文件")

    def test_financial_inputs_are_blank_optional_and_in_closed_sections(self):
        app = make_app()
        self.assertFalse(app.exception)
        amounts = [widget for widget in app.number_input if widget.key != "company_count"]
        self.assertEqual(len(amounts), 14)
        for widget in amounts:
            with self.subTest(label=widget.label):
                self.assertIsNone(widget.value)
                self.assertFalse(widget.proto.required)
                self.assertIn("可选", widget.label)
        self.assertTrue(app.expander)
        self.assertTrue(all(not item.proto.expanded for item in app.expander))
        self.assertTrue(app.get_by_key("demo_mode").value)

    def test_blank_numbers_generate_a_four_agent_report_with_all_downloads(self):
        app = make_app()
        fill_names(app)
        app.button(key="generate").click().run(timeout=60)
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        result = json.loads(app.session_state["result_json"])
        self.assertEqual(result["companies"], ["Alpha", "Beta", "Gamma"])
        self.assertEqual(len(result["main_agents"]), 5)
        self.assertEqual(result["deal_metrics"], {})
        self.assertTrue(all(row["multiple"] is None and row["annualized_irr"] is None
                            and row["proceeds_musd"] is None for row in result["returns"]))
        self.assertLessEqual(len(PdfReader(BytesIO(app.session_state["report_pdf"])).pages), 10)
        self.assertEqual(result["report_max_pages"], 10)
        self.assertTrue(any("研究耗时" in item.value and "$0.0000" in item.value for item in app.caption))
        self.assertEqual(len(app.get("download_button")), 4)

    def test_single_company_and_long_report_settings_reach_the_pipeline(self):
        app = make_app()
        app.number_input(key="company_count").set_value(1)
        app.selectbox(key="report_page_limit").set_value(25).run()
        app.text_input(key="name_0").set_value("Alpha")
        app.button(key="generate").click().run(timeout=60)
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        result = json.loads(app.session_state["result_json"])
        self.assertEqual(result["companies"], ["Alpha"])
        self.assertEqual(result["report_max_pages"], 25)
        app.get_by_key("company_mode").set_value("由研究员推荐").run()
        self.assertFalse(app.exception)
        self.assertEqual(app.number_input(key="company_count").value, 1)

    def test_switching_modes_keeps_two_and_manual_report_contains_both_companies(self):
        app = make_app()
        count = app.number_input(key="company_count")
        self.assertEqual((count.min, count.max, count.step, count.value), (1, 5, 1, 3))
        count.set_value(2).run()
        app.get_by_key("company_mode").set_value("由研究员推荐").run()
        self.assertEqual(app.number_input(key="company_count").value, 2)
        self.assertTrue(any("推荐 2 家" in item.value for item in app.info))
        app.get_by_key("company_mode").set_value("手动指定").run()
        self.assertEqual(app.number_input(key="company_count").value, 2)
        app.text_input(key="name_0").set_value("甲公司")
        app.text_input(key="name_1").set_value("乙公司")
        app.button(key="generate").click().run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        result = json.loads(app.session_state["result_json"])
        self.assertEqual(result["companies"], ["甲公司", "乙公司"])
        self.assertTrue(app.session_state["report_pdf"].startswith(b"%PDF"))

    def test_optional_values_reach_the_backend_without_unit_or_zero_conversion(self):
        app = make_app()
        fill_names(app)
        app.number_input(key="valuation_0").set_value(50.0)
        app.number_input(key="investment_0").set_value(5.0)
        app.number_input(key="round_0").set_value(10.0)
        app.number_input(key="holding_years").set_value(5.0)
        app.number_input(key="exit_dilution").set_value(0.0)
        for name, value in (("乐观", 20.0), ("基线", 60.0), ("悲观", 20.0)):
            app.number_input(key=f"weight_{name}").set_value(value)
        captured = {}
        actual = vcagents.SectorPipeline

        def factory(settings, docs, emit):
            captured["settings"] = settings
            pipeline = actual(settings, docs, emit=emit)
            original = pipeline.run

            def run(request):
                captured["request"] = request
                return original(request)

            pipeline.run = run
            return pipeline

        with patch.object(vcagents, "SectorPipeline", side_effect=factory):
            app.button(key="generate").click().run(timeout=60)
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        settings, request = captured["settings"], captured["request"]
        self.assertEqual(settings.exit_dilution, 0.0, "Explicit zero dilution must remain zero")
        self.assertEqual(settings.holding_years, 5.0)
        self.assertEqual(settings.scenario_probabilities, {"乐观": .2, "基线": .6, "悲观": .2})
        self.assertEqual((request.companies[0].post_money_musd, request.companies[0].investment_musd,
                          request.companies[0].round_size_musd), (50.0, 5.0, 10.0))
        self.assertIsNone(request.companies[1].post_money_musd)
        result = json.loads(app.session_state["result_json"])
        self.assertEqual(result["deal_metrics"]["Alpha"]["ownership"], .1)
        self.assertEqual(result["deal_metrics"]["Alpha"]["pre_money_musd"], 40.0)

    def test_partial_weights_skip_only_the_optional_expected_return(self):
        app = make_app()
        fill_names(app)
        app.number_input(key="weight_乐观").set_value(20.0)
        app.button(key="generate").click().run(timeout=60)
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        self.assertTrue(any("跳过期望倍数" in item.value for item in app.info))
        result = json.loads(app.session_state["result_json"])
        self.assertTrue(all(row["probability"] is None for row in result["returns"]))
        self.assertIn("report_pdf", app.session_state)

    def test_empty_or_duplicate_names_do_not_start_model_work(self):
        for names in (("", "", ""), ("Alpha", "Alpha", "Gamma")):
            with self.subTest(names=names):
                app = make_app()
                for index, name in enumerate(names):
                    app.text_input(key=f"name_{index}").set_value(name)
                with patch.object(vcagents, "SectorPipeline") as factory:
                    app.button(key="generate").click().run()
                self.assertFalse(app.exception)
                self.assertTrue(app.error)
                factory.assert_not_called()

    def test_zero_amount_can_be_cleared_instead_of_becoming_required(self):
        app = make_app()
        fill_names(app)
        app.number_input(key="valuation_0").set_value(0.0)
        with patch.object(vcagents, "SectorPipeline") as factory:
            app.button(key="generate").click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any("清空该数值" in item.value for item in app.error))
        factory.assert_not_called()
        app.number_input(key="valuation_0").set_value(None)
        app.button(key="generate").click().run(timeout=60)
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        self.assertIn("report_pdf", app.session_state)

    def test_legacy_unknown_zero_values_migrate_to_blank_but_real_values_remain(self):
        app = AppTest.from_file(APP, default_timeout=60)
        app.session_state["valuation_0"] = 50.0
        app.session_state["investment_0"] = 0.0
        app.session_state["round_0"] = 0.0
        app.run()
        self.assertFalse(app.exception)
        self.assertEqual(app.number_input(key="valuation_0").value, 50.0)
        self.assertIsNone(app.number_input(key="investment_0").value)
        self.assertIsNone(app.number_input(key="round_0").value)

    def test_researcher_can_recommend_one_company_without_financial_inputs(self):
        app = AppTest.from_file(APP, default_timeout=60)
        app.session_state["company_mode"] = "由研究员推荐"
        app.run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.text_input), 0)
        self.assertTrue(all(widget.value is None for widget in app.number_input if widget.key != "company_count"))
        app.number_input(key="company_count").set_value(1).run()
        app.button(key="generate").click().run(timeout=60)
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        result = json.loads(app.session_state["result_json"])
        self.assertEqual(len(result["companies"]), 1)
        self.assertTrue(app.session_state["report_pdf"].startswith(b"%PDF"))
        self.assertEqual(len(result["main_agents"]), 5)

    def test_setup_failure_is_a_readable_error_instead_of_an_app_crash(self):
        app = make_app()
        app.get_by_key("demo_mode").set_value(False).run()
        fill_names(app)
        app.number_input(key="run_cost_limit").set_value(15.0)
        with patch.object(vcagents, "SectorPipeline", side_effect=RuntimeError("模拟接口配置缺失")) as factory:
            app.button(key="generate").click().run()
        self.assertEqual(factory.call_args.args[0].cost_budget_usd, 15.0)
        self.assertFalse(app.exception)
        self.assertTrue(any("无法开始研究" in item.value and "模拟接口配置缺失" in item.value for item in app.error))


if __name__ == "__main__":
    unittest.main()
