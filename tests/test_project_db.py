import json
import sqlite3
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from vcagents import AnalysisResult, CompanyInput, SectorPipeline, SectorRequest, Settings, save_result
from vcagents.brief import render_brief
from vcagents.project_db import ProjectDatabase
from vcagents.sectors import load_sector


class ProjectDatabaseTest(unittest.TestCase):
    def setUp(self):
        temporary = self.enterContext(tempfile.TemporaryDirectory())
        self.root = Path(temporary)
        self.enterContext(patch("vcagents.project_db.ROOT", self.root))
        self.db = ProjectDatabase()

    def test_free_text_is_persistent_and_presets_keep_their_rubrics(self):
        for key in ("aigc-content", "embodied-ai"):
            preset = load_sector(key)
            self.assertEqual(self.db.resolve_sector(preset.name), preset)
            self.assertEqual(self.db.resolve_sector(key), preset)
        names = ["  工业  ＡＩ / 质检  ", "工业 ai / 质检", "工业 AI / 质检"] * 2
        with ThreadPoolExecutor(4) as pool:
            profiles = list(pool.map(self.db.resolve_sector, names))
        custom = profiles[0]
        self.assertTrue(all(profile == custom for profile in profiles))
        self.assertNotIn("/", custom.key)
        self.assertEqual(len(ProjectDatabase().sector_names()), 3)
        self.assertEqual(ProjectDatabase().resolve_sector(custom.key), custom)
        self.assertAlmostEqual(sum(d.weight for d in custom.dimensions), 1)
        self.assertEqual(custom.benchmarks, ())
        with self.assertRaises(ValueError):
            self.db.resolve_sector("   ")

    def test_backfill_updates_legacy_files_and_keeps_only_metadata(self):
        report = self.root / "outputs" / "old"
        report.mkdir(parents=True)
        metadata = {"sector_key": "aigc-content", "sector_name": "AIGC 内容", "companies": ["甲公司"],
                    "finished_at": "2026-10-30T12:00:00"}
        (report / "result.json").write_text(json.dumps(metadata), encoding="utf-8")
        original = report / "report.md"
        body = "only-in-the-original-markdown"
        original.write_text(body, encoding="utf-8")
        (report / "audit.md").write_text("audit", encoding="utf-8")
        partial = self.root / "outputs" / "unfinished"
        partial.mkdir()
        (partial / "partial-result.json").write_text(json.dumps(metadata), encoding="utf-8")
        (partial / "audit.md").write_text("why", encoding="utf-8")
        evaluation = self.root / "outputs" / "evaluation"
        evaluation.mkdir()
        (evaluation / "result.json").write_text('{"passed": true}', encoding="utf-8")
        downloads = self.root / "output" / "pdf"
        downloads.mkdir(parents=True)
        (downloads / "export.pdf").write_bytes(b"%PDF")
        (downloads / "investment_report_template.pdf").write_bytes(b"%PDF")

        self.assertEqual(self.db.rebuild_report_index(), (3, []))
        self.assertEqual(self.db.rebuild_report_index(), (3, []))
        rows = {row["kind"]: row for row in self.db.reports("aigc-content")}
        self.assertEqual(set(rows), {"report", "partial"})
        self.assertEqual(rows["partial"]["files"], ["outputs/unfinished/audit.md", "outputs/unfinished/partial-result.json"])
        archive = rows["report"]
        self.assertEqual(archive["files"], ["outputs/old/report.md"])
        path = self.db.document_path(archive["files"][0], archive["record_path"])
        self.assertEqual(path.read_text(encoding="utf-8"), body)
        with self.assertRaises(ValueError):
            self.db.document_path("outputs/unfinished/audit.md", archive["record_path"])
        with closing(sqlite3.connect(self.db.path)) as connection:
            self.assertNotIn(body, "\n".join(connection.iterdump()))
            self.assertEqual(connection.execute("PRAGMA quick_check").fetchone(), ("ok",))
        original.rename(report / "revised.md")
        self.assertEqual(self.db.rebuild_report_index(), (3, []))
        updated = next(row for row in self.db.reports() if row["kind"] == "report")
        self.assertEqual(updated["files"], ["outputs/old/revised.md"])

    def test_custom_mock_report_auto_indexes_and_renders_from_its_saved_configuration(self):
        request = SectorRequest("工业视觉质检", [CompanyInput("甲公司")])
        result = SectorPipeline(Settings(mock=True, memory_enabled=False)).run(request)
        folder = save_result(result, self.root / "outputs")
        record, = ProjectDatabase().reports(result.sector_key)
        self.assertEqual((record["sector_name"], record["companies"]), ("工业视觉质检", ["甲公司"]))
        pdf, = (self.db.document_path(f, record["record_path"]) for f in record["files"] if f.endswith(".pdf"))
        self.assertTrue(pdf.read_bytes().startswith(b"%PDF"))
        saved = AnalysisResult(**json.loads((folder / "result.json").read_text(encoding="utf-8")))
        self.assertEqual(saved.sector_config(), self.db.resolve_sector(result.sector_key))
        self.assertIn("工业视觉质检", render_brief(saved))
        self.db.path.unlink()
        rebuilt = ProjectDatabase()
        self.assertEqual(rebuilt.rebuild_report_index(), (1, []))
        self.assertEqual(rebuilt.sector_names()[result.sector_key], "工业视觉质检")
        self.assertEqual(rebuilt.resolve_sector(result.sector_key), saved.sector_config())


if __name__ == "__main__":
    unittest.main()
