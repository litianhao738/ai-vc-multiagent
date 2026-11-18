"""本地赛道目录与报告索引。报告正文、附件和来源快照各自保留在原处。"""
from __future__ import annotations

from contextlib import closing
from dataclasses import asdict, replace
from datetime import datetime
from hashlib import sha256
import json
import logging
from pathlib import Path
import sqlite3
import unicodedata

from .config import ROOT
from .sectors import SECTOR_DIR, Sector, list_sectors, load_sector, sector_from_dict


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _display_name(value: str) -> str:
    name = " ".join(unicodedata.normalize("NFKC", value).split()) if isinstance(value, str) else ""
    if not name or not name.isprintable():
        raise ValueError("请填写可见的赛道名称。")
    return name


class ProjectDatabase:
    def __init__(self, root: Path | None = None):
        self.root = (root or ROOT).resolve()
        self.path = self.root / "data" / "project.sqlite3"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as db, db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS sectors (
                    key TEXT PRIMARY KEY, name TEXT NOT NULL, normalized_name TEXT NOT NULL,
                    origin TEXT NOT NULL, profile_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS sector_names ON sectors(normalized_name);
                CREATE TABLE IF NOT EXISTS reports (
                    record_path TEXT PRIMARY KEY, sector_key TEXT, sector_name TEXT NOT NULL,
                    companies_json TEXT NOT NULL, status TEXT NOT NULL, saved_at TEXT NOT NULL,
                    kind TEXT NOT NULL, run_id TEXT, pages INTEGER, files_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS reports_by_sector ON reports(sector_key, saved_at DESC);
            """)
            for key in list_sectors():
                preset = load_sector(key)
                db.execute("""
                    INSERT INTO sectors VALUES (?, ?, ?, 'preset', ?)
                    ON CONFLICT(key) DO UPDATE SET name=excluded.name, normalized_name=excluded.normalized_name,
                        origin='preset', profile_json=excluded.profile_json
                    WHERE sectors.profile_json <> excluded.profile_json OR sectors.origin <> 'preset'
                """, (preset.key, preset.name, _display_name(preset.name).casefold(), _json(asdict(preset))))

    def _connect(self):
        db = sqlite3.connect(self.path, timeout=15)
        db.row_factory = sqlite3.Row
        return db

    def sector_names(self) -> dict[str, str]:
        with closing(self._connect()) as db:
            return dict(db.execute("SELECT key, name FROM sectors ORDER BY origin='preset' DESC, normalized_name, key"))

    def resolve_sector(self, value: str) -> Sector:
        name = _display_name(value)
        normalized = name.casefold()
        with closing(self._connect()) as db, db:
            row = db.execute("""
                SELECT profile_json FROM sectors WHERE key=? OR normalized_name=?
                ORDER BY (key=?) DESC, (origin='preset') DESC LIMIT 1
            """, (normalized, normalized, normalized)).fetchone()
            if row is None:
                # Text is never used as a filename. Concurrent submissions of equivalent names
                # get the same persistent key and use the spelling saved by the first writer.
                key = "custom-" + sha256(normalized.encode("utf-8")).hexdigest()[:24]
                profile = replace(load_sector("general", SECTOR_DIR / "templates"), key=key, name=name)
                db.execute("""INSERT INTO sectors VALUES (?, ?, ?, 'custom', ?)
                              ON CONFLICT(key) DO NOTHING""", (key, name, normalized, _json(asdict(profile))))
                row = db.execute("SELECT profile_json FROM sectors WHERE key=?", (key,)).fetchone()
        return sector_from_dict(json.loads(row["profile_json"]))

    def _reference(self, path: Path) -> str:
        path = path.resolve()
        return (path.relative_to(self.root) if path.is_relative_to(self.root) else path).as_posix()

    def index_report(self, record: Path) -> bool:
        record = record.resolve()
        relative = self._reference(record)
        if not record.is_file():
            return False
        if record.suffix.lower() == ".pdf":
            data = {}
            files = [relative]
            kind, fallback = "pdf", "独立 PDF"
        elif record.name in ("result.json", "partial-result.json"):
            data = json.loads(record.read_text(encoding="utf-8-sig"))
            if not isinstance(data, dict) or not data.get("sector_key"):
                return False  # Some evaluation fixtures also use the name result.json.
            kind = "partial" if record.name == "partial-result.json" else "report"
            fallback = "未完成" if kind == "partial" else "历史报告"
            if kind == "report" and "mock" in (data.get("models") or {}).values():
                fallback = "演示稿"
            files = [self._reference(path) for path in sorted(record.parent.iterdir())
                     if path.suffix.lower() in (".pdf", ".md") and path.name not in ("audit.md", "revision-summary.md")
                     and path.is_file() and path.resolve().parent == record.parent]
            if kind == "partial":
                audit = record.with_name("audit.md")
                if audit.is_file() and audit.resolve().parent == record.parent:
                    files.append(self._reference(audit))
                files.append(relative)
        else:
            return False
        date = data.get("finished_at") or data.get("started_at")
        date = date or datetime.fromtimestamp(record.stat().st_mtime).isoformat(timespec="seconds")
        values = (relative, data.get("sector_key"), data.get("sector_name") or data.get("sector_key") or "未标注赛道",
                  _json(data.get("companies") or []), data.get("report_status") or fallback, date, kind,
                  data.get("run_id"), data.get("report_pages"), _json(files))
        with closing(self._connect()) as db, db:
            profile = data.get("sector_profile")
            if profile and profile.get("key") == data.get("sector_key"):
                sector = sector_from_dict(profile)
                db.execute("""INSERT INTO sectors VALUES (?, ?, ?, 'custom', ?)
                              ON CONFLICT(key) DO NOTHING""",
                           (sector.key, sector.name, _display_name(sector.name).casefold(), _json(asdict(sector))))
            db.execute("""INSERT INTO reports VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(record_path) DO UPDATE SET sector_key=excluded.sector_key,
                    sector_name=excluded.sector_name, companies_json=excluded.companies_json, status=excluded.status,
                    saved_at=excluded.saved_at, kind=excluded.kind, run_id=excluded.run_id,
                    pages=excluded.pages, files_json=excluded.files_json""", values)
        return True

    def rebuild_report_index(self, archive: Path | None = None) -> tuple[int, list[str]]:
        archive = (archive or self.root / "outputs").resolve()
        records = {p.resolve() for p in archive.rglob("*result.json")
                   if p.name in ("result.json", "partial-result.json")}
        pdf_dir = self.root / "output" / "pdf"
        records.update(p.resolve() for p in pdf_dir.glob("*.pdf") if "template" not in p.stem.casefold())
        with closing(self._connect()) as db:
            tracked = [r[0] for r in db.execute("SELECT record_path FROM reports")]
        errors = []
        count = 0
        for relative in tracked:
            record = self.root / relative
            if record.is_file():
                records.add(record.resolve())
            else:
                with closing(self._connect()) as db, db:
                    db.execute("DELETE FROM reports WHERE record_path=?", (relative,))
        for record in sorted(records):
            try:
                count += self.index_report(record)
            except (OSError, ValueError, sqlite3.Error) as exc:
                errors.append(f"{record.name} ({record.parent.name}) — {exc}")
        return count, errors

    def reports(self, sector_key: str | None = None) -> list[dict]:
        query = "SELECT * FROM reports"
        params = ()
        if sector_key is not None:
            query += " WHERE sector_key=?"
            params = (sector_key,)
        with closing(self._connect()) as db:
            rows = db.execute(query + " ORDER BY saved_at DESC, record_path", params).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["companies"] = json.loads(item.pop("companies_json"))
            item["files"] = json.loads(item.pop("files_json"))
            result.append(item)
        return result

    def document_path(self, reference: str, record_path: str) -> Path:
        record = (self.root / record_path).resolve()
        path = (self.root / reference).resolve()
        document = path.suffix.lower() in (".pdf", ".md") or path.name == "partial-result.json"
        same_export = path == record if record.suffix.lower() == ".pdf" else path.parent == record.parent
        if not document or not same_export:
            raise ValueError("此文件不属于所选归档记录。")
        return path


def index_export(record: Path) -> None:
    """Register the saved export, including a user-selected output directory."""
    try:
        ProjectDatabase().index_report(record)
    except (OSError, ValueError, sqlite3.Error) as exc:
        logging.getLogger(__name__).warning("报告已保存，索引暂未更新，可用 --index-reports 补录。%s", exc)
