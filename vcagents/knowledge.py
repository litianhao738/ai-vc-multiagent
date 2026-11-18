"""显式归档公开来源，保留原文快照并用 SQLite FTS5 检索。"""
from __future__ import annotations

from contextlib import closing
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import re
import sqlite3


MAX_CHUNK_CHARS = 1500
MAX_SEARCH_RESULTS = 20
CHINESE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")
WORDS = re.compile(r"[a-z0-9_]+")


class KnowledgeStore:
    """使用文件数据库；offset 为原始字符串的字符索引，end_offset 不含末字符。"""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as connection, connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS snapshots (
                    snapshot_id TEXT PRIMARY KEY,
                    url TEXT NOT NULL,
                    title TEXT NOT NULL,
                    fetched_at TEXT NOT NULL,
                    content_sha256 TEXT NOT NULL,
                    text TEXT NOT NULL,
                    UNIQUE(url, content_sha256)
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY,
                    snapshot_id TEXT NOT NULL REFERENCES snapshots(snapshot_id),
                    chunk_id INTEGER NOT NULL,
                    start_offset INTEGER NOT NULL,
                    end_offset INTEGER NOT NULL,
                    UNIQUE(snapshot_id, chunk_id)
                );
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
                    USING fts5(title, search_text, tokenize='unicode61');
            """)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    def archive(self, url: str, title: str, text: str) -> dict:
        """同 URL 同内容复用首次快照；内容变化保留新快照，不覆盖历史。"""
        url = url.strip()
        if not url:
            raise ValueError("归档来源 URL 不能为空")
        title = title.strip() or url
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        snapshot_id = hashlib.sha256((url + "\n" + content_hash).encode("utf-8")).hexdigest()
        fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with closing(self._connect()) as connection, connection:
            inserted = connection.execute(
                "INSERT OR IGNORE INTO snapshots VALUES (?, ?, ?, ?, ?, ?)",
                (snapshot_id, url, title, fetched_at, content_hash, text),
            )
            if inserted.rowcount:
                for chunk_id, (start, end) in enumerate(_chunk_spans(text)):
                    row = connection.execute(
                        "INSERT INTO chunks(snapshot_id, chunk_id, start_offset, end_offset) VALUES (?, ?, ?, ?)",
                        (snapshot_id, chunk_id, start, end),
                    )
                    connection.execute(
                        "INSERT INTO chunks_fts(rowid, title, search_text) VALUES (?, ?, ?)",
                        (row.lastrowid, _index_text(title), _index_text(text[start:end])),
                    )
            snapshot = connection.execute("SELECT * FROM snapshots WHERE snapshot_id = ?", (snapshot_id,)).fetchone()
        return _metadata(snapshot)

    def search(self, query: str, limit: int = 5) -> list[dict]:
        """按 BM25 排序；所有历史版本均可检索，最多返回 20 个原文块。"""
        terms = _query_terms(query)
        if not terms or limit <= 0:
            return []
        match = " OR ".join('"' + term + '"' for term in terms)
        with closing(self._connect()) as connection:
            rows = connection.execute("""
                SELECT snapshots.*, chunks.chunk_id, chunks.start_offset, chunks.end_offset,
                       bm25(chunks_fts, 2.0, 1.0) AS bm25_score
                FROM chunks_fts
                JOIN chunks ON chunks.id = chunks_fts.rowid
                JOIN snapshots ON snapshots.snapshot_id = chunks.snapshot_id
                WHERE chunks_fts MATCH ?
                ORDER BY bm25_score, snapshots.fetched_at DESC, chunks.id
                LIMIT ?
            """, (match, min(limit, MAX_SEARCH_RESULTS))).fetchall()
        return [{**_metadata(row), "chunk_id": row["chunk_id"], "start_offset": row["start_offset"],
                 "end_offset": row["end_offset"], "excerpt": row["text"][row["start_offset"]:row["end_offset"]],
                 "bm25_score": row["bm25_score"]} for row in rows]

    def read(self, snapshot_id: str, chunk_id: int | None = None) -> dict | None:
        """读取完整快照或指定块；没有该快照或块时返回 None。"""
        with closing(self._connect()) as connection:
            snapshot = connection.execute("SELECT * FROM snapshots WHERE snapshot_id = ?", (snapshot_id,)).fetchone()
            if snapshot is None:
                return None
            start, end = 0, len(snapshot["text"])
            if chunk_id is not None:
                chunk = connection.execute(
                    "SELECT start_offset, end_offset FROM chunks WHERE snapshot_id = ? AND chunk_id = ?",
                    (snapshot_id, chunk_id),
                ).fetchone()
                if chunk is None:
                    return None
                start, end = chunk["start_offset"], chunk["end_offset"]
        return {**_metadata(snapshot), "chunk_id": chunk_id, "start_offset": start,
                "end_offset": end, "text": snapshot["text"][start:end]}


def _metadata(row: sqlite3.Row) -> dict:
    return {key: row[key] for key in ("snapshot_id", "url", "title", "fetched_at", "content_sha256")}


def _chunk_spans(text: str):
    start = 0
    boundaries = [(match.start(), match.end()) for match in re.finditer(r"\r?\n[ \t]*\r?\n", text)]
    boundaries.append((len(text), len(text)))
    for end, next_start in boundaries:
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        for offset in range(start, end, MAX_CHUNK_CHARS):
            yield offset, min(offset + MAX_CHUNK_CHARS, end)
        start = next_start


def _index_text(text: str) -> str:
    terms = WORDS.findall(text.casefold())
    for run in CHINESE.findall(text):
        terms.extend(run)
        terms.extend(run[i:i + 2] for i in range(len(run) - 1))
    return " ".join(terms)


def _query_terms(query: str) -> list[str]:
    terms = WORDS.findall(query.casefold())
    for run in CHINESE.findall(query):
        if len(run) > 1:
            terms.extend(run[i:i + 2] for i in range(len(run) - 1))
        else:
            terms.append(run)
    return list(dict.fromkeys(terms))
