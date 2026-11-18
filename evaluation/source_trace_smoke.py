"""公开 API → 全文快照 → 断网检索 → 同源摘录匹配；不调用大模型。"""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vcagents.knowledge import KnowledgeStore
from vcagents.llm import ToolCall
from vcagents.tools import Toolbox


def main():
    folder = Path("outputs/evaluation") / f"source_trace_{datetime.now():%Y%m%d_%H%M%S}"
    folder.mkdir(parents=True)
    store = KnowledgeStore(folder / "sources.sqlite3")
    url = "https://api.worldbank.org/v2/country/USA/indicator/NY.GDP.MKTP.CD?date=2024&format=json"
    fetched = Toolbox(knowledge=store).execute(ToolCall("fetch", "fetch_url", {"url": url}))
    if fetched.is_error:
        raise RuntimeError(fetched.content)
    offline = Toolbox(knowledge=KnowledgeStore(folder / "sources.sqlite3"))
    with patch("vcagents.tools.httpx.get", side_effect=AssertionError("离线阶段不得联网")):
        searched = offline.execute(ToolCall("search", "search_knowledge", {"query": "GDP United States 2024"}))
        hits = json.loads(searched.content)
        chosen = hits[0]
        read = offline.execute(ToolCall("read", "read_source", {"snapshot_id": chosen["snapshot_id"], "chunk_id": chosen["chunk_id"]}))
    record = json.loads(read.content)
    evidence = offline.verify_evidence({"sources": [url], "evidence": [
        {"source_ref": url, "excerpt": record["text"], "source_type": "独立来源"}]})["evidence"][0]
    snapshot = store.read(record["snapshot_id"])
    summary = {"url": url, "snapshot_id": record["snapshot_id"], "fetched_at": record["fetched_at"],
               "content_sha256": record["content_sha256"], "archived_chars": len(snapshot["text"]),
               "hash_matches": hashlib.sha256(snapshot["text"].encode("utf-8")).hexdigest() == record["content_sha256"],
               "offline_hits": len(hits), "excerpt_verified": evidence["excerpt_verified"],
               "scope": "数据链路连通及原文追溯验证，不代表投资事实或报告质量评测"}
    (folder / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"folder": str(folder), **summary}, ensure_ascii=False))
    return 0 if summary["hash_matches"] and summary["excerpt_verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
