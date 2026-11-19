"""Source-grounded research memory. Model prose never becomes verified truth.

Facts are promoted only after a supported fact-check and a public text snapshot.
Hypotheses and corrections are separate kinds. Retrieval supplies leads, never
current-run evidence IDs; agents must reopen the source before citing it.
"""
from __future__ import annotations

from contextlib import closing
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sqlite3

from .inquiry import private_material, public_learning_case, replay_method_case
from .learning import LearningJournal


def _now():
    return datetime.now(timezone.utc)


class ResearchMemory:
    def __init__(self, path: str | Path, ttl_days: int = 30):
        if ttl_days < 1:
            raise ValueError("记忆有效期至少为一天")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.ttl_days = ttl_days
        with closing(self._connect()) as db, db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS research_memory (
                    id TEXT PRIMARY KEY, sector TEXT NOT NULL, entity TEXT NOT NULL,
                    kind TEXT NOT NULL, fact_key TEXT NOT NULL, status TEXT NOT NULL,
                    text TEXT NOT NULL, payload TEXT NOT NULL, run_id TEXT NOT NULL,
                    recorded_at TEXT NOT NULL, expires_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS memory_scope ON research_memory(sector, entity, kind, status);
                CREATE TABLE IF NOT EXISTS research_method_memory (
                    id TEXT PRIMARY KEY, sector TEXT NOT NULL, procedure TEXT NOT NULL,
                    question TEXT NOT NULL, limitations TEXT NOT NULL, cases TEXT NOT NULL,
                    recorded_at TEXT NOT NULL, expires_at TEXT NOT NULL
                );
            """)
        self.learning = LearningJournal(self._connect)

    def _connect(self):
        db = sqlite3.connect(self.path, timeout=15)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        return db

    def _put(self, sector, entity, kind, fact_key, text, payload, run_id, status="active"):
        entity = entity.strip().casefold()
        identity = json.dumps([sector, entity, kind, fact_key, text, payload], ensure_ascii=False, sort_keys=True)
        key = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        now = _now()
        with closing(self._connect()) as db, db:
            db.execute("INSERT OR IGNORE INTO research_memory VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                       (key, sector, entity, kind, fact_key, status, text,
                        json.dumps(payload, ensure_ascii=False), run_id, now.isoformat(),
                        (now + timedelta(days=self.ttl_days)).isoformat()))
        return key

    def recall(self, sector: str, entity: str = "", query: str = "", limit: int = 6, kinds: tuple[str, ...] | None = None) -> list[dict]:
        """Scope by sector AND company; stale/conflicting entries are explicit."""
        entity = entity.strip().casefold()
        sql = "SELECT * FROM research_memory WHERE sector=? AND status='active'"
        args = [sector]
        if entity:
            sql += " AND entity=?"
            args.append(entity)
        if kinds:
            if set(kinds) - {"fact", "hypothesis", "correction"}:
                raise ValueError("未知记忆类型")
            sql += " AND kind IN (" + ",".join("?" for _ in kinds) + ")"
            args.extend(kinds)
        sql += " ORDER BY recorded_at DESC LIMIT 200"
        with closing(self._connect()) as db:
            rows = db.execute(sql, args).fetchall()
        fact_variants = {}
        for row in rows:
            if row["kind"] == "fact":
                fact_variants.setdefault((row["entity"], row["fact_key"]), set()).add(row["text"])
        words = query.casefold().split()
        rows = sorted(rows, key=lambda row: (-sum(w in row["text"].casefold() for w in words),
                                            row["kind"] != "correction"))
        return [{"memory_id": row["id"], "entity": row["entity"], "kind": row["kind"],
                 "text": row["text"], "recorded_at": row["recorded_at"],
                 "stale": row["expires_at"] <= _now().isoformat(),
                 "conflicted": len(fact_variants.get((row["entity"], row["fact_key"]), ())) > 1,
                 "requires_source_read": True, "payload": json.loads(row["payload"]),
                 "usage": "仅作检索线索或历史纠错；重新读取来源并获本轮证据编号后才能支持事实"}
                for row in rows[:max(0, min(limit, 12))]]

    def remember_result(self, result) -> dict:
        """Write after checks; private uploads and unsupported prose are excluded."""
        counts = {"facts": 0, "hypotheses": 0, "corrections": 0, "methods": 0}
        if "mock" in result.models.values():
            return counts
        checks = [*result.fact_check.get("claims", []), *result.decision_fact_check.get("claims", [])]
        supported = {}
        for claim in checks:
            if (claim.get("verdict") in ("属实", "公司口径") and claim.get("excerpt_verified")
                    and claim.get("support") == "完全支持"):
                for key in claim.get("evidence_ids", []):
                    supported[key] = claim["verdict"]
        if result.errors or result.unresolved_checks or result.report_checks or result.report_status != "待人工复核的研究报告":
            supported = {}  # Drafts may retain hypotheses/corrections, never promote facts.
        run_id = getattr(result, "run_id", "") or result.started_at
        sheets = [("", result.sector_brief), *result.fact_sheets.items()]
        for entity, sheet in sheets:
            for evidence in sheet.get("evidence", []):
                if (evidence.get("evidence_id") not in supported or not evidence.get("excerpt_verified")
                        or not evidence.get("source_linked")):
                    continue
                snapshots = [s for s in evidence.get("source_snapshots", [])
                             if s.get("snapshot_id") and s.get("content_sha256")
                             and s.get("url", "").startswith(("https://", "http://"))]
                if not snapshots:
                    continue
                # A period is part of the fact key: historic revenue is not current ARR.
                fact_key = json.dumps([evidence.get("fact_type"), evidence.get("as_of"),
                                       [q.get("metric") for q in evidence.get("quantities", [])]], ensure_ascii=False)
                self._put(result.sector_key, entity, "fact", fact_key, evidence["claim"],
                          {"sources": snapshots, "excerpt": evidence.get("excerpt", ""),
                           "as_of": evidence.get("as_of", ""), "verdict": supported[evidence["evidence_id"]],
                           "source_type": evidence.get("source_type", "未知")}, run_id)
                counts["facts"] += 1
        private_context = private_material(result)
        if result.decision and not private_context:
            self._put(result.sector_key, result.decision.get("top_pick", ""), "hypothesis", "investment_thesis",
                      result.decision.get("headline", ""),
                      {"action": result.decision.get("action"), "status": result.report_status,
                       "disproof": [t.get("disproof", "") for t in result.decision.get("top_pick_thesis", [])],
                       "notice": "历史研究判断，不能作为事实证据"}, run_id)
            counts["hypotheses"] += 1
        for claim in checks:
            if private_context:
                break  # Do not persist BP-derived prose in the shared cross-run store.
            if claim.get("verdict") not in ("不实", "无法核实"):
                continue
            text = claim.get("claim") or claim.get("target", "")
            entity = claim.get("source_agent", "").split(":", 1)[-1]
            if entity not in result.fact_sheets:
                entity = ""
            self._put(result.sector_key, entity, "correction", "rejected_claim", text,
                      {"verdict": claim["verdict"], "reason": claim.get("evidence", ""),
                       "notice": "这段旧主张被否定或缺乏支持，禁止复用为事实"}, run_id)
            with closing(self._connect()) as db, db:
                db.execute("UPDATE research_memory SET status='superseded' WHERE sector=? AND entity=? AND kind='fact' AND text=?",
                           (result.sector_key, entity.strip().casefold(), text))
            counts["corrections"] += 1
        # Also preserve issues that disappeared from the last recheck.
        for repair in result.repairs:
            if private_context:
                break
            for claim in repair.get("claims", []):
                if claim.get("verdict") not in ("不实", "无法核实"):
                    continue
                entity = claim.get("source_agent", repair.get("source_agent", "")).split(":", 1)[-1]
                if entity not in result.fact_sheets:
                    entity = ""
                text = claim.get("claim", "")
                self._put(result.sector_key, entity, "correction", "repaired_claim", text,
                          {"notice": "历史主张曾被修正或删除，使用相关内容前重新核查", "verdict": claim["verdict"]}, run_id)
                with closing(self._connect()) as db, db:
                    db.execute("UPDATE research_memory SET status='superseded' WHERE sector=? AND entity=? AND kind='fact' AND text=?",
                               (result.sector_key, entity.strip().casefold(), text))
                counts["corrections"] += 1
        counts["methods"] = self.remember_methods(result)
        if public_learning_case(result):
            counts["method_reuse"] = self._remember_reuse(result)
            counts["theses"] = self.learning.freeze_thesis(result)
        return counts

    def remember_methods(self, result) -> int:
        count = 0
        for method in getattr(result, "research_methods", []):
            evaluation = replay_method_case(result, method)
            method["replay_evaluation"] = evaluation
            if not evaluation["accepted"] or method.get("origin_method_id"):
                continue
            validation = method["validation"]
            identity = json.dumps([result.sector_key, method["procedure"], method["question"], validation["limitations"]],
                                  ensure_ascii=False)
            key = hashlib.sha256(identity.encode("utf-8")).hexdigest()
            case = {"run_id": result.run_id, "reason": validation["reason"]}
            now = _now()
            with closing(self._connect()) as db, db:
                db.execute("BEGIN IMMEDIATE")
                row = db.execute("SELECT cases FROM research_method_memory WHERE id=?", (key,)).fetchone()
                cases = json.loads(row["cases"]) if row else []
                if any(item["run_id"] == result.run_id for item in cases):
                    continue
                cases.append(case)
                db.execute("INSERT OR REPLACE INTO research_method_memory VALUES (?,?,?,?,?,?,?,?)",
                    (key, result.sector_key, method["procedure"], method["question"], validation["limitations"],
                     json.dumps(cases, ensure_ascii=False), now.isoformat(),
                     (now + timedelta(days=self.ttl_days)).isoformat()))
            count += 1
        return count

    def _remember_reuse(self, result):
        recalled = {m["method_id"] for m in result.research_state.get("method_memory", [])}
        research_hash = result.material_versions.get("research", {}).get("sha256")
        if not recalled or not research_hash:
            return 0
        methods = {m["task_id"]: m for m in result.research_methods}
        accepted = {h["phase"] for h in result.handoffs if h.get("status") == "accepted"}
        count = 0
        with closing(self._connect()) as db, db:
            for row in result.research_iterations:
                phase = f"research_followup_{row.get('round')}"
                if (row.get("kind") != "followup" or phase not in accepted
                        or row.get("status") not in ("completed", "no_change")):
                    continue
                for task in row.get("tasks", []):
                    origin = task.get("origin_method_id")
                    if not origin or origin not in recalled:
                        continue
                    method = methods.get(task["task_id"], {})
                    if row["status"] == "no_change":
                        if row.get("after", {}).get("research", {}).get("sha256") != research_hash:
                            continue
                        outcome, reason = "no_change", "本次执行后整批底稿未变化"
                    elif (method.get("origin_method_id") != origin or method.get("phase") != phase
                          or method.get("research_sha256") != research_hash
                          or method.get("status") != "reviewed"):
                        continue
                    elif method["replay_evaluation"]["accepted"]:
                        outcome, reason = "useful", method["validation"]["reason"]
                    elif method["validation"].get("verdict") == "not_demonstrated":
                        outcome, reason = "inconclusive", method["validation"]["reason"]
                    else:
                        continue
                    inserted = db.execute("""INSERT OR IGNORE INTO method_reuse
                        (method_id,run_id,task_id,outcome,reason,batch_seconds)
                        SELECT id,?,?,?,?,? FROM research_method_memory WHERE id=? AND sector=?""",
                        (result.run_id, task["task_id"], outcome, reason, row.get("research_seconds"), origin,
                         result.sector_key)).rowcount
                    if inserted and outcome == "useful":
                        now = _now()
                        until = (now + timedelta(days=self.ttl_days)).isoformat()
                        db.execute("UPDATE research_method_memory SET recorded_at=?, expires_at=MAX(expires_at,?) WHERE id=?",
                                   (now.isoformat(), until, origin))
                    count += inserted
        return count

    def recall_methods(self, sector: str, query: str = "", limit: int = 3) -> list[dict]:
        """Return case-validated procedures separately from facts and old evidence IDs."""
        with closing(self._connect()) as db:
            rows = db.execute("""SELECT m.*, COUNT(DISTINCT u.run_id) AS trials,
                COUNT(DISTINCT CASE WHEN u.outcome='useful' THEN u.run_id END) AS useful,
                COUNT(DISTINCT CASE WHEN u.outcome='no_change' THEN u.run_id END) AS no_change,
                COUNT(DISTINCT CASE WHEN u.outcome='inconclusive' THEN u.run_id END) AS inconclusive
                FROM research_method_memory m LEFT JOIN method_reuse u ON u.method_id=m.id
                WHERE m.sector=? AND m.expires_at>? GROUP BY m.id ORDER BY m.recorded_at DESC LIMIT 100""",
                              (sector, _now().isoformat())).fetchall()
        words = query.casefold().split()
        rows = sorted(rows, key=lambda row: (
            -sum(word in (row["question"] + row["procedure"]).casefold() for word in words),
            -(row["useful"] - row["inconclusive"] - row["no_change"]) / row["trials"] if row["trials"] >= 2 else 0))
        return [{"method_id": row["id"], "version": row["id"][:12], "procedure": row["procedure"],
                 "question": row["question"], "limitations": row["limitations"],
                 "validated_cases": len(json.loads(row["cases"])),
                 "reuse": {key: row[key] for key in ("trials", "useful", "no_change", "inconclusive")},
                 "usage": "仅作方法线索；本案例需重新执行与核验，不作为公司事实或当前证据"}
                for row in rows[:max(0, min(limit, 6))]]
