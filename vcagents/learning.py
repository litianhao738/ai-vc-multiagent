"""投研方法、编辑盲评、审计用例和论点随访。与 ResearchMemory 共用本地 SQLite。"""
from __future__ import annotations

from contextlib import closing
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import secrets
import unicodedata
from urllib.parse import urlsplit

from .config import ROOT


def _now():
    return datetime.now(timezone.utc).isoformat()


def _text(value, label, limit):
    if not isinstance(value, str) or not 0 < len(value.strip()) <= limit:
        raise ValueError(f"{label}须填写 1 至 {limit} 字。")
    return value.strip()


def _company_key(name):
    return " ".join(unicodedata.normalize("NFKC", name).casefold().split())


def editor_template():
    return (ROOT / "prompts" / "lead_editing.md").read_text(encoding="utf-8")


def editor_version(model, template=None):
    from .main_roles import LEAD
    return hashlib.sha256("\n".join((model, LEAD.brief, editor_template() if template is None else template)).encode()).hexdigest()


def editing_prompt(template, rule):
    if not rule:
        return template
    return template + "\n\n经盲评采用的本赛道编辑准则（仅调整表达，沿用当前证据和审计要求）\n" + rule


AUDIT_EXAMPLES = (
    ("金额与指标", "甲公司2025年确认收入为100万美元。",
     "甲公司2025年确认收入为1,000,000美元。", "甲公司2025年ARR为1,000,000美元。"),
    ("标点与负号", "甲公司毛利率为-20.3%。",
     "甲公司毛利率为－２０．３％；", "甲公司毛利率为20.3%。"),
)
OUTCOMES = ("得到支持", "被削弱", "仍无数据")


class LearningJournal:
    def __init__(self, connect):
        self._connect = connect
        with closing(connect()) as db, db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS editing_rules (
                    id INTEGER PRIMARY KEY, sector TEXT NOT NULL, version TEXT NOT NULL,
                    parent_id INTEGER REFERENCES editing_rules(id), origin TEXT NOT NULL, original TEXT NOT NULL,
                    feedback TEXT NOT NULL, rule TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'candidate',
                    created_at TEXT NOT NULL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS one_active_editor ON editing_rules(sector, version)
                    WHERE status='active';
                CREATE UNIQUE INDEX IF NOT EXISTS distinct_edit_feedback ON editing_rules
                    (sector, version, IFNULL(parent_id, 0), origin, original, feedback, rule);
                CREATE TABLE IF NOT EXISTS editing_trials (
                    rule_id INTEGER NOT NULL REFERENCES editing_rules(id), company_key TEXT NOT NULL,
                    company TEXT NOT NULL, material TEXT NOT NULL, sparse INTEGER NOT NULL,
                    text_a TEXT NOT NULL, text_b TEXT NOT NULL, candidate_side TEXT NOT NULL,
                    choice TEXT, grounded INTEGER, created_at TEXT NOT NULL,
                    PRIMARY KEY(rule_id, company_key)
                );
                CREATE TABLE IF NOT EXISTS method_reuse (
                    method_id TEXT NOT NULL REFERENCES research_method_memory(id), run_id TEXT NOT NULL,
                    task_id TEXT NOT NULL, outcome TEXT NOT NULL, reason TEXT NOT NULL, batch_seconds REAL,
                    PRIMARY KEY(method_id, run_id, task_id)
                );
                CREATE TABLE IF NOT EXISTS audit_pairs (
                    id INTEGER PRIMARY KEY, issue TEXT NOT NULL, excerpt TEXT NOT NULL,
                    equivalent TEXT NOT NULL, different TEXT NOT NULL, origin TEXT NOT NULL,
                    UNIQUE(excerpt, equivalent, different)
                );
                CREATE TABLE IF NOT EXISTS audit_replays (
                    id INTEGER PRIMARY KEY, checked_at TEXT NOT NULL, model TEXT NOT NULL,
                    prompt_hash TEXT NOT NULL, results TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS thesis_cards (
                    id INTEGER PRIMARY KEY, sector TEXT NOT NULL, run_id TEXT NOT NULL, company TEXT NOT NULL,
                    company_key TEXT NOT NULL, as_of TEXT NOT NULL, review_on TEXT NOT NULL,
                    original TEXT NOT NULL, UNIQUE(sector, run_id)
                );
                CREATE INDEX IF NOT EXISTS thesis_scope ON thesis_cards(sector, company_key, as_of);
                CREATE TABLE IF NOT EXISTS thesis_observations (
                    id INTEGER PRIMARY KEY, card_id INTEGER NOT NULL REFERENCES thesis_cards(id),
                    observed_on TEXT NOT NULL, outcome TEXT NOT NULL, note TEXT NOT NULL, source TEXT NOT NULL,
                    recorded_at TEXT NOT NULL, UNIQUE(card_id, observed_on, outcome, note, source)
                );
            """)
            db.executemany("""INSERT OR IGNORE INTO audit_pairs(issue,excerpt,equivalent,different,origin)
                VALUES (?,?,?,?,'内置示例')""", AUDIT_EXAMPLES)

    def active_rule(self, sector, version):
        with closing(self._connect()) as db:
            row = db.execute("SELECT id, rule FROM editing_rules WHERE sector=? AND version=? AND status='active'",
                             (sector, version)).fetchone()
        return dict(row) if row else {"id": None, "rule": ""}

    def rules(self, sector, version=None):
        query, args = "SELECT * FROM editing_rules WHERE sector=?", [sector]
        if version is not None:
            query += " AND version=?"
            args.append(version)
        with closing(self._connect()) as db:
            return [dict(row) for row in db.execute(query + " ORDER BY id DESC", args)]

    def propose(self, sector, version, origin, original, feedback, rule, parent_id):
        values = (_text(origin, "反馈来源公司", 150), _text(original, "原句", 2000),
                  _text(feedback, "改稿意见", 1000), _text(rule, "编辑准则", 800) if rule.strip() else "")
        with closing(self._connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            active = db.execute("SELECT id,rule FROM editing_rules WHERE sector=? AND version=? AND status='active'",
                                (sector, version)).fetchone()
            if parent_id != (active["id"] if active else None):
                raise ValueError("当前编辑版本已变化，请基于新版本重新提案。")
            if active and active["rule"] == values[-1]:
                raise ValueError("候选准则与当前版本相同。")
            args = (sector, version, parent_id, *values)
            db.execute("""INSERT OR IGNORE INTO editing_rules
                (sector,version,parent_id,origin,original,feedback,rule,status,created_at) VALUES (?,?,?,?,?,?,?,?,?)""",
                       (*args, "candidate" if values[-1] else "feedback", _now()))
            return db.execute("""SELECT id FROM editing_rules WHERE sector=? AND version=? AND parent_id IS ?
                AND origin=? AND original=? AND feedback=? AND rule=?""", args).fetchone()[0]

    @staticmethod
    def _pending(db, rule_id):
        row = db.execute("SELECT * FROM editing_rules WHERE id=?", (rule_id,)).fetchone()
        if not row or row["status"] != "candidate":
            raise ValueError("请选择待盲评的候选准则。")
        active = db.execute("SELECT id, rule FROM editing_rules WHERE sector=? AND version=? AND status='active'",
                            (row["sector"], row["version"])).fetchone()
        if row["parent_id"] != (active["id"] if active else None):
            raise ValueError("盲评基线已变化，请基于当前版本重新提案。")
        return row, active["rule"] if active else ""

    def _trial_input(self, db, rule_id, company, material):
        company = _text(company, "盲评公司", 150)
        material = _text(material, "冻结的公开材料", 6000)
        rule, baseline = self._pending(db, rule_id)
        if _company_key(rule["origin"]) == _company_key(company):
            raise ValueError("请用反馈来源以外的公司验证准则。")
        if db.execute("SELECT 1 FROM editing_trials WHERE rule_id=? AND (company_key=? OR material=?)",
                      (rule_id, _company_key(company), material)).fetchone():
            raise ValueError("这家公司或这份材料已生成盲评，可直接打开已有样本。")
        return dict(rule), baseline

    def prepare_trial(self, rule_id, company, material):
        with closing(self._connect()) as db:
            return self._trial_input(db, rule_id, company, material)

    def save_trial(self, rule_id, company, material, sparse, baseline, candidate):
        a, b = (_text(baseline, "基线输出", 4000), _text(candidate, "候选输出", 4000))
        side = secrets.choice(("A", "B"))
        if side == "A":
            a, b = b, a
        with closing(self._connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            self._trial_input(db, rule_id, company, material)
            db.execute("""INSERT INTO editing_trials
                (rule_id,company_key,company,material,sparse,text_a,text_b,candidate_side,created_at)
                VALUES (?,?,?,?,?,?,?,?,?)""", (rule_id, _company_key(company), company.strip(), material.strip(),
                                                bool(sparse), a, b, side, _now()))

    def trials(self, rule_id):
        with closing(self._connect()) as db:
            return [dict(row) for row in db.execute("""SELECT company_key,company,material,sparse,text_a,text_b,
                choice,grounded FROM editing_trials WHERE rule_id=? ORDER BY created_at""", (rule_id,))]

    def vote(self, rule_id, company_key, choice, grounded):
        if choice not in ("A", "B", "平局") or type(grounded) is not bool:
            raise ValueError("请选择盲评结果并核对事实依据。")
        with closing(self._connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            rule, _ = self._pending(db, rule_id)
            updated = db.execute("""UPDATE editing_trials SET choice=?,grounded=? WHERE rule_id=? AND
                company_key=? AND choice IS NULL AND (text_a<>text_b OR ?='平局')""",
                                 (choice, grounded, rule_id, company_key, choice))
            if not updated.rowcount:
                raise ValueError("样本已评、已失效，或两份输出相同只能选平局。")
            trials = db.execute("SELECT * FROM editing_trials WHERE rule_id=?", (rule_id,)).fetchall()
            if any(t["choice"] and (t["choice"] != t["candidate_side"] or not t["grounded"]) for t in trials):
                db.execute("UPDATE editing_rules SET status='needs_revision' WHERE id=?", (rule_id,))
                return False
            if (len(trials) < 2 or not any(t["sparse"] for t in trials)
                    or not all(t["choice"] == t["candidate_side"] and t["grounded"] for t in trials)):
                return False
            db.execute("UPDATE editing_rules SET status='superseded' WHERE id=?", (rule["parent_id"],))
            db.execute("UPDATE editing_rules SET status='active' WHERE id=?", (rule_id,))
            return True

    def rollback(self, rule_id):
        with closing(self._connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            rule = db.execute("SELECT parent_id FROM editing_rules WHERE id=? AND status='active'", (rule_id,)).fetchone()
            if not rule:
                raise ValueError("当前版本已变化。")
            db.execute("UPDATE editing_rules SET status='rolled_back' WHERE id=?", (rule_id,))
            db.execute("UPDATE editing_rules SET status='active' WHERE id=?", (rule["parent_id"],))

    def add_audit_pair(self, issue, excerpt, equivalent, different):
        values = tuple(_text(value, label, 700) for value, label in zip(
            (issue, excerpt, equivalent, different), ("退回原因", "同一段原文", "应放行的表述", "应保留退回的表述")))
        if values[2] == values[3]:
            raise ValueError("应放行与应退回的表述相同。")
        with closing(self._connect()) as db, db:
            db.execute("""INSERT OR IGNORE INTO audit_pairs(issue,excerpt,equivalent,different,origin)
                VALUES (?,?,?,?,'人工标注')""", values)

    def audit_pairs(self):
        with closing(self._connect()) as db:
            return [dict(row) for row in db.execute("SELECT * FROM audit_pairs ORDER BY id")]

    def save_audit_replay(self, model, prompt_hash, results):
        with closing(self._connect()) as db, db:
            db.execute("INSERT INTO audit_replays(checked_at,model,prompt_hash,results) VALUES (?,?,?,?)",
                       (_now(), model, prompt_hash, json.dumps(results, ensure_ascii=False)))

    def audit_replays(self):
        with closing(self._connect()) as db:
            return [{**dict(row), "results": json.loads(row["results"])} for row in
                    db.execute("SELECT * FROM audit_replays ORDER BY id DESC LIMIT 10")]

    def reuse_history(self, sector):
        with closing(self._connect()) as db:
            return [dict(row) for row in db.execute("""SELECT m.procedure,m.question,u.run_id,u.outcome,u.reason,u.batch_seconds
                FROM method_reuse u JOIN research_method_memory m ON m.id=u.method_id
                WHERE m.sector=? ORDER BY u.rowid DESC LIMIT 30""", (sector,))]

    def freeze_thesis(self, result):
        decision = result.decision
        company = decision.get("top_pick")
        moat = decision.get("technical_moat") or {}
        first = next(iter(decision.get("top_pick_thesis", [])), {})
        claim = decision.get("headline") or first.get("judgment")
        falsifier = moat.get("would_fail_if") or first.get("disproof")
        if not company or not claim or not falsifier or not result.finished_at:
            return 0
        as_of = date.fromisoformat(result.finished_at[:10])
        from .inquiry import current_evidence
        evidence = current_evidence(result)
        refs = {eid for row in result.decision_fact_check.get("reasoning_checks", [])
                if row.get("target_id") in ("M1", "T1") for eid in row.get("evidence_ids", [])}
        sources = [{k: s[k] for k in ("url", "content_sha256", "snapshot_id") if k in s}
                   for eid in sorted(refs) for s in evidence.get(eid, {}).get("source_snapshots", [])
                   if s.get("content_sha256") and s.get("url", "").startswith(("https://", "http://"))]
        original = {"claim": claim, "evidence": moat.get("evidence") or first.get("evidence", ""),
                    "alternative": decision.get("dissent", ""), "replication": moat.get("replication", ""), "falsifier": falsifier,
                    "strength": moat.get("status", ""), "sources": sources}
        with closing(self._connect()) as db, db:
            return db.execute("""INSERT OR IGNORE INTO thesis_cards
                (sector,run_id,company,company_key,as_of,review_on,original) VALUES (?,?,?,?,?,?,?)""",
                (result.sector_key, result.run_id, company, _company_key(company), as_of.isoformat(),
                 (as_of + timedelta(days=90)).isoformat(), json.dumps(original, ensure_ascii=False))).rowcount

    def observe(self, card_id, observed_on, outcome, note, source=""):
        observed = date.fromisoformat(observed_on)
        note = _text(note, "本次观察", 1000)
        source = source.strip()
        if outcome not in OUTCOMES:
            raise ValueError("未知随访结论。")
        if source or outcome != "仍无数据":
            url = urlsplit(source)
            if url.scheme not in ("http", "https") or not url.hostname or url.username or len(source) > 2000:
                raise ValueError("支持或削弱论点需提供公开来源网址。")
        with closing(self._connect()) as db, db:
            card = db.execute("SELECT as_of FROM thesis_cards WHERE id=?", (card_id,)).fetchone()
            if not card or not date.fromisoformat(card[0]) <= observed <= date.today():
                raise ValueError("观察日期应介于原判断日期和今天之间。")
            db.execute("""INSERT OR IGNORE INTO thesis_observations
                (card_id,observed_on,outcome,note,source,recorded_at) VALUES (?,?,?,?,?,?)""",
                       (card_id, observed.isoformat(), outcome, note, source, _now()))

    def cards(self, sector):
        with closing(self._connect()) as db:
            return [{**dict(row), "original": json.loads(row["original"])} for row in db.execute(
                "SELECT * FROM thesis_cards WHERE sector=? ORDER BY review_on,id DESC", (sector,))]

    def observations(self, card_id):
        with closing(self._connect()) as db:
            return [dict(row) for row in db.execute("SELECT * FROM thesis_observations WHERE card_id=? "
                                                    "ORDER BY observed_on DESC,id DESC", (card_id,))]

    def thesis_leads(self, sector, companies):
        """Return company-scoped observations; peers contribute only a question for re-testing."""
        keys = {_company_key(name) for name in companies}
        if not keys:
            return []
        marks = ",".join("?" for _ in keys)
        with closing(self._connect()) as db:
            rows = db.execute(f"""SELECT c.*,o.outcome,o.note,o.source,o.observed_on FROM thesis_cards c
                LEFT JOIN thesis_observations o ON o.id=(SELECT id FROM thesis_observations WHERE card_id=c.id
                    ORDER BY observed_on DESC,id DESC LIMIT 1)
                WHERE sector=? AND (company_key IN ({marks}) OR (o.source<>'' AND o.outcome IN ('得到支持','被削弱')))
                ORDER BY company_key IN ({marks}) DESC, c.as_of DESC,c.id DESC LIMIT 6""",
                              (sector, *keys, *keys)).fetchall()
        leads = []
        for row in rows:
            original = json.loads(row["original"])
            lead = {"card_id": row["id"], "原案例": row["company"], "待重新检验的问题": original["falsifier"][:500]}
            if row["company_key"] in keys:
                lead.update(原判断=original["claim"][:400], 判断日期=row["as_of"], 随访=row["outcome"] or "尚未观察",
                            观察日期=row["observed_on"], 随访线索=(row["note"] or "")[:500], 待重读来源=row["source"])
            leads.append(lead)
        return leads
