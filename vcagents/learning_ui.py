"""Streamlit 的本地研究记忆工作区，展开时才查询数据。"""
from datetime import date
import sqlite3

import streamlit as st

from .config import Settings
from .governor import RunStopped
from .learning import OUTCOMES, editor_version
from .learning_replay import audit_summary, draft_rule, generate_trial, replay_audit
from .llm import LLMError
from .memory import ResearchMemory


def _saved(message):
    st.session_state["learning_notice"] = message
    st.rerun()


def _editing(journal, sector, settings):
    version = editor_version(settings.role_models["lead"])
    active = journal.active_rule(sector.key, version)
    if active["id"]:
        st.write(f"当前编辑准则 · 版本 {active['id']}", active["rule"])
        if st.button("回退到上一版编辑准则", key=f"rollback_{active['id']}"):
            journal.rollback(active["id"])
            _saved("已回退。下次真实研究将采用上一版编辑准则。")
    with st.form(f"reader_feedback_{sector.key}"):
        origin = st.text_input("反馈来自哪家公司", max_chars=150)
        original = st.text_area("报告原句", max_chars=2000)
        feedback = st.text_area("你希望怎样改，为什么", max_chars=1000)
        rule = st.text_area("候选编辑准则（可选）", max_chars=800,
                            help="适用于其他公司的完整补充准则。留空先存反馈，稍后可由总控提炼。")
        if st.form_submit_button("保存改稿反馈"):
            journal.propose(sector.key, version, origin, original, feedback, rule, active["id"])
            _saved("已保存，可在反馈与版本中查看。")

    rules = journal.rules(sector.key)
    if not rules:
        return
    statuses = {"feedback": "待提炼", "candidate": "待盲评", "active": "使用中",
                "superseded": "旧版", "rolled_back": "已回退", "needs_revision": "需修改"}
    by_id = {r["id"]: r for r in rules}
    titles = {r["id"]: f"{r['origin']} · {statuses[r['status']] if r['version'] == version else '历史模型或模板'} · #{r['id']}"
              for r in rules}
    rid = st.selectbox("反馈与版本", by_id.keys(), format_func=titles.get, key=f"learning_rule_{sector.key}")
    row = by_id[rid]
    st.caption(row["original"])
    st.write(row["feedback"])
    if row["rule"]:
        st.write(row["rule"])
    old_version = row["version"] != version
    if old_version or row["status"] in ("feedback", "needs_revision") or (row["status"] == "candidate" and row["parent_id"] != active["id"]):
        with st.form(f"rule_draft_{rid}"):
            instruction = st.text_area("拟验证的完整准则", value=row["rule"], max_chars=800)
            manual = st.form_submit_button("保存为候选准则")
            automatic = st.form_submit_button("由总控提炼候选准则（调用模型）", disabled=settings.mock)
        if manual or automatic:
            if automatic:
                instruction = draft_rule(settings, row["original"], row["feedback"], active["rule"])
            if not instruction.strip():
                raise ValueError("请填写候选准则。")
            if not old_version and row["status"] == "needs_revision" and instruction.strip() == row["rule"]:
                raise ValueError("请先修订未通过盲评的准则。")
            journal.propose(sector.key, version, row["origin"], row["original"], row["feedback"], instruction, active["id"])
            _saved("候选准则已保存，可选择它开始盲评。")
        if row["status"] == "feedback":
            return

    pending = not old_version and row["status"] == "candidate" and row["parent_id"] == active["id"]
    if pending:
        st.caption("用两家其他公司的冻结公开材料盲评，其中至少一家缺乏壁垒证据。候选两次均获选且有依据后，下次成稿自动采用。")
        with st.form(f"blind_generation_{rid}"):
            company = st.text_input("盲评公司的名称", max_chars=150)
            material = st.text_area("冻结的公开材料", help="粘贴摘录、来源和日期。两份首页会收到完全相同的材料。",
                                    max_chars=6000)
            sparse = st.checkbox("此案例缺乏证明技术壁垒的证据")
            if st.form_submit_button("生成两份匿名首页（调用总控模型两次）", disabled=settings.mock):
                generate_trial(journal, rid, sector, company, material, sparse, settings)
                _saved("两份匿名首页已就绪。")
        if settings.mock:
            st.caption("关闭页面顶部的演示模式后可调用模型生成盲评样本。")
    for trial in journal.trials(rid):
        with st.container(border=True):
            st.markdown(f"**{trial['company']}**")
            st.caption(trial["material"])
            for side, col in zip(("A", "B"), st.columns(2)):
                col.markdown(f"**首页 {side}**")
                col.write(trial["text_" + side.lower()])
            if trial["choice"] is not None:
                st.caption("已评 · " + trial["choice"] + (" · 依据已核对" if trial["grounded"] else " · 依据未确认"))
            elif pending:
                with st.form(f"blind_vote_{rid}_{trial['company_key']}"):
                    choice = st.radio("你更认可哪份首页", ("A", "B", "平局"), index=None, horizontal=True)
                    grounded = st.checkbox("已核对原材料，所选首页的判断有依据")
                    if st.form_submit_button("提交盲评"):
                        promoted = journal.vote(rid, trial["company_key"], choice, grounded)
                        _saved("准则已通过盲评，下次成稿将采用。" if promoted else "已记录盲评。")


def _audit(journal, settings):
    st.caption("每组共用一段原文，分别标注应放行和应退回的表述。所有赛道共用这组写法测试。")
    with st.form("learning_audit_pair"):
        issue = st.text_input("争议点", max_chars=700)
        excerpt = st.text_area("用于核对的原文", max_chars=700)
        allow = st.text_input("应放行的表述", max_chars=700)
        reject = st.text_input("应保留退回的表述", max_chars=700)
        if st.form_submit_button("保存成对案例"):
            journal.add_audit_pair(issue, excerpt, allow, reject)
            _saved("已保存审计案例。")
    pairs = journal.audit_pairs()
    st.caption(f"现有 {len(pairs)} 组案例，一次完整回放最多发起 {(len(pairs) + 1) // 2} 次快速模型调用。")
    with st.expander("查看已标注案例"):
        for pair in pairs:
            st.write(f"#{pair['id']} · {pair['issue']} · {pair['origin']}", pair["excerpt"])
            st.caption("放行 · " + pair["equivalent"] + "　退回 · " + pair["different"])
    if st.button("回放当前审计提示词（调用快速模型）", disabled=settings.mock):
        replay_audit(journal, settings)
        _saved("回放已结束，结果已保存。")
    for record in journal.audit_replays():
        totals = audit_summary(record["results"])
        with st.expander(record["checked_at"][:19] + " · " + "　".join(f"{k} {v}" for k, v in totals.items())):
            st.caption(f"{record['model']} · 提示词 {record['prompt_hash'][:12]}")
            for row in record["results"]:
                if row["actual"] is not row["expected"]:
                    st.write(f"#{row['pair']} · " + ("应放行" if row["expected"] else "应退回"), row["reason"])


def _methods(memory, sector):
    methods = memory.recall_methods(sector, limit=6)
    if not methods:
        st.caption("这个赛道尚无通过案例核验且在有效期内的补查方法。")
    for item in methods:
        with st.container(border=True):
            st.write(item["procedure"])
            st.caption(item["limitations"])
            reuse = item["reuse"]
            st.caption(f"复用 {reuse['trials']} 份报告 · 有效 {reuse['useful']} · 未证实有效 {reuse['inconclusive']} · "
                       f"底稿无变化 {reuse['no_change']}")
    for item in memory.learning.reuse_history(sector):
        st.write(item["procedure"], item["reason"])
        if item["batch_seconds"] is not None:
            st.caption(f"本批补查耗时 {item['batch_seconds']} 秒")


def _theses(journal, sector):
    cards = journal.cards(sector)
    if not cards:
        st.caption("真实研究通过发布检查并归档后，这里会出现原始论点、反证条件和建议回看日期。")
        return
    by_id = {row["id"]: row for row in cards}
    card_id = st.selectbox("要跟踪的论点", by_id.keys(), format_func=lambda k:
                           f"{by_id[k]['company']} · {by_id[k]['as_of']} · #{k}", key=f"learning_thesis_{sector}")
    card = by_id[card_id]
    original = card["original"]
    st.write(original["claim"])
    st.caption("首次建议回看日期　" + card["review_on"])
    for label, key in (("当时的依据", "evidence"), ("最强替代解释", "alternative"),
                       ("竞品复刻条件", "replication"), ("可推翻判断的观察", "falsifier")):
        if original.get(key):
            st.write(label, original[key])
    for source in original["sources"]:
        st.write(source["url"])
    for row in journal.observations(card_id):
        st.write(f"{row['observed_on']} · {row['outcome']}", row["note"])
        if row["source"]:
            st.caption(row["source"])
    with st.form(f"thesis_observe_{card_id}"):
        outcome = st.radio("后续事实对原论点的影响", OUTCOMES, index=None, horizontal=True)
        observed = st.date_input("观察日期", value=date.today(), min_value=date.fromisoformat(card["as_of"]),
                                 max_value=date.today())
        note = st.text_area("观察到了什么", max_chars=1000)
        source = st.text_input("公开来源网址（仍无数据时可留空）", max_chars=2000)
        if st.form_submit_button("保存随访"):
            journal.observe(card_id, observed.isoformat(), outcome, note, source)
            _saved("随访已追加，保留了当初的判断。")


def memory_panel(database, sectors, sector_key, mock):
    panel = st.expander("研究记忆库", icon=":material/psychology:", key="learning_open", on_change="rerun")
    if not panel.open:
        return
    with panel:
        if notice := st.session_state.pop("learning_notice", None):
            st.success(notice)
        categories = ("改稿与盲评", "补查方法", "审计回放", "论点随访")
        tab = st.segmented_control("记忆类型", categories, default=categories[0], required=True, key="learning_tab")
        try:
            settings = Settings.load(mock=mock)
            memory = ResearchMemory(database.root / "data" / "memory.sqlite3", settings.memory_ttl_days)
            if tab == "审计回放":
                _audit(memory.learning, settings)
                return
            selected = st.selectbox("记忆所属赛道", sectors.keys(), format_func=sectors.get, key="learning_sector",
                                    index=list(sectors).index(sector_key) if sector_key in sectors else 0)
            if tab == "改稿与盲评":
                _editing(memory.learning, database.resolve_sector(selected), settings)
            elif tab == "补查方法":
                _methods(memory, selected)
            else:
                _theses(memory.learning, selected)
        except (ValueError, OSError, sqlite3.Error, LLMError, RunStopped) as exc:
            st.error(str(exc))
