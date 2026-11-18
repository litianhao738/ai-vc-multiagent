"""Versioned research questions and reproducible, case-scoped method validation."""
from __future__ import annotations

from copy import deepcopy


def initial_state(plan: dict, user_selected: bool, rationale: str = "") -> dict:
    questions = list(dict.fromkeys(q.strip() for q in plan.get("research_questions", []) if q.strip()))
    if not questions:
        questions = [plan.get("core_question") or "哪些证据会改变本次投资判断？"]
    return {"core_question": plan.get("core_question", ""),
            "initial_hypothesis": plan.get("working_hypothesis", ""),
            "working_hypothesis": plan.get("working_hypothesis", ""),
            "selection_basis": "用户指定研究对象" if user_selected else "由研究主Agent推荐候选对象，推荐不等于投资结论",
            # The user's own reason for choosing these companies is a proposition to test, never a finding.
            "selection_rationale": rationale.strip(),
            "questions": [{"question_id": f"Q{i}", "question": question, "status": "open", "answer": "",
                           "alternative_explanations": [], "supporting_evidence_ids": [],
                           "contradicting_evidence_ids": [], "decision_impact": "", "missing_information": [],
                           "next_step": "", "audit_verdict": "not_reviewed"}
                          for i, question in enumerate(questions[:6], 1)],
            "research_sha256": "", "stop_reason": "", "method_memory": []}


def current_evidence(result) -> dict:
    return {e["evidence_id"]: e for sheet in [result.sector_brief, *result.fact_sheets.values()]
            for e in sheet.get("evidence", [])
            if e.get("evidence_id") and e.get("excerpt_verified") and e.get("source_linked")}


def flagged_evidence(result) -> set[str]:
    """Evidence tied to a current audit finding of false or unverifiable."""
    return {eid for claim in result.fact_check.get("claims", []) if claim.get("verdict") in ("不实", "无法核实")
            for eid in claim.get("evidence_ids", [])}


def usable_evidence(result) -> dict:
    """Located evidence the independent audit has not rejected.

    Early-stage research almost always leaves some claims unverifiable. Those
    claims are excluded from question answers instead of blocking the whole
    question-driven loop; the publication gate still handles them separately.
    """
    flagged = flagged_evidence(result)
    return {eid: e for eid, e in current_evidence(result).items() if eid not in flagged}


def process_digest(result) -> dict:
    """Plain counts of how the research actually unfolded; no model judgment involved."""
    state = result.research_state or {}
    questions = state.get("questions", [])
    rows = result.research_iterations or []
    followups = [row for row in rows if row.get("kind") == "followup"]
    return {"initial_questions": sum(not q.get("added") for q in questions),
            "added_questions": sum(bool(q.get("added")) for q in questions),
            "checkpoints": sum(row.get("kind") == "checkpoint" and row.get("status") == "accepted" for row in rows),
            "followups_completed": sum(row.get("status") == "completed" for row in followups),
            "followup_tasks": [task for row in followups if row.get("status") == "completed" for task in row.get("tasks", [])],
            "supported": sum(q.get("audit_verdict") == "supported" for q in questions),
            "conditional": sum(q.get("audit_verdict") == "conditional" for q in questions),
            "open": sum(q.get("status") in ("open", "blocked") for q in questions),
            "hypothesis_changed": bool(state.get("initial_hypothesis")) and
                                  state.get("initial_hypothesis") != state.get("working_hypothesis"),
            "initial_hypothesis": state.get("initial_hypothesis", ""),
            "working_hypothesis": state.get("working_hypothesis", ""),
            "stop_reason": state.get("stop_reason", "")}


def checkpoint_errors(data: dict, state: dict, evidence_ids: set[str], scopes: set[str]) -> list[str]:
    errors, seen = [], set()
    questions = {q["question_id"] for q in state["questions"]}
    for row in data["question_updates"]:
        qid = row["question_id"]
        if qid not in questions or qid in seen:
            errors.append(f"问题编号 {qid} 不存在或重复")
        seen.add(qid)
        refs = set(row["supporting_evidence_ids"]) | set(row["contradicting_evidence_ids"])
        if refs - evidence_ids:
            errors.append(f"{qid} 只能引用当前已核验的证据")
        if row["status"] == "answered" and (not refs or not row["answer"].strip()):
            errors.append(f"{qid} 已回答需要当前证据及简短回答；未知可保持open或blocked")
    tasks = data["follow_up_tasks"]
    if data["next_action"] == "finish" and tasks:
        errors.append("finish时follow_up_tasks应为空")
    if data["next_action"] == "research" and not tasks:
        errors.append("research时至少给出一项具体任务")
    for task in tasks:
        if task["question_id"] and task["question_id"] not in questions:
            errors.append("补查任务引用了不存在的问题编号")
        if not task["question_id"] and not task["question"].strip():
            errors.append("新问题需要具体表述")
        if task["scope"] not in scopes or not task["method"].strip() or not task["expected_impact"].strip():
            errors.append("补查任务需要已有研究范围、可执行的方法及判断影响")
    return errors


def apply_checkpoint(state: dict, data: dict, round_number: int) -> list[dict]:
    by_id = {q["question_id"]: q for q in state["questions"]}
    for row in data["question_updates"]:
        by_id[row["question_id"]].update(deepcopy(row), audit_verdict="not_reviewed")
    if data["working_hypothesis"].strip():
        state["working_hypothesis"] = data["working_hypothesis"].strip()
    tasks = []
    for i, task in enumerate(data["follow_up_tasks"], 1):
        task = deepcopy(task)
        if not task["question_id"]:
            qid = f"Q{len(state['questions']) + 1}"
            question = initial_state({"research_questions": [task["question"]]}, True)["questions"][0]
            question.update(question_id=qid, added=round_number)
            state["questions"].append(question)
            by_id[qid] = question
            task["question_id"] = qid
        task["task_id"] = f"T{round_number}-{i}"
        task["question"] = by_id[task["question_id"]]["question"]
        tasks.append(task)
    return tasks


def invalidate_answers(state: dict, methods: list[dict], research_hash: str):
    if state.get("research_sha256") and state["research_sha256"] != research_hash:
        for question in state["questions"]:
            if question["status"] == "answered":
                question.update(status="open", audit_verdict="stale", next_step="研究材料已变更，需要按当前证据重新核对")
        for method in methods:
            method["status"] = "stale"
    state["research_sha256"] = research_hash


def audit_errors(data: dict, question_ids: list[str], method_ids: list[str], evidence_ids: set[str]) -> list[str]:
    errors = []
    for key, field, expected in (("question_checks", "question_id", question_ids), ("method_checks", "method_id", method_ids)):
        if sorted(row[field] for row in data[key]) != sorted(expected):
            errors.append(f"{key}需要逐项覆盖本轮编号，不能重复或遗漏")
        for row in data[key]:
            if set(row["evidence_ids"]) - evidence_ids:
                errors.append(f"{row[field]}引用了未核验或非当前证据")
            if row["verdict"] in ("supported", "useful") and not row["evidence_ids"]:
                errors.append(f"{row[field]}通过复核必须列出实际证据")
    return errors


def replay_method_case(result, method: dict) -> dict:
    """Recheck archived execution, independent review and final publication, without an API call."""
    reasons = []
    if (result.status != "completed" or result.report_status != "待人工复核的研究报告"
            or result.errors or result.unresolved_checks or result.report_checks or result.quality_issues
            or result.models.get("lead", result.models.get("chair")) == "mock"):
        reasons.append("报告未完成正式研究复核或为演示数据")
    if any(read.get("kind") == "document" for reads in result.source_reads.values() for read in reads):
        reasons.append("包含私有上传资料，不进入共享方法记忆")
    research_hash = result.material_versions.get("research", {}).get("sha256")
    if not research_hash or method.get("research_sha256") != research_hash or method.get("status") != "reviewed":
        reasons.append("方法评价不对应当前研究版本")
    executed = any(row.get("kind") == "followup" and row.get("status") == "completed"
                   and any(task.get("task_id") == method.get("task_id") and task.get("method") == method.get("procedure")
                           for task in row.get("tasks", [])) for row in result.research_iterations)
    if not executed or not any(h.get("phase") == method.get("phase") and h.get("status") == "accepted" for h in result.handoffs):
        reasons.append("找不到对应的实际补查任务与成功交接")
    validation = method.get("validation", {})
    if validation.get("verdict") != "useful" or not validation.get("reason", "").strip():
        reasons.append("独立审计未确认本案例的方法价值")
    if not validation.get("limitations", "").strip():
        reasons.append("缺少方法局限与适用条件")
    questions = {q["question_id"]: q for q in result.research_state.get("questions", [])}
    question = questions.get(method.get("question_id"), {})
    if question.get("audit_verdict") != "supported":
        reasons.append("对应问题尚未得到独立支持")
    supported = {eid for check in (result.fact_check, result.decision_fact_check) for claim in check.get("claims", [])
                 if claim.get("verdict") in ("属实", "公司口径") and claim.get("excerpt_verified")
                 and claim.get("support") == "完全支持" for eid in claim.get("evidence_ids", [])}
    refs = validation.get("evidence_ids", [])
    evidence = current_evidence(result)
    if not refs or any(eid not in supported or eid not in evidence for eid in refs):
        reasons.append("方法缺少当前通过事实核查的证据")
    if any(not any(s.get("snapshot_id") and s.get("content_sha256") and s.get("url", "").startswith(("https://", "http://"))
                   for s in evidence.get(eid, {}).get("source_snapshots", [])) for eid in refs):
        reasons.append("方法证据缺少公共原文快照")
    return {"method_id": method.get("method_id", ""), "accepted": not reasons, "reasons": reasons}
