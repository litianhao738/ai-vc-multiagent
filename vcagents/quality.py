"""Locate active blockers and route them to the existing four main agents."""
from .corrections import affected_materials
from .report_checks import publication_findings, quality_finding


def owner_for(source):
    if source.startswith("quality_repair:"):
        return source.split(":", 1)[1]
    if source.startswith(("sector_researcher", "company_researcher", "research")):
        return "research"
    if source.startswith(("company_analyst", "comparator", "bull", "bear", "analysis")):
        return "analysis"
    if source.startswith(("fact_checker", "audit")):
        return "audit"
    return "lead"


# 对投资结论影响最大的事实类型，越靠前越优先列入必查清单
_COMPANY_PRIORITY = ("估值", "融资", "收入或ARR", "用户或客户", "合作或订单")
_SECTOR_PRIORITY = ("市场规模", "估值", "融资", "收入或ARR")


def audit_plan(materials: dict, per_company: int) -> dict:
    """Code, not the auditor, decides what a research audit must at least cover.

    Each company gets `per_company` claims and the sector one. The evidence with
    the highest decision weight is named explicitly, so coverage no longer
    depends on which claims a model happens to sample.
    """
    plan = {}
    for name, sheet in materials.items():
        sector = name == "sector_researcher"
        order = _SECTOR_PRIORITY if sector else _COMPANY_PRIORITY
        evidence = sorted(enumerate(sheet.get("evidence", [])), key=lambda row: (
            order.index(row[1].get("fact_type")) if row[1].get("fact_type") in order else len(order), row[0]))
        wanted = 1 if sector else per_company
        targets = [{"evidence_id": e["evidence_id"], "fact_type": e.get("fact_type", "其他"),
                    "statement": e.get("statement") or e.get("claim", "")} for _, e in evidence[:wanted]]
        # A thin sheet still needs one audited claim; it cannot need more than it contains.
        plan[name] = {"min_claims": max(1, len(targets)), "targets": targets}
    return plan


def coverage_gaps(plan: dict, claims: list[dict]) -> list[dict]:
    gaps = []
    for name, need in plan.items():
        own = [claim for claim in claims if claim.get("source_agent") == name]
        checked = {evidence_id for claim in own for evidence_id in claim.get("evidence_ids", [])}
        missing = [t["evidence_id"] for t in need["targets"] if t["evidence_id"] not in checked]
        short = max(0, need["min_claims"] - len(own))
        if short or missing:
            gaps.append({"material": name, "checked_claims": len(own), "missing_claims": short,
                         "missing_evidence": missing})
    return gaps


def coverage_messages(gaps: list[dict]) -> list[str]:
    out = []
    for gap in gaps:
        if gap["missing_claims"]:
            out.append(f"{gap['material']} 还需核查 {gap['missing_claims']} 条主张")
        if gap["missing_evidence"]:
            out.append(f"{gap['material']} 的必查证据 {'、'.join(gap['missing_evidence'])} 尚未核查"
                       "（核查后在该主张的 evidence_ids 中填写对应编号）")
    return out


def diagnose(result, memo, *, review_amount=None):
    issues = publication_findings(result, memo, review_amount=review_amount)
    research = {"sector_researcher": result.sector_brief,
                **{f"company_researcher:{name}": sheet for name, sheet in result.fact_sheets.items()}}
    for name, check in (("research_audit", result.fact_check), ("final_audit", result.decision_fact_check)):
        if not check.get("claims"):
            issues.append(quality_finding("audit_missing", "audit", "关键事实核查未完成", location=name))
        for claim in check.get("claims", []):
            if claim["verdict"] not in ("不实", "无法核实"):
                continue
            # An original statement in a research sheet is a research repair;
            # a later unsupported inference belongs to its narrative author.
            upstream = affected_materials(research, [{"claim": claim["claim"]}])
            source = next(iter(upstream), claim["source_agent"])
            issues.append(quality_finding("fact", owner_for(source), claim["evidence"],
                location=source, target=claim["claim"], claim=claim))
    for unresolved in result.unresolved_checks:
        issues.append(quality_finding("unresolved", owner_for(unresolved["source_agent"]),
            unresolved["problem"] + "：" + unresolved.get("reason", ""),
            location=unresolved["source_agent"], target=unresolved["claim"], unresolved=unresolved))
    for stage, error in result.errors.items():
        issues.append(quality_finding("stage_error", owner_for(stage), error, location=stage))
    return list({issue["issue_id"]: issue for issue in issues}.values())
