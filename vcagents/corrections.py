"""根据明确文本或证据编号定位需要纠正的上游材料。"""
from __future__ import annotations

from copy import deepcopy
import re


def affected_materials(materials: dict, claims: list) -> dict[str, list[dict]]:
    affected = {}
    for source, material in materials.items():
        if source == "chair":
            continue
        texts, evidence_ids = _material_content(material)
        seen = set()
        for claim in claims:
            statement = re.sub(r"\s+", "", claim["claim"])
            if statement in seen:
                continue
            cited = set(claim.get("evidence_ids", []))
            if _contains_statement(texts, statement) or cited.intersection(evidence_ids):
                revised = deepcopy(claim)
                revised["source_agent"] = source
                affected.setdefault(source, []).append(revised)
                seen.add(statement)
    return affected


def _material_content(value) -> tuple[list[str], set[str]]:
    texts, evidence_ids = [], set()
    if isinstance(value, str):
        texts.append(re.sub(r"\s+", "", value))
    elif isinstance(value, list):
        for item in value:
            child_texts, child_ids = _material_content(item)
            texts.extend(child_texts)
            evidence_ids.update(child_ids)
    elif isinstance(value, dict):
        evidence = value.get("evidence")
        if isinstance(evidence, list):
            evidence_ids.update(record["evidence_id"] for record in evidence
                                if isinstance(record, dict) and isinstance(record.get("evidence_id"), str))
        for item in value.values():
            child_texts, child_ids = _material_content(item)
            texts.extend(child_texts)
            evidence_ids.update(child_ids)
    return texts, evidence_ids


def _contains_statement(texts: list[str], statement: str) -> bool:
    if not statement:
        return False
    # 数字前后缀相同不是同一个数，例如主张末尾100与正文1000。
    pattern = ((r"(?<![\d+\-])(?<!\d[.,])" if statement[0].isdigit() else "") + re.escape(statement)
               + (r"(?!\d|[.,]\d)" if statement[-1].isdigit() else ""))
    return any(re.search(pattern, text) for text in texts)
