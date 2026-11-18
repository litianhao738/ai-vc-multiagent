"""赛道评分与研究配置。内置 YAML、项目数据库及已归档报告共用同一套校验。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from .config import ROOT
from .financials import METRIC_LABELS

SECTOR_DIR = ROOT / "sectors"


@dataclass(frozen=True)
class Dimension:
    key: str
    name: str
    weight: float
    rubric: str


@dataclass(frozen=True)
class Sector:
    key: str
    name: str
    description: str
    valuation_basis: str
    dimensions: tuple[Dimension, ...]
    key_metrics: tuple[str, ...]
    metrics: tuple[str, ...]  # 适用于该赛道商业模式、由代码计算的经营指标
    must_check_sources: tuple[str, ...]
    sector_risks: tuple[str, ...]
    benchmarks: tuple[str, ...]

    def brief(self) -> str:
        """拼进提示词的赛道说明，各智能体按同一份标准工作。"""
        dims = "\n".join(f"- {d.name}（{d.key}，权重 {d.weight:.0%}）：{d.rubric}" for d in self.dimensions)
        return (
            f"## 赛道：{self.name}\n{self.description}\n\n"
            f"### 评分维度和标准（1-10 分）\n{dims}\n\n"
            f"### 关键指标\n" + "\n".join(f"- {m}" for m in self.key_metrics) + "\n\n"
            f"### 必查信息来源\n" + "\n".join(f"- {s}" for s in self.must_check_sources) + "\n\n"
            f"### 赛道特有风险\n" + "\n".join(f"- {r}" for r in self.sector_risks) + "\n\n"
            f"### 参照公司\n{'、'.join(self.benchmarks) or '由赛道研究员补充'}\n\n"
            f"### 估值参照\n{self.valuation_basis}"
        )


def list_sectors(directory: Path = SECTOR_DIR) -> list[str]:
    return sorted(p.stem for p in directory.glob("*.yaml"))


def load_sector(key: str, directory: Path = SECTOR_DIR) -> Sector:
    if key not in list_sectors(directory):
        raise ValueError(f"没有赛道配置 {key}，可选：{'、'.join(list_sectors(directory))}")
    path = directory / f"{key}.yaml"
    return sector_from_dict(yaml.safe_load(path.read_text(encoding="utf-8")))


def sector_from_dict(data: dict) -> Sector:
    label = data["name"]
    dims = tuple(Dimension(d["key"], d["name"], float(d["weight"]), d["rubric"]) for d in data["dimensions"])
    if not dims:
        raise ValueError(f"{label} 没有评分维度")
    if len({d.key for d in dims}) != len(dims):
        raise ValueError(f"{label} 的维度 key 有重复")
    metrics = tuple(data.get("metrics") or METRIC_LABELS)
    unknown = set(metrics) - set(METRIC_LABELS)
    if unknown:
        raise ValueError(f"{label} 的 metrics 有未知指标：{'、'.join(sorted(unknown))}")
    total = sum(d.weight for d in dims)
    if abs(total - 1) > 0.01:
        raise ValueError(f"{label} 的维度权重之和是 {total:.2f}，应为 1")
    return Sector(
        key=data["key"],
        name=data["name"],
        description=data.get("description", ""),
        valuation_basis=data.get("valuation_basis", ""),
        dimensions=dims,
        key_metrics=tuple(data.get("key_metrics") or []),
        metrics=metrics,
        must_check_sources=tuple(data.get("must_check_sources") or []),
        sector_risks=tuple(data.get("sector_risks") or []),
        benchmarks=tuple(data.get("benchmarks") or []),
    )
