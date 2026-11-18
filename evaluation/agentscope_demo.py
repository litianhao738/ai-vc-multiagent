"""Offline AgentScope demo, including BP tools, calculations and real PDF pages.

python -m evaluation.agentscope_demo
No model or search request is sent; every company and narrative is simulated.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from vcagents import CompanyInput, DocumentStore, SectorPipeline, SectorRequest, Settings, save_result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="outputs/agentscope-demo")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    companies = [CompanyInput(name) for name in ("Alpha", "Beta", "Gamma", "Delta", "Epsilon")]
    companies[0].financial_data = json.loads((root / "examples/financial_inputs.example.json").read_text(encoding="utf-8"))
    companies[0].post_money_musd = 50
    companies[0].investment_musd = 5
    companies[0].round_size_musd = 10
    market = json.loads((root / "examples/market_inputs.example.json").read_text(encoding="utf-8"))
    bp = DocumentStore()
    bp.add_text("demo-bp.txt", "本文件为虚构演示材料，不代表任何真实公司。Alpha 产品面向内容创作者，团队、收入与融资均为演示假设。")
    settings = Settings(mock=True, max_workers=4, holding_years=5)
    request = SectorRequest("aigc-content", companies, market_data=market,
                            core_idea="检验客户付费、持续价值捕获与估值回报，优先寻找反证")
    result = SectorPipeline(settings, {"Alpha": bp}).run(request)
    folder = save_result(result, args.out)
    filename = "report.pdf" if (folder / "report.pdf").exists() else "research-draft.pdf"
    destination = root / "output/pdf/four_main_agents_demo.pdf"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes((folder / filename).read_bytes())
    print(json.dumps({"folder": str(folder), "pdf": str(destination), "pages": result.report_pages,
                      "status": result.report_status, "agent_instances": len(result.usage),
                      "main_agents": result.main_agents, "phase_calls": len(result.handoffs),
                      "task_nodes": len(result.task_trace), "errors": result.errors,
                      "publication_issues": result.report_checks}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
