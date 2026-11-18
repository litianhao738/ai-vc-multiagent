"""运行配置：每个角色用哪个模型、接口密钥、流程参数。"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# 角色 → 模型。默认都走 OpenRouter；写成 "anthropic:<模型>" 表示直连 Anthropic API。
# 可在项目根目录的 models.yaml 里覆盖，不用改代码。
DEFAULT_ROLE_MODELS: dict[str, str] = {
    "lead": "google/gemini-4-carbon",
    "research": "moonshotai/kimi-k3.1-pro",
    "analysis": "anthropic/claude-fable-5.2",
    "bear": "openai/gpt-6.2-astra",
    "audit": "z-ai/glm-5.3-prime",
    "sector_researcher": "google/gemini-4-argon",
    "company_researcher": "deepseek/deepseek-v4.1-flash",
    "company_analyst": "qwen/qwen3.8-max-0902",
    "comparator": "openai/gpt-6.1-sol",
    "bull": "moonshotai/kimi-k3.1-pro",
    "fact_checker": "z-ai/glm-5.3",
    "chair": "anthropic/claude-opus-5.5",
    "search": "perplexity/sonar",  # 联网搜索工具，不是智能体
    "rejection_review": "google/gemini-3.1-flash-lite",  # 按需核对引用写法和成稿金额
}

DEFAULT_EXIT_DILUTION = 0.5
MAX_COMPANIES = 5
DEFAULT_REPORT_MAX_PAGES = 10
REPORT_PAGE_CAP = 25  # 用户可显式选择的最长篇幅，含来源与附录

DEFAULT_FUND_PROFILE = (
    "HT Investment，一家面向全球市场的风险投资机构，在 3 个国家设有办公室，一只 5 亿美元基金已完成首轮募集。"
    "投资阶段覆盖种子轮到 Pre-IPO，也参与管理层收购、分拆和并购。"
    "重点关注人工智能科技与内容娱乐的交叉领域，包括 AI、机器人、消费电子、科技与 SaaS、社交与娱乐。"
    "据公开报道，已投组合包括演语科技（Liblib，2026 年 6 月跟投 B+ 轮）和极致上下文（Pexo，2025 年 10 月）；"
    "评估新项目时要说明它与已投组合公司的竞争或协同关系。"
)


@dataclass
class Settings:
    openrouter_api_key: str = ""
    anthropic_api_key: str = ""  # 只有某个角色配置成 "anthropic:..." 时才需要
    role_models: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_ROLE_MODELS))
    mock: bool = False  # 演示模式：不调用任何模型和搜索
    debate_rounds: int | None = None  # core 固定两轮，expanded 默认一轮
    max_workers: int = 12
    research_lanes: int = 3  # 每家公司按产品团队、经营融资、竞争风险分工
    max_agent_steps: int = 14
    # 真实运行中研究员和核查员都因 14 步上限失败过；它们要读原文，其他角色不联网
    role_max_steps: dict[str, int] = field(default_factory=lambda: {"fact_checker": 18, "sector_researcher": 18,
                                                                    "company_researcher": 18,
                                                                    "lead": 14, "research": 28, "analysis": 14, "audit": 20})
    exit_dilution: float | None = None  # 投资到退出期间持股被稀释的比例；None 时用系统默认值并在报告中标注
    holding_years: float | None = None
    scenario_probabilities: dict[str, float] = field(default_factory=dict)
    management_fee: float = 0.02  # 年管理费率，用于扣费后回报的近似测算
    carry: float = 0.2  # 超额收益分成比例
    liquidation_preference: float | None = None  # 优先清算倍数（不参与分配型）；None 时不测算优先清算权
    pairwise_checks: bool = False
    knowledge_path: str = str(ROOT / "data" / "sources.sqlite3")
    memory_path: str = str(ROOT / "data" / "memory.sqlite3")
    memory_enabled: bool = True
    memory_ttl_days: int = 30
    report_max_pages: int = DEFAULT_REPORT_MAX_PAGES  # 整份对外报告，含来源与附录
    publication_repair_rounds: int = 2  # 发布检查后的定向修复上限；不创建额外 Agent
    research_refinement_rounds: int = 1  # 保留原流程，成稿前最多追加一轮问题驱动补查；0关闭，最大2
    rejection_review_enabled: bool = True
    rejection_review_max_calls: int = 6  # 每份任务最多复核六组退回；相同输入复用结论
    # 运行管控：整份任务共用，不随 Agent 或阶段数量放大
    run_timeout_seconds: float | None = 5400  # 整份任务总时限；None 或 0 表示不限
    retry_budget: int | None = 12  # 整份任务可用的接口重试总次数；None 表示只受单次请求上限约束
    cost_budget_usd: float | None = None  # 费用硬上限（美元）；只统计接口返回了费用的请求
    provider_concurrency: int = 8  # 同一供应商同时在途的请求数
    provider_rpm: dict[str, int] = field(default_factory=dict)  # 供应商 → 每分钟请求数，如 {"perplexity": 60}
    audit_claims_per_company: int = 2  # 研究核查：每家公司至少核查的关键主张数
    fund_profile: str = DEFAULT_FUND_PROFILE

    @classmethod
    def load(cls, models_file: str | Path | None = None, **overrides) -> "Settings":
        settings = cls(
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        )
        path = Path(models_file) if models_file else ROOT / "models.yaml"
        if path.exists():
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            unknown = set(data) - set(DEFAULT_ROLE_MODELS)
            if unknown:
                raise ValueError(f"{path.name} 里有未知角色：{'、'.join(sorted(unknown))}")
            settings.role_models.update({k: str(v) for k, v in data.items()})
            for core, legacy in {"lead": "chair", "research": "sector_researcher",
                                 "analysis": "company_analyst", "audit": "fact_checker"}.items():
                if core not in data and legacy in data:
                    settings.role_models[core] = str(data[legacy])
        for key, value in overrides.items():
            if value is not None:
                setattr(settings, key, value)
        return settings
