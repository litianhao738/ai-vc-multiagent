"""生成适合文档和论文排版的系统架构图。"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parent
font_manager.fontManager.addfont("C:/Windows/Fonts/msyh.ttc")
plt.rcParams["font.family"] = font_manager.FontProperties(fname="C:/Windows/Fonts/msyh.ttc").get_name()
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(16, 9))
fig.patch.set_facecolor("white")
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis("off")


def box(x, y, width, height, text, *, fill="white", edge="#B5C5D6", size=10, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.015,rounding_size=0.10",
                                linewidth=1, edgecolor=edge, facecolor=fill, zorder=2))
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=size,
            color="#1C3048", weight=weight, linespacing=1.35, zorder=3)


def arrow(start, end, *, color="#77899D", style="-", curvature=0):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=11, linewidth=1.15,
                                 color=color, linestyle=style, connectionstyle=f"arc3,rad={curvature}", zorder=4))


def panel(x, title, fill, edge):
    ax.add_patch(FancyBboxPatch((x, 1.95), 3.67, 5.35, boxstyle="round,pad=0.02,rounding_size=0.14",
                                linewidth=1.2, edgecolor=edge, facecolor=fill, zorder=0))
    ax.text(x + 0.18, 7.05, title, fontsize=13.5, color="#18314E", weight="bold", va="center")


ax.text(0.32, 8.73, "AI 赛道投研多智能体系统架构", fontsize=19, color="#152E4A", weight="bold")
ax.text(15.68, 8.76, "VC RESEARCH · MULTI-AGENT", fontsize=9, color="#70869E", ha="right")

top = [
    (0.34, "用户输入", "赛道、3–5 家公司、官网、BP、可选估值"),
    (4.22, "赛道配置", "评分标准、维度权重、关键指标、风险"),
    (8.10, "研究与核查工具", "Sonar 搜索、网页抓取、BP 文本检索"),
    (11.98, "共享模型运行时", "OpenRouter / 可选 Anthropic、角色模型配置"),
]
for x, title, desc in top:
    box(x, 7.65, 3.67, 0.68, f"{title}\n{desc}", fill="#F6F8FB", edge="#CDD9E5", size=9.3)

panel(0.34, "① 研究与证据", "#F3F8FE", "#A6C9E9")
panel(4.22, "② 分析与比较", "#F3FAF6", "#A6D1BC")
panel(8.10, "③ 辩论、核查与修正", "#FFF9F1", "#E8C697")
panel(11.98, "④ 决策与交付", "#FAF7FD", "#CDBDE0")

box(0.62, 5.98, 3.10, 0.72, "赛道研究员\n市场规模 · 赛道地图 · 候选公司", fill="#E4F1FD", edge="#88B6DF")
box(0.62, 4.97, 3.10, 0.72, "公司研究员 × N\n公司底稿 · 上市状态 · 信息缺口", fill="#E4F1FD", edge="#88B6DF")
box(0.62, 3.64, 3.10, 0.86, "结构化证据\n事实 · 来源编号 · 原文摘录\n时间 · 来源性质", edge="#88B6DF")

box(4.50, 5.98, 3.10, 0.72, "公司分析师 × N\n维度评分 · 置信度 · 证据 ID", fill="#E5F5ED", edge="#82BCA0")
box(4.50, 4.97, 3.10, 0.72, "代码计算\n加权分 · TOPSIS · 敏感性", edge="#82BCA0")
box(4.50, 3.83, 3.10, 0.84, "对比分析师\n排序复核 · 正反顺序两两比较", fill="#E5F5ED", edge="#82BCA0")

box(8.38, 5.98, 3.10, 0.72, "看多方 / 看空方\n两轮论证 · 逐条回应", fill="#FFF0D8", edge="#DFB06E")
box(8.38, 4.72, 3.10, 0.83, "事实核查员\n属实 · 公司口径 · 不实 · 无法核实", fill="#FFF0D8", edge="#DFB06E")
box(8.38, 3.47, 3.10, 0.82, "退回修正与复查\n保存前后材料 · 逐条复查", edge="#DFB06E")

box(12.26, 5.98, 3.10, 0.72, "投委会主席\n结论 · 首选 · 风险 · 尽调", fill="#F0E8FA", edge="#B49DD0")
box(12.26, 4.97, 3.10, 0.72, "主席结论核查\n必要时退回修正", edge="#B49DD0")
box(12.26, 3.83, 3.10, 0.84, "代码计算与报告组装\n回报情景 · 七部分报告 · JSON", fill="#F0E8FA", edge="#B49DD0")

arrow((2.18, 7.65), (2.18, 6.73))
arrow((4.01, 7.92), (3.74, 5.33), curvature=-0.13)
arrow((6.06, 7.65), (6.06, 6.73), style="--")
arrow((0.60, 6.33), (0.60, 4.06), curvature=0.23)
arrow((2.17, 4.96), (2.17, 4.53))
arrow((6.05, 5.97), (6.05, 5.72))
arrow((6.05, 4.96), (6.05, 4.70))
arrow((3.74, 4.18), (4.48, 6.31), curvature=-0.08)
arrow((7.62, 4.25), (8.36, 6.29), curvature=-0.08)
arrow((9.93, 5.97), (9.93, 5.58))
arrow((9.93, 4.71), (9.93, 4.32), color="#C58838")
ax.text(10.10, 4.51, "发现不实", fontsize=8, color="#A46B29", va="center")
arrow((11.50, 5.16), (12.24, 6.31), curvature=-0.06)
arrow((13.81, 5.97), (13.81, 5.72))
arrow((13.81, 4.96), (13.81, 4.70))
arrow((14.85, 4.96), (14.85, 5.97), color="#9A7AB9", style="--", curvature=0.35)
ax.text(15.20, 5.40, "必要时修正", fontsize=8, color="#8666A4", rotation=90, va="center")

arrow((9.93, 3.45), (2.18, 1.98), color="#C58838", style="--", curvature=-0.15)
ax.text(5.85, 2.16, "发现不实：退回相关角色，重跑受影响阶段并逐条复查", fontsize=8.5,
        color="#A46B29", ha="center", va="center", bbox={"facecolor": "white", "edgecolor": "none", "pad": 2})

ax.text(0.36, 1.30, "共享执行能力", fontsize=10.5, weight="bold", color="#28415C")
ax.text(2.02, 1.30, "结构化提交与格式重试  ·  并行执行  ·  使用量与费用记录  ·  不完整阶段在报告中标注",
        fontsize=10, color="#50667D")
ax.text(0.36, 0.72, "实线：主要数据流    虚线：配置或修正反馈    N：公司数量（3–5）", fontsize=9.3, color="#61758A")
ax.text(15.64, 0.72, "来源匹配与抽查不等于事实真实性保证", fontsize=9.3, color="#61758A", ha="right")

fig.savefig(ROOT / "architecture.png", dpi=220, bbox_inches="tight", facecolor="white")
fig.savefig(ROOT / "architecture.svg", bbox_inches="tight", facecolor="white")
