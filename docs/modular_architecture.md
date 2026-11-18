# 当前框架分模块图（2026-11-17）

AgentScope 2.0.11，四个固定主Agent：lead、research、analysis、audit。蓝色为原有能力，绿色为本次新增，橙色为本次增强，灰色为此前加入的辅助复核。比较基准是本轮问题驱动研究增强之前的四主Agent框架。

```mermaid
flowchart TB
    USER[用户输入：赛道、公司、资料、核心思想] --> PLAN
    subgraph M1[① 总控模块：同一个 lead]
        PLAN[原有：核心问题与研究议程]
        CHECK{新增：证据是否足够？<br/>补查是否可能改变判断？}
        WRITE[原有：投资决策与统一定稿]
    end
    subgraph M2[② 研究与分析模块：research + analysis]
        RESEARCH[增强：按问题取证与指定范围补查]
        ANALYSIS[增强：竞争性解释、成本分析<br/>只重算受影响公司的评分]
    end
    subgraph M3[③ 审计与发布模块：同一个 audit + 代码服务]
        FACT[原有：研究事实与原文引用核查]
        METHODCHECK[新增：问题回答与方法复核<br/>仅实际追加研究后执行，否则跳过]
        FINAL[原有：成稿事实与核心推论复核]
        GATE{原有：引用、计算、实际页数等发布检查}
        FIX[原有：定位责任Agent、定向修复<br/>更新受影响下游后重新复核]
        DRAFT[原有：达到修复上限时保留草稿]
        KIMI[此前新增：Kimi K2.5辅助API<br/>只判断标点、写法、等值换算是否误退回]
    end
    subgraph M4[④ 共享状态与记忆模块：代码与存储服务]
        BASE[原有：原文、事实、历史判断、纠错记忆<br/>材料版本、哈希与运行预算]
        STATE[新增：问题工作台<br/>问题、假设、证据、缺口、判断变化]
        METHODS[新增：案例验证的方法经验<br/>按适用条件供后续任务参考]
    end
    PLAN --> RESEARCH
    RESEARCH --> ANALYSIS
    RESEARCH --> FACT
    ANALYSIS --> CHECK
    FACT --> CHECK
    CHECK -->|值得补查，预算与轮数允许| RESEARCH
    CHECK -->|结束追加研究| METHODCHECK
    METHODCHECK --> WRITE
    WRITE --> FINAL
    FINAL --> GATE
    GATE -->|通过| REPORT[原有七部分投资报告：含来源最多25页]
    GATE -->|未通过且有修复额度| FIX
    FIX --> FINAL
    GATE -->|修复额度用完| DRAFT
    RESEARCH -.字段退回时.-> KIMI
    BASE -.材料与管控支撑.-> RESEARCH
    STATE -.- CHECK
    REPORT -.方法案例检查合格后入库.-> METHODS
    METHODS -.下次规划仅作方法线索.-> PLAN
    classDef old fill:#eaf2ff,stroke:#4b75b9,color:#172b4d
    classDef added fill:#e8f7ee,stroke:#278653,color:#173d29
    classDef enhanced fill:#fff2df,stroke:#c27b15,color:#583c12
    classDef previous fill:#f1f2f4,stroke:#7b8390,color:#303846
    class PLAN,WRITE,FACT,FINAL,GATE,FIX,DRAFT,BASE,REPORT old
    class CHECK,METHODCHECK,STATE,METHODS added
    class RESEARCH,ANALYSIS enhanced
    class KIMI previous
```

图为模块与主要流程；同名角色的不同阶段复用同一AgentScope上下文。补查方法复核是有实际追加研究时才执行的可选阶段，分析与研究事实审计可以并行。

| 模块 | 原有能力 | 本次修改提升 |
| --- | --- | --- |
| 总控 | 核心思想、研究议程、决策、定稿 | 成稿前判断补查价值；支持新问题、竞争性解释、假设更新及停止条件 |
| 研究与分析 | 搜索、读取原文、公司底稿、统一评分、比较与代码计算 | 只补查指定范围；公司补查保留其他公司底稿和评分；强化产品差异、成本口径及优势持续性分析 |
| 审计与发布 | 事实和核心推论审计、定向修复、七部分报告与25页出口 | 新增问题与实际方法复核；研究版本改变使旧回答和方法评价失效 |
| 共享状态与记忆 | 原文、事实、判断、纠错、材料板及运行管控 | 问题状态、执行与判断变化记录；方法候选通过执行、独立审计、原文和发布状态检查后入库 |

默认最多追加一轮研究，每轮最多两项任务；可配置0至2轮。补查额度与原事实纠错、发布修复额度分别管理。Kimi只处理非必要退回；总时限、重试和费用管控覆盖所有阶段。

当前落地的进化范围是案例验证后的方法复用，不能据此声称方法普遍有效。新增记录位于result.json的research_state、research_iterations、research_methods，以及audit.md。实现已通过359项离线回归；真实模型研究效果仍需实际案例验证。
