"""Bounded asynchronous DAG scheduling with isolated failures and task traces."""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Callable

from .governor import RunStopped


@dataclass(frozen=True)
class TaskNode:
    key: str
    run: Callable[[], object]
    depends_on: tuple[str, ...] = ()
    allow_failed_dependencies: bool = False
    critical: bool = False  # 失败即整轮作废：立刻取消仍在运行的其他任务


@dataclass
class TaskOutcome:
    value: object = None
    error: str = ""
    exception: BaseException | None = None  # 调用方可按原类型重新抛出


class DAGScheduler:
    def __init__(self, max_concurrency: int = 12, control=None):
        if not 1 <= max_concurrency <= 32:
            raise ValueError("并发 Agent 数量须在 1 到 32 之间")
        self.max_concurrency = max_concurrency
        self.control = control  # RunGovernor：总时限、取消与预算
        self.trace: list[dict] = []

    def run(self, nodes: list[TaskNode]) -> dict[str, TaskOutcome]:
        by_key = {n.key: n for n in nodes}
        if len(by_key) != len(nodes):
            raise ValueError("任务 ID 不能重复")
        visited, visiting = set(), set()

        def visit(key):
            if key not in by_key:
                raise ValueError(f"任务依赖不存在：{key}")
            if key in visiting:
                raise ValueError(f"任务依赖成环：{key}")
            if key in visited:
                return
            visiting.add(key)
            for dep in by_key[key].depends_on:
                visit(dep)
            visiting.remove(key)
            visited.add(key)
        for key in by_key:
            visit(key)
        try:
            outcomes = asyncio.run(self._run(by_key))
        except BaseException:
            # Worker threads outlive an interrupted loop; tell them to stop.
            if self.control is not None:
                self.control.cancel("调度被中断")
            raise
        # Only a critical task that actually ran and failed explains a cancellation;
        # a task merely blocked by a stopped dependency does not.
        critical_failed = any(by_key[key].critical and out.exception is not None
                              and not isinstance(out.exception, RunStopped) for key, out in outcomes.items())
        if not critical_failed:
            # A stop is a property of the whole run, not of the task that noticed it.
            for out in outcomes.values():
                if isinstance(out.exception, RunStopped):
                    raise out.exception
        return outcomes

    async def _run(self, nodes):
        semaphore = asyncio.Semaphore(self.max_concurrency)
        pending = {}
        self.trace = []
        origin = time.monotonic()

        def record(node, status, started=None):
            now = time.monotonic()
            self.trace.append({"task": node.key, "depends_on": list(node.depends_on), "status": status,
                               "start": round((started or now) - origin, 3),
                               "seconds": round(now - started, 3) if started else 0})

        async def execute(node):
            dependencies = await asyncio.gather(*(pending[d] for d in node.depends_on))
            failed = [d for d, out in zip(node.depends_on, dependencies) if out.error]
            if failed and not node.allow_failed_dependencies:
                record(node, "blocked")
                return TaskOutcome(error="依赖失败：" + "、".join(failed))
            async with semaphore:
                stop = self.control.stop_error() if self.control is not None else None
                if stop is not None:
                    record(node, "cancelled")
                    return TaskOutcome(error=f"{type(stop).__name__}: {stop}", exception=stop)
                started = time.monotonic()
                try:
                    out = TaskOutcome(value=await asyncio.to_thread(node.run))
                except Exception as exc:
                    out = TaskOutcome(error=f"{type(exc).__name__}: {exc}", exception=exc)
                    if node.critical and self.control is not None and not isinstance(exc, RunStopped):
                        self.control.cancel(f"关键任务 {node.key} 失败，其余并行任务已取消")
                record(node, ("cancelled" if isinstance(out.exception, RunStopped) else "failed") if out.error
                       else "completed", started)
                return out
        # Tasks start on the next event-loop tick, after all keys are registered.
        for key, node in nodes.items():
            pending[key] = asyncio.create_task(execute(node))
        try:
            return dict(zip(pending, await asyncio.gather(*pending.values())))
        except asyncio.CancelledError:
            if self.control is not None:
                self.control.cancel("调度被中断")
            raise
