"""Run-wide request governance shared by every agent, tool and scheduler task.

One governor lives for one pipeline run. It owns the things a single request
cannot decide on its own: the total deadline, cancellation, how many retries the
whole run may still spend, per-provider pacing, reuse of identical requests and
the hard cost ceiling. It adds no agent and makes no model call.
"""
from __future__ import annotations

import random
import threading
import time
from contextlib import contextmanager


class RunStopped(RuntimeError):
    """The whole run must stop; never downgrade this to a per-agent failure."""


class RunTimeout(RunStopped):
    pass


class RunCancelled(RunStopped):
    pass


class BudgetExceeded(RunStopped):
    pass


class _Gate:
    """Concurrency, request spacing and rate-limit cooldown for one provider."""

    def __init__(self, concurrency: int, rpm: int | None):
        self.semaphore = threading.BoundedSemaphore(concurrency)
        self.interval = 60 / rpm if rpm else 0.0
        self.next_free = 0.0
        self.cooldown_until = 0.0
        self.lock = threading.Lock()

    def reserve(self, now: float) -> float:
        with self.lock:
            start = max(now, self.next_free, self.cooldown_until)
            self.next_free = start + self.interval
            return start - now

    def cool(self, until: float) -> None:
        with self.lock:
            self.cooldown_until = max(self.cooldown_until, until)


class _Flight:
    def __init__(self):
        self.done = threading.Event()
        self.value = None
        self.error: BaseException | None = None


class RunGovernor:
    backoff_base = 0.5  # 第一次重试前的等待秒数，之后每次翻倍
    max_backoff = 8.0
    max_retry_after = 60.0

    def __init__(self, *, timeout_seconds: float | None = None, retry_budget: int | None = None,
                 cost_budget_usd: float | None = None, provider_concurrency: int = 8,
                 provider_rpm: dict[str, int] | None = None, clock=time.monotonic):
        self.timeout_seconds = timeout_seconds or None
        self.retry_budget = retry_budget
        self.cost_budget_usd = cost_budget_usd
        self.provider_concurrency = provider_concurrency
        self.provider_rpm = dict(provider_rpm or {})
        self._clock = clock
        self._lock = threading.Lock()
        self._reset(armed=False)

    @classmethod
    def from_settings(cls, settings) -> "RunGovernor":
        governor = cls()
        governor.configure(settings)
        return governor

    def configure(self, settings) -> None:
        """Re-read the limits; clients and toolboxes keep pointing at this object."""
        self.timeout_seconds = settings.run_timeout_seconds or None
        self.retry_budget = settings.retry_budget
        self.cost_budget_usd = settings.cost_budget_usd
        self.provider_concurrency = settings.provider_concurrency
        self.provider_rpm = dict(settings.provider_rpm)

    def _reset(self, armed: bool) -> None:
        self._cancel = threading.Event()
        self._reason = ""
        self._deadline = self._clock() + self.timeout_seconds if armed and self.timeout_seconds else None
        self._retries_left = self.retry_budget
        self._gates: dict[str, _Gate] = {}
        self._flights: dict[tuple, _Flight] = {}
        self._spent = 0.0
        self._counts = {"requests": 0, "retries": 0, "reused_requests": 0, "rate_limit_waits": 0}

    def start(self) -> None:
        """Arm the deadline and forget the previous run's budgets and cache."""
        with self._lock:
            self._reset(armed=True)

    # ------------------------------------------------------------ 停止条件

    def cancel(self, reason: str = "任务已取消") -> None:
        with self._lock:
            if not self._cancel.is_set():
                self._reason = reason
                self._cancel.set()

    def remaining(self) -> float | None:
        return None if self._deadline is None else self._deadline - self._clock()

    def stop_error(self) -> RunStopped | None:
        if self._cancel.is_set():
            return RunCancelled(self._reason)
        left = self.remaining()
        if left is not None and left <= 0:
            return RunTimeout(f"超过任务总时限 {self.timeout_seconds / 60:g} 分钟，已停止后续模型与工具调用")
        if self.cost_budget_usd is not None and self._spent >= self.cost_budget_usd:
            return BudgetExceeded(f"已用费用 {self._spent:.2f} 美元，达到费用上限 {self.cost_budget_usd:.2f} 美元")
        return None

    @property
    def stopped(self) -> bool:
        return self.stop_error() is not None

    def check(self) -> None:
        error = self.stop_error()
        if error is not None:
            raise error

    def request_timeout(self, default: float) -> float:
        """A single request may never outlive the run."""
        left = self.remaining()
        return default if left is None else max(1.0, min(default, left))

    def _sleep(self, seconds: float) -> None:
        if seconds > 0:
            left = self.remaining()
            self._cancel.wait(seconds if left is None else max(0.0, min(seconds, left)))
        self.check()

    # ------------------------------------------------------------ 供应商配额

    def _gate(self, provider: str) -> _Gate:
        with self._lock:
            if provider not in self._gates:
                self._gates[provider] = _Gate(self.provider_concurrency, self.provider_rpm.get(provider))
            return self._gates[provider]

    @contextmanager
    def slot(self, provider: str):
        gate = self._gate(provider)
        while not gate.semaphore.acquire(timeout=0.2):
            self.check()
        try:
            delay = gate.reserve(self._clock())
            if delay > 0:
                with self._lock:
                    self._counts["rate_limit_waits"] += 1
            self._sleep(delay)
            yield
        finally:
            gate.semaphore.release()

    def _take_retry(self) -> bool:
        with self._lock:
            if self._retries_left is not None:
                if self._retries_left <= 0:
                    return False
                self._retries_left -= 1
            self._counts["retries"] += 1
            return True

    def call(self, provider: str, request, *, retries: int, timeout: float,
             retryable=lambda exc: False, retry_after=lambda exc: None):
        """Send one logical request. `request(timeout)` performs a single attempt.

        Retries are drawn from the run-wide budget, so a degraded provider cannot
        multiply every request of every agent by its own retry count.
        """
        failures = 0
        while True:
            with self.slot(provider):
                with self._lock:
                    self._counts["requests"] += 1
                try:
                    return request(self.request_timeout(timeout))
                except RunStopped:
                    raise
                except Exception as exc:
                    if not retryable(exc):
                        raise
                    failures += 1
                    wait = retry_after(exc)
                    delay = (min(wait, self.max_retry_after) if wait is not None
                             else min(self.max_backoff, self.backoff_base * 2 ** (failures - 1)) * (0.75 + random.random() / 2))
                    if getattr(exc, "status_code", None) == 429:
                        # Everyone queued for this provider waits, instead of each burning a retry.
                        self._gate(provider).cool(self._clock() + delay)
                    if failures > retries or not self._take_retry():
                        raise
            self._sleep(delay)

    # ------------------------------------------------------------ 请求去重

    def shared(self, key: tuple, produce):
        """Identical requests in one run share one execution; only successes are kept."""
        with self._lock:
            flight = self._flights.get(key)
            owner = flight is None
            if owner:
                flight = self._flights[key] = _Flight()
            else:
                self._counts["reused_requests"] += 1
        if owner:
            try:
                flight.value = produce()
            except BaseException as exc:
                flight.error = exc
                with self._lock:
                    self._flights.pop(key, None)
                raise
            finally:
                flight.done.set()
            return flight.value
        while not flight.done.wait(0.2):
            self.check()
        if flight.error is not None:
            raise flight.error
        return flight.value

    # ------------------------------------------------------------ 费用与统计

    def add_cost(self, usd: float) -> None:
        with self._lock:
            self._spent += usd

    def stats(self) -> dict:
        error = self.stop_error()
        with self._lock:
            return {**self._counts, "retry_budget": self.retry_budget, "retries_left": self._retries_left,
                    "timeout_seconds": self.timeout_seconds, "cost_budget_usd": self.cost_budget_usd,
                    "reported_cost_usd": round(self._spent, 6),
                    "stop_reason": f"{type(error).__name__}: {error}" if error else ""}
