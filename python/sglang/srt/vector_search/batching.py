from __future__ import annotations

import asyncio
import time
import math
from dataclasses import dataclass
from typing import Callable, Generic, Protocol, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class MicroBatchConfig:
    max_batch_size: int = 64
    flush_timeout_ms: int = 1

    def validate(self) -> None:
        if self.max_batch_size <= 0:
            raise ValueError("max_batch_size must be > 0")
        if self.flush_timeout_ms < 0:
            raise ValueError("flush_timeout_ms must be >= 0")


class MicroBatcher(Generic[T]):
    def __init__(self, config: MicroBatchConfig):
        config.validate()
        self._cfg = config
        self._queue: asyncio.Queue[T] = asyncio.Queue()

    async def put(self, item: T) -> None:
        await self._queue.put(item)

    async def get_batch(self) -> list[T]:
        first = await self._queue.get()
        batch = [first]
        if self._cfg.max_batch_size == 1:
            return batch

        start = time.monotonic()
        while len(batch) < self._cfg.max_batch_size:
            remaining = (self._cfg.flush_timeout_ms / 1000.0) - (
                time.monotonic() - start
            )
            if remaining <= 0:
                break
            try:
                item = await asyncio.wait_for(self._queue.get(), timeout=remaining)
            except TimeoutError:
                break
            batch.append(item)
        return batch


class BatchItemInfo(Protocol):
    """Protocol for extracting scheduling metadata from an item."""

    def is_prefill(self, item: T) -> bool: ...

    def deadline_ms(self, item: T) -> int: ...


@dataclass(frozen=True)
class StageAwareConfig:
    max_batch_size: int = 64
    flush_timeout_ms: int = 1
    prefill_reserve_ratio: float = 0.25
    prefill_tau_ms: int = 1
    service_time_ema_alpha: float = 0.2

    def validate(self) -> None:
        if self.max_batch_size <= 0:
            raise ValueError("max_batch_size must be > 0")
        if self.flush_timeout_ms < 0:
            raise ValueError("flush_timeout_ms must be >= 0")
        if not (0.0 <= self.prefill_reserve_ratio <= 1.0):
            raise ValueError("prefill_reserve_ratio must be in [0, 1]")
        if self.prefill_tau_ms < 0:
            raise ValueError("prefill_tau_ms must be >= 0")
        if not (0.0 < self.service_time_ema_alpha <= 1.0):
            raise ValueError("service_time_ema_alpha must be in (0, 1]")


@dataclass
class _Entry(Generic[T]):
    item: T
    arrival_ts: float
    deadline_s: float


class StageAwareBatcher(Generic[T]):
    """
    Stage-aware micro-batcher with two queues:
    - Prefill: scheduled by earliest slack (deadline-aware).
    - Decode: scheduled FIFO.

    Implements Trinity-like knobs:
    - prefill_reserve_ratio: reserve fraction of a batch for prefill.
    - prefill_tau_ms: if any prefill arrives, stop waiting after tau to serve it sooner.
    """

    def __init__(
        self,
        config: StageAwareConfig,
        *,
        is_prefill_fn: Callable[[T], bool],
        deadline_ms_fn: Callable[[T], int],
    ):
        config.validate()
        self._cfg = config
        self._is_prefill = is_prefill_fn
        self._deadline_ms = deadline_ms_fn

        self._cond = asyncio.Condition()
        self._prefill: list[_Entry[T]] = []
        self._decode: list[_Entry[T]] = []
        self._avg_service_s: float | None = None

    async def put(self, item: T) -> None:
        now = time.monotonic()
        deadline_ms = max(int(self._deadline_ms(item)), 0)
        entry = _Entry(item=item, arrival_ts=now, deadline_s=now + (deadline_ms / 1000.0))
        async with self._cond:
            if self._is_prefill(item):
                self._prefill.append(entry)
            else:
                self._decode.append(entry)
            self._cond.notify_all()

    def report_batch_service_time(self, *, duration_s: float, batch_size: int) -> None:
        if batch_size <= 0:
            return
        per_item = max(duration_s / batch_size, 0.0)
        if self._avg_service_s is None:
            self._avg_service_s = per_item
            return
        a = self._cfg.service_time_ema_alpha
        self._avg_service_s = (1 - a) * self._avg_service_s + a * per_item

    async def get_batch(self) -> list[T]:
        start = time.monotonic()
        async with self._cond:
            while not self._prefill and not self._decode:
                await self._cond.wait()

            while True:
                total = len(self._prefill) + len(self._decode)
                if total >= self._cfg.max_batch_size:
                    break
                elapsed_ms = (time.monotonic() - start) * 1000.0
                if elapsed_ms >= self._cfg.flush_timeout_ms:
                    break
                if self._prefill and elapsed_ms >= self._cfg.prefill_tau_ms:
                    break

                remaining = (self._cfg.flush_timeout_ms - elapsed_ms) / 1000.0
                if remaining <= 0:
                    break
                try:
                    await asyncio.wait_for(self._cond.wait(), timeout=remaining)
                except TimeoutError:
                    break

            return self._select_batch_locked()

    def _select_batch_locked(self) -> list[T]:
        now = time.monotonic()
        total = len(self._prefill) + len(self._decode)
        n = min(self._cfg.max_batch_size, total)
        if n <= 0:
            return []

        n_pre = int(math.ceil(self._cfg.prefill_reserve_ratio * n))
        n_pre = min(n_pre, n)

        est = self._avg_service_s or 0.0
        pre_sorted = sorted(self._prefill, key=lambda e: (e.deadline_s - (now + est)))
        dec_sorted = list(self._decode)  # FIFO by arrival order

        selected: list[_Entry[T]] = []
        selected.extend(pre_sorted[:n_pre])
        remaining = n - len(selected)
        selected.extend(dec_sorted[:remaining])

        if len(selected) < n:
            remaining = n - len(selected)
            selected.extend(pre_sorted[n_pre : n_pre + remaining])
        if len(selected) < n:
            remaining = n - len(selected)
            used_dec = min(len(dec_sorted), max(0, n - n_pre))
            selected.extend(dec_sorted[used_dec : used_dec + remaining])

        selected_set = {id(e) for e in selected}
        self._prefill = [e for e in self._prefill if id(e) not in selected_set]
        self._decode = [e for e in self._decode if id(e) not in selected_set]

        return [e.item for e in selected]
