from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Generic, TypeVar

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

