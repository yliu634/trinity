from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SearchResult:
    indices: np.ndarray  # int64, shape [B, topk]
    scores: np.ndarray  # float32, shape [B, topk]


class VectorIndex(ABC):
    @abstractmethod
    def search(self, queries: np.ndarray, topk: int) -> SearchResult: ...

