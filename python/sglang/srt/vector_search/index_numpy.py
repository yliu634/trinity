from __future__ import annotations

import numpy as np

from sglang.srt.vector_search.index_base import SearchResult, VectorIndex


class NumpyCosineIndex(VectorIndex):
    """
    Exact cosine search on CPU.

    Assumes database vectors are L2-normalized and queries are L2-normalized.
    """

    def __init__(self, db_vectors: np.ndarray):
        if db_vectors.ndim != 2:
            raise ValueError("db_vectors must be 2D")
        self._db = db_vectors.astype(np.float32, copy=False)

    def search(self, queries: np.ndarray, topk: int) -> SearchResult:
        if queries.ndim != 2:
            raise ValueError("queries must be 2D")
        if topk <= 0:
            raise ValueError("topk must be > 0")
        topk = min(topk, self._db.shape[0])

        scores = queries.astype(np.float32, copy=False) @ self._db.T
        part = np.argpartition(-scores, kth=topk - 1, axis=1)[:, :topk]
        part_scores = np.take_along_axis(scores, part, axis=1)
        order = np.argsort(-part_scores, axis=1)
        top_idx = np.take_along_axis(part, order, axis=1).astype(np.int64, copy=False)
        top_scores = np.take_along_axis(part_scores, order, axis=1).astype(
            np.float32, copy=False
        )
        return SearchResult(indices=top_idx, scores=top_scores)

