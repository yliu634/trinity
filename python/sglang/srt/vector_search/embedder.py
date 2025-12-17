from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class EmbedderConfig:
    model_name: str = "BAAI/bge-m3"
    batch_size: int = 64
    normalize: bool = True
    device: str | None = None  # e.g. "cuda", "cpu"


class SentenceTransformerEmbedder:
    def __init__(self, config: EmbedderConfig):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required for vector-search embedding. "
                "Install it (and torch) before starting the vector-search server."
            ) from e

        self._config = config
        if config.device:
            self._model = SentenceTransformer(config.model_name, device=config.device)
        else:
            self._model = SentenceTransformer(config.model_name)

    def embed_sync(self, texts: Sequence[str]) -> np.ndarray:
        emb = self._model.encode(
            list(texts),
            batch_size=self._config.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self._config.normalize,
        )
        if not isinstance(emb, np.ndarray):
            emb = np.asarray(emb)
        return emb.astype(np.float32, copy=False)

    async def embed(self, texts: Sequence[str]) -> np.ndarray:
        return await asyncio.to_thread(self.embed_sync, texts)

