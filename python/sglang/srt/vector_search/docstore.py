from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class DocChunkRecord:
    chunk_id: str
    doc_id: str
    text: str
    token_count: int
    byte_len: int
    metadata: dict[str, Any]


class DocStore:
    def __init__(self, chunks: dict[str, DocChunkRecord]):
        self._chunks = chunks

    @classmethod
    def load_pickle(cls, path: str | Path) -> "DocStore":
        path = Path(path)
        with path.open("rb") as f:
            obj = pickle.load(f)

        if not isinstance(obj, dict) or "chunks" not in obj:
            raise ValueError(f"Invalid docstore pickle: missing 'chunks' in {path}")
        chunks_obj = obj["chunks"]
        if not isinstance(chunks_obj, dict):
            raise ValueError(f"Invalid docstore pickle: 'chunks' must be dict in {path}")

        chunks: dict[str, DocChunkRecord] = {}
        for chunk_id, record in chunks_obj.items():
            if not isinstance(record, dict):
                continue
            text = record.get("text")
            if not isinstance(text, str):
                continue
            chunks[str(chunk_id)] = DocChunkRecord(
                chunk_id=str(record.get("chunk_id", chunk_id)),
                doc_id=str(record.get("doc_id", "")),
                text=text,
                token_count=int(record.get("token_count", 0)),
                byte_len=int(record.get("byte_len", len(text.encode("utf-8")))),
                metadata=record.get("metadata")
                if isinstance(record.get("metadata"), dict)
                else {},
            )

        if not chunks:
            raise ValueError(f"Docstore has no valid chunks: {path}")
        return cls(chunks)

    def __len__(self) -> int:
        return len(self._chunks)

    def get(self, chunk_id: str) -> DocChunkRecord | None:
        return self._chunks.get(chunk_id)

    def chunk_ids(self) -> list[str]:
        return list(self._chunks.keys())

    def iter_chunks(self) -> Iterable[DocChunkRecord]:
        return self._chunks.values()

