#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from sglang.srt.vector_search.text_splitter import (
    TokenChunkConfig,
    make_hf_tokenizer_adapter,
    split_text_on_tokens,
)


@dataclass(frozen=True)
class DocChunk:
    chunk_id: str
    doc_id: str
    text: str
    token_count: int
    byte_len: int
    metadata: dict[str, Any]


def iter_input_records(input_path: Path) -> Iterable[tuple[str, str, dict[str, Any]]]:
    """
    Yields (doc_id, text, metadata).

    Supported inputs:
    - directory: recursively reads all files as UTF-8 text
    - .jsonl: each line is a JSON object with {id/text} or {doc_id/text} or {text}
    - .txt/.md: treated as a single document
    """
    if input_path.is_dir():
        for file_path in sorted(p for p in input_path.rglob("*") if p.is_file()):
            try:
                text = file_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            doc_id = str(file_path.relative_to(input_path))
            yield doc_id, text, {"source": doc_id}
        return

    if input_path.suffix.lower() == ".jsonl":
        with input_path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                text = obj.get("text") or obj.get("page_content") or obj.get("content")
                if not isinstance(text, str) or not text.strip():
                    continue
                doc_id = (
                    obj.get("id")
                    or obj.get("doc_id")
                    or obj.get("document_id")
                    or f"{input_path.name}:{i}"
                )
                metadata = obj.get("metadata") if isinstance(obj.get("metadata"), dict) else {}
                metadata = {**metadata, "source": str(input_path)}
                yield str(doc_id), text, metadata
        return

    if input_path.suffix.lower() in {".txt", ".md"}:
        text = input_path.read_text(encoding="utf-8")
        yield input_path.name, text, {"source": str(input_path)}
        return

    raise ValueError(f"Unsupported input: {input_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a docstore pickle for Trinity vector search."
    )
    parser.add_argument("--input", type=str, required=True, help="Input path.")
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output pickle path (chunk_id -> chunk record).",
    )
    parser.add_argument(
        "--tokenizer-model",
        type=str,
        default="BAAI/bge-m3",
        help="HuggingFace tokenizer model name/path.",
    )
    parser.add_argument(
        "--chunk-tokens",
        type=int,
        default=256,
        help="Chunk size in tokens.",
    )
    parser.add_argument(
        "--chunk-overlap-tokens",
        type=int,
        default=50,
        help="Chunk overlap in tokens.",
    )
    parser.add_argument(
        "--min-chunk-chars",
        type=int,
        default=20,
        help="Drop very short chunks.",
    )
    args = parser.parse_args()

    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        raise RuntimeError(
            "transformers is required to build the docstore. Install it first."
        ) from e

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model, use_fast=True)
    tokenizer_adapter = make_hf_tokenizer_adapter(tokenizer)

    cfg = TokenChunkConfig(
        chunk_tokens=args.chunk_tokens, chunk_overlap_tokens=args.chunk_overlap_tokens
    )

    chunks: dict[str, dict[str, Any]] = {}
    total_docs = 0
    total_chunks = 0

    for doc_id, text, metadata in iter_input_records(input_path):
        total_docs += 1
        splits = split_text_on_tokens(text=text, tokenizer=tokenizer_adapter, config=cfg)
        for idx, chunk_text in enumerate(splits):
            chunk_text = chunk_text.strip()
            if len(chunk_text) < args.min_chunk_chars:
                continue
            chunk_id = f"{doc_id}::chunk{idx}"
            token_count = len(tokenizer_adapter.encode(chunk_text))
            byte_len = len(chunk_text.encode("utf-8"))
            record = DocChunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                text=chunk_text,
                token_count=token_count,
                byte_len=byte_len,
                metadata=metadata,
            )
            chunks[chunk_id] = {
                "chunk_id": record.chunk_id,
                "doc_id": record.doc_id,
                "text": record.text,
                "token_count": record.token_count,
                "byte_len": record.byte_len,
                "metadata": record.metadata,
            }
            total_chunks += 1

    with output_path.open("wb") as f:
        pickle.dump(
            {
                "version": 1,
                "tokenizer_model": args.tokenizer_model,
                "chunk_tokens": args.chunk_tokens,
                "chunk_overlap_tokens": args.chunk_overlap_tokens,
                "chunks": chunks,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    print(
        json.dumps(
            {
                "output": str(output_path),
                "docs": total_docs,
                "chunks": total_chunks,
            }
        )
    )


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()

