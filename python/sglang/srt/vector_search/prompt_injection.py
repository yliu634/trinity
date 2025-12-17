from __future__ import annotations

from sglang.srt.vector_search.grpc import vector_search_pb2


def format_retrieved_docs(
    docs: list[vector_search_pb2.RetrievedDoc],
    *,
    max_docs: int,
    max_total_bytes: int | None = None,
) -> str:
    parts: list[str] = []
    total = 0
    for i, d in enumerate(docs[:max_docs]):
        text = (d.text or "").strip()
        if not text:
            continue
        piece = f"[{i+1}] {text}"
        if max_total_bytes is not None:
            piece_bytes = len(piece.encode("utf-8"))
            if total > 0 and total + 2 + piece_bytes > max_total_bytes:
                break
            total += piece_bytes if total == 0 else 2 + piece_bytes
        parts.append(piece)
    return "\n\n".join(parts)


def render_context(template: str, docs_text: str) -> str:
    try:
        return template.format(docs=docs_text)
    except Exception:
        return "\n\n[Retrieved Context]\n" + docs_text + "\n"
