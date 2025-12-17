from __future__ import annotations

from sglang.srt.vector_search.grpc import vector_search_pb2


def format_retrieved_docs(
    docs: list[vector_search_pb2.RetrievedDoc],
    *,
    max_docs: int,
) -> str:
    parts: list[str] = []
    for i, d in enumerate(docs[:max_docs]):
        text = (d.text or "").strip()
        if not text:
            continue
        parts.append(f"[{i+1}] {text}")
    return "\n\n".join(parts)


def render_context(template: str, docs_text: str) -> str:
    try:
        return template.format(docs=docs_text)
    except Exception:
        return "\n\n[Retrieved Context]\n" + docs_text + "\n"
