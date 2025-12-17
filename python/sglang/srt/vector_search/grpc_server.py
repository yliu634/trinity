from __future__ import annotations

import argparse
import asyncio
import logging
import signal
from dataclasses import dataclass
from typing import Optional

import grpc

from sglang.srt.vector_search.grpc import vector_search_pb2, vector_search_pb2_grpc
from sglang.srt.vector_search.batching import MicroBatchConfig, MicroBatcher
from sglang.srt.vector_search.docstore import DocStore
from sglang.srt.vector_search.embedder import EmbedderConfig, SentenceTransformerEmbedder
from sglang.srt.vector_search.index_numpy import NumpyCosineIndex

logger = logging.getLogger(__name__)


@dataclass
class _PendingRequest:
    request: vector_search_pb2.SearchRequest
    future: asyncio.Future[vector_search_pb2.SearchResponse]


class VectorSearchService:
    def __init__(
        self,
        *,
        docstore: DocStore,
        embedder: SentenceTransformerEmbedder,
        index: NumpyCosineIndex,
        chunk_ids: list[str],
        micro_batch: MicroBatchConfig,
    ):
        self._docstore = docstore
        self._embedder = embedder
        self._index = index
        self._chunk_ids = chunk_ids
        self._batcher: MicroBatcher[_PendingRequest] = MicroBatcher(micro_batch)
        self._task: Optional[asyncio.Task] = None

    def start(self) -> None:
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._batch_loop())

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None

    async def handle(self, request: vector_search_pb2.SearchRequest) -> vector_search_pb2.SearchResponse:
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[vector_search_pb2.SearchResponse] = loop.create_future()
        await self._batcher.put(_PendingRequest(request=request, future=fut))
        return await fut

    async def _batch_loop(self) -> None:
        while True:
            batch = await self._batcher.get_batch()
            try:
                await self._process_batch(batch)
            except Exception as e:
                for item in batch:
                    if not item.future.done():
                        item.future.set_result(
                            vector_search_pb2.SearchResponse(
                                request_id=item.request.request_id,
                                docs=[],
                                error_message=str(e),
                            )
                        )

    async def _process_batch(self, batch: list[_PendingRequest]) -> None:
        texts = [b.request.text for b in batch]
        embeddings = await self._embedder.embed(texts)

        groups: dict[int, list[int]] = {}
        for i, item in enumerate(batch):
            topk = int(item.request.topk) if item.request.topk > 0 else 4
            groups.setdefault(topk, []).append(i)

        for topk, idxs in groups.items():
            q = embeddings[idxs, :]
            result = self._index.search(q, topk=topk)
            for local_row, batch_i in enumerate(idxs):
                req = batch[batch_i].request
                doc_msgs: list[vector_search_pb2.RetrievedDoc] = []
                for j in range(result.indices.shape[1]):
                    db_ix = int(result.indices[local_row, j])
                    chunk_id = self._chunk_ids[db_ix]
                    chunk = self._docstore.get(chunk_id)
                    if chunk is None:
                        continue
                    doc_msgs.append(
                        vector_search_pb2.RetrievedDoc(
                            doc_id=chunk.chunk_id,
                            text=chunk.text,
                            score=float(result.scores[local_row, j]),
                        )
                    )
                if not batch[batch_i].future.done():
                    batch[batch_i].future.set_result(
                        vector_search_pb2.SearchResponse(
                            request_id=req.request_id,
                            docs=doc_msgs,
                            error_message="",
                        )
                    )


class VectorSearchServicer(vector_search_pb2_grpc.VectorSearchServicer):
    def __init__(self, service: VectorSearchService):
        self._service = service

    async def Search(
        self,
        request: vector_search_pb2.SearchRequest,
        context: grpc.aio.ServicerContext,
    ) -> vector_search_pb2.SearchResponse:
        return await self._service.handle(request)


async def _build_index_and_service(
    *,
    docstore_path: str,
    embedding_model: str,
    embedding_device: Optional[str],
    embedding_batch_size: int,
    micro_batch: MicroBatchConfig,
) -> VectorSearchService:
    docstore = DocStore.load_pickle(docstore_path)
    chunk_ids = docstore.chunk_ids()
    texts = [docstore.get(cid).text for cid in chunk_ids]  # type: ignore[union-attr]

    embedder = SentenceTransformerEmbedder(
        EmbedderConfig(
            model_name=embedding_model,
            batch_size=embedding_batch_size,
            normalize=True,
            device=embedding_device,
        )
    )
    logger.info("Building embeddings for %d chunks...", len(chunk_ids))
    db_emb = await embedder.embed(texts)
    index = NumpyCosineIndex(db_emb)

    service = VectorSearchService(
        docstore=docstore,
        embedder=embedder,
        index=index,
        chunk_ids=chunk_ids,
        micro_batch=micro_batch,
    )
    service.start()
    return service


async def serve(
    host: str,
    port: int,
    *,
    docstore_path: str,
    embedding_model: str,
    embedding_device: Optional[str],
    embedding_batch_size: int,
    max_batch_size: int,
    flush_timeout_ms: int,
) -> None:
    service = await _build_index_and_service(
        docstore_path=docstore_path,
        embedding_model=embedding_model,
        embedding_device=embedding_device,
        embedding_batch_size=embedding_batch_size,
        micro_batch=MicroBatchConfig(
            max_batch_size=max_batch_size, flush_timeout_ms=flush_timeout_ms
        ),
    )
    server = grpc.aio.server()
    vector_search_pb2_grpc.add_VectorSearchServicer_to_server(
        VectorSearchServicer(service), server
    )
    listen_addr = f"{host}:{port}"
    server.add_insecure_port(listen_addr)
    logger.info("VectorSearch gRPC server listening on %s", listen_addr)
    await server.start()
    try:
        await server.wait_for_termination()
    finally:
        await service.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="VectorSearch gRPC server (micro-batch).")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=50070)
    parser.add_argument(
        "--docstore",
        type=str,
        required=True,
        help="Path to docstore pickle produced by sglang.tools.build_docstore.",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="BAAI/bge-m3",
        help="SentenceTransformer/HF model name for embeddings.",
    )
    parser.add_argument(
        "--embedding-device",
        type=str,
        default=None,
        help="Embedding device hint (e.g., cuda or cpu).",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=64,
        help="Embedding batch size.",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=64,
        help="Max micro-batch size for vector search.",
    )
    parser.add_argument(
        "--flush-timeout-ms",
        type=int,
        default=1,
        help="Micro-batch flush timeout in milliseconds.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    asyncio.run(
        serve(
            args.host,
            args.port,
            docstore_path=args.docstore,
            embedding_model=args.embedding_model,
            embedding_device=args.embedding_device,
            embedding_batch_size=args.embedding_batch_size,
            max_batch_size=args.max_batch_size,
            flush_timeout_ms=args.flush_timeout_ms,
        )
    )


if __name__ == "__main__":
    main()
