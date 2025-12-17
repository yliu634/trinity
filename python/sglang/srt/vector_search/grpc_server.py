from __future__ import annotations

import argparse
import asyncio
import logging
import signal
from dataclasses import dataclass
from typing import Optional

import grpc

from sglang.srt.vector_search.grpc import vector_search_pb2, vector_search_pb2_grpc
from sglang.srt.vector_search.batching import (
    MicroBatchConfig,
    MicroBatcher,
    StageAwareBatcher,
    StageAwareConfig,
)
from sglang.srt.vector_search.docstore import DocStore
from sglang.srt.vector_search.embedder import EmbedderConfig, SentenceTransformerEmbedder
from sglang.srt.vector_search.index_numpy import NumpyCosineIndex
from sglang.srt.vector_search.rdma.mooncake_responder import (
    MooncakeRdmaConfig,
    MooncakeRdmaResponder,
)

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
        stage_aware_cfg: Optional[StageAwareConfig],
        rdma_responder: Optional[MooncakeRdmaResponder],
    ):
        self._docstore = docstore
        self._embedder = embedder
        self._index = index
        self._chunk_ids = chunk_ids
        if stage_aware_cfg is not None:
            self._batcher = StageAwareBatcher[_PendingRequest](
                stage_aware_cfg,
                is_prefill_fn=lambda item: item.request.stage
                == vector_search_pb2.PREFILL,
                deadline_ms_fn=lambda item: int(item.request.deadline_ms),
            )
        else:
            self._batcher = MicroBatcher[_PendingRequest](micro_batch)
        self._rdma_responder = rdma_responder
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
            start = asyncio.get_running_loop().time()
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
            finally:
                end = asyncio.get_running_loop().time()
                if hasattr(self._batcher, "report_batch_service_time"):
                    self._batcher.report_batch_service_time(
                        duration_s=end - start, batch_size=len(batch)
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
                    if (
                        self._rdma_responder is not None
                        and req.rdma_session_id
                        and req.rdma_doc_out_addr
                        and req.rdma_meta_out_addr
                    ):
                        doc_bytes, meta_bytes = self._rdma_responder.pack_and_write(
                            request=req, docs=doc_msgs
                        )
                        batch[batch_i].future.set_result(
                            vector_search_pb2.SearchResponse(
                                request_id=req.request_id,
                                docs=[],
                                error_message="",
                                rdma_doc_bytes_written=doc_bytes,
                                rdma_meta_bytes_written=meta_bytes,
                            )
                        )
                    else:
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
    stage_aware_cfg: Optional[StageAwareConfig],
    rdma_responder: Optional[MooncakeRdmaResponder],
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
        stage_aware_cfg=stage_aware_cfg,
        rdma_responder=rdma_responder,
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
    enable_stage_aware_scheduler: bool,
    prefill_reserve_ratio: float,
    prefill_tau_ms: int,
    enable_rdma_response: bool,
    rdma_hostname: Optional[str],
    rdma_ib_device: Optional[str],
    rdma_topk_max: int,
    rdma_doc_slot_bytes: int,
) -> None:
    rdma_responder: Optional[MooncakeRdmaResponder] = None
    if enable_rdma_response:
        if not rdma_hostname:
            raise ValueError(
                "--rdma-hostname is required when --enable-rdma-response is set"
            )
        rdma_responder = MooncakeRdmaResponder(
            MooncakeRdmaConfig(
                rdma_hostname=rdma_hostname,
                ib_device=rdma_ib_device,
                topk_max=rdma_topk_max,
                doc_slot_bytes=rdma_doc_slot_bytes,
            )
        )
        logger.info(
            "RDMA response enabled. server_session_id=%s", rdma_responder.session_id
        )

    stage_aware_cfg: Optional[StageAwareConfig] = None
    if enable_stage_aware_scheduler:
        stage_aware_cfg = StageAwareConfig(
            max_batch_size=max_batch_size,
            flush_timeout_ms=flush_timeout_ms,
            prefill_reserve_ratio=prefill_reserve_ratio,
            prefill_tau_ms=prefill_tau_ms,
        )

    service = await _build_index_and_service(
        docstore_path=docstore_path,
        embedding_model=embedding_model,
        embedding_device=embedding_device,
        embedding_batch_size=embedding_batch_size,
        micro_batch=MicroBatchConfig(
            max_batch_size=max_batch_size, flush_timeout_ms=flush_timeout_ms
        ),
        stage_aware_cfg=stage_aware_cfg,
        rdma_responder=rdma_responder,
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
    parser.add_argument(
        "--enable-stage-aware-scheduler",
        action="store_true",
        help="If set, use Trinity-style stage-aware scheduling (prefill priority) when forming batches.",
    )
    parser.add_argument(
        "--prefill-reserve-ratio",
        type=float,
        default=0.25,
        help="Reserve this fraction of each batch for prefill (stage-aware scheduler).",
    )
    parser.add_argument(
        "--prefill-tau-ms",
        type=int,
        default=1,
        help="Prefill flush timeout (ms) used by stage-aware scheduler to serve prefill quickly.",
    )
    parser.add_argument(
        "--enable-rdma-response",
        action="store_true",
        help="If set, allow callers to receive docs/meta via Mooncake RDMA (requires mooncake-transfer-engine).",
    )
    parser.add_argument(
        "--rdma-hostname",
        type=str,
        default=None,
        help="Hostname/IP used to initialize Mooncake transfer engine on the server.",
    )
    parser.add_argument(
        "--rdma-ib-device",
        type=str,
        default=None,
        help="Optional RDMA ib device string for Mooncake.",
    )
    parser.add_argument(
        "--rdma-topk-max",
        type=int,
        default=8,
        help="Max supported topk for RDMA response packing.",
    )
    parser.add_argument(
        "--rdma-doc-slot-bytes",
        type=int,
        default=8192,
        help="Fixed bytes per doc slot when packing docs for RDMA response.",
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
            enable_stage_aware_scheduler=args.enable_stage_aware_scheduler,
            prefill_reserve_ratio=args.prefill_reserve_ratio,
            prefill_tau_ms=args.prefill_tau_ms,
            enable_rdma_response=args.enable_rdma_response,
            rdma_hostname=args.rdma_hostname,
            rdma_ib_device=args.rdma_ib_device,
            rdma_topk_max=args.rdma_topk_max,
            rdma_doc_slot_bytes=args.rdma_doc_slot_bytes,
        )
    )


if __name__ == "__main__":
    main()
