from __future__ import annotations

import re
from dataclasses import dataclass

import grpc

from sglang.srt.vector_search.grpc import vector_search_pb2, vector_search_pb2_grpc


def _normalize_endpoint(endpoint: str) -> str:
    endpoint = endpoint.strip()
    endpoint = re.sub(r"^https?://", "", endpoint)
    return endpoint


@dataclass(frozen=True)
class VectorSearchClientConfig:
    endpoint: str
    timeout_ms: int = 2000
    # Optional: enable Mooncake RDMA response.
    enable_rdma: bool = False
    rdma_hostname: str | None = None
    rdma_ib_device: str | None = None
    # Fixed buffer sizes for RDMA response packing.
    rdma_topk_max: int = 8
    rdma_doc_slot_bytes: int = 8192
    rdma_meta_bytes: int = 4096


class VectorSearchClient:
    def __init__(self, config: VectorSearchClientConfig):
        self._endpoint = _normalize_endpoint(config.endpoint)
        self._timeout_s = max(config.timeout_ms, 1) / 1000.0
        self._channel: grpc.aio.Channel = grpc.aio.insecure_channel(self._endpoint)
        self._stub = vector_search_pb2_grpc.VectorSearchStub(self._channel)
        self._cfg = config

        self._rdma_engine = None
        self._doc_buf = None
        self._meta_buf = None
        self._doc_addr = 0
        self._meta_addr = 0
        if config.enable_rdma:
            if not config.rdma_hostname:
                raise ValueError("rdma_hostname is required when enable_rdma is True")
            try:
                from sglang.srt.disaggregation.mooncake.transfer_engine import (
                    MooncakeTransferEngine,
                )
                import ctypes
            except Exception as e:  # pragma: no cover
                raise ImportError(
                    "Mooncake RDMA client requires mooncake-transfer-engine and SGLang's Mooncake wrapper."
                ) from e

            self._rdma_engine = MooncakeTransferEngine(
                hostname=config.rdma_hostname, gpu_id=0, ib_device=config.rdma_ib_device
            )
            self._doc_buf = ctypes.create_string_buffer(
                config.rdma_topk_max * config.rdma_doc_slot_bytes
            )
            self._meta_buf = ctypes.create_string_buffer(config.rdma_meta_bytes)
            self._doc_addr = ctypes.addressof(self._doc_buf)
            self._meta_addr = ctypes.addressof(self._meta_buf)
            self._rdma_engine.register(self._doc_addr, len(self._doc_buf))
            self._rdma_engine.register(self._meta_addr, len(self._meta_buf))

    async def aclose(self) -> None:
        await self._channel.close()

    async def search(
        self,
        *,
        request_id: str,
        stage: vector_search_pb2.Stage,
        text: str,
        topk: int,
        deadline_ms: int,
    ) -> vector_search_pb2.SearchResponse:
        req = vector_search_pb2.SearchRequest(
            request_id=request_id,
            stage=stage,
            text=text,
            topk=topk,
            deadline_ms=deadline_ms,
        )
        if self._cfg.enable_rdma and self._rdma_engine is not None:
            req.rdma_session_id = self._rdma_engine.session_id
            req.rdma_doc_out_addr = self._doc_addr
            req.rdma_doc_out_len = self._cfg.rdma_topk_max * self._cfg.rdma_doc_slot_bytes
            req.rdma_meta_out_addr = self._meta_addr
            req.rdma_meta_out_len = self._cfg.rdma_meta_bytes
        return await self._stub.Search(req, timeout=self._timeout_s)

    def parse_rdma_docs(
        self, *, topk: int, encoding: str = "utf-8"
    ) -> list[str]:
        """
        Parse docs from the local RDMA buffers after a search().
        Only valid if enable_rdma=True.
        """
        if not self._cfg.enable_rdma or self._meta_buf is None or self._doc_buf is None:
            raise RuntimeError("RDMA not enabled")
        import struct

        topk = min(topk, self._cfg.rdma_topk_max)
        meta_bytes = bytes(self._meta_buf.raw[: 4 + topk * 12])
        (n,) = struct.unpack_from("<I", meta_bytes, 0)
        n = min(n, topk)
        lens: list[int] = []
        for i in range(n):
            doc_len, score, _ = struct.unpack_from("<IfI", meta_bytes, 4 + i * 12)
            lens.append(int(doc_len))
        docs: list[str] = []
        for i in range(n):
            off = i * self._cfg.rdma_doc_slot_bytes
            raw = bytes(self._doc_buf.raw[off : off + lens[i]])
            docs.append(raw.decode(encoding, errors="ignore"))
        return docs
