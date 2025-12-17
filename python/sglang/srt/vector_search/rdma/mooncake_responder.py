from __future__ import annotations

import ctypes
import logging
import struct
from dataclasses import dataclass
from typing import Optional

from sglang.srt.vector_search.grpc import vector_search_pb2

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MooncakeRdmaConfig:
    rdma_hostname: str
    ib_device: Optional[str] = None
    topk_max: int = 8
    doc_slot_bytes: int = 8192
    meta_bytes: int = 4096

    def validate(self) -> None:
        if not self.rdma_hostname:
            raise ValueError("rdma_hostname is required")
        if self.topk_max <= 0:
            raise ValueError("topk_max must be > 0")
        if self.doc_slot_bytes <= 0:
            raise ValueError("doc_slot_bytes must be > 0")
        if self.meta_bytes <= 0:
            raise ValueError("meta_bytes must be > 0")


class MooncakeRdmaResponder:
    """
    Packs docs into fixed-size slots and RDMA-writes them to the caller.

    doc buffer layout (bytes):
      slot 0: doc0 bytes (truncated) + zero padding to slot_size
      slot 1: doc1 bytes ...
      ...

    meta buffer layout (little endian):
      u32 topk
      repeated topk times:
        u32 doc_len
        f32 score
        u32 reserved (padding)
    """

    def __init__(self, cfg: MooncakeRdmaConfig):
        cfg.validate()
        try:
            from sglang.srt.disaggregation.mooncake.transfer_engine import (
                MooncakeTransferEngine,
            )
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "Mooncake RDMA responder requires mooncake-transfer-engine and SGLang's Mooncake wrapper."
            ) from e

        self._cfg = cfg
        self._engine = MooncakeTransferEngine(
            hostname=cfg.rdma_hostname,
            gpu_id=0,
            ib_device=cfg.ib_device,
        )

        self._doc_buf = ctypes.create_string_buffer(cfg.topk_max * cfg.doc_slot_bytes)
        self._meta_buf = ctypes.create_string_buffer(cfg.meta_bytes)
        self._doc_addr = ctypes.addressof(self._doc_buf)
        self._meta_addr = ctypes.addressof(self._meta_buf)

        self._engine.register(self._doc_addr, len(self._doc_buf))
        self._engine.register(self._meta_addr, len(self._meta_buf))

    @property
    def session_id(self) -> str:
        return self._engine.session_id

    def pack_and_write(
        self,
        *,
        request: vector_search_pb2.SearchRequest,
        docs: list[vector_search_pb2.RetrievedDoc],
    ) -> tuple[int, int]:
        if not request.rdma_session_id:
            raise ValueError("rdma_session_id is empty")
        if request.rdma_doc_out_addr == 0 or request.rdma_meta_out_addr == 0:
            raise ValueError("rdma output addresses are not set")
        if request.topk == 0:
            raise ValueError("topk must be > 0 for RDMA response")

        topk = min(int(request.topk), self._cfg.topk_max, len(docs))
        required_doc_bytes = topk * self._cfg.doc_slot_bytes
        if int(request.rdma_doc_out_len) < required_doc_bytes:
            raise ValueError(
                f"caller doc_out_len too small: {request.rdma_doc_out_len} < {required_doc_bytes}"
            )

        meta_needed = 4 + topk * 12
        if int(request.rdma_meta_out_len) < meta_needed or self._cfg.meta_bytes < meta_needed:
            raise ValueError(
                f"caller meta_out_len too small: {request.rdma_meta_out_len} < {meta_needed}"
            )

        # Clear buffers for deterministic reads on the receiver side.
        ctypes.memset(self._doc_addr, 0, required_doc_bytes)
        ctypes.memset(self._meta_addr, 0, meta_needed)

        # Pack docs into fixed slots.
        for i in range(topk):
            slot_off = i * self._cfg.doc_slot_bytes
            raw = docs[i].text.encode("utf-8", errors="ignore")
            raw = raw[: self._cfg.doc_slot_bytes]
            ctypes.memmove(self._doc_addr + slot_off, raw, len(raw))

        # Pack meta.
        meta = bytearray()
        meta += struct.pack("<I", topk)
        for i in range(topk):
            raw = docs[i].text.encode("utf-8", errors="ignore")[: self._cfg.doc_slot_bytes]
            meta += struct.pack("<IfI", len(raw), float(docs[i].score), 0)
        ctypes.memmove(self._meta_addr, bytes(meta), len(meta))

        # RDMA write docs then meta.
        doc_ret = self._engine.transfer_sync(
            request.rdma_session_id,
            self._doc_addr,
            int(request.rdma_doc_out_addr),
            required_doc_bytes,
        )
        meta_ret = self._engine.transfer_sync(
            request.rdma_session_id,
            self._meta_addr,
            int(request.rdma_meta_out_addr),
            meta_needed,
        )
        if doc_ret < 0 or meta_ret < 0:
            raise RuntimeError("RDMA transfer failed")

        return required_doc_bytes, meta_needed

