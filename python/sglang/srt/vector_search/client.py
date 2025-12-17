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


class VectorSearchClient:
    def __init__(self, config: VectorSearchClientConfig):
        self._endpoint = _normalize_endpoint(config.endpoint)
        self._timeout_s = max(config.timeout_ms, 1) / 1000.0
        self._channel: grpc.aio.Channel = grpc.aio.insecure_channel(self._endpoint)
        self._stub = vector_search_pb2_grpc.VectorSearchStub(self._channel)

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
        return await self._stub.Search(req, timeout=self._timeout_s)

