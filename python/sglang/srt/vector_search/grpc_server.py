from __future__ import annotations

import argparse
import asyncio
import logging

import grpc

from sglang.srt.vector_search.grpc import vector_search_pb2, vector_search_pb2_grpc

logger = logging.getLogger(__name__)


class VectorSearchServicer(vector_search_pb2_grpc.VectorSearchServicer):
    async def Search(
        self,
        request: vector_search_pb2.SearchRequest,
        context: grpc.aio.ServicerContext,
    ) -> vector_search_pb2.SearchResponse:
        # PR1: stub implementation (no embedding, no ANN, no docstore).
        return vector_search_pb2.SearchResponse(
            request_id=request.request_id,
            docs=[],
            error_message="",
        )


async def serve(host: str, port: int) -> None:
    server = grpc.aio.server()
    vector_search_pb2_grpc.add_VectorSearchServicer_to_server(
        VectorSearchServicer(), server
    )
    listen_addr = f"{host}:{port}"
    server.add_insecure_port(listen_addr)
    logger.info("VectorSearch gRPC server listening on %s", listen_addr)
    await server.start()
    await server.wait_for_termination()


def main() -> None:
    parser = argparse.ArgumentParser(description="VectorSearch gRPC stub server (PR1).")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=50070)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    asyncio.run(serve(args.host, args.port))


if __name__ == "__main__":
    main()

