from __future__ import annotations

import asyncio
import contextlib
import multiprocessing as mp
import sys
from enum import Enum
from itertools import chain
from typing import Any, AsyncIterator, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from async_timeout import timeout
from hivemind import (
    DHT,
    MSGPackSerializer,
    P2PContext,
    PeerID,
    deserialize_tensor_stream,
    deserialize_torch_tensor,
    nested_flatten,
    nested_pack,
    serialize_torch_tensor,
)
from hivemind.moe.server.connection_handler import ConnectionHandler
from hivemind.moe.server.module_backend import ModuleBackend
from hivemind.p2p.p2p_daemon import DEFAULT_MAX_MSG_SIZE
from hivemind.proto import runtime_pb2
from hivemind.utils.asyncio import amap_in_executor, anext
from hivemind.utils.logging import get_logger
from hivemind.utils.streaming import split_for_streaming

from vllm.executor.gpu_executor import GPUExecutorAsync

logger = get_logger(__name__)

# we use this to pass the __init__() of ConnectionHandler in hivemind
class EmptyModel(nn.Module):
    def __init__(self):
        super(EmptyModel, self).__init__()

    def forward(self, x):
        return x

class Event(Enum):
    PUSH = 0
    SHUTDOWN = 1

class PipelineConnectionHandler(ConnectionHandler):

    def __init__(self, dht:DHT, 
                 request_timeout: float,
                 executor:GPUExecutorAsync):
        module = EmptyModel()
        empty_module_backend = ModuleBackend('', module)
        super.__init__(dht, empty_module_backend)
        self._listener_task: Optional[asyncio.Task] = None
        self.request_timeout = request_timeout
        self._handler_event_queue = mp.Queue()
        self.request_queue = asyncio.Queue()

        self.executor_backend = executor

    async def add_p2p_handlers(self, *args, **kwargs) -> None:
        if self._listener_task is None:
            # Start listening to our own event queue before we accept any requests
            self._listener_task = asyncio.create_task(self._listen_to_event_queue())
        await super().add_p2p_handlers(*args, **kwargs)

    async def _listen_to_event_queue(self):
        loop = asyncio.get_event_loop()
        while True:
            try:
                event, payload = await loop.run_in_executor(None, self._handler_event_queue.get)
                if event == Event.SHUTDOWN:
                    break
                elif event == Event.PUSH:
                    self.request_queue.put_nowait(payload)
                else:
                    raise RuntimeError(f"Unexpected event: {event}")
            except Exception as e:
                logger.exception(e)

    def _put_into_request_queue(self, request: runtime_pb2.ExpertRequest):
        self._handler_event_queue.put_nowait((Event.PUSH, request))

    async def grpc_push(self, request: runtime_pb2.ExpertRequest, context: P2PContext) -> runtime_pb2.ExpertResponse:
        """Directly push activation tensors from one server to another"""
        self._put_into_request_queue(request)
        return runtime_pb2.ExpertResponse()

    async def push_outputs(
        self, request: runtime_pb2.ExpertRequest, serialized_outputs: runtime_pb2.Tensor, metadata: dict
    ) -> None:
        try:
            next_servers = metadata.get("next_servers")
            if not next_servers:
                return

            next_peer_id = next_servers[0]
            next_peer_id = PeerID.from_base58(next_peer_id)

            # Sending hidden states serialized with output_schema to avoid double serialization
            next_tensors = [serialized_outputs] + request.tensors[1:]
            next_metadata = metadata.copy()
            next_metadata.update(next_servers=next_servers[1:])

            # gRPC call
            stub = self.get_stub(self._p2p, next_peer_id)
            await stub.rpc_push(
                runtime_pb2.ExpertRequest(
                    uid='',
                    tensors=next_tensors,
                    metadata=MSGPackSerializer.dumps(next_metadata),
                ),
                timeout=self.request_timeout,
            )
        except Exception:
            logger.debug(
                f"Failed to push outputs to peer_id={next_peer_id}",
                exc_info=True,
            )
    
    async def execute_inference_step(self, request:runtime_pb2.ExpertRequest, context: P2PContext,):
        # NOTE: handling input
        input_tensors = 0
        can_push = False

        pipeline_outputs = await self.executor_backend.execute_model_async(input_tensors)
        background_tasks = set()
        if can_push:
            task = asyncio.create_task(self._push_outputs(request, output_tensors[0], step_metadata))
            background_tasks.add(task)  # Keep reference until it is done to save it from GC
            task.add_done_callback(background_tasks.discard)
        return runtime_pb2.ExpertResponse(tensors=pipeline_outputs)


