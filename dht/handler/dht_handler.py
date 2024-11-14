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
from hivemind.utils.asyncio import amap_in_executor, anext
from hivemind.utils.logging import get_logger
from hivemind.utils.streaming import split_for_streaming
from hivemind.utils.tensor_descr import DUMMY_BATCH_SIZE, BatchTensorDescriptor

from vllm.executor.gpu_executor import GPUExecutorAsync
from vllm.sequence import ExecuteModelRequest, IntermediateTensors
from dht.proto import runtime_pb2
import msgspec
import json
import numpy as np

logger = get_logger(__name__)

# we use this to pass the __init__() of ConnectionHandler in hivemind
class EmptyModel(nn.Module):
    def __init__(self):
        super(EmptyModel, self).__init__()

    def forward(self, x, *args,**kwargs):
        return x

class Event(Enum):
    PUSH = 0
    SHUTDOWN = 1

class PipelineConnectionHandler(ConnectionHandler):

    def __init__(self, dht:DHT, 
                 request_timeout: float,
                 executor:GPUExecutorAsync,
                 is_petals_head: Optional[bool],
                 is_petals_tail: Optional[bool]
                 ):
        module = EmptyModel()
        descr = BatchTensorDescriptor(1)
        empty_module_backend = ModuleBackend('', module, args_schema = (descr, descr),kwargs_schema =  {'1': descr}, max_batch_size = 1)
        super().__init__(dht, empty_module_backend)
        self._listener_task_event: Optional[asyncio.Task] = None
        self._listener_task_request: Optional[asyncio.Task] = None
        self.request_timeout = request_timeout
        self._handler_event_queue = mp.Queue()
        self.request_queue = mp.Queue()
        self.serving_blocks = []
        self.is_petals_head = is_petals_head
        self.is_petals_tail = is_petals_tail

        self.engine = None
        self.executor_backend = executor

    async def add_p2p_handlers(self, *args, **kwargs) -> None:
        if self._listener_task_event is None:
            self._listener_task_event = asyncio.create_task(self._listen_to_event_queue())
        if self._listener_task_request is None:
            self._listener_task_request = asyncio.create_task(self._listen_to_request_queue())
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

    async def _listen_to_request_queue(self):
        loop = asyncio.get_event_loop()
        while True:
            try:
                request = await loop.run_in_executor(None, self.request_queue.get)
                # unpacking the request to
                execute_model_req = request.execute_model_request
                intermediate_tensors = request.intermediate_tensors
                grpc_metadata = request.grpc_metadata
                
                execute_model_req = msgspec.json.decode(execute_model_req)
                temp = IntermediateTensors()
                for it in intermediate_tensors:
                    key = it.key
                    tensors = torch.from_numpy(np.frombuffer(it.tensor_data))
                    temp.tensors.update({key: tensors})
                intermediate_tensors = temp
                grpc_metadata = json.loads(grpc_metadata.decode('utf-8'))

                print('#' * 100)
                print('#' * 100)
                print('#' * 100)
                print('#' * 100)
                print('#' * 100)
                print('#' * 100)
                print(execute_model_req)
                print(intermediate_tensors)
                print(grpc_metadata)

                res = await self.execute_inference_step(execute_model_req, intermediate_tensors, grpc_metadata)
            except Exception as e:
                logger.exception(e)

    def _put_into_request_queue(self, request: runtime_pb2.GrpcRequestData):
        self._handler_event_queue.put_nowait((Event.PUSH, request))

    async def grpc_push(self, request: runtime_pb2.GrpcRequestData) -> runtime_pb2.GrpcResponseData:
        """Directly push activation tensors from one server to another"""
        self._put_into_request_queue(request)
        return runtime_pb2.GrpcResponseData

    async def push_outputs(
        self, execute_model_req: ExecuteModelRequest, intermediate_tensors, grpc_metadata,
    ) -> None:
        try:
            if len(grpc_metadata) <= 0:
                return
            next_server = grpc_metadata[0]
            # there's no next server

            next_peer_id = PeerID.from_base58(next_server)
            grpc_metadata = grpc_metadata[1:]
            grpc_metadata = json.dumps(grpc_metadata).encode('utf-8')

            execute_model_req.async_callback = None
            bytes_emr = msgspec.json.encode(execute_model_req)

            grpc_intermediate_tensors = runtime_pb2.IntermediateTensors()
            for key, tensors in intermediate_tensors.items():
                grpc_intermediate_tensors.tensors.append(runtime_pb2.TensorEntry(key = key,
                                                                                 tensor_data = tensors.cpu().numpy().tobytes()))

            grpc_request_data = runtime_pb2.GrpcRequestData(execute_model_request = bytes_emr,
                                                            intermediate_tensors = grpc_intermediate_tensors,
                                                            grpc_metadata = grpc_metadata,)

            # gRPC call
            stub = self.get_stub(self._p2p, next_peer_id)
            await stub.grpc_push(
                grpc_request_data
            )
        except Exception:
            logger.debug(
                f"Failed to push outputs to next peer",
                exc_info=True,
            )
    
    async def execute_inference_step(self, execute_model_req: ExecuteModelRequest, intermediate_tensors, grpc_metadata):
        # NOTE: handling input
        input_tensors = 0
        can_push = not self.is_petals_tail
        petals_info_metadata = self.engine.petals_info_metadata
        pipeline_outputs = await self.executor_backend.execute_model_async_petals_pp(execute_model_req, 
                                                                                     intermediate_tensors, 
                                                                                     petals_info_metadata)
        pipeline_outputs = pipeline_outputs[0]

        background_tasks = set()
        if can_push:
            task = asyncio.create_task(self.push_outputs(execute_model_req, pipeline_outputs, grpc_metadata))
            background_tasks.add(task)  # Keep reference until it is done to save it from GC
            task.add_done_callback(background_tasks.discard)
            grpc_result = await task
            return grpc_result
        # case sampler_outpurs
        bytes_sampler_outputs = msgspec.json.encode(pipeline_outputs)
        return runtime_pb2.SamplerOutput(output_data=bytes_sampler_outputs)