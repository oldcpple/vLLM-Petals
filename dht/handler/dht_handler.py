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
from hivemind.proto import runtime_pb2

from vllm.executor.gpu_executor import GPUExecutorAsync
from vllm.sequence import ExecuteModelRequest, IntermediateTensors
from dht.proto import grpc_pb2
import msgspec
import json
import numpy as np
import io

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
        self.dht = dht
        self._listener_task_event: Optional[asyncio.Task] = None
        self._listener_task_request: Optional[asyncio.Task] = None
        self.request_timeout = request_timeout
        self.grpc_input_queue = mp.Queue()
        self.grpc_callback_queue = mp.Queue()
        self.serving_blocks = []
        self.is_petals_head = is_petals_head
        self.is_petals_tail = is_petals_tail

        self.engine = None
        self.executor_backend = executor
        self.test_num = 1

    async def add_p2p_handlers(self, *args, **kwargs) -> None:
        #asyncio.create_task(self._listen_to_input_queue())
        await super().add_p2p_handlers(*args, **kwargs)

    async def _listen_to_input_queue(self):
        while True:
            try:
                #event, payload = await loop.run_in_executor(None, self._handler_event_queue.get)
                result = await self.engine.gprc_output_queue.get()
                await self.stub_callback(result)

            except Exception as e:
                logger.exception(e)

    async def stub_callback(self, result, grpc_metadata):
        if self.is_petals_tail:
            head_peer_id = grpc_metadata.get('head')
            head_peer_id = PeerID.from_base58(head_peer_id)
            _p2p = await self.dht.replicate_p2p()
            stub = self.get_stub(_p2p, head_peer_id)
            await stub.rpc_return(
                result
            )


    async def stub_put_input(self, request: grpc_pb2.GrpcRequestData) -> grpc_pb2.GrpcResponseData:
        #event, request = await self._handler_event_queue.get()
        print('#' * 100)
        print('#' * 100)
        print('triggered this func: stub_put_input')
        execute_model_req = request.execute_model_request
        intermediate_tensors = request.intermediate_tensors
        grpc_metadata = request.grpc_metadata

        execute_model_req = msgspec.json.decode(execute_model_req)
        execute_model_req = decoding_execute_model_req(execute_model_req)
        temp = IntermediateTensors(tensors={})
        temp = {}
        print('l1')
        for it in intermediate_tensors.tensors:
            key = it.key
            byte_tensor = it.tensor_data
            temp.update({key:byte_tensor})
            '''
            #tensors = torch.from_numpy(np.frombuffer(it.tensor_data)).to(torch.float16)
            tensors = torch.load(io.BytesIO(it.tensor_data), map_location='cpu')
            print('l2')
            tensors = torch.load(io.BytesIO(it.tensor_data), map_location='cuda')
            print('l3')
            temp.tensors.update({key: tensors})
            '''
        print(type(temp))
        intermediate_tensors = temp
        grpc_metadata = json.loads(grpc_metadata.decode('utf-8'))

        print('ff' * 100)
        print(self.grpc_input_queue.qsize())
        self.grpc_input_queue.put_nowait((execute_model_req, intermediate_tensors, grpc_metadata))
        print(self.grpc_input_queue.qsize())
        #res = await self.execute_inference_step(execute_model_req, intermediate_tensors, grpc_metadata)
        #return res

    async def stub_put_result(self, result: grpc_pb2.SamplerOutput) -> grpc_pb2.GrpcResponseData:
        print('triggered this func: stub_put_result')
        print('ff' * 100)
        outputs = msgspec.json.decode(result.output_data)
        outputs = [decoding_sampler_outputs(outputs)]
        print(self.grpc_callback_queue.qsize())
        self.grpc_callback_queue.put_nowait(outputs)
        print(self.grpc_callback_queue.qsize())


    async def rpc_push(self, request: grpc_pb2.GrpcRequestData, context: P2PContext) -> grpc_pb2.GrpcResponseData:
        print('m' * 100)
        print('triggered this func: rpc_push')
        await self.stub_put_input(request)
        return grpc_pb2.GrpcResponseData()
    
    async def rpc_return(self, result: grpc_pb2.SamplerOutput, context: P2PContext) -> grpc_pb2.GrpcResponseData:
        print('triggered this func: rpc_return')
        await self.stub_put_result(result)
        return grpc_pb2.GrpcResponseData()

    async def push_outputs(
        self, execute_model_req: ExecuteModelRequest, intermediate_tensors, grpc_metadata,
    ) -> None:
        try:
            server_list = grpc_metadata.get('server_list')
            if len(server_list) <= 0:
                return
            next_server = server_list[0]
            # there's no next server

            next_peer_id = PeerID.from_base58(next_server)
            server_list = server_list[1:]
            grpc_metadata.update({'server_list' : server_list})
            grpc_metadata = json.dumps(grpc_metadata).encode('utf-8')
            execute_model_req.async_callback = None
            bytes_emr = msgspec.json.encode(execute_model_req)

            grpc_intermediate_tensors = grpc_pb2.IntermediateTensors()
            for key, tensors in intermediate_tensors.items():
                buffer = io.BytesIO()
                torch.save(tensors, buffer)
                byte_data = buffer.getvalue()
                grpc_intermediate_tensors.tensors.append(grpc_pb2.TensorEntry(key = key,
                                                                                 tensor_data = byte_data))

            grpc_request_data = grpc_pb2.GrpcRequestData(execute_model_request = bytes_emr,
                                                            intermediate_tensors = grpc_intermediate_tensors,
                                                            grpc_metadata = grpc_metadata,)
            print('z5')
            # gRPC call
            _p2p = await self.dht.replicate_p2p()
            print(_p2p)
            stub = self.get_stub(_p2p, next_peer_id)
            await stub.rpc_push(
                grpc_request_data
            )
        except Exception:
            logger.debug(
                f"Failed to push outputs to next peer",
                exc_info=True,
            )

    async def execute_inference_step(self, execute_model_req: ExecuteModelRequest, intermediate_tensors, grpc_metadata):
        # NOTE: handling input
        can_push = not self.is_petals_tail
        petals_info_metadata = self.engine.petals_info_metadata
        print('%' * 100)
        print('l3')
        print(execute_model_req)
        pipeline_outputs = await self.executor_backend.execute_model_async_petals_pp(execute_model_req, 
                                                                                     intermediate_tensors, 
                                                                                     petals_info_metadata)
        pipeline_outputs = pipeline_outputs[0]

        if not can_push and self.is_petals_head:
            bytes_sampler_outputs = msgspec.json.encode(pipeline_outputs)
            outputs = msgspec.json.decode(bytes_sampler_outputs)
            outputs = [decoding_sampler_outputs(outputs)]
            return outputs

        if can_push:
            '''
            background_tasks = set()
            task = asyncio.create_task(self.push_outputs(execute_model_req, pipeline_outputs, grpc_metadata))
            background_tasks.add(task)  # Keep reference until it is done to save it from GC
            task.add_done_callback(background_tasks.discard)
            '''
            print(type(grpc_metadata))
            print(grpc_metadata)
            await self.push_outputs(execute_model_req, pipeline_outputs, grpc_metadata)
            #grpc_result = await self.push_outputs(execute_model_req, pipeline_outputs, grpc_metadata)
            if self.is_petals_head:
                print('l4')
                grpc_result = self.grpc_callback_queue.get()
                print('l5')
                print(grpc_result)
                return grpc_result
        # case sampler_outpurs
        bytes_sampler_outputs = msgspec.json.encode(pipeline_outputs)
        return grpc_pb2.SamplerOutput(output_data=bytes_sampler_outputs)

from vllm.sequence import (VLLM_INVALID_TOKEN_ID,
                           CompletionSequenceGroupOutput, Logprob,
                           PromptLogprobs, SampleLogprobs, SequenceOutput)
from vllm.model_executor.layers.sampler import SamplerOutput
import math

def decoding_sampler_outputs(msgspec_sampelr_outputs):
    assert len(msgspec_sampelr_outputs) == 11, "wrong length of SamplerOutput"
    outputs = msgspec_sampelr_outputs[0]
    sampled_token_probs = msgspec_sampelr_outputs[1]
    logprobs = msgspec_sampelr_outputs[2]
    deferred_sample_results_args = msgspec_sampelr_outputs[3]
    sampled_token_ids = msgspec_sampelr_outputs[4]
    sampled_token_ids_cpu = msgspec_sampelr_outputs[5]
    spec_decode_worker_metrics = msgspec_sampelr_outputs[6]
    hidden_states = msgspec_sampelr_outputs[7]
    prefill_hidden_states = msgspec_sampelr_outputs[8]
    model_forward_time = msgspec_sampelr_outputs[9]
    model_execute_time = msgspec_sampelr_outputs[10]

    sampler_output: List[CompletionSequenceGroupOutput] = []
    for c_s_g_o in outputs:
        samples: List[SequenceOutput] = []
        samples_data = c_s_g_o[0]
        prompt_logprobs = c_s_g_o[1]
        for s_d in samples_data:
            parent_seq_id = s_d[0]
            output_token = s_d[1]
            logprobs = s_d[2]
            k, v = next(iter(logprobs.items()))

            k = int(k)
            logprob = v.get('logprob')
            if logprob is None:
                logprob = math.inf
            rank = v.get('rank')
            decoded_token = v.get('decoded_token')
            v = Logprob(logprob, rank, decoded_token)
            logprobs = dict({k: v})
            this_sample = SequenceOutput(parent_seq_id, output_token, logprobs)
            samples.append(this_sample)
        this_sample_output = CompletionSequenceGroupOutput(samples, prompt_logprobs)

        sampler_output.append(this_sample_output)

    return SamplerOutput(
        outputs=sampler_output,
        sampled_token_probs=sampled_token_probs,
        sampled_token_ids=sampled_token_ids,
        logprobs=logprobs,
        deferred_sample_results_args=deferred_sample_results_args)

from vllm.sequence import SequenceGroupMetadata, SequenceGroupMetadataDelta, SequenceData, SequenceGroupState
from typing import Union
from vllm.sampling_params import SamplingParams

def decoding_execute_model_req(msgspec_emq):
    assert len(msgspec_emq) == 12, 'Wrong length of ExecuteModelRequest'

    seq_group_metadata_list_raw = msgspec_emq[0]
    blocks_to_swap_in = msgspec_emq[1]
    blocks_to_swap_out = msgspec_emq[2]
    blocks_to_copy = msgspec_emq[3]
    virtual_engine = msgspec_emq[4]
    num_lookahead_slots = msgspec_emq[5]
    running_queue_size = msgspec_emq[6]
    previous_hidden_states = msgspec_emq[7]
    num_steps = msgspec_emq[8]
    finished_requests_ids = msgspec_emq[9]
    last_sampled_token_ids = msgspec_emq[10]
    async_callback = msgspec_emq[11]

    seq_group_metadata_list: List[Union[SequenceGroupMetadata]] = []
    for raw_metadata in seq_group_metadata_list_raw:
        if raw_metadata[0] == 'SequenceGroupMetadata':
            request_id = raw_metadata[1]
            is_prompt = raw_metadata[2]

            seq_data_raw = raw_metadata[3]
            seq_data: Dict[int, SequenceData] = {}
            for key, value in seq_data_raw.items():
                print('d' * 100)
                print(value.get('_prompt_token_ids_tuple'))
                print(value.get('_prompt_token_ids'))
                key = int(key)
                seq_data[key] = SequenceData(
                    _prompt_token_ids=[0] + value.get('_prompt_token_ids_tuple', []),
                    _output_token_ids=value.get('_output_token_ids', []),
                    _cumulative_logprob=0.0,
                    _num_computed_tokens=value.get('_num_computed_tokens', 0)
                )


            sampling_params_raw = raw_metadata[4]
            sampling_params = SamplingParams(
                n=sampling_params_raw.get('n', 1),
                presence_penalty=sampling_params_raw.get('presence_penalty', 0.0),
                frequency_penalty=sampling_params_raw.get('frequency_penalty', 0.0),
                repetition_penalty=sampling_params_raw.get('repetition_penalty', 1.0),
                temperature=sampling_params_raw.get('temperature', 1.0),
                top_p=sampling_params_raw.get('top_p', 1.0),
                top_k=sampling_params_raw.get('top_k', -1),
                max_tokens=sampling_params_raw.get('max_tokens', 16),
                min_tokens=sampling_params_raw.get('min_tokens', 0),
                stop=sampling_params_raw.get('stop', []),
                stop_token_ids=sampling_params_raw.get('stop_token_ids', []),
                ignore_eos=sampling_params_raw.get('ignore_eos', False),
                logprobs=sampling_params_raw.get('logprobs', None),
                prompt_logprobs=sampling_params_raw.get('prompt_logprobs', None),
                skip_special_tokens=sampling_params_raw.get('skip_special_tokens', True),
                spaces_between_special_tokens=sampling_params_raw.get('spaces_between_special_tokens', True)
            )

            block_tables_raw = raw_metadata[5]
            block_tables: Dict[int, List[int]] = {int(k): v for k, v in block_tables_raw.items()}

            state = SequenceGroupState(
                num_steps=num_steps
            )

    seq_group_metadata_list.append(SequenceGroupMetadata(
        request_id=request_id,
        is_prompt=is_prompt,
        seq_data=seq_data,
        sampling_params=sampling_params,
        block_tables=block_tables,
        do_sample=raw_metadata[6],
        token_chunk_size=raw_metadata[7],
        lora_request=raw_metadata[8],
        computed_block_nums=raw_metadata[9],
        state=state,
        multi_modal_data=raw_metadata[11],
        mm_processor_kwargs=raw_metadata[12],
        encoder_seq_data=raw_metadata[13],
        cross_block_table=raw_metadata[14],
        prompt_adapter_request=raw_metadata[15]
    ))

    return ExecuteModelRequest(
        seq_group_metadata_list=seq_group_metadata_list,
        blocks_to_swap_in=blocks_to_swap_in,
        blocks_to_swap_out=blocks_to_swap_out,
        blocks_to_copy=blocks_to_copy,
        virtual_engine=virtual_engine,
        num_lookahead_slots=num_lookahead_slots,
        running_queue_size=running_queue_size,
        previous_hidden_states=previous_hidden_states,
        num_steps=num_steps,
        finished_requests_ids=finished_requests_ids,
        last_sampled_token_ids=last_sampled_token_ids,
        async_callback=async_callback
    )
