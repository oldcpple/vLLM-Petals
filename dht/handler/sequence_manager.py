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
from hivemind.proto import runtime_pb2
from hivemind.utils.asyncio import amap_in_executor, anext
from hivemind.utils.logging import get_logger
from hivemind.utils.streaming import split_for_streaming

from vllm.executor.gpu_executor import GPUExecutorAsync
import math
import threading
import time

class RemoteSequenceManager():

    def __init__(self, dht: DHT, 
                 serving_blocks: List[int], 
                 is_petals_head: Optional[bool],
                 is_petals_tail: Optional[bool]):
        self.dht = dht
        self.serving_blocks = serving_blocks
        self.remote_sequence = []
        self.is_petals_head = is_petals_head
        self.is_petals_tail = is_petals_tail
        self.declare_span()
        thread = threading.Thread(target=self.run_in_background, daemon=True)
        thread.start()

    def declare_span(self):
        num_nodes = self.dht.get('node_index')
        if num_nodes is None:
            num_nodes = 1
        else:
            num_nodes += 1
        self.dht.store('node_info', subkey = self.dht.peer_id.to_base58(), value = self.serving_blocks, expiration_time = math.inf)
    
    def manage_sequence(self):
        remote_sequence_info = self.dht.get('node_info')
        remote_sequence_info = remote_sequence_info.value
        #info_sorted_by_index = dict(sorted(remote_sequence_info.items()))
        remote_sequence = []
        remote_blocks_sequence = []
        for k, v in remote_sequence_info.items():
            if k == self.dht.peer_id.to_base58():
                continue
            remote_sequence.append(k)
            remote_blocks_sequence.append(v.value)
        return remote_sequence
    
    def run_in_background(self):
        while True:
            self.remote_sequence = self.manage_sequence()
            time.sleep(5)


