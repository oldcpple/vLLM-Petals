from typing import Optional

from torch import nn

from vllm.config import (CacheConfig, DeviceConfig, LoadConfig, LoRAConfig,
                         ModelConfig, ParallelConfig, SchedulerConfig)
from vllm.model_executor.model_loader.loader import (BaseModelLoader,
                                                     get_model_loader)
from vllm.model_executor.model_loader.utils import (
    get_architecture_class_name, get_model_architecture)


def get_model(*, model_config: ModelConfig, load_config: LoadConfig,
              device_config: DeviceConfig, parallel_config: ParallelConfig,
              scheduler_config: SchedulerConfig,
              lora_config: Optional[LoRAConfig],
              cache_config: CacheConfig,
              using_petals_pp: Optional[bool] = False,
              is_petals_head: Optional[bool] = False,
              is_petals_tail: Optional[bool] = False,
              petals_tf_layers_range: Optional[list] = [],) -> nn.Module:
    loader = get_model_loader(load_config)
    return loader.load_model(model_config=model_config,
                             device_config=device_config,
                             lora_config=lora_config,
                             parallel_config=parallel_config,
                             scheduler_config=scheduler_config,
                             cache_config=cache_config,
                             using_petals_pp = using_petals_pp,
                             is_petals_head = is_petals_head,
                             is_petals_tail = is_petals_tail,
                             petals_tf_layers_range = petals_tf_layers_range,)


__all__ = [
    "get_model", "get_model_loader", "BaseModelLoader",
    "get_architecture_class_name", "get_model_architecture"
]
