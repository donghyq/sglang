# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/kv_cache.py

import logging
from collections.abc import Mapping
from typing import Union

import torch

from sglang.kernels.ops.quantization.fp8_kernel import is_fp8_fnuz
from sglang.srt.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)

logger = logging.getLogger(__name__)


def _load_kv_scale(param: torch.nn.Parameter, loaded_weight: torch.Tensor) -> None:
    """Load either a scalar scale or one value per local KV head."""
    if loaded_weight.numel() == 1:
        param.data.fill_(loaded_weight.item())
    else:
        if param.shape != loaded_weight.shape:
            raise ValueError(
                f"KV cache scale shape mismatch: expected {tuple(param.shape)}, "
                f"got {tuple(loaded_weight.shape)}"
            )
        param.data.copy_(loaded_weight)


def use_per_head_kv_scale(prefix: str) -> bool:
    from sglang.srt.runtime_context import get_server_args

    try:
        server_args = get_server_args()
    except ValueError:
        # Lightweight layer unit tests may construct modules before publishing
        # ServerArgs. Preserve the historical per-tensor behavior there.
        return False
    skip_patterns = {
        pattern.strip()
        for pattern in (server_args.kv_cache_quant_skip_modules or "").split(",")
        if pattern.strip()
    }
    return server_args.kv_cache_quant_granularity == "per_head" and not any(
        pattern in prefix for pattern in skip_patterns
    )


def set_kv_scale(
    layer: torch.nn.Module,
    scaling_factor: Union[float, torch.Tensor, list, Mapping],
) -> None:
    """Copy calibrated K/V scales without replacing registered parameters.

    ``scaling_factor`` may be a legacy shared scalar/vector, a mapping with
    ``k``/``v`` entries, or a validated object exposing ``.k`` and ``.v``.
    """
    if layer.k_scale is None or layer.v_scale is None:
        # Explicit FP8 KV cache on an otherwise unquantized model historically
        # installed scales only when an external calibration file was loaded.
        # Preserve that path while keeping the scales as registered parameters.
        create_kv_scale_parameters(layer, per_head=False)

    if isinstance(scaling_factor, Mapping):
        k_factor = scaling_factor["k"]
        v_factor = scaling_factor["v"]
    elif hasattr(scaling_factor, "k") and hasattr(scaling_factor, "v"):
        k_factor = scaling_factor.k
        v_factor = scaling_factor.v
    else:
        k_factor = v_factor = scaling_factor

    k_loaded = torch.as_tensor(
        k_factor, dtype=layer.k_scale.dtype, device=layer.k_scale.device
    )
    v_loaded = torch.as_tensor(
        v_factor, dtype=layer.v_scale.dtype, device=layer.v_scale.device
    )
    _load_kv_scale(layer.k_scale, k_loaded)
    _load_kv_scale(layer.v_scale, v_loaded)


def create_kv_scale_parameters(layer: torch.nn.Module, per_head: bool) -> None:
    shape = (layer.tp_k_head_num,) if per_head else ()
    layer.k_scale = torch.nn.Parameter(
        torch.full(shape, -1.0, dtype=torch.float32), requires_grad=False
    )
    layer.v_scale = torch.nn.Parameter(
        torch.full(shape, -1.0, dtype=torch.float32), requires_grad=False
    )
    for scale in (layer.k_scale, layer.v_scale):
        scale._skip_weight_check = True
        scale.weight_loader = _load_kv_scale


class BaseKVCacheMethod(QuantizeMethodBase):
    """
    Quant method that adds `k_scale` and `v_scale` attributes to the
    Attention layer to support loading those scaling factors from checkpoints.
    The k/v_scale will be used to:
        - quantize k/v_cache entries before saving them to the cache
        - dequantize k/v_cache entries before fetching them from the cache

    :param quant_config: the appropriate QuantizationConfig
    """

    def __init__(self, quant_config: QuantizationConfig):
        self.quant_config = quant_config

    def create_weights(self, layer: torch.nn.Module):
        """
        Create "weight" (aka k_scale and v_scale) for an attention layer.
        """
        # Initialize the KV cache scales to -1.0, which is an invalid value.
        # If the k/v_scale appears in the checkpoint, it will be overwritten.
        create_kv_scale_parameters(
            layer, use_per_head_kv_scale(getattr(layer, "prefix", ""))
        )

    def apply(self, layer: torch.nn.Module) -> torch.Tensor:
        raise RuntimeError(f"{self.__class__.__name__}.apply should not be called.")

    def process_weights_after_loading(self, layer) -> None:
        k_scale = layer.k_scale.detach()
        v_scale = layer.v_scale.detach()
        k_valid = k_scale > 0.0
        v_valid = v_scale > 0.0

        # Prefer separate scales. A checkpoint containing only one scale uses
        # the valid value for both K and V, independently for every head.
        if torch.all(k_valid) and torch.all(v_valid):
            pass
        elif torch.all(~k_valid) and torch.all(~v_valid):
            k_scale = torch.ones_like(k_scale)
            v_scale = torch.ones_like(v_scale)
        elif torch.all(k_valid | v_valid):
            shared_scale = torch.maximum(k_scale, v_scale)
            k_scale = shared_scale
            v_scale = shared_scale
        else:
            raise ValueError("FP8 KV cache scales must be positive for every head")

        if is_fp8_fnuz():
            k_scale = k_scale * 2
            v_scale = v_scale * 2

        layer.k_scale.copy_(k_scale)
        layer.v_scale.copy_(v_scale)
        if layer.k_scale.numel() == 1:
            layer.k_scale_float: Union[float, None] = layer.k_scale.item()
            layer.v_scale_float: Union[float, None] = layer.v_scale.item()
        else:
            # Scalar-only backends use the *_float attributes. Per-head scales
            # are consumed directly as tensors by FA3.
            layer.k_scale_float = None
            layer.v_scale_float = None
