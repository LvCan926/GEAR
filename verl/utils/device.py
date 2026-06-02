# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# This code is inspired by the torchtune.
# https://github.com/pytorch/torchtune/blob/main/torchtune/utils/_device.py
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license in https://github.com/pytorch/torchtune/blob/main/LICENSE

import logging
import os

import torch

logger = logging.getLogger(__name__)


def _has_visible_ascend_device() -> bool:
    visible_devices = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
    return visible_devices is not None and visible_devices != ""


def _get_local_rank_from_env(default: int = 0) -> int:
    local_rank = os.environ.get("LOCAL_RANK", os.environ.get("RAY_LOCAL_RANK"))
    if local_rank is None:
        return default
    return int(local_rank)


def is_torch_npu_available() -> bool:
    """Check the availability of NPU"""
    try:
        import torch_npu  # noqa: F401

        if torch.npu.is_available():
            return True
        # Ray NPU workers may expose ASCEND_RT_VISIBLE_DEVICES before torch runtime reports available.
        return _has_visible_ascend_device() and hasattr(torch, "npu")
    except ImportError:
        return False


def _is_cuda_available() -> bool:
    return torch.cuda.is_available()


def _is_npu_available() -> bool:
    return is_torch_npu_available()


# Keep these import-time snapshots for existing callers that treat them as constants.
is_cuda_available = _is_cuda_available()
is_npu_available = _is_npu_available()


def get_device_name() -> str:
    """Function that gets the torch.device based on the current machine.
    This currently only supports CPU, CUDA, NPU.
    Returns:
        device
    """
    if _is_cuda_available():
        device = "cuda"
    elif _is_npu_available():
        device = "npu"
    else:
        device = "cpu"
    return device


def get_torch_device() -> any:
    """Return the corresponding torch attribute based on the device type string.
    Returns:
        module: The corresponding torch device namespace, or torch.cuda if not found.
    """
    device_name = get_device_name()
    try:
        return getattr(torch, device_name)
    except AttributeError:
        logger.warning(f"Device namespace '{device_name}' not found in torch, try to load torch.cuda.")
        return torch.cuda


def get_device_id() -> int:
    """Return current device id based on the device type.
    Returns:
        device index
    """
    return get_torch_device().current_device()


def get_nccl_backend() -> str:
    """Return nccl backend type based on the device type.
    Returns:
        nccl backend type string.
    """
    if _is_cuda_available():
        return "nccl"
    elif _is_npu_available():
        return "hccl"
    else:
        visible_devices = {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "ASCEND_RT_VISIBLE_DEVICES": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        }
        raise RuntimeError(
            "No available accelerator backend found. "
            f"Resolved device type: {get_device_name()}. "
            f"Visible-device env: {visible_devices}."
        )


def get_distributed_backend(include_cpu: bool = False) -> str:
    """Return the torch.distributed backend string for the current runtime device."""
    backend = get_nccl_backend()
    if not include_cpu:
        return backend

    return f"cpu:gloo,{get_device_name()}:{backend}"


def get_hf_attn_kwargs() -> dict:
    """Return safe Hugging Face attention kwargs for the current runtime device."""
    if _is_cuda_available():
        return {"attn_implementation": "flash_attention_2"}
    return {}


def maybe_set_runtime_device() -> None:
    """Best-effort device binding for Ray workers before distributed initialization."""
    local_rank = _get_local_rank_from_env()

    if _has_visible_ascend_device():
        try:
            import torch_npu  # noqa: F401

            torch.npu.set_device(local_rank)
            return
        except Exception as exc:
            logger.warning("Failed to set NPU device to local rank %s: %r", local_rank, exc)

    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, ""):
        try:
            torch.cuda.set_device(local_rank)
        except Exception as exc:
            logger.warning("Failed to set CUDA device to local rank %s: %r", local_rank, exc)
