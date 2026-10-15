# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""RFork loader guards for configurations with mutable model storage."""

from typing import Any

from torch import nn


def load_local_weights(model: nn.Module) -> None:
    """Load module-owned host state before RFork registers or publishes weights.

    Opt-in modules must also expose ``get_rfork_local_state_tensors``. That
    method may report only successfully loaded parameters/buffers, and must
    revoke readiness on reload, storage replacement or teardown. Exceptions
    propagate to the loader's existing fallback path.
    """
    for module in model.modules():
        load = getattr(module, "load_rfork_local_weights", None)
        if callable(load):
            load()


def locally_loaded_state_tensor_ids(model: nn.Module) -> set[int]:
    """Identify explicitly loaded CPU state by ownership, never by weight name."""
    loaded_ids: set[int] = set()
    for module in model.modules():
        report = getattr(module, "get_rfork_local_state_tensors", None)
        if not callable(report) or not callable(getattr(module, "load_rfork_local_weights", None)):
            continue
        owned_ids = {id(tensor) for tensor in (*module.parameters(recurse=False), *module.buffers(recurse=False))}
        loaded_ids.update(id(tensor) for tensor in report() if id(tensor) in owned_ids and tensor.device.type == "cpu")
    return loaded_ids


def mutable_weights_bypass_reason(vllm_config: Any, model_config: Any) -> str | None:
    """Return why RFork must not register weights that can later be replaced."""
    if any(
        bool(getattr(config, "enable_sleep_mode", False))
        for config in (model_config, getattr(vllm_config, "model_config", None))
    ):
        return "sleep mode"
    if getattr(vllm_config, "weight_transfer_config", None) is not None:
        return "online weight transfer (weight_transfer_config)"
    return None
