# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import logging
import sys
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch.nn import Module
from torch.utils.hooks import RemovableHandle

from .rfork_test_support import _load_module, _stub


@pytest.fixture
def rfork_helpers(monkeypatch):
    """Load the real RFork helpers with only their external imports stubbed."""

    def install(name, **attributes):
        module = _stub(monkeypatch, name, **attributes)
        parent_name, _, child_name = name.rpartition(".")
        parent = sys.modules.get(parent_name)
        if parent is not None:
            monkeypatch.setattr(parent, child_name, module, raising=False)
        return module

    def package(name):
        return install(name, __path__=[])

    package("vllm")
    package("vllm.config")
    install("vllm.config", ModelConfig=object, VllmConfig=object)
    install("vllm.config.load", LoadConfig=object)
    install("vllm.distributed", get_tensor_model_parallel_rank=lambda: 0)
    install("vllm.distributed.parallel_state", get_ep_group=lambda: None, get_pp_group=lambda: None)
    install("vllm.logger", logger=logging.getLogger("rfork-restored-lifecycle-test"))

    package("vllm.model_executor")
    package("vllm.model_executor.model_loader")
    install("vllm.model_executor.model_loader", register_model_loader=lambda name: lambda cls: cls)

    class BaseModelLoader:
        def __init__(self, load_config):
            self.load_config = load_config

    install("vllm.model_executor.model_loader.base_loader", BaseModelLoader=BaseModelLoader)
    install(
        "vllm.model_executor.model_loader.utils",
        initialize_model=lambda **kwargs: None,
        process_weights_after_loading=lambda *args, **kwargs: None,
    )
    package("vllm.utils")
    install("vllm.utils.torch_utils", set_default_torch_dtype=lambda dtype: nullcontext())
    package("vllm.model_executor.layers")
    rope_dict = {"baseline": object()}
    install("vllm.model_executor.layers.rotary_embedding", _ROPE_DICT=rope_dict)

    package("vllm_ascend")
    package("vllm_ascend.device")
    install("vllm_ascend.ascend_config", get_ascend_config=lambda: SimpleNamespace(enable_fused_mc2=0))
    install("vllm_ascend.device.hardware_profile", get_current_hardware_profile=lambda: SimpleNamespace())
    package("vllm_ascend.model_loader")
    package("vllm_ascend.model_loader.rfork")

    safety = _load_module(monkeypatch, "rfork_restored_lifecycle_safety", "safety.py")
    install(
        "vllm_ascend.model_loader.rfork.safety",
        mutable_weights_bypass_reason=safety.mutable_weights_bypass_reason,
        load_local_weights=safety.load_local_weights,
    )
    install("vllm_ascend.model_loader.rfork.config", RForkConfig=object)
    install(
        "vllm_ascend.model_loader.rfork.identity",
        _resolve_sharded_dp_rank=lambda *args, **kwargs: None,
        build_compatibility_fingerprint=lambda *args, **kwargs: "fingerprint",
    )
    install("vllm_ascend.model_loader.rfork.session", RForkSession=object)
    install(
        "vllm_ascend.model_loader.rfork.types",
        RForkFallbackCleanupResult=object,
        RForkIdentity=object,
        RForkLifecycleState=object,
        RForkSeedServiceStartResult=object,
    )

    package("vllm_ascend.eplb")
    package("vllm_ascend.eplb.adaptor")

    class VllmEplbAdaptor:
        _registered_moe_layers = []

    adaptor_module = install("vllm_ascend.eplb.adaptor.vllm_adaptor", VllmEplbAdaptor=VllmEplbAdaptor)
    package("vllm_ascend.ops")
    package("vllm_ascend.ops.fused_moe")

    class AscendMoERunner(Module):
        def __init__(self, quant_method=None):
            super().__init__()
            self._quant_method = quant_method

    install("vllm_ascend.ops.fused_moe.fused_moe", AscendMoERunner=AscendMoERunner)

    class AscendRoutedExperts:
        moe_counter = -1

    routed_experts_module = install("vllm_ascend.ops.fused_moe.routed_experts", AscendRoutedExperts=AscendRoutedExperts)
    rotary_module = install(
        "vllm_ascend.ops.rotary_embedding",
        _cos_sin_cache=None,
        _cos_cache=None,
        _sin_cache=None,
    )

    class DynamoHookRegistry(dict):
        pass

    dynamo_hooks = DynamoHookRegistry()
    install("torch._dynamo.convert_frame", _bytecode_hooks=dynamo_hooks)

    loader = _load_module(monkeypatch, "rfork_restored_lifecycle_loader", "rfork_loader.py")
    return SimpleNamespace(
        loader=loader,
        safety=safety,
        AscendMoERunner=AscendMoERunner,
        adaptor_module=adaptor_module,
        routed_experts_module=routed_experts_module,
        rotary_module=rotary_module,
        dynamo_hooks=dynamo_hooks,
        rope_dict=rope_dict,
    )


@pytest.mark.parametrize(
    ("model_sleep", "vllm_sleep", "weight_transfer", "expected"),
    [
        (True, False, False, "sleep mode"),
        (False, True, False, "sleep mode"),
        (False, False, True, "online weight transfer (weight_transfer_config)"),
        (False, False, False, None),
        (True, False, True, "sleep mode"),
    ],
)
def test_mutable_weights_bypass_reason_reports_sleep_and_weight_transfer(
    rfork_helpers, model_sleep, vllm_sleep, weight_transfer, expected
):
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(enable_sleep_mode=vllm_sleep),
        weight_transfer_config=object() if weight_transfer else None,
    )
    model_config = SimpleNamespace(enable_sleep_mode=model_sleep)

    assert rfork_helpers.safety.mutable_weights_bypass_reason(vllm_config, model_config) == expected


def _add_dynamo_hook(hooks, owner):
    handle = RemovableHandle(hooks)
    hooks[handle.id] = owner.bytecode_hook
    owner._bytecode_hook_handle = handle
    return handle.id


def test_fallback_reset_restores_ascend_globals_and_only_removes_new_dynamo_hooks(rfork_helpers):
    class HookOwner(Module):
        def bytecode_hook(self, *args, **kwargs):
            pass

    adaptor = rfork_helpers.adaptor_module.VllmEplbAdaptor
    baseline_layer = object()
    baseline_registry = [baseline_layer]
    adaptor._registered_moe_layers = baseline_registry
    rfork_helpers.routed_experts_module.AscendRoutedExperts.moe_counter = 17

    baseline_caches = (object(), object(), object())
    (
        rfork_helpers.rotary_module._cos_sin_cache,
        rfork_helpers.rotary_module._cos_cache,
        rfork_helpers.rotary_module._sin_cache,
    ) = baseline_caches

    baseline_owner = HookOwner()
    baseline_hook_id = _add_dynamo_hook(rfork_helpers.dynamo_hooks, baseline_owner)
    vllm_config = SimpleNamespace(compilation_config=SimpleNamespace())
    snapshot = rfork_helpers.loader._snapshot_process_global_model_state(vllm_config)

    baseline_registry.append(object())
    adaptor._registered_moe_layers = [object()]
    rfork_helpers.routed_experts_module.AscendRoutedExperts.moe_counter = 999
    dirty_caches = (object(), object(), object())
    (
        rfork_helpers.rotary_module._cos_sin_cache,
        rfork_helpers.rotary_module._cos_cache,
        rfork_helpers.rotary_module._sin_cache,
    ) = dirty_caches

    discarded_owner = HookOwner()
    discarded_hook_id = _add_dynamo_hook(rfork_helpers.dynamo_hooks, discarded_owner)
    assert discarded_hook_id not in snapshot.dynamo_bytecode_hook_ids

    rfork_helpers.loader._reset_process_global_model_state(vllm_config, snapshot=snapshot)

    assert adaptor._registered_moe_layers is baseline_registry
    assert adaptor._registered_moe_layers == [baseline_layer]
    assert rfork_helpers.routed_experts_module.AscendRoutedExperts.moe_counter == 17
    assert (
        rfork_helpers.rotary_module._cos_sin_cache,
        rfork_helpers.rotary_module._cos_cache,
        rfork_helpers.rotary_module._sin_cache,
    ) == baseline_caches
    assert baseline_hook_id in rfork_helpers.dynamo_hooks
    assert rfork_helpers.dynamo_hooks[baseline_hook_id].__self__ is baseline_owner
    assert discarded_hook_id not in rfork_helpers.dynamo_hooks


@pytest.mark.parametrize("failure", [None, "local", "transfer", "seed_miss"])
@pytest.mark.parametrize("processed_layout", [False, True])
def test_local_weights_are_ready_before_registration_and_failures_fall_back(
    rfork_helpers, monkeypatch, failure, processed_layout
):
    loader_module = rfork_helpers.loader
    loader = loader_module.RForkModelLoader.__new__(loader_module.RForkModelLoader)
    loader.load_config = SimpleNamespace(device=None)
    events = []
    model = Module()
    model.weight = torch.nn.Parameter(torch.zeros(1))

    def load_local():
        events.append("local")
        if failure == "local":
            raise RuntimeError("checkpoint read failed")
        model.weight.data.fill_(7)

    model.load_rfork_local_weights = load_local

    def register(*args):
        assert model.weight.item() == 7
        events.append("register")
        return True

    session = SimpleNamespace(
        register_destination=register,
        acquire_seed=lambda: failure != "seed_miss",
        transfer_from_seed=lambda *args: failure != "transfer",
        log_transferred_model_layout=Mock(),
        prepare_for_fallback=lambda: SimpleNamespace(can_schedule_seed=True),
    )
    fallback_model = Module()
    fallback = Mock(return_value=fallback_model)
    publish = Mock()
    monkeypatch.setattr(loader, "_ensure_rfork_session", lambda *args: session)
    monkeypatch.setattr(loader, "_get_target_registered_blocks", lambda *args: [])
    monkeypatch.setattr(loader, "_requires_processed_layout_transfer", lambda *args: processed_layout)
    monkeypatch.setattr(loader_module, "initialize_model", lambda **kwargs: model)
    monkeypatch.setattr(loader_module, "_load_with_default_loader", fallback)
    monkeypatch.setattr(loader_module, "_start_rfork_seed_service", publish)
    monkeypatch.setattr(loader_module, "_snapshot_process_global_model_state", lambda *args: None)
    monkeypatch.setattr(loader_module, "_reset_process_global_model_state", Mock())
    monkeypatch.setattr(loader_module, "_refresh_rfork_flatquant_state", Mock())
    monkeypatch.setattr(loader_module, "_rfork_skip_unquantized_moe_post_load_processing", lambda model: nullcontext())
    monkeypatch.setattr(
        loader_module,
        "get_ascend_config",
        lambda: SimpleNamespace(eplb_config=SimpleNamespace(dynamic_eplb=False, expert_map_record_path=None)),
    )
    monkeypatch.setattr(
        torch, "npu", SimpleNamespace(empty_cache=lambda: None, synchronize=lambda: None), raising=False
    )
    model_config = SimpleNamespace(dtype=torch.float32)
    config = SimpleNamespace(device_config=SimpleNamespace(device="cpu"), model_config=model_config)

    result = loader.load_model(vllm_config=config, model_config=model_config)

    assert events == (["local"] if failure == "local" else ["local", "register"])
    assert result is (model if failure is None else fallback_model)
    assert fallback.call_count == (0 if failure is None else 1)
    publish.assert_called_once()
    assert publish.call_args.args[1] is result
