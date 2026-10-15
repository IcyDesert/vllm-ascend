# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import importlib.util
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from safetensors.torch import save_file

from .rfork_test_support import _load_module, _stub


@pytest.fixture
def engram_module(monkeypatch):
    """Keep the real checkpoint reader, replacing only unavailable runtime imports."""
    _stub(monkeypatch, "vllm.logger", logger=logging.getLogger("rfork-local-test"))
    _stub(
        monkeypatch,
        "vllm.distributed",
        get_tensor_model_parallel_rank=lambda: 0,
        get_tensor_model_parallel_world_size=lambda: 1,
        tensor_model_parallel_all_gather=lambda t: t,
    )
    _stub(
        monkeypatch, "vllm.distributed.parallel_state", get_tp_group=lambda: None, in_the_same_node_as=lambda g: [True]
    )
    _stub(monkeypatch, "vllm.model_executor.utils", set_weight_attrs=lambda *args: None)
    _stub(monkeypatch, "vllm.models.deepseek_v41.common.engram", ParallelEngramEmbedding=torch.nn.Module)
    prefix = "rfork_test_engram"
    _stub(monkeypatch, prefix, __path__=[])
    _stub(
        monkeypatch,
        f"{prefix}.npu",
        HostUvaBuffer=object,
        SharedUvaBuffer=object,
        gather_dequantize_engram_int8=Mock(),
        gather_dequantize_host_uva=Mock(),
        quantize_engram_rows=Mock(),
    )
    _stub(
        monkeypatch,
        f"{prefix}.parallel",
        engram_head_shard_rank=lambda: 0,
        exchange_engram_rows=Mock(),
        gather_engram_hashes=Mock(),
        get_engram_dp_group=lambda: None,
        get_engram_dp_size=lambda: 1,
    )
    path = Path(__file__).resolve().parents[4] / "vllm_ascend/models/deepseek_v41/engram/embedding.py"
    spec = importlib.util.spec_from_file_location(f"{prefix}.embedding", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def make_table(module, tmp_path, dtype=torch.int8):
    table = module.AscendParallelEngramEmbedding.__new__(module.AscendParallelEngramEmbedding)
    torch.nn.Module.__init__(table)
    table.cpu_offload = True
    table.storage_dtype = dtype
    table.scale_dtype = torch.uint8 if dtype == torch.float8_e4m3fn else torch.float32
    table.block_size = 32
    table.dim = 32
    table.vocab_start_idx, table.vocab_end_idx = 1, 3
    table._shared_group = None
    table.weight = torch.nn.Parameter(torch.zeros(2, 32, dtype=dtype), requires_grad=False)
    table.weight_scale_inv = (
        None
        if dtype == torch.bfloat16
        else torch.nn.Parameter(torch.zeros(2, 1, dtype=table.scale_dtype), requires_grad=False)
    )
    table.bind_checkpoint(tmp_path, "layers.1.engram.embed.weight")
    return table


@pytest.mark.parametrize("dtype", [torch.int8, torch.bfloat16, torch.float8_e4m3fn])
def test_engram_loads_real_checkpoint_before_cpu_state_is_accepted(engram_module, tensor_runtime, tmp_path, dtype):
    table = make_table(engram_module, tmp_path, dtype)
    key = "layers.1.engram.embed.weight"
    source = torch.arange(128).reshape(4, 32).to(dtype)
    tensors = {key: source}
    if table.weight_scale_inv is not None:
        tensors[key.removesuffix(".weight") + ".scale"] = torch.full((4, 1), 2, dtype=table.scale_dtype)
        if dtype == torch.float8_e4m3fn:
            tensors[key.removesuffix(".weight") + ".scale"] = tensors[key.removesuffix(".weight") + ".scale"].view(
                torch.float8_e8m0fnu
            )
    save_file(tensors, tmp_path / "table.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: "table.safetensors" for name in tensors}})
    )
    assert "weight" in tensor_runtime.tensor_layout.find_non_npu_state_tensors(table)

    table.load_rfork_local_weights()

    assert torch.equal(table.weight.view(torch.uint8), source[1:3].view(torch.uint8))
    if table.weight_scale_inv is not None:
        expected_scale = tensors[key.removesuffix(".weight") + ".scale"][1:3]
        assert torch.equal(table.weight_scale_inv.view(torch.uint8), expected_scale.view(torch.uint8))
    assert tensor_runtime.tensor_layout.find_non_npu_state_tensors(table) == []
    assert tensor_runtime.tensor_layout.collect_transferable_tensors(table, True) == []
    table._load_into_storage = Mock(side_effect=AssertionError("already loaded"))
    table.load_rfork_local_weights()


def test_unknown_cpu_state_and_replaced_engram_parameter_remain_rejected(engram_module, tensor_runtime, tmp_path):
    table = make_table(engram_module, tmp_path)
    table._load_into_storage = Mock()
    table.load_checkpoint(tmp_path, "unused")
    table.register_buffer("unhandled", torch.ones(1))
    assert tensor_runtime.tensor_layout.find_non_npu_state_tensors(table) == ["unhandled"]
    table.weight = torch.nn.Parameter(torch.zeros_like(table.weight), requires_grad=False)
    assert "weight" in tensor_runtime.tensor_layout.find_non_npu_state_tensors(table)


def test_failed_checkpoint_reload_revokes_local_state(engram_module, tensor_runtime, tmp_path):
    table = make_table(engram_module, tmp_path)
    table._load_into_storage = Mock()
    table.load_checkpoint(tmp_path, "unused")
    table._load_into_storage = Mock(side_effect=RuntimeError("disk read failed"))
    with pytest.raises(RuntimeError, match="disk read failed"):
        table.load_checkpoint(tmp_path, "unused")
    assert "weight" in tensor_runtime.tensor_layout.find_non_npu_state_tensors(table)


def test_shared_checkpoint_failure_never_marks_peer_loaded(engram_module, monkeypatch, tmp_path):
    table = make_table(engram_module, tmp_path)
    table._shared_group = SimpleNamespace(rank_in_group=1, world_size=2, cpu_group=object())
    table._load_into_storage = Mock(side_effect=AssertionError("non-writer"))

    def synchronize(errors, error, group):
        errors[:] = ["disk read failed", None]

    monkeypatch.setattr(engram_module.dist, "all_gather_object", synchronize)
    with pytest.raises(RuntimeError, match="disk read failed"):
        table.load_rfork_local_weights()
    assert table.get_rfork_local_state_tensors() == ()


def test_npu_engram_does_not_read_local_checkpoint(engram_module, tmp_path):
    table = make_table(engram_module, tmp_path)
    table.cpu_offload = False
    table.load_checkpoint = Mock(side_effect=AssertionError("NPU weights must use RFork"))
    table.load_rfork_local_weights()
    assert table.get_rfork_local_state_tensors() == ()


def test_local_weight_preparation_walks_each_module_once(monkeypatch):
    safety = _load_module(monkeypatch, "rfork_local_safety_test", "safety.py")
    table = torch.nn.Module()
    table.load_rfork_local_weights = Mock()
    model = torch.nn.Module()
    model.table = table
    model.shared_alias = table
    safety.load_local_weights(model)
    table.load_rfork_local_weights.assert_called_once_with()


def test_same_name_cpu_parameter_is_not_an_engram_exemption(tensor_runtime):
    model = torch.nn.Module()
    model.engram = torch.nn.Module()
    model.engram.weight = torch.nn.Parameter(torch.ones(1))
    assert tensor_runtime.tensor_layout.find_non_npu_state_tensors(model) == ["engram.weight"]


def test_local_state_report_cannot_exempt_another_modules_storage(tensor_runtime):
    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(torch.ones(1))
    model.child = torch.nn.Module()
    model.child.load_rfork_local_weights = Mock()
    model.child.get_rfork_local_state_tensors = lambda: (model.weight,)
    assert tensor_runtime.tensor_layout.find_non_npu_state_tensors(model) == ["weight"]


@pytest.mark.parametrize("replace_storage", [False, True])
def test_checkpoint_rebind_and_storage_replacement_revoke_readiness(
    engram_module, tensor_runtime, tmp_path, replace_storage
):
    table = make_table(engram_module, tmp_path)
    table._load_into_storage = Mock()
    table.load_checkpoint(tmp_path, "unused")
    if replace_storage:
        table.weight.data = torch.zeros_like(table.weight)
    else:
        table.bind_checkpoint(tmp_path, "another.weight")
    assert "weight" in tensor_runtime.tensor_layout.find_non_npu_state_tensors(table)


def test_shared_checkpoint_success_marks_nonwriter_ready(engram_module, monkeypatch, tmp_path):
    table = make_table(engram_module, tmp_path)
    table._shared_group = SimpleNamespace(rank_in_group=1, world_size=2, cpu_group=object())
    table._load_into_storage = Mock(side_effect=AssertionError("non-writer"))

    def synchronize(errors, error, group):
        errors[:] = [None, None]

    monkeypatch.setattr(engram_module.dist, "all_gather_object", synchronize)
    table.load_rfork_local_weights()
    assert table.get_rfork_local_state_tensors()[0] is table.weight
