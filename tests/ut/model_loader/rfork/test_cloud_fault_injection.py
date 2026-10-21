# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check the optional cloud fault patch using the real session control flow."""

import importlib.util
import os
import signal
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[4]
RFORK = Path("vllm_ascend/model_loader/rfork")
FAULT_PATCH = ROOT / "examples/rfork/validation_rc2/02-receiver-rank4-seed-miss.patch"


@pytest.fixture
def fault_runtime(tmp_path, monkeypatch):
    for name in ("session.py", "types.py"):
        target = tmp_path / RFORK / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / RFORK / name).read_bytes())
    # An absent patch exercises the unmodified baseline for the red regression run.
    if FAULT_PATCH.is_file():
        env = dict(os.environ, GIT_CEILING_DIRECTORIES=str(tmp_path.parent))
        for key in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
            env.pop(key, None)
        subprocess.run(["git", "-C", str(tmp_path), "apply", str(FAULT_PATCH)], env=env, check=True)

    def stub(name, **attrs):
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
        return module

    def load(name, filename):
        spec = importlib.util.spec_from_file_location(name, tmp_path / RFORK / filename)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module

    for package in ("vllm", "vllm_ascend", "vllm_ascend.model_loader", "vllm_ascend.model_loader.rfork"):
        stub(package, __path__=[])
    logger = Mock()
    stub("vllm.logger", logger=logger)
    prefix = "vllm_ascend.model_loader.rfork"
    types = load(f"{prefix}.types", "types.py")
    stub(f"{prefix}.config", RForkConfig=object)
    stub(f"{prefix}.planner_client", RForkPlannerClient=Mock(), lease_log_id=lambda lease: "test-lease")
    stub(f"{prefix}.seed_client", build_seed_url=Mock(), fetch_seed_transfer_info=Mock())
    stub(
        f"{prefix}.seed_server",
        RForkSeedServerHandle=object,
        RForkSeedServerStartupError=RuntimeError,
        start_rfork_server=Mock(),
    )
    stub(f"{prefix}.tensor_layout", build_structural_digest=Mock(), collect_transferable_tensors=Mock())
    stub(f"{prefix}.transfer_backend", RForkTransferBackend=Mock())
    module = load("rfork_cloud_fault_session", "session.py")
    module.atexit = SimpleNamespace(register=Mock(), unregister=Mock())
    diagnostics = SimpleNamespace(register=Mock())
    monkeypatch.setattr(module, "faulthandler", diagnostics, raising=False)
    config = SimpleNamespace(planner_url="http://planner", model_url="test", model_deploy_strategy_name="test")

    def make_session(rank=4, draft=False):
        identity = types.RForkIdentity(tp_rank=rank % 4, global_rank=rank, is_draft_model=draft)
        session = module.RForkSession(config, identity)
        session.state = types.RForkLifecycleState.REGISTERED
        session.planner = Mock()
        session.planner.acquire_seed.return_value = types.SeedLease("127.0.0.1", 1234, "test-user", rank, "test-key")
        session.planner.take_rejected_lease.return_value = None
        return session

    return SimpleNamespace(make_session=make_session, logger=logger, diagnostics=diagnostics, types=types)


def test_selected_main_rank_misses_once_before_any_planner_allocation(fault_runtime):
    runtime = fault_runtime
    session = runtime.make_session()
    assert session.acquire_seed() is False
    assert session.state is runtime.types.RForkLifecycleState.REGISTERED
    assert session.seed_lease is None
    assert session._lease_acquired_at is None
    session.planner.acquire_seed.assert_not_called()
    session.planner.take_rejected_lease.assert_not_called()
    runtime.diagnostics.register.assert_called_once_with(signal.SIGUSR2, all_threads=True)
    runtime.logger.warning.assert_called_once()
    message, rank = runtime.logger.warning.call_args.args
    assert "[RFORK_FAULT_INJECT]" in message and "seed_miss" in message and rank == 4
    assert session.acquire_seed() is True
    assert session.state is runtime.types.RForkLifecycleState.LEASED
    session.planner.acquire_seed.assert_called_once()
    runtime.logger.warning.assert_called_once()
    runtime.diagnostics.register.assert_called_once()


@pytest.mark.parametrize(
    "rank,draft", [(rank, draft) for rank in range(16) for draft in (False, True) if (rank, draft) != (4, False)]
)
def test_other_ranks_and_draft_keep_normal_acquisition(fault_runtime, rank, draft):
    runtime = fault_runtime
    session = runtime.make_session(rank, draft)
    assert session.acquire_seed() is True
    session.planner.acquire_seed.assert_called_once()
    runtime.logger.warning.assert_not_called()
    runtime.diagnostics.register.assert_not_called()


@pytest.mark.parametrize("state", ["INITIALIZED", "FINALIZED"])
def test_invalid_lifecycle_does_not_consume_injection(fault_runtime, state):
    runtime = fault_runtime
    session = runtime.make_session()
    session.state = runtime.types.RForkLifecycleState[state]
    assert session.acquire_seed() is False
    runtime.logger.warning.assert_not_called()
    session.planner.acquire_seed.assert_not_called()
    session.state = runtime.types.RForkLifecycleState.REGISTERED
    assert session.acquire_seed() is False
    session.planner.acquire_seed.assert_not_called()
    runtime.logger.warning.assert_called_once()


def test_stopping_session_does_not_consume_injection(fault_runtime):
    runtime = fault_runtime
    session = runtime.make_session()
    session.lease_release_stop_event.set()
    assert session.acquire_seed() is False
    runtime.logger.warning.assert_not_called()
    session.lease_release_stop_event.clear()
    assert session.acquire_seed() is False
    session.planner.acquire_seed.assert_not_called()
    runtime.logger.warning.assert_called_once()


def test_outstanding_lease_is_handled_before_injection(fault_runtime):
    runtime = fault_runtime
    session = runtime.make_session()
    session.seed_lease = object()
    session._release_seed_locked = Mock(return_value=True)
    assert session.acquire_seed() is False
    session._release_seed_locked.assert_called_once()
    runtime.logger.warning.assert_not_called()
    session.seed_lease = None
    assert session.acquire_seed() is False
    session.planner.acquire_seed.assert_not_called()
    runtime.logger.warning.assert_called_once()


def test_rank_four_with_different_tp_rank_is_not_injected(fault_runtime):
    runtime = fault_runtime
    session = runtime.make_session()
    session.identity = runtime.types.RForkIdentity(tp_rank=4, global_rank=4)
    assert session.acquire_seed() is True
    session.planner.acquire_seed.assert_called_once()
    runtime.logger.warning.assert_not_called()
    runtime.diagnostics.register.assert_not_called()
