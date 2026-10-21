# Two-stage cloud validation for e18789f2

Upload patches 01 and 02 together. No ZIP, patch.py or manifest is required.
Keep SHA256SUMS for copy verification and this guide for the run order.
The runtime files on this delivery branch remain the unmodified target commit;
the optional fault is stored as a patch and is not enabled by checking out the branch.

| File | Apply to | Purpose |
| --- | --- | --- |
| `01-rfork-e18789f2-from-8fde6d4e.patch` | Every seed and receiver node | Complete production delta from pristine rc2 to e18789f2, including the OOM fix. |
| `02-receiver-rank4-seed-miss.patch` | Receiver nodes only, after normal validation | One simulated main-model seed miss on global rank 4 / TP rank 0. |

Pinned source revisions:

- Image baseline: `8fde6d4e335fe300e2b949aab0e31659cce0811d`.
- Target under test: `e18789f237c03949a503ed4e34d722cb6788fa87`.
- Source branch: `yangsonglin13/vllm-ascend`, `codex/v0.30.0rc2-rfork-core`.
- vLLM expected by this rc2 environment: `ced6857afa0ea7b2e3f0846a62e1394e90f15607`.
- Normal test state: exactly the target runtime, including the ModelSlim OOM fix.
  No separate OOM patch is required.

Apply these files only to a fresh, unpatched 0.30.0rc2 image at the baseline
above. No older patches or migration steps are needed. This is not a vLLM 0.31 kit.

## Preparation

Use the previously successful two-node topology per deployment: TP4, DP4,
local DP size 2, EP16, with PCP=PP=1. Both nodes see their eight local NPUs.
The API node has no explicit DP start rank; the headless node starts at DP rank 2.
Use the same topology for seed and receiver and explicitly set:

```text
--engram-config '{"cpu_offload":true,"embedding_across_dp":false,"dp_shared_memory":true}'
```

Bring the working launch scripts, planner.sh, YuanRong wheel and offline
dependencies, and complete checkpoint files. Check host RAM and /dev/shm capacity
before startup. Use a fresh model_url shared by this test cohort, for example
`dsv41-e18789f2-shared`; do not change it only on the receiver between stages.

## Copy verification and normal installation

In the directory containing patches 01 and 02, run on each node:

```bash
sha256sum -c SHA256SUMS
RFORK_PATCH_DIR="$PWD"
ASCEND_SOURCE=/vllm-workspace/vllm-ascend
```

SHA256SUMS covers only patches 01 and 02, so the README is optional on the cloud
machine. On Windows, `Get-ChildItem -Filter '*.patch' | Get-FileHash -Algorithm
SHA256` prints the corresponding hashes. Preserve patch whitespace and final
newlines when copying; do not edit expected checksums to hide differences.

Stop existing workers before applying source changes. Confirm the intended source
is the pristine rc2 baseline with no earlier patch edits. Check and apply 01
separately, continuing only after the preceding command succeeds:

```bash
git -C "$ASCEND_SOURCE" apply --check "$RFORK_PATCH_DIR/01-rfork-e18789f2-from-8fde6d4e.patch"
git -C "$ASCEND_SOURCE" apply "$RFORK_PATCH_DIR/01-rfork-e18789f2-from-8fde6d4e.patch"
```

Do not apply 02 yet. No new fault-injection environment variables or CLI options
are needed. Git HEAD stays at the image revision when patches are applied.

## Stage A: normal shared Engram and transfer

1. Start the planner and seed deployment. Initial seed loading may be local.
2. Start the receiver with the same topology and model identity.
3. Confirm resolved shared-memory configuration and an Engram node-local group
   size of 2. Compare the large /dev/shm/psm_ mappings for DP0/TP0 and DP1/TP0:
   matching device numbers and inodes prove shared backing. `(deleted)` is normal.
4. Confirm receiver main AND draft report `source=transfer`; local/fallback alone
   does not establish a healthy transfer baseline.
5. Check /health and the same fixed Chinese prompts. Save logs from both receiver
   nodes, the actual commands, versions and patch hashes.

Only continue to Stage B after this normal baseline passes. These files prepare a
hardware test; the new target's full NPU behavior is not established by CPU tests.

## Stage B: one receiver rank takes the local fallback path

Stop the entire receiver deployment on both nodes and confirm its old workers
have exited. Leave the original seed deployment running. A receiver from Stage A
may also have advertised itself as a seed: ensure those stopped-receiver entries
have expired or been cleared before the next run. If using a dedicated test
planner, it can be restarted and the still-running original seed allowed to
report again. Verify the original seeds are available; stale or missing seeds
that make every rank load locally invalidate this mixed-outcome test.

The normal session.py SHA256 must be:

```text
85f81b84fb58549a1f3307bfc45c4c1827a39710be8fe8b91e8924fe490578bf
```

Check it before installing the fault:

```bash
sha256sum "$ASCEND_SOURCE/vllm_ascend/model_loader/rfork/session.py"
```

On receiver nodes ONLY, check and apply the optional patch:

```bash
git -C "$ASCEND_SOURCE" apply --check "$RFORK_PATCH_DIR/02-receiver-rank4-seed-miss.patch"
git -C "$ASCEND_SOURCE" apply "$RFORK_PATCH_DIR/02-receiver-rank4-seed-miss.patch"
```

Restart both receiver nodes in fresh processes. Editing files under a running
receiver does not inject a failure into an already loaded model. Seed code and
processes must remain free of patch 02.

Use a bounded startup budget longer than the normal load/reload time. For example,
if 20 minutes covers the measured normal startup, run each receiver script with:

```bash
timeout --signal=TERM --kill-after=30s 20m bash node0.sh > receiver-node0-inject.log 2>&1
```

Use node1.sh and a different logfile on the other receiver node. Set the budget
from the Stage A timings; a slow local checkpoint read is not itself a deadlock.
On timeout, stop the whole receiver deployment and verify all workers exited.

The fault is scoped to main model, global rank 4 and TP rank 0. In this topology
that is the non-leader DP1/TP0 peer of DP0/TP0 on the first receiver node. It fires
once per RForkSession after destination registration and before planner allocation,
returning False without acquiring a lease. Other ranks and draft models keep the
normal path. Look for:

```text
[RFORK_FAULT_INJECT] model=main global_rank=4 action=seed_miss once=1 stack_signal=SIGUSR2
```

The same worker should log that seed acquisition was unsuccessful and local loading
is starting. Its TP0 sharing peer must show successful seed transfer for the test
to demonstrate mixed outcomes. If there is no injection marker, or every rank
misses a seed, do not classify the run as a successful fault reproduction.

## Evidence if loading stalls

The injection registers a Python stack-dump handler in the marked worker only.
From that worker's container, use the CURRENT PID printed next to the injection
marker, not the API server or a PID from an earlier run:

```bash
kill -USR2 INJECTED_WORKER_PID
```

This writes all Python thread stacks to that worker's stderr/startup log and
returns; it does not deliberately terminate the registered process. Never send
this signal to an unmarked worker, because only the marked worker has this handler.

Collect the injection marker, source/fallback logs for both sharing peers, and the
stack dump. A stack blocked in SharedUvaBuffer construction's broadcast/all-gather/
barrier, while its peer has left loading, is evidence of the shared fallback
coordination problem. A timeout alone is not enough to locate the cause.

Distinguish three outcomes: coordinated recovery and successful inference;
explicit, bounded failure of the receiver; or a stuck collective. A clean failure
is not proof that automatic fallback recovered successfully. This experiment
simulates a seed miss, not a mid-RDMA transfer failure or corrupted payload.

## Remove the fault

After collecting evidence, stop both receiver nodes. Reverse ONLY patch 02:

```bash
git -C "$ASCEND_SOURCE" apply --reverse --check "$RFORK_PATCH_DIR/02-receiver-rank4-seed-miss.patch"
git -C "$ASCEND_SOURCE" apply --reverse "$RFORK_PATCH_DIR/02-receiver-rank4-seed-miss.patch"
sha256sum "$ASCEND_SOURCE/vllm_ascend/model_loader/rfork/session.py"
```

The hash must return to `85f81b84fb58549a1f3307bfc45c4c1827a39710be8fe8b91e8924fe490578bf`.
Patch 01, including its OOM fix, remains installed. Restart receiver processes and recheck normal
transfer/inference. The old fault process and its signal handler disappear on exit.

## Local checks

The fault-selection suite executes the real patched RForkSession with only its
external planner/NPU/service dependencies stubbed. The unmodified target fails
five injection regressions; the patched version passes all 37 tests. Coverage
includes every rank/draft combination in world size 16, wrong TP rank, one-shot
behavior, lifecycle preconditions and outstanding leases.

Direct replay on pristine rc2 verifies 01, exact target runtime bytes, optional
02, and byte-identical rollback to the normal state. A local Linux subprocess
check confirms the SIGUSR2 diagnostic prints a stack and survives. No real NPU
fault injection or shared-group recovery result is claimed by these local checks.
