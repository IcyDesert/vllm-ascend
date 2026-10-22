# Coordinated Engram fallback validation on rc2

Copy all three patches and SHA256SUMS before creating the cloud job. No ZIP,
patch.py or manifest is required. Bring the already validated launch scripts,
planner.sh, YuanRong wheel and offline dependencies, and checkpoint files.

The runtime files on this delivery branch remain at e18789f2. Both the fault
injection and the candidate coordination fix are delivered as optional patches;
checking out this branch alone does not enable either change.

## Pinned versions and files

- Fresh image baseline: `8fde6d4e335fe300e2b949aab0e31659cce0811d` (0.30.0rc2).
- Patch 01 target: `e18789f237c03949a503ed4e34d722cb6788fa87`.
- Target source: `yangsonglin13/vllm-ascend`, `codex/v0.30.0rc2-rfork-core`.
- Expected vLLM: `ced6857afa0ea7b2e3f0846a62e1394e90f15607`.

| File | Apply to | Purpose |
| --- | --- | --- |
| `01-rfork-e18789f2-from-8fde6d4e.patch` | A, B, C, D | Complete rc2-to-e18789f2 runtime delta, including the ModelSlim OOM fix. |
| `02-receiver-rank4-seed-miss.patch` | C, D only | One main-model seed miss on receiver global rank 4 / TP rank 0. |
| `03-rfork-coordinated-engram-fallback.patch` | C, D only | Coordinate Engram loading outcomes and cleanup before fallback reconstruction. |

Patches 01 and 02 are unchanged from the previous kit. Patch 03 changes only
`vllm_ascend/model_loader/rfork/rfork_loader.py`; it does not change model code or
session.py. Apply 01 first. Patches 02 and 03 can then be applied in either order.
Do not apply an additional OOM patch or older RFork patches.

## Topology

Use four machines with eight NPUs each:

- A/B: seed deployment; its DP coordinator is A.
- C/D: receiver deployment; its DP coordinator is C.
- Each deployment: TP4, DP4, local DP size 2, EP16, PCP=PP=1.
- A/C use the API launch script; B/D use the headless script with DP start rank 2.
- Both nodes in each deployment expose all eight local NPUs.

Keep the working launch parameters, including:

```text
--engram-config '{"cpu_offload":true,"embedding_across_dp":false,"dp_shared_memory":true}'
```

All four nodes use the actual planner address and the same model_url and deploy
strategy. Use a fresh cohort identity to avoid old seed records. Keep that
identity unchanged between the two rounds. Retain the previously checked host
RAM and /dev/shm capacity. Log the planner directly to a regular file.

## Installation

In the copied patch directory on each node:

```bash
sha256sum -c SHA256SUMS
RFORK_PATCH_DIR="$PWD"
ASCEND_SOURCE=/vllm-workspace/vllm-ascend
```

On Windows, `Get-ChildItem -Filter '*.patch' | Get-FileHash -Algorithm SHA256`
prints the corresponding hashes. Preserve whitespace and final newlines.

Apply 01 on all four fresh containers. Run each check before its apply, and stop
if any command fails:

```bash
git -C "$ASCEND_SOURCE" apply --check "$RFORK_PATCH_DIR/01-rfork-e18789f2-from-8fde6d4e.patch"
git -C "$ASCEND_SOURCE" apply "$RFORK_PATCH_DIR/01-rfork-e18789f2-from-8fde6d4e.patch"
```

On C/D only, with workers stopped, apply the candidate fix and fault injection:

```bash
git -C "$ASCEND_SOURCE" apply --check "$RFORK_PATCH_DIR/03-rfork-coordinated-engram-fallback.patch"
git -C "$ASCEND_SOURCE" apply "$RFORK_PATCH_DIR/03-rfork-coordinated-engram-fallback.patch"
git -C "$ASCEND_SOURCE" apply --check "$RFORK_PATCH_DIR/02-receiver-rank4-seed-miss.patch"
git -C "$ASCEND_SOURCE" apply "$RFORK_PATCH_DIR/02-receiver-rank4-seed-miss.patch"
```

No new environment variable or CLI option enables the fix. Git HEAD remains at
the image revision when these patches are applied.

## Round 1: injected failure with coordinated recovery

The original squash and normal transfer have already been validated. Start this
new job directly with the recovery test:

1. Start the planner and A/B; wait for the original seeds to become available.
2. Start C/D. Confirm shared Engram remains enabled, with node-local DP group size 2.
3. Confirm C's DP1/TP0/EP4 worker prints
   `[RFORK_FAULT_INJECT] model=main global_rank=4 action=seed_miss once=1 stack_signal=SIGUSR2`.
4. Confirm the coordination warning says all deployment ranks will load locally.
   The injected rank's main summary should be `source=local`; peers with successful
   RFork attempts should report `source=fallback`, not a completed `source=transfer`.
   Summary logs normally appear only on TP rank 0. Draft loading is not injected
   and may still complete with `source=transfer`.
5. Confirm the receiver finishes startup, /health succeeds and the fixed Chinese
   prompts produce coherent output. Save C/D logs, parameters and patch hashes.

An explicit startup error is not successful recovery. Use the measured normal
load/reload time to set a bounded startup budget. If loading stalls, collect the
marked worker's stack with `kill -USR2 INJECTED_WORKER_PID`, using its current PID.
Only the worker named by the injection marker installs that signal handler;
never send SIGUSR2 to an unmarked process. Stop both C/D after collecting evidence.

The previously observed failure without 03 was the injected rank rebuilding the
model alone and blocking in Engram's TP same-node detection broadcast while its
peers had left loading. The test injects a seed miss, not process death or payload
corruption. Phase coordination does not replace timeout handling for failures
inside model collectives.

## Round 2: normal transfer with the fix retained

Stop C/D and verify their workers have exited. Keep A/B running. Reverse only 02:

```bash
git -C "$ASCEND_SOURCE" apply --reverse --check "$RFORK_PATCH_DIR/02-receiver-rank4-seed-miss.patch"
git -C "$ASCEND_SOURCE" apply --reverse "$RFORK_PATCH_DIR/02-receiver-rank4-seed-miss.patch"
sha256sum "$ASCEND_SOURCE/vllm_ascend/model_loader/rfork/session.py"
```

The normal session.py hash is:

```text
85f81b84fb58549a1f3307bfc45c4c1827a39710be8fe8b91e8924fe490578bf
```

C/D may have advertised themselves as seeds during Round 1. Clear or wait for
those stopped-receiver records to expire before restarting C/D. With a dedicated
test planner, restarting it and waiting for A/B to report again is another option.
Ensure original A/B seeds are available, so a stale seed failure does not invalidate
the normal-transfer check.

Restart C/D with 01 and 03 retained. Confirm main AND draft `source=transfer`,
/health and coherent Chinese output. This validates the candidate fix's healthy
path; the earlier squash test does not replace it. Both rounds use the same job:
A/B start once and C/D start twice.

## Local validation and hardware status

The candidate has 18 focused coordination tests, including real two-process CPU
Gloo coverage, plus 37 passing fault-selection tests. Modified Python files pass
Ruff and mypy. The broader RFork suite has 19 failures and 5 errors reproduced
unchanged on the unmodified delivery baseline; it is not an all-green CI result.

Replay checks verify pristine rc2 -> 01 -> 03 -> 02 and pristine rc2 -> 01 -> 02 -> 03,
exact candidate loader bytes, and reversal of 02/03 back to e18789f2. The candidate
has not yet passed NPU fault-recovery validation; the two rounds above provide that
remaining evidence.
