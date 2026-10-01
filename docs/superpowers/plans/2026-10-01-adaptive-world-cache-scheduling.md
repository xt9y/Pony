# Stage 11 Adaptive World-Cache Scheduling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Stage-10 CPU round-robin world-probe updates with bounded GPU adaptive scheduling driven by revision state, dirtiness, convergence residual, confidence and frame age.

**Architecture:** One 64-thread selector workgroup partitions all active world probes into disjoint rotating lane subsets and emits at most one winner per lane into the existing `RadianceUpdateList`. `CS_UpdateWorldRadianceCache` keeps Stage-10 stable two-bank propagation but now measures `statistics.y` convergence residual; the CPU stops generating/uploading per-frame world update lists.

**Tech Stack:** C11, NRI/Vulkan, Slang/HLSL to SPIR-V 1.5, SDL3, GitHub Actions verification.

**Spec:** `docs/superpowers/specs/2026-10-01-adaptive-world-cache-scheduling-design.md`

## Global Constraints

- Preserve Stage-10 world cache representation, bindings, 8192 probes, 16384 hash slots, 16 directions, 2 banks and 64 updates/frame.
- Add only `CS_SelectWorldProbeUpdates` for scheduling; keep `CS_UpdateWorldRadianceCache` as the world-cache update entry point.
- No CPU/GPU readback and no per-frame CPU world-update-list upload.
- Reuse `RadianceUpdateList` and `RayCounters[3]`; add no persistent scheduler buffer.
- Selector dispatch is exactly one 64-thread workgroup.
- Keep Stage-10 stable previous-bank semantics unchanged.
- Fix the Stage-10 source-traversal determinism tie-break before adaptive scheduling.
- Keep execution-only workflow/helper files out of the final tree.

## Review Focus

- Equal Stage-10 placement candidates: same cell/clearance/surface must still choose deterministically across libc `qsort` implementations.
- Small caches: 0..63 active probes must not produce duplicate/out-of-range list entries or dispatch invalid work.
- Frame wrap: staleness must remain valid across the 24-bit frame-stamp wrap.
- Lighting revision changes: stale probes must become highest priority without exposing stale radiance to lookup.
- Stable scenes: residual-driven prioritization must not starve low-residual probes forever; rotating partitions and frame age must preserve eventual coverage.

---

### Task 1: Close the Stage-10 deterministic placement tie

**Files:**
- Modify: `sdf.c`
- Modify: `tests/stage10_world_cache.c`

**Interfaces:**
- Consumes: existing `WORLD_PROBE_CANDIDATE` sorting and deterministic synthetic Stage-10 builder test.
- Produces: explicit `source_order` final comparator key with no GPU/runtime ABI change.

- [ ] **Step 1: Extend the permanent Stage-10 regression test**

Add a source-contract assertion that `WORLD_PROBE_CANDIDATE` contains `source_order`, the comparator orders on it after `surface_id`, and candidate traversal assigns monotonically increasing order.

- [ ] **Step 2: Run test to verify RED**

Run the Stage-10 architecture/source check before production modification.

Expected: FAIL because `source_order` is absent.

- [ ] **Step 3: Implement minimal deterministic tie-break**

Add `uint64_t source_order` to the private candidate, assign from one monotonic traversal counter when each accepted candidate is appended, and compare ascending only after equal cell, clearance and surface ID.

- [ ] **Step 4: Run Stage-10 CPU regression + sanitizer + source check**

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add sdf.c tests/stage10_world_cache.c
git commit -m "s10 stabilize world probe placement order"
```

### Task 2: Add adaptive selector and convergence residual

**Files:**
- Modify: `shader.hlsl`
- Modify: `build.c`

**Interfaces:**
- Consumes: Stage-10 `WorldProbeState`, `RadianceUpdateList`, `RayCounters[3]`, `Pass.dispatch.x`, world constants and stable-bank lookup.
- Produces: `CS_SelectWorldProbeUpdates`, residual updates in `statistics.y`, and `radiance_world_select.cs.spv` build job.

- [ ] **Step 1: Add failing Stage-11 shader architecture checks**

Require:

```text
CS_SelectWorldProbeUpdates uses [numthreads(64,1,1)]
rotating base = Pass.dispatch.x % active_count
logical offsets are lane + k*64 and map modulo active_count
age = (Pass.dispatch.x - state.state.x) & 0x00ffffffu
priority classes revision/unpublished > dirty > severe stale > normal
same-class tie order residual desc, confidence asc, age desc, index asc
InterlockedAdd(RayCounters[3], 1u, slot) appends at most one winner/lane
CS_UpdateWorldRadianceCache computes mean relative directional delta
statistics.y first update = measured; established = lerp(old, measured, 0.25)
build.c contains exactly one CS_SelectWorldProbeUpdates job
```

- [ ] **Step 2: Run checks to verify RED**

Expected: FAIL because selector/residual build entry do not exist.

- [ ] **Step 3: Implement selector helper/entry point**

Use one local winner per lane with lexicographic comparison from the spec. `sweep_frames = max((active_count + 63u) / 64u, 1u)` and severe stale threshold is `2 * sweep_frames`.

- [ ] **Step 4: Add residual measurement to world-cache update**

Accumulate `length(sample - old) / (1.0 + length(sample))` over the 16 directions and update `statistics.y` exactly per spec without changing bank publication order.

- [ ] **Step 5: Add build job**

Add `{ "CS_SelectWorldProbeUpdates", "compute", "radiance_world_select.cs.spv", NULL }` adjacent to the world-cache update job.

- [ ] **Step 6: Compile selector/update/invalidation shaders and run architecture check**

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add shader.hlsl build.c
git commit -m "s11 select adaptive world probes"
```

### Task 3: Wire GPU-owned scheduling and remove CPU round-robin uploads

**Files:**
- Modify: `game.h`
- Modify: `render.c`

**Interfaces:**
- Consumes: Task-2 `radiance_world_select.cs.spv` and Stage-10 wavefront/world descriptors.
- Produces: `world_radiance_select_pipeline`, selector→barrier→update frame ordering, and removal of per-frame CPU list generation/upload.

- [ ] **Step 1: Add failing renderer architecture checks**

Require:

```text
RENDERER owns world_radiance_select_pipeline
create_pipelines loads radiance_world_select.cs.spv
pipeline destruction includes world_radiance_select_pipeline
prepare_world_probe_updates is absent
stream_dynamic_data has no world_update_data/upload_world_updates/cpu_update_list streaming
CS_ResetWavefront clears RayCounters[3] to zero
when WORLD_CACHE active: selector dispatch = {1,1,1}, wavefront barrier, world update dispatch = {1,1,1}, world-cache barrier
```

- [ ] **Step 2: Run checks to verify RED**

Expected: FAIL on current CPU scheduling/upload path.

- [ ] **Step 3: Remove CPU scheduler state/use**

Remove `cpu_update_list`, `update_count`, `update_cursor` from `RADIANCE_WORLD_RESOURCES`; remove allocation/free/reset and `prepare_world_probe_updates`; remove update-list streamer data/barriers. Leave GPU `wavefront.update_list` allocation intact.

- [ ] **Step 4: Add selector pipeline ownership**

Add/create/destroy `world_radiance_select_pipeline` with the existing wavefront layout.

- [ ] **Step 5: Change reset and frame order**

`CS_ResetWavefront` sets `RayCounters[3] = 0u`. After reset+wavefront barrier, active world cache dispatches selector once, barriers the wavefront buffers, dispatches world update once, then barriers world radiance before screen-probe work.

- [ ] **Step 6: Run all C syntax checks and Stage-11 architecture checks**

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add game.h render.c shader.hlsl
git commit -m "s11 schedule world cache on gpu"
```

### Task 4: Full verification and cleanup

**Files:**
- Verify: all renderer C translation units
- Verify: all active Slang shader jobs
- Remove: execution-only `.github/stage11_*` helpers/workflow

**Interfaces:**
- Consumes: Tasks 1-3 production tree.
- Produces: clean Stage-11 branch ready for M2 runtime validation.

- [ ] **Step 1: Run permanent Stage-10 CPU regression**

```bash
clang -std=c11 $FLAGS tests/stage10_world_cache.c sdf.c -lm -o /tmp/stage10_world_cache_test && /tmp/stage10_world_cache_test
clang -std=c11 -fsanitize=address,undefined -fno-omit-frame-pointer $FLAGS tests/stage10_world_cache.c sdf.c -lm -o /tmp/stage10_world_cache_asan && /tmp/stage10_world_cache_asan
```

Expected: PASS with no sanitizer report.

- [ ] **Step 2: Run Stage-11 architecture checks**

Expected: PASS all determinism, selector, residual, no-readback/upload and frame-order checks.

- [ ] **Step 3: Syntax-compile all C translation units**

```bash
for f in init.c glb.c gltf.c scene.c sdf.c gpu.c render.c main.c; do clang -std=c11 -fsyntax-only $FLAGS "$f" || exit 1; done
```

Expected: PASS.

- [ ] **Step 4: Compile every normal Slang job**

Include `CS_SelectWorldProbeUpdates` and all existing entries with the same SPIR-V/profile/layout flags as `build.c`.

Expected: PASS and non-empty SPIR-V output for each entry.

- [ ] **Step 5: Run `git diff --check`**

Expected: PASS.

- [ ] **Step 6: Remove execution-only helpers and verify only those cleanup paths changed after the production commit**

Keep permanent docs and `tests/stage10_world_cache.c`.

- [ ] **Step 7: Final branch review**

Review Stage-11 production diff against the spec, with particular attention to the five Review Focus cases.
