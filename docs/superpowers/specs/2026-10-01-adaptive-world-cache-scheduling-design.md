# Pony Radiance Stage 11: adaptive world-cache scheduling

## Goal

Stage 11 replaces the Stage-10 CPU round-robin world-probe selector with GPU-owned adaptive scheduling while preserving the Stage-10 world-cache representation and bounded update budget.

The scheduler should spend the existing 64-probe/frame budget on probes that need work most: unpublished/revision-stale probes first, then dirty probes, then badly stale probes, then high-residual/low-confidence/old probes. It must require no CPU readback and must not change camera-independent probe placement.

## Constraints

- Preserve Stage-10 world-probe placement, hash table, two-bank radiance representation, capacities, descriptor spaces and lookup semantics.
- Keep exactly 8192 maximum active probes, 16384 hash slots, 16 directions/probe, 2 banks and at most 64 probe updates per frame.
- Keep `CS_UpdateWorldRadianceCache` as the world-cache update entry point.
- Add exactly one scheduling entry point: `CS_SelectWorldProbeUpdates`.
- Scheduling is GPU-owned; no per-frame world-update-list upload and no CPU/GPU readback.
- Reuse the existing `RadianceUpdateList` and `RayCounters[3]` buffers; add no persistent scheduler buffer.
- The selector dispatch is exactly one 64-thread workgroup.
- Each active probe is examined by exactly one selector lane per frame.
- Same-frame update order remains irrelevant because Stage-10 stable previous-bank reads are preserved.
- Camera motion must not alter world-probe placement or scheduler identity state.
- Static scene replacement still rebuilds the cache; lighting revision changes still invalidate published radiance logically through revision mismatch.
- Stage 11 does not add targeted dynamic-object spatial invalidation or reflections.
- Fix the remaining Stage-10 placement determinism discrepancy before enabling the adaptive selector: equal cell/clearance/surface candidates must use source traversal order as the final tie-break.

## Stage-10 determinism correction

`WORLD_PROBE_CANDIDATE` gains a monotonically increasing `source_order` assigned from deterministic clipmap/sample traversal before sorting. The comparator order is:

1. integer placement cell x/y/z ascending;
2. clearance descending;
3. canonical surface ID ascending;
4. source traversal order ascending.

This makes the chosen representative independent of implementation-specific `qsort` ordering for otherwise equal candidates.

## GPU selector

`CS_SelectWorldProbeUpdates` uses `[numthreads(64, 1, 1)]` and one workgroup.

For active probe count `N`, lane `L` scans logical offsets:

```text
L, L + 64, L + 128, ... < N
```

with a rotating base:

```text
base = Pass.dispatch.x % N
probe_index = (base + logical_offset) % N
```

This visits every active probe exactly once per frame, partitions candidates without overlap, and rotates which physical probes share a lane over time.

Each lane chooses at most one local winner. A lane with no candidate emits nothing. Winners append to `RadianceUpdateList` through `InterlockedAdd(RayCounters[3], 1, slot)`. Therefore selection produces at most 64 unique probes without a global sort.

Selection order inside `RadianceUpdateList` is intentionally unspecified. Stage-10 double-bank rules make update execution order semantically irrelevant.

## Priority ordering

The selector compares candidates lexicographically.

Priority class, highest first:

```text
3 = unpublished or state.y != current lighting revision
2 = dirty bit set
1 = severely stale
0 = normal
```

Frame age uses the existing 24-bit frame stamp:

```text
age_frames = (Pass.dispatch.x - state.state.x) & 0x00ffffff
sweep_frames = max(ceil(active_count / 64), 1)
severely_stale = age_frames >= 2 * sweep_frames
```

Within equal classes, prefer in order:

1. larger `statistics.y` residual;
2. lower `statistics.x` confidence;
3. larger `age_frames`;
4. smaller probe index for deterministic per-lane tie resolution.

This is deliberately not an exact global top-64 sort. The rotating partition plus age term provides bounded work and eventual coverage without extra buffers or readback.

## Residual measurement

`statistics.y` becomes the Stage-11 convergence residual.

During `CS_UpdateWorldRadianceCache`, each of the 16 directional samples contributes:

```text
relative_delta = length(sample - old_value) / (1 + length(sample))
```

The measured probe residual is the mean of the 16 directional relative deltas.

After the update:

```text
if previously established for the current revision:
    statistics.y = lerp(previous_residual, measured_residual, 0.25)
else:
    statistics.y = measured_residual
```

`statistics.x` remains confidence and `statistics.z` remains completed-update age/count as defined by Stage 10.

## Frame order

The world-cache portion becomes:

```text
CS_ResetWavefront
  -> RayCounters[3] = 0
  -> wavefront UAV barrier
CS_SelectWorldProbeUpdates (1 x 64 threads)
  -> writes RadianceUpdateList + RayCounters[3]
  -> wavefront UAV barrier
CS_UpdateWorldRadianceCache (1 x 64 threads when world cache active)
  -> reads selected count from RayCounters[3]
  -> writes world-probe state/radiance
  -> world-cache UAV barrier
normal screen-probe path
```

The update dispatch is always one workgroup when the world cache is active; shader early-outs for lanes beyond the selected count.

## CPU/runtime changes

The renderer no longer calls `prepare_world_probe_updates`, no longer fills `cpu_update_list`, and no longer streams a world update list each frame.

`RADIANCE_WORLD_RESOURCES` no longer needs CPU scheduler state (`cpu_update_list`, `update_count`, `update_cursor`). GPU resource representation is unchanged.

`RENDERER` gains `world_radiance_select_pipeline`; pipeline creation/destruction includes `radiance_world_select.cs.spv`.

`build.c` adds exactly one shader job for `CS_SelectWorldProbeUpdates`.

## Failure behavior

- Zero active world probes: selector/update dispatches are skipped and Stage-10 feature flags remain disabled.
- Fewer than 64 active probes: at most one winner per occupied logical lane; no out-of-range update-list writes.
- Revision mismatch: stale published data remains rejected by Stage-10 lookup and receives highest scheduling class until refreshed.
- 24-bit frame-stamp wrap: age uses masked subtraction.
- A selector lane never emits the same probe as another lane because the logical-offset partition is disjoint.

## Verification

Stage 11 is complete only when demonstrated:

1. Stage-10 equal-candidate placement has an explicit source-order tie-break;
2. repeated Stage-10 synthetic placement remains byte deterministic;
3. no CPU per-frame world update-list generation/upload remains;
4. selector is exactly one 64-thread workgroup;
5. every active probe belongs to exactly one lane partition per frame;
6. selected count never exceeds 64;
7. selected probe indices are in range and unique by construction;
8. invalid/revision-stale beats dirty, dirty beats severely stale, severely stale beats normal;
9. equal-class comparison follows residual, confidence, frame age, index;
10. frame age uses masked 24-bit subtraction;
11. update computes and smooths `statistics.y` residual;
12. `CS_ResetWavefront` clears `RayCounters[3]` before selection;
13. selector dispatch is followed by a UAV barrier before world update;
14. Stage-10 stable previous-bank lookup behavior is unchanged;
15. all C translation units syntax-compile;
16. all active Slang entries including the new selector compile;
17. `git diff --check` passes;
18. execution-only verification helpers are removed after verification;
19. M2 Cornell runtime remains the final visual/performance validation for convergence quality and scheduling behavior.

## Non-goals

- global GPU sorting/top-k infrastructure
- CPU readback scheduling
- changing probe placement density/capacity
- targeted dynamic-object invalidation regions
- reflection scheduling/denoising
- recursive path tracing
- camera-following world probes
