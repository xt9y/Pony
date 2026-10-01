# Stage 10 World Radiance Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a persistent camera-independent world-radiance cache with surface-adaptive static probe placement and stable iterative multi-bounce diffuse GI.

**Architecture:** Stage 9 static/global-SDF data generates deterministic surface-near world probes. The existing space-6 world-cache buffers become real bounded storage, `CS_UpdateWorldRadianceCache` updates at most 64 probes per frame through two directional history banks, and current screen/offscreen surface shading consumes the newest published world indirect radiance while cache updates consume a coherent previous iteration.

**Tech Stack:** C11, NRI/Vulkan, Slang/HLSL to SPIR-V 1.5, SDL3, existing Pony GPU streamer and wavefront renderer.

**Spec:** `docs/superpowers/specs/2026-10-01-world-radiance-cache-design.md`

## Global Constraints

- Preserve the existing NRI/Vulkan/Slang backend and descriptor-space ABI.
- Keep the existing space-6 bindings and `CS_UpdateWorldRadianceCache`; add no second world-cache shader entry point.
- Placement is camera-independent and anchored only by `STATIC` geometry.
- Probe capacity is exactly 8192; hash capacity is exactly 16384.
- Direction map is exactly 4x4 = 16 directions; radiance uses exactly 2 banks.
- Placement spacing is 0.75 world units; influence radius is 1.125; placement offset is 0.075; minimum accepted clearance is 0.0375.
- Fixed update budget is 64 probes/frame = at most 1024 primary world-probe rays/frame.
- Established texels blend 20% new / 80% old; first/invalidated updates never blend undefined or obsolete history.
- Same-frame update order may not expose newly written bank data as previous-iteration input.
- Zero active probes must keep valid descriptors and disable both `WORLD_CACHE` and `MULTIBOUNCE`.
- Camera motion never rebuilds, clears, invalidates, or re-keys world probes.
- `Pass.dispatch.x` is the actual frame stamp; `Radiance.feature_flags.y` remains the scene/lighting revision. Do not conflate them.
- No recursive arbitrary-depth path tracing and no renderer-wide BRDF rewrite.
- Keep repository output clean: no temporary Stage-10 workflow/helper files remain after verification.

## Review Focus

- Negative world coordinates: CPU and shader placement-cell hashing must use identical two's-complement `uint32_t` semantics and still find probes across the 3x3x3 neighborhood.
- Thin/adjacent surfaces: placement clearance and surface-side filtering must prevent a probe across a wall from becoming the preferred diffuse source.
- Hash/update capacities: 8-slot insertion/lookup and a screen-probe count smaller than 64 must never make world probes unreachable or overrun `RadianceUpdateList`.
- Screen-space secondary hits: they currently bypass `SurfaceReflectedRadiance`; Stage 10 must add world indirect there without replacing the existing screen direct term.
- Invalidation/bank timing: lighting revision changes and dirty state must not corrupt the active-bank bit, and same-frame stable reads require a true per-frame `Pass.dispatch.x`.

---

### Task 1: Lock Stage-10 C ownership and builder interfaces

**Files:**
- Modify: `game.h`
- Test: `tests/stage10_world_cache.c`

**Interfaces:**
- Consumes: existing `GLOBAL_SDF_DATA`, `RADIANCE_SCENE_DATA`, `GPU_OBJECT`, `WORLD_PROBE_STATE`, `RADIANCE_WORLD_RESOURCES`.
- Produces:
  - fixed constants `WORLD_PROBE_CAPACITY`, `WORLD_PROBE_HASH_CAPACITY`, `WORLD_PROBE_DIRECTION_SIZE`, `WORLD_PROBE_DIRECTION_COUNT`, `WORLD_PROBE_BANK_COUNT`, `WORLD_PROBE_UPDATES_PER_FRAME`, `WORLD_PROBE_HASH_PROBE_LIMIT`, `WORLD_PROBE_SPACING`, `WORLD_PROBE_RADIUS`, `WORLD_PROBE_CLEARANCE`, `WORLD_PROBE_MIN_CLEARANCE`, `WORLD_PROBE_BLEND`;
  - expanded `RADIANCE_WORLD_RESOURCES` with `cpu_probes`, `cpu_keys`, `cpu_update_list`, `probe_count`, `probe_capacity`, `hash_capacity`, `direction_count`, `bank_count`, `update_count`, `update_cursor`, and existing GPU handles/descriptors/state;
  - `uint32_t radiance_revision` and `NriPipeline *world_radiance_pipeline` in `RENDERER`;
  - public builder signature:

```c
bool sdf_build_world_probes(
    const GLOBAL_SDF_DATA *global_sdf,
    const RADIANCE_SCENE_DATA *radiance_scene,
    const GPU_OBJECT *objects,
    uint32_t object_count,
    WORLD_PROBE_STATE *out_probes,
    uint32_t probe_capacity,
    uint32_t *out_probe_count,
    uint32_t *out_keys,
    uint32_t key_capacity,
    float spacing,
    float radius,
    float clearance,
    float min_clearance
);
```

- [ ] **Step 1: Write the failing interface test**

Create `tests/stage10_world_cache.c` with `_Static_assert`s for the exact Stage-10 capacities and a compile-time reference to `sdf_build_world_probes`. Assert `WORLD_PROBE_STATE` remains 64 bytes and `RADIANCE_CONSTANTS` remains 256 bytes.

- [ ] **Step 2: Run the test to verify RED**

```bash
clang -std=c11 -fsyntax-only -I. -I/tmp/SDL/include -I/tmp/NRI/Include -I/tmp/NRI/Include/Extensions tests/stage10_world_cache.c
```

Expected: FAIL because Stage-10 constants/resource fields/builder declaration do not exist.

- [ ] **Step 3: Add the exact constants, fields, builder declaration and pipeline handle**

Keep all existing GPU ABI structs unchanged. Do not add another descriptor set or world-cache resource family.

- [ ] **Step 4: Re-run the interface test**

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add game.h tests/stage10_world_cache.c
git commit -m "s10 world cache interfaces"
```

### Task 2: Build deterministic static world-probe placement and hash table

**Files:**
- Modify: `sdf.c`
- Modify: `tests/stage10_world_cache.c`

**Interfaces:**
- Consumes: `sdf_build_world_probes(...)` declared in Task 1 and Stage-9 CPU clipmap arrays/canonical triangles.
- Produces: initialized `WORLD_PROBE_STATE[0..probe_count)` and 16384-entry encoded index table where each occupied value is `probe_index + 1`.

- [ ] **Step 1: Extend the test with synthetic placement cases**

Build a minimal synthetic fine/medium `GLOBAL_SDF_DATA`, canonical triangles and object states. Assert:

```text
static geometry => probe_count > 0 and <= 8192
dynamic-only anchors => probe_count == 0
same unchanged input built twice => byte-identical probe sequence and key table
every occupied key slot => decoded index < probe_count and the referenced probe's identity.x matches its placement-cell key
every probe => finite xyz, radius == 1.125, anchor owner is STATIC
placement distance from static geometry >= 0.0375
negative-coordinate fixture => deterministic non-zero keys and reachable probes
collision fixture => no active probe exists unless reachable within 8 open-addressed slots
```

- [ ] **Step 2: Run the behavior test to verify RED**

```bash
clang -std=c11 -I. -I/tmp/SDL/include -I/tmp/NRI/Include -I/tmp/NRI/Include/Extensions tests/stage10_world_cache.c sdf.c -lm -o /tmp/stage10_world_cache_test && /tmp/stage10_world_cache_test
```

Expected: FAIL because `sdf_build_world_probes` has no implementation.

- [ ] **Step 3: Refactor only reusable Stage-9 static-BVH helpers needed by placement**

Keep `SDF_BUILD` temporary to the build call. Add exact CPU equivalents of shader `Hash32`/`HashCombine` using `uint32_t` overflow semantics. Derive each cell with `floorf(position / spacing)`, cast the signed `int32_t` bit pattern to `uint32_t`, then hash x/y/z exactly like the shader.

- [ ] **Step 4: Implement `sdf_build_world_probes(...)`**

Use only fine/medium clipmap samples with valid canonical surface IDs. Reconstruct voxel positions from `GPU_GLOBAL_SDF_CLIPMAP`, reconstruct the canonical world triangle, compute closest point and winding normal, test `+normal * 0.075` then `-normal * 0.075`, require nearest-static clearance >= 0.0375, deduplicate by 0.75-unit cell, choose greatest clearance then smaller surface ID, sort deterministically by integer cell, cap at 8192, and insert each survivor into the 16384 table with an 8-slot limit. Omit candidates that cannot be inserted.

Initialize each `WORLD_PROBE_STATE` with radius 1.125, anchor IDs/revision, confidence/age zero, active bank zero, and valid-published bit clear.

- [ ] **Step 5: Run the placement test**

Expected: PASS all placement/hash/determinism cases.

- [ ] **Step 6: Run sanitizer build**

```bash
clang -std=c11 -fsanitize=address,undefined -fno-omit-frame-pointer -I. -I/tmp/SDL/include -I/tmp/NRI/Include -I/tmp/NRI/Include/Extensions tests/stage10_world_cache.c sdf.c -lm -o /tmp/stage10_world_cache_asan && /tmp/stage10_world_cache_asan
```

Expected: PASS with no sanitizer report.

- [ ] **Step 7: Commit**

```bash
git add sdf.c tests/stage10_world_cache.c
git commit -m "s10 place world probes"
```

### Task 3: Replace dummy world-cache storage with bounded persistent GPU resources

**Files:**
- Modify: `render.c`
- Modify: `tests/stage10_world_cache.c`

**Interfaces:**
- Consumes: Task-2 probe/key arrays and Task-1 capacities.
- Produces: real `WorldProbes`, two-bank `WorldProbeRadiance`, `WorldProbeKeys`, `InvalidationQueue`, world constants/revision, and valid dummy behavior for zero probes.

- [ ] **Step 1: Add failing renderer-contract checks**

Assert pure Stage-10 sizes:

```text
radiance values = 8192 * 16 * 2
radiance bytes = values * sizeof(float[4])
hash slots = 16384
world update list capacity >= 64 independent of screen-probe count
```

Add source-contract assertions that fail until `create_world_radiance_resources` uses these capacities, `create_wavefront` allocates `update_list` for at least `WORLD_PROBE_UPDATES_PER_FRAME`, and `update_radiance_constants` populates `cache_counts.y/z/w`, `world_probe_config`, `world_probe_params`, and the two feature flags.

- [ ] **Step 2: Run the checks to verify RED**

Expected: FAIL on one-element world buffers, screen-probe-sized update list, and disabled world constants.

- [ ] **Step 3: Implement world-resource creation/destruction and scene build**

`create_world_radiance_resources(RENDERER*)` allocates CPU arrays and bounded GPU capacities. `destroy_world_radiance_resources` frees all CPU/GPU ownership. Add `build_world_radiance_scene(RENDERER*)`, called only after Stage-9 global SDF data exists, to call `sdf_build_world_probes`, upload initial probes/keys, initialize both radiance banks to zero once for the new scene, set `probe_count`, and reset `update_cursor`.

For zero probes, retain valid minimal descriptors but set active count zero and both world flags off. Camera update/resize paths must not call the world-probe builder.

- [ ] **Step 4: Make update-list capacity independent of render resolution**

In `create_wavefront`, allocate `RadianceUpdateList` for `max(screen_probe_capacity, WORLD_PROBE_UPDATES_PER_FRAME)` entries while leaving screen-probe budgets/queues governed by their existing capacities.

- [ ] **Step 5: Add stable scene/lighting revision calculation**

Add `compute_radiance_revision(RENDERER*) -> uint32_t` using deterministic bitwise hashing over current analytic-light GPU data plus emissive-source/object revisions relevant to lighting. Keep the result non-zero and store it in `renderer->radiance_revision` / `Radiance.feature_flags.y`.

If the revision changes after scene setup, invalidate world-probe published state without clearing the 4 MiB radiance buffer: mark CPU probe states dirty/unpublished, upload probe state, reset the update cursor, and let first refreshes write full samples. Do not rebuild positions/keys and do not treat camera motion as a revision.

- [ ] **Step 6: Update descriptors/constants without changing space 6**

Set exactly:

```text
cache_counts.y = probe_count
cache_counts.z = 16384
cache_counts.w = 8192
world_probe_config = {4, 64, 2, 0}
world_probe_params = {0.75, 0.20, 1.125, 0.075}
feature_flags.y = non-zero radiance_revision
```

Enable `WORLD_CACHE | MULTIBOUNCE` only when resources are valid and `probe_count > 0`.

- [ ] **Step 7: Run CPU tests and C syntax compile**

```bash
FLAGS='-I. -I/tmp/SDL/include -I/tmp/NRI/Include -I/tmp/NRI/Include/Extensions'
for f in init.c glb.c gltf.c scene.c sdf.c gpu.c render.c main.c; do clang -std=c11 -fsyntax-only $FLAGS "$f" || exit 1; done
```

Expected: PASS together with Task-2 tests.

- [ ] **Step 8: Commit**

```bash
git add render.c tests/stage10_world_cache.c
git commit -m "s10 allocate world cache"
```

### Task 4: Implement stable two-bank world lookup and multi-bounce shader evaluation

**Files:**
- Modify: `shader.hlsl`

**Interfaces:**
- Consumes: Task-3 constants/resources and unchanged space-6 bindings.
- Produces: latest/stable bank addressing, 3x3x3 lookup, surface-aware diffuse integration, screen-hit world indirect, corrected invalidation bits, and the completed world-probe bounce operator.

- [ ] **Step 1: Add failing shader architecture assertions**

The Stage-10 verification script fails unless shader source expresses these semantics:

```text
WorldProbeRadianceIndex(probe, texel, bank) uses cache_counts.w as fixed bank stride
stable world-update reads previous bank when source.state.x == Pass.dispatch.x
normal renderer reads source.state.z
lookup visits dx/dy/dz in [-1,1] and each cell uses <= 8 open-address probes
surface lookup rejects dot(probe.position - surface.position, surface.normal) <= epsilon
diffuse integration visits all 16 directions with positive cosine weights
screen-trace hit keeps ReflectedDirectAtPixel and adds latest world indirect
world update writes inactive bank then publishes state.z
invalidation changes confidence/dirty flags but never overwrites state.z
```

- [ ] **Step 2: Run architecture assertions to verify RED**

Expected: FAIL because current shader is single-bank, exact-cell, in-place and screen hits contain only direct reflected radiance.

- [ ] **Step 3: Replace single-bank helper semantics**

Implement:

```text
WorldProbeRadianceIndex(probe_index, texel, bank)
FindWorldProbe(position)
FindWorldProbeForSurface(position, normal)
SampleWorldProbeDirectional(probe_index, direction, stable_previous)
IntegrateWorldProbeDiffuse(position, normal, stable_previous)
```

The 3x3x3 search performs the same 8-slot lookup as CPU insertion, validates valid-published state, revision, radius and key, and selects the closest usable probe. Surface lookup additionally requires the probe on the positive normal side.

- [ ] **Step 4: Make world indirect authoritative when world cache is enabled**

Split diffuse evaluation so cached direct may be reused but indirect is freshly obtained from `IntegrateWorldProbeDiffuse`: stable previous view during world updates, latest published view during normal screen/offscreen shading. Apply receiver material diffuse response exactly once using the current renderer convention.

- [ ] **Step 5: Add latest world indirect to screen-space secondary hits**

`CS_WavefrontScreenTrace` currently bypasses `SurfaceReflectedRadiance`. Preserve `ReflectedDirectAtPixel(hit_pixel)` and add only the latest world-cache indirect reflected term derived from the returned `SurfaceHit`; do not replace the existing screen direct term or add raw emissive again.

- [ ] **Step 6: Complete the world-probe bounce operator**

For each selected probe/direction: trace with screen tracing disabled; miss => sky; hit => material emission + analytic diffuse direct + existing emissive-area NEE multiplied by the receiver's existing diffuse material factor + stable previous world indirect. New/dirty/revision-stale probes write the traced sample directly; established probes write `lerp(old, sample, 0.20)` to the inactive bank. Publish bank/valid/current revision/age only after all 16 directions are written.

Use deterministic per-probe/per-direction/frame NEE seed. Never recursively launch another indirect bounce.

- [ ] **Step 7: Correct invalidation semantics**

`CS_InvalidateRadiance` reduces confidence and sets the dirty bit in `state.w` while preserving `state.z`. A revision-stale or dirty probe is not accepted as valid previous indirect until refreshed according to the above rules.

- [ ] **Step 8: Compile affected Slang entries**

```bash
for e in CS_ResetWavefront CS_WavefrontScreenTrace CS_WavefrontDynamicTrace CS_WavefrontGlobalTrace CS_WavefrontLocalTrace CS_ShadeRayHits CS_EmissiveGather CS_UpdateWorldRadianceCache CS_InvalidateRadiance PS_Present; do
  stage=compute; [ "$e" = PS_Present ] && stage=fragment
  slangc shader.hlsl -entry "$e" -stage "$stage" -target spirv -profile sm_6_6 -capability spirv_1_5 -matrix-layout-row-major -fvk-use-dx-layout -O3 -o "/tmp/$e.spv" || exit 1
done
```

Expected: PASS.

- [ ] **Step 9: Re-run shader architecture assertions**

Expected: PASS.

- [ ] **Step 10: Commit**

```bash
git add shader.hlsl
git commit -m "s10 propagate world radiance"
```

### Task 5: Wire fixed scheduling, frame stamp, world dispatch and barriers

**Files:**
- Modify: `render.c`
- Modify: `shader.hlsl`

**Interfaces:**
- Consumes: existing `RADIANCE_WAVEFRONT.update_list`, `RayCounters[3]`, Task-3 CPU update array/cursor and Task-4 world-cache shader.
- Produces: true frame-stamped stable-bank reads, at most 64 selected probes/frame, dispatch before screen-probe world-radiance consumption, and explicit storage visibility afterward.

- [ ] **Step 1: Add failing scheduling/order assertions**

Require:

```text
Pass.dispatch[0] = gpu frame index, not radiance revision
update_count = min(64, probe_count)
indices are (update_cursor + i) % probe_count
Pass.range[0] carries update_count
CS_ResetWavefront sets RayCounters[3] = Pass.range.x while resetting counters 0..2
CPU streams the active cpu_update_list into existing wavefront.update_list
world_radiance_pipeline is created from radiance_world_cache.cs.spv
world update dispatch is after reset and before budget/generate/screen-probe tracing
storage barrier follows world update before screen-probe consumers
update_cursor advances modulo probe_count
zero probes => zero update upload/dispatch
```

- [ ] **Step 2: Run assertions to verify RED**

Expected: FAIL because `Pass.dispatch.x` currently carries revision, no world pipeline is created/dispatched, and reset zeros update count.

- [ ] **Step 3: Implement `prepare_world_probe_updates(RENDERER*)`**

Fill `cpu_update_list[0..update_count)` round-robin and advance `update_cursor`. For zero probes set count/cursor zero. Do not set constant-buffer fields here; this helper owns selection only.

- [ ] **Step 4: Integrate selection into `stream_dynamic_data` without losing it**

Immediately after `stream_dynamic_data` clears `pass_constants`, set:

```text
Pass.dispatch[0] = current gpu frame index
Pass.range[0] = world_radiance.update_count
```

Call `prepare_world_probe_updates` before constructing upload descriptors, then stream the active update array into existing `renderer->wavefront.update_list`. Add COPY_DESTINATION -> STORAGE transition only when update_count > 0. Do not allocate a duplicate GPU update-list buffer.

- [ ] **Step 5: Preserve update count through reset**

Change `CS_ResetWavefront` so counters 0..2 reset normally and `RayCounters[3] = Pass.range.x`. No new shader entry point.

- [ ] **Step 6: Create and dispatch `world_radiance_pipeline`**

Create from existing `build/shaders/radiance_world_cache.cs.spv`. In the wavefront frame command sequence: transition required buffers to storage; dispatch reset; if update_count > 0 dispatch `ceil(update_count / 64)` world-cache groups; issue explicit storage barriers for `WorldProbes`, `WorldProbeRadiance`, and any shared surface-cache state written/read; then continue existing budget/generate/screen/dynamic/global/local/shade/temporal/spatial flow.

- [ ] **Step 7: Handle lighting revision invalidation before scheduling**

After `update_scene_objects`/emissive refresh, recompute `radiance_revision`. If it changed, mark/upload world probe states dirty/unpublished and set `update_cursor = 0` before `prepare_world_probe_updates`. Do not touch positions/keys or clear radiance banks. Camera-only changes must not enter this branch.

- [ ] **Step 8: Run scheduling assertions, C syntax compile and affected shader compiles**

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add render.c shader.hlsl
git commit -m "s10 schedule world cache"
```

### Task 6: Full Stage-10 verification and cleanup

**Files:**
- Verify: `game.h`, `sdf.c`, `render.c`, `shader.hlsl`, `build.c`, `tests/stage10_world_cache.c`
- Remove: any temporary `.github/stage10-*` execution helpers/workflows

**Interfaces:**
- Consumes: completed Tasks 1-5.
- Produces: clean `radiance` branch ready for M2 visual/runtime validation.

- [ ] **Step 1: Run complete CPU behavior suite normally and with ASan/UBSan**

Expected: all placement/hash/determinism/capacity/negative-coordinate tests pass; zero sanitizer findings.

- [ ] **Step 2: Compile every normal shader job in `build.c`**

Use the exact `slangc` flags from `build.c`. Expected: every entry compiles to SPIR-V successfully, with exactly one `CS_UpdateWorldRadianceCache` job.

- [ ] **Step 3: Syntax-compile every renderer C translation unit**

Use the Task-3 command. Expected: zero errors.

- [ ] **Step 4: Run source architecture checks and `git diff --check`**

Cover all non-GPU spec requirements: constants/capacities, static-only deterministic placement, hash reachability, negative coordinates, fixed two-bank addressing, update-list capacity >=64, frame stamp vs revision separation, screen-hit indirect integration, update cap/order, invalidation bank preservation, zero-probe flags and camera-independent lifecycle.

Expected: PASS and `git diff --check` prints nothing.

- [ ] **Step 5: Inspect final diff against the approved spec**

Confirm no duplicate world-cache implementation, no new descriptor space, no camera path rebuilding placement, no recursive bounce call, and no unrelated renderer refactor.

- [ ] **Step 6: Remove execution-only workflow/helper files**

If GitHub Actions helpers were required by this harness, delete them after the verified production commit and verify the cleanup commit changes only those files.

- [ ] **Step 7: Keep the focused CPU regression test**

`tests/stage10_world_cache.c` remains because it protects deterministic placement/hash behavior; it is not execution junk.

- [ ] **Step 8: Report runtime validation without claiming visual success**

Use the user's normal M2 run command, e.g.:

```bash
c build run -- cornell_box.glb
```

Runtime acceptance remains: no NRI/Vulkan validation failure, FPS recorded, and visible world GI should converge over frames from the emissive panel. Compile/tests alone do not prove final lighting quality.
