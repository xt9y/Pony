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
- Established texels blend 20% new / 80% old; first updates never blend undefined history.
- Same-frame update order may not expose newly written bank data as previous-iteration input.
- Zero active probes must keep valid descriptors and disable both `WORLD_CACHE` and `MULTIBOUNCE`.
- Camera motion never rebuilds, clears, or re-keys world probes.
- No recursive arbitrary-depth path tracing and no renderer-wide BRDF rewrite.
- Keep repository output clean: no temporary Stage-10 workflow/helper files remain after verification.

## Review Focus

- Negative world coordinates: CPU and shader placement-cell hashing must use identical two's-complement `uint32_t` semantics and still find probes across the 3x3x3 neighborhood.
- Thin/adjacent surfaces: placement clearance and surface-side filtering must prevent a probe across a wall from becoming the preferred diffuse source.
- Hash collisions at high occupancy: insertion and lookup are both limited to 8 slots; omitted candidates must not leave unreachable active probes.
- Empty/tiny scenes: zero probes must remain a valid renderer state; tiny valid static geometry must not produce NaN positions/radii.
- Invalidation while banks alternate: dirty/confidence updates must not overwrite the active-bank bit or make same-frame reads order-dependent.

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
  - `NriPipeline *world_radiance_pipeline` in `RENDERER`;
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

Run:

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

Add test helpers that build a minimal synthetic `GLOBAL_SDF_DATA` with a fine/medium clipmap, canonical static triangles and object states. Assertions:

```text
static geometry => probe_count > 0 and <= 8192
dynamic-only anchors => probe_count == 0
same unchanged input built twice => byte-identical probe sequence and key table
every occupied key slot => decoded index < probe_count and state.identity.x matches the slot's searched key
every probe => finite xyz, radius == 1.125, anchor owner is STATIC
placement distance from static geometry >= 0.0375
negative-coordinate fixture => deterministic non-zero keys and reachable probes
collision fixture => no active probe exists unless reachable within 8 open-addressed slots
```

- [ ] **Step 2: Run the behavior test to verify RED**

Run:

```bash
clang -std=c11 -I. -I/tmp/SDL/include -I/tmp/NRI/Include -I/tmp/NRI/Include/Extensions tests/stage10_world_cache.c sdf.c -lm -o /tmp/stage10_world_cache_test && /tmp/stage10_world_cache_test
```

Expected: FAIL because `sdf_build_world_probes` has no implementation.

- [ ] **Step 3: Refactor only the reusable Stage-9 static-BVH helpers needed by placement**

Keep `SDF_BUILD` temporary to the build call. Add exact CPU equivalents of shader `Hash32`/`HashCombine` using `uint32_t` arithmetic and derive placement cells with `floorf(position / spacing)` before bit-preserving conversion to `uint32_t`.

- [ ] **Step 4: Implement `sdf_build_world_probes(...)`**

Use only fine/medium clipmap samples with valid canonical surface IDs. Reconstruct candidate voxel positions from `GPU_GLOBAL_SDF_CLIPMAP`, reconstruct canonical world triangles through the owning `GPU_OBJECT.world`, compute closest point + geometric normal, test `+normal * 0.075` then `-normal * 0.075`, require nearest static clearance >= `0.0375`, deduplicate by 0.75-unit integer cell, choose greatest clearance then smaller surface ID, sort deterministically by cell, cap at 8192, and insert each surviving probe into the 16384 table with an 8-slot limit. Omit candidates that cannot be inserted.

Initialize `WORLD_PROBE_STATE` exactly as the spec defines: radius 1.125, anchor IDs/revision, confidence/age zero, active bank zero, no valid-published bit.

- [ ] **Step 5: Run the placement test**

Expected: PASS all placement/hash/determinism cases.

- [ ] **Step 6: Run sanitizer build of the CPU test**

Run:

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
- Produces: real `WorldProbes`, two-bank `WorldProbeRadiance`, `WorldProbeKeys`, `InvalidationQueue`, constants/flags, and valid dummy behavior for zero probes.

- [ ] **Step 1: Add a renderer-contract regression check**

Extend `tests/stage10_world_cache.c` with pure size/count assertions for:

```text
radiance values = 8192 * 16 * 2
radiance bytes = values * sizeof(float[4])
hash slots = 16384
max updates = min(64, active_count)
```

Add a small source-contract check in the Stage-10 CI command that fails until `create_world_radiance_resources` uses the Stage-10 constants instead of one-element allocations and until `update_radiance_constants` sets `cache_counts.y/z/w`, `world_probe_config`, `world_probe_params`, and gates both feature flags on `probe_count > 0`.

- [ ] **Step 2: Run the regression check to verify RED**

Expected: FAIL on the one-element placeholder resources/current zero world-cache constants.

- [ ] **Step 3: Implement world-resource creation/destruction and scene rebuild**

`create_world_radiance_resources(RENDERER*)` allocates CPU arrays and GPU capacities from the exact constants. `destroy_world_radiance_resources` frees CPU/GPU ownership. Add `build_world_radiance_scene(RENDERER*)` that calls `sdf_build_world_probes` after Stage-9 global SDF creation, uploads initialized probes/keys, clears both radiance banks once at scene setup, sets `probe_count`, resets `update_cursor`, and leaves camera movement uninvolved.

For zero probes, retain valid minimal allocations/descriptors but set active count zero and both world-cache flags off.

- [ ] **Step 4: Update descriptors/constants without changing space 6**

Reuse the existing six storage-buffer descriptors. Set:

```text
cache_counts.y = probe_count
cache_counts.z = 16384
cache_counts.w = 8192
world_probe_config = {4, 64, 2, 0}
world_probe_params = {0.75, 0.20, 1.125, 0.075}
```

Enable `WORLD_CACHE | MULTIBOUNCE` only when real resources are valid and `probe_count > 0`.

- [ ] **Step 5: Run C unit/contract tests and syntax compile**

Run the Task-2 test plus:

```bash
FLAGS='-I. -I/tmp/SDL/include -I/tmp/NRI/Include -I/tmp/NRI/Include/Extensions'
for f in init.c glb.c gltf.c scene.c sdf.c gpu.c render.c main.c; do clang -std=c11 -fsyntax-only $FLAGS "$f" || exit 1; done
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add render.c tests/stage10_world_cache.c
git commit -m "s10 allocate world cache"
```

### Task 4: Implement stable two-bank world lookup and multi-bounce shader evaluation

**Files:**
- Modify: `shader.hlsl`
- Modify: `tests/stage10_world_cache.c` only if CPU reference values need extension

**Interfaces:**
- Consumes: Task-3 constants/resources and unchanged space-6 bindings.
- Produces: latest/stable bank addressing, 3x3x3 lookup, surface-aware diffuse integration, corrected invalidation bits, and the completed `CS_UpdateWorldRadianceCache` bounce operator.

- [ ] **Step 1: Add the failing shader architecture assertions**

The Stage-10 verification script must fail unless shader source contains all of these semantics:

```text
WorldProbeRadianceIndex(probe, texel, bank) uses fixed physical capacity cache_counts.w
stable update reads previous bank when source.state.x == current frame
normal renderer reads source.state.z
lookup visits dx/dy/dz in [-1,1]
surface lookup rejects dot(probe-position - hit-position, hit-normal) <= epsilon
diffuse integration loops all 16 world directions with positive cosine weights
invalidation sets dirty bit without assigning active bank
world update writes inactive bank then publishes state.z
```

- [ ] **Step 2: Run architecture assertions to verify RED**

Expected: FAIL because the current shader has single-bank exact-cell lookup/in-place update.

- [ ] **Step 3: Replace single-bank helper semantics**

Keep existing bindings. Implement exact helpers for:

```text
WorldProbeRadianceIndex(probe_index, texel, bank)
FindWorldProbe(position)
FindWorldProbeForSurface(position, normal)
SampleWorldProbeDirectional(probe_index, direction, stable_previous)
IntegrateWorldProbeDiffuse(position, normal, stable_previous)
```

The 3x3x3 search uses an 8-slot open-addressing lookup for each neighboring cell and selects the closest probe inside 1.125 units. Surface lookup also requires the probe to lie on the positive normal side.

- [ ] **Step 4: Make surface indirect world-cache authoritative when enabled**

Refactor the diffuse surface evaluator so cached direct can still be reused, but current world indirect is evaluated from `IntegrateWorldProbeDiffuse`. During `CS_UpdateWorldRadianceCache`, request the stable previous view; normal screen/offscreen shading requests the latest published view. Apply the existing diffuse material convention exactly once.

- [ ] **Step 5: Complete the world-probe bounce operator**

For each selected probe/direction: trace with screen tracing disabled; on miss use sky; on hit combine material emission + analytic diffuse direct + existing emissive-area NEE + stable previous world indirect. New probes write the traced sample directly; established probes write `lerp(old, sample, 0.20)` to the inactive bank. Publish bank/valid/revision/age only after all 16 directions are written.

Use deterministic per-probe/per-direction/frame seed for emissive NEE. Do not recursively launch another indirect bounce.

- [ ] **Step 6: Correct invalidation state semantics**

`CS_InvalidateRadiance` must reduce confidence and set `state.w` dirty bit while preserving `state.z` active bank.

- [ ] **Step 7: Compile all affected Slang entries**

Run at minimum:

```bash
for e in CS_ResetWavefront CS_WavefrontScreenTrace CS_WavefrontDynamicTrace CS_WavefrontGlobalTrace CS_WavefrontLocalTrace CS_ShadeRayHits CS_EmissiveGather CS_UpdateWorldRadianceCache CS_InvalidateRadiance PS_Present; do
  stage=compute; [ "$e" = PS_Present ] && stage=fragment
  slangc shader.hlsl -entry "$e" -stage "$stage" -target spirv -profile sm_6_6 -capability spirv_1_5 -matrix-layout-row-major -fvk-use-dx-layout -O3 -o "/tmp/$e.spv" || exit 1
done
```

Expected: PASS.

- [ ] **Step 8: Re-run architecture assertions**

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add shader.hlsl
git commit -m "s10 propagate world radiance"
```

### Task 5: Wire fixed round-robin scheduling, world update dispatch and barriers

**Files:**
- Modify: `render.c`
- Modify: `shader.hlsl`

**Interfaces:**
- Consumes: existing `RADIANCE_WAVEFRONT.update_list`, `RayCounters[3]`, Task-3 CPU `cpu_update_list/update_cursor`, and Task-4 world-cache shader.
- Produces: at most 64 selected world probes per frame, dispatch before screen-probe world-radiance consumption, and explicit storage visibility afterward.

- [ ] **Step 1: Add failing scheduling/order assertions**

Verification must assert:

```text
update_count = min(64, probe_count)
indices are (update_cursor + i) % probe_count
pass_constants.range[0] carries update_count
CS_ResetWavefront initializes RayCounters[3] from Pass.range.x instead of zero
CPU streams cpu_update_list into wavefront.update_list when update_count > 0
world_radiance_pipeline is created from radiance_world_cache.cs.spv
world update dispatch occurs after wavefront reset and before budget/generate/screen-probe shading
storage barrier follows world update before screen-probe consumers
update_cursor advances modulo probe_count
zero probes dispatch zero world updates
```

- [ ] **Step 2: Run assertions to verify RED**

Expected: FAIL because no world-cache pipeline is created/dispatched/scheduled.

- [ ] **Step 3: Implement `prepare_world_probe_updates(RENDERER*)`**

Exact behavior: fill `cpu_update_list[0..update_count)` round-robin, set `pass_constants.range[0] = update_count`, retain/update cursor modulo active count, and set count/cursor to zero for zero probes.

- [ ] **Step 4: Extend `stream_dynamic_data`**

Stream the active portion of `cpu_update_list` to existing `renderer->wavefront.update_list`; include COPY_DESTINATION -> STORAGE transition for that buffer when an upload occurs. Do not allocate a duplicate GPU update-list buffer.

- [ ] **Step 5: Preserve update count through reset**

Change `CS_ResetWavefront` so counters 0..2 are reset normally and `RayCounters[3] = Pass.range.x`. No new shader entry point is introduced.

- [ ] **Step 6: Create and dispatch `world_radiance_pipeline`**

Create it from existing `build/shaders/radiance_world_cache.cs.spv`. In the wavefront frame setup, dispatch reset, then if update_count > 0 dispatch `ceil(update_count / 64)` groups of `world_radiance_pipeline`, add explicit storage barriers for world probes/radiance/keys and relevant cache state, then continue the existing screen-probe budget/generate/trace chain.

- [ ] **Step 7: Run scheduling assertions, C syntax compile and all affected shader compiles**

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add render.c shader.hlsl
git commit -m "s10 schedule world cache"
```

### Task 6: Full Stage-10 verification, runtime handoff and cleanup

**Files:**
- Verify: `game.h`, `sdf.c`, `render.c`, `shader.hlsl`, `build.c`, `tests/stage10_world_cache.c`
- Remove: any temporary `.github/stage10-*` helpers/workflows created only to execute CI in this environment

**Interfaces:**
- Consumes: completed Tasks 1-5.
- Produces: clean `radiance` branch ready for M2 visual/runtime validation.

- [ ] **Step 1: Run the complete CPU behavior suite with sanitizers**

Run both normal and ASan/UBSan commands from Task 2. Expected: PASS, zero sanitizer findings.

- [ ] **Step 2: Compile every normal shader job from `build.c`**

Use the same `slangc` flags as `build.c` for every listed shader entry. Expected: all SPIR-V compiles succeed.

- [ ] **Step 3: Syntax-compile every renderer C translation unit**

Use the Task-3 command. Expected: zero errors.

- [ ] **Step 4: Run source architecture checks and `git diff --check`**

Checks must cover every spec verification item that does not require a GPU: constants/capacities, static-only placement, deterministic build, hash reachability, bank semantics, update cap/order, invalidation bank preservation, zero-probe flags, no new world-cache entry point, and camera-independent lifecycle.

Expected: PASS and `git diff --check` prints nothing.

- [ ] **Step 5: Inspect final diff against the approved spec**

Confirm `build.c` still has exactly one `CS_UpdateWorldRadianceCache` job and no Stage-10 debug/alternate implementations remain. Confirm no camera-update path calls the world-probe builder.

- [ ] **Step 6: Remove execution-only workflow/helper files**

If GitHub Actions helper files were required by this harness, delete them after the verified production commit and verify that the cleanup commit changes only those helpers.

- [ ] **Step 7: Keep the focused CPU regression test**

`tests/stage10_world_cache.c` remains because it validates the deterministic placement/hash behavior rather than being execution junk.

- [ ] **Step 8: Report the runtime validation command without claiming visual success**

Use the user's normal M2 command, for example:

```bash
c build run -- cornell_box.glb
```

Runtime acceptance remains: no NRI/Vulkan validation failure, FPS recorded, and visible world GI should converge over frames from the emissive panel. Compile/tests do not by themselves prove final lighting quality.
