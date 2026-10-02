# Dustmite Radiance-Structure Port Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor every Dustmite source/shader file into the Radiance branch's code orientation and ownership model while preserving Dustmite rendering, baking, cache compatibility, controls, defaults, and visual output.

**Architecture:** Keep Dustmite algorithms and data as the behavioral source of truth. Introduce the Radiance-style low-level `GPU` object, composition-oriented `SCENE`/`OBJECT` layer, renderer lifecycle, thin `main.c`, and consolidated `shader.hlsl`; retain Dustmite-specific bake/BVH/lightmap/beam/cache modules as focused siblings rather than importing Radiance GI/SDF functionality.

**Tech Stack:** C11, SDL3, SDL3_image, NVIDIA NRI, HLSL, SDL_shadercross, DXC where available, c-build.

**Spec:** `docs/superpowers/specs/2026-10-02-dustmite-radiance-structure-design.md`

## Global Constraints

- Base implementation branch: `dustmite-radiance-structure`, originally created from Dustmite `0ae0e70364210ec302c8b076ea5513d0dd716612`.
- `Dustmite` is the behavioral source of truth; `radiance` is only a structure/style reference.
- Do not import Radiance GI, SDF, world-cache, screen-probe, reflection, or other rendering behavior.
- Preserve `.baked` binary compatibility and scene/layout/volume/beam hash inputs and ordering exactly.
- Preserve all Dustmite default values, controls, pass ordering, shader logic, workgroup sizes, register semantics, bake constants, and cache-reuse decisions.
- Preserve NRI backend/platform behavior and queue/swapchain/upload semantics.
- No intentional quality, performance, or image changes.
- Commit names are concise and lowercase.

## Review Focus

- Existing `.baked` cache files must still load after the ownership/API rewrite; Task 1 records the cache ABI/hash baseline and Task 9 verifies it.
- Window resize/minimize and swapchain recreation must still follow Dustmite semantics after `GPU` extraction; Task 3 adds a frame-lifecycle contract and Task 9 verifies the event/render path.
- Bake-active idle rendering must retain the current grace/render/sleep cadence and must not accidentally become synchronous; Task 5 pins the bake lifecycle and Task 8 verifies `main.c` loop ordering.
- Wave and non-wave shader fallback paths must produce the same build job set even after shader consolidation; Task 6 records and compares every shader entry/define pair.
- macOS/Windows/Linux surface/backend branches must stay behaviorally equivalent after moving NRI setup; Task 3 keeps platform branches together and Task 9 compiles all available paths/CI.

---

### Task 1: Lock the Dustmite behavioral and ABI baseline

**Files:**
- Create: `tests/structure_contract.c`
- Create: `tests/baseline_contract.c`
- Modify: `build.c`
- Reference only: `main.c`, `game.h`, `gpu.h`, `bake.c`, `cache.c`, `shaders/*.hlsl`

**Interfaces:**
- Consumes: existing Dustmite public types and build jobs.
- Produces: compile-time/API contracts used by every later task; a build target that compiles the contract tests without changing the application path.

- [ ] **Step 1: Write the failing architecture contract**

Create `tests/structure_contract.c` that expects the target public API to exist:

```c
#include "game.h"
#include "gpu.h"

static bool (*gpu_init_contract)(GPU *, const char *, int, int) = gpu_init;
static void (*gpu_deinit_contract)(GPU *) = gpu_deinit;
static bool (*renderer_init_contract)(RENDERER *, GPU *) = renderer_init;
static bool (*renderer_set_scene_contract)(RENDERER *, SCENE *) = renderer_set_scene;
static void (*renderer_event_contract)(RENDERER *, const SDL_Event *) = renderer_event;
static bool (*renderer_frame_contract)(RENDERER *) = renderer_frame;
static void (*renderer_deinit_contract)(RENDERER *) = renderer_deinit;
```

Add compile-time construction of `TRANSFORM`, `OBJECT`, and `SCENE`, including `scene_add_model`, `scene_add_light`, `object_set_transform`, `object_mark_dirty`, and `scene_free`.

- [ ] **Step 2: Run the contract and verify it fails before the refactor**

Run the dedicated contract build command added to `build.c` (or direct C11 compile if the build DSL cannot expose a test target).
Expected: FAIL because the Radiance-style API/types do not yet exist.

- [ ] **Step 3: Write the passing baseline behavior contract**

Create `tests/baseline_contract.c` with `_Static_assert` checks for existing serialized/public layout fields that participate in cache/GPU ABI and helper functions that reproduce the exact Dustmite bake/hash constant arrays from `main.c`. The test must compare the exact values and ordering currently present, including `LIGHTMAP_TEXELS_PER_UNIT=24`, `LIGHTMAP_MAX_SIZE=4096`, current `bake_settings`, `volume_bake_settings`, and `beam_settings`.

- [ ] **Step 4: Record the shader job baseline in `build.c` without changing jobs**

Keep the current 27 jobs and their exact `path`, `entry`, `define`, `fallback_define`, `stage`, and `wave` values in one static table that later shader consolidation can reuse.

- [ ] **Step 5: Verify baseline build**

Run: `c build`
Expected: application C compilation and all existing shader jobs succeed in the available environment.

- [ ] **Step 6: Commit**

```bash
git add build.c tests/structure_contract.c tests/baseline_contract.c
git commit -m "lock dustmite refactor baseline"
```

### Task 2: Introduce Radiance-style scene composition without changing rendering

**Files:**
- Create: `scene.c`
- Modify: `game.h`
- Modify: `main.c`
- Modify: `build.c`
- Test: `tests/structure_contract.c`

**Interfaces:**
- Consumes: current `MODEL`, `LIGHT`, `SKY`, and object state values.
- Produces: `TRANSFORM transform_identity(void)`, `OBJECT *scene_add_model(SCENE *, struct MODEL *, OBJECT_STATE, TRANSFORM)`, `OBJECT *scene_add_light(SCENE *, struct LIGHT *, OBJECT_STATE, TRANSFORM)`, `void object_set_transform(OBJECT *, TRANSFORM)`, `void object_mark_dirty(OBJECT *)`, `void scene_free(SCENE *)`.

- [ ] **Step 1: Extend the failing scene contract**

Require `OBJECT` to contain `state`, `type`, `transform`, `data`, `revision`; require `SCENE` to contain object storage plus `SKY`; verify `transform_identity()` yields zero position, identity quaternion, unit scale and object revision starts nonzero.

- [ ] **Step 2: Run contract**

Expected: FAIL against current `game.h`/`main.c`.

- [ ] **Step 3: Implement scene types and `scene.c`**

Use the same API/ownership orientation as Radiance, but do not add dynamic-rendering behavior. Preserve Dustmite's existing `STATIC` model/light usage and current light/sky values.

- [ ] **Step 4: Convert `main.c` construction only**

Build a `SCENE`, add the current model and directional light using the new API, and continue feeding exactly the same underlying mesh/visual/light data to the still-old renderer path.

- [ ] **Step 5: Build and run contracts**

Run: `c build` plus contract target.
Expected: PASS, with no shader/build job change.

- [ ] **Step 6: Commit**

```bash
git add game.h scene.c main.c build.c tests/structure_contract.c
git commit -m "add scene composition layer"
```

### Task 3: Extract generic NRI ownership into `GPU`

**Files:**
- Modify: `gpu.h`
- Modify: `gpu.c`
- Modify: `render.c`
- Modify: `game.h`
- Test: `tests/structure_contract.c`

**Interfaces:**
- Consumes: Dustmite's current NRI device creation, platform surface, queues, swapchain, frame contexts, upload/resource helpers.
- Produces: `bool gpu_init(GPU *, const char *, int, int)`, `void gpu_deinit(GPU *)`, `bool gpu_begin_frame(GPU *, NriCommandBuffer **, NriTexture **, uint32_t *)`, `bool gpu_end_frame(GPU *, NriCommandBuffer *, uint32_t)`, `bool gpu_resize(GPU *)`, generic buffer/texture create/upload/destroy helpers, and swapchain attachment accessors.

- [ ] **Step 1: Expand the failing GPU ownership contract**

Compile a `GPU` separately from `RENDERER`; require `RENDERER` to hold `GPU *gpu`; ensure Dustmite-specific fields such as lightmap textures, bake state, materials, FX/post-process state, probe/beam resources are not members of `GPU`.

- [ ] **Step 2: Run contract**

Expected: FAIL because current `RENDERER` embeds NRI/device/swapchain ownership.

- [ ] **Step 3: Move only generic NRI infrastructure**

Port the proven Dustmite implementations into Radiance-style `GPU` ownership: window/platform view, device/interfaces, queues, swapchain, frame fence/context, generic upload/resource creation/destruction. Preserve queue fallback, swapchain format/VSYNC/queued-frame settings, and platform branches exactly.

- [ ] **Step 4: Adapt renderer call sites mechanically**

Replace direct low-level fields with `renderer->gpu->...` or generic `gpu_*` helpers. Do not move renderer-specific resources yet.

- [ ] **Step 5: Verify C/shader build and frame-lifecycle API**

Run: `c build` plus contract target.
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add gpu.h gpu.c game.h render.c tests/structure_contract.c
git commit -m "separate gpu infrastructure"
```

### Task 4: Normalize `RENDERER` ownership and lifecycle

**Files:**
- Modify: `game.h`
- Modify: `gpu.h`
- Modify: `render.c`
- Modify: `main.c`
- Test: `tests/structure_contract.c`

**Interfaces:**
- Consumes: `GPU`, `SCENE`, existing Dustmite runtime resources and draw logic.
- Produces: `bool renderer_init(RENDERER *, GPU *)`, `bool renderer_set_scene(RENDERER *, SCENE *)`, `void renderer_event(RENDERER *, const SDL_Event *)`, `bool renderer_frame(RENDERER *)`, `void renderer_deinit(RENDERER *)`.

- [ ] **Step 1: Make the lifecycle contract fail on any remaining `r_*` public lifecycle dependency**

The test must compile only against the `renderer_*` public API and construct separate `GPU`, `SCENE`, and `RENDERER` values.

- [ ] **Step 2: Run contract**

Expected: FAIL until all public lifecycle declarations/call sites are migrated.

- [ ] **Step 3: Move renderer-specific runtime ownership into `RENDERER`**

Keep pipelines/layouts, depth/HDR/normal/AO/bloom/LUT/volume/lit resources, materials, draw ranges, vertex data, camera/orbit state, runtime sun/sky/volumetric/peripheral state, debug flags, and renderer-specific descriptors in `RENDERER`, not `GPU`.

- [ ] **Step 4: Replace public `r_init`, `r_build_scene`, `r_event`, `r_draw`, `r_deinit` flow**

Use the exact target lifecycle signatures. `renderer_set_scene` must build/upload the same Dustmite geometry/material/lightmap-facing scene resources, and `renderer_frame` must execute the same runtime pass order as current `r_draw`.

- [ ] **Step 5: Adapt `main.c` to separate `GPU gpu` and `RENDERER renderer`**

Do not alter bake loop cadence or cache/hash logic.

- [ ] **Step 6: Build and verify contracts**

Run: `c build` plus contract target.
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add game.h gpu.h render.c main.c tests/structure_contract.c
git commit -m "normalize renderer lifecycle"
```

### Task 5: Isolate all bake-specific GPU work in `bake.c`

**Files:**
- Modify: `bake.c`
- Modify: `gpu.c`
- Modify: `gpu.h`
- Modify: `render.c`
- Modify: `game.h`
- Test: `tests/baseline_contract.c`

**Interfaces:**
- Consumes: generic `GPU` helpers, Dustmite `BVH`, `LIGHTMAP`, `PROBE_GRID`, `BEAM_GRID`, `CACHED_LIGHTMAP`, renderer-owned bake resources.
- Produces: existing Dustmite bake lifecycle (`bake_start`, `bake_update`, `bake_cancel`, `bake_active`, `bake_update_title`) plus bake-private upload/download/dispatch helpers; public signatures remain behaviorally equivalent.

- [ ] **Step 1: Add a failing ownership contract**

Require `gpu.h` to expose no bake-specific functions (`bake_lightmap`, probe bake functions, `upload_lightmap`, `upload_probes`, `upload_beams`, `download_lightmap`, bake-worker APIs). These declarations must move to renderer/bake-private scope or `game.h` only when genuinely public.

- [ ] **Step 2: Run contract**

Expected: FAIL with current `gpu.h`.

- [ ] **Step 3: Move bake-specific implementations out of `gpu.c`**

Move code mechanically into `bake.c`, keeping constants, dispatch dimensions, barriers, readbacks, waits, probe wavefront flow, lightmap queue flow, cached-secondary-hit behavior, and timing semantics unchanged.

- [ ] **Step 4: Keep generic GPU helpers in `gpu.c`**

If a moved bake helper needs a generic operation, expose the smallest generic `gpu_*` helper instead of reintroducing bake concepts into `GPU`.

- [ ] **Step 5: Verify baseline contract and build**

Run: `c build` plus baseline/structure contracts.
Expected: PASS; exact hash/bake arrays unchanged.

- [ ] **Step 6: Commit**

```bash
git add bake.c gpu.c gpu.h render.c game.h tests/baseline_contract.c
git commit -m "isolate bake gpu work"
```

### Task 6: Consolidate every Dustmite shader into `shader.hlsl`

**Files:**
- Create: `shader.hlsl`
- Modify: `build.c`
- Delete after verification: `shaders/vertex.hlsl`, `shaders/fragment.hlsl`, `shaders/compute.hlsl`, `shaders/compute_base.hlsl`, `shaders/vision_compute.hlsl`, `shaders/lightmap_queue.hlsl`, `shaders/probe_wavefront.hlsl`
- Test: build shader job table from Task 1

**Interfaces:**
- Consumes: exact Task 1 shader job entry/define/stage/wave baseline.
- Produces: the same entry points compiled from `shader.hlsl` with the same output SPIR-V names and fallback behavior.

- [ ] **Step 1: Make shader-path verification fail**

Change only the test/validation layer so it expects every job source path to be `shader.hlsl` while preserving the exact entry/define/fallback/stage/wave tuples.

- [ ] **Step 2: Run shader validation**

Expected: FAIL because jobs still point into `shaders/`.

- [ ] **Step 3: Build `shader.hlsl` mechanically**

Merge shared definitions first, then vertex, fragment, runtime compute, vision compute, lightmap queue, and probe wavefront sections. Resolve duplicate helper/type definitions only by deduplicating identical semantics; do not rewrite algorithms.

- [ ] **Step 4: Point all 27 shader jobs at `shader.hlsl`**

Keep output names, entry names, defines, fallback defines, stage and wave fields unchanged. Set include handling so consolidated code does not depend on deleted `shaders/` paths.

- [ ] **Step 5: Compile every shader job before deletion**

Run: `c build`
Expected: all 27 jobs compile successfully.

- [ ] **Step 6: Remove old shader files and verify no references remain**

Search repository source/build files for `shaders/vertex.hlsl`, `fragment.hlsl`, `compute.hlsl`, `compute_base.hlsl`, `vision_compute.hlsl`, `lightmap_queue.hlsl`, `probe_wavefront.hlsl`.
Expected: zero live references.

- [ ] **Step 7: Rebuild after deletion**

Run: `c build`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add build.c shader.hlsl shaders tests
git commit -m "consolidate dustmite shaders"
```

### Task 7: Apply Radiance code orientation to every remaining engine source file

**Files:**
- Modify: `init.c`
- Modify: `glb.c`
- Modify: `gltf.c`
- Modify: `bvh.c`
- Modify: `lmap.c`
- Modify: `beam.c`
- Modify: `cache.c`
- Modify: `bake.c`
- Modify: `gpu.c`
- Modify: `render.c`
- Modify: `scene.c`
- Modify: `game.h`
- Modify: `gpu.h`
- Modify: `build.c`
- Modify: `main.c`
- Modify: `felix-format` only if its rules contradict the desired Radiance orientation; otherwise leave behavior unchanged.
- Test: both contract tests and full build

**Interfaces:**
- Consumes: completed ownership boundaries from Tasks 2-6.
- Produces: consistent Radiance-style naming, helper ordering, initializer orientation, includes, whitespace, early-return structure, and comments across every textual engine source file.

- [ ] **Step 1: Establish a style-only diff rule**

For each listed file, compare against the pre-Task-7 version and classify every change as naming, ordering, formatting, include cleanup, comment cleanup, or behavior-neutral local extraction. Any algorithmic change is rejected from this task.

- [ ] **Step 2: Clean `init.c`, `glb.c`, `gltf.c`**

Use Radiance orientation: concise includes, static helpers before public consumers, early returns, compact designated initializers, consistent blank-line spacing. Preserve parsing/extraction behavior byte-for-byte in logic.

- [ ] **Step 3: Clean `bvh.c`, `lmap.c`, `beam.c`, `cache.c`**

Preserve algorithms, serialized layouts, sort/build traversal semantics, chart generation, beam compression, and cache I/O. Rename only local/internal helpers when it improves consistency and updates every caller in the same commit.

- [ ] **Step 4: Clean `gpu.c`, `render.c`, `bake.c`, `scene.c`**

Group static helpers by owning public operation, remove obsolete wrapper names, standardize `GPU *gpu` / `RENDERER *renderer` identifiers, and keep backend/platform preprocessor blocks localized.

- [ ] **Step 5: Clean `game.h`, `gpu.h`, `main.c`, `build.c`**

Order declarations by subsystem like Radiance, keep `main.c` composition-first, and keep build jobs/data tables declarative.

- [ ] **Step 6: Run formatting/build/contracts after each file group**

Run the project's formatter where applicable, then `c build` and both contract tests.
Expected: PASS after each group, not only at the end.

- [ ] **Step 7: Commit**

```bash
git add init.c glb.c gltf.c bvh.c lmap.c beam.c cache.c bake.c gpu.c render.c scene.c game.h gpu.h build.c main.c felix-format tests
git commit -m "align engine code orientation"
```

### Task 8: Finish thin application composition and preserve the exact Dustmite loop

**Files:**
- Modify: `main.c`
- Modify: `render.c`
- Modify: `bake.c`
- Test: `tests/baseline_contract.c`

**Interfaces:**
- Consumes: `GPU`, `RENDERER`, `SCENE`, current cache/hash and bake lifecycle.
- Produces: a Radiance-like `main.c` whose responsibilities stop at asset loading, scene/config construction, hash/cache setup, event loop, lifecycle calls, and cleanup.

- [ ] **Step 1: Add loop-order assertions to baseline verification**

Pin the observable order: input/event handling -> conditional render according to existing bake idle cadence -> `bake_update` -> `bake_update_title`; preserve `B`, `F5`, `Tab`, `F11`, wheel/LMB orbit, and `Esc` paths.

- [ ] **Step 2: Move any remaining renderer-internal setup out of `main.c`**

Do not move scene defaults, bake/hash inputs, cache path construction, or application-level bake trigger policy.

- [ ] **Step 3: Ensure cleanup follows ownership**

Order: cancel bake, renderer deinit, GPU deinit, scene/free app data, SDL quit, free cache-path allocation; avoid double-free of model/light data owned externally by `SCENE`.

- [ ] **Step 4: Build/contracts**

Run: `c build` and both contract tests.
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add main.c render.c bake.c tests/baseline_contract.c
git commit -m "finish radiance style application flow"
```

### Task 9: Full regression verification against original Dustmite

**Files:**
- Modify only if a regression is found: the smallest responsible source file.
- Test: full project/build and repository diff against `0ae0e70364210ec302c8b076ea5513d0dd716612`.

**Interfaces:**
- Consumes: all completed tasks.
- Produces: verified restructuring branch with no intentional Dustmite behavior change.

- [ ] **Step 1: Verify repository structure**

Expected textual engine layout: `build.c`, `game.h`, `gpu.h`, `gpu.c`, `init.c`, `glb.c`, `gltf.c`, `scene.c`, `render.c`, `bake.c`, `bvh.c`, `beam.c`, `lmap.c`, `cache.c`, `shader.hlsl`, `main.c`, tests/docs; old shader sources absent.

- [ ] **Step 2: Run all compile contracts and full build**

Run: contract targets and `c build`.
Expected: PASS with strict C warnings and every shader job compiling.

- [ ] **Step 3: Compare behavioral constants/ABI**

Compare original Dustmite and branch for cache structs/serialization, hash inputs/order, default light/sky/volumetric/peripheral values, bake settings, sample counts, controls, and runtime/bake pass ordering.
Expected: no semantic change.

- [ ] **Step 4: Run executable smoke test where GPU execution is available**

Run default `concrete_temple.glb`; verify startup succeeds, cached bake loading still works when a compatible cache is present, controls remain functional, frame rendering succeeds, and a rebake can start/update/cancel without error.

- [ ] **Step 5: Compare bake/runtime logs when available**

Use the same scene and configuration as baseline. Atlas dimensions/valid texels/probe dimensions and major bake stages must match; investigate any unexplained difference before completion.

- [ ] **Step 6: Review final diff for accidental Radiance behavior**

Reject any added Radiance SDF/GI/world-cache/screen-probe/reflection feature, altered rendering constant, altered dispatch/workgroup size, or changed cache format.

- [ ] **Step 7: Final verification commit only if fixes were needed**

```bash
git add <smallest-fixed-files>
git commit -m "fix restructuring regressions"
```

Do not merge into `Dustmite` as part of this plan.

