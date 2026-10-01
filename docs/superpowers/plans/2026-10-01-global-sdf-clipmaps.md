# Pony Radiance Stage 9 Global SDF Clipmaps Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a bounded three-level static world SDF so offscreen static-scene rays use one scene-wide acceleration structure instead of exhaustively testing every static local SDF.

**Architecture:** Build one CPU BVH over transformed `STATIC` `GPUSceneTriangle`s, generate three conservative 4x4x4-brick SDF levels, upload them through the existing `space4` ABI, and activate `CS_WavefrontGlobalTrace` between Stage 8 dynamic-grid tracing and static-local fallback. The SDF only proposes candidate surfaces; every accepted global hit is refined against the canonical `SceneTriangles` geometry before shading.

**Tech Stack:** C11, NRI/Vulkan, Slang/HLSL, SDL3, GitHub Actions verification.

**Spec:** `docs/superpowers/specs/2026-10-01-global-sdf-clipmaps-design.md`

## Global Constraints

- Preserve the existing NRI/Vulkan/Slang renderer.
- Build from transformed world-space `STATIC` triangles, never by resampling local SDF volumes.
- Dynamic objects never enter the global SDF and never trigger a rebuild.
- Use exactly 3 levels, 4x4x4 voxels per brick, and logical brick grids 32^3, 16^3, 8^3.
- Level 2 is dense; levels 0 and 1 allocate a 2-brick influence band around static triangle AABBs.
- Store `max(0, exact_distance - 0.5 * sqrt(3) * voxel_size)` plus the nearest canonical global triangle ID.
- Maximum page-table entries: 37,376. Maximum physical voxels: 2,392,064.
- Page-table entries are **level-local physical brick indices**. `GPUGlobalSDFClipmap.data.z` is the global voxel-array base for that level. This exactly matches `data.z + brick_index * data.y + local` in the frozen shader ABI.
- `RADIANCE_FEATURE_GLOBAL_SDF` is enabled only for valid static clipmaps; zero-static scenes succeed with `sdf_counts.z == 0`.
- `global_sdf_params.z` is `0.65f`.
- `GLOBAL_SDF_DATA` itself owns valid one-element dummy GPU buffers/SRVs for zero-static scenes; the old global fields in `RADIANCE_SCENE_FALLBACKS` are removed rather than kept as a parallel representation.
- Do not add a second global tracer, triangle-identity system, or per-frame global-SDF rebuild.
- Final branch contains no temporary Stage 9 workflow/helper/test files.

## Review Focus

- **Static/dynamic separation:** mixed-state scenes must never place a dynamic triangle ID in any global voxel. Task 2 tests this.
- **Thin geometry / conservative marching:** stored distances never exceed exact center distance; approximate voxel hits never become final surfaces without triangle refinement. Tasks 2 and 4 test this.
- **Empty/static-free scene:** global descriptors remain valid while feature/counts are disabled. Task 3 tests this.
- **Large/degenerate bounds:** tiny extents and maximum logical occupancy do not divide by zero, overflow, or leave partial allocations. Task 2 tests this.
- **Ordering/fallback:** exact global hits bypass exhaustive static-local tracing; unresolved global rays still reach local fallback. Task 4 tests this.

---

### Task 1: Lock Stage 9 ownership and builder interfaces

**Files:**
- Modify: `game.h`
- Temporary test: `.github/stage9_arch_test.py`

**Interfaces:**
- Consumes: existing `GPU_GLOBAL_SDF_CLIPMAP`, `RADIANCE_SCENE_DATA`, `GPU_OBJECT`, NRI types.
- Produces:
  - `GLOBAL_SDF_DATA`
  - `bool sdf_build_global_clipmaps(const RADIANCE_SCENE_DATA *scene, const GPU_OBJECT *objects, uint32_t object_count, GLOBAL_SDF_DATA *out);`
  - `void sdf_free_global_clipmaps(GLOBAL_SDF_DATA *data);`
  - `RENDERER.global_sdf`
  - `RENDERER.wavefront_global_pipeline`

- [ ] **Step 1: Write the failing interface test**

Create `.github/stage9_arch_test.py` and assert the desired `GLOBAL_SDF_DATA`, builder/free declarations, renderer member, and global pipeline member exist. Do not add special RED behavior; the desired-state assertions must fail naturally on the current branch.

- [ ] **Step 2: Run it and verify RED**

Run: `python3 .github/stage9_arch_test.py`

Expected: FAIL because the Stage 9 ownership/API is absent.

- [ ] **Step 3: Add the exact `GLOBAL_SDF_DATA` layout to `game.h`**

```c
typedef struct GLOBAL_SDF_DATA {
    GPU_GLOBAL_SDF_CLIPMAP *cpu_clipmaps;
    uint32_t *cpu_page_table;
    float *cpu_bricks;
    uint32_t *cpu_surface_ids;

    NriBuffer *clipmaps;
    NriBuffer *page_table;
    NriBuffer *bricks;
    NriBuffer *surface_ids;
    NriDescriptor *clipmaps_srv;
    NriDescriptor *page_table_srv;
    NriDescriptor *bricks_srv;
    NriDescriptor *surface_ids_srv;
    NriAccessStage state;

    uint32_t clip_count;
    uint32_t page_table_count;
    uint32_t physical_brick_count;
    uint32_t voxel_count;
    float coarsest_voxel_size;
    bool valid;
} GLOBAL_SDF_DATA;
```

Add the two exact function declarations from Interfaces. Add `GLOBAL_SDF_DATA global_sdf;` beside `dynamic_grid`, and `NriPipeline *wavefront_global_pipeline;` between dynamic and local pipelines.

Remove the four global-SDF buffers/SRVs from `RADIANCE_SCENE_FALLBACKS`; because that struct then has no current fields, remove the struct and `RENDERER.radiance_fallbacks` entirely rather than leave an empty future placeholder.

- [ ] **Step 4: Keep ABI assertions unchanged**

The test must assert `_Static_assert(sizeof(GPU_GLOBAL_SDF_CLIPMAP) == 64u, ...)` still exists. Do not change descriptor-space ABI structures.

- [ ] **Step 5: Run GREEN**

Run: `python3 .github/stage9_arch_test.py`

Expected: PASS for Task 1 assertions.

- [ ] **Step 6: Commit**

```bash
git add game.h .github/stage9_arch_test.py
git commit -m "s9 global sdf interfaces"
```

### Task 2: Build conservative static world clipmaps in `sdf.c`

**Files:**
- Modify: `sdf.c`
- Temporary test: `.github/stage9_sdf_test.c`

**Interfaces:**
- Consumes: Task 1 builder API, `radiance_scene->cpu_triangles`, `GPU_OBJECT.world/state/type`.
- Produces: fully populated CPU arrays/counts in `GLOBAL_SDF_DATA`; no GPU objects.

- [ ] **Step 1: Write the failing CPU regression test**

Create `.github/stage9_sdf_test.c`. Build a synthetic `RADIANCE_SCENE_DATA` containing one triangle owned by a `STATIC` `GPU_OBJECT` and one triangle owned by a `DYNAMIC` `GPU_OBJECT`; use identity matrices first, plus a translation case to pin matrix convention.

Required assertions after `sdf_build_global_clipmaps(...)`:

```c
assert(global.valid);
assert(global.clip_count == 3u);
assert(global.page_table_count == 37376u);
assert(global.physical_brick_count <= 37376u);
assert(global.voxel_count == global.physical_brick_count * 64u);
```

For every non-`UINT32_MAX` surface ID, assert `< scene.triangle_count` and not equal to the dynamic triangle ID. Reconstruct sampled voxel centers and assert `stored_distance <= exact_center_distance + 1e-5f`. For every non-empty page entry, assert its level-local brick index is `< that_level_physical_brick_count` and its computed `data.z + brick * 64 + local` address is `< global.voxel_count`.

Add:
- zero-static case: `valid == false`, `clip_count == 0`, safely freeable;
- translated static triangle case: populated surface band follows the transformed position;
- near-zero scene extent: no divide-by-zero and bounded allocation;
- fully occupied logical-grid accounting: count arithmetic remains within the fixed maxima.

- [ ] **Step 2: Run it and verify RED**

First prepare headers in the verification environment:

```bash
git clone --depth 1 https://github.com/libsdl-org/SDL.git /tmp/SDL
git clone --depth 1 https://github.com/NVIDIA-RTX/NRI.git /tmp/NRI
```

Then run:

```bash
clang -std=c11 -I. -I/tmp/SDL/include -I/tmp/NRI/Include -I/tmp/NRI/Include/Extensions \
  .github/stage9_sdf_test.c sdf.c scene.c -lm -o /tmp/stage9_sdf_test
/tmp/stage9_sdf_test
```

Expected: compile/link FAIL because `sdf_build_global_clipmaps` is not implemented.

- [ ] **Step 3: Refactor the existing nearest-triangle BVH helpers without changing `sdf_build_volume()` behavior**

Keep one internal triangle record with a caller-supplied `surface_id`, one BVH builder, and one nearest-distance query. Existing local-SDF generation must still pass local face IDs; Stage 9 passes canonical global triangle IDs.

Add private `sdf_transform_point(MAT4 matrix, VEC3 point)` using the same matrix convention already used by `render.c::mat4_point`:

```text
x' = x*m[0] + y*m[4] + z*m[8]  + m[12]
y' = x*m[1] + y*m[5] + z*m[9]  + m[13]
z' = x*m[2] + y*m[6] + z*m[10] + m[14]
```

- [ ] **Step 4: Implement `sdf_build_global_clipmaps(...)`**

Required decisions:

- iterate canonical `GPU_SCENE_TRIANGLE` records directly;
- validate `triangle.meta[0] < object_count`;
- include a triangle only when `objects[meta[0]].type == MODEL && objects[meta[0]].state == STATIC`;
- transform `p0/p1/p2` by that object's world matrix;
- keep the triangle-array index as canonical `surface_id`;
- build one world BVH;
- compute one padded cubic static-world bound, using at least one coarse voxel of nonzero padding;
- level configurations are exactly `{32,16,8}` logical bricks and brick resolution `4`;
- fine/medium bitsets mark triangle AABB ranges expanded by `2 * brick_world_size`;
- coarse bitset is fully set;
- page-table bases are cumulative logical entry counts;
- **page-table values are level-local physical brick indices**, starting at 0 for each level;
- `data.z` is the cumulative global voxel base for the level;
- `data.x = 4`, `data.y = 64`;
- missing page entries are `UINT32_MAX`;
- sample each allocated voxel center with the shared world BVH;
- store `max(0, exact - half_diagonal)` and nearest canonical triangle ID;
- set `coarsest_voxel_size` from level 2;
- set `valid = true` only after the complete CPU build succeeds.

All products/sums used for page, brick, voxel, byte, and allocation counts are checked in `uint64_t`/`size_t` before narrowing.

- [ ] **Step 5: Implement `sdf_free_global_clipmaps(...)`**

Free only CPU-owned arrays and zero all CPU/count/valid fields. Do not destroy NRI resources here; renderer ownership does that before calling this function. The function is safe on zero/partial data.

- [ ] **Step 6: Run GREEN CPU tests**

Run the exact compile/run command from Step 2.

Expected: PASS.

- [ ] **Step 7: Re-run current local SDF compilation**

Run: `clang -std=c11 -fsyntax-only -I. -I/tmp/SDL/include -I/tmp/NRI/Include -I/tmp/NRI/Include/Extensions sdf.c`

Expected: PASS; `sdf_build_volume` remains available and unchanged externally.

- [ ] **Step 8: Commit**

```bash
git add sdf.c .github/stage9_sdf_test.c
git commit -m "s9 build static global sdf"
```

### Task 3: Replace placeholder globals with owned Stage 9 GPU resources

**Files:**
- Modify: `render.c`
- Modify: `game.h` only if Task 2 reveals a spec-required ownership field is missing.
- Temporary test: `.github/stage9_arch_test.py`

**Interfaces:**
- Consumes: Task 2 CPU `GLOBAL_SDF_DATA`.
- Produces:
  - `static void destroy_global_sdf_resources(RENDERER *renderer);`
  - `static bool create_global_sdf_resources(RENDERER *renderer);`
  - always-valid `GLOBAL_SDF_DATA` SRVs, real or dummy.

- [ ] **Step 1: Extend the architecture test and verify RED**

Require the two functions above, require `renderer_set_scene()` to build Stage 9 before radiance constants/descriptors, require global descriptor binding to use `renderer->global_sdf`, and require the old `create_radiance_scene_fallbacks`/`destroy_radiance_scene_fallbacks` global placeholder path to be gone.

Run: `python3 .github/stage9_arch_test.py`

Expected: FAIL on Task 3 assertions.

- [ ] **Step 2: Implement `destroy_global_sdf_resources(RENDERER *renderer)`**

Destroy the four SRVs and four buffers if present, then call `sdf_free_global_clipmaps(&renderer->global_sdf)` and zero the owner. Partial initialization is safe.

- [ ] **Step 3: Implement `create_global_sdf_resources(RENDERER *renderer)`**

Start by destroying prior Stage 9 state, then call the Task 2 builder.

For a valid static build, allocate/upload exact-size device SRV buffers with these strides:

```text
clipmaps     sizeof(GPU_GLOBAL_SDF_CLIPMAP)
page_table   sizeof(uint32_t)
bricks       sizeof(float)
surface_ids  sizeof(uint32_t)
```

For a legitimate zero-static scene, allocate one element for each resource and upload:

```text
zero GPUGlobalSDFClipmap
UINT32_MAX page entry
0.0f distance
UINT32_MAX surface id
```

Those dummy SRVs belong to `GLOBAL_SDF_DATA`, while `valid == false` and `clip_count == 0` remain authoritative.

Any real build/allocation/view/upload failure destroys partial Stage 9 state and returns `false`; only zero-static is disabled-success.

- [ ] **Step 4: Bind Stage 9 descriptors directly**

In `update_radiance_scene_descriptors()`, `future_scene[3..6]` must be:

```c
renderer->global_sdf.clipmaps_srv
renderer->global_sdf.page_table_srv
renderer->global_sdf.bricks_srv
renderer->global_sdf.surface_ids_srv
```

No conditional descriptor selection is needed because Task 3 guarantees they are always valid.

- [ ] **Step 5: Wire scene/renderer lifetime exactly once**

`renderer_init()` no longer creates the old global fallback owner. `renderer_set_scene()` creates Stage 8 dynamic grid, then Stage 9 global SDF, then updates radiance constants/descriptors. Scene teardown and `renderer_deinit()` call `destroy_global_sdf_resources()` exactly once per owned lifetime.

- [ ] **Step 6: Populate feature/count constants**

Update the existing radiance constants helper:

```c
constants->sdf_counts[2] = renderer->global_sdf.valid ? renderer->global_sdf.clip_count : 0u;
constants->global_sdf_params[1] = renderer->global_sdf.valid ? renderer->global_sdf.coarsest_voxel_size : 0.25f;
constants->global_sdf_params[2] = 0.65f;
constants->feature_flags[0] &= ~RADIANCE_FEATURE_GLOBAL_SDF;
if (renderer->global_sdf.valid && renderer->global_sdf.clip_count)
    constants->feature_flags[0] |= RADIANCE_FEATURE_GLOBAL_SDF;
```

- [ ] **Step 7: Run Task 3 GREEN checks**

Run: `python3 .github/stage9_arch_test.py`

Then syntax-check all C translation units with:

```bash
FLAGS='-I. -I/tmp/SDL/include -I/tmp/NRI/Include -I/tmp/NRI/Include/Extensions'
for f in init.c glb.c gltf.c scene.c sdf.c gpu.c render.c main.c; do
  clang -std=c11 -fsyntax-only $FLAGS "$f"
done
```

Expected: all PASS.

- [ ] **Step 8: Commit**

```bash
git add game.h render.c .github/stage9_arch_test.py
git commit -m "s9 upload global sdf"
```

### Task 4: Activate exact global tracing and static-local fallback

**Files:**
- Modify: `shader.hlsl`
- Modify: `render.c`
- Verify only: `build.c` (the current branch already contains exactly one `CS_WavefrontGlobalTrace -> radiance_global.cs.spv` job)
- Temporary test: `.github/stage9_arch_test.py`

**Interfaces:**
- Consumes: Task 3 bound global resources/constants.
- Produces: active `wavefront_global_pipeline`, exact global hits, `dynamic -> global -> local fallback -> shade`.

- [ ] **Step 1: Add desired-state shader/order assertions and verify RED**

Require:

- pipeline creation from `build/shaders/radiance_global.cs.spv` with `wavefront_layout`;
- pipeline destruction in the normal pipeline list;
- wavefront dispatch order dynamic, global, local, shade;
- unified query order dynamic, global, local only when global did not resolve;
- global hit acceptance calls `IntersectSceneTriangle`/shared exact refinement before `SurfaceFromTriangle(... TRACE_GLOBAL_SDF)`;
- `CS_WavefrontGlobalTrace` sets a dedicated resolved bit;
- `CS_WavefrontLocalTrace` skips exhaustive static local tracing for that bit;
- `build.c` contains exactly one `CS_WavefrontGlobalTrace` job.

Run: `python3 .github/stage9_arch_test.py`

Expected: FAIL on Stage 9 runtime assertions.

- [ ] **Step 2: Add exact global candidate refinement**

Refactor the existing local exact-intersection utility only enough to share it with global tracing. When `TraceGlobalSDF()` reaches `d <= epsilon`:

1. collect the sampled global surface ID;
2. collect unique neighboring voxel surface IDs from the same finest available level around the current sample; if that level is sparse/unavailable, use the next coarser available level;
3. intersect candidates with `IntersectSceneTriangle` in a bounded window around current `t`;
4. choose the nearest real intersection that improves `best_hit`;
5. create `SurfaceFromTriangle(..., TRACE_GLOBAL_SDF)` from exact position/geometric normal;
6. if no candidate intersects, advance and continue tracing.

Do not use `GlobalSDFNormal()` as final surface geometry.

- [ ] **Step 3: Make the unified query hierarchy match the spec**

`TraceUnifiedRay()` becomes:

```text
optional screen
TraceDynamicGrid
bool global_resolved = TraceGlobalSDF
if (!global_resolved) TraceAllLocalSDFs
```

`TraceGlobalSDF()` returns `true` only when it accepted an exact static hit that improves the current best hit.

- [ ] **Step 4: Create/destroy `wavefront_global_pipeline`**

Use the same `create_compute_pipeline(..., renderer->wavefront_layout, ...)` pattern as dynamic/local and load `build/shaders/radiance_global.cs.spv`. Destroy it with adjacent wavefront pipelines.

- [ ] **Step 5: Dispatch dynamic -> global -> local -> shade**

In `build_wavefront_screen_probes()`, after dynamic dispatch/barrier, dispatch the global pipeline with the same `ray_groups`, then barrier, then local, then shade.

- [ ] **Step 6: Mark globally resolved wavefront rays**

Keep existing `0x80000000u` dynamic-hit bit. Reserve `0x40000000u` as `RAY_FLAG_GLOBAL_RESOLVED`.

`CS_WavefrontGlobalTrace` starts from the current dynamic hit if present, calls `TraceGlobalSDF`, writes any improved hit, and sets `RAY_FLAG_GLOBAL_RESOLVED` only when an exact global hit resolved the static query.

`CS_WavefrontLocalTrace` immediately preserves/returns the current hit when that bit is set; otherwise it performs `TraceAllLocalSDFs` as correctness fallback.

- [ ] **Step 7: Compile every active shader entry**

Use Slang v2026.18.3 with the existing options:

```text
-target spirv -profile sm_6_6 -capability spirv_1_5
-matrix-layout-row-major -fvk-use-dx-layout -O3
```

Compile at minimum:

```text
VS_GBuffer
PS_GBufferFull
CS_RadianceDirect
CS_ResetWavefront
CS_ClassifyRayBudgets
CS_GenerateProbeRays
CS_WavefrontScreenTrace
CS_WavefrontDynamicTrace
CS_WavefrontGlobalTrace
CS_WavefrontLocalTrace
CS_ShadeRayHits
CS_ReprojectScreenProbes
CS_SpatialReuseScreenProbes
CS_ResolveDirectionalProbes
CS_EmissiveGather
CS_CommitScreenProbeHistory
PS_Present
```

Expected: Slang exits 0 and every output SPIR-V is non-empty.

- [ ] **Step 8: Run structural/C gates**

Run:

```bash
python3 .github/stage9_arch_test.py
git diff --check
FLAGS='-I. -I/tmp/SDL/include -I/tmp/NRI/Include -I/tmp/NRI/Include/Extensions'
for f in init.c glb.c gltf.c scene.c sdf.c gpu.c render.c main.c; do
  clang -std=c11 -fsyntax-only $FLAGS "$f"
done
```

Expected: all PASS.

- [ ] **Step 9: Commit**

```bash
git add shader.hlsl render.c build.c .github/stage9_arch_test.py
git commit -m "s9 trace static global sdf"
```

### Task 5: Full verification and cleanup

**Files:**
- Verify: `game.h`, `sdf.c`, `render.c`, `shader.hlsl`, `build.c`
- Remove: `.github/stage9_arch_test.py`, `.github/stage9_sdf_test.c`, and any temporary `stage9-*` workflow/helper files created for remote verification.

**Interfaces:**
- Consumes: complete Stage 9 implementation.
- Produces: clean `radiance` branch with only production code + approved spec/plan.

- [ ] **Step 1: Run CPU regression suite**

Run the exact Task 2 compile/run command.

Expected: PASS.

- [ ] **Step 2: Run architecture regression suite**

Run: `python3 .github/stage9_arch_test.py`

Expected: PASS.

- [ ] **Step 3: Run full shader, C, and diff gates**

Use the Task 4 Slang list, all C translation-unit syntax checks, and `git diff --check`.

Expected: all PASS.

- [ ] **Step 4: Audit spec invariants mechanically**

Verify:

```text
STATIC GPU objects only -> global builder
3 levels -> 32^3 / 16^3 / 8^3 bricks
brick resolution -> 4
coarse -> dense
fine/medium -> 2-brick sparse band
page values -> level-local brick indices
data.z -> level global voxel base
conservative distance -> exact - half diagonal
canonical global triangle IDs
zero-static -> dummy descriptors + feature/count off
wavefront -> dynamic -> global -> local fallback -> shade
unified query -> dynamic -> global -> local fallback
accepted global hit -> exact triangle intersection
no per-frame global rebuild
```

- [ ] **Step 5: Remove temporary verification scaffolding**

Delete only temporary Stage 9 test/workflow/helper files. Keep this plan and the design spec.

- [ ] **Step 6: Verify cleanup did not alter production code**

Compare the final verified production commit to cleanup HEAD. Expected changed paths: temporary Stage 9 verification files only.

- [ ] **Step 7: Commit cleanup**

```bash
git add -A .github
git commit -m "remove stage9 verification helpers"
```

- [ ] **Step 8: Runtime validation on the target GPU**

On the M2/NRI Vulkan path run individually:

```bash
c build run -- cornell_box.glb
c build run -- hospital_hallway.glb
c build run -- poolroom.glb
```

Expected: no pipeline/startup errors, interactive frame rate, camera-independent offscreen static tracing, and no reintroduction of the previous structured green/white wall striping. Runtime visual success is reported separately from compile/structural completion; do not claim it before the M2 run.
