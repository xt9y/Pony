# Pony Radiance Stage 9 Global SDF Clipmaps Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a bounded three-level static world SDF so offscreen static-scene rays use one scene-wide acceleration structure instead of exhaustively testing every static local SDF.

**Architecture:** Build a CPU BVH over transformed `STATIC` triangles, generate three 4x4x4-brick conservative unsigned SDF levels, upload them through the existing `space4` global-SDF ABI, and activate `CS_WavefrontGlobalTrace` between dynamic-grid tracing and static-local fallback. Global voxel hits are only approximate candidates; accepted hits must be refined against canonical `SceneTriangles` before shading.

**Tech Stack:** C11, NRI/Vulkan, Slang/HLSL, SDL3, GitHub Actions verification.

**Spec:** `docs/superpowers/specs/2026-10-01-global-sdf-clipmaps-design.md`

## Global Constraints

- Preserve the existing NRI/Vulkan/Slang rendering backend.
- Build from transformed world-space `STATIC` triangles, never by resampling local SDF volumes.
- Dynamic objects never enter the global SDF and never trigger a rebuild.
- Use exactly 3 levels with 4x4x4 voxels per brick and logical brick grids 32^3, 16^3, and 8^3.
- Level 2 is dense; levels 0 and 1 allocate bricks within a 2-brick influence radius of static triangle AABBs.
- Store conservative distance `max(0, exact_distance - 0.5 * sqrt(3) * voxel_size)` and the nearest canonical global triangle ID.
- Maximum page-table entries: 37,376. Maximum physical voxels: 2,392,064.
- `RADIANCE_FEATURE_GLOBAL_SDF` is enabled only when valid static clipmaps exist; zero-static scenes are successful with `sdf_counts.z == 0`.
- `global_sdf_params.z` starts at `0.65`.
- Keep a valid one-element zero descriptor fallback for zero-static scenes because `space4` descriptor bindings remain mandatory.
- Do not add a second global tracer, triangle-identity system, or per-frame global-SDF rebuild.
- Final repository state contains no temporary Stage 9 helper/workflow files.

## Review Focus

- **Static/dynamic separation:** a scene containing both states must never put a dynamic triangle ID in any global-SDF voxel. Task 2 owns this test.
- **Thin geometry / conservative marching:** stored distances must never exceed exact center distance and exact triangle refinement must reject false voxel hits. Tasks 2 and 4 own these tests.
- **Empty/static-free scene:** descriptor binding remains valid while feature/counts remain disabled. Task 3 owns this test.
- **Large or degenerate scene bounds:** zero/near-zero extent and maximum logical occupancy must not overflow counts or allocate beyond caps. Task 2 owns this test.
- **Ordering/correctness fallback:** global hits must bypass exhaustive static-local tracing while unresolved global rays still reach local fallback. Task 4 owns this test.

---

### Task 1: Lock the Stage 9 CPU/GPU ownership interfaces

**Files:**
- Modify: `game.h`
- Test: temporary `.github/stage9_arch_test.py`

**Interfaces:**
- Consumes: existing `GPU_GLOBAL_SDF_CLIPMAP`, `RADIANCE_SCENE_DATA`, `GPU_OBJECT`, `SCENE`, NRI buffer/descriptor types.
- Produces:
  - `GLOBAL_SDF_DATA`
  - `bool sdf_build_global_clipmaps(const SCENE *scene, const RADIANCE_SCENE_DATA *radiance_scene, const GPU_OBJECT *objects, uint32_t object_count, GLOBAL_SDF_DATA *out);`
  - `void sdf_free_global_clipmaps(GLOBAL_SDF_DATA *data);`
  - `RENDERER.global_sdf`
  - `RENDERER.wavefront_global_pipeline`

- [ ] **Step 1: Write the failing architecture test**

Create `.github/stage9_arch_test.py` with assertions that `game.h` does not yet contain `GLOBAL_SDF_DATA`, `sdf_build_global_clipmaps`, `sdf_free_global_clipmaps`, or `wavefront_global_pipeline`; run it in RED mode by asserting their absence.

- [ ] **Step 2: Run the RED check**

Run: `python3 .github/stage9_arch_test.py --red`

Expected: PASS proving Stage 9 ownership/API is absent before implementation.

- [ ] **Step 3: Add the exact ownership structure and API to `game.h`**

Define `GLOBAL_SDF_DATA` with:

```c
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
```

Add the two builder/free declarations above, add `GLOBAL_SDF_DATA global_sdf;` beside `dynamic_grid`, and add `NriPipeline *wavefront_global_pipeline;` between dynamic and local wavefront pipelines.

- [ ] **Step 4: Extend the architecture test for GREEN assertions**

Assert the exact type/API/renderer members exist and `sizeof(GPU_GLOBAL_SDF_CLIPMAP) == 64u` remains unchanged.

- [ ] **Step 5: Run the GREEN interface check**

Run: `python3 .github/stage9_arch_test.py`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add game.h .github/stage9_arch_test.py
git commit -m "s9 global sdf interfaces"
```

### Task 2: Build conservative static world clipmaps in `sdf.c`

**Files:**
- Modify: `sdf.c`
- Modify: `game.h` only if an implementation-private public test helper is strictly necessary; prefer none.
- Test: temporary `.github/stage9_sdf_test.c`

**Interfaces:**
- Consumes: Task 1 `sdf_build_global_clipmaps(...)`, scene model order, `radiance_scene->cpu_triangles`, `GPU_OBJECT.world`.
- Produces: fully populated CPU fields of `GLOBAL_SDF_DATA`; no GPU resources.

- [ ] **Step 1: Write a failing standalone C test**

Create `.github/stage9_sdf_test.c` with synthetic identity-transformed model geometry containing one `STATIC` triangle and one `DYNAMIC` triangle. Assert after `sdf_build_global_clipmaps(...)`:

```c
assert(global.valid);
assert(global.clip_count == 3u);
assert(global.page_table_count == 37376u);
assert(global.physical_brick_count <= 37376u);
assert(global.voxel_count == global.physical_brick_count * 64u);
```

For every stored non-`UINT32_MAX` surface ID assert it is `< radiance_scene.triangle_count` and never equals the dynamic triangle ID. For sampled voxels, recompute exact point-to-triangle center distance and assert `stored <= exact + 1e-5f`.

Add a zero-static case and assert `valid == false`, `clip_count == 0`, and all CPU arrays are safely freeable.

Add a degenerate tiny static triangle case and assert the builder either creates the bounded 3-level cube successfully or returns `false` without partial allocations; no divide-by-zero/overflow is allowed.

- [ ] **Step 2: Run the test and verify it fails to link**

Run: `cc -std=c11 -I. .github/stage9_sdf_test.c sdf.c scene.c -lm -o /tmp/stage9_sdf_test && /tmp/stage9_sdf_test`

Expected: FAIL because `sdf_build_global_clipmaps` is not implemented.

- [ ] **Step 3: Refactor the existing local-SDF BVH internals into reusable world-triangle helpers**

Keep the existing local builder behavior unchanged. Internal helpers must support a triangle record carrying a caller-supplied canonical `surface_id`, BVH build, nearest-distance query, and nearest surface ID.

- [ ] **Step 4: Implement `sdf_build_global_clipmaps(...)`**

Required algorithm decisions:

- collect only `OBJECT.type == MODEL && OBJECT.state == STATIC`;
- derive each model's canonical triangle base from the existing radiance-scene ordering, not from object IDs;
- transform triangle vertices by `objects[model_index].world`;
- build one world BVH;
- compute one padded cubic scene bound;
- emit logical grids `{32,16,8}` with fixed brick resolution `4`;
- mark level 0/1 requested bricks by expanding each triangle AABB by exactly `2 * brick_world_size`;
- mark all level-2 bricks requested;
- collapse requests with one bitset per level;
- assign physical brick indices contiguously in level order;
- fill `GPU_GLOBAL_SDF_CLIPMAP.grid.w` with each level's page-table base and `data.z` with the level's first voxel-data offset;
- fill missing page entries with `UINT32_MAX`;
- sample each physical voxel center through the world BVH;
- store the conservative distance formula from Global Constraints;
- set `coarsest_voxel_size` from level 2 and `valid = true` only after the full build succeeds.

All multiplication/addition count calculations must be checked in 64-bit before narrowing or allocating.

- [ ] **Step 5: Implement `sdf_free_global_clipmaps(...)`**

Free all four CPU arrays and zero the entire struct. It must be safe on zero/partial data.

- [ ] **Step 6: Run the CPU regression test**

Run: `cc -std=c11 -I. .github/stage9_sdf_test.c sdf.c scene.c -lm -o /tmp/stage9_sdf_test && /tmp/stage9_sdf_test`

Expected: PASS for mixed static/dynamic, zero-static, conservative-distance, bounds, and cleanup assertions.

- [ ] **Step 7: Commit**

```bash
git add sdf.c .github/stage9_sdf_test.c
git commit -m "s9 build static global sdf"
```

### Task 3: Own and upload the real global-SDF GPU resources

**Files:**
- Modify: `render.c`
- Modify: `game.h` only if Task 2 exposed a missing field required by the approved spec.
- Test: temporary `.github/stage9_arch_test.py`

**Interfaces:**
- Consumes: Task 2 CPU `GLOBAL_SDF_DATA` arrays/counts.
- Produces:
  - `static void destroy_global_sdf_resources(RENDERER *renderer);`
  - `static bool create_global_sdf_resources(RENDERER *renderer);`
  - real global-SDF SRVs used by `update_radiance_scene_descriptors()`.

- [ ] **Step 1: Extend the architecture test with failing resource-lifetime assertions**

Require `render.c` to own `destroy_global_sdf_resources`, `create_global_sdf_resources`, call the builder from `renderer_set_scene`, and stop binding the four global descriptors from `radiance_fallbacks` for normal scenes.

- [ ] **Step 2: Run the architecture test**

Run: `python3 .github/stage9_arch_test.py`

Expected: FAIL on the new Stage 9 GPU-resource assertions.

- [ ] **Step 3: Implement `destroy_global_sdf_resources(RENDERER *renderer)`**

Destroy four Stage 9 SRVs/buffers, call `sdf_free_global_clipmaps(&renderer->global_sdf)`, and zero GPU access state. Partial initialization must be safe.

- [ ] **Step 4: Implement `create_global_sdf_resources(RENDERER *renderer)`**

Call `sdf_build_global_clipmaps(...)`. If there are zero static triangles (`valid == false && clip_count == 0`), keep the existing one-element zero fallback descriptors and return success. Otherwise create four `NriBufferUsageBits_SHADER_RESOURCE` device buffers with exact populated sizes, create structured SRVs with strides `sizeof(GPU_GLOBAL_SDF_CLIPMAP)`, `sizeof(uint32_t)`, `sizeof(float)`, `sizeof(uint32_t)`, upload all CPU arrays once, and set compute-read state.

Any allocation/view/upload failure must destroy partial Stage 9 resources and return `false`; do not silently disable the feature.

- [ ] **Step 5: Switch descriptor binding to Stage 9 resources when valid**

In `update_radiance_scene_descriptors()`, choose each global descriptor from `renderer->global_sdf` when `valid`, otherwise from the existing one-element zero fallback. Do not change descriptor counts/ranges.

- [ ] **Step 6: Wire scene lifetime**

In `renderer_set_scene()`, after normal scene resources and Stage 8 dynamic-grid construction, build/upload Stage 9 before `update_radiance_constants()` and descriptor updates. Destroy Stage 9 in both scene-resource teardown and renderer deinit paths exactly once.

- [ ] **Step 7: Populate constants and feature state**

Update the existing radiance-constant helper so:

```c
constants->sdf_counts[2] = renderer->global_sdf.valid ? renderer->global_sdf.clip_count : 0u;
constants->global_sdf_params[1] = renderer->global_sdf.valid ? renderer->global_sdf.coarsest_voxel_size : 0.25f;
constants->global_sdf_params[2] = 0.65f;
```

Clear `RADIANCE_FEATURE_GLOBAL_SDF` first, then set it only when `global_sdf.valid && clip_count != 0`.

- [ ] **Step 8: Run structural checks and C syntax compilation**

Run: `python3 .github/stage9_arch_test.py`

Run the repository's normal C translation-unit syntax gate (same include/link flags as existing CI) and require PASS.

- [ ] **Step 9: Commit**

```bash
git add render.c game.h .github/stage9_arch_test.py
git commit -m "s9 upload global sdf"
```

### Task 4: Activate exact global tracing and local fallback ordering

**Files:**
- Modify: `shader.hlsl`
- Modify: `render.c`
- Verify: `build.c` contains exactly one `CS_WavefrontGlobalTrace` shader job; modify only if it is missing or duplicated.
- Test: temporary `.github/stage9_arch_test.py`

**Interfaces:**
- Consumes: Task 3 bound global buffers and feature/count constants.
- Produces: active `wavefront_global_pipeline`, exact global `SurfaceHit`s, `dynamic -> global -> local fallback -> shade` ordering.

- [ ] **Step 1: Add failing shader/order assertions**

Extend `.github/stage9_arch_test.py` to require:

- `wavefront_global_pipeline` is created from the existing global-trace SPIR-V job and destroyed in the pipeline list;
- `build_wavefront_screen_probes()` dispatch order is dynamic, global, local, shade;
- `TraceUnifiedRay()` order is dynamic, global, then local only if global did not resolve;
- `TraceGlobalSDF()` calls exact triangle intersection/refinement before accepting `TRACE_GLOBAL_SDF`;
- `CS_WavefrontGlobalTrace` sets a dedicated global-resolved flag bit;
- `CS_WavefrontLocalTrace` skips exhaustive static local tracing when that bit is set;
- `build.c` contains one and only one `CS_WavefrontGlobalTrace` shader compile job.

- [ ] **Step 2: Run the test and verify it fails**

Run: `python3 .github/stage9_arch_test.py`

Expected: FAIL because global dispatch/refinement is not active yet.

- [ ] **Step 3: Refine global hits against canonical triangles in `shader.hlsl`**

Reuse `IntersectSceneTriangle(...)`. When global SDF reaches `d <= epsilon`, inspect the sampled triangle ID plus neighboring global voxels from the same/next coarser available brick, find the nearest real intersection within a bounded window around `t`, and only then construct `SurfaceFromTriangle(..., TRACE_GLOBAL_SDF)`. If no candidate intersects, advance and continue sphere tracing; never shade `GlobalSDFNormal()` as authoritative geometry.

- [ ] **Step 4: Make unified tracing use dynamic -> global -> local fallback**

Change `TraceUnifiedRay()` to call `TraceDynamicGrid(ray, best)`, then `bool global_resolved = TraceGlobalSDF(ray, best)`, then call `TraceAllLocalSDFs(ray, best)` only when `!global_resolved`. Screen tracing remains optional and first.

- [ ] **Step 5: Activate the global wavefront pipeline in C**

Create `renderer->wavefront_global_pipeline` from the single global shader job using `renderer->wavefront_layout`; destroy it with the other pipelines. In `build_wavefront_screen_probes()`, dispatch it after dynamic and before local, using the same `ray_groups` and storage barriers as adjacent passes.

- [ ] **Step 6: Mark and honor globally resolved rays**

Reserve `0x40000000u` as `RAY_FLAG_GLOBAL_RESOLVED` (`0x80000000u` remains the existing dynamic-hit flag). `CS_WavefrontGlobalTrace` sets it only when `TraceGlobalSDF()` resolves an exact static hit. `CS_WavefrontLocalTrace` returns without exhaustive static tracing when this bit is set; unresolved rays still execute `TraceAllLocalSDFs`.

- [ ] **Step 7: Compile active Slang entry points**

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

Expected: all SPIR-V outputs are non-empty and Slang exits 0.

- [ ] **Step 8: Run architecture and C gates**

Run: `python3 .github/stage9_arch_test.py`

Run: `git diff --check`

Run the normal C syntax gate.

Expected: all PASS.

- [ ] **Step 9: Commit**

```bash
git add shader.hlsl render.c build.c .github/stage9_arch_test.py
git commit -m "s9 trace static global sdf"
```

### Task 5: End-to-end verification, resource audit, and cleanup

**Files:**
- Verify: `game.h`, `sdf.c`, `render.c`, `shader.hlsl`, `build.c`
- Remove: temporary `.github/stage9_arch_test.py`, `.github/stage9_sdf_test.c`, and any temporary Stage 9 workflow/helper files.

**Interfaces:**
- Consumes: complete Stage 9 implementation.
- Produces: clean `radiance` branch with no temporary Stage 9 scaffolding.

- [ ] **Step 1: Run the CPU builder regression suite again**

Run: `cc -std=c11 -I. .github/stage9_sdf_test.c sdf.c scene.c -lm -o /tmp/stage9_sdf_test && /tmp/stage9_sdf_test`

Expected: PASS.

- [ ] **Step 2: Run the architecture regression suite again**

Run: `python3 .github/stage9_arch_test.py`

Expected: PASS.

- [ ] **Step 3: Run full shader and C compile gates**

Require all active shader jobs, all C translation units, and `git diff --check` to pass.

- [ ] **Step 4: Audit the production diff against the spec**

Verify mechanically:

```text
STATIC only -> global SDF
3 levels -> 32^3 / 16^3 / 8^3 bricks
brick resolution -> 4
coarse level -> dense
fine/medium -> sparse 2-brick band
conservative distance -> exact - half diagonal
canonical global triangle IDs
zero static -> feature off + dummy descriptors
wavefront -> dynamic -> global -> local fallback -> shade
unified query -> dynamic -> global -> local fallback
accepted global hit -> exact triangle intersection
no per-frame global rebuild
```

- [ ] **Step 5: Remove all temporary Stage 9 test/workflow/helper files**

Delete only the temporary files named by this plan. Keep the design spec and this implementation plan.

- [ ] **Step 6: Confirm cleanup did not modify production code**

Compare the verified production commit to cleanup HEAD. Expected changed files: temporary Stage 9 verification files only.

- [ ] **Step 7: Commit cleanup**

```bash
git add -A .github
git commit -m "remove stage9 verification helpers"
```

- [ ] **Step 8: Runtime validation on target scenes**

On the M2/NRI Vulkan path run:

```bash
c build run -- cornell_box.glb
c build run -- hospital_hallway.glb
c build run -- poolroom.glb
```

Expected: interactive frame rate remains reasonable; no startup/pipeline errors; static offscreen tracing remains camera-independent; Stage 9 does not reintroduce green/white striping. Runtime visual success is reported separately from compile/structural success.
