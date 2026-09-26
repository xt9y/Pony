# Dynamic Runtime Lighting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add realtime `DYNAMIC` models to Pony while retaining the permanent static bake as an immutable baseline, with immediate moving sun shadows and a fixed-budget local GI lightmap overlay.

**Architecture:** The static scene keeps separate baked indirect and direct-sun lightmaps. Dynamic models use transformed local BVHs and volume probes; a dynamic-only sun shadow map handles immediate direct visibility. Nearby static lightmap cells are invalidated by transform changes and progressively retraced into a transient overlay under a hard per-frame texel/ray budget; cell generations reject stale work without clearing millions of texels.

**Tech Stack:** C11, NRI/Vulkan/MoltenVK-compatible rendering, SDL3/SDL_shadercross, HLSL compute/raster shaders, Pony software BVH/lightmap/probe systems.

**Spec:** `docs/superpowers/specs/2026-09-26-dynamic-lighting-design.md`

## Global Constraints

- `OBJECT_STATE` remains the public classification: exactly `STATIC` or `DYNAMIC`.
- Permanent `.baked` data describes the static world with dynamic objects absent from static lighting.
- No NRI ray tracing, SER, micromaps, vendor intrinsics, mesh-shader dependency, or other M2-incompatible requirement.
- Runtime GI uses the existing software-BVH style of traversal and must remain correct before the separate `nri` branch is merged.
- Dynamic GI work is capped every frame by `texels_per_frame * rays_per_texel`; standing still never increases that cap.
- New movement supersedes stale queued work. Never intentionally converge a transform that is no longer current.
- Dynamic GI is local: expand the swept dynamic bounds by `gi_radius`; do not globally rebake the level.
- Direct moving sun shadows update immediately and affect static and dynamic receivers.
- Runtime overlay data is transient and is never written into `.baked`.
- No new public header. Public declarations stay in `game.h`; renderer-internal declarations stay in `gpu.h`.
- Keep final `static-dynamic-geom` clean: no temporary CI workflows, test fixtures, plan/spec files, or debug-only code are left on the implementation branch.
- Do not depend on the unmerged `nri` branch. Shape job buffers and recording functions so its persistent contexts, indirect dispatch, wave compaction, and wavefront tracer can replace scheduling later without changing ownership semantics.

## File Structure

- Create `dynamic.c` — CPU dynamic-object registry, TRS transforms, world bounds, local BVHs, lightmap spatial cells, cell generations, dirty-cell jobs, fixed-budget GI job selection.
- Create `shaders/dynamic.hlsl` — dynamic-only directional shadow vertex shader plus fixed-budget runtime GI compute shader and transformed dynamic-BVH traversal.
- Modify `game.h` — public transform/settings types, `OBJECT` transform history, cache second lightmap layer, dynamic renderer API.
- Modify `gpu.h` — opaque/internal dynamic state and GPU resource declarations, dynamic render resources attached to `RENDERER`.
- Modify `build.c` — compile `dynamic.c` only; no new dependency.
- Modify `cache.c` — cache v9 with indirect and direct-sun RGBA16F payloads.
- Modify `render.c` — initialize dynamic lighting, register/remove dynamic objects, synchronize transforms before each frame, pass settings/frame state to GPU.
- Modify `gpu.c` — two-lightmap upload/download, dynamic model GPU resources, dynamic shadow map/pipeline, runtime overlay resources, cell-generation updates, GI pass, dynamic draw submission, lifecycle.
- Modify `main.c` — instantiate `DYNAMIC_LIGHTING`, initialize the runtime subsystem, preserve the existing static scene as the baked world.
- Modify `shaders/compute_base.hlsl` — separate primary direct sun from the permanent indirect/emissive base.
- Modify `shaders/vertex.hlsl` — model/normal transforms and dynamic-object mode for surface draws.
- Modify `shaders/fragment.hlsl` — static indirect/direct/overlay composition, dynamic probe irradiance, static beam visibility, dynamic shadow visibility.

## Review Focus

1. **Object moves again while GI jobs are queued:** old jobs must fail their cell-generation check; the old and new swept regions must fall back to the immutable base immediately.
2. **Two dynamic objects overlap the same spatial cell:** invalidation may discard prior overlay for that cell, but the replacement trace must include the complete current dynamic scene and converge to the combined result.
3. **No dynamic objects registered:** static rendering and bake output must match the pre-feature result apart from the intentional cache-version/lightmap-layer split.
4. **Dirty region exceeds one-frame budget:** frame work must remain bounded; the queue persists across frames without switching into an idle/high-throughput mode.
5. **Rotated/non-uniformly-scaled dynamic model:** conservative world bounds, transformed rays, hit `t`, and inverse-transpose normals must remain correct.

---

### Task 1: Dynamic object model, spatial index, and fixed-budget CPU scheduler

**Files:**
- Create: `dynamic.c`
- Modify: `game.h`
- Modify: `gpu.h`
- Modify: `build.c`
- Modify: `main.c`
- Modify: `render.c`

**Interfaces:**
- Produces public types:
  - `TRANSFORM { VEC3 position; VEC3 rotation; VEC3 scale; }`; rotation is XYZ Euler radians and model matrix order is `T * Rz * Ry * Rx * S`.
  - `DYNAMIC_LIGHTING { uint32_t texels_per_frame; uint32_t rays_per_texel; uint32_t target_samples; uint32_t shadow_map_size; float gi_radius; float shadow_bias; }`.
  - `OBJECT` gains `TRANSFORM transform; TRANSFORM previous_transform;` before `void *data`.
- Produces public API:
  - `bool r_dynamic_init(RENDERER *r, const MESH *static_scene, const LIGHTMAP *lm, const DYNAMIC_LIGHTING *settings);`
  - `bool r_add_dynamic_object(RENDERER *r, OBJECT *object);`
  - `void r_remove_dynamic_object(RENDERER *r, OBJECT *object);`
- Produces renderer-internal API in `gpu.h`:
  - `bool dynamic_sync(RENDERER *r);`
  - `uint32_t dynamic_take_gi_jobs(RENDERER *r, DYNAMIC_GI_JOB *out, uint32_t capacity);`
  - `uint32_t dynamic_take_cell_updates(RENDERER *r, DYNAMIC_CELL_UPDATE *out, uint32_t capacity);`
- `DYNAMIC_GI_JOB` contains `sample_index`, `cell_index`, `generation`, and padding to 16 bytes.
- `DYNAMIC_CELL_UPDATE` contains `cell_index`, `generation`.

- [ ] **Step 1: Add the final API usage to `main.c` before defining the new types/functions**

Add a final default configuration and initialization call:

```c
DYNAMIC_LIGHTING dynamic_lighting = {
    .texels_per_frame = 4096u,
    .rays_per_texel = 1u,
    .target_samples = 16u,
    .shadow_map_size = 1024u,
    .gi_radius = 3.0f,
    .shadow_bias = 0.0025f
};
```

Call `r_dynamic_init(&r, &scene, &lm, &dynamic_lighting)` after `r_build_scene` succeeds and before entering the frame loop.

- [ ] **Step 2: Run the build to verify the interface is RED**

Run: `c build`

Expected: FAIL because `DYNAMIC_LIGHTING` / `r_dynamic_init` are not defined yet.

- [ ] **Step 3: Implement the public types, object history, CPU spatial index, and generation queue**

In `dynamic.c`, build a uniform world-space cell index once from `LIGHTMAP.samples`:

- cell size = `max(0.5f, settings->gi_radius * 0.5f)`;
- allocate one `uint32_t sample_ids[lm->sample_count]` plus prefix ranges per occupied/grid cell;
- map each valid `LMAP_SAMPLE.position.xyz` to one cell;
- keep `uint32_t cell_generation[cell_count]`;
- dirty work is stored as cell jobs (`cell_index`, `generation`, `next_sample`) rather than one persistent queue node per texel;
- `dynamic_take_gi_jobs` expands current cell jobs into at most `settings.texels_per_frame` sample jobs per frame;
- stale cell jobs whose generation no longer equals `cell_generation[cell]` are discarded before consuming GPU budget;
- newly invalidated cells are pushed ahead of older work;
- no code path changes the budget based on object velocity or idle duration.

`dynamic_sync` compares every registered dynamic object's current transform with `previous_transform`. On change, compute old/new world AABBs, union them, expand by `gi_radius`, increment generations for overlapping cells, queue `DYNAMIC_CELL_UPDATE`s, queue fresh cell jobs, and then copy current to previous.

Treat zero scale only as invalid input for registered dynamic objects; return `false` from registration rather than silently collapsing or rewriting the transform. Update the existing static/light objects in `main.c` to explicit identity scale `{1,1,1}`.

- [ ] **Step 4: Add CPU regression checks for the Review Focus cases**

Use a temporary local test harness (not committed in the final branch) to construct a small synthetic `LIGHTMAP` and assert:

- one moved AABB invalidates only intersecting expanded cells;
- re-moving the object makes prior cell jobs stale;
- two invalidations of the same cell expose only newest-generation sample jobs;
- `dynamic_take_gi_jobs(..., capacity=17)` returns at most 17 jobs regardless of queue size;
- transformed bounds contain all eight transformed source AABB corners under rotation/non-uniform scale.

- [ ] **Step 5: Run CPU/build verification**

Run: `c build`

Expected: PASS.

Run the temporary CPU harness.

Expected: all assertions PASS.

- [ ] **Step 6: Commit**

```bash
git add build.c game.h gpu.h main.c render.c dynamic.c
git commit -m "Add dynamic lighting runtime state"
```

---

### Task 2: Split the permanent surface bake into indirect/base and direct sun

**Files:**
- Modify: `game.h`
- Modify: `cache.c`
- Modify: `render.c`
- Modify: `gpu.h`
- Modify: `gpu.c`
- Modify: `shaders/compute_base.hlsl`
- Modify: `shaders/fragment.hlsl`

**Interfaces:**
- `CACHED_LIGHTMAP` keeps `pixels` as **indirect/base RGBA16F** and adds `unsigned char *direct_pixels` as **direct-sun RGBA16F**.
- Cache format becomes `DM_CACHE_VERSION 9u`; both pixel blocks have the same `width * height * 8` byte size and both participate in `payload_hash`.
- `RENDERER::lightmap_texture` becomes the permanent indirect/base texture.
- `RENDERER::lightmap_direct` survives bake cleanup and becomes the permanent direct-sun texture.

- [ ] **Step 1: Write a failing cache/source regression check**

Before modifying cache code, add a temporary check that expects cache version 9, `direct_pixels`, and two lightmap textures to be uploaded on cache load.

- [ ] **Step 2: Run the check to verify RED**

Expected: FAIL on cache version 8 / missing `direct_pixels`.

- [ ] **Step 3: Separate sun direct from the base bake**

In `shaders/compute_base.hlsl`:

- `PHASE_DIRECT` calls `direct_sun(...)` only;
- `PHASE_TRACE` samples `direct_emissive(position, normal, seed) + trace_path_core(position, normal, seed, false)` so primary static emissive lighting remains in the immutable base while primary sun does not;
- secondary-bounce `direct_lighting` remains unchanged, so bounced sunlight is indirect and remains part of the base;
- stop using `PHASE_COMBINE` for the final stored layer;
- reconstruct/filter/dilate the indirect layer and direct-sun layer separately so both have valid chart padding for filtered sampling.

In `gpu.c`, preserve both final textures after `bake_lightmap` and read back both in `download_lightmap`.

- [ ] **Step 4: Upgrade cache serialization and load/reuse paths**

`cache_read_partial`, `cache_write`, `cache_free`, `r_load_cached_lightmap`, and `r_rebake_current_scene` must move/read/hash/reuse both pixel blocks atomically. Any v8 file fails version validation and triggers a normal rebake.

- [ ] **Step 5: Keep static rendering visually equivalent before dynamic shadows exist**

Change the surface material binding so `fragment.hlsl` samples `IndirectLightmap` and `DirectSunLightmap` and composes the old baked diffuse as:

```hlsl
float3 baked = indirect + direct_sun;
```

At this task boundary, dynamic visibility is exactly `1.0`; do not change static scene appearance intentionally.

- [ ] **Step 6: Verify**

Run: `c build`

Compile `BUILD_LIGHTMAP_CS` and `BUILD_SURFACE_FS` with the same SDL_shadercross/SPIR-V path used by Pony.

Expected: PASS.

Run one existing scene, rebake once, restart, and verify the v9 cache loads without rebaking and the static image is materially unchanged.

- [ ] **Step 7: Commit**

```bash
git add game.h cache.c render.c gpu.h gpu.c shaders/compute_base.hlsl shaders/fragment.hlsl
git commit -m "Separate baked indirect and direct sun"
```

---

### Task 3: Register/render dynamic models and trace transformed local BVHs

**Files:**
- Modify: `dynamic.c`
- Modify: `gpu.h`
- Modify: `gpu.c`
- Modify: `shaders/vertex.hlsl`
- Modify: `shaders/fragment.hlsl`
- Create: `shaders/dynamic.hlsl`

**Interfaces:**
- `r_add_dynamic_object` accepts only `object->state == DYNAMIC`, `object->type == MODEL`, and non-null `MODEL.geometry` / `MODEL.visual`.
- Dynamic model resources are deduplicated by `struct MODEL *`; multiple objects may instance one uploaded model/local BVH.
- Internal GPU instance record contains world matrix, inverse world matrix, inverse-transpose normal matrix, world AABB, local node/triangle offsets, and model resource index.
- Dynamic BVH buffers concatenate immutable model-local `BVH_NODE` and `BVH_TRIANGLE` arrays; transform changes update instance records only.

- [ ] **Step 1: Add a temporary registration smoke call and verify RED**

In a local-only smoke edit, create a `DYNAMIC` `OBJECT` backed by a loaded model and call `r_add_dynamic_object`; run `c build` before implementing GPU model creation.

Expected: FAIL at the missing/internal dynamic model GPU path.

- [ ] **Step 2: Implement deduplicated dynamic model resources**

Refactor only the reusable parts of current `upload_scene` needed to create a dynamic model resource without releasing the static scene. Each dynamic model resource owns:

- its vertex buffer (`RENDER_VERTEX` with lightmap UVs zeroed);
- material/texture resources;
- draw ranges;
- one CPU local BVH;
- concatenated GPU BVH offsets.

Do not introduce bindless or GPU-driven draw architecture here; the separate `nri` branch owns that optimization direction.

- [ ] **Step 3: Implement transform matrices and dynamic ray traversal**

In `dynamic.c`, compute column-major world, inverse-world, inverse-transpose-normal matrices and conservative world AABBs.

In `shaders/dynamic.hlsl`, implement:

```hlsl
bool dynamic_trace_any(TraceRay ray);
bool dynamic_trace_closest(TraceRay ray, out TraceHit hit);
```

For each candidate instance, world-AABB reject first, transform the ray origin/direction into model local space **without normalizing the transformed direction** so the affine ray parameter `t` remains comparable to the normalized world ray, traverse the local nodes using instance offsets, then inverse-transpose the selected normal to world space.

The first implementation may linearly scan dynamic instance AABBs; this is the small top-level structure. Keep the instance-record interface compatible with replacing that scan by a GPU TLAS/tree later.

- [ ] **Step 4: Render dynamic surfaces using transforms and probe indirect**

Extend the surface camera/object uniform data with `model`, `normal_model`, and `object_dynamic`.

`vertex.hlsl` transforms positions/normals for dynamic draws.

`fragment.hlsl` uses SH9 volume probes for dynamic indirect irradiance; use cosine convolution factors by band (`l0 = PI`, `l1 = 2*PI/3`, `l2 = PI/4`) before Lambertian diffuse evaluation. Static objects continue to use lightmaps.

- [ ] **Step 5: Verify transformed geometry and traversal**

Run: `c build`

Compile `BUILD_SURFACE_VS`, `BUILD_SURFACE_FS`, and the dynamic shader variant(s).

Use a temporary runtime smoke object with translation + rotation + non-uniform scale. Verify its raster position matches its dynamic BVH bounds/hits and that removing it releases its model resource only after the final instance is gone.

- [ ] **Step 6: Commit**

```bash
git add dynamic.c gpu.h gpu.c shaders/vertex.hlsl shaders/fragment.hlsl shaders/dynamic.hlsl
git commit -m "Render and trace dynamic models"
```

---

### Task 4: Dynamic-only directional sun shadow map

**Files:**
- Modify: `gpu.h`
- Modify: `gpu.c`
- Modify: `shaders/dynamic.hlsl`
- Modify: `shaders/fragment.hlsl`

**Interfaces:**
- `RENDERER` gains a depth-only `dynamic_shadow_texture`, comparison/sampling descriptor, dynamic-shadow pipeline/layout, and size from `DYNAMIC_LIGHTING.shadow_map_size`.
- Shadow projection covers the existing baked beam-grid sun-space prism, so static beam visibility and dynamic shadow visibility use the same directional-light basis/bounds.
- Dynamic shadow map contains **dynamic geometry only**.

- [ ] **Step 1: Add shader-side dynamic visibility sampling before the resource exists**

Temporarily wire `dynamic_shadow_visibility(world_position)` into the static direct term and run shader compilation.

Expected: FAIL because the dynamic shadow texture/uniforms are not bound yet.

- [ ] **Step 2: Create depth-only shadow resources and pipeline**

Create/recreate the shadow texture when `shadow_map_size` changes. Record a depth-only pass every rendered frame when at least one dynamic model exists; do not include static geometry.

The shadow vertex shader transforms the dynamic world position into the baked beam-grid sun-space bounds. Use the configured `shadow_bias` at comparison time.

- [ ] **Step 3: Apply combined static and dynamic direct visibility**

In `fragment.hlsl`:

- compute static sun visibility at `world_position` from the existing beam blocker grid;
- compute dynamic visibility from the dynamic shadow map (3x3 PCF is sufficient);
- static diffuse direct = `DirectSunLightmap * dynamic_visibility` (static visibility is already encoded in the direct lightmap);
- dynamic direct diffuse/specular = realtime BRDF sun * `static_beam_visibility * dynamic_visibility`;
- static direct specular uses `static_beam_visibility * dynamic_visibility` instead of the old baked-luma heuristic.

Never multiply baked indirect/base by dynamic shadow visibility.

- [ ] **Step 4: Verify immediate moving shadows**

Run shader compilation and `c build`.

With a temporary animated dynamic object, verify in the same frame that its shadow moves across a static lightmapped wall/floor and that moving it away immediately restores the baked direct term underneath.

Verify indirect/base illumination does not darken when only direct sun is blocked.

- [ ] **Step 5: Commit**

```bash
git add gpu.h gpu.c shaders/dynamic.hlsl shaders/fragment.hlsl
git commit -m "Add dynamic sun shadows"
```

---

### Task 5: Cell-generation invalidation and transient GI overlay resources

**Files:**
- Modify: `dynamic.c`
- Modify: `gpu.h`
- Modify: `gpu.c`
- Modify: `shaders/dynamic.hlsl`
- Modify: `shaders/fragment.hlsl`

**Interfaces:**
- Runtime overlay texture: RGBA16F, lightmap dimensions; RGB = running radiance estimate, A = accumulated sample count.
- Runtime overlay-generation texture: R32_UINT, lightmap dimensions; value = cell generation used to produce that pixel.
- GPU cell-generation buffer: one `uint` per spatial cell.
- Cell update pass consumes `DYNAMIC_CELL_UPDATE[]` and writes new generation values before any surface draw/GI result can use the old cell.
- Surface shader computes the sample's world cell from `world_position`; overlay is valid only when `overlay_generation(lightmap_uv) == CellGenerations[cell]`.

- [ ] **Step 1: Write a failing overlay-validity shader contract**

Require the static surface path to ignore an overlay pixel whose stored generation differs from its current world-cell generation.

Expected before implementation: shader/resource binding check FAIL.

- [ ] **Step 2: Create overlay/generation/cell resources**

Allocate them in `r_dynamic_init`; release/recreate when the static lightmap layout changes. Upload initial cell generations as zero.

Add `GPU_STORAGE_R32UI` to Pony's generated shader binding prefix for the R32_UINT storage texture.

- [ ] **Step 3: Record immediate cell-generation updates**

Each frame, after `dynamic_sync` and before GI/surface lighting, upload only pending `DYNAMIC_CELL_UPDATE`s and dispatch a tiny compute update. Add the precise compute-to-fragment/compute barriers needed for the changed resources; do not add `QueueWaitIdle` or a new global synchronization policy.

This update is not constrained by `texels_per_frame` because it changes one uint per dirty spatial cell, not one operation per lightmap texel.

- [ ] **Step 4: Compose the overlay on static receivers**

When generation matches and sample count > 0:

```hlsl
confidence = saturate(sample_count / target_samples);
indirect = lerp(base_indirect, overlay_radiance, confidence);
```

When generation differs, immediately use `base_indirect` with no stale contribution.

- [ ] **Step 5: Verify local invalidation**

Move a test object twice before prior work converges. Verify old-region pixels immediately fall back to base and stale GPU jobs cannot restore them because their generation is obsolete.

Move a second object in a disjoint region and verify unrelated cell generations remain unchanged.

- [ ] **Step 6: Commit**

```bash
git add dynamic.c gpu.h gpu.c shaders/dynamic.hlsl shaders/fragment.hlsl
git commit -m "Add transient dynamic GI overlay"
```

---

### Task 6: Fixed-budget runtime GI retracing against static + dynamic geometry

**Files:**
- Modify: `dynamic.c`
- Modify: `gpu.h`
- Modify: `gpu.c`
- Modify: `shaders/dynamic.hlsl`

**Interfaces:**
- `dynamic_take_gi_jobs` returns no more than `DYNAMIC_LIGHTING.texels_per_frame` current-generation jobs.
- GPU dispatch size is exactly the returned job count; shader loops exactly `rays_per_texel` per job.
- Runtime tracer uses at most 2 bounces in this first implementation.
- Rays choose the nearest hit across the existing static BVH and transformed dynamic BVHs.
- Primary static emissive lighting remains eligible and dynamic geometry can block those shadow rays; dynamic-emissive importance sampling is deferred.
- New overlay result is committed only if the job generation still equals the current GPU cell generation.

- [ ] **Step 1: Add a failing GI job shader test**

Create the dispatch/binding contract for `DYNAMIC_GI_JOB`, static/dynamic BVHs, probes, overlay textures, and cell generations; compile before implementing `dynamic_gi_cs`.

Expected: FAIL due missing entry point/resources.

- [ ] **Step 2: Implement union tracing and local path evaluation**

`dynamic_gi_cs` retrieves the `LMAP_SAMPLE` by `job.sample_index`, validates its cell generation, then traces `rays_per_texel` samples.

For each trace, choose the closest of static and dynamic geometry. Match current bake material encoding (`albedo` stored in triangle `.w` values), sky model, static emissive direct sampling, sun direct at secondary bounces, and probe reuse where practical. Keep max runtime bounce count = 2.

- [ ] **Step 3: Accumulate a running mean without changing the fixed budget**

Read overlay RGB/sample count only if generation matches. Add at most `rays_per_texel` samples, write new mean/count and the job generation. Once count reaches `target_samples`, the cell job can advance/finish naturally; do **not** increase samples/dispatch because all objects are stationary.

- [ ] **Step 4: Make movement priority explicit**

CPU cell jobs are LIFO/newest-first. A newly invalidated cell gets a new generation and fresh cursor; older jobs remain cheap stale entries that are discarded before creating GPU jobs. If a cell is invalidated repeatedly, only its newest generation may consume GPU ray budget.

- [ ] **Step 5: Verify the hard frame-work ceiling**

Instrument a temporary counter and create a dirty region larger than 100,000 samples.

For 120 consecutive moving and stationary frames assert:

```text
GPU GI jobs <= texels_per_frame
rays launched <= texels_per_frame * rays_per_texel
```

Verify stationary frames never switch to a larger target, batch, or dispatch size. Remove the instrumentation afterward.

- [ ] **Step 6: Verify visible local GI correction**

Use a moving opaque object near a bright static surface/emissive region. Confirm nearby static lightmap texels progressively change, far-away cells remain on the permanent bake, and moving the object again immediately invalidates the prior overlay before convergence restarts under the same budget.

- [ ] **Step 7: Commit**

```bash
git add dynamic.c gpu.h gpu.c shaders/dynamic.hlsl
git commit -m "Retrace dynamic local GI within fixed budget"
```

---

### Task 7: Frame integration, lifecycle, regression verification, and clean handoff

**Files:**
- Modify: `render.c`
- Modify: `gpu.c`
- Modify: `gpu.h`
- Modify: `main.c`
- Possibly modify: only files above if verification exposes a defect

**Interfaces:**
- Per-frame order is fixed:
  1. synchronize dynamic transforms / invalidate cells;
  2. record cell-generation updates;
  3. update dynamic instance buffer;
  4. render dynamic-only shadow map;
  5. dispatch at most the fixed GI job budget;
  6. render static + dynamic surfaces;
  7. existing volumetrics/post/present.
- `r_deinit` frees dynamic model resources, local BVHs, cell index/queues, shadow resources, overlay/generation resources, and buffers exactly once.

- [ ] **Step 1: Add final integration assertions/guards**

Guard no-dynamic-object scenes, zero dirty jobs, removed objects, cache reload, and resize paths. `r_dynamic_init` failure must leave renderer deinit-safe.

- [ ] **Step 2: Run complete build and shader verification**

Run: `c build`

Compile all affected shader variants:

- `BUILD_LIGHTMAP_CS`
- `BUILD_SURFACE_VS`
- `BUILD_SURFACE_FS`
- dynamic shadow variant
- dynamic cell-update variant
- dynamic GI variant
- existing vision volume/compose variants

Expected: all PASS.

- [ ] **Step 3: Static-scene regression**

Run an existing scene with zero registered dynamic objects from a freshly generated v9 bake and from cache reload.

Expected:

- no dynamic allocations/jobs beyond initialized lightweight state;
- no visible loss of static direct/indirect lighting;
- existing volumetrics still function;
- no new bake or frame-time synchronization waits were introduced.

- [ ] **Step 4: Dynamic end-to-end smoke**

Using a temporary, uncommitted main/test edit, register a dynamic model and continuously translate/rotate it.

Verify:

- object itself receives probe indirect + realtime sun;
- static world receives its immediate sun shadow;
- local GI overlay converges while the budget remains fixed;
- changing direction/speed does not increase the GI dispatch ceiling;
- moving again invalidates stale local GI immediately;
- removing the object restores base/static lighting and releases resources.

- [ ] **Step 5: Check the `nri` merge seam without merging it**

Compare against current `nri` branch interfaces. Confirm dynamic GI exposes a contiguous job buffer/count and recording boundary that can later map to persistent command contexts, indirect dispatch, wave compaction, and wavefront queues. Do not copy its scheduler or submission overhaul into this branch.

- [ ] **Step 6: Clean the implementation branch**

Remove all temporary tests, CI/workflow files, debug counters, and local smoke code. The plan/spec remain only on `dynamic-lighting-design`, not `static-dynamic-geom`.

Run:

```bash
git status --short
git diff --check
```

Expected: clean status after commit; no whitespace errors.

- [ ] **Step 7: Whole-branch verification and final commit cleanup**

Compare `static-dynamic-geom` implementation head against its pre-feature base. Only intended C/HLSL/build files may differ. If implementation used temporary/reviewer commits, squash them into coherent source-only commits before handing the branch back.

Final verification evidence must include the exact `c build` result, affected shader compile results, fixed-budget counter result, and the static/dynamic runtime smoke observations.
