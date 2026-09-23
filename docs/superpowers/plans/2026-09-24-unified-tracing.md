# Unified Tracing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Dustmite's duplicated bake-time geometry intersection logic with one shared tracing model while preserving the rasterized sun-depth fast path and keeping runtime free of triangle tracing.

**Architecture:** CPU and GPU use the same ray semantics over the existing threaded BVH: specialized any-hit and closest-hit queries, with path transport layered above them. Lightmap and SH-probe baking use the GPU query layer; the sun-beam builder keeps its rasterized orthographic depth classification and uses the CPU query layer only for ambiguous cells.

**Tech Stack:** C11, HLSL, SDL3, SDL_GPU, SDL_shadercross, existing threaded BVH and c-buildsystem.

**Spec:** `docs/superpowers/specs/2026-09-23-unified-tracing-design.md`

## Global Constraints

- Preserve the existing threaded BVH layout (`meta = left, next, first triangle, triangle count`).
- Preserve rasterized sun-depth classification in `beam.c`; do not brute-force every beam voxel.
- Keep `trace_any` specialized and early-exiting; do not route visibility rays through closest-hit payload work.
- Keep `trace_closest` specialized for hit distance/normal/albedo payloads.
- Keep path tracing as transport built on `trace_any` and `trace_closest`, not as a separate geometry-intersection implementation.
- Keep runtime `compute.hlsl` volume rendering free of triangle/BVH tracing.
- Do not change bake file format, cache version, BVH layout, sample counts, probe layout, beam layout, or runtime draw architecture unless required by a demonstrated correctness bug.
- Do not add dependencies.
- Preserve current visual output within stochastic bake variance.
- Work only on branch `unified-tracing`.
- Minimize history: use checkpoints without intermediate commits and produce one final squashed implementation commit before opening a PR.

## Review Focus

- Rays parallel to one or more AABB axes must agree between CPU and GPU and must not generate NaNs or false hits.
- Self-intersection handling must reject hits at or below `tmin` consistently in CPU and GPU queries.
- Finite `tmax` must reject triangles beyond the caller's requested segment, especially probe-validity rays.
- Threaded skip pointers must produce the same any-hit/closest-hit result after refactoring as the current traversal.
- Sun-depth coherent tiles must still skip exact tracing; ambiguous silhouettes/openings must still fall back to the shared CPU any-hit query.

---

### Task 1: Define and implement the shared CPU trace API

**Files:**
- Modify: `dustmite.h` in the BVH declarations section.
- Modify: `bvh.c` around the BVH helpers and after `bvh_free`/before `bvh_build` as appropriate.

**Interfaces:**
- Produces: `dm_trace_ray`, containing `vec3 origin`, `float tmin`, `vec3 direction`, `float tmax`.
- Produces: `dm_trace_hit`, containing `float t`, `vec3 normal`, `vec3 albedo`, and `uint32_t triangle`.
- Produces: `bool dm_trace_any(const bvh *tree, dm_trace_ray ray)`.
- Produces: `bool dm_trace_closest(const bvh *tree, dm_trace_ray ray, dm_trace_hit *hit)`.
- Consumes: existing `bvh_node`, `bvh_triangle`, threaded `meta[1]` skip pointers, and `BVH_EPSILON`-compatible numeric rules.

- [ ] **Step 1: Add the shared ray/hit declarations to `dustmite.h`**

Use this public shape so `beam.c` and future CPU tools call the BVH through one query interface:

```c
typedef struct dm_trace_ray {
    vec3 origin;
    float tmin;
    vec3 direction;
    float tmax;
} dm_trace_ray;

typedef struct dm_trace_hit {
    float t;
    vec3 normal;
    vec3 albedo;
    uint32_t triangle;
} dm_trace_hit;

bool dm_trace_any(const bvh *tree, dm_trace_ray ray);
bool dm_trace_closest(const bvh *tree, dm_trace_ray ray, dm_trace_hit *hit);
```

- [ ] **Step 2: Implement a CPU AABB helper with explicit ray bounds**

Implement a private helper in `bvh.c` equivalent to:

```c
static bool trace_box(dm_trace_ray ray, const bvh_node *node, float max_t) {
    float lo = ray.tmin;
    float hi = fminf(ray.tmax, max_t);
    const float o[3] = {ray.origin.x, ray.origin.y, ray.origin.z};
    const float d[3] = {ray.direction.x, ray.direction.y, ray.direction.z};

    for (uint32_t axis = 0; axis < 3u; ++axis) {
        if (fabsf(d[axis]) < 1.0e-7f) {
            if (o[axis] < node->min[axis] || o[axis] > node->max[axis]) return false;
            continue;
        }
        const float inv = 1.0f / d[axis];
        float a = (node->min[axis] - o[axis]) * inv;
        float b = (node->max[axis] - o[axis]) * inv;
        if (a > b) { const float t = a; a = b; b = t; }
        lo = fmaxf(lo, a);
        hi = fminf(hi, b);
        if (lo > hi) return false;
    }
    return hi >= ray.tmin;
}
```

- [ ] **Step 3: Implement a CPU triangle helper with identical `tmin`/`tmax` acceptance**

Use Möller-Trumbore over `bvh_triangle.a/b/c`, rejecting degenerate determinants, barycentrics outside the triangle, `t <= ray.tmin`, and `t >= max_t`.

```c
static bool trace_triangle(dm_trace_ray ray, const bvh_triangle *tri,
                           float max_t, float *hit_t) {
    const vec3 a = v3(tri->a[0], tri->a[1], tri->a[2]);
    const vec3 e1 = v3_sub(v3(tri->b[0], tri->b[1], tri->b[2]), a);
    const vec3 e2 = v3_sub(v3(tri->c[0], tri->c[1], tri->c[2]), a);
    const vec3 p = v3_cross(ray.direction, e2);
    const float det = v3_dot(e1, p);
    if (fabsf(det) < 1.0e-7f) return false;
    const float inv_det = 1.0f / det;
    const vec3 s = v3_sub(ray.origin, a);
    const float u = v3_dot(s, p) * inv_det;
    if (u < 0.0f || u > 1.0f) return false;
    const vec3 q = v3_cross(s, e1);
    const float v = v3_dot(ray.direction, q) * inv_det;
    if (v < 0.0f || u + v > 1.0f) return false;
    const float t = v3_dot(e2, q) * inv_det;
    if (t <= ray.tmin || t >= fminf(ray.tmax, max_t)) return false;
    *hit_t = t;
    return true;
}
```

- [ ] **Step 4: Implement `dm_trace_any` as the early-exit threaded traversal**

Start at node zero; skip with `meta[1]` on AABB miss, descend with `meta[0]` for interior nodes, and immediately return `true` on the first accepted triangle. Return `false` when the threaded traversal reaches `UINT32_MAX`.

- [ ] **Step 5: Implement `dm_trace_closest` with closest-distance pruning**

Track `closest = ray.tmax`. Use `closest` as the AABB/triangle maximum. On a hit, fill `t`, triangle index, normal from `tri->normal.xyz`, and albedo from `tri->a.w/b.w/c.w`. Orient the returned normal against the ray direction exactly as the GPU query does:

```c
if (v3_dot(normal, ray.direction) > 0.0f)
    normal = v3_scale(normal, -1.0f);
```

- [ ] **Step 6: Compile-check the CPU API before changing callers**

Run the normal project build. Expected result: no warnings/errors from the new declarations or `bvh.c`; existing behavior is unchanged because no caller has migrated yet.

- [ ] **Checkpoint:** Do not commit; keep the branch dirty for the final single-commit squash.

---

### Task 2: Migrate sun-beam exact fallback to the shared CPU tracer

**Files:**
- Modify: `beam.c` at the local `intersects_box`, `intersects_triangle`, `shaded`, and ambiguous-cell fallback in `dm_beam_build`.

**Interfaces:**
- Consumes: `dm_trace_ray` and `dm_trace_any` from Task 1.
- Produces: unchanged `dm_beam_grid`, `shadow_depth`, compressed beam cells, and exact-ray count logging.

- [ ] **Step 1: Replace the local exact visibility path**

Delete active use of the local `intersects_box`, `intersects_triangle`, and `shaded` implementation. Preserve a disabled reference copy under `#if 0` only if useful for regression comparison; the production path must call the shared API.

- [ ] **Step 2: Convert ambiguous-cell fallback to `dm_trace_any`**

At the existing ambiguous tile branch, construct:

```c
const dm_trace_ray ray = {
    .origin = p,
    .tmin = 0.001f,
    .direction = sun,
    .tmax = 1.0e20f
};
visible = !dm_trace_any(tree, ray);
```

Keep the existing `raster_depth`, `depth_tile`, `BEAM_TILE`, and `traced` counter logic unchanged.

- [ ] **Step 3: Verify the rasterized depth fast path remains active**

Run a bake on `poolroom.glb`. The log must still contain the sun-depth exact-ray count in the form:

```text
B: sun depth pass selected <traced>/<sample_count> exact visibility rays
```

Expected: `<traced>` is strictly less than `<sample_count>` for the poolroom; if all voxels trace, the depth-map workflow has regressed.

- [ ] **Step 4: Compare beam compression sanity**

Record the resulting `B: compressed sun beams into <count> cells` value and verify the bake completes without a zero-cell result or an order-of-magnitude unexplained explosion in cell count.

- [ ] **Checkpoint:** Do not commit.

---

### Task 3: Refactor the GPU bake tracer into the same query model

**Files:**
- Modify: `shaders/compute_base.hlsl` in the `BUILD_LIGHTMAP_CS || BUILD_PROBE_CS` section.
- Modify: `render.c` only if uniform names/signatures must change; do not otherwise restructure rendering.

**Interfaces:**
- Produces HLSL `TraceRay` and `TraceHit` structures matching the CPU semantics.
- Produces `trace_ray_box`, `trace_ray_triangle`, `trace_any`, and `trace_closest`.
- Consumes those queries from `direct_sun`, `trace_path`, lightmap bake, probe validity, probe sample tracing, and probe sun-visibility tests.

- [ ] **Step 1: Add the HLSL ray/hit structs**

Use:

```hlsl
struct TraceRay
{
    float3 origin;
    float tmin;
    float3 direction;
    float tmax;
};

struct TraceHit
{
    float t;
    float3 normal;
    float3 albedo;
    uint triangle;
};
```

- [ ] **Step 2: Refactor primitive helpers to consume `TraceRay`**

Rename/refactor the current `ray_box` and `ray_triangle` into `trace_ray_box` and `trace_ray_triangle`. Initialize slab `tmin` from `ray.tmin`, clamp `tmax` against the caller-provided pruning distance, and apply the same `t <= ray.tmin` / `t >= min(ray.tmax, max_t)` triangle rule as the CPU backend.

- [ ] **Step 3: Refactor `occluded` into specialized `trace_any`**

Keep the current threaded traversal and immediate return on first accepted triangle. Do not allocate or populate a `TraceHit` in this function.

- [ ] **Step 4: Refactor `trace_closest` to return a `TraceHit`**

Keep closest-distance pruning. Populate triangle index, albedo, hit distance, and normal. Keep the current rule that flips the hit normal when `dot(normal, ray.direction) > 0`.

- [ ] **Step 5: Migrate direct-sun visibility**

Replace the old `occluded(origin, direction, max_t)` call with a `TraceRay` using `tmin = bake_params.x` and `tmax = 1.0e20f`, then call `trace_any`.

- [ ] **Step 6: Migrate `trace_path` bounce intersections**

For each bounce, build a `TraceRay` from `position + normal * bake_params.x`, set `tmin = bake_params.x`, `tmax = 1.0e20f`, call `trace_closest`, and consume `TraceHit.t`, `.normal`, and `.albedo`. Keep bounce count, sky evaluation, throughput update, direct-sun sampling, and RNG unchanged.

- [ ] **Step 7: Migrate probe validity rays**

For each six-axis probe validity test, use `TraceRay{ origin=input.xyz, tmin=bake_params.x, direction=axis, tmax=0.15f }` and `trace_closest`. This pins finite-segment behavior to the shared API.

- [ ] **Step 8: Migrate probe radiance rays and sun visibility**

Use `trace_closest` for sphere directions and `trace_any` for the probe's direct-sun visibility bit. Keep SH projection/reduction unchanged.

- [ ] **Step 9: Shader compile/build check**

Build the project so SDL_shadercross compiles both `BUILD_LIGHTMAP_CS` and `BUILD_PROBE_CS` paths. Expected: no HLSL reflection or shader compile errors.

- [ ] **Checkpoint:** Do not commit.

---

### Task 4: Validate bake/runtime behavior and finish as one commit

**Files:**
- Verify: `dustmite.h`, `bvh.c`, `beam.c`, `shaders/compute_base.hlsl`.
- Verify unchanged unless necessary: `render.c`, `shaders/compute.hlsl`, `cache.c`.

**Interfaces:**
- Consumes all changes from Tasks 1-3.
- Produces a branch whose public behavior and baked-data format remain compatible with `Dustmite`.

- [ ] **Step 1: Run a clean build with strict warnings**

Use the repository's normal c-buildsystem command. Expected: successful build with no new warnings.

- [ ] **Step 2: Run the poolroom without rebaking**

Launch the renderer with `poolroom.glb`. Expected: existing cached bake loads when hashes match, normal raster rendering works, F5 fog toggle works, and no new runtime tracing dispatch/log appears.

- [ ] **Step 3: Run a full poolroom rebake**

Press `B` and let all stages finish. Required stages:

```text
scene geometry
surface lightmap
object probes
volume probes
sun visibility
saving cache
```

Expected: no failure stage, no shader error, and `.baked` saves successfully.

- [ ] **Step 4: Validate the sun fast path after full bake**

Confirm the exact-ray count remains below total beam samples and the beam compressor emits a nonzero set of cells. Visually inspect the poolroom shafts for gross loss of openings/silhouettes.

- [ ] **Step 5: Confirm runtime volume shader still has no triangle tracing**

Inspect `shaders/compute.hlsl` and ensure the `BUILD_VOLUME_CS` path consumes `SunBeams`/`VolumeProbes` only; no BVH node/triangle buffers or `trace_any`/`trace_closest` calls are added there.

- [ ] **Step 6: Diff audit**

Compare branch `unified-tracing` against `Dustmite`. Expected functional file scope: `dustmite.h`, `bvh.c`, `beam.c`, `shaders/compute_base.hlsl`, plus `render.c` only if mechanically necessary. Reject unrelated formatting or renderer changes.

- [ ] **Step 7: Produce one final commit**

Squash/amend all implementation and plan changes on `unified-tracing` into one commit before opening the PR. Suggested message:

```text
Unify bake tracing backend
```

Do not merge or open the PR until the final branch has been reviewed.
