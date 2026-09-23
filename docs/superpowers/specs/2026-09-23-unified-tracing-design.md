# Unified Tracing System Design

## Goal

Replace Dustmite's duplicated bake-time geometry tracing paths with one clean tracing architecture while preserving the existing rasterized sun-depth workflow where rasterization is faster.

The runtime renderer remains rasterized and consumes baked lightmaps, SH probes, and compressed sun-beam data. This change does not add per-frame triangle ray tracing.

## Current State

Dustmite currently has two related but separate tracing implementations:

- GPU bake shaders implement `ray_box`, `ray_triangle`, `trace_closest`, `occluded`, and `trace_path` over the threaded BVH.
- `beam.c` has CPU-side `intersects_box`, `intersects_triangle`, and `shaded` helpers for exact sun-visibility fallback after a rasterized orthographic depth classification pass.

The sun-beam builder already uses the correct high-level optimization: rasterize an orthographic depth field from the sun, classify coherent tiles cheaply, and ray-test only ambiguous silhouettes/openings.

## Design

### 1. Shared trace model

Use one query model everywhere:

- `TraceRay`: origin, direction, minimum distance, maximum distance.
- `TraceHit`: hit distance, normal, albedo/material payload where required.
- `trace_any`: returns on the first accepted triangle hit.
- `trace_closest`: traverses until the closest accepted hit is known.

Path tracing is not a separate intersection system. It is an integrator layered on `trace_any` and `trace_closest`.

### 2. GPU trace backend

The GPU implementation remains the primary tracing backend for bake workloads. The existing threaded BVH representation remains initially because it is compact, stackless, and already used by the lightmap/probe shaders.

Refactor the existing HLSL helpers so lightmap tracing and probe tracing call the same common query functions and use the same ray bounds/epsilon rules.

The new layer should separate:

1. primitive intersection (`trace_ray_box`, `trace_ray_triangle`),
2. BVH queries (`trace_any`, `trace_closest`),
3. transport (`trace_path`, direct-sun evaluation, SH projection).

No generic mode-switching traversal is added; `trace_any` and `trace_closest` stay specialized so fast visibility rays do not pay for closest-hit payload work.

### 3. CPU reference/query backend

`beam.c` still needs CPU geometry queries for the ambiguous-cell fallback in the existing rasterized sun-depth pipeline. Instead of maintaining semantically different ad-hoc tests, give it CPU equivalents of the same trace API and acceptance rules:

- same epsilon convention,
- same AABB slab behavior,
- same triangle acceptance,
- same maximum-distance semantics,
- same threaded-BVH traversal semantics.

The CPU implementation is a backend of the same query model, not a separate tracing design.

### 4. Sun visibility remains hybrid raster + trace

Keep the existing fast path:

1. rasterize first sun-facing depth per sun-space column,
2. classify coherent `BEAM_TILE` regions from the depth field,
3. accept obvious fully lit/shaded samples without tracing,
4. use the shared `trace_any` CPU backend only for ambiguous silhouettes/openings,
5. compress resulting visibility into beam cells as before.

Do not replace this with brute-force ray tracing. Rasterization remains the acceleration path because it cheaply resolves large coherent regions.

### 5. Lightmap path tracing

The lightmap compute shader uses the shared GPU query layer:

- direct sun -> `trace_any`,
- bounce intersection -> `trace_closest`,
- multi-bounce integrator -> `trace_path` built only from those queries.

Progressive accumulation, filtering, dilation, sample counts, and output format stay unchanged.

### 6. Probe path tracing

SH probes use exactly the same GPU query layer as lightmaps:

- probe validity rays -> `trace_closest` with a short maximum distance,
- environment/scene samples -> `trace_closest`,
- bounced radiance -> shared `trace_path`,
- direct sun visibility -> `trace_any`.

SH basis projection and reduction remain unchanged.

### 7. Beam runtime

The runtime volume shader does not perform geometry tracing. It continues to consume baked beam visibility/depth data and performs DDA/analytic integration only.

This preserves the low-end-hardware target.

## Files

Expected implementation changes:

- `dustmite.h`: shared CPU trace ray/hit declarations if needed by multiple C translation units.
- `bvh.c`: CPU trace-query implementation over the existing threaded BVH.
- `beam.c`: remove local duplicate intersection/traversal helpers and call the shared CPU trace API while preserving `raster_depth`, `depth_tile`, and compression.
- `shaders/compute_base.hlsl`: reorganize primitive intersection, any-hit, closest-hit, and path integration into one common GPU trace layer used by both lightmap and probe builds.
- `render.c`: only if names/signatures need to change while binding the same BVH buffers; no renderer architecture change.

`compute.hlsl` volume runtime code should remain free of triangle tracing.

## Performance Constraints

- Preserve early exit for any-hit queries.
- Do not force closest-hit payload work on visibility rays.
- Preserve rasterized sun-depth classification.
- No runtime BVH traversal.
- No new allocations in hot trace loops.
- No new GPU/CPU synchronization in the bake path.
- Keep BVH data layout compatible unless a measured reason requires changing it.

## Correctness Constraints

The CPU and GPU tracing backends must agree on:

- parallel-ray AABB handling,
- triangle edge acceptance,
- self-intersection epsilon,
- `tmin`/`tmax` bounds,
- threaded skip-pointer traversal,
- normal orientation behavior where a closest-hit payload is requested.

## Validation

After implementation:

1. build the project with the existing build system,
2. verify the same GLB scene loads and renders,
3. perform a full bake,
4. verify lightmap/probe bake stages complete,
5. confirm the sun-depth pass still reports a reduced number of exact visibility rays rather than tracing every voxel,
6. compare beam counts and visible shafts before/after for gross regressions,
7. confirm normal runtime rendering performs no BVH trace dispatches.

## Non-Goals

- hardware RT/DXR/Vulkan ray tracing,
- replacing the threaded BVH with SAH/BVH4 in this change,
- runtime path tracing,
- replacing volumetric DDA with triangle tracing,
- removing the rasterized sun-depth fast path.
