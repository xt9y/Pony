# Dynamic Lighting Architecture

## Goal

Add realtime support for `DYNAMIC` objects while keeping Pony's existing static bake as the immutable baseline.

A dynamic object must:

- move without entering the permanent lightmap bake;
- receive baked indirect lighting from volume probes;
- cast realtime direct shadows onto static and dynamic receivers;
- alter nearby indirect lighting on static lightmapped surfaces through a temporary runtime lightmap overlay;
- never trigger an unbounded or idle-time bake burst;
- immediately invalidate stale lighting when it moves again.

The permanent `.baked` data always describes the scene with dynamic objects absent from static lighting.

## Constraints

- Portable to the current M2/NRI Vulkan/MoltenVK path. No NRI ray tracing, SER, micromaps, or vendor-specific implementation is required.
- The current software BVH tracer remains the correctness fallback.
- Runtime dynamic-lighting cost is capped by a fixed per-frame work budget. Standing still never increases the budget.
- New movement supersedes obsolete queued work from older transforms.
- The implementation on `static-dynamic-geom` must not depend on unmerged `nri` branch work, but its queue interfaces should be compatible with later reuse of the `nri` branch's persistent command contexts, indirect dispatch, wave compaction, and wavefront tracing.

## Object and Transform Model

`OBJECT_STATE` remains the only public classification: `STATIC` or `DYNAMIC`.

`OBJECT` gains transform/history data:

```c
typedef struct TRANSFORM {
    VEC3 position;
    VEC3 rotation;
    VEC3 scale;
} TRANSFORM;

typedef struct OBJECT {
    OBJECT_STATE state;
    OBJECT_TYPE type;
    TRANSFORM transform;
    TRANSFORM previous_transform;
    void *data;
} OBJECT;
```

A transform change on a dynamic object is the authoritative movement signal. Rendered depth differences are not used to detect movement.

Each model owns immutable local-space geometry and a local-space BVH. Moving an object changes only its transform and world bounds; its triangle BVH is not rebuilt.

## Static and Dynamic Geometry

### Static models

Static models:

- participate in the permanent surface-lightmap bake;
- participate in the permanent volume-probe bake;
- participate in the static BVH;
- receive indirect lighting from the baked lightmap;
- receive baked/static direct light plus realtime dynamic shadow modulation.

### Dynamic models

Dynamic models:

- are excluded from permanent lightmap geometry;
- are excluded from permanent static lighting contributions;
- keep local-space BVHs that can be queried after transforming rays into model space;
- receive baked indirect lighting from volume probes;
- receive realtime direct sun lighting;
- receive static sun visibility from Pony's existing static visibility data and realtime dynamic visibility from the dynamic shadow path.

## Dynamic Acceleration Structure

Pony keeps the existing static scene BVH unchanged.

Dynamic objects are addressed through a small top-level structure containing their world AABBs. Each leaf references a model-local BVH and object transform.

A dynamic ray query is:

1. test the world ray against the dynamic-object top-level AABBs;
2. transform candidate rays into model-local space;
3. traverse the existing local software BVH;
4. transform the hit result back to world space;
5. select the nearest hit between the static scene BVH and all dynamic candidates.

The top-level dynamic structure is rebuilt or refit from object bounds when transforms change. Triangle BLAS data remains immutable.

The interface should be shaped so the later `nri` branch can replace queue execution and traversal scheduling without changing the dynamic-lighting ownership model.

## Permanent Bake Separation

The permanent lightmap must preserve the direct sun component separately from the baked indirect/base component.

The preferred representation is:

```text
base_indirect_lightmap
static_direct_sun_lightmap
```

The final static surface lighting is:

```text
base_indirect
+ static_direct_sun * dynamic_sun_visibility
+ runtime_indirect_overlay
```

If retaining the existing combined lightmap is materially simpler, Pony may store both the combined base and a separate direct-sun texture and reconstruct equivalent results by subtracting/reapplying the direct term. The implementation must not multiply the entire combined bake by dynamic shadow visibility, because that would incorrectly shadow baked indirect light.

Changing the cache representation invalidates older `.baked` files through the existing bake/cache hash or format versioning path.

## Immediate Direct Shadows

The first realtime direct-light implementation targets the existing directional sun.

Every frame, dynamic models only are rasterized into a dynamic directional shadow map. Static geometry is omitted because its sun visibility is already represented in the static bake/beam data.

For static receivers:

```text
sun_direct = static_direct_sun_lightmap * dynamic_shadow_visibility
```

For dynamic receivers:

```text
sun_direct = BRDF_sun * static_sun_visibility(world_position) * dynamic_shadow_visibility
```

The existing sun visibility/beam representation supplies static-scene visibility for dynamic receivers. The dynamic-only shadow map supplies moving-blocker visibility for both static and dynamic receivers.

Point and spot dynamic-shadow implementations are explicitly deferred, but the dirty-lighting architecture must not assume the sun is the only light forever.

## Runtime Indirect-Light Overlay

Dynamic-object influence on static indirect lighting is stored in a transient overlay, never written into `.baked`.

The overlay maintains, per lightmap texel or compact lightmap sample:

```text
radiance accumulator
sample count
valid generation
```

When valid, the static indirect term is reconstructed from the runtime estimate. During convergence, the runtime estimate blends from the immutable base toward the accumulated dynamic result based on sample count so one low-sample update does not cause a noisy pop.

Conceptually:

```c
float confidence = saturate(sample_count / target_samples);
indirect = lerp(base_indirect, runtime_estimate, confidence);
```

When an overlay texel becomes invalid, rendering immediately falls back to `base_indirect` until new samples are available.

## Dirty Region Detection

When a dynamic object transform changes:

1. compute old world bounds;
2. compute new world bounds;
3. form the swept bounds as their union;
4. invalidate the old and new influence regions;
5. enqueue new local GI work for the current transform only.

Direct sun influence uses the dynamic shadow map and therefore updates immediately.

Indirect influence uses the swept bounds expanded by a configurable GI radius. This intentionally limits dynamic GI to a local correction rather than retracing the entire level for every moving object.

## Static Lightmap Spatial Index

Pony builds a static world-space index from lightmap sample world positions.

The index maps coarse world-space cells to compact lightmap sample IDs. It is built once when scene/lightmap data is prepared and reused for all runtime dirty-region queries.

A dynamic influence AABB therefore resolves to:

```text
influence AABB
-> overlapping spatial cells
-> affected lightmap sample IDs
-> runtime dirty queue
```

The implementation must avoid a full scan of every valid lightmap texel on each dynamic-object movement.

## Fixed Runtime Work Budget

Dynamic indirect updates use an explicit fixed budget, for example:

```c
typedef struct DYNAMIC_LIGHTING {
    uint32_t texels_per_frame;
    uint32_t rays_per_texel;
    uint32_t target_samples;
    float gi_radius;
} DYNAMIC_LIGHTING;
```

Rules:

- process at most `texels_per_frame` dirty samples each frame;
- process at most `rays_per_texel` new rays for each selected texel in that frame;
- never increase either number because objects have stopped moving;
- unused budget remains unused;
- once all dirty texels reach `target_samples`, the system becomes idle naturally;
- movement may occur again at any time without changing the maximum frame workload.

This guarantees that standing still cannot initiate a hidden high-GPU "finish the bake" phase and therefore cannot create a later workload discontinuity when motion resumes.

## Queue Generations and Stale Work

Every dynamic-lighting state change increments a generation counter.

Dirty queue entries carry the generation that created them. Before tracing or committing a result, the current generation is checked.

If an object moves again:

- the new transform increments the generation;
- old queued work becomes stale and is skipped/discarded;
- old overlay regions are invalidated immediately;
- current-transform work is inserted ahead of obsolete work.

No GPU time should be intentionally spent converging lighting for a transform that is no longer current.

## Runtime Tracing

The runtime GI tracer reuses the current compute/software-BVH path.

For each selected static lightmap sample, rays are traced against the union of:

```text
static scene BVH
+ transformed dynamic model BVHs
```

The first version computes a local indirect correction with a small number of rays and a limited bounce count. It does not attempt to make movement globally re-solve every bounce throughout the level.

The runtime tracer should share shading/material evaluation with the existing bake where practical rather than creating a second incompatible GI model.

When the `nri` branch lands, the same logical queues can migrate to:

- persistent compute command contexts;
- GPU-generated indirect dispatch arguments;
- wave/subgroup compaction;
- wavefront ray/hit queues;
- asynchronous compute where supported.

Correctness does not depend on those optimizations.

## Runtime Data Flow

```text
OBJECT transform changes
        |
        +-> update previous/current transform
        +-> update world bounds / dynamic TLAS
        +-> increment lighting generation
        +-> invalidate old/new GI overlay regions
        +-> enqueue current affected lightmap samples

frame rendering
        |
        +-> render dynamic-only sun shadow map
        +-> process <= fixed dynamic GI budget
        +-> shade static geometry:
        |      base indirect / overlay
        |      + static direct sun * dynamic visibility
        |
        +-> shade dynamic geometry:
               volume probes
               + realtime sun using static + dynamic visibility
```

## Failure and Fallback Behavior

- No dynamic objects: runtime path performs no GI tracing and the scene matches the permanent bake.
- Overlay allocation failure: render the immutable base bake and dynamic direct shadows only.
- Dynamic spatial-index query produces no samples: no GI work is scheduled.
- Dynamic BVH/TLAS unavailable for an object: that object may still render, but it must not silently contribute incorrect runtime GI; dynamic GI for that object is disabled and the base bake remains intact.
- No baked lighting loaded: dynamic overlay does not pretend to replace the normal unbaked fallback path.

## First Implementation Scope

Included:

- transform/history on `OBJECT`;
- local-space dynamic model BVHs;
- small dynamic-object top-level bounds structure;
- static + dynamic software ray intersection;
- dynamic objects excluded from permanent lightmap lighting;
- dynamic objects lit by SH volume probes;
- separate static direct-sun data from baked indirect;
- dynamic-only directional shadow map;
- realtime sun shadowing on static and dynamic receivers;
- lightmap world-space spatial index;
- transient runtime indirect overlay;
- fixed per-frame dynamic GI budget;
- generation-based invalidation and stale-work rejection;
- local GI radius limitation;
- integration seams for future `nri` queue/indirect/wavefront work.

Deferred:

- dynamic point-light shadow maps;
- dynamic spot-light shadow maps;
- permanent runtime rebaking or writing runtime state to `.baked`;
- global full-scene GI updates caused by moving objects;
- physics/animation systems;
- hardware ray tracing;
- any idle-time/adaptive increase in the runtime GI budget.

## Verification

The implementation is complete only when:

1. a scene with no dynamic objects visually matches the normal baked result;
2. a moving dynamic blocker produces a same-frame sun shadow on static geometry;
3. the same blocker receives static probe indirect lighting and correct sun visibility;
4. moving the blocker invalidates the old local GI overlay immediately, leaving no ghost lighting at its prior transform;
5. nearby static indirect lighting progressively converges under the fixed frame budget;
6. repeated movement never raises the configured per-frame GI work cap;
7. stopping movement does not raise the work cap;
8. obsolete-generation queue items do not write results after a newer transform exists;
9. the portable software-BVH path works without NRI ray tracing;
10. affected C and HLSL variants compile and the final implementation branch contains no temporary workflow/test-helper files.
