# Pony Radiance Stage 9: static global SDF clipmaps

## Goal

Stage 9 adds one authoritative scene-wide acceleration structure for `STATIC` geometry so offscreen/world probe rays can traverse the static scene without testing every local SDF model.

The result must preserve Pony Radiance's current hierarchy:

1. raster/screen tracing for camera-visible geometry
2. Stage 8 dynamic-object grid for `DYNAMIC` models
3. Stage 9 static global SDF clipmaps for `STATIC` geometry
4. static local-SDF fallback only when the global structure cannot resolve a hit
5. common `SurfaceHit` shading

The global SDF is built once when a scene is assigned. Dynamic objects never enter it and never trigger a rebuild.

Stage 9 is an acceleration and hit-reliability stage. It does not implement the Stage 10 world radiance cache, multi-bounce propagation, adaptive ray scheduling, reflections, GPU clipmap construction, or streaming open-world clipmaps.

## Constraints

- Preserve the existing NRI/Vulkan/Slang rendering backend.
- Keep the permanent shader ABI and descriptor spaces already defined for global SDF resources.
- Build from transformed world-space `STATIC` triangles, not by resampling local SDF volumes.
- Store canonical global triangle IDs so every global hit resolves into the same `GPUSceneTriangle` identity used by raster and local-SDF paths.
- Use conservative unsigned distances. The stored distance must never exceed the exact point-to-triangle distance at the sampled voxel center.
- Do not add a second global tracer or a second triangle-identity system.
- Do not include `DYNAMIC` geometry in global clipmaps.
- Do not rebuild static clipmaps per frame.
- Do not remove local SDFs; they remain the dynamic representation and a correctness fallback for unresolved static global hits.
- Keep resource sizes bounded and fail scene setup cleanly if limits or allocations are exceeded.

## Chosen architecture

Stage 9 uses world-triangle sparse SDF clipmaps built on the CPU.

Alternative designs were rejected for this stage:

- Resampling existing local SDFs would compound approximation error and preserve local-grid artifacts.
- GPU voxelization would add a larger subsystem and synchronization burden than required for the current renderer.

The builder therefore creates one reusable world-space BVH over all transformed `STATIC` triangles, then samples three same-center world-space SDF levels from that BVH.

## Static world geometry source

The builder consumes the same model order and global triangle identity already used by `RADIANCE_SCENE_DATA`.

For every `OBJECT` satisfying:

- `object.type == MODEL`
- `object.state == STATIC`

Stage 9 transforms each model triangle by the corresponding `GPU_OBJECT.world` matrix and creates a world triangle record containing:

- world-space `a`, `b`, `c`
- triangle AABB
- triangle centroid
- canonical global `triangle_id`

The `triangle_id` is the index into `renderer->radiance_scene.cpu_triangles` / `SceneTriangles`.

Dynamic objects are skipped completely.

The world-triangle list is used only during scene construction and is freed after clipmap generation.

## Reusable world BVH

`sdf.c` owns the CPU nearest-triangle implementation.

The existing local-SDF builder already contains point-triangle distance and a BVH nearest-distance traversal. Stage 9 refactors those internals into reusable helpers rather than copying the math into `render.c`.

The world BVH stores:

- node AABB
- leaf triangle range
- left/right child indices

Leaves contain at most 8 triangles, matching the current local SDF builder behavior.

A nearest query returns both:

- exact unsigned squared distance
- nearest canonical global triangle ID

The same query is used for all clipmap levels.

## Clipmap layout

Stage 9 creates three concentric clipmap levels with the same center and world extent but different voxel resolutions.

Each level uses 4 x 4 x 4 voxels per brick.

The initial fixed level configuration is:

| Level | Brick grid | Effective voxel grid | Purpose |
| --- | ---: | ---: | --- |
| 0 | 32^3 | 128^3 | fine surface detail |
| 1 | 16^3 | 64^3 | medium-distance traversal |
| 2 | 8^3 | 32^3 | coarse whole-scene fallback |

The clipmap world cube is computed from the combined world-space AABB of all static triangles. The largest axis determines the cube extent. The cube receives one coarse-level voxel of padding on every side so geometry exactly on the scene bounds remains represented.

All levels share the same center and cube extent. Only voxel size and brick size differ.

This is a scene-scale hierarchy rather than camera-following clipmaps. Camera-relative streaming is outside Stage 9.

## Sparse brick allocation

Level 2 is dense: every 8^3 brick exists. This guarantees a coarse distance field across the full static-world cube and avoids a tiny fixed fallback step in empty regions.

Levels 0 and 1 are sparse.

A fine or medium brick is allocated when its world AABB intersects any static triangle AABB expanded by a level-specific influence radius:

- level 0 radius: 2 bricks
- level 1 radius: 2 bricks

This allocates a narrow distance band around geometry while leaving distant empty regions to the coarser levels.

Brick selection is performed using triangle AABB-to-brick coordinate ranges, not by testing every possible brick against every triangle.

Duplicate brick requests are collapsed through a temporary CPU bitset indexed by the level's brick grid.

## GPU resource representation

Stage 9 replaces the current one-element global-SDF fallback resources with real buffers while preserving the frozen `space4` ABI:

- `GlobalSDFClipmaps[]`
- `GlobalSDFPageTable[]`
- `GlobalSDFBricks[]`
- `GlobalSDFSurfaceIds[]`

`GPUGlobalSDFClipmap` uses the existing shader interpretation:

- `center_extent.xyz`: world-space clip center
- `center_extent.w`: half-extent of the cubic clip volume
- `voxel_brick.x`: voxel size
- `voxel_brick.y`: brick world size
- `grid.xyz`: brick-grid dimensions
- `grid.w`: page-table base offset
- `data.x`: brick resolution (`4`)
- `data.y`: voxels per brick (`64`)
- `data.z`: first voxel-data offset for this level
- remaining fields are zero unless already required by the frozen ABI

The page table contains one `uint32_t` entry per logical brick. `UINT32_MAX` means that brick is not physically allocated.

Allocated bricks receive a packed physical brick index. Distance and surface-ID arrays use identical packed-brick ordering.

## Conservative distance generation

For each voxel center in an allocated brick:

1. query the world BVH for exact nearest-triangle distance `d`
2. compute the voxel half diagonal `h = 0.5 * sqrt(3) * voxel_size`
3. store `max(0, d - h)`
4. store the nearest canonical global triangle ID

Subtracting the half diagonal makes nearest-neighbor sampling conservative: the stored value cannot claim more guaranteed empty space than the voxel represents.

No sign is computed. Stage 9 remains an unsigned distance field because the current tracer only needs conservative surface approach and exact triangle refinement.

## Runtime resource ownership

`game.h` gains a `GLOBAL_SDF_DATA` owner containing CPU/GPU lifetime state:

- CPU clip descriptors while building
- CPU page table
- CPU distance voxels
- CPU surface IDs
- GPU buffers and SRVs
- GPU access state
- clip count
- page-table count
- physical brick count
- voxel count
- enabled/valid state

The existing one-element global fallback buffers are removed from `RADIANCE_SCENE_FALLBACKS`.

`GLOBAL_SDF_DATA` always owns valid bound descriptors. If the scene contains static geometry, those descriptors point at the real Stage 9 buffers. If the scene contains no static geometry, `GLOBAL_SDF_DATA` creates one zero clip descriptor, one `UINT32_MAX` page-table entry, one zero distance value, and one `UINT32_MAX` surface-ID value solely to satisfy NRI descriptor binding; `clip_count` remains zero and `RADIANCE_FEATURE_GLOBAL_SDF` remains disabled. The shader never consumes the dummy data because the count/feature gate is authoritative.

Creation sequence in `renderer_set_scene()` becomes:

1. create normal scene resources
2. create local SDF resources
3. build Stage 8 dynamic grid
4. build Stage 9 static global SDF
5. upload global buffers
6. update radiance constants and descriptors

Destruction releases all CPU arrays, descriptors, and GPU buffers and zeros the owner.

## Feature flags and constants

When valid static clipmaps exist:

- `RADIANCE_FEATURE_GLOBAL_SDF` is enabled
- `Radiance.sdf_counts.z` equals the real clip count (`3`)

When they do not exist:

- the flag is disabled
- `Radiance.sdf_counts.z == 0`

`Radiance.global_sdf_params.y` stores the coarsest voxel size so an out-of-level sample has a scale-aware fallback step.

`Radiance.global_sdf_params.z` remains the global hit epsilon scale. Stage 9 uses an initial value of `0.65`, matching the intent of the existing local-SDF hit threshold while still requiring exact triangle refinement before accepting a hit.

## Exact global-hit refinement

The global SDF is only an acceleration structure. A coarse voxel hit is never accepted directly as final surface geometry.

`TraceGlobalSDF()` must:

1. sphere-trace the hierarchical global field
2. obtain the nearest canonical triangle ID from the sampled voxel
3. when `distance <= epsilon`, intersect the ray against that exact `SceneTriangle`
4. if that triangle misses, inspect the 3 x 3 x 3 neighboring voxels in the sampled clip level, deduplicate their valid surface IDs, and test those exact triangles
5. accept only a real ray/triangle intersection within `t +/- max(3 * epsilon, 2 * voxel_size)` and before the current best hit distance
6. build the final `SurfaceHit` with exact world position, geometric normal, object/material/primitive identity, and object revision
7. otherwise continue tracing rather than shading the approximate voxel

The exact triangle-refinement helper is shared with the existing local-SDF refinement path where practical.

This rule specifically prevents Stage 9 from reintroducing the long structured surface artifacts previously caused by treating coarse SDF positions/normals as authoritative geometry.

## Hierarchical sampling

`SampleGlobalSDF()` examines clip levels from finest to coarsest.

For each level containing the world position:

- if the logical brick exists, sample its stored nearest-neighbor voxel and return its conservative distance/surface ID
- if the logical brick is absent, continue to the next coarser level

The dense coarse level therefore resolves any point inside the static scene cube.

If the position lies outside all clip levels, the function returns the configured coarsest voxel scale and an invalid surface ID so sphere tracing can advance without inventing a hit.

## Active wavefront ordering

The active miss path changes from:

```text
screen -> dynamic -> static local -> shade
```

to:

```text
screen -> dynamic -> global static -> static local fallback -> shade
```

`CS_WavefrontGlobalTrace` becomes an active pipeline immediately after the dynamic pass.

A successful exact global hit sets a dedicated per-ray resolved bit in `RayFlags`. `CS_WavefrontLocalTrace` checks that bit and skips exhaustive static-local tracing for resolved rays. If global tracing fails to produce an exact hit, the bit remains clear and the existing static-local fallback runs normally. This makes the global SDF a real acceleration stage rather than extra work before the old exhaustive path.

Dynamic models remain excluded from `TraceAllLocalSDFs()` whenever the Stage 8 dynamic-grid feature is enabled.

The unified non-wavefront query follows the same logical hierarchy:

```text
screen (optional) -> dynamic -> global static -> static local fallback
```

For unified queries, static local fallback runs only when `TraceGlobalSDF()` did not resolve an exact hit.

This keeps shadow/emissive visibility queries and probe queries consistent.

## Failure handling

Scene setup fails cleanly when:

- transformed static-triangle count overflows supported 32-bit identity/count ranges
- BVH allocation fails
- clip/page/brick/voxel counts overflow `size_t`, `uint32_t`, or buffer size limits
- any CPU allocation fails
- any NRI global-SDF buffer or descriptor creation fails
- any GPU upload fails

Partial Stage 9 resources are destroyed before returning failure.

The implementation must not silently disable the global SDF after a build/upload failure. Only a legitimate zero-static-geometry scene results in a disabled-but-successful Stage 9 state.

## Memory limits

The builder enforces explicit caps before allocation:

- at most 3 clip levels
- brick resolution fixed at 4
- page-table entries fixed by the level grids: `32^3 + 16^3 + 8^3 = 37376`
- total physical bricks may not exceed the total logical brick count
- total physical voxels may not exceed `37376 * 64 = 2392064`

At maximum occupancy, distance storage is about 9.1 MiB and surface-ID storage is about 9.1 MiB, plus approximately 146 KiB for the page table and negligible clip metadata. Sparse fine/medium levels normally use less.

These caps keep Stage 9 bounded even for large source scenes; world extent affects voxel size, not allocation dimensions.

## Expected performance behavior

Scene-load cost increases because the static world field is generated once on the CPU. Frame cost should decrease for offscreen static-world misses because rays no longer loop every static local SDF model before reaching a result.

Stage 9 does not attempt incremental static edits. Changing a `STATIC` object's transform/revision after scene creation is outside the Stage 9 runtime contract. If such editing is later supported, the correct behavior will be explicit global-SDF invalidation/rebuild rather than silently treating the stale structure as valid.

## Validation

The implementation must include structural/regression checks for these invariants:

- only `STATIC` model triangles enter the world BVH
- every stored surface ID is either `UINT32_MAX` or `< radiance_scene.triangle_count`
- no dynamic model triangle is referenced by a global-SDF voxel
- every page-table physical brick index is in range
- every physical brick has exactly 64 distance voxels and 64 surface IDs
- conservative stored distance is `<=` exact center distance within floating-point tolerance
- zero-static scenes produce `clip_count == 0` and disable `RADIANCE_FEATURE_GLOBAL_SDF`
- valid static scenes produce exactly 3 levels
- wavefront ordering is `screen -> dynamic -> global -> local fallback -> shade`
- a resolved global hit skips exhaustive static-local tracing
- unified tracing follows dynamic/global/local-fallback ordering
- global hits undergo exact triangle refinement before becoming final `SurfaceHit`s
- active Slang entry points compile, including `CS_WavefrontGlobalTrace`
- all C translation units syntax-compile
- `git diff --check` passes

Runtime validation should cover at least:

- `cornell_box.glb`
- `hospital_hallway.glb`
- `poolroom.glb`

The key visual check is that enabling Stage 9 does not reintroduce striped/line artifacts on static walls while offscreen static-world tracing remains camera-independent.

## Files and responsibilities

### `game.h`

- declare `GLOBAL_SDF_DATA`
- add global-SDF ownership to `RENDERER`
- add `wavefront_global_pipeline`
- keep ABI structs aligned with the frozen shader

### `sdf.c`

- refactor reusable triangle/BVH nearest-distance helpers
- build transformed static world triangles
- generate clip descriptors, sparse page table, conservative brick distances, and surface IDs
- expose build/free API to renderer code

### `render.c`

- create/upload/destroy global-SDF GPU resources
- replace placeholder global descriptors with real Stage 9 resources
- populate constants and feature flags
- create/destroy `radiance_global.cs.spv` pipeline
- dispatch dynamic -> global -> local fallback
- bind descriptors and maintain resource states

### `shader.hlsl`

- preserve the existing global resource ABI
- make `TraceGlobalSDF()` refine approximate global hits against exact triangles
- mark resolved global rays so local fallback does not retrace the static scene
- keep unified trace ordering consistent with the active wavefront chain

### `build.c`

- retain the existing single `CS_WavefrontGlobalTrace` shader job
- do not create duplicate global-SDF jobs

## Completion criteria

Stage 9 is complete when static scene geometry has a real uploaded three-level global SDF, the wavefront global trace is active between dynamic and local fallback, accepted global hits use exact canonical triangle geometry, dynamic objects are excluded, resolved global hits skip the static local-SDF path, unresolved rays retain the local fallback, all compile/regression checks pass, and no temporary implementation/test files remain in the repository.
