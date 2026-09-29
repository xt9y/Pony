# Pony Radiance shader freeze design

## Goal

Freeze `shader.hlsl` for the defined Pony Radiance roadmap so later milestones can be implemented through `.c`/`.h` resource creation, uploads, constants, dispatches, and scene extraction without changing shader interfaces or shader-side algorithms.

The frozen shader must cover the planned renderer architecture: raster visibility, hierarchical software ray tracing, material-aware world-space surfaces, emissive area lights, surface radiance cache, directional screen probes, temporal reuse, static global SDF clipmaps, rigid-dynamic local SDF traversal, world radiance cache, iterative multi-bounce propagation, adaptive ray scheduling, and reflections.

This freeze applies to the current roadmap. Features outside that roadmap, such as exact skinned-mesh ray tracing, translucent GI, volumetric GI, arbitrary material graphs, or hardware ray tracing, are not covered by the compatibility guarantee.

## Constraints

- Keep one `shader.hlsl` source file.
- Preserve rasterized primary visibility.
- Keep NRI/Vulkan SPIR-V build flow through Slang.
- Do not implement future renderer stages in `.c`/`.h` during this freeze, except ABI structs, constant buffers, feature/debug enums, resource holders, descriptor layouts, dummy resources, and initialization required for the frozen shader to compile and the existing renderer to run.
- Do not modify scene extraction, GLTF loading, SDF generation, dynamic-grid construction, world-cache population, or future scheduling logic in this pass.
- Remove temporary debugging experiments and duplicate/obsolete shader logic from the current branch.
- All future quality controls that may change at runtime must be constants or flags rather than architecture-changing `#define`s.

## Permanent shader architecture

`shader.hlsl` is organized into these logical sections:

1. constants and feature/debug flags
2. GPU ABI structures
3. resource bindings
4. math, depth, projection, hashing, low-discrepancy sampling, octahedral mapping
5. material and surface evaluation
6. emissive-source sampling
7. direct lighting
8. screen/HZB tracing
9. local SDF tracing
10. dynamic-object-grid tracing
11. global SDF clipmap tracing
12. unified ray-query helpers
13. surface-cache helpers
14. world-radiance-cache helpers
15. screen-probe placement and ray generation
16. temporal and spatial probe reuse
17. wavefront queues and adaptive ray budgets
18. reflection helpers
19. raster, HZB, direct-light, tracing, cache, probe, world-cache, invalidation, reflection, and composite entry points
20. runtime debug presentation

Unused future systems remain dormant through counts and feature flags set to zero rather than requiring shader changes later.

## Permanent GPU ABI

The shader defines stable layouts for these concepts now:

- `GPUObject`
- `GPUMaterial`
- `GPUSceneTriangle`
- `GPUEmissiveTriangle`
- `GPUSDFModel`
- `GPUDynamicGridCell`
- `GPUGlobalSDFClipmap`
- `GPULight`
- `SurfaceHit`
- `TraceRay`
- `TraceHit`
- `SurfaceCacheEntry`
- `ScreenProbeState`
- `WorldProbeState`
- `RayBudget`

Matching C declarations are added to `game.h` during the freeze so future uploads cannot drift from shader layout.

### Surface identity

Every ray-resolution path converges on a `SurfaceHit` containing at least:

- world position
- world normal
- distance
- object ID
- material ID
- primitive/surface ID
- surface coordinates or UV-like coordinates
- hit type
- flags/confidence

A tracer may be approximate, but shading always consumes this common representation. This removes the current architectural problem where SDF hits know only distance/object and lose material identity.

## Descriptor layout

The shader uses subsystem-oriented spaces so future stages do not grow the current monolithic trace set indefinitely:

- `space0`: raster scene and G-buffer draw inputs
- `space1`: presentation textures
- `space2`: HZB build
- `space3`: common radiance/G-buffer inputs and currently active outputs
- `space4`: ray-scene data: triangles, materials, emissive triangles, objects, local SDFs, dynamic grid, global SDF clipmaps
- `space5`: wavefront queues, hits, counters, indirect arguments, ray budgets
- `space6`: surface cache and world radiance cache
- `space7`: screen probes and temporal history
- `space8`: reflection resources

Resources required by currently running passes are bound immediately. Future-only resources may be bound to small dummy buffers/textures or left behind non-dispatched entry points until their C-side stage is implemented, provided pipeline creation remains valid.

## Constants

The ABI uses three logical constant blocks:

### `FrameConstants`

Contains only per-frame camera/view state:

- current, inverse, and previous view-projection matrices
- camera position
- resolution and inverse resolution
- frame index
- jitter/current temporal sample state
- debug view

### `RadianceConstants`

Contains renderer-wide GI configuration and resource counts:

- object/material/triangle/emissive counts
- local-SDF count
- dynamic-grid count/dimensions
- global-SDF clip count
- surface/world-cache counts
- probe dimensions and quality settings
- GI distance and biases
- screen-trace thickness/step controls
- temporal parameters
- adaptive-ray limits
- feature flags
- lighting revision

### `PassConstants`

Small generic per-dispatch parameters for ranges, offsets, dimensions, mip/clip indices, and other stage-local values. This avoids introducing a new shader constant layout for every future pass.

Compile-time constants are retained only for hard maximums and packing decisions. Quality values such as active probe rays, directional resolution, tracing budgets, and feature enablement are runtime constants.

## Material-aware GPU scene and emissive lighting

The frozen shader already supports `GPUSceneTriangle` and `GPUEmissiveTriangle` resources.

`GPUEmissiveTriangle` stores enough data for direct area-light sampling without camera visibility:

- triangle positions or reconstruction references
- normal
- object/material IDs
- area
- emitted radiance/power
- sampling weight/CDF metadata as needed

Shader helpers permanently include:

- emissive triangle selection
- barycentric point sampling
- emitted-radiance evaluation
- PDF/geometry-term weighting
- visibility query

Screen and world probes can explicitly sample emissive geometry. If `emissive_triangle_count == 0`, this path contributes zero.

## Unified tracing

The shader exposes hierarchical traversal helpers with the intended order:

1. screen/HZB trace
2. rigid-dynamic candidate grid/local SDF trace
3. remaining local/object SDF trace as appropriate
4. static global SDF clipmap fallback

Each stage can report unresolved rays for compaction. Final shading receives `SurfaceHit` rather than stage-specific hit formats.

The shader contains future entry points for the wavefront chain instead of a monolithic tracer:

- ray generation
- screen trace
- miss compaction
- local/dynamic trace
- miss compaction
- global SDF trace
- hit shading
- probe/cache update

Current C code may dispatch only the subset already implemented.

## Surface radiance cache

The shader-side cache API is geometry-addressed rather than defined by camera-visible world-position hashing.

The permanent semantics support:

- material identity
- emissive radiance
- direct radiance
- indirect radiance
- object/surface revision
- confidence
- age/last update
- invalidation state

Helpers exist for lookup, material initialization, direct update, indirect update, confidence decay, and invalidation.

The C backing representation can evolve by stage as long as it honors the frozen binding/entry semantics.

## Screen probes

Screen probes use a permanent directional representation. The target representation is octahedral directional radiance with a compile-time maximum size sufficient for the planned quality range.

The initial implementation can trace a small subset of directions per frame, but the shader already supports:

- directional texel addressing
- low-discrepancy direction generation
- probe placement from G-buffer surfaces
- per-direction radiance update
- irradiance integration
- depth/normal-aware interpolation
- history age/confidence/variance
- temporal reprojection
- neighbor reuse
- world-cache fallback

The current temporary model of averaging a fixed 4x4 ray block into one RGB value is removed as the architectural definition.

## World radiance cache

The shader includes stable structures and entry points for sparse persistent world probes with directional radiance, age, confidence, variance, revision, and clip/level information.

Future C code supplies allocation, clipmap placement, update lists, and scheduling. Shader-side lookup, sampling, probe-ray generation, and update logic are already present.

## Dynamic invalidation and multi-bounce

The shader supports confidence-based invalidation rather than global clearing. Object/light revisions can reduce confidence for affected cache/probe regions.

Multi-bounce is iterative: surface/world cache entries may include previously accumulated indirect radiance. Ray hits query cached outgoing radiance; they do not recursively launch full shading paths.

The feature remains disabled until C-side stage wiring enables and schedules it.

## Adaptive ray scheduling

The shader ABI includes ray-budget metadata and classification using inputs such as:

- variance
- history age
- motion/disocclusion
- confidence
- lighting revision
- surface complexity

Future C code sets global budgets and dispatches the scheduler. The initial renderer can force a fixed number of rays by constants.

## Reflections

The frozen shader contains the planned roughness-based hierarchy:

- rough surfaces reuse screen/world directional radiance
- medium-rough surfaces query the radiance caches
- smooth surfaces use a dedicated ray path with screen trace then software scene fallback and surface-radiance lookup

Reflection passes remain undispatched until later stages.

## Runtime debug system

Debug selection becomes a runtime enum in constants rather than recompiling `PS_Present` with per-view defines.

Reserved debug modes cover at least:

- final GI
- albedo
- normals
- depth
- roughness/metallic
- velocity
- object/material/primitive IDs
- HZB
- direct radiance
- emissive-source view
- screen-trace classification
- local SDF
- global SDF
- surface cache
- screen-probe directional radiance
- screen-probe irradiance/confidence/variance/history
- world radiance cache
- ray budget/queue occupancy
- reflection result

C-side key handling only changes the enum.

## Files changed in the shader-freeze pass

### `shader.hlsl`

Substantial cleanup/rewrite to the permanent roadmap-complete shader ABI and entry points. Remove diagnostic branches, duplicate functions, stale temporary cache semantics, and milestone-specific defines.

### `game.h`

Add matching GPU ABI structs, constant structs where shared, feature/debug enums, maximums, resource-holder fields, and compile-time size expectations required for the frozen shader contract.

No scene extraction or future-stage behavior is implemented.

### `render.c`

Only compatibility/wiring changes required to run the current renderer with the frozen shader:

- create/update permanent constant buffers
- update descriptor layouts for active/dummy resources
- initialize counts and feature flags to represent only currently implemented systems
- bind dummy resources when required by a compiled entry point
- switch debug view selection to runtime constants
- restore deterministic current rendering after removing diagnostic experiments

No material-aware triangle extraction, emissive list generation, global SDF construction, temporal history algorithm, world cache population, dynamic-grid construction, adaptive scheduling, or reflection dispatch is implemented in this pass.

### `build.c`

Compile the permanent shader entry points required by the roadmap and remove the runtime debug-shader recompilation dependency. Future stages should select/dispatch existing shader binaries rather than add new HLSL entry points.

## Data flow after the freeze

The currently active renderer remains approximately:

```text
G-buffer
  -> HZB
  -> direct radiance
  -> current surface-cache compatibility path
  -> screen probe ray generation
  -> screen trace
  -> local SDF fallback
  -> probe integration
  -> final composite
```

Dormant future resources/counts are zeroed. Later stages progressively replace compatibility paths by uploading real GPU-scene, emissive, cache, temporal, global-SDF, world-probe, scheduler, and reflection data without modifying `shader.hlsl`.

## Error handling and compatibility

- C and HLSL struct layouts use explicit alignment/padding and C `_Static_assert`s.
- Counts of zero must be valid and produce no contribution.
- Dummy descriptors use valid minimal resources so pipeline creation and validation do not depend on a future subsystem being populated.
- Array/index helpers clamp or validate against runtime counts.
- Queue capacities are bounded by C-provided capacities and shader dispatch limits.
- Unsupported feature flags must not access unbound or zero-count resources.

## Verification

The freeze is complete only when all of the following hold:

1. every shader entry point in `build.c` compiles through `slangc` to SPIR-V 1.5
2. C builds cleanly with strict warnings
3. the renderer initializes on the existing NRI Vulkan path
4. the existing Cornell/current scene still renders without crashes or descriptor errors
5. F1/default final GI and existing core debug views are available through runtime debug selection
6. future-system counts default to zero and do not alter the current image
7. C/HLSL structure size assertions pass
8. no temporary diagnostic colors, commented-out alternate algorithms, duplicate cache lookup functions, or milestone-specific hacks remain in `shader.hlsl`
9. the shader source already contains the declared permanent structures, bindings, helpers, and entry points for every roadmap stage listed above
10. subsequent planned milestones can be described entirely as `.c`/`.h` data extraction/resource/dispatch work without requiring an HLSL interface or algorithm change

## Non-goals for this pass

- extracting material IDs into CPU collision faces
- building/uploading real `GPUSceneTriangle` or `GPUEmissiveTriangle` arrays
- implementing emissive CDF construction
- changing the CPU SDF generator
- implementing global SDF clipmaps
- implementing the dynamic object grid
- implementing temporal history allocation/reprojection scheduling
- implementing the world radiance cache allocator/updater
- enabling multi-bounce propagation
- enabling adaptive ray scheduling
- enabling reflections

Those remain later `.c`/`.h` milestones against the frozen shader contract.
