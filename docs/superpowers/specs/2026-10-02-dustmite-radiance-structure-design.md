# Dustmite Radiance-Structure Port Design

## Goal

Refactor the `Dustmite` codebase so its organization, naming, ownership boundaries, and overall renderer layout read like the `radiance` branch while preserving Dustmite behavior exactly.

`Dustmite` is the behavioral source of truth. `radiance` is only the structural and stylistic reference.

## Non-goals

This work must not import Radiance rendering behavior, GI/SDF systems, world-cache behavior, or any other feature that changes what Dustmite renders or how its bake behaves.

This work must not intentionally change:

- final image appearance;
- bake output or convergence behavior;
- `.baked` cache compatibility;
- cache/hash inputs or ordering;
- lightmap atlas generation;
- BVH construction/traversal behavior;
- probe placement or sampling;
- sun-beam generation;
- volumetric lighting;
- post-processing;
- controls and shortcuts;
- default scene, light, sky, volumetric, or peripheral-vision values;
- NRI backend behavior;
- supported platforms;
- runtime or bake pass ordering except for behavior-neutral extraction into different files/functions.

## Branching

All implementation work happens on `dustmite-radiance-structure`, created directly from Dustmite commit `0ae0e70364210ec302c8b076ea5513d0dd716612`.

`Dustmite` itself is not used as the implementation branch.

## Architectural principle

The refactor follows Radiance's ownership model rather than copying Radiance functionality.

The key split is:

- `GPU`: low-level NRI device, queues, swapchain, frame synchronization, resource allocation, uploads, and generic GPU helpers.
- `RENDERER`: Dustmite-specific render resources, scene resources, bake resources, pipelines, camera state, render passes, and frame orchestration.
- `SCENE` / `OBJECT`: application-facing scene composition and object state.
- specialized Dustmite modules: bake, BVH, beam, lightmap, cache.

The specialized Dustmite modules remain separate because they represent real behavior and data ownership, not arbitrary file fragmentation.

## Target source layout

```text
build.c
game.h
gpu.h
gpu.c
init.c
glb.c
gltf.c
scene.c
render.c
bake.c
bvh.c
beam.c
lmap.c
cache.c
shader.hlsl
main.c
```

Large binary scene assets remain unchanged.

The existing `shaders/` directory is removed only after all Dustmite shader entry points are consolidated into `shader.hlsl` without changing shader logic, entry-point behavior, binding semantics, or compile-time defines.

## File responsibilities

### `gpu.h` / `gpu.c`

These files describe and implement only generic GPU/NRI infrastructure.

The Radiance-style `GPU` object owns:

- SDL window and platform surface state;
- NRI device and interfaces;
- graphics/compute/copy queues;
- swapchain and swapchain textures;
- frame contexts and fences;
- generic streamer/upload infrastructure;
- frame begin/end helpers;
- generic buffer/texture creation and upload helpers;
- generic destruction helpers.

Dustmite-specific lightmap, bake, volumetric, material, post-process, and scene resources must not live in the low-level `GPU` object after the migration.

The NRI setup semantics remain equivalent to Dustmite, including queue fallback, swapchain settings, synchronization, upload semantics, and platform handling.

### `game.h`

`game.h` remains the central public engine/application header in the same broad orientation as Radiance.

It owns public math/mesh types, GLB/GLTF types and APIs, scene/object types, lighting/sky/volumetric/peripheral-vision types, public bake/BVH/lightmap/probe/beam/cache types where needed, renderer-facing declarations, and deliberately shared ABI structures.

Internal NRI-heavy renderer implementation state should not be exposed here unless it is required across modules.

### `scene.c`

Introduce Radiance-style scene ownership:

- `TRANSFORM`;
- `SCENE`;
- `OBJECT`;
- `scene_add_model`;
- `scene_add_light`;
- `object_set_transform`;
- `object_mark_dirty`;
- `scene_free`.

For this port the scene abstraction is an ownership/composition layer only. It must not introduce new dynamic-object rendering behavior.

Dustmite's existing model/light flow is represented through this API while preserving the same data and defaults.

### `render.c`

`render.c` becomes the central Dustmite renderer implementation, following the Radiance naming/orientation pattern.

Public lifecycle target:

```c
bool renderer_init(RENDERER *renderer, GPU *gpu);
bool renderer_set_scene(RENDERER *renderer, SCENE *scene);
void renderer_event(RENDERER *renderer, const SDL_Event *event);
bool renderer_frame(RENDERER *renderer);
void renderer_deinit(RENDERER *renderer);
```

Renderer-specific state moved here includes runtime/post-process pipelines and layouts, depth/HDR/normal/AO/bloom/LUT/volume/lit resources, material textures/samplers, draw/material data, scene vertex data, camera/orbit state, runtime sun/sky/volumetric/peripheral-vision state, debug state, renderer-specific descriptors, and runtime pass orchestration.

The internal frame sequence remains behaviorally equivalent to current Dustmite.

### `bake.c`

`bake.c` owns bake orchestration and bake-specific GPU operations that are not generic GPU infrastructure.

This includes the existing bake lifecycle, worker behavior, progress/title handling, lightmap GPU bake dispatch, probe bake dispatch, sun-beam integration, cached partial-reuse decisions, upload/download of bake outputs, and cache serialization orchestration.

Current algorithms and constants stay unchanged. Helpers currently buried in `gpu.c` move here when they are bake-specific.

### `bvh.c`, `beam.c`, `lmap.c`, `cache.c`

These modules keep their current responsibilities and algorithms.

They receive only structural/style cleanup needed to match Radiance orientation: concise includes, consistent naming, helper ordering, whitespace, initializer style, and early-return error handling.

### `shader.hlsl`

Consolidate Dustmite's current shader files into one Radiance-style `shader.hlsl` mechanically.

The consolidation must preserve:

- all Dustmite shader logic;
- functional entry points;
- register/binding layout semantics;
- defines and compile options;
- workgroup sizes and dispatch assumptions;
- bake/runtime shader ordering;
- all visual and bake behavior.

`build.c` is updated atomically with the consolidation. Old shader files are removed only after every replacement entry point compiles successfully.

### `main.c`

`main.c` becomes composition-oriented like Radiance.

It should:

1. initialize SDL;
2. load GLB/GLTF data;
3. construct the same model/light/scene state;
4. construct the same Dustmite sky, volumetric, and peripheral-vision values;
5. compute the same bake path and hashes;
6. initialize `GPU`;
7. initialize `RENDERER`;
8. bind the scene;
9. load the same cached bake or remain unbaked;
10. run the same event/render/bake loop;
11. deinitialize in reverse ownership order.

`main.c` must not directly manage renderer-internal NRI objects.

## Code-style target

The result should visually read like `radiance`.

Conventions include:

- descriptive `renderer_*` public APIs instead of abbreviated `r_*` names;
- explicit `GPU *gpu` ownership inside `RENDERER` rather than embedding all low-level state in the renderer;
- plain-C composition through structs and functions;
- local static helpers grouped near their consumer;
- early returns for invalid state/failures;
- compact initializer-oriented setup;
- limited comments focused on invariants;
- Radiance-like whitespace and declaration orientation;
- minimal preprocessor branching outside platform/backend boundaries.

`felix-format` remains the formatting source where applicable, but the migration must follow actual Radiance orientation rather than blindly reformatting unrelated code.

## Behavior-preservation contract

### Runtime

The following must remain equivalent to Dustmite:

- rendered geometry;
- PBR material behavior;
- directional sun;
- sun beams;
- volume probes;
- fog toggle;
- wireframe toggle;
- fullscreen behavior;
- orbit/zoom behavior;
- peripheral vision;
- HDR, bloom, ACES, LUT, SSAO, and volumetric composition behavior currently present;
- debug views and existing controls.

### Bake

The following must remain equivalent for identical inputs:

- atlas dimensions;
- valid texel count;
- lightmap sample selection;
- target/minimum sample counts;
- probe placement/count;
- probe sampling parameters;
- compressed sun-beam representation;
- direct/indirect separation;
- cached secondary-hit behavior;
- cache reuse decisions;
- `.baked` binary compatibility.

### Hash compatibility

The exact inputs and ordering for scene, layout, volume, and beam hashes must remain unchanged.

Existing Dustmite `.baked` files must continue loading after the refactor.

## Migration sequence

### Stage 1: lock behavioral baseline

Before structural edits:

- record current source SHA and file list;
- compile all C translation units;
- compile all current shader entry points;
- record shader entry-point list;
- record default configuration values and bake/hash arrays;
- if executable GPU testing is available, run at least the default scene and capture startup/bake logs.

### Stage 2: scene layer

Add `SCENE`, `OBJECT`, and `TRANSFORM` APIs and change `main.c` to compose the same model/light data through them. Renderer internals may remain old at this stage.

### Stage 3: low-level `GPU` extraction

Create the Radiance-style `GPU` object and move generic device/swapchain/frame/resource infrastructure out of the current Dustmite renderer ownership.

### Stage 4: renderer lifecycle/ownership normalization

Move Dustmite-specific runtime state into `RENDERER`, introduce `renderer_*` lifecycle functions, and make `main.c` thin/compositional.

### Stage 5: bake ownership isolation

Move bake-specific GPU helpers and state into `bake.c` while preserving bake logic as literally as practical.

### Stage 6: shader consolidation

Create `shader.hlsl`, move all Dustmite shader code into it, update `build.c`, compile every entry point, then remove old shader files only after successful verification.

### Stage 7: style/orientation pass

Apply Radiance naming, helper ordering, initializer style, whitespace, and comment conventions across touched files without semantic changes.

### Stage 8: regression verification

Run all available build/test/CI validation and compare against the Stage 1 baseline.

## Verification strategy

Required checks:

1. all C translation units compile in the available environment;
2. all shader entry points compile;
3. no obsolete shader path remains referenced after consolidation;
4. all expected Dustmite controls remain reachable in the event path;
5. default configuration constants and hash-input arrays remain identical;
6. cache structures/serialization retain compatible layouts;
7. runtime pass order matches the pre-refactor order;
8. bake orchestration order matches the pre-refactor order;
9. repository CI is green where available;
10. when executable GPU testing is available, default-scene startup, bake, and runtime logs show no unexplained regression.

If deterministic image capture is feasible, compare a before/after reference frame and require no meaningful image difference attributable to the refactor.

## Regression handling

If a structural step introduces an unexplained visual, bake, cache, or performance change:

1. isolate the smallest structural change that introduced it;
2. compare directly against the original Dustmite implementation;
3. restore Dustmite semantics;
4. continue only after baseline behavior is recovered.

Radiance behavior is never used as a shortcut to repair a Dustmite regression.

## Performance policy

This is not an optimization pass. No algorithmic optimization or quality/performance trade-off is introduced as part of this refactor.

Incidental improvements are acceptable only when they are obviously behavior-neutral consequences of cleaner ownership.

## Commit strategy

Use a small sequence of reviewable structural commits, preferably one per migration stage or tightly related subset.

Commit names remain concise and lowercase.

Do not collapse the development history into one opaque rewrite before verification. A final cleanup/squash can be considered separately after the port is proven.

## Completion criteria

The port is complete when:

- implementation lives on the isolated branch;
- Dustmite behavior remains the source of truth;
- renderer ownership/lifecycle has Radiance-style structure;
- `main.c` is composition-oriented;
- low-level GPU infrastructure is separated from renderer-specific state;
- scene/object composition exists;
- Dustmite bake/BVH/beam/lightmap/cache behavior remains intact;
- shaders use the Radiance-style single-file orientation without semantic changes;
- existing `.baked` files remain compatible;
- all available verification passes;
- no intentional visual or functional change has been introduced.
