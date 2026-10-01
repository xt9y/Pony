# Pony Radiance Stage 10: world radiance cache and multi-bounce GI

## Goal

Stage 10 adds a persistent, camera-independent world-radiance cache so Pony can retain and propagate offscreen indirect lighting through the scene over time.

The target result is stable iterative diffuse GI rather than recursive path tracing:

1. Stage 9 static global SDF provides a camera-independent world-space placement domain.
2. Surface-adaptive world probes are placed near static geometry and persist across frames.
3. Each probe stores a small directional radiance map.
4. A fixed per-frame update budget traces a subset of probes through the existing unified software ray hierarchy.
5. Updated probes evaluate direct analytic lights, explicit emissive-area-light sampling, and previously accumulated world-cache indirect lighting.
6. Screen-probe and offscreen surface shading can consume the newest published world-cache radiance.
7. Successive frames therefore converge from direct lighting toward multiple diffuse bounces without launching recursive rays from each pixel.

The primary visual target is that an emissive ceiling panel can illuminate a nearby surface directly, that surface can feed the world cache, and later cache iterations can propagate that energy to other walls, the ceiling, corners, and camera-visible screen probes.

Stage 10 is not an adaptive-scheduling stage and is not the reflections stage. Stage 11 will decide which probes deserve more updates; Stage 12 will consume the cache for the reflection hierarchy.

## Constraints

- Preserve the NRI/Vulkan/Slang renderer and the existing permanent descriptor spaces.
- Preserve the existing `WorldProbeState`, `WorldProbeRadiance`, `WorldProbeKeys`, and `InvalidationQueue` bindings in space 6.
- Keep `CS_UpdateWorldRadianceCache` as the world-cache update entry point. Do not create a second world-cache shader path.
- Keep world-cache placement camera-independent. Camera motion must not move, rebuild, or re-key the cache.
- Build placement from static scene data only. `DYNAMIC` objects never become persistent placement anchors.
- Use Stage 9 global-SDF/static-surface information rather than inventing a separate spatial acceleration structure solely for GI placement.
- Hard-cap persistent world probes at 8192.
- Use a 4x4 octahedral directional representation: 16 radiance values per probe.
- Use a 16384-slot power-of-two lookup table so the maximum load factor is 50 percent.
- Use a target world-cell spacing of 0.75 world units.
- Initial probe influence radius is 1.125 world units.
- Initial fixed update budget is 64 probes per frame, or 1024 primary world-probe rays per frame.
- The cache must support zero probes without invalid descriptors or a contribution to lighting.
- No full cache clear is allowed merely because the camera moved.
- Multi-bounce must be iterative and bounded. No recursive shader path may launch another arbitrary-depth path.
- Updating probes in a different GPU thread order must not change which previous-iteration data they read.
- Keep the current material/BRDF conventions. Stage 10 must not become an unrelated material-model rewrite.

## Chosen architecture

Stage 10 uses sparse, surface-adaptive, persistent world probes.

Rejected alternatives:

- A uniform scene-volume grid wastes most probes in empty space and scales poorly with room size.
- Camera-following probe clipmaps make offscreen lighting dependent on the current camera and can visibly change as the view moves.
- A new world-BVH or independent placement acceleration structure would duplicate Stage 9 responsibilities.

The world cache is therefore derived from Stage 9's static global-SDF placement domain and remains in fixed world coordinates for the lifetime of the static scene.

## Resource model

`RADIANCE_WORLD_RESOURCES` becomes real persistent storage instead of four one-element dummy buffers.

The subsystem owns:

- `WorldProbeState[probe_capacity]`
- `WorldProbeRadiance[probe_capacity * direction_count * 2]`
- `WorldProbeKeys[hash_capacity]`
- `InvalidationQueue[...]`
- CPU placement/state data required to construct and upload the persistent probe table
- counts/capacities
- an update cursor for fixed round-robin scheduling

Baseline capacities:

- probe capacity: 8192
- hash capacity: 16384
- direction size: 4
- direction count: 16
- radiance history banks: 2

At full capacity, directional radiance stores 8192 * 16 * 2 `float4` values, roughly 4 MiB.

No new descriptor space or alternate world-cache buffer family is introduced.

## Probe placement

Placement is deterministic and camera-independent.

The CPU builder walks Stage 9 fine and medium global-SDF coverage and considers surface-near samples with valid canonical static triangle IDs. Candidate positions are associated with the nearest static surface and converted into world-space placement cells using the 0.75-unit spacing.

Placement rules:

1. Only canonical surfaces whose owning object is `STATIC` may generate candidates.
2. Candidate positions must be finite and inside the Stage 9 static scene domain.
3. Candidates are deduplicated by their world placement cell.
4. When several candidates map to one cell, the deterministic winner is the candidate with the better usable surface clearance, with stable tie-breaking by canonical surface ID / deterministic traversal order.
5. Candidate count is clipped deterministically to 8192. Rebuilding the same unchanged scene must produce the same probe sequence and hash table.
6. Probe placement does not depend on camera position, visibility, screen resolution, or frame index.

The nearest canonical surface ID is retained in `WorldProbeState.identity` for validation/debugging and future invalidation. Probe world position and 1.125-unit influence radius live in `position_radius`.

The builder may use Stage 9 CPU clipmap data and canonical scene triangles to derive a usable surface-side placement point. It must not create a separate persistent acceleration structure just for Stage 10.

## Hash table and lookup

`WorldProbeKeys` is a 16384-entry open-addressed table.

A probe placement cell is hashed using the same integer-cell hash on CPU and shader. The table stores encoded probe indices (`probe_index + 1`), keeping zero as the empty sentinel. `WorldProbeState.identity.x` stores the corresponding non-zero cell key so hash collisions can be rejected.

Runtime lookup searches the 3x3x3 neighborhood around the query placement cell rather than only the exact cell. For each neighboring cell:

1. compute the cell key;
2. probe the hash table with a bounded open-addressing search;
3. validate the decoded probe index and matching key;
4. reject probes outside their influence radius;
5. choose the closest valid probe.

Surface queries additionally prefer probes on the usable side of the queried surface so thin walls do not freely select a nearby probe from the opposite side. This uses the query surface normal and the vector from surface point to probe; it does not require changing the permanent `WorldProbeState` layout.

If no usable probe exists, the world-cache contribution is zero.

## Directional representation

Each world probe uses a 4x4 octahedral directional map.

The logical directional index remains:

```text
probe_index -> octahedral texel -> float4 radiance
```

The physical radiance buffer contains two banks:

```text
bank 0: probe_count * 16 values
bank 1: probe_count * 16 values
```

A helper computes the address from `(probe_index, texel, bank)`.

`WorldProbeState.state` carries the active-bank and validity/update metadata without changing the struct size. The exact field assignments are fixed by the implementation plan, but they must include:

- latest published bank
- last update frame
- lighting/scene revision
- validity / dirty state as required by the existing invalidation path

## Stable double-bank iteration

The two banks prevent same-dispatch feedback and order-dependent multi-bounce.

For each selected probe:

1. read its currently published bank as the old value;
2. trace 16 directions;
3. evaluate each hit using only stable previous-iteration world-cache data;
4. temporally blend the new directional sample against the probe's old published value;
5. write the result into the inactive bank;
6. publish that bank only after the probe update is complete.

A probe updated earlier in the same frame must not become a newer indirect-light source for another probe being updated later in that same frame.

The shader can satisfy this without a third radiance copy by using `last_update_frame` plus the active-bank bit:

- for normal rendering, sample each probe's currently published bank;
- during world-cache iteration, if a source probe has already published in the current frame, sample its opposite bank, which is the previous iteration;
- otherwise sample its current published bank.

Therefore all world probes updated in frame N consume a coherent frame-N-1 cache state even though their output is written in parallel/in arbitrary execution order.

## World-probe update scheduling

Stage 10 uses fixed round-robin scheduling.

The CPU owns an update cursor. Each frame it selects:

```text
min(64, world_probe_count)
```

probe indices, wrapping at the end of the persistent list.

Those indices are uploaded/written into the existing `RadianceUpdateList`, and `RayCounters[3]` contains the update count expected by `CS_UpdateWorldRadianceCache`.

No variance-based or priority scheduling is added in Stage 10. Stage 11 replaces only this selection policy, not the cache format or update shader contract.

A full 8192-probe cache requires 128 frames for one complete fixed-budget sweep. Smaller indoor scenes complete sweeps proportionally faster.

## World-probe shading

Each directional world-probe ray uses the existing unified hierarchy:

```text
world probe
  -> dynamic grid
  -> static global SDF
  -> static local-SDF fallback if unresolved
  -> common SurfaceHit
```

Screen tracing is disabled for persistent world probes because world-cache convergence must not depend on the current camera.

For a miss, the directional sample is the existing sky radiance.

For a surface hit, Stage 10 evaluates world outgoing radiance from:

1. material emission;
2. existing analytic direct-light diffuse contribution;
3. one explicit emissive-area-light NEE sample using the existing emissive triangle CDF/sampler and visibility query;
4. diffuse indirect contribution integrated from the stable previous world cache.

This is the Stage 10 bounce operator. It does not recursively trace another bounce from the hit.

The existing material conventions are retained. Stage 10 must not silently introduce a different metallic/diffuse BRDF than the rest of the current renderer.

## Diffuse world-cache integration

World-cache diffuse GI is not a single arbitrary directional lookup.

For a surface hit, the shader samples the selected world probe's 4x4 directions and integrates the hemisphere aligned to the hit normal using the existing octahedral directions and cosine weighting. The result is converted into the renderer's current diffuse reflected-radiance convention using the hit material.

This integrated previous-cache term becomes the `indirect` value stored/returned by the geometry-addressed surface-radiance path when appropriate.

A directional single-texel lookup remains useful for later reflections and other directional queries, but diffuse bounce propagation uses a hemisphere integration.

## Surface cache interaction

The geometry-addressed surface cache remains a separate subsystem.

Stage 10 may store the latest evaluated direct and indirect reflected radiance in `SurfaceCacheEntry`, but it must not make stale surface-cache values permanently override fresher world-cache propagation.

The implementation must define one authoritative path for a cache miss/update and preserve scene/light revision validation. There must not be two competing implementations of diffuse indirect lighting.

The intended relationship is:

```text
world radiance cache = persistent scene-space indirect-light field
surface cache        = geometry-addressed memoization of evaluated surfaces
screen probes        = camera-visible directional sampling/reuse
```

## Screen-probe integration

Stage 10 feeds current screen-probe shading without replacing the screen-probe system.

When a screen-probe ray resolves to an offscreen/world surface, the hit's reflected radiance may include the newest published world-cache indirect term.

Camera-visible direct/emissive paths remain as they are; Stage 10 adds indirect energy rather than introducing a second direct-light renderer.

This creates the intended flow:

```text
emissive panel
   -> directly lit floor / wall
   -> world probe update
   -> persistent world radiance
   -> later world probe updates
   -> second and later diffuse bounces
   -> screen-probe hits
   -> final visible GI
```

## Temporal behavior

Established world-probe texels use an initial blend factor of 0.20 for new samples versus their previous published value.

New or invalid probes receive a full first update rather than blending against undefined history.

World-probe state tracks enough metadata to distinguish new/invalid data from established history.

Camera movement does not invalidate the world cache.

Static scene replacement rebuilds placement and resets the cache because canonical static geometry changed.

Dynamic-object movement does not rebuild static placement. It can affect traced visibility and, where existing invalidation data is available, reduce confidence/mark affected world probes dirty rather than globally clearing all world radiance.

Lighting revision changes should invalidate/decay cached confidence consistently with the existing revision/invalidation model. Stage 10 does not require an expensive all-buffer clear each frame.

## Feature flags and constants

`RADIANCE_FEATURE_WORLD_CACHE` is enabled only when at least one valid world probe exists and all world-cache buffers/descriptors are ready.

`RADIANCE_FEATURE_MULTIBOUNCE` is enabled together with the iterative previous-cache bounce contribution once the world cache is active.

Runtime constants use the existing `RadianceConstants` fields:

- `cache_counts.y`: active world-probe count
- `cache_counts.z`: world-probe hash-table capacity
- `world_probe_config.x`: direction-map size (`4`)
- `world_probe_config.y`: maximum updates dispatched this frame (`64`)
- remaining `world_probe_config` fields: fixed implementation metadata if required without changing ABI
- `world_probe_params.x`: placement/hash cell spacing (`0.75`)
- `world_probe_params.y`: temporal blend (`0.20`)
- remaining `world_probe_params` fields: influence radius / implementation parameters as fixed by the implementation plan

Counts are zero when no valid probes exist.

## GPU synchronization

The world-cache update pass reads and writes the same logical subsystem, so explicit storage/UAV synchronization is required.

Before `CS_UpdateWorldRadianceCache`:

- probe state, radiance, keys, update list, counters, and required scene/cache resources must be in compute-readable/storage state as appropriate;
- the update list and update count must be visible to the compute shader.

After the update:

- writes to the new radiance bank and probe state must be visible before later screen-probe shading or presentation-side consumers sample the newest published state.

No CPU/GPU readback is part of the per-frame world-cache loop.

## Zero-probe and failure behavior

Zero valid probes is supported.

The renderer still creates/binds valid minimal dummy resources when required by the permanent descriptor layout, but sets:

```text
cache_counts.y = 0
WORLD_CACHE flag = off
MULTIBOUNCE flag = off
```

The image then follows the Stage 9 behavior with zero world-cache contribution.

Allocation or placement failures during scene setup must fail cleanly without partially enabled feature flags or dangling descriptors.

## Files expected to change

### `game.h`

- expand `RADIANCE_WORLD_RESOURCES` with CPU placement data, capacities/counts, update cursor and persistent state
- add Stage 10 fixed capacities/constants if they belong in the shared renderer contract
- keep existing GPU ABI struct sizes unchanged
- add the world-cache update pipeline handle if not already present

### `render.c`

- replace one-element world-cache placeholders with real bounded GPU buffers
- build deterministic static world-probe placement from Stage 9 global-SDF/static-surface data
- construct and upload the 16384-slot hash table
- initialize both directional radiance banks
- fill world-cache constants/feature flags
- schedule up to 64 probe updates per frame through the existing update list/counter
- create/bind/dispatch the existing `radiance_world_cache.cs.spv` pipeline
- add required storage barriers
- rebuild/reset persistent world-cache state on scene replacement, not on camera movement
- preserve valid dummy resources for the zero-probe case

### `shader.hlsl`

- keep the permanent space-6 bindings and `CS_UpdateWorldRadianceCache` entry point
- extend world-probe indexing to two banks
- implement stable previous-iteration reads
- implement bounded 3x3x3 neighboring-cell lookup
- add surface-aware probe selection for diffuse hit queries
- integrate 4x4 world-probe radiance over a diffuse hemisphere
- make world-probe hit evaluation include analytic direct + emissive NEE + previous world-cache indirect + material emission
- publish the inactive bank after each completed probe update
- keep world tracing camera-independent

### `build.c`

No new world-cache shader entry point is expected. `CS_UpdateWorldRadianceCache` already compiles to `radiance_world_cache.cs.spv`. Stage 10 may only need verification/build-list adjustments if the normal build is not already compiling the required active entry.

## Verification

Stage 10 is complete only when all of the following are demonstrated:

1. unchanged scenes produce deterministic world-probe placement;
2. only static surfaces create persistent placement anchors;
3. world-probe count never exceeds 8192;
4. hash capacity is 16384 and every occupied slot decodes to a valid probe with the matching key;
5. zero-probe scenes keep valid descriptor binding and both world-cache feature flags disabled;
6. each probe owns exactly 16 logical directional texels per bank and no directional address exceeds the allocated two-bank radiance buffer;
7. fixed scheduling updates no more than 64 probes / 1024 primary world rays per frame;
8. probes updated in the same frame consume coherent previous-iteration world-cache data regardless of update order;
9. first updates do not blend against uninitialized radiance;
10. established probe updates use the configured 0.20 temporal blend;
11. diffuse world-cache queries use hemisphere integration rather than a single arbitrary direction;
12. emissive area-light NEE participates in world-probe hit shading;
13. camera movement alone does not rebuild, clear, or re-key the world cache;
14. static scene replacement resets/rebuilds the persistent cache cleanly;
15. all active Slang entry points compile to SPIR-V through the normal build configuration;
16. every renderer C translation unit syntax-compiles with the existing strict build headers;
17. `git diff --check` passes;
18. no temporary Stage 10 workflow/helper/test artifacts remain after verification;
19. an M2 runtime test can render the current Cornell scene without descriptor/validation failure and expose the expected gradual multi-bounce convergence for visual inspection.

Compile/structural verification cannot by itself prove the final lighting quality. Visual convergence and performance on the M2 remain runtime validation items after the implementation is pushed.

## Non-goals

- adaptive variance-driven world-probe scheduling (Stage 11)
- reflection output or reflection denoising (Stage 12)
- camera-following irradiance volumes
- GPU-generated probe placement
- signed-distance reconstruction of Stage 9
- skinned-mesh persistent GI placement
- translucent GI
- recursive path tracing
- changing the global material/BRDF model
- replacing screen probes or the surface cache with the world cache
