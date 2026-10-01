# Pony Radiance Stage 10: world radiance cache and multi-bounce GI

## Goal

Stage 10 adds a persistent, camera-independent world-radiance cache so Pony can retain and propagate offscreen indirect lighting through the scene over time.

The target result is bounded iterative diffuse GI rather than recursive path tracing:

1. Stage 9 static global SDF provides a camera-independent world-space placement domain.
2. Surface-adaptive world probes are placed near static geometry and persist across frames.
3. Each probe stores a 4x4 octahedral directional radiance map.
4. A fixed per-frame budget updates a subset of probes through the existing unified software ray hierarchy.
5. Probe-hit shading combines material emission, analytic direct light, explicit emissive-area-light NEE, and the stable previous world-cache iteration.
6. Screen-probe/offscreen surface shading consumes the newest published world-cache indirect term.
7. Successive frames therefore converge toward multiple diffuse bounces without launching recursive rays from each pixel.

The primary visual target is the current Cornell/emissive-panel case: the panel illuminates nearby floor/walls, those surfaces feed persistent world radiance, and later iterations propagate that energy to the rest of the room.

Stage 10 does not implement adaptive scheduling or reflections. Stage 11 changes scheduling policy; Stage 12 consumes the cache for reflections.

## Constraints

- Preserve NRI/Vulkan/Slang and the existing permanent descriptor spaces.
- Preserve the existing space-6 bindings: `WorldProbes`, `WorldProbeRadiance`, `WorldProbeKeys`, and `InvalidationQueue`.
- Keep `CS_UpdateWorldRadianceCache` as the single world-cache update entry point.
- Placement is camera-independent. Camera movement must not move, rebuild, clear, or re-key the cache.
- Only `STATIC` geometry may anchor persistent probes.
- Reuse Stage 9 static/global-SDF data and canonical triangle identity; do not create a second persistent world acceleration structure.
- Maximum active probes: 8192.
- Hash table capacity: 16384, power-of-two.
- Hash lookup/insertion probe bound: 8 slots, matching the existing `MAX_CACHE_PROBES` limit.
- Direction map: 4x4 octahedral = 16 directions per probe.
- Two directional history banks are stored inside the existing `WorldProbeRadiance` buffer.
- Placement spacing: 0.75 world units.
- Influence radius: 1.125 world units.
- Placement surface offset: 0.075 world units.
- Minimum accepted exact static-geometry clearance: 0.0375 world units.
- Fixed update budget: 64 probes/frame = 1024 primary world-probe rays/frame.
- Established texels blend 20 percent new sample / 80 percent previous value.
- New/invalid probes receive a full first update.
- Zero active probes is a valid state and contributes no world GI.
- Multi-bounce is iterative and bounded; no recursive arbitrary-depth ray launch is permitted.
- Same-frame GPU execution order must not change which previous-iteration radiance an update consumes.
- Stage 10 keeps the renderer's current material/BRDF conventions rather than redefining them.

## Chosen architecture

Stage 10 uses sparse, surface-adaptive, persistent world probes.

Rejected alternatives:

- Uniform scene-volume grid: wastes most probes in empty space.
- Camera-following probe clipmaps: violate camera-independent offscreen GI and can change lighting when the camera moves.
- Independent persistent world BVH/cache-placement structure: duplicates Stage 9 responsibilities.

The world cache is derived from Stage 9 static geometry/global-SDF coverage and remains fixed in world coordinates for the lifetime of that static scene.

## Fixed capacities and constants

```text
WORLD_PROBE_CAPACITY        = 8192
WORLD_PROBE_HASH_CAPACITY   = 16384
WORLD_PROBE_DIRECTION_SIZE  = 4
WORLD_PROBE_DIRECTION_COUNT = 16
WORLD_PROBE_BANK_COUNT      = 2
WORLD_PROBE_UPDATES_FRAME   = 64
WORLD_PROBE_SPACING         = 0.75
WORLD_PROBE_RADIUS          = 1.125
WORLD_PROBE_CLEARANCE       = 0.075
WORLD_PROBE_MIN_CLEARANCE   = 0.0375
WORLD_PROBE_BLEND           = 0.20
```

Existing `RadianceConstants` fields are assigned exactly as follows:

```text
cache_counts.x = surface-cache capacity
cache_counts.y = active world-probe count
cache_counts.z = 16384 world-probe hash capacity
cache_counts.w = 8192 physical world-probe capacity

world_probe_config.x = 4 directional size
world_probe_config.y = 64 maximum update count
world_probe_config.z = 2 history banks
world_probe_config.w = 0 reserved

world_probe_params.x = 0.75 placement/hash spacing
world_probe_params.y = 0.20 temporal update blend
world_probe_params.z = 1.125 influence radius
world_probe_params.w = 0.075 placement offset
```

No new constant-buffer layout is introduced.

## Resource model

`RADIANCE_WORLD_RESOURCES` becomes real persistent storage instead of one-element dummy storage.

It owns:

- CPU `WorldProbeState[active_count]` initialization data;
- CPU 16384-entry hash table;
- GPU `WorldProbeState[8192]` storage;
- GPU `float4 WorldProbeRadiance[8192 * 16 * 2]` storage;
- GPU `uint WorldProbeKeys[16384]` storage;
- existing invalidation queue storage;
- active probe count, capacities and update cursor.

The two radiance banks use about 4 MiB at full capacity.

When a scene has zero valid static probe candidates, minimal valid dummy descriptors remain bound while active world-probe count is zero and the feature flags remain disabled.

## WorldProbeState semantics

The existing 64-byte `WorldProbeState` ABI is retained exactly.

```text
position_radius.xyz = probe world position
position_radius.w   = 1.125 influence radius

identity.x = non-zero placement-cell hash key
identity.y = canonical anchor triangle/surface ID
identity.z = anchor GPU object index
identity.w = anchor object revision at placement build

statistics.x = confidence [0,1]
statistics.y = variance/residual estimate reserved for Stage 11
statistics.z = age in completed probe updates
statistics.w = reserved

state.x = last update frame index
state.y = Radiance.feature_flags.y lighting/scene revision
state.z = active radiance bank (0 or 1)
state.w = flags
```

`state.w` flags:

```text
bit 0 = valid published radiance
bit 1 = dirty / confidence-reduced
```

The frozen invalidation entry point is corrected so world-probe invalidation sets the dirty flag/reduces confidence and never overwrites `state.z`.

Stage 10 does not add a new spatial dynamic-invalidation scheduler. Dynamic objects affect world-probe visibility naturally when probes are refreshed. Targeted dirty-probe scheduling can be layered on later without changing this state layout.

## Probe placement

Placement is deterministic, static-only and camera-independent.

The CPU builder uses Stage 9 fine and medium static global-SDF coverage as its candidate domain. For each relevant SDF sample with a valid canonical surface ID:

1. Reject it unless the canonical triangle owner is `STATIC`.
2. Reconstruct the sample world position and canonical world-space triangle.
3. Compute the closest point on that triangle and geometric normal from canonical winding.
4. First candidate: `closest_point + normal * 0.075`.
5. Compute exact nearest distance from that candidate to all static triangles through a temporary nearest-distance/BVH helper shared/refactored from the Stage 9 builder.
6. Accept that side only if the candidate is finite, lies in the Stage 9 static scene domain, and exact nearest distance is at least 0.0375.
7. If `+normal` fails, test `closest_point - normal * 0.075` with the same rules.
8. If neither side passes, discard the source sample.
9. Convert the accepted world position to `floor(position / 0.75)` integer placement-cell coordinates.
10. Keep at most one candidate per placement cell.

The clearance helper/BVH is temporary CPU build state and is freed after placement; Stage 10 must not leave a second persistent acceleration structure in the renderer.

If both +/- sides are valid, canonical `+normal` wins. If several accepted source samples map to one placement cell, choose the candidate with greatest exact clearance; break equal-clearance ties by smaller canonical surface ID, then deterministic source traversal order.

Candidates are emitted in deterministic integer-cell order and capped at 8192. Rebuilding the same unchanged static scene must produce identical probe positions, identities and hash contents.

Dynamic geometry never creates, removes or relocates these persistent placement anchors.

## Hash table

`WorldProbeKeys` is a 16384-entry open-addressed table with zero as the empty sentinel.

CPU and shader use the same hash from signed integer placement-cell coordinates. The final key is forced non-zero (`hash | 1`). `WorldProbeState.identity.x` stores that key. Hash slots store `probe_index + 1`.

Placement-cell deduplication prevents two probes for the same cell. Different cells may still have the same 32-bit hash key; this is legal. Such probes occupy separate linear-probe slots, and runtime distance/surface filtering disambiguates them.

Insertion examines at most 8 consecutive slots. If a candidate cannot be inserted within 8 slots, omit that candidate before finalizing active indices. Every emitted active probe therefore has a reachable table entry.

## Runtime lookup

World-cache lookup searches the 3x3x3 neighborhood around the query's `floor(position / 0.75)` cell.

For each of the 27 neighboring cells:

1. compute its non-zero key;
2. scan at most 8 linear-probe slots;
3. for each occupied slot, decode `probe_index + 1` and validate index/key/valid bit;
4. reject probes farther than 1.125 from the query;
5. keep the closest valid candidate.

The scan cannot stop merely because another decoded probe has the same hash key: distinct cells can hash-collide. It stops on an empty table slot or after 8 probes.

Diffuse surface queries additionally require:

```text
dot(probe.position - surface.position, surface.normal) > 1e-4
```

This prevents a thin wall from freely using a nearby probe on its opposite side. If no candidate survives, world-cache contribution is zero.

A non-surface-filtered directional helper may remain for Stage 12 reflections, but Stage 10 diffuse GI uses the surface-aware query.

## Directional radiance addressing

Each probe has 16 logical directional texels in each of two physical banks.

```text
bank_base  = bank * cache_counts.w * 16
probe_base = probe_index * 16
address    = bank_base + probe_base + direction_texel
```

`cache_counts.w` is the fixed physical capacity 8192, not active count. Bank addressing therefore never changes with scene probe count.

Helpers reject probe indices outside `cache_counts.y` and bank values outside `[0,1]` before accessing radiance.

## Stable double-bank iteration

For a selected probe:

1. `old_bank = state.z`.
2. `new_bank = old_bank ^ 1`.
3. Trace all 16 directions.
4. Read indirect lighting from a stable previous-iteration view of neighboring probes.
5. Blend/write the 16 values into `new_bank`.
6. After those writes, publish `state.z = new_bank` and mark the probe valid.

A world-cache update sampling another probe selects the stable read bank by:

```text
if source.state.x == current_frame:
    read source.state.z ^ 1
else:
    read source.state.z
```

A probe already published earlier in the same frame is therefore read from its previous bank; a not-yet-updated probe is read from its current published bank. All updates in frame N consume the coherent cache state from before frame N's sweep regardless of GPU execution order.

Normal rendering consumers outside the world-cache update read `state.z`, the newest published bank.

## Scheduling

Stage 10 uses CPU-owned fixed round-robin scheduling.

```text
update_count = min(64, active_probe_count)
```

Each frame the renderer writes the wrapped probe indices into existing `RadianceUpdateList`, uploads/writes `RayCounters[3] = update_count` for the world-cache dispatch, and advances the cursor. The later normal screen-wavefront reset may clear that counter after the world update.

A full 8192-probe cache completes one sweep in 128 frames. Smaller scenes sweep proportionally faster.

Stage 11 replaces this selection policy with adaptive scheduling without changing cache representation.

## World-probe ray tracing

Each selected probe traces 16 octahedral directions using the existing unified world tracer with screen tracing disabled:

```text
world probe
  -> dynamic-object grid
  -> static global SDF
  -> static local-SDF fallback if unresolved
  -> SurfaceHit
```

This keeps placement static while still allowing moving geometry to occlude/refine subsequent probe updates.

A miss returns existing sky radiance.

## World-hit bounce operator

A world-probe surface hit evaluates outgoing radiance from exactly:

1. `material.emissive`;
2. existing `EvaluateSurfaceReflectedDirect(hit)` analytic direct contribution;
3. explicit emissive-area-light NEE;
4. stable previous-world-cache diffuse indirect contribution.

The existing `EvaluateEmissiveSampleForMaterial` returns the emitter transport term including cosine/geometry/PDF and `1/pi`, but not receiver albedo. Stage 10 multiplies that term by the current receiver `material.base_color.rgb`, matching the renderer's existing diffuse analytic-light convention. It does not introduce a new metallic/BRDF rule in this stage.

The existing emitter helper already returns zero on an emissive source material, avoiding obvious self-emitter double counting.

No term recursively launches another bounce. Previous-cache sampling is the only multi-bounce input.

## Diffuse world-cache integration

Diffuse GI uses hemisphere integration, not one arbitrary directional lookup.

For the selected nearby probe and requested bank:

1. read all 16 octahedral directions;
2. compute `weight = max(dot(surface_normal, direction), 0)`;
3. accumulate `radiance * weight` and `weight_sum`;
4. return `sum / weight_sum` when `weight_sum > 0`, exactly matching Pony's existing screen-probe directional integration convention;
5. multiply by receiver `material.base_color.rgb` to produce the current renderer's diffuse reflected indirect term.

A directional single-texel sample remains available for Stage 12, but Stage 10 diffuse propagation uses the 16-direction integrated value.

## Surface-cache interaction

The geometry-addressed surface cache remains separate from the persistent world field.

Stage 10 prevents stale cached indirect values from masking newer world GI:

- valid cached direct radiance may be reused under the existing identity/revision checks;
- with world cache enabled, indirect is evaluated from the requested world-cache view: latest bank for normal screen rendering, stable previous view during world-cache updates;
- surface-cache storage may be refreshed with that evaluated indirect value for memoization/debugging, but an old cached indirect value is not authoritative over the world cache.

There is one world-cache diffuse evaluator shared by world-hit and screen/offscreen hit shading.

Subsystem roles remain:

```text
world cache   = persistent scene-space indirect-light field
surface cache = geometry-addressed surface memoization
screen probes = camera-visible directional sampling/reuse
```

## Screen-probe integration

Stage 10 does not replace screen probes.

Two hit cases must consume latest world-cache indirect:

1. Offscreen/SDF/global hits already reach common `SurfaceHit` shading and add latest world-cache diffuse indirect there.
2. Screen/HZB hits currently use `ReflectedDirectAtPixel` and bypass common `SurfaceHit` shading. Stage 10 adds one shared pixel-surface reflected-radiance helper that reconstructs the visible hit position/normal/material, reuses the existing direct radiance, adds latest world-cache diffuse indirect, and is used by `CS_WavefrontScreenTrace` instead of direct-only reuse.

The existing screen-probe `CS_EmissiveGather` remains the current camera-visible explicit-emitter NEE path. Stage 10 does not add a second screen-emitter gather.

The intended energy flow is:

```text
emissive panel
  -> directly lit floor/wall
  -> world-probe update
  -> persistent directional world radiance
  -> later world-probe update
  -> second/later diffuse bounce
  -> screen-probe screen/world hit
  -> visible GI
```

## Temporal update behavior

Established valid texels use:

```text
new_value = lerp(old_value, traced_sample, 0.20)
```

New/invalid probes use `new_value = traced_sample`.

After a probe update:

- confidence moves toward valid/stable;
- age increments;
- `state.x` becomes current frame;
- `state.y` becomes current lighting/scene revision;
- dirty bit clears;
- new bank publishes through `state.z`;
- valid bit sets.

Camera movement changes none of this state.

Static scene replacement destroys/rebuilds world placement and clears both radiance banks.

Dynamic movement does not rebuild placement. Stage 10 does not add targeted spatial invalidation scheduling; changed dynamic visibility is incorporated as round-robin probe updates revisit affected directions.

If the existing/future invalidation pass is dispatched, it may decay confidence/set dirty state but must preserve the bank bit.

## Feature flags

`RADIANCE_FEATURE_WORLD_CACHE` is enabled only when all real world-cache GPU resources are valid and `cache_counts.y > 0`.

`RADIANCE_FEATURE_MULTIBOUNCE` is enabled only with active world cache and stable previous-bank indirect evaluation.

Zero active probes use:

```text
cache_counts.y = 0
WORLD_CACHE = off
MULTIBOUNCE = off
```

The Stage 9 image path then remains valid with zero world-cache contribution.

## GPU synchronization

Before `CS_UpdateWorldRadianceCache`, compute visibility/state is established for world probes, radiance, keys, update list/counter, and all scene tracing resources.

The update writes probe state and inactive radiance-bank values. A storage/UAV barrier after the dispatch makes those writes visible before current-frame screen-probe shading samples latest published world radiance.

No per-frame CPU/GPU readback is used.

## Frame order

Semantic ordering is:

```text
stream scene/dynamic data
  -> refresh dynamic grid if needed
  -> upload fixed world-probe update list/count
  -> CS_UpdateWorldRadianceCache
  -> world-cache UAV barrier
  -> normal direct + screen-probe wavefront path
  -> screen temporal/spatial reuse
  -> existing screen emissive gather/history
  -> present
```

If shared wavefront bookkeeping requires a small setup/reset immediately before the world-cache dispatch, the implementation plan may do so, but it must not clear `RayCounters[3]` between writing the world update count and dispatching `CS_UpdateWorldRadianceCache`. The normal screen-wavefront reset occurs after the world update.

## Failure behavior

Allocation, placement or hash construction failure must not leave partially enabled flags or dangling descriptors.

Scene setup produces one of:

- valid real world-cache resources plus zero-or-more active probes;
- valid minimal dummy bindings with world-cache features disabled; or
- a clean scene-setup failure if required allocation itself cannot be established.

No partial feature enablement is allowed.

## Files expected to change

### `game.h`

- Stage 10 fixed constants
- real `RADIANCE_WORLD_RESOURCES` CPU/GPU ownership, counts/capacities/update cursor
- world-cache update pipeline handle if absent
- unchanged permanent GPU ABI struct sizes

### `sdf.c`

- refactor/reuse static-triangle nearest-distance/BVH/closest-point helpers for deterministic probe placement clearance
- no second persistent acceleration structure

### `render.c`

- build deterministic static world-probe placement after Stage 9 static data exists
- allocate/upload real world-probe/radiance/key/invalidation resources
- construct bounded 16384-slot table
- initialize both banks
- set cache constants/feature flags
- populate round-robin update list/count each frame
- create/bind/dispatch existing `radiance_world_cache.cs.spv`
- add storage barriers
- rebuild on static scene replacement, not camera movement
- retain valid zero-probe dummy behavior

### `shader.hlsl`

- preserve permanent space-6 bindings and `CS_UpdateWorldRadianceCache`
- two-bank addressing and stable read-bank selection
- 3x3x3/8-slot bounded lookup
- surface-side filtering
- 16-direction diffuse hemisphere integration
- world-hit emission + analytic direct + albedo-weighted emissive NEE + stable previous world indirect
- latest world indirect for screen/HZB hit reuse
- invalidation dirty-bit correction
- inactive-bank publish semantics

### `build.c`

No new world-cache entry point is added. `CS_UpdateWorldRadianceCache` already compiles to `radiance_world_cache.cs.spv`.

## Verification

Stage 10 is complete only when all are demonstrated:

1. unchanged scenes produce deterministic placement;
2. only static canonical surfaces anchor persistent probes;
3. accepted placement candidates satisfy >= 0.0375 exact static clearance;
4. active count never exceeds 8192;
5. hash capacity is exactly 16384;
6. insertion/lookup never examines more than 8 slots per requested cell;
7. every active probe is reachable and every occupied hash slot decodes to a matching active probe;
8. zero-probe scenes keep valid descriptors and both world-cache feature flags disabled;
9. physical radiance addressing is bounded for exactly 8192 * 16 * 2 entries;
10. each frame updates at most 64 probes / 1024 primary world-probe rays;
11. same-frame update order cannot expose newly written bank data as previous-iteration input;
12. first updates do not blend uninitialized history;
13. established updates use 0.20 blend;
14. diffuse queries use 16-direction weighted integration and receiver albedo;
15. explicit emissive NEE participates in world-hit updates with receiver albedo;
16. both screen/HZB and offscreen world hits can consume latest published world indirect;
17. camera movement alone does not rebuild/clear/re-key world cache;
18. static scene replacement rebuilds/resets cleanly;
19. invalidation preserves `state.z` active-bank state;
20. all active Slang entries compile;
21. all renderer C translation units syntax-compile;
22. `git diff --check` passes;
23. temporary Stage 10 verification helpers/workflows are removed after verification;
24. M2 Cornell runtime renders without descriptor/validation failure and exposes gradual multi-bounce convergence for visual/performance inspection.

Compile/structural verification does not prove final lighting quality. Cornell convergence, emitter energy distribution, flicker and FPS remain runtime validation items after implementation is pushed.

## Non-goals

- adaptive/variance-driven scheduling (Stage 11)
- reflections/reflection denoising (Stage 12)
- camera-following irradiance volumes
- GPU-generated placement
- a new signed-distance representation
- skinned-mesh persistent GI anchors
- translucent GI
- recursive path tracing
- targeted dynamic-object spatial invalidation scheduling
- renderer-wide BRDF/material redesign
- replacing screen probes or the surface cache
