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
- Direction map: 4x4 octahedral = 16 directions per probe.
- Two directional history banks are stored inside the existing `WorldProbeRadiance` buffer.
- Placement spacing: 0.75 world units.
- Influence radius: 1.125 world units.
- Placement surface offset/clearance target: 0.075 world units.
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
- Independent world BVH/cache-placement structure: duplicates Stage 9 responsibilities.

The world cache is derived from Stage 9 static geometry/global-SDF coverage and remains fixed in world coordinates for the lifetime of that static scene.

## Fixed capacities and constants

The shared renderer constants are:

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
WORLD_PROBE_BLEND           = 0.20
```

The existing `RadianceConstants` fields are assigned as follows:

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
world_probe_params.w = 0.075 placement clearance
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

When the scene has zero valid static probe candidates, minimal valid dummy descriptors remain bound while active world-probe count is zero and the feature flags remain disabled.

## WorldProbeState semantics

The existing 64-byte `WorldProbeState` ABI is retained exactly.

Stage 10 assigns its fields as follows:

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
state.y = lighting/scene revision used for published radiance
state.z = active radiance bank (0 or 1)
state.w = flags
```

`state.w` flag bits:

```text
bit 0 = valid published radiance
bit 1 = dirty / confidence-reduced and should be refreshed
```

The existing invalidation shader must mark the dirty flag; it must not overwrite `state.z`, because `state.z` is the active-bank selector in Stage 10.

## Probe placement

Placement is deterministic, static-only and camera-independent.

The builder uses Stage 9 fine/medium static coverage as its candidate domain. For every relevant global-SDF sample with a valid canonical surface ID:

1. Reject the sample if its canonical triangle owner is not `STATIC`.
2. Reconstruct the sample world position and canonical world-space triangle.
3. Compute the closest point on that triangle and its geometric normal from canonical winding.
4. Try a probe candidate at `closest_point + normal * 0.075`.
5. If that side fails finite/bounds/clearance validation, try `closest_point - normal * 0.075`.
6. If neither side is usable, discard the candidate.
7. Convert the candidate position into an integer 0.75-unit placement cell.
8. Keep at most one candidate per placement cell.

Clearance validation may use a temporary/static-scene nearest-distance helper shared with the Stage 9 CPU SDF builder. It must not leave a second persistent BVH in the renderer. A candidate is accepted only when it has non-degenerate usable clearance from static geometry; this prevents placing the world probe directly inside/against another surface.

If both +/- candidates are valid, canonical triangle winding (`+normal`) wins. If several source samples map to one placement cell, choose the candidate with greatest validated clearance; break equal-clearance ties by smaller canonical surface ID, then deterministic traversal order.

Candidates are emitted in deterministic cell order and capped at 8192. Rebuilding the same unchanged static scene must produce identical probe positions, identities and hash contents.

Dynamic geometry never creates, removes or relocates these persistent placement anchors.

## Hash table

`WorldProbeKeys` is a 16384-entry open-addressed table with zero as the empty sentinel.

The CPU and shader use the same hash from integer placement-cell coordinates. `WorldProbeState.identity.x` stores the resulting non-zero key. The table stores `probe_index + 1`.

Insertion uses the existing bounded cache-probe limit. If a candidate cannot be inserted within that bounded search, the candidate is omitted rather than creating an unreachable probe. Because the table is at most 50 percent full, this is exceptional but deterministic.

Verification requires every active probe to be reachable through the final table and every occupied slot to decode to a valid active probe with a matching key.

## Runtime lookup

World-cache lookup searches the 3x3x3 neighborhood around the query's 0.75-unit placement cell.

For each neighboring cell:

1. compute its key;
2. perform the bounded open-addressing lookup;
3. validate decoded index/key/valid flag;
4. reject probes farther than their 1.125-unit radius;
5. choose the closest remaining probe.

For diffuse surface queries, candidates must also be on the usable side of the hit surface:

```text
dot(probe.position - surface.position, surface.normal) > small_positive_epsilon
```

This prevents a thin wall from freely using a close probe on its opposite side. If no candidate survives, world-cache contribution is zero.

Directional reflection-oriented lookup may retain a non-surface-filtered helper for Stage 12, but Stage 10 diffuse GI uses the surface-aware query.

## Directional radiance addressing

Each probe has 16 logical directional texels in each of two physical banks.

The physical address is:

```text
bank_base = bank * cache_counts.w * 16
probe_base = probe_index * 16
address = bank_base + probe_base + direction_texel
```

`cache_counts.w` is the fixed physical capacity (8192), not active count, so bank addressing never changes when scenes contain fewer probes.

All address helpers validate against active probe count for probe identity and against physical capacity for buffer addressing.

## Stable double-bank iteration

The two banks prevent same-frame feedback and GPU-order-dependent multi-bounce.

For a selected probe:

1. `old_bank = state.z`.
2. `new_bank = old_bank ^ 1`.
3. Trace all 16 directions.
4. Read indirect lighting from a stable previous-iteration view of neighboring probes.
5. Blend/write the 16 new values into `new_bank`.
6. After all 16 texels for that probe are written, set `state.z = new_bank` and mark the probe valid.

A world-cache update that samples another probe chooses its stable read bank as follows:

```text
if source.state.x == current_frame:
    read source.state.z ^ 1
else:
    read source.state.z
```

Therefore a probe already published earlier in the same frame is read from its previous bank, while a probe not yet updated this frame is read from its currently published bank. Every update in frame N therefore consumes the coherent state from before frame N's update sweep.

Normal renderer consumers outside the world-cache iteration always read `state.z`, the newest published bank.

## Scheduling

Stage 10 uses CPU-owned fixed round-robin scheduling.

Each frame:

```text
update_count = min(64, active_probe_count)
```

The renderer writes those wrapped probe indices into the existing `RadianceUpdateList`, writes the update count expected by `CS_UpdateWorldRadianceCache`, then advances the cursor.

A full 8192-probe cache completes one sweep in 128 frames. Smaller scenes sweep proportionally faster.

Stage 10 does not add variance/priority scheduling. Stage 11 replaces only the selection policy.

## World-probe ray tracing

Each selected probe traces 16 deterministic/oct-directional rays using the existing unified world tracer with screen tracing disabled:

```text
probe ray
  -> dynamic-object grid
  -> static global SDF
  -> static local-SDF fallback if unresolved
  -> SurfaceHit
```

This preserves dynamic occlusion while keeping persistent cache placement static.

A miss returns existing sky radiance.

## World-hit bounce operator

A world-probe surface hit evaluates outgoing radiance from exactly these terms:

1. material emission;
2. existing analytic diffuse direct-light evaluation;
3. one explicit emissive-area-light NEE sample using the existing emitter CDF/barycentric sampler and visibility query;
4. diffuse indirect radiance integrated from the stable previous world-cache view.

The explicit emissive sample is skipped on an emissive source material by the existing emitter helper, avoiding self-light double counting.

The NEE term is converted through the renderer's existing diffuse material convention before being added to outgoing reflected radiance. Stage 10 does not change the renderer-wide BRDF model.

No term recursively launches another indirect bounce. Previous-cache sampling is the only multi-bounce input.

## Diffuse world-cache integration

Diffuse GI is hemisphere integration, not a single directional lookup.

For the selected nearby world probe:

1. read all 16 octahedral directions from the requested stable/latest bank;
2. compute cosine weight against the hit surface normal;
3. accumulate only positive-hemisphere directions;
4. normalize consistently with Pony's existing probe integration convention;
5. apply the hit material's current diffuse convention.

The result is the surface's world-cache indirect reflected-radiance term.

A single directional world-cache sample remains available for the later reflection hierarchy, but Stage 10 diffuse propagation always uses the integrated hemisphere result.

## Surface-cache interaction

The geometry-addressed surface cache remains separate from the persistent world field.

Stage 10 changes surface evaluation so a stale cached indirect value cannot permanently mask newer world GI:

- cached direct radiance may be reused when its existing identity/revision checks succeed;
- when world cache is enabled, indirect radiance is evaluated from the currently requested world-cache view (latest for screen rendering, stable previous view during world updates);
- the surface cache may then store/update that evaluated indirect term for memoization/debugging, but lookup does not treat an old indirect field as more authoritative than the world cache.

There is one diffuse indirect evaluator, shared by world-hit and screen/offscreen hit shading; there is not a second competing GI implementation.

Subsystem roles remain:

```text
world cache   = persistent scene-space indirect-light field
surface cache = geometry-addressed surface memoization
screen probes = camera-visible directional sampling/reuse
```

## Screen-probe integration

Stage 10 does not replace the screen-probe system.

When a screen-probe ray resolves to a real world surface, the surface reflected-radiance evaluation can add the newest published world-cache indirect term. Existing camera-visible direct and screen-probe emissive-gather paths remain intact.

The intended energy flow is:

```text
emissive panel
  -> directly lit floor/wall
  -> world-probe update
  -> persistent directional world radiance
  -> later world-probe update
  -> second/later diffuse bounce
  -> screen-probe world hit
  -> visible GI
```

## Temporal update behavior

For an established valid probe texel:

```text
new_value = lerp(old_value, traced_sample, 0.20)
```

For a new/invalid probe, `new_value = traced_sample`.

After update:

- confidence increases toward valid/stable;
- age increments;
- last update frame is set;
- lighting/scene revision is stored;
- dirty flag is cleared;
- new bank is published.

Camera movement changes none of this state.

Static scene replacement destroys/rebuilds world placement and resets both radiance banks.

Dynamic movement can alter subsequent traced visibility and may mark affected probes dirty through the existing invalidation mechanism, but never rebuilds static placement.

Lighting revision changes reduce confidence/mark affected probes dirty consistently with existing invalidation rather than forcing a per-frame global clear.

## Feature flags

`RADIANCE_FEATURE_WORLD_CACHE` is enabled only when all world-cache GPU resources are valid and `cache_counts.y > 0`.

`RADIANCE_FEATURE_MULTIBOUNCE` is enabled only with the active world cache and stable previous-bank indirect evaluation.

For zero active probes:

```text
cache_counts.y = 0
WORLD_CACHE = off
MULTIBOUNCE = off
```

The Stage 9 image path then remains valid with zero world-cache contribution.

## GPU synchronization

Before `CS_UpdateWorldRadianceCache`, the renderer ensures compute visibility/state for:

- `WorldProbes`
- `WorldProbeRadiance`
- `WorldProbeKeys`
- `RadianceUpdateList`
- update counter
- scene/global-SDF/dynamic-grid resources required by tracing

The update pass performs storage writes to probe state and the inactive radiance bank. A storage/UAV barrier after the world-cache dispatch makes published banks/state visible before subsequent screen-probe shading consumes latest world radiance.

No per-frame CPU/GPU readback is used.

## Frame order

The Stage 10 frame order is:

```text
stream scene/dynamic updates
  -> update dynamic grid if needed
  -> fill fixed world-probe update list
  -> world-radiance-cache update
  -> barrier world cache
  -> normal direct/screen-probe wavefront path
  -> temporal/spatial screen-probe reuse
  -> screen-probe emissive gather/history
  -> present
```

World-cache update runs before current screen-probe shading so the visible frame can consume newly published world radiance while the world update itself still consumes the stable previous iteration.

If actual current render ordering requires shared-buffer reset to happen first, the implementation plan may place the update immediately after the reset/setup required for `RadianceUpdateList`/counter ownership, but it must preserve the semantic ordering above: world-cache update before screen-probe world-radiance consumption and with stable previous-bank reads.

## Failure behavior

Allocation, placement or hash construction failure must not leave partially enabled world-cache flags or dangling descriptors.

Scene setup either produces:

- valid real world-cache resources and zero-or-more active probes; or
- valid dummy/minimal resources with world-cache features disabled; or
- a clean renderer scene-setup failure if mandatory allocation itself fails.

No partial feature enablement is allowed.

## Files expected to change

### `game.h`

- Stage 10 fixed constants
- real `RADIANCE_WORLD_RESOURCES` CPU/GPU ownership, counts, capacities and update cursor
- world-cache update pipeline handle if absent
- unchanged GPU ABI struct sizes

### `sdf.c`

- reuse/refactor static triangle/BVH/closest-point helpers only as needed for deterministic surface candidate placement/clearance
- no second persistent acceleration structure

### `render.c`

- build deterministic static world-probe placement after Stage 9 scene data is available
- allocate/upload real world-probe, radiance, key and invalidation resources
- construct 16384-slot hash table
- initialize two radiance banks
- set world-cache constants/flags
- populate fixed round-robin update list/count each frame
- create/bind/dispatch `radiance_world_cache.cs.spv`
- add required storage barriers
- rebuild on scene replacement, not camera movement
- retain valid zero-probe dummy binding behavior

### `shader.hlsl`

- preserve permanent space-6 bindings and `CS_UpdateWorldRadianceCache`
- two-bank addressing and stable read-bank selection
- 3x3x3 neighboring-cell lookup
- surface-side filtering
- 4x4 diffuse hemisphere integration
- world-hit emission + analytic direct + emissive NEE + stable previous world indirect
- correct invalidation dirty-bit behavior
- inactive-bank publish semantics

### `build.c`

No new world-cache entry point is added. `CS_UpdateWorldRadianceCache` already compiles to `radiance_world_cache.cs.spv`.

## Verification

Stage 10 is complete only when all of these are demonstrated:

1. unchanged scenes produce deterministic world-probe placement;
2. only static canonical surfaces anchor persistent probes;
3. active count never exceeds 8192;
4. hash capacity is exactly 16384;
5. every active probe is reachable by its hash and every occupied hash slot decodes to a matching active probe;
6. zero-probe scenes keep valid descriptors and both world-cache feature flags disabled;
7. physical radiance addressing is bounded for 8192 * 16 * 2 entries;
8. each update traces at most 64 * 16 = 1024 primary world-probe rays;
9. same-frame update order cannot expose newly written bank data as previous-iteration input;
10. first updates do not blend uninitialized history;
11. established updates use 0.20 blend;
12. diffuse queries use 16-direction hemisphere integration;
13. explicit emissive NEE participates in world-hit updates;
14. screen/offscreen surface shading can consume newest published world indirect;
15. camera movement alone does not rebuild/clear/re-key the cache;
16. scene replacement rebuilds and resets cleanly;
17. invalidation marks dirty/confidence state without corrupting active-bank state;
18. all active Slang entry points compile;
19. all renderer C translation units syntax-compile;
20. `git diff --check` passes;
21. temporary Stage 10 verification helpers/workflows are removed after verification;
22. M2 runtime testing renders the Cornell scene without descriptor/validation failure and exposes gradual multi-bounce convergence for visual/performance inspection.

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
- renderer-wide BRDF/material redesign
- replacing the screen-probe or surface-cache subsystems
