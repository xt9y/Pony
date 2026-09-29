# Pony Radiance Shader Freeze Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the experimental `shader.hlsl` with a clean, roadmap-complete shader ABI and algorithms so all planned Pony Radiance stages can be enabled later through C/H resource wiring and dispatches only.

**Architecture:** Keep the currently active descriptor ABI in spaces 0-3 compatible enough for the existing renderer, and add permanent subsystem spaces 4-8 for the future GPU scene, wavefront queues, radiance caches, temporal screen probes, and reflections. The frozen shader contains real helper algorithms and dormant entry points for all planned systems; future-only resources are never accessed unless their counts/features are enabled.

**Tech Stack:** C11, HLSL/Slang, SPIR-V 1.5, NRI Vulkan.

**Spec:** `docs/superpowers/specs/2026-09-29-shader-freeze-design.md`

## Global Constraints

- Keep one `shader.hlsl` source file.
- Preserve rasterized primary visibility and the current NRI/Vulkan path.
- Do not implement future scene extraction, clipmap construction, temporal scheduling, world-cache population, dynamic-grid construction, or reflection dispatch in C during this pass.
- Future quality controls are runtime fields or hard maximums, not milestone-specific shader rewrites.
- Remove temporary debug colors, commented alternate algorithms, and stale investigation code.
- Preserve currently active shader entry-point names so the existing renderer can continue running.

## Review Focus

- Zero-count future resources must never be indexed by active or dormant algorithms.
- Screen/local/global tracing must converge on one material-aware surface-hit representation.
- Emissive sampling must be camera independent once C uploads emissive triangles.
- Queue writes and probe atlas writes must be capacity/bounds guarded.
- Current C-side layouts must remain compatible with the active spaces 0-3 until later stages wire spaces 4-8.

---

### Task 1: Freeze the shader ABI and roadmap algorithms

**Files:**
- Modify: `shader.hlsl`

**Interfaces:**
- Consumes: current spaces 0-3 bindings and current renderer dispatch dimensions.
- Produces: permanent structs, feature/debug constants, spaces 4-8 bindings, unified tracing/surface APIs, emissive sampling, cache/probe/world-cache/scheduler/reflection helpers, and permanent future compute entry points.

- [ ] **Step 1: Establish the RED check**

Check the current source for required freeze symbols (`GPUSceneTriangle`, `GPUEmissiveTriangle`, `SurfaceHit`, `GPUDynamicGridCell`, `GPUGlobalSDFClipmap`, `WorldProbeState`, octahedral helpers, emissive sampling, global SDF trace, temporal probe pass, adaptive budget pass, reflection pass). The current shader must fail this completeness check because those symbols/entry points do not exist.

- [ ] **Step 2: Replace `shader.hlsl` with the clean permanent superset**

Preserve current active entry names and space 0-3 bindings. Add permanent future spaces 4-8, hard maxima, runtime feature gates, material-aware `SurfaceHit`, explicit emissive-area-light sampling, octahedral directional probes, geometry-addressed cache API, local/dynamic/global trace helpers, temporal/spatial reuse, world radiance cache, adaptive scheduler, reflections, runtime debug helper, and dormant compute entry points.

- [ ] **Step 3: Run structural GREEN check**

Verify every required ABI type/helper/entry point exists exactly once and that no temporary diagnostic comments/colors or duplicate cache functions remain.

### Task 2: Mirror the frozen ABI in C without enabling future stages

**Files:**
- Modify: `game.h`

**Interfaces:**
- Consumes: shader ABI layouts from Task 1.
- Produces: matching C structs/enums/hard maxima that later stages can upload without redefining shader-side layouts.

- [ ] **Step 1: Establish the RED check**

Check that `game.h` currently lacks the future GPU-scene, material-aware hit, dynamic-grid, global-SDF, screen/world-probe, and ray-budget structures.

- [ ] **Step 2: Add matching ABI declarations**

Add explicit C layouts for the frozen future structs while leaving active renderer resource ownership and behavior unchanged. Keep existing active structs binary-compatible.

- [ ] **Step 3: Add compile-time size expectations**

Add size macros/comments suitable for `_Static_assert` use by later wiring without forcing unused future resources to be allocated now.

### Task 3: Compile every permanent shader entry point

**Files:**
- Modify: `build.c`

**Interfaces:**
- Consumes: permanent entry points from Task 1.
- Produces: SPIR-V binaries for the current renderer plus every planned future pass, so later stages dispatch existing binaries instead of editing HLSL.

- [ ] **Step 1: Establish the RED check**

Compare the current compile list against the frozen entry-point list; it must be missing future passes.

- [ ] **Step 2: Extend `compile_shaders()`**

Compile current entries plus future ray-generation, dynamic/local/global tracing, hit shading, emissive gather, temporal/spatial probe reuse, world-cache update, invalidation, adaptive budget, and reflection passes using the same SPIR-V 1.5 / SM 6.6 options.

- [ ] **Step 3: Verify compile-list completeness**

Ensure every permanent compute entry has exactly one build command and no removed diagnostic-only shader variants remain.

### Task 4: Verification and cleanup

**Files:**
- Verify: `shader.hlsl`, `game.h`, `build.c`

**Interfaces:**
- Consumes: Tasks 1-3.
- Produces: a branch where shader architecture is frozen and later roadmap stages require only C/H data/resource/dispatch work.

- [ ] **Step 1: Shader compiler verification**

Compile every entry point with Slang using the exact options in `build.c`. Any compiler error is blocking.

- [ ] **Step 2: C syntax/layout verification**

Run the project build where available; at minimum verify the modified header and build file remain syntactically coherent and active struct layouts are unchanged.

- [ ] **Step 3: Final source audit**

Confirm no temporary debug colors, commented-out investigation algorithms, stale `near_self_hit`, duplicate cache lookup implementations, or future-stage TODO stubs remain in `shader.hlsl`.

- [ ] **Step 4: Commit/push**

Commit the implementation on `radiance` and push. The user explicitly authorized direct commit/push and waived an additional review gate.
