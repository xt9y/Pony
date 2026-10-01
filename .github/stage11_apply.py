#!/usr/bin/env python3
from pathlib import Path
import sys


def replace_once(path, old, new):
    p = Path(path)
    text = p.read_text()
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected exactly one match, got {count}\n--- needle ---\n{old[:500]}")
    p.write_text(text.replace(old, new, 1))


def insert_before_once(path, marker, addition):
    replace_once(path, marker, addition + marker)


def task1():
    replace_once(
        "sdf.c",
        """typedef struct WORLD_PROBE_CANDIDATE {\n    int32_t cell[3];\n    VEC3 position;\n    float clearance;\n    uint32_t surface_id;\n    uint32_t object_index;\n    uint32_t revision;\n} WORLD_PROBE_CANDIDATE;\n""",
        """typedef struct WORLD_PROBE_CANDIDATE {\n    int32_t cell[3];\n    VEC3 position;\n    float clearance;\n    uint32_t surface_id;\n    uint32_t object_index;\n    uint32_t revision;\n    uint64_t source_order;\n} WORLD_PROBE_CANDIDATE;\n""",
    )
    replace_once(
        "sdf.c",
        """    if (a->surface_id < b->surface_id) return -1;\n    if (a->surface_id > b->surface_id) return 1;\n    return 0;\n}\n""",
        """    if (a->surface_id < b->surface_id) return -1;\n    if (a->surface_id > b->surface_id) return 1;\n    if (a->source_order < b->source_order) return -1;\n    if (a->source_order > b->source_order) return 1;\n    return 0;\n}\n""",
    )
    replace_once(
        "sdf.c",
        """    WORLD_PROBE_CANDIDATE *candidates = malloc(candidate_capacity * sizeof(*candidates));\n    if (!candidates) goto fail;\n    uint32_t level_count = global_sdf->clip_count < 2u ? global_sdf->clip_count : 2u;\n""",
        """    WORLD_PROBE_CANDIDATE *candidates = malloc(candidate_capacity * sizeof(*candidates));\n    if (!candidates) goto fail;\n    uint64_t source_order = 0u;\n    uint32_t level_count = global_sdf->clip_count < 2u ? global_sdf->clip_count : 2u;\n""",
    )
    replace_once(
        "sdf.c",
        """                        if (!accepted) continue;\n                        if (candidate_count == candidate_capacity) {\n""",
        """                        if (!accepted) continue;\n                        candidate.source_order = source_order++;\n                        if (candidate_count == candidate_capacity) {\n""",
    )
    replace_once(
        "tests/stage10_world_cache.c",
        """    assert(na == nb);\n    assert(memcmp(a, b, (size_t)na * sizeof(*a)) == 0);\n    assert(memcmp(ka, kb, WORLD_PROBE_HASH_CAPACITY * sizeof(*ka)) == 0);\n\n    int saw_negative = 0;\n""",
        """    assert(na == nb);\n    assert(memcmp(a, b, (size_t)na * sizeof(*a)) == 0);\n    assert(memcmp(ka, kb, WORLD_PROBE_HASH_CAPACITY * sizeof(*ka)) == 0);\n    for (uint32_t repeat = 0u; repeat < 8u; ++repeat) {\n        memset(b, 0, WORLD_PROBE_CAPACITY * sizeof(*b));\n        memset(kb, 0, WORLD_PROBE_HASH_CAPACITY * sizeof(*kb));\n        nb = 0u;\n        assert(sdf_build_world_probes(&g, &scene, &object, 1u, b, WORLD_PROBE_CAPACITY, &nb, kb, WORLD_PROBE_HASH_CAPACITY,\n                                      WORLD_PROBE_SPACING, WORLD_PROBE_RADIUS, WORLD_PROBE_CLEARANCE, WORLD_PROBE_MIN_CLEARANCE));\n        assert(na == nb);\n        assert(memcmp(a, b, (size_t)na * sizeof(*a)) == 0);\n        assert(memcmp(ka, kb, WORLD_PROBE_HASH_CAPACITY * sizeof(*ka)) == 0);\n    }\n\n    int saw_negative = 0;\n""",
    )


def task2():
    replace_once(
        "build.c",
        """        {\"CS_CommitScreenProbeHistory\", \"compute\", \"radiance_probe_history.cs.spv\", NULL},\n        {\"CS_UpdateWorldRadianceCache\", \"compute\", \"radiance_world_cache.cs.spv\", NULL},\n""",
        """        {\"CS_CommitScreenProbeHistory\", \"compute\", \"radiance_probe_history.cs.spv\", NULL},\n        {\"CS_SelectWorldProbeUpdates\", \"compute\", \"radiance_world_select.cs.spv\", NULL},\n        {\"CS_UpdateWorldRadianceCache\", \"compute\", \"radiance_world_cache.cs.spv\", NULL},\n""",
    )

    marker = """[numthreads(64, 1, 1)]\nvoid CS_UpdateWorldRadianceCache(uint3 dispatch_id : SV_DispatchThreadID) {\n"""
    selector = r'''uint WorldProbeFrameAge(WorldProbeState state) {
    return (Pass.dispatch.x - state.state.x) & 0x00ffffffu;
}

uint WorldProbePriorityClass(WorldProbeState state, uint sweep_frames) {
    if ((state.state.w & 1u) == 0u || state.state.y != Radiance.feature_flags.y) return 3u;
    if ((state.state.w & 2u) != 0u) return 2u;
    uint age_frames = WorldProbeFrameAge(state);
    return age_frames >= 2u * sweep_frames ? 1u : 0u;
}

bool WorldProbeMoreUrgent(WorldProbeState candidate, uint candidate_index, WorldProbeState best, uint best_index, uint sweep_frames) {
    uint candidate_class = WorldProbePriorityClass(candidate, sweep_frames);
    uint best_class = WorldProbePriorityClass(best, sweep_frames);
    if (candidate_class != best_class) return candidate_class > best_class;
    if (candidate.statistics.y != best.statistics.y) return candidate.statistics.y > best.statistics.y;
    if (candidate.statistics.x != best.statistics.x) return candidate.statistics.x < best.statistics.x;
    uint candidate_age = WorldProbeFrameAge(candidate);
    uint best_age = WorldProbeFrameAge(best);
    if (candidate_age != best_age) return candidate_age > best_age;
    return candidate_index < best_index;
}

[numthreads(64, 1, 1)]
void CS_SelectWorldProbeUpdates(uint3 dispatch_id : SV_DispatchThreadID) {
    uint active_count = Radiance.cache_counts.y;
    uint lane = dispatch_id.x;
    if (!FeatureEnabled(RADIANCE_FEATURE_WORLD_CACHE) || active_count == 0u || lane >= 64u) return;

    uint base = Pass.dispatch.x % active_count;
    uint sweep_frames = max((active_count + 63u) / 64u, 1u);
    uint best_index = INVALID_INDEX;
    WorldProbeState best_state = (WorldProbeState)0;

    for (uint logical_offset = lane; logical_offset < active_count; logical_offset += 64u) {
        uint probe_index = (base + logical_offset) % active_count;
        WorldProbeState state = WorldProbes[probe_index];
        if (best_index == INVALID_INDEX || WorldProbeMoreUrgent(state, probe_index, best_state, best_index, sweep_frames)) {
            best_index = probe_index;
            best_state = state;
        }
    }

    if (best_index == INVALID_INDEX) return;
    uint slot;
    InterlockedAdd(RayCounters[3], 1u, slot);
    if (slot < Radiance.world_probe_config.y) RadianceUpdateList[slot] = best_index;
}

'''
    insert_before_once("shader.hlsl", marker, selector)
    replace_once(
        "shader.hlsl",
        """    bool established = (state.state.w & 1u) != 0u && state.state.y == Radiance.feature_flags.y;\n    uint s = max(Radiance.world_probe_config.x, 1u);\n    for (uint d = 0u; d < WorldProbeDirectionCount(); ++d) {\n""",
        """    bool established = (state.state.w & 1u) != 0u && state.state.y == Radiance.feature_flags.y;\n    uint s = max(Radiance.world_probe_config.x, 1u);\n    float residual_sum = 0.0f;\n    for (uint d = 0u; d < WorldProbeDirectionCount(); ++d) {\n""",
    )
    replace_once(
        "shader.hlsl",
        """        float3 old = WorldProbeRadiance[old_address].rgb;\n        float blend = established ? saturate(Radiance.world_probe_params.y) : 1.0f;\n        WorldProbeRadiance[new_address] = float4(lerp(old, sample, blend), 1.0f);\n    }\n    state.statistics.x = established ? saturate(state.statistics.x + 0.1f) : 1.0f;\n""",
        """        float3 old = WorldProbeRadiance[old_address].rgb;\n        residual_sum += length(sample - old) / (1.0f + length(sample));\n        float blend = established ? saturate(Radiance.world_probe_params.y) : 1.0f;\n        WorldProbeRadiance[new_address] = float4(lerp(old, sample, blend), 1.0f);\n    }\n    float measured_residual = residual_sum / max((float)WorldProbeDirectionCount(), 1.0f);\n    state.statistics.y = established ? lerp(state.statistics.y, measured_residual, 0.25f) : measured_residual;\n    state.statistics.x = established ? saturate(state.statistics.x + 0.1f) : 1.0f;\n""",
    )


def task3():
    replace_once(
        "game.h",
        """typedef struct RADIANCE_WORLD_RESOURCES {\n    WORLD_PROBE_STATE *cpu_probes;\n    uint32_t *cpu_keys;\n    uint32_t *cpu_update_list;\n\n    NriBuffer *probes;\n""",
        """typedef struct RADIANCE_WORLD_RESOURCES {\n    WORLD_PROBE_STATE *cpu_probes;\n    uint32_t *cpu_keys;\n\n    NriBuffer *probes;\n""",
    )
    replace_once(
        "game.h",
        """    uint32_t direction_count;\n    uint32_t bank_count;\n    uint32_t update_count;\n    uint32_t update_cursor;\n} RADIANCE_WORLD_RESOURCES;\n""",
        """    uint32_t direction_count;\n    uint32_t bank_count;\n} RADIANCE_WORLD_RESOURCES;\n""",
    )
    replace_once(
        "game.h",
        """    NriPipeline *wavefront_history_pipeline;\n    NriPipeline *world_radiance_pipeline;\n""",
        """    NriPipeline *wavefront_history_pipeline;\n    NriPipeline *world_radiance_select_pipeline;\n    NriPipeline *world_radiance_pipeline;\n""",
    )

    replace_once("render.c", "    free(w->cpu_update_list);\n", "")
    replace_once(
        "render.c",
        """    w->cpu_probes = calloc(WORLD_PROBE_CAPACITY, sizeof(*w->cpu_probes));\n    w->cpu_keys = calloc(WORLD_PROBE_HASH_CAPACITY, sizeof(*w->cpu_keys));\n    w->cpu_update_list = calloc(WORLD_PROBE_UPDATES_PER_FRAME, sizeof(*w->cpu_update_list));\n    if (!w->cpu_probes || !w->cpu_keys || !w->cpu_update_list ||\n""",
        """    w->cpu_probes = calloc(WORLD_PROBE_CAPACITY, sizeof(*w->cpu_probes));\n    w->cpu_keys = calloc(WORLD_PROBE_HASH_CAPACITY, sizeof(*w->cpu_keys));\n    if (!w->cpu_probes || !w->cpu_keys ||\n""",
    )
    replace_once("render.c", "    renderer->world_radiance.update_cursor = 0u;\n", "")
    replace_once(
        "render.c",
        """    w->update_cursor = 0u;\n    w->update_count = 0u;\n    renderer->radiance_revision = compute_radiance_revision(renderer);\n""",
        """    renderer->radiance_revision = compute_radiance_revision(renderer);\n""",
    )
    replace_once(
        "render.c",
        """static void prepare_world_probe_updates(RENDERER *renderer) {\n    RADIANCE_WORLD_RESOURCES *world = &renderer->world_radiance;\n    world->update_count = world->probe_count < WORLD_PROBE_UPDATES_PER_FRAME ? world->probe_count : WORLD_PROBE_UPDATES_PER_FRAME;\n    if (!world->probe_count) {\n        world->update_cursor = 0u;\n        world->update_count = 0u;\n        renderer->pass_constants.range[0] = 0u;\n        return;\n    }\n    for (uint32_t i = 0u; i < world->update_count; ++i)\n        world->cpu_update_list[i] = (world->update_cursor + i) % world->probe_count;\n    renderer->pass_constants.range[0] = world->update_count;\n    world->update_cursor = (world->update_cursor + world->update_count) % world->probe_count;\n}\n\n\n""",
        "",
    )
    replace_once("render.c", "    prepare_world_probe_updates(renderer);\n", "")
    replace_once(
        "render.c",
        """    const NriDataSize grid_index_data = {.data = renderer->dynamic_grid.cpu_indices, .size = (uint64_t)index_upload_count * sizeof(uint32_t)};\n    const NriDataSize world_update_data = {.data = renderer->world_radiance.cpu_update_list, .size = (uint64_t)renderer->world_radiance.update_count * sizeof(uint32_t)};\n\n    NriStreamBufferDataDesc uploads[9];\n""",
        """    const NriDataSize grid_index_data = {.data = renderer->dynamic_grid.cpu_indices, .size = (uint64_t)index_upload_count * sizeof(uint32_t)};\n\n    NriStreamBufferDataDesc uploads[8];\n""",
    )
    replace_once(
        "render.c",
        """    const bool upload_world_updates = renderer->world_radiance.update_count > 0u;\n    if (upload_world_updates)\n        uploads[upload_count++] = (NriStreamBufferDataDesc){.dataChunks = &world_update_data, .dataChunkNum = 1, .placementAlignment = 16, .copyBatch = batch, .dstBuffer = renderer->wavefront.update_list};\n\n""",
        "",
    )
    replace_once("render.c", "    NriBufferBarrierDesc before[9];\n", "    NriBufferBarrierDesc before[8];\n")
    replace_once("render.c", "    if (upload_world_updates) before[barrier_count++] = (NriBufferBarrierDesc){.buffer = renderer->wavefront.update_list, .before = renderer->wavefront.update_list_state, .after = copy};\n", "")
    replace_once("render.c", "    NriBufferBarrierDesc after[9];\n", "    NriBufferBarrierDesc after[8];\n")
    replace_once("render.c", "    if (upload_world_updates) after[barrier_count++] = (NriBufferBarrierDesc){.buffer = renderer->wavefront.update_list, .before = copy, .after = (NriAccessStage){.access = NriAccessBits_SHADER_RESOURCE_STORAGE, .stages = NriStageBits_COMPUTE_SHADER}};\n", "")
    replace_once("render.c", "    if (upload_world_updates) renderer->wavefront.update_list_state = (NriAccessStage){.access = NriAccessBits_SHADER_RESOURCE_STORAGE, .stages = NriStageBits_COMPUTE_SHADER};\n", "")

    replace_once(
        "render.c",
        """           create_compute_pipeline(renderer, \"build/shaders/radiance_probe_history.cs.spv\", renderer->wavefront_layout, &renderer->wavefront_history_pipeline) &&\n           create_compute_pipeline(renderer, \"build/shaders/radiance_world_cache.cs.spv\", renderer->wavefront_layout, &renderer->world_radiance_pipeline);\n""",
        """           create_compute_pipeline(renderer, \"build/shaders/radiance_probe_history.cs.spv\", renderer->wavefront_layout, &renderer->wavefront_history_pipeline) &&\n           create_compute_pipeline(renderer, \"build/shaders/radiance_world_select.cs.spv\", renderer->wavefront_layout, &renderer->world_radiance_select_pipeline) &&\n           create_compute_pipeline(renderer, \"build/shaders/radiance_world_cache.cs.spv\", renderer->wavefront_layout, &renderer->world_radiance_pipeline);\n""",
    )
    replace_once(
        "render.c",
        """            renderer->emissive_pipeline,\n            renderer->wavefront_history_pipeline,\n            renderer->world_radiance_pipeline\n""",
        """            renderer->emissive_pipeline,\n            renderer->wavefront_history_pipeline,\n            renderer->world_radiance_select_pipeline,\n            renderer->world_radiance_pipeline\n""",
    )
    replace_once(
        "render.c",
        """    RADIANCE_WORLD_RESOURCES *world = &renderer->world_radiance;\n    if (world->update_count && (renderer->radiance_constants.feature_flags[0] & RADIANCE_FEATURE_WORLD_CACHE)) {\n        bind_wavefront(renderer, command_buffer, renderer->world_radiance_pipeline);\n        renderer->gpu->core.CmdDispatch(command_buffer, &(NriDispatchDesc){.workGroupNumX = (world->update_count + 63u) / 64u, .workGroupNumY = 1, .workGroupNumZ = 1});\n        barrier_world_radiance(renderer, command_buffer);\n    }\n\n""",
        """    if (renderer->radiance_constants.feature_flags[0] & RADIANCE_FEATURE_WORLD_CACHE) {\n        bind_wavefront(renderer, command_buffer, renderer->world_radiance_select_pipeline);\n        renderer->gpu->core.CmdDispatch(command_buffer, &(NriDispatchDesc){.workGroupNumX = 1, .workGroupNumY = 1, .workGroupNumZ = 1});\n        barrier_wavefront_buffers(renderer, command_buffer, storage);\n\n        bind_wavefront(renderer, command_buffer, renderer->world_radiance_pipeline);\n        renderer->gpu->core.CmdDispatch(command_buffer, &(NriDispatchDesc){.workGroupNumX = 1, .workGroupNumY = 1, .workGroupNumZ = 1});\n        barrier_world_radiance(renderer, command_buffer);\n    }\n\n""",
    )
    replace_once("shader.hlsl", "    RayCounters[3] = Pass.range.x;\n", "    RayCounters[3] = 0u;\n")


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in {"1", "2", "3"}:
        raise SystemExit("usage: stage11_apply.py {1|2|3}")
    {"1": task1, "2": task2, "3": task3}[sys.argv[1]]()


if __name__ == "__main__":
    main()
