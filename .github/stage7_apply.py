from pathlib import Path
import re


def replace_once(text, old, new, label):
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected 1 match, found {count}")
    return text.replace(old, new, 1)


def sub_once(text, pattern, replacement, label):
    out, count = re.subn(pattern, replacement, text, count=1, flags=re.S)
    if count != 1:
        raise SystemExit(f"{label}: expected 1 match, found {count}")
    return out

# -----------------------------------------------------------------------------
# game.h
# -----------------------------------------------------------------------------
p = Path('game.h')
s = p.read_text()
s = replace_once(
    s,
    '''    NriPipeline *wavefront_shade_pipeline;\n    NriPipeline *wavefront_resolve_pipeline;\n    NriPipeline *emissive_pipeline;\n\n    uint32_t width;\n    uint32_t height;\n    bool has_previous_frame;''',
    '''    NriPipeline *wavefront_shade_pipeline;\n    NriPipeline *wavefront_temporal_pipeline;\n    NriPipeline *wavefront_spatial_pipeline;\n    NriPipeline *wavefront_resolve_pipeline;\n    NriPipeline *emissive_pipeline;\n    NriPipeline *wavefront_history_pipeline;\n\n    uint32_t width;\n    uint32_t height;\n    bool has_previous_frame;\n    bool probe_history_valid;''',
    'game.h pipelines/history state'
)
p.write_text(s)

# -----------------------------------------------------------------------------
# shader.hlsl
# -----------------------------------------------------------------------------
p = Path('shader.hlsl')
s = p.read_text()

s = replace_once(
    s,
    '''[[vk::binding(5, 7)]] RWTexture2D<float4> ProbeCurrentRadiance : register(u0, space7);\n[[vk::binding(6, 7)]] RWTexture2D<float4> ProbeCurrentMeta : register(u1, space7);\n[[vk::binding(7, 7)]] RWTexture2D<float4> ProbeIrradiance : register(u2, space7);''',
    '''[[vk::binding(5, 7)]] RWTexture2D<float4> ProbeCurrentRadiance : register(u0, space7);\n[[vk::binding(6, 7)]] RWTexture2D<float4> ProbeCurrentMeta : register(u1, space7);\n[[vk::binding(7, 7)]] RWTexture2D<float4> ProbeIrradiance : register(u2, space7);\n[[vk::binding(8, 7)]] RWTexture2D<float4> ProbeHistoryRadianceOut : register(u3, space7);\n[[vk::binding(9, 7)]] RWTexture2D<float4> ProbeHistoryMetaOut : register(u4, space7);\n[[vk::binding(10, 7)]] RWTexture2D<float4> ProbePreviousIrradianceOut : register(u5, space7);\n[[vk::binding(11, 7)]] RWTexture2D<float> ProbeHistoryDepthOut : register(u6, space7);\n[[vk::binding(12, 7)]] RWTexture2D<float4> ProbeHistoryNormalOut : register(u7, space7);''',
    'space7 history UAV aliases'
)

s = replace_once(
    s,
    '''float3 IntegrateProbeIrradiance(uint2 probe, float3 normal) {\n    uint size = ProbeDirectionSize();\n    float3 sum = 0.0f;\n    float weight_sum = 0.0f;\n    for (uint y = 0u; y < size; ++y) {\n        for (uint x = 0u; x < size; ++x) {\n            uint2 texel = uint2(x, y);\n            float3 direction = ProbeDirection(texel);\n            float weight = saturate(dot(normal, direction));\n            sum += ProbeCurrentRadiance[ProbeAtlasCoord(probe, texel)].rgb * weight;\n            weight_sum += weight;\n        }\n    }\n    return weight_sum > 0.0f ? sum / weight_sum : 0.0f;\n}''',
    '''float3 ResolvedProbeRadiance(uint2 probe, uint2 texel) {\n    uint2 coord = ProbeAtlasCoord(probe, texel);\n    if (FeatureEnabled(RADIANCE_FEATURE_SPATIAL_PROBES))\n        return ProbeHistoryRadiance.Load(int3(coord, 0)).rgb;\n    return ProbeCurrentRadiance[coord].rgb;\n}\n\nfloat3 IntegrateProbeIrradiance(uint2 probe, float3 normal) {\n    uint size = ProbeDirectionSize();\n    float3 sum = 0.0f;\n    float weight_sum = 0.0f;\n    for (uint y = 0u; y < size; ++y) {\n        for (uint x = 0u; x < size; ++x) {\n            uint2 texel = uint2(x, y);\n            float3 direction = ProbeDirection(texel);\n            float weight = saturate(dot(normal, direction));\n            sum += ResolvedProbeRadiance(probe, texel) * weight;\n            weight_sum += weight;\n        }\n    }\n    return weight_sum > 0.0f ? sum / weight_sum : 0.0f;\n}''',
    'resolved probe radiance source'
)

s = replace_once(
    s,
    'if (FeatureEnabled(RADIANCE_FEATURE_TEMPORAL_PROBES)) {\n        float previous_luma = dot(ProbePreviousIrradiance.Load(int3(probe, 0)).rgb, float3(0.2126f, 0.7152f, 0.0722f));',
    'if (FeatureEnabled(RADIANCE_FEATURE_TEMPORAL_PROBES) && Pass.flags.x != 0u) {\n        float previous_luma = dot(ProbePreviousIrradiance.Load(int3(probe, 0)).rgb, float3(0.2126f, 0.7152f, 0.0722f));',
    'resolve temporal history gate'
)

s = sub_once(
    s,
    r'''\[numthreads\(8, 8, 1\)\]\nvoid CS_ReprojectScreenProbes\(uint3 dispatch_id : SV_DispatchThreadID\) \{.*?\n\}\n\n\[numthreads\(8, 8, 1\)\]\nvoid CS_SpatialReuseScreenProbes\(uint3 dispatch_id : SV_DispatchThreadID\) \{.*?\n\}\n''',
    '''[numthreads(8, 8, 1)]\nvoid CS_ReprojectScreenProbes(uint3 dispatch_id : SV_DispatchThreadID) {\n    uint2 probe = dispatch_id.xy;\n    if (probe.x >= Pass.dimensions.x || probe.y >= Pass.dimensions.y) return;\n    if (!FeatureEnabled(RADIANCE_FEATURE_TEMPORAL_PROBES) || Pass.flags.x == 0u) return;\n\n    uint2 pixel = ProbeRepresentativePixel(probe);\n    float depth = TraceDepth.Load(int3(pixel, 0));\n    if (depth <= 0.0f) return;\n\n    float2 velocity = VelocityTexture.Load(int3(pixel, 0));\n    float2 previous_pixel = float2(pixel) - velocity * TraceFrame.resolution.xy;\n    int2 previous_probe = int2(floor(previous_pixel / (float)ProbeTileSize()));\n    if (any(previous_probe < int2(0, 0)) || previous_probe.x >= (int)Pass.dimensions.x || previous_probe.y >= (int)Pass.dimensions.y) return;\n\n    uint2 pp = uint2(previous_probe);\n    float previous_depth = ProbeHistoryDepth.Load(int3(pp, 0));\n    if (previous_depth <= 0.0f) return;\n\n    float3 previous_normal = ProbeHistoryNormal.Load(int3(pp, 0)).xyz;\n    float previous_normal_length = dot(previous_normal, previous_normal);\n    if (previous_normal_length <= 1.0e-8f) return;\n    previous_normal *= rsqrt(previous_normal_length);\n\n    float3 current_normal = normalize(TraceNormalRoughness.Load(int3(pixel, 0)).xyz);\n    float previous_linear_depth = LinearizeDepth(previous_depth);\n    float current_linear_depth = LinearizeDepth(depth);\n    float relative_depth_error = abs(previous_linear_depth - current_linear_depth) / max(current_linear_depth, 1.0e-3f);\n    float depth_tolerance = max(Radiance.temporal_params.z, 1.0e-4f);\n    if (relative_depth_error > depth_tolerance || dot(previous_normal, current_normal) < Radiance.temporal_params.w) return;\n\n    float history_weight = saturate(Radiance.temporal_params.x);\n    uint size = ProbeDirectionSize();\n    for (uint y = 0u; y < size; ++y) {\n        for (uint x = 0u; x < size; ++x) {\n            uint2 texel = uint2(x, y);\n            float3 history = ProbeHistoryRadiance.Load(int3(ProbeAtlasCoord(pp, texel), 0)).rgb;\n            uint2 dst = ProbeAtlasCoord(probe, texel);\n            float3 current = ProbeCurrentRadiance[dst].rgb;\n            ProbeCurrentRadiance[dst] = float4(lerp(current, history, history_weight), 1.0f);\n        }\n    }\n\n    float4 meta = ProbeHistoryMeta.Load(int3(pp, 0));\n    ProbeCurrentMeta[probe] = float4(meta.x * Radiance.temporal_params.y, meta.y, meta.z + 1.0f, meta.w);\n}\n\n[numthreads(8, 8, 1)]\nvoid CS_SpatialReuseScreenProbes(uint3 dispatch_id : SV_DispatchThreadID) {\n    uint2 probe = dispatch_id.xy;\n    if (probe.x >= Pass.dimensions.x || probe.y >= Pass.dimensions.y) return;\n\n    uint size = ProbeDirectionSize();\n    ScreenProbeState center;\n    bool valid_center = PlaceScreenProbe(probe, center);\n    uint radius = min(Radiance.probe_config.w, 2u);\n\n    for (uint y = 0u; y < size; ++y) {\n        for (uint x = 0u; x < size; ++x) {\n            uint2 texel = uint2(x, y);\n            uint2 destination = ProbeAtlasCoord(probe, texel);\n\n            if (!valid_center) {\n                ProbeHistoryRadianceOut[destination] = 0.0f;\n                continue;\n            }\n\n            float3 sum = ProbeCurrentRadiance[destination].rgb;\n            float weight_sum = 1.0f;\n\n            if (FeatureEnabled(RADIANCE_FEATURE_SPATIAL_PROBES)) {\n                for (int oy = -(int)radius; oy <= (int)radius; ++oy) {\n                    for (int ox = -(int)radius; ox <= (int)radius; ++ox) {\n                        if (ox == 0 && oy == 0) continue;\n                        int2 np = int2(probe) + int2(ox, oy);\n                        if (np.x < 0 || np.y < 0 || np.x >= (int)Pass.dimensions.x || np.y >= (int)Pass.dimensions.y) continue;\n\n                        ScreenProbeState neighbor;\n                        if (!PlaceScreenProbe(uint2(np), neighbor)) continue;\n                        float normal_weight = saturate(dot(center.normal_confidence.xyz, neighbor.normal_confidence.xyz));\n                        if (normal_weight <= 0.35f) continue;\n                        float distance_weight = rcp(1.0f + length(center.position_depth.xyz - neighbor.position_depth.xyz));\n                        float weight = normal_weight * normal_weight * distance_weight;\n                        sum += ProbeCurrentRadiance[ProbeAtlasCoord(uint2(np), texel)].rgb * weight;\n                        weight_sum += weight;\n                    }\n                }\n            }\n\n            ProbeHistoryRadianceOut[destination] = float4(sum / max(weight_sum, 1.0e-6f), 1.0f);\n        }\n    }\n}\n\n[numthreads(8, 8, 1)]\nvoid CS_CommitScreenProbeHistory(uint3 dispatch_id : SV_DispatchThreadID) {\n    uint2 probe = dispatch_id.xy;\n    if (probe.x >= Pass.dimensions.x || probe.y >= Pass.dimensions.y) return;\n\n    ScreenProbeState state;\n    if (!PlaceScreenProbe(probe, state)) {\n        ProbeHistoryMetaOut[probe] = 0.0f;\n        ProbePreviousIrradianceOut[probe] = 0.0f;\n        ProbeHistoryDepthOut[probe] = 0.0f;\n        ProbeHistoryNormalOut[probe] = 0.0f;\n        return;\n    }\n\n    ProbeHistoryMetaOut[probe] = ProbeCurrentMeta[probe];\n    ProbePreviousIrradianceOut[probe] = ProbeIrradiance[probe];\n    ProbeHistoryDepthOut[probe] = state.position_depth.w;\n    ProbeHistoryNormalOut[probe] = float4(state.normal_confidence.xyz, 1.0f);\n}\n''',
    'temporal/spatial/history passes'
)

p.write_text(s)

# -----------------------------------------------------------------------------
# build.c
# -----------------------------------------------------------------------------
p = Path('build.c')
s = p.read_text()
s = replace_once(
    s,
    '''        {"CS_ReprojectScreenProbes", "compute", "radiance_probe_temporal.cs.spv", NULL},\n        {"CS_SpatialReuseScreenProbes", "compute", "radiance_probe_spatial.cs.spv", NULL},\n        {"CS_UpdateWorldRadianceCache", "compute", "radiance_world_cache.cs.spv", NULL},''',
    '''        {"CS_ReprojectScreenProbes", "compute", "radiance_probe_temporal.cs.spv", NULL},\n        {"CS_SpatialReuseScreenProbes", "compute", "radiance_probe_spatial.cs.spv", NULL},\n        {"CS_CommitScreenProbeHistory", "compute", "radiance_probe_history.cs.spv", NULL},\n        {"CS_UpdateWorldRadianceCache", "compute", "radiance_world_cache.cs.spv", NULL},''',
    'build history shader job'
)
p.write_text(s)

# -----------------------------------------------------------------------------
# render.c
# -----------------------------------------------------------------------------
p = Path('render.c')
s = p.read_text()

s = replace_once(
    s,
    '''    constants.probe_config[2] = SCREEN_PROBE_DIRECTION_COUNT;\n    constants.probe_config[3] = 0u;\n    constants.trace_params[0] = 200.0f;''',
    '''    constants.probe_config[2] = SCREEN_PROBE_DIRECTION_COUNT;\n    constants.probe_config[3] = 1u;\n    constants.trace_params[0] = 200.0f;''',
    'spatial radius'
)

s = replace_once(
    s,
    '''    constants.trace_limits[2] = 96u;\n    constants.trace_limits[3] = 128u;\n    constants.feature_flags[0] = RADIANCE_FEATURE_SURFACE_CACHE;''',
    '''    constants.trace_limits[2] = 96u;\n    constants.trace_limits[3] = 128u;\n    constants.temporal_params[0] = 0.85f;\n    constants.temporal_params[1] = 0.95f;\n    constants.temporal_params[2] = 0.02f;\n    constants.temporal_params[3] = 0.90f;\n    constants.feature_flags[0] = RADIANCE_FEATURE_SURFACE_CACHE | RADIANCE_FEATURE_TEMPORAL_PROBES | RADIANCE_FEATURE_SPATIAL_PROBES;''',
    'temporal settings/features'
)

s = replace_once(
    s,
    '''    const NriDescriptorRangeDesc wavefront_probe_ranges[] = {\n        {.baseRegisterIndex = 0, .descriptorNum = 5, .descriptorType = NriDescriptorType_TEXTURE, .shaderStages = NriStageBits_COMPUTE_SHADER},\n        {.baseRegisterIndex = 5, .descriptorNum = 3, .descriptorType = NriDescriptorType_STORAGE_TEXTURE, .shaderStages = NriStageBits_COMPUTE_SHADER}\n    };''',
    '''    const NriDescriptorRangeDesc wavefront_probe_ranges[] = {\n        {.baseRegisterIndex = 0, .descriptorNum = 5, .descriptorType = NriDescriptorType_TEXTURE, .shaderStages = NriStageBits_COMPUTE_SHADER},\n        {.baseRegisterIndex = 5, .descriptorNum = 8, .descriptorType = NriDescriptorType_STORAGE_TEXTURE, .shaderStages = NriStageBits_COMPUTE_SHADER}\n    };''',
    'space7 storage range'
)

s = replace_once(
    s,
    '''           create_compute_pipeline(renderer, "build/shaders/radiance_local.cs.spv", renderer->wavefront_layout, &renderer->wavefront_local_pipeline) &&\n           create_compute_pipeline(renderer, "build/shaders/radiance_shade.cs.spv", renderer->wavefront_layout, &renderer->wavefront_shade_pipeline) &&\n           create_compute_pipeline(renderer, "build/shaders/radiance_probe_resolve.cs.spv", renderer->wavefront_layout, &renderer->wavefront_resolve_pipeline) &&\n           create_compute_pipeline(renderer, "build/shaders/radiance_emissive.cs.spv", renderer->wavefront_layout, &renderer->emissive_pipeline);''',
    '''           create_compute_pipeline(renderer, "build/shaders/radiance_local.cs.spv", renderer->wavefront_layout, &renderer->wavefront_local_pipeline) &&\n           create_compute_pipeline(renderer, "build/shaders/radiance_shade.cs.spv", renderer->wavefront_layout, &renderer->wavefront_shade_pipeline) &&\n           create_compute_pipeline(renderer, "build/shaders/radiance_probe_temporal.cs.spv", renderer->wavefront_layout, &renderer->wavefront_temporal_pipeline) &&\n           create_compute_pipeline(renderer, "build/shaders/radiance_probe_spatial.cs.spv", renderer->wavefront_layout, &renderer->wavefront_spatial_pipeline) &&\n           create_compute_pipeline(renderer, "build/shaders/radiance_probe_resolve.cs.spv", renderer->wavefront_layout, &renderer->wavefront_resolve_pipeline) &&\n           create_compute_pipeline(renderer, "build/shaders/radiance_emissive.cs.spv", renderer->wavefront_layout, &renderer->emissive_pipeline) &&\n           create_compute_pipeline(renderer, "build/shaders/radiance_probe_history.cs.spv", renderer->wavefront_layout, &renderer->wavefront_history_pipeline);''',
    'chapter7 pipelines'
)

s = replace_once(
    s,
    '''    const NriDescriptor *current[] = {p->current_radiance.uav, p->current_meta.uav, renderer->screen_probes.uav};\n    for (uint32_t i = 0; i < 9; ++i) if (!queues[i]) return false;\n    for (uint32_t i = 0; i < 6; ++i) if (!cache[i]) return false;\n    for (uint32_t i = 0; i < 5; ++i) if (!history[i]) return false;\n    for (uint32_t i = 0; i < 3; ++i) if (!current[i]) return false;''',
    '''    const NriDescriptor *current[] = {\n        p->current_radiance.uav, p->current_meta.uav, renderer->screen_probes.uav,\n        p->history_radiance.uav, p->history_meta.uav, p->previous_irradiance.uav,\n        p->history_depth.uav, p->history_normal.uav\n    };\n    for (uint32_t i = 0; i < 9; ++i) if (!queues[i]) return false;\n    for (uint32_t i = 0; i < 6; ++i) if (!cache[i]) return false;\n    for (uint32_t i = 0; i < 5; ++i) if (!history[i]) return false;\n    for (uint32_t i = 0; i < 8; ++i) if (!current[i]) return false;''',
    'chapter7 descriptors'
)

s = replace_once(
    s,
    '''        {.descriptorSet = renderer->wavefront_probe_set, .rangeIndex = 0, .descriptors = history, .descriptorNum = 5},\n        {.descriptorSet = renderer->wavefront_probe_set, .rangeIndex = 1, .descriptors = current, .descriptorNum = 3}\n    };''',
    '''        {.descriptorSet = renderer->wavefront_probe_set, .rangeIndex = 0, .descriptors = history, .descriptorNum = 5},\n        {.descriptorSet = renderer->wavefront_probe_set, .rangeIndex = 1, .descriptors = current, .descriptorNum = 8}\n    };''',
    'chapter7 descriptor update count'
)

s = replace_once(
    s,
    '''static bool create_size_dependent_resources(RENDERER *renderer, uint32_t width, uint32_t height) {\n    destroy_compute_texture(renderer, &renderer->direct_radiance);''',
    '''static bool create_size_dependent_resources(RENDERER *renderer, uint32_t width, uint32_t height) {\n    renderer->probe_history_valid = false;\n    destroy_compute_texture(renderer, &renderer->direct_radiance);''',
    'resize history invalidation'
)

s = replace_once(
    s,
    '''    renderer->pass_constants.dispatch[0] = renderer->radiance_constants.feature_flags[1];\n    renderer->pass_constants.dimensions[0] = renderer->screen_probes.width;\n    renderer->pass_constants.dimensions[1] = renderer->screen_probes.height;''',
    '''    renderer->pass_constants.dispatch[0] = renderer->radiance_constants.feature_flags[1];\n    renderer->pass_constants.dimensions[0] = renderer->screen_probes.width;\n    renderer->pass_constants.dimensions[1] = renderer->screen_probes.height;\n    renderer->pass_constants.flags[0] = renderer->probe_history_valid ? 1u : 0u;''',
    'history-valid pass flag'
)

# Replace the post-shade tail of the wavefront pass with temporal -> spatial -> resolve.
s = replace_once(
    s,
    '''    const NriTextureBarrierDesc probe_sync = {\n        .texture = p->current_radiance.texture,\n        .before = p->current_radiance.state,\n        .after = compute_storage,\n        .mipNum = 1, .layerNum = 1, .planes = NriPlaneBits_COLOR\n    };\n    renderer->gpu->core.CmdBarrier(command_buffer, &(NriBarrierDesc){.textures = &probe_sync, .textureNum = 1});\n\n    bind_wavefront(renderer, command_buffer, renderer->wavefront_resolve_pipeline);\n    renderer->gpu->core.CmdDispatch(command_buffer, &(NriDispatchDesc){\n        .workGroupNumX = (renderer->screen_probes.width + 7u) / 8u,\n        .workGroupNumY = (renderer->screen_probes.height + 7u) / 8u,\n        .workGroupNumZ = 1\n    });\n}\n''',
    '''    const NriTextureBarrierDesc probe_sync = {\n        .texture = p->current_radiance.texture,\n        .before = p->current_radiance.state,\n        .after = compute_storage,\n        .mipNum = 1, .layerNum = 1, .planes = NriPlaneBits_COLOR\n    };\n    renderer->gpu->core.CmdBarrier(command_buffer, &(NriBarrierDesc){.textures = &probe_sync, .textureNum = 1});\n\n    bind_wavefront(renderer, command_buffer, renderer->wavefront_temporal_pipeline);\n    renderer->gpu->core.CmdDispatch(command_buffer, &(NriDispatchDesc){\n        .workGroupNumX = (renderer->screen_probes.width + 7u) / 8u,\n        .workGroupNumY = (renderer->screen_probes.height + 7u) / 8u,\n        .workGroupNumZ = 1\n    });\n\n    const NriTextureBarrierDesc temporal_sync = {\n        .texture = p->current_radiance.texture,\n        .before = compute_storage,\n        .after = compute_storage,\n        .mipNum = 1, .layerNum = 1, .planes = NriPlaneBits_COLOR\n    };\n    renderer->gpu->core.CmdBarrier(command_buffer, &(NriBarrierDesc){.textures = &temporal_sync, .textureNum = 1});\n\n    const NriTextureBarrierDesc spatial_write = {\n        .texture = p->history_radiance.texture,\n        .before = p->history_radiance.state,\n        .after = compute_storage,\n        .mipNum = 1, .layerNum = 1, .planes = NriPlaneBits_COLOR\n    };\n    renderer->gpu->core.CmdBarrier(command_buffer, &(NriBarrierDesc){.textures = &spatial_write, .textureNum = 1});\n    p->history_radiance.state = compute_storage;\n\n    bind_wavefront(renderer, command_buffer, renderer->wavefront_spatial_pipeline);\n    renderer->gpu->core.CmdDispatch(command_buffer, &(NriDispatchDesc){\n        .workGroupNumX = (renderer->screen_probes.width + 7u) / 8u,\n        .workGroupNumY = (renderer->screen_probes.height + 7u) / 8u,\n        .workGroupNumZ = 1\n    });\n\n    const NriTextureBarrierDesc spatial_read = {\n        .texture = p->history_radiance.texture,\n        .before = compute_storage,\n        .after = history_read,\n        .mipNum = 1, .layerNum = 1, .planes = NriPlaneBits_COLOR\n    };\n    renderer->gpu->core.CmdBarrier(command_buffer, &(NriBarrierDesc){.textures = &spatial_read, .textureNum = 1});\n    p->history_radiance.state = history_read;\n\n    bind_wavefront(renderer, command_buffer, renderer->wavefront_resolve_pipeline);\n    renderer->gpu->core.CmdDispatch(command_buffer, &(NriDispatchDesc){\n        .workGroupNumX = (renderer->screen_probes.width + 7u) / 8u,\n        .workGroupNumY = (renderer->screen_probes.height + 7u) / 8u,\n        .workGroupNumZ = 1\n    });\n}\n''',
    'wavefront temporal/spatial ordering'
)

# Emissive stays compute-visible; history commit performs the final transition for presentation.
s = sub_once(
    s,
    r'''static void build_emissive_gather\(RENDERER \*renderer, NriCommandBuffer \*command_buffer\) \{.*?\n\}\n\nstatic void set_fullscreen_view''',
    '''static void build_emissive_gather(RENDERER *renderer, NriCommandBuffer *command_buffer) {\n    const bool enabled = renderer->radiance_scene.emissive_triangle_count &&\n                         (renderer->radiance_constants.feature_flags[0] & RADIANCE_FEATURE_EMISSIVE);\n    if (!enabled) return;\n\n    bind_wavefront(renderer, command_buffer, renderer->emissive_pipeline);\n    renderer->gpu->core.CmdDispatch(command_buffer, &(NriDispatchDesc){\n        .workGroupNumX = (renderer->screen_probes.width + 7u) / 8u,\n        .workGroupNumY = (renderer->screen_probes.height + 7u) / 8u,\n        .workGroupNumZ = 1\n    });\n}\n\nstatic void commit_probe_history(RENDERER *renderer, NriCommandBuffer *command_buffer) {\n    RADIANCE_PROBES *p = &renderer->probes;\n    const NriAccessLayoutStage compute_storage = {\n        .access = NriAccessBits_SHADER_RESOURCE_STORAGE,\n        .layout = NriLayout_SHADER_RESOURCE_STORAGE,\n        .stages = NriStageBits_COMPUTE_SHADER\n    };\n    const NriAccessLayoutStage history_read = {\n        .access = NriAccessBits_SHADER_RESOURCE,\n        .layout = NriLayout_SHADER_RESOURCE,\n        .stages = NriStageBits_COMPUTE_SHADER\n    };\n    const NriAccessLayoutStage present_read = {\n        .access = NriAccessBits_SHADER_RESOURCE,\n        .layout = NriLayout_SHADER_RESOURCE,\n        .stages = NriStageBits_FRAGMENT_SHADER\n    };\n\n    NriTextureBarrierDesc to_write[] = {\n        {.texture = p->history_meta.texture, .before = p->history_meta.state, .after = compute_storage, .mipNum = 1, .layerNum = 1, .planes = NriPlaneBits_COLOR},\n        {.texture = p->previous_irradiance.texture, .before = p->previous_irradiance.state, .after = compute_storage, .mipNum = 1, .layerNum = 1, .planes = NriPlaneBits_COLOR},\n        {.texture = p->history_depth.texture, .before = p->history_depth.state, .after = compute_storage, .mipNum = 1, .layerNum = 1, .planes = NriPlaneBits_COLOR},\n        {.texture = p->history_normal.texture, .before = p->history_normal.state, .after = compute_storage, .mipNum = 1, .layerNum = 1, .planes = NriPlaneBits_COLOR}\n    };\n    renderer->gpu->core.CmdBarrier(command_buffer, &(NriBarrierDesc){.textures = to_write, .textureNum = 4});\n    p->history_meta.state = compute_storage;\n    p->previous_irradiance.state = compute_storage;\n    p->history_depth.state = compute_storage;\n    p->history_normal.state = compute_storage;\n\n    bind_wavefront(renderer, command_buffer, renderer->wavefront_history_pipeline);\n    renderer->gpu->core.CmdDispatch(command_buffer, &(NriDispatchDesc){\n        .workGroupNumX = (renderer->screen_probes.width + 7u) / 8u,\n        .workGroupNumY = (renderer->screen_probes.height + 7u) / 8u,\n        .workGroupNumZ = 1\n    });\n\n    NriTextureBarrierDesc history_ready[] = {\n        {.texture = p->history_meta.texture, .before = compute_storage, .after = history_read, .mipNum = 1, .layerNum = 1, .planes = NriPlaneBits_COLOR},\n        {.texture = p->previous_irradiance.texture, .before = compute_storage, .after = history_read, .mipNum = 1, .layerNum = 1, .planes = NriPlaneBits_COLOR},\n        {.texture = p->history_depth.texture, .before = compute_storage, .after = history_read, .mipNum = 1, .layerNum = 1, .planes = NriPlaneBits_COLOR},\n        {.texture = p->history_normal.texture, .before = compute_storage, .after = history_read, .mipNum = 1, .layerNum = 1, .planes = NriPlaneBits_COLOR},\n        {.texture = renderer->screen_probes.texture, .before = renderer->screen_probes.state, .after = present_read, .mipNum = 1, .layerNum = 1, .planes = NriPlaneBits_COLOR}\n    };\n    renderer->gpu->core.CmdBarrier(command_buffer, &(NriBarrierDesc){.textures = history_ready, .textureNum = 5});\n    p->history_meta.state = history_read;\n    p->previous_irradiance.state = history_read;\n    p->history_depth.state = history_read;\n    p->history_normal.state = history_read;\n    renderer->screen_probes.state = present_read;\n}\n\nstatic void set_fullscreen_view''',
    'emissive/history commit'
)

s = replace_once(
    s,
    '''            renderer->wavefront_local_pipeline,\n            renderer->wavefront_shade_pipeline,\n            renderer->wavefront_resolve_pipeline,\n            renderer->emissive_pipeline''',
    '''            renderer->wavefront_local_pipeline,\n            renderer->wavefront_shade_pipeline,\n            renderer->wavefront_temporal_pipeline,\n            renderer->wavefront_spatial_pipeline,\n            renderer->wavefront_resolve_pipeline,\n            renderer->emissive_pipeline,\n            renderer->wavefront_history_pipeline''',
    'pipeline destruction'
)

s = replace_once(
    s,
    '''    renderer->previous_view_projection = mat4_identity();\n    renderer->has_previous_frame = false;\n\n    if (!clear_radiance_surface_cache(renderer)) return false;''',
    '''    renderer->previous_view_projection = mat4_identity();\n    renderer->has_previous_frame = false;\n    renderer->probe_history_valid = false;\n\n    if (!clear_radiance_surface_cache(renderer)) return false;''',
    'scene history invalidation'
)

s = replace_once(
    s,
    '''    build_wavefront_screen_probes(renderer, command_buffer);\n    build_emissive_gather(renderer, command_buffer);\n    record_present_pass(renderer, command_buffer, swapchain_index);''',
    '''    build_wavefront_screen_probes(renderer, command_buffer);\n    build_emissive_gather(renderer, command_buffer);\n    commit_probe_history(renderer, command_buffer);\n    record_present_pass(renderer, command_buffer, swapchain_index);''',
    'frame history commit ordering'
)

s = replace_once(
    s,
    '''    renderer->previous_view_projection = view_projection;\n    renderer->has_previous_frame = true;\n    renderer->frame_index = renderer->gpu->frame_index;''',
    '''    renderer->previous_view_projection = view_projection;\n    renderer->has_previous_frame = true;\n    renderer->probe_history_valid = true;\n    renderer->frame_index = renderer->gpu->frame_index;''',
    'history becomes valid after completed frame'
)

p.write_text(s)
