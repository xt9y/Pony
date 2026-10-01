from pathlib import Path
import re


def replace_once(text, old, new, label):
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected 1 match, found {count}")
    return text.replace(old, new, 1)


def sub_once(text, pattern, replacement, label):
    result, count = re.subn(pattern, replacement, text, count=1, flags=re.S)
    if count != 1:
        raise SystemExit(f"{label}: expected 1 match, found {count}")
    return result


# -----------------------------------------------------------------------------
# game.h
# -----------------------------------------------------------------------------
p = Path("game.h")
s = p.read_text()

s = sub_once(
    s,
    r"typedef struct TRACE_BUFFER \{.*?\} TRACE_BUFFER;\n\ntypedef struct TRACE_QUEUE \{.*?\} TRACE_QUEUE;\n\n",
    "",
    "remove legacy trace structs",
)

s = replace_once(
    s,
    "typedef struct COMPUTE_TEXTURE {\n    NriTexture *texture;\n    NriDescriptor *srv;\n    NriDescriptor *uav;\n    NriAccessLayoutStage state;\n    uint32_t width;\n    uint32_t height;\n} COMPUTE_TEXTURE;\n",
    "typedef struct COMPUTE_TEXTURE {\n    NriTexture *texture;\n    NriDescriptor *srv;\n    NriDescriptor *uav;\n    NriAccessLayoutStage state;\n    NriFormat format;\n    uint32_t width;\n    uint32_t height;\n} COMPUTE_TEXTURE;\n\n"
    "typedef struct RADIANCE_PROBES {\n"
    "    COMPUTE_TEXTURE current_radiance;\n"
    "    COMPUTE_TEXTURE current_meta;\n"
    "    COMPUTE_TEXTURE history_radiance;\n"
    "    COMPUTE_TEXTURE history_meta;\n"
    "    COMPUTE_TEXTURE previous_irradiance;\n"
    "    COMPUTE_TEXTURE history_depth;\n"
    "    COMPUTE_TEXTURE history_normal;\n"
    "} RADIANCE_PROBES;\n",
    "probe resources",
)

insert = r'''
typedef struct RADIANCE_WAVEFRONT {
    NriBuffer *queue_a;
    NriBuffer *queue_b;
    NriBuffer *surface_hits;
    NriBuffer *counters;
    NriBuffer *dispatch_args;
    NriBuffer *budgets;
    NriBuffer *update_list;
    NriBuffer *radiance;
    NriBuffer *flags;

    NriDescriptor *queue_a_uav;
    NriDescriptor *queue_b_uav;
    NriDescriptor *surface_hits_uav;
    NriDescriptor *counters_uav;
    NriDescriptor *dispatch_args_uav;
    NriDescriptor *budgets_uav;
    NriDescriptor *update_list_uav;
    NriDescriptor *radiance_uav;
    NriDescriptor *flags_uav;

    NriAccessStage state;
    uint32_t ray_capacity;
    uint32_t probe_capacity;
} RADIANCE_WAVEFRONT;

typedef struct RADIANCE_WORLD_RESOURCES {
    NriBuffer *probes;
    NriBuffer *radiance;
    NriBuffer *keys;
    NriBuffer *invalidation_queue;

    NriDescriptor *probes_uav;
    NriDescriptor *radiance_uav;
    NriDescriptor *keys_uav;
    NriDescriptor *invalidation_queue_uav;

    NriAccessStage state;
} RADIANCE_WORLD_RESOURCES;
'''
marker = "typedef struct RENDER_TEXTURE {"
if s.count(marker) != 1:
    raise SystemExit("render texture marker")
s = s.replace(marker, insert + "\n" + marker, 1)

s = replace_once(
    s,
    "    COMPUTE_TEXTURE direct_radiance;\n    COMPUTE_TEXTURE screen_probe_radiance;\n    COMPUTE_TEXTURE screen_probes;\n    TRACE_BUFFER trace_hits;\n    TRACE_QUEUE miss_queue;\n    SDF_GPU_SCENE sdf;\n    SURFACE_CACHE surface_cache;\n    SURFACE_CACHE radiance_surface_cache;\n",
    "    COMPUTE_TEXTURE direct_radiance;\n    COMPUTE_TEXTURE screen_probes;\n    RADIANCE_PROBES probes;\n    RADIANCE_WAVEFRONT wavefront;\n    RADIANCE_WORLD_RESOURCES world_radiance;\n    SDF_GPU_SCENE sdf;\n    SURFACE_CACHE radiance_surface_cache;\n",
    "renderer resources",
)

s = sub_once(
    s,
    r"    NriDescriptorSet \*trace_set;\n    NriDescriptorSet \*radiance_scene_set;\n    NriDescriptorSet \*radiance_direct_trace_set;\n    NriDescriptorSet \*radiance_direct_scene_set;\n    NriDescriptorSet \*radiance_direct_cache_set;\n    NriDescriptorSet \*emissive_trace_set;\n    NriDescriptorSet \*emissive_scene_set;\n    NriDescriptorSet \*emissive_probe_set;\n",
    "    NriDescriptorSet *trace_set;\n"
    "    NriDescriptorSet *wavefront_trace_set;\n"
    "    NriDescriptorSet *wavefront_scene_set;\n"
    "    NriDescriptorSet *wavefront_queue_set;\n"
    "    NriDescriptorSet *wavefront_cache_set;\n"
    "    NriDescriptorSet *wavefront_probe_set;\n",
    "descriptor sets",
)

s = sub_once(
    s,
    r"    NriPipelineLayout \*trace_layout;\n    NriPipelineLayout \*radiance_scene_layout;\n    NriPipelineLayout \*radiance_direct_layout;\n    NriPipelineLayout \*emissive_layout;\n",
    "    NriPipelineLayout *trace_layout;\n    NriPipelineLayout *wavefront_layout;\n",
    "pipeline layouts",
)

s = sub_once(
    s,
    r"    NriPipeline \*screen_trace_pipeline;\n    NriPipeline \*trace_reset_pipeline;\n    NriPipeline \*trace_compact_pipeline;\n    NriPipeline \*trace_args_pipeline;\n    NriPipeline \*sdf_trace_pipeline;\n    NriPipeline \*direct_radiance_pipeline;\n    NriPipeline \*surface_cache_pipeline;\n    NriPipeline \*screen_probes_pipeline;\n    NriPipeline \*emissive_pipeline;\n",
    "    NriPipeline *direct_radiance_pipeline;\n"
    "    NriPipeline *wavefront_reset_pipeline;\n"
    "    NriPipeline *wavefront_budget_pipeline;\n"
    "    NriPipeline *wavefront_generate_pipeline;\n"
    "    NriPipeline *wavefront_screen_pipeline;\n"
    "    NriPipeline *wavefront_local_pipeline;\n"
    "    NriPipeline *wavefront_shade_pipeline;\n"
    "    NriPipeline *wavefront_resolve_pipeline;\n"
    "    NriPipeline *emissive_pipeline;\n",
    "pipelines",
)

p.write_text(s)


# -----------------------------------------------------------------------------
# shader.hlsl -- keep permanent ABI, only make the frozen wavefront path obey
# the intended direct/emissive decomposition and history feature gates.
# -----------------------------------------------------------------------------
p = Path("shader.hlsl")
s = p.read_text()

s = replace_once(
    s,
    "        RaySurfaceHits[index] = hit;\n        RayRadiance[index] = float4(SurfaceReflectedRadiance(hit), 1.0f);\n        return;\n",
    "        RaySurfaceHits[index] = hit;\n"
    "        uint2 hit_pixel = UnpackPixel(hit.meta.z);\n"
    "        RayRadiance[index] = float4(ReflectedDirectAtPixel(hit_pixel), 1.0f);\n"
    "        return;\n",
    "cheap wavefront screen hits",
)

old = """    float3 radiance;\n    if (hit.identity.w == TRACE_MISS || hit.identity.w == TRACE_INACTIVE) radiance = FutureSkyRadiance(ray.direction_tmax.xyz);\n    else radiance = SurfaceOutgoingRadiance(hit);\n    RayRadiance[index] = float4(radiance, 1.0f);\n"""
new = """    float3 radiance;\n    if (hit.identity.w == TRACE_SCREEN) radiance = RayRadiance[index].rgb;\n    else if (hit.identity.w == TRACE_MISS || hit.identity.w == TRACE_INACTIVE) radiance = FutureSkyRadiance(ray.direction_tmax.xyz);\n    else if (ray.type == TRACE_RAY_DIFFUSE) radiance = SurfaceReflectedRadiance(hit);\n    else radiance = SurfaceOutgoingRadiance(hit);\n    RayRadiance[index] = float4(radiance, 1.0f);\n"""
s = replace_once(s, old, new, "wavefront diffuse decomposition")

old = """    float previous_luma = dot(ProbePreviousIrradiance.Load(int3(probe, 0)).rgb, float3(0.2126f, 0.7152f, 0.0722f));\n    float current_luma = dot(irradiance, float3(0.2126f, 0.7152f, 0.0722f));\n    float variance = abs(current_luma - previous_luma);\n"""
new = """    float variance = 0.0f;\n    if (FeatureEnabled(RADIANCE_FEATURE_TEMPORAL_PROBES)) {\n        float previous_luma = dot(ProbePreviousIrradiance.Load(int3(probe, 0)).rgb, float3(0.2126f, 0.7152f, 0.0722f));\n        float current_luma = dot(irradiance, float3(0.2126f, 0.7152f, 0.0722f));\n        variance = abs(current_luma - previous_luma);\n    }\n"""
s = replace_once(s, old, new, "history gate")
p.write_text(s)


# -----------------------------------------------------------------------------
# build.c -- remove the compatibility compute jobs; permanent/future entries
# continue to compile so the frozen shader ABI stays checked.
# -----------------------------------------------------------------------------
p = Path("build.c")
s = p.read_text()
for line in [
    '        {"CS_SurfaceCacheUpdate", "compute", "surface_cache.cs.spv", NULL},\n',
    '        {"CS_ScreenTrace", "compute", "screen_trace.cs.spv", NULL},\n',
    '        {"CS_ResetTraceQueue", "compute", "trace_reset.cs.spv", NULL},\n',
    '        {"CS_CompactTraceMisses", "compute", "trace_compact.cs.spv", NULL},\n',
    '        {"CS_BuildTraceDispatchArgs", "compute", "trace_args.cs.spv", NULL},\n',
    '        {"CS_SDFTrace", "compute", "sdf_trace.cs.spv", NULL},\n',
    '        {"CS_ScreenProbes", "compute", "screen_probes.cs.spv", NULL},\n',
]:
    if s.count(line) != 1:
        raise SystemExit(f"legacy build job missing: {line.strip()}")
    s = s.replace(line, "", 1)
p.write_text(s)


# -----------------------------------------------------------------------------
# render.c
# -----------------------------------------------------------------------------
p = Path("render.c")
s = p.read_text()
s = s.replace('_Static_assert(sizeof(SURFACE_CACHE_ENTRY) == 112u, "SURFACE_CACHE_ENTRY GPU layout changed");\n', '')

s = sub_once(s, r"static bool clear_surface_cache\(RENDERER \*renderer\) \{.*?\n\}\n\n", "", "remove legacy cache clear")

# Generic typed compute texture creator.
s = sub_once(
    s,
    r"static bool create_compute_texture\(RENDERER \*renderer, COMPUTE_TEXTURE \*target, uint32_t width, uint32_t height\) \{.*?\n\}\n\nstatic void destroy_compute_texture",
    r'''static bool create_compute_texture(RENDERER *renderer, COMPUTE_TEXTURE *target, uint32_t width, uint32_t height, NriFormat format) {
    memset(target, 0, sizeof(*target));
    target->width = width;
    target->height = height;
    target->format = format;

    const NriTextureDesc desc = {
        .type = NriTextureType_TEXTURE_2D,
        .usage = NriTextureUsageBits_SHADER_RESOURCE | NriTextureUsageBits_SHADER_RESOURCE_STORAGE,
        .format = format,
        .width = (NriDim_t)width,
        .height = (NriDim_t)height,
        .depth = 1,
        .mipNum = 1,
        .layerNum = 1,
        .sampleNum = 1
    };

    if (!gpu_create_texture(renderer->gpu, &desc, NriMemoryLocation_DEVICE, &target->texture)) return false;

    const NriTextureViewDesc srv = {
        .texture = target->texture,
        .type = NriTextureView_TEXTURE,
        .format = format,
        .mipNum = 1,
        .layerNum = 1,
        .sliceNum = 1,
        .planes = NriPlaneBits_COLOR
    };

    const NriTextureViewDesc uav = {
        .texture = target->texture,
        .type = NriTextureView_STORAGE_TEXTURE,
        .format = format,
        .mipNum = 1,
        .layerNum = 1,
        .sliceNum = 1,
        .planes = NriPlaneBits_COLOR
    };

    if (renderer->gpu->core.CreateTextureView(&srv, &target->srv) != NriResult_SUCCESS ||
        renderer->gpu->core.CreateTextureView(&uav, &target->uav) != NriResult_SUCCESS)
        return false;

    return true;
}

static void destroy_compute_texture''',
    "typed compute texture",
)

# Remove old queue/cache resource implementations.
s = sub_once(
    s,
    r"static void destroy_trace_buffer\(RENDERER \*renderer, TRACE_BUFFER \*buffer\) \{.*?\n\}\n\nstatic bool create_surface_cache\(RENDERER \*renderer\) \{.*?\n\}\n\n",
    "",
    "remove legacy queue cache resources",
)

# New permanent stage-6 resources.
marker = "static bool clear_radiance_surface_cache(RENDERER *renderer) {"
if s.count(marker) != 1:
    raise SystemExit("radiance cache marker")
new_resources = r'''static bool create_storage_uav(
    RENDERER *renderer,
    uint64_t size,
    uint32_t stride,
    NriBufferUsageBits extra_usage,
    NriBuffer **buffer,
    NriDescriptor **uav
) {
    const NriBufferDesc desc = {
        .size = size,
        .structureStride = stride,
        .usage = NriBufferUsageBits_SHADER_RESOURCE_STORAGE | extra_usage
    };
    if (!gpu_create_buffer(renderer->gpu, &desc, NriMemoryLocation_DEVICE, buffer)) return false;
    if (!create_buffer_view(renderer, *buffer, NriBufferView_STORAGE_STRUCTURED_BUFFER, size, stride, uav)) {
        gpu_destroy_buffer(renderer->gpu, *buffer);
        *buffer = NULL;
        return false;
    }
    return true;
}

static void destroy_storage_uav(RENDERER *renderer, NriBuffer **buffer, NriDescriptor **uav) {
    if (*uav) renderer->gpu->core.DestroyDescriptor(*uav);
    if (*buffer) gpu_destroy_buffer(renderer->gpu, *buffer);
    *uav = NULL;
    *buffer = NULL;
}

static void destroy_wavefront(RENDERER *renderer) {
    RADIANCE_WAVEFRONT *w = &renderer->wavefront;
    destroy_storage_uav(renderer, &w->queue_a, &w->queue_a_uav);
    destroy_storage_uav(renderer, &w->queue_b, &w->queue_b_uav);
    destroy_storage_uav(renderer, &w->surface_hits, &w->surface_hits_uav);
    destroy_storage_uav(renderer, &w->counters, &w->counters_uav);
    destroy_storage_uav(renderer, &w->dispatch_args, &w->dispatch_args_uav);
    destroy_storage_uav(renderer, &w->budgets, &w->budgets_uav);
    destroy_storage_uav(renderer, &w->update_list, &w->update_list_uav);
    destroy_storage_uav(renderer, &w->radiance, &w->radiance_uav);
    destroy_storage_uav(renderer, &w->flags, &w->flags_uav);
    memset(w, 0, sizeof(*w));
}

static bool create_wavefront(RENDERER *renderer, uint32_t probe_capacity) {
    destroy_wavefront(renderer);
    RADIANCE_WAVEFRONT *w = &renderer->wavefront;
    if (!probe_capacity || probe_capacity > UINT32_MAX / SCREEN_PROBE_DIRECTION_COUNT) return false;
    w->probe_capacity = probe_capacity;
    w->ray_capacity = probe_capacity * SCREEN_PROBE_DIRECTION_COUNT;

    if (!create_storage_uav(renderer, (uint64_t)w->ray_capacity * sizeof(TRACE_RAY), sizeof(TRACE_RAY), 0, &w->queue_a, &w->queue_a_uav) ||
        !create_storage_uav(renderer, (uint64_t)w->ray_capacity * sizeof(TRACE_RAY), sizeof(TRACE_RAY), 0, &w->queue_b, &w->queue_b_uav) ||
        !create_storage_uav(renderer, (uint64_t)w->ray_capacity * sizeof(SURFACE_HIT), sizeof(SURFACE_HIT), 0, &w->surface_hits, &w->surface_hits_uav) ||
        !create_storage_uav(renderer, 4u * sizeof(uint32_t), sizeof(uint32_t), 0, &w->counters, &w->counters_uav) ||
        !create_storage_uav(renderer, 3u * sizeof(uint32_t), sizeof(uint32_t), NriBufferUsageBits_ARGUMENT, &w->dispatch_args, &w->dispatch_args_uav) ||
        !create_storage_uav(renderer, (uint64_t)probe_capacity * sizeof(RAY_BUDGET), sizeof(RAY_BUDGET), 0, &w->budgets, &w->budgets_uav) ||
        !create_storage_uav(renderer, (uint64_t)probe_capacity * sizeof(uint32_t), sizeof(uint32_t), 0, &w->update_list, &w->update_list_uav) ||
        !create_storage_uav(renderer, (uint64_t)w->ray_capacity * 4u * sizeof(float), 4u * sizeof(float), 0, &w->radiance, &w->radiance_uav) ||
        !create_storage_uav(renderer, (uint64_t)w->ray_capacity * sizeof(uint32_t), sizeof(uint32_t), 0, &w->flags, &w->flags_uav)) {
        destroy_wavefront(renderer);
        return false;
    }
    return true;
}

static void destroy_world_radiance_resources(RENDERER *renderer) {
    RADIANCE_WORLD_RESOURCES *w = &renderer->world_radiance;
    destroy_storage_uav(renderer, &w->probes, &w->probes_uav);
    destroy_storage_uav(renderer, &w->radiance, &w->radiance_uav);
    destroy_storage_uav(renderer, &w->keys, &w->keys_uav);
    destroy_storage_uav(renderer, &w->invalidation_queue, &w->invalidation_queue_uav);
    memset(w, 0, sizeof(*w));
}

static bool create_world_radiance_resources(RENDERER *renderer) {
    destroy_world_radiance_resources(renderer);
    RADIANCE_WORLD_RESOURCES *w = &renderer->world_radiance;
    if (!create_storage_uav(renderer, sizeof(WORLD_PROBE_STATE), sizeof(WORLD_PROBE_STATE), 0, &w->probes, &w->probes_uav) ||
        !create_storage_uav(renderer, 4u * sizeof(float), 4u * sizeof(float), 0, &w->radiance, &w->radiance_uav) ||
        !create_storage_uav(renderer, sizeof(uint32_t), sizeof(uint32_t), 0, &w->keys, &w->keys_uav) ||
        !create_storage_uav(renderer, sizeof(uint32_t), sizeof(uint32_t), 0, &w->invalidation_queue, &w->invalidation_queue_uav)) {
        destroy_world_radiance_resources(renderer);
        return false;
    }
    return true;
}

static void destroy_radiance_probes(RENDERER *renderer) {
    RADIANCE_PROBES *p = &renderer->probes;
    destroy_compute_texture(renderer, &p->current_radiance);
    destroy_compute_texture(renderer, &p->current_meta);
    destroy_compute_texture(renderer, &p->history_radiance);
    destroy_compute_texture(renderer, &p->history_meta);
    destroy_compute_texture(renderer, &p->previous_irradiance);
    destroy_compute_texture(renderer, &p->history_depth);
    destroy_compute_texture(renderer, &p->history_normal);
}

static bool create_radiance_probes(RENDERER *renderer, uint32_t probe_width, uint32_t probe_height) {
    destroy_radiance_probes(renderer);
    const uint32_t atlas_width = probe_width * SCREEN_PROBE_DIRECTION_SIZE;
    const uint32_t atlas_height = probe_height * SCREEN_PROBE_DIRECTION_SIZE;
    RADIANCE_PROBES *p = &renderer->probes;
    return create_compute_texture(renderer, &p->current_radiance, atlas_width, atlas_height, NriFormat_RGBA16_SFLOAT) &&
           create_compute_texture(renderer, &p->current_meta, probe_width, probe_height, NriFormat_RGBA16_SFLOAT) &&
           create_compute_texture(renderer, &p->history_radiance, atlas_width, atlas_height, NriFormat_RGBA16_SFLOAT) &&
           create_compute_texture(renderer, &p->history_meta, probe_width, probe_height, NriFormat_RGBA16_SFLOAT) &&
           create_compute_texture(renderer, &p->previous_irradiance, probe_width, probe_height, NriFormat_RGBA16_SFLOAT) &&
           create_compute_texture(renderer, &p->history_depth, probe_width, probe_height, NriFormat_R32_SFLOAT) &&
           create_compute_texture(renderer, &p->history_normal, probe_width, probe_height, NriFormat_RGBA16_SFLOAT);
}

'''
s = s.replace(marker, new_resources + marker, 1)

# Queue capacity belongs in frozen feature_flags.z; debug mode lives in reserved.x.
s = replace_once(
    s,
    "    constants.feature_flags[1] = 1u;\n    constants.feature_flags[2] = RADIANCE_DEBUG_FINAL_GI;\n    constants.feature_flags[3] = 0u;\n",
    "    constants.feature_flags[1] = 1u;\n"
    "    constants.feature_flags[2] = renderer->wavefront.ray_capacity;\n"
    "    constants.feature_flags[3] = 0u;\n"
    "    constants.reserved[0] = RADIANCE_DEBUG_FINAL_GI;\n",
    "ray capacity constants",
)

# Direct layout: only inputs + full-resolution output.
trace_pattern = r"    const NriDescriptorRangeDesc trace_ranges\[\] = \{.*?    if \(renderer->gpu->core.CreatePipelineLayout\(renderer->gpu->device, &trace_layout, &renderer->trace_layout\) != NriResult_SUCCESS\) return false;\n"
trace_replacement = r'''    const NriDescriptorRangeDesc trace_ranges[] = {
        {.baseRegisterIndex = 0, .descriptorNum = 7, .descriptorType = NriDescriptorType_TEXTURE, .shaderStages = NriStageBits_COMPUTE_SHADER},
        {.baseRegisterIndex = 7, .descriptorNum = 1, .descriptorType = NriDescriptorType_CONSTANT_BUFFER, .shaderStages = NriStageBits_COMPUTE_SHADER},
        {.baseRegisterIndex = 16, .descriptorNum = 1, .descriptorType = NriDescriptorType_STORAGE_TEXTURE, .shaderStages = NriStageBits_COMPUTE_SHADER}
    };

    const NriDescriptorSetDesc trace_set = {
        .registerSpace = 3,
        .ranges = trace_ranges,
        .rangeNum = 3
    };

    const NriPipelineLayoutDesc trace_layout = {
        .descriptorSets = &trace_set,
        .descriptorSetNum = 1,
        .shaderStages = NriStageBits_COMPUTE_SHADER,
        .flags = NriPipelineLayoutBits_IGNORE_GLOBAL_SPIRV_OFFSETS
    };

    if (renderer->gpu->core.CreatePipelineLayout(renderer->gpu->device, &trace_layout, &renderer->trace_layout) != NriResult_SUCCESS) return false;
'''
s = sub_once(s, trace_pattern, trace_replacement, "direct trace layout")

# Replace the three transitional layouts with one complete wavefront layout.
layout_pattern = r"    const NriPipelineLayoutDesc radiance_scene_layout = \{.*?    return renderer->gpu->core.CreatePipelineLayout\(renderer->gpu->device, &emissive_layout, &renderer->emissive_layout\) == NriResult_SUCCESS;\n\}"
layout_replacement = r'''    const NriDescriptorRangeDesc wavefront_trace_ranges[] = {
        {.baseRegisterIndex = 0, .descriptorNum = 7, .descriptorType = NriDescriptorType_TEXTURE, .shaderStages = NriStageBits_COMPUTE_SHADER},
        {.baseRegisterIndex = 7, .descriptorNum = 1, .descriptorType = NriDescriptorType_CONSTANT_BUFFER, .shaderStages = NriStageBits_COMPUTE_SHADER}
    };
    const NriDescriptorSetDesc wavefront_trace_set = {
        .registerSpace = 3,
        .ranges = wavefront_trace_ranges,
        .rangeNum = 2
    };

    const NriDescriptorRangeDesc wavefront_queue_range = {
        .baseRegisterIndex = 0, .descriptorNum = 9,
        .descriptorType = NriDescriptorType_STORAGE_STRUCTURED_BUFFER,
        .shaderStages = NriStageBits_COMPUTE_SHADER
    };
    const NriDescriptorSetDesc wavefront_queue_set = {
        .registerSpace = 5, .ranges = &wavefront_queue_range, .rangeNum = 1
    };

    const NriDescriptorRangeDesc wavefront_cache_range = {
        .baseRegisterIndex = 0, .descriptorNum = 6,
        .descriptorType = NriDescriptorType_STORAGE_STRUCTURED_BUFFER,
        .shaderStages = NriStageBits_COMPUTE_SHADER
    };
    const NriDescriptorSetDesc wavefront_cache_set = {
        .registerSpace = 6, .ranges = &wavefront_cache_range, .rangeNum = 1
    };

    const NriDescriptorRangeDesc wavefront_probe_ranges[] = {
        {.baseRegisterIndex = 0, .descriptorNum = 5, .descriptorType = NriDescriptorType_TEXTURE, .shaderStages = NriStageBits_COMPUTE_SHADER},
        {.baseRegisterIndex = 5, .descriptorNum = 3, .descriptorType = NriDescriptorType_STORAGE_TEXTURE, .shaderStages = NriStageBits_COMPUTE_SHADER}
    };
    const NriDescriptorSetDesc wavefront_probe_set = {
        .registerSpace = 7, .ranges = wavefront_probe_ranges, .rangeNum = 2
    };

    const NriDescriptorSetDesc wavefront_sets[] = {
        wavefront_trace_set,
        radiance_scene_set,
        wavefront_queue_set,
        wavefront_cache_set,
        wavefront_probe_set
    };
    const NriPipelineLayoutDesc wavefront_layout = {
        .descriptorSets = wavefront_sets,
        .descriptorSetNum = 5,
        .shaderStages = NriStageBits_COMPUTE_SHADER,
        .flags = NriPipelineLayoutBits_IGNORE_GLOBAL_SPIRV_OFFSETS
    };

    return renderer->gpu->core.CreatePipelineLayout(renderer->gpu->device, &wavefront_layout, &renderer->wavefront_layout) == NriResult_SUCCESS;
}'''
s = sub_once(s, layout_pattern, layout_replacement, "wavefront layout")

# Descriptor pool and sets.
s = sub_once(
    s,
    r"static bool create_descriptor_pool\(RENDERER \*renderer\) \{.*?\n\}\n\nstatic bool create_compute_pipeline",
    r'''static bool create_descriptor_pool(RENDERER *renderer) {
    const NriDescriptorPoolDesc desc = {
        .descriptorSetMaxNum = 8 + HZB_MAX_MIPS,
        .constantBufferMaxNum = 12,
        .textureMaxNum = 96,
        .storageTextureMaxNum = HZB_MAX_MIPS + 16,
        .structuredBufferMaxNum = 64,
        .storageStructuredBufferMaxNum = 32
    };

    if (renderer->gpu->core.CreateDescriptorPool(renderer->gpu->device, &desc, &renderer->descriptor_pool) != NriResult_SUCCESS) return false;

    return renderer->gpu->core.AllocateDescriptorSets(renderer->descriptor_pool, renderer->gbuffer_layout, 0, &renderer->gbuffer_set, 1, 0) == NriResult_SUCCESS &&
           renderer->gpu->core.AllocateDescriptorSets(renderer->descriptor_pool, renderer->present_layout, 0, &renderer->present_set, 1, 0) == NriResult_SUCCESS &&
           renderer->gpu->core.AllocateDescriptorSets(renderer->descriptor_pool, renderer->hzb_layout, 0, renderer->hzb_sets, HZB_MAX_MIPS, 0) == NriResult_SUCCESS &&
           renderer->gpu->core.AllocateDescriptorSets(renderer->descriptor_pool, renderer->trace_layout, 0, &renderer->trace_set, 1, 0) == NriResult_SUCCESS &&
           renderer->gpu->core.AllocateDescriptorSets(renderer->descriptor_pool, renderer->wavefront_layout, 0, &renderer->wavefront_trace_set, 1, 0) == NriResult_SUCCESS &&
           renderer->gpu->core.AllocateDescriptorSets(renderer->descriptor_pool, renderer->wavefront_layout, 1, &renderer->wavefront_scene_set, 1, 0) == NriResult_SUCCESS &&
           renderer->gpu->core.AllocateDescriptorSets(renderer->descriptor_pool, renderer->wavefront_layout, 2, &renderer->wavefront_queue_set, 1, 0) == NriResult_SUCCESS &&
           renderer->gpu->core.AllocateDescriptorSets(renderer->descriptor_pool, renderer->wavefront_layout, 3, &renderer->wavefront_cache_set, 1, 0) == NriResult_SUCCESS &&
           renderer->gpu->core.AllocateDescriptorSets(renderer->descriptor_pool, renderer->wavefront_layout, 4, &renderer->wavefront_probe_set, 1, 0) == NriResult_SUCCESS;
}

static bool create_compute_pipeline''',
    "descriptor pool",
)

# Only the active direct + stage6 wavefront pipelines.
old_chain = '''    return create_compute_pipeline(renderer, "build/shaders/hzb.cs.spv", renderer->hzb_layout, &renderer->hzb_pipeline) &&
           create_compute_pipeline(renderer, "build/shaders/radiance_direct.cs.spv", renderer->trace_layout, &renderer->direct_radiance_pipeline) &&
           create_compute_pipeline(renderer, "build/shaders/surface_cache.cs.spv", renderer->trace_layout, &renderer->surface_cache_pipeline) &&
           create_compute_pipeline(renderer, "build/shaders/screen_trace.cs.spv", renderer->trace_layout, &renderer->screen_trace_pipeline) &&
           create_compute_pipeline(renderer, "build/shaders/trace_reset.cs.spv", renderer->trace_layout, &renderer->trace_reset_pipeline) &&
           create_compute_pipeline(renderer, "build/shaders/trace_compact.cs.spv", renderer->trace_layout, &renderer->trace_compact_pipeline) &&
           create_compute_pipeline(renderer, "build/shaders/trace_args.cs.spv", renderer->trace_layout, &renderer->trace_args_pipeline) &&
           create_compute_pipeline(renderer, "build/shaders/sdf_trace.cs.spv", renderer->radiance_direct_layout, &renderer->sdf_trace_pipeline) &&
           create_compute_pipeline(renderer, "build/shaders/screen_probes.cs.spv", renderer->trace_layout, &renderer->screen_probes_pipeline) &&
           create_compute_pipeline(renderer, "build/shaders/radiance_emissive.cs.spv", renderer->emissive_layout, &renderer->emissive_pipeline);'''
new_chain = '''    return create_compute_pipeline(renderer, "build/shaders/hzb.cs.spv", renderer->hzb_layout, &renderer->hzb_pipeline) &&
           create_compute_pipeline(renderer, "build/shaders/radiance_direct.cs.spv", renderer->trace_layout, &renderer->direct_radiance_pipeline) &&
           create_compute_pipeline(renderer, "build/shaders/radiance_wave_reset.cs.spv", renderer->wavefront_layout, &renderer->wavefront_reset_pipeline) &&
           create_compute_pipeline(renderer, "build/shaders/radiance_budget.cs.spv", renderer->wavefront_layout, &renderer->wavefront_budget_pipeline) &&
           create_compute_pipeline(renderer, "build/shaders/radiance_generate.cs.spv", renderer->wavefront_layout, &renderer->wavefront_generate_pipeline) &&
           create_compute_pipeline(renderer, "build/shaders/radiance_screen.cs.spv", renderer->wavefront_layout, &renderer->wavefront_screen_pipeline) &&
           create_compute_pipeline(renderer, "build/shaders/radiance_local.cs.spv", renderer->wavefront_layout, &renderer->wavefront_local_pipeline) &&
           create_compute_pipeline(renderer, "build/shaders/radiance_shade.cs.spv", renderer->wavefront_layout, &renderer->wavefront_shade_pipeline) &&
           create_compute_pipeline(renderer, "build/shaders/radiance_probe_resolve.cs.spv", renderer->wavefront_layout, &renderer->wavefront_resolve_pipeline) &&
           create_compute_pipeline(renderer, "build/shaders/radiance_emissive.cs.spv", renderer->wavefront_layout, &renderer->emissive_pipeline);'''
s = replace_once(s, old_chain, new_chain, "pipeline chain")

# Descriptor updates for one active direct set and one complete wavefront set.
s = sub_once(
    s,
    r"static bool update_trace_descriptor_set\(RENDERER \*renderer, NriDescriptorSet \*descriptor_set\) \{.*?\n\}\n\nstatic bool create_size_dependent_resources",
    r'''static bool update_trace_descriptors(RENDERER *renderer) {
    if (!renderer->trace_set || !renderer->depth.srv || !renderer->normal_roughness.srv || !renderer->albedo_metallic.srv || !renderer->emissive.srv ||
        !renderer->hzb.srv || !renderer->object_id.srv || !renderer->direct_radiance.srv || !renderer->frame_srv || !renderer->direct_radiance.uav)
        return false;

    const NriDescriptor *textures[] = {
        renderer->depth.srv, renderer->normal_roughness.srv, renderer->albedo_metallic.srv,
        renderer->emissive.srv, renderer->hzb.srv, renderer->object_id.srv, renderer->direct_radiance.srv
    };
    const NriDescriptor *frame[] = {renderer->frame_srv};
    const NriDescriptor *output[] = {renderer->direct_radiance.uav};
    const NriUpdateDescriptorRangeDesc updates[] = {
        {.descriptorSet = renderer->trace_set, .rangeIndex = 0, .descriptors = textures, .descriptorNum = 7},
        {.descriptorSet = renderer->trace_set, .rangeIndex = 1, .descriptors = frame, .descriptorNum = 1},
        {.descriptorSet = renderer->trace_set, .rangeIndex = 2, .descriptors = output, .descriptorNum = 1}
    };
    renderer->gpu->core.UpdateDescriptorRanges(updates, 3);
    return true;
}

static bool update_wavefront_trace_descriptors(RENDERER *renderer) {
    if (!renderer->wavefront_trace_set || !renderer->depth.srv || !renderer->normal_roughness.srv || !renderer->albedo_metallic.srv || !renderer->emissive.srv ||
        !renderer->hzb.srv || !renderer->object_id.srv || !renderer->direct_radiance.srv || !renderer->frame_srv)
        return false;
    const NriDescriptor *textures[] = {
        renderer->depth.srv, renderer->normal_roughness.srv, renderer->albedo_metallic.srv,
        renderer->emissive.srv, renderer->hzb.srv, renderer->object_id.srv, renderer->direct_radiance.srv
    };
    const NriDescriptor *frame[] = {renderer->frame_srv};
    const NriUpdateDescriptorRangeDesc updates[] = {
        {.descriptorSet = renderer->wavefront_trace_set, .rangeIndex = 0, .descriptors = textures, .descriptorNum = 7},
        {.descriptorSet = renderer->wavefront_trace_set, .rangeIndex = 1, .descriptors = frame, .descriptorNum = 1}
    };
    renderer->gpu->core.UpdateDescriptorRanges(updates, 2);
    return true;
}

static bool update_radiance_scene_descriptors(RENDERER *renderer) {
    if (!renderer || !renderer->wavefront_scene_set || !renderer->radiance_constants_srv || !renderer->pass_constants_srv || !renderer->object_srv || !renderer->material_srv ||
        !renderer->radiance_scene.triangles_srv || !renderer->radiance_scene.emissive_triangles_srv || !renderer->sdf.models_srv || !renderer->sdf.voxels_srv ||
        !renderer->sdf.surface_ids_srv || !renderer->radiance_fallbacks.dynamic_grid_cells_srv || !renderer->radiance_fallbacks.dynamic_grid_indices_srv ||
        !renderer->radiance_fallbacks.global_sdf_clipmaps_srv || !renderer->radiance_fallbacks.global_sdf_page_table_srv || !renderer->radiance_fallbacks.global_sdf_bricks_srv ||
        !renderer->radiance_fallbacks.global_sdf_surface_ids_srv || !renderer->light_srv || !renderer->material_id.srv || !renderer->primitive_id.srv)
        return false;

    const NriDescriptor *constants[] = {renderer->radiance_constants_srv, renderer->pass_constants_srv};
    const NriDescriptor *scene_core[] = {
        renderer->object_srv, renderer->material_srv, renderer->radiance_scene.triangles_srv,
        renderer->radiance_scene.emissive_triangles_srv, renderer->sdf.models_srv, renderer->sdf.voxels_srv
    };
    const NriDescriptor *future_scene[] = {
        renderer->sdf.surface_ids_srv, renderer->radiance_fallbacks.dynamic_grid_cells_srv,
        renderer->radiance_fallbacks.dynamic_grid_indices_srv, renderer->radiance_fallbacks.global_sdf_clipmaps_srv,
        renderer->radiance_fallbacks.global_sdf_page_table_srv, renderer->radiance_fallbacks.global_sdf_bricks_srv,
        renderer->radiance_fallbacks.global_sdf_surface_ids_srv
    };
    const NriDescriptor *lights[] = {renderer->light_srv};
    const NriDescriptor *identity[] = {renderer->material_id.srv, renderer->primitive_id.srv};
    const NriUpdateDescriptorRangeDesc updates[] = {
        {.descriptorSet = renderer->wavefront_scene_set, .rangeIndex = 0, .descriptors = constants, .descriptorNum = 2},
        {.descriptorSet = renderer->wavefront_scene_set, .rangeIndex = 1, .descriptors = scene_core, .descriptorNum = 6},
        {.descriptorSet = renderer->wavefront_scene_set, .rangeIndex = 2, .descriptors = future_scene, .descriptorNum = 7},
        {.descriptorSet = renderer->wavefront_scene_set, .rangeIndex = 3, .descriptors = lights, .descriptorNum = 1},
        {.descriptorSet = renderer->wavefront_scene_set, .rangeIndex = 4, .descriptors = identity, .descriptorNum = 2}
    };
    renderer->gpu->core.UpdateDescriptorRanges(updates, 5);
    return true;
}

static bool update_wavefront_descriptors(RENDERER *renderer) {
    RADIANCE_WAVEFRONT *w = &renderer->wavefront;
    RADIANCE_WORLD_RESOURCES *world = &renderer->world_radiance;
    RADIANCE_PROBES *p = &renderer->probes;
    if (!update_wavefront_trace_descriptors(renderer)) return false;

    const NriDescriptor *queues[] = {
        w->queue_a_uav, w->queue_b_uav, w->surface_hits_uav, w->counters_uav, w->dispatch_args_uav,
        w->budgets_uav, w->update_list_uav, w->radiance_uav, w->flags_uav
    };
    const NriDescriptor *cache[] = {
        renderer->radiance_surface_cache.keys_uav, renderer->radiance_surface_cache.entries_uav,
        world->probes_uav, world->radiance_uav, world->keys_uav, world->invalidation_queue_uav
    };
    const NriDescriptor *history[] = {
        p->history_radiance.srv, p->history_meta.srv, p->previous_irradiance.srv,
        p->history_depth.srv, p->history_normal.srv
    };
    const NriDescriptor *current[] = {p->current_radiance.uav, p->current_meta.uav, renderer->screen_probes.uav};
    for (uint32_t i = 0; i < 9; ++i) if (!queues[i]) return false;
    for (uint32_t i = 0; i < 6; ++i) if (!cache[i]) return false;
    for (uint32_t i = 0; i < 5; ++i) if (!history[i]) return false;
    for (uint32_t i = 0; i < 3; ++i) if (!current[i]) return false;

    const NriUpdateDescriptorRangeDesc updates[] = {
        {.descriptorSet = renderer->wavefront_queue_set, .rangeIndex = 0, .descriptors = queues, .descriptorNum = 9},
        {.descriptorSet = renderer->wavefront_cache_set, .rangeIndex = 0, .descriptors = cache, .descriptorNum = 6},
        {.descriptorSet = renderer->wavefront_probe_set, .rangeIndex = 0, .descriptors = history, .descriptorNum = 5},
        {.descriptorSet = renderer->wavefront_probe_set, .rangeIndex = 1, .descriptors = current, .descriptorNum = 3}
    };
    renderer->gpu->core.UpdateDescriptorRanges(updates, 4);
    return true;
}

static bool create_size_dependent_resources''',
    "descriptor updates",
)

# Size dependent probe + wavefront resources.
s = sub_once(
    s,
    r"static bool create_size_dependent_resources\(RENDERER \*renderer, uint32_t width, uint32_t height\) \{.*?\n\}\n\nstatic bool update_scene_objects",
    r'''static bool create_size_dependent_resources(RENDERER *renderer, uint32_t width, uint32_t height) {
    destroy_compute_texture(renderer, &renderer->direct_radiance);
    destroy_compute_texture(renderer, &renderer->screen_probes);
    destroy_radiance_probes(renderer);
    destroy_wavefront(renderer);

    const uint32_t probe_width = (width + SCREEN_PROBE_TILE_SIZE - 1u) / SCREEN_PROBE_TILE_SIZE;
    const uint32_t probe_height = (height + SCREEN_PROBE_TILE_SIZE - 1u) / SCREEN_PROBE_TILE_SIZE;
    const uint32_t probe_count = probe_width * probe_height;
    const uint32_t atlas_width = probe_width * SCREEN_PROBE_DIRECTION_SIZE;
    const uint32_t atlas_height = probe_height * SCREEN_PROBE_DIRECTION_SIZE;

    if (!create_gbuffer(renderer, width, height) || !create_hzb(renderer, width, height) ||
        !create_screen_trace(renderer, atlas_width, atlas_height) ||
        !create_compute_texture(renderer, &renderer->direct_radiance, width, height, NriFormat_RGBA16_SFLOAT) ||
        !create_compute_texture(renderer, &renderer->screen_probes, probe_width, probe_height, NriFormat_RGBA16_SFLOAT) ||
        !create_radiance_probes(renderer, probe_width, probe_height) ||
        !create_wavefront(renderer, probe_count))
        return false;

    update_hzb_descriptors(renderer);
    update_present_descriptors(renderer);
    if (!update_trace_descriptors(renderer) || !update_wavefront_descriptors(renderer)) return false;
    if (renderer->radiance_scene.triangles_srv && !update_radiance_scene_descriptors(renderer)) return false;
    return true;
}

static bool update_scene_objects''',
    "size resources",
)

# Replace transitional bind with the permanent wavefront bind.
s = sub_once(
    s,
    r"static void bind_radiance_world\(RENDERER \*renderer, NriCommandBuffer \*command_buffer, NriPipeline \*pipeline\) \{.*?\n\}\n\n",
    r'''static void bind_wavefront(RENDERER *renderer, NriCommandBuffer *command_buffer, NriPipeline *pipeline) {
    renderer->gpu->core.CmdSetPipelineLayout(command_buffer, NriBindPoint_COMPUTE, renderer->wavefront_layout);
    renderer->gpu->core.CmdSetPipeline(command_buffer, pipeline);
    const NriSetDescriptorSetDesc sets[] = {
        {.setIndex = 0, .descriptorSet = renderer->wavefront_trace_set, .bindPoint = NriBindPoint_COMPUTE},
        {.setIndex = 1, .descriptorSet = renderer->wavefront_scene_set, .bindPoint = NriBindPoint_COMPUTE},
        {.setIndex = 2, .descriptorSet = renderer->wavefront_queue_set, .bindPoint = NriBindPoint_COMPUTE},
        {.setIndex = 3, .descriptorSet = renderer->wavefront_cache_set, .bindPoint = NriBindPoint_COMPUTE},
        {.setIndex = 4, .descriptorSet = renderer->wavefront_probe_set, .bindPoint = NriBindPoint_COMPUTE}
    };
    for (uint32_t i = 0; i < sizeof(sets) / sizeof(sets[0]); ++i)
        renderer->gpu->core.CmdSetDescriptorSet(command_buffer, &sets[i]);
}

''',
    "wavefront bind",
)

# Replace every legacy probe/queue pass with one wavefront stage.
pass_pattern = r"static void build_surface_cache\(RENDERER \*renderer, NriCommandBuffer \*command_buffer\) \{.*?\n\}\n\nstatic void set_fullscreen_view"
pass_replacement = r'''static void barrier_wavefront_buffers(RENDERER *renderer, NriCommandBuffer *command_buffer, NriAccessStage state) {
    RADIANCE_WAVEFRONT *w = &renderer->wavefront;
    NriBufferBarrierDesc barriers[] = {
        {.buffer = w->queue_a, .before = w->state, .after = state},
        {.buffer = w->queue_b, .before = w->state, .after = state},
        {.buffer = w->surface_hits, .before = w->state, .after = state},
        {.buffer = w->counters, .before = w->state, .after = state},
        {.buffer = w->dispatch_args, .before = w->state, .after = state},
        {.buffer = w->budgets, .before = w->state, .after = state},
        {.buffer = w->update_list, .before = w->state, .after = state},
        {.buffer = w->radiance, .before = w->state, .after = state},
        {.buffer = w->flags, .before = w->state, .after = state}
    };
    renderer->gpu->core.CmdBarrier(command_buffer, &(NriBarrierDesc){.buffers = barriers, .bufferNum = 9});
    w->state = state;
}

static void build_wavefront_screen_probes(RENDERER *renderer, NriCommandBuffer *command_buffer) {
    const NriAccessStage storage = {
        .access = NriAccessBits_SHADER_RESOURCE_STORAGE,
        .stages = NriStageBits_COMPUTE_SHADER
    };
    const NriAccessStage cache_storage = storage;
    const NriAccessLayoutStage texture_storage = {
        .access = NriAccessBits_SHADER_RESOURCE_STORAGE,
        .layout = NriLayout_SHADER_RESOURCE_STORAGE,
        .stages = NriStageBits_COMPUTE_SHADER | NriStageBits_CLEAR_STORAGE
    };
    const NriAccessLayoutStage history_read = {
        .access = NriAccessBits_SHADER_RESOURCE,
        .layout = NriLayout_SHADER_RESOURCE,
        .stages = NriStageBits_COMPUTE_SHADER
    };

    RADIANCE_PROBES *p = &renderer->probes;
    NriTextureBarrierDesc output_barriers[] = {
        {.texture = p->current_radiance.texture, .before = p->current_radiance.state, .after = texture_storage, .mipNum = 1, .layerNum = 1, .planes = NriPlaneBits_COLOR},
        {.texture = p->current_meta.texture, .before = p->current_meta.state, .after = texture_storage, .mipNum = 1, .layerNum = 1, .planes = NriPlaneBits_COLOR},
        {.texture = renderer->screen_probes.texture, .before = renderer->screen_probes.state, .after = texture_storage, .mipNum = 1, .layerNum = 1, .planes = NriPlaneBits_COLOR}
    };
    NriTextureBarrierDesc history_barriers[] = {
        {.texture = p->history_radiance.texture, .before = p->history_radiance.state, .after = history_read, .mipNum = 1, .layerNum = 1, .planes = NriPlaneBits_COLOR},
        {.texture = p->history_meta.texture, .before = p->history_meta.state, .after = history_read, .mipNum = 1, .layerNum = 1, .planes = NriPlaneBits_COLOR},
        {.texture = p->previous_irradiance.texture, .before = p->previous_irradiance.state, .after = history_read, .mipNum = 1, .layerNum = 1, .planes = NriPlaneBits_COLOR},
        {.texture = p->history_depth.texture, .before = p->history_depth.state, .after = history_read, .mipNum = 1, .layerNum = 1, .planes = NriPlaneBits_COLOR},
        {.texture = p->history_normal.texture, .before = p->history_normal.state, .after = history_read, .mipNum = 1, .layerNum = 1, .planes = NriPlaneBits_COLOR}
    };
    renderer->gpu->core.CmdBarrier(command_buffer, &(NriBarrierDesc){
        .textures = output_barriers, .textureNum = 3
    });
    renderer->gpu->core.CmdBarrier(command_buffer, &(NriBarrierDesc){
        .textures = history_barriers, .textureNum = 5
    });
    p->current_radiance.state = texture_storage;
    p->current_meta.state = texture_storage;
    renderer->screen_probes.state = texture_storage;
    p->history_radiance.state = history_read;
    p->history_meta.state = history_read;
    p->previous_irradiance.state = history_read;
    p->history_depth.state = history_read;
    p->history_normal.state = history_read;

    barrier_wavefront_buffers(renderer, command_buffer, storage);

    NriBufferBarrierDesc cache_barriers[] = {
        {.buffer = renderer->radiance_surface_cache.keys, .before = renderer->radiance_surface_cache.keys_state, .after = cache_storage},
        {.buffer = renderer->radiance_surface_cache.entries, .before = renderer->radiance_surface_cache.entries_state, .after = cache_storage},
        {.buffer = renderer->world_radiance.probes, .before = renderer->world_radiance.state, .after = cache_storage},
        {.buffer = renderer->world_radiance.radiance, .before = renderer->world_radiance.state, .after = cache_storage},
        {.buffer = renderer->world_radiance.keys, .before = renderer->world_radiance.state, .after = cache_storage},
        {.buffer = renderer->world_radiance.invalidation_queue, .before = renderer->world_radiance.state, .after = cache_storage}
    };
    renderer->gpu->core.CmdBarrier(command_buffer, &(NriBarrierDesc){.buffers = cache_barriers, .bufferNum = 6});
    renderer->radiance_surface_cache.keys_state = cache_storage;
    renderer->radiance_surface_cache.entries_state = cache_storage;
    renderer->world_radiance.state = cache_storage;

    bind_wavefront(renderer, command_buffer, renderer->wavefront_reset_pipeline);
    const NriClearStorageDesc zero_current = {
        .descriptor = p->current_radiance.uav,
        .value = {.color = {.f = {.x = 0.0f, .y = 0.0f, .z = 0.0f, .w = 0.0f}}},
        .setIndex = 4
    };
    const NriClearStorageDesc zero_meta = {
        .descriptor = p->current_meta.uav,
        .value = {.color = {.f = {.x = 0.0f, .y = 0.0f, .z = 0.0f, .w = 0.0f}}},
        .setIndex = 4
    };
    const NriClearStorageDesc zero_irradiance = {
        .descriptor = renderer->screen_probes.uav,
        .value = {.color = {.f = {.x = 0.0f, .y = 0.0f, .z = 0.0f, .w = 0.0f}}},
        .setIndex = 4
    };
    renderer->gpu->core.CmdClearStorage(command_buffer, &zero_current);
    renderer->gpu->core.CmdClearStorage(command_buffer, &zero_meta);
    renderer->gpu->core.CmdClearStorage(command_buffer, &zero_irradiance);

    const NriAccessLayoutStage compute_storage = {
        .access = NriAccessBits_SHADER_RESOURCE_STORAGE,
        .layout = NriLayout_SHADER_RESOURCE_STORAGE,
        .stages = NriStageBits_COMPUTE_SHADER
    };
    NriTextureBarrierDesc clear_sync[] = {
        {.texture = p->current_radiance.texture, .before = texture_storage, .after = compute_storage, .mipNum = 1, .layerNum = 1, .planes = NriPlaneBits_COLOR},
        {.texture = p->current_meta.texture, .before = texture_storage, .after = compute_storage, .mipNum = 1, .layerNum = 1, .planes = NriPlaneBits_COLOR},
        {.texture = renderer->screen_probes.texture, .before = texture_storage, .after = compute_storage, .mipNum = 1, .layerNum = 1, .planes = NriPlaneBits_COLOR}
    };
    renderer->gpu->core.CmdBarrier(command_buffer, &(NriBarrierDesc){.textures = clear_sync, .textureNum = 3});
    p->current_radiance.state = compute_storage;
    p->current_meta.state = compute_storage;
    renderer->screen_probes.state = compute_storage;

    renderer->gpu->core.CmdDispatch(command_buffer, &(NriDispatchDesc){.workGroupNumX = 1, .workGroupNumY = 1, .workGroupNumZ = 1});
    barrier_wavefront_buffers(renderer, command_buffer, storage);

    bind_wavefront(renderer, command_buffer, renderer->wavefront_budget_pipeline);
    renderer->gpu->core.CmdDispatch(command_buffer, &(NriDispatchDesc){
        .workGroupNumX = (renderer->screen_probes.width + 7u) / 8u,
        .workGroupNumY = (renderer->screen_probes.height + 7u) / 8u,
        .workGroupNumZ = 1
    });
    barrier_wavefront_buffers(renderer, command_buffer, storage);

    bind_wavefront(renderer, command_buffer, renderer->wavefront_generate_pipeline);
    renderer->gpu->core.CmdDispatch(command_buffer, &(NriDispatchDesc){
        .workGroupNumX = (renderer->screen_probes.width + 7u) / 8u,
        .workGroupNumY = (renderer->screen_probes.height + 7u) / 8u,
        .workGroupNumZ = 1
    });
    barrier_wavefront_buffers(renderer, command_buffer, storage);

    const uint32_t ray_groups = (renderer->wavefront.ray_capacity + 63u) / 64u;
    bind_wavefront(renderer, command_buffer, renderer->wavefront_screen_pipeline);
    renderer->gpu->core.CmdDispatch(command_buffer, &(NriDispatchDesc){.workGroupNumX = ray_groups, .workGroupNumY = 1, .workGroupNumZ = 1});
    barrier_wavefront_buffers(renderer, command_buffer, storage);

    bind_wavefront(renderer, command_buffer, renderer->wavefront_local_pipeline);
    renderer->gpu->core.CmdDispatch(command_buffer, &(NriDispatchDesc){.workGroupNumX = ray_groups, .workGroupNumY = 1, .workGroupNumZ = 1});
    barrier_wavefront_buffers(renderer, command_buffer, storage);

    bind_wavefront(renderer, command_buffer, renderer->wavefront_shade_pipeline);
    renderer->gpu->core.CmdDispatch(command_buffer, &(NriDispatchDesc){.workGroupNumX = ray_groups, .workGroupNumY = 1, .workGroupNumZ = 1});
    barrier_wavefront_buffers(renderer, command_buffer, storage);

    const NriTextureBarrierDesc probe_sync = {
        .texture = p->current_radiance.texture,
        .before = p->current_radiance.state,
        .after = compute_storage,
        .mipNum = 1, .layerNum = 1, .planes = NriPlaneBits_COLOR
    };
    renderer->gpu->core.CmdBarrier(command_buffer, &(NriBarrierDesc){.textures = &probe_sync, .textureNum = 1});

    bind_wavefront(renderer, command_buffer, renderer->wavefront_resolve_pipeline);
    renderer->gpu->core.CmdDispatch(command_buffer, &(NriDispatchDesc){
        .workGroupNumX = (renderer->screen_probes.width + 7u) / 8u,
        .workGroupNumY = (renderer->screen_probes.height + 7u) / 8u,
        .workGroupNumZ = 1
    });
}

static void build_emissive_gather(RENDERER *renderer, NriCommandBuffer *command_buffer) {
    const bool enabled = renderer->radiance_scene.emissive_triangle_count &&
                         (renderer->radiance_constants.feature_flags[0] & RADIANCE_FEATURE_EMISSIVE);
    if (enabled) {
        bind_wavefront(renderer, command_buffer, renderer->emissive_pipeline);
        renderer->gpu->core.CmdDispatch(command_buffer, &(NriDispatchDesc){
            .workGroupNumX = (renderer->screen_probes.width + 7u) / 8u,
            .workGroupNumY = (renderer->screen_probes.height + 7u) / 8u,
            .workGroupNumZ = 1
        });
    }

    const NriAccessLayoutStage read = {
        .access = NriAccessBits_SHADER_RESOURCE,
        .layout = NriLayout_SHADER_RESOURCE,
        .stages = NriStageBits_FRAGMENT_SHADER
    };
    const NriTextureBarrierDesc to_read = {
        .texture = renderer->screen_probes.texture,
        .before = renderer->screen_probes.state,
        .after = read,
        .mipNum = 1, .layerNum = 1, .planes = NriPlaneBits_COLOR
    };
    renderer->gpu->core.CmdBarrier(command_buffer, &(NriBarrierDesc){.textures = &to_read, .textureNum = 1});
    renderer->screen_probes.state = read;
}

static void set_fullscreen_view'''
s = sub_once(s, pass_pattern, pass_replacement, "wavefront passes")

# Init/deinit/set-scene/resource ownership.
s = replace_once(
    s,
    "        !create_radiance_scene_fallbacks(renderer) || !create_pipelines(renderer) || !create_surface_cache(renderer) ||\n        !create_radiance_surface_cache(renderer) || !update_radiance_direct_cache_descriptors(renderer) ||\n        !create_size_dependent_resources(renderer, gpu->swapchain_width, gpu->swapchain_height)) {",
    "        !create_radiance_scene_fallbacks(renderer) || !create_world_radiance_resources(renderer) || !create_pipelines(renderer) ||\n"
    "        !create_radiance_surface_cache(renderer) ||\n"
    "        !create_size_dependent_resources(renderer, gpu->swapchain_width, gpu->swapchain_height)) {",
    "renderer init",
)

# Pipeline destruction list.
s = sub_once(
    s,
    r"        NriPipeline \*pipelines\[\] = \{.*?        \};",
    r'''        NriPipeline *pipelines[] = {
            renderer->gbuffer_pipeline,
            renderer->present_pipeline,
            renderer->hzb_pipeline,
            renderer->direct_radiance_pipeline,
            renderer->wavefront_reset_pipeline,
            renderer->wavefront_budget_pipeline,
            renderer->wavefront_generate_pipeline,
            renderer->wavefront_screen_pipeline,
            renderer->wavefront_local_pipeline,
            renderer->wavefront_shade_pipeline,
            renderer->wavefront_resolve_pipeline,
            renderer->emissive_pipeline
        };''',
    "pipeline destruction",
)

for old in [
    "        destroy_trace_queue(renderer);\n",
    "        destroy_trace_buffer(renderer, &renderer->trace_hits);\n",
    "        destroy_surface_cache(renderer);\n",
    "        destroy_compute_texture(renderer, &renderer->screen_probe_radiance);\n",
]:
    s = s.replace(old, "")

s = replace_once(
    s,
    "        destroy_radiance_surface_cache(renderer);\n        destroy_compute_texture(renderer, &renderer->direct_radiance);\n        destroy_compute_texture(renderer, &renderer->screen_probes);\n",
    "        destroy_radiance_surface_cache(renderer);\n"
    "        destroy_world_radiance_resources(renderer);\n"
    "        destroy_wavefront(renderer);\n"
    "        destroy_radiance_probes(renderer);\n"
    "        destroy_compute_texture(renderer, &renderer->direct_radiance);\n"
    "        destroy_compute_texture(renderer, &renderer->screen_probes);\n",
    "resource destruction",
)

for old in [
    "        if (renderer->radiance_scene_layout) renderer->gpu->core.DestroyPipelineLayout(renderer->radiance_scene_layout);\n\n",
    "        if (renderer->radiance_direct_layout) renderer->gpu->core.DestroyPipelineLayout(renderer->radiance_direct_layout);\n\n",
    "        if (renderer->emissive_layout) renderer->gpu->core.DestroyPipelineLayout(renderer->emissive_layout);\n",
]:
    s = s.replace(old, "")

s = replace_once(
    s,
    "        if (renderer->trace_layout) renderer->gpu->core.DestroyPipelineLayout(renderer->trace_layout);\n\n",
    "        if (renderer->trace_layout) renderer->gpu->core.DestroyPipelineLayout(renderer->trace_layout);\n\n"
    "        if (renderer->wavefront_layout) renderer->gpu->core.DestroyPipelineLayout(renderer->wavefront_layout);\n\n",
    "layout destruction",
)

s = replace_once(
    s,
    "    if (!clear_surface_cache(renderer) || !clear_radiance_surface_cache(renderer)) return false;\n    update_gbuffer_descriptors(renderer);\n    update_trace_descriptors(renderer);\n",
    "    if (!clear_radiance_surface_cache(renderer)) return false;\n"
    "    update_gbuffer_descriptors(renderer);\n"
    "    if (!update_trace_descriptors(renderer) || !update_wavefront_descriptors(renderer) || !update_radiance_scene_descriptors(renderer)) return false;\n",
    "set scene cache descriptors",
)

# Active frame: one path, no compatibility queue.
s = replace_once(
    s,
    "    build_direct_radiance(renderer, command_buffer);\n    build_screen_trace(renderer, command_buffer);\n    build_miss_queue(renderer, command_buffer);\n    build_sdf_trace(renderer, command_buffer);\n    finish_screen_trace(renderer, command_buffer);\n    build_screen_probes(renderer, command_buffer);\n    build_emissive_gather(renderer, command_buffer);\n",
    "    build_direct_radiance(renderer, command_buffer);\n"
    "    build_wavefront_screen_probes(renderer, command_buffer);\n"
    "    build_emissive_gather(renderer, command_buffer);\n",
    "active frame",
)

# Any resize must refresh both direct and wavefront descriptors.
s = s.replace("    update_trace_descriptors(renderer);\n\n    if (!update_emissive_probe_descriptors(renderer)) return false;\n", "")

p.write_text(s)
