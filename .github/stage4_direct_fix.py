from pathlib import Path


def replace_once(text, old, new, label):
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one match, got {count}")
    return text.replace(old, new, 1)

# -----------------------------------------------------------------------------
# shader.hlsl: make the already-frozen CS_RadianceDirect pass actually produce
# the active direct-radiance texture. Area emitters belong here, not in the
# screen-probe gather pass.
# -----------------------------------------------------------------------------
p = Path("shader.hlsl")
s = p.read_text()

old = '''[numthreads(8, 8, 1)]
void CS_RadianceDirect(uint3 dispatch_id : SV_DispatchThreadID) {
    uint2 pixel = dispatch_id.xy;
    uint2 resolution = uint2(TraceFrame.resolution.xy);
    if (any(pixel >= resolution)) return;
    float depth = TraceDepth.Load(int3(pixel, 0));
    if (depth <= 0.0f) return;
'''
new = '''[numthreads(8, 8, 1)]
void CS_RadianceDirect(uint3 dispatch_id : SV_DispatchThreadID) {
    uint2 pixel = dispatch_id.xy;
    uint2 resolution = uint2(TraceFrame.resolution.xy);
    if (any(pixel >= resolution)) return;
    float depth = TraceDepth.Load(int3(pixel, 0));
    if (depth <= 0.0f) {
        float2 uv = (float2(pixel) + 0.5f) * TraceFrame.resolution.zw;
        float4 far_world4 = mul(float4(ScreenUVToNDC(uv), 0.0f, 1.0f), TraceFrame.inverse_view_projection);
        float3 direction = normalize(far_world4.xyz / far_world4.w - TraceFrame.camera_position.xyz);
        DirectRadianceOutput[pixel] = float4(FutureSkyRadiance(direction), 1.0f);
        return;
    }
'''
s = replace_once(s, old, new, "CS_RadianceDirect background output")

old = '''    float3 emissive_direct = EvaluateEmissiveSample(position, normal, object_id, pixel, HashCombine(Pass.dispatch.x, PackPixel(pixel)));
    direct += material.base_color.rgb * emissive_direct;
    SurfaceHit hit = MakeSurfaceHit(TRACE_SCREEN, 0.0f);
'''
new = '''    float3 emissive_direct = EvaluateEmissiveSample(position, normal, object_id, pixel, HashCombine(Pass.dispatch.x, PackPixel(pixel)));
    direct += material.base_color.rgb * emissive_direct;
    DirectRadianceOutput[pixel] = float4(direct, 1.0f);
    SurfaceHit hit = MakeSurfaceHit(TRACE_SCREEN, 0.0f);
'''
s = replace_once(s, old, new, "CS_RadianceDirect surface output")
p.write_text(s)

# -----------------------------------------------------------------------------
# game.h: resources for a pipeline layout containing spaces 3, 4 and 6.
# -----------------------------------------------------------------------------
p = Path("game.h")
s = p.read_text()

old = '''    NriDescriptorSet *trace_set;
    NriDescriptorSet *radiance_scene_set;
    NriDescriptorSet *emissive_trace_set;
'''
new = '''    NriDescriptorSet *trace_set;
    NriDescriptorSet *radiance_scene_set;
    NriDescriptorSet *radiance_direct_trace_set;
    NriDescriptorSet *radiance_direct_scene_set;
    NriDescriptorSet *radiance_direct_cache_set;
    NriDescriptorSet *emissive_trace_set;
'''
s = replace_once(s, old, new, "direct descriptor set fields")

old = '''    NriPipelineLayout *trace_layout;
    NriPipelineLayout *radiance_scene_layout;
    NriPipelineLayout *emissive_layout;
'''
new = '''    NriPipelineLayout *trace_layout;
    NriPipelineLayout *radiance_scene_layout;
    NriPipelineLayout *radiance_direct_layout;
    NriPipelineLayout *emissive_layout;
'''
s = replace_once(s, old, new, "direct pipeline layout field")
p.write_text(s)

# -----------------------------------------------------------------------------
# render.c
# -----------------------------------------------------------------------------
p = Path("render.c")
s = p.read_text()

# Permanent direct-light layout: space3 compatibility G-buffer/output, space4
# world ray scene, space6 cache bindings. cache_counts.x remains zero until s5,
# so the existing cache buffers are valid dormant bindings for now.
old = '''    if (renderer->gpu->core.CreatePipelineLayout(renderer->gpu->device, &radiance_scene_layout, &renderer->radiance_scene_layout) != NriResult_SUCCESS) return false;

    const NriDescriptorRangeDesc emissive_probe_range = {
'''
new = '''    if (renderer->gpu->core.CreatePipelineLayout(renderer->gpu->device, &radiance_scene_layout, &renderer->radiance_scene_layout) != NriResult_SUCCESS) return false;

    const NriDescriptorRangeDesc radiance_direct_cache_range = {
        .baseRegisterIndex = 0,
        .descriptorNum = 2,
        .descriptorType = NriDescriptorType_STORAGE_STRUCTURED_BUFFER,
        .shaderStages = NriStageBits_COMPUTE_SHADER
    };

    const NriDescriptorSetDesc radiance_direct_cache_set = {
        .registerSpace = 6,
        .ranges = &radiance_direct_cache_range,
        .rangeNum = 1
    };

    const NriDescriptorSetDesc radiance_direct_sets[] = {trace_set, radiance_scene_set, radiance_direct_cache_set};

    const NriPipelineLayoutDesc radiance_direct_layout = {
        .descriptorSets = radiance_direct_sets,
        .descriptorSetNum = 3,
        .shaderStages = NriStageBits_COMPUTE_SHADER,
        .flags = NriPipelineLayoutBits_IGNORE_GLOBAL_SPIRV_OFFSETS
    };

    if (renderer->gpu->core.CreatePipelineLayout(renderer->gpu->device, &radiance_direct_layout, &renderer->radiance_direct_layout) != NriResult_SUCCESS) return false;

    const NriDescriptorRangeDesc emissive_probe_range = {
'''
s = replace_once(s, old, new, "radiance direct layout")

old = '''        .descriptorSetMaxNum = 7 + HZB_MAX_MIPS,
        .constantBufferMaxNum = 8,
        .textureMaxNum = 64,
        .storageTextureMaxNum = HZB_MAX_MIPS + 9,
        .structuredBufferMaxNum = 40,
        .storageStructuredBufferMaxNum = 16
'''
new = '''        .descriptorSetMaxNum = 10 + HZB_MAX_MIPS,
        .constantBufferMaxNum = 12,
        .textureMaxNum = 80,
        .storageTextureMaxNum = HZB_MAX_MIPS + 16,
        .structuredBufferMaxNum = 64,
        .storageStructuredBufferMaxNum = 24
'''
s = replace_once(s, old, new, "descriptor pool capacity")

old = '''           renderer->gpu->core.AllocateDescriptorSets(renderer->descriptor_pool, renderer->trace_layout, 0, &renderer->trace_set, 1, 0) == NriResult_SUCCESS &&
           renderer->gpu->core.AllocateDescriptorSets(renderer->descriptor_pool, renderer->radiance_scene_layout, 0, &renderer->radiance_scene_set, 1, 0) == NriResult_SUCCESS &&
           renderer->gpu->core.AllocateDescriptorSets(renderer->descriptor_pool, renderer->emissive_layout, 0, &renderer->emissive_trace_set, 1, 0) == NriResult_SUCCESS &&
'''
new = '''           renderer->gpu->core.AllocateDescriptorSets(renderer->descriptor_pool, renderer->trace_layout, 0, &renderer->trace_set, 1, 0) == NriResult_SUCCESS &&
           renderer->gpu->core.AllocateDescriptorSets(renderer->descriptor_pool, renderer->radiance_scene_layout, 0, &renderer->radiance_scene_set, 1, 0) == NriResult_SUCCESS &&
           renderer->gpu->core.AllocateDescriptorSets(renderer->descriptor_pool, renderer->radiance_direct_layout, 0, &renderer->radiance_direct_trace_set, 1, 0) == NriResult_SUCCESS &&
           renderer->gpu->core.AllocateDescriptorSets(renderer->descriptor_pool, renderer->radiance_direct_layout, 1, &renderer->radiance_direct_scene_set, 1, 0) == NriResult_SUCCESS &&
           renderer->gpu->core.AllocateDescriptorSets(renderer->descriptor_pool, renderer->radiance_direct_layout, 2, &renderer->radiance_direct_cache_set, 1, 0) == NriResult_SUCCESS &&
           renderer->gpu->core.AllocateDescriptorSets(renderer->descriptor_pool, renderer->emissive_layout, 0, &renderer->emissive_trace_set, 1, 0) == NriResult_SUCCESS &&
'''
s = replace_once(s, old, new, "direct descriptor allocation")

old = '''    return create_compute_pipeline(renderer, "build/shaders/hzb.cs.spv", renderer->hzb_layout, &renderer->hzb_pipeline) &&
           create_compute_pipeline(renderer, "build/shaders/direct_radiance.cs.spv", renderer->trace_layout, &renderer->direct_radiance_pipeline) &&
'''
new = '''    return create_compute_pipeline(renderer, "build/shaders/hzb.cs.spv", renderer->hzb_layout, &renderer->hzb_pipeline) &&
           create_compute_pipeline(renderer, "build/shaders/radiance_direct.cs.spv", renderer->radiance_direct_layout, &renderer->direct_radiance_pipeline) &&
'''
s = replace_once(s, old, new, "activate permanent direct shader")

old = '''static void update_trace_descriptors(RENDERER *renderer) {
    if (renderer->trace_set) update_trace_descriptor_set(renderer, renderer->trace_set);
    if (renderer->emissive_trace_set) update_trace_descriptor_set(renderer, renderer->emissive_trace_set);
}
'''
new = '''static void update_trace_descriptors(RENDERER *renderer) {
    if (renderer->trace_set) update_trace_descriptor_set(renderer, renderer->trace_set);
    if (renderer->radiance_direct_trace_set) update_trace_descriptor_set(renderer, renderer->radiance_direct_trace_set);
    if (renderer->emissive_trace_set) update_trace_descriptor_set(renderer, renderer->emissive_trace_set);
}
'''
s = replace_once(s, old, new, "direct trace descriptor update")

old = '''static bool update_radiance_scene_descriptors(RENDERER *renderer) {
    if (!update_radiance_scene_descriptor_set(renderer, renderer->radiance_scene_set)) return false;
    if (renderer->emissive_scene_set && !update_radiance_scene_descriptor_set(renderer, renderer->emissive_scene_set)) return false;
    return true;
}

static bool update_emissive_probe_descriptors(RENDERER *renderer) {
'''
new = '''static bool update_radiance_scene_descriptors(RENDERER *renderer) {
    if (!update_radiance_scene_descriptor_set(renderer, renderer->radiance_scene_set)) return false;
    if (renderer->radiance_direct_scene_set && !update_radiance_scene_descriptor_set(renderer, renderer->radiance_direct_scene_set)) return false;
    if (renderer->emissive_scene_set && !update_radiance_scene_descriptor_set(renderer, renderer->emissive_scene_set)) return false;
    return true;
}

static bool update_radiance_direct_cache_descriptors(RENDERER *renderer) {
    if (!renderer || !renderer->radiance_direct_cache_set || !renderer->surface_cache.keys_uav || !renderer->surface_cache.entries_uav) return false;

    const NriDescriptor *descriptors[] = {renderer->surface_cache.keys_uav, renderer->surface_cache.entries_uav};
    const NriUpdateDescriptorRangeDesc update = {
        .descriptorSet = renderer->radiance_direct_cache_set,
        .rangeIndex = 0,
        .descriptors = descriptors,
        .descriptorNum = 2
    };

    renderer->gpu->core.UpdateDescriptorRanges(&update, 1);
    return true;
}

static bool update_emissive_probe_descriptors(RENDERER *renderer) {
'''
s = replace_once(s, old, new, "direct scene/cache descriptor update")

# Bind spaces 3,4,6 for the new active direct pass.
old = '''static void build_hzb(RENDERER *renderer, NriCommandBuffer *command_buffer) {
'''
new = '''static void bind_radiance_direct(RENDERER *renderer, NriCommandBuffer *command_buffer) {
    renderer->gpu->core.CmdSetPipelineLayout(command_buffer, NriBindPoint_COMPUTE, renderer->radiance_direct_layout);
    renderer->gpu->core.CmdSetPipeline(command_buffer, renderer->direct_radiance_pipeline);

    const NriSetDescriptorSetDesc sets[] = {
        {.setIndex = 0, .descriptorSet = renderer->radiance_direct_trace_set, .bindPoint = NriBindPoint_COMPUTE},
        {.setIndex = 1, .descriptorSet = renderer->radiance_direct_scene_set, .bindPoint = NriBindPoint_COMPUTE},
        {.setIndex = 2, .descriptorSet = renderer->radiance_direct_cache_set, .bindPoint = NriBindPoint_COMPUTE}
    };

    for (uint32_t i = 0; i < sizeof(sets) / sizeof(sets[0]); ++i)
        renderer->gpu->core.CmdSetDescriptorSet(command_buffer, &sets[i]);
}

static void build_hzb(RENDERER *renderer, NriCommandBuffer *command_buffer) {
'''
s = replace_once(s, old, new, "bind radiance direct")

old = '''    bind_trace(renderer, command_buffer, renderer->direct_radiance_pipeline);
    renderer->gpu->core.CmdDispatch(
'''
new = '''    bind_radiance_direct(renderer, command_buffer);
    renderer->gpu->core.CmdDispatch(
'''
s = replace_once(s, old, new, "direct pass bind")

old = '''    if (!create_pipeline_layouts(renderer) || !create_descriptor_pool(renderer) || !create_frame_buffer(renderer) || !create_radiance_constant_buffers(renderer) ||
        !create_radiance_scene_fallbacks(renderer) || !create_pipelines(renderer) || !create_surface_cache(renderer) ||
        !create_size_dependent_resources(renderer, gpu->swapchain_width, gpu->swapchain_height)) {
'''
new = '''    if (!create_pipeline_layouts(renderer) || !create_descriptor_pool(renderer) || !create_frame_buffer(renderer) || !create_radiance_constant_buffers(renderer) ||
        !create_radiance_scene_fallbacks(renderer) || !create_pipelines(renderer) || !create_surface_cache(renderer) ||
        !update_radiance_direct_cache_descriptors(renderer) ||
        !create_size_dependent_resources(renderer, gpu->swapchain_width, gpu->swapchain_height)) {
'''
s = replace_once(s, old, new, "initialize direct cache descriptors")

old = '''        if (renderer->radiance_scene_layout) renderer->gpu->core.DestroyPipelineLayout(renderer->radiance_scene_layout);

        if (renderer->emissive_layout) renderer->gpu->core.DestroyPipelineLayout(renderer->emissive_layout);
'''
new = '''        if (renderer->radiance_scene_layout) renderer->gpu->core.DestroyPipelineLayout(renderer->radiance_scene_layout);

        if (renderer->radiance_direct_layout) renderer->gpu->core.DestroyPipelineLayout(renderer->radiance_direct_layout);

        if (renderer->emissive_layout) renderer->gpu->core.DestroyPipelineLayout(renderer->emissive_layout);
'''
s = replace_once(s, old, new, "destroy direct layout")

old = '''    build_screen_probes(renderer, command_buffer);
    build_emissive_gather(renderer, command_buffer);
    record_present_pass(renderer, command_buffer, swapchain_index);
'''
new = '''    build_screen_probes(renderer, command_buffer);
    record_present_pass(renderer, command_buffer, swapchain_index);
'''
s = replace_once(s, old, new, "stop additive screen-probe emitter gather")

p.write_text(s)
