from pathlib import Path


def replace_once(text, old, new, label):
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected 1 match, got {count}")
    return text.replace(old, new, 1)


game = Path("game.h")
text = game.read_text()

text = replace_once(
    text,
    '''typedef struct RADIANCE_SCENE_DATA {
    GPU_SCENE_TRIANGLE *cpu_triangle;
    GPU_EMISSIVE_TRIANGLE *cpu_emissive_triangles;

    uint32_t triangle_count;
    uint32_t emissive_triangle_count;
} RADIANCE_SCENE_DATA;''',
    '''typedef struct RADIANCE_SCENE_DATA {
    GPU_SCENE_TRIANGLE *cpu_triangles;
    GPU_EMISSIVE_TRIANGLE *cpu_emissive_triangles;

    NriBuffer *triangles;
    NriBuffer *emissive_triangles;

    NriDescriptor *triangles_srv;
    NriDescriptor *emissive_triangles_srv;

    NriAccessStage triangles_state;
    NriAccessStage emissive_triangles_state;

    uint32_t triangle_count;
    uint32_t emissive_triangle_count;
} RADIANCE_SCENE_DATA;''',
    "RADIANCE_SCENE_DATA",
)

text = replace_once(
    text,
    '''    NriBuffer *light_buffer;
    NriBuffer *frame_buffer;''',
    '''    NriBuffer *light_buffer;
    NriBuffer *frame_buffer;

    NriBuffer *radiance_constants_buffer;
    NriBuffer *pass_constants_buffer;''',
    "renderer radiance buffers",
)

text = replace_once(
    text,
    '''    NriDescriptor *light_srv;
    NriDescriptor *frame_srv;''',
    '''    NriDescriptor *light_srv;
    NriDescriptor *frame_srv;

    NriDescriptor *radiance_constants_srv;
    NriDescriptor *pass_constants_srv;''',
    "renderer radiance descriptors",
)

text = replace_once(
    text,
    '''    GPU_OBJECT *cpu_objects;
    GPU_LIGHT *cpu_lights;''',
    '''    GPU_OBJECT *cpu_objects;
    GPU_LIGHT *cpu_lights;

    RADIANCE_CONSTANTS radiance_constants;
    PASS_CONSTANTS pass_constants;''',
    "renderer radiance CPU constants",
)

text = replace_once(
    text,
    '''    NriDescriptorSet *hzb_sets[HZB_MAX_MIPS];
    NriDescriptorSet *trace_set;''',
    '''    NriDescriptorSet *hzb_sets[HZB_MAX_MIPS];
    NriDescriptorSet *trace_set;
    NriDescriptorSet *radiance_scene_set;''',
    "renderer radiance descriptor set",
)

text = replace_once(
    text,
    '''    NriPipelineLayout *hzb_layout;
    NriPipelineLayout *trace_layout;''',
    '''    NriPipelineLayout *hzb_layout;
    NriPipelineLayout *trace_layout;
    NriPipelineLayout *radiance_scene_layout;''',
    "renderer radiance pipeline layout",
)

text = replace_once(
    text,
    '''    NriAccessStage object_state;
    NriAccessStage light_state;
    NriAccessStage frame_state;''',
    '''    NriAccessStage object_state;
    NriAccessStage light_state;
    NriAccessStage frame_state;

    NriAccessStage radiance_constants_state;
    NriAccessStage pass_constants_state;''',
    "renderer radiance states",
)

game.write_text(text)

render = Path("render.c")
text = render.read_text()

text = replace_once(
    text,
    '''_Static_assert(sizeof(SURFACE_CACHE_ENTRY) == 112u, "SURFACE_CACHE_ENTRY GPU layout changed");

static MAT4 mat4_identity(void) {''',
    '''_Static_assert(sizeof(SURFACE_CACHE_ENTRY) == 112u, "SURFACE_CACHE_ENTRY GPU layout changed");

static bool update_radiance_constants(RENDERER *renderer);
static bool update_radiance_scene_descriptors(RENDERER *renderer);

static MAT4 mat4_identity(void) {''',
    "radiance forward declarations",
)

if text.count("cpu_triangle") != 2:
    raise SystemExit(f"cpu_triangle rename: expected 2 matches, got {text.count('cpu_triangle')}")
text = text.replace("cpu_triangle", "cpu_triangles")

text = replace_once(
    text,
    '''static void destroy_radiance_scene_data(RENDERER *renderer) {

    if (!renderer) return;

    free(renderer->radiance_scene.cpu_triangles);
    free(renderer->radiance_scene.cpu_emissive_triangles);

    memset(&renderer->radiance_scene, 0, sizeof(renderer->radiance_scene));
}''',
    '''static void destroy_radiance_scene_gpu_resources(RENDERER *renderer) {
    if (!renderer) return;

    RADIANCE_SCENE_DATA *scene = &renderer->radiance_scene;

    if (renderer->gpu) {
        if (scene->triangles_srv) renderer->gpu->core.DestroyDescriptor(scene->triangles_srv);
        if (scene->emissive_triangles_srv) renderer->gpu->core.DestroyDescriptor(scene->emissive_triangles_srv);
        if (scene->triangles) gpu_destroy_buffer(renderer->gpu, scene->triangles);
        if (scene->emissive_triangles) gpu_destroy_buffer(renderer->gpu, scene->emissive_triangles);
    }

    scene->triangles = NULL;
    scene->emissive_triangles = NULL;
    scene->triangles_srv = NULL;
    scene->emissive_triangles_srv = NULL;
    scene->triangles_state = (NriAccessStage){0};
    scene->emissive_triangles_state = (NriAccessStage){0};
}

static void destroy_radiance_scene_data(RENDERER *renderer) {
    if (!renderer) return;

    destroy_radiance_scene_gpu_resources(renderer);
    free(renderer->radiance_scene.cpu_triangles);
    free(renderer->radiance_scene.cpu_emissive_triangles);

    memset(&renderer->radiance_scene, 0, sizeof(renderer->radiance_scene));
}''',
    "radiance scene teardown",
)

marker = '''static void destroy_screen_trace(RENDERER *renderer) {'''
if text.count(marker) != 1:
    raise SystemExit("destroy_screen_trace marker missing")
gpu_scene_fn = r'''static bool create_radiance_scene_gpu_resources(RENDERER *renderer) {
    if (!renderer) return false;

    RADIANCE_SCENE_DATA *scene = &renderer->radiance_scene;
    destroy_radiance_scene_gpu_resources(renderer);

    const uint32_t triangle_capacity = scene->triangle_count ? scene->triangle_count : 1u;
    const uint32_t emissive_capacity = scene->emissive_triangle_count ? scene->emissive_triangle_count : 1u;

    const NriBufferDesc triangle_desc = {
        .size = (uint64_t)triangle_capacity * sizeof(GPU_SCENE_TRIANGLE),
        .structureStride = sizeof(GPU_SCENE_TRIANGLE),
        .usage = NriBufferUsageBits_SHADER_RESOURCE
    };

    const NriBufferDesc emissive_desc = {
        .size = (uint64_t)emissive_capacity * sizeof(GPU_EMISSIVE_TRIANGLE),
        .structureStride = sizeof(GPU_EMISSIVE_TRIANGLE),
        .usage = NriBufferUsageBits_SHADER_RESOURCE
    };

    if (!gpu_create_buffer(renderer->gpu, &triangle_desc, NriMemoryLocation_DEVICE, &scene->triangles) ||
        !gpu_create_buffer(renderer->gpu, &emissive_desc, NriMemoryLocation_DEVICE, &scene->emissive_triangles)) {
        destroy_radiance_scene_gpu_resources(renderer);
        return false;
    }

    if (!create_buffer_view(renderer, scene->triangles, NriBufferView_STRUCTURED_BUFFER, triangle_desc.size, sizeof(GPU_SCENE_TRIANGLE), &scene->triangles_srv) ||
        !create_buffer_view(renderer, scene->emissive_triangles, NriBufferView_STRUCTURED_BUFFER, emissive_desc.size, sizeof(GPU_EMISSIVE_TRIANGLE), &scene->emissive_triangles_srv)) {
        destroy_radiance_scene_gpu_resources(renderer);
        return false;
    }

    const NriAccessStage read = {
        .access = NriAccessBits_SHADER_RESOURCE,
        .stages = NriStageBits_COMPUTE_SHADER
    };

    const GPU_SCENE_TRIANGLE zero_triangle = {0};
    const GPU_EMISSIVE_TRIANGLE zero_emitter = {0};
    const GPU_SCENE_TRIANGLE *triangle_data = scene->triangle_count ? scene->cpu_triangles : &zero_triangle;
    const GPU_EMISSIVE_TRIANGLE *emissive_data = scene->emissive_triangle_count ? scene->cpu_emissive_triangles : &zero_emitter;

    if (!gpu_upload_buffer(renderer->gpu, scene->triangles, triangle_data, read) ||
        !gpu_upload_buffer(renderer->gpu, scene->emissive_triangles, emissive_data, read)) {
        destroy_radiance_scene_gpu_resources(renderer);
        return false;
    }

    scene->triangles_state = read;
    scene->emissive_triangles_state = read;

    SDL_Log("Radiance GPU scene: %u triangles, %u emissive triangles", scene->triangle_count, scene->emissive_triangle_count);

    return true;
}

'''
text = text.replace(marker, gpu_scene_fn + marker, 1)

text = replace_once(
    text,
    '''        const NriAccessStage material_state = {
            .access = NriAccessBits_SHADER_RESOURCE,
            .stages = NriStageBits_FRAGMENT_SHADER
        };''',
    '''        const NriAccessStage material_state = {
            .access = NriAccessBits_SHADER_RESOURCE,
            .stages = NriStageBits_FRAGMENT_SHADER | NriStageBits_COMPUTE_SHADER
        };''',
    "material compute visibility",
)

text = replace_once(
    text,
    '''    if (!ok || !create_sdf_scene(renderer, scene)) {
        destroy_scene_resources(renderer);

        return false;
    }

    return true;
}''',
    '''    if (!ok || !create_sdf_scene(renderer, scene) || !create_radiance_scene_gpu_resources(renderer) || !update_radiance_constants(renderer) ||
        !update_radiance_scene_descriptors(renderer)) {
        destroy_scene_resources(renderer);

        return false;
    }

    return true;
}''',
    "scene resource finalization",
)

marker = '''static bool create_frame_buffer(RENDERER *renderer) {'''
if text.count(marker) != 1:
    raise SystemExit("create_frame_buffer marker missing")
constants_fns = r'''static void destroy_radiance_constant_buffers(RENDERER *renderer) {
    if (!renderer || !renderer->gpu) return;

    if (renderer->radiance_constants_srv) renderer->gpu->core.DestroyDescriptor(renderer->radiance_constants_srv);
    if (renderer->pass_constants_srv) renderer->gpu->core.DestroyDescriptor(renderer->pass_constants_srv);
    if (renderer->radiance_constants_buffer) gpu_destroy_buffer(renderer->gpu, renderer->radiance_constants_buffer);
    if (renderer->pass_constants_buffer) gpu_destroy_buffer(renderer->gpu, renderer->pass_constants_buffer);

    renderer->radiance_constants_srv = NULL;
    renderer->pass_constants_srv = NULL;
    renderer->radiance_constants_buffer = NULL;
    renderer->pass_constants_buffer = NULL;
    renderer->radiance_constants_state = (NriAccessStage){0};
    renderer->pass_constants_state = (NriAccessStage){0};

    memset(&renderer->radiance_constants, 0, sizeof(renderer->radiance_constants));
    memset(&renderer->pass_constants, 0, sizeof(renderer->pass_constants));
}

static bool create_radiance_constant_buffers(RENDERER *renderer) {
    destroy_radiance_constant_buffers(renderer);

    const NriBufferDesc radiance_desc = {
        .size = sizeof(RADIANCE_CONSTANTS),
        .usage = NriBufferUsageBits_CONSTANT
    };

    const NriBufferDesc pass_desc = {
        .size = sizeof(PASS_CONSTANTS),
        .usage = NriBufferUsageBits_CONSTANT
    };

    if (!gpu_create_buffer(renderer->gpu, &radiance_desc, NriMemoryLocation_DEVICE, &renderer->radiance_constants_buffer) ||
        !gpu_create_buffer(renderer->gpu, &pass_desc, NriMemoryLocation_DEVICE, &renderer->pass_constants_buffer)) {
        destroy_radiance_constant_buffers(renderer);
        return false;
    }

    if (!create_buffer_view(renderer, renderer->radiance_constants_buffer, NriBufferView_CONSTANT_BUFFER, sizeof(RADIANCE_CONSTANTS), 0, &renderer->radiance_constants_srv) ||
        !create_buffer_view(renderer, renderer->pass_constants_buffer, NriBufferView_CONSTANT_BUFFER, sizeof(PASS_CONSTANTS), 0, &renderer->pass_constants_srv)) {
        destroy_radiance_constant_buffers(renderer);
        return false;
    }

    memset(&renderer->radiance_constants, 0, sizeof(renderer->radiance_constants));
    memset(&renderer->pass_constants, 0, sizeof(renderer->pass_constants));

    const NriAccessStage read = {
        .access = NriAccessBits_CONSTANT_BUFFER,
        .stages = NriStageBits_COMPUTE_SHADER | NriStageBits_FRAGMENT_SHADER
    };

    if (!gpu_upload_buffer(renderer->gpu, renderer->radiance_constants_buffer, &renderer->radiance_constants, read) ||
        !gpu_upload_buffer(renderer->gpu, renderer->pass_constants_buffer, &renderer->pass_constants, read)) {
        destroy_radiance_constant_buffers(renderer);
        return false;
    }

    renderer->radiance_constants_state = read;
    renderer->pass_constants_state = read;

    return true;
}

static bool update_radiance_constants(RENDERER *renderer) {
    if (!renderer || !renderer->radiance_constants_buffer) return false;

    RADIANCE_CONSTANTS constants = {0};
    constants.scene_counts[0] = renderer->gpu_object_count;
    constants.scene_counts[1] = renderer->material_count;
    constants.scene_counts[2] = renderer->radiance_scene.triangle_count;
    constants.scene_counts[3] = renderer->radiance_scene.emissive_triangle_count;
    constants.sdf_counts[0] = renderer->sdf.model_count;
    constants.sdf_counts[1] = 0u;
    constants.sdf_counts[2] = 0u;
    constants.sdf_counts[3] = renderer->light_count;
    constants.cache_counts[0] = 0u;
    constants.cache_counts[1] = 0u;
    constants.cache_counts[2] = 0u;
    constants.cache_counts[3] = 0u;
    constants.probe_config[0] = SCREEN_PROBE_TILE_SIZE;
    constants.probe_config[1] = SCREEN_PROBE_DIRECTION_SIZE;
    constants.probe_config[2] = SCREEN_PROBE_DIRECTION_COUNT;
    constants.probe_config[3] = 0u;
    constants.trace_params[0] = 200.0f;
    constants.trace_params[1] = 0.10f;
    constants.trace_params[2] = 0.05f;
    constants.trace_params[3] = 0.05f;
    constants.trace_limits[0] = 128u;
    constants.trace_limits[1] = 5u;
    constants.trace_limits[2] = 96u;
    constants.trace_limits[3] = 128u;
    constants.feature_flags[0] = 0u;
    constants.feature_flags[1] = 1u;
    constants.feature_flags[2] = RADIANCE_DEBUG_FINAL_GI;
    constants.feature_flags[3] = 0u;
    constants.global_sdf_params[2] = 1.0e-4f;

    renderer->radiance_constants = constants;

    const NriAccessStage read = {
        .access = NriAccessBits_CONSTANT_BUFFER,
        .stages = NriStageBits_COMPUTE_SHADER | NriStageBits_FRAGMENT_SHADER
    };

    if (!gpu_upload_buffer(renderer->gpu, renderer->radiance_constants_buffer, &renderer->radiance_constants, read)) return false;

    renderer->radiance_constants_state = read;
    return true;
}

'''
text = text.replace(marker, constants_fns + marker, 1)

text = replace_once(
    text,
    '''    return renderer->gpu->core.CreatePipelineLayout(renderer->gpu->device, &trace_layout, &renderer->trace_layout) == NriResult_SUCCESS;
}''',
    '''    if (renderer->gpu->core.CreatePipelineLayout(renderer->gpu->device, &trace_layout, &renderer->trace_layout) != NriResult_SUCCESS) return false;

    const NriStageBits radiance_stages = NriStageBits_COMPUTE_SHADER | NriStageBits_FRAGMENT_SHADER;

    const NriDescriptorRangeDesc radiance_scene_ranges[] = {
        {.baseRegisterIndex = 0, .descriptorNum = 2, .descriptorType = NriDescriptorType_CONSTANT_BUFFER, .shaderStages = radiance_stages},
        {.baseRegisterIndex = 2, .descriptorNum = 6, .descriptorType = NriDescriptorType_STRUCTURED_BUFFER, .shaderStages = radiance_stages},
        {.baseRegisterIndex = 8, .descriptorNum = 7, .descriptorType = NriDescriptorType_STRUCTURED_BUFFER, .shaderStages = radiance_stages},
        {.baseRegisterIndex = 15, .descriptorNum = 1, .descriptorType = NriDescriptorType_STRUCTURED_BUFFER, .shaderStages = radiance_stages},
        {.baseRegisterIndex = 16, .descriptorNum = 2, .descriptorType = NriDescriptorType_TEXTURE, .shaderStages = radiance_stages}
    };

    const NriDescriptorSetDesc radiance_scene_set = {
        .registerSpace = 4,
        .ranges = radiance_scene_ranges,
        .rangeNum = sizeof(radiance_scene_ranges) / sizeof(radiance_scene_ranges[0])
    };

    const NriPipelineLayoutDesc radiance_scene_layout = {
        .descriptorSets = &radiance_scene_set,
        .descriptorSetNum = 1,
        .shaderStages = radiance_stages,
        .flags = NriPipelineLayoutBits_IGNORE_GLOBAL_SPIRV_OFFSETS
    };

    return renderer->gpu->core.CreatePipelineLayout(renderer->gpu->device, &radiance_scene_layout, &renderer->radiance_scene_layout) == NriResult_SUCCESS;
}''',
    "space4 pipeline layout",
)

text = replace_once(
    text,
    '''static bool create_descriptor_pool(RENDERER *renderer) {
    const NriDescriptorPoolDesc desc = {
        .descriptorSetMaxNum = 3 + HZB_MAX_MIPS,
        .constantBufferMaxNum = 3,
        .textureMaxNum = 40,
        .storageTextureMaxNum = HZB_MAX_MIPS + 4,
        .structuredBufferMaxNum = 8,
        .storageStructuredBufferMaxNum = 8
    };

    if (renderer->gpu->core.CreateDescriptorPool(renderer->gpu->device, &desc, &renderer->descriptor_pool) != NriResult_SUCCESS) return false;

    return renderer->gpu->core.AllocateDescriptorSets(renderer->descriptor_pool, renderer->gbuffer_layout, 0, &renderer->gbuffer_set, 1, 0) == NriResult_SUCCESS &&
           renderer->gpu->core.AllocateDescriptorSets(renderer->descriptor_pool, renderer->present_layout, 0, &renderer->present_set, 1, 0) == NriResult_SUCCESS &&
           renderer->gpu->core.AllocateDescriptorSets(renderer->descriptor_pool, renderer->hzb_layout, 0, renderer->hzb_sets, HZB_MAX_MIPS, 0) == NriResult_SUCCESS &&
           renderer->gpu->core.AllocateDescriptorSets(renderer->descriptor_pool, renderer->trace_layout, 0, &renderer->trace_set, 1, 0) == NriResult_SUCCESS;
}''',
    '''static bool create_descriptor_pool(RENDERER *renderer) {
    const NriDescriptorPoolDesc desc = {
        .descriptorSetMaxNum = 4 + HZB_MAX_MIPS,
        .constantBufferMaxNum = 4,
        .textureMaxNum = 48,
        .storageTextureMaxNum = HZB_MAX_MIPS + 4,
        .structuredBufferMaxNum = 24,
        .storageStructuredBufferMaxNum = 8
    };

    if (renderer->gpu->core.CreateDescriptorPool(renderer->gpu->device, &desc, &renderer->descriptor_pool) != NriResult_SUCCESS) return false;

    return renderer->gpu->core.AllocateDescriptorSets(renderer->descriptor_pool, renderer->gbuffer_layout, 0, &renderer->gbuffer_set, 1, 0) == NriResult_SUCCESS &&
           renderer->gpu->core.AllocateDescriptorSets(renderer->descriptor_pool, renderer->present_layout, 0, &renderer->present_set, 1, 0) == NriResult_SUCCESS &&
           renderer->gpu->core.AllocateDescriptorSets(renderer->descriptor_pool, renderer->hzb_layout, 0, renderer->hzb_sets, HZB_MAX_MIPS, 0) == NriResult_SUCCESS &&
           renderer->gpu->core.AllocateDescriptorSets(renderer->descriptor_pool, renderer->trace_layout, 0, &renderer->trace_set, 1, 0) == NriResult_SUCCESS &&
           renderer->gpu->core.AllocateDescriptorSets(renderer->descriptor_pool, renderer->radiance_scene_layout, 0, &renderer->radiance_scene_set, 1, 0) == NriResult_SUCCESS;
}''',
    "descriptor pool",
)

marker = '''static bool create_size_dependent_resources(RENDERER *renderer, uint32_t width, uint32_t height) {'''
if text.count(marker) != 1:
    raise SystemExit("create_size_dependent_resources marker missing")
descriptor_fn = r'''static bool update_radiance_scene_descriptors(RENDERER *renderer) {
    if (!renderer || !renderer->radiance_scene_set || !renderer->radiance_constants_srv || !renderer->pass_constants_srv || !renderer->object_srv ||
        !renderer->material_srv || !renderer->radiance_scene.triangles_srv || !renderer->radiance_scene.emissive_triangles_srv || !renderer->sdf.models_srv ||
        !renderer->sdf.voxels_srv || !renderer->light_srv) {
        return false;
    }

    const NriDescriptor *constants[] = {renderer->radiance_constants_srv, renderer->pass_constants_srv};
    const NriDescriptor *scene_core[] = {
        renderer->object_srv,
        renderer->material_srv,
        renderer->radiance_scene.triangles_srv,
        renderer->radiance_scene.emissive_triangles_srv,
        renderer->sdf.models_srv,
        renderer->sdf.voxels_srv
    };
    const NriDescriptor *lights[] = {renderer->light_srv};

    const NriUpdateDescriptorRangeDesc updates[] = {
        {.descriptorSet = renderer->radiance_scene_set, .rangeIndex = 0, .descriptors = constants, .descriptorNum = 2},
        {.descriptorSet = renderer->radiance_scene_set, .rangeIndex = 1, .descriptors = scene_core, .descriptorNum = 6},
        {.descriptorSet = renderer->radiance_scene_set, .rangeIndex = 3, .descriptors = lights, .descriptorNum = 1}
    };

    renderer->gpu->core.UpdateDescriptorRanges(updates, sizeof(updates) / sizeof(updates[0]));
    return true;
}

'''
text = text.replace(marker, descriptor_fn + marker, 1)

text = replace_once(
    text,
    '''    const NriAccessStage object_read = {
        .access = NriAccessBits_SHADER_RESOURCE,
        .stages = NriStageBits_VERTEX_SHADER
    };''',
    '''    const NriAccessStage object_read = {
        .access = NriAccessBits_SHADER_RESOURCE,
        .stages = NriStageBits_VERTEX_SHADER | NriStageBits_COMPUTE_SHADER
    };''',
    "object compute visibility",
)

text = replace_once(
    text,
    '''    if (!create_pipeline_layouts(renderer) || !create_descriptor_pool(renderer) || !create_frame_buffer(renderer) || !create_pipelines(renderer) ||
        !create_surface_cache(renderer) || !create_size_dependent_resources(renderer, gpu->swapchain_width, gpu->swapchain_height)) {''',
    '''    if (!create_pipeline_layouts(renderer) || !create_descriptor_pool(renderer) || !create_frame_buffer(renderer) || !create_radiance_constant_buffers(renderer) ||
        !create_pipelines(renderer) || !create_surface_cache(renderer) || !create_size_dependent_resources(renderer, gpu->swapchain_width, gpu->swapchain_height)) {''',
    "renderer init constants",
)

text = replace_once(
    text,
    '''        destroy_hzb(renderer);
        destroy_gbuffer(renderer);

        if (renderer->frame_srv) renderer->gpu->core.DestroyDescriptor(renderer->frame_srv);''',
    '''        destroy_hzb(renderer);
        destroy_gbuffer(renderer);
        destroy_radiance_constant_buffers(renderer);

        if (renderer->frame_srv) renderer->gpu->core.DestroyDescriptor(renderer->frame_srv);''',
    "renderer deinit constants",
)

text = replace_once(
    text,
    '''        if (renderer->trace_layout) renderer->gpu->core.DestroyPipelineLayout(renderer->trace_layout);
    }''',
    '''        if (renderer->trace_layout) renderer->gpu->core.DestroyPipelineLayout(renderer->trace_layout);

        if (renderer->radiance_scene_layout) renderer->gpu->core.DestroyPipelineLayout(renderer->radiance_scene_layout);
    }''',
    "renderer deinit radiance layout",
)

render.write_text(text)
