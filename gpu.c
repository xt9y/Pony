#include "game.h"
#include "render_internal.h"

#include <SDL3_image/SDL_image.h>

#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BAKE_TARGET_SAMPLES 128u
#define BAKE_MAX_BOUNCES 3u
#define BAKE_BATCH_SAMPLES 8u
#define BAKE_DILATION_PASSES 3u
#define FRAME_QUEUE_DEPTH 2u
#define WORK_QUEUE_DEPTH 8u
#define UNIFORM_RING_BYTES (1024u * 1024u)
#define TIMESTAMP_CAPACITY 16u
#define UPLOAD_RING_SIZE 2u
#define UPLOAD_CHUNK_BYTES (32u * 1024u * 1024u)
#define UPLOAD_SLOW_LOG_MS 5000u
#define PHASE_CLEAR 0u
#define PHASE_TRACE 1u
#define PHASE_FILTER 2u
#define PHASE_DILATE 3u
#define PHASE_DIRECT 4u
#define PHASE_COMBINE 5u
#define PHASE_RECONSTRUCT 6u

typedef struct CAMERA_UNIFORMS {
    float mvp[16];
    float view[16];
} CAMERA_UNIFORMS;

typedef struct SKY_UNIFORMS {
    float camera_right[4];
    float camera_up[4];
    float camera_forward[4];
    float sky_zenith[4];
    float sky_horizon[4];
    float sun_direction_intensity[4];
    float sun_color_radius[4];
} SKY_UNIFORMS;

typedef struct MATERIAL_UNIFORMS {
    float base_color_factor[4];
    float emissive_metallic[4];
    float roughness_normal_ao_sun[4];
    float sun_direction[4];
    float sun_color[4];
    float camera_position[4];
} MATERIAL_UNIFORMS;

typedef struct SSAO_UNIFORMS {
    Uint32 width, height, ao_width, ao_height;
    float tan_half_fov, aspect, radius, bias;
} SSAO_UNIFORMS;

typedef struct BLOOM_UNIFORMS {
    Uint32 src_width, src_height, dst_width, dst_height;
    Uint32 phase, _pad0, _pad1, _pad2;
    float threshold, knee, strength, _pad3;
} BLOOM_UNIFORMS;

typedef struct COMPOSE_UNIFORMS {
    float exposure;
    float ao_strength;
    float bloom_strength;
    float _pad;
} COMPOSE_UNIFORMS;

typedef struct VOLUME_UNIFORMS {
    float eye_density[4], right_tan[4], up_tan[4], forward_g[4];
    float sun_intensity[4], sun_color[4], grid_origin_spacing[4];
    Uint32 grid_dims_width[4], height_debug[4];
    float beam_origin[4], beam_step[4];
    float volume_params[4], volume_radii[4], volume_filter[4];
    Uint32 volume_quality[4], volume_strides[4];
} VOLUME_UNIFORMS;

typedef struct VOLUME_COMPOSE_UNIFORMS {
    Uint32 width, height, debug_view, bypass_volume;

    float volume_radii[4];
    float volume_filter[4];
    Uint32 volume_strides[4];
} VOLUME_COMPOSE_UNIFORMS;

struct RENDER_MATERIAL {
    GLTF_MATERIAL data;
    NriTexture *base_color;
    NriTexture *metallic_roughness;
    NriTexture *normal;
    NriTexture *occlusion;
    NriTexture *emissive;
};

struct SWAPCHAIN_TEXTURE {
    NriTexture *texture;
    NriDescriptor *color_attachment;
    NriFence *acquire;
    NriFence *release;
};

struct FRAME_CONTEXT {
    NriCommandAllocator *allocator;
    NriCommandBuffer *command_buffer;
    NriDescriptorPool *descriptor_pool;
    NriDescriptor **temporary_descriptors;
    NriBuffer **temporary_buffers;

    uint32_t temporary_descriptor_num, temporary_descriptor_cap;
    uint32_t temporary_buffer_num, temporary_buffer_cap;

    NriBuffer *uniform_buffer;
    uint64_t uniform_offset;
    uint64_t fence_value;
};

typedef struct UPLOAD_SLOT {
    NriBuffer *staging;
    NriCommandAllocator *allocator;
    NriCommandBuffer *command_buffer;
    uint64_t fence_value;
} UPLOAD_SLOT;

struct UPLOAD_CONTEXT {
    NriFence *fence;
    UPLOAD_SLOT slots[UPLOAD_RING_SIZE];
    uint64_t next_fence_value;
    uint32_t next_slot;
};

struct TEXTURE_STATE {
    NriTexture *texture;
    NriAccessLayoutStage state;
};

static Uint8 *load_spirv(const char *define, size_t *size) {
    char path[256];
    int written = snprintf(path, sizeof(path), "build/shaders/%s.spv", define);

    if (written < 0 || (size_t)written >= sizeof(path)) return NULL;

    Uint8 *spirv = SDL_LoadFile(path, size);
    if (!spirv) SDL_Log("SPIR-V load failed for %s: %s", path, SDL_GetError());

    return spirv;
}

static void free_probe_grid(PROBE_GRID *grid) {
    if (!grid) return;
    free(grid->probes);
    memset(grid, 0, sizeof(*grid));
}

static bool create_pipeline_cache(RENDERER *r) {
    if (!r || !r->gpu->device) return false;

    if (r->gpu->pipeline_cache) return true;

    const NriPipelineCacheDesc desc = {0};
    NriResult result = r->gpu->core.CreatePipelineCache(r->gpu->device, &desc, &r->gpu->pipeline_cache);

    if (result == NriResult_UNSUPPORTED) {
        r->gpu->pipeline_cache = NULL;

        return true;
    }

    return result == NriResult_SUCCESS;
}

static void destroy_pipeline_cache(RENDERER *r) {
    if (!r || !r->gpu->pipeline_cache) return;
    r->gpu->core.DestroyPipelineCache(r->gpu->pipeline_cache);
    r->gpu->pipeline_cache = NULL;
}

static bool acquire_queues(RENDERER *r) {
    if (!r || !r->gpu->device) return false;

    if (r->gpu->core.GetQueue(r->gpu->device, NriQueueType_GRAPHICS, 0, &r->gpu->graphics_queue) != NriResult_SUCCESS) return false;

    if (r->gpu->core.GetQueue(r->gpu->device, NriQueueType_COMPUTE, 0, &r->gpu->compute_queue) != NriResult_SUCCESS)
        r->gpu->compute_queue = r->gpu->graphics_queue;

    if (r->gpu->core.GetQueue(r->gpu->device, NriQueueType_COPY, 0, &r->gpu->copy_queue) != NriResult_SUCCESS) r->gpu->copy_queue = r->gpu->graphics_queue;

    r->gpu->work_queue = r->gpu->compute_queue ? r->gpu->compute_queue : r->gpu->graphics_queue;

    return true;
}

static bool create_gpu_timestamps(RENDERER *r) {
    if (!r || !r->gpu->device) return false;

    const NriDeviceDesc *device = r->gpu->core.GetDeviceDesc(r->gpu->device);

    if (!device || !device->features.timestamp || !device->other.timestampFrequencyHz) return true;

    const NriQueryPoolDesc query_desc = {.queryType = NriQueryType_TIMESTAMP, .capacity = TIMESTAMP_CAPACITY};

    if (r->gpu->core.CreateQueryPool(r->gpu->device, &query_desc, &r->gpu->timestamp_pool) != NriResult_SUCCESS) return true;

    r->gpu->timestamp_query_size = r->gpu->core.GetQuerySize(r->gpu->timestamp_pool);

    if (!r->gpu->timestamp_query_size) {
        r->gpu->core.DestroyQueryPool(r->gpu->timestamp_pool);
        r->gpu->timestamp_pool = NULL;

        return true;
    }

    const NriBufferDesc readback_desc = {.size = (uint64_t)r->gpu->timestamp_query_size * TIMESTAMP_CAPACITY};

    if (r->gpu->core.CreateCommittedBuffer(r->gpu->device, NriMemoryLocation_HOST_READBACK, 0.0f, &readback_desc, &r->gpu->timestamp_readback) !=
        NriResult_SUCCESS) {
        r->gpu->core.DestroyQueryPool(r->gpu->timestamp_pool);
        r->gpu->timestamp_pool = NULL;
        r->gpu->timestamp_query_size = 0u;

        return true;
    }

    r->gpu->timestamp_supported = true;

    return true;
}

static void destroy_gpu_timestamps(RENDERER *r) {
    if (!r) return;

    if (r->gpu->timestamp_readback) r->gpu->core.DestroyBuffer(r->gpu->timestamp_readback);

    if (r->gpu->timestamp_pool) r->gpu->core.DestroyQueryPool(r->gpu->timestamp_pool);
    r->gpu->timestamp_readback = NULL;
    r->gpu->timestamp_pool = NULL;
    r->gpu->timestamp_query_size = 0u;
    r->gpu->timestamp_supported = false;
}

bool gpu_timestamp_begin(RENDERER *r, NriCommandBuffer *cmd, uint32_t slot) {
    if (!r || !cmd || !r->gpu->timestamp_supported) return true;

    if (slot + 1u >= TIMESTAMP_CAPACITY) return false;
    r->gpu->core.CmdResetQueries(cmd, r->gpu->timestamp_pool, slot, 2u);
    r->gpu->core.CmdEndQuery(cmd, r->gpu->timestamp_pool, slot);

    return true;
}

bool gpu_timestamp_end(RENDERER *r, NriCommandBuffer *cmd, uint32_t slot) {
    if (!r || !cmd || !r->gpu->timestamp_supported) return true;

    if (slot + 1u >= TIMESTAMP_CAPACITY) return false;
    r->gpu->core.CmdEndQuery(cmd, r->gpu->timestamp_pool, slot + 1u);
    r->gpu->core.CmdCopyQueries(cmd, r->gpu->timestamp_pool, slot, 2u, r->gpu->timestamp_readback, (uint64_t)slot * r->gpu->timestamp_query_size);

    return true;
}

void gpu_timestamp_log(RENDERER *r, uint32_t slot, const char *label) {
    if (!r || !r->gpu->timestamp_supported || slot + 1u >= TIMESTAMP_CAPACITY || !label) return;

    const NriDeviceDesc *device = r->gpu->core.GetDeviceDesc(r->gpu->device);

    if (!device || !device->other.timestampFrequencyHz) return;

    const uint64_t offset = (uint64_t)slot * r->gpu->timestamp_query_size;
    const uint64_t bytes = (uint64_t)r->gpu->timestamp_query_size * 2u;
    const uint8_t *mapped = r->gpu->core.MapBuffer(r->gpu->timestamp_readback, offset, bytes);

    if (!mapped) return;

    uint64_t begin = 0u;
    uint64_t end = 0u;

    memcpy(&begin, mapped, sizeof(begin));
    memcpy(&end, mapped + r->gpu->timestamp_query_size, sizeof(end));
    r->gpu->core.UnmapBuffer(r->gpu->timestamp_readback);

    if (end >= begin) {
        const double ms = (double)(end - begin) * 1000.0 / (double)device->other.timestampFrequencyHz;

        SDL_Log("GPU: %s %.3f ms", label, ms);
    }
}

static NriShaderDesc compile_shader(const char *path, const char *entrypoint, const char *define, NriStageBits stage) {
    (void)path;

    size_t spirv_size = 0;
    Uint8 *spirv = load_spirv(define, &spirv_size);
    if (!spirv) return (NriShaderDesc){0};

    return (NriShaderDesc){.stage = stage, .bytecode = spirv, .size = spirv_size, .entryPointName = entrypoint};
}

NriPipeline *gpu_compile_compute(RENDERER *r, NriPipelineLayout *layout, const char *path, const char *entrypoint, const char *define) {
    (void)path;

    size_t spirv_size = 0;
    Uint8 *spirv = load_spirv(define, &spirv_size);

    if (!spirv) return NULL;

    const NriShaderDesc shader = {.stage = NriStageBits_COMPUTE_SHADER, .bytecode = spirv, .size = spirv_size, .entryPointName = entrypoint};
    const NriComputePipelineDesc desc = {.pipelineLayout = layout, .shader = shader, .cache = r->gpu->pipeline_cache};
    NriPipeline *pipeline = NULL;
    NriResult result = r->gpu->core.CreateComputePipeline(r->gpu->device, &desc, &pipeline);

    SDL_free(spirv);

    if (result != NriResult_SUCCESS) {
        SDL_Log("compute pipeline creation failed for %s:%s", define, entrypoint);

        return NULL;
    }

    return pipeline;
}

static NriPipeline *make_surface_pipeline(RENDERER *r, NriCoreInterface *core, NriPipelineLayout *layout, const NriShaderDesc *vs, const NriShaderDesc *ps) {

    const NriVertexStreamDesc vb = {.bindingSlot = 0, .stepRate = NriVertexStreamStepRate_PER_VERTEX, .stride = (uint16_t)sizeof(RENDER_VERTEX)};

    const NriVertexAttributeDesc attrs[4] = {{.d3d = {.semanticName = "TEXCOORD", .semanticIndex = 0},
                                              .vk = {.location = 0},
                                              .offset = (uint32_t)offsetof(RENDER_VERTEX, x),
                                              .format = NriFormat_RGB32_SFLOAT,
                                              .streamIndex = 0},
                                             {.d3d = {.semanticName = "TEXCOORD", .semanticIndex = 1},
                                              .vk = {.location = 1},
                                              .offset = (uint32_t)offsetof(RENDER_VERTEX, nx),
                                              .format = NriFormat_RGB32_SFLOAT,
                                              .streamIndex = 0},
                                             {.d3d = {.semanticName = "TEXCOORD", .semanticIndex = 2},
                                              .vk = {.location = 2},
                                              .offset = (uint32_t)offsetof(RENDER_VERTEX, u),
                                              .format = NriFormat_RG32_SFLOAT,
                                              .streamIndex = 0},
                                             {.d3d = {.semanticName = "TEXCOORD", .semanticIndex = 3},
                                              .vk = {.location = 3},
                                              .offset = (uint32_t)offsetof(RENDER_VERTEX, lu),
                                              .format = NriFormat_RG32_SFLOAT,
                                              .streamIndex = 0}};

    const NriVertexInputDesc vertex_input = {.attributes = attrs, .attributeNum = 4, .streams = &vb, .streamNum = 1};

    const NriColorAttachmentDesc targets[2] = {{.format = NriFormat_RGBA16_SFLOAT, .colorWriteMask = NriColorWriteBits_RGBA},
                                               {.format = NriFormat_RGBA16_SFLOAT, .colorWriteMask = NriColorWriteBits_RGBA}};

    const NriMultisampleDesc multisample = {.sampleMask = NRI_ALL, .sampleNum = 1};

    const NriShaderDesc shaders[2] = {*vs, *ps};

    const NriGraphicsPipelineDesc desc = {
        .pipelineLayout = layout,

        .vertexInput = &vertex_input,

        .inputAssembly = {.topology = NriTopology_TRIANGLE_LIST},

        .rasterization = {.fillMode = NriFillMode_SOLID, .cullMode = NriCullMode_NONE, .frontCounterClockwise = true, .depthClamp = false},

        .multisample = &multisample,

        .outputMerger = {.colors = targets,
                         .colorNum = 2,

                         .depth = {.compareOp = NriCompareOp_LESS, .write = true},

                         .depthStencilFormat = r->depth_format},

        .shaders = shaders,
        .shaderNum = 2,
        .cache = r->gpu->pipeline_cache};

    NriPipeline *pipeline = NULL;

    if (core->CreateGraphicsPipeline(r->gpu->device, &desc, &pipeline) != NriResult_SUCCESS) {
        return NULL;
    }

    return pipeline;
}

void gpu_clear_temporary(RENDERER *r);
static bool begin_work_commands(RENDERER *r, NriCommandAllocator **allocator, NriCommandBuffer **command_buffer);
static bool submit_work_commands(RENDERER *r, NriCommandAllocator *allocator, NriCommandBuffer *command_buffer, bool wait);
static void abort_work_commands(RENDERER *r, NriCommandAllocator *allocator, NriCommandBuffer *command_buffer);
static void work_drain(RENDERER *r);

NriResult gpu_begin_commands(RENDERER *r, NriCommandAllocator **allocator, NriCommandBuffer **command_buffer) {
    return begin_work_commands(r, allocator, command_buffer) ? NriResult_SUCCESS : NriResult_FAILURE;
}

bool gpu_submit_commands(RENDERER *r, NriCommandAllocator *allocator, NriCommandBuffer *command_buffer) {
    return submit_work_commands(r, allocator, command_buffer, true);
}

static bool submit_commands_async(RENDERER *r, NriCommandAllocator *allocator, NriCommandBuffer *command_buffer) {
    return submit_work_commands(r, allocator, command_buffer, false);
}

void gpu_abort_commands(RENDERER *r, NriCommandAllocator *allocator, NriCommandBuffer *cmd) {
    abort_work_commands(r, allocator, cmd);
}

/* Descriptor sets match the HLSL register spaces in shaders/. */
bool gpu_create_pipeline_layout(RENDERER *r, NriPipelineLayout **out, const NriDescriptorType *types[4], const uint8_t counts[4], NriStageBits stages) {
    NriDescriptorRangeDesc ranges[4][16] = {0};
    NriDescriptorSetDesc sets[4] = {0};

    for (uint32_t set = 0; set < 4; ++set) {
        sets[set].registerSpace = set;
        sets[set].ranges = ranges[set];
        sets[set].rangeNum = counts[set];

        for (uint32_t i = 0; i < counts[set]; ++i) {
            NriDescriptorType type = types[set][i];
            uint32_t reg = 0;

            for (uint32_t j = 0; j < i; ++j) {
                const NriDescriptorType previous = types[set][j];
                const bool sampled_namespace = (type == NriDescriptorType_TEXTURE || type == NriDescriptorType_STRUCTURED_BUFFER) &&
                                               (previous == NriDescriptorType_TEXTURE || previous == NriDescriptorType_STRUCTURED_BUFFER);

                const bool storage_namespace = (type == NriDescriptorType_STORAGE_TEXTURE || type == NriDescriptorType_STORAGE_STRUCTURED_BUFFER) &&
                                               (previous == NriDescriptorType_STORAGE_TEXTURE || previous == NriDescriptorType_STORAGE_STRUCTURED_BUFFER);

                if (previous == type || sampled_namespace || storage_namespace) ++reg;
            }

            ranges[set][i] = (NriDescriptorRangeDesc){.baseRegisterIndex = reg, .descriptorNum = 1, .descriptorType = type, .shaderStages = stages};
        }
    }

    const NriPipelineLayoutDesc desc = {.descriptorSets = sets, .descriptorSetNum = 4, .rootRegisterSpace = 4, .shaderStages = stages};

    return r->gpu->core.CreatePipelineLayout(r->gpu->device, &desc, out) == NriResult_SUCCESS;
}

static bool create_pipeline_layouts(RENDERER *r);

static bool create_surface_layout(RENDERER *r) {
    static const NriDescriptorType camera[] = {NriDescriptorType_CONSTANT_BUFFER};

    static const NriDescriptorType material[] = {NriDescriptorType_TEXTURE, NriDescriptorType_TEXTURE, NriDescriptorType_TEXTURE, NriDescriptorType_TEXTURE,
                                                 NriDescriptorType_TEXTURE, NriDescriptorType_TEXTURE, NriDescriptorType_SAMPLER, NriDescriptorType_SAMPLER,
                                                 NriDescriptorType_SAMPLER, NriDescriptorType_SAMPLER, NriDescriptorType_SAMPLER, NriDescriptorType_SAMPLER};

    static const NriDescriptorType uniform[] = {NriDescriptorType_CONSTANT_BUFFER};

    const NriDescriptorType *sets[4] = {NULL, camera, material, uniform};
    const uint8_t counts[4] = {0, 1, 12, 1};

    return gpu_create_pipeline_layout(r, &r->surface_layout, sets, counts, NriStageBits_VERTEX_SHADER | NriStageBits_FRAGMENT_SHADER);
}

static bool create_line_layout(RENDERER *r) {
    static const NriDescriptorType camera[] = {NriDescriptorType_CONSTANT_BUFFER};
    const NriDescriptorType *sets[4] = {NULL, camera, NULL, NULL};
    const uint8_t counts[4] = {0, 1, 0, 0};

    return gpu_create_pipeline_layout(r, &r->line_layout, sets, counts, NriStageBits_VERTEX_SHADER | NriStageBits_FRAGMENT_SHADER);
}

static bool create_sky_layout(RENDERER *r) {
    static const NriDescriptorType uniform[] = {NriDescriptorType_CONSTANT_BUFFER};

    const NriDescriptorType *sets[4] = {NULL, NULL, NULL, uniform};
    const uint8_t counts[4] = {0, 0, 0, 1};

    return gpu_create_pipeline_layout(r, &r->sky_layout, sets, counts, NriStageBits_VERTEX_SHADER | NriStageBits_FRAGMENT_SHADER);
}

bool gpu_create_compute_layout(RENDERER *r, NriPipelineLayout **out, const NriDescriptorType *sources, uint8_t source_num, NriDescriptorType output_type,
                               bool has_uniform) {
    static const NriDescriptorType output_texture[] = {NriDescriptorType_STORAGE_TEXTURE};

    static const NriDescriptorType output_buffer[] = {NriDescriptorType_STORAGE_STRUCTURED_BUFFER};

    static const NriDescriptorType uniform[] = {NriDescriptorType_CONSTANT_BUFFER};

    const NriDescriptorType *sets[4] = {sources, output_type == NriDescriptorType_STORAGE_TEXTURE ? output_texture : output_buffer,
                                        has_uniform ? uniform : NULL, NULL};

    const uint8_t counts[4] = {source_num, 1, has_uniform ? 1 : 0, 0};

    return gpu_create_pipeline_layout(r, out, sets, counts, NriStageBits_COMPUTE_SHADER);
}

static bool create_ssao_layout(RENDERER *r) {
    static const NriDescriptorType src[] = {NriDescriptorType_TEXTURE, NriDescriptorType_SAMPLER};

    return gpu_create_compute_layout(r, &r->ssao_layout, src, 2, NriDescriptorType_STORAGE_TEXTURE, true);
}

static bool create_bloom_layout(RENDERER *r) {
    static const NriDescriptorType src[] = {NriDescriptorType_TEXTURE, NriDescriptorType_SAMPLER};

    return gpu_create_compute_layout(r, &r->bloom_layout, src, 2, NriDescriptorType_STORAGE_TEXTURE, true);
}

static bool create_grade_layout(RENDERER *r) {
    return gpu_create_compute_layout(r, &r->grade_layout, NULL, 0, NriDescriptorType_STORAGE_TEXTURE, false);
}

static bool create_volume_layout(RENDERER *r) {
    static const NriDescriptorType src[] = {NriDescriptorType_TEXTURE, NriDescriptorType_SAMPLER, NriDescriptorType_STRUCTURED_BUFFER,
                                            NriDescriptorType_STRUCTURED_BUFFER};

    return gpu_create_compute_layout(r, &r->volume_layout, src, 4, NriDescriptorType_STORAGE_TEXTURE, true);
}

static bool create_volume_compose_layout(RENDERER *r) {
    static const NriDescriptorType src[] = {NriDescriptorType_TEXTURE, NriDescriptorType_TEXTURE, NriDescriptorType_TEXTURE,
                                            NriDescriptorType_SAMPLER, NriDescriptorType_SAMPLER, NriDescriptorType_SAMPLER};

    return gpu_create_compute_layout(r, &r->volume_compose_layout, src, 6, NriDescriptorType_STORAGE_TEXTURE, true);
}

static bool create_compose_layout(RENDERER *r) {
    static const NriDescriptorType src[] = {NriDescriptorType_TEXTURE, NriDescriptorType_TEXTURE, NriDescriptorType_TEXTURE, NriDescriptorType_TEXTURE,
                                            NriDescriptorType_SAMPLER, NriDescriptorType_SAMPLER, NriDescriptorType_SAMPLER, NriDescriptorType_SAMPLER};

    static const NriDescriptorType uniform[] = {NriDescriptorType_CONSTANT_BUFFER};

    const NriDescriptorType *sets[4] = {NULL, NULL, src, uniform};
    const uint8_t counts[4] = {0, 0, 8, 1};

    return gpu_create_pipeline_layout(r, &r->compose_layout, sets, counts, NriStageBits_VERTEX_SHADER | NriStageBits_FRAGMENT_SHADER);
}

static bool create_descriptor_pool_object(RENDERER *r, NriDescriptorPool **pool) {
    const NriDescriptorPoolDesc desc = {.descriptorSetMaxNum = 8192,
                                        .samplerMaxNum = 8192,
                                        .textureMaxNum = 8192,
                                        .storageTextureMaxNum = 8192,
                                        .structuredBufferMaxNum = 8192,
                                        .storageStructuredBufferMaxNum = 8192,
                                        .constantBufferMaxNum = 8192};

    return r->gpu->core.CreateDescriptorPool(r->gpu->device, &desc, pool) == NriResult_SUCCESS;
}

static bool create_descriptor_pool(RENDERER *r) {
    return create_descriptor_pool_object(r, &r->gpu->descriptor_pool);
}

static bool track_descriptor_array(RENDERER *r, NriDescriptor ***items, uint32_t *count, uint32_t *capacity, NriDescriptor *descriptor) {
    if (!descriptor) return false;

    if (*count == *capacity) {
        uint32_t cap = *capacity ? *capacity * 2u : 64u;
        NriDescriptor **data = realloc(*items, cap * sizeof(*data));

        if (!data) {
            r->gpu->core.DestroyDescriptor(descriptor);

            return false;
        }
        *items = data;
        *capacity = cap;
    }

    (*items)[(*count)++] = descriptor;

    return true;
}

static bool track_buffer_array(RENDERER *r, NriBuffer ***items, uint32_t *count, uint32_t *capacity, NriBuffer *buffer) {
    if (!buffer) return false;

    if (*count == *capacity) {
        uint32_t cap = *capacity ? *capacity * 2u : 64u;
        NriBuffer **data = realloc(*items, cap * sizeof(*data));

        if (!data) {
            r->gpu->core.DestroyBuffer(buffer);

            return false;
        }
        *items = data;
        *capacity = cap;
    }

    (*items)[(*count)++] = buffer;

    return true;
}

static bool track_descriptor(RENDERER *r, NriDescriptor *descriptor) {
    if (r->gpu->active_frame) {
        FRAME_CONTEXT *frame = r->gpu->active_frame;

        return track_descriptor_array(r, &frame->temporary_descriptors, &frame->temporary_descriptor_num, &frame->temporary_descriptor_cap, descriptor);
    }

    return track_descriptor_array(r, &r->gpu->temporary_descriptors, &r->gpu->temporary_descriptor_num, &r->gpu->temporary_descriptor_cap, descriptor);
}

static bool track_buffer(RENDERER *r, NriBuffer *buffer) {
    if (r->gpu->active_frame) {
        FRAME_CONTEXT *frame = r->gpu->active_frame;

        return track_buffer_array(r, &frame->temporary_buffers, &frame->temporary_buffer_num, &frame->temporary_buffer_cap, buffer);
    }

    return track_buffer_array(r, &r->gpu->temporary_buffers, &r->gpu->temporary_buffer_num, &r->gpu->temporary_buffer_cap, buffer);
}

void gpu_clear_temporary(RENDERER *r) {
    for (uint32_t i = 0; i < r->gpu->temporary_descriptor_num; ++i) r->gpu->core.DestroyDescriptor(r->gpu->temporary_descriptors[i]);

    for (uint32_t i = 0; i < r->gpu->temporary_buffer_num; ++i) r->gpu->core.DestroyBuffer(r->gpu->temporary_buffers[i]);
    r->gpu->temporary_descriptor_num = r->gpu->temporary_buffer_num = 0;

    if (r->gpu->descriptor_pool) r->gpu->core.ResetDescriptorPool(r->gpu->descriptor_pool);
}

static void clear_frame_temporary(RENDERER *r, FRAME_CONTEXT *frame) {
    if (!frame) return;

    for (uint32_t i = 0; i < frame->temporary_descriptor_num; ++i) r->gpu->core.DestroyDescriptor(frame->temporary_descriptors[i]);

    for (uint32_t i = 0; i < frame->temporary_buffer_num; ++i) r->gpu->core.DestroyBuffer(frame->temporary_buffers[i]);
    frame->temporary_descriptor_num = 0;
    frame->temporary_buffer_num = 0;
    frame->uniform_offset = 0u;

    if (frame->descriptor_pool) r->gpu->core.ResetDescriptorPool(frame->descriptor_pool);
}

static bool create_uniform_ring(RENDERER *r, FRAME_CONTEXT *context) {
    if (!r || !context) return false;

    const NriBufferDesc desc = {.size = UNIFORM_RING_BYTES, .usage = NriBufferUsageBits_CONSTANT};

    return r->gpu->core.CreateCommittedBuffer(r->gpu->device, NriMemoryLocation_HOST_UPLOAD, 0.0f, &desc, &context->uniform_buffer) == NriResult_SUCCESS;
}

static bool create_work_contexts(RENDERER *r) {
    if (!r || !r->gpu->device || !r->gpu->work_queue) return false;

    r->gpu->work_contexts = calloc(WORK_QUEUE_DEPTH, sizeof(*r->gpu->work_contexts));

    if (!r->gpu->work_contexts) return false;

    if (r->gpu->core.CreateFence(r->gpu->device, 0u, &r->gpu->work_fence) != NriResult_SUCCESS) return false;

    for (uint32_t i = 0; i < WORK_QUEUE_DEPTH; ++i) {
        FRAME_CONTEXT *work = &r->gpu->work_contexts[i];

        if (!create_descriptor_pool_object(r, &work->descriptor_pool) || !create_uniform_ring(r, work) ||
            r->gpu->core.CreateCommandAllocator(r->gpu->work_queue, &work->allocator) != NriResult_SUCCESS ||
            r->gpu->core.CreateCommandBuffer(work->allocator, &work->command_buffer) != NriResult_SUCCESS)
            return false;
    }

    r->gpu->work_next_fence = 1u;

    return true;
}

static void work_drain(RENDERER *r) {
    if (!r || !r->gpu->work_fence || r->gpu->work_next_fence <= 1u) return;
    r->gpu->core.Wait(r->gpu->work_fence, r->gpu->work_next_fence - 1u);
}

static void destroy_work_contexts(RENDERER *r) {
    if (!r) return;
    work_drain(r);

    if (r->gpu->work_contexts) {
        for (uint32_t i = 0; i < WORK_QUEUE_DEPTH; ++i) {
            FRAME_CONTEXT *work = &r->gpu->work_contexts[i];
            clear_frame_temporary(r, work);

            if (work->command_buffer) r->gpu->core.DestroyCommandBuffer(work->command_buffer);

            if (work->allocator) r->gpu->core.DestroyCommandAllocator(work->allocator);

            if (work->descriptor_pool) r->gpu->core.DestroyDescriptorPool(work->descriptor_pool);

            if (work->uniform_buffer) r->gpu->core.DestroyBuffer(work->uniform_buffer);

            free(work->temporary_descriptors);
            free(work->temporary_buffers);
        }

        free(r->gpu->work_contexts);
    }

    r->gpu->work_contexts = NULL;
    r->gpu->active_work = NULL;

    if (r->gpu->work_fence) r->gpu->core.DestroyFence(r->gpu->work_fence);
    r->gpu->work_fence = NULL;
}

static bool begin_work_commands(RENDERER *r, NriCommandAllocator **allocator, NriCommandBuffer **command_buffer) {
    if (!r || !r->gpu->work_contexts || !r->gpu->work_fence || !r->gpu->work_queue || !allocator || !command_buffer) return false;

    FRAME_CONTEXT *work = &r->gpu->work_contexts[r->gpu->work_index % WORK_QUEUE_DEPTH];

    if (work->fence_value && r->gpu->core.GetFenceValue(r->gpu->work_fence) < work->fence_value) r->gpu->core.Wait(r->gpu->work_fence, work->fence_value);

    clear_frame_temporary(r, work);
    r->gpu->core.ResetCommandAllocator(work->allocator);
    r->current_graphics_layout = r->current_compute_layout = NULL;
    r->gpu->active_work = work;
    r->gpu->active_frame = work;

    if (r->gpu->core.BeginCommandBuffer(work->command_buffer, work->descriptor_pool) != NriResult_SUCCESS) {
        r->gpu->active_work = NULL;
        r->gpu->active_frame = NULL;

        return false;
    }

    *allocator = work->allocator;
    *command_buffer = work->command_buffer;
    return true;
}

static bool submit_work_commands(RENDERER *r, NriCommandAllocator *allocator, NriCommandBuffer *command_buffer, bool wait) {
    FRAME_CONTEXT *work = r ? r->gpu->active_work : NULL;

    if (!r || !work || work->allocator != allocator || work->command_buffer != command_buffer) return false;

    bool good = r->gpu->core.EndCommandBuffer(command_buffer) == NriResult_SUCCESS;
    uint64_t value = 0u;

    if (good) {
        value = r->gpu->work_next_fence++;
        const NriFenceSubmitDesc signal = {.fence = r->gpu->work_fence, .value = value};
        NriFenceSubmitDesc upload_wait = {0};
        uint32_t upload_wait_num = 0u;

        if (r->gpu->upload && r->gpu->upload->fence && r->gpu->upload->next_fence_value > 1u) {
            upload_wait = (NriFenceSubmitDesc){.fence = r->gpu->upload->fence, .value = r->gpu->upload->next_fence_value - 1u, .stages = NriStageBits_ALL};
            upload_wait_num = 1u;
        }

        const NriQueueSubmitDesc submit = {.waitFences = upload_wait_num ? &upload_wait : NULL,
                                           .waitFenceNum = upload_wait_num,
                                           .commandBuffers = (const NriCommandBuffer *const *)&command_buffer,
                                           .commandBufferNum = 1u,
                                           .signalFences = &signal,
                                           .signalFenceNum = 1u};

        good = r->gpu->core.QueueSubmit(r->gpu->work_queue, &submit) == NriResult_SUCCESS;
    }

    if (good) {
        work->fence_value = value;
        r->gpu->work_index++;
    }

    r->gpu->active_work = NULL;
    r->gpu->active_frame = NULL;

    if (good && wait) r->gpu->core.Wait(r->gpu->work_fence, value);

    return good;
}

static void abort_work_commands(RENDERER *r, NriCommandAllocator *allocator, NriCommandBuffer *command_buffer) {
    if (!r) return;

    FRAME_CONTEXT *work = r->gpu->active_work;

    if (!work || work->allocator != allocator || work->command_buffer != command_buffer) return;

    if (work->command_buffer) r->gpu->core.DestroyCommandBuffer(work->command_buffer);

    if (work->allocator) r->gpu->core.DestroyCommandAllocator(work->allocator);

    work->command_buffer = NULL;
    work->allocator = NULL;
    clear_frame_temporary(r, work);
    r->gpu->active_work = NULL;
    r->gpu->active_frame = NULL;

    if (r->gpu->core.CreateCommandAllocator(r->gpu->work_queue, &work->allocator) == NriResult_SUCCESS)
        r->gpu->core.CreateCommandBuffer(work->allocator, &work->command_buffer);
}

static bool create_frame_contexts(RENDERER *r) {
    r->gpu->frame_contexts = calloc(FRAME_QUEUE_DEPTH, sizeof(*r->gpu->frame_contexts));

    if (!r->gpu->frame_contexts) return false;

    if (r->gpu->core.CreateFence(r->gpu->device, 0, &r->gpu->frame_fence) != NriResult_SUCCESS) return false;

    for (uint32_t i = 0; i < FRAME_QUEUE_DEPTH; ++i) {
        FRAME_CONTEXT *frame = &r->gpu->frame_contexts[i];

        if (!create_descriptor_pool_object(r, &frame->descriptor_pool) || !create_uniform_ring(r, frame) ||
            r->gpu->core.CreateCommandAllocator(r->gpu->graphics_queue, &frame->allocator) != NriResult_SUCCESS ||
            r->gpu->core.CreateCommandBuffer(frame->allocator, &frame->command_buffer) != NriResult_SUCCESS)
            return false;
    }

    return true;
}

static void destroy_frame_contexts(RENDERER *r) {
    if (r->gpu->frame_contexts) {
        for (uint32_t i = 0; i < FRAME_QUEUE_DEPTH; ++i) {
            FRAME_CONTEXT *frame = &r->gpu->frame_contexts[i];
            clear_frame_temporary(r, frame);

            if (frame->command_buffer) r->gpu->core.DestroyCommandBuffer(frame->command_buffer);

            if (frame->allocator) r->gpu->core.DestroyCommandAllocator(frame->allocator);

            if (frame->descriptor_pool) r->gpu->core.DestroyDescriptorPool(frame->descriptor_pool);

            if (frame->uniform_buffer) r->gpu->core.DestroyBuffer(frame->uniform_buffer);
            free(frame->temporary_descriptors);
            free(frame->temporary_buffers);
        }

        free(r->gpu->frame_contexts);
    }

    r->gpu->frame_contexts = NULL;
    r->gpu->active_frame = NULL;

    if (r->gpu->frame_fence) r->gpu->core.DestroyFence(r->gpu->frame_fence);
    r->gpu->frame_fence = NULL;
}

static bool begin_frame_commands(RENDERER *r, FRAME_CONTEXT **out_frame, NriCommandBuffer **out_command_buffer) {
    if (!r || !r->gpu->frame_contexts || !r->gpu->frame_fence || !out_frame || !out_command_buffer) return false;

    const uint64_t wait_value = r->gpu->frame_index >= FRAME_QUEUE_DEPTH ? 1u + r->gpu->frame_index - FRAME_QUEUE_DEPTH : 0u;

    r->gpu->core.Wait(r->gpu->frame_fence, wait_value);

    FRAME_CONTEXT *frame = &r->gpu->frame_contexts[r->gpu->frame_index % FRAME_QUEUE_DEPTH];
    clear_frame_temporary(r, frame);
    r->gpu->core.ResetCommandAllocator(frame->allocator);
    r->current_graphics_layout = r->current_compute_layout = NULL;
    r->gpu->active_frame = frame;

    if (r->gpu->core.BeginCommandBuffer(frame->command_buffer, frame->descriptor_pool) != NriResult_SUCCESS) {
        r->gpu->active_frame = NULL;

        return false;
    }

    *out_frame = frame;
    *out_command_buffer = frame->command_buffer;
    return true;
}

static void abort_frame_commands(RENDERER *r, FRAME_CONTEXT *frame) {
    if (!r || !frame) return;

    if (frame->command_buffer) r->gpu->core.DestroyCommandBuffer(frame->command_buffer);

    if (frame->allocator) r->gpu->core.DestroyCommandAllocator(frame->allocator);
    frame->command_buffer = NULL;
    frame->allocator = NULL;
    r->gpu->active_frame = NULL;
    clear_frame_temporary(r, frame);

    if (r->gpu->core.CreateCommandAllocator(r->gpu->graphics_queue, &frame->allocator) == NriResult_SUCCESS)
        r->gpu->core.CreateCommandBuffer(frame->allocator, &frame->command_buffer);
}

NriDescriptor *gpu_create_texture_view(RENDERER *r, NriTexture *texture, NriTextureView type) {
    if (!texture) return NULL;

    NriDescriptor *view = NULL;

    const NriTextureViewDesc desc = {
        .texture = texture, .type = type, .format = r->gpu->core.GetTextureDesc(texture)->format, .mipNum = 1, .layerNum = 1, .sliceNum = 1};

    if (r->gpu->core.CreateTextureView(&desc, &view) != NriResult_SUCCESS) return NULL;

    return track_descriptor(r, view) ? view : NULL;
}

NriDescriptor *gpu_create_buffer_view(RENDERER *r, NriBuffer *buffer, NriBufferView type, uint32_t stride) {
    if (!buffer) return NULL;

    NriDescriptor *view = NULL;

    const NriBufferViewDesc desc = {.buffer = buffer, .type = type, .offset = 0, .size = r->gpu->core.GetBufferDesc(buffer)->size, .structureStride = stride};

    if (r->gpu->core.CreateBufferView(&desc, &view) != NriResult_SUCCESS) return NULL;

    return track_descriptor(r, view) ? view : NULL;
}

static NriDescriptor *uniform_view(RENDERER *r, const void *data, size_t size) {
    if (!r || !data || !size) return NULL;

    FRAME_CONTEXT *context = r->gpu->active_frame;
    const NriDeviceDesc *device = r->gpu->core.GetDeviceDesc(r->gpu->device);
    uint64_t alignment = device ? device->memoryAlignment.constantBufferOffset : 256u;

    if (alignment < 16u) alignment = 16u;

    const uint64_t view_size = ((uint64_t)size + alignment - 1u) / alignment * alignment;

    if (context && context->uniform_buffer) {
        const uint64_t offset = (context->uniform_offset + alignment - 1u) / alignment * alignment;

        if (offset + view_size <= UNIFORM_RING_BYTES) {
            void *mapped = r->gpu->core.MapBuffer(context->uniform_buffer, offset, view_size);

            if (mapped) {
                memset(mapped, 0, view_size);
                memcpy(mapped, data, size);
                r->gpu->core.UnmapBuffer(context->uniform_buffer);

                NriDescriptor *view = NULL;
                const NriBufferViewDesc desc = {.buffer = context->uniform_buffer, .type = NriBufferView_CONSTANT_BUFFER, .offset = offset, .size = view_size};

                if (r->gpu->core.CreateBufferView(&desc, &view) == NriResult_SUCCESS && track_descriptor(r, view)) {
                    context->uniform_offset = offset + view_size;

                    return view;
                }
            }
        }
    }

    const NriBufferDesc desc = {.size = view_size, .usage = NriBufferUsageBits_CONSTANT};
    NriBuffer *buffer = NULL;

    if (r->gpu->core.CreateCommittedBuffer(r->gpu->device, NriMemoryLocation_HOST_UPLOAD, 0.0f, &desc, &buffer) != NriResult_SUCCESS) return NULL;

    if (!track_buffer(r, buffer)) return NULL;

    void *mapped = r->gpu->core.MapBuffer(buffer, 0, desc.size);

    if (!mapped) return NULL;
    memset(mapped, 0, desc.size);
    memcpy(mapped, data, size);
    r->gpu->core.UnmapBuffer(buffer);

    return gpu_create_buffer_view(r, buffer, NriBufferView_CONSTANT_BUFFER, 0);
}

bool gpu_bind_descriptor_set(RENDERER *r, NriCommandBuffer *cmd, NriPipelineLayout *layout, NriBindPoint point, uint32_t set_index,
                             NriDescriptor *const *descriptors, uint32_t count) {
    NriDescriptorSet *set = NULL;
    NriDescriptorPool *pool = r->gpu->active_frame ? r->gpu->active_frame->descriptor_pool : r->gpu->descriptor_pool;

    if (r->gpu->core.AllocateDescriptorSets(pool, layout, set_index, &set, 1, 0) != NriResult_SUCCESS) return false;

    for (uint32_t i = 0; i < count; ++i) {
        if (!descriptors[i]) return false;

        const NriDescriptor *d = descriptors[i];

        const NriUpdateDescriptorRangeDesc update = {.descriptorSet = set, .rangeIndex = i, .descriptors = &d, .descriptorNum = 1};

        r->gpu->core.UpdateDescriptorRanges(&update, 1);
    }

    NriPipelineLayout **current = point == NriBindPoint_GRAPHICS ? &r->current_graphics_layout : &r->current_compute_layout;

    if (*current != layout) {
        r->gpu->core.CmdSetPipelineLayout(cmd, point, layout);
        *current = layout;
    }

    r->gpu->core.CmdSetDescriptorSet(cmd, &(NriSetDescriptorSetDesc){.setIndex = set_index, .descriptorSet = set, .bindPoint = point});

    return true;
}

bool gpu_bind_uniform_data(RENDERER *r, NriCommandBuffer *cmd, NriPipelineLayout *layout, NriBindPoint point, uint32_t set, const void *data, size_t size) {
    NriDescriptor *view = uniform_view(r, data, size);

    return view && gpu_bind_descriptor_set(r, cmd, layout, point, set, &view, 1);
}

bool gpu_transition_texture(RENDERER *r, NriCommandBuffer *cmd, NriTexture *texture, NriAccessBits access, NriLayout layout, NriStageBits stages);

static bool bind_fx_resources(RENDERER *r, NriCommandBuffer *cmd, NriPipeline *pipeline, NriTexture *source, NriTexture *destination,
                              NriDescriptor *sampler_desc, const void *uniforms, uint32_t size) {
    if ((source && !gpu_transition_texture(r, cmd, source, NriAccessBits_SHADER_RESOURCE, NriLayout_SHADER_RESOURCE, NriStageBits_COMPUTE_SHADER)) ||
        !gpu_transition_texture(r, cmd, destination, NriAccessBits_SHADER_RESOURCE_STORAGE, NriLayout_SHADER_RESOURCE_STORAGE, NriStageBits_COMPUTE_SHADER))
        return false;

    NriPipelineLayout *layout = pipeline == r->fx.grade_pipeline ? r->grade_layout : pipeline == r->fx.bloom_pipeline ? r->bloom_layout : r->ssao_layout;

    NriDescriptor *dst = gpu_create_texture_view(r, destination, NriTextureView_STORAGE_TEXTURE);

    if (source) {
        NriDescriptor *src[] = {gpu_create_texture_view(r, source, NriTextureView_TEXTURE), sampler_desc};

        if (!gpu_bind_descriptor_set(r, cmd, layout, NriBindPoint_COMPUTE, 0, src, 2)) return false;
    }

    return gpu_bind_descriptor_set(r, cmd, layout, NriBindPoint_COMPUTE, 1, &dst, 1) &&
           (!size || gpu_bind_uniform_data(r, cmd, layout, NriBindPoint_COMPUTE, 2, uniforms, size));
}

static bool bind_volume_resources(RENDERER *r, NriCommandBuffer *cmd, NriTexture *normal, NriDescriptor *sampler_desc, NriBuffer *probes, NriBuffer *beams,
                                  NriTexture *output, const void *uniforms, uint32_t size) {
    if (!gpu_transition_texture(r, cmd, normal, NriAccessBits_SHADER_RESOURCE, NriLayout_SHADER_RESOURCE, NriStageBits_COMPUTE_SHADER) ||
        !gpu_transition_texture(r, cmd, output, NriAccessBits_SHADER_RESOURCE_STORAGE, NriLayout_SHADER_RESOURCE_STORAGE, NriStageBits_COMPUTE_SHADER))
        return false;

    NriDescriptor *src[] = {gpu_create_texture_view(r, normal, NriTextureView_TEXTURE), sampler_desc,
                            gpu_create_buffer_view(r, probes, NriBufferView_STRUCTURED_BUFFER, sizeof(PROBE)),
                            gpu_create_buffer_view(r, beams, NriBufferView_STRUCTURED_BUFFER, sizeof(float))};

    NriDescriptor *dst = gpu_create_texture_view(r, output, NriTextureView_STORAGE_TEXTURE);

    return gpu_bind_descriptor_set(r, cmd, r->volume_layout, NriBindPoint_COMPUTE, 0, src, 4) &&
           gpu_bind_descriptor_set(r, cmd, r->volume_layout, NriBindPoint_COMPUTE, 1, &dst, 1) &&
           gpu_bind_uniform_data(r, cmd, r->volume_layout, NriBindPoint_COMPUTE, 2, uniforms, size);
}

static bool bind_volume_compose_resources(RENDERER *r, NriCommandBuffer *cmd, NriTexture *hdr, NriTexture *volume, NriTexture *normal,
                                          NriDescriptor *sampler_desc, NriDescriptor *depth_sampler, NriTexture *output, const void *uniforms, uint32_t size) {
    NriTexture *sources[] = {hdr, volume, normal};

    for (uint32_t i = 0; i < 3; ++i)
        if (!gpu_transition_texture(r, cmd, sources[i], NriAccessBits_SHADER_RESOURCE, NriLayout_SHADER_RESOURCE, NriStageBits_COMPUTE_SHADER)) return false;

    if (!gpu_transition_texture(r, cmd, output, NriAccessBits_SHADER_RESOURCE_STORAGE, NriLayout_SHADER_RESOURCE_STORAGE, NriStageBits_COMPUTE_SHADER))
        return false;

    NriDescriptor *src[] = {gpu_create_texture_view(r, hdr, NriTextureView_TEXTURE),
                            gpu_create_texture_view(r, volume, NriTextureView_TEXTURE),
                            gpu_create_texture_view(r, normal, NriTextureView_TEXTURE),
                            sampler_desc,
                            sampler_desc,
                            depth_sampler};

    NriDescriptor *dst = gpu_create_texture_view(r, output, NriTextureView_STORAGE_TEXTURE);

    return gpu_bind_descriptor_set(r, cmd, r->volume_compose_layout, NriBindPoint_COMPUTE, 0, src, 6) &&
           gpu_bind_descriptor_set(r, cmd, r->volume_compose_layout, NriBindPoint_COMPUTE, 1, &dst, 1) &&
           gpu_bind_uniform_data(r, cmd, r->volume_compose_layout, NriBindPoint_COMPUTE, 2, uniforms, size);
}

static bool bind_sky_resources(RENDERER *r, NriCommandBuffer *cmd, const void *data, size_t size) {
    return gpu_bind_uniform_data(r, cmd, r->sky_layout, NriBindPoint_GRAPHICS, 3, data, size);
}

static bool bind_camera_resources(RENDERER *r, NriCommandBuffer *cmd, const void *data, size_t size) {
    return gpu_bind_uniform_data(r, cmd, r->surface_layout, NriBindPoint_GRAPHICS, 1, data, size);
}

static bool bind_surface_resources(RENDERER *r, NriCommandBuffer *cmd, const RENDER_MATERIAL *material, NriTexture *lightmap, NriDescriptor *material_sampler,
                                   NriDescriptor *lightmap_sampler, const void *uniforms, size_t size) {
    NriDescriptor *src[] = {gpu_create_texture_view(r, material->base_color, NriTextureView_TEXTURE),
                            gpu_create_texture_view(r, material->metallic_roughness, NriTextureView_TEXTURE),
                            gpu_create_texture_view(r, material->normal, NriTextureView_TEXTURE),
                            gpu_create_texture_view(r, material->occlusion, NriTextureView_TEXTURE),
                            gpu_create_texture_view(r, material->emissive, NriTextureView_TEXTURE),
                            gpu_create_texture_view(r, lightmap, NriTextureView_TEXTURE),
                            material_sampler,
                            material_sampler,
                            material_sampler,
                            material_sampler,
                            material_sampler,
                            lightmap_sampler};

    return gpu_bind_descriptor_set(r, cmd, r->surface_layout, NriBindPoint_GRAPHICS, 2, src, 12) &&
           gpu_bind_uniform_data(r, cmd, r->surface_layout, NriBindPoint_GRAPHICS, 3, uniforms, size);
}

static bool bind_line_resources(RENDERER *r, NriCommandBuffer *cmd, const void *data, size_t size) {
    return gpu_bind_uniform_data(r, cmd, r->line_layout, NriBindPoint_GRAPHICS, 1, data, size);
}

static void free_shader(NriShaderDesc *shader) {
    if (shader && shader->bytecode) SDL_free((void *)shader->bytecode);

    if (shader) *shader = (NriShaderDesc){0};
}

static TEXTURE_STATE *find_texture_state(RENDERER *r, NriTexture *texture);
static bool create_swapchain(RENDERER *r, uint32_t width, uint32_t height) {
    if (!r->gpu->window || !width || !height) return false;

    NriWindow window = {0};
    const SDL_PropertiesID props = SDL_GetWindowProperties(r->gpu->window);
#if defined(__APPLE__)
    (void)props;

    if (!r->gpu->metal_view) r->gpu->metal_view = SDL_Metal_CreateView(r->gpu->window);

    if (!r->gpu->metal_view) return false;
    window.metal.caMetalLayer = SDL_Metal_GetLayer(r->gpu->metal_view);
#elif defined(_WIN32)
    window.windows.hwnd = SDL_GetPointerProperty(props, SDL_PROP_WINDOW_WIN32_HWND_POINTER, NULL);
#else
    window.wayland.display = SDL_GetPointerProperty(props, SDL_PROP_WINDOW_WAYLAND_DISPLAY_POINTER, NULL);

    if (window.wayland.display) {
        window.wayland.surface = SDL_GetPointerProperty(props, SDL_PROP_WINDOW_WAYLAND_SURFACE_POINTER, NULL);
    } else {
        window.x11.dpy = SDL_GetPointerProperty(props, SDL_PROP_WINDOW_X11_DISPLAY_POINTER, NULL);
        window.x11.window = SDL_GetNumberProperty(props, SDL_PROP_WINDOW_X11_WINDOW_NUMBER, 0);

        if (!window.x11.dpy || !window.x11.window) return false;
    }
#endif
    const NriSwapChainDesc desc = {.window = window,
                                   .queue = r->gpu->graphics_queue,
                                   .width = (NriDim_t)width,
                                   .height = (NriDim_t)height,
                                   .textureNum = FRAME_QUEUE_DEPTH + 1u,
                                   .format = NriSwapChainFormat_BT709_G22_8BIT,
                                   .flags = NriSwapChainBits_VSYNC,
                                   .queuedFrameNum = FRAME_QUEUE_DEPTH};

    if (r->gpu->swapchain_api.CreateSwapChain(r->gpu->device, &desc, &r->gpu->swapchain) != NriResult_SUCCESS) return false;

    uint32_t count = 0;

    NriTexture *const *textures = r->gpu->swapchain_api.GetSwapChainTextures(r->gpu->swapchain, &count);

    if (!textures || !count) return false;
    r->gpu->swapchain_textures = calloc(count, sizeof(*r->gpu->swapchain_textures));
    r->gpu->swapchain_frames = calloc(count, sizeof(*r->gpu->swapchain_frames));

    if (!r->gpu->swapchain_textures || !r->gpu->swapchain_frames) return false;
    r->gpu->swapchain_texture_count = count;
    r->gpu->swapchain_format = r->gpu->core.GetTextureDesc(textures[0])->format;
    r->gpu->swapchain_width = width;
    r->gpu->swapchain_height = height;

    for (uint32_t i = 0; i < count; ++i) {
        r->gpu->swapchain_textures[i] = textures[i];

        TEXTURE_STATE *state = find_texture_state(r, textures[i]);

        if (!state) return false;
        state->state = (NriAccessLayoutStage){.layout = NriLayout_UNDEFINED, .stages = NriStageBits_NONE};

        SWAPCHAIN_TEXTURE *frame = &r->gpu->swapchain_frames[i];
        frame->texture = textures[i];

        const NriTextureViewDesc view = {
            .texture = textures[i], .type = NriTextureView_COLOR_ATTACHMENT, .format = r->gpu->swapchain_format, .mipNum = 1, .layerNum = 1, .sliceNum = 1};

        if (r->gpu->core.CreateTextureView(&view, &frame->color_attachment) != NriResult_SUCCESS ||
            r->gpu->core.CreateFence(r->gpu->device, NRI_SWAPCHAIN_SEMAPHORE, &frame->acquire) != NriResult_SUCCESS ||
            r->gpu->core.CreateFence(r->gpu->device, NRI_SWAPCHAIN_SEMAPHORE, &frame->release) != NriResult_SUCCESS)
            return false;
    }

    return true;
}

static void destroy_swapchain(RENDERER *r) {
    for (uint32_t i = 0; i < r->gpu->swapchain_texture_count; ++i) {
        for (uint32_t j = 0; j < r->gpu->texture_state_num; ++j)
            if (r->gpu->texture_states[j].texture == r->gpu->swapchain_textures[i]) {
                r->gpu->texture_states[j] = r->gpu->texture_states[--r->gpu->texture_state_num];

                break;
            }

        SWAPCHAIN_TEXTURE *frame = &r->gpu->swapchain_frames[i];

        if (frame->color_attachment) r->gpu->core.DestroyDescriptor(frame->color_attachment);

        if (frame->acquire) r->gpu->core.DestroyFence(frame->acquire);

        if (frame->release) r->gpu->core.DestroyFence(frame->release);
    }

    free(r->gpu->swapchain_frames);
    free(r->gpu->swapchain_textures);
    r->gpu->swapchain_frames = NULL;
    r->gpu->swapchain_textures = NULL;
    r->gpu->swapchain_texture_count = 0;

    if (r->gpu->swapchain) r->gpu->swapchain_api.DestroySwapChain(r->gpu->swapchain);
    r->gpu->swapchain = NULL;
}

static bool acquire_swapchain_texture(RENDERER *r, uint32_t *index) {
    NriFence *acquire = r->gpu->swapchain_frames[r->gpu->frame_index % r->gpu->swapchain_texture_count].acquire;

    NriResult result = r->gpu->swapchain_api.AcquireNextTexture(r->gpu->swapchain, acquire, index);

    return result == NriResult_SUCCESS && *index < r->gpu->swapchain_texture_count;
}

static TEXTURE_STATE *find_texture_state(RENDERER *r, NriTexture *texture) {
    for (uint32_t i = 0; i < r->gpu->texture_state_num; ++i)
        if (r->gpu->texture_states[i].texture == texture) return &r->gpu->texture_states[i];

    if (r->gpu->texture_state_num == r->gpu->texture_state_cap) {
        uint32_t cap = r->gpu->texture_state_cap ? r->gpu->texture_state_cap * 2 : 32;
        TEXTURE_STATE *items = realloc(r->gpu->texture_states, cap * sizeof(*items));

        if (!items) return NULL;
        r->gpu->texture_states = items;
        r->gpu->texture_state_cap = cap;
    }

    TEXTURE_STATE *item = &r->gpu->texture_states[r->gpu->texture_state_num++];
    *item = (TEXTURE_STATE){.texture = texture};

    return item;
}

static bool texture_barrier(RENDERER *r, NriCommandBuffer *cmd, NriTexture *texture, NriAccessLayoutStage before, NriAccessLayoutStage after) {
    TEXTURE_STATE *item = find_texture_state(r, texture);

    if (!item) return false;

    if (item->state.layout) before = item->state;

    const NriTextureBarrierDesc barrier = {.texture = texture, .before = before, .after = after, .mipNum = 1, .layerNum = 1};

    r->gpu->core.CmdBarrier(cmd, &(NriBarrierDesc){.textures = &barrier, .textureNum = 1});
    item->state = after;

    return true;
}

bool gpu_transition_texture(RENDERER *r, NriCommandBuffer *cmd, NriTexture *texture, NriAccessBits access, NriLayout layout, NriStageBits stages) {
    if (!texture) return false;

    return texture_barrier(r, cmd, texture, (NriAccessLayoutStage){0}, (NriAccessLayoutStage){.access = access, .layout = layout, .stages = stages});
}

static bool begin_scene_rendering(RENDERER *r, NriCommandBuffer *cmd, NriTexture *hdr, NriTexture *normal, NriTexture *depth, uint32_t width, uint32_t height) {
    NriDescriptor *hdr_view = gpu_create_texture_view(r, hdr, NriTextureView_COLOR_ATTACHMENT);

    NriDescriptor *normal_view = gpu_create_texture_view(r, normal, NriTextureView_COLOR_ATTACHMENT);

    NriDescriptor *depth_view = gpu_create_texture_view(r, depth, NriTextureView_DEPTH_STENCIL_ATTACHMENT);

    if (!hdr_view || !normal_view || !depth_view) return false;

    const NriAccessLayoutStage color = {NriAccessBits_COLOR_ATTACHMENT, NriLayout_COLOR_ATTACHMENT, NriStageBits_COLOR_ATTACHMENT};

    const NriAccessLayoutStage depth_state = {NriAccessBits_DEPTH_STENCIL_ATTACHMENT, NriLayout_DEPTH_STENCIL_ATTACHMENT,
                                              NriStageBits_DEPTH_STENCIL_ATTACHMENT};

    if (!texture_barrier(r, cmd, hdr, (NriAccessLayoutStage){0}, color) || !texture_barrier(r, cmd, normal, (NriAccessLayoutStage){0}, color) ||
        !texture_barrier(r, cmd, depth, (NriAccessLayoutStage){0}, depth_state))
        return false;

    const NriAttachmentDesc colors[2] = {{.descriptor = hdr_view, .loadOp = NriLoadOp_CLEAR, .storeOp = NriStoreOp_STORE},
                                         {.descriptor = normal_view, .loadOp = NriLoadOp_CLEAR, .storeOp = NriStoreOp_STORE}};

    const NriRenderingDesc desc = {
        .colors = colors,
        .colorNum = 2,
        .depth = {.descriptor = depth_view, .loadOp = NriLoadOp_CLEAR, .storeOp = NriStoreOp_STORE, .clearValue = {.depthStencil = {.depth = 1.0f}}}};

    r->gpu->core.CmdSetViewports(cmd, &(NriViewport){.width = (float)width, .height = (float)height, .depthMax = 1.0f}, 1);
    r->gpu->core.CmdSetScissors(cmd, &(NriRect){.width = (NriDim_t)width, .height = (NriDim_t)height}, 1);
    r->gpu->core.CmdBeginRendering(cmd, &desc);

    return true;
}

static bool begin_compose_rendering(RENDERER *r, NriCommandBuffer *cmd, NriTexture *swap, NriTexture *hdr, NriTexture *ao, NriTexture *bloom, NriTexture *lut,
                                    NriDescriptor *sampler_desc, const void *uniforms, size_t size) {
    NriTexture *sources[] = {hdr, ao, bloom, lut};

    for (uint32_t i = 0; i < 4; ++i)
        if (!gpu_transition_texture(r, cmd, sources[i], NriAccessBits_SHADER_RESOURCE, NriLayout_SHADER_RESOURCE, NriStageBits_FRAGMENT_SHADER)) return false;

    NriDescriptor *src[] = {gpu_create_texture_view(r, hdr, NriTextureView_TEXTURE),
                            gpu_create_texture_view(r, ao, NriTextureView_TEXTURE),
                            gpu_create_texture_view(r, bloom, NriTextureView_TEXTURE),
                            gpu_create_texture_view(r, lut, NriTextureView_TEXTURE),
                            sampler_desc,
                            sampler_desc,
                            sampler_desc,
                            sampler_desc};

    if (!gpu_bind_descriptor_set(r, cmd, r->compose_layout, NriBindPoint_GRAPHICS, 2, src, 8) ||
        !gpu_bind_uniform_data(r, cmd, r->compose_layout, NriBindPoint_GRAPHICS, 3, uniforms, size))
        return false;

    if (!texture_barrier(
            r, cmd, swap, (NriAccessLayoutStage){.layout = NriLayout_UNDEFINED, .stages = NriStageBits_NONE},
            (NriAccessLayoutStage){.access = NriAccessBits_COLOR_ATTACHMENT, .layout = NriLayout_COLOR_ATTACHMENT, .stages = NriStageBits_COLOR_ATTACHMENT}))
        return false;

    const NriAttachmentDesc color = {
        .descriptor = r->gpu->swapchain_frames[r->gpu->current_swap_index].color_attachment, .loadOp = NriLoadOp_CLEAR, .storeOp = NriStoreOp_STORE};

    const NriRenderingDesc desc = {.colors = &color, .colorNum = 1};

    r->gpu->core.CmdBeginRendering(cmd, &desc);

    return true;
}

static bool submit_frame(RENDERER *r, FRAME_CONTEXT *frame, NriCommandBuffer *cmd, uint32_t index) {
    bool good =
        frame && texture_barrier(r, cmd, r->gpu->swapchain_textures[index],
                                 (NriAccessLayoutStage){
                                     .access = NriAccessBits_COLOR_ATTACHMENT, .layout = NriLayout_COLOR_ATTACHMENT, .stages = NriStageBits_COLOR_ATTACHMENT},
                                 (NriAccessLayoutStage){.layout = NriLayout_PRESENT, .stages = NriStageBits_NONE});
    if (good) good = r->gpu->core.EndCommandBuffer(cmd) == NriResult_SUCCESS;

    const uint64_t frame_value = 1u + r->gpu->frame_index;
    NriFenceSubmitDesc waits[2] = {
        {.fence = r->gpu->swapchain_frames[r->gpu->frame_index % r->gpu->swapchain_texture_count].acquire, .stages = NriStageBits_COLOR_ATTACHMENT}, {0}};
    uint32_t wait_num = 1u;

    if (r->gpu->upload && r->gpu->upload->fence && r->gpu->upload->next_fence_value > 1u) {
        waits[wait_num++] = (NriFenceSubmitDesc){.fence = r->gpu->upload->fence, .value = r->gpu->upload->next_fence_value - 1u, .stages = NriStageBits_ALL};
    }

    const NriFenceSubmitDesc signals[2] = {{.fence = r->gpu->swapchain_frames[index].release}, {.fence = r->gpu->frame_fence, .value = frame_value}};

    const NriQueueSubmitDesc submit = {.waitFences = waits,
                                       .waitFenceNum = wait_num,
                                       .commandBuffers = (const NriCommandBuffer *const *)&cmd,
                                       .commandBufferNum = 1,
                                       .signalFences = signals,
                                       .signalFenceNum = 2};

    bool submitted = false;

    if (good) {
        submitted = r->gpu->core.QueueSubmit(r->gpu->graphics_queue, &submit) == NriResult_SUCCESS;

        good = submitted;
    }

    if (good) good = r->gpu->swapchain_api.QueuePresent(r->gpu->swapchain, r->gpu->swapchain_frames[index].release, frame_value) == NriResult_SUCCESS;

    r->gpu->active_frame = NULL;

    if (submitted)
        r->gpu->frame_index++;
    else if (!good)
        abort_frame_commands(r, frame);

    return good;
}

NriDescriptor *gpu_create_sampler(RENDERER *r, NriFilter min_filter, NriFilter mag_filter, NriAddressMode address) {
    const NriSamplerDesc desc = {.filters = {.min = min_filter, .mag = mag_filter, .mip = NriFilter_NEAREST},
                                 .addressModes = {address, address, address},
                                 .mipMin = 0.0f,
                                 .mipMax = 16.0f};

    NriDescriptor *result = NULL;

    if (r->gpu->core.CreateSampler(r->gpu->device, &desc, &result) != NriResult_SUCCESS) return NULL;

    return result;
}

static NriPipeline *make_line_pipeline(RENDERER *r, NriPipelineLayout *layout, const NriShaderDesc *vs, const NriShaderDesc *ps) {
    const NriVertexStreamDesc vb = {.bindingSlot = 0, .stepRate = NriVertexStreamStepRate_PER_VERTEX, .stride = (uint16_t)sizeof(RENDER_VERTEX)};

    const NriVertexAttributeDesc attrs[2] = {{.d3d = {.semanticName = "TEXCOORD", .semanticIndex = 0},
                                              .vk = {.location = 0},
                                              .offset = (uint32_t)offsetof(RENDER_VERTEX, x),
                                              .format = NriFormat_RGB32_SFLOAT,
                                              .streamIndex = 0},
                                             {.d3d = {.semanticName = "TEXCOORD", .semanticIndex = 1},
                                              .vk = {.location = 1},
                                              .offset = (uint32_t)offsetof(RENDER_VERTEX, r),
                                              .format = NriFormat_RGBA32_SFLOAT,
                                              .streamIndex = 0}};

    const NriVertexInputDesc vertex_input = {.attributes = attrs, .attributeNum = 2, .streams = &vb, .streamNum = 1};

    const NriColorAttachmentDesc targets[2] = {{.format = NriFormat_RGBA16_SFLOAT, .colorWriteMask = NriColorWriteBits_RGBA},
                                               {.format = NriFormat_RGBA16_SFLOAT, .colorWriteMask = NriColorWriteBits_NONE}};

    const NriMultisampleDesc multisample = {.sampleMask = NRI_ALL, .sampleNum = 1};

    const NriShaderDesc shaders[2] = {*vs, *ps};

    const NriGraphicsPipelineDesc desc = {
        .pipelineLayout = layout,
        .vertexInput = &vertex_input,
        .inputAssembly = {.topology = NriTopology_LINE_LIST},
        .rasterization = {.fillMode = NriFillMode_SOLID, .cullMode = NriCullMode_NONE, .frontCounterClockwise = true, .depthClamp = false},
        .multisample = &multisample,
        .outputMerger = {.colors = targets,
                         .colorNum = 2,
                         .depth = {.compareOp = NriCompareOp_LESS_EQUAL, .write = false},
                         .depthStencilFormat = r->depth_format},
        .shaders = shaders,
        .shaderNum = 2,
        .cache = r->gpu->pipeline_cache};

    NriPipeline *pipeline = NULL;

    if (r->gpu->core.CreateGraphicsPipeline(r->gpu->device, &desc, &pipeline) != NriResult_SUCCESS) return NULL;

    return pipeline;
}

static NriPipeline *make_sky_pipeline(RENDERER *r, NriPipelineLayout *layout, const NriShaderDesc *vs, const NriShaderDesc *ps) {
    const NriColorAttachmentDesc targets[2] = {{.format = NriFormat_RGBA16_SFLOAT, .colorWriteMask = NriColorWriteBits_RGBA},
                                               {.format = NriFormat_RGBA16_SFLOAT, .colorWriteMask = NriColorWriteBits_RGBA}};

    const NriMultisampleDesc multisample = {.sampleMask = NRI_ALL, .sampleNum = 1};

    const NriShaderDesc shaders[2] = {*vs, *ps};
    const NriGraphicsPipelineDesc desc = {
        .pipelineLayout = layout,
        .inputAssembly = {.topology = NriTopology_TRIANGLE_LIST},
        .rasterization = {.fillMode = NriFillMode_SOLID, .cullMode = NriCullMode_NONE, .frontCounterClockwise = true, .depthClamp = false},
        .multisample = &multisample,
        .outputMerger = {.colors = targets, .colorNum = 2, .depthStencilFormat = r->depth_format},
        .shaders = shaders,
        .shaderNum = 2,
        .cache = r->gpu->pipeline_cache};

    NriPipeline *pipeline = NULL;

    if (r->gpu->core.CreateGraphicsPipeline(r->gpu->device, &desc, &pipeline) != NriResult_SUCCESS) return NULL;

    return pipeline;
}

NriTexture *gpu_create_texture(RENDERER *r, NriFormat format, NriTextureUsageBits usage, Uint32 width, Uint32 height) {
    if (!r || !r->gpu->device || !width || !height) return NULL;

    const NriTextureDesc desc = {.type = NriTextureType_TEXTURE_2D,
                                 .usage = usage,
                                 .format = format,
                                 .width = (NriDim_t)width,
                                 .height = (NriDim_t)height,
                                 .depth = 1,
                                 .mipNum = 1,
                                 .layerNum = 1,
                                 .sampleNum = 1};

    NriTexture *result = NULL;

    if (r->gpu->core.CreateCommittedTexture(r->gpu->device, NriMemoryLocation_DEVICE, 1.0f, &desc, &result) != NriResult_SUCCESS) return NULL;

    if (!find_texture_state(r, result)) {
        release_texture(r, result);

        return NULL;
    }

    return result;
}

static uint64_t upload_align(uint64_t value, uint64_t alignment) {
    if (!alignment) return value;

    return (value + alignment - 1u) / alignment * alignment;
}

static void destroy_upload_context(RENDERER *r) {
    if (!r || !r->gpu->upload) return;

    UPLOAD_CONTEXT *upload = r->gpu->upload;

    for (uint32_t i = 0; i < UPLOAD_RING_SIZE; ++i) {
        UPLOAD_SLOT *slot = &upload->slots[i];

        if (slot->command_buffer) r->gpu->core.DestroyCommandBuffer(slot->command_buffer);

        if (slot->allocator) r->gpu->core.DestroyCommandAllocator(slot->allocator);

        if (slot->staging) r->gpu->core.DestroyBuffer(slot->staging);
    }

    if (upload->fence) r->gpu->core.DestroyFence(upload->fence);

    free(upload);
    r->gpu->upload = NULL;
}

static bool create_upload_context(RENDERER *r) {
    if (!r || !r->gpu->device || !r->gpu->graphics_queue) return false;

    if (r->gpu->upload) return true;

    UPLOAD_CONTEXT *upload = calloc(1, sizeof(*upload));

    if (!upload) return false;
    r->gpu->upload = upload;

    if (r->gpu->core.CreateFence(r->gpu->device, 0u, &upload->fence) != NriResult_SUCCESS) goto fail;

    const NriBufferDesc staging_desc = {.size = UPLOAD_CHUNK_BYTES, .usage = NriBufferUsageBits_NONE};

    for (uint32_t i = 0; i < UPLOAD_RING_SIZE; ++i) {
        UPLOAD_SLOT *slot = &upload->slots[i];

        if (r->gpu->core.CreateCommittedBuffer(r->gpu->device, NriMemoryLocation_HOST_UPLOAD, 0.0f, &staging_desc, &slot->staging) != NriResult_SUCCESS ||
            r->gpu->core.CreateCommandAllocator(r->gpu->graphics_queue, &slot->allocator) != NriResult_SUCCESS ||
            r->gpu->core.CreateCommandBuffer(slot->allocator, &slot->command_buffer) != NriResult_SUCCESS)
            goto fail;
    }

    upload->next_fence_value = 1u;

    return true;

fail:
    destroy_upload_context(r);

    return false;
}

static bool upload_wait_slot(RENDERER *r, UPLOAD_SLOT *slot) {
    if (!r || !r->gpu->upload || !slot || !slot->fence_value) return true;

    const Uint64 started = SDL_GetTicks();
    bool logged = false;

    while (r->gpu->core.GetFenceValue(r->gpu->upload->fence) < slot->fence_value) {
        if (!logged && SDL_GetTicks() - started >= UPLOAD_SLOW_LOG_MS) {
            SDL_Log("GPU upload chunk is still pending after %u ms; continuing "
                    "without NRI's hard fence timeout",
                    UPLOAD_SLOW_LOG_MS);
            logged = true;
        }

        SDL_Delay(1u);
    }

    slot->fence_value = 0u;

    return true;
}

static bool upload_begin_slot(RENDERER *r, UPLOAD_SLOT **out) {
    if (!out || !create_upload_context(r)) return false;

    UPLOAD_CONTEXT *upload = r->gpu->upload;
    UPLOAD_SLOT *slot = &upload->slots[upload->next_slot];
    upload->next_slot = (upload->next_slot + 1u) % UPLOAD_RING_SIZE;

    if (!upload_wait_slot(r, slot)) return false;
    r->gpu->core.ResetCommandAllocator(slot->allocator);

    if (r->gpu->core.BeginCommandBuffer(slot->command_buffer, NULL) != NriResult_SUCCESS) return false;

    *out = slot;
    return true;
}

static bool upload_submit_slot(RENDERER *r, UPLOAD_SLOT *slot) {
    if (!r || !r->gpu->upload || !slot) return false;

    if (r->gpu->core.EndCommandBuffer(slot->command_buffer) != NriResult_SUCCESS) return false;

    const uint64_t value = r->gpu->upload->next_fence_value++;

    const NriFenceSubmitDesc signal = {.fence = r->gpu->upload->fence, .value = value};

    NriCommandBuffer *command = slot->command_buffer;

    const NriQueueSubmitDesc submit = {
        .commandBuffers = (const NriCommandBuffer *const *)&command, .commandBufferNum = 1u, .signalFences = &signal, .signalFenceNum = 1u};

    if (r->gpu->core.QueueSubmit(r->gpu->graphics_queue, &submit) != NriResult_SUCCESS) return false;

    slot->fence_value = value;

    return true;
}

static bool upload_drain(RENDERER *r) {
    if (!r || !r->gpu->upload) return true;

    for (uint32_t i = 0; i < UPLOAD_RING_SIZE; ++i)
        if (!upload_wait_slot(r, &r->gpu->upload->slots[i])) return false;

    return true;
}

static NriAccessStage uploaded_buffer_state(NriBufferUsageBits usage) {
    NriAccessStage state = {0};

    if (usage & NriBufferUsageBits_SHADER_RESOURCE) {
        state.access |= NriAccessBits_SHADER_RESOURCE;
        state.stages |= NriStageBits_COMPUTE_SHADER | NriStageBits_FRAGMENT_SHADER;
    }

    if (usage & NriBufferUsageBits_VERTEX) {
        state.access |= NriAccessBits_VERTEX_BUFFER;
        state.stages |= NriStageBits_VERTEX_SHADER;
    }

    return state;
}

NriBuffer *gpu_upload_buffer(RENDERER *r, NriBufferUsageBits usage, const void *data, size_t bytes, uint32_t stride) {
    if (!r || !r->gpu->device || !r->gpu->graphics_queue || !data || !bytes) return NULL;

    const NriBufferDesc desc = {.size = bytes, .structureStride = stride, .usage = usage};

    NriBuffer *buffer = NULL;

    if (r->gpu->core.CreateCommittedBuffer(r->gpu->device, NriMemoryLocation_DEVICE, 1.0f, &desc, &buffer) != NriResult_SUCCESS) return NULL;

    const NriAccessStage copy_state = {.access = NriAccessBits_COPY_DESTINATION, .stages = NriStageBits_ALL};

    const NriAccessStage final_state = uploaded_buffer_state(usage);
    const Uint8 *source = data;

    size_t offset = 0u;

    while (offset < bytes) {
        const size_t chunk = bytes - offset > UPLOAD_CHUNK_BYTES ? UPLOAD_CHUNK_BYTES : bytes - offset;

        UPLOAD_SLOT *slot = NULL;

        if (!upload_begin_slot(r, &slot)) goto fail;

        void *mapped = r->gpu->core.MapBuffer(slot->staging, 0u, chunk);

        if (!mapped) {
            (void)r->gpu->core.EndCommandBuffer(slot->command_buffer);

            goto fail;
        }

        memcpy(mapped, source + offset, chunk);
        r->gpu->core.UnmapBuffer(slot->staging);

        if (!offset) {
            const NriBufferBarrierDesc barrier = {.buffer = buffer, .before = {0}, .after = copy_state};

            r->gpu->core.CmdBarrier(slot->command_buffer, &(NriBarrierDesc){.buffers = &barrier, .bufferNum = 1u});
        }

        r->gpu->core.CmdCopyBuffer(slot->command_buffer, buffer, offset, slot->staging, 0u, chunk);

        if (offset + chunk == bytes) {
            const NriBufferBarrierDesc barrier = {.buffer = buffer, .before = copy_state, .after = final_state};

            r->gpu->core.CmdBarrier(slot->command_buffer, &(NriBarrierDesc){.buffers = &barrier, .bufferNum = 1u});
        }

        if (!upload_submit_slot(r, slot)) goto fail;
        offset += chunk;
    }

    return buffer;

fail:
    (void)upload_drain(r);
    r->gpu->core.DestroyBuffer(buffer);

    return NULL;
}

bool gpu_upload_texture_data(RENDERER *r, NriTexture *texture, const void *data, uint32_t row_pitch, uint32_t slice_pitch, NriAccessBits access,
                             NriLayout layout, NriStageBits stages) {
    if (!r || !texture || !data || !row_pitch || !slice_pitch || slice_pitch % row_pitch) return false;

    const NriTextureDesc *desc = r->gpu->core.GetTextureDesc(texture);
    const NriDeviceDesc *device = r->gpu->core.GetDeviceDesc(r->gpu->device);
    const uint32_t row_alignment = device->memoryAlignment.uploadBufferTextureRow;
    const uint32_t slice_alignment = device->memoryAlignment.uploadBufferTextureSlice;

    const uint64_t aligned_row = upload_align(row_pitch, row_alignment);
    const uint32_t row_count = slice_pitch / row_pitch;

    if (!desc || !row_count || aligned_row > UPLOAD_CHUNK_BYTES) return false;

    const NriAccessLayoutStage copy_state = {.access = NriAccessBits_COPY_DESTINATION, .layout = NriLayout_COPY_DESTINATION, .stages = NriStageBits_ALL};

    const NriAccessLayoutStage final_state = {.access = access, .layout = layout, .stages = stages};

    const Uint8 *source = data;

    uint32_t first_row = 0u;

    while (first_row < row_count) {
        uint32_t rows = (uint32_t)(UPLOAD_CHUNK_BYTES / aligned_row);
        const uint32_t remaining = row_count - first_row;

        if (!rows) rows = 1u;

        if (rows > remaining) rows = remaining;

        uint64_t staging_bytes = upload_align(aligned_row * rows, slice_alignment);

        while (rows > 1u && staging_bytes > UPLOAD_CHUNK_BYTES) {
            --rows;

            staging_bytes = upload_align(aligned_row * rows, slice_alignment);
        }

        if (staging_bytes > UPLOAD_CHUNK_BYTES) return false;

        UPLOAD_SLOT *slot = NULL;

        if (!upload_begin_slot(r, &slot)) goto fail;

        Uint8 *mapped = r->gpu->core.MapBuffer(slot->staging, 0u, staging_bytes);

        if (!mapped) {
            (void)r->gpu->core.EndCommandBuffer(slot->command_buffer);

            goto fail;
        }

        for (uint32_t row = 0; row < rows; ++row) memcpy(mapped + (size_t)row * aligned_row, source + (size_t)(first_row + row) * row_pitch, row_pitch);
        r->gpu->core.UnmapBuffer(slot->staging);

        if (!first_row && !texture_barrier(r, slot->command_buffer, texture, (NriAccessLayoutStage){0}, copy_state)) goto fail;

        const NriTextureDataLayoutDesc source_layout = {.offset = 0u, .rowPitch = (uint32_t)aligned_row, .slicePitch = (uint32_t)staging_bytes};

        const NriTextureRegionDesc region = {
            .x = 0u, .y = (NriDim_t)first_row, .z = 0u, .width = desc->width, .height = (NriDim_t)rows, .depth = 1u, .mipOffset = 0u, .layerOffset = 0u};

        r->gpu->core.CmdUploadBufferToTexture(slot->command_buffer, texture, &region, slot->staging, &source_layout);

        if (first_row + rows == row_count && !texture_barrier(r, slot->command_buffer, texture, copy_state, final_state)) goto fail;

        if (!upload_submit_slot(r, slot)) goto fail;
        first_row += rows;
    }

    return true;

fail:
    (void)upload_drain(r);

    return false;
}

static NriTexture *pixel_texture(RENDERER *r, Uint8 red, Uint8 green, Uint8 blue, Uint8 alpha) {
    const Uint8 pixels[4] = {red, green, blue, alpha};

    NriTexture *result = gpu_create_texture(r, NriFormat_RGBA8_UNORM, NriTextureUsageBits_SHADER_RESOURCE, 1, 1);

    if (!result) return NULL;

    if (!gpu_upload_texture_data(r, result, pixels, 4, 4, NriAccessBits_SHADER_RESOURCE, NriLayout_SHADER_RESOURCE, NriStageBits_ALL)) {
        release_texture(r, result);

        return NULL;
    }

    return result;
}

void release_texture(RENDERER *r, NriTexture *value) {
    if (!r || !value) return;

    for (uint32_t i = 0; i < r->gpu->texture_state_num; ++i) {
        if (r->gpu->texture_states[i].texture == value) {
            r->gpu->texture_states[i] = r->gpu->texture_states[--r->gpu->texture_state_num];

            break;
        }
    }

    r->gpu->core.DestroyTexture(value);
}

void release_buffer(RENDERER *r, NriBuffer *value) {
    if (r && value) r->gpu->core.DestroyBuffer(value);
}

static bool ensure_depth_texture(RENDERER *r, Uint32 width, Uint32 height) {
    if (r->depth_texture && r->depth_width == width && r->depth_height == height) return true;

    release_texture(r, r->depth_texture);
    r->depth_texture = gpu_create_texture(r, r->depth_format, NriTextureUsageBits_DEPTH_STENCIL_ATTACHMENT, width, height);

    if (!r->depth_texture) return false;

    r->depth_width = width;
    r->depth_height = height;

    return true;
}

static const char *image_type(const char *mime) {
    if (!mime || !*mime) return NULL;

    if (strstr(mime, "png")) return "PNG";

    if (strstr(mime, "jpeg") || strstr(mime, "jpg")) return "JPG";

    if (strstr(mime, "webp")) return "WEBP";

    if (strstr(mime, "avif")) return "AVIF";

    return NULL;
}

static NriTexture *load_image(RENDERER *r, const GLTF_IMAGE *image) {
    if (!image || !image->bytes.data || !image->bytes.size) return NULL;

    SDL_IOStream *io = SDL_IOFromConstMem(image->bytes.data, image->bytes.size);

    if (!io) return NULL;

    SDL_Surface *decoded = IMG_LoadTyped_IO(io, true, image_type(image->mime));

    if (!decoded) return NULL;

    SDL_Surface *rgba = SDL_ConvertSurface(decoded, SDL_PIXELFORMAT_RGBA32);

    SDL_DestroySurface(decoded);

    if (!rgba) return NULL;

    NriTexture *result = gpu_create_texture(r, NriFormat_RGBA8_UNORM, NriTextureUsageBits_SHADER_RESOURCE, (Uint32)rgba->w, (Uint32)rgba->h);

    if (result && !gpu_upload_texture_data(r, result, rgba->pixels, (uint32_t)rgba->pitch, (uint32_t)(rgba->pitch * rgba->h), NriAccessBits_SHADER_RESOURCE,
                                           NriLayout_SHADER_RESOURCE, NriStageBits_FRAGMENT_SHADER)) {
        release_texture(r, result);

        result = NULL;
    }

    SDL_DestroySurface(rgba);

    return result;
}

static bool load_images(RENDERER *r, const GLTF_SCENE *visual) {
    r->image_texture_count = visual->image_count;

    if (!visual->image_count) return true;

    r->image_textures = calloc(visual->image_count, sizeof(*r->image_textures));

    if (!r->image_textures) return false;

    for (uint32_t i = 0; i < visual->image_count; ++i) {
        const GLTF_IMAGE *image = &visual->images[i];

        if (!image->bytes.data || !image->bytes.size) continue;

        r->image_textures[i] = load_image(r, image);

        if (!r->image_textures[i]) {
            SDL_Log("SDL_image could not decode GLB image %u (%s): %s", i, image->mime[0] ? image->mime : "unknown", SDL_GetError());
        }
    }

    return true;
}

static NriTexture *resolve_texture(RENDERER *r, const GLTF_SCENE *visual, int32_t texture_index, NriTexture *fallback) {
    if (texture_index < 0 || (uint32_t)texture_index >= visual->texture_count) return fallback;

    const int32_t image = visual->textures[texture_index].image;

    if (image < 0 || (uint32_t)image >= r->image_texture_count || !r->image_textures[image]) return fallback;

    return r->image_textures[image];
}

static void release_scene_resources(RENDERER *r) {
    if (!r || !r->gpu->device) return;

    if (r->image_textures) {
        for (uint32_t i = 0; i < r->image_texture_count; ++i) release_texture(r, r->image_textures[i]);
    }

    free(r->image_textures);
    r->image_textures = NULL;
    r->image_texture_count = 0;

    free(r->materials);
    r->materials = NULL;
    r->material_count = 0;

    release_texture(r, r->default_white);
    release_texture(r, r->default_normal);

    if (r->material_sampler) r->gpu->core.DestroyDescriptor(r->material_sampler);
    release_buffer(r, r->vertex_buffer);
    release_texture(r, r->lightmap_texture);

    if (r->lightmap_sampler) r->gpu->core.DestroyDescriptor(r->lightmap_sampler);

    r->default_white = NULL;
    r->default_normal = NULL;
    r->material_sampler = NULL;
    r->vertex_buffer = NULL;
    r->lightmap_texture = NULL;
    r->lightmap_sampler = NULL;
}

bool upload_scene(RENDERER *r, const GLTF_SCENE *visual) {
    if (!r || !r->gpu->device || !visual || !r->vertices || !r->vertex_count) return false;

    release_scene_resources(r);

    r->default_white = pixel_texture(r, 255, 255, 255, 255);
    r->default_normal = pixel_texture(r, 128, 128, 255, 255);
    r->material_sampler = gpu_create_sampler(r, NriFilter_LINEAR, NriFilter_LINEAR, NriAddressMode_REPEAT);

    if (!r->default_white || !r->default_normal || !r->material_sampler) return false;

    const uint64_t vertex_bytes = (uint64_t)r->vertex_count * sizeof(*r->vertices);

    r->vertex_buffer = gpu_upload_buffer(r, NriBufferUsageBits_VERTEX, r->vertices, (size_t)vertex_bytes, 0);

    if (!r->vertex_buffer) return false;

    free(r->vertices);
    r->vertices = NULL;
    r->vertex_capacity = 0u;

    if (!load_images(r, visual)) return false;

    r->material_count = visual->material_count;
    r->materials = calloc(r->material_count, sizeof(*r->materials));

    if (!r->materials) return false;

    for (uint32_t i = 0; i < r->material_count; ++i) {
        RENDER_MATERIAL *m = &r->materials[i];
        m->data = visual->materials[i];
        m->base_color = resolve_texture(r, visual, m->data.base_color_texture, r->default_white);
        m->metallic_roughness = resolve_texture(r, visual, m->data.metallic_roughness_texture, r->default_white);
        m->normal = resolve_texture(r, visual, m->data.normal_texture, r->default_normal);
        m->occlusion = resolve_texture(r, visual, m->data.occlusion_texture, r->default_white);
        m->emissive = resolve_texture(r, visual, m->data.emissive_texture, r->default_white);
    }

    r->lightmap_sampler = gpu_create_sampler(r, NriFilter_LINEAR, NriFilter_LINEAR, NriAddressMode_CLAMP_TO_EDGE);
    r->lightmap_texture = pixel_texture(r, 0, 0, 0, 255);

    return r->lightmap_sampler && r->lightmap_texture;
}

static bool dispatch_one(FX_STATE *fx, NriCommandBuffer *cmd, NriPipeline *pipeline, NriTexture *source, NriTexture *destination, const void *uniforms,
                         Uint32 uniform_size, Uint32 width, Uint32 height) {
    if (!fx || !fx->owner || !cmd || !pipeline || !destination) return false;

    RENDERER *r = fx->owner;

    if (!bind_fx_resources(r, cmd, pipeline, source, destination, fx->sampler, uniforms, uniform_size)) return false;

    r->gpu->core.CmdSetPipeline(cmd, pipeline);
    r->gpu->core.CmdDispatch(cmd, &(NriDispatchDesc){.workGroupNumX = (width + 7u) / 8u, .workGroupNumY = (height + 7u) / 8u, .workGroupNumZ = 1});

    return true;
}

static void release_frame_textures(FX_STATE *fx) {
    if (!fx || !fx->owner) return;

    RENDERER *r = fx->owner;

    release_texture(r, fx->hdr);
    release_texture(r, fx->normal_depth);
    release_texture(r, fx->ao);
    release_texture(r, fx->bloom_a);
    release_texture(r, fx->bloom_b);
    release_texture(r, fx->volume);
    release_texture(r, fx->lit);

    fx->hdr = NULL;
    fx->normal_depth = NULL;
    fx->ao = NULL;
    fx->bloom_a = NULL;
    fx->bloom_b = NULL;
    fx->volume = NULL;
    fx->lit = NULL;
    fx->width = fx->height = fx->ao_width = fx->ao_height = 0;
    fx->volume_ready = false;
}

static void fx_deinit(FX_STATE *fx) {
    if (!fx || !fx->owner) return;

    RENDERER *r = fx->owner;

    release_frame_textures(fx);

    release_texture(r, fx->lut);

    if (fx->sampler) r->gpu->core.DestroyDescriptor(fx->sampler);

    if (fx->depth_sampler) r->gpu->core.DestroyDescriptor(fx->depth_sampler);

    if (fx->compose_pipeline) r->gpu->core.DestroyPipeline(fx->compose_pipeline);

    if (fx->ssao_pipeline) r->gpu->core.DestroyPipeline(fx->ssao_pipeline);

    if (fx->bloom_pipeline) r->gpu->core.DestroyPipeline(fx->bloom_pipeline);

    if (fx->grade_pipeline) r->gpu->core.DestroyPipeline(fx->grade_pipeline);

    if (fx->volume_pipeline) r->gpu->core.DestroyPipeline(fx->volume_pipeline);

    if (fx->volume_compose_pipeline) r->gpu->core.DestroyPipeline(fx->volume_compose_pipeline);

    memset(fx, 0, sizeof(*fx));
}

static bool make_compose_pipeline(RENDERER *r, NriShaderDesc *vs, NriShaderDesc *ps, NriFormat swap_format, NriPipeline **out) {
    const NriColorAttachmentDesc target = {.format = swap_format, .colorWriteMask = NriColorWriteBits_RGBA};

    const NriShaderDesc shaders[2] = {*vs, *ps};
    const NriGraphicsPipelineDesc desc = {
        .pipelineLayout = r->compose_layout,
        .inputAssembly = {.topology = NriTopology_TRIANGLE_LIST},
        .rasterization = {.fillMode = NriFillMode_SOLID, .cullMode = NriCullMode_NONE, .frontCounterClockwise = true, .depthClamp = false},
        .outputMerger = {.colors = &target, .colorNum = 1},
        .shaders = shaders,
        .shaderNum = 2,
        .cache = r->gpu->pipeline_cache};

    return r->gpu->core.CreateGraphicsPipeline(r->gpu->device, &desc, out) == NriResult_SUCCESS;
}

static bool fx_init(FX_STATE *fx, RENDERER *r) {
    if (!fx || !r || !r->gpu->device) return false;

    NriShaderDesc vs = {0};
    NriShaderDesc ps = {0};
    NriCommandAllocator *allocator = NULL;
    NriCommandBuffer *cmd = NULL;

    memset(fx, 0, sizeof(*fx));
    fx->owner = r;

    fx->sampler = gpu_create_sampler(r, NriFilter_LINEAR, NriFilter_LINEAR, NriAddressMode_CLAMP_TO_EDGE);

    fx->depth_sampler = gpu_create_sampler(r, NriFilter_NEAREST, NriFilter_NEAREST, NriAddressMode_CLAMP_TO_EDGE);

    if (!fx->sampler || !fx->depth_sampler) goto fail;

    vs = compile_shader("shaders/vertex.hlsl", "fullscreen_vs", "BUILD_FULLSCREEN_VS", NriStageBits_VERTEX_SHADER);

    ps = compile_shader("shaders/fragment.hlsl", "compose_fs", "BUILD_COMPOSE_FS", NriStageBits_FRAGMENT_SHADER);

    fx->ssao_pipeline = gpu_compile_compute(r, r->ssao_layout, "shaders/compute.hlsl", "ssao_cs", "BUILD_SSAO_CS");

    fx->bloom_pipeline = gpu_compile_compute(r, r->bloom_layout, "shaders/compute.hlsl", "bloom_cs", "BUILD_BLOOM_CS");

    fx->grade_pipeline = gpu_compile_compute(r, r->grade_layout, "shaders/compute.hlsl", "grade_cs", "BUILD_GRADE_CS");

    fx->volume_pipeline = gpu_compile_compute(r, r->volume_layout, "shaders/vision_compute.hlsl", "volume_cs", "BUILD_VISION_VOLUME_CS");

    fx->volume_compose_pipeline =
        gpu_compile_compute(r, r->volume_compose_layout, "shaders/vision_compute.hlsl", "volume_compose_cs", "BUILD_VISION_COMPOSE_CS");

    if (!vs.bytecode || !ps.bytecode || !fx->ssao_pipeline || !fx->bloom_pipeline || !fx->grade_pipeline || !fx->volume_pipeline ||
        !fx->volume_compose_pipeline)
        goto fail;

    if (!make_compose_pipeline(r, &vs, &ps, r->gpu->swapchain_format, &fx->compose_pipeline)) goto fail;

    free_shader(&vs);
    free_shader(&ps);

    fx->lut = gpu_create_texture(r, NriFormat_RGBA16_SFLOAT, NriTextureUsageBits_SHADER_RESOURCE | NriTextureUsageBits_SHADER_RESOURCE_STORAGE, 256u, 16u);

    if (!fx->lut) goto fail;

    if (gpu_begin_commands(r, &allocator, &cmd) != NriResult_SUCCESS || !dispatch_one(fx, cmd, fx->grade_pipeline, NULL, fx->lut, NULL, 0, 256u, 16u))
        goto fail;

    const bool submitted = gpu_submit_commands(r, allocator, cmd);
    cmd = NULL;
    allocator = NULL;

    if (!submitted) goto fail;

    return true;

fail:
    free_shader(&vs);
    free_shader(&ps);
    gpu_abort_commands(r, allocator, cmd);
    fx_deinit(fx);

    return false;
}

static bool fx_ensure(FX_STATE *fx, Uint32 width, Uint32 height) {
    if (!fx || !fx->owner || !width || !height) return false;

    if (fx->hdr && fx->width == width && fx->height == height) return true;

    RENDERER *r = fx->owner;

    release_frame_textures(fx);
    fx->width = width;
    fx->height = height;
    fx->ao_width = (width + 1u) / 2u;
    fx->ao_height = (height + 1u) / 2u;

    const NriTextureUsageBits rt = NriTextureUsageBits_COLOR_ATTACHMENT | NriTextureUsageBits_SHADER_RESOURCE;

    const NriTextureUsageBits compute = NriTextureUsageBits_SHADER_RESOURCE | NriTextureUsageBits_SHADER_RESOURCE_STORAGE;

    fx->hdr = gpu_create_texture(r, NriFormat_RGBA16_SFLOAT, rt, width, height);
    fx->normal_depth = gpu_create_texture(r, NriFormat_RGBA16_SFLOAT, rt, width, height);
    fx->ao = gpu_create_texture(r, NriFormat_RGBA16_SFLOAT, compute, fx->ao_width, fx->ao_height);
    fx->bloom_a = gpu_create_texture(r, NriFormat_RGBA16_SFLOAT, compute, fx->ao_width, fx->ao_height);
    fx->bloom_b = gpu_create_texture(r, NriFormat_RGBA16_SFLOAT, compute, fx->ao_width, fx->ao_height);
    fx->volume = gpu_create_texture(r, NriFormat_RGBA16_SFLOAT, compute, fx->ao_width, fx->ao_height);
    fx->lit = gpu_create_texture(r, NriFormat_RGBA16_SFLOAT, compute, width, height);

    if (!fx->hdr || !fx->normal_depth || !fx->ao || !fx->bloom_a || !fx->bloom_b || !fx->volume || !fx->lit) {
        release_frame_textures(fx);

        return false;
    }

    return true;
}

static bool fx_volume(FX_STATE *fx, NriCommandBuffer *cmd, NriBuffer *probes, NriBuffer *beams, const PROBE_GRID *grid, const BEAM_GRID *beam_grid,
                      const RENDER_FRAME *frame) {
    if (!fx || !fx->owner || !cmd || !probes || !beams || !grid || !beam_grid || !grid->probes || !fx->volume) return false;

    RENDERER *r = fx->owner;

    const VOLUME_UNIFORMS u = {
        .eye_density = {frame->eye.x, frame->eye.y, frame->eye.z, 0.0f},
        .right_tan = {frame->right.x * frame->tan_half_fov * frame->aspect, frame->right.y * frame->tan_half_fov * frame->aspect,
                      frame->right.z * frame->tan_half_fov * frame->aspect, 0},
        .up_tan = {frame->up.x * frame->tan_half_fov, frame->up.y * frame->tan_half_fov, frame->up.z * frame->tan_half_fov, 0},
        .forward_g = {frame->forward.x, frame->forward.y, frame->forward.z, 0.0f},
        .sun_intensity = {frame->sun.direction.x, frame->sun.direction.y, frame->sun.direction.z, frame->sun.intensity},
        .sun_color = {frame->sun.color.x, frame->sun.color.y, frame->sun.color.z, 1.0f},
        .grid_origin_spacing = {grid->origin.x, grid->origin.y, grid->origin.z, grid->spacing},
        .grid_dims_width = {grid->count_x, grid->count_y, grid->count_z, fx->ao_width},
        .height_debug = {fx->ao_height, fx->debug_view, beam_grid->depth, 0},
        .beam_origin = {beam_grid->origin.x, beam_grid->origin.y, beam_grid->origin.z, 0},
        .beam_step = {beam_grid->step.x, beam_grid->step.y, beam_grid->step.z, 0},
        .volume_params = {frame->volumetrics.density, frame->volumetrics.anisotropy, frame->volumetrics.probe_intensity, frame->volumetrics.max_distance},
        .volume_radii = {frame->vision.center_radius, frame->vision.middle_radius, frame->vision.center_transition_width,
                         frame->vision.middle_transition_width},
        .volume_filter = {frame->vision.jitter_strength, frame->vision.volume_blur_strength, 0.0f, 0.0f},
        .volume_quality = {frame->vision.center_steps, frame->vision.middle_steps, frame->vision.peripheral_steps, 0u},
        .volume_strides = {frame->vision.center_stride, frame->vision.middle_stride, frame->vision.peripheral_stride, 0u}};

    if (!bind_volume_resources(r, cmd, fx->normal_depth, fx->depth_sampler, probes, beams, fx->volume, &u, sizeof(u))) return false;

    r->gpu->core.CmdSetPipeline(cmd, fx->volume_pipeline);

    r->gpu->core.CmdDispatch(cmd,
                             &(NriDispatchDesc){.workGroupNumX = (fx->ao_width + 7u) / 8u, .workGroupNumY = (fx->ao_height + 7u) / 8u, .workGroupNumZ = 1});

    const VOLUME_COMPOSE_UNIFORMS compose = {
        .width = fx->width,
        .height = fx->height,
        .debug_view = fx->debug_view,
        .bypass_volume = 0u,
        .volume_radii = {frame->vision.center_radius, frame->vision.middle_radius, frame->vision.center_transition_width,
                         frame->vision.middle_transition_width},
        .volume_filter = {frame->vision.jitter_strength, frame->vision.volume_blur_strength, 0.0f, 0.0f},
        .volume_strides = {frame->vision.center_stride, frame->vision.middle_stride, frame->vision.peripheral_stride, 0u},
    };

    if (!bind_volume_compose_resources(r, cmd, fx->hdr, fx->volume, fx->normal_depth, fx->sampler, fx->depth_sampler, fx->lit, &compose, sizeof(compose)))
        return false;

    r->gpu->core.CmdSetPipeline(cmd, fx->volume_compose_pipeline);

    r->gpu->core.CmdDispatch(cmd, &(NriDispatchDesc){.workGroupNumX = (fx->width + 7u) / 8u, .workGroupNumY = (fx->height + 7u) / 8u, .workGroupNumZ = 1});

    fx->volume_ready = true;

    return true;
}

static bool run_vision_only(FX_STATE *fx, NriCommandBuffer *cmd) {
    if (!fx || !fx->owner || !cmd) return false;

    RENDERER *r = fx->owner;

    const VOLUME_COMPOSE_UNIFORMS compose = {
        .width = fx->width,
        .height = fx->height,
        .debug_view = fx->debug_view,
        .bypass_volume = 1u,
    };

    if (!bind_volume_compose_resources(r, cmd, fx->hdr, fx->volume, fx->normal_depth, fx->sampler, fx->depth_sampler, fx->lit, &compose, sizeof(compose)))
        return false;

    r->gpu->core.CmdSetPipeline(cmd, fx->volume_compose_pipeline);

    r->gpu->core.CmdDispatch(cmd, &(NriDispatchDesc){.workGroupNumX = (fx->width + 7u) / 8u, .workGroupNumY = (fx->height + 7u) / 8u, .workGroupNumZ = 1});

    return true;
}

static bool bloom_pass(FX_STATE *fx, NriCommandBuffer *cmd, NriTexture *source, NriTexture *destination, Uint32 src_width, Uint32 src_height, Uint32 phase) {
    const BLOOM_UNIFORMS u = {.src_width = src_width,
                              .src_height = src_height,
                              .dst_width = fx->ao_width,
                              .dst_height = fx->ao_height,
                              .phase = phase,
                              .threshold = 1.0f,
                              .knee = 0.55f,
                              .strength = 1.0f * 1e2};

    return dispatch_one(fx, cmd, fx->bloom_pipeline, source, destination, &u, sizeof(u), fx->ao_width, fx->ao_height);
}

static bool fx_apply_base(FX_STATE *fx, NriCommandBuffer *cmd, NriTexture *swap, float tan_half_fov, float aspect) {
    if (!fx || !fx->owner || !cmd || !swap || !fx->hdr || !fx->normal_depth) return false;

    RENDERER *r = fx->owner;

    const SSAO_UNIFORMS ao = {.width = fx->width,
                              .height = fx->height,
                              .ao_width = fx->ao_width,
                              .ao_height = fx->ao_height,
                              .tan_half_fov = tan_half_fov,
                              .aspect = aspect,
                              .radius = 0.65f,
                              .bias = 0.035f};

    NriTexture *hdr = fx->volume_ready ? fx->lit : fx->hdr;

    if (!dispatch_one(fx, cmd, fx->ssao_pipeline, fx->normal_depth, fx->ao, &ao, sizeof(ao), fx->ao_width, fx->ao_height) ||
        !bloom_pass(fx, cmd, fx->hdr, fx->bloom_a, fx->width, fx->height, 0u) ||
        !bloom_pass(fx, cmd, fx->bloom_a, fx->bloom_b, fx->ao_width, fx->ao_height, 1u) ||
        !bloom_pass(fx, cmd, fx->bloom_b, fx->bloom_a, fx->ao_width, fx->ao_height, 2u))
        return false;

    const COMPOSE_UNIFORMS u = {.exposure = 1.0f, .ao_strength = fx->debug_view >= 3u ? 0.0f : 0.62f, .bloom_strength = fx->debug_view >= 3u ? 0.0f : 0.22f};

    if (!begin_compose_rendering(r, cmd, swap, hdr, fx->ao, fx->bloom_a, fx->lut, fx->sampler, &u, sizeof(u))) return false;

    r->gpu->core.CmdSetPipeline(cmd, fx->compose_pipeline);
    r->gpu->core.CmdDraw(cmd, &(NriDrawDesc){.vertexNum = 3, .instanceNum = 1, .baseVertex = 0, .baseInstance = 0});

    r->gpu->core.CmdEndRendering(cmd);

    return true;
}

static bool fx_apply(FX_STATE *fx, NriCommandBuffer *cmd, NriTexture *swap, float tan_half_fov, float aspect) {
    if (!fx || !cmd || !swap) return false;

    const bool had_volume = fx->volume_ready;

    if (!had_volume) {
        if (!run_vision_only(fx, cmd)) return false;
        fx->volume_ready = true;
    }

    const bool ok = fx_apply_base(fx, cmd, swap, tan_half_fov, aspect);

    if (!had_volume) fx->volume_ready = false;

    return ok;
}

/*
 * Pipeline layouts mirror the descriptor spaces used by the shaders.
 * These helpers are intentionally named by the shader job they describe.
 * Each one mirrors the register spaces already present in your HLSL.
 *
 * The binding helper implementations are kept in gpu.c as well so render.c
 * never learns about NRI descriptor sets/views.
 */
static bool create_pipeline_layouts(RENDERER *r) {
    return create_surface_layout(r) && create_line_layout(r) && create_sky_layout(r) && bake_gpu_layouts_init(r) && create_ssao_layout(r) &&
           create_bloom_layout(r) && create_grade_layout(r) && create_volume_layout(r) && create_volume_compose_layout(r) && create_compose_layout(r);
}

static void destroy_pipeline_layouts(RENDERER *r) {
    if (!r) return;

    bake_gpu_layouts_deinit(r);

    NriPipelineLayout **layouts[] = {&r->surface_layout, &r->line_layout,           &r->sky_layout,    &r->ssao_layout, &r->bloom_layout, &r->grade_layout,
                                     &r->volume_layout,  &r->volume_compose_layout, &r->compose_layout};

    for (uint32_t i = 0; i < sizeof(layouts) / sizeof(layouts[0]); ++i) {
        if (*layouts[i]) {
            r->gpu->core.DestroyPipelineLayout(*layouts[i]);
            *layouts[i] = NULL;
        }
    }
}

bool gpu_init_worker(GPU *gpu) {
    if (!gpu) return false;

    memset(gpu, 0, sizeof(*gpu));

    NriDeviceCreationDesc device_desc = {0};
    device_desc.graphicsAPI = NriGraphicsAPI_VK;
    device_desc.vkBindingOffsets = (NriVKBindingOffsets){.sRegister = 0, .tRegister = 16, .bRegister = 32, .uRegister = 48};

    if (nriCreateDevice(&device_desc, &gpu->device) != NriResult_SUCCESS) {
        gpu_deinit(gpu);
        return false;
    }

    if (nriGetInterface(gpu->device, NRI_INTERFACE(NriCoreInterface), &gpu->core) != NriResult_SUCCESS ||
        nriGetInterface(gpu->device, NRI_INTERFACE(NriHelperInterface), &gpu->helper) != NriResult_SUCCESS) {
        gpu_deinit(gpu);
        return false;
    }

    RENDERER shell = {.gpu = gpu};

    if (!acquire_queues(&shell) || !create_pipeline_cache(&shell) || !create_gpu_timestamps(&shell) || !create_descriptor_pool(&shell) ||
        !create_work_contexts(&shell)) {
        gpu_deinit(gpu);
        return false;
    }

    return true;
}

bool gpu_init(GPU *gpu, const char *title, int width, int height) {
    if (!gpu || !title || width <= 0 || height <= 0) return false;

    memset(gpu, 0, sizeof(*gpu));
    gpu->window = SDL_CreateWindow(title, width, height,
                                   SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY
#if defined(__APPLE__)
                                       | SDL_WINDOW_METAL
#else
                                       | SDL_WINDOW_VULKAN
#endif
    );
    if (!gpu->window) return false;

    NriDeviceCreationDesc device_desc = {0};
    device_desc.graphicsAPI = NriGraphicsAPI_VK;
    device_desc.enableNRIValidation = false;
    device_desc.enableGraphicsAPIValidation = false;
    device_desc.vkBindingOffsets = (NriVKBindingOffsets){.sRegister = 0, .tRegister = 16, .bRegister = 32, .uRegister = 48};

    if (nriCreateDevice(&device_desc, &gpu->device) != NriResult_SUCCESS) {
        SDL_Log("NRI device creation failed");
        gpu_deinit(gpu);
        return false;
    }

    if (nriGetInterface(gpu->device, NRI_INTERFACE(NriCoreInterface), &gpu->core) != NriResult_SUCCESS ||
        nriGetInterface(gpu->device, NRI_INTERFACE(NriHelperInterface), &gpu->helper) != NriResult_SUCCESS ||
        nriGetInterface(gpu->device, NRI_INTERFACE(NriSwapChainInterface), &gpu->swapchain_api) != NriResult_SUCCESS) {
        SDL_Log("NRI interface acquisition failed");
        gpu_deinit(gpu);
        return false;
    }

    RENDERER shell = {.gpu = gpu};

    if (!acquire_queues(&shell) || !create_pipeline_cache(&shell) || !create_gpu_timestamps(&shell) || !create_swapchain(&shell, width, height) ||
        !create_descriptor_pool(&shell) || !create_work_contexts(&shell) || !create_frame_contexts(&shell)) {
        gpu_deinit(gpu);
        return false;
    }

    return true;
}

bool renderer_gpu_resources_init(RENDERER *r) {
    if (!r || !r->gpu || !r->gpu->device) return false;

    r->show_volume = true;
    r->yaw = -0.78f;
    r->pitch = 0.34f;
    r->distance = 14.0f;
    r->target = v3(0.0f, 1.0f, 0.0f);

    if (!create_pipeline_layouts(r)) {
        renderer_gpu_resources_deinit(r);
        return false;
    }

    if (r->gpu->core.GetFormatSupport(r->gpu->device, NriFormat_D32_SFLOAT) & NriFormatSupportBits_DEPTH_STENCIL_ATTACHMENT) {
        r->depth_format = NriFormat_D32_SFLOAT;
    } else if (r->gpu->core.GetFormatSupport(r->gpu->device, NriFormat_D24_UNORM_S8_UINT) & NriFormatSupportBits_DEPTH_STENCIL_ATTACHMENT) {
        r->depth_format = NriFormat_D24_UNORM_S8_UINT;
    } else {
        r->depth_format = NriFormat_D16_UNORM;
    }

    NriShaderDesc surface_vs = compile_shader("shaders/vertex.hlsl", "surface_vs", "BUILD_SURFACE_VS", NriStageBits_VERTEX_SHADER);
    NriShaderDesc surface_ps = compile_shader("shaders/fragment.hlsl", "surface_fs", "BUILD_SURFACE_FS", NriStageBits_FRAGMENT_SHADER);
    NriShaderDesc line_vs = compile_shader("shaders/vertex.hlsl", "wireframe_vs", "BUILD_WIREFRAME_VS", NriStageBits_VERTEX_SHADER);
    NriShaderDesc line_ps = compile_shader("shaders/fragment.hlsl", "wireframe_fs", "BUILD_WIREFRAME_FS", NriStageBits_FRAGMENT_SHADER);
    NriShaderDesc sky_vs = compile_shader("shaders/vertex.hlsl", "fullscreen_vs", "BUILD_FULLSCREEN_VS", NriStageBits_VERTEX_SHADER);
    NriShaderDesc sky_ps = compile_shader("shaders/fragment.hlsl", "sky_fs", "BUILD_SKY_FS", NriStageBits_FRAGMENT_SHADER);

    if (!surface_vs.bytecode || !surface_ps.bytecode || !line_vs.bytecode || !line_ps.bytecode || !sky_vs.bytecode || !sky_ps.bytecode) {
        free_shader(&surface_vs);
        free_shader(&surface_ps);
        free_shader(&line_vs);
        free_shader(&line_ps);
        free_shader(&sky_vs);
        free_shader(&sky_ps);
        renderer_gpu_resources_deinit(r);
        return false;
    }

    r->solid_pipeline = make_surface_pipeline(r, &r->gpu->core, r->surface_layout, &surface_vs, &surface_ps);
    r->line_pipeline = make_line_pipeline(r, r->line_layout, &line_vs, &line_ps);
    r->sky_pipeline = make_sky_pipeline(r, r->sky_layout, &sky_vs, &sky_ps);

    free_shader(&surface_vs);
    free_shader(&surface_ps);
    free_shader(&line_vs);
    free_shader(&line_ps);
    free_shader(&sky_vs);
    free_shader(&sky_ps);

    if (!r->solid_pipeline || !r->line_pipeline || !r->sky_pipeline || !fx_init(&r->fx, r)) {
        renderer_gpu_resources_deinit(r);
        return false;
    }

    SDL_Log("GPU backend: NRI");
    SDL_Log("depth format: %s", r->depth_format == NriFormat_D32_SFLOAT ? "D32_FLOAT" : r->depth_format == NriFormat_D24_UNORM_S8_UINT ? "D24S8" : "D16_UNORM");
    SDL_Log("SDL_image: %d", IMG_Version());

    return true;
}

bool draw_frame(RENDERER *r, const RENDER_FRAME *frame) {
    if (!r || !frame || !r->gpu->device || !r->solid_pipeline || !r->sky_pipeline || !r->vertex_buffer || !r->lightmap_texture || !r->lightmap_sampler)
        return false;

    uint32_t width = 0;
    uint32_t height = 0;

    SDL_GetWindowSizeInPixels(r->gpu->window, (int *)&width, (int *)&height);

    if (!width || !height) return true;

    if (!r->gpu->swapchain || width != r->gpu->swapchain_width || height != r->gpu->swapchain_height) {
        if (r->gpu->core.QueueWaitIdle(r->gpu->graphics_queue) != NriResult_SUCCESS) return false;
        destroy_swapchain(r);

        if (!create_swapchain(r, width, height)) return false;
    }

    if (!fx_ensure(&r->fx, width, height) || !ensure_depth_texture(r, width, height)) return false;

    uint32_t swap_index = 0;

    if (!acquire_swapchain_texture(r, &swap_index)) {
        destroy_swapchain(r);

        return false;
    }

    r->gpu->current_swap_index = swap_index;

    NriTexture *swap = r->gpu->swapchain_textures[swap_index];

    FRAME_CONTEXT *queued_frame = NULL;
    NriCommandBuffer *cmd = NULL;

    if (!begin_frame_commands(r, &queued_frame, &cmd)) goto failed_frame;

    CAMERA_UNIFORMS camera = {0};

    memcpy(camera.mvp, frame->mvp, sizeof(camera.mvp));
    memcpy(camera.view, frame->view, sizeof(camera.view));

    const SKY_UNIFORMS sky = {.camera_right = {frame->right.x * frame->tan_half_fov * frame->aspect, frame->right.y * frame->tan_half_fov * frame->aspect,
                                               frame->right.z * frame->tan_half_fov * frame->aspect, 0},
                              .camera_up = {frame->up.x * frame->tan_half_fov, frame->up.y * frame->tan_half_fov, frame->up.z * frame->tan_half_fov, 0},
                              .camera_forward = {frame->forward.x, frame->forward.y, frame->forward.z, 0},
                              .sky_zenith = {frame->sky.zenith.x, frame->sky.zenith.y, frame->sky.zenith.z, frame->sky.intensity},
                              .sky_horizon = {frame->sky.horizon.x, frame->sky.horizon.y, frame->sky.horizon.z, 1.0f},
                              .sun_direction_intensity = {frame->sun.direction.x, frame->sun.direction.y, frame->sun.direction.z, frame->sun.intensity},
                              .sun_color_radius = {frame->sun.color.x, frame->sun.color.y, frame->sun.color.z, frame->sun.angular_radius}};

    if (!begin_scene_rendering(r, cmd, r->fx.hdr, r->fx.normal_depth, r->depth_texture, width, height)) goto failed_frame;

    if (!bind_sky_resources(r, cmd, &sky, sizeof(sky))) goto failed_frame;
    r->gpu->core.CmdSetPipeline(cmd, r->sky_pipeline);

    r->gpu->core.CmdDraw(cmd, &(NriDrawDesc){.vertexNum = 3, .instanceNum = 1, .baseVertex = 0, .baseInstance = 0});

    const NriVertexBufferDesc vertex = {.buffer = r->vertex_buffer, .offset = 0, .stride = sizeof(RENDER_VERTEX)};

    r->gpu->core.CmdSetVertexBuffers(cmd, 0, &vertex, 1);
    r->gpu->core.CmdSetPipeline(cmd, r->solid_pipeline);

    if (!bind_camera_resources(r, cmd, &camera, sizeof(camera))) goto failed_frame;

    for (uint32_t i = 0; i < r->draw_count; ++i) {
        const DRAW_RANGE *draw = &r->draws[i];
        const RENDER_MATERIAL *m = &r->materials[draw->material];

        const MATERIAL_UNIFORMS material = {
            .base_color_factor = {m->data.base_color[0], m->data.base_color[1], m->data.base_color[2], m->data.base_color[3]},
            .emissive_metallic = {m->data.emissive[0], m->data.emissive[1], m->data.emissive[2], m->data.metallic},
            .roughness_normal_ao_sun = {m->data.roughness, m->data.normal_scale, m->data.occlusion_strength, frame->sun.intensity},
            .sun_direction = {frame->sun.direction.x, frame->sun.direction.y, frame->sun.direction.z, 0},
            .sun_color = {frame->sun.color.x, frame->sun.color.y, frame->sun.color.z, 1},
            .camera_position = {frame->eye.x, frame->eye.y, frame->eye.z, r->debug_view == 1u ? 2.0f : (r->has_bake ? 1.0f : 0.0f)}};

        if (!bind_surface_resources(r, cmd, m, r->lightmap_texture, r->material_sampler, r->lightmap_sampler, &material, sizeof(material))) goto failed_frame;

        r->gpu->core.CmdDraw(cmd, &(NriDrawDesc){.vertexNum = draw->count, .instanceNum = 1, .baseVertex = draw->first, .baseInstance = 0});
    }

    if (r->show_debug && r->debug_vertex_count) {
        r->gpu->core.CmdSetPipeline(cmd, r->line_pipeline);

        if (!bind_line_resources(r, cmd, camera.mvp, sizeof(camera.mvp))) goto failed_frame;

        r->gpu->core.CmdDraw(cmd, &(NriDrawDesc){.vertexNum = r->debug_vertex_count, .instanceNum = 1, .baseVertex = r->debug_vertex_start, .baseInstance = 0});
    }

    r->gpu->core.CmdEndRendering(cmd);

    r->fx.volume_ready = false;
    r->fx.debug_view = r->debug_view;

    static uint32_t last_logged_view = UINT32_MAX;

    if (last_logged_view != r->fx.debug_view) {
        SDL_Log("GPU debug view: %u | fog: %d | bake: %d", r->fx.debug_view, r->show_volume, r->has_bake);

        last_logged_view = r->fx.debug_view;
    }

    if (r->show_volume && r->has_bake && (r->debug_view == 0u || r->debug_view >= 3u) && r->volume_probe_buffer && r->beam_buffer &&
        !fx_volume(&r->fx, cmd, r->volume_probe_buffer, r->beam_buffer, &r->volume_probes, &r->beams, frame))
        goto failed_frame;

    if (!fx_apply(&r->fx, cmd, swap, frame->tan_half_fov, frame->aspect)) goto failed_frame;

    if (!submit_frame(r, queued_frame, cmd, swap_index)) return false;

    return true;

failed_frame:
    abort_frame_commands(r, queued_frame);
    destroy_swapchain(r);

    return false;
}

void renderer_gpu_resources_deinit(RENDERER *r) {
    if (!r) return;

    GPU *gpu = r->gpu;
    free(r->vertices);
    free(r->draws);
    free_probe_grid(&r->volume_probes);
    beam_free(&r->beams);

    if (gpu && gpu->device) {
        if (gpu->graphics_queue) gpu->core.QueueWaitIdle(gpu->graphics_queue);

        gpu_clear_temporary(r);
        probe_wavefront_scratch_destroy(r);
        fx_deinit(&r->fx);

        if (r->image_textures) {
            for (uint32_t i = 0; i < r->image_texture_count; ++i) release_texture(r, r->image_textures[i]);
        }

        free(r->image_textures);
        free(r->materials);
        release_texture(r, r->default_white);
        release_texture(r, r->default_normal);

        if (r->material_sampler) gpu->core.DestroyDescriptor(r->material_sampler);

        release_buffer(r, r->vertex_buffer);
        release_bake_resources(r);
        release_buffer(r, r->volume_probe_buffer);
        release_buffer(r, r->beam_buffer);
        release_texture(r, r->depth_texture);
        release_texture(r, r->lightmap_texture);

        if (r->lightmap_sampler) gpu->core.DestroyDescriptor(r->lightmap_sampler);
        if (r->sky_pipeline) gpu->core.DestroyPipeline(r->sky_pipeline);
        if (r->solid_pipeline) gpu->core.DestroyPipeline(r->solid_pipeline);
        if (r->line_pipeline) gpu->core.DestroyPipeline(r->line_pipeline);

        destroy_pipeline_layouts(r);
    }

    memset(r, 0, sizeof(*r));
}

void gpu_deinit(GPU *gpu) {
    if (!gpu) return;

    RENDERER shell = {.gpu = gpu};

    if (gpu->device) {
        if (gpu->graphics_queue) gpu->core.QueueWaitIdle(gpu->graphics_queue);

        destroy_upload_context(&shell);
        destroy_work_contexts(&shell);
        gpu_clear_temporary(&shell);
        destroy_frame_contexts(&shell);
        destroy_gpu_timestamps(&shell);
        destroy_swapchain(&shell);

        if (gpu->descriptor_pool) gpu->core.DestroyDescriptorPool(gpu->descriptor_pool);
        destroy_pipeline_cache(&shell);
        nriDestroyDevice(gpu->device);
    }

    free(gpu->temporary_descriptors);
    free(gpu->temporary_buffers);
    free(gpu->texture_states);

#if defined(__APPLE__)
    if (gpu->metal_view) SDL_Metal_DestroyView(gpu->metal_view);
#endif

    if (gpu->window) SDL_DestroyWindow(gpu->window);
    memset(gpu, 0, sizeof(*gpu));
}
