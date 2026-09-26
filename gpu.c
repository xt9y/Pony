#include "game.h"

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

typedef struct BAKE_UNIFORMS {
    Uint32 item_count;
    Uint32 lightmap_width;
    Uint32 lightmap_height;
    Uint32 dispatch_width;
    Uint32 iteration;
    Uint32 phase;
    Uint32 max_bounces;
    Uint32 batch_count;
    float sun_direction_intensity[4];
    float sun_color_radius[4];
    float sky_zenith[4];
    float sky_horizon[4];
    float bake_params[4];
    float probe_origin_spacing[4];
    Uint32 probe_dims_mode[4];
    float emissive_data[4];
} BAKE_UNIFORMS;

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

static bool create_pipeline_cache(renderer *r) {
    if (!r || !r->device) return false;

    if (r->pipeline_cache) return true;

    const NriPipelineCacheDesc desc = {0};
    NriResult result = r->core.CreatePipelineCache(r->device, &desc, &r->pipeline_cache);

    if (result == NriResult_UNSUPPORTED) {
        r->pipeline_cache = NULL;

        return true;
    }

    return result == NriResult_SUCCESS;
}

static void destroy_pipeline_cache(renderer *r) {
    if (!r || !r->pipeline_cache) return;
    r->core.DestroyPipelineCache(r->pipeline_cache);
    r->pipeline_cache = NULL;
}

static bool acquire_queues(renderer *r) {
    if (!r || !r->device) return false;

    if (r->core.GetQueue(r->device, NriQueueType_GRAPHICS, 0, &r->graphics_queue) != NriResult_SUCCESS) return false;

    if (r->core.GetQueue(r->device, NriQueueType_COMPUTE, 0, &r->compute_queue) != NriResult_SUCCESS) r->compute_queue = r->graphics_queue;

    if (r->core.GetQueue(r->device, NriQueueType_COPY, 0, &r->copy_queue) != NriResult_SUCCESS) r->copy_queue = r->graphics_queue;

    r->work_queue = r->compute_queue ? r->compute_queue : r->graphics_queue;

    return true;
}

static bool create_gpu_timestamps(renderer *r) {
    if (!r || !r->device) return false;

    const NriDeviceDesc *device = r->core.GetDeviceDesc(r->device);

    if (!device || !device->features.timestamp || !device->other.timestampFrequencyHz) return true;

    const NriQueryPoolDesc query_desc = {.queryType = NriQueryType_TIMESTAMP, .capacity = TIMESTAMP_CAPACITY};

    if (r->core.CreateQueryPool(r->device, &query_desc, &r->timestamp_pool) != NriResult_SUCCESS) return true;

    r->timestamp_query_size = r->core.GetQuerySize(r->timestamp_pool);

    if (!r->timestamp_query_size) {
        r->core.DestroyQueryPool(r->timestamp_pool);
        r->timestamp_pool = NULL;

        return true;
    }

    const NriBufferDesc readback_desc = {.size = (uint64_t)r->timestamp_query_size * TIMESTAMP_CAPACITY};

    if (r->core.CreateCommittedBuffer(r->device, NriMemoryLocation_HOST_READBACK, 0.0f, &readback_desc, &r->timestamp_readback) != NriResult_SUCCESS) {
        r->core.DestroyQueryPool(r->timestamp_pool);
        r->timestamp_pool = NULL;
        r->timestamp_query_size = 0u;

        return true;
    }

    r->timestamp_supported = true;

    return true;
}

static void destroy_gpu_timestamps(renderer *r) {
    if (!r) return;

    if (r->timestamp_readback) r->core.DestroyBuffer(r->timestamp_readback);

    if (r->timestamp_pool) r->core.DestroyQueryPool(r->timestamp_pool);
    r->timestamp_readback = NULL;
    r->timestamp_pool = NULL;
    r->timestamp_query_size = 0u;
    r->timestamp_supported = false;
}

static bool gpu_timestamp_begin(renderer *r, NriCommandBuffer *cmd, uint32_t slot) {
    if (!r || !cmd || !r->timestamp_supported) return true;

    if (slot + 1u >= TIMESTAMP_CAPACITY) return false;
    r->core.CmdResetQueries(cmd, r->timestamp_pool, slot, 2u);
    r->core.CmdEndQuery(cmd, r->timestamp_pool, slot);

    return true;
}

static bool gpu_timestamp_end(renderer *r, NriCommandBuffer *cmd, uint32_t slot) {
    if (!r || !cmd || !r->timestamp_supported) return true;

    if (slot + 1u >= TIMESTAMP_CAPACITY) return false;
    r->core.CmdEndQuery(cmd, r->timestamp_pool, slot + 1u);
    r->core.CmdCopyQueries(cmd, r->timestamp_pool, slot, 2u, r->timestamp_readback, (uint64_t)slot * r->timestamp_query_size);

    return true;
}

static void gpu_timestamp_log(renderer *r, uint32_t slot, const char *label) {
    if (!r || !r->timestamp_supported || slot + 1u >= TIMESTAMP_CAPACITY || !label) return;

    const NriDeviceDesc *device = r->core.GetDeviceDesc(r->device);

    if (!device || !device->other.timestampFrequencyHz) return;

    const uint64_t offset = (uint64_t)slot * r->timestamp_query_size;
    const uint64_t bytes = (uint64_t)r->timestamp_query_size * 2u;
    const uint8_t *mapped = r->core.MapBuffer(r->timestamp_readback, offset, bytes);

    if (!mapped) return;

    uint64_t begin = 0u;
    uint64_t end = 0u;

    memcpy(&begin, mapped, sizeof(begin));
    memcpy(&end, mapped + r->timestamp_query_size, sizeof(end));
    r->core.UnmapBuffer(r->timestamp_readback);

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

    return (NriShaderDesc){
        .stage = stage,
        .bytecode = spirv,
        .size = spirv_size,
        .entryPointName = entrypoint
    };
}

static NriPipeline *compile_compute(renderer *r, NriPipelineLayout *layout, const char *path, const char *entrypoint, const char *define) {
    (void)path;

    size_t spirv_size = 0;
    Uint8 *spirv = load_spirv(define, &spirv_size);

    if (!spirv) return NULL;

    const NriShaderDesc shader = {.stage = NriStageBits_COMPUTE_SHADER, .bytecode = spirv, .size = spirv_size, .entryPointName = entrypoint};
    const NriComputePipelineDesc desc = {.pipelineLayout = layout, .shader = shader, .cache = r->pipeline_cache};
    NriPipeline *pipeline = NULL;
    NriResult result = r->core.CreateComputePipeline(r->device, &desc, &pipeline);

    SDL_free(spirv);

    if (result != NriResult_SUCCESS) {
        SDL_Log("compute pipeline creation failed for %s:%s", define, entrypoint);

        return NULL;
    }

    return pipeline;
}

static NriPipeline *make_surface_pipeline(RENDERER *r, NriCoreInterface *core, NriPipelineLayout *layout, const NriShaderDesc *vs, const NriShaderDesc *ps) {

    const NriVertexStreamDesc vb = {
        .bindingSlot = 0,
        .stepRate = NriVertexStreamStepRate_PER_VERTEX,
        .stride = (uint16_t)sizeof(RENDER_VERTEX)
    };

    const NriVertexAttributeDesc attrs[4] = {
        {
            .d3d = {
                .semanticName = "TEXCOORD",
                .semanticIndex = 0
            },
            .vk = {
                .location = 0
            },
            .offset = (uint32_t)offsetof(RENDER_VERTEX, x),
            .format = NriFormat_RGB32_SFLOAT,
            .streamIndex = 0
        },
        {
            .d3d = {
                .semanticName = "TEXCOORD",
                .semanticIndex = 1
            },
            .vk = {
                .location = 1
            },
            .offset = (uint32_t)offsetof(RENDER_VERTEX, nx),
            .format = NriFormat_RGB32_SFLOAT,
            .streamIndex = 0
        },
        {
            .d3d = {
                .semanticName = "TEXCOORD",
                .semanticIndex = 2
            },
            .vk = {
                .location = 2
            },
            .offset = (uint32_t)offsetof(RENDER_VERTEX, u),
            .format = NriFormat_RG32_SFLOAT,
            .streamIndex = 0
        },
        {
            .d3d = {
                .semanticName = "TEXCOORD",
                .semanticIndex = 3
            },
            .vk = {
                .location = 3
            },
            .offset = (uint32_t)offsetof(RENDER_VERTEX, lu),
            .format = NriFormat_RG32_SFLOAT,
            .streamIndex = 0
        }
    };

    const NriVertexInputDesc vertex_input = {
        .attributes = attrs,
        .attributeNum = 4,
        .streams = &vb,
        .streamNum = 1
    };

    const NriColorAttachmentDesc targets[2] = {
        {
            .format = NriFormat_RGBA16_SFLOAT,
            .colorWriteMask = NriColorWriteBits_RGBA
        }, {
            .format = NriFormat_RGBA16_SFLOAT,
            .colorWriteMask = NriColorWriteBits_RGBA
        }
    };

    const NriMultisampleDesc multisample = {
        .sampleMask = NRI_ALL,
        .sampleNum = 1
    };

    const NriShaderDesc shaders[2] = {*vs, *ps};

    const NriGraphicsPipelineDesc desc = {.pipelineLayout = layout,

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
                                          .cache = r->pipeline_cache};

    NriPipeline *pipeline = NULL;

    if (core->CreateGraphicsPipeline(r->device, &desc, &pipeline) != NriResult_SUCCESS) {
        return NULL;
    }

    return pipeline;
}

static void clear_temporary(renderer *r);
static bool begin_work_commands(renderer *r, NriCommandAllocator **allocator, NriCommandBuffer **command_buffer);
static bool submit_work_commands(renderer *r, NriCommandAllocator *allocator, NriCommandBuffer *command_buffer, bool wait);
static void abort_work_commands(renderer *r, NriCommandAllocator *allocator, NriCommandBuffer *command_buffer);
static void work_drain(renderer *r);

static NriResult begin_commands(renderer *r, NriCommandAllocator **allocator, NriCommandBuffer **command_buffer) {
    return begin_work_commands(r, allocator, command_buffer) ? NriResult_SUCCESS : NriResult_FAILURE;
}

static bool submit_commands(renderer *r, NriCommandAllocator *allocator, NriCommandBuffer *command_buffer) {
    return submit_work_commands(r, allocator, command_buffer, true);
}

static bool submit_commands_async(renderer *r, NriCommandAllocator *allocator, NriCommandBuffer *command_buffer) {
    return submit_work_commands(r, allocator, command_buffer, false);
}

static void abort_commands(renderer *r, NriCommandAllocator *allocator, NriCommandBuffer *cmd) {
    abort_work_commands(r, allocator, cmd);
}

/* Descriptor sets match the HLSL register spaces in shaders/. */
static bool create_pipeline_layout(RENDERER *r, NriPipelineLayout **out, const NriDescriptorType *types[4], const uint8_t counts[4], NriStageBits stages) {
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

            ranges[set][i] = (NriDescriptorRangeDesc){
                .baseRegisterIndex = reg,
                .descriptorNum = 1,
                .descriptorType = type,
                .shaderStages = stages
            };
        }
    }

    const NriPipelineLayoutDesc desc = {
        .descriptorSets = sets,
        .descriptorSetNum = 4,
        .rootRegisterSpace = 4,
        .shaderStages = stages
    };

    return r->core.CreatePipelineLayout(r->device, &desc, out) == NriResult_SUCCESS;
}

static bool create_pipeline_layouts(RENDERER *r);

static bool create_surface_layout(RENDERER *r) {
    static const NriDescriptorType camera[] = {NriDescriptorType_CONSTANT_BUFFER};

    static const NriDescriptorType material[] = {
        NriDescriptorType_TEXTURE,
        NriDescriptorType_TEXTURE,
        NriDescriptorType_TEXTURE,
        NriDescriptorType_TEXTURE,
        NriDescriptorType_TEXTURE,
        NriDescriptorType_TEXTURE,
        NriDescriptorType_SAMPLER,
        NriDescriptorType_SAMPLER,
        NriDescriptorType_SAMPLER,
        NriDescriptorType_SAMPLER,
        NriDescriptorType_SAMPLER,
        NriDescriptorType_SAMPLER
    };

    static const NriDescriptorType uniform[] = {NriDescriptorType_CONSTANT_BUFFER};

    const NriDescriptorType *sets[4] = {NULL, camera, material, uniform};
    const uint8_t counts[4] = {0, 1, 12, 1};

    return create_pipeline_layout(r, &r->surface_layout, sets, counts, NriStageBits_VERTEX_SHADER | NriStageBits_FRAGMENT_SHADER);
}

static bool create_line_layout(RENDERER *r) {
    static const NriDescriptorType camera[] = {NriDescriptorType_CONSTANT_BUFFER};
    const NriDescriptorType *sets[4] = {NULL, camera, NULL, NULL};
    const uint8_t counts[4] = {0, 1, 0, 0};

    return create_pipeline_layout(r, &r->line_layout, sets, counts, NriStageBits_VERTEX_SHADER | NriStageBits_FRAGMENT_SHADER);
}

static bool create_sky_layout(RENDERER *r) {
    static const NriDescriptorType uniform[] = {NriDescriptorType_CONSTANT_BUFFER};

    const NriDescriptorType *sets[4] = {NULL, NULL, NULL, uniform};
    const uint8_t counts[4] = {0, 0, 0, 1};

    return create_pipeline_layout(r, &r->sky_layout, sets, counts, NriStageBits_VERTEX_SHADER | NriStageBits_FRAGMENT_SHADER);
}

static bool create_compute_layout(RENDERER *r, NriPipelineLayout **out, const NriDescriptorType *sources, uint8_t source_num, NriDescriptorType output_type, bool has_uniform) {
    static const NriDescriptorType output_texture[] = {NriDescriptorType_STORAGE_TEXTURE};

    static const NriDescriptorType output_buffer[] = {NriDescriptorType_STORAGE_STRUCTURED_BUFFER};

    static const NriDescriptorType uniform[] = {NriDescriptorType_CONSTANT_BUFFER};

    const NriDescriptorType *sets[4] = {sources, output_type == NriDescriptorType_STORAGE_TEXTURE ? output_texture : output_buffer, has_uniform ? uniform : NULL, NULL};

    const uint8_t counts[4] = {source_num, 1, has_uniform ? 1 : 0, 0};

    return create_pipeline_layout(r, out, sets, counts, NriStageBits_COMPUTE_SHADER);
}

static bool create_bake_layout(RENDERER *r) {
    static const NriDescriptorType src[] = {NriDescriptorType_TEXTURE,           NriDescriptorType_SAMPLER,           NriDescriptorType_TEXTURE,
                                            NriDescriptorType_SAMPLER,           NriDescriptorType_STRUCTURED_BUFFER, NriDescriptorType_STRUCTURED_BUFFER,
                                            NriDescriptorType_STRUCTURED_BUFFER, NriDescriptorType_STRUCTURED_BUFFER, NriDescriptorType_STRUCTURED_BUFFER,
                                            NriDescriptorType_STRUCTURED_BUFFER, NriDescriptorType_STRUCTURED_BUFFER, NriDescriptorType_STRUCTURED_BUFFER};

    static const NriDescriptorType dst[] = {NriDescriptorType_STORAGE_TEXTURE, NriDescriptorType_STORAGE_STRUCTURED_BUFFER, NriDescriptorType_STORAGE_STRUCTURED_BUFFER};
    static const NriDescriptorType uniform[] = {NriDescriptorType_CONSTANT_BUFFER};
    const NriDescriptorType *sets[4] = {src, dst, uniform, NULL};
    const uint8_t counts[4] = {12, 3, 1, 0};

    return create_pipeline_layout(r, &r->bake_layout, sets, counts, NriStageBits_COMPUTE_SHADER);
}

static bool create_lightmap_queue_layouts(renderer *r) {
    static const NriDescriptorType read_count[] = {NriDescriptorType_STRUCTURED_BUFFER};
    static const NriDescriptorType write_buffer[] = {NriDescriptorType_STORAGE_STRUCTURED_BUFFER};
    static const NriDescriptorType uniform[] = {NriDescriptorType_CONSTANT_BUFFER};

    const NriDescriptorType *reset_sets[4] = {NULL, write_buffer, NULL, NULL};
    const uint8_t reset_counts[4] = {0, 1, 0, 0};
    const NriDescriptorType *args_sets[4] = {read_count, write_buffer, uniform, NULL};
    const uint8_t args_counts[4] = {1, 1, 1, 0};

    return create_pipeline_layout(r, &r->lightmap_queue_reset_layout, reset_sets, reset_counts, NriStageBits_COMPUTE_SHADER) &&
           create_pipeline_layout(r, &r->lightmap_queue_args_layout, args_sets, args_counts, NriStageBits_COMPUTE_SHADER);
}

static bool create_probe_layout(RENDERER *r) {
    static const NriDescriptorType src[] = {NriDescriptorType_STRUCTURED_BUFFER, NriDescriptorType_STRUCTURED_BUFFER, NriDescriptorType_STRUCTURED_BUFFER};

    return create_compute_layout(r, &r->probe_layout, src, 3, NriDescriptorType_STORAGE_STRUCTURED_BUFFER, true);
}

static bool create_ssao_layout(RENDERER *r) {
    static const NriDescriptorType src[] = {NriDescriptorType_TEXTURE, NriDescriptorType_SAMPLER};

    return create_compute_layout(r, &r->ssao_layout, src, 2, NriDescriptorType_STORAGE_TEXTURE, true);
}

static bool create_bloom_layout(RENDERER *r) {
    static const NriDescriptorType src[] = {NriDescriptorType_TEXTURE, NriDescriptorType_SAMPLER};

    return create_compute_layout(r, &r->bloom_layout, src, 2, NriDescriptorType_STORAGE_TEXTURE, true);
}

static bool create_grade_layout(RENDERER *r) {
    return create_compute_layout(r, &r->grade_layout, NULL, 0, NriDescriptorType_STORAGE_TEXTURE, false);
}

static bool create_volume_layout(RENDERER *r) {
    static const NriDescriptorType src[] = {NriDescriptorType_TEXTURE, NriDescriptorType_SAMPLER, NriDescriptorType_STRUCTURED_BUFFER, NriDescriptorType_STRUCTURED_BUFFER};

    return create_compute_layout(r, &r->volume_layout, src, 4, NriDescriptorType_STORAGE_TEXTURE, true);
}

static bool create_volume_compose_layout(RENDERER *r) {
    static const NriDescriptorType src[] = {
        NriDescriptorType_TEXTURE, NriDescriptorType_TEXTURE, NriDescriptorType_TEXTURE, NriDescriptorType_SAMPLER, NriDescriptorType_SAMPLER, NriDescriptorType_SAMPLER
    };

    return create_compute_layout(r, &r->volume_compose_layout, src, 6, NriDescriptorType_STORAGE_TEXTURE, true);
}

static bool create_compose_layout(RENDERER *r) {
    static const NriDescriptorType src[] = {
        NriDescriptorType_TEXTURE,
        NriDescriptorType_TEXTURE,
        NriDescriptorType_TEXTURE,
        NriDescriptorType_TEXTURE,
        NriDescriptorType_SAMPLER,
        NriDescriptorType_SAMPLER,
        NriDescriptorType_SAMPLER,
        NriDescriptorType_SAMPLER
    };

    static const NriDescriptorType uniform[] = {NriDescriptorType_CONSTANT_BUFFER};

    const NriDescriptorType *sets[4] = {NULL, NULL, src, uniform};
    const uint8_t counts[4] = {0, 0, 8, 1};

    return create_pipeline_layout(r, &r->compose_layout, sets, counts, NriStageBits_VERTEX_SHADER | NriStageBits_FRAGMENT_SHADER);
}

static bool create_descriptor_pool_object(RENDERER *r, NriDescriptorPool **pool) {
    const NriDescriptorPoolDesc desc = {
        .descriptorSetMaxNum = 8192,
        .samplerMaxNum = 8192,
        .textureMaxNum = 8192,
        .storageTextureMaxNum = 8192,
        .structuredBufferMaxNum = 8192,
        .storageStructuredBufferMaxNum = 8192,
        .constantBufferMaxNum = 8192
    };

    return r->core.CreateDescriptorPool(r->device, &desc, pool) == NriResult_SUCCESS;
}

static bool create_descriptor_pool(RENDERER *r) {
    return create_descriptor_pool_object(r, &r->descriptor_pool);
}

static bool track_descriptor_array(RENDERER *r, NriDescriptor ***items, uint32_t *count, uint32_t *capacity, NriDescriptor *descriptor) {
    if (!descriptor) return false;

    if (*count == *capacity) {
        uint32_t cap = *capacity ? *capacity * 2u : 64u;
        NriDescriptor **data = realloc(*items, cap * sizeof(*data));

        if (!data) {
            r->core.DestroyDescriptor(descriptor);

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
            r->core.DestroyBuffer(buffer);

            return false;
        }
        *items = data;
        *capacity = cap;
    }

    (*items)[(*count)++] = buffer;

    return true;
}

static bool track_descriptor(RENDERER *r, NriDescriptor *descriptor) {
    if (r->active_frame) {
        FRAME_CONTEXT *frame = r->active_frame;

        return track_descriptor_array(r, &frame->temporary_descriptors, &frame->temporary_descriptor_num, &frame->temporary_descriptor_cap, descriptor);
    }

    return track_descriptor_array(r, &r->temporary_descriptors, &r->temporary_descriptor_num, &r->temporary_descriptor_cap, descriptor);
}

static bool track_buffer(RENDERER *r, NriBuffer *buffer) {
    if (r->active_frame) {
        FRAME_CONTEXT *frame = r->active_frame;

        return track_buffer_array(r, &frame->temporary_buffers, &frame->temporary_buffer_num, &frame->temporary_buffer_cap, buffer);
    }

    return track_buffer_array(r, &r->temporary_buffers, &r->temporary_buffer_num, &r->temporary_buffer_cap, buffer);
}

static void clear_temporary(RENDERER *r) {
    for (uint32_t i = 0; i < r->temporary_descriptor_num; ++i)
        r->core.DestroyDescriptor(r->temporary_descriptors[i]);

    for (uint32_t i = 0; i < r->temporary_buffer_num; ++i)
        r->core.DestroyBuffer(r->temporary_buffers[i]);
    r->temporary_descriptor_num = r->temporary_buffer_num = 0;

    if (r->descriptor_pool) r->core.ResetDescriptorPool(r->descriptor_pool);
}

static void clear_frame_temporary(RENDERER *r, FRAME_CONTEXT *frame) {
    if (!frame) return;

    for (uint32_t i = 0; i < frame->temporary_descriptor_num; ++i)
        r->core.DestroyDescriptor(frame->temporary_descriptors[i]);

    for (uint32_t i = 0; i < frame->temporary_buffer_num; ++i)
        r->core.DestroyBuffer(frame->temporary_buffers[i]);
    frame->temporary_descriptor_num = 0;
    frame->temporary_buffer_num = 0;
    frame->uniform_offset = 0u;

    if (frame->descriptor_pool) r->core.ResetDescriptorPool(frame->descriptor_pool);
}

static bool create_uniform_ring(renderer *r, frame_context *context) {
    if (!r || !context) return false;

    const NriBufferDesc desc = {.size = UNIFORM_RING_BYTES, .usage = NriBufferUsageBits_CONSTANT};

    return r->core.CreateCommittedBuffer(r->device, NriMemoryLocation_HOST_UPLOAD, 0.0f, &desc, &context->uniform_buffer) == NriResult_SUCCESS;
}

static bool create_work_contexts(renderer *r) {
    if (!r || !r->device || !r->work_queue) return false;

    r->work_contexts = calloc(WORK_QUEUE_DEPTH, sizeof(*r->work_contexts));

    if (!r->work_contexts) return false;

    if (r->core.CreateFence(r->device, 0u, &r->work_fence) != NriResult_SUCCESS) return false;

    for (uint32_t i = 0; i < WORK_QUEUE_DEPTH; ++i) {
        frame_context *work = &r->work_contexts[i];

        if (!create_descriptor_pool_object(r, &work->descriptor_pool) || !create_uniform_ring(r, work) ||
            r->core.CreateCommandAllocator(r->work_queue, &work->allocator) != NriResult_SUCCESS ||
            r->core.CreateCommandBuffer(work->allocator, &work->command_buffer) != NriResult_SUCCESS)
            return false;
    }

    r->work_next_fence = 1u;

    return true;
}

static void work_drain(renderer *r) {
    if (!r || !r->work_fence || r->work_next_fence <= 1u) return;
    r->core.Wait(r->work_fence, r->work_next_fence - 1u);
}

static void destroy_work_contexts(renderer *r) {
    if (!r) return;
    work_drain(r);

    if (r->work_contexts) {
        for (uint32_t i = 0; i < WORK_QUEUE_DEPTH; ++i) {
            frame_context *work = &r->work_contexts[i];
            clear_frame_temporary(r, work);

            if (work->command_buffer) r->core.DestroyCommandBuffer(work->command_buffer);

            if (work->allocator) r->core.DestroyCommandAllocator(work->allocator);

            if (work->descriptor_pool) r->core.DestroyDescriptorPool(work->descriptor_pool);

            if (work->uniform_buffer) r->core.DestroyBuffer(work->uniform_buffer);

            free(work->temporary_descriptors);
            free(work->temporary_buffers);
        }

        free(r->work_contexts);
    }

    r->work_contexts = NULL;
    r->active_work = NULL;

    if (r->work_fence) r->core.DestroyFence(r->work_fence);
    r->work_fence = NULL;
}

static bool begin_work_commands(renderer *r, NriCommandAllocator **allocator, NriCommandBuffer **command_buffer) {
    if (!r || !r->work_contexts || !r->work_fence || !r->work_queue || !allocator || !command_buffer) return false;

    frame_context *work = &r->work_contexts[r->work_index % WORK_QUEUE_DEPTH];

    if (work->fence_value && r->core.GetFenceValue(r->work_fence) < work->fence_value) r->core.Wait(r->work_fence, work->fence_value);

    clear_frame_temporary(r, work);
    r->core.ResetCommandAllocator(work->allocator);
    r->current_graphics_layout = r->current_compute_layout = NULL;
    r->active_work = work;
    r->active_frame = work;

    if (r->core.BeginCommandBuffer(work->command_buffer, work->descriptor_pool) != NriResult_SUCCESS) {
        r->active_work = NULL;
        r->active_frame = NULL;

        return false;
    }

    *allocator = work->allocator;
    *command_buffer = work->command_buffer;
    return true;
}

static bool submit_work_commands(renderer *r, NriCommandAllocator *allocator, NriCommandBuffer *command_buffer, bool wait) {
    frame_context *work = r ? r->active_work : NULL;

    if (!r || !work || work->allocator != allocator || work->command_buffer != command_buffer) return false;

    bool good = r->core.EndCommandBuffer(command_buffer) == NriResult_SUCCESS;
    uint64_t value = 0u;

    if (good) {
        value = r->work_next_fence++;
        const NriFenceSubmitDesc signal = {.fence = r->work_fence, .value = value};
        NriFenceSubmitDesc upload_wait = {0};
        uint32_t upload_wait_num = 0u;

        if (r->upload && r->upload->fence && r->upload->next_fence_value > 1u) {
            upload_wait = (NriFenceSubmitDesc){.fence = r->upload->fence, .value = r->upload->next_fence_value - 1u, .stages = NriStageBits_ALL};
            upload_wait_num = 1u;
        }

        const NriQueueSubmitDesc submit = {.waitFences = upload_wait_num ? &upload_wait : NULL,
                                           .waitFenceNum = upload_wait_num,
                                           .commandBuffers = (const NriCommandBuffer *const *)&command_buffer,
                                           .commandBufferNum = 1u,
                                           .signalFences = &signal,
                                           .signalFenceNum = 1u};

        good = r->core.QueueSubmit(r->work_queue, &submit) == NriResult_SUCCESS;
    }

    if (good) {
        work->fence_value = value;
        r->work_index++;
    }

    r->active_work = NULL;
    r->active_frame = NULL;

    if (good && wait) r->core.Wait(r->work_fence, value);

    return good;
}

static void abort_work_commands(renderer *r, NriCommandAllocator *allocator, NriCommandBuffer *command_buffer) {
    if (!r) return;

    frame_context *work = r->active_work;

    if (!work || work->allocator != allocator || work->command_buffer != command_buffer) return;

    if (work->command_buffer) r->core.DestroyCommandBuffer(work->command_buffer);

    if (work->allocator) r->core.DestroyCommandAllocator(work->allocator);

    work->command_buffer = NULL;
    work->allocator = NULL;
    clear_frame_temporary(r, work);
    r->active_work = NULL;
    r->active_frame = NULL;

    if (r->core.CreateCommandAllocator(r->work_queue, &work->allocator) == NriResult_SUCCESS) r->core.CreateCommandBuffer(work->allocator, &work->command_buffer);
}

static bool create_frame_contexts(renderer *r) {
    r->frame_contexts = calloc(FRAME_QUEUE_DEPTH, sizeof(*r->frame_contexts));

    if (!r->frame_contexts) return false;

    if (r->core.CreateFence(r->device, 0, &r->frame_fence) != NriResult_SUCCESS) return false;

    for (uint32_t i = 0; i < FRAME_QUEUE_DEPTH; ++i) {
        FRAME_CONTEXT *frame = &r->frame_contexts[i];

        if (!create_descriptor_pool_object(r, &frame->descriptor_pool) || !create_uniform_ring(r, frame) ||
            r->core.CreateCommandAllocator(r->graphics_queue, &frame->allocator) != NriResult_SUCCESS ||
            r->core.CreateCommandBuffer(frame->allocator, &frame->command_buffer) != NriResult_SUCCESS)
            return false;
    }

    return true;
}

static void destroy_frame_contexts(RENDERER *r) {
    if (r->frame_contexts) {
        for (uint32_t i = 0; i < FRAME_QUEUE_DEPTH; ++i) {
            FRAME_CONTEXT *frame = &r->frame_contexts[i];
            clear_frame_temporary(r, frame);

            if (frame->command_buffer) r->core.DestroyCommandBuffer(frame->command_buffer);

            if (frame->allocator) r->core.DestroyCommandAllocator(frame->allocator);

            if (frame->descriptor_pool) r->core.DestroyDescriptorPool(frame->descriptor_pool);

            if (frame->uniform_buffer) r->core.DestroyBuffer(frame->uniform_buffer);
            free(frame->temporary_descriptors);
            free(frame->temporary_buffers);
        }

        free(r->frame_contexts);
    }

    r->frame_contexts = NULL;
    r->active_frame = NULL;

    if (r->frame_fence) r->core.DestroyFence(r->frame_fence);
    r->frame_fence = NULL;
}

static bool begin_frame_commands(RENDERER *r, FRAME_CONTEXT **out_frame, NriCommandBuffer **out_command_buffer) {
    if (!r || !r->frame_contexts || !r->frame_fence || !out_frame || !out_command_buffer) return false;

    const uint64_t wait_value = r->frame_index >= FRAME_QUEUE_DEPTH ? 1u + r->frame_index - FRAME_QUEUE_DEPTH : 0u;

    r->core.Wait(r->frame_fence, wait_value);

    FRAME_CONTEXT *frame = &r->frame_contexts[r->frame_index % FRAME_QUEUE_DEPTH];
    clear_frame_temporary(r, frame);
    r->core.ResetCommandAllocator(frame->allocator);
    r->current_graphics_layout = r->current_compute_layout = NULL;
    r->active_frame = frame;

    if (r->core.BeginCommandBuffer(frame->command_buffer, frame->descriptor_pool) != NriResult_SUCCESS) {
        r->active_frame = NULL;

        return false;
    }

    *out_frame = frame;
    *out_command_buffer = frame->command_buffer;
    return true;
}

static void abort_frame_commands(RENDERER *r, FRAME_CONTEXT *frame) {
    if (!r || !frame) return;

    if (frame->command_buffer) r->core.DestroyCommandBuffer(frame->command_buffer);

    if (frame->allocator) r->core.DestroyCommandAllocator(frame->allocator);
    frame->command_buffer = NULL;
    frame->allocator = NULL;
    r->active_frame = NULL;
    clear_frame_temporary(r, frame);

    if (r->core.CreateCommandAllocator(r->graphics_queue, &frame->allocator) == NriResult_SUCCESS) r->core.CreateCommandBuffer(frame->allocator, &frame->command_buffer);
}

static NriDescriptor *create_texture_view(RENDERER *r, NriTexture *texture, NriTextureView type) {
    if (!texture) return NULL;

    NriDescriptor *view = NULL;

    const NriTextureViewDesc desc = {
        .texture = texture,
        .type = type,
        .format = r->core.GetTextureDesc(texture)->format,
        .mipNum = 1,
        .layerNum = 1,
        .sliceNum = 1
    };

    if (r->core.CreateTextureView(&desc, &view) != NriResult_SUCCESS) return NULL;

    return track_descriptor(r, view) ? view : NULL;
}

static NriDescriptor *create_buffer_view(RENDERER *r, NriBuffer *buffer, NriBufferView type, uint32_t stride) {
    if (!buffer) return NULL;

    NriDescriptor *view = NULL;

    const NriBufferViewDesc desc = {
        .buffer = buffer,
        .type = type,
        .offset = 0,
        .size = r->core.GetBufferDesc(buffer)->size,
        .structureStride = stride
    };

    if (r->core.CreateBufferView(&desc, &view) != NriResult_SUCCESS) return NULL;

    return track_descriptor(r, view) ? view : NULL;
}

static NriDescriptor *uniform_view(renderer *r, const void *data, size_t size) {
    if (!r || !data || !size) return NULL;

    frame_context *context = r->active_frame;
    const NriDeviceDesc *device = r->core.GetDeviceDesc(r->device);
    uint64_t alignment = device ? device->memoryAlignment.constantBufferOffset : 256u;

    if (alignment < 16u) alignment = 16u;

    const uint64_t view_size = ((uint64_t)size + alignment - 1u) / alignment * alignment;

    if (context && context->uniform_buffer) {
        const uint64_t offset = (context->uniform_offset + alignment - 1u) / alignment * alignment;

        if (offset + view_size <= UNIFORM_RING_BYTES) {
            void *mapped = r->core.MapBuffer(context->uniform_buffer, offset, view_size);

            if (mapped) {
                memset(mapped, 0, view_size);
                memcpy(mapped, data, size);
                r->core.UnmapBuffer(context->uniform_buffer);

                NriDescriptor *view = NULL;
                const NriBufferViewDesc desc = {.buffer = context->uniform_buffer, .type = NriBufferView_CONSTANT_BUFFER, .offset = offset, .size = view_size};

                if (r->core.CreateBufferView(&desc, &view) == NriResult_SUCCESS && track_descriptor(r, view)) {
                    context->uniform_offset = offset + view_size;

                    return view;
                }
            }
        }
    }

    const NriBufferDesc desc = {.size = view_size, .usage = NriBufferUsageBits_CONSTANT};
    NriBuffer *buffer = NULL;

    if (r->core.CreateCommittedBuffer(r->device, NriMemoryLocation_HOST_UPLOAD, 0.0f, &desc, &buffer) != NriResult_SUCCESS) return NULL;

    if (!track_buffer(r, buffer)) return NULL;

    void *mapped = r->core.MapBuffer(buffer, 0, desc.size);

    if (!mapped) return NULL;
    memset(mapped, 0, desc.size);
    memcpy(mapped, data, size);
    r->core.UnmapBuffer(buffer);

    return create_buffer_view(r, buffer, NriBufferView_CONSTANT_BUFFER, 0);
}

static bool
bind_descriptor_set(RENDERER *r, NriCommandBuffer *cmd, NriPipelineLayout *layout, NriBindPoint point, uint32_t set_index, NriDescriptor *const *descriptors, uint32_t count) {
    NriDescriptorSet *set = NULL;
    NriDescriptorPool *pool = r->active_frame ? r->active_frame->descriptor_pool : r->descriptor_pool;

    if (r->core.AllocateDescriptorSets(pool, layout, set_index, &set, 1, 0) != NriResult_SUCCESS) return false;

    for (uint32_t i = 0; i < count; ++i) {
        if (!descriptors[i]) return false;

        const NriDescriptor *d = descriptors[i];

        const NriUpdateDescriptorRangeDesc update = {
            .descriptorSet = set,
            .rangeIndex = i,
            .descriptors = &d,
            .descriptorNum = 1
        };

        r->core.UpdateDescriptorRanges(&update, 1);
    }

    NriPipelineLayout **current = point == NriBindPoint_GRAPHICS ? &r->current_graphics_layout : &r->current_compute_layout;

    if (*current != layout) {
        r->core.CmdSetPipelineLayout(cmd, point, layout);
        *current = layout;
    }

    r->core.CmdSetDescriptorSet(cmd, &(NriSetDescriptorSetDesc){
        .setIndex = set_index,
        .descriptorSet = set,
        .bindPoint = point
    });

    return true;
}

static bool bind_uniform_data(RENDERER *r, NriCommandBuffer *cmd, NriPipelineLayout *layout, NriBindPoint point, uint32_t set, const void *data, size_t size) {
    NriDescriptor *view = uniform_view(r, data, size);

    return view && bind_descriptor_set(r, cmd, layout, point, set, &view, 1);
}

static bool transition_texture(RENDERER *r, NriCommandBuffer *cmd, NriTexture *texture, NriAccessBits access, NriLayout layout, NriStageBits stages);

static bool bind_bake_resources_ex(renderer *r, NriCommandBuffer *cmd, NriTexture *source, NriTexture *destination, NriBuffer *active_in, NriBuffer *active_count,
                                   NriBuffer *active_out, NriBuffer *active_out_count, const bake_uniforms *uniforms, size_t size) {
    if (!r || !cmd || !source || !destination || !r->lightmap_sampler || !r->bvh_node_buffer || !r->bvh_triangle_buffer || !r->lightmap_sample_buffer ||
        !r->lightmap_probe_buffer || !active_in || !active_count || !active_out || !active_out_count)
        return false;

    NriTexture *direct = r->lightmap_direct && r->lightmap_direct != destination ? r->lightmap_direct : source;

    if (!transition_texture(r, cmd, source, NriAccessBits_SHADER_RESOURCE, NriLayout_SHADER_RESOURCE, NriStageBits_COMPUTE_SHADER) ||
        (direct != source && !transition_texture(r, cmd, direct, NriAccessBits_SHADER_RESOURCE, NriLayout_SHADER_RESOURCE, NriStageBits_COMPUTE_SHADER)) ||
        !transition_texture(r, cmd, destination, NriAccessBits_SHADER_RESOURCE_STORAGE, NriLayout_SHADER_RESOURCE_STORAGE, NriStageBits_COMPUTE_SHADER))
        return false;

    NriBuffer *patch_map = r->lightmap_patch_map_buffer ? r->lightmap_patch_map_buffer : r->lightmap_sample_buffer;

    NriBuffer *patch_anchors = r->lightmap_patch_anchor_buffer ? r->lightmap_patch_anchor_buffer : r->lightmap_sample_buffer;

    NriDescriptor *src[] = {create_texture_view(r, source, NriTextureView_TEXTURE),
                            r->lightmap_sampler,
                            create_texture_view(r, direct, NriTextureView_TEXTURE),
                            r->lightmap_sampler,
                            create_buffer_view(r, r->bvh_node_buffer, NriBufferView_STRUCTURED_BUFFER, sizeof(BVH_NODE)),
                            create_buffer_view(r, r->bvh_triangle_buffer, NriBufferView_STRUCTURED_BUFFER, sizeof(BVH_TRIANGLE)),
                            create_buffer_view(r, r->lightmap_sample_buffer, NriBufferView_STRUCTURED_BUFFER, sizeof(LMAP_SAMPLE)),
                            create_buffer_view(r, r->lightmap_probe_buffer, NriBufferView_STRUCTURED_BUFFER, sizeof(PROBE)),
                            create_buffer_view(r, patch_map, NriBufferView_STRUCTURED_BUFFER, sizeof(Uint32)),
                            create_buffer_view(r, patch_anchors, NriBufferView_STRUCTURED_BUFFER, sizeof(Uint32[4])),
                            create_buffer_view(r, active_in, NriBufferView_STRUCTURED_BUFFER, sizeof(Uint32)),
                            create_buffer_view(r, active_count, NriBufferView_STRUCTURED_BUFFER, sizeof(Uint32))};

    NriDescriptor *dst[] = {create_texture_view(r, destination, NriTextureView_STORAGE_TEXTURE),
                            create_buffer_view(r, active_out, NriBufferView_STORAGE_STRUCTURED_BUFFER, sizeof(Uint32)),
                            create_buffer_view(r, active_out_count, NriBufferView_STORAGE_STRUCTURED_BUFFER, sizeof(Uint32))};

    return bind_descriptor_set(r, cmd, r->bake_layout, NriBindPoint_COMPUTE, 0, src, 12) && bind_descriptor_set(r, cmd, r->bake_layout, NriBindPoint_COMPUTE, 1, dst, 3) &&
           bind_uniform_data(r, cmd, r->bake_layout, NriBindPoint_COMPUTE, 2, uniforms, size);
}

static bool bind_bake_resources(renderer *r, NriCommandBuffer *cmd, NriTexture *source, NriTexture *destination, const bake_uniforms *uniforms, size_t size) {
    return bind_bake_resources_ex(r, cmd, source, destination, r->lightmap_active_buffer[0], r->lightmap_active_count[0], r->lightmap_active_buffer[1], r->lightmap_active_count[1],
                                  uniforms, size);
}

static bool bind_probe_resources(renderer *r, NriCommandBuffer *cmd, NriBuffer *input, NriBuffer *nodes, NriBuffer *triangles, NriBuffer *output, const bake_uniforms *uniforms,
                                 size_t size) {
    const NriBufferBarrierDesc barrier = {.buffer = output, .after = {.access = NriAccessBits_SHADER_RESOURCE_STORAGE, .stages = NriStageBits_COMPUTE_SHADER}};
    r->core.CmdBarrier(cmd, &(NriBarrierDesc){.buffers = &barrier, .bufferNum = 1});

    r->core.CmdBarrier(cmd, &(NriBarrierDesc){
        .buffers = &barrier,
        .bufferNum = 1
    });

    NriDescriptor *src[] = {
        create_buffer_view(r, input, NriBufferView_STRUCTURED_BUFFER, sizeof(float[4])),
        create_buffer_view(r, nodes, NriBufferView_STRUCTURED_BUFFER, sizeof(BVH_NODE)),
        create_buffer_view(r, triangles, NriBufferView_STRUCTURED_BUFFER, sizeof(BVH_TRIANGLE))
    };

    NriDescriptor *dst = create_buffer_view(r, output, NriBufferView_STORAGE_STRUCTURED_BUFFER, sizeof(float[4]));

    return bind_descriptor_set(r, cmd, r->probe_layout, NriBindPoint_COMPUTE, 0, src, 3) && bind_descriptor_set(r, cmd, r->probe_layout, NriBindPoint_COMPUTE, 1, &dst, 1) &&
           bind_uniform_data(r, cmd, r->probe_layout, NriBindPoint_COMPUTE, 2, uniforms, size);
}

static bool read_buffer(RENDERER *r, NriBuffer *output, PROBE_GRID *grid, uint32_t bytes, uint32_t count) {
    const NriBufferDesc desc = {
        .size = bytes
    };

    NriBuffer *staging = NULL;

    if (r->core.CreateCommittedBuffer(r->device, NriMemoryLocation_HOST_READBACK, 1.0f, &desc, &staging) != NriResult_SUCCESS) return false;

    NriCommandAllocator *allocator = NULL;
    NriCommandBuffer *cmd = NULL;
    bool good = false;

    if (begin_commands(r, &allocator, &cmd) == NriResult_SUCCESS) {
        const NriBufferBarrierDesc barrier = {
            .buffer = output,
            .before = {
                .access = NriAccessBits_SHADER_RESOURCE_STORAGE,
                .stages = NriStageBits_COMPUTE_SHADER
            },
            .after = {
                .access = NriAccessBits_COPY_SOURCE,
                .stages = NriStageBits_COPY
            }
        };

        r->core.CmdBarrier(cmd, &(NriBarrierDesc){
            .buffers = &barrier,
            .bufferNum = 1
        });
        r->core.CmdCopyBuffer(cmd, staging, 0, output, 0, bytes);

        good = submit_commands(r, allocator, cmd);
    }

    if (good) {
        const float (*values)[4] = r->core.MapBuffer(staging, 0, bytes);

        good = values != NULL;

        if (good) {
            for (uint32_t i = 0; i < count; ++i) {
                for (uint32_t j = 0; j < 9; ++j)
                    memcpy(grid->probes[i].coefficients[j], values[i * 9u + j], sizeof(float[4]));
                grid->probes[i].position[3] = values[i * 9u][3];
            }

            r->core.UnmapBuffer(staging);
        }
    }

    r->core.DestroyBuffer(staging);

    return good;
}

static bool bind_fx_resources(
    RENDERER *r, NriCommandBuffer *cmd, NriPipeline *pipeline, NriTexture *source, NriTexture *destination, NriDescriptor *sampler_desc, const void *uniforms, uint32_t size
) {
    if ((source && !transition_texture(r, cmd, source, NriAccessBits_SHADER_RESOURCE, NriLayout_SHADER_RESOURCE, NriStageBits_COMPUTE_SHADER)) ||
        !transition_texture(r, cmd, destination, NriAccessBits_SHADER_RESOURCE_STORAGE, NriLayout_SHADER_RESOURCE_STORAGE, NriStageBits_COMPUTE_SHADER))
        return false;

    NriPipelineLayout *layout = pipeline == r->fx.grade_pipeline ? r->grade_layout : pipeline == r->fx.bloom_pipeline ? r->bloom_layout : r->ssao_layout;

    NriDescriptor *dst = create_texture_view(r, destination, NriTextureView_STORAGE_TEXTURE);

    if (source) {
        NriDescriptor *src[] = {create_texture_view(r, source, NriTextureView_TEXTURE), sampler_desc};

        if (!bind_descriptor_set(r, cmd, layout, NriBindPoint_COMPUTE, 0, src, 2)) return false;
    }

    return bind_descriptor_set(r, cmd, layout, NriBindPoint_COMPUTE, 1, &dst, 1) && (!size || bind_uniform_data(r, cmd, layout, NriBindPoint_COMPUTE, 2, uniforms, size));
}

static bool bind_volume_resources(
    RENDERER *r,
    NriCommandBuffer *cmd,
    NriTexture *normal,
    NriDescriptor *sampler_desc,
    NriBuffer *probes,
    NriBuffer *beams,
    NriTexture *output,
    const void *uniforms,
    uint32_t size
) {
    if (!transition_texture(r, cmd, normal, NriAccessBits_SHADER_RESOURCE, NriLayout_SHADER_RESOURCE, NriStageBits_COMPUTE_SHADER) ||
        !transition_texture(r, cmd, output, NriAccessBits_SHADER_RESOURCE_STORAGE, NriLayout_SHADER_RESOURCE_STORAGE, NriStageBits_COMPUTE_SHADER))
        return false;

    NriDescriptor *src[] = {
        create_texture_view(r, normal, NriTextureView_TEXTURE),
        sampler_desc,
        create_buffer_view(r, probes, NriBufferView_STRUCTURED_BUFFER, sizeof(PROBE)),
        create_buffer_view(r, beams, NriBufferView_STRUCTURED_BUFFER, sizeof(float))
    };

    NriDescriptor *dst = create_texture_view(r, output, NriTextureView_STORAGE_TEXTURE);

    return bind_descriptor_set(r, cmd, r->volume_layout, NriBindPoint_COMPUTE, 0, src, 4) && bind_descriptor_set(r, cmd, r->volume_layout, NriBindPoint_COMPUTE, 1, &dst, 1) &&
           bind_uniform_data(r, cmd, r->volume_layout, NriBindPoint_COMPUTE, 2, uniforms, size);
}

static bool bind_volume_compose_resources(
    RENDERER *r,
    NriCommandBuffer *cmd,
    NriTexture *hdr,
    NriTexture *volume,
    NriTexture *normal,
    NriDescriptor *sampler_desc,
    NriDescriptor *depth_sampler,
    NriTexture *output,
    const void *uniforms,
    uint32_t size
) {
    NriTexture *sources[] = {hdr, volume, normal};

    for (uint32_t i = 0; i < 3; ++i)
        if (!transition_texture(r, cmd, sources[i], NriAccessBits_SHADER_RESOURCE, NriLayout_SHADER_RESOURCE, NriStageBits_COMPUTE_SHADER)) return false;

    if (!transition_texture(r, cmd, output, NriAccessBits_SHADER_RESOURCE_STORAGE, NriLayout_SHADER_RESOURCE_STORAGE, NriStageBits_COMPUTE_SHADER)) return false;

    NriDescriptor *src[] = {
        create_texture_view(r, hdr, NriTextureView_TEXTURE),
        create_texture_view(r, volume, NriTextureView_TEXTURE),
        create_texture_view(r, normal, NriTextureView_TEXTURE),
        sampler_desc,
        sampler_desc,
        depth_sampler
    };

    NriDescriptor *dst = create_texture_view(r, output, NriTextureView_STORAGE_TEXTURE);

    return bind_descriptor_set(r, cmd, r->volume_compose_layout, NriBindPoint_COMPUTE, 0, src, 6) &&
           bind_descriptor_set(r, cmd, r->volume_compose_layout, NriBindPoint_COMPUTE, 1, &dst, 1) &&
           bind_uniform_data(r, cmd, r->volume_compose_layout, NriBindPoint_COMPUTE, 2, uniforms, size);
}

static bool bind_sky_resources(RENDERER *r, NriCommandBuffer *cmd, const void *data, size_t size) {
    return bind_uniform_data(r, cmd, r->sky_layout, NriBindPoint_GRAPHICS, 3, data, size);
}

static bool bind_camera_resources(RENDERER *r, NriCommandBuffer *cmd, const void *data, size_t size) {
    return bind_uniform_data(r, cmd, r->surface_layout, NriBindPoint_GRAPHICS, 1, data, size);
}

static bool bind_surface_resources(
    RENDERER *r,
    NriCommandBuffer *cmd,
    const RENDER_MATERIAL *material,
    NriTexture *lightmap,
    NriDescriptor *material_sampler,
    NriDescriptor *lightmap_sampler,
    const void *uniforms,
    size_t size
) {
    NriDescriptor *src[] = {
        create_texture_view(r, material->base_color, NriTextureView_TEXTURE),
        create_texture_view(r, material->metallic_roughness, NriTextureView_TEXTURE),
        create_texture_view(r, material->normal, NriTextureView_TEXTURE),
        create_texture_view(r, material->occlusion, NriTextureView_TEXTURE),
        create_texture_view(r, material->emissive, NriTextureView_TEXTURE),
        create_texture_view(r, lightmap, NriTextureView_TEXTURE),
        material_sampler,
        material_sampler,
        material_sampler,
        material_sampler,
        material_sampler,
        lightmap_sampler
    };

    return bind_descriptor_set(r, cmd, r->surface_layout, NriBindPoint_GRAPHICS, 2, src, 12) &&
           bind_uniform_data(r, cmd, r->surface_layout, NriBindPoint_GRAPHICS, 3, uniforms, size);
}

static bool bind_line_resources(RENDERER *r, NriCommandBuffer *cmd, const void *data, size_t size) {
    return bind_uniform_data(r, cmd, r->line_layout, NriBindPoint_GRAPHICS, 1, data, size);
}

static void free_shader(NriShaderDesc *shader) {
    if (shader && shader->bytecode) SDL_free((void *)shader->bytecode);

    if (shader) *shader = (NriShaderDesc){0};
}

static TEXTURE_STATE *find_texture_state(RENDERER *r, NriTexture *texture);
static bool create_swapchain(RENDERER *r, uint32_t width, uint32_t height) {
    if (!r->window || !width || !height) return false;

    NriWindow window = {0};
    const SDL_PropertiesID props = SDL_GetWindowProperties(r->window);
#if defined(__APPLE__)
    (void)props;

    if (!r->metal_view) r->metal_view = SDL_Metal_CreateView(r->window);

    if (!r->metal_view) return false;
    window.metal.caMetalLayer = SDL_Metal_GetLayer(r->metal_view);
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
    const NriSwapChainDesc desc = {
        .window = window,
        .queue = r->graphics_queue,
        .width = (NriDim_t)width,
        .height = (NriDim_t)height,
        .textureNum = FRAME_QUEUE_DEPTH + 1u,
        .format = NriSwapChainFormat_BT709_G22_8BIT,
        .flags = NriSwapChainBits_VSYNC,
        .queuedFrameNum = FRAME_QUEUE_DEPTH
    };

    if (r->swapchain_api.CreateSwapChain(r->device, &desc, &r->swapchain) != NriResult_SUCCESS) return false;

    uint32_t count = 0;

    NriTexture *const *textures = r->swapchain_api.GetSwapChainTextures(r->swapchain, &count);

    if (!textures || !count) return false;
    r->swapchain_textures = calloc(count, sizeof(*r->swapchain_textures));
    r->swapchain_frames = calloc(count, sizeof(*r->swapchain_frames));

    if (!r->swapchain_textures || !r->swapchain_frames) return false;
    r->swapchain_texture_count = count;
    r->swapchain_format = r->core.GetTextureDesc(textures[0])->format;
    r->swapchain_width = width;
    r->swapchain_height = height;

    for (uint32_t i = 0; i < count; ++i) {
        r->swapchain_textures[i] = textures[i];

        TEXTURE_STATE *state = find_texture_state(r, textures[i]);

        if (!state) return false;
        state->state = (NriAccessLayoutStage){
            .layout = NriLayout_UNDEFINED,
            .stages = NriStageBits_NONE
        };

        SWAPCHAIN_TEXTURE *frame = &r->swapchain_frames[i];
        frame->texture = textures[i];

        const NriTextureViewDesc view = {
            .texture = textures[i],
            .type = NriTextureView_COLOR_ATTACHMENT,
            .format = r->swapchain_format,
            .mipNum = 1,
            .layerNum = 1,
            .sliceNum = 1
        };

        if (r->core.CreateTextureView(&view, &frame->color_attachment) != NriResult_SUCCESS ||
            r->core.CreateFence(r->device, NRI_SWAPCHAIN_SEMAPHORE, &frame->acquire) != NriResult_SUCCESS ||
            r->core.CreateFence(r->device, NRI_SWAPCHAIN_SEMAPHORE, &frame->release) != NriResult_SUCCESS)
            return false;
    }

    return true;
}

static void destroy_swapchain(RENDERER *r) {
    for (uint32_t i = 0; i < r->swapchain_texture_count; ++i) {
        for (uint32_t j = 0; j < r->texture_state_num; ++j)
            if (r->texture_states[j].texture == r->swapchain_textures[i]) {
                r->texture_states[j] = r->texture_states[--r->texture_state_num];

                break;
            }

        SWAPCHAIN_TEXTURE *frame = &r->swapchain_frames[i];

        if (frame->color_attachment) r->core.DestroyDescriptor(frame->color_attachment);

        if (frame->acquire) r->core.DestroyFence(frame->acquire);

        if (frame->release) r->core.DestroyFence(frame->release);
    }

    free(r->swapchain_frames);
    free(r->swapchain_textures);
    r->swapchain_frames = NULL;
    r->swapchain_textures = NULL;
    r->swapchain_texture_count = 0;

    if (r->swapchain) r->swapchain_api.DestroySwapChain(r->swapchain);
    r->swapchain = NULL;
}

static bool acquire_swapchain_texture(RENDERER *r, uint32_t *index) {
    NriFence *acquire = r->swapchain_frames[r->frame_index % r->swapchain_texture_count].acquire;

    NriResult result = r->swapchain_api.AcquireNextTexture(r->swapchain, acquire, index);

    return result == NriResult_SUCCESS && *index < r->swapchain_texture_count;
}

static TEXTURE_STATE *find_texture_state(RENDERER *r, NriTexture *texture) {
    for (uint32_t i = 0; i < r->texture_state_num; ++i)
        if (r->texture_states[i].texture == texture) return &r->texture_states[i];

    if (r->texture_state_num == r->texture_state_cap) {
        uint32_t cap = r->texture_state_cap ? r->texture_state_cap * 2 : 32;
        TEXTURE_STATE *items = realloc(r->texture_states, cap * sizeof(*items));

        if (!items) return NULL;
        r->texture_states = items;
        r->texture_state_cap = cap;
    }

    TEXTURE_STATE *item = &r->texture_states[r->texture_state_num++];
    *item = (TEXTURE_STATE){
        .texture = texture
    };

    return item;
}

static bool texture_barrier(RENDERER *r, NriCommandBuffer *cmd, NriTexture *texture, NriAccessLayoutStage before, NriAccessLayoutStage after) {
    TEXTURE_STATE *item = find_texture_state(r, texture);

    if (!item) return false;

    if (item->state.layout) before = item->state;

    const NriTextureBarrierDesc barrier = {
        .texture = texture,
        .before = before,
        .after = after,
        .mipNum = 1,
        .layerNum = 1
    };

    r->core.CmdBarrier(cmd, &(NriBarrierDesc){
        .textures = &barrier,
        .textureNum = 1
    });
    item->state = after;

    return true;
}

static bool transition_texture(RENDERER *r, NriCommandBuffer *cmd, NriTexture *texture, NriAccessBits access, NriLayout layout, NriStageBits stages) {
    if (!texture) return false;

    return texture_barrier(r, cmd, texture, (NriAccessLayoutStage){0}, (NriAccessLayoutStage){
        .access = access,
        .layout = layout,
        .stages = stages
    });
}

static bool begin_scene_rendering(RENDERER *r, NriCommandBuffer *cmd, NriTexture *hdr, NriTexture *normal, NriTexture *depth, uint32_t width, uint32_t height) {
    NriDescriptor *hdr_view = create_texture_view(r, hdr, NriTextureView_COLOR_ATTACHMENT);

    NriDescriptor *normal_view = create_texture_view(r, normal, NriTextureView_COLOR_ATTACHMENT);

    NriDescriptor *depth_view = create_texture_view(r, depth, NriTextureView_DEPTH_STENCIL_ATTACHMENT);

    if (!hdr_view || !normal_view || !depth_view) return false;

    const NriAccessLayoutStage color = {NriAccessBits_COLOR_ATTACHMENT, NriLayout_COLOR_ATTACHMENT, NriStageBits_COLOR_ATTACHMENT};

    const NriAccessLayoutStage depth_state = {NriAccessBits_DEPTH_STENCIL_ATTACHMENT, NriLayout_DEPTH_STENCIL_ATTACHMENT, NriStageBits_DEPTH_STENCIL_ATTACHMENT};

    if (!texture_barrier(r, cmd, hdr, (NriAccessLayoutStage){0}, color) || !texture_barrier(r, cmd, normal, (NriAccessLayoutStage){0}, color) ||
        !texture_barrier(r, cmd, depth, (NriAccessLayoutStage){0}, depth_state))
        return false;

    const NriAttachmentDesc colors[2] = {
        {
            .descriptor = hdr_view,
            .loadOp = NriLoadOp_CLEAR,
            .storeOp = NriStoreOp_STORE
        }, {
            .descriptor = normal_view,
            .loadOp = NriLoadOp_CLEAR,
            .storeOp = NriStoreOp_STORE
        }
    };

    const NriRenderingDesc desc = {
        .colors = colors,
        .colorNum = 2,
        .depth = {
            .descriptor = depth_view,
            .loadOp = NriLoadOp_CLEAR,
            .storeOp = NriStoreOp_STORE,
            .clearValue = {
                .depthStencil = {
                    .depth = 1.0f
                }
            }
        }
    };

    r->core.CmdSetViewports(cmd, &(NriViewport){
        .width = (float)width,
        .height = (float)height,
        .depthMax = 1.0f
    }, 1);
    r->core.CmdSetScissors(cmd, &(NriRect){
        .width = (NriDim_t)width,
        .height = (NriDim_t)height
    }, 1);
    r->core.CmdBeginRendering(cmd, &desc);

    return true;
}

static bool begin_compose_rendering(
    RENDERER *r,
    NriCommandBuffer *cmd,
    NriTexture *swap,
    NriTexture *hdr,
    NriTexture *ao,
    NriTexture *bloom,
    NriTexture *lut,
    NriDescriptor *sampler_desc,
    const void *uniforms,
    size_t size
) {
    NriTexture *sources[] = {hdr, ao, bloom, lut};

    for (uint32_t i = 0; i < 4; ++i)
        if (!transition_texture(r, cmd, sources[i], NriAccessBits_SHADER_RESOURCE, NriLayout_SHADER_RESOURCE, NriStageBits_FRAGMENT_SHADER)) return false;

    NriDescriptor *src[] = {
        create_texture_view(r, hdr, NriTextureView_TEXTURE),
        create_texture_view(r, ao, NriTextureView_TEXTURE),
        create_texture_view(r, bloom, NriTextureView_TEXTURE),
        create_texture_view(r, lut, NriTextureView_TEXTURE),
        sampler_desc,
        sampler_desc,
        sampler_desc,
        sampler_desc
    };

    if (!bind_descriptor_set(r, cmd, r->compose_layout, NriBindPoint_GRAPHICS, 2, src, 8) ||
        !bind_uniform_data(r, cmd, r->compose_layout, NriBindPoint_GRAPHICS, 3, uniforms, size))
        return false;

    if (!texture_barrier(
            r,
            cmd,
            swap,
            (NriAccessLayoutStage){
                .layout = NriLayout_UNDEFINED,
                .stages = NriStageBits_NONE
            },
            (NriAccessLayoutStage){
                .access = NriAccessBits_COLOR_ATTACHMENT,
                .layout = NriLayout_COLOR_ATTACHMENT,
                .stages = NriStageBits_COLOR_ATTACHMENT
            }
        ))
        return false;

    const NriAttachmentDesc color = {
        .descriptor = r->swapchain_frames[r->current_swap_index].color_attachment,
        .loadOp = NriLoadOp_CLEAR,
        .storeOp = NriStoreOp_STORE
    };

    const NriRenderingDesc desc = {
        .colors = &color,
        .colorNum = 1
    };

    r->core.CmdBeginRendering(cmd, &desc);

    return true;
}

static bool submit_frame(RENDERER *r, FRAME_CONTEXT *frame, NriCommandBuffer *cmd, uint32_t index) {
    bool good = frame && texture_barrier(
                             r,
                             cmd,
                             r->swapchain_textures[index],
                             (NriAccessLayoutStage){
                                 .access = NriAccessBits_COLOR_ATTACHMENT,
                                 .layout = NriLayout_COLOR_ATTACHMENT,
                                 .stages = NriStageBits_COLOR_ATTACHMENT
                             },
                             (NriAccessLayoutStage){
                                 .layout = NriLayout_PRESENT,
                                 .stages = NriStageBits_NONE
                             }
                         );
    if (good) good = r->core.EndCommandBuffer(cmd) == NriResult_SUCCESS;

    const uint64_t frame_value = 1u + r->frame_index;
    NriFenceSubmitDesc waits[2] = {{.fence = r->swapchain_frames[r->frame_index % r->swapchain_texture_count].acquire, .stages = NriStageBits_COLOR_ATTACHMENT}, {0}};
    uint32_t wait_num = 1u;

    if (r->upload && r->upload->fence && r->upload->next_fence_value > 1u) {
        waits[wait_num++] = (NriFenceSubmitDesc){.fence = r->upload->fence, .value = r->upload->next_fence_value - 1u, .stages = NriStageBits_ALL};
    }

    const NriFenceSubmitDesc wait = {
        .fence = r->swapchain_frames[r->frame_index % r->swapchain_texture_count].acquire,
        .stages = NriStageBits_COLOR_ATTACHMENT
    };

    const NriFenceSubmitDesc signals[2] = {{
    .fence = r->swapchain_frames[index].release
}, {
    .fence = r->frame_fence,
    .value = frame_value
}};

    const NriQueueSubmitDesc submit = {.waitFences = waits,
                                       .waitFenceNum = wait_num,
                                       .commandBuffers = (const NriCommandBuffer *const *)&cmd,
                                       .commandBufferNum = 1,
                                       .signalFences = signals,
                                       .signalFenceNum = 2};

    bool submitted = false;

    if (good) {
        submitted = r->core.QueueSubmit(r->graphics_queue, &submit) == NriResult_SUCCESS;

        good = submitted;
    }

    if (good) good = r->swapchain_api.QueuePresent(r->swapchain, r->swapchain_frames[index].release, frame_value) == NriResult_SUCCESS;

    r->active_frame = NULL;

    if (submitted) r->frame_index++;
    else if (!good) abort_frame_commands(r, frame);

    return good;
}

static NriDescriptor *sampler(RENDERER *r, NriFilter min_filter, NriFilter mag_filter, NriAddressMode address) {
    const NriSamplerDesc desc = {
        .filters = {
            .min = min_filter,
            .mag = mag_filter,
            .mip = NriFilter_NEAREST
        },
        .addressModes = {address, address, address},
        .mipMin = 0.0f,
        .mipMax = 16.0f
    };

    NriDescriptor *result = NULL;

    if (r->core.CreateSampler(r->device, &desc, &result) != NriResult_SUCCESS) return NULL;

    return result;
}

static NriPipeline *make_line_pipeline(RENDERER *r, NriPipelineLayout *layout, const NriShaderDesc *vs, const NriShaderDesc *ps) {
    const NriVertexStreamDesc vb = {
        .bindingSlot = 0,
        .stepRate = NriVertexStreamStepRate_PER_VERTEX,
        .stride = (uint16_t)sizeof(RENDER_VERTEX)
    };

    const NriVertexAttributeDesc attrs[2] = {
        {
            .d3d = {
                .semanticName = "TEXCOORD",
                .semanticIndex = 0
            },
            .vk = {
                .location = 0
            },
            .offset = (uint32_t)offsetof(RENDER_VERTEX, x),
            .format = NriFormat_RGB32_SFLOAT,
            .streamIndex = 0
        },
        {
            .d3d = {
                .semanticName = "TEXCOORD",
                .semanticIndex = 1
            },
            .vk = {
                .location = 1
            },
            .offset = (uint32_t)offsetof(RENDER_VERTEX, r),
            .format = NriFormat_RGBA32_SFLOAT,
            .streamIndex = 0
        }
    };

    const NriVertexInputDesc vertex_input = {
        .attributes = attrs,
        .attributeNum = 2,
        .streams = &vb,
        .streamNum = 1
    };

    const NriColorAttachmentDesc targets[2] = {
        {
            .format = NriFormat_RGBA16_SFLOAT,
            .colorWriteMask = NriColorWriteBits_RGBA
        }, {
            .format = NriFormat_RGBA16_SFLOAT,
            .colorWriteMask = NriColorWriteBits_NONE
        }
    };

    const NriMultisampleDesc multisample = {
        .sampleMask = NRI_ALL,
        .sampleNum = 1
    };

    const NriShaderDesc shaders[2] = {*vs, *ps};

    const NriGraphicsPipelineDesc desc = {
        .pipelineLayout = layout,
        .vertexInput = &vertex_input,
        .inputAssembly = {
            .topology = NriTopology_LINE_LIST
        },
        .rasterization = {
            .fillMode = NriFillMode_SOLID,
            .cullMode = NriCullMode_NONE,
            .frontCounterClockwise = true,
            .depthClamp = false
        },
        .multisample = &multisample,
        .outputMerger = {
            .colors = targets,
            .colorNum = 2,
            .depth = {
                .compareOp = NriCompareOp_LESS_EQUAL,
                .write = false
            },
            .depthStencilFormat = r->depth_format
        },
        .shaders = shaders,
        .shaderNum = 2,
        .cache = r->pipeline_cache};

    NriPipeline *pipeline = NULL;

    if (r->core.CreateGraphicsPipeline(r->device, &desc, &pipeline) != NriResult_SUCCESS) return NULL;

    return pipeline;
}

static NriPipeline *make_sky_pipeline(RENDERER *r, NriPipelineLayout *layout, const NriShaderDesc *vs, const NriShaderDesc *ps) {
    const NriColorAttachmentDesc targets[2] = {
        {
            .format = NriFormat_RGBA16_SFLOAT,
            .colorWriteMask = NriColorWriteBits_RGBA
        }, {
            .format = NriFormat_RGBA16_SFLOAT,
            .colorWriteMask = NriColorWriteBits_RGBA
        }
    };

    const NriMultisampleDesc multisample = {
        .sampleMask = NRI_ALL,
        .sampleNum = 1
    };

    const NriShaderDesc shaders[2] = {*vs, *ps};
    const NriGraphicsPipelineDesc desc = {.pipelineLayout = layout,
                                          .inputAssembly = {.topology = NriTopology_TRIANGLE_LIST},
                                          .rasterization = {.fillMode = NriFillMode_SOLID, .cullMode = NriCullMode_NONE, .frontCounterClockwise = true, .depthClamp = false},
                                          .multisample = &multisample,
                                          .outputMerger = {.colors = targets, .colorNum = 2, .depthStencilFormat = r->depth_format},
                                          .shaders = shaders,
                                          .shaderNum = 2,
                                          .cache = r->pipeline_cache};

    NriPipeline *pipeline = NULL;

    if (r->core.CreateGraphicsPipeline(r->device, &desc, &pipeline) != NriResult_SUCCESS) return NULL;

    return pipeline;
}

static NriTexture *create_texture(RENDERER *r, NriFormat format, NriTextureUsageBits usage, Uint32 width, Uint32 height) {
    if (!r || !r->device || !width || !height) return NULL;

    const NriTextureDesc desc = {
        .type = NriTextureType_TEXTURE_2D,
        .usage = usage,
        .format = format,
        .width = (NriDim_t)width,
        .height = (NriDim_t)height,
        .depth = 1,
        .mipNum = 1,
        .layerNum = 1,
        .sampleNum = 1
    };

    NriTexture *result = NULL;

    if (r->core.CreateCommittedTexture(r->device, NriMemoryLocation_DEVICE, 1.0f, &desc, &result) != NriResult_SUCCESS) return NULL;

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
    if (!r || !r->upload) return;

    UPLOAD_CONTEXT *upload = r->upload;

    for (uint32_t i = 0; i < UPLOAD_RING_SIZE; ++i) {
        UPLOAD_SLOT *slot = &upload->slots[i];

        if (slot->command_buffer) r->core.DestroyCommandBuffer(slot->command_buffer);

        if (slot->allocator) r->core.DestroyCommandAllocator(slot->allocator);

        if (slot->staging) r->core.DestroyBuffer(slot->staging);
    }

    if (upload->fence) r->core.DestroyFence(upload->fence);

    free(upload);
    r->upload = NULL;
}

static bool create_upload_context(RENDERER *r) {
    if (!r || !r->device || !r->graphics_queue) return false;

    if (r->upload) return true;

    UPLOAD_CONTEXT *upload = calloc(1, sizeof(*upload));

    if (!upload) return false;
    r->upload = upload;

    if (r->core.CreateFence(r->device, 0u, &upload->fence) != NriResult_SUCCESS) goto fail;

    const NriBufferDesc staging_desc = {
        .size = UPLOAD_CHUNK_BYTES,
        .usage = NriBufferUsageBits_NONE
    };

    for (uint32_t i = 0; i < UPLOAD_RING_SIZE; ++i) {
        UPLOAD_SLOT *slot = &upload->slots[i];

        if (r->core.CreateCommittedBuffer(r->device, NriMemoryLocation_HOST_UPLOAD, 0.0f, &staging_desc, &slot->staging) != NriResult_SUCCESS ||
            r->core.CreateCommandAllocator(r->graphics_queue, &slot->allocator) != NriResult_SUCCESS ||
            r->core.CreateCommandBuffer(slot->allocator, &slot->command_buffer) != NriResult_SUCCESS)
            goto fail;
    }

    upload->next_fence_value = 1u;

    return true;

fail:
    destroy_upload_context(r);

    return false;
}

static bool upload_wait_slot(RENDERER *r, UPLOAD_SLOT *slot) {
    if (!r || !r->upload || !slot || !slot->fence_value) return true;

    const Uint64 started = SDL_GetTicks();
    bool logged = false;

    while (r->core.GetFenceValue(r->upload->fence) < slot->fence_value) {
        if (!logged && SDL_GetTicks() - started >= UPLOAD_SLOW_LOG_MS) {
            SDL_Log(
                "GPU upload chunk is still pending after %u ms; continuing "
                "without NRI's hard fence timeout",
                UPLOAD_SLOW_LOG_MS
            );
            logged = true;
        }

        SDL_Delay(1u);
    }

    slot->fence_value = 0u;

    return true;
}

static bool upload_begin_slot(RENDERER *r, UPLOAD_SLOT **out) {
    if (!out || !create_upload_context(r)) return false;

    UPLOAD_CONTEXT *upload = r->upload;
    UPLOAD_SLOT *slot = &upload->slots[upload->next_slot];
    upload->next_slot = (upload->next_slot + 1u) % UPLOAD_RING_SIZE;

    if (!upload_wait_slot(r, slot)) return false;
    r->core.ResetCommandAllocator(slot->allocator);

    if (r->core.BeginCommandBuffer(slot->command_buffer, NULL) != NriResult_SUCCESS) return false;

    *out = slot;
    return true;
}

static bool upload_submit_slot(RENDERER *r, UPLOAD_SLOT *slot) {
    if (!r || !r->upload || !slot) return false;

    if (r->core.EndCommandBuffer(slot->command_buffer) != NriResult_SUCCESS) return false;

    const uint64_t value = r->upload->next_fence_value++;

    const NriFenceSubmitDesc signal = {
        .fence = r->upload->fence,
        .value = value
    };

    NriCommandBuffer *command = slot->command_buffer;

    const NriQueueSubmitDesc submit = {
        .commandBuffers = (const NriCommandBuffer *const *)&command,
        .commandBufferNum = 1u,
        .signalFences = &signal,
        .signalFenceNum = 1u
    };

    if (r->core.QueueSubmit(r->graphics_queue, &submit) != NriResult_SUCCESS) return false;

    slot->fence_value = value;

    return true;
}

static bool upload_drain(RENDERER *r) {
    if (!r || !r->upload) return true;

    for (uint32_t i = 0; i < UPLOAD_RING_SIZE; ++i)
        if (!upload_wait_slot(r, &r->upload->slots[i])) return false;

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

static NriBuffer *upload_buffer(RENDERER *r, NriBufferUsageBits usage, const void *data, size_t bytes, uint32_t stride) {
    if (!r || !r->device || !r->graphics_queue || !data || !bytes) return NULL;

    const NriBufferDesc desc = {
        .size = bytes,
        .structureStride = stride,
        .usage = usage
    };

    NriBuffer *buffer = NULL;

    if (r->core.CreateCommittedBuffer(r->device, NriMemoryLocation_DEVICE, 1.0f, &desc, &buffer) != NriResult_SUCCESS) return NULL;

    const NriAccessStage copy_state = {
        .access = NriAccessBits_COPY_DESTINATION,
        .stages = NriStageBits_ALL
    };

    const NriAccessStage final_state = uploaded_buffer_state(usage);
    const Uint8 *source = data;

    size_t offset = 0u;

    while (offset < bytes) {
        const size_t chunk = bytes - offset > UPLOAD_CHUNK_BYTES ? UPLOAD_CHUNK_BYTES : bytes - offset;

        UPLOAD_SLOT *slot = NULL;

        if (!upload_begin_slot(r, &slot)) goto fail;

        void *mapped = r->core.MapBuffer(slot->staging, 0u, chunk);

        if (!mapped) {
            (void)r->core.EndCommandBuffer(slot->command_buffer);

            goto fail;
        }

        memcpy(mapped, source + offset, chunk);
        r->core.UnmapBuffer(slot->staging);

        if (!offset) {
            const NriBufferBarrierDesc barrier = {
                .buffer = buffer,
                .before = {0},
                .after = copy_state
            };

            r->core.CmdBarrier(slot->command_buffer, &(NriBarrierDesc){
                .buffers = &barrier,
                .bufferNum = 1u
            });
        }

        r->core.CmdCopyBuffer(slot->command_buffer, buffer, offset, slot->staging, 0u, chunk);

        if (offset + chunk == bytes) {
            const NriBufferBarrierDesc barrier = {
                .buffer = buffer,
                .before = copy_state,
                .after = final_state
            };

            r->core.CmdBarrier(slot->command_buffer, &(NriBarrierDesc){
                .buffers = &barrier,
                .bufferNum = 1u
            });
        }

        if (!upload_submit_slot(r, slot)) goto fail;
        offset += chunk;
    }

    return buffer;

fail:
    (void)upload_drain(r);
    r->core.DestroyBuffer(buffer);

    return NULL;
}

static bool
upload_texture_data(RENDERER *r, NriTexture *texture, const void *data, uint32_t row_pitch, uint32_t slice_pitch, NriAccessBits access, NriLayout layout, NriStageBits stages) {
    if (!r || !texture || !data || !row_pitch || !slice_pitch || slice_pitch % row_pitch) return false;

    const NriTextureDesc *desc = r->core.GetTextureDesc(texture);
    const NriDeviceDesc *device = r->core.GetDeviceDesc(r->device);
    const uint32_t row_alignment = device->memoryAlignment.uploadBufferTextureRow;
    const uint32_t slice_alignment = device->memoryAlignment.uploadBufferTextureSlice;

    const uint64_t aligned_row = upload_align(row_pitch, row_alignment);
    const uint32_t row_count = slice_pitch / row_pitch;

    if (!desc || !row_count || aligned_row > UPLOAD_CHUNK_BYTES) return false;

    const NriAccessLayoutStage copy_state = {
        .access = NriAccessBits_COPY_DESTINATION,
        .layout = NriLayout_COPY_DESTINATION,
        .stages = NriStageBits_ALL
    };

    const NriAccessLayoutStage final_state = {
        .access = access,
        .layout = layout,
        .stages = stages
    };

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

        Uint8 *mapped = r->core.MapBuffer(slot->staging, 0u, staging_bytes);

        if (!mapped) {
            (void)r->core.EndCommandBuffer(slot->command_buffer);

            goto fail;
        }

        for (uint32_t row = 0; row < rows; ++row)
            memcpy(mapped + (size_t)row * aligned_row, source + (size_t)(first_row + row) * row_pitch, row_pitch);
        r->core.UnmapBuffer(slot->staging);

        if (!first_row && !texture_barrier(r, slot->command_buffer, texture, (NriAccessLayoutStage){0}, copy_state)) goto fail;

        const NriTextureDataLayoutDesc source_layout = {
            .offset = 0u,
            .rowPitch = (uint32_t)aligned_row,
            .slicePitch = (uint32_t)staging_bytes
        };

        const NriTextureRegionDesc region = {
            .x = 0u,
            .y = (NriDim_t)first_row,
            .z = 0u,
            .width = desc->width,
            .height = (NriDim_t)rows,
            .depth = 1u,
            .mipOffset = 0u,
            .layerOffset = 0u
        };

        r->core.CmdUploadBufferToTexture(slot->command_buffer, texture, &region, slot->staging, &source_layout);

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

    NriTexture *result = create_texture(r, NriFormat_RGBA8_UNORM, NriTextureUsageBits_SHADER_RESOURCE, 1, 1);

    if (!result) return NULL;

    if (!upload_texture_data(r, result, pixels, 4, 4, NriAccessBits_SHADER_RESOURCE, NriLayout_SHADER_RESOURCE, NriStageBits_ALL)) {
        release_texture(r, result);

        return NULL;
    }

    return result;
}

void release_texture(RENDERER *r, NriTexture *value) {
    if (!r || !value) return;

    for (uint32_t i = 0; i < r->texture_state_num; ++i) {
        if (r->texture_states[i].texture == value) {
            r->texture_states[i] = r->texture_states[--r->texture_state_num];

            break;
        }
    }

    r->core.DestroyTexture(value);
}

void release_buffer(RENDERER *r, NriBuffer *value) {
    if (r && value) r->core.DestroyBuffer(value);
}

static bool ensure_depth_texture(RENDERER *r, Uint32 width, Uint32 height) {
    if (r->depth_texture && r->depth_width == width && r->depth_height == height) return true;

    release_texture(r, r->depth_texture);
    r->depth_texture = create_texture(r, r->depth_format, NriTextureUsageBits_DEPTH_STENCIL_ATTACHMENT, width, height);

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

    NriTexture *result = create_texture(r, NriFormat_RGBA8_UNORM, NriTextureUsageBits_SHADER_RESOURCE, (Uint32)rgba->w, (Uint32)rgba->h);

    if (result && !upload_texture_data(
                      r,
                      result,
                      rgba->pixels,
                      (uint32_t)rgba->pitch,
                      (uint32_t)(rgba->pitch * rgba->h),
                      NriAccessBits_SHADER_RESOURCE,
                      NriLayout_SHADER_RESOURCE,
                      NriStageBits_FRAGMENT_SHADER
                  )) {
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
    if (!r || !r->device) return;

    if (r->image_textures) {
        for (uint32_t i = 0; i < r->image_texture_count; ++i)
            release_texture(r, r->image_textures[i]);
    }

    free(r->image_textures);
    r->image_textures = NULL;
    r->image_texture_count = 0;

    free(r->materials);
    r->materials = NULL;
    r->material_count = 0;

    release_texture(r, r->default_white);
    release_texture(r, r->default_normal);

    if (r->material_sampler) r->core.DestroyDescriptor(r->material_sampler);
    release_buffer(r, r->vertex_buffer);
    release_texture(r, r->lightmap_texture);

    if (r->lightmap_sampler) r->core.DestroyDescriptor(r->lightmap_sampler);

    r->default_white = NULL;
    r->default_normal = NULL;
    r->material_sampler = NULL;
    r->vertex_buffer = NULL;
    r->lightmap_texture = NULL;
    r->lightmap_sampler = NULL;
}

bool upload_scene(RENDERER *r, const GLTF_SCENE *visual) {
    if (!r || !r->device || !visual || !r->vertices || !r->vertex_count) return false;

    release_scene_resources(r);

    r->default_white = pixel_texture(r, 255, 255, 255, 255);
    r->default_normal = pixel_texture(r, 128, 128, 255, 255);
    r->material_sampler = sampler(r, NriFilter_LINEAR, NriFilter_LINEAR, NriAddressMode_REPEAT);

    if (!r->default_white || !r->default_normal || !r->material_sampler) return false;

    const uint64_t vertex_bytes = (uint64_t)r->vertex_count * sizeof(*r->vertices);

    r->vertex_buffer = upload_buffer(r, NriBufferUsageBits_VERTEX, r->vertices, (size_t)vertex_bytes, 0);

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

    r->lightmap_sampler = sampler(r, NriFilter_LINEAR, NriFilter_LINEAR, NriAddressMode_CLAMP_TO_EDGE);
    r->lightmap_texture = pixel_texture(r, 0, 0, 0, 255);

    return r->lightmap_sampler && r->lightmap_texture;
}

static NriTexture *create_lightmap_texture(RENDERER *r, Uint32 width, Uint32 height) {
    return create_texture(r, NriFormat_RGBA16_SFLOAT, NriTextureUsageBits_SHADER_RESOURCE | NriTextureUsageBits_SHADER_RESOURCE_STORAGE, width, height);
}

static bool transfer_size(uint32_t width, uint32_t height, Uint32 *out) {
    const uint64_t bytes = (uint64_t)width * (uint64_t)height * 8u;

    if (!width || !height || bytes > UINT32_MAX) return false;
    *out = (Uint32)bytes;
    return true;
}

NriTexture *upload_lightmap(RENDERER *r, const CACHED_LIGHTMAP *cached) {
    if (!r || !r->device || !cached || !cached->pixels) return NULL;

    Uint32 bytes = 0;

    if (!transfer_size(cached->width, cached->height, &bytes)) return NULL;

    NriTexture *result = create_lightmap_texture(r, cached->width, cached->height);

    if (!result) return NULL;

    if (!upload_texture_data(r, result, cached->pixels, cached->width * 8u, bytes, NriAccessBits_SHADER_RESOURCE, NriLayout_SHADER_RESOURCE, NriStageBits_ALL)) {
        release_texture(r, result);

        return NULL;
    }

    return result;
}

static bool read_rgba16f_texture(RENDERER *r, NriTexture *texture, Uint32 width, Uint32 height, Uint8 **pixels) {
    if (!r || !r->device || !texture || !width || !height || !pixels) return false;
    *pixels = NULL;

    Uint32 tight_bytes = 0;

    if (!transfer_size(width, height, &tight_bytes)) return false;

    const uint32_t row_bytes = width * 8u;
    const NriDeviceDesc *device = r->core.GetDeviceDesc(r->device);
    const uint32_t row_alignment = device->memoryAlignment.uploadBufferTextureRow;
    const uint32_t slice_alignment = device->memoryAlignment.uploadBufferTextureSlice;

    if (!row_alignment || !slice_alignment) return false;

    const uint64_t row_pitch = ((uint64_t)row_bytes + row_alignment - 1u) / row_alignment * row_alignment;
    const uint64_t slice_bytes = row_pitch * height;
    const uint64_t staging_bytes = (slice_bytes + slice_alignment - 1u) / slice_alignment * slice_alignment;

    if (staging_bytes > UINT32_MAX) return false;

    const NriBufferDesc desc = {
        .size = staging_bytes,
        .usage = NriBufferUsageBits_NONE
    };

    NriBuffer *readback = NULL;

    if (r->core.CreateCommittedBuffer(r->device, NriMemoryLocation_HOST_READBACK, 1.0f, &desc, &readback) != NriResult_SUCCESS) return false;

    NriCommandAllocator *allocator = NULL;
    NriCommandBuffer *cmd = NULL;
    bool good = begin_commands(r, &allocator, &cmd) == NriResult_SUCCESS;

    if (good) {
        const NriTextureDataLayoutDesc layout = {
            .offset = 0,
            .rowPitch = (uint32_t)row_pitch,
            .slicePitch = (uint32_t)staging_bytes
        };

        const NriTextureRegionDesc region = {
            .width = (NriDim_t)width,
            .height = (NriDim_t)height,
            .depth = 1,
            .mipOffset = 0,
            .layerOffset = 0
        };

        good = transition_texture(r, cmd, texture, NriAccessBits_COPY_SOURCE, NriLayout_COPY_SOURCE, NriStageBits_COPY);

        if (good) {
            r->core.CmdReadbackTextureToBuffer(cmd, readback, &layout, texture, &region);

            good = submit_commands(r, allocator, cmd);
            allocator = NULL;
            cmd = NULL;
        }
    }

    if (!good) {
        abort_commands(r, allocator, cmd);
        r->core.DestroyBuffer(readback);

        return false;
    }

    const Uint8 *mapped = r->core.MapBuffer(readback, 0, staging_bytes);

    if (!mapped) {
        r->core.DestroyBuffer(readback);

        return false;
    }

    Uint8 *data = malloc(tight_bytes);

    if (data) {
        for (uint32_t y = 0; y < height; ++y)
            memcpy(data + (size_t)y * row_bytes, mapped + (size_t)y * row_pitch, row_bytes);
    }

    r->core.UnmapBuffer(readback);
    r->core.DestroyBuffer(readback);
    *pixels = data;
    return data != NULL;
}

bool download_lightmap(RENDERER *r, CACHED_LIGHTMAP *out) {
    if (!r || !out) return false;

    Uint8 *pixels = NULL;

    if (!read_rgba16f_texture(r, r->lightmap_texture, r->lightmap_width, r->lightmap_height, &pixels)) return false;
    out->pixels = pixels;
    out->width = r->lightmap_width;
    out->height = r->lightmap_height;

    return true;
}

bool upload_bvh(RENDERER *r, const BVH *tree) {
    if (!r || !tree || !tree->nodes || !tree->node_count || !tree->triangles || !tree->triangle_count) return false;

    release_buffer(r, r->bvh_node_buffer);
    release_buffer(r, r->bvh_triangle_buffer);
    r->bvh_node_buffer = NULL;
    r->bvh_triangle_buffer = NULL;

    r->bvh_node_buffer = upload_buffer(r, NriBufferUsageBits_SHADER_RESOURCE, tree->nodes, (size_t)tree->node_count * sizeof(*tree->nodes), sizeof(BVH_NODE));

    r->bvh_triangle_buffer = upload_buffer(r, NriBufferUsageBits_SHADER_RESOURCE, tree->triangles, (size_t)tree->triangle_count * sizeof(*tree->triangles), sizeof(BVH_TRIANGLE));

    const bool good = r->bvh_node_buffer && r->bvh_triangle_buffer;

    r->bvh_triangle_count = good ? tree->triangle_count : 0u;
    r->bvh_emissive_weight = good ? tree->emissive_weight : 0.0f;

    return good;
}

NriBuffer *upload_probes(RENDERER *r, const PROBE_GRID *grid) {
    if (!grid || !grid->probes) return NULL;

    const size_t count = (size_t)grid->count_x * grid->count_y * grid->count_z;

    if (!count) return NULL;

    return upload_buffer(r, NriBufferUsageBits_SHADER_RESOURCE, grid->probes, count * sizeof(PROBE), sizeof(PROBE));
}

NriBuffer *upload_beams(RENDERER *r, const BEAM_GRID *grid) {
    if (!grid) return NULL;

    float *visibility = beam_expand(grid);

    if (!visibility || !grid->shadow_depth) {
        free(visibility);

        return NULL;
    }

    const size_t beam_count = (size_t)grid->width * grid->height * grid->depth;
    const size_t depth_count = (size_t)grid->width * grid->height;

    float *data = realloc(visibility, (beam_count + depth_count) * sizeof(float));

    if (!data) {
        free(visibility);

        return NULL;
    }

    memcpy(data + beam_count, grid->shadow_depth, depth_count * sizeof(float));

    NriBuffer *buffer = upload_buffer(r, NriBufferUsageBits_SHADER_RESOURCE, data, (beam_count + depth_count) * sizeof(float), sizeof(float));

    free(data);

    return buffer;
}

typedef struct lightmap_queue_uniforms {
    Uint32 dispatch_width;

    Uint32 pad0, pad1, pad2;
} lightmap_queue_uniforms;

static NriBuffer *lightmap_queue_buffer(renderer *r, uint64_t bytes, NriBufferUsageBits usage) {
    const NriBufferDesc desc = {.size = bytes, .structureStride = sizeof(Uint32), .usage = usage};
    NriBuffer *buffer = NULL;

    return r->core.CreateCommittedBuffer(r->device, NriMemoryLocation_DEVICE, 1.0f, &desc, &buffer) == NriResult_SUCCESS ? buffer : NULL;
}

static bool lightmap_queue_ensure(renderer *r, Uint32 capacity) {
    if (!r || !capacity) return false;

    if (!r->lightmap_queue_reset_pipeline)
        r->lightmap_queue_reset_pipeline =
            compile_compute(r, r->lightmap_queue_reset_layout, "shaders/lightmap_queue.hlsl", "lightmap_queue_reset_cs", "BUILD_LIGHTMAP_QUEUE_RESET_CS");

    if (!r->lightmap_queue_args_pipeline)
        r->lightmap_queue_args_pipeline =
            compile_compute(r, r->lightmap_queue_args_layout, "shaders/lightmap_queue.hlsl", "lightmap_queue_args_cs", "BUILD_LIGHTMAP_QUEUE_ARGS_CS");

    if (!r->lightmap_queue_reset_pipeline || !r->lightmap_queue_args_pipeline) return false;

    const uint64_t bytes = (uint64_t)capacity * sizeof(Uint32);

    if (r->lightmap_active_capacity >= bytes && r->lightmap_active_buffer[0] && r->lightmap_active_buffer[1] && r->lightmap_active_count[0] && r->lightmap_active_count[1] &&
        r->lightmap_dispatch_args)
        return true;

    for (uint32_t i = 0; i < 2u; ++i) {
        release_buffer(r, r->lightmap_active_buffer[i]);
        release_buffer(r, r->lightmap_active_count[i]);
        r->lightmap_active_buffer[i] = NULL;
        r->lightmap_active_count[i] = NULL;
    }

    release_buffer(r, r->lightmap_dispatch_args);
    r->lightmap_dispatch_args = NULL;
    r->lightmap_active_capacity = 0u;

    const NriBufferUsageBits queue_usage = NriBufferUsageBits_SHADER_RESOURCE | NriBufferUsageBits_SHADER_RESOURCE_STORAGE;
    const NriBufferUsageBits args_usage = NriBufferUsageBits_SHADER_RESOURCE_STORAGE | NriBufferUsageBits_ARGUMENT;

    for (uint32_t i = 0; i < 2u; ++i) {
        r->lightmap_active_buffer[i] = lightmap_queue_buffer(r, bytes, queue_usage);
        r->lightmap_active_count[i] = lightmap_queue_buffer(r, sizeof(Uint32), queue_usage);
    }

    r->lightmap_dispatch_args = lightmap_queue_buffer(r, 3u * sizeof(Uint32), args_usage);

    if (!r->lightmap_active_buffer[0] || !r->lightmap_active_buffer[1] || !r->lightmap_active_count[0] || !r->lightmap_active_count[1] || !r->lightmap_dispatch_args) return false;

    r->lightmap_active_capacity = bytes;

    return true;
}

static void dispatch_shape_trace(Uint32 items, Uint32 *groups_x, Uint32 *groups_y, Uint32 *dispatch_width) {
    Uint32 gx = (items + 63u) / 64u;

    if (!gx) gx = 1u;

    if (gx > 256u) gx = 256u;

    const Uint32 width = gx * 64u;
    const Uint32 gy = (items + width - 1u) / width;
    *groups_x = gx;
    *groups_y = gy ? gy : 1u;
    *dispatch_width = width;
}

static void lightmap_buffer_barrier(renderer *r, NriCommandBuffer *cmd, NriBuffer *buffer, NriAccessBits before_access, NriStageBits before_stages, NriAccessBits after_access,
                                    NriStageBits after_stages) {
    const NriBufferBarrierDesc barrier = {
        .buffer = buffer, .before = {.access = before_access, .stages = before_stages}, .after = {.access = after_access, .stages = after_stages}};
    r->core.CmdBarrier(cmd, &(NriBarrierDesc){.buffers = &barrier, .bufferNum = 1u});
}

static bool record_lightmap_queue_reset(renderer *r, NriCommandBuffer *cmd, uint32_t index, bool reused) {
    NriBuffer *count = r->lightmap_active_count[index];
    lightmap_buffer_barrier(r, cmd, count, reused ? NriAccessBits_SHADER_RESOURCE : NriAccessBits_NONE, reused ? NriStageBits_COMPUTE_SHADER : NriStageBits_NONE,
                            NriAccessBits_SHADER_RESOURCE_STORAGE, NriStageBits_COMPUTE_SHADER);
    NriDescriptor *dst = create_buffer_view(r, count, NriBufferView_STORAGE_STRUCTURED_BUFFER, sizeof(Uint32));

    if (!dst || !bind_descriptor_set(r, cmd, r->lightmap_queue_reset_layout, NriBindPoint_COMPUTE, 1, &dst, 1)) return false;
    r->core.CmdSetPipeline(cmd, r->lightmap_queue_reset_pipeline);
    r->core.CmdDispatch(cmd, &(NriDispatchDesc){.workGroupNumX = 1u, .workGroupNumY = 1u, .workGroupNumZ = 1u});

    return true;
}

static bool record_lightmap_queue_args(renderer *r, NriCommandBuffer *cmd, uint32_t index, Uint32 dispatch_width, bool args_reused) {
    NriBuffer *count = r->lightmap_active_count[index];
    lightmap_buffer_barrier(r, cmd, count, NriAccessBits_SHADER_RESOURCE_STORAGE, NriStageBits_COMPUTE_SHADER, NriAccessBits_SHADER_RESOURCE, NriStageBits_COMPUTE_SHADER);
    lightmap_buffer_barrier(r, cmd, r->lightmap_dispatch_args, args_reused ? NriAccessBits_ARGUMENT_BUFFER : NriAccessBits_NONE,
                            args_reused ? NriStageBits_INDIRECT : NriStageBits_NONE, NriAccessBits_SHADER_RESOURCE_STORAGE, NriStageBits_COMPUTE_SHADER);

    NriDescriptor *src = create_buffer_view(r, count, NriBufferView_STRUCTURED_BUFFER, sizeof(Uint32));
    NriDescriptor *dst = create_buffer_view(r, r->lightmap_dispatch_args, NriBufferView_STORAGE_STRUCTURED_BUFFER, sizeof(Uint32));
    const lightmap_queue_uniforms uniforms = {.dispatch_width = dispatch_width};

    if (!src || !dst || !bind_descriptor_set(r, cmd, r->lightmap_queue_args_layout, NriBindPoint_COMPUTE, 0, &src, 1) ||
        !bind_descriptor_set(r, cmd, r->lightmap_queue_args_layout, NriBindPoint_COMPUTE, 1, &dst, 1) ||
        !bind_uniform_data(r, cmd, r->lightmap_queue_args_layout, NriBindPoint_COMPUTE, 2, &uniforms, sizeof(uniforms)))
        return false;

    r->core.CmdSetPipeline(cmd, r->lightmap_queue_args_pipeline);
    r->core.CmdDispatch(cmd, &(NriDispatchDesc){.workGroupNumX = 1u, .workGroupNumY = 1u, .workGroupNumZ = 1u});
    lightmap_buffer_barrier(r, cmd, r->lightmap_dispatch_args, NriAccessBits_SHADER_RESOURCE_STORAGE, NriStageBits_COMPUTE_SHADER, NriAccessBits_ARGUMENT_BUFFER,
                            NriStageBits_INDIRECT);
    return true;
}

static void dispatch_shape(Uint32 items, Uint32 *groups_x, Uint32 *groups_y, Uint32 *dispatch_width) {
    Uint32 gx = (items + 63u) / 64u;

    if (gx == 0u) gx = 1u;

    if (gx > 4096u) gx = 4096u;

    const Uint32 width = gx * 64u;
    const Uint32 gy = (items + width - 1u) / width;

    *groups_x = gx;
    *groups_y = gy ? gy : 1u;
    *dispatch_width = width;
}

static BAKE_UNIFORMS bake_data(RENDERER *r, Uint32 phase, Uint32 iteration, Uint32 item_count, Uint32 dispatch_width, Uint32 batch_count) {
    const DIRECTIONAL_LIGHT sun = r->sun;
    const SKY sky = r->sky;

    return (BAKE_UNIFORMS){
        .item_count = item_count,
        .lightmap_width = r->lightmap_width,
        .lightmap_height = r->lightmap_height,
        .dispatch_width = dispatch_width,
        .iteration = iteration,
        .phase = phase,
        .max_bounces = BAKE_MAX_BOUNCES,
        .batch_count = batch_count,
        .sun_direction_intensity = {sun.direction.x, sun.direction.y, sun.direction.z, sun.intensity},
        .sun_color_radius = {sun.color.x, sun.color.y, sun.color.z, sun.angular_radius},
        .sky_zenith = {sky.zenith.x, sky.zenith.y, sky.zenith.z, 1.0f},
        .sky_horizon = {sky.horizon.x, sky.horizon.y, sky.horizon.z, 1.0f},
        .bake_params = {r->bake_epsilon, 0.72f, sky.intensity, (float)r->lightmap_min_samples},
        .probe_origin_spacing = {r->lightmap_probe_origin.x, r->lightmap_probe_origin.y, r->lightmap_probe_origin.z, r->lightmap_probe_spacing},
        .probe_dims_mode = {r->lightmap_probe_count_x, r->lightmap_probe_count_y, r->lightmap_probe_count_z, 0u},
        .emissive_data = {r->bvh_emissive_weight, (float)r->bvh_triangle_count, r->volumetrics.emissive_probe_intensity, 0.0f}
    };
}

static bool
record_bake_pass(RENDERER *r, NriCommandBuffer *cmd, NriTexture *source, NriTexture *destination, Uint32 phase, Uint32 iteration, Uint32 item_count, Uint32 batch_count) {
    Uint32 groups_x, groups_y, dispatch_width;
    dispatch_shape(item_count, &groups_x, &groups_y, &dispatch_width);

    const BAKE_UNIFORMS uniforms = bake_data(r, phase, iteration, item_count, dispatch_width, batch_count);

    if (!bind_bake_resources(r, cmd, source, destination, &uniforms, sizeof(uniforms))) return false;

    r->core.CmdSetPipeline(cmd, r->bake_pipeline);
    r->core.CmdDispatch(cmd, &(NriDispatchDesc){
        .workGroupNumX = groups_x,
        .workGroupNumY = groups_y,
        .workGroupNumZ = 1
    });

    return true;
}

static void swap_lightmaps(RENDERER *r) {
    NriTexture *tmp = r->lightmap_texture;

    r->lightmap_texture = r->lightmap_scratch;
    r->lightmap_scratch = tmp;
}

static bool record_trace_batch(renderer *r, NriCommandBuffer *cmd, Uint32 first, Uint32 count, Uint32 items, Uint32 batch_index, Uint32 groups_x, Uint32 groups_y,
                               Uint32 dispatch_width) {
    if (!r || !cmd || !items || !count) return false;

    const uint32_t output_index = batch_index & 1u;
    const uint32_t input_index = output_index ^ 1u;
    const bool active_mode = batch_index != 0u;
    const bool output_reused = batch_index >= 2u;

    if (!record_lightmap_queue_reset(r, cmd, output_index, output_reused)) return false;
    lightmap_buffer_barrier(r, cmd, r->lightmap_active_buffer[output_index], output_reused ? NriAccessBits_SHADER_RESOURCE : NriAccessBits_NONE,
                            output_reused ? NriStageBits_COMPUTE_SHADER : NriStageBits_NONE, NriAccessBits_SHADER_RESOURCE_STORAGE, NriStageBits_COMPUTE_SHADER);

    if (active_mode) {
        lightmap_buffer_barrier(r, cmd, r->lightmap_active_buffer[input_index], NriAccessBits_SHADER_RESOURCE_STORAGE, NriStageBits_COMPUTE_SHADER, NriAccessBits_SHADER_RESOURCE,
                                NriStageBits_COMPUTE_SHADER);
    }

    bake_uniforms uniforms = bake_data(r, PHASE_TRACE, first, items, dispatch_width, count);

    uniforms.probe_dims_mode[3] = active_mode ? 1u : 0u;

    if (!bind_bake_resources_ex(r, cmd, r->lightmap_texture, r->lightmap_scratch, r->lightmap_active_buffer[input_index], r->lightmap_active_count[input_index],
                                r->lightmap_active_buffer[output_index], r->lightmap_active_count[output_index], &uniforms, sizeof(uniforms)))
        return false;

    r->core.CmdSetPipeline(cmd, r->bake_pipeline);

    if (active_mode) r->core.CmdDispatchIndirect(cmd, r->lightmap_dispatch_args, 0u);
    else r->core.CmdDispatch(cmd, &(NriDispatchDesc){.workGroupNumX = groups_x, .workGroupNumY = groups_y, .workGroupNumZ = 1u});

    if (!record_lightmap_queue_args(r, cmd, output_index, dispatch_width, batch_index != 0u)) return false;
    swap_lightmaps(r);

    return true;
}

static float decode_half(Uint16 h) {
    Uint32 exponent = (h >> 10u) & 31u;
    Uint32 mantissa = h & 1023u;
    float value = exponent == 0u ? ldexpf((float)mantissa, -24) : exponent == 31u ? INFINITY : ldexpf((float)(1024u + mantissa), (int)exponent - 25);

    return (h & 0x8000u) ? -value : value;
}

static float direct_luma(const Uint8 *pixels, Uint32 pixel) {
    Uint16 channels[3];

    memcpy(channels, pixels + (size_t)pixel * 8u, sizeof(channels));

    return decode_half(channels[0]) * 0.2126f + decode_half(channels[1]) * 0.7152f + decode_half(channels[2]) * 0.0722f;
}

static bool build_lightmap_patches(RENDERER *r, const LIGHTMAP *lm) {
    Uint8 *pixels = NULL;

    if (!read_rgba16f_texture(r, r->lightmap_direct, lm->width, lm->height, &pixels)) return false;

    const size_t pixel_count = (size_t)lm->width * lm->height;
    const size_t tile_width = (lm->width + 3u) / 4u;
    const size_t tile_count = tile_width * ((lm->height + 3u) / 4u);
    Uint32 *pixel_sample = malloc(pixel_count * sizeof(*pixel_sample));
    Uint32 *mapping = malloc((size_t)lm->sample_count * sizeof(*mapping));

    Uint32(*anchors)[4] = calloc(tile_count, sizeof(*anchors));

    Uint8 *keep = malloc(lm->sample_count);
    bool ok = false;

    if (!pixel_sample || !mapping || !anchors || !keep) goto cleanup;
    memset(pixel_sample, 0xff, pixel_count * sizeof(*pixel_sample));
    memset(mapping, 0xff, (size_t)lm->sample_count * sizeof(*mapping));
    memset(keep, 1, lm->sample_count);

    for (Uint32 i = 0; i < lm->sample_count; ++i) {
        Uint32 pixel;

        memcpy(&pixel, &lm->samples[i].position[3], sizeof(pixel));

        pixel_sample[pixel] = i;
    }

    Uint32 reduced = 0;
    Uint32 active_tiles = 0;

    for (Uint32 y = 0; y + 3u < lm->height; y += 4u) {
        for (Uint32 x = 0; x + 3u < lm->width; x += 4u) {
            Uint32 indices[16];
            float lo = INFINITY, hi = -INFINITY;
            bool smooth = true;
            Uint32 base_index = pixel_sample[(size_t)y * lm->width + x];

            if (base_index == UINT32_MAX) continue;

            const LMAP_SAMPLE *base = &lm->samples[base_index];

            for (Uint32 dy = 0; dy < 4u && smooth; ++dy) {
                for (Uint32 dx = 0; dx < 4u; ++dx) {
                    Uint32 pixel = (y + dy) * lm->width + x + dx;
                    Uint32 index = pixel_sample[pixel];

                    if (index == UINT32_MAX) {
                        smooth = false;

                        break;
                    }

                    const LMAP_SAMPLE *sample = &lm->samples[index];
                    float dot = base->normal[0] * sample->normal[0] + base->normal[1] * sample->normal[1] + base->normal[2] * sample->normal[2];

                    float plane = (sample->position[0] - base->position[0]) * base->normal[0] + (sample->position[1] - base->position[1]) * base->normal[1] +
                                  (sample->position[2] - base->position[2]) * base->normal[2];

                    if (sample->normal[3] != base->normal[3] || dot < 0.995f || fabsf(plane) > 0.025f) {
                        smooth = false;

                        break;
                    }

                    indices[dy * 4u + dx] = index;
                    float luma = direct_luma(pixels, pixel);
                    lo = fminf(lo, luma);
                    hi = fmaxf(hi, luma);
                }
            }

            if (!smooth || !isfinite(hi) || hi - lo > 0.06f + 0.1f * hi) continue;

            Uint32 tile = (y / 4u) * (Uint32)tile_width + x / 4u;
            const Uint32 corners[4] = {0u, 3u, 12u, 15u};

            for (Uint32 j = 0; j < 4u; ++j) {
                keep[indices[corners[j]]] = 1u;

                anchors[tile][j] = (y + corners[j] / 4u) * lm->width + x + corners[j] % 4u;
            }

            for (Uint32 j = 0; j < 16u; ++j) {
                mapping[indices[j]] = tile;

                if (j != 0u && j != 3u && j != 12u && j != 15u) keep[indices[j]] = 0u;
            }

            reduced += 12u;
            active_tiles++;
        }
    }

    ok = true;

    if (reduced) {
        Uint32 count = lm->sample_count - reduced;
        LMAP_SAMPLE *sparse = malloc((size_t)count * sizeof(*sparse));
        NriBuffer *sample_buffer = NULL;
        NriBuffer *map_buffer = NULL;
        NriBuffer *anchor_buffer = NULL;

        if (sparse) {
            Uint32 next = 0;

            for (Uint32 i = 0; i < lm->sample_count; ++i)
                if (keep[i]) sparse[next++] = lm->samples[i];

            sample_buffer = upload_buffer(r, NriBufferUsageBits_SHADER_RESOURCE, sparse, (size_t)count * sizeof(*sparse), sizeof(LMAP_SAMPLE));
            map_buffer = upload_buffer(r, NriBufferUsageBits_SHADER_RESOURCE, mapping, (size_t)lm->sample_count * sizeof(*mapping), sizeof(Uint32));
            anchor_buffer = upload_buffer(r, NriBufferUsageBits_SHADER_RESOURCE, anchors, tile_count * sizeof(*anchors), sizeof(Uint32[4]));
        }

        free(sparse);

        ok = sample_buffer && map_buffer && anchor_buffer;

        if (ok) {
            r->lightmap_full_sample_buffer = r->lightmap_sample_buffer;
            r->lightmap_sample_buffer = sample_buffer;
            r->lightmap_patch_map_buffer = map_buffer;
            r->lightmap_patch_anchor_buffer = anchor_buffer;
            r->lightmap_trace_count = count;
            SDL_Log("B: lightmap 4x4 patches: %u chart-safe tiles, %u/%u indirect texels", active_tiles, count, lm->sample_count);
        } else {
            release_buffer(r, sample_buffer);
            release_buffer(r, map_buffer);
            release_buffer(r, anchor_buffer);
        }
    }

cleanup:
    free(keep);
    free(anchors);
    free(mapping);
    free(pixel_sample);
    free(pixels);

    return ok;
}

static bool bake_lightmap_once(RENDERER *r, const LIGHTMAP *lm) {
    const Uint32 pixels = r->lightmap_width * r->lightmap_height;

    SDL_Log("B: lightmap trace samples per pass: %u", BAKE_BATCH_SAMPLES);
    bake_progress(r, "surface lightmap", 0u, r->bake_target_samples);

    NriCommandAllocator *allocator = NULL;
    NriCommandBuffer *cmd = NULL;

    if (begin_commands(r, &allocator, &cmd) != NriResult_SUCCESS) return false;

    if (!record_bake_pass(r, cmd, r->lightmap_scratch, r->lightmap_texture, PHASE_CLEAR, 0, pixels, 0) ||
        !record_bake_pass(r, cmd, r->lightmap_texture, r->lightmap_scratch, PHASE_CLEAR, 0, pixels, 0) ||
        !record_bake_pass(r, cmd, r->lightmap_scratch, r->lightmap_direct, PHASE_CLEAR, 0, pixels, 0) ||
        !record_bake_pass(r, cmd, r->lightmap_scratch, r->lightmap_direct, PHASE_DIRECT, 0, r->lightmap_sample_count, 0)) {
        abort_commands(r, allocator, cmd);

        return false;
    }

    if (!submit_commands(r, allocator, cmd)) return false;

    if (!build_lightmap_patches(r, lm)) return false;

    if (r->lightmap_trace_count) {
        allocator = NULL;
        cmd = NULL;

        if (begin_commands(r, &allocator, &cmd) != NriResult_SUCCESS) return false;

        if (!gpu_timestamp_begin(r, cmd, 2u)) {
            abort_commands(r, allocator, cmd);

            return false;
        }

        Uint32 trace_groups_x = 0u, trace_groups_y = 0u, trace_dispatch_width = 0u;

        dispatch_shape_trace(r->lightmap_trace_count, &trace_groups_x, &trace_groups_y, &trace_dispatch_width);

        Uint32 batch_index = 0u;

        for (Uint32 first = 0; first < r->bake_target_samples; first += BAKE_BATCH_SAMPLES, ++batch_index) {
            Uint32 count = r->bake_target_samples - first;

            if (count > BAKE_BATCH_SAMPLES) count = BAKE_BATCH_SAMPLES;

            if (!record_trace_batch(r, cmd, first, count, r->lightmap_trace_count, batch_index, trace_groups_x, trace_groups_y, trace_dispatch_width)) {
                abort_commands(r, allocator, cmd);

                return false;
            }
        }

        if (!gpu_timestamp_end(r, cmd, 2u) || !submit_commands(r, allocator, cmd)) return false;
        gpu_timestamp_log(r, 2u, "surface lightmap trace");
        bake_progress(r, "surface lightmap", r->bake_target_samples, r->bake_target_samples);
    }

    bake_progress(r, "filtering lightmap", 0u, 0u);

    allocator = NULL;
    cmd = NULL;

    if (begin_commands(r, &allocator, &cmd) != NriResult_SUCCESS) return false;

    if (r->lightmap_full_sample_buffer) {
        r->lightmap_sparse_sample_buffer = r->lightmap_sample_buffer;
        r->lightmap_sample_buffer = r->lightmap_full_sample_buffer;
        r->lightmap_full_sample_buffer = NULL;

        if (!record_bake_pass(r, cmd, r->lightmap_texture, r->lightmap_scratch, PHASE_CLEAR, 0, pixels, 0) ||
            !record_bake_pass(r, cmd, r->lightmap_texture, r->lightmap_scratch, PHASE_RECONSTRUCT, 0, r->lightmap_sample_count, 0)) {
            abort_commands(r, allocator, cmd);

            return false;
        }

        swap_lightmaps(r);
    }

    if (!record_bake_pass(r, cmd, r->lightmap_texture, r->lightmap_scratch, PHASE_COMBINE, 0, pixels, 0)) {
        abort_commands(r, allocator, cmd);

        return false;
    }

    swap_lightmaps(r);

    if (!record_bake_pass(r, cmd, r->lightmap_texture, r->lightmap_scratch, PHASE_FILTER, 0, pixels, 0)) {
        abort_commands(r, allocator, cmd);

        return false;
    }

    swap_lightmaps(r);

    for (Uint32 i = 0; i < BAKE_DILATION_PASSES; ++i) {
        if (!record_bake_pass(r, cmd, r->lightmap_texture, r->lightmap_scratch, PHASE_DILATE, 0, pixels, 0)) {
            abort_commands(r, allocator, cmd);

            return false;
        }

        swap_lightmaps(r);
    }

    if (!submit_commands(r, allocator, cmd)) return false;
    release_texture(r, r->lightmap_scratch);
    r->lightmap_scratch = NULL;

    return true;
}

bool bake_lightmap(RENDERER *r, const BVH *tree, const LIGHTMAP *lm, const PROBE_GRID *probes) {
    if (!r || !r->device || !tree || !tree->node_count || !lm || !lm->width || !lm->height || !lm->samples || !lm->sample_count || !r->lightmap_sampler || !probes ||
        !probes->probes || !probes->spacing || !probes->count_x || !probes->count_y || !probes->count_z)
        return false;

    r->lightmap_width = lm->width;
    r->lightmap_height = lm->height;
    r->lightmap_sample_count = lm->sample_count;
    r->lightmap_trace_count = lm->sample_count;
    r->bake_target_samples = BAKE_TARGET_SAMPLES;
    r->lightmap_min_samples = 32u;
    r->lightmap_probe_origin = probes->origin;
    r->lightmap_probe_spacing = probes->spacing;
    r->lightmap_probe_count_x = probes->count_x;
    r->lightmap_probe_count_y = probes->count_y;
    r->lightmap_probe_count_z = probes->count_z;
    SDL_Log("B: lightmap minimum samples: %u", r->lightmap_min_samples);

    r->lightmap_probe_buffer = upload_probes(r, probes);

    if (!r->lightmap_probe_buffer) return false;

    const BVH_NODE *root = &tree->nodes[0];
    const float sx = root->max[0] - root->min[0];
    const float sy = root->max[1] - root->min[1];
    const float sz = root->max[2] - root->min[2];
    float scene_scale = fmaxf(sx, fmaxf(sy, sz));

    if (scene_scale < 1.0f) scene_scale = 1.0f;

    r->bake_epsilon = scene_scale * 2.0e-5f;

    r->lightmap_texture = create_lightmap_texture(r, lm->width, lm->height);
    r->lightmap_scratch = create_lightmap_texture(r, lm->width, lm->height);
    r->lightmap_direct = create_lightmap_texture(r, lm->width, lm->height);

    if (!r->lightmap_texture || !r->lightmap_scratch || !r->lightmap_direct) return false;

    if (!upload_bvh(r, tree)) return false;
    r->lightmap_sample_buffer = upload_buffer(r, NriBufferUsageBits_SHADER_RESOURCE, lm->samples, (size_t)lm->sample_count * sizeof(*lm->samples), sizeof(LMAP_SAMPLE));

    if (!r->lightmap_sample_buffer) return false;

    if (!lightmap_queue_ensure(r, lm->sample_count)) return false;

    if (!r->bake_pipeline) {
        const NriDeviceDesc *device = r->core.GetDeviceDesc(r->device);
        const bool wave_ops = device && (device->wave.waveOpsStages & NriStageBits_COMPUTE_SHADER) != 0;

        r->bake_pipeline = compile_compute(r, r->bake_layout, "shaders/compute.hlsl", "lightmap_cs", wave_ops ? "BUILD_LIGHTMAP_WAVE_CS" : "BUILD_LIGHTMAP_CS");
    }

    if (!r->bake_pipeline) return false;

    SDL_Log("lightmap: %ux%u, %u charts, %u valid texels, %.2f texels/unit", lm->width, lm->height, lm->chart_count, lm->sample_count, lm->texel_density);

    return bake_lightmap_once(r, lm);
}

#define PROBE_BLOCK_SAMPLES 128u
#define PROBE_MAX_BOUNCES 3u
#define PROBE_PACKED_NODE_BYTES 32u
#define PROBE_PACKED_TRIANGLE_BYTES 48u
#define PROBE_RAY_STATE_BYTES 32u
#define PROBE_ACCUM_BYTES 32u
#define PROBE_OUTPUT_STRIDE_BYTES (9u * 16u)

typedef struct PROBE_WAVEFRONT_UNIFORMS {
    Uint32 probe_count;
    Uint32 sample_offset;
    Uint32 samples_per_block;
    Uint32 total_samples;
    Uint32 node_count;
    Uint32 triangle_count;
    Uint32 bounce_index;
    Uint32 max_bounces;
    Uint32 beam_depth;

    Uint32 emissive_samples, pad1, pad2;

    float sun_direction_intensity[4];
    float sun_color_radius[4];
    float sky_zenith[4];
    float sky_horizon[4];
    float bake_params[4];
    float emissive_params[4];
    float beam_origin[4];
    float beam_step[4];
} PROBE_WAVEFRONT_UNIFORMS;

typedef struct PROBE_WAVEFRONT_BUFFER {
    NriBuffer *buffer;
    uint32_t stride;
    NriAccessBits access;
    NriStageBits stages;
    NriBufferUsageBits usage;
    uint64_t capacity;
} probe_wavefront_buffer;

typedef struct PROBE_WAVEFRONT_STAGE {
    NriPipelineLayout *layout;
    NriPipeline *pipeline;
    uint8_t read_count;
    uint8_t write_count;
} PROBE_WAVEFRONT_STAGE;

typedef struct probe_wavefront_pipelines {
    probe_wavefront_stage prepare;
    probe_wavefront_stage reset;
    probe_wavefront_stage validate;
    probe_wavefront_stage primary;
    probe_wavefront_stage args;
    probe_wavefront_stage bounce;
    probe_wavefront_stage reduce;
} probe_wavefront_pipelines;

struct probe_wavefront_scratch {
    probe_wavefront_buffer packed_nodes;
    probe_wavefront_buffer packed_triangles;
    probe_wavefront_buffer states_a;
    probe_wavefront_buffer states_b;
    probe_wavefront_buffer results;
    probe_wavefront_buffer accums;
    probe_wavefront_buffer coefficients;
    probe_wavefront_buffer counters;
    probe_wavefront_buffer dispatch_args;
    probe_wavefront_pipelines pipelines;
    NriBuffer *counter_readback;
    NriBuffer *output_readback;
    uint64_t output_readback_capacity;
    bool pipelines_ready;
};

static probe_wavefront_buffer probe_wavefront_uploaded(renderer *r, const void *data, uint64_t bytes, uint32_t stride) {
    probe_wavefront_buffer result = {0};

    if (!data || !bytes || bytes > SIZE_MAX) return result;
    result.buffer = upload_buffer(r, NriBufferUsageBits_SHADER_RESOURCE, data, (size_t)bytes, stride);

    if (result.buffer) {
        result.stride = stride;
        result.access = NriAccessBits_SHADER_RESOURCE;
        result.stages = NriStageBits_COMPUTE_SHADER;
        result.usage = NriBufferUsageBits_SHADER_RESOURCE;
        result.capacity = bytes;
    }

    return result;
}

static PROBE_WAVEFRONT_BUFFER probe_wavefront_storage(RENDERER *r, uint64_t bytes, uint32_t stride) {
    PROBE_WAVEFRONT_BUFFER result = {0};

    if (!r || !r->device || !bytes) return result;

    const NriBufferUsageBits usage = NriBufferUsageBits_SHADER_RESOURCE | NriBufferUsageBits_SHADER_RESOURCE_STORAGE;
    const NriBufferDesc desc = {.size = bytes, .structureStride = stride, .usage = usage};

    if (r->core.CreateCommittedBuffer(r->device, NriMemoryLocation_DEVICE, 1.0f, &desc, &result.buffer) == NriResult_SUCCESS) {
        result.stride = stride;
        result.usage = usage;
        result.capacity = bytes;
    }

    return result;
}

static probe_wavefront_buffer probe_wavefront_argument(renderer *r) {
    probe_wavefront_buffer result = {0};
    const NriBufferDesc desc = {
        .size = 3u * sizeof(uint32_t), .structureStride = sizeof(uint32_t), .usage = NriBufferUsageBits_SHADER_RESOURCE_STORAGE | NriBufferUsageBits_ARGUMENT};

    if (r && r->device && r->core.CreateCommittedBuffer(r->device, NriMemoryLocation_DEVICE, 1.0f, &desc, &result.buffer) == NriResult_SUCCESS) {
        result.stride = sizeof(uint32_t);
        result.usage = desc.usage;
        result.capacity = desc.size;
    }

    return result;
}

static void probe_wavefront_release_buffer(RENDERER *r, PROBE_WAVEFRONT_BUFFER *buffer) {
    if (!buffer) return;
    release_buffer(r, buffer->buffer);
    *buffer = (PROBE_WAVEFRONT_BUFFER){0};
}

static bool probe_wavefront_reserve(renderer *r, probe_wavefront_buffer *buffer, uint64_t bytes, uint32_t stride, NriBufferUsageBits usage) {
    if (!r || !buffer || !bytes || !stride) return false;

    if (buffer->buffer && buffer->capacity >= bytes && buffer->stride == stride && buffer->usage == usage) return true;

    probe_wavefront_release_buffer(r, buffer);

    const NriBufferDesc desc = {.size = bytes, .structureStride = stride, .usage = usage};

    if (r->core.CreateCommittedBuffer(r->device, NriMemoryLocation_DEVICE, 1.0f, &desc, &buffer->buffer) != NriResult_SUCCESS) return false;

    buffer->stride = stride;
    buffer->usage = usage;
    buffer->capacity = bytes;

    return true;
}

static bool probe_wavefront_reserve_readback(renderer *r, NriBuffer **buffer, uint64_t *capacity, uint64_t bytes) {
    if (!r || !buffer || !capacity || !bytes) return false;

    if (*buffer && *capacity >= bytes) return true;

    if (*buffer) r->core.DestroyBuffer(*buffer);
    *buffer = NULL;
    *capacity = 0u;

    const NriBufferDesc desc = {.size = bytes};

    if (r->core.CreateCommittedBuffer(r->device, NriMemoryLocation_HOST_READBACK, 0.0f, &desc, buffer) != NriResult_SUCCESS) return false;

    *capacity = bytes;
    return true;
}

static bool probe_wavefront_layout(renderer *r, NriPipelineLayout **layout, uint8_t reads, uint8_t writes) {
    NriDescriptorType read_types[4] = {0};
    NriDescriptorType write_types[3] = {0};
    static const NriDescriptorType uniform[] = {NriDescriptorType_CONSTANT_BUFFER};

    if (reads > 4u || writes > 3u) return false;

    for (uint8_t i = 0; i < reads; ++i)
        read_types[i] = NriDescriptorType_STRUCTURED_BUFFER;

    for (uint8_t i = 0; i < writes; ++i)
        write_types[i] = NriDescriptorType_STORAGE_STRUCTURED_BUFFER;
    const NriDescriptorType *sets[4] = {reads ? read_types : NULL, writes ? write_types : NULL, uniform, NULL};

    const uint8_t counts[4] = {reads, writes, 1u, 0u};

    return create_pipeline_layout(r, layout, sets, counts, NriStageBits_COMPUTE_SHADER);
}

static bool probe_wavefront_stage_init(RENDERER *r, PROBE_WAVEFRONT_STAGE *stage, const char *entrypoint, const char *define, uint8_t reads, uint8_t writes) {
    if (!probe_wavefront_layout(r, &stage->layout, reads, writes)) return false;
    stage->pipeline = compile_compute(r, stage->layout, "shaders/probe_wavefront.hlsl", entrypoint, define);
    stage->read_count = reads;
    stage->write_count = writes;

    return stage->pipeline != NULL;
}

static void probe_wavefront_stage_deinit(RENDERER *r, PROBE_WAVEFRONT_STAGE *stage) {
    if (!stage) return;

    if (stage->pipeline) r->core.DestroyPipeline(stage->pipeline);

    if (stage->layout) r->core.DestroyPipelineLayout(stage->layout);
    *stage = (PROBE_WAVEFRONT_STAGE){0};
}

static bool probe_wavefront_pipelines_init(RENDERER *r, PROBE_WAVEFRONT_PIPELINES *p) {
    memset(p, 0, sizeof(*p));

    const NriDeviceDesc *device = r->core.GetDeviceDesc(r->device);
    const bool wave_ops = device && (device->wave.waveOpsStages & NriStageBits_COMPUTE_SHADER) != 0;
    const char *primary_define = wave_ops ? "BUILD_PROBE_PRIMARY_WAVE_CS" : "BUILD_PROBE_PRIMARY_CS";
    const char *bounce_define = wave_ops ? "BUILD_PROBE_BOUNCE_WAVE_CS" : "BUILD_PROBE_BOUNCE_CS";

    return probe_wavefront_stage_init(r, &p->prepare, "probe_prepare_cs", "BUILD_PROBE_PREP_CS", 2, 2) &&
           probe_wavefront_stage_init(r, &p->reset, "probe_reset_cs", "BUILD_PROBE_RESET_CS", 0, 3) &&
           probe_wavefront_stage_init(r, &p->validate, "probe_validate_cs", "BUILD_PROBE_VALIDATE_CS", 3, 1) &&
           probe_wavefront_stage_init(r, &p->primary, "probe_primary_cs", primary_define, 4, 3) &&
           probe_wavefront_stage_init(r, &p->args, "probe_args_cs", "BUILD_PROBE_ARGS_CS", 1, 1) &&
           probe_wavefront_stage_init(r, &p->bounce, "probe_bounce_cs", bounce_define, 4, 3) &&
           probe_wavefront_stage_init(r, &p->reduce, "probe_reduce_cs", "BUILD_PROBE_REDUCE_CS", 1, 3);
}

static void probe_wavefront_pipelines_deinit(RENDERER *r, PROBE_WAVEFRONT_PIPELINES *p) {
    probe_wavefront_stage_deinit(r, &p->emissive);
    probe_wavefront_stage_deinit(r, &p->reduce);
    probe_wavefront_stage_deinit(r, &p->bounce);
    probe_wavefront_stage_deinit(r, &p->args);
    probe_wavefront_stage_deinit(r, &p->primary);
    probe_wavefront_stage_deinit(r, &p->validate);
    probe_wavefront_stage_deinit(r, &p->reset);
    probe_wavefront_stage_deinit(r, &p->prepare);
}

static bool probe_wavefront_scratch_ensure(renderer *r, uint64_t node_bytes, uint64_t triangle_bytes, uint64_t state_bytes, uint64_t result_bytes, uint64_t accum_bytes,
                                           uint64_t output_bytes) {
    if (!r) return false;

    if (!r->probe_scratch) {
        r->probe_scratch = calloc(1, sizeof(*r->probe_scratch));

        if (!r->probe_scratch) return false;
    }

    probe_wavefront_scratch *scratch = r->probe_scratch;
    const NriBufferUsageBits storage = NriBufferUsageBits_SHADER_RESOURCE | NriBufferUsageBits_SHADER_RESOURCE_STORAGE;
    const NriBufferUsageBits argument = NriBufferUsageBits_SHADER_RESOURCE_STORAGE | NriBufferUsageBits_ARGUMENT;

    if (!probe_wavefront_reserve(r, &scratch->packed_nodes, node_bytes, PROBE_PACKED_NODE_BYTES, storage) ||
        !probe_wavefront_reserve(r, &scratch->packed_triangles, triangle_bytes, PROBE_PACKED_TRIANGLE_BYTES, storage) ||
        !probe_wavefront_reserve(r, &scratch->states_a, state_bytes, PROBE_RAY_STATE_BYTES, storage) ||
        !probe_wavefront_reserve(r, &scratch->states_b, state_bytes, PROBE_RAY_STATE_BYTES, storage) ||
        !probe_wavefront_reserve(r, &scratch->results, result_bytes, sizeof(float[4]), storage) ||
        !probe_wavefront_reserve(r, &scratch->accums, accum_bytes, PROBE_ACCUM_BYTES, storage) ||
        !probe_wavefront_reserve(r, &scratch->coefficients, output_bytes, sizeof(float[4]), storage) ||
        !probe_wavefront_reserve(r, &scratch->counters, 4u * sizeof(uint32_t), sizeof(uint32_t), storage) ||
        !probe_wavefront_reserve(r, &scratch->dispatch_args, 3u * sizeof(uint32_t), sizeof(uint32_t), argument))
        return false;

    uint64_t counter_capacity = scratch->counter_readback ? 4u * sizeof(uint32_t) : 0u;

    if (!probe_wavefront_reserve_readback(r, &scratch->counter_readback, &counter_capacity, 4u * sizeof(uint32_t)) ||
        !probe_wavefront_reserve_readback(r, &scratch->output_readback, &scratch->output_readback_capacity, output_bytes))
        return false;

    if (!scratch->pipelines_ready) {
        if (!probe_wavefront_pipelines_init(r, &scratch->pipelines)) return false;
        scratch->pipelines_ready = true;
    }

    return true;
}

static void probe_wavefront_scratch_destroy(renderer *r) {
    if (!r || !r->probe_scratch) return;

    probe_wavefront_scratch *scratch = r->probe_scratch;

    if (scratch->pipelines_ready) probe_wavefront_pipelines_deinit(r, &scratch->pipelines);
    probe_wavefront_release_buffer(r, &scratch->dispatch_args);
    probe_wavefront_release_buffer(r, &scratch->counters);
    probe_wavefront_release_buffer(r, &scratch->coefficients);
    probe_wavefront_release_buffer(r, &scratch->accums);
    probe_wavefront_release_buffer(r, &scratch->results);
    probe_wavefront_release_buffer(r, &scratch->states_b);
    probe_wavefront_release_buffer(r, &scratch->states_a);
    probe_wavefront_release_buffer(r, &scratch->packed_triangles);
    probe_wavefront_release_buffer(r, &scratch->packed_nodes);

    if (scratch->counter_readback) r->core.DestroyBuffer(scratch->counter_readback);

    if (scratch->output_readback) r->core.DestroyBuffer(scratch->output_readback);
    free(scratch);
    r->probe_scratch = NULL;
}

static bool probe_wavefront_transition(renderer *r, NriCommandBuffer *cmd, probe_wavefront_buffer *const *reads, uint8_t read_count, probe_wavefront_buffer *const *writes,
                                       uint8_t write_count) {
    NriBufferBarrierDesc barriers[7] = {0};
    uint32_t count = 0;

    for (uint8_t i = 0; i < read_count; ++i) {
        PROBE_WAVEFRONT_BUFFER *buffer = reads[i];

        if (!buffer || !buffer->buffer) return false;

        barriers[count++] = (NriBufferBarrierDesc){
            .buffer = buffer->buffer,
            .before = {
                .access = buffer->access,
                .stages = buffer->stages
            },
            .after = {
                .access = NriAccessBits_SHADER_RESOURCE,
                .stages = NriStageBits_COMPUTE_SHADER
            }
        };

        buffer->access = NriAccessBits_SHADER_RESOURCE;
        buffer->stages = NriStageBits_COMPUTE_SHADER;
    }

    for (uint8_t i = 0; i < write_count; ++i) {
        PROBE_WAVEFRONT_BUFFER *buffer = writes[i];

        if (!buffer || !buffer->buffer) return false;

        barriers[count++] = (NriBufferBarrierDesc){
            .buffer = buffer->buffer,
            .before = {
                .access = buffer->access,
                .stages = buffer->stages
            },
            .after = {
                .access = NriAccessBits_SHADER_RESOURCE_STORAGE,
                .stages = NriStageBits_COMPUTE_SHADER
            }
        };

        buffer->access = NriAccessBits_SHADER_RESOURCE_STORAGE;
        buffer->stages = NriStageBits_COMPUTE_SHADER;
    }

    if (count) r->core.CmdBarrier(cmd, &(NriBarrierDesc){
        .buffers = barriers,
        .bufferNum = count
    });

    return true;
}

static bool probe_wavefront_dispatch(
    RENDERER *r,
    NriCommandBuffer *cmd,
    const PROBE_WAVEFRONT_STAGE *stage,
    PROBE_WAVEFRONT_BUFFER *const *reads,
    PROBE_WAVEFRONT_BUFFER *const *writes,
    const PROBE_WAVEFRONT_UNIFORMS *uniforms,
    uint32_t groups_x
) {
    if (!r || !cmd || !stage || !stage->pipeline || !stage->layout || !uniforms || !groups_x) return false;

    if (!probe_wavefront_transition(r, cmd, reads, stage->read_count, writes, stage->write_count)) return false;

    if (stage->read_count) {
        NriDescriptor *descriptors[4] = {0};

        for (uint8_t i = 0; i < stage->read_count; ++i)
            descriptors[i] = create_buffer_view(r, reads[i]->buffer, NriBufferView_STRUCTURED_BUFFER, reads[i]->stride);

        if (!bind_descriptor_set(r, cmd, stage->layout, NriBindPoint_COMPUTE, 0, descriptors, stage->read_count)) return false;
    }

    if (stage->write_count) {
        NriDescriptor *descriptors[3] = {0};

        for (uint8_t i = 0; i < stage->write_count; ++i)
            descriptors[i] = create_buffer_view(r, writes[i]->buffer, NriBufferView_STORAGE_STRUCTURED_BUFFER, writes[i]->stride);

        if (!bind_descriptor_set(r, cmd, stage->layout, NriBindPoint_COMPUTE, 1, descriptors, stage->write_count)) return false;
    }

    if (!bind_uniform_data(r, cmd, stage->layout, NriBindPoint_COMPUTE, 2, uniforms, sizeof(*uniforms))) return false;

    r->core.CmdSetPipeline(cmd, stage->pipeline);
    r->core.CmdDispatch(cmd, &(NriDispatchDesc){
        .workGroupNumX = groups_x,
        .workGroupNumY = 1,
        .workGroupNumZ = 1
    });

    return true;
}

static bool probe_wavefront_dispatch_indirect(renderer *r, NriCommandBuffer *cmd, const probe_wavefront_stage *stage, probe_wavefront_buffer *const *reads,
                                              probe_wavefront_buffer *const *writes, const probe_wavefront_uniforms *uniforms, probe_wavefront_buffer *arguments) {
    if (!r || !cmd || !stage || !stage->pipeline || !stage->layout || !uniforms || !arguments || !arguments->buffer) return false;

    if (!probe_wavefront_transition(r, cmd, reads, stage->read_count, writes, stage->write_count)) return false;

    if (stage->read_count) {
        NriDescriptor *descriptors[4] = {0};

        for (uint8_t i = 0; i < stage->read_count; ++i)
            descriptors[i] = create_buffer_view(r, reads[i]->buffer, NriBufferView_STRUCTURED_BUFFER, reads[i]->stride);

        if (!bind_descriptor_set(r, cmd, stage->layout, NriBindPoint_COMPUTE, 0, descriptors, stage->read_count)) return false;
    }

    if (stage->write_count) {
        NriDescriptor *descriptors[3] = {0};

        for (uint8_t i = 0; i < stage->write_count; ++i)
            descriptors[i] = create_buffer_view(r, writes[i]->buffer, NriBufferView_STORAGE_STRUCTURED_BUFFER, writes[i]->stride);

        if (!bind_descriptor_set(r, cmd, stage->layout, NriBindPoint_COMPUTE, 1, descriptors, stage->write_count)) return false;
    }

    if (!bind_uniform_data(r, cmd, stage->layout, NriBindPoint_COMPUTE, 2, uniforms, sizeof(*uniforms))) return false;

    const NriBufferBarrierDesc barrier = {.buffer = arguments->buffer,
                                          .before = {.access = arguments->access, .stages = arguments->stages},
                                          .after = {.access = NriAccessBits_ARGUMENT_BUFFER, .stages = NriStageBits_INDIRECT}};
    r->core.CmdBarrier(cmd, &(NriBarrierDesc){.buffers = &barrier, .bufferNum = 1});
    arguments->access = NriAccessBits_ARGUMENT_BUFFER;
    arguments->stages = NriStageBits_INDIRECT;

    r->core.CmdSetPipeline(cmd, stage->pipeline);
    r->core.CmdDispatchIndirect(cmd, arguments->buffer, 0u);

    return true;
}

static uint32_t probe_wavefront_groups64(uint64_t threads) {
    const uint64_t groups = (threads + 63u) / 64u;

    return groups && groups <= UINT32_MAX ? (uint32_t)groups : 0u;
}

static PROBE_WAVEFRONT_UNIFORMS probe_wavefront_data(
    const BVH *tree,
    const BEAM_GRID *beams,
    DIRECTIONAL_LIGHT sun,
    SKY sky,
    VOLUMETRICS_LIGHTING volumetrics,
    uint32_t probe_count,
    uint32_t sample_offset,
    uint32_t block_samples,
    uint32_t bounce_index
) {
    const BVH_NODE *root = &tree->nodes[0];
    float scene_scale = fmaxf(root->max[0] - root->min[0], fmaxf(root->max[1] - root->min[1], root->max[2] - root->min[2]));

    if (scene_scale < 1.0f) scene_scale = 1.0f;

    const float epsilon = scene_scale * 2.0e-5f;

    return (PROBE_WAVEFRONT_UNIFORMS){
        .probe_count = probe_count,
        .sample_offset = sample_offset,
        .samples_per_block = block_samples,
        .total_samples = volumetrics.probe_samples,
        .node_count = tree->node_count,
        .triangle_count = tree->triangle_count,
        .bounce_index = bounce_index,
        .max_bounces = PROBE_MAX_BOUNCES,
        .beam_depth = beams->depth,
        .emissive_samples = volumetrics.emissive_samples,
        .sun_direction_intensity = {sun.direction.x, sun.direction.y, sun.direction.z, sun.intensity},
        .sun_color_radius = {sun.color.x, sun.color.y, sun.color.z, sun.angular_radius},
        .sky_zenith = {sky.zenith.x, sky.zenith.y, sky.zenith.z, 1.0f},
        .sky_horizon = {sky.horizon.x, sky.horizon.y, sky.horizon.z, 1.0f},
        .bake_params = {epsilon, 0.72f, sky.intensity, tree->emissive_weight},
        .emissive_params = {volumetrics.emissive_probe_intensity, 0.0f, 0.0f, 0.0f},
        .beam_origin = {beams->origin.x, beams->origin.y, beams->origin.z, 0.0f},
        .beam_step = {beams->step.x, beams->step.y, beams->step.z, 0.0f}
    };
}

static NriBuffer *probe_wavefront_readback_buffer(RENDERER *r, uint64_t bytes) {
    const NriBufferDesc desc = {
        .size = bytes
    };

    NriBuffer *buffer = NULL;

    if (r->core.CreateCommittedBuffer(r->device, NriMemoryLocation_HOST_READBACK, 1.0f, &desc, &buffer) != NriResult_SUCCESS) return NULL;

    return buffer;
}

static bool probe_wavefront_copy_to_readback(RENDERER *r, NriCommandBuffer *cmd, PROBE_WAVEFRONT_BUFFER *source, NriBuffer *destination, uint64_t bytes) {
    if (!source || !source->buffer || !destination) return false;

    const NriBufferBarrierDesc barrier = {
        .buffer = source->buffer,
        .before = {
            .access = source->access,
            .stages = source->stages
        },
        .after = {
            .access = NriAccessBits_COPY_SOURCE,
            .stages = NriStageBits_COPY
        }
    };

    r->core.CmdBarrier(cmd, &(NriBarrierDesc){
        .buffers = &barrier,
        .bufferNum = 1
    });
    source->access = NriAccessBits_COPY_SOURCE;
    source->stages = NriStageBits_COPY;
    r->core.CmdCopyBuffer(cmd, destination, 0, source->buffer, 0, bytes);

    return true;
}

bool bake_probe_grid_fast(RENDERER *r, PROBE_GRID *grid, const BVH *tree, const BEAM_GRID *beams, PROBE_BAKE_PROGRESS_FN progress) {
    if (!r || !r->device || !grid || !grid->probes || !tree || !tree->node_count || !tree->triangle_count || !beams || !beams->shadow_depth) return false;

    const uint64_t probe_count64 = (uint64_t)grid->count_x * grid->count_y * grid->count_z;

    if (!probe_count64 || probe_count64 > UINT32_MAX) return false;

    const uint32_t probe_count = (uint32_t)probe_count64;
    const uint64_t max_rays = probe_count64 * PROBE_BLOCK_SAMPLES;
    const uint64_t output_bytes = probe_count64 * PROBE_OUTPUT_STRIDE_BYTES;
    const uint64_t state_bytes = max_rays * PROBE_RAY_STATE_BYTES;
    const uint64_t result_bytes = max_rays * sizeof(float[4]);
    const uint64_t accum_bytes = probe_count64 * PROBE_ACCUM_BYTES;

    float (*positions)[4] = malloc((size_t)probe_count * sizeof(float[4]));

    if (!positions) return false;

    for (uint32_t i = 0; i < probe_count; ++i)
        memcpy(positions[i], grid->probes[i].position, sizeof(float[4]));

    float *beam_data = beam_expand(beams);

    if (!beam_data || !beams->shadow_depth) {
        free(positions);
        free(beam_data);

        return false;
    }

    const size_t beam_count = (size_t)beams->width * beams->height * beams->depth;
    const size_t depth_count = (size_t)beams->width * beams->height;
    float *expanded = realloc(beam_data, (beam_count + depth_count) * sizeof(float));

    if (!expanded) {
        free(positions);
        free(beam_data);

        return false;
    }

    beam_data = expanded;

    memcpy(beam_data + beam_count, beams->shadow_depth, depth_count * sizeof(float));

    PROBE_WAVEFRONT_BUFFER position_buffer = probe_wavefront_uploaded(r, positions, probe_count64 * sizeof(float[4]), sizeof(float[4]));
    PROBE_WAVEFRONT_BUFFER source_nodes = probe_wavefront_uploaded(r, tree->nodes, (uint64_t)tree->node_count * sizeof(*tree->nodes), sizeof(BVH_NODE));
    PROBE_WAVEFRONT_BUFFER source_triangles = probe_wavefront_uploaded(r, tree->triangles, (uint64_t)tree->triangle_count * sizeof(*tree->triangles), sizeof(BVH_TRIANGLE));
    PROBE_WAVEFRONT_BUFFER sun_beams = probe_wavefront_uploaded(r, beam_data, (beam_count + depth_count) * sizeof(float), sizeof(float));

    free(positions);
    free(beam_data);

    const bool scratch_ready = probe_wavefront_scratch_ensure(r, (uint64_t)tree->node_count * PROBE_PACKED_NODE_BYTES, (uint64_t)tree->triangle_count * PROBE_PACKED_TRIANGLE_BYTES,
                                                              state_bytes, result_bytes, accum_bytes, output_bytes);
    probe_wavefront_scratch *scratch = r->probe_scratch;

    probe_wavefront_buffer packed_nodes = scratch_ready ? scratch->packed_nodes : (probe_wavefront_buffer){0};
    probe_wavefront_buffer packed_triangles = scratch_ready ? scratch->packed_triangles : (probe_wavefront_buffer){0};
    probe_wavefront_buffer states_a = scratch_ready ? scratch->states_a : (probe_wavefront_buffer){0};
    probe_wavefront_buffer states_b = scratch_ready ? scratch->states_b : (probe_wavefront_buffer){0};
    probe_wavefront_buffer results = scratch_ready ? scratch->results : (probe_wavefront_buffer){0};
    probe_wavefront_buffer accums = scratch_ready ? scratch->accums : (probe_wavefront_buffer){0};
    probe_wavefront_buffer coefficients = scratch_ready ? scratch->coefficients : (probe_wavefront_buffer){0};
    probe_wavefront_buffer counters = scratch_ready ? scratch->counters : (probe_wavefront_buffer){0};
    probe_wavefront_buffer dispatch_args = scratch_ready ? scratch->dispatch_args : (probe_wavefront_buffer){0};
    NriBuffer *counter_readback = scratch_ready ? scratch->counter_readback : NULL;
    NriBuffer *output_readback = scratch_ready ? scratch->output_readback : NULL;
    probe_wavefront_pipelines pipelines = scratch_ready ? scratch->pipelines : (probe_wavefront_pipelines){0};

    bool good = position_buffer.buffer && source_nodes.buffer && source_triangles.buffer && sun_beams.buffer && scratch_ready;

    const uint32_t prep_groups = probe_wavefront_groups64(tree->node_count > tree->triangle_count ? tree->node_count : tree->triangle_count);
    const uint32_t probe_groups = probe_wavefront_groups64(probe_count);
    probe_wavefront_uniforms uniforms = probe_wavefront_data(tree, beams, probe_count, 0u, PROBE_BLOCK_SAMPLES, 0u);

    if (!prep_groups || !probe_groups) good = false;

    if (good) {
        NriCommandAllocator *allocator = NULL;
        NriCommandBuffer *cmd = NULL;
        good = begin_commands(r, &allocator, &cmd) == NriResult_SUCCESS;

        if (good) {
            good = gpu_timestamp_begin(r, cmd, 0u);
            probe_wavefront_buffer *prepare_reads[] = {&source_nodes, &source_triangles};

            PROBE_WAVEFRONT_BUFFER *prepare_writes[] = {&packed_nodes, &packed_triangles};

            PROBE_WAVEFRONT_BUFFER *reset_writes[] = {&counters, &accums, &coefficients};

            PROBE_WAVEFRONT_BUFFER *validate_reads[] = {&position_buffer, &packed_nodes, &packed_triangles};

            PROBE_WAVEFRONT_BUFFER *validate_writes[] = {&accums};
            good = probe_wavefront_dispatch(r, cmd, &pipelines.prepare, prepare_reads, prepare_writes, &uniforms, prep_groups) &&
                   probe_wavefront_dispatch(r, cmd, &pipelines.reset, NULL, reset_writes, &uniforms, probe_groups) &&
                   probe_wavefront_dispatch(r, cmd, &pipelines.validate, validate_reads, validate_writes, &uniforms, probe_groups);

            if (good) good = gpu_timestamp_end(r, cmd, 0u);

            if (good) good = submit_commands(r, allocator, cmd);
            else abort_commands(r, allocator, cmd);
        }

        if (good) gpu_timestamp_log(r, 0u, "probe prepare + validate");
    }

    uint32_t completed = 0u;
    uint32_t active = probe_count;
    const uint32_t max_samples = r->volumetrics.probe_samples;

    if (!max_samples || max_samples > 65536u) good = false;

    if (good && progress && !progress(0u, max_samples, active)) good = false;

    while (good && completed < max_samples && active) {
        const uint32_t block = max_samples - completed > PROBE_BLOCK_SAMPLES ? PROBE_BLOCK_SAMPLES : max_samples - completed;

        uniforms = probe_wavefront_data(tree, beams, r->sun, r->sky, r->volumetrics, probe_count, completed, block, 0u);

        NriCommandAllocator *allocator = NULL;
        NriCommandBuffer *cmd = NULL;
        good = begin_commands(r, &allocator, &cmd) == NriResult_SUCCESS;

        if (!good) break;

        good = gpu_timestamp_begin(r, cmd, 2u);

        if (completed) {
            PROBE_WAVEFRONT_BUFFER *reset_writes[] = {&counters, &accums, &coefficients};

            good = probe_wavefront_dispatch(r, cmd, &pipelines.reset, NULL, reset_writes, &uniforms, probe_groups);
        }

        PROBE_WAVEFRONT_BUFFER *primary_reads[] = {&position_buffer, &accums, &packed_nodes, &packed_triangles};

        PROBE_WAVEFRONT_BUFFER *primary_writes[] = {&results, &states_a, &counters};

        if (good) good = probe_wavefront_dispatch(r, cmd, &pipelines.primary, primary_reads, primary_writes, &uniforms, probe_wavefront_groups64((uint64_t)probe_count * block));

        for (uint32_t bounce = 0; good && bounce < PROBE_MAX_BOUNCES; ++bounce) {
            uniforms.bounce_index = bounce;

            probe_wavefront_buffer *args_reads[] = {&counters};
            probe_wavefront_buffer *args_writes[] = {&dispatch_args};
            good = probe_wavefront_dispatch(r, cmd, &pipelines.args, args_reads, args_writes, &uniforms, 1u);

            probe_wavefront_buffer *input = bounce & 1u ? &states_b : &states_a;
            probe_wavefront_buffer *output = bounce & 1u ? &states_a : &states_b;
            probe_wavefront_buffer *bounce_reads[] = {&packed_nodes, &packed_triangles, input, &sun_beams};
            probe_wavefront_buffer *bounce_writes[] = {&results, output, &counters};

            if (good) good = probe_wavefront_dispatch_indirect(r, cmd, &pipelines.bounce, bounce_reads, bounce_writes, &uniforms, &dispatch_args);
        }

        uniforms.bounce_index = 0u;

        PROBE_WAVEFRONT_BUFFER *reduce_reads[] = {&results};
        PROBE_WAVEFRONT_BUFFER *reduce_writes[] = {&accums, &coefficients, &counters};

        if (good) good = probe_wavefront_dispatch(r, cmd, &pipelines.reduce, reduce_reads, reduce_writes, &uniforms, probe_count);

        if (good) good = gpu_timestamp_end(r, cmd, 2u);

        if (good) good = probe_wavefront_copy_to_readback(r, cmd, &counters, counter_readback, 4u * sizeof(uint32_t));

        if (good) good = submit_commands(r, allocator, cmd);
        else abort_commands(r, allocator, cmd);

        if (!good) break;
        gpu_timestamp_log(r, 2u, "probe trace block");

        const uint32_t *values = r->core.MapBuffer(counter_readback, 0, 4u * sizeof(uint32_t));

        if (!values) {
            good = false;

            break;
        }

        active = values[3];

        r->core.UnmapBuffer(counter_readback);
        completed += block;

        const uint32_t progress_total = active ? max_samples : completed;

        if (progress && !progress(completed, progress_total, active)) good = false;

        SDL_Log("B: probe wavefront %u/%u spp | %u active probes", completed, max_samples, active);
    }

    if (good && uniforms.emissive_samples && tree->emissive_weight > 0.0f) {
        NriCommandAllocator *allocator = NULL;
        NriCommandBuffer *cmd = NULL;
        good = begin_commands(r, &allocator, &cmd) == NriResult_SUCCESS;

        if (good) {
            PROBE_WAVEFRONT_BUFFER *emissive_reads[] = {&position_buffer, &packed_nodes, &packed_triangles};
            PROBE_WAVEFRONT_BUFFER *emissive_writes[] = {&coefficients};
            good = probe_wavefront_dispatch(r, cmd, &pipelines.emissive, emissive_reads, emissive_writes, &uniforms, probe_groups);

            if (good) good = submit_commands(r, allocator, cmd);
            else abort_commands(r, allocator, cmd);
        }
    }

    if (good) {
        NriCommandAllocator *allocator = NULL;
        NriCommandBuffer *cmd = NULL;
        good = begin_commands(r, &allocator, &cmd) == NriResult_SUCCESS;

        if (good) {
            good = probe_wavefront_copy_to_readback(r, cmd, &coefficients, output_readback, output_bytes);

            if (good) good = submit_commands(r, allocator, cmd);
            else abort_commands(r, allocator, cmd);
        }
    }

    if (good) {
        const float (*values)[4] = r->core.MapBuffer(output_readback, 0, output_bytes);

        if (!values) good = false;
        else {
            for (uint32_t i = 0; i < probe_count; ++i) {
                for (uint32_t coefficient = 0; coefficient < 9u; ++coefficient)
                    memcpy(grid->probes[i].coefficients[coefficient], values[i * 9u + coefficient], sizeof(float[4]));
                grid->probes[i].position[3] = values[i * 9u][3];
            }

            r->core.UnmapBuffer(output_readback);
        }
    }

    if (scratch_ready) {
        scratch->packed_nodes = packed_nodes;
        scratch->packed_triangles = packed_triangles;
        scratch->states_a = states_a;
        scratch->states_b = states_b;
        scratch->results = results;
        scratch->accums = accums;
        scratch->coefficients = coefficients;
        scratch->counters = counters;
        scratch->dispatch_args = dispatch_args;
    }

    probe_wavefront_release_buffer(r, &sun_beams);
    probe_wavefront_release_buffer(r, &source_triangles);
    probe_wavefront_release_buffer(r, &source_nodes);
    probe_wavefront_release_buffer(r, &position_buffer);

    return good;
}

bool bake_probe_grid(RENDERER *r, PROBE_GRID *grid, Uint32 samples) {
    if (!r || !grid || !grid->probes) return false;

    const uint64_t count = (uint64_t)grid->count_x * grid->count_y * grid->count_z;

    if (!count || count > UINT32_MAX) return false;

    const uint64_t output_size = count * 9u * sizeof(float[4]);
    const uint64_t input_size = count * sizeof(float[4]);

    if (output_size > UINT32_MAX || input_size > UINT32_MAX) return false;

    const Uint32 output_bytes = (Uint32)output_size;
    const Uint32 input_bytes = (Uint32)input_size;

    float (*positions)[4] = malloc(input_bytes);

    if (!positions) return false;

    for (uint32_t i = 0; i < (uint32_t)count; ++i)
        memcpy(positions[i], grid->probes[i].position, sizeof(positions[i]));

    NriBuffer *input = upload_buffer(r, NriBufferUsageBits_SHADER_RESOURCE, positions, input_bytes, sizeof(float[4]));

    free(positions);

    const NriBufferDesc output_desc = {
        .size = output_bytes,
        .structureStride = sizeof(float[4]),
        .usage = NriBufferUsageBits_SHADER_RESOURCE_STORAGE
    };

    NriBuffer *output = NULL;

    if (r->core.CreateCommittedBuffer(r->device, NriMemoryLocation_DEVICE, 1.0f, &output_desc, &output) != NriResult_SUCCESS) output = NULL;

    NriPipeline *pipeline = compile_compute(r, r->probe_layout, "shaders/compute.hlsl", "probe_cs", "BUILD_PROBE_CS");

    bool good = input && output && pipeline;

    if (good) {
        NriCommandAllocator *allocator = NULL;
        NriCommandBuffer *cmd = NULL;
        good = begin_commands(r, &allocator, &cmd) == NriResult_SUCCESS;

        if (good) {
            const BAKE_UNIFORMS u = bake_data(r, 0u, 0u, samples, 0u, 0u);

            good = bind_probe_resources(r, cmd, input, r->bvh_node_buffer, r->bvh_triangle_buffer, output, &u, sizeof(u));

            if (good) {
                r->core.CmdSetPipeline(cmd, pipeline);

                r->core.CmdDispatch(cmd, &(NriDispatchDesc){
                    .workGroupNumX = (Uint32)count,
                    .workGroupNumY = 1,
                    .workGroupNumZ = 1
                });

                good = submit_commands(r, allocator, cmd);
            } else {
                abort_commands(r, allocator, cmd);
            }
        }
    }

    if (good) good = read_buffer(r, output, grid, output_bytes, (uint32_t)count);

    if (pipeline) r->core.DestroyPipeline(pipeline);
    release_buffer(r, output);
    release_buffer(r, input);

    return good;
}

void release_bake_resources(RENDERER *r) {
    if (!r || !r->device) return;

    release_buffer(r, r->bvh_node_buffer);
    release_buffer(r, r->bvh_triangle_buffer);
    release_buffer(r, r->lightmap_sample_buffer);
    release_buffer(r, r->lightmap_full_sample_buffer);
    release_buffer(r, r->lightmap_sparse_sample_buffer);
    release_buffer(r, r->lightmap_probe_buffer);
    release_buffer(r, r->lightmap_patch_map_buffer);
    release_buffer(r, r->lightmap_patch_anchor_buffer);

    for (uint32_t i = 0; i < 2u; ++i) {
        release_buffer(r, r->lightmap_active_buffer[i]);
        release_buffer(r, r->lightmap_active_count[i]);
    }

    release_buffer(r, r->lightmap_dispatch_args);
    release_texture(r, r->lightmap_scratch);
    release_texture(r, r->lightmap_direct);

    if (r->bake_pipeline) r->core.DestroyPipeline(r->bake_pipeline);

    if (r->lightmap_queue_reset_pipeline) r->core.DestroyPipeline(r->lightmap_queue_reset_pipeline);

    if (r->lightmap_queue_args_pipeline) r->core.DestroyPipeline(r->lightmap_queue_args_pipeline);

    r->bvh_node_buffer = NULL;
    r->bvh_triangle_buffer = NULL;
    r->bvh_triangle_count = 0u;
    r->bvh_emissive_weight = 0.0f;
    r->lightmap_sample_buffer = NULL;
    r->lightmap_full_sample_buffer = NULL;
    r->lightmap_sparse_sample_buffer = NULL;
    r->lightmap_probe_buffer = NULL;
    r->lightmap_patch_map_buffer = NULL;
    r->lightmap_patch_anchor_buffer = NULL;

    for (uint32_t i = 0; i < 2u; ++i) {
        r->lightmap_active_buffer[i] = NULL;
        r->lightmap_active_count[i] = NULL;
    }

    r->lightmap_dispatch_args = NULL;
    r->lightmap_active_capacity = 0u;
    r->lightmap_scratch = NULL;
    r->lightmap_direct = NULL;
    r->bake_pipeline = NULL;
    r->lightmap_queue_reset_pipeline = NULL;
    r->lightmap_queue_args_pipeline = NULL;
}

static bool dispatch_one(
    FX_STATE *fx, NriCommandBuffer *cmd, NriPipeline *pipeline, NriTexture *source, NriTexture *destination, const void *uniforms, Uint32 uniform_size, Uint32 width, Uint32 height
) {
    if (!fx || !fx->owner || !cmd || !pipeline || !destination) return false;

    RENDERER *r = fx->owner;

    if (!bind_fx_resources(r, cmd, pipeline, source, destination, fx->sampler, uniforms, uniform_size)) return false;

    r->core.CmdSetPipeline(cmd, pipeline);
    r->core.CmdDispatch(cmd, &(NriDispatchDesc){
        .workGroupNumX = (width + 7u) / 8u,
        .workGroupNumY = (height + 7u) / 8u,
        .workGroupNumZ = 1
    });

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

    if (fx->sampler) r->core.DestroyDescriptor(fx->sampler);

    if (fx->depth_sampler) r->core.DestroyDescriptor(fx->depth_sampler);

    if (fx->compose_pipeline) r->core.DestroyPipeline(fx->compose_pipeline);

    if (fx->ssao_pipeline) r->core.DestroyPipeline(fx->ssao_pipeline);

    if (fx->bloom_pipeline) r->core.DestroyPipeline(fx->bloom_pipeline);

    if (fx->grade_pipeline) r->core.DestroyPipeline(fx->grade_pipeline);

    if (fx->volume_pipeline) r->core.DestroyPipeline(fx->volume_pipeline);

    if (fx->volume_compose_pipeline) r->core.DestroyPipeline(fx->volume_compose_pipeline);

    memset(fx, 0, sizeof(*fx));
}

static bool make_compose_pipeline(RENDERER *r, NriShaderDesc *vs, NriShaderDesc *ps, NriFormat swap_format, NriPipeline **out) {
    const NriColorAttachmentDesc target = {
        .format = swap_format,
        .colorWriteMask = NriColorWriteBits_RGBA
    };

    const NriShaderDesc shaders[2] = {*vs, *ps};
    const NriGraphicsPipelineDesc desc = {.pipelineLayout = r->compose_layout,
                                          .inputAssembly = {.topology = NriTopology_TRIANGLE_LIST},
                                          .rasterization = {.fillMode = NriFillMode_SOLID, .cullMode = NriCullMode_NONE, .frontCounterClockwise = true, .depthClamp = false},
                                          .outputMerger = {.colors = &target, .colorNum = 1},
                                          .shaders = shaders,
                                          .shaderNum = 2,
                                          .cache = r->pipeline_cache};

    return r->core.CreateGraphicsPipeline(r->device, &desc, out) == NriResult_SUCCESS;
}

static bool fx_init(FX_STATE *fx, RENDERER *r) {
    if (!fx || !r || !r->device) return false;

    NriShaderDesc vs = {0};
    NriShaderDesc ps = {0};
    NriCommandAllocator *allocator = NULL;
    NriCommandBuffer *cmd = NULL;

    memset(fx, 0, sizeof(*fx));
    fx->owner = r;

    fx->sampler = sampler(r, NriFilter_LINEAR, NriFilter_LINEAR, NriAddressMode_CLAMP_TO_EDGE);

    fx->depth_sampler = sampler(r, NriFilter_NEAREST, NriFilter_NEAREST, NriAddressMode_CLAMP_TO_EDGE);

    if (!fx->sampler || !fx->depth_sampler) goto fail;

    vs = compile_shader("shaders/vertex.hlsl", "fullscreen_vs", "BUILD_FULLSCREEN_VS", NriStageBits_VERTEX_SHADER);

    ps = compile_shader("shaders/fragment.hlsl", "compose_fs", "BUILD_COMPOSE_FS", NriStageBits_FRAGMENT_SHADER);

    fx->ssao_pipeline = compile_compute(r, r->ssao_layout, "shaders/compute.hlsl", "ssao_cs", "BUILD_SSAO_CS");

    fx->bloom_pipeline = compile_compute(r, r->bloom_layout, "shaders/compute.hlsl", "bloom_cs", "BUILD_BLOOM_CS");

    fx->grade_pipeline = compile_compute(r, r->grade_layout, "shaders/compute.hlsl", "grade_cs", "BUILD_GRADE_CS");

    fx->volume_pipeline = compile_compute(r, r->volume_layout, "shaders/vision_compute.hlsl", "volume_cs", "BUILD_VISION_VOLUME_CS");

    fx->volume_compose_pipeline = compile_compute(r, r->volume_compose_layout, "shaders/vision_compute.hlsl", "volume_compose_cs", "BUILD_VISION_COMPOSE_CS");

    if (!vs.bytecode || !ps.bytecode || !fx->ssao_pipeline || !fx->bloom_pipeline || !fx->grade_pipeline || !fx->volume_pipeline || !fx->volume_compose_pipeline) goto fail;

    if (!make_compose_pipeline(r, &vs, &ps, r->swapchain_format, &fx->compose_pipeline)) goto fail;

    free_shader(&vs);
    free_shader(&ps);

    fx->lut = create_texture(r, NriFormat_RGBA16_SFLOAT, NriTextureUsageBits_SHADER_RESOURCE | NriTextureUsageBits_SHADER_RESOURCE_STORAGE, 256u, 16u);

    if (!fx->lut) goto fail;

    if (begin_commands(r, &allocator, &cmd) != NriResult_SUCCESS || !dispatch_one(fx, cmd, fx->grade_pipeline, NULL, fx->lut, NULL, 0, 256u, 16u)) goto fail;

    const bool submitted = submit_commands(r, allocator, cmd);
    cmd = NULL;
    allocator = NULL;

    if (!submitted) goto fail;

    return true;

fail:
    free_shader(&vs);
    free_shader(&ps);
    abort_commands(r, allocator, cmd);
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

    fx->hdr = create_texture(r, NriFormat_RGBA16_SFLOAT, rt, width, height);
    fx->normal_depth = create_texture(r, NriFormat_RGBA16_SFLOAT, rt, width, height);
    fx->ao = create_texture(r, NriFormat_RGBA16_SFLOAT, compute, fx->ao_width, fx->ao_height);
    fx->bloom_a = create_texture(r, NriFormat_RGBA16_SFLOAT, compute, fx->ao_width, fx->ao_height);
    fx->bloom_b = create_texture(r, NriFormat_RGBA16_SFLOAT, compute, fx->ao_width, fx->ao_height);
    fx->volume = create_texture(r, NriFormat_RGBA16_SFLOAT, compute, fx->ao_width, fx->ao_height);
    fx->lit = create_texture(r, NriFormat_RGBA16_SFLOAT, compute, width, height);

    if (!fx->hdr || !fx->normal_depth || !fx->ao || !fx->bloom_a || !fx->bloom_b || !fx->volume || !fx->lit) {
        release_frame_textures(fx);

        return false;
    }

    return true;
}

static bool fx_volume(FX_STATE *fx, NriCommandBuffer *cmd, NriBuffer *probes, NriBuffer *beams, const PROBE_GRID *grid, const BEAM_GRID *beam_grid, const RENDER_FRAME *frame) {
    if (!fx || !fx->owner || !cmd || !probes || !beams || !grid || !beam_grid || !grid->probes || !fx->volume) return false;

    RENDERER *r = fx->owner;

    const VOLUME_UNIFORMS u = {
        .eye_density = {frame->eye.x, frame->eye.y, frame->eye.z, 0.0f},
        .right_tan =
        {frame->right.x * frame->tan_half_fov * frame->aspect, frame->right.y * frame->tan_half_fov * frame->aspect, frame->right.z * frame->tan_half_fov * frame->aspect, 0},
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
        .volume_radii = {frame->vision.center_radius, frame->vision.middle_radius, frame->vision.center_transition_width, frame->vision.middle_transition_width},
        .volume_filter = {frame->vision.jitter_strength, frame->vision.volume_blur_strength, 0.0f, 0.0f},
        .volume_quality = {frame->vision.center_steps, frame->vision.middle_steps, frame->vision.peripheral_steps, 0u},
        .volume_strides = {frame->vision.center_stride, frame->vision.middle_stride, frame->vision.peripheral_stride, 0u}
    };

    if (!bind_volume_resources(r, cmd, fx->normal_depth, fx->depth_sampler, probes, beams, fx->volume, &u, sizeof(u))) return false;

    r->core.CmdSetPipeline(cmd, fx->volume_pipeline);

    r->core.CmdDispatch(cmd, &(NriDispatchDesc){
        .workGroupNumX = (fx->ao_width + 7u) / 8u,
        .workGroupNumY = (fx->ao_height + 7u) / 8u,
        .workGroupNumZ = 1
    });

    const VOLUME_COMPOSE_UNIFORMS compose = {
        .width = fx->width,
        .height = fx->height,
        .debug_view = fx->debug_view,
        .bypass_volume = 0u,
        .volume_radii = {frame->vision.center_radius, frame->vision.middle_radius, frame->vision.center_transition_width, frame->vision.middle_transition_width},
        .volume_filter = {frame->vision.jitter_strength, frame->vision.volume_blur_strength, 0.0f, 0.0f},
        .volume_strides = {frame->vision.center_stride, frame->vision.middle_stride, frame->vision.peripheral_stride, 0u},
    };

    if (!bind_volume_compose_resources(r, cmd, fx->hdr, fx->volume, fx->normal_depth, fx->sampler, fx->depth_sampler, fx->lit, &compose, sizeof(compose))) return false;

    r->core.CmdSetPipeline(cmd, fx->volume_compose_pipeline);

    r->core.CmdDispatch(cmd, &(NriDispatchDesc){
        .workGroupNumX = (fx->width + 7u) / 8u,
        .workGroupNumY = (fx->height + 7u) / 8u,
        .workGroupNumZ = 1
    });

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

    if (!bind_volume_compose_resources(r, cmd, fx->hdr, fx->volume, fx->normal_depth, fx->sampler, fx->depth_sampler, fx->lit, &compose, sizeof(compose))) return false;

    r->core.CmdSetPipeline(cmd, fx->volume_compose_pipeline);

    r->core.CmdDispatch(cmd, &(NriDispatchDesc){
        .workGroupNumX = (fx->width + 7u) / 8u,
        .workGroupNumY = (fx->height + 7u) / 8u,
        .workGroupNumZ = 1
    });

    return true;
}

static bool bloom_pass(FX_STATE *fx, NriCommandBuffer *cmd, NriTexture *source, NriTexture *destination, Uint32 src_width, Uint32 src_height, Uint32 phase) {
    const BLOOM_UNIFORMS u = {
        .src_width = src_width,
        .src_height = src_height,
        .dst_width = fx->ao_width,
        .dst_height = fx->ao_height,
        .phase = phase,
        .threshold = 1.0f,
        .knee = 0.55f,
        .strength = 1.0f * 1e2
    };

    return dispatch_one(fx, cmd, fx->bloom_pipeline, source, destination, &u, sizeof(u), fx->ao_width, fx->ao_height);
}

static bool fx_apply_base(FX_STATE *fx, NriCommandBuffer *cmd, NriTexture *swap, float tan_half_fov, float aspect) {
    if (!fx || !fx->owner || !cmd || !swap || !fx->hdr || !fx->normal_depth) return false;

    RENDERER *r = fx->owner;

    const SSAO_UNIFORMS ao = {
        .width = fx->width,
        .height = fx->height,
        .ao_width = fx->ao_width,
        .ao_height = fx->ao_height,
        .tan_half_fov = tan_half_fov,
        .aspect = aspect,
        .radius = 0.65f,
        .bias = 0.035f
    };

    NriTexture *hdr = fx->volume_ready ? fx->lit : fx->hdr;

    if (!dispatch_one(fx, cmd, fx->ssao_pipeline, fx->normal_depth, fx->ao, &ao, sizeof(ao), fx->ao_width, fx->ao_height) ||
        !bloom_pass(fx, cmd, fx->hdr, fx->bloom_a, fx->width, fx->height, 0u) || !bloom_pass(fx, cmd, fx->bloom_a, fx->bloom_b, fx->ao_width, fx->ao_height, 1u) ||
        !bloom_pass(fx, cmd, fx->bloom_b, fx->bloom_a, fx->ao_width, fx->ao_height, 2u))
        return false;

    const COMPOSE_UNIFORMS u = {
        .exposure = 1.0f,
        .ao_strength = fx->debug_view >= 3u ? 0.0f : 0.62f,
        .bloom_strength = fx->debug_view >= 3u ? 0.0f : 0.22f
    };

    if (!begin_compose_rendering(r, cmd, swap, hdr, fx->ao, fx->bloom_a, fx->lut, fx->sampler, &u, sizeof(u))) return false;

    r->core.CmdSetPipeline(cmd, fx->compose_pipeline);
    r->core.CmdDraw(cmd, &(NriDrawDesc){
        .vertexNum = 3,
        .instanceNum = 1,
        .baseVertex = 0,
        .baseInstance = 0
    });

    r->core.CmdEndRendering(cmd);

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
static bool create_pipeline_layouts(renderer *r) {
    return create_surface_layout(r) && create_line_layout(r) && create_sky_layout(r) && create_bake_layout(r) && create_lightmap_queue_layouts(r) && create_probe_layout(r) &&
           create_ssao_layout(r) && create_bloom_layout(r) && create_grade_layout(r) && create_volume_layout(r) && create_volume_compose_layout(r) && create_compose_layout(r);
}

static void destroy_pipeline_layouts(RENDERER *r) {
    if (!r) return;

    NriPipelineLayout **layouts[] = {&r->surface_layout,
                                     &r->line_layout,
                                     &r->sky_layout,
                                     &r->bake_layout,
                                     &r->lightmap_queue_reset_layout,
                                     &r->lightmap_queue_args_layout,
                                     &r->probe_layout,
                                     &r->ssao_layout,
                                     &r->bloom_layout,
                                     &r->grade_layout,
                                     &r->volume_layout,
                                     &r->volume_compose_layout,
                                     &r->compose_layout};

    for (uint32_t i = 0; i < sizeof(layouts) / sizeof(layouts[0]); ++i) {
        if (*layouts[i]) {
            r->core.DestroyPipelineLayout(*layouts[i]);
            *layouts[i] = NULL;
        }
    }
}

bool r_init(RENDERER *r, const char *title, int width, int height) {
    if (!r) return false;

    memset(r, 0, sizeof(*r));
    r->show_volume = true;
    r->yaw = -0.78f;
    r->pitch = 0.34f;
    r->distance = 14.0f;
    r->target = v3(0.0f, 1.0f, 0.0f);

    r->window = SDL_CreateWindow(title, width, height,
                                 SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY
#if defined(__APPLE__)
            | SDL_WINDOW_METAL
#else
            | SDL_WINDOW_VULKAN
#endif
    );
    if (!r->window) {
        r_deinit(r);

        return false;
    }

    NriDeviceCreationDesc device_desc = {0};
#if defined(__APPLE__)
    device_desc.graphicsAPI = NriGraphicsAPI_VK;
#else
    device_desc.graphicsAPI = NriGraphicsAPI_VK;
#endif
    device_desc.enableNRIValidation = false;
    device_desc.enableGraphicsAPIValidation = false;
    device_desc.vkBindingOffsets = (NriVKBindingOffsets){
        .sRegister = 0,
        .tRegister = 16,
        .bRegister = 32,
        .uRegister = 48
    };

    if (nriCreateDevice(&device_desc, &r->device) != NriResult_SUCCESS) {
        SDL_Log("NRI device creation failed");
        r_deinit(r);

        return false;
    }

    if (nriGetInterface(r->device, NRI_INTERFACE(NriCoreInterface), &r->core) != NriResult_SUCCESS ||
        nriGetInterface(r->device, NRI_INTERFACE(NriHelperInterface), &r->helper) != NriResult_SUCCESS ||
        nriGetInterface(r->device, NRI_INTERFACE(NriSwapChainInterface), &r->swapchain_api) != NriResult_SUCCESS) {
        SDL_Log("NRI interface acquisition failed");
        r_deinit(r);

        return false;
    }

    if (!acquire_queues(r)) {
        r_deinit(r);

        return false;
    }

    if (!create_pipeline_cache(r) || !create_gpu_timestamps(r) || !create_swapchain(r, width, height) || !create_descriptor_pool(r) || !create_work_contexts(r) ||
        !create_frame_contexts(r) || !create_pipeline_layouts(r)) {
        r_deinit(r);

        return false;
    }

    if (r->core.GetFormatSupport(r->device, NriFormat_D32_SFLOAT) & NriFormatSupportBits_DEPTH_STENCIL_ATTACHMENT) {
        r->depth_format = NriFormat_D32_SFLOAT;
    } else if (r->core.GetFormatSupport(r->device, NriFormat_D24_UNORM_S8_UINT) & NriFormatSupportBits_DEPTH_STENCIL_ATTACHMENT) {
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
        r_deinit(r);

        return false;
    }

    r->solid_pipeline = make_surface_pipeline(r, &r->core, r->surface_layout, &surface_vs, &surface_ps);
    r->line_pipeline = make_line_pipeline(r, r->line_layout, &line_vs, &line_ps);
    r->sky_pipeline = make_sky_pipeline(r, r->sky_layout, &sky_vs, &sky_ps);

    free_shader(&surface_vs);
    free_shader(&surface_ps);
    free_shader(&line_vs);
    free_shader(&line_ps);
    free_shader(&sky_vs);
    free_shader(&sky_ps);

    if (!r->solid_pipeline || !r->line_pipeline || !r->sky_pipeline || !fx_init(&r->fx, r)) {
        r_deinit(r);

        return false;
    }

    SDL_Log("GPU backend: NRI");
    SDL_Log("depth format: %s", r->depth_format == NriFormat_D32_SFLOAT ? "D32_FLOAT" : r->depth_format == NriFormat_D24_UNORM_S8_UINT ? "D24S8" : "D16_UNORM");
    SDL_Log("SDL_image: %d", IMG_Version());

    return true;
}

bool draw_frame(RENDERER *r, const RENDER_FRAME *frame) {
    if (!r || !frame || !r->device || !r->solid_pipeline || !r->sky_pipeline || !r->vertex_buffer || !r->lightmap_texture || !r->lightmap_sampler) return false;

    uint32_t width = 0;
    uint32_t height = 0;

    SDL_GetWindowSizeInPixels(r->window, (int *)&width, (int *)&height);

    if (!width || !height) return true;

    if (!r->swapchain || width != r->swapchain_width || height != r->swapchain_height) {
        if (r->core.QueueWaitIdle(r->graphics_queue) != NriResult_SUCCESS) return false;
        destroy_swapchain(r);

        if (!create_swapchain(r, width, height)) return false;
    }

    if (!fx_ensure(&r->fx, width, height) || !ensure_depth_texture(r, width, height)) return false;

    uint32_t swap_index = 0;

    if (!acquire_swapchain_texture(r, &swap_index)) {
        destroy_swapchain(r);

        return false;
    }

    r->current_swap_index = swap_index;

    NriTexture *swap = r->swapchain_textures[swap_index];

    FRAME_CONTEXT *queued_frame = NULL;
    NriCommandBuffer *cmd = NULL;

    if (!begin_frame_commands(r, &queued_frame, &cmd)) goto failed_frame;

    CAMERA_UNIFORMS camera = {0};

    memcpy(camera.mvp, frame->mvp, sizeof(camera.mvp));
    memcpy(camera.view, frame->view, sizeof(camera.view));

    const SKY_UNIFORMS sky = {
        .camera_right =
        {frame->right.x * frame->tan_half_fov * frame->aspect, frame->right.y * frame->tan_half_fov * frame->aspect, frame->right.z * frame->tan_half_fov * frame->aspect, 0},
        .camera_up = {frame->up.x * frame->tan_half_fov, frame->up.y * frame->tan_half_fov, frame->up.z * frame->tan_half_fov, 0},
        .camera_forward = {frame->forward.x, frame->forward.y, frame->forward.z, 0},
        .sky_zenith = {frame->sky.zenith.x, frame->sky.zenith.y, frame->sky.zenith.z, frame->sky.intensity},
        .sky_horizon = {frame->sky.horizon.x, frame->sky.horizon.y, frame->sky.horizon.z, 1.0f},
        .sun_direction_intensity = {frame->sun.direction.x, frame->sun.direction.y, frame->sun.direction.z, frame->sun.intensity},
        .sun_color_radius = {frame->sun.color.x, frame->sun.color.y, frame->sun.color.z, frame->sun.angular_radius}
    };

    if (!begin_scene_rendering(r, cmd, r->fx.hdr, r->fx.normal_depth, r->depth_texture, width, height)) goto failed_frame;

    if (!bind_sky_resources(r, cmd, &sky, sizeof(sky))) goto failed_frame;
    r->core.CmdSetPipeline(cmd, r->sky_pipeline);

    r->core.CmdDraw(cmd, &(NriDrawDesc){
        .vertexNum = 3,
        .instanceNum = 1,
        .baseVertex = 0,
        .baseInstance = 0
    });

    const NriVertexBufferDesc vertex = {
        .buffer = r->vertex_buffer,
        .offset = 0,
        .stride = sizeof(RENDER_VERTEX)
    };

    r->core.CmdSetVertexBuffers(cmd, 0, &vertex, 1);
    r->core.CmdSetPipeline(cmd, r->solid_pipeline);

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
            .camera_position = {frame->eye.x, frame->eye.y, frame->eye.z, r->debug_view == 1u ? 2.0f : (r->has_bake ? 1.0f : 0.0f)}
        };

        if (!bind_surface_resources(r, cmd, m, r->lightmap_texture, r->material_sampler, r->lightmap_sampler, &material, sizeof(material))) goto failed_frame;

        r->core.CmdDraw(cmd, &(NriDrawDesc){
            .vertexNum = draw->count,
            .instanceNum = 1,
            .baseVertex = draw->first,
            .baseInstance = 0
        });
    }

    if (r->show_debug && r->debug_vertex_count) {
        r->core.CmdSetPipeline(cmd, r->line_pipeline);

        if (!bind_line_resources(r, cmd, camera.mvp, sizeof(camera.mvp))) goto failed_frame;

        r->core.CmdDraw(cmd, &(NriDrawDesc){
            .vertexNum = r->debug_vertex_count,
            .instanceNum = 1,
            .baseVertex = r->debug_vertex_start,
            .baseInstance = 0
        });
    }

    r->core.CmdEndRendering(cmd);

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

bool bake_worker_init(RENDERER *r) {
    if (!r) return false;
    memset(r, 0, sizeof(*r));

    NriDeviceCreationDesc device_desc = {0};

    device_desc.graphicsAPI = NriGraphicsAPI_VK;
    device_desc.vkBindingOffsets = (NriVKBindingOffsets){
        .sRegister = 0,
        .tRegister = 16,
        .bRegister = 32,
        .uRegister = 48
    };

    if (nriCreateDevice(&device_desc, &r->device) != NriResult_SUCCESS) return false;

    if (nriGetInterface(r->device, NRI_INTERFACE(NriCoreInterface), &r->core) != NriResult_SUCCESS ||
        nriGetInterface(r->device, NRI_INTERFACE(NriHelperInterface), &r->helper) != NriResult_SUCCESS || !acquire_queues(r)) {
        bake_worker_deinit(r);

        return false;
    }

    if (!create_pipeline_cache(r) || !create_gpu_timestamps(r) || !create_descriptor_pool(r) || !create_work_contexts(r) || !create_pipeline_layouts(r)) {
        bake_worker_deinit(r);

        return false;
    }

    r->lightmap_sampler = sampler(r, NriFilter_LINEAR, NriFilter_LINEAR, NriAddressMode_CLAMP_TO_EDGE);

    r->lightmap_texture = create_texture(r, NriFormat_RGBA16_SFLOAT, NriTextureUsageBits_SHADER_RESOURCE, 1, 1);

    if (!r->lightmap_sampler || !r->lightmap_texture) {
        bake_worker_deinit(r);

        return false;
    }

    return true;
}

void bake_worker_deinit(RENDERER *r) {
    if (!r) return;

    free_probe_grid(&r->volume_probes);
    beam_free(&r->beams);

    if (r->device) {
        if (r->graphics_queue) r->core.QueueWaitIdle(r->graphics_queue);

        destroy_upload_context(r);
        destroy_work_contexts(r);
        clear_temporary(r);

        probe_wavefront_scratch_destroy(r);
        release_bake_resources(r);
        destroy_gpu_timestamps(r);
        release_buffer(r, r->volume_probe_buffer);
        release_buffer(r, r->beam_buffer);
        release_texture(r, r->lightmap_texture);

        if (r->lightmap_sampler) r->core.DestroyDescriptor(r->lightmap_sampler);

        destroy_pipeline_layouts(r);

        if (r->descriptor_pool) r->core.DestroyDescriptorPool(r->descriptor_pool);
        destroy_pipeline_cache(r);

        nriDestroyDevice(r->device);
    }

    free(r->temporary_descriptors);
    free(r->temporary_buffers);
    free(r->texture_states);

    memset(r, 0, sizeof(*r));
}

void r_deinit(RENDERER *r) {
    if (!r) return;

    free(r->vertices);
    free(r->draws);
    free_probe_grid(&r->volume_probes);
    beam_free(&r->beams);

    if (r->device) {
        if (r->graphics_queue) r->core.QueueWaitIdle(r->graphics_queue);

        destroy_upload_context(r);
        destroy_work_contexts(r);
        clear_temporary(r);
        destroy_frame_contexts(r);
        probe_wavefront_scratch_destroy(r);
        destroy_gpu_timestamps(r);

        fx_deinit(&r->fx);

        if (r->image_textures) {
            for (uint32_t i = 0; i < r->image_texture_count; ++i)
                release_texture(r, r->image_textures[i]);
        }

        free(r->image_textures);
        free(r->materials);

        release_texture(r, r->default_white);
        release_texture(r, r->default_normal);

        if (r->material_sampler) r->core.DestroyDescriptor(r->material_sampler);

        release_buffer(r, r->vertex_buffer);
        release_bake_resources(r);
        release_buffer(r, r->volume_probe_buffer);
        release_buffer(r, r->beam_buffer);
        release_texture(r, r->depth_texture);
        release_texture(r, r->lightmap_texture);

        if (r->lightmap_sampler) r->core.DestroyDescriptor(r->lightmap_sampler);

        if (r->sky_pipeline) r->core.DestroyPipeline(r->sky_pipeline);

        if (r->solid_pipeline) r->core.DestroyPipeline(r->solid_pipeline);

        if (r->line_pipeline) r->core.DestroyPipeline(r->line_pipeline);

        destroy_swapchain(r);
        destroy_pipeline_layouts(r);

        if (r->descriptor_pool) r->core.DestroyDescriptorPool(r->descriptor_pool);
        destroy_pipeline_cache(r);

        nriDestroyDevice(r->device);
    }

    free(r->temporary_descriptors);
    free(r->temporary_buffers);
    free(r->texture_states);

#if defined(__APPLE__)
    if (r->metal_view) SDL_Metal_DestroyView(r->metal_view);
#endif

    if (r->window) SDL_DestroyWindow(r->window);

    memset(r, 0, sizeof(*r));
}
