#include "dustmite.h"

#include <SDL3_image/SDL_image.h>
#include <SDL3_shadercross/SDL_shadercross.h>

#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BAKE_TARGET_SAMPLES 128u
#define BAKE_MAX_BOUNCES 3u
#define BAKE_BATCH_SAMPLES 8u
#define BAKE_DILATION_PASSES 3u
#define PHASE_CLEAR 0u
#define PHASE_TRACE 1u
#define PHASE_FILTER 2u
#define PHASE_DILATE 3u

typedef struct camera_uniforms {
    float mvp[16];
    float view[16];
} camera_uniforms;

typedef struct bake_uniforms {
    Uint32 item_count;
    Uint32 lightmap_width;
    Uint32 lightmap_height;
    Uint32 dispatch_width;
    Uint32 iteration;
    Uint32 phase;
    Uint32 max_bounces;
    Uint32 padding;
    float sun_direction_intensity[4];
    float sun_color_radius[4];
    float sky_zenith[4];
    float sky_horizon[4];
    float bake_params[4];
} bake_uniforms;

typedef struct sky_uniforms {
    float camera_right[4];
    float camera_up[4];
    float camera_forward[4];
    float sky_zenith[4];
    float sky_horizon[4];
    float sun_direction_intensity[4];
    float sun_color_radius[4];
} sky_uniforms;

typedef struct material_uniforms {
    float base_color_factor[4];
    float emissive_metallic[4];
    float roughness_normal_ao_sun[4];
    float sun_direction[4];
    float sun_color[4];
    float camera_position[4];
} material_uniforms;

typedef struct ssao_uniforms {
    Uint32 width, height, ao_width, ao_height;
    float tan_half_fov, aspect, radius, bias;
} ssao_uniforms;

typedef struct bloom_uniforms {
    Uint32 src_width, src_height, dst_width, dst_height;
    Uint32 phase, _pad0, _pad1, _pad2;
    float threshold, knee, strength, _pad3;
} bloom_uniforms;

typedef struct compose_uniforms {
    float exposure;
    float ao_strength;
    float bloom_strength;
    float _pad;
} compose_uniforms;

typedef struct volume_uniforms {
    float eye_density[4], right_tan[4], up_tan[4], forward_g[4];
    float sun_intensity[4], grid_origin_spacing[4];
    Uint32 grid_dims_width[4], height_debug[4];
    float beam_origin[4], beam_step[4];
} volume_uniforms;

struct render_material {
    gltf_material data;
    TEXTURE *base_color;
    TEXTURE *metallic_roughness;
    TEXTURE *normal;
    TEXTURE *occlusion;
    TEXTURE *emissive;
};

static char *shader_source(const char *path) {
    size_t size = 0;
    char *body = SDL_LoadFile(path, &size);
    if (!body) {
        SDL_Log("shader source load failed for %s: %s", path, SDL_GetError());
        return NULL;
    }
#ifdef DUSTMITE_GPU_SDL
    const char *bindings =
        "#define GPU_BIND_S(n,s)\n#define GPU_BIND_T(n,s)\n"
        "#define GPU_BIND_B(n,s)\n#define GPU_BIND_U(n,s)\n"
        "#define GPU_STORAGE_RGBA16F\n";
#else
    const char *bindings =
        "#define GPU_BIND_S(n,s) [[vk::binding(n, s)]]\n"
        "#define GPU_BIND_T(n,s) [[vk::binding(n + 16, s)]]\n"
        "#define GPU_BIND_B(n,s) [[vk::binding(n + 32, s)]]\n"
        "#define GPU_BIND_U(n,s) [[vk::binding(n + 48, s)]]\n"
        "#define GPU_STORAGE_RGBA16F [[vk::image_format(\"rgba16f\")]]\n";
#endif
    const size_t prefix = strlen(bindings);
    char *source = SDL_malloc(prefix + size + 1);
    if (source) {
        memcpy(source, bindings, prefix);
        memcpy(source + prefix, body, size);
        source[prefix + size] = '\0';
    }
    SDL_free(body);
    return source;
}

static vec3 sun_direction(void) {
    return v3_normalize(v3(0.38f, 0.30f, 0.32f));
}

static void free_probe_grid(dm_probe_grid *grid) {
    if (!grid) return;
    free(grid->probes);
    memset(grid, 0, sizeof(*grid));
}

#ifndef DUSTMITE_GPU_SDL
static SDL_ShaderCross_ShaderStage get_shadercross_stage(NriStageBits stage) {


    if (stage == NriStageBits_VERTEX_SHADER)
        return SDL_SHADERCROSS_SHADERSTAGE_VERTEX;

    if (stage == NriStageBits_FRAGMENT_SHADER)
        return SDL_SHADERCROSS_SHADERSTAGE_FRAGMENT;

    if (stage == NriStageBits_COMPUTE_SHADER)
        return SDL_SHADERCROSS_SHADERSTAGE_COMPUTE;

    SDL_Log("unsupported NRI shader stage: %u", (unsigned)stage);

    return SDL_SHADERCROSS_SHADERSTAGE_VERTEX;

}


static Uint8 *compile_spirv(const char *path, 
                            const char *entrypoint,
                            const char *define,
                            NriStageBits stage, 
                            size_t *size) {

    char *source = shader_source(path);
    if (!source) {
        SDL_Log("shader source load failed for %s: %s", path, SDL_GetError());
        return NULL;
    }

    SDL_ShaderCross_HLSL_Define defines[2] = {
        {.name = (char *)define, .value = NULL}, {0}
    };


    const SDL_ShaderCross_HLSL_Info hlsl = {
        .source = source,
        .entrypoint = entrypoint,
        .include_dir = "shaders",
        .defines = defines,
        .shader_stage = get_shadercross_stage(stage),
        .props = 0
    };

    Uint8 *spirv = SDL_ShaderCross_CompileSPIRVFromHLSL(&hlsl, size);

    SDL_free(source);

    if (!spirv) {
        SDL_Log("shadercross HLSL->SPIR-V failed for %s:%s: %s",
                path, entrypoint, SDL_GetError());
    }

    return spirv;
}

static NriShaderDesc compile_shader(const char *path,
                                    const char *entrypoint,
                                    const char *define,
                                    NriStageBits stage) {
    size_t spirv_size = 0;

    Uint8 *spirv = compile_spirv(
        path,
        entrypoint,
        define,
        stage,
        &spirv_size
    );
    if (!spirv) return (NriShaderDesc){0};


    return (NriShaderDesc){
        .stage = stage,
        .bytecode = spirv,
        .size = spirv_size,
        .entryPointName = entrypoint
    };
}

static NriPipeline *compile_compute(NriCoreInterface *core,
                                    DEVICE *device,
                                    NriPipelineLayout *layout,
                                    const char *path,
                                    const char *entrypoint,
                                    const char *define) {
    size_t spirv_size = 0;

    Uint8 *spirv = compile_spirv(
        path,
        entrypoint,
        define,
        NriStageBits_COMPUTE_SHADER,
        &spirv_size
    );

    if (!spirv) return NULL;

 
    NriShaderDesc shader = {
        .stage = NriStageBits_COMPUTE_SHADER,
        .bytecode = spirv,
        .size = spirv_size,
        .entryPointName = entrypoint
    };

    NriComputePipelineDesc desc = {
        .pipelineLayout = layout,
        .shader = shader
    };

    NriPipeline *pipeline = NULL;

    NriResult result = core->CreateComputePipeline(
        device,
        &desc,
        &pipeline
    );


    SDL_free(spirv);
    
    if (result != NriResult_SUCCESS) {
        SDL_Log(
            "compute pipeline creation failed for %s:%s",
            path,
            entrypoint
        );

        return NULL;
    }

    return pipeline;
}

static NriPipeline *make_surface_pipeline(renderer *r,
                                                NriCoreInterface *core,
                                                NriPipelineLayout *layout,
                                                const NriShaderDesc *vs,
                                                const NriShaderDesc *ps) {
    
    const NriVertexStreamDesc vb = {
        .bindingSlot = 0,
        .stepRate = NriVertexStreamStepRate_PER_VERTEX,
        .stride = (uint16_t)sizeof(render_vertex)
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
            .offset = (uint32_t)offsetof(render_vertex, x),
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
            .offset = (uint32_t)offsetof(render_vertex, nx),
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
            .offset = (uint32_t)offsetof(render_vertex, u),
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
            .offset = (uint32_t)offsetof(render_vertex, lu),
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
        },
        {
            .format = NriFormat_RGBA16_SFLOAT,
            .colorWriteMask = NriColorWriteBits_RGBA
        }
    };

    const NriMultisampleDesc multisample = {
        .sampleMask = NRI_ALL,
        .sampleNum = 1
    };

    const NriShaderDesc shaders[2] = {
        *vs,
        *ps
    };

    const NriGraphicsPipelineDesc desc = {
        .pipelineLayout = layout,

        .vertexInput = &vertex_input,

        .inputAssembly = {
            .topology = NriTopology_TRIANGLE_LIST
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
                .compareOp = NriCompareOp_LESS,
                .write = true
            },

            .depthStencilFormat = r->depth_format
        },

        .shaders = shaders,
        .shaderNum = 2
    };

    NriPipeline *pipeline = NULL;

    if (core->CreateGraphicsPipeline(
            r->device,
            &desc,
            &pipeline
        ) != NriResult_SUCCESS) {
        return NULL;
    }

    return pipeline;

}


static void clear_temporary(renderer *r);

static NriResult begin_commands(renderer *r, NriCommandAllocator **allocator,
                                NriCommandBuffer **command_buffer) {
    if (!r || !r->graphics_queue) return NriResult_INVALID_ARGUMENT;
    r->current_graphics_layout = r->current_compute_layout = NULL;

    NriResult result = r->core.CreateCommandAllocator(
        r->graphics_queue, allocator);
    if (result != NriResult_SUCCESS) return result;

    result = r->core.CreateCommandBuffer(*allocator, command_buffer);
    if (result != NriResult_SUCCESS) {
        r->core.DestroyCommandAllocator(*allocator);
        *allocator = NULL;
        return result;
    }

    result = r->core.BeginCommandBuffer(*command_buffer, r->descriptor_pool);
    if (result != NriResult_SUCCESS) {
        r->core.DestroyCommandBuffer(*command_buffer);
        r->core.DestroyCommandAllocator(*allocator);
        *command_buffer = NULL;
        *allocator = NULL;
    }

    return result;
}

static bool submit_commands(renderer *r, NriCommandAllocator *allocator,
                            NriCommandBuffer *command_buffer) {
    if (!r || !allocator || !command_buffer) return false;

    bool good = r->core.EndCommandBuffer(command_buffer) == NriResult_SUCCESS;
    if (good) {
        const NriQueueSubmitDesc submit = {
            .commandBuffers = (const NriCommandBuffer *const *)&command_buffer,
            .commandBufferNum = 1
        };
        good = r->core.QueueSubmit(r->graphics_queue, &submit) == NriResult_SUCCESS;
        if (good) good = r->core.QueueWaitIdle(r->graphics_queue) == NriResult_SUCCESS;
    }

    r->core.DestroyCommandBuffer(command_buffer);
    r->core.DestroyCommandAllocator(allocator);
    clear_temporary(r);
    return good;
}

static void abort_commands(renderer *r, NriCommandAllocator *allocator,
                           NriCommandBuffer *cmd) {
    if (cmd) r->core.DestroyCommandBuffer(cmd);
    if (allocator) r->core.DestroyCommandAllocator(allocator);
    clear_temporary(r);
}

/* Descriptor sets match the HLSL register spaces in shaders/. */
static bool make_layout(renderer *r, NriPipelineLayout **out,
                        const NriDescriptorType *types[4],
                        const uint8_t counts[4],
                        NriStageBits stages) {
    NriDescriptorRangeDesc ranges[4][16] = {0};
    NriDescriptorSetDesc sets[4] = {0};
    for (uint32_t set = 0; set < 4; ++set) {
        sets[set].registerSpace = set;
        sets[set].ranges = ranges[set];
        sets[set].rangeNum = counts[set];
        for (uint32_t i = 0; i < counts[set]; ++i) {
            NriDescriptorType type = types[set][i];
            uint32_t reg = 0;
            for (uint32_t j = 0; j < i; ++j)
                if (types[set][j] == type ||
                    (type == NriDescriptorType_STRUCTURED_BUFFER &&
                     types[set][j] == NriDescriptorType_TEXTURE)) ++reg;
            ranges[set][i] = (NriDescriptorRangeDesc){
                .baseRegisterIndex = reg, .descriptorNum = 1,
                .descriptorType = type, .shaderStages = stages
            };
        }
    }
    const NriPipelineLayoutDesc desc = {
        .descriptorSets = sets, .descriptorSetNum = 4,
        .rootRegisterSpace = 4,
        .shaderStages = stages
    };
    return r->core.CreatePipelineLayout(r->device, &desc, out) == NriResult_SUCCESS;
}

static bool create_pipeline_layouts(renderer *r);

static bool create_surface_layout(renderer *r) {
    static const NriDescriptorType camera[] = {NriDescriptorType_CONSTANT_BUFFER};
    static const NriDescriptorType material[] = {
        NriDescriptorType_TEXTURE, NriDescriptorType_TEXTURE,
        NriDescriptorType_TEXTURE, NriDescriptorType_TEXTURE,
        NriDescriptorType_TEXTURE, NriDescriptorType_TEXTURE,
        NriDescriptorType_SAMPLER, NriDescriptorType_SAMPLER,
        NriDescriptorType_SAMPLER, NriDescriptorType_SAMPLER,
        NriDescriptorType_SAMPLER, NriDescriptorType_SAMPLER
    };
    static const NriDescriptorType uniform[] = {NriDescriptorType_CONSTANT_BUFFER};
    const NriDescriptorType *sets[4] = {NULL, camera, material, uniform};
    const uint8_t counts[4] = {0, 1, 12, 1};
    return make_layout(r, &r->surface_layout, sets, counts,
                       NriStageBits_VERTEX_SHADER | NriStageBits_FRAGMENT_SHADER);
}
static bool create_line_layout(renderer *r) {
    static const NriDescriptorType camera[] = {NriDescriptorType_CONSTANT_BUFFER};
    const NriDescriptorType *sets[4] = {NULL, camera, NULL, NULL};
    const uint8_t counts[4] = {0, 1, 0, 0};
    return make_layout(r, &r->line_layout, sets, counts,
                       NriStageBits_VERTEX_SHADER | NriStageBits_FRAGMENT_SHADER);
}
static bool create_sky_layout(renderer *r) {
    static const NriDescriptorType uniform[] = {NriDescriptorType_CONSTANT_BUFFER};
    const NriDescriptorType *sets[4] = {NULL, NULL, NULL, uniform};
    const uint8_t counts[4] = {0, 0, 0, 1};
    return make_layout(r, &r->sky_layout, sets, counts, NriStageBits_VERTEX_SHADER | NriStageBits_FRAGMENT_SHADER);
}
static bool create_compute_layout(renderer *r, NriPipelineLayout **out,
                                  const NriDescriptorType *sources, uint8_t source_num,
                                  NriDescriptorType output_type, bool has_uniform) {
    static const NriDescriptorType output_texture[] = {NriDescriptorType_STORAGE_TEXTURE};
    static const NriDescriptorType output_buffer[] = {NriDescriptorType_STORAGE_STRUCTURED_BUFFER};
    static const NriDescriptorType uniform[] = {NriDescriptorType_CONSTANT_BUFFER};
    const NriDescriptorType *sets[4] = {sources,
        output_type == NriDescriptorType_STORAGE_TEXTURE ? output_texture : output_buffer,
        has_uniform ? uniform : NULL, NULL};
    const uint8_t counts[4] = {source_num, 1, has_uniform ? 1 : 0, 0};
    return make_layout(r, out, sets, counts, NriStageBits_COMPUTE_SHADER);
}
static bool create_bake_layout(renderer *r) {
    static const NriDescriptorType src[] = {NriDescriptorType_TEXTURE, NriDescriptorType_SAMPLER,
        NriDescriptorType_STRUCTURED_BUFFER, NriDescriptorType_STRUCTURED_BUFFER,
        NriDescriptorType_STRUCTURED_BUFFER};
    return create_compute_layout(r, &r->bake_layout, src, 5, NriDescriptorType_STORAGE_TEXTURE, true);
}
static bool create_probe_layout(renderer *r) {
    static const NriDescriptorType src[] = {NriDescriptorType_STRUCTURED_BUFFER,
        NriDescriptorType_STRUCTURED_BUFFER, NriDescriptorType_STRUCTURED_BUFFER};
    return create_compute_layout(r, &r->probe_layout, src, 3, NriDescriptorType_STORAGE_STRUCTURED_BUFFER, true);
}
static bool create_ssao_layout(renderer *r) {
    static const NriDescriptorType src[] = {NriDescriptorType_TEXTURE, NriDescriptorType_SAMPLER};
    return create_compute_layout(r, &r->ssao_layout, src, 2, NriDescriptorType_STORAGE_TEXTURE, true);
}
static bool create_bloom_layout(renderer *r) {
    static const NriDescriptorType src[] = {NriDescriptorType_TEXTURE, NriDescriptorType_SAMPLER};
    return create_compute_layout(r, &r->bloom_layout, src, 2, NriDescriptorType_STORAGE_TEXTURE, true);
}
static bool create_grade_layout(renderer *r) {
    return create_compute_layout(r, &r->grade_layout, NULL, 0, NriDescriptorType_STORAGE_TEXTURE, false);
}
static bool create_volume_layout(renderer *r) {
    static const NriDescriptorType src[] = {NriDescriptorType_TEXTURE, NriDescriptorType_SAMPLER,
        NriDescriptorType_STRUCTURED_BUFFER, NriDescriptorType_STRUCTURED_BUFFER};
    return create_compute_layout(r, &r->volume_layout, src, 4, NriDescriptorType_STORAGE_TEXTURE, true);
}
static bool create_volume_compose_layout(renderer *r) {
    static const NriDescriptorType src[] = {NriDescriptorType_TEXTURE, NriDescriptorType_TEXTURE,
        NriDescriptorType_TEXTURE, NriDescriptorType_SAMPLER,
        NriDescriptorType_SAMPLER, NriDescriptorType_SAMPLER};
    return create_compute_layout(r, &r->volume_compose_layout, src, 6,
                                 NriDescriptorType_STORAGE_TEXTURE, true);
}
static bool create_compose_layout(renderer *r) {
    static const NriDescriptorType src[] = {NriDescriptorType_TEXTURE, NriDescriptorType_TEXTURE,
        NriDescriptorType_TEXTURE, NriDescriptorType_TEXTURE,
        NriDescriptorType_SAMPLER, NriDescriptorType_SAMPLER,
        NriDescriptorType_SAMPLER, NriDescriptorType_SAMPLER};
    static const NriDescriptorType uniform[] = {NriDescriptorType_CONSTANT_BUFFER};
    const NriDescriptorType *sets[4] = {NULL, NULL, src, uniform};
    const uint8_t counts[4] = {0, 0, 8, 1};
    return make_layout(r, &r->compose_layout, sets, counts, NriStageBits_VERTEX_SHADER | NriStageBits_FRAGMENT_SHADER);
}
static bool create_descriptor_pool(renderer *r) {
    const NriDescriptorPoolDesc desc = {
        .descriptorSetMaxNum = 8192, .samplerMaxNum = 8192,
        .textureMaxNum = 8192, .storageTextureMaxNum = 8192,
        .structuredBufferMaxNum = 8192, .storageStructuredBufferMaxNum = 8192,
        .constantBufferMaxNum = 8192
    };
    return r->core.CreateDescriptorPool(r->device, &desc, &r->descriptor_pool) == NriResult_SUCCESS;
}

static bool track_descriptor(renderer *r, NriDescriptor *desc) {
    if (!desc) return false;
    if (r->temporary_descriptor_num == r->temporary_descriptor_cap) {
        uint32_t cap = r->temporary_descriptor_cap ? r->temporary_descriptor_cap * 2 : 64;
        NriDescriptor **data = realloc(r->temporary_descriptors, cap * sizeof(*data));
        if (!data) { r->core.DestroyDescriptor(desc); return false; }
        r->temporary_descriptors = data;
        r->temporary_descriptor_cap = cap;
    }
    r->temporary_descriptors[r->temporary_descriptor_num++] = desc;
    return true;
}
static bool track_buffer(renderer *r, NriBuffer *buffer) {
    if (r->temporary_buffer_num == r->temporary_buffer_cap) {
        uint32_t cap = r->temporary_buffer_cap ? r->temporary_buffer_cap * 2 : 64;
        NriBuffer **data = realloc(r->temporary_buffers, cap * sizeof(*data));
        if (!data) { r->core.DestroyBuffer(buffer); return false; }
        r->temporary_buffers = data;
        r->temporary_buffer_cap = cap;
    }
    r->temporary_buffers[r->temporary_buffer_num++] = buffer;
    return true;
}
static void clear_temporary(renderer *r) {
    for (uint32_t i = 0; i < r->temporary_descriptor_num; ++i)
        r->core.DestroyDescriptor(r->temporary_descriptors[i]);
    for (uint32_t i = 0; i < r->temporary_buffer_num; ++i)
        r->core.DestroyBuffer(r->temporary_buffers[i]);
    r->temporary_descriptor_num = r->temporary_buffer_num = 0;
    if (r->descriptor_pool) r->core.ResetDescriptorPool(r->descriptor_pool);
}
static NriDescriptor *texture_view(renderer *r, TEXTURE *texture, NriTextureView type) {
    if (!texture) return NULL;
    NriDescriptor *view = NULL;
    const NriTextureViewDesc desc = {
        .texture = texture, .type = type, .format = r->core.GetTextureDesc(texture)->format,
        .mipNum = 1, .layerNum = 1, .sliceNum = 1
    };
    if (r->core.CreateTextureView(&desc, &view) != NriResult_SUCCESS) return NULL;
    return track_descriptor(r, view) ? view : NULL;
}
static NriDescriptor *buffer_view(renderer *r, BUFFER *buffer, NriBufferView type, uint32_t stride) {
    if (!buffer) return NULL;
    NriDescriptor *view = NULL;
    const NriBufferViewDesc desc = {
        .buffer = buffer, .type = type, .offset = 0,
        .size = r->core.GetBufferDesc(buffer)->size, .structureStride = stride
    };
    if (r->core.CreateBufferView(&desc, &view) != NriResult_SUCCESS) return NULL;
    return track_descriptor(r, view) ? view : NULL;
}
static NriDescriptor *uniform_view(renderer *r, const void *data, size_t size) {
    if (!data || !size) return NULL;
    const NriBufferDesc desc = {.size = (size + 255u) & ~(uint64_t)255u,
                                .usage = NriBufferUsageBits_CONSTANT};
    NriBuffer *buffer = NULL;
    if (r->core.CreateCommittedBuffer(r->device, NriMemoryLocation_HOST_UPLOAD,
        1.0f, &desc, &buffer) != NriResult_SUCCESS) return NULL;
    if (!track_buffer(r, buffer)) return NULL;
    void *mapped = r->core.MapBuffer(buffer, 0, desc.size);
    if (!mapped) return NULL;
    memset(mapped, 0, desc.size);
    memcpy(mapped, data, size);
    r->core.UnmapBuffer(buffer);
    return buffer_view(r, buffer, NriBufferView_CONSTANT_BUFFER, 0);
}
static bool bind_set(renderer *r, NriCommandBuffer *cmd, NriPipelineLayout *layout,
                     NriBindPoint point, uint32_t set_index,
                     NriDescriptor *const *descriptors, uint32_t count) {
    NriDescriptorSet *set = NULL;
    if (r->core.AllocateDescriptorSets(r->descriptor_pool, layout, set_index,
                                       &set, 1, 0) != NriResult_SUCCESS) return false;
    for (uint32_t i = 0; i < count; ++i) {
        if (!descriptors[i]) return false;
        const NriDescriptor *d = descriptors[i];
        const NriUpdateDescriptorRangeDesc update = {
            .descriptorSet = set, .rangeIndex = i, .descriptors = &d, .descriptorNum = 1
        };
        r->core.UpdateDescriptorRanges(&update, 1);
    }
    NriPipelineLayout **current = point == NriBindPoint_GRAPHICS ?
        &r->current_graphics_layout : &r->current_compute_layout;
    if (*current != layout) {
        r->core.CmdSetPipelineLayout(cmd, point, layout);
        *current = layout;
    }
    r->core.CmdSetDescriptorSet(cmd, &(NriSetDescriptorSetDesc){
        .setIndex = set_index, .descriptorSet = set, .bindPoint = point});
    return true;
}
static bool bind_uniform(renderer *r, NriCommandBuffer *cmd, NriPipelineLayout *layout,
                         NriBindPoint point, uint32_t set, const void *data, size_t size) {
    NriDescriptor *view = uniform_view(r, data, size);
    return view && bind_set(r, cmd, layout, point, set, &view, 1);
}

static bool transition_texture(renderer *r, NriCommandBuffer *cmd, TEXTURE *texture,
                               NriAccessBits access, NriLayout layout,
                               NriStageBits stages);

static bool bind_bake_resources(renderer *r, NriCommandBuffer *cmd,
                                TEXTURE *source, TEXTURE *destination,
                                BUFFER *nodes, BUFFER *triangles, BUFFER *samples,
                                const bake_uniforms *uniforms, size_t size) {
    if (!transition_texture(r, cmd, source, NriAccessBits_SHADER_RESOURCE,
                            NriLayout_SHADER_RESOURCE, NriStageBits_COMPUTE_SHADER) ||
        !transition_texture(r, cmd, destination, NriAccessBits_SHADER_RESOURCE_STORAGE,
                            NriLayout_SHADER_RESOURCE_STORAGE, NriStageBits_COMPUTE_SHADER)) return false;
    NriDescriptor *src[] = {
        texture_view(r, source, NriTextureView_TEXTURE), r->lightmap_sampler,
        buffer_view(r, nodes, NriBufferView_STRUCTURED_BUFFER, sizeof(bvh_node)),
        buffer_view(r, triangles, NriBufferView_STRUCTURED_BUFFER, sizeof(bvh_triangle)),
        buffer_view(r, samples, NriBufferView_STRUCTURED_BUFFER, sizeof(lmap_sample))
    };
    NriDescriptor *dst = texture_view(r, destination, NriTextureView_STORAGE_TEXTURE);
    return bind_set(r, cmd, r->bake_layout, NriBindPoint_COMPUTE, 0, src, 5) &&
           bind_set(r, cmd, r->bake_layout, NriBindPoint_COMPUTE, 1, &dst, 1) &&
           bind_uniform(r, cmd, r->bake_layout, NriBindPoint_COMPUTE, 2, uniforms, size);
}
static bool bind_probe_resources(renderer *r, NriCommandBuffer *cmd,
                                 BUFFER *input, BUFFER *nodes, BUFFER *triangles,
                                 BUFFER *output, const bake_uniforms *uniforms,
                                 size_t size) {
    const NriBufferBarrierDesc barrier = {
        .buffer = output,
        .after = {.access = NriAccessBits_SHADER_RESOURCE_STORAGE,
                  .stages = NriStageBits_COMPUTE_SHADER}
    };
    r->core.CmdBarrier(cmd, &(NriBarrierDesc){.buffers = &barrier, .bufferNum = 1});
    NriDescriptor *src[] = {
        buffer_view(r, input, NriBufferView_STRUCTURED_BUFFER, sizeof(float[4])),
        buffer_view(r, nodes, NriBufferView_STRUCTURED_BUFFER, sizeof(bvh_node)),
        buffer_view(r, triangles, NriBufferView_STRUCTURED_BUFFER, sizeof(bvh_triangle))
    };
    NriDescriptor *dst = buffer_view(r, output, NriBufferView_STORAGE_STRUCTURED_BUFFER, sizeof(float[4]));
    return bind_set(r, cmd, r->probe_layout, NriBindPoint_COMPUTE, 0, src, 3) &&
           bind_set(r, cmd, r->probe_layout, NriBindPoint_COMPUTE, 1, &dst, 1) &&
           bind_uniform(r, cmd, r->probe_layout, NriBindPoint_COMPUTE, 2, uniforms, size);
}
static bool read_buffer(renderer *r, BUFFER *output, dm_probe_grid *grid,
                        uint32_t bytes, uint32_t count) {
    const NriBufferDesc desc = {.size = bytes};
    BUFFER *staging = NULL;
    if (r->core.CreateCommittedBuffer(r->device, NriMemoryLocation_HOST_READBACK,
        1.0f, &desc, &staging) != NriResult_SUCCESS) return false;
    NriCommandAllocator *allocator = NULL;
    NriCommandBuffer *cmd = NULL;
    bool good = false;
    if (begin_commands(r, &allocator, &cmd) == NriResult_SUCCESS) {
        const NriBufferBarrierDesc barrier = {
            .buffer = output,
            .before = {.access = NriAccessBits_SHADER_RESOURCE_STORAGE, .stages = NriStageBits_COMPUTE_SHADER},
            .after = {.access = NriAccessBits_COPY_SOURCE, .stages = NriStageBits_COPY}
        };
        r->core.CmdBarrier(cmd, &(NriBarrierDesc){.buffers = &barrier, .bufferNum = 1});
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
static bool bind_fx_resources(renderer *r, NriCommandBuffer *cmd,
                              NriPipeline *pipeline, TEXTURE *source,
                              TEXTURE *destination, NriDescriptor *sampler_desc,
                              const void *uniforms, uint32_t size) {
    if ((source && !transition_texture(r, cmd, source, NriAccessBits_SHADER_RESOURCE,
                        NriLayout_SHADER_RESOURCE, NriStageBits_COMPUTE_SHADER)) ||
        !transition_texture(r, cmd, destination, NriAccessBits_SHADER_RESOURCE_STORAGE,
                            NriLayout_SHADER_RESOURCE_STORAGE, NriStageBits_COMPUTE_SHADER)) return false;
    NriPipelineLayout *layout = pipeline == r->fx.grade_pipeline ? r->grade_layout :
        pipeline == r->fx.bloom_pipeline ? r->bloom_layout : r->ssao_layout;
    NriDescriptor *dst = texture_view(r, destination, NriTextureView_STORAGE_TEXTURE);
    if (source) {
        NriDescriptor *src[] = {texture_view(r, source, NriTextureView_TEXTURE), sampler_desc};
        if (!bind_set(r, cmd, layout, NriBindPoint_COMPUTE, 0, src, 2)) return false;
    }
    return bind_set(r, cmd, layout, NriBindPoint_COMPUTE, 1, &dst, 1) &&
           (!size || bind_uniform(r, cmd, layout, NriBindPoint_COMPUTE, 2, uniforms, size));
}
static bool bind_volume_resources(renderer *r, NriCommandBuffer *cmd,
                                  TEXTURE *normal, NriDescriptor *sampler_desc,
                                  BUFFER *probes, BUFFER *beams, TEXTURE *output,
                                  const void *uniforms, uint32_t size) {
    if (!transition_texture(r, cmd, normal, NriAccessBits_SHADER_RESOURCE,
                            NriLayout_SHADER_RESOURCE, NriStageBits_COMPUTE_SHADER) ||
        !transition_texture(r, cmd, output, NriAccessBits_SHADER_RESOURCE_STORAGE,
                            NriLayout_SHADER_RESOURCE_STORAGE, NriStageBits_COMPUTE_SHADER)) return false;
    NriDescriptor *src[] = {texture_view(r, normal, NriTextureView_TEXTURE), sampler_desc,
        buffer_view(r, probes, NriBufferView_STRUCTURED_BUFFER, sizeof(dm_probe)),
        buffer_view(r, beams, NriBufferView_STRUCTURED_BUFFER, sizeof(float))};
    NriDescriptor *dst = texture_view(r, output, NriTextureView_STORAGE_TEXTURE);
    return bind_set(r, cmd, r->volume_layout, NriBindPoint_COMPUTE, 0, src, 4) &&
           bind_set(r, cmd, r->volume_layout, NriBindPoint_COMPUTE, 1, &dst, 1) &&
           bind_uniform(r, cmd, r->volume_layout, NriBindPoint_COMPUTE, 2, uniforms, size);
}
static bool bind_volume_compose_resources(renderer *r, NriCommandBuffer *cmd,
                                          TEXTURE *hdr, TEXTURE *volume, TEXTURE *normal,
                                          NriDescriptor *sampler_desc, NriDescriptor *depth_sampler,
                                          TEXTURE *output, const void *uniforms, uint32_t size) {
    TEXTURE *sources[] = {hdr, volume, normal};
    for (uint32_t i = 0; i < 3; ++i)
        if (!transition_texture(r, cmd, sources[i], NriAccessBits_SHADER_RESOURCE,
                                NriLayout_SHADER_RESOURCE, NriStageBits_COMPUTE_SHADER)) return false;
    if (!transition_texture(r, cmd, output, NriAccessBits_SHADER_RESOURCE_STORAGE,
                            NriLayout_SHADER_RESOURCE_STORAGE, NriStageBits_COMPUTE_SHADER)) return false;
    NriDescriptor *src[] = {texture_view(r, hdr, NriTextureView_TEXTURE),
        texture_view(r, volume, NriTextureView_TEXTURE),
        texture_view(r, normal, NriTextureView_TEXTURE),
        sampler_desc, sampler_desc, depth_sampler};
    NriDescriptor *dst = texture_view(r, output, NriTextureView_STORAGE_TEXTURE);
    return bind_set(r, cmd, r->volume_compose_layout, NriBindPoint_COMPUTE, 0, src, 6) &&
           bind_set(r, cmd, r->volume_compose_layout, NriBindPoint_COMPUTE, 1, &dst, 1) &&
           bind_uniform(r, cmd, r->volume_compose_layout, NriBindPoint_COMPUTE, 2, uniforms, size);
}
static bool bind_sky_resources(renderer *r, NriCommandBuffer *cmd, const void *data, size_t size) {
    return bind_uniform(r, cmd, r->sky_layout, NriBindPoint_GRAPHICS, 3, data, size);
}
static bool bind_camera_resources(renderer *r, NriCommandBuffer *cmd, const void *data, size_t size) {
    return bind_uniform(r, cmd, r->surface_layout, NriBindPoint_GRAPHICS, 1, data, size);
}
static bool bind_surface_resources(renderer *r, NriCommandBuffer *cmd,
                                   const render_material *material, TEXTURE *lightmap,
                                   NriDescriptor *material_sampler, NriDescriptor *lightmap_sampler,
                                   const void *uniforms, size_t size) {
    NriDescriptor *src[] = {
        texture_view(r, material->base_color, NriTextureView_TEXTURE),
        texture_view(r, material->metallic_roughness, NriTextureView_TEXTURE),
        texture_view(r, material->normal, NriTextureView_TEXTURE),
        texture_view(r, material->occlusion, NriTextureView_TEXTURE),
        texture_view(r, material->emissive, NriTextureView_TEXTURE),
        texture_view(r, lightmap, NriTextureView_TEXTURE),
        material_sampler, material_sampler, material_sampler,
        material_sampler, material_sampler, lightmap_sampler
    };
    return bind_set(r, cmd, r->surface_layout, NriBindPoint_GRAPHICS, 2, src, 12) &&
           bind_uniform(r, cmd, r->surface_layout, NriBindPoint_GRAPHICS, 3, uniforms, size);
}
static bool bind_line_resources(renderer *r, NriCommandBuffer *cmd, const void *data, size_t size) {
    return bind_uniform(r, cmd, r->line_layout, NriBindPoint_GRAPHICS, 1, data, size);
}

static void free_shader(NriShaderDesc *shader) {
    if (shader && shader->bytecode) SDL_free((void *)shader->bytecode);
    if (shader) *shader = (NriShaderDesc){0};
}
static texture_state *find_texture_state(renderer *r, TEXTURE *texture);
static bool create_swapchain(renderer *r, uint32_t width, uint32_t height) {
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
        .window = window, .queue = r->graphics_queue,
        .width = (NriDim_t)width, .height = (NriDim_t)height,
        .textureNum = 2, .format = NriSwapChainFormat_BT709_G22_8BIT,
        .flags = NriSwapChainBits_VSYNC
    };
    if (r->swapchain_api.CreateSwapChain(r->device, &desc, &r->swapchain) != NriResult_SUCCESS)
        return false;
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
        texture_state *state = find_texture_state(r, textures[i]);
        if (!state) return false;
        state->state = (NriAccessLayoutStage){.layout = NriLayout_UNDEFINED,
                                               .stages = NriStageBits_NONE};
        swapchain_texture *frame = &r->swapchain_frames[i];
        frame->texture = textures[i];
        const NriTextureViewDesc view = {
            .texture = textures[i], .type = NriTextureView_COLOR_ATTACHMENT,
            .format = r->swapchain_format, .mipNum = 1, .layerNum = 1, .sliceNum = 1
        };
        if (r->core.CreateTextureView(&view, &frame->color_attachment) != NriResult_SUCCESS ||
            r->core.CreateFence(r->device, NRI_SWAPCHAIN_SEMAPHORE, &frame->acquire) != NriResult_SUCCESS ||
            r->core.CreateFence(r->device, NRI_SWAPCHAIN_SEMAPHORE, &frame->release) != NriResult_SUCCESS)
            return false;
    }
    return true;
}
static void destroy_swapchain(renderer *r) {
    for (uint32_t i = 0; i < r->swapchain_texture_count; ++i) {
        for (uint32_t j = 0; j < r->texture_state_num; ++j)
            if (r->texture_states[j].texture == r->swapchain_textures[i]) {
                r->texture_states[j] = r->texture_states[--r->texture_state_num];
                break;
            }
        swapchain_texture *frame = &r->swapchain_frames[i];
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
static bool acquire_swapchain_texture(renderer *r, uint32_t *index) {
    NriFence *acquire = r->swapchain_frames[r->frame_index % r->swapchain_texture_count].acquire;
    NriResult result = r->swapchain_api.AcquireNextTexture(r->swapchain, acquire, index);
    return result == NriResult_SUCCESS && *index < r->swapchain_texture_count;
}
static texture_state *find_texture_state(renderer *r, TEXTURE *texture) {
    for (uint32_t i = 0; i < r->texture_state_num; ++i)
        if (r->texture_states[i].texture == texture) return &r->texture_states[i];
    if (r->texture_state_num == r->texture_state_cap) {
        uint32_t cap = r->texture_state_cap ? r->texture_state_cap * 2 : 32;
        texture_state *items = realloc(r->texture_states, cap * sizeof(*items));
        if (!items) return NULL;
        r->texture_states = items;
        r->texture_state_cap = cap;
    }
    texture_state *item = &r->texture_states[r->texture_state_num++];
    *item = (texture_state){.texture = texture};
    return item;
}
static bool texture_barrier(renderer *r, NriCommandBuffer *cmd, TEXTURE *texture,
                            NriAccessLayoutStage before, NriAccessLayoutStage after) {
    texture_state *item = find_texture_state(r, texture);
    if (!item) return false;
    if (item->state.layout) before = item->state;
    const NriTextureBarrierDesc barrier = {
        .texture = texture, .before = before, .after = after,
        .mipNum = 1, .layerNum = 1
    };
    r->core.CmdBarrier(cmd, &(NriBarrierDesc){.textures = &barrier, .textureNum = 1});
    item->state = after;
    return true;
}
static bool transition_texture(renderer *r, NriCommandBuffer *cmd, TEXTURE *texture,
                               NriAccessBits access, NriLayout layout,
                               NriStageBits stages) {
    if (!texture) return false;
    return texture_barrier(r, cmd, texture, (NriAccessLayoutStage){0},
                           (NriAccessLayoutStage){.access = access,
                               .layout = layout, .stages = stages});
}
static bool begin_scene_rendering(renderer *r, NriCommandBuffer *cmd,
                                  TEXTURE *hdr, TEXTURE *normal, TEXTURE *depth,
                                  uint32_t width, uint32_t height) {
    NriDescriptor *hdr_view = texture_view(r, hdr, NriTextureView_COLOR_ATTACHMENT);
    NriDescriptor *normal_view = texture_view(r, normal, NriTextureView_COLOR_ATTACHMENT);
    NriDescriptor *depth_view = texture_view(r, depth, NriTextureView_DEPTH_STENCIL_ATTACHMENT);
    if (!hdr_view || !normal_view || !depth_view) return false;
    const NriAccessLayoutStage color = {NriAccessBits_COLOR_ATTACHMENT,
                                       NriLayout_COLOR_ATTACHMENT, NriStageBits_COLOR_ATTACHMENT};
    const NriAccessLayoutStage depth_state = {NriAccessBits_DEPTH_STENCIL_ATTACHMENT,
                                  NriLayout_DEPTH_STENCIL_ATTACHMENT, NriStageBits_DEPTH_STENCIL_ATTACHMENT};
    if (!texture_barrier(r, cmd, hdr, (NriAccessLayoutStage){0}, color) ||
        !texture_barrier(r, cmd, normal, (NriAccessLayoutStage){0}, color) ||
        !texture_barrier(r, cmd, depth, (NriAccessLayoutStage){0}, depth_state)) return false;
    const NriAttachmentDesc colors[2] = {
        {.descriptor = hdr_view, .loadOp = NriLoadOp_CLEAR, .storeOp = NriStoreOp_STORE},
        {.descriptor = normal_view, .loadOp = NriLoadOp_CLEAR, .storeOp = NriStoreOp_STORE}
    };
    const NriRenderingDesc desc = {
        .colors = colors, .colorNum = 2,
        .depth = {.descriptor = depth_view, .loadOp = NriLoadOp_CLEAR,
                  .storeOp = NriStoreOp_STORE, .clearValue = {.depthStencil = {.depth = 1.0f}}}
    };
    r->core.CmdSetViewports(cmd, &(NriViewport){.width = (float)width,
                            .height = (float)height, .depthMax = 1.0f}, 1);
    r->core.CmdSetScissors(cmd, &(NriRect){.width = (NriDim_t)width,
                           .height = (NriDim_t)height}, 1);
    r->core.CmdBeginRendering(cmd, &desc);
    return true;
}
static bool begin_compose_rendering(renderer *r, NriCommandBuffer *cmd, TEXTURE *swap,
                                    TEXTURE *hdr, TEXTURE *ao, TEXTURE *bloom,
                                    TEXTURE *lut, NriDescriptor *sampler_desc,
                                    const void *uniforms, size_t size) {
    TEXTURE *sources[] = {hdr, ao, bloom, lut};
    for (uint32_t i = 0; i < 4; ++i)
        if (!transition_texture(r, cmd, sources[i], NriAccessBits_SHADER_RESOURCE,
                                NriLayout_SHADER_RESOURCE, NriStageBits_FRAGMENT_SHADER)) return false;
    NriDescriptor *src[] = {
        texture_view(r, hdr, NriTextureView_TEXTURE),
        texture_view(r, ao, NriTextureView_TEXTURE),
        texture_view(r, bloom, NriTextureView_TEXTURE),
        texture_view(r, lut, NriTextureView_TEXTURE),
        sampler_desc, sampler_desc, sampler_desc, sampler_desc
    };
    if (!bind_set(r, cmd, r->compose_layout, NriBindPoint_GRAPHICS, 2, src, 8) ||
        !bind_uniform(r, cmd, r->compose_layout, NriBindPoint_GRAPHICS, 3, uniforms, size))
        return false;
    if (!texture_barrier(r, cmd, swap,
        (NriAccessLayoutStage){.layout = NriLayout_UNDEFINED,
            .stages = NriStageBits_NONE},
        (NriAccessLayoutStage){.access = NriAccessBits_COLOR_ATTACHMENT,
            .layout = NriLayout_COLOR_ATTACHMENT, .stages = NriStageBits_COLOR_ATTACHMENT})) return false;
    const NriAttachmentDesc color = {
        .descriptor = r->swapchain_frames[r->current_swap_index].color_attachment,
        .loadOp = NriLoadOp_CLEAR, .storeOp = NriStoreOp_STORE
    };
    const NriRenderingDesc desc = {.colors = &color, .colorNum = 1};
    r->core.CmdBeginRendering(cmd, &desc);
    return true;
}
static bool submit_frame(renderer *r, NriCommandAllocator *allocator,
                         NriCommandBuffer *cmd, uint32_t index) {
    bool good = texture_barrier(r, cmd, r->swapchain_textures[index],
        (NriAccessLayoutStage){.access = NriAccessBits_COLOR_ATTACHMENT,
            .layout = NriLayout_COLOR_ATTACHMENT, .stages = NriStageBits_COLOR_ATTACHMENT},
        (NriAccessLayoutStage){.layout = NriLayout_PRESENT,
            .stages = NriStageBits_NONE});
    if (good) good = r->core.EndCommandBuffer(cmd) == NriResult_SUCCESS;
    const NriFenceSubmitDesc wait = {
        .fence = r->swapchain_frames[r->frame_index % r->swapchain_texture_count].acquire,
        .stages = NriStageBits_COLOR_ATTACHMENT
    };
    const NriFenceSubmitDesc signal = {.fence = r->swapchain_frames[index].release};
    const NriQueueSubmitDesc submit = {
        .waitFences = &wait, .waitFenceNum = 1,
        .commandBuffers = (const NriCommandBuffer *const *)&cmd, .commandBufferNum = 1,
        .signalFences = &signal, .signalFenceNum = 1
    };
    if (good) good = r->core.QueueSubmit(r->graphics_queue, &submit) == NriResult_SUCCESS;
    if (good) good = r->swapchain_api.QueuePresent(r->swapchain, r->swapchain_frames[index].release,
                                                    ++r->frame_index) == NriResult_SUCCESS;
    if (r->core.QueueWaitIdle(r->graphics_queue) != NriResult_SUCCESS) good = false;
    r->core.DestroyCommandBuffer(cmd);
    r->core.DestroyCommandAllocator(allocator);
    clear_temporary(r);
    if (!good) destroy_swapchain(r);
    return good;
}

static NriDescriptor *sampler(renderer *r, NriFilter min_filter,
                              NriFilter mag_filter, NriAddressMode address) {
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
    if (r->core.CreateSampler(r->device, &desc, &result) != NriResult_SUCCESS)
        return NULL;
    return result;
}

static NriPipeline *make_line_pipeline(renderer *r,
                                       NriPipelineLayout *layout,
                                       const NriShaderDesc *vs,
                                       const NriShaderDesc *ps) {
    const NriVertexStreamDesc vb = {
        .bindingSlot = 0,
        .stepRate = NriVertexStreamStepRate_PER_VERTEX,
        .stride = (uint16_t)sizeof(render_vertex)
    };

    const NriVertexAttributeDesc attrs[2] = {
        {
            .d3d = {.semanticName = "TEXCOORD", .semanticIndex = 0},
            .vk = {.location = 0},
            .offset = (uint32_t)offsetof(render_vertex, x),
            .format = NriFormat_RGB32_SFLOAT,
            .streamIndex = 0
        },
        {
            .d3d = {.semanticName = "TEXCOORD", .semanticIndex = 1},
            .vk = {.location = 1},
            .offset = (uint32_t)offsetof(render_vertex, r),
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
        },
        {
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
        .inputAssembly = {.topology = NriTopology_LINE_LIST},
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
        .shaderNum = 2
    };

    NriPipeline *pipeline = NULL;
    if (r->core.CreateGraphicsPipeline(
            r->device, &desc, &pipeline) != NriResult_SUCCESS)
        return NULL;

    return pipeline;
}

static NriPipeline *make_sky_pipeline(renderer *r,
                                      NriPipelineLayout *layout,
                                      const NriShaderDesc *vs,
                                      const NriShaderDesc *ps) {
    const NriColorAttachmentDesc targets[2] = {
        {
            .format = NriFormat_RGBA16_SFLOAT,
            .colorWriteMask = NriColorWriteBits_RGBA
        },
        {
            .format = NriFormat_RGBA16_SFLOAT,
            .colorWriteMask = NriColorWriteBits_RGBA
        }
    };

    const NriMultisampleDesc multisample = {
        .sampleMask = NRI_ALL,
        .sampleNum = 1
    };

    const NriShaderDesc shaders[2] = {*vs, *ps};
    const NriGraphicsPipelineDesc desc = {
        .pipelineLayout = layout,
        .inputAssembly = {.topology = NriTopology_TRIANGLE_LIST},
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
            .depthStencilFormat = r->depth_format
        },
        .shaders = shaders,
        .shaderNum = 2
    };

    NriPipeline *pipeline = NULL;
    if (r->core.CreateGraphicsPipeline(
            r->device, &desc, &pipeline) != NriResult_SUCCESS)
        return NULL;

    return pipeline;
}

static TEXTURE *texture(renderer *r, NriFormat format,
                        NriTextureUsageBits usage,
                        Uint32 width, Uint32 height) {
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

    TEXTURE *result = NULL;
    if (r->core.CreateCommittedTexture(
            r->device,
            NriMemoryLocation_DEVICE,
            1.0f,
            &desc,
            &result) != NriResult_SUCCESS)
        return NULL;

    if (!find_texture_state(r, result)) {
        release_texture(r, result);
        return NULL;
    }

    return result;
}

static BUFFER *upload_buffer(renderer *r, NriBufferUsageBits usage,
                             const void *data, size_t bytes, uint32_t stride) {
    if (!r || !r->device || !r->graphics_queue || !data || !bytes)
        return NULL;

    const NriBufferDesc desc = {
        .size = bytes,
        .structureStride = stride,
        .usage = usage
    };

    BUFFER *buffer = NULL;
    if (r->core.CreateCommittedBuffer(
            r->device,
            NriMemoryLocation_DEVICE,
            1.0f,
            &desc,
            &buffer) != NriResult_SUCCESS)
        return NULL;

    const NriBufferUploadDesc upload = {
        .data = data,
        .buffer = buffer,
        .after = {
            .access = (usage & NriBufferUsageBits_SHADER_RESOURCE ?
                       NriAccessBits_SHADER_RESOURCE : 0) |
                      (usage & NriBufferUsageBits_VERTEX ?
                       NriAccessBits_VERTEX_BUFFER : 0),
            .stages = usage & NriBufferUsageBits_VERTEX ?
                      NriStageBits_VERTEX_SHADER : NriStageBits_COMPUTE_SHADER |
                      NriStageBits_FRAGMENT_SHADER
        }
    };

    if (r->helper.UploadData(
            r->graphics_queue,
            NULL, 0,
            &upload, 1) != NriResult_SUCCESS) {
        r->core.DestroyBuffer(buffer);
        return NULL;
    }

    return buffer;
}

static bool upload_texture_data(renderer *r, TEXTURE *texture,
                                const void *data,
                                uint32_t row_pitch,
                                uint32_t slice_pitch,
                                NriAccessBits access,
                                NriLayout layout,
                                NriStageBits stages) {
    const NriTextureSubresourceUploadDesc subresource = {
        .slices = data,
        .sliceNum = 1,
        .rowPitch = row_pitch,
        .slicePitch = slice_pitch
    };

    const NriTextureUploadDesc upload = {
        .subresources = &subresource,
        .texture = texture,
        .after = {
            .access = access,
            .layout = layout,
            .stages = stages
        }
    };

    bool uploaded = r->helper.UploadData(
        r->graphics_queue,
        &upload, 1,
        NULL, 0) == NriResult_SUCCESS;
    if (uploaded) {
        texture_state *state = find_texture_state(r, texture);
        if (state) state->state = (NriAccessLayoutStage){access, layout, stages};
    }
    return uploaded;
}

static TEXTURE *pixel_texture(renderer *r, Uint8 red, Uint8 green,
                              Uint8 blue, Uint8 alpha) {
    const Uint8 pixels[4] = {red, green, blue, alpha};

    TEXTURE *result = texture(
        r,
        NriFormat_RGBA8_UNORM,
        NriTextureUsageBits_SHADER_RESOURCE,
        1, 1);
    if (!result) return NULL;

    if (!upload_texture_data(
            r, result, pixels, 4, 4,
            NriAccessBits_SHADER_RESOURCE,
            NriLayout_SHADER_RESOURCE,
            NriStageBits_ALL)) {
        release_texture(r, result);
        return NULL;
    }

    return result;
}

void release_texture(renderer *r, TEXTURE *value) {
    if (!r || !value) return;
    for (uint32_t i = 0; i < r->texture_state_num; ++i) {
        if (r->texture_states[i].texture == value) {
            r->texture_states[i] = r->texture_states[--r->texture_state_num];
            break;
        }
    }
    r->core.DestroyTexture(value);
}

void release_buffer(renderer *r, BUFFER *value) {
    if (r && value) r->core.DestroyBuffer(value);
}

static bool ensure_depth_texture(renderer *r, Uint32 width, Uint32 height) {
    if (r->depth_texture && r->depth_width == width &&
        r->depth_height == height)
        return true;

    release_texture(r, r->depth_texture);
    r->depth_texture = texture(
        r,
        r->depth_format,
        NriTextureUsageBits_DEPTH_STENCIL_ATTACHMENT,
        width, height);

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

static TEXTURE *load_image(renderer *r, const gltf_image *image) {
    if (!image || !image->bytes.data || !image->bytes.size) return NULL;

    SDL_IOStream *io = SDL_IOFromConstMem(image->bytes.data, image->bytes.size);
    if (!io) return NULL;

    SDL_Surface *decoded = IMG_LoadTyped_IO(
        io, true, image_type(image->mime));
    if (!decoded) return NULL;

    SDL_Surface *rgba = SDL_ConvertSurface(decoded, SDL_PIXELFORMAT_RGBA32);
    SDL_DestroySurface(decoded);
    if (!rgba) return NULL;

    TEXTURE *result = texture(
        r,
        NriFormat_RGBA8_UNORM,
        NriTextureUsageBits_SHADER_RESOURCE,
        (Uint32)rgba->w,
        (Uint32)rgba->h);

    if (result && !upload_texture_data(
            r,
            result,
            rgba->pixels,
            (uint32_t)rgba->pitch,
            (uint32_t)(rgba->pitch * rgba->h),
            NriAccessBits_SHADER_RESOURCE,
            NriLayout_SHADER_RESOURCE,
            NriStageBits_FRAGMENT_SHADER)) {
        release_texture(r, result);
        result = NULL;
    }

    SDL_DestroySurface(rgba);
    return result;
}

static bool load_images(renderer *r, const gltf_scene *visual) {
    r->image_texture_count = visual->image_count;
    if (!visual->image_count) return true;

    r->image_textures = calloc(
        visual->image_count, sizeof(*r->image_textures));
    if (!r->image_textures) return false;

    for (uint32_t i = 0; i < visual->image_count; ++i) {
        const gltf_image *image = &visual->images[i];
        if (!image->bytes.data || !image->bytes.size) continue;

        r->image_textures[i] = load_image(r, image);
        if (!r->image_textures[i]) {
            SDL_Log("SDL_image could not decode GLB image %u (%s): %s",
                    i, image->mime[0] ? image->mime : "unknown",
                    SDL_GetError());
        }
    }

    return true;
}

static TEXTURE *resolve_texture(renderer *r, const gltf_scene *visual,
                                int32_t texture_index, TEXTURE *fallback) {
    if (texture_index < 0 ||
        (uint32_t)texture_index >= visual->texture_count)
        return fallback;

    const int32_t image = visual->textures[texture_index].image;
    if (image < 0 || (uint32_t)image >= r->image_texture_count ||
        !r->image_textures[image])
        return fallback;

    return r->image_textures[image];
}

static void release_scene_resources(renderer *r) {
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
    if (r->material_sampler)
        r->core.DestroyDescriptor(r->material_sampler);
    release_buffer(r, r->vertex_buffer);
    release_texture(r, r->lightmap_texture);
    if (r->lightmap_sampler)
        r->core.DestroyDescriptor(r->lightmap_sampler);

    r->default_white = NULL;
    r->default_normal = NULL;
    r->material_sampler = NULL;
    r->vertex_buffer = NULL;
    r->lightmap_texture = NULL;
    r->lightmap_sampler = NULL;
}

bool upload_scene(renderer *r, const gltf_scene *visual) {
    if (!r || !r->device || !visual || !r->vertices || !r->vertex_count)
        return false;

    release_scene_resources(r);

    r->default_white = pixel_texture(r, 255, 255, 255, 255);
    r->default_normal = pixel_texture(r, 128, 128, 255, 255);

    r->material_sampler = sampler(
        r,
        NriFilter_LINEAR,
        NriFilter_LINEAR,
        NriAddressMode_REPEAT);

    if (!r->default_white || !r->default_normal || !r->material_sampler ||
        !load_images(r, visual))
        return false;

    r->material_count = visual->material_count;
    r->materials = calloc(r->material_count, sizeof(*r->materials));
    if (!r->materials) return false;

    for (uint32_t i = 0; i < r->material_count; ++i) {
        render_material *m = &r->materials[i];
        m->data = visual->materials[i];
        m->base_color = resolve_texture(
            r, visual, m->data.base_color_texture, r->default_white);
        m->metallic_roughness = resolve_texture(
            r, visual, m->data.metallic_roughness_texture, r->default_white);
        m->normal = resolve_texture(
            r, visual, m->data.normal_texture, r->default_normal);
        m->occlusion = resolve_texture(
            r, visual, m->data.occlusion_texture, r->default_white);
        m->emissive = resolve_texture(
            r, visual, m->data.emissive_texture, r->default_white);
    }

    r->vertex_buffer = upload_buffer(
        r,
        NriBufferUsageBits_VERTEX,
        r->vertices,
        (size_t)r->vertex_count * sizeof(*r->vertices), 0);

    r->lightmap_sampler = sampler(
        r,
        NriFilter_LINEAR,
        NriFilter_LINEAR,
        NriAddressMode_CLAMP_TO_EDGE);

    r->lightmap_texture = pixel_texture(r, 0, 0, 0, 255);

    return r->vertex_buffer && r->lightmap_sampler && r->lightmap_texture;
}

static TEXTURE *create_lightmap_texture(renderer *r,
                                        Uint32 width, Uint32 height) {
    return texture(
        r,
        NriFormat_RGBA16_SFLOAT,
        NriTextureUsageBits_SHADER_RESOURCE |
        NriTextureUsageBits_SHADER_RESOURCE_STORAGE,
        width, height);
}

static bool transfer_size(uint32_t width, uint32_t height, Uint32 *out) {
    const uint64_t bytes = (uint64_t)width * (uint64_t)height * 8u;
    if (!width || !height || bytes > UINT32_MAX) return false;
    *out = (Uint32)bytes;
    return true;
}

TEXTURE *upload_lightmap(renderer *r, const dm_cached_lightmap *cached) {
    if (!r || !r->device || !cached || !cached->pixels) return NULL;

    Uint32 bytes = 0;
    if (!transfer_size(cached->width, cached->height, &bytes)) return NULL;

    TEXTURE *result = create_lightmap_texture(
        r, cached->width, cached->height);
    if (!result) return NULL;

    if (!upload_texture_data(
            r,
            result,
            cached->pixels,
            cached->width * 8u,
            bytes,
            NriAccessBits_SHADER_RESOURCE,
            NriLayout_SHADER_RESOURCE,
            NriStageBits_ALL)) {
        release_texture(r, result);
        return NULL;
    }

    return result;
}

bool download_lightmap(renderer *r, dm_cached_lightmap *out) {
    if (!r || !r->device || !r->lightmap_texture || !out) return false;

    Uint32 bytes = 0;
    if (!transfer_size(
            r->lightmap_width, r->lightmap_height, &bytes))
        return false;

    const uint32_t row_bytes = r->lightmap_width * 8u;
    const uint32_t alignment = r->core.GetDeviceDesc(r->device)->memoryAlignment.uploadBufferTextureRow;
    if (!alignment) return false;
    const uint64_t row_pitch = ((uint64_t)row_bytes + alignment - 1u) / alignment * alignment;
    const uint32_t slice_alignment = r->core.GetDeviceDesc(r->device)->memoryAlignment.uploadBufferTextureSlice;
    if (!slice_alignment) return false;
    const uint64_t slice_bytes = row_pitch * r->lightmap_height;
    const uint64_t staging_bytes = (slice_bytes + slice_alignment - 1u) /
                                   slice_alignment * slice_alignment;
    if (staging_bytes > UINT32_MAX) return false;

    const NriBufferDesc readback_desc = {
        .size = staging_bytes,
        .usage = NriBufferUsageBits_NONE
    };

    BUFFER *readback = NULL;
    if (r->core.CreateCommittedBuffer(
            r->device,
            NriMemoryLocation_HOST_READBACK,
            1.0f,
            &readback_desc,
            &readback) != NriResult_SUCCESS)
        return false;

    NriCommandAllocator *allocator = NULL;
    NriCommandBuffer *cmd = NULL;
    if (begin_commands(r, &allocator, &cmd) != NriResult_SUCCESS) {
        r->core.DestroyBuffer(readback);
        return false;
    }

    const NriTextureDataLayoutDesc layout = {
        .offset = 0,
        .rowPitch = (uint32_t)row_pitch,
        .slicePitch = (uint32_t)staging_bytes
    };


    const NriTextureRegionDesc region = {
        .width = (NriDim_t)r->lightmap_width,
        .height = (NriDim_t)r->lightmap_height,
        .depth = 1,
        .mipOffset = 0,
        .layerOffset = 0
    };

    if (!transition_texture(r, cmd, r->lightmap_texture, NriAccessBits_COPY_SOURCE,
                            NriLayout_COPY_SOURCE, NriStageBits_COPY)) {
        abort_commands(r, allocator, cmd);
        r->core.DestroyBuffer(readback);
        return false;
    }
    r->core.CmdReadbackTextureToBuffer(
        cmd, readback, &layout, r->lightmap_texture, &region);

    if (!submit_commands(r, allocator, cmd)) {
        r->core.DestroyBuffer(readback);
        return false;
    }

    void *mapped = r->core.MapBuffer(readback, 0, staging_bytes);
    if (!mapped) {
        r->core.DestroyBuffer(readback);
        return false;
    }

    out->pixels = malloc(bytes);
    if (out->pixels) {
        for (uint32_t y = 0; y < r->lightmap_height; ++y)
            memcpy((uint8_t *)out->pixels + (size_t)y * row_bytes,
                   (const uint8_t *)mapped + (size_t)y * row_pitch, row_bytes);
        out->width = r->lightmap_width;
        out->height = r->lightmap_height;
    }

    r->core.UnmapBuffer(readback);
    r->core.DestroyBuffer(readback);
    return out->pixels != NULL;
}

bool upload_bvh(renderer *r, const bvh *tree) {
    if (!r || !tree || !tree->nodes || !tree->node_count ||
        !tree->triangles || !tree->triangle_count)
        return false;

    release_buffer(r, r->bvh_node_buffer);
    release_buffer(r, r->bvh_triangle_buffer);
    r->bvh_node_buffer = NULL;
    r->bvh_triangle_buffer = NULL;

    r->bvh_node_buffer = upload_buffer(
        r,
        NriBufferUsageBits_SHADER_RESOURCE,
        tree->nodes,
        (size_t)tree->node_count * sizeof(*tree->nodes), sizeof(bvh_node));

    r->bvh_triangle_buffer = upload_buffer(
        r,
        NriBufferUsageBits_SHADER_RESOURCE,
        tree->triangles,
        (size_t)tree->triangle_count * sizeof(*tree->triangles), sizeof(bvh_triangle));

    return r->bvh_node_buffer && r->bvh_triangle_buffer;
}

BUFFER *upload_probes(renderer *r, const dm_probe_grid *grid) {
    if (!grid || !grid->probes) return NULL;

    const size_t count =
        (size_t)grid->count_x * grid->count_y * grid->count_z;
    if (!count) return NULL;

    return upload_buffer(
        r,
        NriBufferUsageBits_SHADER_RESOURCE,
        grid->probes,
        count * sizeof(dm_probe), sizeof(dm_probe));
}

BUFFER *upload_beams(renderer *r, const dm_beam_grid *grid) {
    if (!grid) return NULL;

    float *visibility = dm_beam_expand(grid);
    if (!visibility || !grid->shadow_depth) {
        free(visibility);
        return NULL;
    }

    const size_t beam_count =
        (size_t)grid->width * grid->height * grid->depth;
    const size_t depth_count = (size_t)grid->width * grid->height;

    float *data = realloc(
        visibility, (beam_count + depth_count) * sizeof(float));
    if (!data) {
        free(visibility);
        return NULL;
    }

    memcpy(data + beam_count,
           grid->shadow_depth,
           depth_count * sizeof(float));

    BUFFER *buffer = upload_buffer(
        r,
        NriBufferUsageBits_SHADER_RESOURCE,
        data,
        (beam_count + depth_count) * sizeof(float), sizeof(float));

    free(data);
    return buffer;
}

static void dispatch_shape(Uint32 items, Uint32 *groups_x, Uint32 *groups_y,
                           Uint32 *dispatch_width) {
    Uint32 gx = (items + 63u) / 64u;
    if (gx == 0u) gx = 1u;
    if (gx > 4096u) gx = 4096u;
    const Uint32 width = gx * 64u;
    const Uint32 gy = (items + width - 1u) / width;

    *groups_x = gx;
    *groups_y = gy ? gy : 1u;
    *dispatch_width = width;
}

static bake_uniforms bake_data(renderer *r, Uint32 phase, Uint32 iteration,
                               Uint32 item_count, Uint32 dispatch_width) {
    const vec3 sun = sun_direction();
    return (bake_uniforms){
        .item_count = item_count,
        .lightmap_width = r->lightmap_width,
        .lightmap_height = r->lightmap_height,
        .dispatch_width = dispatch_width,
        .iteration = iteration,
        .phase = phase,
        .max_bounces = BAKE_MAX_BOUNCES,
        .padding = phase == PHASE_TRACE ? 1u : 0u,
        .sun_direction_intensity = {sun.x, sun.y, sun.z, 2.4f},
        .sun_color_radius = {1.00f, 0.94f, 0.84f, 0.00465f},
        .sky_zenith = {0.22f, 0.42f, 0.78f, 1.0f},
        .sky_horizon = {0.68f, 0.76f, 0.88f, 1.0f},
        .bake_params = {r->bake_epsilon, 0.72f, 1.0f, 0.0f}
    };
}

static bool record_bake_pass(renderer *r, NriCommandBuffer *cmd,
                             TEXTURE *source, TEXTURE *destination,
                             Uint32 phase, Uint32 iteration,
                             Uint32 item_count) {
    Uint32 groups_x, groups_y, dispatch_width;
    dispatch_shape(item_count, &groups_x, &groups_y, &dispatch_width);

    const bake_uniforms uniforms = bake_data(
        r, phase, iteration, item_count, dispatch_width);

    if (!bind_bake_resources(
            r,
            cmd,
            source,
            destination,
            r->bvh_node_buffer,
            r->bvh_triangle_buffer,
            r->lightmap_sample_buffer,
            &uniforms,
            sizeof(uniforms)))
        return false;

    r->core.CmdSetPipeline(
        cmd, r->bake_pipeline);

    r->core.CmdDispatch(
        cmd,
        &(NriDispatchDesc){
            .workGroupNumX = groups_x,
            .workGroupNumY = groups_y,
            .workGroupNumZ = 1
        }
    );


    return true;
}

static void swap_lightmaps(renderer *r) {
    TEXTURE *tmp = r->lightmap_texture;
    r->lightmap_texture = r->lightmap_scratch;
    r->lightmap_scratch = tmp;
}

static bool submit_trace_batch(renderer *r, Uint32 first, Uint32 count) {
    NriCommandAllocator *allocator = NULL;
    NriCommandBuffer *cmd = NULL;
    if (begin_commands(r, &allocator, &cmd) != NriResult_SUCCESS)
        return false;

    for (Uint32 i = 0; i < count; ++i) {
        if (!record_bake_pass(
                r, cmd,
                r->lightmap_texture,
                r->lightmap_scratch,
                PHASE_TRACE,
                first + i,
                r->lightmap_sample_count)) {
            abort_commands(r, allocator, cmd);
            return false;
        }
        swap_lightmaps(r);
    }

    return submit_commands(r, allocator, cmd);
}

static bool bake_lightmap_once(renderer *r) {
    const Uint32 pixels = r->lightmap_width * r->lightmap_height;
    bake_progress(r, "surface lightmap", 0u, r->bake_target_samples);

    NriCommandAllocator *allocator = NULL;
    NriCommandBuffer *cmd = NULL;
    if (begin_commands(r, &allocator, &cmd) != NriResult_SUCCESS)
        return false;

    if (!record_bake_pass(
            r, cmd,
            r->lightmap_scratch,
            r->lightmap_texture,
            PHASE_CLEAR, 0, pixels) ||
        !record_bake_pass(
            r, cmd,
            r->lightmap_texture,
            r->lightmap_scratch,
            PHASE_CLEAR, 0, pixels)) {
        abort_commands(r, allocator, cmd);
        return false;
    }

    if (!submit_commands(r, allocator, cmd)) return false;

    for (Uint32 first = 0; first < r->bake_target_samples;
         first += BAKE_BATCH_SAMPLES) {
        Uint32 count = r->bake_target_samples - first;
        if (count > BAKE_BATCH_SAMPLES) count = BAKE_BATCH_SAMPLES;
        if (!submit_trace_batch(r, first, count)) return false;
        bake_progress(
            r, "surface lightmap",
            first + count,
            r->bake_target_samples);
    }

    bake_progress(r, "filtering lightmap", 0u, 0u);

    allocator = NULL;
    cmd = NULL;
    if (begin_commands(r, &allocator, &cmd) != NriResult_SUCCESS)
        return false;

    if (!record_bake_pass(
            r, cmd,
            r->lightmap_texture,
            r->lightmap_scratch,
            PHASE_FILTER, 0, pixels)) {
        abort_commands(r, allocator, cmd);
        return false;
    }

    swap_lightmaps(r);

    for (Uint32 i = 0; i < BAKE_DILATION_PASSES; ++i) {
        if (!record_bake_pass(
                r, cmd,
                r->lightmap_texture,
                r->lightmap_scratch,
                PHASE_DILATE, 0, pixels)) {
            abort_commands(r, allocator, cmd);
            return false;
        }
        swap_lightmaps(r);
    }

    if (!submit_commands(r, allocator, cmd)) return false;

    release_texture(r, r->lightmap_scratch);
    r->lightmap_scratch = NULL;

    if (r->bake_pipeline)
        r->core.DestroyPipeline(r->bake_pipeline);
    r->bake_pipeline = NULL;

    return true;
}

bool bake_lightmap(renderer *r, const bvh *tree, const lightmap *lm) {
    if (!r || !r->device || !tree || !tree->node_count || !lm ||
        !lm->width || !lm->height || !lm->samples || !lm->sample_count ||
        !r->lightmap_sampler)
        return false;

    r->lightmap_width = lm->width;
    r->lightmap_height = lm->height;
    r->lightmap_sample_count = lm->sample_count;
    r->bake_target_samples = BAKE_TARGET_SAMPLES;

    const bvh_node *root = &tree->nodes[0];
    const float sx = root->max[0] - root->min[0];
    const float sy = root->max[1] - root->min[1];
    const float sz = root->max[2] - root->min[2];
    float scene_scale = fmaxf(sx, fmaxf(sy, sz));
    if (scene_scale < 1.0f) scene_scale = 1.0f;
    r->bake_epsilon = scene_scale * 2.0e-5f;

    r->lightmap_texture = create_lightmap_texture(r, lm->width, lm->height);
    r->lightmap_scratch = create_lightmap_texture(r, lm->width, lm->height);
    if (!r->lightmap_texture || !r->lightmap_scratch) return false;

    if (!upload_bvh(r, tree)) return false;

    r->lightmap_sample_buffer = upload_buffer(
        r,
        NriBufferUsageBits_SHADER_RESOURCE,
        lm->samples,
        (size_t)lm->sample_count * sizeof(*lm->samples), sizeof(lmap_sample));
    if (!r->lightmap_sample_buffer) return false;

    r->bake_pipeline = compile_compute(
        &r->core,
        r->device,
        r->bake_layout,
        "shaders/compute.hlsl",
        "lightmap_cs",
        "BUILD_LIGHTMAP_CS");
    if (!r->bake_pipeline) return false;

    SDL_Log("lightmap: %ux%u, %u charts, %u valid texels, %.2f texels/unit",
            lm->width, lm->height, lm->chart_count,
            lm->sample_count, lm->texel_density);

    return bake_lightmap_once(r);
}

bool bake_probe_grid(renderer *r, dm_probe_grid *grid, Uint32 samples) {
    if (!r || !grid || !grid->probes) return false;

    const uint64_t count =
        (uint64_t)grid->count_x * grid->count_y * grid->count_z;
    if (!count || count > UINT32_MAX) return false;

    const uint64_t output_size = count * 9u * sizeof(float[4]);
    const uint64_t input_size = count * sizeof(float[4]);
    if (output_size > UINT32_MAX || input_size > UINT32_MAX)
        return false;

    const Uint32 output_bytes = (Uint32)output_size;
    const Uint32 input_bytes = (Uint32)input_size;

    float (*positions)[4] = malloc(input_bytes);
    if (!positions) return false;

    for (uint32_t i = 0; i < (uint32_t)count; ++i)
        memcpy(positions[i], grid->probes[i].position, sizeof(positions[i]));

    BUFFER *input = upload_buffer(
        r,
        NriBufferUsageBits_SHADER_RESOURCE,
        positions,
        input_bytes, sizeof(float[4]));
    free(positions);

    const NriBufferDesc output_desc = {
        .size = output_bytes,
        .structureStride = sizeof(float[4]),
        .usage = NriBufferUsageBits_SHADER_RESOURCE_STORAGE
    };

    BUFFER *output = NULL;
    if (r->core.CreateCommittedBuffer(
            r->device,
            NriMemoryLocation_DEVICE,
            1.0f,
            &output_desc,
            &output) != NriResult_SUCCESS)
        output = NULL;

    NriPipeline *pipeline = compile_compute(
        &r->core,
        r->device,
        r->probe_layout,
        "shaders/compute.hlsl",
        "probe_cs",
        "BUILD_PROBE_CS");

    bool good = input && output && pipeline;

    if (good) {
        NriCommandAllocator *allocator = NULL;
        NriCommandBuffer *cmd = NULL;
        good = begin_commands(r, &allocator, &cmd) == NriResult_SUCCESS;

        if (good) {
            const bake_uniforms u = bake_data(r, 0u, 0u, samples, 0u);

            good = bind_probe_resources(
                r, cmd,
                input,
                r->bvh_node_buffer,
                r->bvh_triangle_buffer,
                output,
                &u,
                sizeof(u));

            if (good) {
                r->core.CmdSetPipeline(cmd, pipeline);
                

               r->core.CmdDispatch(
                    cmd,
                    &(NriDispatchDesc){
                        .workGroupNumX = (Uint32)count,
                        .workGroupNumY = 1,
                        .workGroupNumZ = 1
                    }
                );

                good = submit_commands(r, allocator, cmd);
            } else {
                abort_commands(r, allocator, cmd);
            }
        }
    }

    if (good)
        good = read_buffer(r, output, grid, output_bytes, (uint32_t)count);

    if (pipeline) r->core.DestroyPipeline(pipeline);
    release_buffer(r, output);
    release_buffer(r, input);
    return good;
}

void release_bake_resources(renderer *r) {
    if (!r || !r->device) return;

    release_buffer(r, r->bvh_node_buffer);
    release_buffer(r, r->bvh_triangle_buffer);
    release_buffer(r, r->lightmap_sample_buffer);
    release_texture(r, r->lightmap_scratch);
    if (r->bake_pipeline)
        r->core.DestroyPipeline(r->bake_pipeline);

    r->bvh_node_buffer = NULL;
    r->bvh_triangle_buffer = NULL;
    r->lightmap_sample_buffer = NULL;
    r->lightmap_scratch = NULL;
    r->bake_pipeline = NULL;
}

static bool dispatch_one(fx_state *fx, NriCommandBuffer *cmd,
                         NriPipeline *pipeline, TEXTURE *source,
                         TEXTURE *destination, const void *uniforms,
                         Uint32 uniform_size, Uint32 width, Uint32 height) {
    if (!fx || !fx->owner || !cmd || !pipeline || !destination)
        return false;

    renderer *r = fx->owner;

    if (!bind_fx_resources(
            r, cmd,
            pipeline,
            source,
            destination,
            fx->sampler,
            uniforms,
            uniform_size))
        return false;

    r->core.CmdSetPipeline(cmd, pipeline);
    r->core.CmdDispatch(
        cmd,
        &(NriDispatchDesc){
            .workGroupNumX = (width + 7u) / 8u,
            .workGroupNumY = (height + 7u) / 8u,
            .workGroupNumZ = 1
        }
    );

    return true;
}

static void release_frame_textures(fx_state *fx) {
    if (!fx || !fx->owner) return;

    renderer *r = fx->owner;

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

static void fx_deinit(fx_state *fx) {
    if (!fx || !fx->owner) return;

    renderer *r = fx->owner;
    release_frame_textures(fx);

    release_texture(r, fx->lut);

    if (fx->sampler) r->core.DestroyDescriptor(fx->sampler);
    if (fx->depth_sampler) r->core.DestroyDescriptor(fx->depth_sampler);

    if (fx->compose_pipeline)
        r->core.DestroyPipeline(fx->compose_pipeline);
    if (fx->ssao_pipeline)
        r->core.DestroyPipeline(fx->ssao_pipeline);
    if (fx->bloom_pipeline)
        r->core.DestroyPipeline(fx->bloom_pipeline);
    if (fx->grade_pipeline)
        r->core.DestroyPipeline(fx->grade_pipeline);
    if (fx->volume_pipeline)
        r->core.DestroyPipeline(fx->volume_pipeline);
    if (fx->volume_compose_pipeline)
        r->core.DestroyPipeline(fx->volume_compose_pipeline);

    memset(fx, 0, sizeof(*fx));
}

static bool make_compose_pipeline(renderer *r,
                                  NriShaderDesc *vs,
                                  NriShaderDesc *ps,
                                  NriFormat swap_format,
                                  NriPipeline **out) {
    const NriColorAttachmentDesc target = {
        .format = swap_format,
        .colorWriteMask = NriColorWriteBits_RGBA
    };

    const NriShaderDesc shaders[2] = {*vs, *ps};
    const NriGraphicsPipelineDesc desc = {
        .pipelineLayout = r->compose_layout,
        .inputAssembly = {.topology = NriTopology_TRIANGLE_LIST},
        .rasterization = {
            .fillMode = NriFillMode_SOLID,
            .cullMode = NriCullMode_NONE,
            .frontCounterClockwise = true,
            .depthClamp = false
        },
        .outputMerger = {
            .colors = &target,
            .colorNum = 1
        },
        .shaders = shaders,
        .shaderNum = 2
    };

    return r->core.CreateGraphicsPipeline(
        r->device, &desc, out) == NriResult_SUCCESS;
}

static bool fx_init(fx_state *fx, renderer *r) {
    if (!fx || !r || !r->device) return false;
    NriShaderDesc vs = {0};
    NriShaderDesc ps = {0};
    NriCommandAllocator *allocator = NULL;
    NriCommandBuffer *cmd = NULL;

    memset(fx, 0, sizeof(*fx));
    fx->owner = r;

    fx->sampler = sampler(
        r, NriFilter_LINEAR, NriFilter_LINEAR,
        NriAddressMode_CLAMP_TO_EDGE);

    fx->depth_sampler = sampler(
        r, NriFilter_NEAREST, NriFilter_NEAREST,
        NriAddressMode_CLAMP_TO_EDGE);

    if (!fx->sampler || !fx->depth_sampler) goto fail;

    vs = compile_shader(
        "shaders/vertex.hlsl", "fullscreen_vs",
        "BUILD_FULLSCREEN_VS", NriStageBits_VERTEX_SHADER);

    ps = compile_shader(
        "shaders/fragment.hlsl", "compose_fs",
        "BUILD_COMPOSE_FS", NriStageBits_FRAGMENT_SHADER);

    fx->ssao_pipeline = compile_compute(
        &r->core, r->device, r->ssao_layout,
        "shaders/compute.hlsl", "ssao_cs", "BUILD_SSAO_CS");

    fx->bloom_pipeline = compile_compute(
        &r->core, r->device, r->bloom_layout,
        "shaders/compute.hlsl", "bloom_cs", "BUILD_BLOOM_CS");

    fx->grade_pipeline = compile_compute(
        &r->core, r->device, r->grade_layout,
        "shaders/compute.hlsl", "grade_cs", "BUILD_GRADE_CS");

    fx->volume_pipeline = compile_compute(
        &r->core, r->device, r->volume_layout,
        "shaders/vision_compute.hlsl", "volume_cs",
        "BUILD_VISION_VOLUME_CS");

    fx->volume_compose_pipeline = compile_compute(
        &r->core, r->device, r->volume_compose_layout,
        "shaders/vision_compute.hlsl", "volume_compose_cs",
        "BUILD_VISION_COMPOSE_CS");

    if (!vs.bytecode || !ps.bytecode || !fx->ssao_pipeline ||
        !fx->bloom_pipeline || !fx->grade_pipeline ||
        !fx->volume_pipeline || !fx->volume_compose_pipeline)
        goto fail;

    if (!make_compose_pipeline(
            r, &vs, &ps, r->swapchain_format,
            &fx->compose_pipeline))
        goto fail;

    free_shader(&vs);
    free_shader(&ps);

    fx->lut = texture(
        r,
        NriFormat_RGBA16_SFLOAT,
        NriTextureUsageBits_SHADER_RESOURCE |
        NriTextureUsageBits_SHADER_RESOURCE_STORAGE,
        256u, 16u);
    if (!fx->lut) goto fail;

    if (begin_commands(r, &allocator, &cmd) != NriResult_SUCCESS ||
        !dispatch_one(
            fx, cmd,
            fx->grade_pipeline,
            NULL, fx->lut,
            NULL, 0,
            256u, 16u))
        goto fail;

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

static bool fx_ensure(fx_state *fx, Uint32 width, Uint32 height) {
    if (!fx || !fx->owner || !width || !height) return false;
    if (fx->hdr && fx->width == width && fx->height == height) return true;

    renderer *r = fx->owner;

    release_frame_textures(fx);
    fx->width = width;
    fx->height = height;
    fx->ao_width = (width + 1u) / 2u;
    fx->ao_height = (height + 1u) / 2u;

    const NriTextureUsageBits rt =
        NriTextureUsageBits_COLOR_ATTACHMENT |
        NriTextureUsageBits_SHADER_RESOURCE;

    const NriTextureUsageBits compute =
        NriTextureUsageBits_SHADER_RESOURCE |
        NriTextureUsageBits_SHADER_RESOURCE_STORAGE;

    fx->hdr = texture(r, NriFormat_RGBA16_SFLOAT, rt, width, height);
    fx->normal_depth = texture(
        r, NriFormat_RGBA16_SFLOAT, rt, width, height);
    fx->ao = texture(
        r, NriFormat_RGBA16_SFLOAT, compute,
        fx->ao_width, fx->ao_height);
    fx->bloom_a = texture(
        r, NriFormat_RGBA16_SFLOAT, compute,
        fx->ao_width, fx->ao_height);
    fx->bloom_b = texture(
        r, NriFormat_RGBA16_SFLOAT, compute,
        fx->ao_width, fx->ao_height);
    fx->volume = texture(
        r, NriFormat_RGBA16_SFLOAT, compute,
        fx->ao_width, fx->ao_height);
    fx->lit = texture(
        r, NriFormat_RGBA16_SFLOAT, compute,
        width, height);

    if (!fx->hdr || !fx->normal_depth || !fx->ao || !fx->bloom_a ||
        !fx->bloom_b || !fx->volume || !fx->lit) {
        release_frame_textures(fx);
        return false;
    }

    return true;
}

static bool fx_volume(fx_state *fx, NriCommandBuffer *cmd,
                      BUFFER *probes, BUFFER *beams,
                      const dm_probe_grid *grid,
                      const dm_beam_grid *beam_grid,
                      const render_frame *frame) {
    if (!fx || !fx->owner || !cmd || !probes || !beams || !grid ||
        !beam_grid || !grid->probes || !fx->volume)
        return false;

    renderer *r = fx->owner;

    const volume_uniforms u = {
        .eye_density = {frame->eye.x, frame->eye.y, frame->eye.z, 0.045f},
        .right_tan = {
            frame->right.x * frame->tan_half_fov * frame->aspect,
            frame->right.y * frame->tan_half_fov * frame->aspect,
            frame->right.z * frame->tan_half_fov * frame->aspect, 0
        },
        .up_tan = {
            frame->up.x * frame->tan_half_fov,
            frame->up.y * frame->tan_half_fov,
            frame->up.z * frame->tan_half_fov, 0
        },
        .forward_g = {
            frame->forward.x, frame->forward.y, frame->forward.z, 0.55f
        },
        .sun_intensity = {
            frame->sun.x, frame->sun.y, frame->sun.z, 2.4f
        },
        .grid_origin_spacing = {
            grid->origin.x, grid->origin.y, grid->origin.z, grid->spacing
        },
        .grid_dims_width = {
            grid->count_x, grid->count_y, grid->count_z, fx->ao_width
        },
        .height_debug = {
            fx->ao_height, fx->debug_view, beam_grid->depth, 0
        },
        .beam_origin = {
            beam_grid->origin.x, beam_grid->origin.y, beam_grid->origin.z, 0
        },
        .beam_step = {
            beam_grid->step.x, beam_grid->step.y, beam_grid->step.z, 0
        }
    };

    if (!bind_volume_resources(
            r, cmd,
            fx->normal_depth,
            fx->depth_sampler,
            probes, beams,
            fx->volume,
            &u, sizeof(u)))
        return false;

    r->core.CmdSetPipeline(cmd, fx->volume_pipeline);

    r->core.CmdDispatch(
        cmd,
        &(NriDispatchDesc){
            .workGroupNumX = (fx->ao_width + 7u) / 8u,
            .workGroupNumY = (fx->ao_height + 7u) / 8u,
            .workGroupNumZ = 1
        }
    );

    const Uint32 dimensions[4] = {
        fx->width, fx->height, fx->debug_view, 0
    };

    if (!bind_volume_compose_resources(
            r, cmd,
            fx->hdr, fx->volume, fx->normal_depth,
            fx->sampler, fx->depth_sampler,
            fx->lit,
            dimensions, sizeof(dimensions)))
        return false;

    r->core.CmdSetPipeline(cmd, fx->volume_compose_pipeline);


    r->core.CmdDispatch(
        cmd,
        &(NriDispatchDesc){
            .workGroupNumX = (fx->width + 7u) / 8u,
            .workGroupNumY = (fx->height + 7u) / 8u,
            .workGroupNumZ = 1
        }
    );

    fx->volume_ready = true;
    return true;
}

static bool run_vision_only(fx_state *fx, NriCommandBuffer *cmd) {
    if (!fx || !fx->owner || !cmd) return false;

    renderer *r = fx->owner;
    const Uint32 dimensions[4] = {
        fx->width, fx->height, fx->debug_view, 1u
    };

    if (!bind_volume_compose_resources(
            r, cmd,
            fx->hdr, fx->volume, fx->normal_depth,
            fx->sampler, fx->depth_sampler,
            fx->lit,
            dimensions, sizeof(dimensions)))
        return false;

    r->core.CmdSetPipeline(cmd, fx->volume_compose_pipeline);
    
    r->core.CmdDispatch(
        cmd,
        &(NriDispatchDesc){
            .workGroupNumX = (fx->width + 7u) / 8u,
            .workGroupNumY = (fx->height + 7u) / 8u,
            .workGroupNumZ = 1
        }
    );

    return true;
}

static bool bloom_pass(fx_state *fx, NriCommandBuffer *cmd,
                       TEXTURE *source, TEXTURE *destination,
                       Uint32 src_width, Uint32 src_height, Uint32 phase) {
    const bloom_uniforms u = {
        .src_width = src_width,
        .src_height = src_height,
        .dst_width = fx->ao_width,
        .dst_height = fx->ao_height,
        .phase = phase,
        .threshold = 1.0f,
        .knee = 0.55f,
        .strength = 1.0f * 1e2
    };

    return dispatch_one(
        fx, cmd,
        fx->bloom_pipeline,
        source, destination,
        &u, sizeof(u),
        fx->ao_width, fx->ao_height);
}

static bool fx_apply_base(fx_state *fx, NriCommandBuffer *cmd,
                          TEXTURE *swap, float tan_half_fov, float aspect) {
    if (!fx || !fx->owner || !cmd || !swap || !fx->hdr ||
        !fx->normal_depth)
        return false;

    renderer *r = fx->owner;

    const ssao_uniforms ao = {
        .width = fx->width,
        .height = fx->height,
        .ao_width = fx->ao_width,
        .ao_height = fx->ao_height,
        .tan_half_fov = tan_half_fov,
        .aspect = aspect,
        .radius = 0.65f,
        .bias = 0.035f
    };

    TEXTURE *hdr = fx->volume_ready ? fx->lit : fx->hdr;

    if (!dispatch_one(
            fx, cmd, fx->ssao_pipeline,
            fx->normal_depth, fx->ao,
            &ao, sizeof(ao),
            fx->ao_width, fx->ao_height) ||
        !bloom_pass(
            fx, cmd, fx->hdr, fx->bloom_a,
            fx->width, fx->height, 0u) ||
        !bloom_pass(
            fx, cmd, fx->bloom_a, fx->bloom_b,
            fx->ao_width, fx->ao_height, 1u) ||
        !bloom_pass(
            fx, cmd, fx->bloom_b, fx->bloom_a,
            fx->ao_width, fx->ao_height, 2u))
        return false;

    const compose_uniforms u = {
        .exposure = 1.0f,
        .ao_strength = fx->debug_view >= 3u ? 0.0f : 0.62f,
        .bloom_strength = fx->debug_view >= 3u ? 0.0f : 0.22f
    };

    if (!begin_compose_rendering(
            r, cmd, swap,
            hdr, fx->ao, fx->bloom_a, fx->lut,
            fx->sampler,
            &u, sizeof(u)))
        return false;

    r->core.CmdSetPipeline(cmd, fx->compose_pipeline);
    r->core.CmdDraw(
        cmd,
        &(NriDrawDesc){
            .vertexNum = 3,
            .instanceNum = 1,
            .baseVertex = 0,
            .baseInstance = 0
        }
    );

    r->core.CmdEndRendering(cmd);
    return true;
}

static bool fx_apply(fx_state *fx, NriCommandBuffer *cmd,
                     TEXTURE *swap, float tan_half_fov, float aspect) {
    if (!fx || !cmd || !swap) return false;

    const bool had_volume = fx->volume_ready;
    if (!had_volume) {
        if (!run_vision_only(fx, cmd)) return false;
        fx->volume_ready = true;
    }

    const bool ok = fx_apply_base(
        fx, cmd, swap, tan_half_fov, aspect);

    if (!had_volume) fx->volume_ready = false;
    return ok;
}

/*
 * Pipeline layouts are where NRI differs most from SDL_GPU.
 * These helpers are intentionally named by the shader job they describe.
 * Each one mirrors the register spaces already present in your HLSL.
 *
 * The binding helper implementations are kept in gpu.c as well so render.c
 * never learns about NRI descriptor sets/views.
 */
static bool create_pipeline_layouts(renderer *r) {
    return create_surface_layout(r) &&
           create_line_layout(r) &&
           create_sky_layout(r) &&
           create_bake_layout(r) &&
           create_probe_layout(r) &&
           create_ssao_layout(r) &&
           create_bloom_layout(r) &&
           create_grade_layout(r) &&
           create_volume_layout(r) &&
           create_volume_compose_layout(r) &&
           create_compose_layout(r);
}

static void destroy_pipeline_layouts(renderer *r) {
    if (!r) return;

    NriPipelineLayout **layouts[] = {
        &r->surface_layout,
        &r->line_layout,
        &r->sky_layout,
        &r->bake_layout,
        &r->probe_layout,
        &r->ssao_layout,
        &r->bloom_layout,
        &r->grade_layout,
        &r->volume_layout,
        &r->volume_compose_layout,
        &r->compose_layout
    };

    for (uint32_t i = 0; i < sizeof(layouts) / sizeof(layouts[0]); ++i) {
        if (*layouts[i]) {
            r->core.DestroyPipelineLayout(*layouts[i]);
            *layouts[i] = NULL;
        }
    }
}

bool r_init(renderer *r, const char *title, int width, int height) {
    if (!r) return false;

    memset(r, 0, sizeof(*r));
    r->show_volume = true;
    r->yaw = -0.78f;
    r->pitch = 0.34f;
    r->distance = 14.0f;
    r->target = v3(0.0f, 1.0f, 0.0f);

    if (!SDL_ShaderCross_Init()) return false;

    r->window = SDL_CreateWindow(
        title, width, height,
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
    device_desc.enableNRIValidation = true;
    device_desc.enableGraphicsAPIValidation = true;
    device_desc.vkBindingOffsets = (NriVKBindingOffsets){
        .sRegister = 0, .tRegister = 16,
        .bRegister = 32, .uRegister = 48
    };

    if (nriCreateDevice(&device_desc, &r->device) != NriResult_SUCCESS) {
        SDL_Log("NRI device creation failed");
        r_deinit(r);
        return false;
    }

    if (nriGetInterface(
            r->device,
            NRI_INTERFACE(NriCoreInterface),
            &r->core) != NriResult_SUCCESS ||
        nriGetInterface(
            r->device,
            NRI_INTERFACE(NriHelperInterface),
            &r->helper) != NriResult_SUCCESS ||
        nriGetInterface(
            r->device,
            NRI_INTERFACE(NriSwapChainInterface),
            &r->swapchain_api) != NriResult_SUCCESS) {
        SDL_Log("NRI interface acquisition failed");
        r_deinit(r);
        return false;
    }

    if (r->core.GetQueue(
            r->device,
            NriQueueType_GRAPHICS,
            0,
            &r->graphics_queue) != NriResult_SUCCESS) {
        r_deinit(r);
        return false;
    }

    if (!create_swapchain(r, width, height) ||
        !create_descriptor_pool(r) ||
        !create_pipeline_layouts(r)) {
        r_deinit(r);
        return false;
    }

    if (r->core.GetFormatSupport(
            r->device, NriFormat_D32_SFLOAT) &
        NriFormatSupportBits_DEPTH_STENCIL_ATTACHMENT) {
        r->depth_format = NriFormat_D32_SFLOAT;
    } else if (r->core.GetFormatSupport(
                   r->device, NriFormat_D24_UNORM_S8_UINT) &
               NriFormatSupportBits_DEPTH_STENCIL_ATTACHMENT) {
        r->depth_format = NriFormat_D24_UNORM_S8_UINT;
    } else {
        r->depth_format = NriFormat_D16_UNORM;
    }

    NriShaderDesc surface_vs = compile_shader(
        "shaders/vertex.hlsl", "surface_vs",
        "BUILD_SURFACE_VS", NriStageBits_VERTEX_SHADER);
    NriShaderDesc surface_ps = compile_shader(
        "shaders/fragment.hlsl", "surface_fs",
        "BUILD_SURFACE_FS", NriStageBits_FRAGMENT_SHADER);
    NriShaderDesc line_vs = compile_shader(
        "shaders/vertex.hlsl", "wireframe_vs",
        "BUILD_WIREFRAME_VS", NriStageBits_VERTEX_SHADER);
    NriShaderDesc line_ps = compile_shader(
        "shaders/fragment.hlsl", "wireframe_fs",
        "BUILD_WIREFRAME_FS", NriStageBits_FRAGMENT_SHADER);
    NriShaderDesc sky_vs = compile_shader(
        "shaders/vertex.hlsl", "fullscreen_vs",
        "BUILD_FULLSCREEN_VS", NriStageBits_VERTEX_SHADER);
    NriShaderDesc sky_ps = compile_shader(
        "shaders/fragment.hlsl", "sky_fs",
        "BUILD_SKY_FS", NriStageBits_FRAGMENT_SHADER);

    if (!surface_vs.bytecode || !surface_ps.bytecode ||
        !line_vs.bytecode || !line_ps.bytecode ||
        !sky_vs.bytecode || !sky_ps.bytecode) {
        free_shader(&surface_vs);
        free_shader(&surface_ps);
        free_shader(&line_vs);
        free_shader(&line_ps);
        free_shader(&sky_vs);
        free_shader(&sky_ps);
        r_deinit(r);
        return false;
    }

    r->solid_pipeline = make_surface_pipeline(
        r, &r->core, r->surface_layout,
        &surface_vs, &surface_ps);
    r->line_pipeline = make_line_pipeline(
        r, r->line_layout, &line_vs, &line_ps);
    r->sky_pipeline = make_sky_pipeline(
        r, r->sky_layout, &sky_vs, &sky_ps);

    free_shader(&surface_vs);
    free_shader(&surface_ps);
    free_shader(&line_vs);
    free_shader(&line_ps);
    free_shader(&sky_vs);
    free_shader(&sky_ps);

    if (!r->solid_pipeline || !r->line_pipeline || !r->sky_pipeline ||
        !fx_init(&r->fx, r)) {
        r_deinit(r);
        return false;
    }

    SDL_Log("GPU backend: NRI");
    SDL_Log("depth format: %s",
            r->depth_format == NriFormat_D32_SFLOAT ? "D32_FLOAT" :
            r->depth_format == NriFormat_D24_UNORM_S8_UINT ? "D24S8" :
            "D16_UNORM");
    SDL_Log("SDL_image: %d", IMG_Version());
    return true;
}

bool draw_frame(renderer *r, const render_frame *frame) {
    if (!r || !frame || !r->device || !r->solid_pipeline ||
        !r->sky_pipeline || !r->vertex_buffer || !r->lightmap_texture ||
        !r->lightmap_sampler)
        return false;

    uint32_t width = 0;
    uint32_t height = 0;
    SDL_GetWindowSizeInPixels(
        r->window, (int *)&width, (int *)&height);
    if (!width || !height) return true;

    if (!r->swapchain || width != r->swapchain_width || height != r->swapchain_height) {
        if (r->core.QueueWaitIdle(r->graphics_queue) != NriResult_SUCCESS) return false;
        destroy_swapchain(r);
        if (!create_swapchain(r, width, height)) return false;
    }

    if (!fx_ensure(&r->fx, width, height) ||
        !ensure_depth_texture(r, width, height)) return false;

    uint32_t swap_index = 0;
    if (!acquire_swapchain_texture(r, &swap_index)) {
        destroy_swapchain(r);
        return false;
    }
    r->current_swap_index = swap_index;

    TEXTURE *swap = r->swapchain_textures[swap_index];

    NriCommandAllocator *allocator = NULL;
    NriCommandBuffer *cmd = NULL;
    if (begin_commands(r, &allocator, &cmd) != NriResult_SUCCESS)
        goto failed_frame;

    camera_uniforms camera = {0};
    memcpy(camera.mvp, frame->mvp, sizeof(camera.mvp));
    memcpy(camera.view, frame->view, sizeof(camera.view));

    const sky_uniforms sky = {
        .camera_right = {
            frame->right.x * frame->tan_half_fov * frame->aspect,
            frame->right.y * frame->tan_half_fov * frame->aspect,
            frame->right.z * frame->tan_half_fov * frame->aspect, 0
        },
        .camera_up = {
            frame->up.x * frame->tan_half_fov,
            frame->up.y * frame->tan_half_fov,
            frame->up.z * frame->tan_half_fov, 0
        },
        .camera_forward = {
            frame->forward.x, frame->forward.y, frame->forward.z, 0
        },
        .sky_zenith = {0.22f, 0.42f, 0.78f, 1},
        .sky_horizon = {0.68f, 0.76f, 0.88f, 1},
        .sun_direction_intensity = {
            frame->sun.x, frame->sun.y, frame->sun.z, 2.4f
        },
        .sun_color_radius = {1.00f, 0.94f, 0.84f, 0.00465f}
    };

    if (!begin_scene_rendering(
            r, cmd,
            r->fx.hdr,
            r->fx.normal_depth,
            r->depth_texture,
            width, height)) goto failed_frame;

    if (!bind_sky_resources(r, cmd, &sky, sizeof(sky)))
        goto failed_frame;
    r->core.CmdSetPipeline(cmd, r->sky_pipeline);

    r->core.CmdDraw(
        cmd,
        &(NriDrawDesc){
            .vertexNum = 3,
            .instanceNum = 1,
            .baseVertex = 0,
            .baseInstance = 0
        }
    );

    const NriVertexBufferDesc vertex = {
        .buffer = r->vertex_buffer,
        .offset = 0,
        .stride = sizeof(render_vertex)
    };
    r->core.CmdSetVertexBuffers(cmd, 0, &vertex, 1);
    r->core.CmdSetPipeline(cmd, r->solid_pipeline);

    if (!bind_camera_resources(r, cmd, &camera, sizeof(camera)))
        goto failed_frame;

    for (uint32_t i = 0; i < r->draw_count; ++i) {
        const draw_range *draw = &r->draws[i];
        const render_material *m = &r->materials[draw->material];

        const material_uniforms material = {
            .base_color_factor = {
                m->data.base_color[0], m->data.base_color[1],
                m->data.base_color[2], m->data.base_color[3]
            },
            .emissive_metallic = {
                m->data.emissive[0], m->data.emissive[1],
                m->data.emissive[2], m->data.metallic
            },
            .roughness_normal_ao_sun = {
                m->data.roughness, m->data.normal_scale,
                m->data.occlusion_strength, 2.4f
            },
            .sun_direction = {
                frame->sun.x, frame->sun.y, frame->sun.z, 0
            },
            .sun_color = {1.00f, 0.94f, 0.84f, 1},
            .camera_position = {
                frame->eye.x, frame->eye.y, frame->eye.z,
                r->debug_view == 1u ? 2.0f :
                (r->has_bake ? 1.0f : 0.0f)
            }
        };

        if (!bind_surface_resources(
                r, cmd, m,
                r->lightmap_texture,
                r->material_sampler,
                r->lightmap_sampler,
                &material,
                sizeof(material)))
            goto failed_frame;


        r->core.CmdDraw(
            cmd,
            &(NriDrawDesc){
                .vertexNum = draw->count,
                .instanceNum = 1,
                .baseVertex = draw->first,
                .baseInstance = 0
            }
        );
    }

    if (r->show_debug && r->debug_vertex_count) {
        r->core.CmdSetPipeline(cmd, r->line_pipeline);
        if (!bind_line_resources(
                r, cmd, camera.mvp, sizeof(camera.mvp)))
            goto failed_frame;


        r->core.CmdDraw(
            cmd,
            &(NriDrawDesc){
                .vertexNum = r->debug_vertex_count,
                .instanceNum = 1,
                .baseVertex = r->debug_vertex_start,
                .baseInstance = 0
            }
        );
    }

    r->core.CmdEndRendering(cmd);

    r->fx.volume_ready = false;
    r->fx.debug_view = r->debug_view;

    static uint32_t last_logged_view = UINT32_MAX;
    if (last_logged_view != r->fx.debug_view) {
        SDL_Log("GPU debug view: %u | fog: %d | bake: %d",
                r->fx.debug_view, r->show_volume, r->has_bake);
        last_logged_view = r->fx.debug_view;
    }

    if (r->show_volume && r->has_bake &&
        (r->debug_view == 0u || r->debug_view >= 3u) &&
        r->volume_probe_buffer && r->beam_buffer &&
        !fx_volume(
            &r->fx, cmd,
            r->volume_probe_buffer,
            r->beam_buffer,
            &r->volume_probes,
            &r->beams,
            frame))
        goto failed_frame;

    if (!fx_apply(
            &r->fx, cmd, swap,
            frame->tan_half_fov,
            frame->aspect))
        goto failed_frame;

    if (!submit_frame(r, allocator, cmd, swap_index))
        return false;

    return true;

failed_frame:
    abort_commands(r, allocator, cmd);
    destroy_swapchain(r);
    return false;
}

bool bake_worker_init(renderer *r) {
    if (!r) return false;
    memset(r, 0, sizeof(*r));

    NriDeviceCreationDesc device_desc = {0};
    device_desc.graphicsAPI = NriGraphicsAPI_VK;
    device_desc.vkBindingOffsets = (NriVKBindingOffsets){
        .sRegister = 0, .tRegister = 16,
        .bRegister = 32, .uRegister = 48
    };

    if (nriCreateDevice(&device_desc, &r->device) != NriResult_SUCCESS)
        return false;

    if (nriGetInterface(
            r->device,
            NRI_INTERFACE(NriCoreInterface),
            &r->core) != NriResult_SUCCESS ||
        nriGetInterface(
            r->device,
            NRI_INTERFACE(NriHelperInterface),
            &r->helper) != NriResult_SUCCESS ||
        r->core.GetQueue(
            r->device,
            NriQueueType_GRAPHICS,
            0,
            &r->graphics_queue) != NriResult_SUCCESS) {
        bake_worker_deinit(r);
        return false;
    }

    if (!create_descriptor_pool(r) || !create_pipeline_layouts(r)) {
        bake_worker_deinit(r);
        return false;
    }

    r->lightmap_sampler = sampler(
        r,
        NriFilter_LINEAR,
        NriFilter_LINEAR,
        NriAddressMode_CLAMP_TO_EDGE);

    r->lightmap_texture = texture(
        r,
        NriFormat_RGBA16_SFLOAT,
        NriTextureUsageBits_SHADER_RESOURCE,
        1, 1);

    if (!r->lightmap_sampler || !r->lightmap_texture) {
        bake_worker_deinit(r);
        return false;
    }

    return true;
}

void bake_worker_deinit(renderer *r) {
    if (!r) return;

    free_probe_grid(&r->volume_probes);
    dm_beam_free(&r->beams);

    if (r->device) {
        if (r->graphics_queue)
            r->core.QueueWaitIdle(r->graphics_queue);

        clear_temporary(r);

        release_bake_resources(r);
        release_buffer(r, r->volume_probe_buffer);
        release_buffer(r, r->beam_buffer);
        release_texture(r, r->lightmap_texture);

        if (r->lightmap_sampler)
            r->core.DestroyDescriptor(r->lightmap_sampler);

        destroy_pipeline_layouts(r);

        if (r->descriptor_pool)
            r->core.DestroyDescriptorPool(r->descriptor_pool);

        nriDestroyDevice(r->device);
    }

    free(r->temporary_descriptors);
    free(r->temporary_buffers);
    free(r->texture_states);

    memset(r, 0, sizeof(*r));
}

void r_deinit(renderer *r) {
    if (!r) return;

    free(r->vertices);
    free(r->draws);
    free_probe_grid(&r->volume_probes);
    dm_beam_free(&r->beams);

    if (r->device) {
        if (r->graphics_queue)
            r->core.QueueWaitIdle(r->graphics_queue);

        clear_temporary(r);

        fx_deinit(&r->fx);

        if (r->image_textures) {
            for (uint32_t i = 0; i < r->image_texture_count; ++i)
                release_texture(r, r->image_textures[i]);
        }

        free(r->image_textures);
        free(r->materials);

        release_texture(r, r->default_white);
        release_texture(r, r->default_normal);
        if (r->material_sampler)
            r->core.DestroyDescriptor(r->material_sampler);

        release_buffer(r, r->vertex_buffer);
        release_bake_resources(r);
        release_buffer(r, r->volume_probe_buffer);
        release_buffer(r, r->beam_buffer);
        release_texture(r, r->depth_texture);
        release_texture(r, r->lightmap_texture);

        if (r->lightmap_sampler)
            r->core.DestroyDescriptor(r->lightmap_sampler);

        if (r->sky_pipeline)
            r->core.DestroyPipeline(r->sky_pipeline);
        if (r->solid_pipeline)
            r->core.DestroyPipeline(r->solid_pipeline);
        if (r->line_pipeline)
            r->core.DestroyPipeline(r->line_pipeline);

        destroy_swapchain(r);
        destroy_pipeline_layouts(r);

        if (r->descriptor_pool)
            r->core.DestroyDescriptorPool(r->descriptor_pool);

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
    SDL_ShaderCross_Quit();
}

#else
static Uint8 *compile_spirv(const char *path, const char *entrypoint,
                            const char *define,
                            SDL_ShaderCross_ShaderStage stage, size_t *size) {
    char *source = shader_source(path);
    if (!source) {
        SDL_Log("SDL_LoadFile(%s) failed: %s", path, SDL_GetError());
        return NULL;
    }

    SDL_ShaderCross_HLSL_Define defines[2] = {
        {.name = (char *)define, .value = NULL}, {0}
    };
    const SDL_ShaderCross_HLSL_Info hlsl = {
        .source = source,
        .entrypoint = entrypoint,
        .include_dir = "shaders",
        .defines = defines,
        .shader_stage = stage,
        .props = 0
    };

    Uint8 *spirv = SDL_ShaderCross_CompileSPIRVFromHLSL(&hlsl, size);
    SDL_free(source);
    if (!spirv) {
        SDL_Log("shadercross HLSL->SPIR-V failed for %s:%s: %s",
                path, entrypoint, SDL_GetError());
    }
    return spirv;
}

static SDL_GPUShader *compile_shader(DEVICE *device, const char *path,
                                     const char *entrypoint, const char *define,
                                     SDL_ShaderCross_ShaderStage stage) {
    size_t spirv_size = 0;
    Uint8 *spirv = compile_spirv(path, entrypoint, define, stage, &spirv_size);
    if (!spirv) return NULL;

    SDL_ShaderCross_GraphicsShaderMetadata *meta =
        SDL_ShaderCross_ReflectGraphicsSPIRV(spirv, spirv_size, 0);
    if (!meta) {
        SDL_Log("shader reflection failed for %s:%s: %s",
                path, entrypoint, SDL_GetError());
        SDL_free(spirv);
        return NULL;
    }

    const SDL_ShaderCross_SPIRV_Info info = {
        .bytecode = spirv,
        .bytecode_size = spirv_size,
        .entrypoint = entrypoint,
        .shader_stage = stage,
        .props = 0
    };
    SDL_GPUShader *shader = SDL_ShaderCross_CompileGraphicsShaderFromSPIRV(
        device, &info, &meta->resource_info, 0);

    SDL_free(meta);
    SDL_free(spirv);
    return shader;
}

static COMPUTE_PIPELINE *compile_compute(DEVICE *device, const char *path,
                                         const char *entrypoint,
                                         const char *define) {
    size_t spirv_size = 0;
    Uint8 *spirv = compile_spirv(path, entrypoint, define,
                                 SDL_SHADERCROSS_SHADERSTAGE_COMPUTE,
                                 &spirv_size);
    if (!spirv) return NULL;

    SDL_ShaderCross_ComputePipelineMetadata *meta =
        SDL_ShaderCross_ReflectComputeSPIRV(spirv, spirv_size, 0);
    if (!meta) {
        SDL_Log("compute reflection failed for %s:%s: %s",
                path, entrypoint, SDL_GetError());
        SDL_free(spirv);
        return NULL;
    }

    const SDL_ShaderCross_SPIRV_Info info = {
        .bytecode = spirv,
        .bytecode_size = spirv_size,
        .entrypoint = entrypoint,
        .shader_stage = SDL_SHADERCROSS_SHADERSTAGE_COMPUTE,
        .props = 0
    };
    COMPUTE_PIPELINE *pipeline =
        SDL_ShaderCross_CompileComputePipelineFromSPIRV(device, &info, meta, 0);

    SDL_free(meta);
    SDL_free(spirv);
    return pipeline;
}

static GRAPHICS_PIPELINE *make_surface_pipeline(renderer *r,
                                                SDL_GPUShader *vs,
                                                SDL_GPUShader *ps) {
    const SDL_GPUVertexBufferDescription vb = {
        .slot = 0,
        .pitch = (Uint32)sizeof(render_vertex),
        .input_rate = SDL_GPU_VERTEXINPUTRATE_VERTEX
    };
    const SDL_GPUVertexAttribute attrs[4] = {
        {.location = 0, .buffer_slot = 0,
         .format = SDL_GPU_VERTEXELEMENTFORMAT_FLOAT3,
         .offset = (Uint32)offsetof(render_vertex, x)},
        {.location = 1, .buffer_slot = 0,
         .format = SDL_GPU_VERTEXELEMENTFORMAT_FLOAT3,
         .offset = (Uint32)offsetof(render_vertex, nx)},
        {.location = 2, .buffer_slot = 0,
         .format = SDL_GPU_VERTEXELEMENTFORMAT_FLOAT2,
         .offset = (Uint32)offsetof(render_vertex, u)},
        {.location = 3, .buffer_slot = 0,
         .format = SDL_GPU_VERTEXELEMENTFORMAT_FLOAT2,
         .offset = (Uint32)offsetof(render_vertex, lu)}
    };
    const SDL_GPUColorTargetDescription targets[2] = {
        {.format = SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT},
        {.format = SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT}
    };

    return SDL_CreateGPUGraphicsPipeline(r->device,
        &(SDL_GPUGraphicsPipelineCreateInfo){
            .vertex_shader = vs,
            .fragment_shader = ps,
            .vertex_input_state = {
                .vertex_buffer_descriptions = &vb,
                .num_vertex_buffers = 1,
                .vertex_attributes = attrs,
                .num_vertex_attributes = 4
            },
            .primitive_type = SDL_GPU_PRIMITIVETYPE_TRIANGLELIST,
            .rasterizer_state = {
                .fill_mode = SDL_GPU_FILLMODE_FILL,
                .cull_mode = SDL_GPU_CULLMODE_NONE,
                .front_face = SDL_GPU_FRONTFACE_COUNTER_CLOCKWISE,
                .enable_depth_clip = true
            },
            .multisample_state = {.sample_count = SDL_GPU_SAMPLECOUNT_1},
            .depth_stencil_state = {
                .compare_op = SDL_GPU_COMPAREOP_LESS,
                .enable_depth_test = true,
                .enable_depth_write = true
            },
            .target_info = {
                .color_target_descriptions = targets,
                .num_color_targets = 2,
                .depth_stencil_format = r->depth_format,
                .has_depth_stencil_target = true
            }
        });
}

static GRAPHICS_PIPELINE *make_line_pipeline(renderer *r,
                                             SDL_GPUShader *vs,
                                             SDL_GPUShader *ps) {
    const SDL_GPUVertexBufferDescription vb = {
        .slot = 0,
        .pitch = (Uint32)sizeof(render_vertex),
        .input_rate = SDL_GPU_VERTEXINPUTRATE_VERTEX
    };
    const SDL_GPUVertexAttribute attrs[2] = {
        {.location = 0, .buffer_slot = 0,
         .format = SDL_GPU_VERTEXELEMENTFORMAT_FLOAT3,
         .offset = (Uint32)offsetof(render_vertex, x)},
        {.location = 1, .buffer_slot = 0,
         .format = SDL_GPU_VERTEXELEMENTFORMAT_FLOAT4,
         .offset = (Uint32)offsetof(render_vertex, r)}
    };
    const SDL_GPUColorTargetDescription targets[2] = {
        {.format = SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT},
        {.format = SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT,
         .blend_state = {.color_write_mask = 0,
                         .enable_color_write_mask = true}}
    };

    return SDL_CreateGPUGraphicsPipeline(r->device,
        &(SDL_GPUGraphicsPipelineCreateInfo){
            .vertex_shader = vs,
            .fragment_shader = ps,
            .vertex_input_state = {
                .vertex_buffer_descriptions = &vb,
                .num_vertex_buffers = 1,
                .vertex_attributes = attrs,
                .num_vertex_attributes = 2
            },
            .primitive_type = SDL_GPU_PRIMITIVETYPE_LINELIST,
            .rasterizer_state = {
                .fill_mode = SDL_GPU_FILLMODE_FILL,
                .cull_mode = SDL_GPU_CULLMODE_NONE,
                .front_face = SDL_GPU_FRONTFACE_COUNTER_CLOCKWISE,
                .enable_depth_clip = true
            },
            .multisample_state = {.sample_count = SDL_GPU_SAMPLECOUNT_1},
            .depth_stencil_state = {
                .compare_op = SDL_GPU_COMPAREOP_LESS_OR_EQUAL,
                .enable_depth_test = true,
                .enable_depth_write = false
            },
            .target_info = {
                .color_target_descriptions = targets,
                .num_color_targets = 2,
                .depth_stencil_format = r->depth_format,
                .has_depth_stencil_target = true
            }
        });
}

static GRAPHICS_PIPELINE *make_sky_pipeline(renderer *r,
                                            SDL_GPUShader *vs,
                                            SDL_GPUShader *ps) {
    const SDL_GPUColorTargetDescription targets[2] = {
        {.format = SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT},
        {.format = SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT}
    };

    return SDL_CreateGPUGraphicsPipeline(r->device,
        &(SDL_GPUGraphicsPipelineCreateInfo){
            .vertex_shader = vs,
            .fragment_shader = ps,
            .primitive_type = SDL_GPU_PRIMITIVETYPE_TRIANGLELIST,
            .rasterizer_state = {
                .fill_mode = SDL_GPU_FILLMODE_FILL,
                .cull_mode = SDL_GPU_CULLMODE_NONE,
                .front_face = SDL_GPU_FRONTFACE_COUNTER_CLOCKWISE,
                .enable_depth_clip = true
            },
            .multisample_state = {.sample_count = SDL_GPU_SAMPLECOUNT_1},
            .target_info = {
                .color_target_descriptions = targets,
                .num_color_targets = 2,
                .depth_stencil_format = r->depth_format,
                .has_depth_stencil_target = true
            }
        });
}

static TEXTURE *texture(DEVICE *device, SDL_GPUTextureFormat format,
                        SDL_GPUTextureUsageFlags usage,
                        Uint32 width, Uint32 height) {
    return SDL_CreateGPUTexture(device, &(SDL_GPUTextureCreateInfo){
        .type = SDL_GPU_TEXTURETYPE_2D,
        .format = format,
        .usage = usage,
        .width = width,
        .height = height,
        .layer_count_or_depth = 1,
        .num_levels = 1,
        .sample_count = SDL_GPU_SAMPLECOUNT_1
    });
}

static BUFFER *upload_buffer(renderer *r, SDL_GPUBufferUsageFlags usage,
                             const void *data, size_t bytes) {
    if (!r || !r->device || !data || !bytes || bytes > UINT32_MAX) return NULL;

    BUFFER *buffer = SDL_CreateGPUBuffer(r->device,
        &(SDL_GPUBufferCreateInfo){.usage = usage, .size = (Uint32)bytes});
    if (!buffer) return NULL;

    SDL_GPUTransferBuffer *transfer = SDL_CreateGPUTransferBuffer(r->device,
        &(SDL_GPUTransferBufferCreateInfo){
            .usage = SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD,
            .size = (Uint32)bytes
        });
    if (!transfer) {
        SDL_ReleaseGPUBuffer(r->device, buffer);
        return NULL;
    }

    void *dst = SDL_MapGPUTransferBuffer(r->device, transfer, false);
    if (!dst) {
        SDL_ReleaseGPUTransferBuffer(r->device, transfer);
        SDL_ReleaseGPUBuffer(r->device, buffer);
        return NULL;
    }
    memcpy(dst, data, bytes);
    SDL_UnmapGPUTransferBuffer(r->device, transfer);

    SDL_GPUCommandBuffer *cmd = SDL_AcquireGPUCommandBuffer(r->device);
    if (!cmd) {
        SDL_ReleaseGPUTransferBuffer(r->device, transfer);
        SDL_ReleaseGPUBuffer(r->device, buffer);
        return NULL;
    }

    SDL_GPUCopyPass *copy = SDL_BeginGPUCopyPass(cmd);
    if (!copy) {
        SDL_CancelGPUCommandBuffer(cmd);
        SDL_ReleaseGPUTransferBuffer(r->device, transfer);
        SDL_ReleaseGPUBuffer(r->device, buffer);
        return NULL;
    }

    SDL_UploadToGPUBuffer(copy,
        &(SDL_GPUTransferBufferLocation){.transfer_buffer = transfer, .offset = 0},
        &(SDL_GPUBufferRegion){.buffer = buffer, .offset = 0, .size = (Uint32)bytes},
        false);
    SDL_EndGPUCopyPass(copy);

    if (!SDL_SubmitGPUCommandBuffer(cmd)) {
        SDL_ReleaseGPUTransferBuffer(r->device, transfer);
        SDL_ReleaseGPUBuffer(r->device, buffer);
        return NULL;
    }

    SDL_ReleaseGPUTransferBuffer(r->device, transfer);
    return buffer;
}

static TEXTURE *pixel_texture(renderer *r, Uint8 red, Uint8 green,
                              Uint8 blue, Uint8 alpha) {
    TEXTURE *result = texture(r->device, SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM,
                              SDL_GPU_TEXTUREUSAGE_SAMPLER, 1, 1);
    if (!result) return NULL;

    SDL_GPUTransferBuffer *transfer = SDL_CreateGPUTransferBuffer(r->device,
        &(SDL_GPUTransferBufferCreateInfo){
            .usage = SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD,
            .size = 4
        });
    if (!transfer) {
        SDL_ReleaseGPUTexture(r->device, result);
        return NULL;
    }

    Uint8 *pixels = SDL_MapGPUTransferBuffer(r->device, transfer, false);
    if (!pixels) {
        SDL_ReleaseGPUTransferBuffer(r->device, transfer);
        SDL_ReleaseGPUTexture(r->device, result);
        return NULL;
    }
    pixels[0] = red;
    pixels[1] = green;
    pixels[2] = blue;
    pixels[3] = alpha;
    SDL_UnmapGPUTransferBuffer(r->device, transfer);

    SDL_GPUCommandBuffer *cmd = SDL_AcquireGPUCommandBuffer(r->device);
    if (!cmd) {
        SDL_ReleaseGPUTransferBuffer(r->device, transfer);
        SDL_ReleaseGPUTexture(r->device, result);
        return NULL;
    }

    SDL_GPUCopyPass *copy = SDL_BeginGPUCopyPass(cmd);
    if (!copy) {
        SDL_CancelGPUCommandBuffer(cmd);
        SDL_ReleaseGPUTransferBuffer(r->device, transfer);
        SDL_ReleaseGPUTexture(r->device, result);
        return NULL;
    }

    SDL_UploadToGPUTexture(copy,
        &(SDL_GPUTextureTransferInfo){.transfer_buffer = transfer},
        &(SDL_GPUTextureRegion){
            .texture = result,
            .mip_level = 0,
            .layer = 0,
            .x = 0, .y = 0, .z = 0,
            .w = 1, .h = 1, .d = 1
        }, false);
    SDL_EndGPUCopyPass(copy);

    if (!SDL_SubmitGPUCommandBuffer(cmd)) {
        SDL_ReleaseGPUTransferBuffer(r->device, transfer);
        SDL_ReleaseGPUTexture(r->device, result);
        return NULL;
    }

    SDL_ReleaseGPUTransferBuffer(r->device, transfer);
    return result;
}

void release_texture(renderer *r, TEXTURE *value) {
    if (r && r->device && value) SDL_ReleaseGPUTexture(r->device, value);
}

void release_buffer(renderer *r, BUFFER *value) {
    if (r && r->device && value) SDL_ReleaseGPUBuffer(r->device, value);
}

static bool ensure_depth_texture(renderer *r, Uint32 width, Uint32 height) {
    if (r->depth_texture && r->depth_width == width && r->depth_height == height)
        return true;

    release_texture(r, r->depth_texture);
    r->depth_texture = texture(r->device, r->depth_format,
                               SDL_GPU_TEXTUREUSAGE_DEPTH_STENCIL_TARGET,
                               width, height);
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

static bool load_images(renderer *r, const gltf_scene *visual) {
    r->image_texture_count = visual->image_count;
    if (!visual->image_count) return true;

    r->image_textures = calloc(visual->image_count, sizeof(*r->image_textures));
    if (!r->image_textures) return false;

    SDL_GPUCommandBuffer *cmd = SDL_AcquireGPUCommandBuffer(r->device);
    if (!cmd) return false;
    SDL_GPUCopyPass *copy = SDL_BeginGPUCopyPass(cmd);
    if (!copy) {
        SDL_CancelGPUCommandBuffer(cmd);
        return false;
    }

    for (uint32_t i = 0; i < visual->image_count; ++i) {
        const gltf_image *image = &visual->images[i];
        if (!image->bytes.data || !image->bytes.size) continue;
        SDL_IOStream *io = SDL_IOFromConstMem(image->bytes.data, image->bytes.size);
        if (!io) continue;

        r->image_textures[i] = IMG_LoadGPUTextureTyped_IO(
            r->device, copy, io, true, image_type(image->mime), NULL, NULL);
        if (!r->image_textures[i]) {
            SDL_Log("SDL_image could not decode GLB image %u (%s): %s",
                    i, image->mime[0] ? image->mime : "unknown", SDL_GetError());
        }
    }

    SDL_EndGPUCopyPass(copy);
    return SDL_SubmitGPUCommandBuffer(cmd);
}

static TEXTURE *resolve_texture(renderer *r, const gltf_scene *visual,
                                int32_t texture_index, TEXTURE *fallback) {
    if (texture_index < 0 || (uint32_t)texture_index >= visual->texture_count)
        return fallback;
    const int32_t image = visual->textures[texture_index].image;
    if (image < 0 || (uint32_t)image >= r->image_texture_count ||
        !r->image_textures[image]) return fallback;
    return r->image_textures[image];
}

static void release_scene_resources(renderer *r) {
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
    if (r->material_sampler) SDL_ReleaseGPUSampler(r->device, r->material_sampler);
    release_buffer(r, r->vertex_buffer);
    release_texture(r, r->lightmap_texture);
    if (r->lightmap_sampler) SDL_ReleaseGPUSampler(r->device, r->lightmap_sampler);

    r->default_white = NULL;
    r->default_normal = NULL;
    r->material_sampler = NULL;
    r->vertex_buffer = NULL;
    r->lightmap_texture = NULL;
    r->lightmap_sampler = NULL;
}

bool upload_scene(renderer *r, const gltf_scene *visual) {
    if (!r || !r->device || !visual || !r->vertices || !r->vertex_count)
        return false;

    release_scene_resources(r);

    r->default_white = pixel_texture(r, 255, 255, 255, 255);
    r->default_normal = pixel_texture(r, 128, 128, 255, 255);
    r->material_sampler = SDL_CreateGPUSampler(r->device,
        &(SDL_GPUSamplerCreateInfo){
            .min_filter = SDL_GPU_FILTER_LINEAR,
            .mag_filter = SDL_GPU_FILTER_LINEAR,
            .mipmap_mode = SDL_GPU_SAMPLERMIPMAPMODE_NEAREST,
            .address_mode_u = SDL_GPU_SAMPLERADDRESSMODE_REPEAT,
            .address_mode_v = SDL_GPU_SAMPLERADDRESSMODE_REPEAT,
            .address_mode_w = SDL_GPU_SAMPLERADDRESSMODE_REPEAT
        });
    if (!r->default_white || !r->default_normal || !r->material_sampler ||
        !load_images(r, visual)) return false;

    r->material_count = visual->material_count;
    r->materials = calloc(r->material_count, sizeof(*r->materials));
    if (!r->materials) return false;

    for (uint32_t i = 0; i < r->material_count; ++i) {
        render_material *m = &r->materials[i];
        m->data = visual->materials[i];
        m->base_color = resolve_texture(r, visual, m->data.base_color_texture,
                                        r->default_white);
        m->metallic_roughness = resolve_texture(
            r, visual, m->data.metallic_roughness_texture, r->default_white);
        m->normal = resolve_texture(r, visual, m->data.normal_texture,
                                    r->default_normal);
        m->occlusion = resolve_texture(r, visual, m->data.occlusion_texture,
                                       r->default_white);
        m->emissive = resolve_texture(r, visual, m->data.emissive_texture,
                                      r->default_white);
    }

    r->vertex_buffer = upload_buffer(
        r, SDL_GPU_BUFFERUSAGE_VERTEX, r->vertices,
        (size_t)r->vertex_count * sizeof(*r->vertices));

    r->lightmap_sampler = SDL_CreateGPUSampler(r->device,
        &(SDL_GPUSamplerCreateInfo){
            .min_filter = SDL_GPU_FILTER_LINEAR,
            .mag_filter = SDL_GPU_FILTER_LINEAR,
            .mipmap_mode = SDL_GPU_SAMPLERMIPMAPMODE_NEAREST,
            .address_mode_u = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE,
            .address_mode_v = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE,
            .address_mode_w = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE
        });
    r->lightmap_texture = pixel_texture(r, 0, 0, 0, 255);

    return r->vertex_buffer && r->lightmap_sampler && r->lightmap_texture;
}

static TEXTURE *create_lightmap_texture(renderer *r, Uint32 width, Uint32 height) {
    return texture(r->device, SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT,
                   SDL_GPU_TEXTUREUSAGE_SAMPLER |
                       SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_WRITE,
                   width, height);
}

static bool transfer_size(uint32_t width, uint32_t height, Uint32 *out) {
    const uint64_t bytes = (uint64_t)width * (uint64_t)height * 8u;
    if (!width || !height || bytes > UINT32_MAX) return false;
    *out = (Uint32)bytes;
    return true;
}

TEXTURE *upload_lightmap(renderer *r, const dm_cached_lightmap *cached) {
    if (!r || !r->device || !cached || !cached->pixels) return NULL;

    Uint32 bytes = 0;
    if (!transfer_size(cached->width, cached->height, &bytes)) return NULL;

    TEXTURE *result = create_lightmap_texture(r, cached->width, cached->height);
    SDL_GPUTransferBuffer *transfer = SDL_CreateGPUTransferBuffer(r->device,
        &(SDL_GPUTransferBufferCreateInfo){
            .usage = SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD,
            .size = bytes
        });
    if (!result || !transfer) goto fail;

    void *mapped = SDL_MapGPUTransferBuffer(r->device, transfer, false);
    if (!mapped) goto fail;
    memcpy(mapped, cached->pixels, bytes);
    SDL_UnmapGPUTransferBuffer(r->device, transfer);

    SDL_GPUCommandBuffer *cmd = SDL_AcquireGPUCommandBuffer(r->device);
    if (!cmd) goto fail;
    SDL_GPUCopyPass *pass = SDL_BeginGPUCopyPass(cmd);
    if (!pass) {
        SDL_CancelGPUCommandBuffer(cmd);
        goto fail;
    }

    SDL_UploadToGPUTexture(pass,
        &(SDL_GPUTextureTransferInfo){.transfer_buffer = transfer},
        &(SDL_GPUTextureRegion){
            .texture = result,
            .w = cached->width,
            .h = cached->height,
            .d = 1
        }, false);
    SDL_EndGPUCopyPass(pass);
    if (!SDL_SubmitGPUCommandBuffer(cmd)) goto fail;

    SDL_ReleaseGPUTransferBuffer(r->device, transfer);
    return result;

fail:
    if (transfer) SDL_ReleaseGPUTransferBuffer(r->device, transfer);
    release_texture(r, result);
    return NULL;
}

bool download_lightmap(renderer *r, dm_cached_lightmap *out) {
    if (!r || !r->device || !r->lightmap_texture || !out) return false;

    Uint32 bytes = 0;
    if (!transfer_size(r->lightmap_width, r->lightmap_height, &bytes)) return false;

    SDL_GPUTransferBuffer *transfer = SDL_CreateGPUTransferBuffer(r->device,
        &(SDL_GPUTransferBufferCreateInfo){
            .usage = SDL_GPU_TRANSFERBUFFERUSAGE_DOWNLOAD,
            .size = bytes
        });
    if (!transfer) return false;

    SDL_GPUCommandBuffer *cmd = SDL_AcquireGPUCommandBuffer(r->device);
    if (!cmd) {
        SDL_ReleaseGPUTransferBuffer(r->device, transfer);
        return false;
    }

    SDL_GPUCopyPass *pass = SDL_BeginGPUCopyPass(cmd);
    if (!pass) {
        SDL_CancelGPUCommandBuffer(cmd);
        SDL_ReleaseGPUTransferBuffer(r->device, transfer);
        return false;
    }

    SDL_DownloadFromGPUTexture(pass,
        &(SDL_GPUTextureRegion){
            .texture = r->lightmap_texture,
            .w = r->lightmap_width,
            .h = r->lightmap_height,
            .d = 1
        },
        &(SDL_GPUTextureTransferInfo){.transfer_buffer = transfer});
    SDL_EndGPUCopyPass(pass);

    SDL_GPUFence *fence = SDL_SubmitGPUCommandBufferAndAcquireFence(cmd);
    if (!fence) {
        SDL_ReleaseGPUTransferBuffer(r->device, transfer);
        return false;
    }

    bool good = SDL_WaitForGPUFences(r->device, true, &fence, 1);
    SDL_ReleaseGPUFence(r->device, fence);

    if (good) {
        void *mapped = SDL_MapGPUTransferBuffer(r->device, transfer, false);
        if (!mapped) {
            good = false;
        } else {
            out->pixels = malloc(bytes);
            if (out->pixels) {
                memcpy(out->pixels, mapped, bytes);
                out->width = r->lightmap_width;
                out->height = r->lightmap_height;
            } else {
                good = false;
            }
            SDL_UnmapGPUTransferBuffer(r->device, transfer);
        }
    }

    SDL_ReleaseGPUTransferBuffer(r->device, transfer);
    return good;
}

bool upload_bvh(renderer *r, const bvh *tree) {
    if (!r || !tree || !tree->nodes || !tree->node_count ||
        !tree->triangles || !tree->triangle_count) return false;

    release_buffer(r, r->bvh_node_buffer);
    release_buffer(r, r->bvh_triangle_buffer);
    r->bvh_node_buffer = NULL;
    r->bvh_triangle_buffer = NULL;

    r->bvh_node_buffer = upload_buffer(
        r, SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_READ,
        tree->nodes, (size_t)tree->node_count * sizeof(*tree->nodes));
    r->bvh_triangle_buffer = upload_buffer(
        r, SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_READ,
        tree->triangles, (size_t)tree->triangle_count * sizeof(*tree->triangles));

    return r->bvh_node_buffer && r->bvh_triangle_buffer;
}

BUFFER *upload_probes(renderer *r, const dm_probe_grid *grid) {
    if (!grid || !grid->probes) return NULL;
    const size_t count = (size_t)grid->count_x * grid->count_y * grid->count_z;
    if (!count) return NULL;
    return upload_buffer(r, SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_READ,
                         grid->probes, count * sizeof(dm_probe));
}

BUFFER *upload_beams(renderer *r, const dm_beam_grid *grid) {
    if (!grid) return NULL;

    float *visibility = dm_beam_expand(grid);
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

    BUFFER *buffer = upload_buffer(
        r, SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_READ,
        data, (beam_count + depth_count) * sizeof(float));
    free(data);
    return buffer;
}

static void dispatch_shape(Uint32 items, Uint32 *groups_x, Uint32 *groups_y,
                           Uint32 *dispatch_width) {
    Uint32 gx = (items + 63u) / 64u;
    if (gx == 0u) gx = 1u;
    if (gx > 4096u) gx = 4096u;
    const Uint32 width = gx * 64u;
    const Uint32 gy = (items + width - 1u) / width;

    *groups_x = gx;
    *groups_y = gy ? gy : 1u;
    *dispatch_width = width;
}

static bake_uniforms bake_data(renderer *r, Uint32 phase, Uint32 iteration,
                               Uint32 item_count, Uint32 dispatch_width) {
    const vec3 sun = sun_direction();
    return (bake_uniforms){
        .item_count = item_count,
        .lightmap_width = r->lightmap_width,
        .lightmap_height = r->lightmap_height,
        .dispatch_width = dispatch_width,
        .iteration = iteration,
        .phase = phase,
        .max_bounces = BAKE_MAX_BOUNCES,
        .padding = phase == PHASE_TRACE ? 1u : 0u,
        .sun_direction_intensity = {sun.x, sun.y, sun.z, 2.4f},
        .sun_color_radius = {1.00f, 0.94f, 0.84f, 0.00465f},
        .sky_zenith = {0.22f, 0.42f, 0.78f, 1.0f},
        .sky_horizon = {0.68f, 0.76f, 0.88f, 1.0f},
        .bake_params = {r->bake_epsilon, 0.72f, 1.0f, 0.0f}
    };
}

static bool record_bake_pass(renderer *r, SDL_GPUCommandBuffer *cmd,
                             TEXTURE *source, TEXTURE *destination,
                             Uint32 phase, Uint32 iteration,
                             Uint32 item_count) {
    Uint32 groups_x, groups_y, dispatch_width;
    dispatch_shape(item_count, &groups_x, &groups_y, &dispatch_width);

    const bake_uniforms uniforms = bake_data(
        r, phase, iteration, item_count, dispatch_width);
    const SDL_GPUStorageTextureReadWriteBinding output = {
        .texture = destination,
        .mip_level = 0,
        .layer = 0,
        .cycle = false
    };

    SDL_GPUComputePass *pass = SDL_BeginGPUComputePass(cmd, &output, 1, NULL, 0);
    if (!pass) return false;

    SDL_BindGPUComputePipeline(pass, r->bake_pipeline);
    const SDL_GPUTextureSamplerBinding source_binding = {
        .texture = source,
        .sampler = r->lightmap_sampler
    };
    SDL_BindGPUComputeSamplers(pass, 0, &source_binding, 1);

    BUFFER *buffers[3] = {
        r->bvh_node_buffer,
        r->bvh_triangle_buffer,
        r->lightmap_sample_buffer
    };
    SDL_BindGPUComputeStorageBuffers(pass, 0, buffers, 3);
    SDL_PushGPUComputeUniformData(cmd, 0, &uniforms, sizeof(uniforms));
    SDL_DispatchGPUCompute(pass, groups_x, groups_y, 1);
    SDL_EndGPUComputePass(pass);
    return true;
}

static void swap_lightmaps(renderer *r) {
    TEXTURE *tmp = r->lightmap_texture;
    r->lightmap_texture = r->lightmap_scratch;
    r->lightmap_scratch = tmp;
}

static bool submit_trace_batch(renderer *r, Uint32 first, Uint32 count) {
    SDL_GPUCommandBuffer *cmd = SDL_AcquireGPUCommandBuffer(r->device);
    if (!cmd) return false;

    for (Uint32 i = 0; i < count; ++i) {
        if (!record_bake_pass(r, cmd, r->lightmap_texture,
                              r->lightmap_scratch, PHASE_TRACE,
                              first + i, r->lightmap_sample_count)) {
            SDL_CancelGPUCommandBuffer(cmd);
            return false;
        }
        swap_lightmaps(r);
    }

    return SDL_SubmitGPUCommandBuffer(cmd);
}

static bool bake_lightmap_once(renderer *r) {
    const Uint32 pixels = r->lightmap_width * r->lightmap_height;
    bake_progress(r, "surface lightmap", 0u, r->bake_target_samples);

    SDL_GPUCommandBuffer *clear = SDL_AcquireGPUCommandBuffer(r->device);
    if (!clear) return false;
    if (!record_bake_pass(r, clear, r->lightmap_scratch,
                          r->lightmap_texture, PHASE_CLEAR, 0, pixels) ||
        !record_bake_pass(r, clear, r->lightmap_texture,
                          r->lightmap_scratch, PHASE_CLEAR, 0, pixels)) {
        SDL_CancelGPUCommandBuffer(clear);
        return false;
    }
    if (!SDL_SubmitGPUCommandBuffer(clear)) return false;

    for (Uint32 first = 0; first < r->bake_target_samples;
         first += BAKE_BATCH_SAMPLES) {
        Uint32 count = r->bake_target_samples - first;
        if (count > BAKE_BATCH_SAMPLES) count = BAKE_BATCH_SAMPLES;
        if (!submit_trace_batch(r, first, count)) return false;
        bake_progress(r, "surface lightmap", first + count, r->bake_target_samples);
    }

    bake_progress(r, "filtering lightmap", 0u, 0u);
    SDL_GPUCommandBuffer *post = SDL_AcquireGPUCommandBuffer(r->device);
    if (!post) return false;
    if (!record_bake_pass(r, post, r->lightmap_texture,
                          r->lightmap_scratch, PHASE_FILTER, 0, pixels)) {
        SDL_CancelGPUCommandBuffer(post);
        return false;
    }

    swap_lightmaps(r);
    for (Uint32 i = 0; i < BAKE_DILATION_PASSES; ++i) {
        if (!record_bake_pass(r, post, r->lightmap_texture,
                              r->lightmap_scratch, PHASE_DILATE, 0, pixels)) {
            SDL_CancelGPUCommandBuffer(post);
            return false;
        }
        swap_lightmaps(r);
    }

    if (!SDL_SubmitGPUCommandBuffer(post)) return false;

    release_texture(r, r->lightmap_scratch);
    r->lightmap_scratch = NULL;
    if (r->bake_pipeline) SDL_ReleaseGPUComputePipeline(r->device, r->bake_pipeline);
    r->bake_pipeline = NULL;
    return true;
}

bool bake_lightmap(renderer *r, const bvh *tree, const lightmap *lm) {
    if (!r || !r->device || !tree || !tree->node_count || !lm ||
        !lm->width || !lm->height || !lm->samples || !lm->sample_count ||
        !r->lightmap_sampler) return false;

    r->lightmap_width = lm->width;
    r->lightmap_height = lm->height;
    r->lightmap_sample_count = lm->sample_count;
    r->bake_target_samples = BAKE_TARGET_SAMPLES;

    const bvh_node *root = &tree->nodes[0];
    const float sx = root->max[0] - root->min[0];
    const float sy = root->max[1] - root->min[1];
    const float sz = root->max[2] - root->min[2];
    float scene_scale = fmaxf(sx, fmaxf(sy, sz));
    if (scene_scale < 1.0f) scene_scale = 1.0f;
    r->bake_epsilon = scene_scale * 2.0e-5f;

    r->lightmap_texture = create_lightmap_texture(r, lm->width, lm->height);
    r->lightmap_scratch = create_lightmap_texture(r, lm->width, lm->height);
    if (!r->lightmap_texture || !r->lightmap_scratch) return false;

    if (!upload_bvh(r, tree)) return false;
    r->lightmap_sample_buffer = upload_buffer(
        r, SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_READ,
        lm->samples, (size_t)lm->sample_count * sizeof(*lm->samples));
    if (!r->lightmap_sample_buffer) return false;

    r->bake_pipeline = compile_compute(
        r->device, "shaders/compute.hlsl", "lightmap_cs", "BUILD_LIGHTMAP_CS");
    if (!r->bake_pipeline) return false;

    SDL_Log("lightmap: %ux%u, %u charts, %u valid texels, %.2f texels/unit",
            lm->width, lm->height, lm->chart_count,
            lm->sample_count, lm->texel_density);
    return bake_lightmap_once(r);
}

bool bake_probe_grid(renderer *r, dm_probe_grid *grid, Uint32 samples) {
    if (!r || !grid || !grid->probes) return false;

    const uint64_t count = (uint64_t)grid->count_x *
                           grid->count_y * grid->count_z;
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

    BUFFER *input = upload_buffer(
        r, SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_READ, positions, input_bytes);
    free(positions);

    BUFFER *output = SDL_CreateGPUBuffer(r->device,
        &(SDL_GPUBufferCreateInfo){
            .usage = SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_WRITE,
            .size = output_bytes
        });
    SDL_GPUTransferBuffer *transfer = SDL_CreateGPUTransferBuffer(r->device,
        &(SDL_GPUTransferBufferCreateInfo){
            .usage = SDL_GPU_TRANSFERBUFFERUSAGE_DOWNLOAD,
            .size = output_bytes
        });
    COMPUTE_PIPELINE *pipeline = compile_compute(
        r->device, "shaders/compute.hlsl", "probe_cs", "BUILD_PROBE_CS");

    bool good = input && output && transfer && pipeline;
    if (good) {
        SDL_GPUCommandBuffer *cmd = SDL_AcquireGPUCommandBuffer(r->device);
        if (!cmd) {
            good = false;
        } else {
            const SDL_GPUStorageBufferReadWriteBinding binding = {.buffer = output};
            SDL_GPUComputePass *pass = SDL_BeginGPUComputePass(
                cmd, NULL, 0, &binding, 1);
            if (!pass) {
                SDL_CancelGPUCommandBuffer(cmd);
                good = false;
            } else {
                SDL_BindGPUComputePipeline(pass, pipeline);
                BUFFER *buffers[3] = {
                    input, r->bvh_node_buffer, r->bvh_triangle_buffer
                };
                SDL_BindGPUComputeStorageBuffers(pass, 0, buffers, 3);
                bake_uniforms u = bake_data(r, 0u, 0u, samples, 0u);
                SDL_PushGPUComputeUniformData(cmd, 0, &u, sizeof(u));
                SDL_DispatchGPUCompute(pass, (Uint32)count, 1u, 1u);
                SDL_EndGPUComputePass(pass);

                SDL_GPUCopyPass *copy = SDL_BeginGPUCopyPass(cmd);
                if (!copy) {
                    SDL_CancelGPUCommandBuffer(cmd);
                    good = false;
                } else {
                    SDL_DownloadFromGPUBuffer(copy,
                        &(SDL_GPUBufferRegion){
                            .buffer = output,
                            .size = output_bytes
                        },
                        &(SDL_GPUTransferBufferLocation){
                            .transfer_buffer = transfer
                        });
                    SDL_EndGPUCopyPass(copy);

                    SDL_GPUFence *fence =
                        SDL_SubmitGPUCommandBufferAndAcquireFence(cmd);
                    if (!fence) {
                        good = false;
                    } else {
                        good = SDL_WaitForGPUFences(
                            r->device, true, &fence, 1u);
                        SDL_ReleaseGPUFence(r->device, fence);
                    }
                }
            }
        }
    }

    if (good) {
        const float (*values)[4] = SDL_MapGPUTransferBuffer(
            r->device, transfer, false);
        if (!values) {
            good = false;
        } else {
            for (uint32_t i = 0; i < (uint32_t)count; ++i) {
                for (uint32_t j = 0; j < 9u; ++j) {
                    memcpy(grid->probes[i].coefficients[j],
                           values[i * 9u + j], sizeof(float[4]));
                }
                grid->probes[i].position[3] = values[i * 9u][3];
            }
            SDL_UnmapGPUTransferBuffer(r->device, transfer);
        }
    }

    if (pipeline) SDL_ReleaseGPUComputePipeline(r->device, pipeline);
    if (transfer) SDL_ReleaseGPUTransferBuffer(r->device, transfer);
    release_buffer(r, output);
    release_buffer(r, input);
    return good;
}

void release_bake_resources(renderer *r) {
    if (!r || !r->device) return;

    release_buffer(r, r->bvh_node_buffer);
    release_buffer(r, r->bvh_triangle_buffer);
    release_buffer(r, r->lightmap_sample_buffer);
    release_texture(r, r->lightmap_scratch);
    if (r->bake_pipeline) SDL_ReleaseGPUComputePipeline(r->device, r->bake_pipeline);

    r->bvh_node_buffer = NULL;
    r->bvh_triangle_buffer = NULL;
    r->lightmap_sample_buffer = NULL;
    r->lightmap_scratch = NULL;
    r->bake_pipeline = NULL;
}

static bool dispatch_one(fx_state *fx, SDL_GPUCommandBuffer *cmd,
                         COMPUTE_PIPELINE *pipeline, TEXTURE *source,
                         TEXTURE *destination, const void *uniforms,
                         Uint32 uniform_size, Uint32 width, Uint32 height) {
    const SDL_GPUStorageTextureReadWriteBinding output = {
        .texture = destination,
        .mip_level = 0,
        .layer = 0,
        .cycle = false
    };
    SDL_GPUComputePass *pass = SDL_BeginGPUComputePass(cmd, &output, 1, NULL, 0);
    if (!pass) return false;

    SDL_BindGPUComputePipeline(pass, pipeline);
    if (source) {
        const SDL_GPUTextureSamplerBinding input = {
            .texture = source,
            .sampler = fx->sampler
        };
        SDL_BindGPUComputeSamplers(pass, 0, &input, 1);
    }
    if (uniforms && uniform_size)
        SDL_PushGPUComputeUniformData(cmd, 0, uniforms, uniform_size);

    SDL_DispatchGPUCompute(pass, (width + 7u) / 8u,
                          (height + 7u) / 8u, 1);
    SDL_EndGPUComputePass(pass);
    return true;
}

static void release_frame_textures(fx_state *fx) {
    if (!fx || !fx->device) return;

    if (fx->hdr) SDL_ReleaseGPUTexture(fx->device, fx->hdr);
    if (fx->normal_depth) SDL_ReleaseGPUTexture(fx->device, fx->normal_depth);
    if (fx->ao) SDL_ReleaseGPUTexture(fx->device, fx->ao);
    if (fx->bloom_a) SDL_ReleaseGPUTexture(fx->device, fx->bloom_a);
    if (fx->bloom_b) SDL_ReleaseGPUTexture(fx->device, fx->bloom_b);
    if (fx->volume) SDL_ReleaseGPUTexture(fx->device, fx->volume);
    if (fx->lit) SDL_ReleaseGPUTexture(fx->device, fx->lit);

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

static void fx_deinit(fx_state *fx) {
    if (!fx) return;
    if (fx->device) {
        release_frame_textures(fx);
        if (fx->lut) SDL_ReleaseGPUTexture(fx->device, fx->lut);
        if (fx->sampler) SDL_ReleaseGPUSampler(fx->device, fx->sampler);
        if (fx->depth_sampler) SDL_ReleaseGPUSampler(fx->device, fx->depth_sampler);
        if (fx->compose_pipeline)
            SDL_ReleaseGPUGraphicsPipeline(fx->device, fx->compose_pipeline);
        if (fx->ssao_pipeline)
            SDL_ReleaseGPUComputePipeline(fx->device, fx->ssao_pipeline);
        if (fx->bloom_pipeline)
            SDL_ReleaseGPUComputePipeline(fx->device, fx->bloom_pipeline);
        if (fx->grade_pipeline)
            SDL_ReleaseGPUComputePipeline(fx->device, fx->grade_pipeline);
        if (fx->volume_pipeline)
            SDL_ReleaseGPUComputePipeline(fx->device, fx->volume_pipeline);
        if (fx->volume_compose_pipeline)
            SDL_ReleaseGPUComputePipeline(fx->device, fx->volume_compose_pipeline);
    }
    memset(fx, 0, sizeof(*fx));
}

static bool fx_init(fx_state *fx, DEVICE *device, WINDOW *window) {
    if (!fx || !device || !window) return false;
    memset(fx, 0, sizeof(*fx));
    fx->device = device;

    fx->sampler = SDL_CreateGPUSampler(device, &(SDL_GPUSamplerCreateInfo){
        .min_filter = SDL_GPU_FILTER_LINEAR,
        .mag_filter = SDL_GPU_FILTER_LINEAR,
        .mipmap_mode = SDL_GPU_SAMPLERMIPMAPMODE_NEAREST,
        .address_mode_u = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE,
        .address_mode_v = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE,
        .address_mode_w = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE
    });
    fx->depth_sampler = SDL_CreateGPUSampler(device, &(SDL_GPUSamplerCreateInfo){
        .min_filter = SDL_GPU_FILTER_NEAREST,
        .mag_filter = SDL_GPU_FILTER_NEAREST,
        .mipmap_mode = SDL_GPU_SAMPLERMIPMAPMODE_NEAREST,
        .address_mode_u = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE,
        .address_mode_v = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE
    });
    if (!fx->sampler || !fx->depth_sampler) goto fail;

    SDL_GPUShader *vs = compile_shader(
        device, "shaders/vertex.hlsl", "fullscreen_vs",
        "BUILD_FULLSCREEN_VS", SDL_SHADERCROSS_SHADERSTAGE_VERTEX);
    SDL_GPUShader *ps = compile_shader(
        device, "shaders/fragment.hlsl", "compose_fs",
        "BUILD_COMPOSE_FS", SDL_SHADERCROSS_SHADERSTAGE_FRAGMENT);

    fx->ssao_pipeline = compile_compute(
        device, "shaders/compute.hlsl", "ssao_cs", "BUILD_SSAO_CS");
    fx->bloom_pipeline = compile_compute(
        device, "shaders/compute.hlsl", "bloom_cs", "BUILD_BLOOM_CS");
    fx->grade_pipeline = compile_compute(
        device, "shaders/compute.hlsl", "grade_cs", "BUILD_GRADE_CS");
    fx->volume_pipeline = compile_compute(
        device, "shaders/vision_compute.hlsl", "volume_cs",
        "BUILD_VISION_VOLUME_CS");
    fx->volume_compose_pipeline = compile_compute(
        device, "shaders/vision_compute.hlsl", "volume_compose_cs",
        "BUILD_VISION_COMPOSE_CS");

    if (!vs || !ps || !fx->ssao_pipeline || !fx->bloom_pipeline ||
        !fx->grade_pipeline || !fx->volume_pipeline ||
        !fx->volume_compose_pipeline) {
        if (vs) SDL_ReleaseGPUShader(device, vs);
        if (ps) SDL_ReleaseGPUShader(device, ps);
        goto fail;
    }

    const SDL_GPUColorTargetDescription target = {
        .format = SDL_GetGPUSwapchainTextureFormat(device, window)
    };
    fx->compose_pipeline = SDL_CreateGPUGraphicsPipeline(device,
        &(SDL_GPUGraphicsPipelineCreateInfo){
            .vertex_shader = vs,
            .fragment_shader = ps,
            .primitive_type = SDL_GPU_PRIMITIVETYPE_TRIANGLELIST,
            .rasterizer_state = {
                .fill_mode = SDL_GPU_FILLMODE_FILL,
                .cull_mode = SDL_GPU_CULLMODE_NONE,
                .front_face = SDL_GPU_FRONTFACE_COUNTER_CLOCKWISE,
                .enable_depth_clip = true
            },
            .target_info = {
                .color_target_descriptions = &target,
                .num_color_targets = 1
            }
        });
    SDL_ReleaseGPUShader(device, vs);
    SDL_ReleaseGPUShader(device, ps);
    if (!fx->compose_pipeline) goto fail;

    fx->lut = texture(device, SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT,
                      SDL_GPU_TEXTUREUSAGE_SAMPLER |
                          SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_WRITE,
                      256u, 16u);
    if (!fx->lut) goto fail;

    SDL_GPUCommandBuffer *cmd = SDL_AcquireGPUCommandBuffer(device);
    if (!cmd || !dispatch_one(fx, cmd, fx->grade_pipeline,
                              NULL, fx->lut, NULL, 0, 256u, 16u)) {
        if (cmd) SDL_CancelGPUCommandBuffer(cmd);
        goto fail;
    }
    if (!SDL_SubmitGPUCommandBuffer(cmd)) goto fail;
    return true;

fail:
    fx_deinit(fx);
    return false;
}

static bool fx_ensure(fx_state *fx, Uint32 width, Uint32 height) {
    if (!fx || !width || !height) return false;
    if (fx->hdr && fx->width == width && fx->height == height) return true;

    release_frame_textures(fx);
    fx->width = width;
    fx->height = height;
    fx->ao_width = (width + 1u) / 2u;
    fx->ao_height = (height + 1u) / 2u;

    const SDL_GPUTextureUsageFlags rt =
        SDL_GPU_TEXTUREUSAGE_COLOR_TARGET | SDL_GPU_TEXTUREUSAGE_SAMPLER;
    const SDL_GPUTextureUsageFlags compute =
        SDL_GPU_TEXTUREUSAGE_SAMPLER | SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_WRITE;

    fx->hdr = texture(fx->device, SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT,
                      rt, width, height);
    fx->normal_depth = texture(
        fx->device, SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT,
        rt, width, height);
    fx->ao = texture(fx->device, SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT,
                     compute, fx->ao_width, fx->ao_height);
    fx->bloom_a = texture(
        fx->device, SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT,
        compute, fx->ao_width, fx->ao_height);
    fx->bloom_b = texture(
        fx->device, SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT,
        compute, fx->ao_width, fx->ao_height);
    fx->volume = texture(
        fx->device, SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT,
        compute, fx->ao_width, fx->ao_height);
    fx->lit = texture(fx->device, SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT,
                      compute, width, height);

    if (!fx->hdr || !fx->normal_depth || !fx->ao || !fx->bloom_a ||
        !fx->bloom_b || !fx->volume || !fx->lit) {
        release_frame_textures(fx);
        return false;
    }
    return true;
}

static bool fx_volume(fx_state *fx, SDL_GPUCommandBuffer *cmd,
                      BUFFER *probes, BUFFER *beams,
                      const dm_probe_grid *grid,
                      const dm_beam_grid *beam_grid,
                      const render_frame *frame) {
    if (!fx || !cmd || !probes || !beams || !grid || !beam_grid ||
        !grid->probes || !fx->volume) return false;

    const SDL_GPUStorageTextureReadWriteBinding target = {.texture = fx->volume};
    SDL_GPUComputePass *pass = SDL_BeginGPUComputePass(cmd, &target, 1, NULL, 0);
    if (!pass) return false;

    SDL_BindGPUComputePipeline(pass, fx->volume_pipeline);
    const SDL_GPUTextureSamplerBinding source = {
        .texture = fx->normal_depth,
        .sampler = fx->depth_sampler
    };
    SDL_BindGPUComputeSamplers(pass, 0, &source, 1);
    BUFFER *storage[2] = {probes, beams};
    SDL_BindGPUComputeStorageBuffers(pass, 0, storage, 2);

    const volume_uniforms u = {
        .eye_density = {frame->eye.x, frame->eye.y, frame->eye.z, 0.045f},
        .right_tan = {
            frame->right.x * frame->tan_half_fov * frame->aspect,
            frame->right.y * frame->tan_half_fov * frame->aspect,
            frame->right.z * frame->tan_half_fov * frame->aspect, 0
        },
        .up_tan = {
            frame->up.x * frame->tan_half_fov,
            frame->up.y * frame->tan_half_fov,
            frame->up.z * frame->tan_half_fov, 0
        },
        .forward_g = {
            frame->forward.x, frame->forward.y, frame->forward.z, 0.55f
        },
        .sun_intensity = {
            frame->sun.x, frame->sun.y, frame->sun.z, 2.4f
        },
        .grid_origin_spacing = {
            grid->origin.x, grid->origin.y, grid->origin.z, grid->spacing
        },
        .grid_dims_width = {
            grid->count_x, grid->count_y, grid->count_z, fx->ao_width
        },
        .height_debug = {
            fx->ao_height, fx->debug_view, beam_grid->depth, 0
        },
        .beam_origin = {
            beam_grid->origin.x, beam_grid->origin.y, beam_grid->origin.z, 0
        },
        .beam_step = {
            beam_grid->step.x, beam_grid->step.y, beam_grid->step.z, 0
        }
    };
    SDL_PushGPUComputeUniformData(cmd, 0, &u, sizeof(u));
    SDL_DispatchGPUCompute(pass, (fx->ao_width + 7u) / 8u,
                          (fx->ao_height + 7u) / 8u, 1);
    SDL_EndGPUComputePass(pass);

    const SDL_GPUStorageTextureReadWriteBinding lit = {.texture = fx->lit};
    pass = SDL_BeginGPUComputePass(cmd, &lit, 1, NULL, 0);
    if (!pass) return false;
    SDL_BindGPUComputePipeline(pass, fx->volume_compose_pipeline);

    const SDL_GPUTextureSamplerBinding inputs[3] = {
        {.texture = fx->hdr, .sampler = fx->sampler},
        {.texture = fx->volume, .sampler = fx->sampler},
        {.texture = fx->normal_depth, .sampler = fx->depth_sampler}
    };
    SDL_BindGPUComputeSamplers(pass, 0, inputs, 3);
    const Uint32 dimensions[4] = {fx->width, fx->height, fx->debug_view, 0};
    SDL_PushGPUComputeUniformData(cmd, 0, dimensions, sizeof(dimensions));
    SDL_DispatchGPUCompute(pass, (fx->width + 7u) / 8u,
                          (fx->height + 7u) / 8u, 1);
    SDL_EndGPUComputePass(pass);
    fx->volume_ready = true;
    return true;
}

static bool run_vision_only(fx_state *fx, SDL_GPUCommandBuffer *cmd) {
    const SDL_GPUStorageTextureReadWriteBinding lit = {.texture = fx->lit};
    SDL_GPUComputePass *pass = SDL_BeginGPUComputePass(cmd, &lit, 1, NULL, 0);
    if (!pass) return false;

    SDL_BindGPUComputePipeline(pass, fx->volume_compose_pipeline);
    const SDL_GPUTextureSamplerBinding inputs[3] = {
        {.texture = fx->hdr, .sampler = fx->sampler},
        {.texture = fx->volume, .sampler = fx->sampler},
        {.texture = fx->normal_depth, .sampler = fx->depth_sampler}
    };
    SDL_BindGPUComputeSamplers(pass, 0, inputs, 3);

    const Uint32 dimensions[4] = {fx->width, fx->height, fx->debug_view, 1u};
    SDL_PushGPUComputeUniformData(cmd, 0, dimensions, sizeof(dimensions));
    SDL_DispatchGPUCompute(pass, (fx->width + 7u) / 8u,
                          (fx->height + 7u) / 8u, 1);
    SDL_EndGPUComputePass(pass);
    return true;
}

static bool bloom_pass(fx_state *fx, SDL_GPUCommandBuffer *cmd,
                       TEXTURE *source, TEXTURE *destination,
                       Uint32 src_width, Uint32 src_height, Uint32 phase) {
    const bloom_uniforms u = {
        .src_width = src_width,
        .src_height = src_height,
        .dst_width = fx->ao_width,
        .dst_height = fx->ao_height,
        .phase = phase,
        .threshold = 1.0f,
        .knee = 0.55f,
        .strength = 1.0f * 1e2
    };
    return dispatch_one(fx, cmd, fx->bloom_pipeline, source, destination,
                        &u, sizeof(u), fx->ao_width, fx->ao_height);
}

static bool fx_apply_base(fx_state *fx, SDL_GPUCommandBuffer *cmd,
                          TEXTURE *swap, float tan_half_fov, float aspect) {
    if (!fx || !cmd || !swap || !fx->hdr || !fx->normal_depth) return false;

    const ssao_uniforms ao = {
        .width = fx->width,
        .height = fx->height,
        .ao_width = fx->ao_width,
        .ao_height = fx->ao_height,
        .tan_half_fov = tan_half_fov,
        .aspect = aspect,
        .radius = 0.65f,
        .bias = 0.035f
    };
    TEXTURE *hdr = fx->volume_ready ? fx->lit : fx->hdr;
    if (!dispatch_one(fx, cmd, fx->ssao_pipeline, fx->normal_depth, fx->ao,
                      &ao, sizeof(ao), fx->ao_width, fx->ao_height) ||
        !bloom_pass(fx, cmd, fx->hdr, fx->bloom_a,
                    fx->width, fx->height, 0u) ||
        !bloom_pass(fx, cmd, fx->bloom_a, fx->bloom_b,
                    fx->ao_width, fx->ao_height, 1u) ||
        !bloom_pass(fx, cmd, fx->bloom_b, fx->bloom_a,
                    fx->ao_width, fx->ao_height, 2u)) {
        return false;
    }

    const SDL_GPUColorTargetInfo color = {
        .texture = swap,
        .load_op = SDL_GPU_LOADOP_DONT_CARE,
        .store_op = SDL_GPU_STOREOP_STORE
    };
    SDL_GPURenderPass *pass = SDL_BeginGPURenderPass(cmd, &color, 1, NULL);
    if (!pass) return false;

    SDL_BindGPUGraphicsPipeline(pass, fx->compose_pipeline);
    const SDL_GPUTextureSamplerBinding bindings[4] = {
        {.texture = hdr, .sampler = fx->sampler},
        {.texture = fx->ao, .sampler = fx->sampler},
        {.texture = fx->bloom_a, .sampler = fx->sampler},
        {.texture = fx->lut, .sampler = fx->sampler}
    };
    SDL_BindGPUFragmentSamplers(pass, 0, bindings, 4);

    const compose_uniforms u = {
        .exposure = 1.0f,
        .ao_strength = fx->debug_view >= 3u ? 0.0f : 0.62f,
        .bloom_strength = fx->debug_view >= 3u ? 0.0f : 0.22f
    };
    SDL_PushGPUFragmentUniformData(cmd, 0, &u, sizeof(u));
    SDL_DrawGPUPrimitives(pass, 3, 1, 0, 0);
    SDL_EndGPURenderPass(pass);
    return true;
}

static bool fx_apply(fx_state *fx, SDL_GPUCommandBuffer *cmd,
                     TEXTURE *swap, float tan_half_fov, float aspect) {
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

bool r_init(renderer *r, const char *title, int width, int height) {
    if (!r) return false;

    memset(r, 0, sizeof(*r));
    r->show_volume = true;
    r->yaw = -0.78f;
    r->pitch = 0.34f;
    r->distance = 14.0f;
    r->target = v3(0.0f, 1.0f, 0.0f);

    if (!SDL_ShaderCross_Init()) return false;

    r->window = SDL_CreateWindow(title, width, height,
        SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY);
    if (!r->window) {
        r_deinit(r);
        return false;
    }

    const SDL_GPUShaderFormat formats = SDL_ShaderCross_GetSPIRVShaderFormats();
    r->device = SDL_CreateGPUDevice(formats, true, NULL);
    if (!r->device || !SDL_ClaimWindowForGPUDevice(r->device, r->window)) {
        SDL_Log("GPU initialization failed: %s", SDL_GetError());
        r_deinit(r);
        return false;
    }

    if (SDL_GPUTextureSupportsFormat(r->device, SDL_GPU_TEXTUREFORMAT_D32_FLOAT,
                                     SDL_GPU_TEXTURETYPE_2D,
                                     SDL_GPU_TEXTUREUSAGE_DEPTH_STENCIL_TARGET)) {
        r->depth_format = SDL_GPU_TEXTUREFORMAT_D32_FLOAT;
    } else if (SDL_GPUTextureSupportsFormat(
                   r->device, SDL_GPU_TEXTUREFORMAT_D24_UNORM,
                   SDL_GPU_TEXTURETYPE_2D,
                   SDL_GPU_TEXTUREUSAGE_DEPTH_STENCIL_TARGET)) {
        r->depth_format = SDL_GPU_TEXTUREFORMAT_D24_UNORM;
    } else {
        r->depth_format = SDL_GPU_TEXTUREFORMAT_D16_UNORM;
    }

    SDL_GPUShader *surface_vs = compile_shader(
        r->device, "shaders/vertex.hlsl", "surface_vs",
        "BUILD_SURFACE_VS", SDL_SHADERCROSS_SHADERSTAGE_VERTEX);
    SDL_GPUShader *surface_ps = compile_shader(
        r->device, "shaders/fragment.hlsl", "surface_fs",
        "BUILD_SURFACE_FS", SDL_SHADERCROSS_SHADERSTAGE_FRAGMENT);
    SDL_GPUShader *line_vs = compile_shader(
        r->device, "shaders/vertex.hlsl", "wireframe_vs",
        "BUILD_WIREFRAME_VS", SDL_SHADERCROSS_SHADERSTAGE_VERTEX);
    SDL_GPUShader *line_ps = compile_shader(
        r->device, "shaders/fragment.hlsl", "wireframe_fs",
        "BUILD_WIREFRAME_FS", SDL_SHADERCROSS_SHADERSTAGE_FRAGMENT);
    SDL_GPUShader *sky_vs = compile_shader(
        r->device, "shaders/vertex.hlsl", "fullscreen_vs",
        "BUILD_FULLSCREEN_VS", SDL_SHADERCROSS_SHADERSTAGE_VERTEX);
    SDL_GPUShader *sky_ps = compile_shader(
        r->device, "shaders/fragment.hlsl", "sky_fs",
        "BUILD_SKY_FS", SDL_SHADERCROSS_SHADERSTAGE_FRAGMENT);

    if (!surface_vs || !surface_ps || !line_vs || !line_ps ||
        !sky_vs || !sky_ps) {
        if (surface_vs) SDL_ReleaseGPUShader(r->device, surface_vs);
        if (surface_ps) SDL_ReleaseGPUShader(r->device, surface_ps);
        if (line_vs) SDL_ReleaseGPUShader(r->device, line_vs);
        if (line_ps) SDL_ReleaseGPUShader(r->device, line_ps);
        if (sky_vs) SDL_ReleaseGPUShader(r->device, sky_vs);
        if (sky_ps) SDL_ReleaseGPUShader(r->device, sky_ps);
        r_deinit(r);
        return false;
    }

    r->solid_pipeline = make_surface_pipeline(r, surface_vs, surface_ps);
    r->line_pipeline = make_line_pipeline(r, line_vs, line_ps);
    r->sky_pipeline = make_sky_pipeline(r, sky_vs, sky_ps);

    SDL_ReleaseGPUShader(r->device, surface_vs);
    SDL_ReleaseGPUShader(r->device, surface_ps);
    SDL_ReleaseGPUShader(r->device, line_vs);
    SDL_ReleaseGPUShader(r->device, line_ps);
    SDL_ReleaseGPUShader(r->device, sky_vs);
    SDL_ReleaseGPUShader(r->device, sky_ps);

    if (!r->solid_pipeline || !r->line_pipeline || !r->sky_pipeline ||
        !fx_init(&r->fx, r->device, r->window)) {
        r_deinit(r);
        return false;
    }

    SDL_Log("GPU backend: %s", SDL_GetGPUDeviceDriver(r->device));
    SDL_Log("depth format: %s",
            r->depth_format == SDL_GPU_TEXTUREFORMAT_D32_FLOAT ? "D32_FLOAT" :
            r->depth_format == SDL_GPU_TEXTUREFORMAT_D24_UNORM ? "D24_UNORM" :
            "D16_UNORM");
    SDL_Log("SDL_image: %d", IMG_Version());
    return true;
}

bool draw_frame(renderer *r, const render_frame *frame) {
    if (!r || !frame || !r->device || !r->solid_pipeline ||
        !r->sky_pipeline || !r->vertex_buffer || !r->lightmap_texture ||
        !r->lightmap_sampler) return false;

    SDL_GPUCommandBuffer *cmd = SDL_AcquireGPUCommandBuffer(r->device);
    if (!cmd) return false;

    TEXTURE *swap = NULL;
    Uint32 width = 0;
    Uint32 height = 0;
    if (!SDL_WaitAndAcquireGPUSwapchainTexture(
            cmd, r->window, &swap, &width, &height)) {
        SDL_CancelGPUCommandBuffer(cmd);
        return false;
    }

    if (!swap || !width || !height)
        return SDL_SubmitGPUCommandBuffer(cmd);

    if (!fx_ensure(&r->fx, width, height) ||
        !ensure_depth_texture(r, width, height)) {
        SDL_CancelGPUCommandBuffer(cmd);
        return false;
    }

    camera_uniforms camera = {0};
    memcpy(camera.mvp, frame->mvp, sizeof(camera.mvp));
    memcpy(camera.view, frame->view, sizeof(camera.view));

    const sky_uniforms sky = {
        .camera_right = {
            frame->right.x * frame->tan_half_fov * frame->aspect,
            frame->right.y * frame->tan_half_fov * frame->aspect,
            frame->right.z * frame->tan_half_fov * frame->aspect, 0
        },
        .camera_up = {
            frame->up.x * frame->tan_half_fov,
            frame->up.y * frame->tan_half_fov,
            frame->up.z * frame->tan_half_fov, 0
        },
        .camera_forward = {
            frame->forward.x, frame->forward.y, frame->forward.z, 0
        },
        .sky_zenith = {0.22f, 0.42f, 0.78f, 1},
        .sky_horizon = {0.68f, 0.76f, 0.88f, 1},
        .sun_direction_intensity = {
            frame->sun.x, frame->sun.y, frame->sun.z, 2.4f
        },
        .sun_color_radius = {1.00f, 0.94f, 0.84f, 0.00465f}
    };

    const SDL_GPUColorTargetInfo colors[2] = {
        {.texture = r->fx.hdr,
         .load_op = SDL_GPU_LOADOP_DONT_CARE,
         .store_op = SDL_GPU_STOREOP_STORE},
        {.texture = r->fx.normal_depth,
         .load_op = SDL_GPU_LOADOP_DONT_CARE,
         .store_op = SDL_GPU_STOREOP_STORE}
    };
    const SDL_GPUDepthStencilTargetInfo depth = {
        .texture = r->depth_texture,
        .clear_depth = 1.0f,
        .load_op = SDL_GPU_LOADOP_CLEAR,
        .store_op = SDL_GPU_STOREOP_DONT_CARE,
        .stencil_load_op = SDL_GPU_LOADOP_DONT_CARE,
        .stencil_store_op = SDL_GPU_STOREOP_DONT_CARE
    };

    SDL_GPURenderPass *pass = SDL_BeginGPURenderPass(cmd, colors, 2, &depth);
    if (!pass) {
        SDL_CancelGPUCommandBuffer(cmd);
        return false;
    }

    SDL_BindGPUGraphicsPipeline(pass, r->sky_pipeline);
    SDL_PushGPUFragmentUniformData(cmd, 0, &sky, sizeof(sky));
    SDL_DrawGPUPrimitives(pass, 3, 1, 0, 0);

    SDL_BindGPUVertexBuffers(pass, 0,
        &(SDL_GPUBufferBinding){.buffer = r->vertex_buffer, .offset = 0}, 1);
    SDL_BindGPUGraphicsPipeline(pass, r->solid_pipeline);
    SDL_PushGPUVertexUniformData(cmd, 0, &camera, sizeof(camera));

    const SDL_GPUTextureSamplerBinding lightmap_binding = {
        .texture = r->lightmap_texture,
        .sampler = r->lightmap_sampler
    };
    SDL_BindGPUFragmentSamplers(pass, 5, &lightmap_binding, 1);

    for (uint32_t i = 0; i < r->draw_count; ++i) {
        const draw_range *draw = &r->draws[i];
        const render_material *m = &r->materials[draw->material];
        const SDL_GPUTextureSamplerBinding bindings[5] = {
            {.texture = m->base_color, .sampler = r->material_sampler},
            {.texture = m->metallic_roughness, .sampler = r->material_sampler},
            {.texture = m->normal, .sampler = r->material_sampler},
            {.texture = m->occlusion, .sampler = r->material_sampler},
            {.texture = m->emissive, .sampler = r->material_sampler}
        };
        SDL_BindGPUFragmentSamplers(pass, 0, bindings, 5);

        const material_uniforms material = {
            .base_color_factor = {
                m->data.base_color[0], m->data.base_color[1],
                m->data.base_color[2], m->data.base_color[3]
            },
            .emissive_metallic = {
                m->data.emissive[0], m->data.emissive[1],
                m->data.emissive[2], m->data.metallic
            },
            .roughness_normal_ao_sun = {
                m->data.roughness, m->data.normal_scale,
                m->data.occlusion_strength, 2.4f
            },
            .sun_direction = {
                frame->sun.x, frame->sun.y, frame->sun.z, 0
            },
            .sun_color = {1.00f, 0.94f, 0.84f, 1},
            .camera_position = {
                frame->eye.x, frame->eye.y, frame->eye.z,
                r->debug_view == 1u ? 2.0f : (r->has_bake ? 1.0f : 0.0f)
            }
        };

        SDL_PushGPUFragmentUniformData(cmd, 0, &material, sizeof(material));
        SDL_DrawGPUPrimitives(pass, draw->count, 1, draw->first, 0);
    }

    if (r->show_debug && r->debug_vertex_count) {
        SDL_BindGPUGraphicsPipeline(pass, r->line_pipeline);
        SDL_PushGPUVertexUniformData(cmd, 0, camera.mvp, sizeof(camera.mvp));
        SDL_DrawGPUPrimitives(pass, r->debug_vertex_count, 1,
                              r->debug_vertex_start, 0);
    }
    SDL_EndGPURenderPass(pass);

    r->fx.volume_ready = false;
    r->fx.debug_view = r->debug_view;

    static uint32_t last_logged_view = UINT32_MAX;
    if (last_logged_view != r->fx.debug_view) {
        SDL_Log("GPU debug view: %u | fog: %d | bake: %d",
                r->fx.debug_view, r->show_volume, r->has_bake);
        last_logged_view = r->fx.debug_view;
    }

    if (r->show_volume && r->has_bake &&
        (r->debug_view == 0u || r->debug_view >= 3u) &&
        r->volume_probe_buffer && r->beam_buffer &&
        !fx_volume(&r->fx, cmd, r->volume_probe_buffer, r->beam_buffer,
                   &r->volume_probes, &r->beams, frame)) {
        SDL_CancelGPUCommandBuffer(cmd);
        return false;
    }

    if (!fx_apply(&r->fx, cmd, swap, frame->tan_half_fov, frame->aspect)) {
        SDL_CancelGPUCommandBuffer(cmd);
        return false;
    }

    return SDL_SubmitGPUCommandBuffer(cmd);
}

bool bake_worker_init(renderer *r) {
    if (!r) return false;
    memset(r, 0, sizeof(*r));

    const SDL_GPUShaderFormat formats = SDL_ShaderCross_GetSPIRVShaderFormats();
    r->device = SDL_CreateGPUDevice(formats, false, NULL);
    if (!r->device) return false;

    r->lightmap_sampler = SDL_CreateGPUSampler(r->device,
        &(SDL_GPUSamplerCreateInfo){
            .min_filter = SDL_GPU_FILTER_LINEAR,
            .mag_filter = SDL_GPU_FILTER_LINEAR,
            .mipmap_mode = SDL_GPU_SAMPLERMIPMAPMODE_NEAREST,
            .address_mode_u = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE,
            .address_mode_v = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE,
            .address_mode_w = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE
        });
    r->lightmap_texture = texture(
        r->device, SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT,
        SDL_GPU_TEXTUREUSAGE_SAMPLER, 1, 1);

    if (!r->lightmap_sampler || !r->lightmap_texture) {
        bake_worker_deinit(r);
        return false;
    }
    return true;
}

void bake_worker_deinit(renderer *r) {
    if (!r) return;

    free_probe_grid(&r->volume_probes);
    dm_beam_free(&r->beams);

    if (r->device) {
        SDL_WaitForGPUIdle(r->device);
        release_bake_resources(r);
        release_buffer(r, r->volume_probe_buffer);
        release_buffer(r, r->beam_buffer);
        release_texture(r, r->lightmap_texture);
        if (r->lightmap_sampler)
            SDL_ReleaseGPUSampler(r->device, r->lightmap_sampler);
        SDL_DestroyGPUDevice(r->device);
    }

    memset(r, 0, sizeof(*r));
}

void r_deinit(renderer *r) {
    if (!r) return;

    free(r->vertices);
    free(r->draws);
    free_probe_grid(&r->volume_probes);
    dm_beam_free(&r->beams);

    if (r->device) {
        SDL_WaitForGPUIdle(r->device);
        fx_deinit(&r->fx);

        if (r->image_textures) {
            for (uint32_t i = 0; i < r->image_texture_count; ++i)
                release_texture(r, r->image_textures[i]);
        }
        free(r->image_textures);
        free(r->materials);

        release_texture(r, r->default_white);
        release_texture(r, r->default_normal);
        if (r->material_sampler)
            SDL_ReleaseGPUSampler(r->device, r->material_sampler);
        release_buffer(r, r->vertex_buffer);
        release_bake_resources(r);
        release_buffer(r, r->volume_probe_buffer);
        release_buffer(r, r->beam_buffer);
        release_texture(r, r->depth_texture);
        release_texture(r, r->lightmap_texture);
        if (r->lightmap_sampler)
            SDL_ReleaseGPUSampler(r->device, r->lightmap_sampler);
        if (r->sky_pipeline)
            SDL_ReleaseGPUGraphicsPipeline(r->device, r->sky_pipeline);
        if (r->solid_pipeline)
            SDL_ReleaseGPUGraphicsPipeline(r->device, r->solid_pipeline);
        if (r->line_pipeline)
            SDL_ReleaseGPUGraphicsPipeline(r->device, r->line_pipeline);
        if (r->window) SDL_ReleaseWindowFromGPUDevice(r->device, r->window);
        SDL_DestroyGPUDevice(r->device);
    }

    if (r->window) SDL_DestroyWindow(r->window);
    memset(r, 0, sizeof(*r));
    SDL_ShaderCross_Quit();
}
#endif
