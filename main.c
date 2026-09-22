#include "sdl.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>

#define ARRAY_COUNT(a) ((uint32_t)(sizeof(a) / sizeof((a)[0])))

typedef struct VEC3 {
    float x, y, z;
} VEC3;

typedef struct VERTEX {
    float position[3];
    float normal[3];
    float uv[2];
    float lightmap_uv[2];
} VERTEX;

typedef struct AABB {
    VEC3 min;
    VEC3 max;
} AABB;

typedef struct MESH {
    const VERTEX *vertices; 
    uint32_t vertex_count;
    const uint32_t *indices;
    uint32_t index_count;
    AABB bounds;

    SDL_GPUBuffer *vertex_buffer;
    SDL_GPUBuffer *index_buffer;
} MESH;

typedef struct DRAW_UNIFORM {
    float mvp[16];
    float model[16];
    float albedo[4];
} DRAW_UNIFORM;

typedef struct BAKE_UNIFORM {
    uint32_t width;
    uint32_t height;
    float exposure;
    float padding;
} BAKE_UNIFORM;

typedef struct CAM {
    VEC3 position;
    float yaw;
    float pitch;
} CAM;

typedef struct APP {
    SDL_Window *window;
    SDL_GPUDevice *device;
    SDL_GPUTextureFormat swapchain_format;
    SDL_GPUTextureFormat depth_format;

    SDL_GPUTexture *depth;
    uint32_t depth_width;
    uint32_t depth_height;

    SDL_GPUShader *vertex_shader;
    SDL_GPUShader *fragment_shader;
    SDL_GPUGraphicsPipeline *graphics_pipeline;

    SDL_GPUComputePipeline *lightmap_pipeline;
    SDL_GPUTexture *lightmap;
    SDL_GPUSampler *lightmap_sampler;
    uint32_t lightmap_size;

    MESH cube;
    MESH floor;

    bool shadercross_ready;
    bool window_claimed;
} APP;

#define SHADER_VERTEX_PATH "s_vertex.hlsl"
#define SHADER_FRAGMENT_PATH "s_fragment.hlsl"
#define SHADER_LIGHTMAP_PATH "s_lightmap.hlsl"

static char *shader_load_file_nul(const char *path, size_t *out_size)
{
    size_t size = 0;
    void *data = SDL_LoadFile(path, &size);
    if (!data) return NULL;

    char *text = (char *)SDL_malloc(size + 1);
    if (!text) {
        SDL_free(data);
        return NULL;
    }
    SDL_memcpy(text, data, size);
    text[size] = '\0';
    SDL_free(data);
    if (out_size) *out_size = size;
    return text;
}

static char *shader_load_source(const char *filename)
{
    char *source = shader_load_file_nul(filename, NULL);
    if (source) return source;

    const char *base = SDL_GetBasePath();
    if (base) {
        char *candidate = NULL;
        if (SDL_asprintf(&candidate, "%s%s", base, filename) >= 0 && candidate) {
            source = shader_load_file_nul(candidate, NULL);
            SDL_free(candidate);
            if (source) return source;
        }
        /* Executable lives in build/debug/, shaders live in the project root. */
        if (SDL_asprintf(&candidate, "%s../../%s", base, filename) >= 0 && candidate) {
            source = shader_load_file_nul(candidate, NULL);
            SDL_free(candidate);
            if (source) return source;
        }
    }

    SDL_LogError(
        SDL_LOG_CATEGORY_APPLICATION,
        "could not open shader file '%s': %s",
        filename,
        SDL_GetError());
    return NULL;
}

#define VERTEX(px, py, pz, nx, ny, nz, u, v, lu, lv) \
    {{px, py, pz}, {nx, ny, nz}, {u, v}, {lu, lv}}

static const VERTEX k_cube_vertices[] = {
    VERTEX( 0.5f, -0.5f, -0.5f,  1.0f,  0.0f,  0.0f, 1.0f, 0.0f, 1.0f, 0.0f),
    VERTEX( 0.5f,  0.5f, -0.5f,  1.0f,  0.0f,  0.0f, 1.0f, 1.0f, 1.0f, 1.0f),
    VERTEX( 0.5f,  0.5f,  0.5f,  1.0f,  0.0f,  0.0f, 0.0f, 1.0f, 0.0f, 1.0f),
    VERTEX( 0.5f, -0.5f,  0.5f,  1.0f,  0.0f,  0.0f, 0.0f, 0.0f, 0.0f, 0.0f),

    VERTEX(-0.5f, -0.5f,  0.5f, -1.0f,  0.0f,  0.0f, 1.0f, 0.0f, 1.0f, 0.0f),
    VERTEX(-0.5f,  0.5f,  0.5f, -1.0f,  0.0f,  0.0f, 1.0f, 1.0f, 1.0f, 1.0f),
    VERTEX(-0.5f,  0.5f, -0.5f, -1.0f,  0.0f,  0.0f, 0.0f, 1.0f, 0.0f, 1.0f),
    VERTEX(-0.5f, -0.5f, -0.5f, -1.0f,  0.0f,  0.0f, 0.0f, 0.0f, 0.0f, 0.0f),

    VERTEX(-0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f, 1.0f, 0.0f, 1.0f, 0.0f),
    VERTEX(-0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f, 1.0f, 1.0f, 1.0f, 1.0f),
    VERTEX( 0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f, 0.0f, 1.0f, 0.0f, 1.0f),
    VERTEX( 0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f, 0.0f, 0.0f, 0.0f, 0.0f),

    VERTEX( 0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f, 1.0f, 0.0f, 1.0f, 0.0f),
    VERTEX( 0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f, 1.0f, 1.0f, 1.0f, 1.0f),
    VERTEX(-0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f, 0.0f, 1.0f, 0.0f, 1.0f),
    VERTEX(-0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f, 0.0f, 0.0f, 0.0f, 0.0f),

    VERTEX( 0.5f, -0.5f,  0.5f,  0.0f,  0.0f,  1.0f, 1.0f, 0.0f, 1.0f, 0.0f),
    VERTEX( 0.5f,  0.5f,  0.5f,  0.0f,  0.0f,  1.0f, 1.0f, 1.0f, 1.0f, 1.0f),
    VERTEX(-0.5f,  0.5f,  0.5f,  0.0f,  0.0f,  1.0f, 0.0f, 1.0f, 0.0f, 1.0f),
    VERTEX(-0.5f, -0.5f,  0.5f,  0.0f,  0.0f,  1.0f, 0.0f, 0.0f, 0.0f, 0.0f),

    VERTEX(-0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f, 1.0f, 0.0f, 1.0f, 0.0f),
    VERTEX(-0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f),
    VERTEX( 0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f, 0.0f, 1.0f, 0.0f, 1.0f),
    VERTEX( 0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f)
};

static const uint32_t k_cube_indices[] = {
     0,  1,  2,  0,  2,  3,
     4,  5,  6,  4,  6,  7,
     8,  9, 10,  8, 10, 11,
    12, 13, 14, 12, 14, 15,
    16, 17, 18, 16, 18, 19,
    20, 21, 22, 20, 22, 23
};

static const VERTEX k_floor_vertices[] = {
    VERTEX(-6.0f, -0.5f, -6.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f),
    VERTEX(-6.0f, -0.5f,  6.0f, 0.0f, 1.0f, 0.0f, 0.0f, 6.0f, 0.0f, 1.0f),
    VERTEX( 6.0f, -0.5f,  6.0f, 0.0f, 1.0f, 0.0f, 6.0f, 6.0f, 1.0f, 1.0f),
    VERTEX( 6.0f, -0.5f, -6.0f, 0.0f, 1.0f, 0.0f, 6.0f, 0.0f, 1.0f, 0.0f)
};

#undef VERTEX

static const uint32_t k_floor_indices[] = {0, 1, 2, 0, 2, 3};

static VEC3 v3(float x, float y, float z)
{
    VEC3 v = {x, y, z};
    return v;
}

static VEC3 v3_add(VEC3 a, VEC3 b)
{
    return v3(a.x + b.x, a.y + b.y, a.z + b.z);
}

static VEC3 v3_sub(VEC3 a, VEC3 b)
{
    return v3(a.x - b.x, a.y - b.y, a.z - b.z);
}

static VEC3 v3_mul(VEC3 v, float s)
{
    return v3(v.x * s, v.y * s, v.z * s);
}

static float v3_dot(VEC3 a, VEC3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

static VEC3 v3_cross(VEC3 a, VEC3 b)
{
    return v3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x);
}

static VEC3 v3_norm(VEC3 v)
{
    float len2 = v3_dot(v, v);
    if (len2 <= 1.0e-20f) return v3(0.0f, 0.0f, 0.0f);
    return v3_mul(v, 1.0f / sqrtf(len2));
}

static void m4_identity(float m[16])
{
    SDL_memset(m, 0, sizeof(float) * 16);
    m[0] = 1.0f;
    m[5] = 1.0f;
    m[10] = 1.0f;
    m[15] = 1.0f;
}

static void m4_mul(float out[16], const float a[16], const float b[16])
{
    float tmp[16];
    for (int c = 0; c < 4; ++c) {
        for (int r = 0; r < 4; ++r) {
            tmp[c * 4 + r] =
                a[r] * b[c * 4] +
                a[4 + r] * b[c * 4 + 1] +
                a[8 + r] * b[c * 4 + 2] +
                a[12 + r] * b[c * 4 + 3];
        }
    }
    SDL_memcpy(out, tmp, sizeof(tmp));
}

static void m4_translation(float out[16], float x, float y, float z)
{
    m4_identity(out);
    out[12] = x;
    out[13] = y;
    out[14] = z;
}

static void m4_perspective(float out[16], float fov_y, float aspect, float near_z, float far_z)
{
    float f = 1.0f / tanf(fov_y * 0.5f);
    SDL_memset(out, 0, sizeof(float) * 16);
    out[0] = f / aspect;
    out[5] = f;
    out[10] = far_z / (near_z - far_z);
    out[11] = -1.0f;
    out[14] = -(far_z * near_z) / (far_z - near_z);
}

static void m4_look_at(float out[16], VEC3 eye, VEC3 target, VEC3 up)
{
    VEC3 forward = v3_norm(v3_sub(target, eye));
    VEC3 side = v3_norm(v3_cross(forward, up));
    VEC3 camera_up = v3_cross(side, forward);

    m4_identity(out);
    out[0] = side.x;
    out[1] = camera_up.x;
    out[2] = -forward.x;
    out[4] = side.y;
    out[5] = camera_up.y;
    out[6] = -forward.y;
    out[8] = side.z;
    out[9] = camera_up.z;
    out[10] = -forward.z;
    out[12] = -v3_dot(side, eye);
    out[13] = -v3_dot(camera_up, eye);
    out[14] = v3_dot(forward, eye);
}

static SDL_GPUShader *shader_compile_graphics(
    SDL_GPUDevice *device,
    const char *source,
    SDL_ShaderCross_ShaderStage stage)
{
    SDL_ShaderCross_HLSL_Info hlsl = {
        .source = source,
        .entrypoint = "main",
        .include_dir = NULL,
        .defines = NULL,
        .shader_stage = stage,
        .props = 0
    };

    size_t spirv_size = 0;
    void *spirv = SDL_ShaderCross_CompileSPIRVFromHLSL(&hlsl, &spirv_size);
    if (!spirv) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "HLSL -> SPIR-V failed: %s", SDL_GetError());
        return NULL;
    }

    SDL_ShaderCross_GraphicsShaderMetadata *metadata =
        SDL_ShaderCross_ReflectGraphicsSPIRV(spirv, spirv_size, 0);
    if (!metadata) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "shader reflection failed: %s", SDL_GetError());
        SDL_free(spirv);
        return NULL;
    }

    SDL_ShaderCross_SPIRV_Info info = {
        .bytecode = spirv,
        .bytecode_size = spirv_size,
        .entrypoint = "main",
        .shader_stage = stage,
        .props = 0
    };

    SDL_GPUShader *shader = SDL_ShaderCross_CompileGraphicsShaderFromSPIRV(
        device, &info, &metadata->resource_info, 0);
    if (!shader) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "GPU shader compilation failed: %s", SDL_GetError());
    }

    SDL_free(metadata);
    SDL_free(spirv);
    return shader;
}

static SDL_GPUComputePipeline *shader_compile_compute(SDL_GPUDevice *device, const char *source)
{
    SDL_ShaderCross_HLSL_Info hlsl = {
        .source = source,
        .entrypoint = "main",
        .include_dir = NULL,
        .defines = NULL,
        .shader_stage = SDL_SHADERCROSS_SHADERSTAGE_COMPUTE,
        .props = 0
    };

    size_t spirv_size = 0;
    void *spirv = SDL_ShaderCross_CompileSPIRVFromHLSL(&hlsl, &spirv_size);
    if (!spirv) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "compute HLSL -> SPIR-V failed: %s", SDL_GetError());
        return NULL;
    }

    SDL_ShaderCross_ComputePipelineMetadata *metadata =
        SDL_ShaderCross_ReflectComputeSPIRV(spirv, spirv_size, 0);
    if (!metadata) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "compute reflection failed: %s", SDL_GetError());
        SDL_free(spirv);
        return NULL;
    }

    SDL_ShaderCross_SPIRV_Info info = {
        .bytecode = spirv,
        .bytecode_size = spirv_size,
        .entrypoint = "main",
        .shader_stage = SDL_SHADERCROSS_SHADERSTAGE_COMPUTE,
        .props = 0
    };

    SDL_GPUComputePipeline *pipeline = SDL_ShaderCross_CompileComputePipelineFromSPIRV(
        device, &info, metadata, 0);
    if (!pipeline) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "compute pipeline compilation failed: %s", SDL_GetError());
    }

    SDL_free(metadata);
    SDL_free(spirv);
    return pipeline;
}

static bool mesh_calculate_bounds(MESH *mesh)
{
    if (!mesh || !mesh->vertices || mesh->vertex_count == 0) return false;

    AABB bounds = {
        .min = {INFINITY, INFINITY, INFINITY},
        .max = {-INFINITY, -INFINITY, -INFINITY}
    };

    for (uint32_t i = 0; i < mesh->vertex_count; ++i) {
        const float *p = mesh->vertices[i].position;
        bounds.min.x = SDL_min(bounds.min.x, p[0]);
        bounds.min.y = SDL_min(bounds.min.y, p[1]);
        bounds.min.z = SDL_min(bounds.min.z, p[2]);
        bounds.max.x = SDL_max(bounds.max.x, p[0]);
        bounds.max.y = SDL_max(bounds.max.y, p[1]);
        bounds.max.z = SDL_max(bounds.max.z, p[2]);
    }

    mesh->bounds = bounds;
    return true;
}

static bool mesh_upload(SDL_GPUDevice *device, MESH *mesh)
{
    if (!mesh_calculate_bounds(mesh)) return false;
    if (!mesh->indices || mesh->index_count == 0 || mesh->index_count % 3u != 0u) return false;

    uint32_t vertex_bytes = mesh->vertex_count * (uint32_t)sizeof(VERTEX);
    uint32_t index_bytes = mesh->index_count * (uint32_t)sizeof(uint32_t);
    uint32_t total_bytes = vertex_bytes + index_bytes;

    SDL_GPUBufferCreateInfo vertex_info = {
        .usage = SDL_GPU_BUFFERUSAGE_VERTEX | SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_READ,
        .size = vertex_bytes,
        .props = 0
    };
    SDL_GPUBufferCreateInfo index_info = {
        .usage = SDL_GPU_BUFFERUSAGE_INDEX | SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_READ,
        .size = index_bytes,
        .props = 0
    };

    mesh->vertex_buffer = SDL_CreateGPUBuffer(device, &vertex_info);
    mesh->index_buffer = SDL_CreateGPUBuffer(device, &index_info);
    if (!mesh->vertex_buffer || !mesh->index_buffer) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "mesh buffer creation failed: %s", SDL_GetError());
        return false;
    }

    SDL_GPUTransferBufferCreateInfo transfer_info = {
        .usage = SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD,
        .size = total_bytes,
        .props = 0
    };
    SDL_GPUTransferBuffer *transfer = SDL_CreateGPUTransferBuffer(device, &transfer_info);
    if (!transfer) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "transfer buffer creation failed: %s", SDL_GetError());
        return false;
    }

    void *mapped = SDL_MapGPUTransferBuffer(device, transfer, false);
    if (!mapped) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "transfer buffer map failed: %s", SDL_GetError());
        SDL_ReleaseGPUTransferBuffer(device, transfer);
        return false;
    }
    SDL_memcpy(mapped, mesh->vertices, vertex_bytes);
    SDL_memcpy((uint8_t *)mapped + vertex_bytes, mesh->indices, index_bytes);
    SDL_UnmapGPUTransferBuffer(device, transfer);

    SDL_GPUCommandBuffer *cmd = SDL_AcquireGPUCommandBuffer(device);
    if (!cmd) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "copy command buffer failed: %s", SDL_GetError());
        SDL_ReleaseGPUTransferBuffer(device, transfer);
        return false;
    }

    SDL_GPUCopyPass *copy = SDL_BeginGPUCopyPass(cmd);
    SDL_GPUTransferBufferLocation vertex_src = {.transfer_buffer = transfer, .offset = 0};
    SDL_GPUBufferRegion vertex_dst = {.buffer = mesh->vertex_buffer, .offset = 0, .size = vertex_bytes};
    SDL_GPUTransferBufferLocation index_src = {.transfer_buffer = transfer, .offset = vertex_bytes};
    SDL_GPUBufferRegion index_dst = {.buffer = mesh->index_buffer, .offset = 0, .size = index_bytes};
    SDL_UploadToGPUBuffer(copy, &vertex_src, &vertex_dst, false);
    SDL_UploadToGPUBuffer(copy, &index_src, &index_dst, false);
    SDL_EndGPUCopyPass(copy);

    if (!SDL_SubmitGPUCommandBuffer(cmd)) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "mesh upload submission failed: %s", SDL_GetError());
        SDL_ReleaseGPUTransferBuffer(device, transfer);
        return false;
    }

    SDL_ReleaseGPUTransferBuffer(device, transfer);
    return true;
}

static void mesh_destroy(SDL_GPUDevice *device, MESH *mesh)
{
    if (!device || !mesh) return;
    if (mesh->index_buffer) SDL_ReleaseGPUBuffer(device, mesh->index_buffer);
    if (mesh->vertex_buffer) SDL_ReleaseGPUBuffer(device, mesh->vertex_buffer);
    mesh->index_buffer = NULL;
    mesh->vertex_buffer = NULL;
}

static bool app_create_depth(APP *app, uint32_t width, uint32_t height)
{
    if (width == 0 || height == 0) return true;
    if (app->depth && app->depth_width == width && app->depth_height == height) return true;

    if (app->depth) {
        SDL_ReleaseGPUTexture(app->device, app->depth);
        app->depth = NULL;
    }

    SDL_GPUTextureCreateInfo info = {
        .type = SDL_GPU_TEXTURETYPE_2D,
        .format = app->depth_format,
        .usage = SDL_GPU_TEXTUREUSAGE_DEPTH_STENCIL_TARGET,
        .width = width,
        .height = height,
        .layer_count_or_depth = 1,
        .num_levels = 1,
        .sample_count = SDL_GPU_SAMPLECOUNT_1,
        .props = 0
    };

    app->depth = SDL_CreateGPUTexture(app->device, &info);
    if (!app->depth) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "depth texture creation failed: %s", SDL_GetError());
        return false;
    }

    app->depth_width = width;
    app->depth_height = height;
    return true;
}

static bool app_create_graphics_pipeline(APP *app)
{
    char *vertex_source = shader_load_source(SHADER_VERTEX_PATH);
    if (!vertex_source) return false;
    char *fragment_source = shader_load_source(SHADER_FRAGMENT_PATH);
    if (!fragment_source) {
        SDL_free(vertex_source);
        return false;
    }

    app->vertex_shader = shader_compile_graphics(
        app->device, vertex_source, SDL_SHADERCROSS_SHADERSTAGE_VERTEX);
    app->fragment_shader = shader_compile_graphics(
        app->device, fragment_source, SDL_SHADERCROSS_SHADERSTAGE_FRAGMENT);
    SDL_free(vertex_source);
    SDL_free(fragment_source);
    if (!app->vertex_shader || !app->fragment_shader) return false;

    SDL_GPUVertexBufferDescription vertex_buffer = {
        .slot = 0,
        .pitch = sizeof(VERTEX),
        .input_rate = SDL_GPU_VERTEXINPUTRATE_VERTEX,
        .instance_step_rate = 0
    };

    SDL_GPUVertexAttribute attributes[] = {
        {.location = 0, .buffer_slot = 0, .format = SDL_GPU_VERTEXELEMENTFORMAT_FLOAT3, .offset = offsetof(VERTEX, position)},
        {.location = 1, .buffer_slot = 0, .format = SDL_GPU_VERTEXELEMENTFORMAT_FLOAT3, .offset = offsetof(VERTEX, normal)},
        {.location = 2, .buffer_slot = 0, .format = SDL_GPU_VERTEXELEMENTFORMAT_FLOAT2, .offset = offsetof(VERTEX, uv)},
        {.location = 3, .buffer_slot = 0, .format = SDL_GPU_VERTEXELEMENTFORMAT_FLOAT2, .offset = offsetof(VERTEX, lightmap_uv)}
    };

    SDL_GPUColorTargetDescription color_target = {.format = app->swapchain_format};

    SDL_GPUGraphicsPipelineCreateInfo info = {
        .vertex_shader = app->vertex_shader,
        .fragment_shader = app->fragment_shader,
        .vertex_input_state = {
            .vertex_buffer_descriptions = &vertex_buffer,
            .num_vertex_buffers = 1,
            .vertex_attributes = attributes,
            .num_vertex_attributes = ARRAY_COUNT(attributes)
        },
        .primitive_type = SDL_GPU_PRIMITIVETYPE_TRIANGLELIST,
        .rasterizer_state = {
            .fill_mode = SDL_GPU_FILLMODE_FILL,
            .cull_mode = SDL_GPU_CULLMODE_BACK,
            .front_face = SDL_GPU_FRONTFACE_COUNTER_CLOCKWISE,
            .depth_bias_constant_factor = 0.0f,
            .depth_bias_clamp = 0.0f,
            .depth_bias_slope_factor = 0.0f,
            .enable_depth_bias = false,
            .enable_depth_clip = true
        },
        .multisample_state = {
            .sample_count = SDL_GPU_SAMPLECOUNT_1,
            .sample_mask = 0,
            .enable_mask = false,
            .enable_alpha_to_coverage = false
        },
        .depth_stencil_state = {
            .compare_op = SDL_GPU_COMPAREOP_LESS,
            .back_stencil_state = {0},
            .front_stencil_state = {0},
            .compare_mask = 0,
            .write_mask = 0,
            .enable_depth_test = true,
            .enable_depth_write = true,
            .enable_stencil_test = false
        },
        .target_info = {
            .color_target_descriptions = &color_target,
            .num_color_targets = 1,
            .depth_stencil_format = app->depth_format,
            .has_depth_stencil_target = true
        },
        .props = 0
    };

    app->graphics_pipeline = SDL_CreateGPUGraphicsPipeline(app->device, &info);
    if (!app->graphics_pipeline) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "graphics pipeline creation failed: %s", SDL_GetError());
        return false;
    }
    return true;
}

static bool app_create_lightmap(APP *app)
{
    app->lightmap_size = 256;
    SDL_GPUTextureUsageFlags usage =
        SDL_GPU_TEXTUREUSAGE_SAMPLER | SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_WRITE;

    SDL_GPUTextureFormat format = SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT;
    if (!SDL_GPUTextureSupportsFormat(app->device, format, SDL_GPU_TEXTURETYPE_2D, usage)) {
        format = SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM;
    }
    if (!SDL_GPUTextureSupportsFormat(app->device, format, SDL_GPU_TEXTURETYPE_2D, usage)) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "no sampleable compute-writable lightmap format available");
        return false;
    }

    SDL_GPUTextureCreateInfo texture_info = {
        .type = SDL_GPU_TEXTURETYPE_2D,
        .format = format,
        .usage = usage,
        .width = app->lightmap_size,
        .height = app->lightmap_size,
        .layer_count_or_depth = 1,
        .num_levels = 1,
        .sample_count = SDL_GPU_SAMPLECOUNT_1,
        .props = 0
    };
    app->lightmap = SDL_CreateGPUTexture(app->device, &texture_info);
    if (!app->lightmap) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "lightmap texture creation failed: %s", SDL_GetError());
        return false;
    }

    SDL_GPUSamplerCreateInfo sampler_info = {
        .min_filter = SDL_GPU_FILTER_LINEAR,
        .mag_filter = SDL_GPU_FILTER_LINEAR,
        .mipmap_mode = SDL_GPU_SAMPLERMIPMAPMODE_LINEAR,
        .address_mode_u = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE,
        .address_mode_v = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE,
        .address_mode_w = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE,
        .mip_lod_bias = 0.0f,
        .max_anisotropy = 1.0f,
        .compare_op = SDL_GPU_COMPAREOP_ALWAYS,
        .min_lod = 0.0f,
        .max_lod = 0.0f,
        .enable_anisotropy = false,
        .enable_compare = false,
        .props = 0
    };
    app->lightmap_sampler = SDL_CreateGPUSampler(app->device, &sampler_info);
    if (!app->lightmap_sampler) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "lightmap sampler creation failed: %s", SDL_GetError());
        return false;
    }

    char *lightmap_source = shader_load_source(SHADER_LIGHTMAP_PATH);
    if (!lightmap_source) return false;
    app->lightmap_pipeline = shader_compile_compute(app->device, lightmap_source);
    SDL_free(lightmap_source);
    return app->lightmap_pipeline != NULL;
}

static bool app_bake_lightmap(APP *app)
{
    SDL_GPUCommandBuffer *cmd = SDL_AcquireGPUCommandBuffer(app->device);
    if (!cmd) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "lightmap command buffer failed: %s", SDL_GetError());
        return false;
    }

    SDL_GPUStorageTextureReadWriteBinding target = {
        .texture = app->lightmap,
        .mip_level = 0,
        .layer = 0,
        .cycle = false
    };
    SDL_GPUComputePass *pass = SDL_BeginGPUComputePass(cmd, &target, 1, NULL, 0);
    SDL_BindGPUComputePipeline(pass, app->lightmap_pipeline);

    BAKE_UNIFORM bake = {
        .width = app->lightmap_size,
        .height = app->lightmap_size,
        .exposure = 0.80f,
        .padding = 0.0f
    };
    SDL_PushGPUComputeUniformData(cmd, 0, &bake, sizeof(bake));
    SDL_DispatchGPUCompute(pass, (bake.width + 7u) / 8u, (bake.height + 7u) / 8u, 1);
    SDL_EndGPUComputePass(pass);

    if (!SDL_SubmitGPUCommandBuffer(cmd)) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "lightmap submission failed: %s", SDL_GetError());
        return false;
    }
    return true;
}

static bool app_init(APP *app)
{
    SDL_memset(app, 0, sizeof(*app));

    if (!SDL_Init(SDL_INIT_VIDEO)) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "SDL_Init failed: %s", SDL_GetError());
        return false;
    }

    app->window = SDL_CreateWindow(
        "Untitled",
        1280,
        720,
        SDL_WINDOW_RESIZABLE);
    if (!app->window) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "window creation failed: %s", SDL_GetError());
        return false;
    }

    if (!SDL_ShaderCross_Init()) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "SDL_shadercross init failed: %s", SDL_GetError());
        return false;
    }
    app->shadercross_ready = true;

    SDL_GPUShaderFormat formats = SDL_ShaderCross_GetHLSLShaderFormats();
    app->device = SDL_CreateGPUDevice(formats, true, NULL);
    if (!app->device) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "GPU device creation failed: %s", SDL_GetError());
        return false;
    }

    if (!SDL_ClaimWindowForGPUDevice(app->device, app->window)) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "GPU window claim failed: %s", SDL_GetError());
        return false;
    }
    app->window_claimed = true;
    app->swapchain_format = SDL_GetGPUSwapchainTextureFormat(app->device, app->window);

    app->depth_format = SDL_GPU_TEXTUREFORMAT_D16_UNORM;
    if (SDL_GPUTextureSupportsFormat(
            app->device,
            SDL_GPU_TEXTUREFORMAT_D32_FLOAT,
            SDL_GPU_TEXTURETYPE_2D,
            SDL_GPU_TEXTUREUSAGE_DEPTH_STENCIL_TARGET)) {
        app->depth_format = SDL_GPU_TEXTUREFORMAT_D32_FLOAT;
    }

    int pixel_w = 0;
    int pixel_h = 0;
    if (!SDL_GetWindowSizeInPixels(app->window, &pixel_w, &pixel_h)) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "window pixel size failed: %s", SDL_GetError());
        return false;
    }
    if (!app_create_depth(app, (uint32_t)pixel_w, (uint32_t)pixel_h)) return false;

    if (!app_create_graphics_pipeline(app)) return false;
    if (!app_create_lightmap(app)) return false;

    app->cube = (MESH){
        .vertices = k_cube_vertices,
        .vertex_count = ARRAY_COUNT(k_cube_vertices),
        .indices = k_cube_indices,
        .index_count = ARRAY_COUNT(k_cube_indices)
    };
    app->floor = (MESH){
        .vertices = k_floor_vertices,
        .vertex_count = ARRAY_COUNT(k_floor_vertices),
        .indices = k_floor_indices,
        .index_count = ARRAY_COUNT(k_floor_indices)
    };

    if (!mesh_upload(app->device, &app->cube)) return false;
    if (!mesh_upload(app->device, &app->floor)) return false;
    if (!app_bake_lightmap(app)) return false;

    SDL_SetWindowRelativeMouseMode(app->window, true);
    SDL_Log("GPU backend: %s", SDL_GetGPUDeviceDriver(app->device));
    SDL_Log("WASD move, Q/E down/up, mouse look, ESC quit");
    return true;
}

static void app_destroy(APP *app)
{
    if (!app) return;

    if (app->device) {
        SDL_WaitForGPUIdle(app->device);

        mesh_destroy(app->device, &app->floor);
        mesh_destroy(app->device, &app->cube);

        if (app->lightmap_pipeline) SDL_ReleaseGPUComputePipeline(app->device, app->lightmap_pipeline);
        if (app->lightmap_sampler) SDL_ReleaseGPUSampler(app->device, app->lightmap_sampler);
        if (app->lightmap) SDL_ReleaseGPUTexture(app->device, app->lightmap);

        if (app->graphics_pipeline) SDL_ReleaseGPUGraphicsPipeline(app->device, app->graphics_pipeline);
        if (app->fragment_shader) SDL_ReleaseGPUShader(app->device, app->fragment_shader);
        if (app->vertex_shader) SDL_ReleaseGPUShader(app->device, app->vertex_shader);
        if (app->depth) SDL_ReleaseGPUTexture(app->device, app->depth);

        if (app->window_claimed && app->window) {
            SDL_ReleaseWindowFromGPUDevice(app->device, app->window);
        }
        SDL_DestroyGPUDevice(app->device);
    }

    if (app->shadercross_ready) SDL_ShaderCross_Quit();
    if (app->window) SDL_DestroyWindow(app->window);
    SDL_Quit();
    SDL_memset(app, 0, sizeof(*app));
}

static VEC3 camera_forward(const CAM *camera)
{
    float cp = cosf(camera->pitch);
    return v3_norm(v3(
        -sinf(camera->yaw) * cp,
        sinf(camera->pitch),
        -cosf(camera->yaw) * cp));
}

static void camera_update(CAM *camera, float dt)
{
    const bool *keys = SDL_GetKeyboardState(NULL);
    VEC3 forward = camera_forward(camera);
    VEC3 flat_forward = v3_norm(v3(forward.x, 0.0f, forward.z));
    VEC3 right = v3_norm(v3_cross(flat_forward, v3(0.0f, 1.0f, 0.0f)));

    VEC3 move = v3(0.0f, 0.0f, 0.0f);
    if (keys[SDL_SCANCODE_W]) move = v3_add(move, forward);
    if (keys[SDL_SCANCODE_S]) move = v3_sub(move, forward);
    if (keys[SDL_SCANCODE_D]) move = v3_add(move, right);
    if (keys[SDL_SCANCODE_A]) move = v3_sub(move, right);

    if (v3_dot(move, move) > 0.0f) move = v3_norm(move);
    float speed = keys[SDL_SCANCODE_LSHIFT] ? 8.0f : 3.0f;
    camera->position = v3_add(camera->position, v3_mul(move, speed * dt));
}

static void render_mesh(
    SDL_GPUCommandBuffer *cmd,
    SDL_GPURenderPass *pass,
    const MESH *mesh,
    const float view[16],
    const float projection[16],
    const float model[16],
    const float albedo[4])
{
    float view_model[16];
    DRAW_UNIFORM uniform;
    m4_mul(view_model, view, model);
    m4_mul(uniform.mvp, projection, view_model);
    SDL_memcpy(uniform.model, model, sizeof(uniform.model));
    SDL_memcpy(uniform.albedo, albedo, sizeof(uniform.albedo));

    SDL_PushGPUVertexUniformData(cmd, 0, &uniform, sizeof(uniform));

    SDL_GPUBufferBinding vertex_binding = {.buffer = mesh->vertex_buffer, .offset = 0};
    SDL_GPUBufferBinding index_binding = {.buffer = mesh->index_buffer, .offset = 0};
    SDL_BindGPUVertexBuffers(pass, 0, &vertex_binding, 1);
    SDL_BindGPUIndexBuffer(pass, &index_binding, SDL_GPU_INDEXELEMENTSIZE_32BIT);
    SDL_DrawGPUIndexedPrimitives(pass, mesh->index_count, 1, 0, 0, 0);
}

static bool app_render(APP *app, const CAM *camera)
{
    SDL_GPUCommandBuffer *cmd = SDL_AcquireGPUCommandBuffer(app->device);
    if (!cmd) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "frame command buffer failed: %s", SDL_GetError());
        return false;
    }

    SDL_GPUTexture *swapchain = NULL;
    uint32_t width = 0;
    uint32_t height = 0;
    if (!SDL_WaitAndAcquireGPUSwapchainTexture(cmd, app->window, &swapchain, &width, &height)) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "swapchain acquire failed: %s", SDL_GetError());
        SDL_CancelGPUCommandBuffer(cmd);
        return false;
    }

    if (!swapchain) {
        return SDL_SubmitGPUCommandBuffer(cmd);
    }

    if (!app_create_depth(app, width, height)) {
        SDL_SubmitGPUCommandBuffer(cmd);
        return false;
    }

    SDL_GPUColorTargetInfo color_target = {
        .texture = swapchain,
        .mip_level = 0,
        .layer_or_depth_plane = 0,
        .clear_color = {0.018f, 0.022f, 0.030f, 1.0f},
        .load_op = SDL_GPU_LOADOP_CLEAR,
        .store_op = SDL_GPU_STOREOP_STORE,
        .resolve_texture = NULL,
        .resolve_mip_level = 0,
        .resolve_layer = 0,
        .cycle = false,
        .cycle_resolve_texture = false
    };
    SDL_GPUDepthStencilTargetInfo depth_target = {
        .texture = app->depth,
        .clear_depth = 1.0f,
        .load_op = SDL_GPU_LOADOP_CLEAR,
        .store_op = SDL_GPU_STOREOP_DONT_CARE,
        .stencil_load_op = SDL_GPU_LOADOP_DONT_CARE,
        .stencil_store_op = SDL_GPU_STOREOP_DONT_CARE,
        .cycle = false,
        .clear_stencil = 0
    };

    SDL_GPURenderPass *pass = SDL_BeginGPURenderPass(cmd, &color_target, 1, &depth_target);
    SDL_BindGPUGraphicsPipeline(pass, app->graphics_pipeline);

    SDL_GPUTextureSamplerBinding lightmap_binding = {
        .texture = app->lightmap,
        .sampler = app->lightmap_sampler
    };
    SDL_BindGPUFragmentSamplers(pass, 0, &lightmap_binding, 1);

    float projection[16];
    float view[16];
    float model[16];
    VEC3 forward = camera_forward(camera);
    m4_perspective(projection, 60.0f * (3.14159265358979323846f / 180.0f), (float)width / (float)height, 0.05f, 100.0f);
    m4_look_at(view, camera->position, v3_add(camera->position, forward), v3(0.0f, 1.0f, 0.0f));

    static const float floor_color[4] = {0.58f, 0.60f, 0.62f, 1.0f};
    static const float cube_color[4] = {0.72f, 0.46f, 0.28f, 1.0f};

    m4_identity(model);
    render_mesh(cmd, pass, &app->floor, view, projection, model, floor_color);

    m4_translation(model, 0.0f, 0.0f, 0.0f);
    render_mesh(cmd, pass, &app->cube, view, projection, model, cube_color);

    m4_translation(model, 1.7f, 0.35f, -1.4f);
    render_mesh(cmd, pass, &app->cube, view, projection, model, cube_color);

    SDL_EndGPURenderPass(pass);
    if (!SDL_SubmitGPUCommandBuffer(cmd)) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "frame submission failed: %s", SDL_GetError());
        return false;
    }
    return true;
}

static int app_run(APP *app)
{
    CAM camera = {
        .position = {0.0f, 1.1f, 4.0f},
        .yaw = 0.0f,
        .pitch = -0.12f
    };

    bool running = true;
    uint64_t previous_ticks = SDL_GetTicks();

    while (running) {
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_EVENT_QUIT) running = false;
            if (event.type == SDL_EVENT_KEY_DOWN && event.key.scancode == SDL_SCANCODE_ESCAPE) running = false;
            if (event.type == SDL_EVENT_MOUSE_MOTION) {
                camera.yaw -= event.motion.xrel * 0.0025f;
                camera.pitch -= event.motion.yrel * 0.0025f;
                camera.pitch = SDL_clamp(camera.pitch, -1.50f, 1.50f);
            }
        }

        uint64_t now = SDL_GetTicks();
        float dt = (float)(now - previous_ticks) / 1000.0f;
        previous_ticks = now;
        dt = SDL_min(dt, 0.05f);

        camera_update(&camera, dt);
        if (running && !app_render(app, &camera)) return 1;
    }

    return 0;
}

int main(int argc, char **argv)
{
    (void)argc;
    (void)argv;

    APP app;
    if (!app_init(&app)) {
        app_destroy(&app);
        return 1;
    }

    int result = app_run(&app);
    app_destroy(&app);
    return result;
}
