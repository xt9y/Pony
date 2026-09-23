#include "dustmite.h"
#include "cache.h"

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

static void bake_progress(renderer *r, const char *stage, Uint32 done, Uint32 total) {

    if (!r || !r->window) return;

    r->bake_stage = stage;

    char title[160];
    if (total) {
        snprintf(title, sizeof(title), "Dustmite - B baking %s: %u/%u", stage, done, total);

        const double valid_texels = (double)r->lightmap_sample_count;
        const double completed_work = valid_texels * (double)done / (double)total;
        printf("frame time: %.2f ms | bake: %.2e/%.2e (%s)\n",
               r->frame_time_ms, completed_work, valid_texels, stage);
        fflush(stdout);
    } else {
        snprintf(title, sizeof(title), "Dustmite - B baking %s...", stage);
    }

    SDL_SetWindowTitle(r->window, title);
    SDL_PumpEvents();

}

static void bake_timing(const char *stage, Uint64 started) {

    const double elapsed = (double)(SDL_GetPerformanceCounter() - started) * 1000.0 /
                           (double)SDL_GetPerformanceFrequency();
    SDL_Log("B: %s took %.2f ms", stage, elapsed);
}

typedef struct mat4 {
    float m[16];
} mat4;
typedef struct color4 {
    float r, g, b, a;
} color4;

typedef struct camera_uniforms {
    mat4 mvp;
    mat4 view;
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
    float camera_forward[4];
} material_uniforms;

struct gpu_material {
    gltf_material data;
    SDL_GPUTexture *base_color;
    SDL_GPUTexture *metallic_roughness;
    SDL_GPUTexture *normal;
    SDL_GPUTexture *occlusion;
    SDL_GPUTexture *emissive;
};

struct draw_range {
    uint32_t first;
    uint32_t count;
    uint32_t material;
};

static vec3 scene_sun_direction(void) {

    return v3_normalize(v3(0.38f, 0.30f, 0.32f));

}

static mat4 m4_identity(void) {
    mat4 r = {0};
    r.m[0] = r.m[5] = r.m[10] = r.m[15] = 1.0f;
    return r;

}

static mat4 m4_mul(mat4 a, mat4 b) {
    mat4 r = {0};

    for (int c = 0; c < 4; ++c) {
        for (int row = 0; row < 4; ++row) {
            r.m[c * 4 + row] = a.m[row] * b.m[c * 4] + a.m[4 + row] * b.m[c * 4 + 1] + a.m[8 + row] * b.m[c * 4 + 2] + a.m[12 + row] * b.m[c * 4 + 3];
        }
    }

    return r;

}

static mat4 m4_perspective(float fov_y, float aspect, float znear, float zfar) {
    const float f = 1.0f / tanf(fov_y * 0.5f);

    mat4 r = {0};
    r.m[0] = f / aspect;
    r.m[5] = f;
    r.m[10] = zfar / (znear - zfar);
    r.m[11] = -1.0f;
    r.m[14] = (znear * zfar) / (znear - zfar);

    return r;

}

static mat4 m4_look_at(vec3 eye, vec3 target, vec3 up) {

    const vec3 f = v3_normalize(v3_sub(target, eye));
    const vec3 s = v3_normalize(v3_cross(f, up));
    const vec3 u = v3_cross(s, f);
    mat4 r = m4_identity();

    r.m[0] = s.x;
    r.m[1] = u.x;
    r.m[2] = -f.x;
    r.m[4] = s.y;
    r.m[5] = u.y;
    r.m[6] = -f.y;
    r.m[8] = s.z;
    r.m[9] = u.z;
    r.m[10] = -f.z;
    r.m[12] = -v3_dot(s, eye);
    r.m[13] = -v3_dot(u, eye);
    r.m[14] = v3_dot(f, eye);

    return r;
}

static Uint8 *compile_spirv(const char *path, const char *entrypoint, const char *define, SDL_ShaderCross_ShaderStage stage, size_t *size) {

    size_t source_size = 0;
    char *source = SDL_LoadFile(path, &source_size);
    if (!source) {
        SDL_Log("SDL_LoadFile(%s) failed: %s", path, SDL_GetError());
        return NULL;
    }

    SDL_ShaderCross_HLSL_Define defines[2] = {{.name = (char *)define, .value = NULL}, {0}};
    const SDL_ShaderCross_HLSL_Info hlsl = {.source = source, .entrypoint = entrypoint, .include_dir = "shaders", .defines = defines, .shader_stage = stage, .props = 0};
    Uint8 *spirv = SDL_ShaderCross_CompileSPIRVFromHLSL(&hlsl, size);
    SDL_free(source);
    if (!spirv) SDL_Log("shadercross HLSL->SPIR-V failed for %s:%s: %s", path, entrypoint, SDL_GetError());
    
    return spirv;

}

static SDL_GPUShader *compile_graphics_shader(renderer *r, const char *path, const char *entrypoint, const char *define, SDL_ShaderCross_ShaderStage stage) {
    
    size_t spirv_size = 0;
    Uint8 *spirv = compile_spirv(path, entrypoint, define, stage, &spirv_size);
    if (!spirv) return NULL;
    
    SDL_ShaderCross_GraphicsShaderMetadata *meta = SDL_ShaderCross_ReflectGraphicsSPIRV(spirv, spirv_size, 0);
    if (!meta) {
        SDL_Log("shader reflection failed for %s:%s: %s", path, entrypoint, SDL_GetError());
        SDL_free(spirv);
        return NULL;
    }
    
    const SDL_ShaderCross_SPIRV_Info info = {.bytecode = spirv, .bytecode_size = spirv_size, .entrypoint = entrypoint, .shader_stage = stage, .props = 0};
    SDL_GPUShader *shader = SDL_ShaderCross_CompileGraphicsShaderFromSPIRV(r->device, &info, &meta->resource_info, 0);
    
    SDL_free(meta);
    SDL_free(spirv);
    
    return shader;
}

static SDL_GPUComputePipeline *compile_compute_pipeline(renderer *r, const char *path, const char *entrypoint, const char *define) {
    
    size_t spirv_size = 0;
    Uint8 *spirv = compile_spirv(path, entrypoint, define, SDL_SHADERCROSS_SHADERSTAGE_COMPUTE, &spirv_size);
    if (!spirv) return NULL;
    
    SDL_ShaderCross_ComputePipelineMetadata *meta = SDL_ShaderCross_ReflectComputeSPIRV(spirv, spirv_size, 0);
    if (!meta) {
        SDL_Log("compute reflection failed for %s:%s: %s", path, entrypoint, SDL_GetError());
        SDL_free(spirv);
        return NULL;
    }
    
    const SDL_ShaderCross_SPIRV_Info info = {
        .bytecode = spirv, .bytecode_size = spirv_size, .entrypoint = entrypoint, .shader_stage = SDL_SHADERCROSS_SHADERSTAGE_COMPUTE, .props = 0
    };
    SDL_GPUComputePipeline *pipeline = SDL_ShaderCross_CompileComputePipelineFromSPIRV(r->device, &info, meta, 0);
    
    SDL_free(meta);
    SDL_free(spirv);
    
    return pipeline;
}

static SDL_GPUGraphicsPipeline *make_surface_pipeline(renderer *r, SDL_GPUShader *vs, SDL_GPUShader *ps) {
    
    const SDL_GPUVertexBufferDescription vb = {.slot = 0, .pitch = (Uint32)sizeof(render_vertex), .input_rate = SDL_GPU_VERTEXINPUTRATE_VERTEX};
    const SDL_GPUVertexAttribute attrs[4] = {
        {.location = 0, .buffer_slot = 0, .format = SDL_GPU_VERTEXELEMENTFORMAT_FLOAT3, .offset = (Uint32)offsetof(render_vertex, x)},
        {.location = 1, .buffer_slot = 0, .format = SDL_GPU_VERTEXELEMENTFORMAT_FLOAT3, .offset = (Uint32)offsetof(render_vertex, nx)},
        {.location = 2, .buffer_slot = 0, .format = SDL_GPU_VERTEXELEMENTFORMAT_FLOAT2, .offset = (Uint32)offsetof(render_vertex, u)},
        {.location = 3, .buffer_slot = 0, .format = SDL_GPU_VERTEXELEMENTFORMAT_FLOAT2, .offset = (Uint32)offsetof(render_vertex, lu)}
    };
    const SDL_GPUColorTargetDescription targets[2] = {{.format = SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT}, {.format = SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT}};
    return SDL_CreateGPUGraphicsPipeline(r->device, &(SDL_GPUGraphicsPipelineCreateInfo){
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
       .multisample_state = {
            .sample_count = SDL_GPU_SAMPLECOUNT_1
        },
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

static SDL_GPUGraphicsPipeline *make_line_pipeline(renderer *r, SDL_GPUShader *vs, SDL_GPUShader *ps) {
    
    const SDL_GPUVertexBufferDescription vb = {.slot = 0, .pitch = (Uint32)sizeof(render_vertex), .input_rate = SDL_GPU_VERTEXINPUTRATE_VERTEX};
    const SDL_GPUVertexAttribute attrs[2] = {
        {.location = 0, .buffer_slot = 0, .format = SDL_GPU_VERTEXELEMENTFORMAT_FLOAT3, .offset = (Uint32)offsetof(render_vertex, x)},
        {.location = 1, .buffer_slot = 0, .format = SDL_GPU_VERTEXELEMENTFORMAT_FLOAT4, .offset = (Uint32)offsetof(render_vertex, r)}
    };
    const SDL_GPUColorTargetDescription targets[2] = {
        {.format = SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT},
        {.format = SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT, .blend_state = {.color_write_mask = 0, .enable_color_write_mask = true}}
    };
    
    return SDL_CreateGPUGraphicsPipeline(r->device, &(SDL_GPUGraphicsPipelineCreateInfo){
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
       .multisample_state = {
            .sample_count = SDL_GPU_SAMPLECOUNT_1
        },
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

static SDL_GPUGraphicsPipeline *make_sky_pipeline(renderer *r, SDL_GPUShader *vs, SDL_GPUShader *ps) {
    
    const SDL_GPUColorTargetDescription targets[2] = {{.format = SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT}, {.format = SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT}};
    return SDL_CreateGPUGraphicsPipeline(r->device, &(SDL_GPUGraphicsPipelineCreateInfo){
       .vertex_shader = vs,
       .fragment_shader = ps,
       .primitive_type = SDL_GPU_PRIMITIVETYPE_TRIANGLELIST,
       .rasterizer_state = {
            .fill_mode = SDL_GPU_FILLMODE_FILL, 
            .cull_mode = SDL_GPU_CULLMODE_NONE, 
            .front_face = SDL_GPU_FRONTFACE_COUNTER_CLOCKWISE, 
            .enable_depth_clip = true
        },
       .multisample_state = {
            .sample_count = SDL_GPU_SAMPLECOUNT_1
        },
       .target_info = {
            .color_target_descriptions = targets, 
            .num_color_targets = 2, 
            .depth_stencil_format = r->depth_format, 
            .has_depth_stencil_target = true
        }
   });

}

static bool ensure_depth_texture(renderer *r, Uint32 width, Uint32 height) {
    if (r->depth_texture && r->depth_width == width && r->depth_height == height) return true;
    if (r->depth_texture) SDL_ReleaseGPUTexture(r->device, r->depth_texture);
    r->depth_texture = SDL_CreateGPUTexture(r->device, &(SDL_GPUTextureCreateInfo){
        .type = SDL_GPU_TEXTURETYPE_2D,
        .format = r->depth_format,
        .usage = SDL_GPU_TEXTUREUSAGE_DEPTH_STENCIL_TARGET,
        .width = width,
        .height = height,
        .layer_count_or_depth = 1,
        .num_levels = 1,
        .sample_count = SDL_GPU_SAMPLECOUNT_1
   });

    if (!r->depth_texture) return false;
    r->depth_width = width;
    r->depth_height = height;

    return true;
}

static SDL_GPUBuffer *upload_buffer(renderer *r, SDL_GPUBufferUsageFlags usage, const void *data, size_t bytes) {
    
    if (!data || !bytes || bytes > UINT32_MAX) return NULL;
    SDL_GPUBuffer *buffer = SDL_CreateGPUBuffer(r->device, &(SDL_GPUBufferCreateInfo){.usage = usage, .size = (Uint32)bytes});
    
    if (!buffer) return NULL;
    SDL_GPUTransferBuffer *transfer = SDL_CreateGPUTransferBuffer(r->device, &(SDL_GPUTransferBufferCreateInfo){
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
    SDL_UploadToGPUBuffer(copy, &(SDL_GPUTransferBufferLocation){
        .transfer_buffer = transfer, 
        .offset = 0
    }, &(SDL_GPUBufferRegion){
        .buffer = buffer, 
        .offset = 0, 
        .size = (Uint32)bytes
    }, false);
    
    SDL_EndGPUCopyPass(copy);
    if (!SDL_SubmitGPUCommandBuffer(cmd)) {
        SDL_ReleaseGPUTransferBuffer(r->device, transfer);
        SDL_ReleaseGPUBuffer(r->device, buffer);
        return NULL;
    
    }
    SDL_ReleaseGPUTransferBuffer(r->device, transfer);
    
    return buffer;

}

static SDL_GPUTexture *pixel_texture(renderer *r, Uint8 red, Uint8 green, Uint8 blue, Uint8 alpha) {
    SDL_GPUTexture *texture = SDL_CreateGPUTexture(r->device, &(SDL_GPUTextureCreateInfo){
        .type = SDL_GPU_TEXTURETYPE_2D,
        .format = SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM,
        .usage = SDL_GPU_TEXTUREUSAGE_SAMPLER,
        .width = 1,
        .height = 1,
        .layer_count_or_depth = 1,
        .num_levels = 1,
        .sample_count = SDL_GPU_SAMPLECOUNT_1
    });
    if (!texture) return NULL;

    SDL_GPUTransferBuffer *transfer = SDL_CreateGPUTransferBuffer(r->device, &(SDL_GPUTransferBufferCreateInfo){.usage = SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD, .size = 4});
    if (!transfer) {
        SDL_ReleaseGPUTexture(r->device, texture);
        return NULL;
    }

    Uint8 *pixels = SDL_MapGPUTransferBuffer(r->device, transfer, false);
    if (!pixels) {
        SDL_ReleaseGPUTransferBuffer(r->device, transfer);
        SDL_ReleaseGPUTexture(r->device, texture);
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
        SDL_ReleaseGPUTexture(r->device, texture);
        return NULL;
    }

    SDL_GPUCopyPass *copy = SDL_BeginGPUCopyPass(cmd);
    SDL_UploadToGPUTexture(copy, &(SDL_GPUTextureTransferInfo){.transfer_buffer = transfer}, &(SDL_GPUTextureRegion){
        .texture = texture, 
        .mip_level = 0, 
        .layer = 0, 
        .x = 0, .y = 0, .z = 0, .w = 1, .h = 1, .d = 1
    }, false);
    
    SDL_EndGPUCopyPass(copy);
    if (!SDL_SubmitGPUCommandBuffer(cmd)) {
        SDL_ReleaseGPUTransferBuffer(r->device, transfer);
        SDL_ReleaseGPUTexture(r->device, texture);
        return NULL;
    }
    
    SDL_ReleaseGPUTransferBuffer(r->device, transfer);
    return texture;

}

static bool reserve_vertices(renderer *r, uint32_t needed) {

    if (needed <= r->vertex_capacity) return true;

    uint32_t capacity = r->vertex_capacity ? r->vertex_capacity : 1024u;
    while (capacity < needed) {
        if (capacity > UINT32_MAX / 2u) return false;
        capacity *= 2u;
    }

    render_vertex *vertices = realloc(r->vertices, (size_t)capacity * sizeof(*vertices));
    if (!vertices) return false;
    r->vertices = vertices;
    r->vertex_capacity = capacity;
    
    return true;
}

static bool push_surface(renderer *r, const gltf_vertex *v, lmap_uv uv) {

    if (!reserve_vertices(r, r->vertex_count + 1u)) return false;
    r->vertices[r->vertex_count++] = (render_vertex){
        .x = v->position.x,
        .y = v->position.y,
        .z = v->position.z,
        .nx = v->normal.x,
        .ny = v->normal.y,
        .nz = v->normal.z,
        .u = v->u,
        .v = v->v,
        .lu = uv.u,
        .lv = uv.v,
        .r = 1,
        .g = 1,
        .b = 1,
        .a = 1
    };
    return true;
}

static bool push_line_vertex(renderer *r, vec3 p, color4 c) {

    if (!reserve_vertices(r, r->vertex_count + 1u)) return false;
    r->vertices[r->vertex_count++] = (render_vertex){.x = p.x, .y = p.y, .z = p.z, .r = c.r, .g = c.g, .b = c.b, .a = c.a};
    return true;
}

static bool add_wire_triangle(renderer *r, vec3 a, vec3 b, vec3 c, color4 color) {

    return push_line_vertex(r, a, color) && 
           push_line_vertex(r, b, color) && 
           push_line_vertex(r, b, color) && 
           push_line_vertex(r, c, color) && 
           push_line_vertex(r, c, color) &&
           push_line_vertex(r, a, color);
}

static bool upload_vertices(renderer *r) {

    if (r->vertex_buffer) SDL_ReleaseGPUBuffer(r->device, r->vertex_buffer);
    r->vertex_buffer = upload_buffer(r, SDL_GPU_BUFFERUSAGE_VERTEX, r->vertices, (size_t)r->vertex_count * sizeof(*r->vertices));
    return r->vertex_buffer != NULL;
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
        r->image_textures[i] = IMG_LoadGPUTextureTyped_IO(r->device, copy, io, true, image_type(image->mime), NULL, NULL);
        if (!r->image_textures[i]) SDL_Log("SDL_image could not decode GLB image %u (%s): %s", i, image->mime[0] ? image->mime : "unknown", SDL_GetError());
    }

    SDL_EndGPUCopyPass(copy);
    return SDL_SubmitGPUCommandBuffer(cmd);
}

static SDL_GPUTexture *resolve_texture(renderer *r, const gltf_scene *visual, int32_t texture_index, SDL_GPUTexture *fallback) {

    if (texture_index < 0 || (uint32_t)texture_index >= visual->texture_count) return fallback;
    const int32_t image = visual->textures[texture_index].image;
    if (image < 0 || (uint32_t)image >= r->image_texture_count || !r->image_textures[image]) return fallback;
    return r->image_textures[image];
}

static bool setup_materials(renderer *r, const gltf_scene *visual) {
    r->default_white = pixel_texture(r, 255, 255, 255, 255);
    r->default_normal = pixel_texture(r, 128, 128, 255, 255);
    r->material_sampler = SDL_CreateGPUSampler(r->device, &(SDL_GPUSamplerCreateInfo){
        .min_filter = SDL_GPU_FILTER_LINEAR,
        .mag_filter = SDL_GPU_FILTER_LINEAR,
        .mipmap_mode = SDL_GPU_SAMPLERMIPMAPMODE_NEAREST,
        .address_mode_u = SDL_GPU_SAMPLERADDRESSMODE_REPEAT,
        .address_mode_v = SDL_GPU_SAMPLERADDRESSMODE_REPEAT,
        .address_mode_w = SDL_GPU_SAMPLERADDRESSMODE_REPEAT
    });
    if (!r->default_white || !r->default_normal || !r->material_sampler || !load_images(r, visual)) return false;

    r->material_count = visual->material_count;
    r->materials = calloc(r->material_count, sizeof(*r->materials));
    if (!r->materials) return false;
    for (uint32_t i = 0; i < r->material_count; ++i) {
        gpu_material *m = &r->materials[i];
        m->data = visual->materials[i];
        m->base_color = resolve_texture(r, visual, m->data.base_color_texture, r->default_white);
        m->metallic_roughness = resolve_texture(r, visual, m->data.metallic_roughness_texture, r->default_white);
        m->normal = resolve_texture(r, visual, m->data.normal_texture, r->default_normal);
        m->occlusion = resolve_texture(r, visual, m->data.occlusion_texture, r->default_white);
        m->emissive = resolve_texture(r, visual, m->data.emissive_texture, r->default_white);
    }
    return true;
}

static SDL_GPUTexture *create_lightmap_texture(renderer *r, Uint32 width, Uint32 height) {
    return SDL_CreateGPUTexture(r->device, &(SDL_GPUTextureCreateInfo){
        .type = SDL_GPU_TEXTURETYPE_2D,
        .format = SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT,
        .usage = SDL_GPU_TEXTUREUSAGE_SAMPLER | SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_WRITE,
        .width = width,
        .height = height,
        .layer_count_or_depth = 1,
        .num_levels = 1,
        .sample_count = SDL_GPU_SAMPLECOUNT_1
    });
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

static bake_uniforms bake_data(renderer *r, Uint32 phase, Uint32 iteration, Uint32 item_count, Uint32 dispatch_width) {

    const vec3 sun = scene_sun_direction();
    return (bake_uniforms){
        .item_count = item_count,
        .lightmap_width = r->lightmap_width,
        .lightmap_height = r->lightmap_height,
        .dispatch_width = dispatch_width,
        .iteration = iteration,
        .phase = phase,
        .max_bounces = BAKE_MAX_BOUNCES,
        .padding = 0u,
        .sun_direction_intensity = {sun.x, sun.y, sun.z, 2.4f},
        .sun_color_radius = {1.00f, 0.94f, 0.84f, 0.00465f},
        .sky_zenith = {0.22f, 0.42f, 0.78f, 1.0f},
        .sky_horizon = {0.68f, 0.76f, 0.88f, 1.0f},
        .bake_params = {r->bake_epsilon, 0.72f, 1.0f, 0.0f}
    };
}

static bool record_bake_pass(renderer *r, SDL_GPUCommandBuffer *cmd, SDL_GPUTexture *source, SDL_GPUTexture *destination, Uint32 phase, Uint32 iteration, Uint32 item_count) {

    Uint32 groups_x, groups_y, dispatch_width;
    dispatch_shape(item_count, &groups_x, &groups_y, &dispatch_width);

    const bake_uniforms uniforms = bake_data(r, phase, iteration, item_count, dispatch_width);
    const SDL_GPUStorageTextureReadWriteBinding output = {.texture = destination, .mip_level = 0, .layer = 0, .cycle = false};
    SDL_GPUComputePass *pass = SDL_BeginGPUComputePass(cmd, &output, 1, NULL, 0);
    if (!pass) return false;

    SDL_BindGPUComputePipeline(pass, r->bake_pipeline);


    const SDL_GPUTextureSamplerBinding source_binding = {.texture = source, .sampler = r->lightmap_sampler};
    SDL_BindGPUComputeSamplers(pass, 0, &source_binding, 1);
    SDL_GPUBuffer *buffers[3] = {r->bvh_node_buffer, r->bvh_triangle_buffer, r->lightmap_sample_buffer};
    SDL_BindGPUComputeStorageBuffers(pass, 0, buffers, 3);
    SDL_PushGPUComputeUniformData(cmd, 0, &uniforms, sizeof(uniforms));
    SDL_DispatchGPUCompute(pass, groups_x, groups_y, 1);
    SDL_EndGPUComputePass(pass);

    return true;
}

static void swap_lightmaps(renderer *r) {

    SDL_GPUTexture *tmp = r->lightmap_texture;
    r->lightmap_texture = r->lightmap_scratch;
    r->lightmap_scratch = tmp;
}

static void release_bake_buffers(renderer *r) {

    if (r->bvh_node_buffer) SDL_ReleaseGPUBuffer(r->device, r->bvh_node_buffer);
    if (r->bvh_triangle_buffer) SDL_ReleaseGPUBuffer(r->device, r->bvh_triangle_buffer);
    if (r->lightmap_sample_buffer) SDL_ReleaseGPUBuffer(r->device, r->lightmap_sample_buffer);

    r->bvh_node_buffer = r->bvh_triangle_buffer = r->lightmap_sample_buffer = NULL;

}

static bool submit_trace_batch(renderer *r, Uint32 first, Uint32 count) {

    SDL_GPUCommandBuffer *cmd = SDL_AcquireGPUCommandBuffer(r->device);
    if (!cmd) return false;

    for (Uint32 i = 0; i < count; ++i) {

        if (!record_bake_pass(r, cmd, r->lightmap_texture, r->lightmap_scratch, PHASE_TRACE, first + i, r->lightmap_sample_count)) {
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
    if (!record_bake_pass(r, clear, r->lightmap_scratch, r->lightmap_texture, PHASE_CLEAR, 0, pixels) ||
        !record_bake_pass(r, clear, r->lightmap_texture, r->lightmap_scratch, PHASE_CLEAR, 0, pixels)) {
        SDL_CancelGPUCommandBuffer(clear);
        return false;
    }

    if (!SDL_SubmitGPUCommandBuffer(clear)) return false;


    for (Uint32 first = 0; first < r->bake_target_samples; first += BAKE_BATCH_SAMPLES) {

        Uint32 count = r->bake_target_samples - first;
        if (count > BAKE_BATCH_SAMPLES) count = BAKE_BATCH_SAMPLES;
        if (!submit_trace_batch(r, first, count) || !SDL_WaitForGPUIdle(r->device)) return false;

        bake_progress(r, "surface lightmap", first + count, r->bake_target_samples);
    }

    bake_progress(r, "filtering lightmap", 0u, 0u);

    SDL_GPUCommandBuffer *post = SDL_AcquireGPUCommandBuffer(r->device);
    if (!post) return false;
    if (!record_bake_pass(r, post, r->lightmap_texture, r->lightmap_scratch, PHASE_FILTER, 0, pixels)) {
        SDL_CancelGPUCommandBuffer(post);
        return false;
    }


    swap_lightmaps(r);
    for (Uint32 i = 0; i < BAKE_DILATION_PASSES; ++i) {

        if (!record_bake_pass(r, post, r->lightmap_texture, r->lightmap_scratch, PHASE_DILATE, 0, pixels)) {
            SDL_CancelGPUCommandBuffer(post);
            return false;
        }
        swap_lightmaps(r);
    }

    if (!SDL_SubmitGPUCommandBuffer(post) || !SDL_WaitForGPUIdle(r->device)) return false;

    if (r->lightmap_scratch) {
        SDL_ReleaseGPUTexture(r->device, r->lightmap_scratch);
        r->lightmap_scratch = NULL;
    }
    if (r->bake_pipeline) {
        SDL_ReleaseGPUComputePipeline(r->device, r->bake_pipeline);
        r->bake_pipeline = NULL;
    }

    return true;

}

static bool setup_lightmap(renderer *r, const bvh *tree, const lightmap *lm) {

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
    if (!r->lightmap_texture || !r->lightmap_scratch || !r->lightmap_sampler) return false;

    r->bvh_node_buffer = upload_buffer(r, SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_READ, tree->nodes, (size_t)tree->node_count * sizeof(*tree->nodes));
    r->bvh_triangle_buffer = upload_buffer(r, SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_READ, tree->triangles, (size_t)tree->triangle_count * sizeof(*tree->triangles));
    r->lightmap_sample_buffer = upload_buffer(r, SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_READ, lm->samples, (size_t)lm->sample_count * sizeof(*lm->samples));
    if (!r->bvh_node_buffer || !r->bvh_triangle_buffer || !r->lightmap_sample_buffer) return false;

    SDL_Log("lightmap: %ux%u, %u charts, %u valid texels, %.2f texels/unit", lm->width, lm->height, lm->chart_count, lm->sample_count, lm->texel_density);
    return bake_lightmap_once(r);
}

static bool transfer_size(uint32_t width, uint32_t height, Uint32 *out) {
    const uint64_t bytes = (uint64_t)width * (uint64_t)height * 8u;
    if (!width || !height || bytes > UINT32_MAX) return false;
    *out = (Uint32)bytes;
    return true;
}

static bool upload_cached_texture(renderer *r, const dm_cached_lightmap *cached,
                                  SDL_GPUTexture **out) {
    Uint32 bytes;
    if (!transfer_size(cached->width, cached->height, &bytes)) return false;
    SDL_GPUTexture *texture = create_lightmap_texture(r, cached->width, cached->height);
    SDL_GPUTransferBuffer *transfer = SDL_CreateGPUTransferBuffer(r->device,
        &(SDL_GPUTransferBufferCreateInfo){.usage = SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD,
                                           .size = bytes});
    if (!texture || !transfer) goto fail;
    void *mapped = SDL_MapGPUTransferBuffer(r->device, transfer, false);
    if (!mapped) goto fail;
    memcpy(mapped, cached->pixels, bytes);
    SDL_UnmapGPUTransferBuffer(r->device, transfer);
    SDL_GPUCommandBuffer *cmd = SDL_AcquireGPUCommandBuffer(r->device);
    if (!cmd) goto fail;
    SDL_GPUCopyPass *pass = SDL_BeginGPUCopyPass(cmd);
    if (!pass) { SDL_CancelGPUCommandBuffer(cmd); goto fail; }
    SDL_UploadToGPUTexture(pass,
        &(SDL_GPUTextureTransferInfo){.transfer_buffer = transfer},
        &(SDL_GPUTextureRegion){.texture = texture, .w = cached->width,
                                .h = cached->height, .d = 1}, false);
    SDL_EndGPUCopyPass(pass);
    if (!SDL_SubmitGPUCommandBuffer(cmd)) goto fail;
    SDL_ReleaseGPUTransferBuffer(r->device, transfer);
    *out = texture;
    return true;
fail:
    if (transfer) SDL_ReleaseGPUTransferBuffer(r->device, transfer);
    if (texture) SDL_ReleaseGPUTexture(r->device, texture);
    return false;
}

static bool download_current_lightmap(renderer *r, dm_cached_lightmap *out) {
    Uint32 bytes;
    if (!transfer_size(r->lightmap_width, r->lightmap_height, &bytes)) return false;
    SDL_GPUTransferBuffer *transfer = SDL_CreateGPUTransferBuffer(r->device,
        &(SDL_GPUTransferBufferCreateInfo){.usage = SDL_GPU_TRANSFERBUFFERUSAGE_DOWNLOAD,
                                           .size = bytes});
    if (!transfer) return false;
    SDL_GPUCommandBuffer *cmd = SDL_AcquireGPUCommandBuffer(r->device);
    if (!cmd) { SDL_ReleaseGPUTransferBuffer(r->device, transfer); return false; }
    SDL_GPUCopyPass *pass = SDL_BeginGPUCopyPass(cmd);
    if (!pass) {
        SDL_CancelGPUCommandBuffer(cmd);
        SDL_ReleaseGPUTransferBuffer(r->device, transfer);
        return false;
    }
    SDL_DownloadFromGPUTexture(pass,
        &(SDL_GPUTextureRegion){.texture = r->lightmap_texture,
                                .w = r->lightmap_width, .h = r->lightmap_height, .d = 1},
        &(SDL_GPUTextureTransferInfo){.transfer_buffer = transfer});
    SDL_EndGPUCopyPass(pass);
    SDL_GPUFence *fence = SDL_SubmitGPUCommandBufferAndAcquireFence(cmd);
    if (!fence) { SDL_ReleaseGPUTransferBuffer(r->device, transfer); return false; }
    bool good = SDL_WaitForGPUFences(r->device, true, &fence, 1);
    SDL_ReleaseGPUFence(r->device, fence);
    if (good) {
        void *mapped = SDL_MapGPUTransferBuffer(r->device, transfer, false);
        if (mapped) {
            out->pixels = malloc(bytes);
            if (out->pixels) {
                memcpy(out->pixels, mapped, bytes);
                out->width = r->lightmap_width;
                out->height = r->lightmap_height;
            } else good = false;
            SDL_UnmapGPUTransferBuffer(r->device, transfer);
        } else good = false;
    }
    SDL_ReleaseGPUTransferBuffer(r->device, transfer);
    return good;
}

static void free_probe_grid(dm_probe_grid *grid) {
    if (!grid) return;
    free(grid->probes);
    memset(grid, 0, sizeof(*grid));
}

static SDL_GPUBuffer *upload_beams(renderer *r, const dm_beam_grid *grid) {

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

    SDL_GPUBuffer *buffer = upload_buffer(r, SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_READ,
        data, (beam_count + depth_count) * sizeof(float));
    free(data);
    return buffer;
}

static bool make_probe_grid(const mesh *m, float spacing, dm_probe_grid *grid) {

    if (!m || !grid || spacing <= 0.0f) return false;
    memset(grid, 0, sizeof(*grid));
    const vec3 extent = v3_sub(m->bounds.max, m->bounds.min);
    if (!isfinite(extent.x) || !isfinite(extent.y) || !isfinite(extent.z) ||
        extent.x < 0.0f || extent.y < 0.0f || extent.z < 0.0f) return false;
    if (extent.x / spacing > 16384.0f || extent.y / spacing > 16384.0f ||
        extent.z / spacing > 16384.0f) return false;
    grid->count_x = (uint32_t)ceilf(extent.x / spacing) + 1u;
    grid->count_y = (uint32_t)ceilf(extent.y / spacing) + 1u;
    grid->count_z = (uint32_t)ceilf(extent.z / spacing) + 1u;
    uint64_t count = (uint64_t)grid->count_x * grid->count_y * grid->count_z;
    if (!count || count > 16384u) return false;
    grid->origin = m->bounds.min;
    grid->spacing = spacing;
    grid->probes = calloc((size_t)count, sizeof(*grid->probes));
    if (!grid->probes) return false;
    for (uint32_t z = 0; z < grid->count_z; ++z)
        for (uint32_t y = 0; y < grid->count_y; ++y)
            for (uint32_t x = 0; x < grid->count_x; ++x) {
                const size_t index = x + (size_t)grid->count_x *
                    (y + (size_t)grid->count_y * z);
                dm_probe *p = &grid->probes[index];
                p->position[0] = grid->origin.x + x * spacing;
                p->position[1] = grid->origin.y + y * spacing;
                p->position[2] = grid->origin.z + z * spacing;
                p->position[3] = 1.0f;
            }
    return true;
}

static bool bake_probe_grid(renderer *r, dm_probe_grid *grid, Uint32 samples) {

    uint64_t count = (uint64_t)grid->count_x * grid->count_y * grid->count_z;
    const Uint32 output_bytes = (Uint32)(count * 9u * sizeof(float[4]));
    const Uint32 input_bytes = (Uint32)(count * sizeof(float[4]));
    float (*positions)[4] = malloc(input_bytes);
    if (!positions) return false;
    for (uint32_t i = 0; i < (uint32_t)count; ++i)
        memcpy(positions[i], grid->probes[i].position, sizeof(positions[i]));
    SDL_GPUBuffer *input = upload_buffer(r, SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_READ,
                                          positions, input_bytes);
    free(positions);
    SDL_GPUBuffer *output = SDL_CreateGPUBuffer(r->device,
        &(SDL_GPUBufferCreateInfo){.usage = SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_WRITE,
                                   .size = output_bytes});
    SDL_GPUTransferBuffer *transfer = SDL_CreateGPUTransferBuffer(r->device,
        &(SDL_GPUTransferBufferCreateInfo){.usage = SDL_GPU_TRANSFERBUFFERUSAGE_DOWNLOAD,
                                           .size = output_bytes});
    SDL_GPUComputePipeline *pipeline = compile_compute_pipeline(r, "shaders/compute.hlsl",
                                                                  "probe_cs", "BUILD_PROBE_CS");
    bool good = input && output && transfer && pipeline;
    if (good) {
        SDL_GPUCommandBuffer *cmd = SDL_AcquireGPUCommandBuffer(r->device);
        if (!cmd) good = false;
        if (cmd) {
            SDL_GPUStorageBufferReadWriteBinding binding = {.buffer = output};
            SDL_GPUComputePass *pass = SDL_BeginGPUComputePass(cmd, NULL, 0, &binding, 1);
            if (!pass) { SDL_CancelGPUCommandBuffer(cmd); good = false; }
            else {
                SDL_BindGPUComputePipeline(pass, pipeline);
                SDL_GPUBuffer *buffers[3] = {input, r->bvh_node_buffer, r->bvh_triangle_buffer};
                SDL_BindGPUComputeStorageBuffers(pass, 0, buffers, 3);
                bake_uniforms u = bake_data(r, 0u, 0u, samples, 0u);
                SDL_PushGPUComputeUniformData(cmd, 0, &u, sizeof(u));
                SDL_DispatchGPUCompute(pass, (Uint32)count, 1u, 1u);
                SDL_EndGPUComputePass(pass);
                SDL_GPUCopyPass *copy = SDL_BeginGPUCopyPass(cmd);
                if (!copy) { SDL_CancelGPUCommandBuffer(cmd); good = false; }
                else {
                    SDL_DownloadFromGPUBuffer(copy,
                        &(SDL_GPUBufferRegion){.buffer = output, .size = output_bytes},
                        &(SDL_GPUTransferBufferLocation){.transfer_buffer = transfer});
                    SDL_EndGPUCopyPass(copy);
                    SDL_GPUFence *fence = SDL_SubmitGPUCommandBufferAndAcquireFence(cmd);
                    if (!fence) good = false;
                    else {
                        good = SDL_WaitForGPUFences(r->device, true, &fence, 1u);
                        SDL_ReleaseGPUFence(r->device, fence);
                    }
                }
            }
        }
    }
    if (good) {
        const float (*values)[4] = SDL_MapGPUTransferBuffer(r->device, transfer, false);
        if (!values) good = false;
        else {
            for (uint32_t i = 0; i < (uint32_t)count; ++i) {
                for (uint32_t j = 0; j < 9u; ++j)
                    memcpy(grid->probes[i].coefficients[j], values[i * 9u + j],
                           sizeof(float[4]));
                grid->probes[i].position[3] = values[i * 9u][3];
            }
            SDL_UnmapGPUTransferBuffer(r->device, transfer);
        }
    }
    if (pipeline) SDL_ReleaseGPUComputePipeline(r->device, pipeline);
    if (transfer) SDL_ReleaseGPUTransferBuffer(r->device, transfer);
    if (output) SDL_ReleaseGPUBuffer(r->device, output);
    if (input) SDL_ReleaseGPUBuffer(r->device, input);
    return good;
}

bool r_load_cached_lightmap(renderer *r, const char *path, uint64_t scene_hash,
                            uint64_t layout_hash, const lightmap *lm) {
    if (!r || !r->device || !lm) return false;
    dm_cached_lightmap cached = {0};
    if (!dm_cache_read(path, scene_hash, layout_hash, &cached)) return false;
    SDL_GPUTexture *replacement = NULL;
    bool good = cached.width == lm->width && cached.height == lm->height &&
                upload_cached_texture(r, &cached, &replacement);
    SDL_GPUBuffer *volume_buffer = NULL;
    SDL_GPUBuffer *beam_buffer = NULL;
    if (good) {
        uint64_t count = (uint64_t)cached.volume_probes.count_x *
            cached.volume_probes.count_y * cached.volume_probes.count_z;
        volume_buffer = upload_buffer(r, SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_READ,
            cached.volume_probes.probes, (size_t)count * sizeof(dm_probe));
        good = volume_buffer != NULL;
    }
    if (good) {
        beam_buffer = upload_beams(r, &cached.beams);
        good = beam_buffer != NULL;
    }
    if (good) {
        SDL_GPUTexture *old = r->lightmap_texture;
        SDL_GPUBuffer *old_volume = r->volume_probe_buffer;
        SDL_GPUBuffer *old_beam = r->beam_buffer;
        r->lightmap_texture = replacement;
        r->volume_probe_buffer = volume_buffer;
        r->beam_buffer = beam_buffer;
        r->lightmap_width = cached.width;
        r->lightmap_height = cached.height;
        free_probe_grid(&r->object_probes);
        free_probe_grid(&r->volume_probes);
        r->object_probes = cached.object_probes;
        r->volume_probes = cached.volume_probes;
        dm_beam_free(&r->beams);
        r->beams = cached.beams;
        cached.beams.cells = NULL;
        cached.beams.shadow_depth = NULL;
        cached.object_probes.probes = NULL;
        cached.volume_probes.probes = NULL;
        r->has_bake = true;
        SDL_ReleaseGPUTexture(r->device, old);
        if (old_volume) SDL_ReleaseGPUBuffer(r->device, old_volume);
        if (old_beam) SDL_ReleaseGPUBuffer(r->device, old_beam);
    }
    if (!good) {
        if (replacement) SDL_ReleaseGPUTexture(r->device, replacement);
        if (volume_buffer) SDL_ReleaseGPUBuffer(r->device, volume_buffer);
        if (beam_buffer) SDL_ReleaseGPUBuffer(r->device, beam_buffer);
    }
    dm_cache_free(&cached);
    return good;
}

bool r_rebake_current_scene(renderer *r, const mesh *m, const gltf_scene *visual,
                            const lightmap *lm,
                            const char *path, uint64_t scene_hash, uint64_t layout_hash) {

    if (!r || !m || !lm || !r->device) return false;

    bake_progress(r, "scene geometry", 0u, 0u);

    Uint64 started = SDL_GetPerformanceCounter();
    bvh tree = {0};
    if (!bvh_build(&tree, m, visual)) return false;
    // SDL_Log("probe grid %ux%ux%u spacing=%.2f", grid->count_x, grid->count_y, grid->count_z, grid->spacing);
    
    bake_timing("scene geometry", started);

    bake_progress(r, "lightmap shader", 0u, 0u);

    SDL_GPUTexture *old = r->lightmap_texture;
    Uint32 old_width = r->lightmap_width, old_height = r->lightmap_height;
    bool had_bake = r->has_bake;
    r->lightmap_texture = NULL;
    r->bake_pipeline = compile_compute_pipeline(r, "shaders/compute.hlsl",
                                                  "lightmap_cs", "BUILD_LIGHTMAP_CS");
    started = SDL_GetPerformanceCounter();
    bool good = r->bake_pipeline && setup_lightmap(r, &tree, lm);
    if (good) bake_timing("surface lightmap", started);
    dm_probe_grid object_candidate = {0}, volume_candidate = {0};
    dm_beam_grid beam_candidate = {0};

    if (good) bake_progress(r, "object probes", 0u, 0u);
    started = SDL_GetPerformanceCounter();
    if (good) good = make_probe_grid(m, 2.0f, &object_candidate) &&
                     bake_probe_grid(r, &object_candidate, 1024u);
    if (good) bake_timing("object probes", started);

    if (good) bake_progress(r, "volume probes", 0u, 0u);
    started = SDL_GetPerformanceCounter();
    if (good) good = make_probe_grid(m, 4.0f, &volume_candidate) &&
                     bake_probe_grid(r, &volume_candidate, 1024u);
    if (good) bake_timing("volume probes", started);

    if (good) bake_progress(r, "sun visibility", 0u, 0u);
    started = SDL_GetPerformanceCounter();
    if (good) good = dm_beam_build(&beam_candidate, m, &tree, scene_sun_direction());
    if (good) bake_timing("sun visibility", started);
    if (good) SDL_Log("B: compressed sun beams into %u cells", beam_candidate.count);

    dm_cached_lightmap candidate = {0};
    SDL_GPUBuffer *volume_buffer = NULL;
    SDL_GPUBuffer *beam_buffer = NULL;
    if (good) {
        const size_t count = (size_t)volume_candidate.count_x *
            volume_candidate.count_y * volume_candidate.count_z;
        volume_buffer = upload_buffer(r, SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_READ,
                                      volume_candidate.probes, count * sizeof(dm_probe));
        good = volume_buffer != NULL;
    }
    if (good) {
        beam_buffer = upload_beams(r, &beam_candidate);
        good = beam_buffer != NULL;
    }
    if (good) {
        bake_progress(r, "saving cache", 0u, 0u);
        started = SDL_GetPerformanceCounter();

        candidate.object_probes = object_candidate;
        candidate.volume_probes = volume_candidate;
        candidate.beams = beam_candidate;
        good = download_current_lightmap(r, &candidate) &&
               dm_cache_write(path, scene_hash, layout_hash, &candidate);
        candidate.object_probes.probes = NULL;
        candidate.volume_probes.probes = NULL;
        candidate.beams.cells = NULL;
        candidate.beams.shadow_depth = NULL;
        if (good) bake_timing("saving cache", started);
    }
    dm_cache_free(&candidate);
    release_bake_buffers(r);
    if (r->lightmap_scratch) SDL_ReleaseGPUTexture(r->device, r->lightmap_scratch);
    r->lightmap_scratch = NULL;
    if (r->bake_pipeline) SDL_ReleaseGPUComputePipeline(r->device, r->bake_pipeline);
    r->bake_pipeline = NULL;
    bvh_free(&tree);
    if (good) {
        SDL_GPUBuffer *old_volume = r->volume_probe_buffer;
        SDL_GPUBuffer *old_beam = r->beam_buffer;
        r->volume_probe_buffer = volume_buffer;
        r->beam_buffer = beam_buffer;
        free_probe_grid(&r->object_probes);
        free_probe_grid(&r->volume_probes);
        r->object_probes = object_candidate;
        r->volume_probes = volume_candidate;
        dm_beam_free(&r->beams);
        r->beams = beam_candidate;
        r->has_bake = true;
        SDL_ReleaseGPUTexture(r->device, old);
        if (old_volume) SDL_ReleaseGPUBuffer(r->device, old_volume);
        if (old_beam) SDL_ReleaseGPUBuffer(r->device, old_beam);
    } else {
        if (volume_buffer) SDL_ReleaseGPUBuffer(r->device, volume_buffer);
        if (beam_buffer) SDL_ReleaseGPUBuffer(r->device, beam_buffer);
        dm_beam_free(&beam_candidate);
        free_probe_grid(&object_candidate);
        free_probe_grid(&volume_candidate);
        if (r->lightmap_texture) SDL_ReleaseGPUTexture(r->device, r->lightmap_texture);
        r->lightmap_texture = old;
        r->lightmap_width = old_width;
        r->lightmap_height = old_height;
        r->has_bake = had_bake;
    }
    return good;
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
    r->window = SDL_CreateWindow(title, width, height, SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY);
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

    if (SDL_GPUTextureSupportsFormat(r->device, SDL_GPU_TEXTUREFORMAT_D32_FLOAT, SDL_GPU_TEXTURETYPE_2D, SDL_GPU_TEXTUREUSAGE_DEPTH_STENCIL_TARGET))
        r->depth_format = SDL_GPU_TEXTUREFORMAT_D32_FLOAT;
    else if (SDL_GPUTextureSupportsFormat(r->device, SDL_GPU_TEXTUREFORMAT_D24_UNORM, SDL_GPU_TEXTURETYPE_2D, SDL_GPU_TEXTUREUSAGE_DEPTH_STENCIL_TARGET))
        r->depth_format = SDL_GPU_TEXTUREFORMAT_D24_UNORM;
    else r->depth_format = SDL_GPU_TEXTUREFORMAT_D16_UNORM;

    SDL_GPUShader *surface_vs = compile_graphics_shader(r, "shaders/vertex.hlsl", "surface_vs", "BUILD_SURFACE_VS", SDL_SHADERCROSS_SHADERSTAGE_VERTEX);
    SDL_GPUShader *surface_ps = compile_graphics_shader(r, "shaders/fragment.hlsl", "surface_fs", "BUILD_SURFACE_FS", SDL_SHADERCROSS_SHADERSTAGE_FRAGMENT);
    SDL_GPUShader *line_vs = compile_graphics_shader(r, "shaders/vertex.hlsl", "wireframe_vs", "BUILD_WIREFRAME_VS", SDL_SHADERCROSS_SHADERSTAGE_VERTEX);
    SDL_GPUShader *line_ps = compile_graphics_shader(r, "shaders/fragment.hlsl", "wireframe_fs", "BUILD_WIREFRAME_FS", SDL_SHADERCROSS_SHADERSTAGE_FRAGMENT);
    SDL_GPUShader *sky_vs = compile_graphics_shader(r, "shaders/vertex.hlsl", "fullscreen_vs", "BUILD_FULLSCREEN_VS", SDL_SHADERCROSS_SHADERSTAGE_VERTEX);
    SDL_GPUShader *sky_ps = compile_graphics_shader(r, "shaders/fragment.hlsl", "sky_fs", "BUILD_SKY_FS", SDL_SHADERCROSS_SHADERSTAGE_FRAGMENT);
    if (!surface_vs || !surface_ps || !line_vs || !line_ps || !sky_vs || !sky_ps) {

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

    if (!r->solid_pipeline || !r->line_pipeline || !r->sky_pipeline || !fx_init(&r->fx, r->device, r->window)) {

        r_deinit(r);
        return false;
    }

    SDL_Log("GPU backend: %s", SDL_GetGPUDeviceDriver(r->device));
    SDL_Log("depth format: %s", r->depth_format == SDL_GPU_TEXTUREFORMAT_D32_FLOAT ? "D32_FLOAT" : r->depth_format == SDL_GPU_TEXTUREFORMAT_D24_UNORM ? "D24_UNORM" : "D16_UNORM");
    SDL_Log("SDL_image: %d", IMG_Version());
    return true;
}

bool r_build_scene(renderer *r, const mesh *m, const gltf_scene *visual, const lightmap *lm) {

    if (!r || !m || !visual || !lm || !visual->vertex_count || visual->vertex_count % 3u || visual->vertex_count / 3u != m->faces.count || !visual->material_count ||
        !lm->uvs) {
        SDL_Log("render/lightmap geometry mismatch");
        return false;
    }

    r->target = m->bounds.center;
    r->scene_radius = fmaxf(m->bounds.extents.x, fmaxf(m->bounds.extents.y, m->bounds.extents.z));
    if (r->scene_radius < 1.0f) r->scene_radius = 1.0f;
    r->distance = r->scene_radius * 2.15f;

    if (!setup_materials(r, visual)) return false;
    r->draws = calloc(visual->material_count, sizeof(*r->draws));
    if (!r->draws) return false;

    r->vertex_count = 0;
    const size_t triangle_count = visual->vertex_count / 3u;
    for (uint32_t material = 0; material < visual->material_count; ++material) {

        const uint32_t first = r->vertex_count;
        for (size_t triangle = 0; triangle < triangle_count; ++triangle) {

            const gltf_vertex *v = &visual->vertices[triangle * 3u];
            if (v[0].material != material) continue;
            const lmap_uv *uv = &lm->uvs[triangle * 3u];

            if (!push_surface(r, &v[0], uv[0]) || !push_surface(r, &v[1], uv[1]) || !push_surface(r, &v[2], uv[2])) return false;
        
        }

        const uint32_t count = r->vertex_count - first;
        if (count) r->draws[r->draw_count++] = (draw_range){first, count, material};

    }

    r->debug_vertex_start = r->vertex_count;
    const color4 wire = {0.18f, 0.95f, 0.24f, 1.0f};
    for (size_t triangle = 0; triangle < triangle_count; ++triangle) {

        const gltf_vertex *v = &visual->vertices[triangle * 3u];
        if (!add_wire_triangle(r, v[0].position, v[1].position, v[2].position, wire)) return false;
    }

    r->debug_vertex_count = r->vertex_count - r->debug_vertex_start;

    if (!upload_vertices(r)) return false;
    r->lightmap_sampler = SDL_CreateGPUSampler(r->device, &(SDL_GPUSamplerCreateInfo){
        .min_filter = SDL_GPU_FILTER_LINEAR,
        .mag_filter = SDL_GPU_FILTER_LINEAR,
        .mipmap_mode = SDL_GPU_SAMPLERMIPMAPMODE_NEAREST,
        .address_mode_u = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE,
        .address_mode_v = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE,
        .address_mode_w = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE
    });
    /* Valid before any bake exists; the shader's unbaked branch uses fixed ambient light. */
    r->lightmap_texture = pixel_texture(r, 0, 0, 0, 255);
    if (!r->lightmap_sampler || !r->lightmap_texture) return false;
    r->has_bake = false;
    SDL_Log("materials: %u | material draw ranges: %u | embedded images: %u", r->material_count, r->draw_count, r->image_texture_count);
    return true;
}

void r_event(renderer *r, const SDL_Event *event) {

    if (!r || !event) return;

    switch (event->type) {

        case SDL_EVENT_MOUSE_BUTTON_DOWN:
            if (event->button.button == SDL_BUTTON_LEFT) r->dragging = true;
            break;

        case SDL_EVENT_MOUSE_BUTTON_UP:
            if (event->button.button == SDL_BUTTON_LEFT) r->dragging = false;
            break;

        case SDL_EVENT_MOUSE_MOTION:
            if (r->dragging) {
                r->yaw += event->motion.xrel * 0.0075f;
                r->pitch += event->motion.yrel * 0.0075f;
                if (r->pitch > 1.45f) r->pitch = 1.45f;
                if (r->pitch < -1.45f) r->pitch = -1.45f;
            }
            break;

        case SDL_EVENT_MOUSE_WHEEL:
            r->distance -= event->wheel.y * (r->distance * 0.08f);
            if (r->distance < r->scene_radius * 0.05f) r->distance = r->scene_radius * 0.05f;
            if (r->distance > r->scene_radius * 20.0f) r->distance = r->scene_radius * 20.0f;
            break;

        case SDL_EVENT_KEY_DOWN:
            if (!event->key.repeat && event->key.key == SDLK_TAB) r->show_debug = !r->show_debug;
            if (!event->key.repeat && event->key.key == SDLK_F5) r->show_volume = !r->show_volume;
            if (!event->key.repeat && (event->key.key == SDLK_F1 || event->key.key == SDLK_F3 || event->key.key == SDLK_F4)) {
                uint32_t view = (uint32_t)(event->key.key - SDLK_F1) + 1u;
                r->debug_view = r->debug_view == view ? 0u : view;
            }
            break;

        default:
            break;

    }

}

bool r_draw(renderer *r) {

    if (!r || !r->device || !r->solid_pipeline || !r->sky_pipeline || !r->vertex_buffer || !r->lightmap_texture || !r->lightmap_sampler) return false;

    // r->debug_view = 3u;
    // r->show_volume = true;

    SDL_GPUCommandBuffer *cmd = SDL_AcquireGPUCommandBuffer(r->device);
    if (!cmd) return false;
    SDL_GPUTexture *swap = NULL;
    Uint32 width = 0, height = 0;

    if (!SDL_WaitAndAcquireGPUSwapchainTexture(cmd, r->window, &swap, &width, &height)) {
        SDL_CancelGPUCommandBuffer(cmd);
        return false;
    }

    if (!swap || !width || !height) return SDL_SubmitGPUCommandBuffer(cmd);
    if (!fx_ensure(&r->fx, width, height) || !ensure_depth_texture(r, width, height)) {
        SDL_CancelGPUCommandBuffer(cmd);
        return false;
    }

    const float fov = 62.0f * 3.14159265358979323846f / 180.0f;
    const float cp = cosf(r->pitch);
    const vec3 eye = v3(r->target.x + r->distance * cp * cosf(r->yaw), r->target.y + r->distance * sinf(r->pitch), r->target.z + r->distance * cp * sinf(r->yaw));
    const vec3 forward = v3_normalize(v3_sub(r->target, eye));
    const vec3 right = v3_normalize(v3_cross(forward, v3(0, 1, 0)));
    const vec3 up = v3_cross(right, forward);
    const float aspect = (float)width / (float)height;
    const float tan_half = tanf(fov * 0.5f);
    const float znear = fmaxf(0.02f, r->scene_radius * 0.005f);
    const float zfar = fmaxf(100.0f, r->scene_radius * 10.0f);
    const mat4 view = m4_look_at(eye, r->target, v3(0, 1, 0));
    const mat4 proj = m4_perspective(fov, aspect, znear, zfar);
    const camera_uniforms camera = {
        .mvp = m4_mul(proj, view), 
        .view = view, 
    };

    const vec3 sun = scene_sun_direction();
    const sky_uniforms sky = {
        .camera_right = {right.x * tan_half * aspect, right.y * tan_half * aspect, right.z * tan_half * aspect, 0},
        .camera_up = {up.x * tan_half, up.y * tan_half, up.z * tan_half, 0},
        .camera_forward = {forward.x, forward.y, forward.z, 0},
        .sky_zenith = {0.22f, 0.42f, 0.78f, 1},
        .sky_horizon = {0.68f, 0.76f, 0.88f, 1},
        .sun_direction_intensity = {sun.x, sun.y, sun.z, 2.4f},
        .sun_color_radius = {1.00f, 0.94f, 0.84f, 0.00465f}
    };

    const SDL_GPUColorTargetInfo colors[2] = {
        {.texture = r->fx.hdr, .load_op = SDL_GPU_LOADOP_DONT_CARE, .store_op = SDL_GPU_STOREOP_STORE},
        {.texture = r->fx.normal_depth, .load_op = SDL_GPU_LOADOP_DONT_CARE, .store_op = SDL_GPU_STOREOP_STORE}
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

    SDL_BindGPUVertexBuffers(pass, 0, &(SDL_GPUBufferBinding){.buffer = r->vertex_buffer, .offset = 0}, 1);
    SDL_BindGPUGraphicsPipeline(pass, r->solid_pipeline);
    SDL_PushGPUVertexUniformData(cmd, 0, &camera, sizeof(camera));
    const SDL_GPUTextureSamplerBinding lightmap_binding = {.texture = r->lightmap_texture, .sampler = r->lightmap_sampler};
    SDL_BindGPUFragmentSamplers(pass, 5, &lightmap_binding, 1);

    for (uint32_t i = 0; i < r->draw_count; ++i) {

        const draw_range *draw = &r->draws[i];
        const gpu_material *m = &r->materials[draw->material];
        const SDL_GPUTextureSamplerBinding bindings[5] = {
            {.texture = m->base_color, .sampler = r->material_sampler},
            {.texture = m->metallic_roughness, .sampler = r->material_sampler},
            {.texture = m->normal, .sampler = r->material_sampler},
            {.texture = m->occlusion, .sampler = r->material_sampler},
            {.texture = m->emissive, .sampler = r->material_sampler}
        };

        SDL_BindGPUFragmentSamplers(pass, 0, bindings, 5);
        const material_uniforms material = {
            .base_color_factor = {m->data.base_color[0], m->data.base_color[1], m->data.base_color[2], m->data.base_color[3]},
            .emissive_metallic = {m->data.emissive[0], m->data.emissive[1], m->data.emissive[2], m->data.metallic},
            .roughness_normal_ao_sun = {m->data.roughness, m->data.normal_scale, m->data.occlusion_strength, 2.4f},
            .sun_direction = {sun.x, sun.y, sun.z, 0},
            .sun_color = {1.00f, 0.94f, 0.84f, 1},
            .camera_position = {eye.x, eye.y, eye.z,
                                r->debug_view==1u ? 2.0f : (r->has_bake ? 1.0f : 0.0f)},

            .camera_forward = {forward.x, forward.y, forward.z, 0}
        };

        SDL_PushGPUFragmentUniformData(cmd, 0, &material, sizeof(material));
        SDL_DrawGPUPrimitives(pass, draw->count, 1, draw->first, 0);

        // SDL_Log("BVH: %u nodes, %u triangles", tree.node_count, tree.triangle_count);
    
    }

    if (r->show_debug && r->debug_vertex_count) {
        SDL_BindGPUGraphicsPipeline(pass, r->line_pipeline);
        SDL_PushGPUVertexUniformData(cmd, 0, &camera.mvp, sizeof(camera.mvp));
        SDL_DrawGPUPrimitives(pass, r->debug_vertex_count, 1, r->debug_vertex_start, 0);
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

    if (/* false &&  */r->show_volume && r->has_bake &&
    (r->debug_view == 0u || r->debug_view >= 3u) &&
        r->volume_probe_buffer && r->beam_buffer &&
        !fx_volume(&r->fx, cmd, r->volume_probe_buffer, r->beam_buffer,
               &r->volume_probes, &r->beams,
               eye, right, up, forward, sun, tan_half, aspect)) {
    // if (r->show_volume && r->has_bake && (r->debug_view==0u || r->debug_view>=3u) &&
    //     r->volume_probe_buffer && r->beam_buffer &&
    //     !fx_volume(&r->fx, cmd, r->volume_probe_buffer, r->beam_buffer,
    //                &r->volume_probes, &r->beams,
    //                eye, right, up, forward, sun, tan_half, aspect)) {
        SDL_CancelGPUCommandBuffer(cmd);
        return false;
    }
    if (!fx_apply(&r->fx, cmd, swap, tan_half, aspect)) {
        SDL_CancelGPUCommandBuffer(cmd);
        return false;
    }

    return SDL_SubmitGPUCommandBuffer(cmd);
}

void r_deinit(renderer *r) {

    if (!r) return;

    free(r->vertices);
    free(r->materials);
    free(r->draws);
    free_probe_grid(&r->object_probes);
    free_probe_grid(&r->volume_probes);
    dm_beam_free(&r->beams);
    if (r->device) {

        SDL_WaitForGPUIdle(r->device);
        fx_deinit(&r->fx);
        if (r->image_textures) {
            for (uint32_t i = 0; i < r->image_texture_count; ++i)
                if (r->image_textures[i]) SDL_ReleaseGPUTexture(r->device, r->image_textures[i]);
        }

        free(r->image_textures);
        if (r->default_white) SDL_ReleaseGPUTexture(r->device, r->default_white);
        if (r->default_normal) SDL_ReleaseGPUTexture(r->device, r->default_normal);
        if (r->material_sampler) SDL_ReleaseGPUSampler(r->device, r->material_sampler);
        if (r->vertex_buffer) SDL_ReleaseGPUBuffer(r->device, r->vertex_buffer);
        if (r->bvh_node_buffer) SDL_ReleaseGPUBuffer(r->device, r->bvh_node_buffer);
        if (r->bvh_triangle_buffer) SDL_ReleaseGPUBuffer(r->device, r->bvh_triangle_buffer);
        if (r->lightmap_sample_buffer) SDL_ReleaseGPUBuffer(r->device, r->lightmap_sample_buffer);
        if (r->volume_probe_buffer) SDL_ReleaseGPUBuffer(r->device, r->volume_probe_buffer);
        if (r->beam_buffer) SDL_ReleaseGPUBuffer(r->device, r->beam_buffer);
        if (r->depth_texture) SDL_ReleaseGPUTexture(r->device, r->depth_texture);
        if (r->lightmap_texture) SDL_ReleaseGPUTexture(r->device, r->lightmap_texture);
        if (r->lightmap_scratch) SDL_ReleaseGPUTexture(r->device, r->lightmap_scratch);
        if (r->lightmap_sampler) SDL_ReleaseGPUSampler(r->device, r->lightmap_sampler);
        if (r->sky_pipeline) SDL_ReleaseGPUGraphicsPipeline(r->device, r->sky_pipeline);
        if (r->solid_pipeline) SDL_ReleaseGPUGraphicsPipeline(r->device, r->solid_pipeline);
        if (r->line_pipeline) SDL_ReleaseGPUGraphicsPipeline(r->device, r->line_pipeline);
        if (r->bake_pipeline) SDL_ReleaseGPUComputePipeline(r->device, r->bake_pipeline);
        if (r->window) SDL_ReleaseWindowFromGPUDevice(r->device, r->window);
        SDL_DestroyGPUDevice(r->device);

    }


    if (r->window) SDL_DestroyWindow(r->window);
    memset(r, 0, sizeof(*r));

    SDL_ShaderCross_Quit();
}
