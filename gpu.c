#include "dustmite.h"

#include <SDL3_image/SDL_image.h>
#include <SDL3_shadercross/SDL_shadercross.h>

#include <stddef.h>
#include <string.h>

/*
 * GPU backend selection is intentionally manual.
 *
 * SDL_GPU is the active implementation. When adding NRI, keep its native
 * calls directly beside the SDL_GPU calls and comment/uncomment the backend
 * you want. There is deliberately no wrapper API, backend enum, dispatch
 * table, compile-time selector, or runtime selector here.
 */

static Uint8 *compile_spirv(const char *path, const char *entrypoint,
                            const char *define,
                            SDL_ShaderCross_ShaderStage stage, size_t *size) {
    size_t source_size = 0;
    char *source = SDL_LoadFile(path, &source_size);
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
    if (!spirv)
        SDL_Log("shadercross HLSL->SPIR-V failed for %s:%s: %s",
                path, entrypoint, SDL_GetError());
    return spirv;
}

static SDL_GPUShader *compile_graphics_shader(renderer *r, const char *path,
                                               const char *entrypoint,
                                               const char *define,
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
        r->device, &info, &meta->resource_info, 0);

    SDL_free(meta);
    SDL_free(spirv);
    return shader;
}

static SDL_GPUGraphicsPipeline *make_surface_pipeline(renderer *r,
                                                       SDL_GPUShader *vs,
                                                       SDL_GPUShader *ps) {
    const SDL_GPUVertexBufferDescription vb = {
        .slot = 0,
        .pitch = (Uint32)sizeof(render_vertex),
        .input_rate = SDL_GPU_VERTEXINPUTRATE_VERTEX
    };
    const SDL_GPUVertexAttribute attrs[4] = {
        {.location = 0, .buffer_slot = 0, .format = SDL_GPU_VERTEXELEMENTFORMAT_FLOAT3,
         .offset = (Uint32)offsetof(render_vertex, x)},
        {.location = 1, .buffer_slot = 0, .format = SDL_GPU_VERTEXELEMENTFORMAT_FLOAT3,
         .offset = (Uint32)offsetof(render_vertex, nx)},
        {.location = 2, .buffer_slot = 0, .format = SDL_GPU_VERTEXELEMENTFORMAT_FLOAT2,
         .offset = (Uint32)offsetof(render_vertex, u)},
        {.location = 3, .buffer_slot = 0, .format = SDL_GPU_VERTEXELEMENTFORMAT_FLOAT2,
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

static SDL_GPUGraphicsPipeline *make_line_pipeline(renderer *r,
                                                    SDL_GPUShader *vs,
                                                    SDL_GPUShader *ps) {
    const SDL_GPUVertexBufferDescription vb = {
        .slot = 0,
        .pitch = (Uint32)sizeof(render_vertex),
        .input_rate = SDL_GPU_VERTEXINPUTRATE_VERTEX
    };
    const SDL_GPUVertexAttribute attrs[2] = {
        {.location = 0, .buffer_slot = 0, .format = SDL_GPU_VERTEXELEMENTFORMAT_FLOAT3,
         .offset = (Uint32)offsetof(render_vertex, x)},
        {.location = 1, .buffer_slot = 0, .format = SDL_GPU_VERTEXELEMENTFORMAT_FLOAT4,
         .offset = (Uint32)offsetof(render_vertex, r)}
    };
    const SDL_GPUColorTargetDescription targets[2] = {
        {.format = SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT},
        {.format = SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT,
         .blend_state = {.color_write_mask = 0, .enable_color_write_mask = true}}
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

static SDL_GPUGraphicsPipeline *make_sky_pipeline(renderer *r,
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

    /* SDL_GPU */
    const SDL_GPUShaderFormat formats = SDL_ShaderCross_GetSPIRVShaderFormats();
    r->device = SDL_CreateGPUDevice(formats, true, NULL);
    if (!r->device || !SDL_ClaimWindowForGPUDevice(r->device, r->window)) {
        SDL_Log("GPU initialization failed: %s", SDL_GetError());
        r_deinit(r);
        return false;
    }

    /* NRI: put the native NRI device/swapchain setup here and comment SDL above. */

    if (SDL_GPUTextureSupportsFormat(r->device, SDL_GPU_TEXTUREFORMAT_D32_FLOAT,
                                     SDL_GPU_TEXTURETYPE_2D,
                                     SDL_GPU_TEXTUREUSAGE_DEPTH_STENCIL_TARGET))
        r->depth_format = SDL_GPU_TEXTUREFORMAT_D32_FLOAT;
    else if (SDL_GPUTextureSupportsFormat(r->device, SDL_GPU_TEXTUREFORMAT_D24_UNORM,
                                          SDL_GPU_TEXTURETYPE_2D,
                                          SDL_GPU_TEXTUREUSAGE_DEPTH_STENCIL_TARGET))
        r->depth_format = SDL_GPU_TEXTUREFORMAT_D24_UNORM;
    else
        r->depth_format = SDL_GPU_TEXTUREFORMAT_D16_UNORM;

    SDL_GPUShader *surface_vs = compile_graphics_shader(
        r, "shaders/vertex.hlsl", "surface_vs", "BUILD_SURFACE_VS",
        SDL_SHADERCROSS_SHADERSTAGE_VERTEX);
    SDL_GPUShader *surface_ps = compile_graphics_shader(
        r, "shaders/fragment.hlsl", "surface_fs", "BUILD_SURFACE_FS",
        SDL_SHADERCROSS_SHADERSTAGE_FRAGMENT);
    SDL_GPUShader *line_vs = compile_graphics_shader(
        r, "shaders/vertex.hlsl", "wireframe_vs", "BUILD_WIREFRAME_VS",
        SDL_SHADERCROSS_SHADERSTAGE_VERTEX);
    SDL_GPUShader *line_ps = compile_graphics_shader(
        r, "shaders/fragment.hlsl", "wireframe_fs", "BUILD_WIREFRAME_FS",
        SDL_SHADERCROSS_SHADERSTAGE_FRAGMENT);
    SDL_GPUShader *sky_vs = compile_graphics_shader(
        r, "shaders/vertex.hlsl", "fullscreen_vs", "BUILD_FULLSCREEN_VS",
        SDL_SHADERCROSS_SHADERSTAGE_VERTEX);
    SDL_GPUShader *sky_ps = compile_graphics_shader(
        r, "shaders/fragment.hlsl", "sky_fs", "BUILD_SKY_FS",
        SDL_SHADERCROSS_SHADERSTAGE_FRAGMENT);

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
