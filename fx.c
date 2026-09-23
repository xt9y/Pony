#include "dustmite.h"

#include <SDL3_shadercross/SDL_shadercross.h>

#include <string.h>

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

    if (!spirv) SDL_Log("shadercross failed for %s:%s: %s", path, entrypoint, SDL_GetError());

    return spirv;
}

static SDL_GPUShader *compile_shader(SDL_GPUDevice *device, const char *path, const char *entrypoint, const char *define, SDL_ShaderCross_ShaderStage stage) {

    size_t size = 0;
    Uint8 *spirv = compile_spirv(path, entrypoint, define, stage, &size);

    if (!spirv) return NULL;
    SDL_ShaderCross_GraphicsShaderMetadata *meta = SDL_ShaderCross_ReflectGraphicsSPIRV(spirv, size, 0);

    if (!meta) {
        SDL_free(spirv);
        return NULL;
    }

    const SDL_ShaderCross_SPIRV_Info info = {.bytecode = spirv, .bytecode_size = size, .entrypoint = entrypoint, .shader_stage = stage, .props = 0};
    SDL_GPUShader *shader = SDL_ShaderCross_CompileGraphicsShaderFromSPIRV(device, &info, &meta->resource_info, 0);

    SDL_free(meta);
    SDL_free(spirv);

    return shader;
}

static SDL_GPUComputePipeline *compile_compute(SDL_GPUDevice *device, const char *path, const char *entrypoint, const char *define) {

    size_t size = 0;
    Uint8 *spirv = compile_spirv(path, entrypoint, define, SDL_SHADERCROSS_SHADERSTAGE_COMPUTE, &size);

    if (!spirv) return NULL;
    SDL_ShaderCross_ComputePipelineMetadata *meta = SDL_ShaderCross_ReflectComputeSPIRV(spirv, size, 0);

    if (!meta) {
        SDL_free(spirv);
        return NULL;
    }


    const SDL_ShaderCross_SPIRV_Info info = {.bytecode = spirv, .bytecode_size = size, .entrypoint = entrypoint, .shader_stage = SDL_SHADERCROSS_SHADERSTAGE_COMPUTE, .props = 0};

    SDL_GPUComputePipeline *pipeline = SDL_ShaderCross_CompileComputePipelineFromSPIRV(device, &info, meta, 0);

    SDL_free(meta);
    SDL_free(spirv);

    return pipeline;
}

static SDL_GPUTexture *texture(SDL_GPUDevice *device, SDL_GPUTextureFormat format, SDL_GPUTextureUsageFlags usage, Uint32 width, Uint32 height) {

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

static bool dispatch_one(fx_state *fx, SDL_GPUCommandBuffer *cmd, SDL_GPUComputePipeline *pipeline, SDL_GPUTexture *source, SDL_GPUTexture *destination, const void *uniforms,
                         Uint32 uniform_size, Uint32 width, Uint32 height) {

    const SDL_GPUStorageTextureReadWriteBinding output = {.texture = destination, .mip_level = 0, .layer = 0, .cycle = false};


    SDL_GPUComputePass *pass = SDL_BeginGPUComputePass(cmd, &output, 1, NULL, 0);
    if (!pass) return false;
    SDL_BindGPUComputePipeline(pass, pipeline);


    if (source) {
        const SDL_GPUTextureSamplerBinding input = {.texture = source, .sampler = fx->sampler};
        SDL_BindGPUComputeSamplers(pass, 0, &input, 1);
    }

    if (uniforms && uniform_size) {
        SDL_PushGPUComputeUniformData(cmd, 0, uniforms, uniform_size);
    }


    SDL_DispatchGPUCompute(pass, (width + 7u) / 8u, (height + 7u) / 8u, 1);
    SDL_EndGPUComputePass(pass);

    return true;
}

static void release_frame_textures(fx_state *fx) {

    if (fx->hdr) SDL_ReleaseGPUTexture(fx->device, fx->hdr);
    if (fx->normal_depth) SDL_ReleaseGPUTexture(fx->device, fx->normal_depth);
    if (fx->ao) SDL_ReleaseGPUTexture(fx->device, fx->ao);
    if (fx->bloom_a) SDL_ReleaseGPUTexture(fx->device, fx->bloom_a);
    if (fx->bloom_b) SDL_ReleaseGPUTexture(fx->device, fx->bloom_b);
    if (fx->volume) SDL_ReleaseGPUTexture(fx->device, fx->volume);
    if (fx->lit) SDL_ReleaseGPUTexture(fx->device, fx->lit);

    fx->hdr = fx->normal_depth = fx->ao = fx->bloom_a = fx->bloom_b = fx->volume = fx->lit = NULL;
    fx->width = fx->height = fx->ao_width = fx->ao_height = 0;
    fx->volume_ready = false;
}

bool fx_init(fx_state *fx, SDL_GPUDevice *device, SDL_Window *window) {

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

    SDL_GPUShader *vs = compile_shader(device, "shaders/vertex.hlsl", "fullscreen_vs", "BUILD_FULLSCREEN_VS", SDL_SHADERCROSS_SHADERSTAGE_VERTEX);
    SDL_GPUShader *ps = compile_shader(device, "shaders/fragment.hlsl", "compose_fs", "BUILD_COMPOSE_FS", SDL_SHADERCROSS_SHADERSTAGE_FRAGMENT);
    fx->ssao_pipeline = compile_compute(device, "shaders/compute.hlsl", "ssao_cs", "BUILD_SSAO_CS");
    fx->bloom_pipeline = compile_compute(device, "shaders/compute.hlsl", "bloom_cs", "BUILD_BLOOM_CS");
    fx->grade_pipeline = compile_compute(device, "shaders/compute.hlsl", "grade_cs", "BUILD_GRADE_CS");
    fx->volume_pipeline = compile_compute(device, "shaders/compute.hlsl", "volume_cs", "BUILD_VOLUME_CS");
    fx->volume_compose_pipeline = compile_compute(device, "shaders/compute.hlsl", "volume_compose_cs", "BUILD_VOLUME_COMPOSE_CS");

    if (!vs || !ps || !fx->ssao_pipeline || !fx->bloom_pipeline || !fx->grade_pipeline ||
        !fx->volume_pipeline || !fx->volume_compose_pipeline) {

        if (vs) SDL_ReleaseGPUShader(device, vs);
        if (ps) SDL_ReleaseGPUShader(device, ps);
        goto fail;
    }


    const SDL_GPUColorTargetDescription target = {.format = SDL_GetGPUSwapchainTextureFormat(device, window)};
    fx->compose_pipeline = SDL_CreateGPUGraphicsPipeline(
        device, &(SDL_GPUGraphicsPipelineCreateInfo){
                    .vertex_shader = vs,
                    .fragment_shader = ps,
                    .primitive_type = SDL_GPU_PRIMITIVETYPE_TRIANGLELIST,
                    .rasterizer_state =
                        {.fill_mode = SDL_GPU_FILLMODE_FILL, .cull_mode = SDL_GPU_CULLMODE_NONE, .front_face = SDL_GPU_FRONTFACE_COUNTER_CLOCKWISE, .enable_depth_clip = true},
                    .target_info = {.color_target_descriptions = &target, .num_color_targets = 1}
                });
    SDL_ReleaseGPUShader(device, vs);
    SDL_ReleaseGPUShader(device, ps);
    if (!fx->compose_pipeline) goto fail;

    fx->lut = texture(device, SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT, SDL_GPU_TEXTUREUSAGE_SAMPLER | SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_WRITE, 256u, 16u);
    if (!fx->lut) goto fail;

    SDL_GPUCommandBuffer *cmd = SDL_AcquireGPUCommandBuffer(device);
    if (!cmd || !dispatch_one(fx, cmd, fx->grade_pipeline, NULL, fx->lut, NULL, 0, 256u, 16u)) {
        if (cmd) SDL_CancelGPUCommandBuffer(cmd);
        goto fail;
    }
    if (!SDL_SubmitGPUCommandBuffer(cmd)) goto fail;
    return true;


fail:
    fx_deinit(fx);
    return false;
}

bool fx_ensure(fx_state *fx, Uint32 width, Uint32 height) {

    if (!fx || !width || !height) return false;
    if (fx->hdr && fx->width == width && fx->height == height) return true;


    release_frame_textures(fx);

    fx->width = width;
    fx->height = height;
    fx->ao_width = (width + 1u) / 2u;
    fx->ao_height = (height + 1u) / 2u;

    const SDL_GPUTextureUsageFlags rt = SDL_GPU_TEXTUREUSAGE_COLOR_TARGET | SDL_GPU_TEXTUREUSAGE_SAMPLER;
    const SDL_GPUTextureUsageFlags compute = SDL_GPU_TEXTUREUSAGE_SAMPLER | SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_WRITE;


    fx->hdr = texture(fx->device, SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT, rt, width, height);
    fx->normal_depth = texture(fx->device, SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT, rt, width, height);
    fx->ao = texture(fx->device, SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT, compute, fx->ao_width, fx->ao_height);
    fx->bloom_a = texture(fx->device, SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT, compute, fx->ao_width, fx->ao_height);
    fx->bloom_b = texture(fx->device, SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT, compute, fx->ao_width, fx->ao_height);
    fx->volume = texture(fx->device, SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT, compute, fx->ao_width, fx->ao_height);
    fx->lit = texture(fx->device, SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT, compute, width, height);

    if (!fx->hdr || !fx->normal_depth || !fx->ao || !fx->bloom_a || !fx->bloom_b || !fx->volume || !fx->lit) {
        release_frame_textures(fx);
        return false;
    }

    return true;
}

bool fx_volume(fx_state *fx, SDL_GPUCommandBuffer *cmd, SDL_GPUBuffer *probes,
               SDL_GPUBuffer *beams, const dm_probe_grid *grid,
               const dm_beam_grid *beam_grid, vec3 eye, vec3 right, vec3 up,
               vec3 forward, vec3 sun, float tan_half_fov, float aspect) {
    if (!fx || !cmd || !probes || !beams || !grid || !beam_grid ||
        !grid->probes || !fx->volume) return false;
    const SDL_GPUStorageTextureReadWriteBinding target = {.texture = fx->volume};
    SDL_GPUComputePass *pass = SDL_BeginGPUComputePass(cmd, &target, 1, NULL, 0);
    if (!pass) return false;
    SDL_BindGPUComputePipeline(pass, fx->volume_pipeline);
    const SDL_GPUTextureSamplerBinding source = {.texture = fx->normal_depth, .sampler = fx->depth_sampler};
    SDL_BindGPUComputeSamplers(pass, 0, &source, 1);
    SDL_GPUBuffer *storage[2]={probes,beams};
    SDL_BindGPUComputeStorageBuffers(pass, 0, storage, 2);
    const volume_uniforms u = {
        .eye_density = {eye.x, eye.y, eye.z, 0.045f},
        .right_tan = {right.x*tan_half_fov*aspect, right.y*tan_half_fov*aspect, right.z*tan_half_fov*aspect, 0},
        .up_tan = {up.x*tan_half_fov, up.y*tan_half_fov, up.z*tan_half_fov, 0},
        .forward_g = {forward.x, forward.y, forward.z, 0.55f},
        .sun_intensity = {sun.x, sun.y, sun.z, 2.4f},
        .grid_origin_spacing = {grid->origin.x, grid->origin.y, grid->origin.z, grid->spacing},
        .grid_dims_width = {grid->count_x, grid->count_y, grid->count_z, fx->ao_width},
        .height_debug = {fx->ao_height, fx->debug_view, beam_grid->depth, 0},
        .beam_origin = {beam_grid->origin.x,beam_grid->origin.y,beam_grid->origin.z,0},
        .beam_step = {beam_grid->step.x,beam_grid->step.y,beam_grid->step.z,0}
    };
    SDL_PushGPUComputeUniformData(cmd, 0, &u, sizeof(u));
    SDL_DispatchGPUCompute(pass, (fx->ao_width+7u)/8u, (fx->ao_height+7u)/8u, 1);
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
    SDL_DispatchGPUCompute(pass, (fx->width+7u)/8u, (fx->height+7u)/8u, 1);
    SDL_EndGPUComputePass(pass);
    fx->volume_ready = true;
    return true;
}

static bool bloom_pass(fx_state *fx, SDL_GPUCommandBuffer *cmd, SDL_GPUTexture *source, SDL_GPUTexture *destination, Uint32 src_width, Uint32 src_height, Uint32 phase) {

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
    return dispatch_one(fx, cmd, fx->bloom_pipeline, source, destination, &u, sizeof(u), fx->ao_width, fx->ao_height);
}

bool fx_apply(fx_state *fx, SDL_GPUCommandBuffer *cmd, SDL_GPUTexture *swap, float tan_half_fov, float aspect) {

    if (!fx || !cmd || !swap || !fx->hdr || !fx->normal_depth) {
        return false;
    }


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
    SDL_GPUTexture *hdr = fx->volume_ready ? fx->lit : fx->hdr;
    if (!dispatch_one(fx, cmd, fx->ssao_pipeline, fx->normal_depth, fx->ao, &ao, sizeof(ao), fx->ao_width, fx->ao_height) ||
        !bloom_pass(fx, cmd, fx->hdr, fx->bloom_a, fx->width, fx->height, 0u) || !bloom_pass(fx, cmd, fx->bloom_a, fx->bloom_b, fx->ao_width, fx->ao_height, 1u) ||
        !bloom_pass(fx, cmd, fx->bloom_b, fx->bloom_a, fx->ao_width, fx->ao_height, 2u)) {
        return false;
    }


    const SDL_GPUColorTargetInfo color = {.texture = swap, .load_op = SDL_GPU_LOADOP_DONT_CARE, .store_op = SDL_GPU_STOREOP_STORE};
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
        .exposure = -1.0f,
        .ao_strength = 0.0f,
        .bloom_strength = 0.0f
    };
    // const compose_uniforms u = {
    //      .exposure = 1.0f, 
    //      .ao_strength = fx->debug_view >= 3u ? 0.0f : 0.62f,
    //      .bloom_strength = fx->debug_view >= 3u ? 0.0f : 0.22f
    //  };
    SDL_PushGPUFragmentUniformData(cmd, 0, &u, sizeof(u));
    SDL_DrawGPUPrimitives(pass, 3, 1, 0, 0);
    SDL_EndGPURenderPass(pass);


    return true;
}

void fx_deinit(fx_state *fx) {

    if (!fx) return;
    if (fx->device) {

        release_frame_textures(fx);
        if (fx->lut) SDL_ReleaseGPUTexture(fx->device, fx->lut);
        if (fx->sampler) SDL_ReleaseGPUSampler(fx->device, fx->sampler);
        if (fx->depth_sampler) SDL_ReleaseGPUSampler(fx->device, fx->depth_sampler);
        if (fx->compose_pipeline) SDL_ReleaseGPUGraphicsPipeline(fx->device, fx->compose_pipeline);
        if (fx->ssao_pipeline) SDL_ReleaseGPUComputePipeline(fx->device, fx->ssao_pipeline);
        if (fx->bloom_pipeline) SDL_ReleaseGPUComputePipeline(fx->device, fx->bloom_pipeline);
        if (fx->grade_pipeline) SDL_ReleaseGPUComputePipeline(fx->device, fx->grade_pipeline);
        if (fx->volume_pipeline) SDL_ReleaseGPUComputePipeline(fx->device, fx->volume_pipeline);
        if (fx->volume_compose_pipeline) SDL_ReleaseGPUComputePipeline(fx->device, fx->volume_compose_pipeline);
    }

    memset(fx, 0, sizeof(*fx));
}
