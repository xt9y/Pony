#define fx_init fx_base_init
#define fx_ensure fx_base_ensure
#define fx_volume fx_base_volume
#define fx_apply fx_base_apply
#define fx_deinit fx_base_deinit
#include "fx.c"
#undef fx_init
#undef fx_ensure
#undef fx_volume
#undef fx_apply
#undef fx_deinit

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

    /* The fourth word is unused by the original compose shader.  The vision
     * shader uses it as a bypass-volume flag so DOF/peripheral reconstruction
     * also runs when volumetric fog is disabled. */
    const Uint32 dimensions[4] = {fx->width, fx->height, fx->debug_view, 1u};
    SDL_PushGPUComputeUniformData(cmd, 0, dimensions, sizeof(dimensions));
    SDL_DispatchGPUCompute(pass, (fx->width + 7u) / 8u, (fx->height + 7u) / 8u, 1);
    SDL_EndGPUComputePass(pass);
    return true;
}

bool fx_init(fx_state *fx, SDL_GPUDevice *device, SDL_Window *window) {
    if (!fx_base_init(fx, device, window)) return false;

    SDL_GPUComputePipeline *volume = compile_compute(
        device, "shaders/vision_compute.hlsl", "volume_cs", "BUILD_VISION_VOLUME_CS");
    SDL_GPUComputePipeline *compose = compile_compute(
        device, "shaders/vision_compute.hlsl", "volume_compose_cs", "BUILD_VISION_COMPOSE_CS");

    if (!volume || !compose) {
        if (volume) SDL_ReleaseGPUComputePipeline(device, volume);
        if (compose) SDL_ReleaseGPUComputePipeline(device, compose);
        fx_base_deinit(fx);
        return false;
    }

    SDL_ReleaseGPUComputePipeline(device, fx->volume_pipeline);
    SDL_ReleaseGPUComputePipeline(device, fx->volume_compose_pipeline);
    fx->volume_pipeline = volume;
    fx->volume_compose_pipeline = compose;
    return true;
}

bool fx_ensure(fx_state *fx, Uint32 width, Uint32 height) {
    return fx_base_ensure(fx, width, height);
}

bool fx_volume(fx_state *fx, SDL_GPUCommandBuffer *cmd, SDL_GPUBuffer *probes,
               SDL_GPUBuffer *beams, const dm_probe_grid *grid,
               const dm_beam_grid *beam_grid, vec3 eye, vec3 right, vec3 up,
               vec3 forward, vec3 sun, float tan_half_fov, float aspect) {
    return fx_base_volume(fx, cmd, probes, beams, grid, beam_grid, eye, right, up,
                          forward, sun, tan_half_fov, aspect);
}

bool fx_apply(fx_state *fx, SDL_GPUCommandBuffer *cmd, SDL_GPUTexture *swap,
              float tan_half_fov, float aspect) {
    if (!fx || !cmd || !swap) return false;

    const bool had_volume = fx->volume_ready;
    if (!had_volume) {
        if (!run_vision_only(fx, cmd)) return false;
        fx->volume_ready = true;
    }

    const bool ok = fx_base_apply(fx, cmd, swap, tan_half_fov, aspect);
    if (!had_volume) fx->volume_ready = false;
    return ok;
}

void fx_deinit(fx_state *fx) {
    fx_base_deinit(fx);
}
