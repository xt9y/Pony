#include "game.h"

#include <string.h>

bool renderer_init(RENDERER *renderer, GPU *gpu) {
    if (!renderer || !gpu || !gpu->device) return false;

    memset(renderer, 0, sizeof(*renderer));

    renderer->gpu = gpu;
    renderer->camera = (CAMERA){
        .position = {0.0f, 1.0f, 5.0f},
        .forward = {0.0f, 0.0f, -1.0f},
        .up = {0.0f, 1.0f, 0.0f},
        .fov_y = 62.0f,
        .near_plane = 0.05f,
        .far_plane = 1000.0f
    };

    return true;
}

void renderer_deinit(RENDERER *renderer) {
    if (!renderer) return;

    memset(renderer, 0, sizeof(*renderer));
}

bool renderer_set_scene(RENDERER *renderer, SCENE *scene) {
    if (!renderer || !renderer->gpu || !scene) return false;

    renderer->scene = scene;

    return true;
}

void renderer_event(RENDERER *renderer, const SDL_Event *event) {
    (void)renderer;
    (void)event;
}

bool renderer_frame(RENDERER *renderer) {
    if (!renderer || !renderer->gpu || !renderer->gpu->device || !renderer->scene) return false;

    NriCommandBuffer *command_buffer = NULL;
    NriTexture *swapchain_texture = NULL;
    uint32_t swapchain_index = 0;

    if (!gpu_begin_frame(renderer->gpu, &command_buffer, &swapchain_texture, &swapchain_index)) return false;

    if (!command_buffer || !swapchain_texture) return true;

    const NriColor32f clear_color = {
        .x = 0.025f,
        .y = 0.035f,
        .z = 0.055f,
        .w = 1.0f
    };

    if (!gpu_clear_frame(renderer->gpu, command_buffer, swapchain_index, clear_color) ||
        !gpu_end_frame(renderer->gpu, command_buffer, swapchain_index)) {
        return false;
    }

    renderer->frame_index = renderer->gpu->frame_index;

    return true;
}
