#include "game.h"

#include <string.h>


bool renderer_init(RENDERER *renderer, GPU *gpu) {
    if (!renderer || !gpu || !gpu->device) return false;

    memset(renderer, 0, sizeof(*renderer));

    renderer->gpu = gpu;

    renderer->camera =
        (CAMERA){
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
    if (!renderer || !renderer->gpu || !renderer->gpu->device || !renderer->scene) {
        return false;
    }

    ++renderer->frame_index;

    return true;
}
