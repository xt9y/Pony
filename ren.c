#ifndef REN_H
#define REN_H

#include "sdl.h"

typedef struct Renderer {
    SDL_Window *win;
    SDL_GPUDevice *device;
    SDL_GPUTextureFormat swp_format;
    uint32_t w;
    uint32_t h;
} RENDERER;

int r_init(RENDERER *r) {
    if (!SDL_Init(SDL_INIT_VIDEO)) {
        SDL_Log("SDL_Init failed: %s", SDL_GetError());
        return 0;
    }

    r->win = SDL_CreateWindow("Untitled", r->w, r->h, SDL_WINDOW_RESIZABLE);
    if (!r->win) {
        SDL_Log("window creation failed: %s", SDL_GetError());
        return 0;
    }

    if (!SDL_ShaderCross_Init()) {
        SDL_Log("shadercross init failed: %s", SDL_GetError());
        return 0;
    }

    SDL_GPUShaderFormat formats = SDL_ShaderCross_GetHLSLShaderFormats();

    r->device = SDL_CreateGPUDevice(formats, true, NULL);
    if (!r->device) {
        SDL_Log("GPU device creation failed: %s", SDL_GetError());
        return 0;
    }

    if (!SDL_ClaimWindowForGPUDevice(r->device, r->win)) {
        SDL_Log("claim window failed: %s", SDL_GetError());
        return 0;
    }

    r->swp_format = SDL_GetGPUSwapchainTextureFormat(r->device, r->win);
    return 1;
}

void r_deinit(RENDERER *r) {
    SDL_ReleaseWindowFromGPUDevice(r->device, r->win);
    SDL_DestroyGPUDevice(r->device);
    SDL_DestroyWindow(r->win);
    SDL_Quit();
}

#endif // REN_H
