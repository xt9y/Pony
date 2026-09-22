#include <SDL3/SDL.h>
#include <SDL3/SDL_gpu.h>
#include <SDL3/SDL_main.h>

int main() {
    if (!SDL_Init(SDL_INIT_VIDEO)) {
        SDL_Log("SDL_Init failed: %s", SDL_GetError());
        return 1;
    }

    SDL_Window *window =
        SDL_CreateWindow("SDL_GPU", 1280, 720, SDL_WINDOW_RESIZABLE);

    if (!window) {
        SDL_Log("SDL_CreateWindow failed: %s", SDL_GetError());
        SDL_Quit();
        return 1;
    }

    SDL_GPUDevice *gpu = SDL_CreateGPUDevice(SDL_GPU_SHADERFORMAT_SPIRV |
                                                 SDL_GPU_SHADERFORMAT_DXIL |
                                                 SDL_GPU_SHADERFORMAT_MSL,
                                             false, NULL);

    if (!gpu) {
        SDL_Log("SDL_CreateGPUDevice failed: %s", SDL_GetError());
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }

    if (!SDL_ClaimWindowForGPUDevice(gpu, window)) {
        SDL_Log("SDL_ClaimWindowForGPUDevice failed: %s", SDL_GetError());
        SDL_DestroyGPUDevice(gpu);
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }

    bool running = true;

    while (running) {
        SDL_Event event;

        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_EVENT_QUIT) {
                running = false;
            }
        }
    }

    SDL_ReleaseWindowFromGPUDevice(gpu, window);
    SDL_DestroyGPUDevice(gpu);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}
