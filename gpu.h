#ifndef GPU_H
#define GPU_H

#include <stdbool.h>

#include <SDL3/SDL.h>

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#pragma clang diagnostic ignored "-Wvariadic-macro-arguments-omitted"
#pragma clang diagnostic ignored "-Wstrict-prototypes"
#endif

#include <NRI.h>
#include <Extensions/NRIDeviceCreation.h>
#include <Extensions/NRIHelper.h>
#include <Extensions/NRISwapChain.h>

#if defined(__clang__)
#pragma clang diagnostic pop
#endif


typedef struct GPU {
    SDL_Window *window;

    NriDevice *device;
    NriCoreInterface core;
    NriHelperInterface helper;
    NriSwapChainInterface swapchain;

    NriQueue *graphics_queue;
    NriQueue *compute_queue;
    NriQueue *copy_queue;
} GPU;


bool gpu_init(GPU *gpu, const char *title, int width, int height);
void gpu_deinit(GPU *gpu);


#endif
