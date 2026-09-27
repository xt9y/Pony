#ifndef GPU_H
#define GPU_H

#include <stdbool.h>

#include <SDL3/SDL.h>
#include <stdint.h>

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

typedef struct SWAPCHAIN_FRAME SWAPCHAIN_FRAME;
typedef struct FRAME_CONTEXT FRAME_CONTEXT;

typedef struct GPU {
    SDL_Window *window;

    NriDevice *device;
    NriCoreInterface core;
    NriHelperInterface helper;
    NriSwapChainInterface swapchain_api;

    NriSwapChain *swapchain;
    NriTexture **swapchain_textures;
    SWAPCHAIN_FRAME *swapchain_frames;

    uint32_t texture_state_num;
    uint32_t swapchain_texture_count;
    uint32_t swapchain_width;
    uint32_t swapchain_height;

    NriFormat swapchain_format;

    FRAME_CONTEXT *frame_contexts;
    NriFence *frame_fence;

    uint32_t frame_index;

    NriQueue *graphics_queue;
    NriQueue *compute_queue;
    NriQueue *copy_queue;
} GPU;


bool gpu_init(GPU *gpu, const char *title, int width, int height);
void gpu_deinit(GPU *gpu);

bool gpu_begin_frame(GPU *gpu, NriCommandBuffer **command_buffer, NriTexture **swapchain_texture, uint32_t *swapchain_index);
bool gpu_end_frame(GPU *gpu, NriCommandBuffer *command_buffer, uint32_t *swapchain_index);

bool gpu_resize(GPU *gpu);


#endif
