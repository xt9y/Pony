#ifndef GPU_H
#define GPU_H

#include <stdbool.h>
#include <stdint.h>

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

typedef struct SWAPCHAIN_FRAME SWAPCHAIN_FRAME;
typedef struct FRAME_CONTEXT FRAME_CONTEXT;

typedef struct GPU {
    SDL_Window *window;

#if defined(__APPLE__)
    SDL_MetalView metal_view;
#endif

    NriDevice *device;
    NriCoreInterface core;
    NriHelperInterface helper;
    NriSwapChainInterface swapchain_api;

    NriQueue *graphics_queue;
    NriQueue *compute_queue;
    NriQueue *copy_queue;

    NriSwapChain *swapchain;
    NriTexture **swapchain_textures;
    SWAPCHAIN_FRAME *swapchain_frames;

    uint32_t swapchain_texture_count;
    uint32_t swapchain_width;
    uint32_t swapchain_height;
    NriFormat swapchain_format;

    FRAME_CONTEXT *frame_contexts;
    NriFence *frame_fence;
    uint64_t frame_index;
} GPU;

bool gpu_init(GPU *gpu, const char *title, int width, int height);
void gpu_deinit(GPU *gpu);

bool gpu_begin_frame(
    GPU *gpu,
    NriCommandBuffer **command_buffer,
    NriTexture **swapchain_texture,
    uint32_t *swapchain_index
);

bool gpu_clear_frame(
    GPU *gpu,
    NriCommandBuffer *command_buffer,
    uint32_t swapchain_index,
    NriColor32f clear_color
);

bool gpu_end_frame(
    GPU *gpu,
    NriCommandBuffer *command_buffer,
    uint32_t swapchain_index
);

bool gpu_resize(GPU *gpu);

bool gpu_create_buffer(
    GPU *gpu,
    const NriBufferDesc *desc,
    NriMemoryLocation memory,
    NriBuffer **buffer
);

bool gpu_create_texture(
    GPU *gpu,
    const NriTextureDesc *desc,
    NriMemoryLocation memory,
    NriTexture **texture
);

bool gpu_upload_buffer(
    GPU *gpu,
    NriBuffer *buffer,
    const void *data,
    NriAccessStage after
);

bool gpu_upload_texture(
    GPU *gpu,
    NriTexture *texture,
    const NriTextureSubresourceUploadDesc *subresources,
    NriPlaneBits planes,
    NriAccessLayoutStage after
);

void gpu_destroy_buffer(GPU *gpu, NriBuffer *buffer);
void gpu_destroy_texture(GPU *gpu, NriTexture *texture);

#endif
