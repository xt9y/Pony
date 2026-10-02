#ifndef GPU_H
#define GPU_H

#include <SDL3/SDL.h>

#include <stdbool.h>
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

typedef struct SWAPCHAIN_TEXTURE SWAPCHAIN_TEXTURE;
typedef struct FRAME_CONTEXT FRAME_CONTEXT;
typedef struct UPLOAD_CONTEXT UPLOAD_CONTEXT;
typedef struct TEXTURE_STATE TEXTURE_STATE;

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
    NriQueue *work_queue;

    NriSwapChain *swapchain;
    NriDescriptorPool *descriptor_pool;
    NriFence *frame_fence;
    NriFence *work_fence;

    FRAME_CONTEXT *frame_contexts;
    FRAME_CONTEXT *work_contexts;
    FRAME_CONTEXT *active_frame;
    FRAME_CONTEXT *active_work;
    uint64_t work_index;
    uint64_t work_next_fence;
    UPLOAD_CONTEXT *upload;

    SWAPCHAIN_TEXTURE *swapchain_frames;
    NriTexture **swapchain_textures;
    uint32_t swapchain_texture_count;

    NriPipelineCache *pipeline_cache;
    NriQueryPool *timestamp_pool;
    NriBuffer *timestamp_readback;
    uint32_t timestamp_query_size;
    bool timestamp_supported;

    uint32_t swapchain_width;
    uint32_t swapchain_height;
    uint32_t current_swap_index;
    uint64_t frame_index;
    NriFormat swapchain_format;

    NriDescriptor **temporary_descriptors;
    NriBuffer **temporary_buffers;
    uint32_t temporary_descriptor_num;
    uint32_t temporary_descriptor_cap;
    uint32_t temporary_buffer_num;
    uint32_t temporary_buffer_cap;

    TEXTURE_STATE *texture_states;
    uint32_t texture_state_num;
    uint32_t texture_state_cap;
} GPU;

bool gpu_init(GPU *gpu, const char *title, int width, int height);
void gpu_deinit(GPU *gpu);

#endif
