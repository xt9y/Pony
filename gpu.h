#ifndef GPU_H
#define GPU_H

#include "NRIDescs.h"
#include <stdbool.h>
#include <stdint.h>

#include <SDL3/SDL.h>

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#pragma clang diagnostic ignored "-Wvariadic-macro-arguments-omitted"
#pragma clang diagnostic ignored "-Wstrict-prototypes"
#endif

// cant use Externsions/NriRayTracing.h because my M2 doesnt support that :(
#include <NRI.h>
#include <Extensions/NRIDeviceCreation.h>
#include <Extensions/NRIHelper.h>
#include <Extensions/NRISwapChain.h>
#include <Extensions/NRIStreamer.h>

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
    NriStreamerInterface streamer_api;

    NriStreamer *streamer;

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

typedef struct MAT4 {
    float m[16];
} MAT4;

typedef struct GPU_OBJECT {
    MAT4 world;
    MAT4 previous_world;
    MAT4 normal_world;

    uint32_t first_vertex;
    uint32_t vertex_count;
    uint32_t object_id;
    uint32_t material_offset;

    uint32_t revision;
    uint32_t state;
    uint32_t type;
    uint32_t padding;
} GPU_OBJECT;

typedef struct GPU_MATERIAL {
    float base_color[4];
    float emissive[3];
    float metallic;
    float roughness;
    float padding[3];
} GPU_MATERIAL;

NriDescriptor *gpu_swapchain_color_attachment(GPU *gpu, uint32_t swapchain_index);

bool gpu_init(GPU *gpu, const char *title, int width, int height);
void gpu_deinit(GPU *gpu);

bool gpu_begin_frame(GPU *gpu, NriCommandBuffer **command_buffer, NriTexture **swapchain_texture, uint32_t *swapchain_index);

bool gpu_clear_frame(GPU *gpu, NriCommandBuffer *command_buffer, uint32_t swapchain_index, NriColor32f clear_color);

bool gpu_end_frame(GPU *gpu, NriCommandBuffer *command_buffer, uint32_t swapchain_index);

bool gpu_resize(GPU *gpu);

bool gpu_create_buffer(GPU *gpu, const NriBufferDesc *desc, NriMemoryLocation memory, NriBuffer **buffer);

bool gpu_create_texture(GPU *gpu, const NriTextureDesc *desc, NriMemoryLocation memory, NriTexture **texture);

bool gpu_upload_buffer(GPU *gpu, NriBuffer *buffer, const void *data, NriAccessStage after);

bool gpu_upload_texture(GPU *gpu, NriTexture *texture, const NriTextureSubresourceUploadDesc *subresources, NriPlaneBits planes, NriAccessLayoutStage after);

void gpu_destroy_buffer(GPU *gpu, NriBuffer *buffer);
void gpu_destroy_texture(GPU *gpu, NriTexture *texture);

#endif
