#include "gpu.h"
#include "Extensions/NRISwapChain.h"
#include "NRIDescs.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define FRAME_QUEUE_DEPTH 2u

struct FRAME_CONTEXT {
    NriCommandAllocator *allocator;
    NriCommandBuffer *command_buffer;
};

struct SWAPCHAIN_FRAME {
    NriDescriptor *color_attachment;
    NriFence *acquire;
    NriFence *release;
    NriTexture *texture;
};

static bool acquire_queues(GPU *gpu) {
    if (!gpu || !gpu->device) return false;

    if (gpu->core.GetQueue(gpu->device, NriQueueType_GRAPHICS, 0, &gpu->graphics_queue) != NriResult_SUCCESS) {
        return false;
    }

    if (gpu->core.GetQueue(gpu->device, NriQueueType_COMPUTE, 0, &gpu->compute_queue) != NriResult_SUCCESS) {
        gpu->compute_queue = gpu->graphics_queue;
    }

    if (gpu->core.GetQueue(gpu->device, NriQueueType_COPY, 0, &gpu->copy_queue) != NriResult_SUCCESS) {
        gpu->copy_queue = gpu->graphics_queue;
    }

    return true;
}


static bool create_swapchain(GPU *gpu, uint32_t width, uint32_t height) {
    if (!gpu->window) {
        return false;
    }

    NriWindow window = {0};

#if defined(_WIN32)
    SDL_PropertiesID props = SDL_GetWindowProperties(gpu->window);

    window.windows.hwnd = SDL_GetPointerProperty(props, SDL_PROP_WINDOW_WIN32_HWND_POINTER, NULL);
#endif

    NriSwapChainDesc swapchain_desc = {
        .window = window,
        .queue = gpu->graphics_queue,
        .format = NriSwapChainFormat_BT709_G22_8BIT,
        .flags = NriSwapChainBits_VSYNC,
        .width = (NriDim_t)width,
        .height = (NriDim_t)height,
        .textureNum = FRAME_QUEUE_DEPTH + 1u,
        .queuedFrameNum = FRAME_QUEUE_DEPTH
    };

    if (gpu->swapchain_api.CreateSwapChain(gpu->device, &swapchain_desc, &gpu->swapchain) != NriResult_SUCCESS) {
        return false;
    }

    uint32_t count;

    NriTexture *const *textures = gpu->swapchain_api.GetSwapChainTextures(gpu->swapchain, &count);

    if (!textures || !count) {
        return false;
    }

    gpu->swapchain_textures = calloc(count, sizeof(*gpu->swapchain_textures));
    gpu->swapchain_frames = calloc(count, sizeof(*gpu->swapchain_frames));

    if (!gpu->swapchain_textures || !gpu->swapchain_frames) {
        return false;
    }

    gpu->swapchain_texture_count = count;
    gpu->swapchain_format = gpu->core.GetTextureDesc(textures[0])->format;
    gpu->swapchain_height = height;
    gpu->swapchain_width = width;

    for (uint32_t i = 0; i < count; ++i) {
        gpu->swapchain_textures[i] = textures[i];

        NriTextureViewDesc view_desc = {
            .texture = textures[i],
            .type = NriTextureView_COLOR_ATTACHMENT,
            .format = gpu->swapchain_format,
            .mipNum = 1,
            .layerNum = 1,
            .sliceNum = 1,
        };

        SWAPCHAIN_FRAME *frame = &gpu->swapchain_frames[i];
        frame->texture = textures[i];

        if (gpu->core.CreateTextureView(&view_desc, &frame->color_attachment) != NriResult_SUCCESS ||
            gpu->core.CreateFence(gpu->device, NRI_SWAPCHAIN_SEMAPHORE, &frame->acquire) != NriResult_SUCCESS ||
            gpu->core.CreateFence(gpu->device, NRI_SWAPCHAIN_SEMAPHORE, &frame->release) != NriResult_SUCCESS) {
            return false;
        }
    }


    return true;
}

static void destroy_swapchain(GPU *gpu) {
    for (uint32_t i = 0; i < gpu->swapchain_texture_count; ++i) {
        SWAPCHAIN_FRAME *frame = &gpu->swapchain_frames[i];

        if (frame->color_attachment) {
            gpu->core.DestroyDescriptor(frame->color_attachment);
        }

        if (frame->acquire) {
            gpu->core.DestroyFence(frame->acquire);
        }

        if (frame->release) {
            gpu->core.DestroyFence(frame->release);
        }
    }

    free(gpu->swapchain_textures);
    free(gpu->swapchain_frames);

    gpu->swapchain_frames = NULL;
    gpu->swapchain_textures = NULL;
    gpu->swapchain_texture_count = 0;

    if (gpu->swapchain) gpu->swapchain_api.DestroySwapChain(gpu->swapchain);
    gpu->swapchain = NULL;
}

static bool create_frame_contexts(GPU *gpu) {
    gpu->frame_contexts = calloc(FRAME_QUEUE_DEPTH, sizeof(*gpu->frame_contexts));

    for (uint32_t i = 0; i < FRAME_QUEUE_DEPTH; ++i) {
        FRAME_CONTEXT *frame = &gpu->frame_contexts[i];

        if (gpu->core.CreateCommandAllocator(gpu->graphics_queue, &frame->allocator) != NriResult_SUCCESS) {
            return false;
        }

        if (gpu->core.CreateCommandBuffer(frame->allocator, &frame->command_buffer) != NriResult_SUCCESS) {
            return false;
        }
    }

    if (!gpu->frame_contexts) {
        return false;
    }

    if (gpu->core.CreateFence(gpu->device, 0, &gpu->frame_fence) != NriResult_SUCCESS) {
        return false;
    }

    return true;
}

static void destroy_frame_contexts(GPU *gpu) {
    if (!gpu) return;

    if (gpu->frame_contexts) {
        for (uint32_t i = 0; i < FRAME_QUEUE_DEPTH; ++i) {
            FRAME_CONTEXT *frame = &gpu->frame_contexts[i];

            if (frame->command_buffer) {
                gpu->core.DestroyCommandBuffer(frame->command_buffer);
            }

            if (frame->allocator) {
                gpu->core.DestroyCommandAllocator(frame->allocator);
            }
        }
    }

    free(gpu->frame_contexts);
    gpu->frame_contexts = NULL;

    if (gpu->frame_fence) {
        gpu->core.DestroyFence(gpu->frame_fence);
        gpu->frame_fence = NULL;
    }
}

// bool gpu_begin_frame(GPU *gpu, NriCommandBuffer **command_buffer, NriTexture **swapchain_texture, uint32_t *swapchain_index) {
//
// }
// bool gpu_end_frame(GPU *gpu, NriCommandBuffer *command_buffer, uint32_t *swapchain_index) {
//
// }
// bool gpu_resize(GPU *gpu) {
//
// }


bool gpu_init(GPU *gpu, const char *title, int width, int height) {
    if (!gpu || !title || width <= 0 || height <= 0) return false;

    memset(gpu, 0, sizeof(*gpu));

    Uint64 window_flags = SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY;

#if defined(__APPLE__)
    window_flags |= SDL_WINDOW_METAL;
#else
    window_flags |= SDL_WINDOW_VULKAN;
#endif

    gpu->window = SDL_CreateWindow(title, width, height, window_flags);

    if (!gpu->window) return false;

    NriDeviceCreationDesc device_desc = {0};

    device_desc.graphicsAPI = NriGraphicsAPI_VK;
    device_desc.enableNRIValidation = false;
    device_desc.enableGraphicsAPIValidation = false;

    device_desc.vkBindingOffsets = (NriVKBindingOffsets){
        .sRegister = 0,
        .tRegister = 16,
        .bRegister = 32,
        .uRegister = 48
    };

    if (nriCreateDevice(&device_desc, &gpu->device) != NriResult_SUCCESS) {
        SDL_Log("NRI device creation failed");
        gpu_deinit(gpu);

        return false;
    }

    if (nriGetInterface(gpu->device, NRI_INTERFACE(NriCoreInterface), &gpu->core) != NriResult_SUCCESS ||
        nriGetInterface(gpu->device, NRI_INTERFACE(NriHelperInterface), &gpu->helper) != NriResult_SUCCESS ||
        nriGetInterface(gpu->device, NRI_INTERFACE(NriSwapChainInterface), &gpu->swapchain_api) != NriResult_SUCCESS) {
        SDL_Log("NRI interface acquisition failed");
        gpu_deinit(gpu);

        return false;
    }

    if (!acquire_queues(gpu)) {
        SDL_Log("NRI queue acquisition failed");
        gpu_deinit(gpu);

        return false;
    }

    if (!create_swapchain(gpu, (uint32_t)width, (uint32_t)height) || !create_frame_contexts(gpu)) {
        SDL_Log("NRI swapchain/ fram context creation failed");
        gpu_deinit(gpu);

        return false;
    }

    SDL_Log("GPU backend: NRI Vulkan");

    return true;
}


void gpu_deinit(GPU *gpu) {
    if (!gpu) return;

    if (gpu->device) {
        if (gpu->graphics_queue) {
            gpu->core.QueueWaitIdle(gpu->graphics_queue);
        }

        destroy_frame_contexts(gpu);
        destroy_swapchain(gpu);

        nriDestroyDevice(gpu->device);
    }

    if (gpu->window) {
        SDL_DestroyWindow(gpu->window);
    }

    memset(gpu, 0, sizeof(*gpu));
}
