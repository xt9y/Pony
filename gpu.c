#include "gpu.h"

#include <string.h>


static bool acquire_queues(GPU *gpu) {
    if (!gpu || !gpu->device) return false;

    if (gpu->core.GetQueue(
            gpu->device,
            NriQueueType_GRAPHICS,
            0,
            &gpu->graphics_queue
        ) != NriResult_SUCCESS) {
        return false;
    }

    if (gpu->core.GetQueue(
            gpu->device,
            NriQueueType_COMPUTE,
            0,
            &gpu->compute_queue
        ) != NriResult_SUCCESS) {
        gpu->compute_queue = gpu->graphics_queue;
    }

    if (gpu->core.GetQueue(
            gpu->device,
            NriQueueType_COPY,
            0,
            &gpu->copy_queue
        ) != NriResult_SUCCESS) {
        gpu->copy_queue = gpu->graphics_queue;
    }

    return true;
}


bool gpu_init(GPU *gpu, const char *title, int width, int height) {
    if (!gpu || !title || width <= 0 || height <= 0) return false;

    memset(gpu, 0, sizeof(*gpu));

    Uint64 window_flags =
        SDL_WINDOW_RESIZABLE |
        SDL_WINDOW_HIGH_PIXEL_DENSITY;

#if defined(__APPLE__)
    window_flags |= SDL_WINDOW_METAL;
#else
    window_flags |= SDL_WINDOW_VULKAN;
#endif

    gpu->window = SDL_CreateWindow(
        title,
        width,
        height,
        window_flags
    );

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

    if (nriCreateDevice(
            &device_desc,
            &gpu->device
        ) != NriResult_SUCCESS) {
        SDL_Log("NRI device creation failed");
        gpu_deinit(gpu);

        return false;
    }

    if (nriGetInterface(
            gpu->device,
            NRI_INTERFACE(NriCoreInterface),
            &gpu->core
        ) != NriResult_SUCCESS ||
        nriGetInterface(
            gpu->device,
            NRI_INTERFACE(NriHelperInterface),
            &gpu->helper
        ) != NriResult_SUCCESS ||
        nriGetInterface(
            gpu->device,
            NRI_INTERFACE(NriSwapChainInterface),
            &gpu->swapchain
        ) != NriResult_SUCCESS) {
        SDL_Log("NRI interface acquisition failed");
        gpu_deinit(gpu);

        return false;
    }

    if (!acquire_queues(gpu)) {
        SDL_Log("NRI queue acquisition failed");
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

        nriDestroyDevice(gpu->device);
    }

    if (gpu->window) {
        SDL_DestroyWindow(gpu->window);
    }

    memset(gpu, 0, sizeof(*gpu));
}
