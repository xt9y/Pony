#include "gpu.h"
#include "game.h"

#include "NRI.h"
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
    NriAccessLayoutStage state;
};

NriDescriptor *gpu_swapchain_color_attachment(GPU *gpu, uint32_t swapchain_index) {
    if (!gpu || !gpu->swapchain_frames || swapchain_index >= gpu->swapchain_texture_count) {
        return NULL;
    }

    return gpu->swapchain_frames[swapchain_index].color_attachment;
}

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

static void destroy_swapchain(GPU *gpu);
static void destroy_frame_contexts(GPU *gpu);

static bool create_swapchain(GPU *gpu, uint32_t width, uint32_t height) {
    if (!gpu || !gpu->window || !gpu->device || !gpu->graphics_queue || !width || !height) return false;

    NriWindow window = {0};
    const SDL_PropertiesID props = SDL_GetWindowProperties(gpu->window);

#if defined(__APPLE__)
    (void)props;

    if (!gpu->metal_view) gpu->metal_view = SDL_Metal_CreateView(gpu->window);

    if (!gpu->metal_view) return false;

    window.metal.caMetalLayer = SDL_Metal_GetLayer(gpu->metal_view);
#elif defined(_WIN32)
    window.windows.hwnd = SDL_GetPointerProperty(props, SDL_PROP_WINDOW_WIN32_HWND_POINTER, NULL);

    if (!window.windows.hwnd) return false;
#else
    window.wayland.display = SDL_GetPointerProperty(props, SDL_PROP_WINDOW_WAYLAND_DISPLAY_POINTER, NULL);

    if (window.wayland.display) {
        window.wayland.surface = SDL_GetPointerProperty(props, SDL_PROP_WINDOW_WAYLAND_SURFACE_POINTER, NULL);

        if (!window.wayland.surface) return false;
    } else {
        window.x11.dpy = SDL_GetPointerProperty(props, SDL_PROP_WINDOW_X11_DISPLAY_POINTER, NULL);
        window.x11.window = (uint64_t)SDL_GetNumberProperty(props, SDL_PROP_WINDOW_X11_WINDOW_NUMBER, 0);

        if (!window.x11.dpy || !window.x11.window) return false;
    }
#endif

    const NriSwapChainDesc desc = {
        .window = window,
        .queue = gpu->graphics_queue,
        .width = (NriDim_t)width,
        .height = (NriDim_t)height,
        .textureNum = FRAME_QUEUE_DEPTH + 1u,
        .format = NriSwapChainFormat_BT709_G22_8BIT,
        .flags = NriSwapChainBits_VSYNC,
        .queuedFrameNum = FRAME_QUEUE_DEPTH
    };

    if (gpu->swapchain_api.CreateSwapChain(gpu->device, &desc, &gpu->swapchain) != NriResult_SUCCESS) return false;

    uint32_t count = 0;

    NriTexture *const *textures = gpu->swapchain_api.GetSwapChainTextures(gpu->swapchain, &count);

    if (!textures || !count) {
        destroy_swapchain(gpu);

        return false;
    }

    gpu->swapchain_texture_count = count;
    gpu->swapchain_textures = calloc(count, sizeof(*gpu->swapchain_textures));
    gpu->swapchain_frames = calloc(count, sizeof(*gpu->swapchain_frames));

    if (!gpu->swapchain_textures || !gpu->swapchain_frames) {
        destroy_swapchain(gpu);

        return false;
    }

    gpu->swapchain_format = gpu->core.GetTextureDesc(textures[0])->format;
    gpu->swapchain_width = width;
    gpu->swapchain_height = height;

    for (uint32_t i = 0; i < count; ++i) {
        SWAPCHAIN_FRAME *frame = &gpu->swapchain_frames[i];

        gpu->swapchain_textures[i] = textures[i];
        frame->texture = textures[i];
        frame->state = (NriAccessLayoutStage){
            .access = NriAccessBits_NONE,
            .layout = NriLayout_UNDEFINED,
            .stages = NriStageBits_NONE
        };

        const NriTextureViewDesc view_desc = {
            .texture = textures[i],
            .type = NriTextureView_COLOR_ATTACHMENT,
            .format = gpu->swapchain_format,
            .mipNum = 1,
            .layerNum = 1,
            .sliceNum = 1
        };

        if (gpu->core.CreateTextureView(&view_desc, &frame->color_attachment) != NriResult_SUCCESS ||
            gpu->core.CreateFence(gpu->device, NRI_SWAPCHAIN_SEMAPHORE, &frame->acquire) != NriResult_SUCCESS ||
            gpu->core.CreateFence(gpu->device, NRI_SWAPCHAIN_SEMAPHORE, &frame->release) != NriResult_SUCCESS) {
            destroy_swapchain(gpu);

            return false;
        }
    }

    return true;
}

static void destroy_swapchain(GPU *gpu) {
    if (!gpu) return;

    if (gpu->swapchain_frames) {
        for (uint32_t i = 0; i < gpu->swapchain_texture_count; ++i) {
            SWAPCHAIN_FRAME *frame = &gpu->swapchain_frames[i];

            if (frame->color_attachment) gpu->core.DestroyDescriptor(frame->color_attachment);

            if (frame->acquire) gpu->core.DestroyFence(frame->acquire);

            if (frame->release) gpu->core.DestroyFence(frame->release);
        }
    }

    free(gpu->swapchain_textures);
    free(gpu->swapchain_frames);

    gpu->swapchain_textures = NULL;
    gpu->swapchain_frames = NULL;
    gpu->swapchain_texture_count = 0;
    gpu->swapchain_width = 0;
    gpu->swapchain_height = 0;
    gpu->swapchain_format = NriFormat_UNKNOWN;

    if (gpu->swapchain) {
        gpu->swapchain_api.DestroySwapChain(gpu->swapchain);
        gpu->swapchain = NULL;
    }
}

static bool recreate_swapchain(GPU *gpu, uint32_t width, uint32_t height) {
    if (gpu->core.QueueWaitIdle(gpu->graphics_queue) != NriResult_SUCCESS) return false;

    destroy_swapchain(gpu);

    return create_swapchain(gpu, width, height);
}

static bool create_frame_contexts(GPU *gpu) {
    if (!gpu || !gpu->device || !gpu->graphics_queue) return false;

    gpu->frame_contexts = calloc(FRAME_QUEUE_DEPTH, sizeof(*gpu->frame_contexts));

    if (!gpu->frame_contexts) return false;

    if (gpu->core.CreateFence(gpu->device, 0, &gpu->frame_fence) != NriResult_SUCCESS) {
        destroy_frame_contexts(gpu);

        return false;
    }

    for (uint32_t i = 0; i < FRAME_QUEUE_DEPTH; ++i) {
        FRAME_CONTEXT *frame = &gpu->frame_contexts[i];

        if (gpu->core.CreateCommandAllocator(gpu->graphics_queue, &frame->allocator) != NriResult_SUCCESS ||
            gpu->core.CreateCommandBuffer(frame->allocator, &frame->command_buffer) != NriResult_SUCCESS) {
            destroy_frame_contexts(gpu);

            return false;
        }
    }

    return true;
}

static void destroy_frame_contexts(GPU *gpu) {
    if (!gpu) return;

    if (gpu->frame_contexts) {
        for (uint32_t i = 0; i < FRAME_QUEUE_DEPTH; ++i) {
            FRAME_CONTEXT *frame = &gpu->frame_contexts[i];

            if (frame->command_buffer) gpu->core.DestroyCommandBuffer(frame->command_buffer);

            if (frame->allocator) gpu->core.DestroyCommandAllocator(frame->allocator);
        }
    }

    free(gpu->frame_contexts);
    gpu->frame_contexts = NULL;

    if (gpu->frame_fence) {
        gpu->core.DestroyFence(gpu->frame_fence);
        gpu->frame_fence = NULL;
    }
}

static void transition_swapchain_texture(GPU *gpu, NriCommandBuffer *command_buffer, uint32_t swapchain_index, NriAccessLayoutStage after) {
    SWAPCHAIN_FRAME *frame = &gpu->swapchain_frames[swapchain_index];

    const NriTextureBarrierDesc texture_barrier = {
        .texture = frame->texture,
        .before = frame->state,
        .after = after,
        .mipNum = 1,
        .layerNum = 1,
        .planes = NriPlaneBits_COLOR
    };

    const NriBarrierDesc barrier = {
        .textures = &texture_barrier,
        .textureNum = 1
    };

    gpu->core.CmdBarrier(command_buffer, &barrier);
    frame->state = after;
}

static bool create_streamer(GPU *gpu) {
    const NriStreamerDesc streamer_desc = {
        .constantBufferMemoryLocation = NriMemoryLocation_HOST_UPLOAD,
        .constantBufferSize = 1024 * 1024,
        .dynamicBufferMemoryLocation = NriMemoryLocation_HOST_UPLOAD,
        .dynamicBufferDesc = {
            .usage = NriBufferUsageBits_NONE
        },
        .queuedFrameNum = FRAME_QUEUE_DEPTH,
        .hostDataCapacity = 0
    };

    if (gpu->streamer_api.CreateStreamer(gpu->device, &streamer_desc, &gpu->streamer) != NriResult_SUCCESS) {
        return false;
    }

    return true;
}

static void destroy_streamer(GPU *gpu) {
    if (gpu->streamer) {
        gpu->streamer_api.DestroyStreamer(gpu->streamer);
    }
}

bool gpu_begin_frame(GPU *gpu, NriCommandBuffer **command_buffer, NriTexture **swapchain_texture, uint32_t *swapchain_index) {
    if (!gpu || !gpu->device || !gpu->swapchain || !gpu->frame_contexts || !gpu->frame_fence || !command_buffer || !swapchain_texture || !swapchain_index) {
        return false;
    }

    *command_buffer = NULL;
    *swapchain_texture = NULL;
    *swapchain_index = 0;

    int width = 0;
    int height = 0;

    if (!SDL_GetWindowSizeInPixels(gpu->window, &width, &height)) return false;

    if (width <= 0 || height <= 0) return true;

    if ((uint32_t)width != gpu->swapchain_width || (uint32_t)height != gpu->swapchain_height) {
        if (!recreate_swapchain(gpu, (uint32_t)width, (uint32_t)height)) return false;
    }

    const uint64_t wait_value = gpu->frame_index >= FRAME_QUEUE_DEPTH ? 1u + gpu->frame_index - FRAME_QUEUE_DEPTH : 0u;

    gpu->core.Wait(gpu->frame_fence, wait_value);

    FRAME_CONTEXT *frame = &gpu->frame_contexts[gpu->frame_index % FRAME_QUEUE_DEPTH];
    gpu->core.ResetCommandAllocator(frame->allocator);

    const uint32_t semaphore_index = (uint32_t)(gpu->frame_index % gpu->swapchain_texture_count);
    NriFence *acquire = gpu->swapchain_frames[semaphore_index].acquire;

    uint32_t index = 0;
    NriResult result = gpu->swapchain_api.AcquireNextTexture(gpu->swapchain, acquire, &index);

    if (result == NriResult_OUT_OF_DATE) {
        if (!recreate_swapchain(gpu, (uint32_t)width, (uint32_t)height)) return false;

        acquire = gpu->swapchain_frames[semaphore_index % gpu->swapchain_texture_count].acquire;
        result = gpu->swapchain_api.AcquireNextTexture(gpu->swapchain, acquire, &index);
    }

    if (result != NriResult_SUCCESS || index >= gpu->swapchain_texture_count) return false;

    if (gpu->core.BeginCommandBuffer(frame->command_buffer, NULL) != NriResult_SUCCESS) return false;

    transition_swapchain_texture(
        gpu,
        frame->command_buffer,
        index,
        (NriAccessLayoutStage){
            .access = NriAccessBits_COLOR_ATTACHMENT_WRITE,
            .layout = NriLayout_COLOR_ATTACHMENT,
            .stages = NriStageBits_COLOR_ATTACHMENT
        }
    );

    *command_buffer = frame->command_buffer;
    *swapchain_texture = gpu->swapchain_textures[index];
    *swapchain_index = index;

    return true;
}

bool gpu_clear_frame(GPU *gpu, NriCommandBuffer *command_buffer, uint32_t swapchain_index, NriColor32f clear_color) {
    if (!gpu || !command_buffer || swapchain_index >= gpu->swapchain_texture_count) return false;

    const NriAttachmentDesc color = {
        .descriptor = gpu->swapchain_frames[swapchain_index].color_attachment,
        .clearValue = {
            .color = {
                .f = clear_color
            }
        },
        .loadOp = NriLoadOp_CLEAR,
        .storeOp = NriStoreOp_STORE
    };

    const NriRenderingDesc rendering = {
        .colors = &color,
        .colorNum = 1
    };

    gpu->core.CmdBeginRendering(command_buffer, &rendering);
    gpu->core.CmdEndRendering(command_buffer);

    return true;
}

bool gpu_end_frame(GPU *gpu, NriCommandBuffer *command_buffer, uint32_t swapchain_index) {
    if (!gpu || !command_buffer || !gpu->swapchain || swapchain_index >= gpu->swapchain_texture_count) return false;

    transition_swapchain_texture(
        gpu, command_buffer, swapchain_index, (NriAccessLayoutStage){
            .access = NriAccessBits_NONE,
            .layout = NriLayout_PRESENT,
            .stages = NriStageBits_NONE
        }
    );

    if (gpu->core.EndCommandBuffer(command_buffer) != NriResult_SUCCESS) return false;

    const uint32_t semaphore_index = (uint32_t)(gpu->frame_index % gpu->swapchain_texture_count);
    NriFence *acquire = gpu->swapchain_frames[semaphore_index].acquire;
    NriFence *release = gpu->swapchain_frames[swapchain_index].release;
    const uint64_t present_id = gpu->frame_index + 1u;

    const NriFenceSubmitDesc wait_fence = {
        .fence = acquire,
        .stages = NriStageBits_COLOR_ATTACHMENT
    };

    const NriFenceSubmitDesc signal_fences[] = {{
    .fence = release
}, {
    .fence = gpu->frame_fence,
    .value = present_id
}};

    const NriCommandBuffer *const command_buffers[] = {command_buffer};

    const NriQueueSubmitDesc submit = {
        .waitFences = &wait_fence,
        .waitFenceNum = 1,
        .commandBuffers = command_buffers,
        .commandBufferNum = 1,
        .signalFences = signal_fences,
        .signalFenceNum = 2
    };

    if (gpu->core.QueueSubmit(gpu->graphics_queue, &submit) != NriResult_SUCCESS) return false;

    const NriResult result = gpu->swapchain_api.QueuePresent(gpu->swapchain, release, present_id);

    gpu->frame_index = present_id;

    if (result == NriResult_OUT_OF_DATE) {
        int width = 0;
        int height = 0;

        if (!SDL_GetWindowSizeInPixels(gpu->window, &width, &height)) return false;

        if (width <= 0 || height <= 0) return true;

        return recreate_swapchain(gpu, (uint32_t)width, (uint32_t)height);
    }

    return result == NriResult_SUCCESS;
}

bool gpu_resize(GPU *gpu) {
    if (!gpu || !gpu->window || !gpu->device || !gpu->graphics_queue) return false;

    int width = 0;
    int height = 0;

    if (!SDL_GetWindowSizeInPixels(gpu->window, &width, &height)) return false;

    if (width <= 0 || height <= 0) return true;

    if (gpu->swapchain && gpu->swapchain_width == (uint32_t)width && gpu->swapchain_height == (uint32_t)height) return true;

    return recreate_swapchain(gpu, (uint32_t)width, (uint32_t)height);
}

bool gpu_create_buffer(GPU *gpu, const NriBufferDesc *desc, NriMemoryLocation memory, NriBuffer **buffer) {
    if (!gpu || !gpu->device || !desc || !buffer) return false;

    return gpu->core.CreateCommittedBuffer(gpu->device, memory, 0.0f, desc, buffer) == NriResult_SUCCESS;
}

bool gpu_create_texture(GPU *gpu, const NriTextureDesc *desc, NriMemoryLocation memory, NriTexture **texture) {
    if (!gpu || !gpu->device || !desc || !texture) return false;

    return gpu->core.CreateCommittedTexture(gpu->device, memory, 0.0f, desc, texture) == NriResult_SUCCESS;
}

bool gpu_upload_buffer(GPU *gpu, NriBuffer *buffer, const void *data, NriAccessStage after) {
    if (!gpu || !gpu->graphics_queue || !buffer || !data) return false;

    const NriBufferUploadDesc upload = {
        .data = data,
        .buffer = buffer,
        .after = after
    };

    return gpu->helper.UploadData(gpu->graphics_queue, NULL, 0, &upload, 1) == NriResult_SUCCESS;
}

bool gpu_upload_texture(GPU *gpu, NriTexture *texture, const NriTextureSubresourceUploadDesc *subresources, NriPlaneBits planes, NriAccessLayoutStage after) {
    if (!gpu || !gpu->graphics_queue || !texture || !subresources) return false;

    const NriTextureUploadDesc upload = {
        .subresources = subresources,
        .texture = texture,
        .after = after,
        .planes = planes
    };

    return gpu->helper.UploadData(gpu->graphics_queue, &upload, 1, NULL, 0) == NriResult_SUCCESS;
}

void gpu_destroy_buffer(GPU *gpu, NriBuffer *buffer) {
    if (!gpu || !buffer) return;

    gpu->core.DestroyBuffer(buffer);
}

void gpu_destroy_texture(GPU *gpu, NriTexture *texture) {
    if (!gpu || !texture) return;

    gpu->core.DestroyTexture(texture);
}

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

    NriDeviceCreationDesc device_desc = {
        .graphicsAPI = NriGraphicsAPI_VK,
        .enableNRIValidation = false,
        .enableGraphicsAPIValidation = false,
        .vkBindingOffsets = {
            .sRegister = 0,
            .tRegister = 16,
            .bRegister = 32,
            .uRegister = 48
        }
    };

    NriVertexAttributeDesc attributes_desc[] = {
        {
            .d3d = {"POSITION", 0},
            .vk = {0},
            .offset = 0,
            .format = NriFormat_RGB32_SFLOAT,
            .streamIndex = 0
        },
        {
            .d3d = {"NOR  MAL", 0},
            .vk = {1},
            .offset = 12,
            .format = NriFormat_RGB32_SFLOAT,
            .streamIndex = 0
        },
        {
            .d3d = {"TEXCOORD", 0},
            .vk = {2},
            .offset = 24,
            .format = NriFormat_RG32_SFLOAT,
            .streamIndex = 0
        },
        {
            .d3d = {"MATERIAL", 0},
            .vk = {3},
            .offset = 32,
            .format = NriFormat_R32_UINT,
            .streamIndex = 0
        }
    };

    NriVertexStreamDesc stream_desc = {
        .bindingSlot = 0,
        .stride = sizeof(GLTF_VERTEX),
        .stepRate = NriVertexStreamStepRate_PER_VERTEX
    };

    if (nriCreateDevice(&device_desc, &gpu->device) != NriResult_SUCCESS) {
        SDL_Log("NRI device creation failed");
        gpu_deinit(gpu);

        return false;
    }

    if (nriGetInterface(gpu->device, NRI_INTERFACE(NriStreamerInterface), &gpu->streamer_api) != NriResult_SUCCESS ||
        nriGetInterface(gpu->device, NRI_INTERFACE(NriCoreInterface), &gpu->core) != NriResult_SUCCESS ||
        nriGetInterface(gpu->device, NRI_INTERFACE(NriHelperInterface), &gpu->helper) != NriResult_SUCCESS ||
        nriGetInterface(gpu->device, NRI_INTERFACE(NriSwapChainInterface), &gpu->swapchain_api) != NriResult_SUCCESS) {
        SDL_Log("NRI interface acquisition failed");
        gpu_deinit(gpu);

        return false;
    }

    if (!acquire_queues(gpu) || !create_frame_contexts(gpu)) {
        SDL_Log("NRI frame creation failed");
        gpu_deinit(gpu);

        return false;
    }

    if (!create_swapchain(gpu, (uint32_t)width, (uint32_t)height)) {
        SDL_Log("NRI swapchain creation failed");
        gpu_deinit(gpu);

        return false;
    }

    if (!create_streamer(gpu)) {
        SDL_Log("NRI stramer creation failed");
        gpu_deinit(gpu);

        return false;
    }

    SDL_Log("GPU backend: NRI Vulkan");

    return true;
}

void gpu_deinit(GPU *gpu) {
    if (!gpu) return;

    if (gpu->device) {
        if (gpu->graphics_queue) gpu->core.QueueWaitIdle(gpu->graphics_queue);

        destroy_frame_contexts(gpu);
        destroy_swapchain(gpu);
        destroy_streamer(gpu);

        nriDestroyDevice(gpu->device);
    }

#if defined(__APPLE__)
    if (gpu->metal_view) SDL_Metal_DestroyView(gpu->metal_view);
#endif

    if (gpu->window) SDL_DestroyWindow(gpu->window);

    memset(gpu, 0, sizeof(*gpu));
}
