#include "game.h"
#include "render_internal.h"

#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FRAME_QUEUE_DEPTH 2u
#define WORK_QUEUE_DEPTH 8u
#define UNIFORM_RING_BYTES (1024u * 1024u)
#define TIMESTAMP_CAPACITY 16u
#define UPLOAD_RING_SIZE 2u
#define UPLOAD_CHUNK_BYTES (32u * 1024u * 1024u)
#define UPLOAD_SLOW_LOG_MS 5000u

struct SWAPCHAIN_TEXTURE {
    NriTexture *texture;
    NriDescriptor *color_attachment;
    NriFence *acquire;
    NriFence *release;
};

struct FRAME_CONTEXT {
    NriCommandAllocator *allocator;
    NriCommandBuffer *command_buffer;
    NriDescriptorPool *descriptor_pool;
    NriDescriptor **temporary_descriptors;
    NriBuffer **temporary_buffers;

    uint32_t temporary_descriptor_num, temporary_descriptor_cap;
    uint32_t temporary_buffer_num, temporary_buffer_cap;

    NriBuffer *uniform_buffer;
    uint64_t uniform_offset;
    uint64_t fence_value;
};

typedef struct UPLOAD_SLOT {
    NriBuffer *staging;
    NriCommandAllocator *allocator;
    NriCommandBuffer *command_buffer;
    uint64_t fence_value;
} UPLOAD_SLOT;

struct UPLOAD_CONTEXT {
    NriFence *fence;
    UPLOAD_SLOT slots[UPLOAD_RING_SIZE];
    uint64_t next_fence_value;
    uint32_t next_slot;
};

struct TEXTURE_STATE {
    NriTexture *texture;
    NriAccessLayoutStage state;
};

static Uint8 *load_spirv(const char *define, size_t *size) {
    char path[256];
    int written = snprintf(path, sizeof(path), "build/shaders/%s.spv", define);

    if (written < 0 || (size_t)written >= sizeof(path)) return NULL;

    Uint8 *spirv = SDL_LoadFile(path, size);
    if (!spirv) SDL_Log("SPIR-V load failed for %s: %s", path, SDL_GetError());

    return spirv;
}

static bool create_pipeline_cache(RENDERER *r) {
    if (!r || !r->gpu->device) return false;

    if (r->gpu->pipeline_cache) return true;

    const NriPipelineCacheDesc desc = {0};
    NriResult result = r->gpu->core.CreatePipelineCache(r->gpu->device, &desc, &r->gpu->pipeline_cache);

    if (result == NriResult_UNSUPPORTED) {
        r->gpu->pipeline_cache = NULL;

        return true;
    }

    return result == NriResult_SUCCESS;
}

static void destroy_pipeline_cache(RENDERER *r) {
    if (!r || !r->gpu->pipeline_cache) return;
    r->gpu->core.DestroyPipelineCache(r->gpu->pipeline_cache);
    r->gpu->pipeline_cache = NULL;
}

static bool acquire_queues(RENDERER *r) {
    if (!r || !r->gpu->device) return false;

    if (r->gpu->core.GetQueue(r->gpu->device, NriQueueType_GRAPHICS, 0, &r->gpu->graphics_queue) != NriResult_SUCCESS) return false;

    if (r->gpu->core.GetQueue(r->gpu->device, NriQueueType_COMPUTE, 0, &r->gpu->compute_queue) != NriResult_SUCCESS)
        r->gpu->compute_queue = r->gpu->graphics_queue;

    if (r->gpu->core.GetQueue(r->gpu->device, NriQueueType_COPY, 0, &r->gpu->copy_queue) != NriResult_SUCCESS) r->gpu->copy_queue = r->gpu->graphics_queue;

    r->gpu->work_queue = r->gpu->compute_queue ? r->gpu->compute_queue : r->gpu->graphics_queue;

    return true;
}

static bool create_gpu_timestamps(RENDERER *r) {
    if (!r || !r->gpu->device) return false;

    const NriDeviceDesc *device = r->gpu->core.GetDeviceDesc(r->gpu->device);

    if (!device || !device->features.timestamp || !device->other.timestampFrequencyHz) return true;

    const NriQueryPoolDesc query_desc = {.queryType = NriQueryType_TIMESTAMP, .capacity = TIMESTAMP_CAPACITY};

    if (r->gpu->core.CreateQueryPool(r->gpu->device, &query_desc, &r->gpu->timestamp_pool) != NriResult_SUCCESS) return true;

    r->gpu->timestamp_query_size = r->gpu->core.GetQuerySize(r->gpu->timestamp_pool);

    if (!r->gpu->timestamp_query_size) {
        r->gpu->core.DestroyQueryPool(r->gpu->timestamp_pool);
        r->gpu->timestamp_pool = NULL;

        return true;
    }

    const NriBufferDesc readback_desc = {.size = (uint64_t)r->gpu->timestamp_query_size * TIMESTAMP_CAPACITY};

    if (r->gpu->core.CreateCommittedBuffer(r->gpu->device, NriMemoryLocation_HOST_READBACK, 0.0f, &readback_desc, &r->gpu->timestamp_readback) !=
        NriResult_SUCCESS) {
        r->gpu->core.DestroyQueryPool(r->gpu->timestamp_pool);
        r->gpu->timestamp_pool = NULL;
        r->gpu->timestamp_query_size = 0u;

        return true;
    }

    r->gpu->timestamp_supported = true;

    return true;
}

static void destroy_gpu_timestamps(RENDERER *r) {
    if (!r) return;

    if (r->gpu->timestamp_readback) r->gpu->core.DestroyBuffer(r->gpu->timestamp_readback);

    if (r->gpu->timestamp_pool) r->gpu->core.DestroyQueryPool(r->gpu->timestamp_pool);
    r->gpu->timestamp_readback = NULL;
    r->gpu->timestamp_pool = NULL;
    r->gpu->timestamp_query_size = 0u;
    r->gpu->timestamp_supported = false;
}

bool gpu_timestamp_begin(RENDERER *r, NriCommandBuffer *cmd, uint32_t slot) {
    if (!r || !cmd || !r->gpu->timestamp_supported) return true;

    if (slot + 1u >= TIMESTAMP_CAPACITY) return false;
    r->gpu->core.CmdResetQueries(cmd, r->gpu->timestamp_pool, slot, 2u);
    r->gpu->core.CmdEndQuery(cmd, r->gpu->timestamp_pool, slot);

    return true;
}

bool gpu_timestamp_end(RENDERER *r, NriCommandBuffer *cmd, uint32_t slot) {
    if (!r || !cmd || !r->gpu->timestamp_supported) return true;

    if (slot + 1u >= TIMESTAMP_CAPACITY) return false;
    r->gpu->core.CmdEndQuery(cmd, r->gpu->timestamp_pool, slot + 1u);
    r->gpu->core.CmdCopyQueries(cmd, r->gpu->timestamp_pool, slot, 2u, r->gpu->timestamp_readback, (uint64_t)slot * r->gpu->timestamp_query_size);

    return true;
}

void gpu_timestamp_log(RENDERER *r, uint32_t slot, const char *label) {
    if (!r || !r->gpu->timestamp_supported || slot + 1u >= TIMESTAMP_CAPACITY || !label) return;

    const NriDeviceDesc *device = r->gpu->core.GetDeviceDesc(r->gpu->device);

    if (!device || !device->other.timestampFrequencyHz) return;

    const uint64_t offset = (uint64_t)slot * r->gpu->timestamp_query_size;
    const uint64_t bytes = (uint64_t)r->gpu->timestamp_query_size * 2u;
    const uint8_t *mapped = r->gpu->core.MapBuffer(r->gpu->timestamp_readback, offset, bytes);

    if (!mapped) return;

    uint64_t begin = 0u;
    uint64_t end = 0u;

    memcpy(&begin, mapped, sizeof(begin));
    memcpy(&end, mapped + r->gpu->timestamp_query_size, sizeof(end));
    r->gpu->core.UnmapBuffer(r->gpu->timestamp_readback);

    if (end >= begin) {
        const double ms = (double)(end - begin) * 1000.0 / (double)device->other.timestampFrequencyHz;

        SDL_Log("GPU: %s %.3f ms", label, ms);
    }
}

NriShaderDesc gpu_load_shader(const char *entrypoint, const char *define, NriStageBits stage) {
    size_t spirv_size = 0;
    Uint8 *spirv = load_spirv(define, &spirv_size);
    if (!spirv) return (NriShaderDesc){0};

    return (NriShaderDesc){.stage = stage, .bytecode = spirv, .size = spirv_size, .entryPointName = entrypoint};
}

NriPipeline *gpu_compile_compute(RENDERER *r, NriPipelineLayout *layout, const char *path, const char *entrypoint, const char *define) {
    (void)path;

    size_t spirv_size = 0;
    Uint8 *spirv = load_spirv(define, &spirv_size);

    if (!spirv) return NULL;

    const NriShaderDesc shader = {.stage = NriStageBits_COMPUTE_SHADER, .bytecode = spirv, .size = spirv_size, .entryPointName = entrypoint};
    const NriComputePipelineDesc desc = {.pipelineLayout = layout, .shader = shader, .cache = r->gpu->pipeline_cache};
    NriPipeline *pipeline = NULL;
    NriResult result = r->gpu->core.CreateComputePipeline(r->gpu->device, &desc, &pipeline);

    SDL_free(spirv);

    if (result != NriResult_SUCCESS) {
        SDL_Log("compute pipeline creation failed for %s:%s", define, entrypoint);

        return NULL;
    }

    return pipeline;
}

void gpu_clear_temporary(RENDERER *r);
static bool begin_work_commands(RENDERER *r, NriCommandAllocator **allocator, NriCommandBuffer **command_buffer);
static bool submit_work_commands(RENDERER *r, NriCommandAllocator *allocator, NriCommandBuffer *command_buffer, bool wait);
static void abort_work_commands(RENDERER *r, NriCommandAllocator *allocator, NriCommandBuffer *command_buffer);
static void work_drain(RENDERER *r);

NriResult gpu_begin_commands(RENDERER *r, NriCommandAllocator **allocator, NriCommandBuffer **command_buffer) {
    return begin_work_commands(r, allocator, command_buffer) ? NriResult_SUCCESS : NriResult_FAILURE;
}

bool gpu_submit_commands(RENDERER *r, NriCommandAllocator *allocator, NriCommandBuffer *command_buffer) {
    return submit_work_commands(r, allocator, command_buffer, true);
}

static bool submit_commands_async(RENDERER *r, NriCommandAllocator *allocator, NriCommandBuffer *command_buffer) {
    return submit_work_commands(r, allocator, command_buffer, false);
}

void gpu_abort_commands(RENDERER *r, NriCommandAllocator *allocator, NriCommandBuffer *cmd) {
    abort_work_commands(r, allocator, cmd);
}

/* Descriptor sets match the HLSL register spaces in shaders/. */
bool gpu_create_pipeline_layout(RENDERER *r, NriPipelineLayout **out, const NriDescriptorType *types[4], const uint8_t counts[4], NriStageBits stages) {
    enum { MAX_LAYOUT_RANGES = 32 };
    NriDescriptorRangeDesc ranges[4][MAX_LAYOUT_RANGES] = {0};
    NriDescriptorSetDesc sets[4] = {0};

    for (uint32_t set = 0; set < 4; ++set) {
        if (counts[set] > MAX_LAYOUT_RANGES) {
            SDL_SetError("pipeline layout set %u has %u ranges; maximum is %u", set, counts[set], MAX_LAYOUT_RANGES);
            return false;
        }

        sets[set].registerSpace = set;
        sets[set].ranges = ranges[set];
        sets[set].rangeNum = counts[set];

        for (uint32_t i = 0; i < counts[set]; ++i) {
            NriDescriptorType type = types[set][i];
            uint32_t reg = 0;

            for (uint32_t j = 0; j < i; ++j) {
                const NriDescriptorType previous = types[set][j];
                const bool sampled_namespace = (type == NriDescriptorType_TEXTURE || type == NriDescriptorType_STRUCTURED_BUFFER) &&
                                               (previous == NriDescriptorType_TEXTURE || previous == NriDescriptorType_STRUCTURED_BUFFER);

                const bool storage_namespace = (type == NriDescriptorType_STORAGE_TEXTURE || type == NriDescriptorType_STORAGE_STRUCTURED_BUFFER) &&
                                               (previous == NriDescriptorType_STORAGE_TEXTURE || previous == NriDescriptorType_STORAGE_STRUCTURED_BUFFER);

                if (previous == type || sampled_namespace || storage_namespace) ++reg;
            }

            ranges[set][i] = (NriDescriptorRangeDesc){.baseRegisterIndex = reg, .descriptorNum = 1, .descriptorType = type, .shaderStages = stages};
        }
    }

    const NriPipelineLayoutDesc desc = {.descriptorSets = sets, .descriptorSetNum = 4, .rootRegisterSpace = 4, .shaderStages = stages};

    return r->gpu->core.CreatePipelineLayout(r->gpu->device, &desc, out) == NriResult_SUCCESS;
}

bool gpu_create_compute_layout(RENDERER *r, NriPipelineLayout **out, const NriDescriptorType *sources, uint8_t source_num, NriDescriptorType output_type,
                               bool has_uniform) {
    static const NriDescriptorType output_texture[] = {NriDescriptorType_STORAGE_TEXTURE};

    static const NriDescriptorType output_buffer[] = {NriDescriptorType_STORAGE_STRUCTURED_BUFFER};

    static const NriDescriptorType uniform[] = {NriDescriptorType_CONSTANT_BUFFER};

    const NriDescriptorType *sets[4] = {sources, output_type == NriDescriptorType_STORAGE_TEXTURE ? output_texture : output_buffer,
                                        has_uniform ? uniform : NULL, NULL};

    const uint8_t counts[4] = {source_num, 1, has_uniform ? 1 : 0, 0};

    return gpu_create_pipeline_layout(r, out, sets, counts, NriStageBits_COMPUTE_SHADER);
}

static bool create_descriptor_pool_object(RENDERER *r, NriDescriptorPool **pool) {
    const NriDescriptorPoolDesc desc = {.descriptorSetMaxNum = 8192,
                                        .samplerMaxNum = 8192,
                                        .textureMaxNum = 8192,
                                        .storageTextureMaxNum = 8192,
                                        .structuredBufferMaxNum = 8192,
                                        .storageStructuredBufferMaxNum = 8192,
                                        .constantBufferMaxNum = 8192};

    return r->gpu->core.CreateDescriptorPool(r->gpu->device, &desc, pool) == NriResult_SUCCESS;
}

static bool create_descriptor_pool(RENDERER *r) {
    return create_descriptor_pool_object(r, &r->gpu->descriptor_pool);
}

static bool track_descriptor_array(RENDERER *r, NriDescriptor ***items, uint32_t *count, uint32_t *capacity, NriDescriptor *descriptor) {
    if (!descriptor) return false;

    if (*count == *capacity) {
        uint32_t cap = *capacity ? *capacity * 2u : 64u;
        NriDescriptor **data = realloc(*items, cap * sizeof(*data));

        if (!data) {
            r->gpu->core.DestroyDescriptor(descriptor);

            return false;
        }
        *items = data;
        *capacity = cap;
    }

    (*items)[(*count)++] = descriptor;

    return true;
}

static bool track_buffer_array(RENDERER *r, NriBuffer ***items, uint32_t *count, uint32_t *capacity, NriBuffer *buffer) {
    if (!buffer) return false;

    if (*count == *capacity) {
        uint32_t cap = *capacity ? *capacity * 2u : 64u;
        NriBuffer **data = realloc(*items, cap * sizeof(*data));

        if (!data) {
            r->gpu->core.DestroyBuffer(buffer);

            return false;
        }
        *items = data;
        *capacity = cap;
    }

    (*items)[(*count)++] = buffer;

    return true;
}

static bool track_descriptor(RENDERER *r, NriDescriptor *descriptor) {
    if (r->gpu->active_frame) {
        FRAME_CONTEXT *frame = r->gpu->active_frame;

        return track_descriptor_array(r, &frame->temporary_descriptors, &frame->temporary_descriptor_num, &frame->temporary_descriptor_cap, descriptor);
    }

    return track_descriptor_array(r, &r->gpu->temporary_descriptors, &r->gpu->temporary_descriptor_num, &r->gpu->temporary_descriptor_cap, descriptor);
}

static bool track_buffer(RENDERER *r, NriBuffer *buffer) {
    if (r->gpu->active_frame) {
        FRAME_CONTEXT *frame = r->gpu->active_frame;

        return track_buffer_array(r, &frame->temporary_buffers, &frame->temporary_buffer_num, &frame->temporary_buffer_cap, buffer);
    }

    return track_buffer_array(r, &r->gpu->temporary_buffers, &r->gpu->temporary_buffer_num, &r->gpu->temporary_buffer_cap, buffer);
}

void gpu_clear_temporary(RENDERER *r) {
    for (uint32_t i = 0; i < r->gpu->temporary_descriptor_num; ++i) r->gpu->core.DestroyDescriptor(r->gpu->temporary_descriptors[i]);

    for (uint32_t i = 0; i < r->gpu->temporary_buffer_num; ++i) r->gpu->core.DestroyBuffer(r->gpu->temporary_buffers[i]);
    r->gpu->temporary_descriptor_num = r->gpu->temporary_buffer_num = 0;

    if (r->gpu->descriptor_pool) r->gpu->core.ResetDescriptorPool(r->gpu->descriptor_pool);
}

static void clear_frame_temporary(RENDERER *r, FRAME_CONTEXT *frame) {
    if (!frame) return;

    for (uint32_t i = 0; i < frame->temporary_descriptor_num; ++i) r->gpu->core.DestroyDescriptor(frame->temporary_descriptors[i]);

    for (uint32_t i = 0; i < frame->temporary_buffer_num; ++i) r->gpu->core.DestroyBuffer(frame->temporary_buffers[i]);
    frame->temporary_descriptor_num = 0;
    frame->temporary_buffer_num = 0;
    frame->uniform_offset = 0u;

    if (frame->descriptor_pool) r->gpu->core.ResetDescriptorPool(frame->descriptor_pool);
}

static bool create_uniform_ring(RENDERER *r, FRAME_CONTEXT *context) {
    if (!r || !context) return false;

    const NriBufferDesc desc = {.size = UNIFORM_RING_BYTES, .usage = NriBufferUsageBits_CONSTANT};

    return r->gpu->core.CreateCommittedBuffer(r->gpu->device, NriMemoryLocation_HOST_UPLOAD, 0.0f, &desc, &context->uniform_buffer) == NriResult_SUCCESS;
}

static bool create_work_contexts(RENDERER *r) {
    if (!r || !r->gpu->device || !r->gpu->work_queue) return false;

    r->gpu->work_contexts = calloc(WORK_QUEUE_DEPTH, sizeof(*r->gpu->work_contexts));

    if (!r->gpu->work_contexts) return false;

    if (r->gpu->core.CreateFence(r->gpu->device, 0u, &r->gpu->work_fence) != NriResult_SUCCESS) return false;

    for (uint32_t i = 0; i < WORK_QUEUE_DEPTH; ++i) {
        FRAME_CONTEXT *work = &r->gpu->work_contexts[i];

        if (!create_descriptor_pool_object(r, &work->descriptor_pool) || !create_uniform_ring(r, work) ||
            r->gpu->core.CreateCommandAllocator(r->gpu->work_queue, &work->allocator) != NriResult_SUCCESS ||
            r->gpu->core.CreateCommandBuffer(work->allocator, &work->command_buffer) != NriResult_SUCCESS)
            return false;
    }

    r->gpu->work_next_fence = 1u;

    return true;
}

static void work_drain(RENDERER *r) {
    if (!r || !r->gpu->work_fence || r->gpu->work_next_fence <= 1u) return;
    r->gpu->core.Wait(r->gpu->work_fence, r->gpu->work_next_fence - 1u);
}

static void destroy_work_contexts(RENDERER *r) {
    if (!r) return;
    work_drain(r);

    if (r->gpu->work_contexts) {
        for (uint32_t i = 0; i < WORK_QUEUE_DEPTH; ++i) {
            FRAME_CONTEXT *work = &r->gpu->work_contexts[i];
            clear_frame_temporary(r, work);

            if (work->command_buffer) r->gpu->core.DestroyCommandBuffer(work->command_buffer);

            if (work->allocator) r->gpu->core.DestroyCommandAllocator(work->allocator);

            if (work->descriptor_pool) r->gpu->core.DestroyDescriptorPool(work->descriptor_pool);

            if (work->uniform_buffer) r->gpu->core.DestroyBuffer(work->uniform_buffer);

            free(work->temporary_descriptors);
            free(work->temporary_buffers);
        }

        free(r->gpu->work_contexts);
    }

    r->gpu->work_contexts = NULL;
    r->gpu->active_work = NULL;

    if (r->gpu->work_fence) r->gpu->core.DestroyFence(r->gpu->work_fence);
    r->gpu->work_fence = NULL;
}

static bool begin_work_commands(RENDERER *r, NriCommandAllocator **allocator, NriCommandBuffer **command_buffer) {
    if (!r || !r->gpu->work_contexts || !r->gpu->work_fence || !r->gpu->work_queue || !allocator || !command_buffer) return false;

    FRAME_CONTEXT *work = &r->gpu->work_contexts[r->gpu->work_index % WORK_QUEUE_DEPTH];

    if (work->fence_value && r->gpu->core.GetFenceValue(r->gpu->work_fence) < work->fence_value) r->gpu->core.Wait(r->gpu->work_fence, work->fence_value);

    clear_frame_temporary(r, work);
    r->gpu->core.ResetCommandAllocator(work->allocator);
    r->current_graphics_layout = r->current_compute_layout = NULL;
    r->gpu->active_work = work;
    r->gpu->active_frame = work;

    if (r->gpu->core.BeginCommandBuffer(work->command_buffer, work->descriptor_pool) != NriResult_SUCCESS) {
        r->gpu->active_work = NULL;
        r->gpu->active_frame = NULL;

        return false;
    }

    *allocator = work->allocator;
    *command_buffer = work->command_buffer;
    return true;
}

static bool submit_work_commands(RENDERER *r, NriCommandAllocator *allocator, NriCommandBuffer *command_buffer, bool wait) {
    FRAME_CONTEXT *work = r ? r->gpu->active_work : NULL;

    if (!r || !work || work->allocator != allocator || work->command_buffer != command_buffer) return false;

    bool good = r->gpu->core.EndCommandBuffer(command_buffer) == NriResult_SUCCESS;
    uint64_t value = 0u;

    if (good) {
        value = r->gpu->work_next_fence++;
        const NriFenceSubmitDesc signal = {.fence = r->gpu->work_fence, .value = value};
        NriFenceSubmitDesc upload_wait = {0};
        uint32_t upload_wait_num = 0u;

        if (r->gpu->upload && r->gpu->upload->fence && r->gpu->upload->next_fence_value > 1u) {
            upload_wait = (NriFenceSubmitDesc){.fence = r->gpu->upload->fence, .value = r->gpu->upload->next_fence_value - 1u, .stages = NriStageBits_ALL};
            upload_wait_num = 1u;
        }

        const NriQueueSubmitDesc submit = {.waitFences = upload_wait_num ? &upload_wait : NULL,
                                           .waitFenceNum = upload_wait_num,
                                           .commandBuffers = (const NriCommandBuffer *const *)&command_buffer,
                                           .commandBufferNum = 1u,
                                           .signalFences = &signal,
                                           .signalFenceNum = 1u};

        good = r->gpu->core.QueueSubmit(r->gpu->work_queue, &submit) == NriResult_SUCCESS;
    }

    if (good) {
        work->fence_value = value;
        r->gpu->work_index++;
    }

    r->gpu->active_work = NULL;
    r->gpu->active_frame = NULL;

    if (good && wait) r->gpu->core.Wait(r->gpu->work_fence, value);

    return good;
}

static void abort_work_commands(RENDERER *r, NriCommandAllocator *allocator, NriCommandBuffer *command_buffer) {
    if (!r) return;

    FRAME_CONTEXT *work = r->gpu->active_work;

    if (!work || work->allocator != allocator || work->command_buffer != command_buffer) return;

    if (work->command_buffer) r->gpu->core.DestroyCommandBuffer(work->command_buffer);

    if (work->allocator) r->gpu->core.DestroyCommandAllocator(work->allocator);

    work->command_buffer = NULL;
    work->allocator = NULL;
    clear_frame_temporary(r, work);
    r->gpu->active_work = NULL;
    r->gpu->active_frame = NULL;

    if (r->gpu->core.CreateCommandAllocator(r->gpu->work_queue, &work->allocator) == NriResult_SUCCESS)
        r->gpu->core.CreateCommandBuffer(work->allocator, &work->command_buffer);
}

static bool create_frame_contexts(RENDERER *r) {
    r->gpu->frame_contexts = calloc(FRAME_QUEUE_DEPTH, sizeof(*r->gpu->frame_contexts));

    if (!r->gpu->frame_contexts) return false;

    if (r->gpu->core.CreateFence(r->gpu->device, 0, &r->gpu->frame_fence) != NriResult_SUCCESS) return false;

    for (uint32_t i = 0; i < FRAME_QUEUE_DEPTH; ++i) {
        FRAME_CONTEXT *frame = &r->gpu->frame_contexts[i];

        if (!create_descriptor_pool_object(r, &frame->descriptor_pool) || !create_uniform_ring(r, frame) ||
            r->gpu->core.CreateCommandAllocator(r->gpu->graphics_queue, &frame->allocator) != NriResult_SUCCESS ||
            r->gpu->core.CreateCommandBuffer(frame->allocator, &frame->command_buffer) != NriResult_SUCCESS)
            return false;
    }

    return true;
}

static void destroy_frame_contexts(RENDERER *r) {
    if (r->gpu->frame_contexts) {
        for (uint32_t i = 0; i < FRAME_QUEUE_DEPTH; ++i) {
            FRAME_CONTEXT *frame = &r->gpu->frame_contexts[i];
            clear_frame_temporary(r, frame);

            if (frame->command_buffer) r->gpu->core.DestroyCommandBuffer(frame->command_buffer);

            if (frame->allocator) r->gpu->core.DestroyCommandAllocator(frame->allocator);

            if (frame->descriptor_pool) r->gpu->core.DestroyDescriptorPool(frame->descriptor_pool);

            if (frame->uniform_buffer) r->gpu->core.DestroyBuffer(frame->uniform_buffer);
            free(frame->temporary_descriptors);
            free(frame->temporary_buffers);
        }

        free(r->gpu->frame_contexts);
    }

    r->gpu->frame_contexts = NULL;
    r->gpu->active_frame = NULL;

    if (r->gpu->frame_fence) r->gpu->core.DestroyFence(r->gpu->frame_fence);
    r->gpu->frame_fence = NULL;
}

static bool begin_frame_commands(RENDERER *r, FRAME_CONTEXT **out_frame, NriCommandBuffer **out_command_buffer) {
    if (!r || !r->gpu->frame_contexts || !r->gpu->frame_fence || !out_frame || !out_command_buffer) return false;

    const uint64_t wait_value = r->gpu->frame_index >= FRAME_QUEUE_DEPTH ? 1u + r->gpu->frame_index - FRAME_QUEUE_DEPTH : 0u;

    r->gpu->core.Wait(r->gpu->frame_fence, wait_value);

    FRAME_CONTEXT *frame = &r->gpu->frame_contexts[r->gpu->frame_index % FRAME_QUEUE_DEPTH];
    clear_frame_temporary(r, frame);
    r->gpu->core.ResetCommandAllocator(frame->allocator);
    r->current_graphics_layout = r->current_compute_layout = NULL;
    r->gpu->active_frame = frame;

    if (r->gpu->core.BeginCommandBuffer(frame->command_buffer, frame->descriptor_pool) != NriResult_SUCCESS) {
        r->gpu->active_frame = NULL;

        return false;
    }

    *out_frame = frame;
    *out_command_buffer = frame->command_buffer;
    return true;
}

static void abort_frame_commands(RENDERER *r, FRAME_CONTEXT *frame) {
    if (!r || !frame) return;

    if (frame->command_buffer) r->gpu->core.DestroyCommandBuffer(frame->command_buffer);

    if (frame->allocator) r->gpu->core.DestroyCommandAllocator(frame->allocator);
    frame->command_buffer = NULL;
    frame->allocator = NULL;
    r->gpu->active_frame = NULL;
    clear_frame_temporary(r, frame);

    if (r->gpu->core.CreateCommandAllocator(r->gpu->graphics_queue, &frame->allocator) == NriResult_SUCCESS)
        r->gpu->core.CreateCommandBuffer(frame->allocator, &frame->command_buffer);
}

NriDescriptor *gpu_create_texture_view(RENDERER *r, NriTexture *texture, NriTextureView type) {
    if (!texture) return NULL;

    NriDescriptor *view = NULL;

    const NriTextureViewDesc desc = {
        .texture = texture, .type = type, .format = r->gpu->core.GetTextureDesc(texture)->format, .mipNum = 1, .layerNum = 1, .sliceNum = 1};

    if (r->gpu->core.CreateTextureView(&desc, &view) != NriResult_SUCCESS) return NULL;

    return track_descriptor(r, view) ? view : NULL;
}

NriDescriptor *gpu_create_buffer_view(RENDERER *r, NriBuffer *buffer, NriBufferView type, uint32_t stride) {
    if (!buffer) return NULL;

    NriDescriptor *view = NULL;

    const NriBufferViewDesc desc = {.buffer = buffer, .type = type, .offset = 0, .size = r->gpu->core.GetBufferDesc(buffer)->size, .structureStride = stride};

    if (r->gpu->core.CreateBufferView(&desc, &view) != NriResult_SUCCESS) return NULL;

    return track_descriptor(r, view) ? view : NULL;
}

static NriDescriptor *uniform_view(RENDERER *r, const void *data, size_t size) {
    if (!r || !data || !size) return NULL;

    FRAME_CONTEXT *context = r->gpu->active_frame;
    const NriDeviceDesc *device = r->gpu->core.GetDeviceDesc(r->gpu->device);
    uint64_t alignment = device ? device->memoryAlignment.constantBufferOffset : 256u;

    if (alignment < 16u) alignment = 16u;

    const uint64_t view_size = ((uint64_t)size + alignment - 1u) / alignment * alignment;

    if (context && context->uniform_buffer) {
        const uint64_t offset = (context->uniform_offset + alignment - 1u) / alignment * alignment;

        if (offset + view_size <= UNIFORM_RING_BYTES) {
            void *mapped = r->gpu->core.MapBuffer(context->uniform_buffer, offset, view_size);

            if (mapped) {
                memset(mapped, 0, view_size);
                memcpy(mapped, data, size);
                r->gpu->core.UnmapBuffer(context->uniform_buffer);

                NriDescriptor *view = NULL;
                const NriBufferViewDesc desc = {.buffer = context->uniform_buffer, .type = NriBufferView_CONSTANT_BUFFER, .offset = offset, .size = view_size};

                if (r->gpu->core.CreateBufferView(&desc, &view) == NriResult_SUCCESS && track_descriptor(r, view)) {
                    context->uniform_offset = offset + view_size;

                    return view;
                }
            }
        }
    }

    const NriBufferDesc desc = {.size = view_size, .usage = NriBufferUsageBits_CONSTANT};
    NriBuffer *buffer = NULL;

    if (r->gpu->core.CreateCommittedBuffer(r->gpu->device, NriMemoryLocation_HOST_UPLOAD, 0.0f, &desc, &buffer) != NriResult_SUCCESS) return NULL;

    if (!track_buffer(r, buffer)) return NULL;

    void *mapped = r->gpu->core.MapBuffer(buffer, 0, desc.size);

    if (!mapped) return NULL;
    memset(mapped, 0, desc.size);
    memcpy(mapped, data, size);
    r->gpu->core.UnmapBuffer(buffer);

    return gpu_create_buffer_view(r, buffer, NriBufferView_CONSTANT_BUFFER, 0);
}

bool gpu_bind_descriptor_set(RENDERER *r, NriCommandBuffer *cmd, NriPipelineLayout *layout, NriBindPoint point, uint32_t set_index,
                             NriDescriptor *const *descriptors, uint32_t count) {
    NriDescriptorSet *set = NULL;
    NriDescriptorPool *pool = r->gpu->active_frame ? r->gpu->active_frame->descriptor_pool : r->gpu->descriptor_pool;

    if (r->gpu->core.AllocateDescriptorSets(pool, layout, set_index, &set, 1, 0) != NriResult_SUCCESS) return false;

    for (uint32_t i = 0; i < count; ++i) {
        if (!descriptors[i]) return false;

        const NriDescriptor *d = descriptors[i];

        const NriUpdateDescriptorRangeDesc update = {.descriptorSet = set, .rangeIndex = i, .descriptors = &d, .descriptorNum = 1};

        r->gpu->core.UpdateDescriptorRanges(&update, 1);
    }

    NriPipelineLayout **current = point == NriBindPoint_GRAPHICS ? &r->current_graphics_layout : &r->current_compute_layout;

    if (*current != layout) {
        r->gpu->core.CmdSetPipelineLayout(cmd, point, layout);
        *current = layout;
    }

    r->gpu->core.CmdSetDescriptorSet(cmd, &(NriSetDescriptorSetDesc){.setIndex = set_index, .descriptorSet = set, .bindPoint = point});

    return true;
}

bool gpu_bind_uniform_data(RENDERER *r, NriCommandBuffer *cmd, NriPipelineLayout *layout, NriBindPoint point, uint32_t set, const void *data, size_t size) {
    NriDescriptor *view = uniform_view(r, data, size);

    return view && gpu_bind_descriptor_set(r, cmd, layout, point, set, &view, 1);
}

bool gpu_transition_texture(RENDERER *r, NriCommandBuffer *cmd, NriTexture *texture, NriAccessBits access, NriLayout layout, NriStageBits stages);

void gpu_free_shader(NriShaderDesc *shader) {
    if (shader && shader->bytecode) SDL_free((void *)shader->bytecode);

    if (shader) *shader = (NriShaderDesc){0};
}

static TEXTURE_STATE *find_texture_state(RENDERER *r, NriTexture *texture);
static bool create_swapchain(RENDERER *r, uint32_t width, uint32_t height) {
    if (!r->gpu->window || !width || !height) return false;

    NriWindow window = {0};
    const SDL_PropertiesID props = SDL_GetWindowProperties(r->gpu->window);
#if defined(__APPLE__)
    (void)props;

    if (!r->gpu->metal_view) r->gpu->metal_view = SDL_Metal_CreateView(r->gpu->window);

    if (!r->gpu->metal_view) return false;
    window.metal.caMetalLayer = SDL_Metal_GetLayer(r->gpu->metal_view);
#elif defined(_WIN32)
    window.windows.hwnd = SDL_GetPointerProperty(props, SDL_PROP_WINDOW_WIN32_HWND_POINTER, NULL);
#else
    window.wayland.display = SDL_GetPointerProperty(props, SDL_PROP_WINDOW_WAYLAND_DISPLAY_POINTER, NULL);

    if (window.wayland.display) {
        window.wayland.surface = SDL_GetPointerProperty(props, SDL_PROP_WINDOW_WAYLAND_SURFACE_POINTER, NULL);
    } else {
        window.x11.dpy = SDL_GetPointerProperty(props, SDL_PROP_WINDOW_X11_DISPLAY_POINTER, NULL);
        window.x11.window = SDL_GetNumberProperty(props, SDL_PROP_WINDOW_X11_WINDOW_NUMBER, 0);

        if (!window.x11.dpy || !window.x11.window) return false;
    }
#endif
    const NriSwapChainDesc desc = {.window = window,
                                   .queue = r->gpu->graphics_queue,
                                   .width = (NriDim_t)width,
                                   .height = (NriDim_t)height,
                                   .textureNum = FRAME_QUEUE_DEPTH + 1u,
                                   .format = NriSwapChainFormat_BT709_G22_8BIT,
                                   .flags = NriSwapChainBits_VSYNC,
                                   .queuedFrameNum = FRAME_QUEUE_DEPTH};

    if (r->gpu->swapchain_api.CreateSwapChain(r->gpu->device, &desc, &r->gpu->swapchain) != NriResult_SUCCESS) return false;

    uint32_t count = 0;

    NriTexture *const *textures = r->gpu->swapchain_api.GetSwapChainTextures(r->gpu->swapchain, &count);

    if (!textures || !count) return false;
    r->gpu->swapchain_textures = calloc(count, sizeof(*r->gpu->swapchain_textures));
    r->gpu->swapchain_frames = calloc(count, sizeof(*r->gpu->swapchain_frames));

    if (!r->gpu->swapchain_textures || !r->gpu->swapchain_frames) return false;
    r->gpu->swapchain_texture_count = count;
    r->gpu->swapchain_format = r->gpu->core.GetTextureDesc(textures[0])->format;
    r->gpu->swapchain_width = width;
    r->gpu->swapchain_height = height;

    for (uint32_t i = 0; i < count; ++i) {
        r->gpu->swapchain_textures[i] = textures[i];

        TEXTURE_STATE *state = find_texture_state(r, textures[i]);

        if (!state) return false;
        state->state = (NriAccessLayoutStage){.layout = NriLayout_UNDEFINED, .stages = NriStageBits_NONE};

        SWAPCHAIN_TEXTURE *frame = &r->gpu->swapchain_frames[i];
        frame->texture = textures[i];

        const NriTextureViewDesc view = {
            .texture = textures[i], .type = NriTextureView_COLOR_ATTACHMENT, .format = r->gpu->swapchain_format, .mipNum = 1, .layerNum = 1, .sliceNum = 1};

        if (r->gpu->core.CreateTextureView(&view, &frame->color_attachment) != NriResult_SUCCESS ||
            r->gpu->core.CreateFence(r->gpu->device, NRI_SWAPCHAIN_SEMAPHORE, &frame->acquire) != NriResult_SUCCESS ||
            r->gpu->core.CreateFence(r->gpu->device, NRI_SWAPCHAIN_SEMAPHORE, &frame->release) != NriResult_SUCCESS)
            return false;
    }

    return true;
}

static void destroy_swapchain(RENDERER *r) {
    for (uint32_t i = 0; i < r->gpu->swapchain_texture_count; ++i) {
        for (uint32_t j = 0; j < r->gpu->texture_state_num; ++j)
            if (r->gpu->texture_states[j].texture == r->gpu->swapchain_textures[i]) {
                r->gpu->texture_states[j] = r->gpu->texture_states[--r->gpu->texture_state_num];

                break;
            }

        SWAPCHAIN_TEXTURE *frame = &r->gpu->swapchain_frames[i];

        if (frame->color_attachment) r->gpu->core.DestroyDescriptor(frame->color_attachment);

        if (frame->acquire) r->gpu->core.DestroyFence(frame->acquire);

        if (frame->release) r->gpu->core.DestroyFence(frame->release);
    }

    free(r->gpu->swapchain_frames);
    free(r->gpu->swapchain_textures);
    r->gpu->swapchain_frames = NULL;
    r->gpu->swapchain_textures = NULL;
    r->gpu->swapchain_texture_count = 0;

    if (r->gpu->swapchain) r->gpu->swapchain_api.DestroySwapChain(r->gpu->swapchain);
    r->gpu->swapchain = NULL;
}

static bool acquire_swapchain_texture(RENDERER *r, uint32_t *index) {
    NriFence *acquire = r->gpu->swapchain_frames[r->gpu->frame_index % r->gpu->swapchain_texture_count].acquire;

    NriResult result = r->gpu->swapchain_api.AcquireNextTexture(r->gpu->swapchain, acquire, index);

    return result == NriResult_SUCCESS && *index < r->gpu->swapchain_texture_count;
}

static TEXTURE_STATE *find_texture_state(RENDERER *r, NriTexture *texture) {
    for (uint32_t i = 0; i < r->gpu->texture_state_num; ++i)
        if (r->gpu->texture_states[i].texture == texture) return &r->gpu->texture_states[i];

    if (r->gpu->texture_state_num == r->gpu->texture_state_cap) {
        uint32_t cap = r->gpu->texture_state_cap ? r->gpu->texture_state_cap * 2 : 32;
        TEXTURE_STATE *items = realloc(r->gpu->texture_states, cap * sizeof(*items));

        if (!items) return NULL;
        r->gpu->texture_states = items;
        r->gpu->texture_state_cap = cap;
    }

    TEXTURE_STATE *item = &r->gpu->texture_states[r->gpu->texture_state_num++];
    *item = (TEXTURE_STATE){.texture = texture};

    return item;
}

bool gpu_texture_barrier(RENDERER *r, NriCommandBuffer *cmd, NriTexture *texture, NriAccessLayoutStage before, NriAccessLayoutStage after) {
    TEXTURE_STATE *item = find_texture_state(r, texture);

    if (!item) return false;

    if (item->state.layout) before = item->state;

    const NriTextureBarrierDesc barrier = {.texture = texture, .before = before, .after = after, .mipNum = 1, .layerNum = 1};

    r->gpu->core.CmdBarrier(cmd, &(NriBarrierDesc){.textures = &barrier, .textureNum = 1});
    item->state = after;

    return true;
}

bool gpu_transition_texture(RENDERER *r, NriCommandBuffer *cmd, NriTexture *texture, NriAccessBits access, NriLayout layout, NriStageBits stages) {
    if (!texture) return false;

    return gpu_texture_barrier(r, cmd, texture, (NriAccessLayoutStage){0}, (NriAccessLayoutStage){.access = access, .layout = layout, .stages = stages});
}

static bool submit_frame(RENDERER *r, FRAME_CONTEXT *frame, NriCommandBuffer *cmd, uint32_t index) {
    bool good = frame && gpu_texture_barrier(r, cmd, r->gpu->swapchain_textures[index],
                                             (NriAccessLayoutStage){.access = NriAccessBits_COLOR_ATTACHMENT,
                                                                    .layout = NriLayout_COLOR_ATTACHMENT,
                                                                    .stages = NriStageBits_COLOR_ATTACHMENT},
                                             (NriAccessLayoutStage){.layout = NriLayout_PRESENT, .stages = NriStageBits_NONE});
    if (good) good = r->gpu->core.EndCommandBuffer(cmd) == NriResult_SUCCESS;

    const uint64_t frame_value = 1u + r->gpu->frame_index;
    NriFenceSubmitDesc waits[2] = {
        {.fence = r->gpu->swapchain_frames[r->gpu->frame_index % r->gpu->swapchain_texture_count].acquire, .stages = NriStageBits_COLOR_ATTACHMENT}, {0}};
    uint32_t wait_num = 1u;

    if (r->gpu->upload && r->gpu->upload->fence && r->gpu->upload->next_fence_value > 1u) {
        waits[wait_num++] = (NriFenceSubmitDesc){.fence = r->gpu->upload->fence, .value = r->gpu->upload->next_fence_value - 1u, .stages = NriStageBits_ALL};
    }

    const NriFenceSubmitDesc signals[2] = {{.fence = r->gpu->swapchain_frames[index].release}, {.fence = r->gpu->frame_fence, .value = frame_value}};

    const NriQueueSubmitDesc submit = {.waitFences = waits,
                                       .waitFenceNum = wait_num,
                                       .commandBuffers = (const NriCommandBuffer *const *)&cmd,
                                       .commandBufferNum = 1,
                                       .signalFences = signals,
                                       .signalFenceNum = 2};

    bool submitted = false;

    if (good) {
        submitted = r->gpu->core.QueueSubmit(r->gpu->graphics_queue, &submit) == NriResult_SUCCESS;

        good = submitted;
    }

    if (good) good = r->gpu->swapchain_api.QueuePresent(r->gpu->swapchain, r->gpu->swapchain_frames[index].release, frame_value) == NriResult_SUCCESS;

    r->gpu->active_frame = NULL;

    if (submitted)
        r->gpu->frame_index++;
    else if (!good)
        abort_frame_commands(r, frame);

    return good;
}

NriDescriptor *gpu_create_sampler(RENDERER *r, NriFilter min_filter, NriFilter mag_filter, NriAddressMode address) {
    const NriSamplerDesc desc = {.filters = {.min = min_filter, .mag = mag_filter, .mip = NriFilter_NEAREST},
                                 .addressModes = {address, address, address},
                                 .mipMin = 0.0f,
                                 .mipMax = 16.0f};

    NriDescriptor *result = NULL;

    if (r->gpu->core.CreateSampler(r->gpu->device, &desc, &result) != NriResult_SUCCESS) return NULL;

    return result;
}

NriTexture *gpu_create_texture(RENDERER *r, NriFormat format, NriTextureUsageBits usage, Uint32 width, Uint32 height) {
    if (!r || !r->gpu->device || !width || !height) return NULL;

    const NriTextureDesc desc = {.type = NriTextureType_TEXTURE_2D,
                                 .usage = usage,
                                 .format = format,
                                 .width = (NriDim_t)width,
                                 .height = (NriDim_t)height,
                                 .depth = 1,
                                 .mipNum = 1,
                                 .layerNum = 1,
                                 .sampleNum = 1};

    NriTexture *result = NULL;

    if (r->gpu->core.CreateCommittedTexture(r->gpu->device, NriMemoryLocation_DEVICE, 1.0f, &desc, &result) != NriResult_SUCCESS) return NULL;

    if (!find_texture_state(r, result)) {
        release_texture(r, result);

        return NULL;
    }

    return result;
}

static uint64_t upload_align(uint64_t value, uint64_t alignment) {
    if (!alignment) return value;

    return (value + alignment - 1u) / alignment * alignment;
}

static void destroy_upload_context(RENDERER *r) {
    if (!r || !r->gpu->upload) return;

    UPLOAD_CONTEXT *upload = r->gpu->upload;

    for (uint32_t i = 0; i < UPLOAD_RING_SIZE; ++i) {
        UPLOAD_SLOT *slot = &upload->slots[i];

        if (slot->command_buffer) r->gpu->core.DestroyCommandBuffer(slot->command_buffer);

        if (slot->allocator) r->gpu->core.DestroyCommandAllocator(slot->allocator);

        if (slot->staging) r->gpu->core.DestroyBuffer(slot->staging);
    }

    if (upload->fence) r->gpu->core.DestroyFence(upload->fence);

    free(upload);
    r->gpu->upload = NULL;
}

static bool create_upload_context(RENDERER *r) {
    if (!r || !r->gpu->device || !r->gpu->graphics_queue) return false;

    if (r->gpu->upload) return true;

    UPLOAD_CONTEXT *upload = calloc(1, sizeof(*upload));

    if (!upload) return false;
    r->gpu->upload = upload;

    if (r->gpu->core.CreateFence(r->gpu->device, 0u, &upload->fence) != NriResult_SUCCESS) goto fail;

    const NriBufferDesc staging_desc = {.size = UPLOAD_CHUNK_BYTES, .usage = NriBufferUsageBits_NONE};

    for (uint32_t i = 0; i < UPLOAD_RING_SIZE; ++i) {
        UPLOAD_SLOT *slot = &upload->slots[i];

        if (r->gpu->core.CreateCommittedBuffer(r->gpu->device, NriMemoryLocation_HOST_UPLOAD, 0.0f, &staging_desc, &slot->staging) != NriResult_SUCCESS ||
            r->gpu->core.CreateCommandAllocator(r->gpu->graphics_queue, &slot->allocator) != NriResult_SUCCESS ||
            r->gpu->core.CreateCommandBuffer(slot->allocator, &slot->command_buffer) != NriResult_SUCCESS)
            goto fail;
    }

    upload->next_fence_value = 1u;

    return true;

fail:
    destroy_upload_context(r);

    return false;
}

static bool upload_wait_slot(RENDERER *r, UPLOAD_SLOT *slot) {
    if (!r || !r->gpu->upload || !slot || !slot->fence_value) return true;

    const Uint64 started = SDL_GetTicks();
    bool logged = false;

    while (r->gpu->core.GetFenceValue(r->gpu->upload->fence) < slot->fence_value) {
        if (!logged && SDL_GetTicks() - started >= UPLOAD_SLOW_LOG_MS) {
            SDL_Log("GPU upload chunk is still pending after %u ms; continuing "
                    "without NRI's hard fence timeout",
                    UPLOAD_SLOW_LOG_MS);
            logged = true;
        }

        SDL_Delay(1u);
    }

    slot->fence_value = 0u;

    return true;
}

static bool upload_begin_slot(RENDERER *r, UPLOAD_SLOT **out) {
    if (!out || !create_upload_context(r)) return false;

    UPLOAD_CONTEXT *upload = r->gpu->upload;
    UPLOAD_SLOT *slot = &upload->slots[upload->next_slot];
    upload->next_slot = (upload->next_slot + 1u) % UPLOAD_RING_SIZE;

    if (!upload_wait_slot(r, slot)) return false;
    r->gpu->core.ResetCommandAllocator(slot->allocator);

    if (r->gpu->core.BeginCommandBuffer(slot->command_buffer, NULL) != NriResult_SUCCESS) return false;

    *out = slot;
    return true;
}

static bool upload_submit_slot(RENDERER *r, UPLOAD_SLOT *slot) {
    if (!r || !r->gpu->upload || !slot) return false;

    if (r->gpu->core.EndCommandBuffer(slot->command_buffer) != NriResult_SUCCESS) return false;

    const uint64_t value = r->gpu->upload->next_fence_value++;

    const NriFenceSubmitDesc signal = {.fence = r->gpu->upload->fence, .value = value};

    NriCommandBuffer *command = slot->command_buffer;

    const NriQueueSubmitDesc submit = {
        .commandBuffers = (const NriCommandBuffer *const *)&command, .commandBufferNum = 1u, .signalFences = &signal, .signalFenceNum = 1u};

    if (r->gpu->core.QueueSubmit(r->gpu->graphics_queue, &submit) != NriResult_SUCCESS) return false;

    slot->fence_value = value;

    return true;
}

static bool upload_drain(RENDERER *r) {
    if (!r || !r->gpu->upload) return true;

    for (uint32_t i = 0; i < UPLOAD_RING_SIZE; ++i)
        if (!upload_wait_slot(r, &r->gpu->upload->slots[i])) return false;

    return true;
}

static NriAccessStage uploaded_buffer_state(NriBufferUsageBits usage) {
    NriAccessStage state = {0};

    if (usage & NriBufferUsageBits_SHADER_RESOURCE) {
        state.access |= NriAccessBits_SHADER_RESOURCE;
        state.stages |= NriStageBits_COMPUTE_SHADER | NriStageBits_FRAGMENT_SHADER;
    }

    if (usage & NriBufferUsageBits_VERTEX) {
        state.access |= NriAccessBits_VERTEX_BUFFER;
        state.stages |= NriStageBits_VERTEX_SHADER;
    }

    return state;
}

NriBuffer *gpu_upload_buffer(RENDERER *r, NriBufferUsageBits usage, const void *data, size_t bytes, uint32_t stride) {
    if (!r || !r->gpu->device || !r->gpu->graphics_queue || !data || !bytes) return NULL;

    const NriBufferDesc desc = {.size = bytes, .structureStride = stride, .usage = usage};

    NriBuffer *buffer = NULL;

    if (r->gpu->core.CreateCommittedBuffer(r->gpu->device, NriMemoryLocation_DEVICE, 1.0f, &desc, &buffer) != NriResult_SUCCESS) return NULL;

    const NriAccessStage copy_state = {.access = NriAccessBits_COPY_DESTINATION, .stages = NriStageBits_ALL};

    const NriAccessStage final_state = uploaded_buffer_state(usage);
    const Uint8 *source = data;

    size_t offset = 0u;

    while (offset < bytes) {
        const size_t chunk = bytes - offset > UPLOAD_CHUNK_BYTES ? UPLOAD_CHUNK_BYTES : bytes - offset;

        UPLOAD_SLOT *slot = NULL;

        if (!upload_begin_slot(r, &slot)) goto fail;

        void *mapped = r->gpu->core.MapBuffer(slot->staging, 0u, chunk);

        if (!mapped) {
            (void)r->gpu->core.EndCommandBuffer(slot->command_buffer);

            goto fail;
        }

        memcpy(mapped, source + offset, chunk);
        r->gpu->core.UnmapBuffer(slot->staging);

        if (!offset) {
            const NriBufferBarrierDesc barrier = {.buffer = buffer, .before = {0}, .after = copy_state};

            r->gpu->core.CmdBarrier(slot->command_buffer, &(NriBarrierDesc){.buffers = &barrier, .bufferNum = 1u});
        }

        r->gpu->core.CmdCopyBuffer(slot->command_buffer, buffer, offset, slot->staging, 0u, chunk);

        if (offset + chunk == bytes) {
            const NriBufferBarrierDesc barrier = {.buffer = buffer, .before = copy_state, .after = final_state};

            r->gpu->core.CmdBarrier(slot->command_buffer, &(NriBarrierDesc){.buffers = &barrier, .bufferNum = 1u});
        }

        if (!upload_submit_slot(r, slot)) goto fail;
        offset += chunk;
    }

    return buffer;

fail:
    (void)upload_drain(r);
    r->gpu->core.DestroyBuffer(buffer);

    return NULL;
}

bool gpu_clear_texture_zero(RENDERER *r, NriTexture *texture, uint32_t bytes_per_texel, NriAccessBits access, NriLayout layout, NriStageBits stages) {
    if (!r || !texture || !bytes_per_texel) return false;

    const NriTextureDesc *desc = r->gpu->core.GetTextureDesc(texture);
    const NriDeviceDesc *device = r->gpu->core.GetDeviceDesc(r->gpu->device);

    if (!desc || !device || !desc->width || !desc->height) return false;
    if ((uint64_t)desc->width * bytes_per_texel > UINT32_MAX) return false;

    const uint32_t row_pitch = (uint32_t)desc->width * bytes_per_texel;
    const uint64_t aligned_row = upload_align(row_pitch, device->memoryAlignment.uploadBufferTextureRow);

    if (!aligned_row || aligned_row > UPLOAD_CHUNK_BYTES) return false;

    const NriAccessLayoutStage copy_state = {.access = NriAccessBits_COPY_DESTINATION, .layout = NriLayout_COPY_DESTINATION, .stages = NriStageBits_ALL};
    const NriAccessLayoutStage final_state = {.access = access, .layout = layout, .stages = stages};

    uint32_t first_row = 0u;
    const uint32_t row_count = (uint32_t)desc->height;

    while (first_row < row_count) {
        uint32_t rows = (uint32_t)(UPLOAD_CHUNK_BYTES / aligned_row);
        const uint32_t remaining = row_count - first_row;

        if (!rows) rows = 1u;
        if (rows > remaining) rows = remaining;

        uint64_t staging_bytes = upload_align(aligned_row * rows, device->memoryAlignment.uploadBufferTextureSlice);

        while (rows > 1u && staging_bytes > UPLOAD_CHUNK_BYTES) {
            --rows;
            staging_bytes = upload_align(aligned_row * rows, device->memoryAlignment.uploadBufferTextureSlice);
        }

        if (staging_bytes > UPLOAD_CHUNK_BYTES) return false;

        UPLOAD_SLOT *slot = NULL;
        if (!upload_begin_slot(r, &slot)) goto fail;

        void *mapped = r->gpu->core.MapBuffer(slot->staging, 0u, staging_bytes);

        if (!mapped) {
            (void)r->gpu->core.EndCommandBuffer(slot->command_buffer);
            goto fail;
        }

        memset(mapped, 0, (size_t)staging_bytes);
        r->gpu->core.UnmapBuffer(slot->staging);

        if (!first_row && !gpu_texture_barrier(r, slot->command_buffer, texture, (NriAccessLayoutStage){0}, copy_state)) goto fail;

        const NriTextureDataLayoutDesc source_layout = {.offset = 0u, .rowPitch = (uint32_t)aligned_row, .slicePitch = (uint32_t)staging_bytes};
        const NriTextureRegionDesc region = {
            .x = 0u, .y = (NriDim_t)first_row, .z = 0u, .width = desc->width, .height = (NriDim_t)rows, .depth = 1u, .mipOffset = 0u, .layerOffset = 0u};

        r->gpu->core.CmdUploadBufferToTexture(slot->command_buffer, texture, &region, slot->staging, &source_layout);

        if (first_row + rows == row_count && !gpu_texture_barrier(r, slot->command_buffer, texture, copy_state, final_state)) goto fail;
        if (!upload_submit_slot(r, slot)) goto fail;

        first_row += rows;
    }

    return true;

fail:
    (void)upload_drain(r);
    return false;
}

bool gpu_upload_texture_data(RENDERER *r, NriTexture *texture, const void *data, uint32_t row_pitch, uint32_t slice_pitch, NriAccessBits access,
                             NriLayout layout, NriStageBits stages) {
    if (!r || !texture || !data || !row_pitch || !slice_pitch || slice_pitch % row_pitch) return false;

    const NriTextureDesc *desc = r->gpu->core.GetTextureDesc(texture);
    const NriDeviceDesc *device = r->gpu->core.GetDeviceDesc(r->gpu->device);
    const uint32_t row_alignment = device->memoryAlignment.uploadBufferTextureRow;
    const uint32_t slice_alignment = device->memoryAlignment.uploadBufferTextureSlice;

    const uint64_t aligned_row = upload_align(row_pitch, row_alignment);
    const uint32_t row_count = slice_pitch / row_pitch;

    if (!desc || !row_count || aligned_row > UPLOAD_CHUNK_BYTES) return false;

    const NriAccessLayoutStage copy_state = {.access = NriAccessBits_COPY_DESTINATION, .layout = NriLayout_COPY_DESTINATION, .stages = NriStageBits_ALL};

    const NriAccessLayoutStage final_state = {.access = access, .layout = layout, .stages = stages};

    const Uint8 *source = data;

    uint32_t first_row = 0u;

    while (first_row < row_count) {
        uint32_t rows = (uint32_t)(UPLOAD_CHUNK_BYTES / aligned_row);
        const uint32_t remaining = row_count - first_row;

        if (!rows) rows = 1u;

        if (rows > remaining) rows = remaining;

        uint64_t staging_bytes = upload_align(aligned_row * rows, slice_alignment);

        while (rows > 1u && staging_bytes > UPLOAD_CHUNK_BYTES) {
            --rows;

            staging_bytes = upload_align(aligned_row * rows, slice_alignment);
        }

        if (staging_bytes > UPLOAD_CHUNK_BYTES) return false;

        UPLOAD_SLOT *slot = NULL;

        if (!upload_begin_slot(r, &slot)) goto fail;

        Uint8 *mapped = r->gpu->core.MapBuffer(slot->staging, 0u, staging_bytes);

        if (!mapped) {
            (void)r->gpu->core.EndCommandBuffer(slot->command_buffer);

            goto fail;
        }

        for (uint32_t row = 0; row < rows; ++row) memcpy(mapped + (size_t)row * aligned_row, source + (size_t)(first_row + row) * row_pitch, row_pitch);
        r->gpu->core.UnmapBuffer(slot->staging);

        if (!first_row && !gpu_texture_barrier(r, slot->command_buffer, texture, (NriAccessLayoutStage){0}, copy_state)) goto fail;

        const NriTextureDataLayoutDesc source_layout = {.offset = 0u, .rowPitch = (uint32_t)aligned_row, .slicePitch = (uint32_t)staging_bytes};

        const NriTextureRegionDesc region = {
            .x = 0u, .y = (NriDim_t)first_row, .z = 0u, .width = desc->width, .height = (NriDim_t)rows, .depth = 1u, .mipOffset = 0u, .layerOffset = 0u};

        r->gpu->core.CmdUploadBufferToTexture(slot->command_buffer, texture, &region, slot->staging, &source_layout);

        if (first_row + rows == row_count && !gpu_texture_barrier(r, slot->command_buffer, texture, copy_state, final_state)) goto fail;

        if (!upload_submit_slot(r, slot)) goto fail;
        first_row += rows;
    }

    return true;

fail:
    (void)upload_drain(r);

    return false;
}

void release_texture(RENDERER *r, NriTexture *value) {
    if (!r || !value) return;

    for (uint32_t i = 0; i < r->gpu->texture_state_num; ++i) {
        if (r->gpu->texture_states[i].texture == value) {
            r->gpu->texture_states[i] = r->gpu->texture_states[--r->gpu->texture_state_num];

            break;
        }
    }

    r->gpu->core.DestroyTexture(value);
}

void release_buffer(RENDERER *r, NriBuffer *value) {
    if (r && value) r->gpu->core.DestroyBuffer(value);
}

/*
 * Pipeline layouts mirror the descriptor spaces used by the shaders.
 * These helpers are intentionally named by the shader job they describe.
 * Each one mirrors the register spaces already present in your HLSL.
 *
 * The binding helper implementations are kept in gpu.c as well so render.c
 * never learns about NRI descriptor sets/views.
 */
bool gpu_ensure_swapchain(RENDERER *r, uint32_t width, uint32_t height) {
    if (!r || !r->gpu || !width || !height) return false;

    if (r->gpu->swapchain && width == r->gpu->swapchain_width && height == r->gpu->swapchain_height) return true;

    if (r->gpu->graphics_queue && r->gpu->core.QueueWaitIdle(r->gpu->graphics_queue) != NriResult_SUCCESS) return false;

    destroy_swapchain(r);
    return create_swapchain(r, width, height);
}

bool gpu_begin_render_frame(RENDERER *r, FRAME_CONTEXT **frame, NriCommandBuffer **command_buffer, NriTexture **swapchain_texture, uint32_t *swapchain_index) {
    if (!r || !r->gpu || !frame || !command_buffer || !swapchain_texture || !swapchain_index) return false;

    uint32_t index = 0;

    if (!acquire_swapchain_texture(r, &index)) {
        destroy_swapchain(r);
        return false;
    }

    r->gpu->current_swap_index = index;

    if (!begin_frame_commands(r, frame, command_buffer)) {
        destroy_swapchain(r);
        return false;
    }

    *swapchain_texture = r->gpu->swapchain_textures[index];
    *swapchain_index = index;
    return true;
}

bool gpu_submit_render_frame(RENDERER *r, FRAME_CONTEXT *frame, NriCommandBuffer *command_buffer, uint32_t swapchain_index) {
    return submit_frame(r, frame, command_buffer, swapchain_index);
}

void gpu_abort_render_frame(RENDERER *r, FRAME_CONTEXT *frame) {
    if (!r) return;

    abort_frame_commands(r, frame);
    destroy_swapchain(r);
}

NriDescriptor *gpu_swapchain_color_attachment(RENDERER *r, uint32_t swapchain_index) {
    if (!r || !r->gpu || !r->gpu->swapchain_frames || swapchain_index >= r->gpu->swapchain_texture_count) return NULL;
    return r->gpu->swapchain_frames[swapchain_index].color_attachment;
}

bool gpu_init_worker(GPU *gpu) {
    if (!gpu) return false;

    memset(gpu, 0, sizeof(*gpu));

    NriDeviceCreationDesc device_desc = {0};
    device_desc.graphicsAPI = NriGraphicsAPI_VK;
    device_desc.vkBindingOffsets = (NriVKBindingOffsets){.sRegister = 0, .tRegister = 16, .bRegister = 32, .uRegister = 48};

    if (nriCreateDevice(&device_desc, &gpu->device) != NriResult_SUCCESS) {
        gpu_deinit(gpu);
        return false;
    }

    if (nriGetInterface(gpu->device, NRI_INTERFACE(NriCoreInterface), &gpu->core) != NriResult_SUCCESS ||
        nriGetInterface(gpu->device, NRI_INTERFACE(NriHelperInterface), &gpu->helper) != NriResult_SUCCESS) {
        gpu_deinit(gpu);
        return false;
    }

    RENDERER shell = {.gpu = gpu};

    if (!acquire_queues(&shell) || !create_pipeline_cache(&shell) || !create_gpu_timestamps(&shell) || !create_descriptor_pool(&shell) ||
        !create_work_contexts(&shell)) {
        gpu_deinit(gpu);
        return false;
    }

    return true;
}

bool gpu_init(GPU *gpu, const char *title, int width, int height) {
    if (!gpu || !title || width <= 0 || height <= 0) return false;

    memset(gpu, 0, sizeof(*gpu));
    gpu->window = SDL_CreateWindow(title, width, height,
                                   SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY
#if defined(__APPLE__)
                                       | SDL_WINDOW_METAL
#else
                                       | SDL_WINDOW_VULKAN
#endif
    );
    if (!gpu->window) return false;

    NriDeviceCreationDesc device_desc = {0};
    device_desc.graphicsAPI = NriGraphicsAPI_VK;
    device_desc.enableNRIValidation = false;
    device_desc.enableGraphicsAPIValidation = false;
    device_desc.vkBindingOffsets = (NriVKBindingOffsets){.sRegister = 0, .tRegister = 16, .bRegister = 32, .uRegister = 48};

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

    RENDERER shell = {.gpu = gpu};

    if (!acquire_queues(&shell) || !create_pipeline_cache(&shell) || !create_gpu_timestamps(&shell) || !create_swapchain(&shell, width, height) ||
        !create_descriptor_pool(&shell) || !create_work_contexts(&shell) || !create_frame_contexts(&shell)) {
        gpu_deinit(gpu);
        return false;
    }

    return true;
}

void gpu_deinit(GPU *gpu) {
    if (!gpu) return;

    RENDERER shell = {.gpu = gpu};

    if (gpu->device) {
        if (gpu->graphics_queue) gpu->core.QueueWaitIdle(gpu->graphics_queue);

        destroy_upload_context(&shell);
        destroy_work_contexts(&shell);
        gpu_clear_temporary(&shell);
        destroy_frame_contexts(&shell);
        destroy_gpu_timestamps(&shell);
        destroy_swapchain(&shell);

        if (gpu->descriptor_pool) gpu->core.DestroyDescriptorPool(gpu->descriptor_pool);
        destroy_pipeline_cache(&shell);
        nriDestroyDevice(gpu->device);
    }

    free(gpu->temporary_descriptors);
    free(gpu->temporary_buffers);
    free(gpu->texture_states);

#if defined(__APPLE__)
    if (gpu->metal_view) SDL_Metal_DestroyView(gpu->metal_view);
#endif

    if (gpu->window) SDL_DestroyWindow(gpu->window);
    memset(gpu, 0, sizeof(*gpu));
}
