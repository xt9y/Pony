#include "game.h"
#include "render_internal.h"

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BAKE_TARGET_SAMPLES 128u
#define BAKE_MAX_BOUNCES 3u
#define BAKE_BATCH_SAMPLES 8u
#define BAKE_DILATION_PASSES 3u
#define PHASE_CLEAR 0u
#define PHASE_TRACE 1u
#define PHASE_FILTER 2u
#define PHASE_DILATE 3u
#define PHASE_DIRECT 4u
#define PHASE_COMBINE 5u
#define PHASE_RECONSTRUCT 6u

static NriTexture *create_lightmap_texture(RENDERER *r, Uint32 width, Uint32 height) {
    return gpu_create_texture(r, NriFormat_RGBA16_SFLOAT, NriTextureUsageBits_SHADER_RESOURCE | NriTextureUsageBits_SHADER_RESOURCE_STORAGE, width, height);
}

static bool transfer_size(uint32_t width, uint32_t height, Uint32 *out) {
    const uint64_t bytes = (uint64_t)width * (uint64_t)height * 8u;

    if (!width || !height || bytes > UINT32_MAX) return false;
    *out = (Uint32)bytes;
    return true;
}

NriTexture *upload_lightmap(RENDERER *r, const CACHED_LIGHTMAP *cached) {
    if (!r || !r->gpu->device || !cached || !cached->pixels) return NULL;

    Uint32 bytes = 0;

    if (!transfer_size(cached->width, cached->height, &bytes)) return NULL;

    NriTexture *result = create_lightmap_texture(r, cached->width, cached->height);

    if (!result) return NULL;

    if (!gpu_upload_texture_data(r, result, cached->pixels, cached->width * 8u, bytes, NriAccessBits_SHADER_RESOURCE, NriLayout_SHADER_RESOURCE,
                                 NriStageBits_ALL)) {
        release_texture(r, result);

        return NULL;
    }

    return result;
}

static bool read_rgba16f_texture(RENDERER *r, NriTexture *texture, Uint32 width, Uint32 height, Uint8 **pixels) {
    if (!r || !r->gpu->device || !texture || !width || !height || !pixels) return false;
    *pixels = NULL;

    Uint32 tight_bytes = 0;

    if (!transfer_size(width, height, &tight_bytes)) return false;

    const uint32_t row_bytes = width * 8u;
    const NriDeviceDesc *device = r->gpu->core.GetDeviceDesc(r->gpu->device);
    const uint32_t row_alignment = device->memoryAlignment.uploadBufferTextureRow;
    const uint32_t slice_alignment = device->memoryAlignment.uploadBufferTextureSlice;

    if (!row_alignment || !slice_alignment) return false;

    const uint64_t row_pitch = ((uint64_t)row_bytes + row_alignment - 1u) / row_alignment * row_alignment;
    const uint64_t slice_bytes = row_pitch * height;
    const uint64_t staging_bytes = (slice_bytes + slice_alignment - 1u) / slice_alignment * slice_alignment;

    if (staging_bytes > UINT32_MAX) return false;

    const NriBufferDesc desc = {.size = staging_bytes, .usage = NriBufferUsageBits_NONE};

    NriBuffer *readback = NULL;

    if (r->gpu->core.CreateCommittedBuffer(r->gpu->device, NriMemoryLocation_HOST_READBACK, 1.0f, &desc, &readback) != NriResult_SUCCESS) return false;

    NriCommandAllocator *allocator = NULL;
    NriCommandBuffer *cmd = NULL;
    bool good = gpu_begin_commands(r, &allocator, &cmd) == NriResult_SUCCESS;

    if (good) {
        const NriTextureDataLayoutDesc layout = {.offset = 0, .rowPitch = (uint32_t)row_pitch, .slicePitch = (uint32_t)staging_bytes};

        const NriTextureRegionDesc region = {.width = (NriDim_t)width, .height = (NriDim_t)height, .depth = 1, .mipOffset = 0, .layerOffset = 0};

        good = gpu_transition_texture(r, cmd, texture, NriAccessBits_COPY_SOURCE, NriLayout_COPY_SOURCE, NriStageBits_COPY);

        if (good) {
            r->gpu->core.CmdReadbackTextureToBuffer(cmd, readback, &layout, texture, &region);

            good = gpu_submit_commands(r, allocator, cmd);
            allocator = NULL;
            cmd = NULL;
        }
    }

    if (!good) {
        gpu_abort_commands(r, allocator, cmd);
        r->gpu->core.DestroyBuffer(readback);

        return false;
    }

    const Uint8 *mapped = r->gpu->core.MapBuffer(readback, 0, staging_bytes);

    if (!mapped) {
        r->gpu->core.DestroyBuffer(readback);

        return false;
    }

    Uint8 *data = malloc(tight_bytes);

    if (data) {
        for (uint32_t y = 0; y < height; ++y) memcpy(data + (size_t)y * row_bytes, mapped + (size_t)y * row_pitch, row_bytes);
    }

    r->gpu->core.UnmapBuffer(readback);
    r->gpu->core.DestroyBuffer(readback);
    *pixels = data;
    return data != NULL;
}

bool download_lightmap(RENDERER *r, CACHED_LIGHTMAP *out) {
    if (!r || !out) return false;

    Uint8 *pixels = NULL;

    if (!read_rgba16f_texture(r, r->lightmap_texture, r->lightmap_width, r->lightmap_height, &pixels)) return false;
    out->pixels = pixels;
    out->width = r->lightmap_width;
    out->height = r->lightmap_height;

    return true;
}

bool upload_bvh(RENDERER *r, const BVH *tree) {
    if (!r || !tree || !tree->nodes || !tree->node_count || !tree->triangles || !tree->triangle_count) return false;

    release_buffer(r, r->bvh_node_buffer);
    release_buffer(r, r->bvh_triangle_buffer);
    r->bvh_node_buffer = NULL;
    r->bvh_triangle_buffer = NULL;

    r->bvh_node_buffer =
        gpu_upload_buffer(r, NriBufferUsageBits_SHADER_RESOURCE, tree->nodes, (size_t)tree->node_count * sizeof(*tree->nodes), sizeof(BVH_NODE));

    r->bvh_triangle_buffer = gpu_upload_buffer(r, NriBufferUsageBits_SHADER_RESOURCE, tree->triangles, (size_t)tree->triangle_count * sizeof(*tree->triangles),
                                               sizeof(BVH_TRIANGLE));

    const bool good = r->bvh_node_buffer && r->bvh_triangle_buffer;

    r->bvh_triangle_count = good ? tree->triangle_count : 0u;
    r->bvh_emissive_weight = good ? tree->emissive_weight : 0.0f;

    return good;
}

NriBuffer *upload_probes(RENDERER *r, const PROBE_GRID *grid) {
    if (!grid || !grid->probes) return NULL;

    const size_t count = (size_t)grid->count_x * grid->count_y * grid->count_z;

    if (!count) return NULL;

    return gpu_upload_buffer(r, NriBufferUsageBits_SHADER_RESOURCE, grid->probes, count * sizeof(PROBE), sizeof(PROBE));
}

NriBuffer *upload_beams(RENDERER *r, const BEAM_GRID *grid) {
    if (!grid) return NULL;

    float *visibility = beam_expand(grid);

    if (!visibility || !grid->shadow_depth) {
        free(visibility);

        return NULL;
    }

    const size_t beam_count = (size_t)grid->width * grid->height * grid->depth;
    const size_t depth_count = (size_t)grid->width * grid->height;

    float *data = realloc(visibility, (beam_count + depth_count) * sizeof(float));

    if (!data) {
        free(visibility);

        return NULL;
    }

    memcpy(data + beam_count, grid->shadow_depth, depth_count * sizeof(float));

    NriBuffer *buffer = gpu_upload_buffer(r, NriBufferUsageBits_SHADER_RESOURCE, data, (beam_count + depth_count) * sizeof(float), sizeof(float));

    free(data);

    return buffer;
}

typedef struct LIGHTMAP_QUEUE_UNIFORMS {
    Uint32 dispatch_width;

    Uint32 pad0, pad1, pad2;
} LIGHTMAP_QUEUE_UNIFORMS;

static NriBuffer *lightmap_queue_buffer(RENDERER *r, uint64_t bytes, NriBufferUsageBits usage) {
    const NriBufferDesc desc = {.size = bytes, .structureStride = sizeof(Uint32), .usage = usage};
    NriBuffer *buffer = NULL;

    return r->gpu->core.CreateCommittedBuffer(r->gpu->device, NriMemoryLocation_DEVICE, 1.0f, &desc, &buffer) == NriResult_SUCCESS ? buffer : NULL;
}

static bool lightmap_queue_ensure(RENDERER *r, Uint32 capacity) {
    if (!r || !capacity) return false;

    if (!r->lightmap_queue_reset_pipeline)
        r->lightmap_queue_reset_pipeline =
            gpu_compile_compute(r, r->lightmap_queue_reset_layout, "shaders/lightmap_queue.hlsl", "lightmap_queue_reset_cs", "BUILD_LIGHTMAP_QUEUE_RESET_CS");

    if (!r->lightmap_queue_args_pipeline)
        r->lightmap_queue_args_pipeline =
            gpu_compile_compute(r, r->lightmap_queue_args_layout, "shaders/lightmap_queue.hlsl", "lightmap_queue_args_cs", "BUILD_LIGHTMAP_QUEUE_ARGS_CS");

    if (!r->lightmap_queue_reset_pipeline || !r->lightmap_queue_args_pipeline) return false;

    const uint64_t bytes = (uint64_t)capacity * sizeof(Uint32);

    if (r->lightmap_active_capacity >= bytes && r->lightmap_active_buffer[0] && r->lightmap_active_buffer[1] && r->lightmap_active_count[0] &&
        r->lightmap_active_count[1] && r->lightmap_dispatch_args)
        return true;

    for (uint32_t i = 0; i < 2u; ++i) {
        release_buffer(r, r->lightmap_active_buffer[i]);
        release_buffer(r, r->lightmap_active_count[i]);
        r->lightmap_active_buffer[i] = NULL;
        r->lightmap_active_count[i] = NULL;
    }

    release_buffer(r, r->lightmap_dispatch_args);
    r->lightmap_dispatch_args = NULL;
    r->lightmap_active_capacity = 0u;

    const NriBufferUsageBits queue_usage = NriBufferUsageBits_SHADER_RESOURCE | NriBufferUsageBits_SHADER_RESOURCE_STORAGE;
    const NriBufferUsageBits args_usage = NriBufferUsageBits_SHADER_RESOURCE_STORAGE | NriBufferUsageBits_ARGUMENT;

    for (uint32_t i = 0; i < 2u; ++i) {
        r->lightmap_active_buffer[i] = lightmap_queue_buffer(r, bytes, queue_usage);
        r->lightmap_active_count[i] = lightmap_queue_buffer(r, sizeof(Uint32), queue_usage);
    }

    r->lightmap_dispatch_args = lightmap_queue_buffer(r, 3u * sizeof(Uint32), args_usage);

    if (!r->lightmap_active_buffer[0] || !r->lightmap_active_buffer[1] || !r->lightmap_active_count[0] || !r->lightmap_active_count[1] ||
        !r->lightmap_dispatch_args)
        return false;

    r->lightmap_active_capacity = bytes;

    return true;
}

static void dispatch_shape_trace(Uint32 items, Uint32 *groups_x, Uint32 *groups_y, Uint32 *dispatch_width) {
    Uint32 gx = (items + 63u) / 64u;

    if (!gx) gx = 1u;

    if (gx > 256u) gx = 256u;

    const Uint32 width = gx * 64u;
    const Uint32 gy = (items + width - 1u) / width;
    *groups_x = gx;
    *groups_y = gy ? gy : 1u;
    *dispatch_width = width;
}

static void lightmap_buffer_barrier(RENDERER *r, NriCommandBuffer *cmd, NriBuffer *buffer, NriAccessBits before_access, NriStageBits before_stages,
                                    NriAccessBits after_access, NriStageBits after_stages) {
    const NriBufferBarrierDesc barrier = {
        .buffer = buffer, .before = {.access = before_access, .stages = before_stages}, .after = {.access = after_access, .stages = after_stages}};
    r->gpu->core.CmdBarrier(cmd, &(NriBarrierDesc){.buffers = &barrier, .bufferNum = 1u});
}

static bool record_lightmap_queue_reset(RENDERER *r, NriCommandBuffer *cmd, uint32_t index, bool reused) {
    NriBuffer *count = r->lightmap_active_count[index];
    lightmap_buffer_barrier(r, cmd, count, reused ? NriAccessBits_SHADER_RESOURCE : NriAccessBits_NONE,
                            reused ? NriStageBits_COMPUTE_SHADER : NriStageBits_NONE, NriAccessBits_SHADER_RESOURCE_STORAGE, NriStageBits_COMPUTE_SHADER);
    NriDescriptor *dst = gpu_create_buffer_view(r, count, NriBufferView_STORAGE_STRUCTURED_BUFFER, sizeof(Uint32));

    if (!dst || !gpu_bind_descriptor_set(r, cmd, r->lightmap_queue_reset_layout, NriBindPoint_COMPUTE, 1, &dst, 1)) return false;
    r->gpu->core.CmdSetPipeline(cmd, r->lightmap_queue_reset_pipeline);
    r->gpu->core.CmdDispatch(cmd, &(NriDispatchDesc){.workGroupNumX = 1u, .workGroupNumY = 1u, .workGroupNumZ = 1u});

    return true;
}

static bool record_lightmap_queue_args(RENDERER *r, NriCommandBuffer *cmd, uint32_t index, Uint32 dispatch_width, bool args_reused) {
    NriBuffer *count = r->lightmap_active_count[index];
    lightmap_buffer_barrier(r, cmd, count, NriAccessBits_SHADER_RESOURCE_STORAGE, NriStageBits_COMPUTE_SHADER, NriAccessBits_SHADER_RESOURCE,
                            NriStageBits_COMPUTE_SHADER);
    lightmap_buffer_barrier(r, cmd, r->lightmap_dispatch_args, args_reused ? NriAccessBits_ARGUMENT_BUFFER : NriAccessBits_NONE,
                            args_reused ? NriStageBits_INDIRECT : NriStageBits_NONE, NriAccessBits_SHADER_RESOURCE_STORAGE, NriStageBits_COMPUTE_SHADER);

    NriDescriptor *src = gpu_create_buffer_view(r, count, NriBufferView_STRUCTURED_BUFFER, sizeof(Uint32));
    NriDescriptor *dst = gpu_create_buffer_view(r, r->lightmap_dispatch_args, NriBufferView_STORAGE_STRUCTURED_BUFFER, sizeof(Uint32));
    const LIGHTMAP_QUEUE_UNIFORMS uniforms = {.dispatch_width = dispatch_width};

    if (!src || !dst || !gpu_bind_descriptor_set(r, cmd, r->lightmap_queue_args_layout, NriBindPoint_COMPUTE, 0, &src, 1) ||
        !gpu_bind_descriptor_set(r, cmd, r->lightmap_queue_args_layout, NriBindPoint_COMPUTE, 1, &dst, 1) ||
        !gpu_bind_uniform_data(r, cmd, r->lightmap_queue_args_layout, NriBindPoint_COMPUTE, 2, &uniforms, sizeof(uniforms)))
        return false;

    r->gpu->core.CmdSetPipeline(cmd, r->lightmap_queue_args_pipeline);
    r->gpu->core.CmdDispatch(cmd, &(NriDispatchDesc){.workGroupNumX = 1u, .workGroupNumY = 1u, .workGroupNumZ = 1u});
    lightmap_buffer_barrier(r, cmd, r->lightmap_dispatch_args, NriAccessBits_SHADER_RESOURCE_STORAGE, NriStageBits_COMPUTE_SHADER,
                            NriAccessBits_ARGUMENT_BUFFER, NriStageBits_INDIRECT);
    return true;
}

static void dispatch_shape(Uint32 items, Uint32 *groups_x, Uint32 *groups_y, Uint32 *dispatch_width) {
    Uint32 gx = (items + 63u) / 64u;

    if (gx == 0u) gx = 1u;

    if (gx > 4096u) gx = 4096u;

    const Uint32 width = gx * 64u;
    const Uint32 gy = (items + width - 1u) / width;

    *groups_x = gx;
    *groups_y = gy ? gy : 1u;
    *dispatch_width = width;
}

static BAKE_UNIFORMS bake_data(RENDERER *r, Uint32 phase, Uint32 iteration, Uint32 item_count, Uint32 dispatch_width, Uint32 batch_count) {
    const DIRECTIONAL_LIGHT sun = r->sun;
    const SKY sky = r->sky;

    return (BAKE_UNIFORMS){
        .item_count = item_count,
        .lightmap_width = r->lightmap_width,
        .lightmap_height = r->lightmap_height,
        .dispatch_width = dispatch_width,
        .iteration = iteration,
        .phase = phase,
        .max_bounces = BAKE_MAX_BOUNCES,
        .batch_count = batch_count,
        .sun_direction_intensity = {sun.direction.x, sun.direction.y, sun.direction.z, sun.intensity},
        .sun_color_radius = {sun.color.x, sun.color.y, sun.color.z, sun.angular_radius},
        .sky_zenith = {sky.zenith.x, sky.zenith.y, sky.zenith.z, 1.0f},
        .sky_horizon = {sky.horizon.x, sky.horizon.y, sky.horizon.z, 1.0f},
        .bake_params = {r->bake_epsilon, 0.72f, sky.intensity, (float)r->lightmap_min_samples},
        .probe_origin_spacing = {r->lightmap_probe_origin.x, r->lightmap_probe_origin.y, r->lightmap_probe_origin.z, r->lightmap_probe_spacing},
        .probe_dims_mode = {r->lightmap_probe_count_x, r->lightmap_probe_count_y, r->lightmap_probe_count_z, 0u},
        .emissive_data = {r->bvh_emissive_weight, (float)r->bvh_triangle_count, r->volumetrics.emissive_probe_intensity, 0.0f}};
}

static bool record_bake_pass(RENDERER *r, NriCommandBuffer *cmd, NriTexture *source, NriTexture *destination, Uint32 phase, Uint32 iteration, Uint32 item_count,
                             Uint32 batch_count) {
    Uint32 groups_x, groups_y, dispatch_width;
    dispatch_shape(item_count, &groups_x, &groups_y, &dispatch_width);

    const BAKE_UNIFORMS uniforms = bake_data(r, phase, iteration, item_count, dispatch_width, batch_count);

    if (!bake_bind_resources(r, cmd, source, destination, &uniforms, sizeof(uniforms))) return false;

    r->gpu->core.CmdSetPipeline(cmd, r->bake_pipeline);
    r->gpu->core.CmdDispatch(cmd, &(NriDispatchDesc){.workGroupNumX = groups_x, .workGroupNumY = groups_y, .workGroupNumZ = 1});

    return true;
}

static void swap_lightmaps(RENDERER *r) {
    NriTexture *tmp = r->lightmap_texture;

    r->lightmap_texture = r->lightmap_scratch;
    r->lightmap_scratch = tmp;
}

static bool record_trace_batch(RENDERER *r, NriCommandBuffer *cmd, Uint32 first, Uint32 count, Uint32 items, Uint32 batch_index, Uint32 groups_x,
                               Uint32 groups_y, Uint32 dispatch_width) {
    if (!r || !cmd || !items || !count) return false;

    const uint32_t output_index = batch_index & 1u;
    const uint32_t input_index = output_index ^ 1u;
    const bool active_mode = batch_index != 0u;
    const bool output_reused = batch_index >= 2u;

    if (!record_lightmap_queue_reset(r, cmd, output_index, output_reused)) return false;
    lightmap_buffer_barrier(r, cmd, r->lightmap_active_buffer[output_index], output_reused ? NriAccessBits_SHADER_RESOURCE : NriAccessBits_NONE,
                            output_reused ? NriStageBits_COMPUTE_SHADER : NriStageBits_NONE, NriAccessBits_SHADER_RESOURCE_STORAGE,
                            NriStageBits_COMPUTE_SHADER);

    if (active_mode) {
        lightmap_buffer_barrier(r, cmd, r->lightmap_active_buffer[input_index], NriAccessBits_SHADER_RESOURCE_STORAGE, NriStageBits_COMPUTE_SHADER,
                                NriAccessBits_SHADER_RESOURCE, NriStageBits_COMPUTE_SHADER);
    }

    BAKE_UNIFORMS uniforms = bake_data(r, PHASE_TRACE, first, items, dispatch_width, count);

    uniforms.probe_dims_mode[3] = active_mode ? 1u : 0u;

    if (!bake_bind_resources_ex(r, cmd, r->lightmap_texture, r->lightmap_scratch, r->lightmap_active_buffer[input_index], r->lightmap_active_count[input_index],
                                r->lightmap_active_buffer[output_index], r->lightmap_active_count[output_index], &uniforms, sizeof(uniforms)))
        return false;

    r->gpu->core.CmdSetPipeline(cmd, r->bake_pipeline);

    if (active_mode)
        r->gpu->core.CmdDispatchIndirect(cmd, r->lightmap_dispatch_args, 0u);
    else
        r->gpu->core.CmdDispatch(cmd, &(NriDispatchDesc){.workGroupNumX = groups_x, .workGroupNumY = groups_y, .workGroupNumZ = 1u});

    if (!record_lightmap_queue_args(r, cmd, output_index, dispatch_width, batch_index != 0u)) return false;
    swap_lightmaps(r);

    return true;
}

static float decode_half(Uint16 h) {
    Uint32 exponent = (h >> 10u) & 31u;
    Uint32 mantissa = h & 1023u;
    float value = exponent == 0u ? ldexpf((float)mantissa, -24) : exponent == 31u ? INFINITY : ldexpf((float)(1024u + mantissa), (int)exponent - 25);

    return (h & 0x8000u) ? -value : value;
}

static float direct_luma(const Uint8 *pixels, Uint32 pixel) {
    Uint16 channels[3];

    memcpy(channels, pixels + (size_t)pixel * 8u, sizeof(channels));

    return decode_half(channels[0]) * 0.2126f + decode_half(channels[1]) * 0.7152f + decode_half(channels[2]) * 0.0722f;
}

static bool build_lightmap_patches(RENDERER *r, const LIGHTMAP *lm) {
    Uint8 *pixels = NULL;

    if (!read_rgba16f_texture(r, r->lightmap_direct, lm->width, lm->height, &pixels)) return false;

    const size_t pixel_count = (size_t)lm->width * lm->height;
    const size_t tile_width = (lm->width + 3u) / 4u;
    const size_t tile_count = tile_width * ((lm->height + 3u) / 4u);
    Uint32 *pixel_sample = malloc(pixel_count * sizeof(*pixel_sample));
    Uint32 *mapping = malloc((size_t)lm->sample_count * sizeof(*mapping));

    Uint32(*anchors)[4] = calloc(tile_count, sizeof(*anchors));

    Uint8 *keep = malloc(lm->sample_count);
    bool ok = false;

    if (!pixel_sample || !mapping || !anchors || !keep) goto cleanup;
    memset(pixel_sample, 0xff, pixel_count * sizeof(*pixel_sample));
    memset(mapping, 0xff, (size_t)lm->sample_count * sizeof(*mapping));
    memset(keep, 1, lm->sample_count);

    for (Uint32 i = 0; i < lm->sample_count; ++i) {
        Uint32 pixel;

        memcpy(&pixel, &lm->samples[i].position[3], sizeof(pixel));

        pixel_sample[pixel] = i;
    }

    Uint32 reduced = 0;
    Uint32 active_tiles = 0;

    for (Uint32 y = 0; y + 3u < lm->height; y += 4u) {
        for (Uint32 x = 0; x + 3u < lm->width; x += 4u) {
            Uint32 indices[16];
            float lo = INFINITY, hi = -INFINITY;
            bool smooth = true;
            Uint32 base_index = pixel_sample[(size_t)y * lm->width + x];

            if (base_index == UINT32_MAX) continue;

            const LMAP_SAMPLE *base = &lm->samples[base_index];

            for (Uint32 dy = 0; dy < 4u && smooth; ++dy) {
                for (Uint32 dx = 0; dx < 4u; ++dx) {
                    Uint32 pixel = (y + dy) * lm->width + x + dx;
                    Uint32 index = pixel_sample[pixel];

                    if (index == UINT32_MAX) {
                        smooth = false;

                        break;
                    }

                    const LMAP_SAMPLE *sample = &lm->samples[index];
                    float dot = base->normal[0] * sample->normal[0] + base->normal[1] * sample->normal[1] + base->normal[2] * sample->normal[2];

                    float plane = (sample->position[0] - base->position[0]) * base->normal[0] + (sample->position[1] - base->position[1]) * base->normal[1] +
                                  (sample->position[2] - base->position[2]) * base->normal[2];

                    if (sample->normal[3] != base->normal[3] || dot < 0.995f || fabsf(plane) > 0.025f) {
                        smooth = false;

                        break;
                    }

                    indices[dy * 4u + dx] = index;
                    float luma = direct_luma(pixels, pixel);
                    lo = fminf(lo, luma);
                    hi = fmaxf(hi, luma);
                }
            }

            if (!smooth || !isfinite(hi) || hi - lo > 0.06f + 0.1f * hi) continue;

            Uint32 tile = (y / 4u) * (Uint32)tile_width + x / 4u;
            const Uint32 corners[4] = {0u, 3u, 12u, 15u};

            for (Uint32 j = 0; j < 4u; ++j) {
                keep[indices[corners[j]]] = 1u;

                anchors[tile][j] = (y + corners[j] / 4u) * lm->width + x + corners[j] % 4u;
            }

            for (Uint32 j = 0; j < 16u; ++j) {
                mapping[indices[j]] = tile;

                if (j != 0u && j != 3u && j != 12u && j != 15u) keep[indices[j]] = 0u;
            }

            reduced += 12u;
            active_tiles++;
        }
    }

    ok = true;

    if (reduced) {
        Uint32 count = lm->sample_count - reduced;
        LMAP_SAMPLE *sparse = malloc((size_t)count * sizeof(*sparse));
        NriBuffer *sample_buffer = NULL;
        NriBuffer *map_buffer = NULL;
        NriBuffer *anchor_buffer = NULL;

        if (sparse) {
            Uint32 next = 0;

            for (Uint32 i = 0; i < lm->sample_count; ++i)
                if (keep[i]) sparse[next++] = lm->samples[i];

            sample_buffer = gpu_upload_buffer(r, NriBufferUsageBits_SHADER_RESOURCE, sparse, (size_t)count * sizeof(*sparse), sizeof(LMAP_SAMPLE));
            map_buffer = gpu_upload_buffer(r, NriBufferUsageBits_SHADER_RESOURCE, mapping, (size_t)lm->sample_count * sizeof(*mapping), sizeof(Uint32));
            anchor_buffer = gpu_upload_buffer(r, NriBufferUsageBits_SHADER_RESOURCE, anchors, tile_count * sizeof(*anchors), sizeof(Uint32[4]));
        }

        free(sparse);

        ok = sample_buffer && map_buffer && anchor_buffer;

        if (ok) {
            r->lightmap_full_sample_buffer = r->lightmap_sample_buffer;
            r->lightmap_sample_buffer = sample_buffer;
            r->lightmap_patch_map_buffer = map_buffer;
            r->lightmap_patch_anchor_buffer = anchor_buffer;
            r->lightmap_trace_count = count;
            SDL_Log("B: lightmap 4x4 patches: %u chart-safe tiles, %u/%u indirect texels", active_tiles, count, lm->sample_count);
        } else {
            release_buffer(r, sample_buffer);
            release_buffer(r, map_buffer);
            release_buffer(r, anchor_buffer);
        }
    }

cleanup:
    free(keep);
    free(anchors);
    free(mapping);
    free(pixel_sample);
    free(pixels);

    return ok;
}

static bool bake_lightmap_once(RENDERER *r, const LIGHTMAP *lm) {
    const Uint32 pixels = r->lightmap_width * r->lightmap_height;

    SDL_Log("B: lightmap trace samples per pass: %u", BAKE_BATCH_SAMPLES);
    bake_progress(r, "surface lightmap", 0u, r->bake_target_samples);

    NriCommandAllocator *allocator = NULL;
    NriCommandBuffer *cmd = NULL;

    if (gpu_begin_commands(r, &allocator, &cmd) != NriResult_SUCCESS) return false;

    if (!record_bake_pass(r, cmd, r->lightmap_scratch, r->lightmap_texture, PHASE_CLEAR, 0, pixels, 0) ||
        !record_bake_pass(r, cmd, r->lightmap_texture, r->lightmap_scratch, PHASE_CLEAR, 0, pixels, 0) ||
        !record_bake_pass(r, cmd, r->lightmap_scratch, r->lightmap_direct, PHASE_CLEAR, 0, pixels, 0) ||
        !record_bake_pass(r, cmd, r->lightmap_scratch, r->lightmap_direct, PHASE_DIRECT, 0, r->lightmap_sample_count, 0)) {
        gpu_abort_commands(r, allocator, cmd);

        return false;
    }

    if (!gpu_submit_commands(r, allocator, cmd)) return false;

    if (!build_lightmap_patches(r, lm)) return false;

    if (r->lightmap_trace_count) {
        Uint32 trace_groups_x = 0u, trace_groups_y = 0u, trace_dispatch_width = 0u;

        dispatch_shape_trace(r->lightmap_trace_count, &trace_groups_x, &trace_groups_y, &trace_dispatch_width);

        Uint32 batch_index = 0u;

        for (Uint32 first = 0; first < r->bake_target_samples; first += BAKE_BATCH_SAMPLES, ++batch_index) {
            Uint32 count = r->bake_target_samples - first;

            if (count > BAKE_BATCH_SAMPLES) count = BAKE_BATCH_SAMPLES;

            allocator = NULL;
            cmd = NULL;

            if (gpu_begin_commands(r, &allocator, &cmd) != NriResult_SUCCESS) return false;

            if (!record_trace_batch(r, cmd, first, count, r->lightmap_trace_count, batch_index, trace_groups_x, trace_groups_y, trace_dispatch_width)) {
                gpu_abort_commands(r, allocator, cmd);

                return false;
            }

            if (!gpu_submit_commands(r, allocator, cmd)) return false;
            bake_progress(r, "surface lightmap", first + count, r->bake_target_samples);
        }
    }

    bake_progress(r, "filtering lightmap", 0u, 0u);

    allocator = NULL;
    cmd = NULL;

    if (gpu_begin_commands(r, &allocator, &cmd) != NriResult_SUCCESS) return false;

    if (r->lightmap_full_sample_buffer) {
        r->lightmap_sparse_sample_buffer = r->lightmap_sample_buffer;
        r->lightmap_sample_buffer = r->lightmap_full_sample_buffer;
        r->lightmap_full_sample_buffer = NULL;

        if (!record_bake_pass(r, cmd, r->lightmap_texture, r->lightmap_scratch, PHASE_CLEAR, 0, pixels, 0) ||
            !record_bake_pass(r, cmd, r->lightmap_texture, r->lightmap_scratch, PHASE_RECONSTRUCT, 0, r->lightmap_sample_count, 0)) {
            gpu_abort_commands(r, allocator, cmd);

            return false;
        }

        swap_lightmaps(r);
    }

    if (!record_bake_pass(r, cmd, r->lightmap_texture, r->lightmap_scratch, PHASE_COMBINE, 0, pixels, 0)) {
        gpu_abort_commands(r, allocator, cmd);

        return false;
    }

    swap_lightmaps(r);

    if (!record_bake_pass(r, cmd, r->lightmap_texture, r->lightmap_scratch, PHASE_FILTER, 0, pixels, 0)) {
        gpu_abort_commands(r, allocator, cmd);

        return false;
    }

    swap_lightmaps(r);

    for (Uint32 i = 0; i < BAKE_DILATION_PASSES; ++i) {
        if (!record_bake_pass(r, cmd, r->lightmap_texture, r->lightmap_scratch, PHASE_DILATE, 0, pixels, 0)) {
            gpu_abort_commands(r, allocator, cmd);

            return false;
        }

        swap_lightmaps(r);
    }

    if (!gpu_submit_commands(r, allocator, cmd)) return false;
    release_texture(r, r->lightmap_scratch);
    r->lightmap_scratch = NULL;

    return true;
}

bool bake_lightmap(RENDERER *r, const BVH *tree, const LIGHTMAP *lm, const PROBE_GRID *probes) {
    if (!r || !r->gpu->device || !tree || !tree->node_count || !lm || !lm->width || !lm->height || !lm->samples || !lm->sample_count || !r->lightmap_sampler ||
        !probes || !probes->probes || !probes->spacing || !probes->count_x || !probes->count_y || !probes->count_z)
        return false;

    r->lightmap_width = lm->width;
    r->lightmap_height = lm->height;
    r->lightmap_sample_count = lm->sample_count;
    r->lightmap_trace_count = lm->sample_count;
    r->bake_target_samples = BAKE_TARGET_SAMPLES;
    r->lightmap_min_samples = 32u;
    r->lightmap_probe_origin = probes->origin;
    r->lightmap_probe_spacing = probes->spacing;
    r->lightmap_probe_count_x = probes->count_x;
    r->lightmap_probe_count_y = probes->count_y;
    r->lightmap_probe_count_z = probes->count_z;
    SDL_Log("B: lightmap minimum samples: %u", r->lightmap_min_samples);

    r->lightmap_probe_buffer = upload_probes(r, probes);

    if (!r->lightmap_probe_buffer) return false;

    const BVH_NODE *root = &tree->nodes[0];
    const float sx = root->max[0] - root->min[0];
    const float sy = root->max[1] - root->min[1];
    const float sz = root->max[2] - root->min[2];
    float scene_scale = fmaxf(sx, fmaxf(sy, sz));

    if (scene_scale < 1.0f) scene_scale = 1.0f;

    r->bake_epsilon = scene_scale * 2.0e-5f;

    r->lightmap_texture = create_lightmap_texture(r, lm->width, lm->height);
    r->lightmap_scratch = create_lightmap_texture(r, lm->width, lm->height);
    r->lightmap_direct = create_lightmap_texture(r, lm->width, lm->height);

    if (!r->lightmap_texture || !r->lightmap_scratch || !r->lightmap_direct) return false;

    if (!upload_bvh(r, tree)) return false;
    r->lightmap_sample_buffer =
        gpu_upload_buffer(r, NriBufferUsageBits_SHADER_RESOURCE, lm->samples, (size_t)lm->sample_count * sizeof(*lm->samples), sizeof(LMAP_SAMPLE));

    if (!r->lightmap_sample_buffer) return false;

    if (!lightmap_queue_ensure(r, lm->sample_count)) return false;

    if (!r->bake_pipeline) {
        const NriDeviceDesc *device = r->gpu->core.GetDeviceDesc(r->gpu->device);
        const bool wave_ops = device && (device->wave.waveOpsStages & NriStageBits_COMPUTE_SHADER) != 0;

        r->bake_pipeline =
            gpu_compile_compute(r, r->bake_layout, "shaders/compute.hlsl", "lightmap_cs", wave_ops ? "BUILD_LIGHTMAP_WAVE_CS" : "BUILD_LIGHTMAP_CS");
    }

    if (!r->bake_pipeline) return false;

    SDL_Log("lightmap: %ux%u, %u charts, %u valid texels, %.2f texels/unit", lm->width, lm->height, lm->chart_count, lm->sample_count, lm->texel_density);

    return bake_lightmap_once(r, lm);
}

#define PROBE_BLOCK_SAMPLES 128u
#define PROBE_MAX_BOUNCES 3u
#define PROBE_PACKED_NODE_BYTES 32u
#define PROBE_PACKED_TRIANGLE_BYTES 80u
#define PROBE_RAY_STATE_BYTES 32u
#define PROBE_ACCUM_BYTES 32u
#define PROBE_OUTPUT_STRIDE_BYTES (9u * 16u)

typedef struct PROBE_WAVEFRONT_UNIFORMS {
    Uint32 probe_count;
    Uint32 sample_offset;
    Uint32 samples_per_block;
    Uint32 total_samples;
    Uint32 node_count;
    Uint32 triangle_count;
    Uint32 bounce_index;
    Uint32 max_bounces;
    Uint32 beam_depth;

    Uint32 emissive_samples, pad1, pad2;

    float sun_direction_intensity[4];
    float sun_color_radius[4];
    float sky_zenith[4];
    float sky_horizon[4];
    float bake_params[4];
    float emissive_params[4];
    float beam_origin[4];
    float beam_step[4];
} PROBE_WAVEFRONT_UNIFORMS;

typedef struct PROBE_WAVEFRONT_BUFFER {
    NriBuffer *buffer;
    uint32_t stride;
    NriAccessBits access;
    NriStageBits stages;
    NriBufferUsageBits usage;
    uint64_t capacity;
} PROBE_WAVEFRONT_BUFFER;

typedef struct PROBE_WAVEFRONT_STAGE {
    NriPipelineLayout *layout;
    NriPipeline *pipeline;
    uint8_t read_count;
    uint8_t write_count;
} PROBE_WAVEFRONT_STAGE;

typedef struct PROBE_WAVEFRONT_PIPELINES {
    PROBE_WAVEFRONT_STAGE prepare;
    PROBE_WAVEFRONT_STAGE reset;
    PROBE_WAVEFRONT_STAGE validate;
    PROBE_WAVEFRONT_STAGE primary;
    PROBE_WAVEFRONT_STAGE args;
    PROBE_WAVEFRONT_STAGE bounce;
    PROBE_WAVEFRONT_STAGE reduce;
    PROBE_WAVEFRONT_STAGE emissive;
} PROBE_WAVEFRONT_PIPELINES;

struct PROBE_WAVEFRONT_SCRATCH {
    PROBE_WAVEFRONT_BUFFER packed_nodes;
    PROBE_WAVEFRONT_BUFFER packed_triangles;
    PROBE_WAVEFRONT_BUFFER states_a;
    PROBE_WAVEFRONT_BUFFER states_b;
    PROBE_WAVEFRONT_BUFFER results;
    PROBE_WAVEFRONT_BUFFER accums;
    PROBE_WAVEFRONT_BUFFER coefficients;
    PROBE_WAVEFRONT_BUFFER counters;
    PROBE_WAVEFRONT_BUFFER dispatch_args;
    PROBE_WAVEFRONT_PIPELINES pipelines;
    NriBuffer *counter_readback;
    NriBuffer *output_readback;
    uint64_t output_readback_capacity;
    bool pipelines_ready;
};

static PROBE_WAVEFRONT_BUFFER probe_wavefront_uploaded(RENDERER *r, const void *data, uint64_t bytes, uint32_t stride) {
    PROBE_WAVEFRONT_BUFFER result = {0};

    if (!data || !bytes || bytes > SIZE_MAX) return result;
    result.buffer = gpu_upload_buffer(r, NriBufferUsageBits_SHADER_RESOURCE, data, (size_t)bytes, stride);

    if (result.buffer) {
        result.stride = stride;
        result.access = NriAccessBits_SHADER_RESOURCE;
        result.stages = NriStageBits_COMPUTE_SHADER;
        result.usage = NriBufferUsageBits_SHADER_RESOURCE;
        result.capacity = bytes;
    }

    return result;
}

static PROBE_WAVEFRONT_BUFFER probe_wavefront_storage(RENDERER *r, uint64_t bytes, uint32_t stride) {
    PROBE_WAVEFRONT_BUFFER result = {0};

    if (!r || !r->gpu->device || !bytes) return result;

    const NriBufferUsageBits usage = NriBufferUsageBits_SHADER_RESOURCE | NriBufferUsageBits_SHADER_RESOURCE_STORAGE;
    const NriBufferDesc desc = {.size = bytes, .structureStride = stride, .usage = usage};

    if (r->gpu->core.CreateCommittedBuffer(r->gpu->device, NriMemoryLocation_DEVICE, 1.0f, &desc, &result.buffer) == NriResult_SUCCESS) {
        result.stride = stride;
        result.usage = usage;
        result.capacity = bytes;
    }

    return result;
}

static PROBE_WAVEFRONT_BUFFER probe_wavefront_argument(RENDERER *r) {
    PROBE_WAVEFRONT_BUFFER result = {0};
    const NriBufferDesc desc = {
        .size = 3u * sizeof(uint32_t), .structureStride = sizeof(uint32_t), .usage = NriBufferUsageBits_SHADER_RESOURCE_STORAGE | NriBufferUsageBits_ARGUMENT};

    if (r && r->gpu->device && r->gpu->core.CreateCommittedBuffer(r->gpu->device, NriMemoryLocation_DEVICE, 1.0f, &desc, &result.buffer) == NriResult_SUCCESS) {
        result.stride = sizeof(uint32_t);
        result.usage = desc.usage;
        result.capacity = desc.size;
    }

    return result;
}

static void probe_wavefront_release_buffer(RENDERER *r, PROBE_WAVEFRONT_BUFFER *buffer) {
    if (!buffer) return;
    release_buffer(r, buffer->buffer);
    *buffer = (PROBE_WAVEFRONT_BUFFER){0};
}

static bool probe_wavefront_reserve(RENDERER *r, PROBE_WAVEFRONT_BUFFER *buffer, uint64_t bytes, uint32_t stride, NriBufferUsageBits usage) {
    if (!r || !buffer || !bytes || !stride) return false;

    if (buffer->buffer && buffer->capacity >= bytes && buffer->stride == stride && buffer->usage == usage) return true;

    probe_wavefront_release_buffer(r, buffer);

    const NriBufferDesc desc = {.size = bytes, .structureStride = stride, .usage = usage};

    if (r->gpu->core.CreateCommittedBuffer(r->gpu->device, NriMemoryLocation_DEVICE, 1.0f, &desc, &buffer->buffer) != NriResult_SUCCESS) return false;

    buffer->stride = stride;
    buffer->usage = usage;
    buffer->capacity = bytes;

    return true;
}

static bool probe_wavefront_reserve_readback(RENDERER *r, NriBuffer **buffer, uint64_t *capacity, uint64_t bytes) {
    if (!r || !buffer || !capacity || !bytes) return false;

    if (*buffer && *capacity >= bytes) return true;

    if (*buffer) r->gpu->core.DestroyBuffer(*buffer);
    *buffer = NULL;
    *capacity = 0u;

    const NriBufferDesc desc = {.size = bytes};

    if (r->gpu->core.CreateCommittedBuffer(r->gpu->device, NriMemoryLocation_HOST_READBACK, 0.0f, &desc, buffer) != NriResult_SUCCESS) return false;

    *capacity = bytes;
    return true;
}

static bool probe_wavefront_layout(RENDERER *r, NriPipelineLayout **layout, uint8_t reads, uint8_t writes) {
    NriDescriptorType read_types[4] = {0};
    NriDescriptorType write_types[3] = {0};
    static const NriDescriptorType uniform[] = {NriDescriptorType_CONSTANT_BUFFER};

    if (reads > 4u || writes > 3u) return false;

    for (uint8_t i = 0; i < reads; ++i) read_types[i] = NriDescriptorType_STRUCTURED_BUFFER;

    for (uint8_t i = 0; i < writes; ++i) write_types[i] = NriDescriptorType_STORAGE_STRUCTURED_BUFFER;
    const NriDescriptorType *sets[4] = {reads ? read_types : NULL, writes ? write_types : NULL, uniform, NULL};

    const uint8_t counts[4] = {reads, writes, 1u, 0u};

    return gpu_create_pipeline_layout(r, layout, sets, counts, NriStageBits_COMPUTE_SHADER);
}

static bool probe_wavefront_stage_init(RENDERER *r, PROBE_WAVEFRONT_STAGE *stage, const char *entrypoint, const char *define, uint8_t reads, uint8_t writes) {
    if (!probe_wavefront_layout(r, &stage->layout, reads, writes)) return false;
    stage->pipeline = gpu_compile_compute(r, stage->layout, "shaders/probe_wavefront.hlsl", entrypoint, define);
    stage->read_count = reads;
    stage->write_count = writes;

    return stage->pipeline != NULL;
}

static void probe_wavefront_stage_deinit(RENDERER *r, PROBE_WAVEFRONT_STAGE *stage) {
    if (!stage) return;

    if (stage->pipeline) r->gpu->core.DestroyPipeline(stage->pipeline);

    if (stage->layout) r->gpu->core.DestroyPipelineLayout(stage->layout);
    *stage = (PROBE_WAVEFRONT_STAGE){0};
}

static bool probe_wavefront_pipelines_init(RENDERER *r, PROBE_WAVEFRONT_PIPELINES *p) {
    memset(p, 0, sizeof(*p));

    const NriDeviceDesc *device = r->gpu->core.GetDeviceDesc(r->gpu->device);
    const bool wave_ops = device && (device->wave.waveOpsStages & NriStageBits_COMPUTE_SHADER) != 0;
    const char *primary_define = wave_ops ? "BUILD_PROBE_PRIMARY_WAVE_CS" : "BUILD_PROBE_PRIMARY_CS";
    const char *bounce_define = wave_ops ? "BUILD_PROBE_BOUNCE_WAVE_CS" : "BUILD_PROBE_BOUNCE_CS";

    return probe_wavefront_stage_init(r, &p->prepare, "probe_prepare_cs", "BUILD_PROBE_PREP_CS", 2, 2) &&
           probe_wavefront_stage_init(r, &p->reset, "probe_reset_cs", "BUILD_PROBE_RESET_CS", 0, 3) &&
           probe_wavefront_stage_init(r, &p->validate, "probe_validate_cs", "BUILD_PROBE_VALIDATE_CS", 3, 1) &&
           probe_wavefront_stage_init(r, &p->primary, "probe_primary_cs", primary_define, 4, 3) &&
           probe_wavefront_stage_init(r, &p->args, "probe_args_cs", "BUILD_PROBE_ARGS_CS", 1, 1) &&
           probe_wavefront_stage_init(r, &p->bounce, "probe_bounce_cs", bounce_define, 4, 3) &&
           probe_wavefront_stage_init(r, &p->reduce, "probe_reduce_cs", "BUILD_PROBE_REDUCE_CS", 1, 3) &&
           probe_wavefront_stage_init(r, &p->emissive, "probe_emissive_cs", "BUILD_PROBE_EMISSIVE_CS", 3, 1);
}

static void probe_wavefront_pipelines_deinit(RENDERER *r, PROBE_WAVEFRONT_PIPELINES *p) {
    probe_wavefront_stage_deinit(r, &p->emissive);
    probe_wavefront_stage_deinit(r, &p->reduce);
    probe_wavefront_stage_deinit(r, &p->bounce);
    probe_wavefront_stage_deinit(r, &p->args);
    probe_wavefront_stage_deinit(r, &p->primary);
    probe_wavefront_stage_deinit(r, &p->validate);
    probe_wavefront_stage_deinit(r, &p->reset);
    probe_wavefront_stage_deinit(r, &p->prepare);
}

static bool probe_wavefront_scratch_ensure(RENDERER *r, uint64_t node_bytes, uint64_t triangle_bytes, uint64_t state_bytes, uint64_t result_bytes,
                                           uint64_t accum_bytes, uint64_t output_bytes) {
    if (!r) return false;

    if (!r->probe_scratch) {
        r->probe_scratch = calloc(1, sizeof(*r->probe_scratch));

        if (!r->probe_scratch) return false;
    }

    PROBE_WAVEFRONT_SCRATCH *scratch = r->probe_scratch;
    const NriBufferUsageBits storage = NriBufferUsageBits_SHADER_RESOURCE | NriBufferUsageBits_SHADER_RESOURCE_STORAGE;
    const NriBufferUsageBits argument = NriBufferUsageBits_SHADER_RESOURCE_STORAGE | NriBufferUsageBits_ARGUMENT;

    if (!probe_wavefront_reserve(r, &scratch->packed_nodes, node_bytes, PROBE_PACKED_NODE_BYTES, storage) ||
        !probe_wavefront_reserve(r, &scratch->packed_triangles, triangle_bytes, PROBE_PACKED_TRIANGLE_BYTES, storage) ||
        !probe_wavefront_reserve(r, &scratch->states_a, state_bytes, PROBE_RAY_STATE_BYTES, storage) ||
        !probe_wavefront_reserve(r, &scratch->states_b, state_bytes, PROBE_RAY_STATE_BYTES, storage) ||
        !probe_wavefront_reserve(r, &scratch->results, result_bytes, sizeof(float[4]), storage) ||
        !probe_wavefront_reserve(r, &scratch->accums, accum_bytes, PROBE_ACCUM_BYTES, storage) ||
        !probe_wavefront_reserve(r, &scratch->coefficients, output_bytes, sizeof(float[4]), storage) ||
        !probe_wavefront_reserve(r, &scratch->counters, 4u * sizeof(uint32_t), sizeof(uint32_t), storage) ||
        !probe_wavefront_reserve(r, &scratch->dispatch_args, 3u * sizeof(uint32_t), sizeof(uint32_t), argument))
        return false;

    uint64_t counter_capacity = scratch->counter_readback ? 4u * sizeof(uint32_t) : 0u;

    if (!probe_wavefront_reserve_readback(r, &scratch->counter_readback, &counter_capacity, 4u * sizeof(uint32_t)) ||
        !probe_wavefront_reserve_readback(r, &scratch->output_readback, &scratch->output_readback_capacity, output_bytes))
        return false;

    if (!scratch->pipelines_ready) {
        if (!probe_wavefront_pipelines_init(r, &scratch->pipelines)) return false;
        scratch->pipelines_ready = true;
    }

    return true;
}

void probe_wavefront_scratch_destroy(RENDERER *r) {
    if (!r || !r->probe_scratch) return;

    PROBE_WAVEFRONT_SCRATCH *scratch = r->probe_scratch;

    if (scratch->pipelines_ready) probe_wavefront_pipelines_deinit(r, &scratch->pipelines);
    probe_wavefront_release_buffer(r, &scratch->dispatch_args);
    probe_wavefront_release_buffer(r, &scratch->counters);
    probe_wavefront_release_buffer(r, &scratch->coefficients);
    probe_wavefront_release_buffer(r, &scratch->accums);
    probe_wavefront_release_buffer(r, &scratch->results);
    probe_wavefront_release_buffer(r, &scratch->states_b);
    probe_wavefront_release_buffer(r, &scratch->states_a);
    probe_wavefront_release_buffer(r, &scratch->packed_triangles);
    probe_wavefront_release_buffer(r, &scratch->packed_nodes);

    if (scratch->counter_readback) r->gpu->core.DestroyBuffer(scratch->counter_readback);

    if (scratch->output_readback) r->gpu->core.DestroyBuffer(scratch->output_readback);
    free(scratch);
    r->probe_scratch = NULL;
}

static bool probe_wavefront_transition(RENDERER *r, NriCommandBuffer *cmd, PROBE_WAVEFRONT_BUFFER *const *reads, uint8_t read_count,
                                       PROBE_WAVEFRONT_BUFFER *const *writes, uint8_t write_count) {
    NriBufferBarrierDesc barriers[7] = {0};
    uint32_t count = 0;

    for (uint8_t i = 0; i < read_count; ++i) {
        PROBE_WAVEFRONT_BUFFER *buffer = reads[i];

        if (!buffer || !buffer->buffer) return false;

        barriers[count++] = (NriBufferBarrierDesc){.buffer = buffer->buffer,
                                                   .before = {.access = buffer->access, .stages = buffer->stages},
                                                   .after = {.access = NriAccessBits_SHADER_RESOURCE, .stages = NriStageBits_COMPUTE_SHADER}};

        buffer->access = NriAccessBits_SHADER_RESOURCE;
        buffer->stages = NriStageBits_COMPUTE_SHADER;
    }

    for (uint8_t i = 0; i < write_count; ++i) {
        PROBE_WAVEFRONT_BUFFER *buffer = writes[i];

        if (!buffer || !buffer->buffer) return false;

        barriers[count++] = (NriBufferBarrierDesc){.buffer = buffer->buffer,
                                                   .before = {.access = buffer->access, .stages = buffer->stages},
                                                   .after = {.access = NriAccessBits_SHADER_RESOURCE_STORAGE, .stages = NriStageBits_COMPUTE_SHADER}};

        buffer->access = NriAccessBits_SHADER_RESOURCE_STORAGE;
        buffer->stages = NriStageBits_COMPUTE_SHADER;
    }

    if (count) r->gpu->core.CmdBarrier(cmd, &(NriBarrierDesc){.buffers = barriers, .bufferNum = count});

    return true;
}

static bool probe_wavefront_dispatch(RENDERER *r, NriCommandBuffer *cmd, const PROBE_WAVEFRONT_STAGE *stage, PROBE_WAVEFRONT_BUFFER *const *reads,
                                     PROBE_WAVEFRONT_BUFFER *const *writes, const PROBE_WAVEFRONT_UNIFORMS *uniforms, uint32_t groups_x) {
    if (!r || !cmd || !stage || !stage->pipeline || !stage->layout || !uniforms || !groups_x) return false;

    if (!probe_wavefront_transition(r, cmd, reads, stage->read_count, writes, stage->write_count)) return false;

    if (stage->read_count) {
        NriDescriptor *descriptors[4] = {0};

        for (uint8_t i = 0; i < stage->read_count; ++i)
            descriptors[i] = gpu_create_buffer_view(r, reads[i]->buffer, NriBufferView_STRUCTURED_BUFFER, reads[i]->stride);

        if (!gpu_bind_descriptor_set(r, cmd, stage->layout, NriBindPoint_COMPUTE, 0, descriptors, stage->read_count)) return false;
    }

    if (stage->write_count) {
        NriDescriptor *descriptors[3] = {0};

        for (uint8_t i = 0; i < stage->write_count; ++i)
            descriptors[i] = gpu_create_buffer_view(r, writes[i]->buffer, NriBufferView_STORAGE_STRUCTURED_BUFFER, writes[i]->stride);

        if (!gpu_bind_descriptor_set(r, cmd, stage->layout, NriBindPoint_COMPUTE, 1, descriptors, stage->write_count)) return false;
    }

    if (!gpu_bind_uniform_data(r, cmd, stage->layout, NriBindPoint_COMPUTE, 2, uniforms, sizeof(*uniforms))) return false;

    r->gpu->core.CmdSetPipeline(cmd, stage->pipeline);
    r->gpu->core.CmdDispatch(cmd, &(NriDispatchDesc){.workGroupNumX = groups_x, .workGroupNumY = 1, .workGroupNumZ = 1});

    return true;
}

static bool probe_wavefront_dispatch_indirect(RENDERER *r, NriCommandBuffer *cmd, const PROBE_WAVEFRONT_STAGE *stage, PROBE_WAVEFRONT_BUFFER *const *reads,
                                              PROBE_WAVEFRONT_BUFFER *const *writes, const PROBE_WAVEFRONT_UNIFORMS *uniforms,
                                              PROBE_WAVEFRONT_BUFFER *arguments) {
    if (!r || !cmd || !stage || !stage->pipeline || !stage->layout || !uniforms || !arguments || !arguments->buffer) return false;

    if (!probe_wavefront_transition(r, cmd, reads, stage->read_count, writes, stage->write_count)) return false;

    if (stage->read_count) {
        NriDescriptor *descriptors[4] = {0};

        for (uint8_t i = 0; i < stage->read_count; ++i)
            descriptors[i] = gpu_create_buffer_view(r, reads[i]->buffer, NriBufferView_STRUCTURED_BUFFER, reads[i]->stride);

        if (!gpu_bind_descriptor_set(r, cmd, stage->layout, NriBindPoint_COMPUTE, 0, descriptors, stage->read_count)) return false;
    }

    if (stage->write_count) {
        NriDescriptor *descriptors[3] = {0};

        for (uint8_t i = 0; i < stage->write_count; ++i)
            descriptors[i] = gpu_create_buffer_view(r, writes[i]->buffer, NriBufferView_STORAGE_STRUCTURED_BUFFER, writes[i]->stride);

        if (!gpu_bind_descriptor_set(r, cmd, stage->layout, NriBindPoint_COMPUTE, 1, descriptors, stage->write_count)) return false;
    }

    if (!gpu_bind_uniform_data(r, cmd, stage->layout, NriBindPoint_COMPUTE, 2, uniforms, sizeof(*uniforms))) return false;

    const NriBufferBarrierDesc barrier = {.buffer = arguments->buffer,
                                          .before = {.access = arguments->access, .stages = arguments->stages},
                                          .after = {.access = NriAccessBits_ARGUMENT_BUFFER, .stages = NriStageBits_INDIRECT}};
    r->gpu->core.CmdBarrier(cmd, &(NriBarrierDesc){.buffers = &barrier, .bufferNum = 1});
    arguments->access = NriAccessBits_ARGUMENT_BUFFER;
    arguments->stages = NriStageBits_INDIRECT;

    r->gpu->core.CmdSetPipeline(cmd, stage->pipeline);
    r->gpu->core.CmdDispatchIndirect(cmd, arguments->buffer, 0u);

    return true;
}

static uint32_t probe_wavefront_groups64(uint64_t threads) {
    const uint64_t groups = (threads + 63u) / 64u;

    return groups && groups <= UINT32_MAX ? (uint32_t)groups : 0u;
}

static PROBE_WAVEFRONT_UNIFORMS probe_wavefront_data(const BVH *tree, const BEAM_GRID *beams, DIRECTIONAL_LIGHT sun, SKY sky, VOLUMETRICS_LIGHTING volumetrics,
                                                     uint32_t probe_count, uint32_t sample_offset, uint32_t block_samples, uint32_t bounce_index) {
    const BVH_NODE *root = &tree->nodes[0];
    float scene_scale = fmaxf(root->max[0] - root->min[0], fmaxf(root->max[1] - root->min[1], root->max[2] - root->min[2]));

    if (scene_scale < 1.0f) scene_scale = 1.0f;

    const float epsilon = scene_scale * 2.0e-5f;

    return (PROBE_WAVEFRONT_UNIFORMS){.probe_count = probe_count,
                                      .sample_offset = sample_offset,
                                      .samples_per_block = block_samples,
                                      .total_samples = volumetrics.probe_samples,
                                      .node_count = tree->node_count,
                                      .triangle_count = tree->triangle_count,
                                      .bounce_index = bounce_index,
                                      .max_bounces = PROBE_MAX_BOUNCES,
                                      .beam_depth = beams->depth,
                                      .emissive_samples = volumetrics.emissive_samples,
                                      .sun_direction_intensity = {sun.direction.x, sun.direction.y, sun.direction.z, sun.intensity},
                                      .sun_color_radius = {sun.color.x, sun.color.y, sun.color.z, sun.angular_radius},
                                      .sky_zenith = {sky.zenith.x, sky.zenith.y, sky.zenith.z, 1.0f},
                                      .sky_horizon = {sky.horizon.x, sky.horizon.y, sky.horizon.z, 1.0f},
                                      .bake_params = {epsilon, 0.72f, sky.intensity, tree->emissive_weight},
                                      .emissive_params = {volumetrics.emissive_probe_intensity, 0.0f, 0.0f, 0.0f},
                                      .beam_origin = {beams->origin.x, beams->origin.y, beams->origin.z, 0.0f},
                                      .beam_step = {beams->step.x, beams->step.y, beams->step.z, 0.0f}};
}

static NriBuffer *probe_wavefront_readback_buffer(RENDERER *r, uint64_t bytes) {
    const NriBufferDesc desc = {.size = bytes};

    NriBuffer *buffer = NULL;

    if (r->gpu->core.CreateCommittedBuffer(r->gpu->device, NriMemoryLocation_HOST_READBACK, 1.0f, &desc, &buffer) != NriResult_SUCCESS) return NULL;

    return buffer;
}

static bool probe_wavefront_copy_to_readback(RENDERER *r, NriCommandBuffer *cmd, PROBE_WAVEFRONT_BUFFER *source, NriBuffer *destination, uint64_t bytes) {
    if (!source || !source->buffer || !destination) return false;

    const NriBufferBarrierDesc barrier = {.buffer = source->buffer,
                                          .before = {.access = source->access, .stages = source->stages},
                                          .after = {.access = NriAccessBits_COPY_SOURCE, .stages = NriStageBits_COPY}};

    r->gpu->core.CmdBarrier(cmd, &(NriBarrierDesc){.buffers = &barrier, .bufferNum = 1});
    source->access = NriAccessBits_COPY_SOURCE;
    source->stages = NriStageBits_COPY;
    r->gpu->core.CmdCopyBuffer(cmd, destination, 0, source->buffer, 0, bytes);

    return true;
}

bool bake_probe_grid_fast(RENDERER *r, PROBE_GRID *grid, const BVH *tree, const BEAM_GRID *beams, PROBE_BAKE_PROGRESS_FN progress) {
    if (!r || !r->gpu->device || !grid || !grid->probes || !tree || !tree->node_count || !tree->triangle_count || !beams || !beams->shadow_depth) return false;

    const uint64_t probe_count64 = (uint64_t)grid->count_x * grid->count_y * grid->count_z;

    if (!probe_count64 || probe_count64 > UINT32_MAX) return false;

    const uint32_t probe_count = (uint32_t)probe_count64;
    const uint64_t max_rays = probe_count64 * PROBE_BLOCK_SAMPLES;
    const uint64_t output_bytes = probe_count64 * PROBE_OUTPUT_STRIDE_BYTES;
    const uint64_t state_bytes = max_rays * PROBE_RAY_STATE_BYTES;
    const uint64_t result_bytes = max_rays * sizeof(float[4]);
    const uint64_t accum_bytes = probe_count64 * PROBE_ACCUM_BYTES;

    float(*positions)[4] = malloc((size_t)probe_count * sizeof(float[4]));

    if (!positions) return false;

    for (uint32_t i = 0; i < probe_count; ++i) memcpy(positions[i], grid->probes[i].position, sizeof(float[4]));

    float *beam_data = beam_expand(beams);

    if (!beam_data || !beams->shadow_depth) {
        free(positions);
        free(beam_data);

        return false;
    }

    const size_t beam_count = (size_t)beams->width * beams->height * beams->depth;
    const size_t depth_count = (size_t)beams->width * beams->height;
    float *expanded = realloc(beam_data, (beam_count + depth_count) * sizeof(float));

    if (!expanded) {
        free(positions);
        free(beam_data);

        return false;
    }

    beam_data = expanded;

    memcpy(beam_data + beam_count, beams->shadow_depth, depth_count * sizeof(float));

    PROBE_WAVEFRONT_BUFFER position_buffer = probe_wavefront_uploaded(r, positions, probe_count64 * sizeof(float[4]), sizeof(float[4]));
    PROBE_WAVEFRONT_BUFFER source_nodes = probe_wavefront_uploaded(r, tree->nodes, (uint64_t)tree->node_count * sizeof(*tree->nodes), sizeof(BVH_NODE));
    PROBE_WAVEFRONT_BUFFER source_triangles =
        probe_wavefront_uploaded(r, tree->triangles, (uint64_t)tree->triangle_count * sizeof(*tree->triangles), sizeof(BVH_TRIANGLE));
    PROBE_WAVEFRONT_BUFFER sun_beams = probe_wavefront_uploaded(r, beam_data, (beam_count + depth_count) * sizeof(float), sizeof(float));

    free(positions);
    free(beam_data);

    const bool scratch_ready =
        probe_wavefront_scratch_ensure(r, (uint64_t)tree->node_count * PROBE_PACKED_NODE_BYTES, (uint64_t)tree->triangle_count * PROBE_PACKED_TRIANGLE_BYTES,
                                       state_bytes, result_bytes, accum_bytes, output_bytes);
    PROBE_WAVEFRONT_SCRATCH *scratch = r->probe_scratch;

    PROBE_WAVEFRONT_BUFFER packed_nodes = scratch_ready ? scratch->packed_nodes : (PROBE_WAVEFRONT_BUFFER){0};
    PROBE_WAVEFRONT_BUFFER packed_triangles = scratch_ready ? scratch->packed_triangles : (PROBE_WAVEFRONT_BUFFER){0};
    PROBE_WAVEFRONT_BUFFER states_a = scratch_ready ? scratch->states_a : (PROBE_WAVEFRONT_BUFFER){0};
    PROBE_WAVEFRONT_BUFFER states_b = scratch_ready ? scratch->states_b : (PROBE_WAVEFRONT_BUFFER){0};
    PROBE_WAVEFRONT_BUFFER results = scratch_ready ? scratch->results : (PROBE_WAVEFRONT_BUFFER){0};
    PROBE_WAVEFRONT_BUFFER accums = scratch_ready ? scratch->accums : (PROBE_WAVEFRONT_BUFFER){0};
    PROBE_WAVEFRONT_BUFFER coefficients = scratch_ready ? scratch->coefficients : (PROBE_WAVEFRONT_BUFFER){0};
    PROBE_WAVEFRONT_BUFFER counters = scratch_ready ? scratch->counters : (PROBE_WAVEFRONT_BUFFER){0};
    PROBE_WAVEFRONT_BUFFER dispatch_args = scratch_ready ? scratch->dispatch_args : (PROBE_WAVEFRONT_BUFFER){0};
    NriBuffer *counter_readback = scratch_ready ? scratch->counter_readback : NULL;
    NriBuffer *output_readback = scratch_ready ? scratch->output_readback : NULL;
    PROBE_WAVEFRONT_PIPELINES pipelines = scratch_ready ? scratch->pipelines : (PROBE_WAVEFRONT_PIPELINES){0};

    bool good = position_buffer.buffer && source_nodes.buffer && source_triangles.buffer && sun_beams.buffer && scratch_ready;

    const uint32_t prep_groups = probe_wavefront_groups64(tree->node_count > tree->triangle_count ? tree->node_count : tree->triangle_count);
    const uint32_t probe_groups = probe_wavefront_groups64(probe_count);
    PROBE_WAVEFRONT_UNIFORMS uniforms = probe_wavefront_data(tree, beams, r->sun, r->sky, r->volumetrics, probe_count, 0u, PROBE_BLOCK_SAMPLES, 0u);

    if (!prep_groups || !probe_groups) good = false;

    if (good) {
        NriCommandAllocator *allocator = NULL;
        NriCommandBuffer *cmd = NULL;
        good = gpu_begin_commands(r, &allocator, &cmd) == NriResult_SUCCESS;

        if (good) {
            good = gpu_timestamp_begin(r, cmd, 0u);
            PROBE_WAVEFRONT_BUFFER *prepare_reads[] = {&source_nodes, &source_triangles};

            PROBE_WAVEFRONT_BUFFER *prepare_writes[] = {&packed_nodes, &packed_triangles};

            PROBE_WAVEFRONT_BUFFER *reset_writes[] = {&counters, &accums, &coefficients};

            PROBE_WAVEFRONT_BUFFER *validate_reads[] = {&position_buffer, &packed_nodes, &packed_triangles};

            PROBE_WAVEFRONT_BUFFER *validate_writes[] = {&accums};
            good = probe_wavefront_dispatch(r, cmd, &pipelines.prepare, prepare_reads, prepare_writes, &uniforms, prep_groups) &&
                   probe_wavefront_dispatch(r, cmd, &pipelines.reset, NULL, reset_writes, &uniforms, probe_groups) &&
                   probe_wavefront_dispatch(r, cmd, &pipelines.validate, validate_reads, validate_writes, &uniforms, probe_groups);

            if (good) good = gpu_timestamp_end(r, cmd, 0u);

            if (good)
                good = gpu_submit_commands(r, allocator, cmd);
            else
                gpu_abort_commands(r, allocator, cmd);
        }

        if (good) gpu_timestamp_log(r, 0u, "probe prepare + validate");
    }

    uint32_t completed = 0u;
    uint32_t active = probe_count;
    const uint32_t max_samples = r->volumetrics.probe_samples;

    if (!max_samples || max_samples > 65536u) good = false;

    if (good && progress && !progress(0u, max_samples, active)) good = false;

    while (good && completed < max_samples && active) {
        const uint32_t block = max_samples - completed > PROBE_BLOCK_SAMPLES ? PROBE_BLOCK_SAMPLES : max_samples - completed;

        uniforms = probe_wavefront_data(tree, beams, r->sun, r->sky, r->volumetrics, probe_count, completed, block, 0u);

        NriCommandAllocator *allocator = NULL;
        NriCommandBuffer *cmd = NULL;
        good = gpu_begin_commands(r, &allocator, &cmd) == NriResult_SUCCESS;

        if (!good) break;

        good = gpu_timestamp_begin(r, cmd, 2u);

        if (completed) {
            PROBE_WAVEFRONT_BUFFER *reset_writes[] = {&counters, &accums, &coefficients};

            good = probe_wavefront_dispatch(r, cmd, &pipelines.reset, NULL, reset_writes, &uniforms, probe_groups);
        }

        PROBE_WAVEFRONT_BUFFER *primary_reads[] = {&position_buffer, &accums, &packed_nodes, &packed_triangles};

        PROBE_WAVEFRONT_BUFFER *primary_writes[] = {&results, &states_a, &counters};

        if (good)
            good = probe_wavefront_dispatch(r, cmd, &pipelines.primary, primary_reads, primary_writes, &uniforms,
                                            probe_wavefront_groups64((uint64_t)probe_count * block));

        for (uint32_t bounce = 0; good && bounce < PROBE_MAX_BOUNCES; ++bounce) {
            uniforms.bounce_index = bounce;

            PROBE_WAVEFRONT_BUFFER *args_reads[] = {&counters};
            PROBE_WAVEFRONT_BUFFER *args_writes[] = {&dispatch_args};
            good = probe_wavefront_dispatch(r, cmd, &pipelines.args, args_reads, args_writes, &uniforms, 1u);

            PROBE_WAVEFRONT_BUFFER *input = bounce & 1u ? &states_b : &states_a;
            PROBE_WAVEFRONT_BUFFER *output = bounce & 1u ? &states_a : &states_b;
            PROBE_WAVEFRONT_BUFFER *bounce_reads[] = {&packed_nodes, &packed_triangles, input, &sun_beams};
            PROBE_WAVEFRONT_BUFFER *bounce_writes[] = {&results, output, &counters};

            if (good) good = probe_wavefront_dispatch_indirect(r, cmd, &pipelines.bounce, bounce_reads, bounce_writes, &uniforms, &dispatch_args);
        }

        uniforms.bounce_index = 0u;

        PROBE_WAVEFRONT_BUFFER *reduce_reads[] = {&results};
        PROBE_WAVEFRONT_BUFFER *reduce_writes[] = {&accums, &coefficients, &counters};

        if (good) good = probe_wavefront_dispatch(r, cmd, &pipelines.reduce, reduce_reads, reduce_writes, &uniforms, probe_count);

        if (good) good = gpu_timestamp_end(r, cmd, 2u);

        if (good) good = probe_wavefront_copy_to_readback(r, cmd, &counters, counter_readback, 4u * sizeof(uint32_t));

        if (good)
            good = gpu_submit_commands(r, allocator, cmd);
        else
            gpu_abort_commands(r, allocator, cmd);

        if (!good) break;
        gpu_timestamp_log(r, 2u, "probe trace block");

        const uint32_t *values = r->gpu->core.MapBuffer(counter_readback, 0, 4u * sizeof(uint32_t));

        if (!values) {
            good = false;

            break;
        }

        active = values[3];

        r->gpu->core.UnmapBuffer(counter_readback);
        completed += block;

        const uint32_t progress_total = active ? max_samples : completed;

        if (progress && !progress(completed, progress_total, active)) good = false;

        SDL_Log("B: probe wavefront %u/%u spp | %u active probes", completed, max_samples, active);
    }

    if (good && uniforms.emissive_samples && tree->emissive_weight > 0.0f) {
        NriCommandAllocator *allocator = NULL;
        NriCommandBuffer *cmd = NULL;
        good = gpu_begin_commands(r, &allocator, &cmd) == NriResult_SUCCESS;

        if (good) {
            PROBE_WAVEFRONT_BUFFER *emissive_reads[] = {&position_buffer, &packed_nodes, &packed_triangles};
            PROBE_WAVEFRONT_BUFFER *emissive_writes[] = {&coefficients};
            good = probe_wavefront_dispatch(r, cmd, &pipelines.emissive, emissive_reads, emissive_writes, &uniforms, probe_groups);

            if (good)
                good = gpu_submit_commands(r, allocator, cmd);
            else
                gpu_abort_commands(r, allocator, cmd);
        }
    }

    if (good) {
        NriCommandAllocator *allocator = NULL;
        NriCommandBuffer *cmd = NULL;
        good = gpu_begin_commands(r, &allocator, &cmd) == NriResult_SUCCESS;

        if (good) {
            good = probe_wavefront_copy_to_readback(r, cmd, &coefficients, output_readback, output_bytes);

            if (good)
                good = gpu_submit_commands(r, allocator, cmd);
            else
                gpu_abort_commands(r, allocator, cmd);
        }
    }

    if (good) {
        const float(*values)[4] = r->gpu->core.MapBuffer(output_readback, 0, output_bytes);

        if (!values)
            good = false;
        else {
            for (uint32_t i = 0; i < probe_count; ++i) {
                for (uint32_t coefficient = 0; coefficient < 9u; ++coefficient)
                    memcpy(grid->probes[i].coefficients[coefficient], values[i * 9u + coefficient], sizeof(float[4]));
                grid->probes[i].position[3] = values[i * 9u][3];
            }

            r->gpu->core.UnmapBuffer(output_readback);
        }
    }

    if (scratch_ready) {
        scratch->packed_nodes = packed_nodes;
        scratch->packed_triangles = packed_triangles;
        scratch->states_a = states_a;
        scratch->states_b = states_b;
        scratch->results = results;
        scratch->accums = accums;
        scratch->coefficients = coefficients;
        scratch->counters = counters;
        scratch->dispatch_args = dispatch_args;
    }

    probe_wavefront_release_buffer(r, &sun_beams);
    probe_wavefront_release_buffer(r, &source_triangles);
    probe_wavefront_release_buffer(r, &source_nodes);
    probe_wavefront_release_buffer(r, &position_buffer);

    return good;
}

bool bake_probe_grid(RENDERER *r, PROBE_GRID *grid, Uint32 samples) {
    if (!r || !grid || !grid->probes) return false;

    const uint64_t count = (uint64_t)grid->count_x * grid->count_y * grid->count_z;

    if (!count || count > UINT32_MAX) return false;

    const uint64_t output_size = count * 9u * sizeof(float[4]);
    const uint64_t input_size = count * sizeof(float[4]);

    if (output_size > UINT32_MAX || input_size > UINT32_MAX) return false;

    const Uint32 output_bytes = (Uint32)output_size;
    const Uint32 input_bytes = (Uint32)input_size;

    float(*positions)[4] = malloc(input_bytes);

    if (!positions) return false;

    for (uint32_t i = 0; i < (uint32_t)count; ++i) memcpy(positions[i], grid->probes[i].position, sizeof(positions[i]));

    NriBuffer *input = gpu_upload_buffer(r, NriBufferUsageBits_SHADER_RESOURCE, positions, input_bytes, sizeof(float[4]));

    free(positions);

    const NriBufferDesc output_desc = {.size = output_bytes, .structureStride = sizeof(float[4]), .usage = NriBufferUsageBits_SHADER_RESOURCE_STORAGE};

    NriBuffer *output = NULL;

    if (r->gpu->core.CreateCommittedBuffer(r->gpu->device, NriMemoryLocation_DEVICE, 1.0f, &output_desc, &output) != NriResult_SUCCESS) output = NULL;

    NriPipeline *pipeline = gpu_compile_compute(r, r->probe_layout, "shaders/compute.hlsl", "probe_cs", "BUILD_PROBE_CS");

    bool good = input && output && pipeline;

    if (good) {
        NriCommandAllocator *allocator = NULL;
        NriCommandBuffer *cmd = NULL;
        good = gpu_begin_commands(r, &allocator, &cmd) == NriResult_SUCCESS;

        if (good) {
            const BAKE_UNIFORMS u = bake_data(r, 0u, 0u, samples, 0u, 0u);

            good = bake_bind_probe_resources(r, cmd, input, r->bvh_node_buffer, r->bvh_triangle_buffer, output, &u, sizeof(u));

            if (good) {
                r->gpu->core.CmdSetPipeline(cmd, pipeline);

                r->gpu->core.CmdDispatch(cmd, &(NriDispatchDesc){.workGroupNumX = (Uint32)count, .workGroupNumY = 1, .workGroupNumZ = 1});

                good = gpu_submit_commands(r, allocator, cmd);
            } else {
                gpu_abort_commands(r, allocator, cmd);
            }
        }
    }

    if (good) good = bake_read_probe_buffer(r, output, grid, output_bytes, (uint32_t)count);

    if (pipeline) r->gpu->core.DestroyPipeline(pipeline);
    release_buffer(r, output);
    release_buffer(r, input);

    return good;
}

void release_bake_resources(RENDERER *r) {
    if (!r || !r->gpu->device) return;

    release_buffer(r, r->bvh_node_buffer);
    release_buffer(r, r->bvh_triangle_buffer);
    release_buffer(r, r->lightmap_sample_buffer);
    release_buffer(r, r->lightmap_full_sample_buffer);
    release_buffer(r, r->lightmap_sparse_sample_buffer);
    release_buffer(r, r->lightmap_probe_buffer);
    release_buffer(r, r->lightmap_patch_map_buffer);
    release_buffer(r, r->lightmap_patch_anchor_buffer);

    for (uint32_t i = 0; i < 2u; ++i) {
        release_buffer(r, r->lightmap_active_buffer[i]);
        release_buffer(r, r->lightmap_active_count[i]);
    }

    release_buffer(r, r->lightmap_dispatch_args);
    release_texture(r, r->lightmap_scratch);
    release_texture(r, r->lightmap_direct);

    if (r->bake_pipeline) r->gpu->core.DestroyPipeline(r->bake_pipeline);

    if (r->lightmap_queue_reset_pipeline) r->gpu->core.DestroyPipeline(r->lightmap_queue_reset_pipeline);

    if (r->lightmap_queue_args_pipeline) r->gpu->core.DestroyPipeline(r->lightmap_queue_args_pipeline);

    r->bvh_node_buffer = NULL;
    r->bvh_triangle_buffer = NULL;
    r->bvh_triangle_count = 0u;
    r->bvh_emissive_weight = 0.0f;
    r->lightmap_sample_buffer = NULL;
    r->lightmap_full_sample_buffer = NULL;
    r->lightmap_sparse_sample_buffer = NULL;
    r->lightmap_probe_buffer = NULL;
    r->lightmap_patch_map_buffer = NULL;
    r->lightmap_patch_anchor_buffer = NULL;

    for (uint32_t i = 0; i < 2u; ++i) {
        r->lightmap_active_buffer[i] = NULL;
        r->lightmap_active_count[i] = NULL;
    }

    r->lightmap_dispatch_args = NULL;
    r->lightmap_active_capacity = 0u;
    r->lightmap_scratch = NULL;
    r->lightmap_direct = NULL;
    r->bake_pipeline = NULL;
    r->lightmap_queue_reset_pipeline = NULL;
    r->lightmap_queue_args_pipeline = NULL;
}

typedef enum BAKE_PHASE { BAKE_PHASE_INIT = 0, BAKE_PHASE_SUN, BAKE_PHASE_PROBES, BAKE_PHASE_SEED, BAKE_PHASE_LIGHTMAP } BAKE_PHASE;

typedef struct BAKE_JOB {
    RENDERER *renderer;
    const MESH *scene;
    const GLTF_SCENE *visual;
    const LIGHTMAP *layout;
    const struct LIGHT *light;
    SKY sky;
    VOLUMETRICS_LIGHTING volumetrics;
    char *path;
    char *worker_path;
    uint64_t scene_hash;
    uint64_t layout_hash;
    uint64_t volume_hash;
    uint64_t beam_hash;
    SDL_Thread *thread;
    SDL_AtomicInt cancel;
    SDL_AtomicInt done;
    SDL_AtomicInt success;
    SDL_AtomicInt phase;
    SDL_AtomicInt phase_done;
    SDL_AtomicInt phase_total;
    SDL_AtomicInt phase_active;
    SDL_AtomicInt phase_started_ms;
    Uint64 started;
    uint64_t previous_cache_bytes;
    Uint64 save_seen_at;
    uint64_t save_seen_bytes;
    char error[256];
} BAKE_JOB;

static BAKE_JOB *g_bake;
static Uint64 g_title_tick;

static double bake_elapsed_ms(Uint64 started) {
    return (double)(SDL_GetPerformanceCounter() - started) * 1000.0 / (double)SDL_GetPerformanceFrequency();
}

static char *bake_path_suffix(const char *path, const char *suffix) {
    if (!path || !suffix) return NULL;

    const size_t a = strlen(path);
    const size_t b = strlen(suffix);

    if (a > SIZE_MAX - b - 1u) return NULL;

    char *result = malloc(a + b + 1u);

    if (!result) return NULL;

    memcpy(result, path, a);
    memcpy(result + a, suffix, b + 1u);

    return result;
}

static uint64_t bake_file_size(const char *path) {
    if (!path) return 0u;

    FILE *file = fopen(path, "rb");

    if (!file) return 0u;

    if (fseek(file, 0, SEEK_END) != 0) {
        fclose(file);
        return 0u;
    }

    const long size = ftell(file);
    fclose(file);

    return size > 0 ? (uint64_t)size : 0u;
}

static const char *bake_phase_name(int phase) {
    switch ((BAKE_PHASE)phase) {
    case BAKE_PHASE_SUN:
        return "SUN";
    case BAKE_PHASE_PROBES:
        return "PROBE";
    case BAKE_PHASE_SEED:
        return "CACHE";
    case BAKE_PHASE_LIGHTMAP:
        return "LIGHTMAP";
    default:
        return "BAKE";
    }
}

static void bake_set_phase(BAKE_PHASE phase, Uint32 done, Uint32 total, Uint32 active) {
    BAKE_JOB *job = g_bake;

    if (!job) return;

    const int previous = SDL_GetAtomicInt(&job->phase);

    if (previous != (int)phase) SDL_SetAtomicInt(&job->phase_started_ms, (int)(Uint32)SDL_GetTicks());

    SDL_SetAtomicInt(&job->phase_done, (int)done);
    SDL_SetAtomicInt(&job->phase_total, (int)total);
    SDL_SetAtomicInt(&job->phase_active, (int)active);
    SDL_SetAtomicInt(&job->phase, (int)phase);
}

static bool bake_cancelled(void) {
    return g_bake && SDL_GetAtomicInt(&g_bake->cancel) != 0;
}

static bool bake_probe_stats(const RENDERER *renderer, Uint32 *minimum, double *average, Uint32 *maximum, Uint32 *valid_count) {
    if (!renderer || !renderer->volume_probes.probes) return false;

    const uint64_t probe_count = (uint64_t)renderer->volume_probes.count_x * renderer->volume_probes.count_y * renderer->volume_probes.count_z;

    if (!probe_count) return false;

    Uint32 min_samples = UINT32_MAX;
    Uint32 max_samples = 0u;
    uint64_t sample_sum = 0u;
    Uint32 measured = 0u;

    for (uint64_t i = 0u; i < probe_count; ++i) {
        const PROBE *probe = &renderer->volume_probes.probes[i];

        if (probe->position[3] <= 0.0f) continue;

        const float encoded = probe->coefficients[2][3];

        if (!isfinite(encoded) || encoded < 1.0f || encoded > 65536.0f) continue;

        const Uint32 samples = (Uint32)(encoded + 0.5f);

        if (samples < min_samples) min_samples = samples;
        if (samples > max_samples) max_samples = samples;

        sample_sum += samples;
        measured++;
    }

    if (!measured) return false;

    if (minimum) *minimum = min_samples;
    if (average) *average = (double)sample_sum / (double)measured;
    if (maximum) *maximum = max_samples;
    if (valid_count) *valid_count = measured;

    return true;
}

static void bake_remove_worker_files(const char *path) {
    if (!path) return;

    remove(path);

    char *temporary = bake_path_suffix(path, ".tmp");

    if (temporary) {
        remove(temporary);
        free(temporary);
    }
}

static bool bake_copy_cache(const char *source_path, const char *destination_path) {
    FILE *source = fopen(source_path, "rb");

    if (!source) return false;

    FILE *destination = fopen(destination_path, "wb");

    if (!destination) {
        fclose(source);
        return false;
    }

    unsigned char *buffer = malloc(1024u * 1024u);
    bool good = buffer != NULL;

    while (good && !bake_cancelled()) {
        const size_t count = fread(buffer, 1, 1024u * 1024u, source);

        if (count && fwrite(buffer, 1, count, destination) != count) good = false;

        if (count < 1024u * 1024u) {
            if (ferror(source)) good = false;
            break;
        }
    }

    free(buffer);

    if (fclose(source) != 0) good = false;
    if (fclose(destination) != 0) good = false;
    if (bake_cancelled()) good = false;
    if (!good) remove(destination_path);

    return good;
}

static void bake_set_error(BAKE_JOB *job, const char *message) {
    if (!job) return;

    const char *text = message && *message ? message : "unknown bake error";
    snprintf(job->error, sizeof(job->error), "%s", text);
}

static bool bake_probe_progress(Uint32 done, Uint32 total, Uint32 active) {
    bake_set_phase(BAKE_PHASE_PROBES, done, total, active);
    return !bake_cancelled();
}

static bool bake_make_probe_grid(const MESH *mesh, float spacing, PROBE_GRID *grid) {
    if (!mesh || !grid || spacing <= 0.0f) return false;

    memset(grid, 0, sizeof(*grid));

    const VEC3 extent = v3_sub(mesh->bounds.max, mesh->bounds.min);

    if (!isfinite(extent.x) || !isfinite(extent.y) || !isfinite(extent.z) || extent.x < 0.0f || extent.y < 0.0f || extent.z < 0.0f) return false;

    grid->count_x = (uint32_t)ceilf(extent.x / spacing) + 1u;
    grid->count_y = (uint32_t)ceilf(extent.y / spacing) + 1u;
    grid->count_z = (uint32_t)ceilf(extent.z / spacing) + 1u;

    const uint64_t count = (uint64_t)grid->count_x * grid->count_y * grid->count_z;

    if (!count || count > 16384u) return false;

    grid->origin = mesh->bounds.min;
    grid->spacing = spacing;
    grid->probes = calloc((size_t)count, sizeof(*grid->probes));

    if (!grid->probes) return false;

    for (uint32_t z = 0; z < grid->count_z; ++z) {
        for (uint32_t y = 0; y < grid->count_y; ++y) {
            for (uint32_t x = 0; x < grid->count_x; ++x) {
                const size_t index = x + (size_t)grid->count_x * (y + (size_t)grid->count_y * z);
                PROBE *probe = &grid->probes[index];

                probe->position[0] = grid->origin.x + x * spacing;
                probe->position[1] = grid->origin.y + y * spacing;
                probe->position[2] = grid->origin.z + z * spacing;
                probe->position[3] = 1.0f;
            }
        }
    }

    return true;
}

static bool bake_write_fast_seed(BAKE_JOB *job, PROBE_GRID *probes, BEAM_GRID *beams) {
    unsigned char black_pixel[8] = {0};
    CACHED_LIGHTMAP seed = {0};

    seed.width = 1u;
    seed.height = 1u;
    seed.pixels = black_pixel;
    seed.volume_probes = *probes;
    seed.beams = *beams;

    const uint64_t stale_layout = job->layout_hash ^ UINT64_C(0x9e3779b97f4a7c15);

    bake_set_phase(BAKE_PHASE_SEED, 0u, 0u, 0u);

    return cache_write(job->worker_path, job->scene_hash, stale_layout, job->volume_hash, job->beam_hash, &seed);
}

static bool bake_prepare_fast_components(BAKE_JOB *job, RENDERER *worker) {
    Uint64 started = SDL_GetPerformanceCounter();
    BVH tree = {0};
    PROBE_GRID probes = {0};
    BEAM_GRID beams = {0};

    if (!job->light || job->light->type != LIGHT_DIRECTIONAL) return false;

    worker->sun = job->light->directional;
    worker->sun.direction = v3_normalize(worker->sun.direction);
    worker->sky = job->sky;
    worker->volumetrics = job->volumetrics;

    if (v3_len_sq(worker->sun.direction) <= 0.0f) return false;

    bool good = bvh_build(&tree, job->scene, job->visual);

    if (good) SDL_Log("B: fast probe BVH built in %.2f ms", bake_elapsed_ms(started));

    bake_set_phase(BAKE_PHASE_SUN, 0u, 0u, 0u);
    started = SDL_GetPerformanceCounter();

    if (good) good = beam_build(&beams, job->scene, &tree, worker->sun.direction);
    if (good) SDL_Log("B: fast sun field took %.2f ms", bake_elapsed_ms(started));

    started = SDL_GetPerformanceCounter();

    if (good)
        good = bake_make_probe_grid(job->scene, job->volumetrics.probe_spacing, &probes) &&
               bake_probe_grid_fast(worker, &probes, &tree, &beams, bake_probe_progress);
    if (good) SDL_Log("B: wavefront volume probes took %.2f ms", bake_elapsed_ms(started));
    if (good) good = bake_write_fast_seed(job, &probes, &beams);

    bvh_free(&tree);
    free(probes.probes);
    beam_free(&beams);

    return good;
}

static int SDLCALL bake_thread_main(void *userdata) {
    BAKE_JOB *job = userdata;
    RENDERER worker = {0};

    if (!bake_worker_init(&worker)) {
        bake_set_error(job, SDL_GetError());
        SDL_SetAtomicInt(&job->done, 1);
        return 1;
    }

    SDL_Log("B: offscreen bake GPU device started");

    bool fast = !bake_cancelled() && bake_prepare_fast_components(job, &worker);

    if (!fast && !bake_cancelled()) {
        SDL_Log("B: fast probe path unavailable; falling back to original bake: %s", *SDL_GetError() ? SDL_GetError() : "unknown error");
        bake_remove_worker_files(job->worker_path);

        if (job->path && bake_copy_cache(job->path, job->worker_path)) SDL_Log("B: fallback seeded previous cache");
    }

    bake_set_phase(BAKE_PHASE_LIGHTMAP, 0u, 0u, 0u);

    bool good = !bake_cancelled() && renderer_rebake_current_scene(&worker, job->scene, job->visual, job->layout, job->light, &job->sky, &job->volumetrics,
                                                                   job->worker_path, job->scene_hash, job->layout_hash, job->volume_hash, job->beam_hash);

    if (!good) bake_set_error(job, SDL_GetError());

    bake_worker_deinit(&worker);

    if (bake_cancelled()) good = false;
    if (!good) bake_remove_worker_files(job->worker_path);

    SDL_SetAtomicInt(&job->success, good ? 1 : 0);
    SDL_SetAtomicInt(&job->done, 1);
    SDL_Log("B: offscreen bake %s after %.2f ms", good ? "finished" : "stopped", bake_elapsed_ms(job->started));

    return good ? 0 : 1;
}

static bool bake_publish_cache(const char *worker_path, const char *path) {
    char *backup = bake_path_suffix(path, ".previous");

    if (!backup) return false;

    remove(backup);

    bool had_previous = false;

    if (rename(path, backup) == 0) {
        had_previous = true;
    } else if (errno != ENOENT) {
        SDL_SetError("could not preserve previous bake cache: %s", strerror(errno));
        free(backup);
        return false;
    }

    if (rename(worker_path, path) != 0) {
        const int publish_error = errno;

        if (had_previous) rename(backup, path);

        SDL_SetError("could not publish bake cache: %s", strerror(publish_error));
        free(backup);
        return false;
    }

    if (had_previous) remove(backup);

    free(backup);
    return true;
}

static void bake_free_job(BAKE_JOB *job) {
    if (!job) return;

    free(job->worker_path);
    free(job->path);
    free(job);
}

bool bake_start(RENDERER *renderer, const MESH *scene, const GLTF_SCENE *visual, const LIGHTMAP *layout, const struct LIGHT *light, const SKY *sky,
                const VOLUMETRICS_LIGHTING *volumetrics, const char *path, uint64_t scene_hash, uint64_t layout_hash, uint64_t volume_hash,
                uint64_t beam_hash) {
    if (!renderer || !renderer->gpu->device || !scene || !visual || !layout || !light || light->type != LIGHT_DIRECTIONAL || !sky || !volumetrics || !path)
        return false;

    if (g_bake) {
        SDL_SetError("a bake is already in progress");
        return false;
    }

    BAKE_JOB *job = calloc(1, sizeof(*job));

    if (!job) return false;

    job->renderer = renderer;
    job->scene = scene;
    job->visual = visual;
    job->layout = layout;
    job->light = light;
    job->sky = *sky;
    job->volumetrics = *volumetrics;
    job->scene_hash = scene_hash;
    job->layout_hash = layout_hash;
    job->volume_hash = volume_hash;
    job->beam_hash = beam_hash;
    job->started = SDL_GetPerformanceCounter();
    job->path = bake_path_suffix(path, "");
    job->worker_path = bake_path_suffix(path, ".worker");
    job->previous_cache_bytes = bake_file_size(path);

    if (!job->path || !job->worker_path) {
        bake_free_job(job);
        SDL_SetError("could not initialize offscreen bake job");
        return false;
    }

    bake_remove_worker_files(job->worker_path);
    g_bake = job;
    SDL_SetAtomicInt(&job->phase_started_ms, (int)(Uint32)SDL_GetTicks());
    job->thread = SDL_CreateThread(bake_thread_main, "pony-bake", job);

    if (!job->thread) {
        g_bake = NULL;
        bake_remove_worker_files(job->worker_path);
        bake_free_job(job);
        return false;
    }

    renderer->bake_stage = "offscreen GPU bake";

    if (renderer->gpu->window) SDL_SetWindowTitle(renderer->gpu->window, "BAKE | 0.0s");

    SDL_Log("B: full-speed offscreen rebake started; render device remains independent");
    return true;
}

bool bake_active(RENDERER *renderer) {
    return g_bake && (!renderer || g_bake->renderer == renderer);
}

void bake_update_title(RENDERER *renderer) {
    if (!renderer || !renderer->gpu->window) return;

    const Uint64 now = SDL_GetTicks();

    if (now - g_title_tick < 100u) return;

    g_title_tick = now;

    const double frame_ms = renderer->frame_time_ms;
    const double fps = frame_ms > 0.001 ? 1000.0 / frame_ms : 0.0;
    char title[256];
    BAKE_JOB *job = g_bake;

    if (job && job->renderer == renderer) {
        const int phase = SDL_GetAtomicInt(&job->phase);
        const Uint32 done = (Uint32)SDL_GetAtomicInt(&job->phase_done);
        const Uint32 total = (Uint32)SDL_GetAtomicInt(&job->phase_total);
        const Uint32 active = (Uint32)SDL_GetAtomicInt(&job->phase_active);
        const Uint32 phase_started = (Uint32)SDL_GetAtomicInt(&job->phase_started_ms);
        const double phase_seconds = (double)((Uint32)now - phase_started) / 1000.0;
        char *temporary = bake_path_suffix(job->worker_path, ".tmp");
        const uint64_t save_bytes = temporary ? bake_file_size(temporary) : 0u;

        free(temporary);

        if (phase == BAKE_PHASE_LIGHTMAP && save_bytes) {
            if (!job->save_seen_at) {
                job->save_seen_at = now;
                job->save_seen_bytes = save_bytes;
            }

            const double save_seconds = (double)(now - job->save_seen_at) / 1000.0;
            const double mib = (double)save_bytes / (1024.0 * 1024.0);
            const double expected = job->previous_cache_bytes ? (double)job->previous_cache_bytes / (1024.0 * 1024.0) : 0.0;
            const double rate = save_seconds > 0.05 ? ((double)(save_bytes - job->save_seen_bytes) / (1024.0 * 1024.0)) / save_seconds : 0.0;

            if (expected > 0.0) {
                const double percent = fmin(100.0, mib * 100.0 / expected);
                snprintf(title, sizeof(title), "%.1fms | %.1ffps | SAVE %.1f/%.1fMiB %.0f%% | %.0fMiB/s", frame_ms, fps, mib, expected, percent, rate);
            } else {
                snprintf(title, sizeof(title), "%.1fms | %.1ffps | SAVE %.1fMiB | %.0fMiB/s", frame_ms, fps, mib, rate);
            }
        } else if (phase == BAKE_PHASE_PROBES && total) {
            const double percent = 100.0 * (double)done / (double)total;
            const double eta = done && done < total ? phase_seconds * (double)(total - done) / (double)done : 0.0;

            if (eta > 0.0) {
                snprintf(title, sizeof(title), "%.1fms | %.1ffps | PROBE %u/%u %.0f%% | A%u | %.1fs ETA %.1fs | F%u D%u", frame_ms, fps, done, total, percent,
                         active, phase_seconds, eta, renderer->show_volume ? 1u : 0u, renderer->debug_view);
            } else {
                snprintf(title, sizeof(title), "%.1fms | %.1ffps | PROBE %u/%u %.0f%% | A%u | %.1fs | F%u D%u", frame_ms, fps, done, total, percent, active,
                         phase_seconds, renderer->show_volume ? 1u : 0u, renderer->debug_view);
            }
        } else {
            snprintf(title, sizeof(title), "%.1fms | %.1ffps | %s %.1fs | F%u D%u", frame_ms, fps, bake_phase_name(phase), phase_seconds,
                     renderer->show_volume ? 1u : 0u, renderer->debug_view);
        }
    } else {
        Uint32 min_samples = 0u;
        Uint32 max_samples = 0u;
        Uint32 measured = 0u;
        double avg_samples = 0.0;

        if (renderer->has_bake && bake_probe_stats(renderer, &min_samples, &avg_samples, &max_samples, &measured)) {
            snprintf(title, sizeof(title), "%.1fms | %.1ffps | READY | LM %ux%u | P %u/%.0f/%u x%u | F%u D%u", frame_ms, fps, renderer->lightmap_width,
                     renderer->lightmap_height, min_samples, avg_samples, max_samples, measured, renderer->show_volume ? 1u : 0u, renderer->debug_view);
        } else {
            snprintf(title, sizeof(title), "%.1fms | %.1ffps | %s | LM %ux%u | F%u D%u", frame_ms, fps, renderer->has_bake ? "READY" : "UNBAKED",
                     renderer->lightmap_width, renderer->lightmap_height, renderer->show_volume ? 1u : 0u, renderer->debug_view);
        }
    }

    SDL_SetWindowTitle(renderer->gpu->window, title);
}

void bake_update(RENDERER *renderer) {
    BAKE_JOB *job = g_bake;

    if (!job || job->renderer != renderer || !SDL_GetAtomicInt(&job->done)) return;

    SDL_WaitThread(job->thread, NULL);
    job->thread = NULL;

    bool good = SDL_GetAtomicInt(&job->success) != 0 && SDL_GetAtomicInt(&job->cancel) == 0;

    if (good) {
        good = renderer_load_cached_lightmap(renderer, job->worker_path, job->scene_hash, job->layout_hash, job->volume_hash, job->beam_hash, job->layout);

        if (!good) bake_set_error(job, SDL_GetError());
    }

    if (good && !bake_publish_cache(job->worker_path, job->path)) SDL_Log("B: new lighting is active but cache publish failed: %s", SDL_GetError());

    if (good) {
        renderer->bake_stage = NULL;

        Uint32 min_samples = 0u;
        Uint32 max_samples = 0u;
        Uint32 measured = 0u;
        double avg_samples = 0.0;

        if (bake_probe_stats(renderer, &min_samples, &avg_samples, &max_samples, &measured)) {
            SDL_Log("B: probe samples %u/%.1f/%u across %u valid probes", min_samples, avg_samples, max_samples, measured);
        }

        SDL_Log("B: bake ready on render device after %.2f ms", bake_elapsed_ms(job->started));
    } else {
        bake_remove_worker_files(job->worker_path);
        renderer->bake_stage = NULL;
        SDL_Log("B: offscreen bake failed; previous lighting retained: %s", job->error[0] ? job->error : "unknown bake error");
    }

    g_bake = NULL;
    bake_free_job(job);
}

void bake_cancel(RENDERER *renderer) {
    BAKE_JOB *job = g_bake;

    if (!job || (renderer && job->renderer != renderer)) return;

    SDL_SetAtomicInt(&job->cancel, 1);

    if (job->thread) {
        SDL_Log("B: waiting for offscreen GPU bake to stop");
        SDL_WaitThread(job->thread, NULL);
        job->thread = NULL;
    }

    bake_remove_worker_files(job->worker_path);
    g_bake = NULL;
    bake_free_job(job);
    SDL_Log("B: offscreen bake cancelled");
}
