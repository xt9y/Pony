#include "NRIDescs.h"
#include "SDL3/SDL_log.h"
#include "game.h"
#include "gpu.h"

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FRAME_CONSTANTS_BUFFER_SIZE 256u
#define FAR_PLANE 10000.0f
#define NEAR_PLANE 0.05f

typedef struct FRAME_CONSTANTS {
    MAT4 view_projection;
    MAT4 inverse_view_projection;
    MAT4 previous_view_projection;
    float camera_position[4];
    float resolution[4];
    float trace_params[4];
    uint32_t trace_limits[4];
} FRAME_CONSTANTS;

_Static_assert(sizeof(GLTF_VERTEX) == 36u, "GLTF_VERTEX GPU layout changed");
_Static_assert(sizeof(GPU_OBJECT) == 224u, "GPU_OBJECT GPU layout changed");
_Static_assert(sizeof(GPU_MATERIAL) == 48u, "GPU_MATERIAL GPU layout changed");
_Static_assert(sizeof(FRAME_CONSTANTS) == 256u, "FRAME_CONSTANTS GPU layout changed");
_Static_assert(sizeof(TRACE_HIT) == 32u, "TRACE_HIT GPU layout changed");
_Static_assert(sizeof(TRACE_RAY) == 48u, "TRACE_RAY GPU layout changed");
_Static_assert(sizeof(GPU_SDF_MODEL) == 128u, "GPU_SDF_MODEL GPU layout changed");
_Static_assert(sizeof(GPU_LIGHT) == 64u, "GPU_LIGHT GPU layout changed");
_Static_assert(sizeof(SURFACE_CACHE_ENTRY) == 112u, "SURFACE_CACHE_ENTRY GPU layout changed");

static bool update_radiance_constants(RENDERER *renderer);
static bool update_radiance_scene_descriptors(RENDERER *renderer);

static MAT4 mat4_identity(void) {
    return (MAT4){.m[0] = 1.0f, .m[5] = 1.0f, .m[10] = 1.0f, .m[15] = 1.0f};
}

static MAT4 mat4_mul(MAT4 a, MAT4 b) {
    MAT4 r = {{0}};

    for (uint32_t row = 0; row < 4; ++row)
        for (uint32_t column = 0; column < 4; ++column)
            for (uint32_t k = 0; k < 4; ++k)
                r.m[row * 4 + column] += a.m[row * 4 + k] * b.m[k * 4 + column];

    return r;
}

static MAT4 mat4_transform(TRANSFORM t) {
    const float x = t.rotation[0], y = t.rotation[1], z = t.rotation[2], w = t.rotation[3];
    const float x2 = x + x, y2 = y + y, z2 = z + z;
    const float xx = x * x2, yy = y * y2, zz = z * z2;
    const float xy = x * y2, xz = x * z2, yz = y * z2;
    const float wx = w * x2, wy = w * y2, wz = w * z2;
    MAT4 r = mat4_identity();
    r.m[0] = (1.0f - (yy + zz)) * t.scale.x;
    r.m[1] = (xy + wz) * t.scale.x;
    r.m[2] = (xz - wy) * t.scale.x;
    r.m[4] = (xy - wz) * t.scale.y;
    r.m[5] = (1.0f - (xx + zz)) * t.scale.y;
    r.m[6] = (yz + wx) * t.scale.y;
    r.m[8] = (xz + wy) * t.scale.z;
    r.m[9] = (yz - wx) * t.scale.z;
    r.m[10] = (1.0f - (xx + yy)) * t.scale.z;
    r.m[12] = t.position.x;
    r.m[13] = t.position.y;
    r.m[14] = t.position.z;

    return r;
}

static VEC3 mat4_point(MAT4 matrix, VEC3 point) {
    return v3(
        point.x * matrix.m[0] + point.y * matrix.m[4] + point.z * matrix.m[8] + matrix.m[12],
        point.x * matrix.m[1] + point.y * matrix.m[5] + point.z * matrix.m[9] + matrix.m[13],
        point.x * matrix.m[2] + point.y * matrix.m[6] + point.z * matrix.m[10] + matrix.m[14]
    );
}

static float triangle_area(VEC3 a, VEC3 b, VEC3 c) {
    const VEC3 ab = v3_sub(b, a);
    const VEC3 ac = v3_sub(c, a);

    return 0.5f * sqrtf(v3_len_sq(v3_cross(ab, ac)));
}

static float emissive_luminance(const float emissive[3]) {
    return emissive[0] * 0.2126f + emissive[1] * 0.7152f + emissive[2] * 0.0722f;
}

static MAT4 mat4_normal_transform(TRANSFORM t) {
    const float x = t.rotation[0], y = t.rotation[1], z = t.rotation[2], w = t.rotation[3];
    const float x2 = x + x, y2 = y + y, z2 = z + z;
    const float xx = x * x2, yy = y * y2, zz = z * z2;
    const float xy = x * y2, xz = x * z2, yz = y * z2;
    const float wx = w * x2, wy = w * y2, wz = w * z2;
    const float sx = fabsf(t.scale.x) > 1.0e-8f ? 1.0f / t.scale.x : 0.0f;
    const float sy = fabsf(t.scale.y) > 1.0e-8f ? 1.0f / t.scale.y : 0.0f;
    const float sz = fabsf(t.scale.z) > 1.0e-8f ? 1.0f / t.scale.z : 0.0f;
    MAT4 r = mat4_identity();
    r.m[0] = (1.0f - (yy + zz)) * sx;
    r.m[1] = (xy + wz) * sx;
    r.m[2] = (xz - wy) * sx;
    r.m[4] = (xy - wz) * sy;
    r.m[5] = (1.0f - (xx + zz)) * sy;
    r.m[6] = (yz + wx) * sy;
    r.m[8] = (xz + wy) * sz;
    r.m[9] = (yz - wx) * sz;
    r.m[10] = (1.0f - (xx + yy)) * sz;

    return r;
}

static MAT4 mat4_view(CAMERA camera) {
    const VEC3 forward = v3_normalize(camera.forward);
    const VEC3 right = v3_normalize(v3_cross(forward, camera.up));
    const VEC3 up = v3_cross(right, forward);
    const VEC3 pos = camera.position;
    MAT4 r = {{0}};
    r.m[0] = right.x;
    r.m[4] = right.y;
    r.m[8] = right.z;
    r.m[1] = up.x;
    r.m[5] = up.y;
    r.m[9] = up.z;
    r.m[2] = -forward.x;
    r.m[6] = -forward.y;
    r.m[10] = -forward.z;
    r.m[12] = -(pos.x * right.x + pos.y * right.y + pos.z * right.z);
    r.m[13] = -(pos.x * up.x + pos.y * up.y + pos.z * up.z);
    r.m[14] = pos.x * forward.x + pos.y * forward.y + pos.z * forward.z;
    r.m[15] = 1.0f;

    return r;
}

static MAT4 mat4_reverse_z_projection(float fov_y, float aspect, float near_p, float far_p) {
    const float half_rad = fov_y * 0.00872664625997164788f;
    const float y = 1.0f / tanf(half_rad);
    const float x = y / aspect;
    const float inv_depth = 1.0f / (far_p - near_p);
    MAT4 r = {{0}};
    r.m[0] = x;
    r.m[5] = y;
    r.m[10] = near_p * inv_depth;
    r.m[11] = -1.0f;
    r.m[14] = near_p * far_p * inv_depth;

    return r;
}

static bool mat4_inverse(MAT4 matrix, MAT4 *inverse) {
    float a[4][8];

    for (uint32_t row = 0; row < 4; ++row) {
        for (uint32_t column = 0; column < 4; ++column)
            a[row][column] = matrix.m[row * 4 + column];

        for (uint32_t column = 0; column < 4; ++column)
            a[row][4 + column] = row == column ? 1.0f : 0.0f;
    }

    for (uint32_t column = 0; column < 4; ++column) {
        uint32_t pivot = column;
        float pivot_size = fabsf(a[pivot][column]);

        for (uint32_t row = column + 1; row < 4; ++row) {
            const float size = fabsf(a[row][column]);

            if (size > pivot_size) {
                pivot = row;
                pivot_size = size;
            }
        }

        if (pivot_size < 1.0e-8f) return false;

        if (pivot != column) {
            for (uint32_t i = 0; i < 8; ++i) {
                const float temporary = a[column][i];

                a[column][i] = a[pivot][i];
                a[pivot][i] = temporary;
            }
        }

        const float scale = 1.0f / a[column][column];

        for (uint32_t i = 0; i < 8; ++i)
            a[column][i] *= scale;

        for (uint32_t row = 0; row < 4; ++row) {
            if (row == column) continue;

            const float factor = a[row][column];

            for (uint32_t i = 0; i < 8; ++i)
                a[row][i] -= factor * a[column][i];
        }
    }

    for (uint32_t row = 0; row < 4; ++row)
        for (uint32_t column = 0; column < 4; ++column)
            inverse->m[row * 4 + column] = a[row][4 + column];

    return true;
}


static bool resolve_face_material(const MESH_FACE *face, const GLTF_SCENE *visual, uint32_t *material) {

    if (!face || !visual || !material) return false;

    uint32_t index = face->material;

    if (index == UINT32_MAX) index = visual->default_material;

    if (index >= visual->material_count) return false;

    *material = index;

    return true;
}

static void update_orbit_camera(RENDERER *renderer) {
    const float cp = cosf(renderer->camera.pitch);

    renderer->camera.position =
        v3(renderer->camera.target.x + renderer->camera.distance * cp * cosf(renderer->camera.yaw),
           renderer->camera.target.y + renderer->camera.distance * sinf(renderer->camera.pitch),
           renderer->camera.target.z + renderer->camera.distance * cp * sinf(renderer->camera.yaw));
    renderer->camera.forward = v3_normalize(v3_sub(renderer->camera.target, renderer->camera.position));
    renderer->camera.up = v3(0.0f, 1.0f, 0.0f);
}

static bool load_shader(const char *path, void **data, size_t *size) {
    FILE *file = fopen(path, "rb");

    if (!file) return false;

    if (fseek(file, 0, SEEK_END) != 0) {
        fclose(file);

        return false;
    }

    const long length = ftell(file);

    if (length <= 0) {
        fclose(file);

        return false;
    }

    rewind(file);

    void *bytes = malloc((size_t)length);

    if (!bytes) {
        fclose(file);

        return false;
    }

    if (fread(bytes, 1, (size_t)length, file) != (size_t)length) {
        free(bytes);
        fclose(file);

        return false;
    }

    fclose(file);
    *data = bytes;
    *size = (size_t)length;
    return true;
}

static bool compile_present_shader(const char *define) {
    char command[1024];
    const int length = snprintf(
        command,
        sizeof(command),
        "slangc shader.hlsl -entry PS_Present -stage fragment -target spirv -profile sm_6_6 -capability spirv_1_5 -matrix-layout-row-major -fvk-use-dx-layout -O3 -D%s -o "
        "build/shaders/present.runtime.ps.spv",
        define
    );
    return length > 0 && (size_t)length < sizeof(command) && system(command) == 0;
}

static bool clear_surface_cache(RENDERER *renderer) {

    SURFACE_CACHE *cache = &renderer->surface_cache;

    if (!cache->keys || !cache->capacity) return false;

    uint32_t *zero_keys = calloc(cache->capacity, sizeof(*zero_keys));

    if (!zero_keys) return false;

    const NriAccessStage storage = {
        .access = NriAccessBits_SHADER_RESOURCE_STORAGE,
        .stages = NriStageBits_COMPUTE_SHADER
    };

    const bool ok = gpu_upload_buffer(renderer->gpu, cache->keys, zero_keys, storage);

    free(zero_keys);

    if (!ok) return false;

    cache->keys_state = storage;

    return true;
}

static bool create_render_texture(RENDERER *renderer, RENDER_TEXTURE *target, NriFormat format, NriTextureUsageBits usage, NriTextureView attachment_type, NriPlaneBits planes) {
    memset(target, 0, sizeof(*target));
    target->format = format;

    const NriTextureDesc desc = {
        .type = NriTextureType_TEXTURE_2D,
        .usage = usage,
        .format = format,
        .width = (NriDim_t)renderer->width,
        .height = (NriDim_t)renderer->height,
        .depth = 1,
        .mipNum = 1,
        .layerNum = 1,
        .sampleNum = 1
    };

    if (!gpu_create_texture(renderer->gpu, &desc, NriMemoryLocation_DEVICE, &target->texture)) return false;

    const NriTextureViewDesc attachment = {
        .texture = target->texture,
        .type = attachment_type,
        .format = format,
        .mipNum = 1,
        .layerNum = 1,
        .sliceNum = 1,
        .planes = planes
    };

    if (renderer->gpu->core.CreateTextureView(&attachment, &target->attachment) != NriResult_SUCCESS) return false;

    const NriTextureViewDesc srv = {
        .texture = target->texture,
        .type = NriTextureView_TEXTURE,
        .format = format,
        .mipNum = 1,
        .layerNum = 1,
        .sliceNum = 1,
        .planes = planes
    };

    if (renderer->gpu->core.CreateTextureView(&srv, &target->srv) != NriResult_SUCCESS) return false;

    return true;
}

static void destroy_render_texture(RENDERER *renderer, RENDER_TEXTURE *target) {
    if (target->attachment) renderer->gpu->core.DestroyDescriptor(target->attachment);

    if (target->srv) renderer->gpu->core.DestroyDescriptor(target->srv);

    if (target->texture) gpu_destroy_texture(renderer->gpu, target->texture);
    memset(target, 0, sizeof(*target));
}

static bool create_compute_texture(RENDERER *renderer, COMPUTE_TEXTURE *target, uint32_t width, uint32_t height) {
    memset(target, 0, sizeof(*target));
    target->width = width;
    target->height = height;

    const NriTextureDesc desc = {
        .type = NriTextureType_TEXTURE_2D,
        .usage = NriTextureUsageBits_SHADER_RESOURCE | NriTextureUsageBits_SHADER_RESOURCE_STORAGE,
        .format = NriFormat_RGBA16_SFLOAT,
        .width = (NriDim_t)width,
        .height = (NriDim_t)height,
        .depth = 1,
        .mipNum = 1,
        .layerNum = 1,
        .sampleNum = 1
    };

    if (!gpu_create_texture(renderer->gpu, &desc, NriMemoryLocation_DEVICE, &target->texture)) return false;

    const NriTextureViewDesc srv = {
        .texture = target->texture,
        .type = NriTextureView_TEXTURE,
        .format = NriFormat_RGBA16_SFLOAT,
        .mipNum = 1,
        .layerNum = 1,
        .sliceNum = 1,
        .planes = NriPlaneBits_COLOR
    };

    const NriTextureViewDesc uav = {
        .texture = target->texture,
        .type = NriTextureView_STORAGE_TEXTURE,
        .format = NriFormat_RGBA16_SFLOAT,
        .mipNum = 1,
        .layerNum = 1,
        .sliceNum = 1,
        .planes = NriPlaneBits_COLOR
    };

    if (renderer->gpu->core.CreateTextureView(&srv, &target->srv) != NriResult_SUCCESS || renderer->gpu->core.CreateTextureView(&uav, &target->uav) != NriResult_SUCCESS)
        return false;

    return true;
}

static void destroy_compute_texture(RENDERER *renderer, COMPUTE_TEXTURE *target) {
    if (target->uav) renderer->gpu->core.DestroyDescriptor(target->uav);

    if (target->srv) renderer->gpu->core.DestroyDescriptor(target->srv);

    if (target->texture) gpu_destroy_texture(renderer->gpu, target->texture);
    memset(target, 0, sizeof(*target));
}

static bool create_buffer_view(RENDERER *renderer, NriBuffer *buffer, NriBufferView type, uint64_t size, uint32_t stride, NriDescriptor **descriptor) {
    const NriBufferViewDesc view = {
        .buffer = buffer,
        .type = type,
        .offset = 0,
        .size = size,
        .structureStride = stride
    };

    return renderer->gpu->core.CreateBufferView(&view, descriptor) == NriResult_SUCCESS;
}

static void destroy_trace_buffer(RENDERER *renderer, TRACE_BUFFER *buffer) {
    if (buffer->uav) renderer->gpu->core.DestroyDescriptor(buffer->uav);

    if (buffer->srv) renderer->gpu->core.DestroyDescriptor(buffer->srv);

    if (buffer->buffer) gpu_destroy_buffer(renderer->gpu, buffer->buffer);
    memset(buffer, 0, sizeof(*buffer));
}

static bool create_trace_buffer(RENDERER *renderer, uint32_t capacity) {
    destroy_trace_buffer(renderer, &renderer->trace_hits);
    renderer->trace_hits.capacity = capacity;

    const uint64_t size = (uint64_t)capacity * sizeof(TRACE_HIT);

    const NriBufferDesc desc = {
        .size = size,
        .structureStride = sizeof(TRACE_HIT),
        .usage = NriBufferUsageBits_SHADER_RESOURCE | NriBufferUsageBits_SHADER_RESOURCE_STORAGE
    };

    if (!gpu_create_buffer(renderer->gpu, &desc, NriMemoryLocation_DEVICE, &renderer->trace_hits.buffer)) return false;

    if (!create_buffer_view(renderer, renderer->trace_hits.buffer, NriBufferView_STRUCTURED_BUFFER, size, sizeof(TRACE_HIT), &renderer->trace_hits.srv)) return false;

    if (!create_buffer_view(renderer, renderer->trace_hits.buffer, NriBufferView_STORAGE_STRUCTURED_BUFFER, size, sizeof(TRACE_HIT), &renderer->trace_hits.uav)) return false;

    return true;
}

static void destroy_trace_queue(RENDERER *renderer) {
    TRACE_QUEUE *queue = &renderer->miss_queue;

    if (queue->rays_uav) renderer->gpu->core.DestroyDescriptor(queue->rays_uav);

    if (queue->count_uav) renderer->gpu->core.DestroyDescriptor(queue->count_uav);

    if (queue->dispatch_args_uav) renderer->gpu->core.DestroyDescriptor(queue->dispatch_args_uav);

    if (queue->rays) gpu_destroy_buffer(renderer->gpu, queue->rays);

    if (queue->count) gpu_destroy_buffer(renderer->gpu, queue->count);

    if (queue->dispatch_args) gpu_destroy_buffer(renderer->gpu, queue->dispatch_args);
    memset(queue, 0, sizeof(*queue));
}

static bool create_trace_queue(RENDERER *renderer, uint32_t capacity) {
    destroy_trace_queue(renderer);

    TRACE_QUEUE *queue = &renderer->miss_queue;
    queue->capacity = capacity;

    const NriBufferDesc rays_desc = {
        .size = (uint64_t)capacity * sizeof(TRACE_RAY),
        .structureStride = sizeof(TRACE_RAY),
        .usage = NriBufferUsageBits_SHADER_RESOURCE_STORAGE
    };

    const NriBufferDesc count_desc = {
        .size = sizeof(uint32_t),
        .structureStride = sizeof(uint32_t),
        .usage = NriBufferUsageBits_SHADER_RESOURCE_STORAGE
    };

    const NriBufferDesc args_desc = {
        .size = 3u * sizeof(uint32_t),
        .structureStride = sizeof(uint32_t),
        .usage = NriBufferUsageBits_SHADER_RESOURCE_STORAGE | NriBufferUsageBits_ARGUMENT
    };

    if (!gpu_create_buffer(renderer->gpu, &rays_desc, NriMemoryLocation_DEVICE, &queue->rays) ||
        !gpu_create_buffer(renderer->gpu, &count_desc, NriMemoryLocation_DEVICE, &queue->count) ||
        !gpu_create_buffer(renderer->gpu, &args_desc, NriMemoryLocation_DEVICE, &queue->dispatch_args))
        return false;

    if (!create_buffer_view(renderer, queue->rays, NriBufferView_STORAGE_STRUCTURED_BUFFER, rays_desc.size, sizeof(TRACE_RAY), &queue->rays_uav) ||
        !create_buffer_view(renderer, queue->count, NriBufferView_STORAGE_STRUCTURED_BUFFER, count_desc.size, sizeof(uint32_t), &queue->count_uav) ||
        !create_buffer_view(renderer, queue->dispatch_args, NriBufferView_STORAGE_STRUCTURED_BUFFER, args_desc.size, sizeof(uint32_t), &queue->dispatch_args_uav))
        return false;

    return true;
}

static void destroy_surface_cache(RENDERER *renderer) {
    SURFACE_CACHE *cache = &renderer->surface_cache;

    if (cache->keys_uav) renderer->gpu->core.DestroyDescriptor(cache->keys_uav);

    if (cache->entries_uav) renderer->gpu->core.DestroyDescriptor(cache->entries_uav);

    if (cache->keys) gpu_destroy_buffer(renderer->gpu, cache->keys);

    if (cache->entries) gpu_destroy_buffer(renderer->gpu, cache->entries);
    memset(cache, 0, sizeof(*cache));
}

static bool create_surface_cache(RENDERER *renderer) {
    destroy_surface_cache(renderer);

    SURFACE_CACHE *cache = &renderer->surface_cache;
    cache->capacity = SURFACE_CACHE_CAPACITY;

    const NriBufferDesc keys_desc = {
        .size = (uint64_t)cache->capacity * sizeof(uint32_t),
        .structureStride = sizeof(uint32_t),
        .usage = NriBufferUsageBits_SHADER_RESOURCE_STORAGE
    };

    const NriBufferDesc entries_desc = {
        .size = (uint64_t)cache->capacity * sizeof(SURFACE_CACHE_ENTRY),
        .structureStride = sizeof(SURFACE_CACHE_ENTRY),
        .usage = NriBufferUsageBits_SHADER_RESOURCE_STORAGE
    };

    if (!gpu_create_buffer(renderer->gpu, &keys_desc, NriMemoryLocation_DEVICE, &cache->keys) ||
        !gpu_create_buffer(renderer->gpu, &entries_desc, NriMemoryLocation_DEVICE, &cache->entries))
        return false;

    if (!create_buffer_view(renderer, cache->keys, NriBufferView_STORAGE_STRUCTURED_BUFFER, keys_desc.size, sizeof(uint32_t), &cache->keys_uav) ||
        !create_buffer_view(renderer, cache->entries, NriBufferView_STORAGE_STRUCTURED_BUFFER, entries_desc.size, sizeof(SURFACE_CACHE_ENTRY), &cache->entries_uav))
        return false;

    return clear_surface_cache(renderer);
}

static bool clear_radiance_surface_cache(RENDERER *renderer) {
    SURFACE_CACHE *cache = &renderer->radiance_surface_cache;
    if (!cache->keys || !cache->capacity) return false;
    uint32_t *zero_keys = calloc(cache->capacity, sizeof(*zero_keys));
    if (!zero_keys) return false;
    const NriAccessStage storage = {
        .access = NriAccessBits_SHADER_RESOURCE_STORAGE,
        .stages = NriStageBits_COMPUTE_SHADER
    };
    const bool ok = gpu_upload_buffer(renderer->gpu, cache->keys, zero_keys, storage);
    free(zero_keys);
    if (!ok) return false;
    cache->keys_state = storage;
    return true;
}

static void destroy_radiance_surface_cache(RENDERER *renderer) {
    SURFACE_CACHE *cache = &renderer->radiance_surface_cache;
    if (cache->keys_uav) renderer->gpu->core.DestroyDescriptor(cache->keys_uav);
    if (cache->entries_uav) renderer->gpu->core.DestroyDescriptor(cache->entries_uav);
    if (cache->keys) gpu_destroy_buffer(renderer->gpu, cache->keys);
    if (cache->entries) gpu_destroy_buffer(renderer->gpu, cache->entries);
    memset(cache, 0, sizeof(*cache));
}

static bool create_radiance_surface_cache(RENDERER *renderer) {
    destroy_radiance_surface_cache(renderer);
    SURFACE_CACHE *cache = &renderer->radiance_surface_cache;
    cache->capacity = SURFACE_CACHE_CAPACITY;

    const NriBufferDesc keys_desc = {
        .size = (uint64_t)cache->capacity * sizeof(uint32_t),
        .structureStride = sizeof(uint32_t),
        .usage = NriBufferUsageBits_SHADER_RESOURCE_STORAGE
    };
    const NriBufferDesc entries_desc = {
        .size = (uint64_t)cache->capacity * sizeof(SURFACE_RADIANCE_ENTRY),
        .structureStride = sizeof(SURFACE_RADIANCE_ENTRY),
        .usage = NriBufferUsageBits_SHADER_RESOURCE_STORAGE
    };

    if (!gpu_create_buffer(renderer->gpu, &keys_desc, NriMemoryLocation_DEVICE, &cache->keys) ||
        !gpu_create_buffer(renderer->gpu, &entries_desc, NriMemoryLocation_DEVICE, &cache->entries))
        return false;
    if (!create_buffer_view(renderer, cache->keys, NriBufferView_STORAGE_STRUCTURED_BUFFER, keys_desc.size, sizeof(uint32_t), &cache->keys_uav) ||
        !create_buffer_view(renderer, cache->entries, NriBufferView_STORAGE_STRUCTURED_BUFFER, entries_desc.size, sizeof(SURFACE_RADIANCE_ENTRY), &cache->entries_uav))
        return false;

    return clear_radiance_surface_cache(renderer);
}

static void destroy_radiance_scene_gpu_resources(RENDERER *renderer) {
    if (!renderer) return;

    RADIANCE_SCENE_DATA *scene = &renderer->radiance_scene;

    if (renderer->gpu) {
        if (scene->triangles_srv) renderer->gpu->core.DestroyDescriptor(scene->triangles_srv);
        if (scene->emissive_triangles_srv) renderer->gpu->core.DestroyDescriptor(scene->emissive_triangles_srv);
        if (scene->triangles) gpu_destroy_buffer(renderer->gpu, scene->triangles);
        if (scene->emissive_triangles) gpu_destroy_buffer(renderer->gpu, scene->emissive_triangles);
    }

    scene->triangles = NULL;
    scene->emissive_triangles = NULL;
    scene->triangles_srv = NULL;
    scene->emissive_triangles_srv = NULL;
    scene->triangles_state = (NriAccessStage){0};
    scene->emissive_triangles_state = (NriAccessStage){0};
}

static void destroy_radiance_scene_data(RENDERER *renderer) {
    if (!renderer) return;

    destroy_radiance_scene_gpu_resources(renderer);
    free(renderer->radiance_scene.cpu_triangles);
    free(renderer->radiance_scene.cpu_emissive_triangles);

    memset(&renderer->radiance_scene, 0, sizeof(renderer->radiance_scene));
}

static bool build_radiance_scene_data(RENDERER *renderer, SCENE *scene) {

    destroy_radiance_scene_data(renderer);

    if (!renderer || !scene || !renderer->cpu_objects) return false;

    uint64_t triangle_count = 0;
    uint64_t possible_emitter_count = 0;

    for (uint32_t scene_index = 0; scene_index < scene->object_count; ++scene_index) {
        OBJECT *object = &scene->objects[scene_index];

        if (object->type != MODEL) continue;

        struct MODEL *model = object->data;

        if (!model || !model->geometry || !model->visual || !model->geometry->faces.buffer || !model->geometry->vertices.buffer || !model->visual->vertices ||
            !model->visual->materials) {
            return false;
        }

        MESH *geometry = model->geometry;
        GLTF_SCENE *visual = model->visual;

        if (geometry->faces.count > SIZE_MAX / 3u) {
            return false;
        }

        if (visual->vertex_count != geometry->faces.count * 3u) {
            SDL_Log("Radiance triangle mismatch: %zu mesh faces vs %zu visual vertices", geometry->faces.count, visual->vertex_count);

            return false;
        }

        triangle_count += geometry->faces.count;

        if (triangle_count > UINT32_MAX) return false;

        MESH_FACE *faces = geometry->faces.buffer;

        for (size_t face_index = 0; face_index < geometry->faces.count; ++face_index) {
            uint32_t material_index;

            if (!resolve_face_material(&faces[face_index], visual, &material_index)) {
                return false;
            }

            const GLTF_MATERIAL *material = &visual->materials[material_index];

            if (emissive_luminance(material->emissive) > 1.0e-6f) {
                ++possible_emitter_count;
            }
        }
    }

    if (!triangle_count) return false;

    if (possible_emitter_count > UINT32_MAX) return false;

    GPU_SCENE_TRIANGLE *triangles = calloc((size_t)triangle_count, sizeof(*triangles));

    GPU_EMISSIVE_TRIANGLE *emitters = possible_emitter_count ? calloc((size_t)possible_emitter_count, sizeof(*emitters)) : NULL;

    if (!triangles || (possible_emitter_count && !emitters)) {
        free(triangles);
        free(emitters);

        return false;
    }

    uint32_t triangle_index = 0;
    uint32_t emitter_index = 0;
    uint32_t object_index = 0;
    uint32_t material_offset = 0;

    double total_emissive_weight = 0.0;

    for (uint32_t scene_index = 0; scene_index < scene->object_count; ++scene_index) {
        OBJECT *object = &scene->objects[scene_index];

        if (object->type != MODEL) continue;

        struct MODEL *model = object->data;
        MESH *geometry = model->geometry;
        GLTF_SCENE *visual = model->visual;
        POINT *points = geometry->vertices.buffer;
        MESH_FACE *faces = geometry->faces.buffer;
        GPU_OBJECT *gpu_object = &renderer->cpu_objects[object_index];
        gpu_object->triangle_offset = triangle_index;

        for (uint32_t face_index = 0; face_index < (uint32_t)geometry->faces.count; ++face_index) {
            MESH_FACE *face = &faces[face_index];

            uint32_t local_material;

            if (!resolve_face_material(face, visual, &local_material)) {
                free(triangles);
                free(emitters);

                return false;
            }

            const uint32_t global_material = material_offset + local_material;

            const VEC3 a = points[face->indices[0]].p;
            const VEC3 b = points[face->indices[1]].p;
            const VEC3 c = points[face->indices[2]].p;

            const GLTF_VERTEX *v0 = &visual->vertices[(size_t)face_index * 3u + 0u];
            const GLTF_VERTEX *v1 = &visual->vertices[(size_t)face_index * 3u + 1u];
            const GLTF_VERTEX *v2 = &visual->vertices[(size_t)face_index * 3u + 2u];

            GPU_SCENE_TRIANGLE *triangle = &triangles[triangle_index];

            triangle->p0[0] = a.x;
            triangle->p0[1] = a.y;
            triangle->p0[2] = a.z;
            triangle->p0[3] = 1.0f;

            triangle->p1[0] = b.x;
            triangle->p1[1] = b.y;
            triangle->p1[2] = b.z;
            triangle->p1[3] = 1.0f;

            triangle->p2[0] = c.x;
            triangle->p2[1] = c.y;
            triangle->p2[2] = c.z;
            triangle->p2[3] = 1.0f;

            triangle->uv01[0] = v0->u;
            triangle->uv01[1] = v0->v;
            triangle->uv01[2] = v1->u;
            triangle->uv01[3] = v1->v;

            triangle->uv2_area[0] = v2->u;
            triangle->uv2_area[1] = v2->v;
            triangle->uv2_area[2] = triangle_area(a, b, c);
            triangle->uv2_area[3] = 0.0f;

            triangle->meta[0] = object_index;
            triangle->meta[1] = global_material;
            triangle->meta[2] = face_index;
            triangle->meta[3] = (uint32_t)object->state;

            const GLTF_MATERIAL *material = &visual->materials[local_material];

            const float luminance = emissive_luminance(material->emissive);

            if (luminance > 1.0e-6f) {
                const VEC3 world_a = mat4_point(gpu_object->world, a);
                const VEC3 world_b = mat4_point(gpu_object->world, b);
                const VEC3 world_c = mat4_point(gpu_object->world, c);

                const float world_area = triangle_area(world_a, world_b, world_c);

                if (world_area > 1.0e-8f) {
                    GPU_EMISSIVE_TRIANGLE *emitter = &emitters[emitter_index++];

                    emitter->meta[0] = triangle_index;
                    emitter->meta[1] = global_material;
                    emitter->meta[2] = object_index;
                    emitter->meta[3] = object->revision;

                    emitter->radiance_area[0] = material->emissive[0];
                    emitter->radiance_area[1] = material->emissive[1];
                    emitter->radiance_area[2] = material->emissive[2];
                    emitter->radiance_area[3] = world_area;

                    const float weight = world_area * luminance;

                    emitter->sampling[2] = weight;

                    total_emissive_weight += (double)weight;
                }
            }

            ++triangle_index;
        }

        material_offset += visual->material_count;

        ++object_index;
    }

    if (emitter_index && total_emissive_weight > 0.0) {

        double cumulative = 0.0;

        for (uint32_t i = 0; i < emitter_index; ++i) {
            GPU_EMISSIVE_TRIANGLE *emitter = &emitters[i];

            const float probability = (float)((double)emitter->sampling[2] / total_emissive_weight);

            cumulative += probability;

            emitter->sampling[0] = i + 1u == emitter_index ? 1.0f : (float)cumulative;
            emitter->sampling[1] = probability;
            // emitter->sampling[2] = emitter->sampling[2];
            emitter->sampling[3] = 0.0f;
        }
    } else {
        free(emitters);

        emitters = NULL;
        emitter_index = 0;
    }

    renderer->radiance_scene.cpu_triangles = triangles;
    renderer->radiance_scene.cpu_emissive_triangles = emitters;
    renderer->radiance_scene.triangle_count = triangle_index;
    renderer->radiance_scene.emissive_triangle_count = emitter_index;

    SDL_Log("Radiance scene : %u triangles, %u emissive triangles", triangle_index, emitter_index);

    return true;
}

static bool create_radiance_scene_gpu_resources(RENDERER *renderer) {
    if (!renderer) return false;

    RADIANCE_SCENE_DATA *scene = &renderer->radiance_scene;
    destroy_radiance_scene_gpu_resources(renderer);

    const uint32_t triangle_capacity = scene->triangle_count ? scene->triangle_count : 1u;
    const uint32_t emissive_capacity = scene->emissive_triangle_count ? scene->emissive_triangle_count : 1u;

    const NriBufferDesc triangle_desc = {
        .size = (uint64_t)triangle_capacity * sizeof(GPU_SCENE_TRIANGLE),
        .structureStride = sizeof(GPU_SCENE_TRIANGLE),
        .usage = NriBufferUsageBits_SHADER_RESOURCE
    };

    const NriBufferDesc emissive_desc = {
        .size = (uint64_t)emissive_capacity * sizeof(GPU_EMISSIVE_TRIANGLE),
        .structureStride = sizeof(GPU_EMISSIVE_TRIANGLE),
        .usage = NriBufferUsageBits_SHADER_RESOURCE
    };

    if (!gpu_create_buffer(renderer->gpu, &triangle_desc, NriMemoryLocation_DEVICE, &scene->triangles) ||
        !gpu_create_buffer(renderer->gpu, &emissive_desc, NriMemoryLocation_DEVICE, &scene->emissive_triangles)) {
        destroy_radiance_scene_gpu_resources(renderer);
        return false;
    }

    if (!create_buffer_view(renderer, scene->triangles, NriBufferView_STRUCTURED_BUFFER, triangle_desc.size, sizeof(GPU_SCENE_TRIANGLE), &scene->triangles_srv) ||
        !create_buffer_view(renderer, scene->emissive_triangles, NriBufferView_STRUCTURED_BUFFER, emissive_desc.size, sizeof(GPU_EMISSIVE_TRIANGLE), &scene->emissive_triangles_srv)) {
        destroy_radiance_scene_gpu_resources(renderer);
        return false;
    }

    const NriAccessStage read = {
        .access = NriAccessBits_SHADER_RESOURCE,
        .stages = NriStageBits_COMPUTE_SHADER
    };

    const GPU_SCENE_TRIANGLE zero_triangle = {0};
    const GPU_EMISSIVE_TRIANGLE zero_emitter = {0};
    const GPU_SCENE_TRIANGLE *triangle_data = scene->triangle_count ? scene->cpu_triangles : &zero_triangle;
    const GPU_EMISSIVE_TRIANGLE *emissive_data = scene->emissive_triangle_count ? scene->cpu_emissive_triangles : &zero_emitter;

    if (!gpu_upload_buffer(renderer->gpu, scene->triangles, triangle_data, read) ||
        !gpu_upload_buffer(renderer->gpu, scene->emissive_triangles, emissive_data, read)) {
        destroy_radiance_scene_gpu_resources(renderer);
        return false;
    }

    scene->triangles_state = read;
    scene->emissive_triangles_state = read;

    SDL_Log("Radiance GPU scene: %u triangles, %u emissive triangles", scene->triangle_count, scene->emissive_triangle_count);

    return true;
}


static void destroy_radiance_scene_fallbacks(RENDERER *renderer) {
    if (!renderer || !renderer->gpu) return;

    RADIANCE_SCENE_FALLBACKS *fallbacks = &renderer->radiance_fallbacks;

    if (fallbacks->dynamic_grid_cells_srv) renderer->gpu->core.DestroyDescriptor(fallbacks->dynamic_grid_cells_srv);
    if (fallbacks->dynamic_grid_indices_srv) renderer->gpu->core.DestroyDescriptor(fallbacks->dynamic_grid_indices_srv);
    if (fallbacks->global_sdf_clipmaps_srv) renderer->gpu->core.DestroyDescriptor(fallbacks->global_sdf_clipmaps_srv);
    if (fallbacks->global_sdf_page_table_srv) renderer->gpu->core.DestroyDescriptor(fallbacks->global_sdf_page_table_srv);
    if (fallbacks->global_sdf_bricks_srv) renderer->gpu->core.DestroyDescriptor(fallbacks->global_sdf_bricks_srv);
    if (fallbacks->global_sdf_surface_ids_srv) renderer->gpu->core.DestroyDescriptor(fallbacks->global_sdf_surface_ids_srv);

    if (fallbacks->dynamic_grid_cells) gpu_destroy_buffer(renderer->gpu, fallbacks->dynamic_grid_cells);
    if (fallbacks->dynamic_grid_indices) gpu_destroy_buffer(renderer->gpu, fallbacks->dynamic_grid_indices);
    if (fallbacks->global_sdf_clipmaps) gpu_destroy_buffer(renderer->gpu, fallbacks->global_sdf_clipmaps);
    if (fallbacks->global_sdf_page_table) gpu_destroy_buffer(renderer->gpu, fallbacks->global_sdf_page_table);
    if (fallbacks->global_sdf_bricks) gpu_destroy_buffer(renderer->gpu, fallbacks->global_sdf_bricks);
    if (fallbacks->global_sdf_surface_ids) gpu_destroy_buffer(renderer->gpu, fallbacks->global_sdf_surface_ids);

    memset(fallbacks, 0, sizeof(*fallbacks));
}

static bool create_radiance_fallback_buffer(RENDERER *renderer, uint32_t stride, NriBuffer **buffer, NriDescriptor **srv) {
    if (!stride || stride > sizeof(GPU_GLOBAL_SDF_CLIPMAP)) return false;

    const NriBufferDesc desc = {
        .size = stride,
        .structureStride = stride,
        .usage = NriBufferUsageBits_SHADER_RESOURCE
    };

    if (!gpu_create_buffer(renderer->gpu, &desc, NriMemoryLocation_DEVICE, buffer)) return false;
    if (!create_buffer_view(renderer, *buffer, NriBufferView_STRUCTURED_BUFFER, desc.size, stride, srv)) return false;

    const uint8_t zero[sizeof(GPU_GLOBAL_SDF_CLIPMAP)] = {0};
    const NriAccessStage read = {
        .access = NriAccessBits_SHADER_RESOURCE,
        .stages = NriStageBits_COMPUTE_SHADER
    };

    return gpu_upload_buffer(renderer->gpu, *buffer, zero, read);
}

static bool create_radiance_scene_fallbacks(RENDERER *renderer) {
    destroy_radiance_scene_fallbacks(renderer);

    RADIANCE_SCENE_FALLBACKS *fallbacks = &renderer->radiance_fallbacks;

    if (!create_radiance_fallback_buffer(renderer, sizeof(GPU_DYNAMIC_GRID_CELL), &fallbacks->dynamic_grid_cells, &fallbacks->dynamic_grid_cells_srv) ||
        !create_radiance_fallback_buffer(renderer, sizeof(uint32_t), &fallbacks->dynamic_grid_indices, &fallbacks->dynamic_grid_indices_srv) ||
        !create_radiance_fallback_buffer(renderer, sizeof(GPU_GLOBAL_SDF_CLIPMAP), &fallbacks->global_sdf_clipmaps, &fallbacks->global_sdf_clipmaps_srv) ||
        !create_radiance_fallback_buffer(renderer, sizeof(uint32_t), &fallbacks->global_sdf_page_table, &fallbacks->global_sdf_page_table_srv) ||
        !create_radiance_fallback_buffer(renderer, sizeof(float), &fallbacks->global_sdf_bricks, &fallbacks->global_sdf_bricks_srv) ||
        !create_radiance_fallback_buffer(renderer, sizeof(uint32_t), &fallbacks->global_sdf_surface_ids, &fallbacks->global_sdf_surface_ids_srv)) {
        destroy_radiance_scene_fallbacks(renderer);
        return false;
    }

    fallbacks->state = (NriAccessStage){
        .access = NriAccessBits_SHADER_RESOURCE,
        .stages = NriStageBits_COMPUTE_SHADER
    };

    return true;
}

static bool refresh_emissive_sampling(RENDERER *renderer) {
    RADIANCE_SCENE_DATA *scene = &renderer->radiance_scene;

    if (!scene->emissive_triangle_count) return true;
    if (!scene->cpu_emissive_triangles || !scene->cpu_triangles || !scene->emissive_triangles) return false;

    bool dirty = false;

    for (uint32_t i = 0; i < scene->emissive_triangle_count; ++i) {
        GPU_EMISSIVE_TRIANGLE *emitter = &scene->cpu_emissive_triangles[i];
        const uint32_t object_index = emitter->meta[2];

        if (object_index >= renderer->gpu_object_count) return false;

        if (emitter->meta[3] != renderer->cpu_objects[object_index].revision) {
            dirty = true;
            break;
        }
    }

    if (!dirty) return true;

    double total_weight = 0.0;

    for (uint32_t i = 0; i < scene->emissive_triangle_count; ++i) {
        GPU_EMISSIVE_TRIANGLE *emitter = &scene->cpu_emissive_triangles[i];
        const uint32_t triangle_id = emitter->meta[0];
        const uint32_t object_index = emitter->meta[2];

        if (triangle_id >= scene->triangle_count || object_index >= renderer->gpu_object_count) return false;

        const GPU_SCENE_TRIANGLE *triangle = &scene->cpu_triangles[triangle_id];
        const MAT4 world = renderer->cpu_objects[object_index].world;
        const VEC3 a = mat4_point(world, v3(triangle->p0[0], triangle->p0[1], triangle->p0[2]));
        const VEC3 b = mat4_point(world, v3(triangle->p1[0], triangle->p1[1], triangle->p1[2]));
        const VEC3 c = mat4_point(world, v3(triangle->p2[0], triangle->p2[1], triangle->p2[2]));
        const float area = triangle_area(a, b, c);
        const float luminance = emitter->radiance_area[0] * 0.2126f + emitter->radiance_area[1] * 0.7152f + emitter->radiance_area[2] * 0.0722f;
        const float weight = area * luminance;

        emitter->radiance_area[3] = area;
        emitter->sampling[2] = weight;
        emitter->meta[3] = renderer->cpu_objects[object_index].revision;
        total_weight += (double)weight;
    }

    if (total_weight <= 1.0e-12) return false;

    double cumulative = 0.0;

    for (uint32_t i = 0; i < scene->emissive_triangle_count; ++i) {
        GPU_EMISSIVE_TRIANGLE *emitter = &scene->cpu_emissive_triangles[i];
        const float probability = (float)((double)emitter->sampling[2] / total_weight);
        cumulative += probability;
        emitter->sampling[0] = i + 1u == scene->emissive_triangle_count ? 1.0f : (float)cumulative;
        emitter->sampling[1] = probability;
        emitter->sampling[3] = 0.0f;
    }

    const NriAccessStage read = {
        .access = NriAccessBits_SHADER_RESOURCE,
        .stages = NriStageBits_COMPUTE_SHADER
    };

    if (!gpu_upload_buffer(renderer->gpu, scene->emissive_triangles, scene->cpu_emissive_triangles, read)) return false;

    scene->emissive_triangles_state = read;
    return true;
}

static void destroy_screen_trace(RENDERER *renderer) {
    if (renderer->screen_trace.uav) renderer->gpu->core.DestroyDescriptor(renderer->screen_trace.uav);

    if (renderer->screen_trace.srv) renderer->gpu->core.DestroyDescriptor(renderer->screen_trace.srv);

    if (renderer->screen_trace.texture) gpu_destroy_texture(renderer->gpu, renderer->screen_trace.texture);
    memset(&renderer->screen_trace, 0, sizeof(renderer->screen_trace));
}

static bool create_screen_trace(RENDERER *renderer, uint32_t width, uint32_t height) {
    destroy_screen_trace(renderer);
    renderer->screen_trace.width = width;
    renderer->screen_trace.height = height;

    const NriTextureDesc desc = {
        .type = NriTextureType_TEXTURE_2D,
        .usage = NriTextureUsageBits_SHADER_RESOURCE | NriTextureUsageBits_SHADER_RESOURCE_STORAGE,
        .format = NriFormat_RGBA16_SFLOAT,
        .width = (NriDim_t)width,
        .height = (NriDim_t)height,
        .depth = 1,
        .mipNum = 1,
        .layerNum = 1,
        .sampleNum = 1
    };

    if (!gpu_create_texture(renderer->gpu, &desc, NriMemoryLocation_DEVICE, &renderer->screen_trace.texture)) return false;

    const NriTextureViewDesc srv = {
        .texture = renderer->screen_trace.texture,
        .type = NriTextureView_TEXTURE,
        .format = NriFormat_RGBA16_SFLOAT,
        .mipNum = 1,
        .layerNum = 1,
        .sliceNum = 1,
        .planes = NriPlaneBits_COLOR
    };

    const NriTextureViewDesc uav = {
        .texture = renderer->screen_trace.texture,
        .type = NriTextureView_STORAGE_TEXTURE,
        .format = NriFormat_RGBA16_SFLOAT,
        .mipNum = 1,
        .layerNum = 1,
        .sliceNum = 1,
        .planes = NriPlaneBits_COLOR
    };

    return renderer->gpu->core.CreateTextureView(&srv, &renderer->screen_trace.srv) == NriResult_SUCCESS &&
           renderer->gpu->core.CreateTextureView(&uav, &renderer->screen_trace.uav) == NriResult_SUCCESS;
}

static void destroy_hzb(RENDERER *renderer) {
    for (uint32_t mip = 0; mip < HZB_MAX_MIPS; ++mip) {
        if (renderer->hzb.mip_srvs[mip]) renderer->gpu->core.DestroyDescriptor(renderer->hzb.mip_srvs[mip]);

        if (renderer->hzb.mip_uavs[mip]) renderer->gpu->core.DestroyDescriptor(renderer->hzb.mip_uavs[mip]);
    }

    if (renderer->hzb.srv) renderer->gpu->core.DestroyDescriptor(renderer->hzb.srv);

    if (renderer->hzb.texture) gpu_destroy_texture(renderer->gpu, renderer->hzb.texture);
    memset(&renderer->hzb, 0, sizeof(renderer->hzb));
}

static uint32_t hzb_mip_count(uint32_t width, uint32_t height) {
    uint32_t size = width > height ? width : height;
    uint32_t count = 1;

    while (size > 1) {
        size >>= 1;
        ++count;
    }

    return count;
}

static bool create_hzb(RENDERER *renderer, uint32_t width, uint32_t height) {
    destroy_hzb(renderer);
    renderer->hzb.width = width > 1 ? width >> 1 : 1;
    renderer->hzb.height = height > 1 ? height >> 1 : 1;
    renderer->hzb.mip_count = hzb_mip_count(renderer->hzb.width, renderer->hzb.height);

    if (renderer->hzb.mip_count > HZB_MAX_MIPS) return false;

    const NriTextureDesc desc = {
        .type = NriTextureType_TEXTURE_2D,
        .usage = NriTextureUsageBits_SHADER_RESOURCE | NriTextureUsageBits_SHADER_RESOURCE_STORAGE,
        .format = NriFormat_R32_SFLOAT,
        .width = (NriDim_t)renderer->hzb.width,
        .height = (NriDim_t)renderer->hzb.height,
        .depth = 1,
        .mipNum = (NriDim_t)renderer->hzb.mip_count,
        .layerNum = 1,
        .sampleNum = 1
    };

    if (!gpu_create_texture(renderer->gpu, &desc, NriMemoryLocation_DEVICE, &renderer->hzb.texture)) return false;

    const NriTextureViewDesc srv = {
        .texture = renderer->hzb.texture,
        .type = NriTextureView_TEXTURE,
        .format = NriFormat_R32_SFLOAT,
        .mipNum = (NriDim_t)renderer->hzb.mip_count,
        .layerNum = 1,
        .sliceNum = 1,
        .planes = NriPlaneBits_COLOR
    };

    if (renderer->gpu->core.CreateTextureView(&srv, &renderer->hzb.srv) != NriResult_SUCCESS) return false;

    for (uint32_t mip = 0; mip < renderer->hzb.mip_count; ++mip) {
        const NriTextureViewDesc mip_srv = {
            .texture = renderer->hzb.texture,
            .type = NriTextureView_TEXTURE,
            .format = NriFormat_R32_SFLOAT,
            .mipOffset = (NriDim_t)mip,
            .mipNum = 1,
            .layerNum = 1,
            .sliceNum = 1,
            .planes = NriPlaneBits_COLOR
        };

        const NriTextureViewDesc mip_uav = {
            .texture = renderer->hzb.texture,
            .type = NriTextureView_STORAGE_TEXTURE,
            .format = NriFormat_R32_SFLOAT,
            .mipOffset = (NriDim_t)mip,
            .mipNum = 1,
            .layerNum = 1,
            .sliceNum = 1,
            .planes = NriPlaneBits_COLOR
        };

        if (renderer->gpu->core.CreateTextureView(&mip_srv, &renderer->hzb.mip_srvs[mip]) != NriResult_SUCCESS ||
            renderer->gpu->core.CreateTextureView(&mip_uav, &renderer->hzb.mip_uavs[mip]) != NriResult_SUCCESS)
            return false;
    }

    return true;
}

static void destroy_gbuffer(RENDERER *renderer) {
    destroy_render_texture(renderer, &renderer->depth);
    destroy_render_texture(renderer, &renderer->normal_roughness);
    destroy_render_texture(renderer, &renderer->albedo_metallic);
    destroy_render_texture(renderer, &renderer->emissive);
    destroy_render_texture(renderer, &renderer->velocity);
    destroy_render_texture(renderer, &renderer->object_id);
    destroy_render_texture(renderer, &renderer->material_id);
    destroy_render_texture(renderer, &renderer->primitive_id);
}

static bool create_gbuffer(RENDERER *renderer, uint32_t width, uint32_t height) {
    destroy_gbuffer(renderer);
    renderer->width = width;
    renderer->height = height;

    return create_render_texture(
               renderer,
               &renderer->depth,
               NriFormat_D32_SFLOAT,
               NriTextureUsageBits_DEPTH_STENCIL_ATTACHMENT | NriTextureUsageBits_SHADER_RESOURCE,
               NriTextureView_DEPTH_STENCIL_ATTACHMENT,
               NriPlaneBits_DEPTH
           ) &&
           create_render_texture(
               renderer,
               &renderer->normal_roughness,
               NriFormat_RGBA16_SFLOAT,
               NriTextureUsageBits_COLOR_ATTACHMENT | NriTextureUsageBits_SHADER_RESOURCE,
               NriTextureView_COLOR_ATTACHMENT,
               NriPlaneBits_COLOR
           ) &&
           create_render_texture(
               renderer,
               &renderer->albedo_metallic,
               NriFormat_RGBA8_UNORM,
               NriTextureUsageBits_COLOR_ATTACHMENT | NriTextureUsageBits_SHADER_RESOURCE,
               NriTextureView_COLOR_ATTACHMENT,
               NriPlaneBits_COLOR
           ) &&
           create_render_texture(
               renderer,
               &renderer->emissive,
               NriFormat_RGBA16_SFLOAT,
               NriTextureUsageBits_COLOR_ATTACHMENT | NriTextureUsageBits_SHADER_RESOURCE,
               NriTextureView_COLOR_ATTACHMENT,
               NriPlaneBits_COLOR
           ) &&
           create_render_texture(
               renderer,
               &renderer->velocity,
               NriFormat_RG16_SFLOAT,
               NriTextureUsageBits_COLOR_ATTACHMENT | NriTextureUsageBits_SHADER_RESOURCE,
               NriTextureView_COLOR_ATTACHMENT,
               NriPlaneBits_COLOR
           ) &&
           create_render_texture(
               renderer,
               &renderer->object_id,
               NriFormat_R32_UINT,
               NriTextureUsageBits_COLOR_ATTACHMENT | NriTextureUsageBits_SHADER_RESOURCE,
               NriTextureView_COLOR_ATTACHMENT,
               NriPlaneBits_COLOR
           ) &&
           create_render_texture(
               renderer,
               &renderer->material_id,
               NriFormat_R32_UINT,
               NriTextureUsageBits_COLOR_ATTACHMENT | NriTextureUsageBits_SHADER_RESOURCE,
               NriTextureView_COLOR_ATTACHMENT,
               NriPlaneBits_COLOR
           ) &&
           create_render_texture(
               renderer,
               &renderer->primitive_id,
               NriFormat_R32_UINT,
               NriTextureUsageBits_COLOR_ATTACHMENT | NriTextureUsageBits_SHADER_RESOURCE,
               NriTextureView_COLOR_ATTACHMENT,
               NriPlaneBits_COLOR
           );
}

static void destroy_sdf_scene(RENDERER *renderer) {
    if (renderer->sdf.models_srv) renderer->gpu->core.DestroyDescriptor(renderer->sdf.models_srv);
    if (renderer->sdf.voxels_srv) renderer->gpu->core.DestroyDescriptor(renderer->sdf.voxels_srv);
    if (renderer->sdf.surface_ids_srv) renderer->gpu->core.DestroyDescriptor(renderer->sdf.surface_ids_srv);

    if (renderer->sdf.models) gpu_destroy_buffer(renderer->gpu, renderer->sdf.models);
    if (renderer->sdf.voxels) gpu_destroy_buffer(renderer->gpu, renderer->sdf.voxels);
    if (renderer->sdf.surface_ids) gpu_destroy_buffer(renderer->gpu, renderer->sdf.surface_ids);

    free(renderer->sdf.cpu_models);
    memset(&renderer->sdf, 0, sizeof(renderer->sdf));
}

static bool create_sdf_scene(RENDERER *renderer, SCENE *scene) {
    destroy_sdf_scene(renderer);

    uint32_t model_count = 0;

    for (uint32_t i = 0; i < scene->object_count; ++i)
        if (scene->objects[i].type == MODEL) ++model_count;

    if (!model_count) return false;

    SDF_VOLUME *volumes = calloc(model_count, sizeof(*volumes));
    GPU_SDF_MODEL *models = calloc(model_count, sizeof(*models));
    uint32_t *triangle_bases = calloc(model_count, sizeof(*triangle_bases));

    if (!volumes || !models || !triangle_bases) {
        free(volumes);
        free(models);
        free(triangle_bases);
        return false;
    }

    uint64_t total_voxels = 0;
    uint64_t triangle_cursor = 0;
    uint32_t model_index = 0;
    bool ok = true;

    for (uint32_t scene_index = 0; scene_index < scene->object_count; ++scene_index) {
        OBJECT *object = &scene->objects[scene_index];

        if (object->type != MODEL) continue;

        struct MODEL *model = object->data;

        if (!model || !model->geometry || !model->geometry->faces.buffer || model->geometry->faces.count > UINT32_MAX ||
            !sdf_build_volume(model->geometry, SDF_DEFAULT_RESOLUTION, &volumes[model_index])) {
            ok = false;
            break;
        }

        triangle_bases[model_index] = (uint32_t)triangle_cursor;
        triangle_cursor += model->geometry->faces.count;

        if (triangle_cursor > UINT32_MAX || triangle_cursor > renderer->radiance_scene.triangle_count) {
            ok = false;
            break;
        }

        const uint64_t volume_voxels =
            (uint64_t)volumes[model_index].resolution * volumes[model_index].resolution * volumes[model_index].resolution;

        if (total_voxels + volume_voxels > UINT32_MAX) {
            ok = false;
            break;
        }

        GPU_SDF_MODEL *gpu_model = &models[model_index];
        MAT4 world_to_local;

        if (!mat4_inverse(mat4_transform(object->transform), &world_to_local)) {
            ok = false;
            break;
        }

        gpu_model->world_to_local = world_to_local;
        gpu_model->bounds_min[0] = volumes[model_index].bounds.min.x;
        gpu_model->bounds_min[1] = volumes[model_index].bounds.min.y;
        gpu_model->bounds_min[2] = volumes[model_index].bounds.min.z;
        gpu_model->bounds_min[3] = 0.0f;
        gpu_model->bounds_max[0] = volumes[model_index].bounds.max.x;
        gpu_model->bounds_max[1] = volumes[model_index].bounds.max.y;
        gpu_model->bounds_max[2] = volumes[model_index].bounds.max.z;
        gpu_model->bounds_max[3] = 0.0f;
        gpu_model->voxel_offset = (uint32_t)total_voxels;
        gpu_model->resolution = volumes[model_index].resolution;
        gpu_model->object_id = scene_index + 1u;
        gpu_model->state = (uint32_t)object->state;
        gpu_model->revision = object->revision;
        gpu_model->padding[0] = 0u;
        gpu_model->padding[1] = 0u;
        gpu_model->padding[2] = 0u;

        total_voxels += volume_voxels;
        ++model_index;
    }

    if (ok && triangle_cursor != renderer->radiance_scene.triangle_count) ok = false;

    float *voxel_data = NULL;
    uint32_t *surface_id_data = NULL;

    if (ok) {
        voxel_data = malloc((size_t)total_voxels * sizeof(*voxel_data));
        surface_id_data = malloc((size_t)total_voxels * sizeof(*surface_id_data));

        if (!voxel_data || !surface_id_data) ok = false;
    }

    if (ok) {
        for (uint32_t i = 0; i < model_count && ok; ++i) {
            const uint64_t count = (uint64_t)volumes[i].resolution * volumes[i].resolution * volumes[i].resolution;
            const uint32_t offset = models[i].voxel_offset;

            memcpy(voxel_data + offset, volumes[i].distance, (size_t)count * sizeof(float));

            for (uint64_t j = 0; j < count; ++j) {
                const uint32_t local_surface = volumes[i].surface_id[j];

                if (local_surface == UINT32_MAX) {
                    surface_id_data[offset + j] = UINT32_MAX;
                    continue;
                }

                const uint64_t global_surface = (uint64_t)triangle_bases[i] + local_surface;

                if (global_surface >= renderer->radiance_scene.triangle_count) {
                    ok = false;
                    break;
                }

                surface_id_data[offset + j] = (uint32_t)global_surface;
            }
        }
    }

    if (ok) {
        const NriBufferDesc models_desc = {
            .size = (uint64_t)model_count * sizeof(GPU_SDF_MODEL),
            .structureStride = sizeof(GPU_SDF_MODEL),
            .usage = NriBufferUsageBits_SHADER_RESOURCE
        };

        const NriBufferDesc voxels_desc = {
            .size = total_voxels * sizeof(float),
            .structureStride = sizeof(float),
            .usage = NriBufferUsageBits_SHADER_RESOURCE
        };

        const NriBufferDesc surface_ids_desc = {
            .size = total_voxels * sizeof(uint32_t),
            .structureStride = sizeof(uint32_t),
            .usage = NriBufferUsageBits_SHADER_RESOURCE
        };

        ok = gpu_create_buffer(renderer->gpu, &models_desc, NriMemoryLocation_DEVICE, &renderer->sdf.models) &&
             gpu_create_buffer(renderer->gpu, &voxels_desc, NriMemoryLocation_DEVICE, &renderer->sdf.voxels) &&
             gpu_create_buffer(renderer->gpu, &surface_ids_desc, NriMemoryLocation_DEVICE, &renderer->sdf.surface_ids) &&
             create_buffer_view(renderer, renderer->sdf.models, NriBufferView_STRUCTURED_BUFFER, models_desc.size, sizeof(GPU_SDF_MODEL), &renderer->sdf.models_srv) &&
             create_buffer_view(renderer, renderer->sdf.voxels, NriBufferView_STRUCTURED_BUFFER, voxels_desc.size, sizeof(float), &renderer->sdf.voxels_srv) &&
             create_buffer_view(renderer, renderer->sdf.surface_ids, NriBufferView_STRUCTURED_BUFFER, surface_ids_desc.size, sizeof(uint32_t), &renderer->sdf.surface_ids_srv);

        if (ok) {
            const NriAccessStage read = {
                .access = NriAccessBits_SHADER_RESOURCE,
                .stages = NriStageBits_COMPUTE_SHADER
            };

            ok = gpu_upload_buffer(renderer->gpu, renderer->sdf.voxels, voxel_data, read) &&
                 gpu_upload_buffer(renderer->gpu, renderer->sdf.surface_ids, surface_id_data, read);

            renderer->sdf.voxels_state = read;
            renderer->sdf.surface_ids_state = read;
        }
    }

    for (uint32_t i = 0; i < model_count; ++i)
        sdf_free_volume(&volumes[i]);

    free(volumes);
    free(voxel_data);
    free(surface_id_data);
    free(triangle_bases);

    if (!ok) {
        free(models);
        destroy_sdf_scene(renderer);
        return false;
    }

    renderer->sdf.cpu_models = models;
    renderer->sdf.model_count = model_count;
    renderer->sdf.voxel_count = (uint32_t)total_voxels;
    renderer->sdf.clipmap_count = 0u;
    renderer->sdf.clipmap_resolution = 0u;
    renderer->sdf.clipmap_base_extent = 0.0f;

    SDL_Log("SDF scene: %u models, %u voxels with surface IDs", model_count, renderer->sdf.voxel_count);
    return true;
}

static void destroy_scene_resources(RENDERER *renderer) {
    destroy_radiance_scene_data(renderer);
    destroy_sdf_scene(renderer);

    if (renderer->object_srv) renderer->gpu->core.DestroyDescriptor(renderer->object_srv);

    if (renderer->material_srv) renderer->gpu->core.DestroyDescriptor(renderer->material_srv);

    if (renderer->light_srv) renderer->gpu->core.DestroyDescriptor(renderer->light_srv);

    if (renderer->vertex_buffer) gpu_destroy_buffer(renderer->gpu, renderer->vertex_buffer);

    if (renderer->material_buffer) gpu_destroy_buffer(renderer->gpu, renderer->material_buffer);

    if (renderer->object_buffer) gpu_destroy_buffer(renderer->gpu, renderer->object_buffer);

    if (renderer->light_buffer) gpu_destroy_buffer(renderer->gpu, renderer->light_buffer);
    free(renderer->cpu_objects);
    free(renderer->cpu_lights);
    renderer->object_srv = NULL;
    renderer->material_srv = NULL;
    renderer->light_srv = NULL;
    renderer->vertex_buffer = NULL;
    renderer->material_buffer = NULL;
    renderer->object_buffer = NULL;
    renderer->light_buffer = NULL;
    renderer->cpu_objects = NULL;
    renderer->cpu_lights = NULL;
    renderer->gpu_object_count = 0;
    renderer->vertex_count = 0;
    renderer->material_count = 0;
    renderer->light_count = 0;
    renderer->object_state = (NriAccessStage){0};
    renderer->light_state = (NriAccessStage){0};
}

static void fill_light(GPU_LIGHT *dst, const OBJECT *object) {
    memset(dst, 0, sizeof(*dst));

    const struct LIGHT *light = object->data;

    if (!light) return;
    dst->position_range[0] = object->transform.position.x;
    dst->position_range[1] = object->transform.position.y;
    dst->position_range[2] = object->transform.position.z;
    dst->direction_type[3] = (float)light->type;

    if (light->type == LIGHT_DIRECTIONAL) {
        dst->direction_type[0] = light->directional.direction.x;
        dst->direction_type[1] = light->directional.direction.y;
        dst->direction_type[2] = light->directional.direction.z;
        dst->color_intensity[0] = light->directional.color.x;
        dst->color_intensity[1] = light->directional.color.y;
        dst->color_intensity[2] = light->directional.color.z;
        dst->color_intensity[3] = light->directional.intensity;
    } else if (light->type == LIGHT_POINT) {
        dst->position_range[3] = light->point.range;
        dst->color_intensity[0] = light->point.color.x;
        dst->color_intensity[1] = light->point.color.y;
        dst->color_intensity[2] = light->point.color.z;
        dst->color_intensity[3] = light->point.intensity;
    } else {
        dst->position_range[3] = light->spot.range;
        dst->direction_type[0] = light->spot.direction.x;
        dst->direction_type[1] = light->spot.direction.y;
        dst->direction_type[2] = light->spot.direction.z;
        dst->color_intensity[0] = light->spot.color.x;
        dst->color_intensity[1] = light->spot.color.y;
        dst->color_intensity[2] = light->spot.color.z;
        dst->color_intensity[3] = light->spot.intensity;
        dst->spot_angles[0] = cosf(light->spot.inner_angle);
        dst->spot_angles[1] = cosf(light->spot.outer_angle);
    }
}

static bool create_scene_resources(RENDERER *renderer, SCENE *scene) {
    uint64_t total_vertices = 0;
    uint64_t total_materials = 0;
    uint32_t model_count = 0;
    uint32_t scene_light_count = 0;

    for (uint32_t i = 0; i < scene->object_count; ++i) {
        OBJECT *object = &scene->objects[i];

        if (object->type == MODEL) {
            struct MODEL *model = object->data;

            if (!model || !model->visual || !model->visual->vertices || !model->visual->materials) return false;
            total_vertices += model->visual->vertex_count;
            total_materials += model->visual->material_count;
            ++model_count;
        } else if (object->type == LIGHT) {
            ++scene_light_count;
        }
    }

    if (!model_count || total_vertices > UINT32_MAX || total_materials > UINT32_MAX) return false;

    GLTF_VERTEX *vertices = malloc((size_t)total_vertices * sizeof(*vertices));
    GPU_MATERIAL *materials = calloc((size_t)total_materials, sizeof(*materials));
    GPU_OBJECT *objects = calloc(model_count, sizeof(*objects));
    GPU_LIGHT *lights = calloc(scene_light_count + 1u, sizeof(*lights));

    if (!vertices || !materials || !objects || !lights) {
        free(vertices);
        free(materials);
        free(objects);
        free(lights);

        return false;
    }

    uint32_t vertex_offset = 0, material_offset = 0, object_index = 0, light_index = 0;

    for (uint32_t scene_index = 0; scene_index < scene->object_count; ++scene_index) {
        OBJECT *object = &scene->objects[scene_index];

        if (object->type == MODEL) {
            struct MODEL *model = object->data;
            GLTF_SCENE *visual = model->visual;
            memcpy(&vertices[vertex_offset], visual->vertices, visual->vertex_count * sizeof(*vertices));

            for (uint32_t material_index = 0; material_index < visual->material_count; ++material_index) {
                const GLTF_MATERIAL *src = &visual->materials[material_index];
                GPU_MATERIAL *dst = &materials[material_offset + material_index];
                memcpy(dst->base_color, src->base_color, sizeof(dst->base_color));
                memcpy(dst->emissive, src->emissive, sizeof(dst->emissive));
                dst->metallic = src->metallic;
                dst->roughness = src->roughness;
            }

            GPU_OBJECT *gpu_object = &objects[object_index++];
            gpu_object->world = mat4_transform(object->transform);
            gpu_object->previous_world = gpu_object->world;
            gpu_object->normal_world = mat4_normal_transform(object->transform);
            gpu_object->first_vertex = vertex_offset;
            gpu_object->vertex_count = (uint32_t)visual->vertex_count;
            gpu_object->object_id = scene_index + 1u;
            gpu_object->material_offset = material_offset;
            gpu_object->revision = object->revision;
            gpu_object->state = (uint32_t)object->state;
            gpu_object->type = (uint32_t)object->type;
            vertex_offset += (uint32_t)visual->vertex_count;
            material_offset += visual->material_count;
        } else if (object->type == LIGHT) {
            fill_light(&lights[light_index++], object);
        }
    }

    GPU_LIGHT *sky = &lights[light_index++];
    sky->direction_type[3] = 3.0f;
    sky->color_intensity[0] = scene->sky.zenith.x;
    sky->color_intensity[1] = scene->sky.zenith.y;
    sky->color_intensity[2] = scene->sky.zenith.z;
    sky->color_intensity[3] = scene->sky.intensity;
    sky->spot_angles[0] = scene->sky.horizon.x;
    sky->spot_angles[1] = scene->sky.horizon.y;
    sky->spot_angles[2] = scene->sky.horizon.z;

    renderer->cpu_objects = objects;
    renderer->cpu_lights = lights;
    renderer->gpu_object_count = model_count;
    renderer->vertex_count = (uint32_t)total_vertices;
    renderer->material_count = (uint32_t)total_materials;
    renderer->light_count = light_index;

    if (!build_radiance_scene_data(renderer, scene)) {
        free(vertices);
        free(materials);

        destroy_scene_resources(renderer);

        return false;
    }

    const NriBufferDesc vertex_desc = {
        .size = (uint64_t)renderer->vertex_count * sizeof(GLTF_VERTEX),
        .usage = NriBufferUsageBits_VERTEX
    };

    const NriBufferDesc material_desc = {
        .size = (uint64_t)renderer->material_count * sizeof(GPU_MATERIAL),
        .structureStride = sizeof(GPU_MATERIAL),
        .usage = NriBufferUsageBits_SHADER_RESOURCE
    };

    const NriBufferDesc object_desc = {
        .size = (uint64_t)renderer->gpu_object_count * sizeof(GPU_OBJECT),
        .structureStride = sizeof(GPU_OBJECT),
        .usage = NriBufferUsageBits_SHADER_RESOURCE
    };

    const NriBufferDesc light_desc = {
        .size = (uint64_t)renderer->light_count * sizeof(GPU_LIGHT),
        .structureStride = sizeof(GPU_LIGHT),
        .usage = NriBufferUsageBits_SHADER_RESOURCE
    };

    bool ok = gpu_create_buffer(renderer->gpu, &vertex_desc, NriMemoryLocation_DEVICE, &renderer->vertex_buffer) &&
              gpu_create_buffer(renderer->gpu, &material_desc, NriMemoryLocation_DEVICE, &renderer->material_buffer) &&
              gpu_create_buffer(renderer->gpu, &object_desc, NriMemoryLocation_DEVICE, &renderer->object_buffer) &&
              gpu_create_buffer(renderer->gpu, &light_desc, NriMemoryLocation_DEVICE, &renderer->light_buffer);

    if (ok) {
        const NriAccessStage vertex_state = {
            .access = NriAccessBits_VERTEX_BUFFER,
            .stages = NriStageBits_VERTEX_SHADER
        };

        const NriAccessStage material_state = {
            .access = NriAccessBits_SHADER_RESOURCE,
            .stages = NriStageBits_FRAGMENT_SHADER | NriStageBits_COMPUTE_SHADER
        };

        ok = gpu_upload_buffer(renderer->gpu, renderer->vertex_buffer, vertices, vertex_state) &&
             gpu_upload_buffer(renderer->gpu, renderer->material_buffer, materials, material_state);
    }

    if (ok) {
        ok = create_buffer_view(renderer, renderer->object_buffer, NriBufferView_STRUCTURED_BUFFER, object_desc.size, sizeof(GPU_OBJECT), &renderer->object_srv) &&
             create_buffer_view(renderer, renderer->material_buffer, NriBufferView_STRUCTURED_BUFFER, material_desc.size, sizeof(GPU_MATERIAL), &renderer->material_srv) &&
             create_buffer_view(renderer, renderer->light_buffer, NriBufferView_STRUCTURED_BUFFER, light_desc.size, sizeof(GPU_LIGHT), &renderer->light_srv);
    }

    free(vertices);
    free(materials);

    if (!ok || !create_sdf_scene(renderer, scene) || !create_radiance_scene_gpu_resources(renderer) || !update_radiance_constants(renderer) ||
        !update_radiance_scene_descriptors(renderer)) {
        destroy_scene_resources(renderer);

        return false;
    }

    return true;
}

static void destroy_radiance_constant_buffers(RENDERER *renderer) {
    if (!renderer || !renderer->gpu) return;

    if (renderer->radiance_constants_srv) renderer->gpu->core.DestroyDescriptor(renderer->radiance_constants_srv);
    if (renderer->pass_constants_srv) renderer->gpu->core.DestroyDescriptor(renderer->pass_constants_srv);
    if (renderer->radiance_constants_buffer) gpu_destroy_buffer(renderer->gpu, renderer->radiance_constants_buffer);
    if (renderer->pass_constants_buffer) gpu_destroy_buffer(renderer->gpu, renderer->pass_constants_buffer);

    renderer->radiance_constants_srv = NULL;
    renderer->pass_constants_srv = NULL;
    renderer->radiance_constants_buffer = NULL;
    renderer->pass_constants_buffer = NULL;
    renderer->radiance_constants_state = (NriAccessStage){0};
    renderer->pass_constants_state = (NriAccessStage){0};

    memset(&renderer->radiance_constants, 0, sizeof(renderer->radiance_constants));
    memset(&renderer->pass_constants, 0, sizeof(renderer->pass_constants));
}

static bool create_radiance_constant_buffers(RENDERER *renderer) {
    destroy_radiance_constant_buffers(renderer);

    const NriBufferDesc radiance_desc = {
        .size = sizeof(RADIANCE_CONSTANTS),
        .usage = NriBufferUsageBits_CONSTANT
    };

    const NriBufferDesc pass_desc = {
        .size = sizeof(PASS_CONSTANTS),
        .usage = NriBufferUsageBits_CONSTANT
    };

    if (!gpu_create_buffer(renderer->gpu, &radiance_desc, NriMemoryLocation_DEVICE, &renderer->radiance_constants_buffer) ||
        !gpu_create_buffer(renderer->gpu, &pass_desc, NriMemoryLocation_DEVICE, &renderer->pass_constants_buffer)) {
        destroy_radiance_constant_buffers(renderer);
        return false;
    }

    if (!create_buffer_view(renderer, renderer->radiance_constants_buffer, NriBufferView_CONSTANT_BUFFER, sizeof(RADIANCE_CONSTANTS), 0, &renderer->radiance_constants_srv) ||
        !create_buffer_view(renderer, renderer->pass_constants_buffer, NriBufferView_CONSTANT_BUFFER, sizeof(PASS_CONSTANTS), 0, &renderer->pass_constants_srv)) {
        destroy_radiance_constant_buffers(renderer);
        return false;
    }

    memset(&renderer->radiance_constants, 0, sizeof(renderer->radiance_constants));
    memset(&renderer->pass_constants, 0, sizeof(renderer->pass_constants));

    const NriAccessStage read = {
        .access = NriAccessBits_CONSTANT_BUFFER,
        .stages = NriStageBits_COMPUTE_SHADER | NriStageBits_FRAGMENT_SHADER
    };

    if (!gpu_upload_buffer(renderer->gpu, renderer->radiance_constants_buffer, &renderer->radiance_constants, read) ||
        !gpu_upload_buffer(renderer->gpu, renderer->pass_constants_buffer, &renderer->pass_constants, read)) {
        destroy_radiance_constant_buffers(renderer);
        return false;
    }

    renderer->radiance_constants_state = read;
    renderer->pass_constants_state = read;

    return true;
}

static bool update_radiance_constants(RENDERER *renderer) {
    if (!renderer || !renderer->radiance_constants_buffer) return false;

    RADIANCE_CONSTANTS constants = {0};
    constants.scene_counts[0] = renderer->gpu_object_count;
    constants.scene_counts[1] = renderer->material_count;
    constants.scene_counts[2] = renderer->radiance_scene.triangle_count;
    constants.scene_counts[3] = renderer->radiance_scene.emissive_triangle_count;
    constants.sdf_counts[0] = renderer->sdf.model_count;
    constants.sdf_counts[1] = 0u;
    constants.sdf_counts[2] = 0u;
    constants.sdf_counts[3] = renderer->light_count;
    constants.cache_counts[0] = renderer->radiance_surface_cache.capacity;
    constants.cache_counts[1] = 0u;
    constants.cache_counts[2] = 0u;
    constants.cache_counts[3] = 0u;
    constants.probe_config[0] = SCREEN_PROBE_TILE_SIZE;
    constants.probe_config[1] = SCREEN_PROBE_DIRECTION_SIZE;
    constants.probe_config[2] = SCREEN_PROBE_DIRECTION_COUNT;
    constants.probe_config[3] = 0u;
    constants.trace_params[0] = 200.0f;
    constants.trace_params[1] = 0.10f;
    constants.trace_params[2] = 0.05f;
    constants.trace_params[3] = 0.05f;
    constants.trace_limits[0] = 128u;
    constants.trace_limits[1] = 5u;
    constants.trace_limits[2] = 96u;
    constants.trace_limits[3] = 128u;
    constants.feature_flags[0] = RADIANCE_FEATURE_SURFACE_CACHE;
    if (renderer->radiance_scene.emissive_triangle_count) constants.feature_flags[0] |= RADIANCE_FEATURE_EMISSIVE;
    constants.feature_flags[1] = 1u;
    constants.feature_flags[2] = RADIANCE_DEBUG_FINAL_GI;
    constants.feature_flags[3] = 0u;
    constants.global_sdf_params[2] = 1.0e-4f;

    renderer->radiance_constants = constants;

    const NriAccessStage read = {
        .access = NriAccessBits_CONSTANT_BUFFER,
        .stages = NriStageBits_COMPUTE_SHADER | NriStageBits_FRAGMENT_SHADER
    };

    if (!gpu_upload_buffer(renderer->gpu, renderer->radiance_constants_buffer, &renderer->radiance_constants, read)) return false;

    renderer->radiance_constants_state = read;
    return true;
}

static bool create_frame_buffer(RENDERER *renderer) {
    const NriBufferDesc desc = {
        .size = FRAME_CONSTANTS_BUFFER_SIZE,
        .usage = NriBufferUsageBits_CONSTANT
    };

    if (!gpu_create_buffer(renderer->gpu, &desc, NriMemoryLocation_DEVICE, &renderer->frame_buffer)) return false;

    return create_buffer_view(renderer, renderer->frame_buffer, NriBufferView_CONSTANT_BUFFER, FRAME_CONSTANTS_BUFFER_SIZE, 0, &renderer->frame_srv);
}

static bool create_pipeline_layouts(RENDERER *renderer) {
    const NriDescriptorRangeDesc gbuffer_ranges[] = {
        {
            .baseRegisterIndex = 0,
            .descriptorNum = 1,
            .descriptorType = NriDescriptorType_STRUCTURED_BUFFER,
            .shaderStages = NriStageBits_VERTEX_SHADER
        },
        {
            .baseRegisterIndex = 1,
            .descriptorNum = 1,
            .descriptorType = NriDescriptorType_STRUCTURED_BUFFER,
            .shaderStages = NriStageBits_FRAGMENT_SHADER
        },
        {
            .baseRegisterIndex = 2,
            .descriptorNum = 1,
            .descriptorType = NriDescriptorType_CONSTANT_BUFFER,
            .shaderStages = NriStageBits_VERTEX_SHADER
        }
    };

    const NriDescriptorSetDesc gbuffer_set = {
        .registerSpace = 0,
        .ranges = gbuffer_ranges,
        .rangeNum = 3
    };

    const NriRootConstantDesc draw_constants = {
        .registerIndex = 0,
        .size = sizeof(uint32_t),
        .shaderStages = NriStageBits_VERTEX_SHADER
    };

    const NriPipelineLayoutDesc gbuffer_layout = {
        .rootRegisterSpace = 1,
        .rootConstants = &draw_constants,
        .rootConstantNum = 1,
        .descriptorSets = &gbuffer_set,
        .descriptorSetNum = 1,
        .shaderStages = NriStageBits_VERTEX_SHADER | NriStageBits_FRAGMENT_SHADER,
        .flags = NriPipelineLayoutBits_IGNORE_GLOBAL_SPIRV_OFFSETS
    };

    if (renderer->gpu->core.CreatePipelineLayout(renderer->gpu->device, &gbuffer_layout, &renderer->gbuffer_layout) != NriResult_SUCCESS) return false;

    const NriDescriptorRangeDesc present_range = {
        .baseRegisterIndex = 0,
        .descriptorNum = 9,
        .descriptorType = NriDescriptorType_TEXTURE,
        .shaderStages = NriStageBits_FRAGMENT_SHADER
    };

    const NriDescriptorSetDesc present_set = {
        .registerSpace = 1,
        .ranges = &present_range,
        .rangeNum = 1
    };

    const NriPipelineLayoutDesc present_layout = {
        .descriptorSets = &present_set,
        .descriptorSetNum = 1,
        .shaderStages = NriStageBits_VERTEX_SHADER | NriStageBits_FRAGMENT_SHADER,
        .flags = NriPipelineLayoutBits_IGNORE_GLOBAL_SPIRV_OFFSETS
    };

    if (renderer->gpu->core.CreatePipelineLayout(renderer->gpu->device, &present_layout, &renderer->present_layout) != NriResult_SUCCESS) return false;

    const NriDescriptorRangeDesc hzb_ranges[] = {
        {
            .baseRegisterIndex = 0,
            .descriptorNum = 1,
            .descriptorType = NriDescriptorType_TEXTURE,
            .shaderStages = NriStageBits_COMPUTE_SHADER
        },
        {
            .baseRegisterIndex = 1,
            .descriptorNum = 1,
            .descriptorType = NriDescriptorType_STORAGE_TEXTURE,
            .shaderStages = NriStageBits_COMPUTE_SHADER
        }
    };

    const NriDescriptorSetDesc hzb_set = {
        .registerSpace = 2,
        .ranges = hzb_ranges,
        .rangeNum = 2
    };

    const NriPipelineLayoutDesc hzb_layout = {
        .descriptorSets = &hzb_set,
        .descriptorSetNum = 1,
        .shaderStages = NriStageBits_COMPUTE_SHADER,
        .flags = NriPipelineLayoutBits_IGNORE_GLOBAL_SPIRV_OFFSETS
    };

    if (renderer->gpu->core.CreatePipelineLayout(renderer->gpu->device, &hzb_layout, &renderer->hzb_layout) != NriResult_SUCCESS) return false;

    const NriDescriptorRangeDesc trace_ranges[] = {
        {
            .baseRegisterIndex = 0,
            .descriptorNum = 7,
            .descriptorType = NriDescriptorType_TEXTURE,
            .shaderStages = NriStageBits_COMPUTE_SHADER
        },
        {
            .baseRegisterIndex = 7,
            .descriptorNum = 1,
            .descriptorType = NriDescriptorType_CONSTANT_BUFFER,
            .shaderStages = NriStageBits_COMPUTE_SHADER
        },
        {
            .baseRegisterIndex = 8,
            .descriptorNum = 1,
            .descriptorType = NriDescriptorType_STORAGE_TEXTURE,
            .shaderStages = NriStageBits_COMPUTE_SHADER
        },
        {
            .baseRegisterIndex = 9,
            .descriptorNum = 4,
            .descriptorType = NriDescriptorType_STORAGE_STRUCTURED_BUFFER,
            .shaderStages = NriStageBits_COMPUTE_SHADER
        },
        {
            .baseRegisterIndex = 13,
            .descriptorNum = 3,
            .descriptorType = NriDescriptorType_STRUCTURED_BUFFER,
            .shaderStages = NriStageBits_COMPUTE_SHADER
        },
        {
            .baseRegisterIndex = 16,
            .descriptorNum = 1,
            .descriptorType = NriDescriptorType_STORAGE_TEXTURE,
            .shaderStages = NriStageBits_COMPUTE_SHADER
        },
        {
            .baseRegisterIndex = 17,
            .descriptorNum = 2,
            .descriptorType = NriDescriptorType_STORAGE_STRUCTURED_BUFFER,
            .shaderStages = NriStageBits_COMPUTE_SHADER
        },
        {
            .baseRegisterIndex = 19,
            .descriptorNum = 2,
            .descriptorType = NriDescriptorType_STORAGE_TEXTURE,
            .shaderStages = NriStageBits_COMPUTE_SHADER
        }
    };

    const NriDescriptorSetDesc trace_set = {
        .registerSpace = 3,
        .ranges = trace_ranges,
        .rangeNum = 8
    };

    const NriPipelineLayoutDesc trace_layout = {
        .descriptorSets = &trace_set,
        .descriptorSetNum = 1,
        .shaderStages = NriStageBits_COMPUTE_SHADER,
        .flags = NriPipelineLayoutBits_IGNORE_GLOBAL_SPIRV_OFFSETS
    };

    if (renderer->gpu->core.CreatePipelineLayout(renderer->gpu->device, &trace_layout, &renderer->trace_layout) != NriResult_SUCCESS) return false;

    const NriStageBits radiance_stages = NriStageBits_COMPUTE_SHADER | NriStageBits_FRAGMENT_SHADER;

    const NriDescriptorRangeDesc radiance_scene_ranges[] = {
        {.baseRegisterIndex = 0, .descriptorNum = 2, .descriptorType = NriDescriptorType_CONSTANT_BUFFER, .shaderStages = radiance_stages},
        {.baseRegisterIndex = 2, .descriptorNum = 6, .descriptorType = NriDescriptorType_STRUCTURED_BUFFER, .shaderStages = radiance_stages},
        {.baseRegisterIndex = 8, .descriptorNum = 7, .descriptorType = NriDescriptorType_STRUCTURED_BUFFER, .shaderStages = radiance_stages},
        {.baseRegisterIndex = 15, .descriptorNum = 1, .descriptorType = NriDescriptorType_STRUCTURED_BUFFER, .shaderStages = radiance_stages},
        {.baseRegisterIndex = 16, .descriptorNum = 2, .descriptorType = NriDescriptorType_TEXTURE, .shaderStages = radiance_stages}
    };

    const NriDescriptorSetDesc radiance_scene_set = {
        .registerSpace = 4,
        .ranges = radiance_scene_ranges,
        .rangeNum = sizeof(radiance_scene_ranges) / sizeof(radiance_scene_ranges[0])
    };

    const NriPipelineLayoutDesc radiance_scene_layout = {
        .descriptorSets = &radiance_scene_set,
        .descriptorSetNum = 1,
        .shaderStages = radiance_stages,
        .flags = NriPipelineLayoutBits_IGNORE_GLOBAL_SPIRV_OFFSETS
    };

    if (renderer->gpu->core.CreatePipelineLayout(renderer->gpu->device, &radiance_scene_layout, &renderer->radiance_scene_layout) != NriResult_SUCCESS) return false;

    const NriDescriptorRangeDesc radiance_direct_cache_range = {
        .baseRegisterIndex = 0,
        .descriptorNum = 2,
        .descriptorType = NriDescriptorType_STORAGE_STRUCTURED_BUFFER,
        .shaderStages = NriStageBits_COMPUTE_SHADER
    };

    const NriDescriptorSetDesc radiance_direct_cache_set = {
        .registerSpace = 6,
        .ranges = &radiance_direct_cache_range,
        .rangeNum = 1
    };

    const NriDescriptorSetDesc radiance_direct_sets[] = {trace_set, radiance_scene_set, radiance_direct_cache_set};

    const NriPipelineLayoutDesc radiance_direct_layout = {
        .descriptorSets = radiance_direct_sets,
        .descriptorSetNum = 3,
        .shaderStages = NriStageBits_COMPUTE_SHADER,
        .flags = NriPipelineLayoutBits_IGNORE_GLOBAL_SPIRV_OFFSETS
    };

    if (renderer->gpu->core.CreatePipelineLayout(renderer->gpu->device, &radiance_direct_layout, &renderer->radiance_direct_layout) != NriResult_SUCCESS) return false;

    const NriDescriptorRangeDesc emissive_probe_range = {
        .baseRegisterIndex = 7,
        .descriptorNum = 1,
        .descriptorType = NriDescriptorType_STORAGE_TEXTURE,
        .shaderStages = NriStageBits_COMPUTE_SHADER
    };

    const NriDescriptorSetDesc emissive_probe_set = {
        .registerSpace = 7,
        .ranges = &emissive_probe_range,
        .rangeNum = 1
    };

    const NriDescriptorSetDesc emissive_sets[] = {trace_set, radiance_scene_set, emissive_probe_set};

    const NriPipelineLayoutDesc emissive_layout = {
        .descriptorSets = emissive_sets,
        .descriptorSetNum = 3,
        .shaderStages = NriStageBits_COMPUTE_SHADER,
        .flags = NriPipelineLayoutBits_IGNORE_GLOBAL_SPIRV_OFFSETS
    };

    return renderer->gpu->core.CreatePipelineLayout(renderer->gpu->device, &emissive_layout, &renderer->emissive_layout) == NriResult_SUCCESS;
}

static bool create_descriptor_pool(RENDERER *renderer) {
    const NriDescriptorPoolDesc desc = {
        .descriptorSetMaxNum = 10 + HZB_MAX_MIPS,
        .constantBufferMaxNum = 12,
        .textureMaxNum = 80,
        .storageTextureMaxNum = HZB_MAX_MIPS + 16,
        .structuredBufferMaxNum = 64,
        .storageStructuredBufferMaxNum = 24
    };

    if (renderer->gpu->core.CreateDescriptorPool(renderer->gpu->device, &desc, &renderer->descriptor_pool) != NriResult_SUCCESS) return false;

    return renderer->gpu->core.AllocateDescriptorSets(renderer->descriptor_pool, renderer->gbuffer_layout, 0, &renderer->gbuffer_set, 1, 0) == NriResult_SUCCESS &&
           renderer->gpu->core.AllocateDescriptorSets(renderer->descriptor_pool, renderer->present_layout, 0, &renderer->present_set, 1, 0) == NriResult_SUCCESS &&
           renderer->gpu->core.AllocateDescriptorSets(renderer->descriptor_pool, renderer->hzb_layout, 0, renderer->hzb_sets, HZB_MAX_MIPS, 0) == NriResult_SUCCESS &&
           renderer->gpu->core.AllocateDescriptorSets(renderer->descriptor_pool, renderer->trace_layout, 0, &renderer->trace_set, 1, 0) == NriResult_SUCCESS &&
           renderer->gpu->core.AllocateDescriptorSets(renderer->descriptor_pool, renderer->radiance_scene_layout, 0, &renderer->radiance_scene_set, 1, 0) == NriResult_SUCCESS &&
           renderer->gpu->core.AllocateDescriptorSets(renderer->descriptor_pool, renderer->radiance_direct_layout, 0, &renderer->radiance_direct_trace_set, 1, 0) == NriResult_SUCCESS &&
           renderer->gpu->core.AllocateDescriptorSets(renderer->descriptor_pool, renderer->radiance_direct_layout, 1, &renderer->radiance_direct_scene_set, 1, 0) == NriResult_SUCCESS &&
           renderer->gpu->core.AllocateDescriptorSets(renderer->descriptor_pool, renderer->radiance_direct_layout, 2, &renderer->radiance_direct_cache_set, 1, 0) == NriResult_SUCCESS &&
           renderer->gpu->core.AllocateDescriptorSets(renderer->descriptor_pool, renderer->emissive_layout, 0, &renderer->emissive_trace_set, 1, 0) == NriResult_SUCCESS &&
           renderer->gpu->core.AllocateDescriptorSets(renderer->descriptor_pool, renderer->emissive_layout, 1, &renderer->emissive_scene_set, 1, 0) == NriResult_SUCCESS &&
           renderer->gpu->core.AllocateDescriptorSets(renderer->descriptor_pool, renderer->emissive_layout, 2, &renderer->emissive_probe_set, 1, 0) == NriResult_SUCCESS;
}

static bool create_compute_pipeline(RENDERER *renderer, const char *path, NriPipelineLayout *layout, NriPipeline **pipeline) {
    void *code = NULL;
    size_t size = 0;

    if (!load_shader(path, &code, &size)) return false;

    const NriComputePipelineDesc desc = {
        .pipelineLayout = layout,
        .shader = {
            .stage = NriStageBits_COMPUTE_SHADER,
            .bytecode = code,
            .size = size,
            .entryPointName = "main"
        }
    };

    const bool ok = renderer->gpu->core.CreateComputePipeline(renderer->gpu->device, &desc, pipeline) == NriResult_SUCCESS;

    free(code);

    return ok;
}

static bool create_present_pipeline(RENDERER *renderer, const char *fragment_path, NriPipeline **pipeline) {
    void *vs = NULL, *ps = NULL;
    size_t vs_size = 0, ps_size = 0;

    if (!load_shader("build/shaders/present.vs.spv", &vs, &vs_size) || !load_shader(fragment_path, &ps, &ps_size)) {
        free(vs);
        free(ps);

        return false;
    }

    const NriShaderDesc shaders[] = {
        {
            .stage = NriStageBits_VERTEX_SHADER,
            .bytecode = vs,
            .size = vs_size,
            .entryPointName = "main"
        },
        {
            .stage = NriStageBits_FRAGMENT_SHADER,
            .bytecode = ps,
            .size = ps_size,
            .entryPointName = "main"
        }
    };

    const NriColorAttachmentDesc color = {
        .format = renderer->gpu->swapchain_format,
        .colorWriteMask = NriColorWriteBits_RGBA
    };

    const NriGraphicsPipelineDesc desc = {
        .pipelineLayout = renderer->present_layout,
        .inputAssembly = {
            .topology = NriTopology_TRIANGLE_LIST,
            .primitiveRestart = NriPrimitiveRestart_DISABLED
        },
        .rasterization = {
            .fillMode = NriFillMode_SOLID,
            .cullMode = NriCullMode_NONE
        },
        .outputMerger = {
            .colors = &color,
            .colorNum = 1
        },
        .shaders = shaders,
        .shaderNum = 2
    };

    const bool ok = renderer->gpu->core.CreateGraphicsPipeline(renderer->gpu->device, &desc, pipeline) == NriResult_SUCCESS;

    free(vs);
    free(ps);

    return ok;
}

static bool create_pipelines(RENDERER *renderer) {
    void *vs = NULL, *ps = NULL;
    size_t vs_size = 0, ps_size = 0;

    if (!load_shader("build/shaders/gbuffer.vs.spv", &vs, &vs_size) || !load_shader("build/shaders/gbuffer_full.ps.spv", &ps, &ps_size)) {
        free(vs);
        free(ps);

        return false;
    }

    const NriShaderDesc shaders[] = {
        {
            .stage = NriStageBits_VERTEX_SHADER,
            .bytecode = vs,
            .size = vs_size,
            .entryPointName = "main"
        },
        {
            .stage = NriStageBits_FRAGMENT_SHADER,
            .bytecode = ps,
            .size = ps_size,
            .entryPointName = "main"
        }
    };

    const NriVertexAttributeDesc attributes[] = {
        {
            .d3d = {
                .semanticName = "POSITION",
                .semanticIndex = 0
            },
            .vk = {
                .location = 0
            },
            .offset = offsetof(GLTF_VERTEX, position),
            .format = NriFormat_RGB32_SFLOAT,
            .streamIndex = 0
        },
        {
            .d3d = {
                .semanticName = "NORMAL",
                .semanticIndex = 0
            },
            .vk = {
                .location = 1
            },
            .offset = offsetof(GLTF_VERTEX, normal),
            .format = NriFormat_RGB32_SFLOAT,
            .streamIndex = 0
        },
        {
            .d3d = {
                .semanticName = "TEXCOORD",
                .semanticIndex = 0
            },
            .vk = {
                .location = 2
            },
            .offset = offsetof(GLTF_VERTEX, u),
            .format = NriFormat_RG32_SFLOAT,
            .streamIndex = 0
        },
        {
            .d3d = {
                .semanticName = "TEXCOORD",
                .semanticIndex = 1
            },
            .vk = {
                .location = 3
            },
            .offset = offsetof(GLTF_VERTEX, material),
            .format = NriFormat_R32_UINT,
            .streamIndex = 0
        }
    };

    const NriVertexStreamDesc stream = {
        .bindingSlot = 0,
        .stepRate = NriVertexStreamStepRate_PER_VERTEX,
        .stride = sizeof(GLTF_VERTEX)
    };

    const NriVertexInputDesc vertex_input = {
        .attributes = attributes,
        .attributeNum = 4,
        .streams = &stream,
        .streamNum = 1
    };

    const NriColorAttachmentDesc colors[] = {
        {
            .format = NriFormat_RGBA16_SFLOAT,
            .colorWriteMask = NriColorWriteBits_RGBA
        },
        {
            .format = NriFormat_RGBA8_UNORM,
            .colorWriteMask = NriColorWriteBits_RGBA
        },
        {
            .format = NriFormat_RG16_SFLOAT,
            .colorWriteMask = NriColorWriteBits_RGBA
        },
        {
            .format = NriFormat_R32_UINT,
            .colorWriteMask = NriColorWriteBits_RGBA
        },
        {
            .format = NriFormat_RGBA16_SFLOAT,
            .colorWriteMask = NriColorWriteBits_RGBA
        },
        {
            .format = NriFormat_R32_UINT,
            .colorWriteMask = NriColorWriteBits_RGBA
        },
        {
            .format = NriFormat_R32_UINT,
            .colorWriteMask = NriColorWriteBits_RGBA
        }
    };

    const NriGraphicsPipelineDesc gbuffer = {
        .pipelineLayout = renderer->gbuffer_layout,
        .vertexInput = &vertex_input,
        .inputAssembly = {
            .topology = NriTopology_TRIANGLE_LIST,
            .primitiveRestart = NriPrimitiveRestart_DISABLED
        },
        .rasterization = {
            .fillMode = NriFillMode_SOLID,
            .cullMode = NriCullMode_NONE
        },
        .outputMerger = {
            .colors = colors,
            .colorNum = 7,
            .depth = {
                .compareOp = NriCompareOp_GREATER,
                .write = true
            },
            .depthStencilFormat = NriFormat_D32_SFLOAT
        },
        .shaders = shaders,
        .shaderNum = 2
    };

    bool ok = renderer->gpu->core.CreateGraphicsPipeline(renderer->gpu->device, &gbuffer, &renderer->gbuffer_pipeline) == NriResult_SUCCESS;

    free(vs);
    free(ps);

    if (!ok || !create_present_pipeline(renderer, "build/shaders/present.ps.spv", &renderer->present_pipeline)) return false;

    return create_compute_pipeline(renderer, "build/shaders/hzb.cs.spv", renderer->hzb_layout, &renderer->hzb_pipeline) &&
           create_compute_pipeline(renderer, "build/shaders/radiance_direct.cs.spv", renderer->radiance_direct_layout, &renderer->direct_radiance_pipeline) &&
           create_compute_pipeline(renderer, "build/shaders/surface_cache.cs.spv", renderer->trace_layout, &renderer->surface_cache_pipeline) &&
           create_compute_pipeline(renderer, "build/shaders/screen_trace.cs.spv", renderer->trace_layout, &renderer->screen_trace_pipeline) &&
           create_compute_pipeline(renderer, "build/shaders/trace_reset.cs.spv", renderer->trace_layout, &renderer->trace_reset_pipeline) &&
           create_compute_pipeline(renderer, "build/shaders/trace_compact.cs.spv", renderer->trace_layout, &renderer->trace_compact_pipeline) &&
           create_compute_pipeline(renderer, "build/shaders/trace_args.cs.spv", renderer->trace_layout, &renderer->trace_args_pipeline) &&
           create_compute_pipeline(renderer, "build/shaders/sdf_trace.cs.spv", renderer->radiance_direct_layout, &renderer->sdf_trace_pipeline) &&
           create_compute_pipeline(renderer, "build/shaders/screen_probes.cs.spv", renderer->trace_layout, &renderer->screen_probes_pipeline) &&
           create_compute_pipeline(renderer, "build/shaders/radiance_emissive.cs.spv", renderer->emissive_layout, &renderer->emissive_pipeline);
}

static void update_gbuffer_descriptors(RENDERER *renderer) {
    if (!renderer->gbuffer_set || !renderer->object_srv || !renderer->material_srv || !renderer->frame_srv) return;

    const NriDescriptor *a[] = {renderer->object_srv};
    const NriDescriptor *b[] = {renderer->material_srv};
    const NriDescriptor *c[] = {renderer->frame_srv};

    const NriUpdateDescriptorRangeDesc updates[] = {
        {
            .descriptorSet = renderer->gbuffer_set,
            .rangeIndex = 0,
            .descriptors = a,
            .descriptorNum = 1
        },
        {
            .descriptorSet = renderer->gbuffer_set,
            .rangeIndex = 1,
            .descriptors = b,
            .descriptorNum = 1
        },
        {
            .descriptorSet = renderer->gbuffer_set,
            .rangeIndex = 2,
            .descriptors = c,
            .descriptorNum = 1
        }
    };

    renderer->gpu->core.UpdateDescriptorRanges(updates, 3);
}

static void update_hzb_descriptors(RENDERER *renderer) {
    NriUpdateDescriptorRangeDesc updates[HZB_MAX_MIPS * 2];
    const NriDescriptor *sources[HZB_MAX_MIPS];
    const NriDescriptor *destinations[HZB_MAX_MIPS];
    uint32_t count = 0;

    for (uint32_t mip = 0; mip < renderer->hzb.mip_count; ++mip) {
        sources[mip] = mip == 0 ? renderer->depth.srv : renderer->hzb.mip_srvs[mip - 1];
        destinations[mip] = renderer->hzb.mip_uavs[mip];

        updates[count++] = (NriUpdateDescriptorRangeDesc){
            .descriptorSet = renderer->hzb_sets[mip],
            .rangeIndex = 0,
            .descriptors = &sources[mip],
            .descriptorNum = 1
        };

        updates[count++] = (NriUpdateDescriptorRangeDesc){
            .descriptorSet = renderer->hzb_sets[mip],
            .rangeIndex = 1,
            .descriptors = &destinations[mip],
            .descriptorNum = 1
        };
    }

    renderer->gpu->core.UpdateDescriptorRanges(updates, count);
}

static void update_present_descriptors(RENDERER *renderer) {
    if (!renderer->present_set || !renderer->depth.srv || !renderer->normal_roughness.srv || !renderer->albedo_metallic.srv || !renderer->velocity.srv ||
        !renderer->object_id.srv || !renderer->hzb.srv || !renderer->screen_trace.srv || !renderer->direct_radiance.srv || !renderer->screen_probes.srv)
        return;

    const NriDescriptor *descriptors[] = {
        renderer->depth.srv,
        renderer->normal_roughness.srv,
        renderer->albedo_metallic.srv,
        renderer->velocity.srv,
        renderer->object_id.srv,
        renderer->hzb.srv,
        renderer->screen_trace.srv,
        renderer->direct_radiance.srv,
        renderer->screen_probes.srv
    };

    const NriUpdateDescriptorRangeDesc update = {
        .descriptorSet = renderer->present_set,
        .rangeIndex = 0,
        .descriptors = descriptors,
        .descriptorNum = 9
    };

    renderer->gpu->core.UpdateDescriptorRanges(&update, 1);
}

static bool update_trace_descriptor_set(RENDERER *renderer, NriDescriptorSet *descriptor_set) {
    if (!descriptor_set || !renderer->depth.srv || !renderer->normal_roughness.srv || !renderer->albedo_metallic.srv || !renderer->emissive.srv || !renderer->hzb.srv ||
        !renderer->object_id.srv || !renderer->direct_radiance.srv || !renderer->frame_srv || !renderer->screen_trace.uav || !renderer->trace_hits.uav ||
        !renderer->miss_queue.rays_uav || !renderer->miss_queue.count_uav || !renderer->miss_queue.dispatch_args_uav || !renderer->sdf.models_srv || !renderer->sdf.voxels_srv ||
        !renderer->light_srv || !renderer->direct_radiance.uav || !renderer->surface_cache.keys_uav || !renderer->surface_cache.entries_uav || !renderer->screen_probes.uav ||
        !renderer->screen_probe_radiance.uav) {
        return false;
    }

    const NriDescriptor *textures[] = {
        renderer->depth.srv,
        renderer->normal_roughness.srv,
        renderer->albedo_metallic.srv,
        renderer->emissive.srv,
        renderer->hzb.srv,
        renderer->object_id.srv,
        renderer->direct_radiance.srv
    };

    const NriDescriptor *frame[] = {renderer->frame_srv};
    const NriDescriptor *screen_output[] = {renderer->screen_trace.uav};
    const NriDescriptor *trace_storage[] = {renderer->trace_hits.uav, renderer->miss_queue.rays_uav, renderer->miss_queue.count_uav, renderer->miss_queue.dispatch_args_uav};
    const NriDescriptor *scene[] = {renderer->sdf.models_srv, renderer->sdf.voxels_srv, renderer->light_srv};
    const NriDescriptor *radiance_output[] = {renderer->direct_radiance.uav};
    const NriDescriptor *cache[] = {renderer->surface_cache.keys_uav, renderer->surface_cache.entries_uav};
    const NriDescriptor *probes[] = {renderer->screen_probe_radiance.uav, renderer->screen_probes.uav};

    const NriUpdateDescriptorRangeDesc updates[] = {
        {.descriptorSet = descriptor_set, .rangeIndex = 0, .descriptors = textures, .descriptorNum = 7},
        {.descriptorSet = descriptor_set, .rangeIndex = 1, .descriptors = frame, .descriptorNum = 1},
        {.descriptorSet = descriptor_set, .rangeIndex = 2, .descriptors = screen_output, .descriptorNum = 1},
        {.descriptorSet = descriptor_set, .rangeIndex = 3, .descriptors = trace_storage, .descriptorNum = 4},
        {.descriptorSet = descriptor_set, .rangeIndex = 4, .descriptors = scene, .descriptorNum = 3},
        {.descriptorSet = descriptor_set, .rangeIndex = 5, .descriptors = radiance_output, .descriptorNum = 1},
        {.descriptorSet = descriptor_set, .rangeIndex = 6, .descriptors = cache, .descriptorNum = 2},
        {.descriptorSet = descriptor_set, .rangeIndex = 7, .descriptors = probes, .descriptorNum = 2}
    };

    renderer->gpu->core.UpdateDescriptorRanges(updates, sizeof(updates) / sizeof(updates[0]));
    return true;
}

static void update_trace_descriptors(RENDERER *renderer) {
    if (renderer->trace_set) update_trace_descriptor_set(renderer, renderer->trace_set);
    if (renderer->radiance_direct_trace_set) update_trace_descriptor_set(renderer, renderer->radiance_direct_trace_set);
    if (renderer->emissive_trace_set) update_trace_descriptor_set(renderer, renderer->emissive_trace_set);
}

static bool update_radiance_scene_descriptor_set(RENDERER *renderer, NriDescriptorSet *descriptor_set) {
    if (!renderer || !descriptor_set || !renderer->radiance_constants_srv || !renderer->pass_constants_srv || !renderer->object_srv || !renderer->material_srv ||
        !renderer->radiance_scene.triangles_srv || !renderer->radiance_scene.emissive_triangles_srv || !renderer->sdf.models_srv || !renderer->sdf.voxels_srv ||
        !renderer->sdf.surface_ids_srv || !renderer->radiance_fallbacks.dynamic_grid_cells_srv || !renderer->radiance_fallbacks.dynamic_grid_indices_srv ||
        !renderer->radiance_fallbacks.global_sdf_clipmaps_srv || !renderer->radiance_fallbacks.global_sdf_page_table_srv || !renderer->radiance_fallbacks.global_sdf_bricks_srv ||
        !renderer->radiance_fallbacks.global_sdf_surface_ids_srv || !renderer->light_srv || !renderer->material_id.srv || !renderer->primitive_id.srv) {
        return false;
    }

    const NriDescriptor *constants[] = {renderer->radiance_constants_srv, renderer->pass_constants_srv};
    const NriDescriptor *scene_core[] = {
        renderer->object_srv,
        renderer->material_srv,
        renderer->radiance_scene.triangles_srv,
        renderer->radiance_scene.emissive_triangles_srv,
        renderer->sdf.models_srv,
        renderer->sdf.voxels_srv
    };
    const NriDescriptor *future_scene[] = {
        renderer->sdf.surface_ids_srv,
        renderer->radiance_fallbacks.dynamic_grid_cells_srv,
        renderer->radiance_fallbacks.dynamic_grid_indices_srv,
        renderer->radiance_fallbacks.global_sdf_clipmaps_srv,
        renderer->radiance_fallbacks.global_sdf_page_table_srv,
        renderer->radiance_fallbacks.global_sdf_bricks_srv,
        renderer->radiance_fallbacks.global_sdf_surface_ids_srv
    };
    const NriDescriptor *lights[] = {renderer->light_srv};
    const NriDescriptor *identity_textures[] = {renderer->material_id.srv, renderer->primitive_id.srv};

    const NriUpdateDescriptorRangeDesc updates[] = {
        {.descriptorSet = descriptor_set, .rangeIndex = 0, .descriptors = constants, .descriptorNum = 2},
        {.descriptorSet = descriptor_set, .rangeIndex = 1, .descriptors = scene_core, .descriptorNum = 6},
        {.descriptorSet = descriptor_set, .rangeIndex = 2, .descriptors = future_scene, .descriptorNum = 7},
        {.descriptorSet = descriptor_set, .rangeIndex = 3, .descriptors = lights, .descriptorNum = 1},
        {.descriptorSet = descriptor_set, .rangeIndex = 4, .descriptors = identity_textures, .descriptorNum = 2}
    };

    renderer->gpu->core.UpdateDescriptorRanges(updates, sizeof(updates) / sizeof(updates[0]));
    return true;
}

static bool update_radiance_scene_descriptors(RENDERER *renderer) {
    if (!update_radiance_scene_descriptor_set(renderer, renderer->radiance_scene_set)) return false;
    if (renderer->radiance_direct_scene_set && !update_radiance_scene_descriptor_set(renderer, renderer->radiance_direct_scene_set)) return false;
    if (renderer->emissive_scene_set && !update_radiance_scene_descriptor_set(renderer, renderer->emissive_scene_set)) return false;
    return true;
}

static bool update_radiance_direct_cache_descriptors(RENDERER *renderer) {
    if (!renderer || !renderer->radiance_direct_cache_set || !renderer->radiance_surface_cache.keys_uav || !renderer->radiance_surface_cache.entries_uav) return false;

    const NriDescriptor *descriptors[] = {renderer->radiance_surface_cache.keys_uav, renderer->radiance_surface_cache.entries_uav};
    const NriUpdateDescriptorRangeDesc update = {
        .descriptorSet = renderer->radiance_direct_cache_set,
        .rangeIndex = 0,
        .descriptors = descriptors,
        .descriptorNum = 2
    };

    renderer->gpu->core.UpdateDescriptorRanges(&update, 1);
    return true;
}

static bool update_emissive_probe_descriptors(RENDERER *renderer) {
    if (!renderer || !renderer->emissive_probe_set || !renderer->screen_probes.uav) return false;

    const NriDescriptor *probe[] = {renderer->screen_probes.uav};
    const NriUpdateDescriptorRangeDesc update = {
        .descriptorSet = renderer->emissive_probe_set,
        .rangeIndex = 0,
        .descriptors = probe,
        .descriptorNum = 1
    };

    renderer->gpu->core.UpdateDescriptorRanges(&update, 1);
    return true;
}

static bool create_size_dependent_resources(RENDERER *renderer, uint32_t width, uint32_t height) {
    destroy_compute_texture(renderer, &renderer->direct_radiance);
    destroy_compute_texture(renderer, &renderer->screen_probe_radiance);
    destroy_compute_texture(renderer, &renderer->screen_probes);

    const uint32_t probe_width = (width + SCREEN_PROBE_TILE_SIZE - 1u) / SCREEN_PROBE_TILE_SIZE;

    const uint32_t probe_height = (height + SCREEN_PROBE_TILE_SIZE - 1u) / SCREEN_PROBE_TILE_SIZE;

    const uint32_t ray_width = probe_width * SCREEN_PROBE_DIRECTION_SIZE;

    const uint32_t ray_height = probe_height * SCREEN_PROBE_DIRECTION_SIZE;

    const uint32_t ray_count = ray_width * ray_height;

    if (!create_gbuffer(renderer, width, height) || !create_hzb(renderer, width, height) || !create_screen_trace(renderer, ray_width, ray_height) ||

        /* full-resolution direct lighting */
        !create_compute_texture(renderer, &renderer->direct_radiance, width, height) ||

        /* 4x4 directional samples per 8x8 probe */
        !create_compute_texture(renderer, &renderer->screen_probe_radiance, ray_width, ray_height) ||

        /* one resolved value per screen probe */
        !create_compute_texture(renderer, &renderer->screen_probes, probe_width, probe_height) ||

        !create_trace_buffer(renderer, ray_count) || !create_trace_queue(renderer, ray_count)) {
        return false;
    }

    update_hzb_descriptors(renderer);
    update_present_descriptors(renderer);
    update_trace_descriptors(renderer);

    if (!update_emissive_probe_descriptors(renderer)) return false;
    if (renderer->radiance_scene.triangles_srv && !update_radiance_scene_descriptors(renderer)) return false;

    return true;
}

static bool update_scene_objects(RENDERER *renderer) {
    uint32_t gpu_index = 0;
    uint32_t light_index = 0;
    uint32_t sdf_index = 0;

    for (uint32_t scene_index = 0; scene_index < renderer->scene->object_count; ++scene_index) {
        OBJECT *object = &renderer->scene->objects[scene_index];

        if (object->type == MODEL) {
            if (gpu_index >= renderer->gpu_object_count || sdf_index >= renderer->sdf.model_count) return false;

            GPU_OBJECT *gpu_object = &renderer->cpu_objects[gpu_index++];
            const MAT4 world = mat4_transform(object->transform);
            gpu_object->previous_world = renderer->has_previous_frame ? gpu_object->world : world;
            gpu_object->world = world;
            gpu_object->normal_world = mat4_normal_transform(object->transform);
            gpu_object->object_id = scene_index + 1u;
            gpu_object->revision = object->revision;
            gpu_object->state = (uint32_t)object->state;
            gpu_object->type = (uint32_t)object->type;

            MAT4 inverse;

            if (!mat4_inverse(world, &inverse)) return false;
            renderer->sdf.cpu_models[sdf_index].world_to_local = inverse;
            renderer->sdf.cpu_models[sdf_index].object_id = scene_index + 1u;
            renderer->sdf.cpu_models[sdf_index].state = (uint32_t)object->state;
            renderer->sdf.cpu_models[sdf_index].revision = object->revision;
            ++sdf_index;
        } else if (object->type == LIGHT) {
            if (light_index + 1u >= renderer->light_count) return false;
            fill_light(&renderer->cpu_lights[light_index++], object);
        }
    }

    GPU_LIGHT *sky = &renderer->cpu_lights[light_index++];
    memset(sky, 0, sizeof(*sky));
    sky->direction_type[3] = 3.0f;
    sky->color_intensity[0] = renderer->scene->sky.zenith.x;
    sky->color_intensity[1] = renderer->scene->sky.zenith.y;
    sky->color_intensity[2] = renderer->scene->sky.zenith.z;
    sky->color_intensity[3] = renderer->scene->sky.intensity;
    sky->spot_angles[0] = renderer->scene->sky.horizon.x;
    sky->spot_angles[1] = renderer->scene->sky.horizon.y;
    sky->spot_angles[2] = renderer->scene->sky.horizon.z;

    return gpu_index == renderer->gpu_object_count && light_index == renderer->light_count && sdf_index == renderer->sdf.model_count;
}

static bool make_frame_constants(RENDERER *renderer, MAT4 *view_projection, FRAME_CONSTANTS *frame) {
    const MAT4 view = mat4_view(renderer->camera);
    const MAT4 projection = mat4_reverse_z_projection(renderer->camera.fov_y, (float)renderer->width / renderer->height, renderer->camera.near_plane, renderer->camera.far_plane);
    *view_projection = mat4_mul(view, projection);
    MAT4 inverse;

    if (!mat4_inverse(*view_projection, &inverse)) return false;
    *frame = (FRAME_CONSTANTS){
        .view_projection = *view_projection,
        .inverse_view_projection = inverse,
        .previous_view_projection = renderer->has_previous_frame ? renderer->previous_view_projection : *view_projection,
        .camera_position = {renderer->camera.position.x, renderer->camera.position.y, renderer->camera.position.z, (float)(renderer->gpu->frame_index & 0x00ffffffu)},
        .resolution =
        {(float)renderer->width, (float)renderer->height, 1.0f / renderer->width, 1.0f / renderer->height}, // ^^ carries frame stamp for SURFACE_CACHE_ENTRY.last_frame
        .trace_params = {200.0f, 0.05f, 0.10f, 0.05f},
        .trace_limits = {128u, 5u, renderer->light_count, renderer->sdf.model_count}
    };

    return true;
}

static bool stream_dynamic_data(RENDERER *renderer, NriCommandBuffer *command_buffer, const FRAME_CONSTANTS *frame) {
    memset(&renderer->pass_constants, 0, sizeof(renderer->pass_constants));
    renderer->pass_constants.dispatch[0] = renderer->radiance_constants.feature_flags[1];
    renderer->pass_constants.dimensions[0] = renderer->screen_probes.width;
    renderer->pass_constants.dimensions[1] = renderer->screen_probes.height;

    NriStreamerCopyBatch batch = renderer->gpu->streamer_api.BeginStreamerCopyBatch(renderer->gpu->streamer);

    if (!batch) return false;

    const NriDataSize object_data = {
        .data = renderer->cpu_objects,
        .size = (uint64_t)renderer->gpu_object_count * sizeof(GPU_OBJECT)
    };

    const NriDataSize light_data = {
        .data = renderer->cpu_lights,
        .size = (uint64_t)renderer->light_count * sizeof(GPU_LIGHT)
    };

    const NriDataSize sdf_data = {
        .data = renderer->sdf.cpu_models,
        .size = (uint64_t)renderer->sdf.model_count * sizeof(GPU_SDF_MODEL)
    };

    const NriDataSize frame_data = {
        .data = frame,
        .size = sizeof(*frame)
    };

    const NriDataSize pass_data = {
        .data = &renderer->pass_constants,
        .size = sizeof(renderer->pass_constants)
    };

    const NriStreamBufferDataDesc uploads[] = {
        {
            .dataChunks = &object_data,
            .dataChunkNum = 1,
            .placementAlignment = 16,
            .copyBatch = batch,
            .dstBuffer = renderer->object_buffer
        },
        {
            .dataChunks = &light_data,
            .dataChunkNum = 1,
            .placementAlignment = 16,
            .copyBatch = batch,
            .dstBuffer = renderer->light_buffer
        },
        {
            .dataChunks = &sdf_data,
            .dataChunkNum = 1,
            .placementAlignment = 16,
            .copyBatch = batch,
            .dstBuffer = renderer->sdf.models
        },
        {
            .dataChunks = &frame_data,
            .dataChunkNum = 1,
            .placementAlignment = 16,
            .copyBatch = batch,
            .dstBuffer = renderer->frame_buffer
        },
        {
            .dataChunks = &pass_data,
            .dataChunkNum = 1,
            .placementAlignment = 16,
            .copyBatch = batch,
            .dstBuffer = renderer->pass_constants_buffer
        }
    };

    for (uint32_t i = 0; i < 5; ++i) {
        const NriBufferOffset streamed = renderer->gpu->streamer_api.StreamBufferData(renderer->gpu->streamer, &uploads[i]);

        if (!streamed.buffer) return false;
    }

    const NriAccessStage copy = {
        .access = NriAccessBits_COPY_DESTINATION,
        .stages = NriStageBits_COPY
    };

    const NriBufferBarrierDesc before[] = {
        {
            .buffer = renderer->object_buffer,
            .before = renderer->object_state,
            .after = copy
        },
        {
            .buffer = renderer->light_buffer,
            .before = renderer->light_state,
            .after = copy
        },
        {
            .buffer = renderer->sdf.models,
            .before = renderer->sdf.models_state,
            .after = copy
        },
        {
            .buffer = renderer->frame_buffer,
            .before = renderer->frame_state,
            .after = copy
        },
        {
            .buffer = renderer->pass_constants_buffer,
            .before = renderer->pass_constants_state,
            .after = copy
        }
    };

    renderer->gpu->core.CmdBarrier(command_buffer, &(NriBarrierDesc){
        .buffers = before,
        .bufferNum = 5
    });
    renderer->gpu->streamer_api.CmdCopyStreamedData(command_buffer, renderer->gpu->streamer, batch);

    const NriAccessStage object_read = {
        .access = NriAccessBits_SHADER_RESOURCE,
        .stages = NriStageBits_VERTEX_SHADER | NriStageBits_COMPUTE_SHADER
    };

    const NriAccessStage compute_read = {
        .access = NriAccessBits_SHADER_RESOURCE,
        .stages = NriStageBits_COMPUTE_SHADER
    };

    const NriAccessStage frame_read = {
        .access = NriAccessBits_CONSTANT_BUFFER,
        .stages = NriStageBits_VERTEX_SHADER | NriStageBits_COMPUTE_SHADER
    };

    const NriAccessStage pass_read = {
        .access = NriAccessBits_CONSTANT_BUFFER,
        .stages = NriStageBits_COMPUTE_SHADER
    };

    const NriBufferBarrierDesc after[] = {
        {
            .buffer = renderer->object_buffer,
            .before = copy,
            .after = object_read
        },
        {
            .buffer = renderer->light_buffer,
            .before = copy,
            .after = compute_read
        },
        {
            .buffer = renderer->sdf.models,
            .before = copy,
            .after = compute_read
        },
        {
            .buffer = renderer->frame_buffer,
            .before = copy,
            .after = frame_read
        },
        {
            .buffer = renderer->pass_constants_buffer,
            .before = copy,
            .after = pass_read
        }
    };

    renderer->gpu->core.CmdBarrier(command_buffer, &(NriBarrierDesc){
        .buffers = after,
        .bufferNum = 5
    });
    renderer->object_state = object_read;
    renderer->light_state = compute_read;
    renderer->sdf.models_state = compute_read;
    renderer->frame_state = frame_read;
    renderer->pass_constants_state = pass_read;

    return true;
}

static void transition_gbuffer_for_render(RENDERER *renderer, NriCommandBuffer *command_buffer) {
    const NriAccessLayoutStage color = {
        .access = NriAccessBits_COLOR_ATTACHMENT_WRITE,
        .layout = NriLayout_COLOR_ATTACHMENT,
        .stages = NriStageBits_COLOR_ATTACHMENT
    };

    const NriAccessLayoutStage depth = {
        .access = NriAccessBits_DEPTH_STENCIL_ATTACHMENT_WRITE,
        .layout = NriLayout_DEPTH_STENCIL_ATTACHMENT,
        .stages = NriStageBits_DEPTH_STENCIL_ATTACHMENT
    };

    NriTextureBarrierDesc barriers[] = {
        {
            .texture = renderer->normal_roughness.texture,
            .before = renderer->normal_roughness.state,
            .after = color,
            .mipNum = 1,
            .layerNum = 1,
            .planes = NriPlaneBits_COLOR
        },
        {
            .texture = renderer->albedo_metallic.texture,
            .before = renderer->albedo_metallic.state,
            .after = color,
            .mipNum = 1,
            .layerNum = 1,
            .planes = NriPlaneBits_COLOR
        },
        {
            .texture = renderer->emissive.texture,
            .before = renderer->emissive.state,
            .after = color,
            .mipNum = 1,
            .layerNum = 1,
            .planes = NriPlaneBits_COLOR
        },
        {
            .texture = renderer->velocity.texture,
            .before = renderer->velocity.state,
            .after = color,
            .mipNum = 1,
            .layerNum = 1,
            .planes = NriPlaneBits_COLOR
        },
        {
            .texture = renderer->object_id.texture,
            .before = renderer->object_id.state,
            .after = color,
            .mipNum = 1,
            .layerNum = 1,
            .planes = NriPlaneBits_COLOR
        },
        {
            .texture = renderer->material_id.texture,
            .before = renderer->material_id.state,
            .after = color,
            .mipNum = 1,
            .layerNum = 1,
            .planes = NriPlaneBits_COLOR
        },
        {
            .texture = renderer->primitive_id.texture,
            .before = renderer->primitive_id.state,
            .after = color,
            .mipNum = 1,
            .layerNum = 1,
            .planes = NriPlaneBits_COLOR
        },
        {
            .texture = renderer->depth.texture,
            .before = renderer->depth.state,
            .after = depth,
            .mipNum = 1,
            .layerNum = 1,
            .planes = NriPlaneBits_DEPTH
        }
    };

    renderer->gpu->core.CmdBarrier(command_buffer, &(NriBarrierDesc){
        .textures = barriers,
        .textureNum = 8
    });
    renderer->normal_roughness.state = color;
    renderer->albedo_metallic.state = color;
    renderer->emissive.state = color;
    renderer->velocity.state = color;
    renderer->object_id.state = color;
    renderer->material_id.state = color;
    renderer->primitive_id.state = color;
    renderer->depth.state = depth;
}

static void transition_depth_for_hzb(RENDERER *renderer, NriCommandBuffer *command_buffer) {
    const NriAccessLayoutStage read = {
        .access = NriAccessBits_SHADER_RESOURCE,
        .layout = NriLayout_SHADER_RESOURCE,
        .stages = NriStageBits_COMPUTE_SHADER | NriStageBits_FRAGMENT_SHADER
    };

    const NriTextureBarrierDesc barrier = {
        .texture = renderer->depth.texture,
        .before = renderer->depth.state,
        .after = read,
        .mipNum = 1,
        .layerNum = 1,
        .planes = NriPlaneBits_DEPTH
    };

    renderer->gpu->core.CmdBarrier(command_buffer, &(NriBarrierDesc){
        .textures = &barrier,
        .textureNum = 1
    });
    renderer->depth.state = read;
}

static void transition_gbuffer_for_read(RENDERER *renderer, NriCommandBuffer *command_buffer) {
    const NriAccessLayoutStage read = {
        .access = NriAccessBits_SHADER_RESOURCE,
        .layout = NriLayout_SHADER_RESOURCE,
        .stages = NriStageBits_COMPUTE_SHADER | NriStageBits_FRAGMENT_SHADER
    };

    NriTextureBarrierDesc barriers[] = {
        {
            .texture = renderer->normal_roughness.texture,
            .before = renderer->normal_roughness.state,
            .after = read,
            .mipNum = 1,
            .layerNum = 1,
            .planes = NriPlaneBits_COLOR
        },
        {
            .texture = renderer->albedo_metallic.texture,
            .before = renderer->albedo_metallic.state,
            .after = read,
            .mipNum = 1,
            .layerNum = 1,
            .planes = NriPlaneBits_COLOR
        },
        {
            .texture = renderer->emissive.texture,
            .before = renderer->emissive.state,
            .after = read,
            .mipNum = 1,
            .layerNum = 1,
            .planes = NriPlaneBits_COLOR
        },
        {
            .texture = renderer->velocity.texture,
            .before = renderer->velocity.state,
            .after = read,
            .mipNum = 1,
            .layerNum = 1,
            .planes = NriPlaneBits_COLOR
        },
        {
            .texture = renderer->object_id.texture,
            .before = renderer->object_id.state,
            .after = read,
            .mipNum = 1,
            .layerNum = 1,
            .planes = NriPlaneBits_COLOR
        },
        {
            .texture = renderer->material_id.texture,
            .before = renderer->material_id.state,
            .after = read,
            .mipNum = 1,
            .layerNum = 1,
            .planes = NriPlaneBits_COLOR
        },
        {
            .texture = renderer->primitive_id.texture,
            .before = renderer->primitive_id.state,
            .after = read,
            .mipNum = 1,
            .layerNum = 1,
            .planes = NriPlaneBits_COLOR
        }
    };

    renderer->gpu->core.CmdBarrier(command_buffer, &(NriBarrierDesc){
        .textures = barriers,
        .textureNum = 7
    });
    renderer->normal_roughness.state = read;
    renderer->albedo_metallic.state = read;
    renderer->emissive.state = read;
    renderer->velocity.state = read;
    renderer->object_id.state = read;
    renderer->material_id.state = read;
    renderer->primitive_id.state = read;
}

static void bind_trace(RENDERER *renderer, NriCommandBuffer *command_buffer, NriPipeline *pipeline) {
    renderer->gpu->core.CmdSetPipelineLayout(command_buffer, NriBindPoint_COMPUTE, renderer->trace_layout);
    renderer->gpu->core.CmdSetPipeline(command_buffer, pipeline);

    const NriSetDescriptorSetDesc set = {
        .setIndex = 0,
        .descriptorSet = renderer->trace_set,
        .bindPoint = NriBindPoint_COMPUTE
    };

    renderer->gpu->core.CmdSetDescriptorSet(command_buffer, &set);
}

static void bind_radiance_world(RENDERER *renderer, NriCommandBuffer *command_buffer, NriPipeline *pipeline) {
    renderer->gpu->core.CmdSetPipelineLayout(command_buffer, NriBindPoint_COMPUTE, renderer->radiance_direct_layout);
    renderer->gpu->core.CmdSetPipeline(command_buffer, pipeline);

    const NriSetDescriptorSetDesc sets[] = {
        {.setIndex = 0, .descriptorSet = renderer->radiance_direct_trace_set, .bindPoint = NriBindPoint_COMPUTE},
        {.setIndex = 1, .descriptorSet = renderer->radiance_direct_scene_set, .bindPoint = NriBindPoint_COMPUTE},
        {.setIndex = 2, .descriptorSet = renderer->radiance_direct_cache_set, .bindPoint = NriBindPoint_COMPUTE}
    };

    for (uint32_t i = 0; i < sizeof(sets) / sizeof(sets[0]); ++i)
        renderer->gpu->core.CmdSetDescriptorSet(command_buffer, &sets[i]);
}

static void build_hzb(RENDERER *renderer, NriCommandBuffer *command_buffer) {
    const NriAccessLayoutStage write = {
        .access = NriAccessBits_SHADER_RESOURCE_STORAGE,
        .layout = NriLayout_SHADER_RESOURCE_STORAGE,
        .stages = NriStageBits_COMPUTE_SHADER
    };

    const NriAccessLayoutStage read = {
        .access = NriAccessBits_SHADER_RESOURCE,
        .layout = NriLayout_SHADER_RESOURCE,
        .stages = NriStageBits_COMPUTE_SHADER | NriStageBits_FRAGMENT_SHADER
    };

    renderer->gpu->core.CmdSetPipelineLayout(command_buffer, NriBindPoint_COMPUTE, renderer->hzb_layout);
    renderer->gpu->core.CmdSetPipeline(command_buffer, renderer->hzb_pipeline);

    for (uint32_t mip = 0; mip < renderer->hzb.mip_count; ++mip) {
        const NriTextureBarrierDesc to_write = {
            .texture = renderer->hzb.texture,
            .before = renderer->hzb.mip_states[mip],
            .after = write,
            .mipOffset = (NriDim_t)mip,
            .mipNum = 1,
            .layerNum = 1,
            .planes = NriPlaneBits_COLOR
        };

        renderer->gpu->core.CmdBarrier(command_buffer, &(NriBarrierDesc){
            .textures = &to_write,
            .textureNum = 1
        });
        const NriSetDescriptorSetDesc set = {
            .setIndex = 0,
            .descriptorSet = renderer->hzb_sets[mip],
            .bindPoint = NriBindPoint_COMPUTE
        };

        renderer->gpu->core.CmdSetDescriptorSet(command_buffer, &set);

        const uint32_t w = max(1u, renderer->hzb.width >> mip);
        const uint32_t h = max(1u, renderer->hzb.height >> mip);

        renderer->gpu->core.CmdDispatch(command_buffer, &(NriDispatchDesc){
            .workGroupNumX = (w + 7u) / 8u,
            .workGroupNumY = (h + 7u) / 8u,
            .workGroupNumZ = 1
        });
        const NriTextureBarrierDesc to_read = {
            .texture = renderer->hzb.texture,
            .before = write,
            .after = read,
            .mipOffset = (NriDim_t)mip,
            .mipNum = 1,
            .layerNum = 1,
            .planes = NriPlaneBits_COLOR
        };

        renderer->gpu->core.CmdBarrier(command_buffer, &(NriBarrierDesc){
            .textures = &to_read,
            .textureNum = 1
        });
        renderer->hzb.mip_states[mip] = read;
    }
}

static void build_direct_radiance(RENDERER *renderer, NriCommandBuffer *command_buffer) {
    const NriAccessLayoutStage write = {
        .access = NriAccessBits_SHADER_RESOURCE_STORAGE,
        .layout = NriLayout_SHADER_RESOURCE_STORAGE,
        .stages = NriStageBits_COMPUTE_SHADER
    };

    const NriTextureBarrierDesc to_write = {
        .texture = renderer->direct_radiance.texture,
        .before = renderer->direct_radiance.state,
        .after = write,
        .mipNum = 1,
        .layerNum = 1,
        .planes = NriPlaneBits_COLOR
    };

    const NriAccessStage cache_storage = {
        .access = NriAccessBits_SHADER_RESOURCE_STORAGE,
        .stages = NriStageBits_COMPUTE_SHADER
    };
    const NriBufferBarrierDesc cache_barriers[] = {
        {.buffer = renderer->radiance_surface_cache.keys, .before = renderer->radiance_surface_cache.keys_state, .after = cache_storage},
        {.buffer = renderer->radiance_surface_cache.entries, .before = renderer->radiance_surface_cache.entries_state, .after = cache_storage}
    };

    renderer->gpu->core.CmdBarrier(command_buffer, &(NriBarrierDesc){
        .textures = &to_write,
        .textureNum = 1,
        .buffers = cache_barriers,
        .bufferNum = 2
    });
    renderer->radiance_surface_cache.keys_state = cache_storage;
    renderer->radiance_surface_cache.entries_state = cache_storage;
    bind_radiance_world(renderer, command_buffer, renderer->direct_radiance_pipeline);
    renderer->gpu->core.CmdDispatch(
        command_buffer, &(NriDispatchDesc){
            .workGroupNumX = (renderer->width + 7u) / 8u,
            .workGroupNumY = (renderer->height + 7u) / 8u,
            .workGroupNumZ = 1
        }
    );
    const NriAccessLayoutStage read = {
        .access = NriAccessBits_SHADER_RESOURCE,
        .layout = NriLayout_SHADER_RESOURCE,
        .stages = NriStageBits_COMPUTE_SHADER | NriStageBits_FRAGMENT_SHADER
    };

    const NriTextureBarrierDesc to_read = {
        .texture = renderer->direct_radiance.texture,
        .before = write,
        .after = read,
        .mipNum = 1,
        .layerNum = 1,
        .planes = NriPlaneBits_COLOR
    };

    renderer->gpu->core.CmdBarrier(command_buffer, &(NriBarrierDesc){
        .textures = &to_read,
        .textureNum = 1
    });
    renderer->direct_radiance.state = read;
}

static void build_surface_cache(RENDERER *renderer, NriCommandBuffer *command_buffer) {
    const NriAccessStage storage = {
        .access = NriAccessBits_SHADER_RESOURCE_STORAGE,
        .stages = NriStageBits_COMPUTE_SHADER
    };

    const NriBufferBarrierDesc barriers[] = {
        {
            .buffer = renderer->surface_cache.keys,
            .before = renderer->surface_cache.keys_state,
            .after = storage
        },
        {
            .buffer = renderer->surface_cache.entries,
            .before = renderer->surface_cache.entries_state,
            .after = storage
        }
    };

    renderer->gpu->core.CmdBarrier(command_buffer, &(NriBarrierDesc){
        .buffers = barriers,
        .bufferNum = 2
    });
    renderer->surface_cache.keys_state = storage;
    renderer->surface_cache.entries_state = storage;
    bind_trace(renderer, command_buffer, renderer->surface_cache_pipeline);
    renderer->gpu->core.CmdDispatch(
        command_buffer, &(NriDispatchDesc){
            .workGroupNumX = (renderer->width + 7u) / 8u,
            .workGroupNumY = (renderer->height + 7u) / 8u,
            .workGroupNumZ = 1
        }
    );
}

static void build_screen_trace(RENDERER *renderer, NriCommandBuffer *command_buffer) {
    const NriAccessLayoutStage texture_write = {
        .access = NriAccessBits_SHADER_RESOURCE_STORAGE,
        .layout = NriLayout_SHADER_RESOURCE_STORAGE,
        .stages = NriStageBits_COMPUTE_SHADER
    };

    const NriAccessStage buffer_write = {
        .access = NriAccessBits_SHADER_RESOURCE_STORAGE,
        .stages = NriStageBits_COMPUTE_SHADER
    };

    const NriTextureBarrierDesc texture_barriers[] = {
        {
            .texture = renderer->screen_trace.texture,
            .before = renderer->screen_trace.state,
            .after = texture_write,
            .mipNum = 1,
            .layerNum = 1,
            .planes = NriPlaneBits_COLOR
        },
        {
            .texture = renderer->screen_probe_radiance.texture,
            .before = renderer->screen_probe_radiance.state,
            .after = texture_write,
            .mipNum = 1,
            .layerNum = 1,
            .planes = NriPlaneBits_COLOR
        }
    };

    const NriBufferBarrierDesc buffer_barrier = {
        .buffer = renderer->trace_hits.buffer,
        .before = renderer->trace_hits.state,
        .after = buffer_write
    };

    renderer->gpu->core.CmdBarrier(command_buffer, &(NriBarrierDesc){
        .textures = texture_barriers,
        .textureNum = 2,
        .buffers = &buffer_barrier,
        .bufferNum = 1
    });

    renderer->screen_trace.state = texture_write;
    renderer->screen_probe_radiance.state = texture_write;
    renderer->trace_hits.state = buffer_write;
    bind_trace(renderer, command_buffer, renderer->screen_trace_pipeline);
    renderer->gpu->core.CmdDispatch(
        command_buffer,
        &(NriDispatchDesc){
            .workGroupNumX = (renderer->screen_probe_radiance.width + 7u) / 8u,
            .workGroupNumY = (renderer->screen_probe_radiance.height + 7u) / 8u,
            .workGroupNumZ = 1
        }
    );
}

static void build_miss_queue(RENDERER *renderer, NriCommandBuffer *command_buffer) {
    const NriAccessStage storage = {
        .access = NriAccessBits_SHADER_RESOURCE_STORAGE,
        .stages = NriStageBits_COMPUTE_SHADER
    };

    const NriBufferBarrierDesc to_storage[] = {
        {
            .buffer = renderer->miss_queue.rays,
            .before = renderer->miss_queue.rays_state,
            .after = storage
        },
        {
            .buffer = renderer->miss_queue.count,
            .before = renderer->miss_queue.count_state,
            .after = storage
        },
        {
            .buffer = renderer->miss_queue.dispatch_args,
            .before = renderer->miss_queue.dispatch_args_state,
            .after = storage
        },
        {
            .buffer = renderer->trace_hits.buffer,
            .before = renderer->trace_hits.state,
            .after = storage
        }
    };

    renderer->gpu->core.CmdBarrier(command_buffer, &(NriBarrierDesc){
        .buffers = to_storage,
        .bufferNum = 4
    });
    renderer->miss_queue.rays_state = storage;
    renderer->miss_queue.count_state = storage;
    renderer->miss_queue.dispatch_args_state = storage;
    renderer->trace_hits.state = storage;

    bind_trace(renderer, command_buffer, renderer->trace_reset_pipeline);
    renderer->gpu->core.CmdDispatch(command_buffer, &(NriDispatchDesc){
        .workGroupNumX = 1,
        .workGroupNumY = 1,
        .workGroupNumZ = 1
    });
    const NriBufferBarrierDesc sync_reset[] = {
        {
            .buffer = renderer->miss_queue.count,
            .before = storage,
            .after = storage
        }, {
            .buffer = renderer->miss_queue.dispatch_args,
            .before = storage,
            .after = storage
        }
    };

    renderer->gpu->core.CmdBarrier(command_buffer, &(NriBarrierDesc){
        .buffers = sync_reset,
        .bufferNum = 2
    });

    bind_trace(renderer, command_buffer, renderer->trace_compact_pipeline);
    renderer->gpu->core.CmdDispatch(
        command_buffer,
        &(NriDispatchDesc){
            .workGroupNumX = (renderer->screen_probe_radiance.width + 7u) / 8u,
            .workGroupNumY = (renderer->screen_probe_radiance.height + 7u) / 8u,
            .workGroupNumZ = 1
        }
    );
    const NriBufferBarrierDesc sync_compact[] = {
        {
            .buffer = renderer->miss_queue.count,
            .before = storage,
            .after = storage
        }, {
            .buffer = renderer->miss_queue.rays,
            .before = storage,
            .after = storage
        }
    };

    renderer->gpu->core.CmdBarrier(command_buffer, &(NriBarrierDesc){
        .buffers = sync_compact,
        .bufferNum = 2
    });

    bind_trace(renderer, command_buffer, renderer->trace_args_pipeline);
    renderer->gpu->core.CmdDispatch(command_buffer, &(NriDispatchDesc){
        .workGroupNumX = 1,
        .workGroupNumY = 1,
        .workGroupNumZ = 1
    });
    const NriAccessStage argument = {
        .access = NriAccessBits_ARGUMENT_BUFFER,
        .stages = NriStageBits_INDIRECT
    };

    const NriBufferBarrierDesc args_barrier = {
        .buffer = renderer->miss_queue.dispatch_args,
        .before = storage,
        .after = argument
    };

    renderer->gpu->core.CmdBarrier(command_buffer, &(NriBarrierDesc){
        .buffers = &args_barrier,
        .bufferNum = 1
    });
    renderer->miss_queue.dispatch_args_state = argument;
}

static void build_sdf_trace(RENDERER *renderer, NriCommandBuffer *command_buffer) {
    const NriAccessStage storage = {
        .access = NriAccessBits_SHADER_RESOURCE_STORAGE,
        .stages = NriStageBits_COMPUTE_SHADER
    };

    const NriAccessLayoutStage texture_storage = {
        .access = NriAccessBits_SHADER_RESOURCE_STORAGE,
        .layout = NriLayout_SHADER_RESOURCE_STORAGE,
        .stages = NriStageBits_COMPUTE_SHADER
    };

    const NriBufferBarrierDesc cache_sync[] = {
        {
            .buffer = renderer->radiance_surface_cache.keys,
            .before = renderer->radiance_surface_cache.keys_state,
            .after = storage
        },
        {
            .buffer = renderer->radiance_surface_cache.entries,
            .before = renderer->radiance_surface_cache.entries_state,
            .after = storage
        }
    };

    const NriTextureBarrierDesc probe_sync = {
        .texture = renderer->screen_probe_radiance.texture,
        .before = renderer->screen_probe_radiance.state,
        .after = texture_storage,
        .mipNum = 1,
        .layerNum = 1,
        .planes = NriPlaneBits_COLOR
    };

    renderer->gpu->core.CmdBarrier(command_buffer, &(NriBarrierDesc){
        .buffers = cache_sync,
        .bufferNum = 2,
        .textures = &probe_sync,
        .textureNum = 1
    });

    renderer->screen_probe_radiance.state = texture_storage;
    renderer->radiance_surface_cache.keys_state = storage;
    renderer->radiance_surface_cache.entries_state = storage;

    bind_radiance_world(renderer, command_buffer, renderer->sdf_trace_pipeline);
    renderer->gpu->core.CmdDispatchIndirect(command_buffer, renderer->miss_queue.dispatch_args, 0u);
}

static void finish_screen_trace(RENDERER *renderer, NriCommandBuffer *command_buffer) {
    const NriAccessLayoutStage read = {
        .access = NriAccessBits_SHADER_RESOURCE,
        .layout = NriLayout_SHADER_RESOURCE,
        .stages = NriStageBits_FRAGMENT_SHADER
    };

    const NriTextureBarrierDesc barrier = {
        .texture = renderer->screen_trace.texture,
        .before = renderer->screen_trace.state,
        .after = read,
        .mipNum = 1,
        .layerNum = 1,
        .planes = NriPlaneBits_COLOR
    };

    renderer->gpu->core.CmdBarrier(command_buffer, &(NriBarrierDesc){
        .textures = &barrier,
        .textureNum = 1
    });
    renderer->screen_trace.state = read;
}

static void build_screen_probes(RENDERER *renderer, NriCommandBuffer *command_buffer) {

    const NriAccessLayoutStage storage = {
        .access = NriAccessBits_SHADER_RESOURCE_STORAGE,
        .layout = NriLayout_SHADER_RESOURCE_STORAGE,
        .stages = NriStageBits_COMPUTE_SHADER
    };

    const NriTextureBarrierDesc barriers[] = {
        {
            .texture = renderer->screen_probe_radiance.texture,
            .before = renderer->screen_probe_radiance.state,
            .after = storage,
            .mipNum = 1,
            .layerNum = 1,
            .planes = NriPlaneBits_COLOR
        },
        {
            .texture = renderer->screen_probes.texture,
            .before = renderer->screen_probes.state,
            .after = storage,
            .mipNum = 1,
            .layerNum = 1,
            .planes = NriPlaneBits_COLOR
        }
    };

    renderer->gpu->core.CmdBarrier(command_buffer, &(NriBarrierDesc){
        .textures = barriers,
        .textureNum = 2
    });

    renderer->screen_probe_radiance.state = storage;
    renderer->screen_probes.state = storage;

    bind_trace(renderer, command_buffer, renderer->screen_probes_pipeline);

    renderer->gpu->core.CmdDispatch(
        command_buffer,
        &(NriDispatchDesc){
            .workGroupNumX = (renderer->screen_probes.width + 7u) / 8u,
            .workGroupNumY = (renderer->screen_probes.height + 7u) / 8u,
            .workGroupNumZ = 1
        }
    );

    const NriAccessLayoutStage read = {
        .access = NriAccessBits_SHADER_RESOURCE,
        .layout = NriLayout_SHADER_RESOURCE,
        .stages = NriStageBits_FRAGMENT_SHADER
    };

    const NriTextureBarrierDesc to_read = {
        .texture = renderer->screen_probes.texture,
        .before = storage,
        .after = read,
        .mipNum = 1,
        .layerNum = 1,
        .planes = NriPlaneBits_COLOR
    };

    renderer->gpu->core.CmdBarrier(command_buffer, &(NriBarrierDesc){
        .textures = &to_read,
        .textureNum = 1
    });
    renderer->screen_probes.state = read;
}


static void bind_emissive(RENDERER *renderer, NriCommandBuffer *command_buffer) {
    renderer->gpu->core.CmdSetPipelineLayout(command_buffer, NriBindPoint_COMPUTE, renderer->emissive_layout);
    renderer->gpu->core.CmdSetPipeline(command_buffer, renderer->emissive_pipeline);

    const NriSetDescriptorSetDesc sets[] = {
        {.setIndex = 0, .descriptorSet = renderer->emissive_trace_set, .bindPoint = NriBindPoint_COMPUTE},
        {.setIndex = 1, .descriptorSet = renderer->emissive_scene_set, .bindPoint = NriBindPoint_COMPUTE},
        {.setIndex = 2, .descriptorSet = renderer->emissive_probe_set, .bindPoint = NriBindPoint_COMPUTE}
    };

    for (uint32_t i = 0; i < sizeof(sets) / sizeof(sets[0]); ++i)
        renderer->gpu->core.CmdSetDescriptorSet(command_buffer, &sets[i]);
}

static void build_emissive_gather(RENDERER *renderer, NriCommandBuffer *command_buffer) {
    if (!renderer->radiance_scene.emissive_triangle_count ||
        !(renderer->radiance_constants.feature_flags[0] & RADIANCE_FEATURE_EMISSIVE)) {
        return;
    }

    const NriAccessLayoutStage storage = {
        .access = NriAccessBits_SHADER_RESOURCE_STORAGE,
        .layout = NriLayout_SHADER_RESOURCE_STORAGE,
        .stages = NriStageBits_COMPUTE_SHADER
    };

    const NriTextureBarrierDesc to_storage = {
        .texture = renderer->screen_probes.texture,
        .before = renderer->screen_probes.state,
        .after = storage,
        .mipNum = 1,
        .layerNum = 1,
        .planes = NriPlaneBits_COLOR
    };

    renderer->gpu->core.CmdBarrier(command_buffer, &(NriBarrierDesc){
        .textures = &to_storage,
        .textureNum = 1
    });
    renderer->screen_probes.state = storage;

    bind_emissive(renderer, command_buffer);
    renderer->gpu->core.CmdDispatch(
        command_buffer,
        &(NriDispatchDesc){
            .workGroupNumX = (renderer->screen_probes.width + 7u) / 8u,
            .workGroupNumY = (renderer->screen_probes.height + 7u) / 8u,
            .workGroupNumZ = 1
        }
    );

    const NriAccessLayoutStage read = {
        .access = NriAccessBits_SHADER_RESOURCE,
        .layout = NriLayout_SHADER_RESOURCE,
        .stages = NriStageBits_FRAGMENT_SHADER
    };

    const NriTextureBarrierDesc to_read = {
        .texture = renderer->screen_probes.texture,
        .before = storage,
        .after = read,
        .mipNum = 1,
        .layerNum = 1,
        .planes = NriPlaneBits_COLOR
    };

    renderer->gpu->core.CmdBarrier(command_buffer, &(NriBarrierDesc){
        .textures = &to_read,
        .textureNum = 1
    });
    renderer->screen_probes.state = read;
}

static void set_fullscreen_view(RENDERER *renderer, NriCommandBuffer *command_buffer) {
    const NriViewport viewport = {
        .x = 0.0f,
        .y = 0.0f,
        .width = (float)renderer->width,
        .height = (float)renderer->height,
        .depthMin = 0.0f,
        .depthMax = 1.0f
    };

    const NriRect scissor = {
        .x = 0,
        .y = 0,
        .width = (NriDim_t)renderer->width,
        .height = (NriDim_t)renderer->height
    };

    renderer->gpu->core.CmdSetViewports(command_buffer, &viewport, 1);
    renderer->gpu->core.CmdSetScissors(command_buffer, &scissor, 1);
}

static void record_gbuffer_pass(RENDERER *renderer, NriCommandBuffer *command_buffer) {
    const NriAttachmentDesc colors[] = {
        {
            .descriptor = renderer->normal_roughness.attachment,
            .clearValue = {
                .color = {
                    .f = {
                        .x = 0.0f,
                        .y = 0.0f,
                        .z = 0.0f,
                        .w = 1.0f
                    }
                }
            },
            .loadOp = NriLoadOp_CLEAR,
            .storeOp = NriStoreOp_STORE
        },
        {
            .descriptor = renderer->albedo_metallic.attachment,
            .clearValue = {
                .color = {
                    .f = {
                        .x = 0.0f,
                        .y = 0.0f,
                        .z = 0.0f,
                        .w = 0.0f
                    }
                }
            },
            .loadOp = NriLoadOp_CLEAR,
            .storeOp = NriStoreOp_STORE
        },
        {
            .descriptor = renderer->velocity.attachment,
            .clearValue = {
                .color = {
                    .f = {
                        .x = 0.0f,
                        .y = 0.0f,
                        .z = 0.0f,
                        .w = 0.0f
                    }
                }
            },
            .loadOp = NriLoadOp_CLEAR,
            .storeOp = NriStoreOp_STORE
        },
        {
            .descriptor = renderer->object_id.attachment,
            .clearValue = {
                .color = {
                    .ui = {
                        .x = 0u,
                        .y = 0u,
                        .z = 0u,
                        .w = 0u
                    }
                }
            },
            .loadOp = NriLoadOp_CLEAR,
            .storeOp = NriStoreOp_STORE
        },
        {
            .descriptor = renderer->emissive.attachment,
            .clearValue = {
                .color = {
                    .f = {
                        .x = 0.0f,
                        .y = 0.0f,
                        .z = 0.0f,
                        .w = 1.0f
                    }
                }
            },
            .loadOp = NriLoadOp_CLEAR,
            .storeOp = NriStoreOp_STORE
        },
        {
            .descriptor = renderer->material_id.attachment,
            .clearValue = {
                .color = {
                    .ui = {
                        .x = 0u,
                        .y = 0u,
                        .z = 0u,
                        .w = 0u
                    }
                }
            },
            .loadOp = NriLoadOp_CLEAR,
            .storeOp = NriStoreOp_STORE
        },
        {
            .descriptor = renderer->primitive_id.attachment,
            .clearValue = {
                .color = {
                    .ui = {
                        .x = 0u,
                        .y = 0u,
                        .z = 0u,
                        .w = 0u
                    }
                }
            },
            .loadOp = NriLoadOp_CLEAR,
            .storeOp = NriStoreOp_STORE
        }
    };

    const NriAttachmentDesc depth = {
        .descriptor = renderer->depth.attachment,
        .clearValue = {
            .depthStencil = {
                .depth = 0.0f,
                .stencil = 0
            }
        },
        .loadOp = NriLoadOp_CLEAR,
        .storeOp = NriStoreOp_STORE
    };

    const NriRenderingDesc rendering = {
        .colors = colors,
        .colorNum = 7,
        .depth = depth
    };

    renderer->gpu->core.CmdBeginRendering(command_buffer, &rendering);
    renderer->gpu->core.CmdSetPipelineLayout(command_buffer, NriBindPoint_GRAPHICS, renderer->gbuffer_layout);
    renderer->gpu->core.CmdSetPipeline(command_buffer, renderer->gbuffer_pipeline);

    const NriSetDescriptorSetDesc set = {
        .setIndex = 0,
        .descriptorSet = renderer->gbuffer_set,
        .bindPoint = NriBindPoint_GRAPHICS
    };

    renderer->gpu->core.CmdSetDescriptorSet(command_buffer, &set);

    const NriVertexBufferDesc vertex = {
        .buffer = renderer->vertex_buffer,
        .offset = 0,
        .stride = sizeof(GLTF_VERTEX)
    };

    renderer->gpu->core.CmdSetVertexBuffers(command_buffer, 0, &vertex, 1);
    set_fullscreen_view(renderer, command_buffer);

    for (uint32_t object_index = 0; object_index < renderer->gpu_object_count; ++object_index) {
        const GPU_OBJECT *object = &renderer->cpu_objects[object_index];

        const NriSetRootConstantsDesc root = {
            .rootConstantIndex = 0,
            .data = &object_index,
            .size = sizeof(object_index),
            .bindPoint = NriBindPoint_GRAPHICS
        };

        renderer->gpu->core.CmdSetRootConstants(command_buffer, &root);

        const NriDrawDesc draw = {
            .vertexNum = object->vertex_count,
            .instanceNum = 1,
            .baseVertex = object->first_vertex
        };

        renderer->gpu->core.CmdDraw(command_buffer, &draw);
    }

    renderer->gpu->core.CmdEndRendering(command_buffer);
}

static void record_present_pass(RENDERER *renderer, NriCommandBuffer *command_buffer, uint32_t swapchain_index) {
    const NriAttachmentDesc color = {
        .descriptor = gpu_swapchain_color_attachment(renderer->gpu, swapchain_index),
        .clearValue = {
            .color = {
                .f = {
                    .x = 0.025f,
                    .y = 0.035f,
                    .z = 0.044f,
                    .w = 1.0f
                }
            }
        },
        .loadOp = NriLoadOp_CLEAR,
        .storeOp = NriStoreOp_STORE
    };

    const NriRenderingDesc rendering = {
        .colors = &color,
        .colorNum = 1
    };

    renderer->gpu->core.CmdBeginRendering(command_buffer, &rendering);
    renderer->gpu->core.CmdSetPipelineLayout(command_buffer, NriBindPoint_GRAPHICS, renderer->present_layout);
    renderer->gpu->core.CmdSetPipeline(command_buffer, renderer->present_pipeline);

    const NriSetDescriptorSetDesc set = {
        .setIndex = 0,
        .descriptorSet = renderer->present_set,
        .bindPoint = NriBindPoint_GRAPHICS
    };

    renderer->gpu->core.CmdSetDescriptorSet(command_buffer, &set);
    set_fullscreen_view(renderer, command_buffer);
    renderer->gpu->core.CmdDraw(command_buffer, &(NriDrawDesc){
        .vertexNum = 3,
        .instanceNum = 1
    });
    renderer->gpu->core.CmdEndRendering(command_buffer);
}

static bool set_debug_view(RENDERER *renderer, const char *define) {
    if (!compile_present_shader(define)) return false;

    NriPipeline *pipeline = NULL;

    if (!create_present_pipeline(renderer, "build/shaders/present.runtime.ps.spv", &pipeline)) return false;

    if (renderer->gpu->core.QueueWaitIdle(renderer->gpu->graphics_queue) != NriResult_SUCCESS) {
        renderer->gpu->core.DestroyPipeline(pipeline);

        return false;
    }

    if (renderer->present_pipeline) renderer->gpu->core.DestroyPipeline(renderer->present_pipeline);
    renderer->present_pipeline = pipeline;

    return true;
}

bool renderer_init(RENDERER *renderer, GPU *gpu) {
    if (!renderer || !gpu || !gpu->device) return false;
    memset(renderer, 0, sizeof(*renderer));
    renderer->gpu = gpu;
    renderer->camera =
        (CAMERA){
            .position = {0.0f, 1.0f, 5.0f},
            .forward = {0.0f, 0.0f, -1.0f},
            .up = {0.0f, 1.0f, 0.0f},
            .fov_y = 62.0f,
            .near_plane = NEAR_PLANE,
            .far_plane = FAR_PLANE
        };

    renderer->previous_camera = renderer->camera;
    renderer->previous_view_projection = mat4_identity();

    if (!create_pipeline_layouts(renderer) || !create_descriptor_pool(renderer) || !create_frame_buffer(renderer) || !create_radiance_constant_buffers(renderer) ||
        !create_radiance_scene_fallbacks(renderer) || !create_pipelines(renderer) || !create_surface_cache(renderer) ||
        !create_radiance_surface_cache(renderer) || !update_radiance_direct_cache_descriptors(renderer) ||
        !create_size_dependent_resources(renderer, gpu->swapchain_width, gpu->swapchain_height)) {
        renderer_deinit(renderer);

        return false;
    }

    return true;
}

void renderer_deinit(RENDERER *renderer) {
    if (!renderer) return;

    if (renderer->gpu && renderer->gpu->device) {
        if (renderer->gpu->graphics_queue) renderer->gpu->core.QueueWaitIdle(renderer->gpu->graphics_queue);

        NriPipeline *pipelines[] = {
            renderer->gbuffer_pipeline,
            renderer->present_pipeline,
            renderer->hzb_pipeline,
            renderer->screen_trace_pipeline,
            renderer->trace_reset_pipeline,
            renderer->trace_compact_pipeline,
            renderer->trace_args_pipeline,
            renderer->sdf_trace_pipeline,
            renderer->direct_radiance_pipeline,
            renderer->surface_cache_pipeline,
            renderer->screen_probes_pipeline,
            renderer->emissive_pipeline
        };

        for (uint32_t i = 0; i < sizeof(pipelines) / sizeof(pipelines[0]); ++i) {
            if (pipelines[i]) renderer->gpu->core.DestroyPipeline(pipelines[i]);
        }

        destroy_scene_resources(renderer);
        destroy_radiance_scene_fallbacks(renderer);
        destroy_trace_queue(renderer);
        destroy_trace_buffer(renderer, &renderer->trace_hits);
        destroy_surface_cache(renderer);
        destroy_radiance_surface_cache(renderer);
        destroy_compute_texture(renderer, &renderer->direct_radiance);
        destroy_compute_texture(renderer, &renderer->screen_probe_radiance);
        destroy_compute_texture(renderer, &renderer->screen_probes);
        destroy_screen_trace(renderer);
        destroy_hzb(renderer);
        destroy_gbuffer(renderer);
        destroy_radiance_constant_buffers(renderer);

        if (renderer->frame_srv) renderer->gpu->core.DestroyDescriptor(renderer->frame_srv);

        if (renderer->frame_buffer) gpu_destroy_buffer(renderer->gpu, renderer->frame_buffer);

        if (renderer->descriptor_pool) renderer->gpu->core.DestroyDescriptorPool(renderer->descriptor_pool);

        if (renderer->gbuffer_layout) renderer->gpu->core.DestroyPipelineLayout(renderer->gbuffer_layout);

        if (renderer->present_layout) renderer->gpu->core.DestroyPipelineLayout(renderer->present_layout);

        if (renderer->hzb_layout) renderer->gpu->core.DestroyPipelineLayout(renderer->hzb_layout);

        if (renderer->trace_layout) renderer->gpu->core.DestroyPipelineLayout(renderer->trace_layout);

        if (renderer->radiance_scene_layout) renderer->gpu->core.DestroyPipelineLayout(renderer->radiance_scene_layout);

        if (renderer->radiance_direct_layout) renderer->gpu->core.DestroyPipelineLayout(renderer->radiance_direct_layout);

        if (renderer->emissive_layout) renderer->gpu->core.DestroyPipelineLayout(renderer->emissive_layout);
    }

    memset(renderer, 0, sizeof(*renderer));
}

bool renderer_set_scene(RENDERER *renderer, SCENE *scene) {
    if (!renderer || !renderer->gpu || !renderer->gpu->device || !scene) return false;

    if (renderer->vertex_buffer && renderer->gpu->core.QueueWaitIdle(renderer->gpu->graphics_queue) != NriResult_SUCCESS) return false;
    renderer->scene = NULL;
    destroy_scene_resources(renderer);

    if (!create_scene_resources(renderer, scene)) return false;
    renderer->scene = scene;

    for (uint32_t i = 0; i < scene->object_count; ++i) {
        OBJECT *object = &scene->objects[i];

        if (object->type != MODEL) continue;

        struct MODEL *model = object->data;

        if (!model || !model->geometry) continue;
        renderer->camera.target = model->geometry->bounds.center;
        renderer->scene->radius = fmaxf(model->geometry->bounds.extents.x, fmaxf(model->geometry->bounds.extents.y, model->geometry->bounds.extents.z));

        if (renderer->scene->radius < 1.0f) renderer->scene->radius = 1.0f;
        renderer->camera.distance = renderer->scene->radius * 2.15f;

        break;
    }

    renderer->camera.yaw = 1.57079632679f;
    renderer->camera.pitch = 0.0f;
    update_orbit_camera(renderer);
    renderer->previous_camera = renderer->camera;
    renderer->previous_view_projection = mat4_identity();
    renderer->has_previous_frame = false;

    if (!clear_surface_cache(renderer) || !clear_radiance_surface_cache(renderer)) return false;
    update_gbuffer_descriptors(renderer);
    update_trace_descriptors(renderer);

    return true;
}

void renderer_event(RENDERER *renderer, const SDL_Event *event) {
    if (!renderer || !event) return;

    switch (event->type) {
        case SDL_EVENT_MOUSE_BUTTON_DOWN:
            if (event->button.button == SDL_BUTTON_LEFT) renderer->camera.dragging = true;

            break;
        case SDL_EVENT_MOUSE_BUTTON_UP:
            if (event->button.button == SDL_BUTTON_LEFT) renderer->camera.dragging = false;

            break;
        case SDL_EVENT_MOUSE_MOTION:
            if (renderer->camera.dragging) {
                renderer->camera.yaw += event->motion.xrel * 0.0075f;
                renderer->camera.pitch += event->motion.yrel * 0.0075f;
                renderer->camera.pitch = fmaxf(-1.45f, fminf(1.45f, renderer->camera.pitch));
            }

            break;
        case SDL_EVENT_MOUSE_WHEEL:
            renderer->camera.distance -= event->wheel.y * renderer->camera.distance * 0.08f;
            renderer->camera.distance = fmaxf(renderer->scene->radius * 0.05f, fminf(renderer->scene->radius * 20.0f, renderer->camera.distance));

            break;
        case SDL_EVENT_KEY_DOWN:
            if (event->key.repeat) break;

            switch (event->key.key) {
                case SDLK_0:
                    set_debug_view(renderer, "ALBEDO");

                    break;
                case SDLK_1:
                    set_debug_view(renderer, "NORMALS");

                    break;
                case SDLK_2:
                    set_debug_view(renderer, "DEPTH");

                    break;
                case SDLK_3:
                    set_debug_view(renderer, "ROUGHNESS");

                    break;
                case SDLK_4:
                    set_debug_view(renderer, "VELOCITY");

                    break;
                case SDLK_5:
                    set_debug_view(renderer, "OBJECT_ID");

                    break;
                case SDLK_6:
                    set_debug_view(renderer, "HZB");

                    break;
                case SDLK_7:
                    set_debug_view(renderer, "SCREEN_TRACE");

                    break;
                case SDLK_8:
                    set_debug_view(renderer, "DIRECT_RADIANCE");

                    break;
                case SDLK_9:
                    set_debug_view(renderer, "SCREEN_PROBES");

                    break;
                case SDLK_F1:
                    set_debug_view(renderer, "FINAL_GI");

                    break;
                default:
                    break;
            }

            break;
        default:
            break;
    }
}

bool renderer_frame(RENDERER *renderer) {
    if (!renderer || !renderer->gpu || !renderer->gpu->device || !renderer->scene) return false;

    if (!update_scene_objects(renderer) || !refresh_emissive_sampling(renderer)) return false;
    update_orbit_camera(renderer);

    NriCommandBuffer *command_buffer = NULL;
    NriTexture *swapchain_texture = NULL;
    uint32_t swapchain_index = 0;

    if (!gpu_begin_frame(renderer->gpu, &command_buffer, &swapchain_texture, &swapchain_index)) return false;

    if (!command_buffer || !swapchain_texture) return true;

    if (renderer->width != renderer->gpu->swapchain_width || renderer->height != renderer->gpu->swapchain_height) {
        if (renderer->gpu->core.QueueWaitIdle(renderer->gpu->graphics_queue) != NriResult_SUCCESS ||
            !create_size_dependent_resources(renderer, renderer->gpu->swapchain_width, renderer->gpu->swapchain_height))
            return false;
    }

    MAT4 view_projection;
    FRAME_CONSTANTS frame;

    if (!make_frame_constants(renderer, &view_projection, &frame)) return false;

    renderer->gpu->core.CmdSetDescriptorPool(command_buffer, renderer->descriptor_pool);

    if (!stream_dynamic_data(renderer, command_buffer, &frame)) return false;

    transition_gbuffer_for_render(renderer, command_buffer);
    record_gbuffer_pass(renderer, command_buffer);
    transition_depth_for_hzb(renderer, command_buffer);
    build_hzb(renderer, command_buffer);
    transition_gbuffer_for_read(renderer, command_buffer);
    build_direct_radiance(renderer, command_buffer);
    build_screen_trace(renderer, command_buffer);
    build_miss_queue(renderer, command_buffer);
    build_sdf_trace(renderer, command_buffer);
    finish_screen_trace(renderer, command_buffer);
    build_screen_probes(renderer, command_buffer);
    record_present_pass(renderer, command_buffer, swapchain_index);

    const bool frame_finished = gpu_end_frame(renderer->gpu, command_buffer, swapchain_index);

    renderer->gpu->streamer_api.EndStreamerFrame(renderer->gpu->streamer);

    if (!frame_finished) return false;

    renderer->previous_camera = renderer->camera;
    renderer->previous_view_projection = view_projection;
    renderer->has_previous_frame = true;
    renderer->frame_index = renderer->gpu->frame_index;

    return true;
}
