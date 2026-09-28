#include "NRIDescs.h"
#include "game.h"
#include "gpu.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct FRAME_CONSTATS {
    MAT4 view_projection;
    MAT4 previous_view_projection;

    float camera_position[4];
    float resolution[4];
} FRAME_CONSTATS;

static MAT4 mat4_identity(void) {

    return (MAT4){
        .m[0] = 1.0f,
        .m[4] = 1.0f,
        .m[10] = 1.0f,
        .m[15] = 1.0f,
    };
}

static MAT4 mat4_mul(MAT4 a, MAT4 b) {

    MAT4 dest;

    dest.m[0] = a.m[0] * b.m[0] + a.m[4] * b.m[1] + a.m[8] * b.m[2] + a.m[12] * b.m[3];
    dest.m[1] = a.m[1] * b.m[0] + a.m[5] * b.m[1] + a.m[9] * b.m[2] + a.m[13] * b.m[3];
    dest.m[2] = a.m[2] * b.m[0] + a.m[6] * b.m[1] + a.m[10] * b.m[2] + a.m[14] * b.m[3];
    dest.m[3] = a.m[3] * b.m[0] + a.m[7] * b.m[1] + a.m[11] * b.m[2] + a.m[15] * b.m[3];

    dest.m[4] = a.m[0] * b.m[4] + a.m[4] * b.m[5] + a.m[8] * b.m[6] + a.m[12] * b.m[7];
    dest.m[5] = a.m[1] * b.m[4] + a.m[5] * b.m[5] + a.m[9] * b.m[6] + a.m[13] * b.m[7];
    dest.m[6] = a.m[2] * b.m[4] + a.m[6] * b.m[5] + a.m[10] * b.m[6] + a.m[14] * b.m[7];
    dest.m[7] = a.m[3] * b.m[4] + a.m[7] * b.m[5] + a.m[11] * b.m[6] + a.m[15] * b.m[7];

    dest.m[8] = a.m[0] * b.m[8] + a.m[4] * b.m[9] + a.m[8] * b.m[10] + a.m[12] * b.m[11];
    dest.m[9] = a.m[1] * b.m[8] + a.m[5] * b.m[9] + a.m[9] * b.m[10] + a.m[13] * b.m[11];
    dest.m[10] = a.m[2] * b.m[8] + a.m[6] * b.m[9] + a.m[10] * b.m[10] + a.m[14] * b.m[11];
    dest.m[11] = a.m[3] * b.m[8] + a.m[7] * b.m[9] + a.m[11] * b.m[10] + a.m[15] * b.m[11];

    dest.m[12] = a.m[0] * b.m[12] + a.m[4] * b.m[13] + a.m[8] * b.m[14] + a.m[12] * b.m[15];
    dest.m[13] = a.m[1] * b.m[12] + a.m[5] * b.m[13] + a.m[9] * b.m[14] + a.m[13] * b.m[15];
    dest.m[14] = a.m[2] * b.m[12] + a.m[6] * b.m[13] + a.m[10] * b.m[14] + a.m[14] * b.m[15];
    dest.m[15] = a.m[3] * b.m[12] + a.m[7] * b.m[13] + a.m[11] * b.m[14] + a.m[15] * b.m[15];

    return dest;
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
    r.m[2] = (xz - wy) * t.scale.z;

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
    r.m[14] = (pos.x * forward.x + pos.y * forward.y + pos.z * forward.z);

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
    r.m[14] = (near_p * far_p) * inv_depth;

    return r;
}


static bool load_shader(const char *path, void **data, size_t *size) {

    FILE *file = fopen(path, "rb");

    if (!file) {
        SDL_Log("Could not open shader file: %s", path);

        return false;
    }

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

    *data = bytes;
    *size = (size_t)length;

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

    if (!gpu_create_texture(renderer->gpu, &desc, NriMemoryLocation_DEVICE, &target->texture)) {
        return false;
    }

    const NriTextureViewDesc attachment = {
        .texture = target->texture,
        .type = attachment_type,
        .format = format,
        .mipNum = 1,
        .layerNum = 1,
        .sliceNum = 1,
        .planes = planes
    };

    if (renderer->gpu->core.CreateTextureView(&attachment, &target->attachment) != NriResult_SUCCESS) {
        return false;
    }

    const NriTextureViewDesc srv = {
        .texture = target->texture,
        .type = NriTextureView_TEXTURE,
        .format = format,
        .mipNum = 1,
        .layerNum = 1,
        .sliceNum = 1,
        .planes = planes
    };

    if (renderer->gpu->core.CreateTextureView(&srv, &target->srv) != NriResult_SUCCESS) {
        return false;
    }

    target->state = (NriAccessLayoutStage){
        .access = NriAccessBits_NONE,
        .layout = NriLayout_UNDEFINED,
        .stages = NriStageBits_NONE
    };

    return true;
}

static void destroy_render_texture(RENDERER *renderer, RENDER_TEXTURE *target) {
    if (target->attachment) {
        renderer->gpu->core.DestroyDescriptor(target->attachment);
    }

    if (target->srv) {
        renderer->gpu->core.DestroyDescriptor(target->srv);
    }

    if (target->texture) {
        gpu_destroy_texture(renderer->gpu, target->texture);
    }

    memset(target, 0, sizeof(*target));
}

static void destroy_gbuffer(RENDERER *renderer) {

    destroy_render_texture(renderer, &renderer->depth);
    destroy_render_texture(renderer, &renderer->normal_roughness);
    destroy_render_texture(renderer, &renderer->albedo_metallic);
    destroy_render_texture(renderer, &renderer->velocity);
    destroy_render_texture(renderer, &renderer->object_id);

    renderer->width = 0;
    renderer->height = 0;
}

static void update_present_descriptor(RENDERER *renderer) {

    const NriDescriptor *descriptors[] = {renderer->albedo_metallic.srv};

    const NriUpdateDescriptorRangeDesc update = {
        .descriptorSet = renderer->present_set,
        .rangeIndex = 0,
        .baseDescriptor = 0,
        .descriptors = descriptors,
        .descriptorNum = 1,
    };

    renderer->gpu->core.UpdateDescriptorRanges(&update, 1);
}

static bool create_gbuffer(RENDERER *renderer, uint32_t width, uint32_t height) {

    destroy_gbuffer(renderer);

    renderer->width = width;
    renderer->height = height;

    if (!create_render_texture(
            renderer,
            &renderer->depth,
            NriFormat_D32_SFLOAT,
            NriTextureUsageBits_DEPTH_STENCIL_ATTACHMENT | NriTextureUsageBits_SHADER_RESOURCE,
            NriTextureView_DEPTH_STENCIL_ATTACHMENT,
            NriPlaneBits_DEPTH
        )) {
        return false;
    }

    if (!create_render_texture(
            renderer,
            &renderer->normal_roughness,
            NriFormat_RGBA16_SFLOAT,
            NriTextureUsageBits_COLOR_ATTACHMENT | NriTextureUsageBits_SHADER_RESOURCE,
            NriTextureView_COLOR_ATTACHMENT,
            NriPlaneBits_COLOR
        )) {
        return false;
    }

    if (!create_render_texture(
            renderer,
            &renderer->albedo_metallic,
            NriFormat_RGBA8_UNORM,
            NriTextureUsageBits_COLOR_ATTACHMENT | NriTextureUsageBits_SHADER_RESOURCE,
            NriTextureView_COLOR_ATTACHMENT,
            NriPlaneBits_COLOR
        )) {
        return false;
    }

    if (!create_render_texture(
            renderer,
            &renderer->velocity,
            NriFormat_RG16_SFLOAT,
            NriTextureUsageBits_COLOR_ATTACHMENT | NriTextureUsageBits_SHADER_RESOURCE,
            NriTextureView_COLOR_ATTACHMENT,
            NriPlaneBits_COLOR
        )) {
        return false;
    }

    if (!create_render_texture(
            renderer,
            &renderer->object_id,
            NriFormat_R32_UINT,
            NriTextureUsageBits_COLOR_ATTACHMENT | NriTextureUsageBits_SHADER_RESOURCE,
            NriTextureView_COLOR_ATTACHMENT,
            NriPlaneBits_COLOR
        )) {
        return false;
    }

    if (renderer->present_set) {
        update_present_descriptor(renderer);
    }

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
            .near_plane = 0.05f,
            .far_plane = 1000.0f
        };

    return true;
}

void renderer_deinit(RENDERER *renderer) {
    if (!renderer) return;

    memset(renderer, 0, sizeof(*renderer));
}

bool renderer_set_scene(RENDERER *renderer, SCENE *scene) {
    if (!renderer || !renderer->gpu || !scene) return false;

    renderer->scene = scene;

    return true;
}

void renderer_event(RENDERER *renderer, const SDL_Event *event) {
    (void)renderer;
    (void)event;
}

bool renderer_frame(RENDERER *renderer) {
    if (!renderer || !renderer->gpu || !renderer->gpu->device || !renderer->scene) return false;

    NriCommandBuffer *command_buffer = NULL;
    NriTexture *swapchain_texture = NULL;
    uint32_t swapchain_index = 0;

    if (!gpu_begin_frame(renderer->gpu, &command_buffer, &swapchain_texture, &swapchain_index)) return false;

    if (!command_buffer || !swapchain_texture) return true;

    const NriColor32f clear_color = {
        .x = 0.025f,
        .y = 0.035f,
        .z = 0.055f,
        .w = 1.0f
    };

    if (!gpu_clear_frame(renderer->gpu, command_buffer, swapchain_index, clear_color) || !gpu_end_frame(renderer->gpu, command_buffer, swapchain_index)) {
        return false;
    }

    renderer->frame_index = renderer->gpu->frame_index;

    return true;
}
