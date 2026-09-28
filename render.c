#include "NRIDescs.h"
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

typedef struct FRAME_CONSTANTS {
    MAT4 view_projection;
    MAT4 previous_view_projection;

    float camera_position[4];
    float resolution[4];
} FRAME_CONSTANTS;

_Static_assert(sizeof(GLTF_VERTEX) == 36u, "GLTF_VERTEX GPU layout changed");
_Static_assert(sizeof(GPU_OBJECT) == 224u, "GPU_OBJECT GPU layout changed");
_Static_assert(sizeof(GPU_MATERIAL) == 48u, "GPU_MATERIAL GPU layout changed");
_Static_assert(sizeof(FRAME_CONSTANTS) == 160u, "FRAME_CONSTANTS GPU layout changed");

static MAT4 mat4_identity(void) {

    return (MAT4){
        .m[0] = 1.0f,
        .m[4] = 1.0f,
        .m[10] = 1.0f,
        .m[15] = 1.0f,
    };
}

static MAT4 mat4_mul(MAT4 a, MAT4 b) {

    MAT4 r = {{0}};

    for (uint32_t row = 0; row < 4; ++row) {

        for (uint32_t column = 0; column < 4; ++column) {

            for (uint32_t k = 0; k < 4; ++k) {

                r.m[row * 4 + column] += a.m[row * 4 + k] * b.m[k * 4 + column];
            }
        }
    }

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
        gpu_destroy_texture(renderer->gpu, target->texture);
        target->texture = NULL;

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
        renderer->gpu->core.DestroyDescriptor(target->attachment);
        target->attachment = NULL;

        gpu_destroy_texture(renderer->gpu, target->texture);
        target->texture = NULL;

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
    if (!renderer->present_set || !renderer->albedo_metallic.srv) return;

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
    if (!width || !height) return false;

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
        destroy_gbuffer(renderer);

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
        destroy_gbuffer(renderer);

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
        destroy_gbuffer(renderer);

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
        destroy_gbuffer(renderer);

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
        destroy_gbuffer(renderer);

        return false;
    }

    if (renderer->present_set) {
        update_present_descriptor(renderer);
    }

    return true;
}

static void destroy_scene_resources(RENDERER *renderer) {

    if (renderer->object_srv) {
        renderer->gpu->core.DestroyDescriptor(renderer->object_srv);
        renderer->object_srv = NULL;
    }

    if (renderer->material_srv) {
        renderer->gpu->core.DestroyDescriptor(renderer->material_srv);
        renderer->material_srv = NULL;
    }

    if (renderer->vertex_buffer) {
        gpu_destroy_buffer(renderer->gpu, renderer->vertex_buffer);
        renderer->vertex_buffer = NULL;
    }

    if (renderer->material_buffer) {
        gpu_destroy_buffer(renderer->gpu, renderer->material_buffer);
        renderer->material_buffer = NULL;
    }

    if (renderer->object_buffer) {
        gpu_destroy_buffer(renderer->gpu, renderer->object_buffer);
        renderer->object_buffer = NULL;
    }

    free(renderer->cpu_objects);
    renderer->cpu_objects = NULL;
    renderer->gpu_object_count = 0;
    renderer->vertex_count = 0;
    renderer->material_count = 0;
    renderer->object_state = (NriAccessStage){0};
}

static bool create_frame_buffer(RENDERER *renderer) {

    const NriBufferDesc buffer_desc = {
        .size = FRAME_CONSTANTS_BUFFER_SIZE,
        .usage = NriBufferUsageBits_CONSTANT
    };

    if (!gpu_create_buffer(renderer->gpu, &buffer_desc, NriMemoryLocation_DEVICE, &renderer->frame_buffer)) {
        return false;
    }

    const NriBufferViewDesc view_desc = {
        .buffer = renderer->frame_buffer,
        .type = NriBufferView_CONSTANT_BUFFER,
        .offset = 0,
        .size = FRAME_CONSTANTS_BUFFER_SIZE
    };

    if (renderer->gpu->core.CreateBufferView(&view_desc, &renderer->frame_srv) != NriResult_SUCCESS) {
        gpu_destroy_buffer(renderer->gpu, renderer->frame_buffer);
        renderer->frame_buffer = NULL;

        return false;
    }

    renderer->frame_state = (NriAccessStage){0};

    return true;
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

    if (renderer->gpu->core.CreatePipelineLayout(renderer->gpu->device, &gbuffer_layout, &renderer->gbuffer_layout) != NriResult_SUCCESS) {
        return false;
    }

    const NriDescriptorRangeDesc present_range = {
        .baseRegisterIndex = 0,
        .descriptorNum = 1,
        .descriptorType = NriDescriptorType_TEXTURE,
        .shaderStages = NriStageBits_FRAGMENT_SHADER
    };

    const NriDescriptorSetDesc present_set = {
        .registerSpace = 1,
        .ranges = &present_range,
        .rangeNum = 1
    };

    const NriPipelineLayoutDesc present_layout = {
        .rootRegisterSpace = 0,
        .descriptorSets = &present_set,
        .descriptorSetNum = 1,
        .shaderStages = NriStageBits_VERTEX_SHADER | NriStageBits_FRAGMENT_SHADER,
        .flags = NriPipelineLayoutBits_IGNORE_GLOBAL_SPIRV_OFFSETS
    };

    return renderer->gpu->core.CreatePipelineLayout(renderer->gpu->device, &present_layout, &renderer->present_layout) == NriResult_SUCCESS;
}

static bool create_descriptor_pool(RENDERER *renderer) {

    const NriDescriptorPoolDesc pool_desc = {
        .descriptorSetMaxNum = 2,
        .constantBufferMaxNum = 1,
        .textureMaxNum = 1,
        .structuredBufferMaxNum = 2
    };

    if (renderer->gpu->core.CreateDescriptorPool(renderer->gpu->device, &pool_desc, &renderer->descriptor_pool) != NriResult_SUCCESS) {
        return false;
    }

    if (renderer->gpu->core.AllocateDescriptorSets(renderer->descriptor_pool, renderer->gbuffer_layout, 0, &renderer->gbuffer_set, 1, 0) != NriResult_SUCCESS) {
        return false;
    }

    if (renderer->gpu->core.AllocateDescriptorSets(renderer->descriptor_pool, renderer->present_layout, 0, &renderer->present_set, 1, 0) != NriResult_SUCCESS) {
        return false;
    }

    return true;
}

static bool create_pipelines(RENDERER *renderer) {

    void *gbuffer_vs = NULL;
    void *gbuffer_ps = NULL;
    void *present_vs = NULL;
    void *present_ps = NULL;

    size_t gbuffer_vs_size = 0;
    size_t gbuffer_ps_size = 0;
    size_t present_vs_size = 0;
    size_t present_ps_size = 0;

    bool success = false;

    if (!load_shader("build/shaders/gbuffer.vs.spv", &gbuffer_vs, &gbuffer_vs_size)) {
        goto cleanup;
    }

    if (!load_shader("build/shaders/gbuffer.ps.spv", &gbuffer_ps, &gbuffer_ps_size)) {
        goto cleanup;
    }

    if (!load_shader("build/shaders/present.vs.spv", &present_vs, &present_vs_size)) {
        goto cleanup;
    }

    if (!load_shader("build/shaders/present.ps.spv", &present_ps, &present_ps_size)) {
        goto cleanup;
    }

    const NriShaderDesc gbuffer_shaders[] = {
        {
            .stage = NriStageBits_VERTEX_SHADER,
            .bytecode = gbuffer_vs,
            .size = (uint64_t)gbuffer_vs_size,
            .entryPointName = "main"
        },
        {
            .stage = NriStageBits_FRAGMENT_SHADER,
            .bytecode = gbuffer_ps,
            .size = (uint64_t)gbuffer_ps_size,
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
            .offset = (uint32_t)offsetof(GLTF_VERTEX, position),
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
            .offset = (uint32_t)offsetof(GLTF_VERTEX, normal),
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
            .offset = (uint32_t)offsetof(GLTF_VERTEX, u),
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
            .offset = (uint32_t)offsetof(GLTF_VERTEX, material),
            .format = NriFormat_R32_UINT,
            .streamIndex = 0
        }
    };

    const NriVertexStreamDesc stream = {
        .bindingSlot = 0,
        .stepRate = NriVertexStreamStepRate_PER_VERTEX,
        .stride = (uint16_t)sizeof(GLTF_VERTEX)
    };

    const NriVertexInputDesc vertex_input = {
        .attributes = attributes,
        .attributeNum = 4,
        .streams = &stream,
        .streamNum = 1
    };

    const NriColorAttachmentDesc gbuffer_colors[] = {
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
        }
    };

    const NriGraphicsPipelineDesc gbuffer_pipeline = {
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
            .colors = gbuffer_colors,
            .colorNum = 4,
            .depth = {
                .compareOp = NriCompareOp_GREATER,
                .write = true
            },
            .depthStencilFormat = NriFormat_D32_SFLOAT
        },
        .shaders = gbuffer_shaders,
        .shaderNum = 2
    };

    if (renderer->gpu->core.CreateGraphicsPipeline(renderer->gpu->device, &gbuffer_pipeline, &renderer->gbuffer_pipeline) != NriResult_SUCCESS) {
        goto cleanup;
    }

    const NriShaderDesc present_shaders[] = {
        {
            .stage = NriStageBits_VERTEX_SHADER,
            .bytecode = present_vs,
            .size = (uint64_t)present_vs_size,
            .entryPointName = "main"
        },
        {
            .stage = NriStageBits_FRAGMENT_SHADER,
            .bytecode = present_ps,
            .size = (uint64_t)present_ps_size,
            .entryPointName = "main"
        }
    };

    const NriColorAttachmentDesc present_color = {
        .format = renderer->gpu->swapchain_format,
        .colorWriteMask = NriColorWriteBits_RGBA
    };

    const NriGraphicsPipelineDesc present_pipeline = {
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
            .colors = &present_color,
            .colorNum = 1
        },
        .shaders = present_shaders,
        .shaderNum = 2
    };

    if (renderer->gpu->core.CreateGraphicsPipeline(renderer->gpu->device, &present_pipeline, &renderer->present_pipeline) != NriResult_SUCCESS) {
        goto cleanup;
    }

    success = true;

cleanup:
    free(gbuffer_vs);
    free(gbuffer_ps);
    free(present_vs);
    free(present_ps);

    return success;
}

static void update_gbuffer_descriptors(RENDERER *renderer) {

    const NriDescriptor *objects[] = {renderer->object_srv};

    const NriDescriptor *materials[] = {renderer->material_srv};

    const NriDescriptor *frame[] = {renderer->frame_srv};

    const NriUpdateDescriptorRangeDesc updates[] = {
        {
            .descriptorSet = renderer->gbuffer_set,
            .rangeIndex = 0,
            .baseDescriptor = 0,
            .descriptors = objects,
            .descriptorNum = 1
        },
        {
            .descriptorSet = renderer->gbuffer_set,
            .rangeIndex = 1,
            .baseDescriptor = 0,
            .descriptors = materials,
            .descriptorNum = 1
        },
        {
            .descriptorSet = renderer->gbuffer_set,
            .rangeIndex = 2,
            .baseDescriptor = 0,
            .descriptors = frame,
            .descriptorNum = 1
        }
    };

    renderer->gpu->core.UpdateDescriptorRanges(updates, 3);
}

static bool create_scene_resources(RENDERER *renderer, SCENE *scene) {

    uint64_t total_vertices = 0;
    uint64_t total_materials = 0;
    uint32_t model_count = 0;

    for (uint32_t i = 0; i < scene->object_count; ++i) {
        OBJECT *object = &scene->objects[i];

        if (object->type != MODEL) continue;

        struct MODEL *model = object->data;

        if (!model || !model->visual || !model->visual->vertices || !model->visual->materials || !model->visual->vertex_count || !model->visual->material_count) {
            SDL_Log("Invalid MODEL object at scene index %u", i);

            return false;
        }

        total_vertices += model->visual->vertex_count;
        total_materials += model->visual->material_count;
        ++model_count;

        if (total_vertices > UINT32_MAX || total_materials > UINT32_MAX) {
            SDL_Log("Stage 2 scene exceeds 32-bit renderer limits");

            return false;
        }
    }

    if (!model_count) {
        SDL_Log("Stage 2 scene contains no MODEL objects");

        return false;
    }

    GLTF_VERTEX *vertices = malloc((size_t)total_vertices * sizeof(*vertices));
    GPU_MATERIAL *materials = calloc((size_t)total_materials, sizeof(*materials));
    GPU_OBJECT *objects = calloc(model_count, sizeof(*objects));

    if (!vertices || !materials || !objects) {
        free(vertices);
        free(materials);
        free(objects);

        return false;
    }

    uint32_t vertex_offset = 0;
    uint32_t material_offset = 0;
    uint32_t object_index = 0;

    for (uint32_t scene_index = 0; scene_index < scene->object_count; ++scene_index) {
        OBJECT *object = &scene->objects[scene_index];

        if (object->type != MODEL) continue;

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

        GPU_OBJECT *gpu_object = &objects[object_index];

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

        ++object_index;
    }

    renderer->cpu_objects = objects;
    renderer->gpu_object_count = model_count;
    renderer->vertex_count = (uint32_t)total_vertices;
    renderer->material_count = (uint32_t)total_materials;

    const NriBufferDesc vertex_desc = {
        .size = (uint64_t)renderer->vertex_count * sizeof(GLTF_VERTEX),
        .usage = NriBufferUsageBits_VERTEX
    };

    if (!gpu_create_buffer(renderer->gpu, &vertex_desc, NriMemoryLocation_DEVICE, &renderer->vertex_buffer)) {
        goto fail;
    }

    const NriBufferDesc material_desc = {
        .size = (uint64_t)renderer->material_count * sizeof(GPU_MATERIAL),
        .structureStride = (uint32_t)sizeof(GPU_MATERIAL),
        .usage = NriBufferUsageBits_SHADER_RESOURCE
    };

    if (!gpu_create_buffer(renderer->gpu, &material_desc, NriMemoryLocation_DEVICE, &renderer->material_buffer)) {
        goto fail;
    }

    const NriBufferDesc object_desc = {
        .size = (uint64_t)renderer->gpu_object_count * sizeof(GPU_OBJECT),
        .structureStride = (uint32_t)sizeof(GPU_OBJECT),
        .usage = NriBufferUsageBits_SHADER_RESOURCE
    };

    if (!gpu_create_buffer(renderer->gpu, &object_desc, NriMemoryLocation_DEVICE, &renderer->object_buffer)) {
        goto fail;
    }

    const NriAccessStage vertex_state = {
        .access = NriAccessBits_VERTEX_BUFFER,
        .stages = NriStageBits_VERTEX_SHADER
    };

    if (!gpu_upload_buffer(renderer->gpu, renderer->vertex_buffer, vertices, vertex_state)) {
        goto fail;
    }

    const NriAccessStage material_state = {
        .access = NriAccessBits_SHADER_RESOURCE,
        .stages = NriStageBits_FRAGMENT_SHADER
    };

    if (!gpu_upload_buffer(renderer->gpu, renderer->material_buffer, materials, material_state)) {
        goto fail;
    }

    const NriBufferViewDesc object_view = {
        .buffer = renderer->object_buffer,
        .type = NriBufferView_STRUCTURED_BUFFER,
        .offset = 0,
        .size = (uint64_t)renderer->gpu_object_count * sizeof(GPU_OBJECT),
        .structureStride = (uint32_t)sizeof(GPU_OBJECT)
    };

    if (renderer->gpu->core.CreateBufferView(&object_view, &renderer->object_srv) != NriResult_SUCCESS) {
        goto fail;
    }

    const NriBufferViewDesc material_view = {
        .buffer = renderer->material_buffer,
        .type = NriBufferView_STRUCTURED_BUFFER,
        .offset = 0,
        .size = (uint64_t)renderer->material_count * sizeof(GPU_MATERIAL),
        .structureStride = (uint32_t)sizeof(GPU_MATERIAL)
    };

    if (renderer->gpu->core.CreateBufferView(&material_view, &renderer->material_srv) != NriResult_SUCCESS) {
        goto fail;
    }

    renderer->object_state = (NriAccessStage){0};

    update_gbuffer_descriptors(renderer);

    free(vertices);
    free(materials);

    return true;

fail:
    free(vertices);
    free(materials);

    destroy_scene_resources(renderer);

    return false;
}

static bool update_scene_objects(RENDERER *renderer) {

    uint32_t gpu_index = 0;

    for (uint32_t scene_index = 0; scene_index < renderer->scene->object_count; ++scene_index) {

        OBJECT *object = &renderer->scene->objects[scene_index];

        if (object->type != MODEL) continue;

        if (gpu_index >= renderer->gpu_object_count) {
            return false;
        }

        struct MODEL *model = object->data;

        if (!model || !model->visual) {
            return false;
        }

        GPU_OBJECT *gpu_object = &renderer->cpu_objects[gpu_index];

        const MAT4 world = mat4_transform(object->transform);

        gpu_object->previous_world = renderer->has_previous_frame ? gpu_object->world : world;
        gpu_object->world = world;
        gpu_object->normal_world = mat4_normal_transform(object->transform);
        gpu_object->object_id = scene_index + 1u;
        gpu_object->revision = object->revision;
        gpu_object->state = (uint32_t)object->state;
        gpu_object->type = (uint32_t)object->type;

        ++gpu_index;
    }

    if (gpu_index != renderer->gpu_object_count) {
        SDL_Log("Model topology changed; call renderer_set_scene() again");

        return false;
    }

    return true;
}

static FRAME_CONSTANTS make_frame_constants(RENDERER *renderer, MAT4 *view_projection) {

    const MAT4 view = mat4_view(renderer->camera);

    const MAT4 projection =
        mat4_reverse_z_projection(renderer->camera.fov_y, (float)renderer->width / (float)renderer->height, renderer->camera.near_plane, renderer->camera.far_plane);

    *view_projection = mat4_mul(view, projection);

    FRAME_CONSTANTS frame = {
        .view_projection = *view_projection,
        .previous_view_projection = renderer->has_previous_frame ? renderer->previous_view_projection : *view_projection,
        .camera_position = {renderer->camera.position.x, renderer->camera.position.y, renderer->camera.position.z, 1.0f},
        .resolution = {(float)renderer->width, (float)renderer->height, 1.0f / (float)renderer->width, 1.0f / (float)renderer->height},
    };

    return frame;
}

static bool stream_dynamic_data(RENDERER *renderer, NriCommandBuffer *command_buffer, const FRAME_CONSTANTS *frame) {

    NriStreamerCopyBatch copy_batch = renderer->gpu->streamer_api.BeginStreamerCopyBatch(renderer->gpu->streamer);

    if (!copy_batch) {
        return false;
    }

    const NriDataSize object_data = {
        .data = renderer->cpu_objects,
        .size = (uint64_t)renderer->gpu_object_count * sizeof(GPU_OBJECT)
    };

    const NriStreamBufferDataDesc object_upload = {
        .dataChunks = &object_data,
        .dataChunkNum = 1,
        .placementAlignment = 16,
        .copyBatch = copy_batch,
        .dstBuffer = renderer->object_buffer,
        .dstOffset = 0
    };

    const NriBufferOffset streamed_objects = renderer->gpu->streamer_api.StreamBufferData(renderer->gpu->streamer, &object_upload);

    if (!streamed_objects.buffer) {
        renderer->gpu->streamer_api.EndStreamerFrame(renderer->gpu->streamer);

        return false;
    }

    const NriDataSize frame_data = {
        .data = frame,
        .size = sizeof(*frame)
    };

    const NriStreamBufferDataDesc frame_upload = {
        .dataChunks = &frame_data,
        .dataChunkNum = 1,
        .placementAlignment = 16,
        .copyBatch = copy_batch,
        .dstBuffer = renderer->frame_buffer,
        .dstOffset = 0
    };

    const NriBufferOffset streamed_frame = renderer->gpu->streamer_api.StreamBufferData(renderer->gpu->streamer, &frame_upload);

    if (!streamed_frame.buffer) {
        renderer->gpu->streamer_api.EndStreamerFrame(renderer->gpu->streamer);
        return false;
    }

    const NriAccessStage copy_state = {
        .access = NriAccessBits_COPY_DESTINATION,
        .stages = NriStageBits_COPY
    };

    const NriBufferBarrierDesc to_copy[] = {
        {
            .buffer = renderer->object_buffer,
            .before = renderer->object_state,
            .after = copy_state
        },
        {
            .buffer = renderer->frame_buffer,
            .before = renderer->frame_state,
            .after = copy_state
        }
    };

    const NriBarrierDesc before_copy = {
        .buffers = to_copy,
        .bufferNum = 2
    };

    renderer->gpu->core.CmdBarrier(command_buffer, &before_copy);

    renderer->gpu->streamer_api.CmdCopyStreamedData(command_buffer, renderer->gpu->streamer, copy_batch);

    const NriAccessStage object_read = {
        .access = NriAccessBits_SHADER_RESOURCE,
        .stages = NriStageBits_VERTEX_SHADER
    };

    const NriAccessStage frame_read = {
        .access = NriAccessBits_CONSTANT_BUFFER,
        .stages = NriStageBits_VERTEX_SHADER
    };

    const NriBufferBarrierDesc to_render[] = {
        {
            .buffer = renderer->object_buffer,
            .before = copy_state,
            .after = object_read
        }, {
            .buffer = renderer->frame_buffer,
            .before = copy_state,
            .after = frame_read
        }
    };

    const NriBarrierDesc after_copy = {
        .buffers = to_render,
        .bufferNum = 2,
    };

    renderer->gpu->core.CmdBarrier(command_buffer, &after_copy);

    renderer->object_state = object_read;
    renderer->frame_state = frame_read;

    return true;
}

static void transition_gbuffer_for_render(RENDERER *renderer, NriCommandBuffer *command_buffer) {

    const NriAccessLayoutStage color_state = {
        .access = NriAccessBits_COLOR_ATTACHMENT_WRITE,
        .layout = NriLayout_COLOR_ATTACHMENT,
        .stages = NriStageBits_COLOR_ATTACHMENT
    };

    const NriAccessLayoutStage depth_state = {
        .access = NriAccessBits_DEPTH_STENCIL_ATTACHMENT_WRITE,
        .layout = NriLayout_DEPTH_STENCIL_ATTACHMENT,
        .stages = NriStageBits_DEPTH_STENCIL_ATTACHMENT
    };

    const NriTextureBarrierDesc barriers[] = {
        {
            .texture = renderer->normal_roughness.texture,
            .before = renderer->normal_roughness.state,
            .after = color_state,
            .mipNum = 1,
            .layerNum = 1,
            .planes = NriPlaneBits_COLOR
        },
        {
            .texture = renderer->albedo_metallic.texture,
            .before = renderer->albedo_metallic.state,
            .after = color_state,
            .mipNum = 1,
            .layerNum = 1,
            .planes = NriPlaneBits_COLOR
        },
        {
            .texture = renderer->velocity.texture,
            .before = renderer->velocity.state,
            .after = color_state,
            .mipNum = 1,
            .layerNum = 1,
            .planes = NriPlaneBits_COLOR
        },
        {
            .texture = renderer->object_id.texture,
            .before = renderer->object_id.state,
            .after = color_state,
            .mipNum = 1,
            .layerNum = 1,
            .planes = NriPlaneBits_COLOR
        },
        {
            .texture = renderer->depth.texture,
            .before = renderer->depth.state,
            .after = depth_state,
            .mipNum = 1,
            .layerNum = 1,
            .planes = NriPlaneBits_DEPTH
        }
    };

    const NriBarrierDesc barrier = {
        .textures = barriers,
        .textureNum = 5
    };

    renderer->gpu->core.CmdBarrier(command_buffer, &barrier);

    renderer->normal_roughness.state = color_state;
    renderer->albedo_metallic.state = color_state;
    renderer->velocity.state = color_state;
    renderer->object_id.state = color_state;
    renderer->depth.state = depth_state;
}

static void transition_albedo_for_present(RENDERER *renderer, NriCommandBuffer *command_buffer) {

    const NriAccessLayoutStage shader_state = {
        .access = NriAccessBits_SHADER_RESOURCE,
        .layout = NriLayout_SHADER_RESOURCE,
        .stages = NriStageBits_FRAGMENT_SHADER
    };

    const NriTextureBarrierDesc texture_barrier = {
        .texture = renderer->albedo_metallic.texture,
        .before = renderer->albedo_metallic.state,
        .after = shader_state,
        .mipNum = 1,
        .layerNum = 1,
        .planes = NriPlaneBits_COLOR
    };

    const NriBarrierDesc barrier = {
        .textures = &texture_barrier,
        .textureNum = 1
    };

    renderer->gpu->core.CmdBarrier(command_buffer, &barrier);

    renderer->albedo_metallic.state = shader_state;
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
                        .x = 0,
                        .y = 0,
                        .z = 0,
                        .w = 0
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
        .colorNum = 4,
        .depth = depth
    };

    renderer->gpu->core.CmdBeginRendering(command_buffer, &rendering);

    renderer->gpu->core.CmdSetPipelineLayout(command_buffer, NriBindPoint_GRAPHICS, renderer->gbuffer_layout);

    renderer->gpu->core.CmdSetPipeline(command_buffer, renderer->gbuffer_pipeline);

    const NriSetDescriptorSetDesc descriptor_set = {
        .setIndex = 0,
        .descriptorSet = renderer->gbuffer_set,
        .bindPoint = NriBindPoint_GRAPHICS
    };

    renderer->gpu->core.CmdSetDescriptorSet(command_buffer, &descriptor_set);

    const NriVertexBufferDesc vertex_buffer = {
        .buffer = renderer->vertex_buffer,
        .offset = 0,
        .stride = (uint32_t)sizeof(GLTF_VERTEX)
    };

    renderer->gpu->core.CmdSetVertexBuffers(command_buffer, 0, &vertex_buffer, 1);

    set_fullscreen_view(renderer, command_buffer);

    for (uint32_t object_index = 0; object_index < renderer->gpu_object_count; ++object_index) {

        const GPU_OBJECT *object = &renderer->cpu_objects[object_index];

        const NriSetRootConstantsDesc root_constants = {
            .rootConstantIndex = 0,
            .data = &object_index,
            .size = sizeof(object_index),
            .offset = 0,
            .bindPoint = NriBindPoint_GRAPHICS
        };

        renderer->gpu->core.CmdSetRootConstants(command_buffer, &root_constants);

        const NriDrawDesc draw = {
            .vertexNum = object->vertex_count,
            .instanceNum = 1,
            .baseVertex = object->first_vertex,
            .baseInstance = 0
        };

        renderer->gpu->core.CmdDraw(command_buffer, &draw);
    }

    renderer->gpu->core.CmdEndRendering(command_buffer);
}

static void record_present_pass(RENDERER *renderer, NriCommandBuffer *command_buffer, uint32_t swapchain_index) {

    const NriAttachmentDesc color = {
        .descriptor = gpu_swapchain_color_attachment(renderer->gpu, swapchain_index),
        .clearValue =
        {
            .color =
            {
                .f =
                {
                    .x = 0.025f,
                    .y = 0.035f,
                    .z = 0.044f,
                    .w = 1.0f,
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

    const NriSetDescriptorSetDesc descriptor_set = {
        .setIndex = 0,
        .descriptorSet = renderer->present_set,
        .bindPoint = NriBindPoint_GRAPHICS
    };

    renderer->gpu->core.CmdSetDescriptorSet(command_buffer, &descriptor_set);

    set_fullscreen_view(renderer, command_buffer);

    const NriDrawDesc draw = {
        .vertexNum = 3,
        .instanceNum = 1
    };

    renderer->gpu->core.CmdDraw(command_buffer, &draw);

    renderer->gpu->core.CmdEndRendering(command_buffer);
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

    renderer->previous_camera = renderer->camera;

    renderer->previous_view_projection = mat4_identity();

    if (!create_pipeline_layouts(renderer) || !create_descriptor_pool(renderer) || !create_frame_buffer(renderer) || !create_pipelines(renderer) ||
        !create_gbuffer(renderer, gpu->swapchain_width, gpu->swapchain_height)) {
        renderer_deinit(renderer);
        return false;
    }

    return true;
}

void renderer_deinit(RENDERER *renderer) {
    if (!renderer) return;

    if (renderer->gpu && renderer->gpu->device) {
        if (renderer->gpu->graphics_queue) {
            renderer->gpu->core.QueueWaitIdle(renderer->gpu->graphics_queue);
        }

        if (renderer->gbuffer_pipeline) {
            renderer->gpu->core.DestroyPipeline(renderer->gbuffer_pipeline);
        }

        if (renderer->present_pipeline) {
            renderer->gpu->core.DestroyPipeline(renderer->present_pipeline);
        }

        if (renderer->descriptor_pool) {
            renderer->gpu->core.DestroyDescriptorPool(renderer->descriptor_pool);
        }

        if (renderer->gbuffer_layout) {
            renderer->gpu->core.DestroyPipelineLayout(renderer->gbuffer_layout);
        }

        if (renderer->present_layout) {
            renderer->gpu->core.DestroyPipelineLayout(renderer->present_layout);
        }

        destroy_scene_resources(renderer);

        if (renderer->frame_srv) {
            renderer->gpu->core.DestroyDescriptor(renderer->frame_srv);
        }

        if (renderer->frame_buffer) {
            renderer->gpu->core.DestroyBuffer(renderer->frame_buffer);
        }

        destroy_gbuffer(renderer);
    } else free(renderer->cpu_objects);

    memset(renderer, 0, sizeof(*renderer));
}

bool renderer_set_scene(RENDERER *renderer, SCENE *scene) {
    if (!renderer || !renderer->gpu || !renderer->gpu->device || !scene) return false;

    if (renderer->vertex_buffer || renderer->material_buffer || renderer->object_buffer) {
        if (renderer->gpu->core.QueueWaitIdle(renderer->gpu->graphics_queue) != NriResult_SUCCESS) {
            return false;
        }
    }

    renderer->scene = NULL;

    destroy_scene_resources(renderer);

    if (!create_scene_resources(renderer, scene)) {
        return false;
    }

    renderer->scene = scene;

    renderer->previous_camera = renderer->camera;

    renderer->previous_view_projection = mat4_identity();

    renderer->has_previous_frame = false;

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

    if (renderer->width != renderer->gpu->swapchain_width || renderer->height != renderer->gpu->swapchain_height) {
        if (!create_gbuffer(renderer, renderer->gpu->swapchain_width, renderer->gpu->swapchain_height)) {
            return false;
        }
    }

    if (!update_scene_objects(renderer)) {
        return false;
    }

    MAT4 view_projection;
    const FRAME_CONSTANTS frame = make_frame_constants(renderer, &view_projection);

    renderer->gpu->core.CmdSetDescriptorPool(command_buffer, renderer->descriptor_pool);

    if (!stream_dynamic_data(renderer, command_buffer, &frame)) {
        return false;
    }

    transition_gbuffer_for_render(renderer, command_buffer);

    record_gbuffer_pass(renderer, command_buffer);

    transition_albedo_for_present(renderer, command_buffer);

    record_present_pass(renderer, command_buffer, swapchain_index);

    const bool frame_finished = gpu_end_frame(renderer->gpu, command_buffer, swapchain_index);

    renderer->gpu->streamer_api.EndStreamerFrame(renderer->gpu->streamer);

    if (!frame_finished) return false;

    renderer->previous_camera = renderer->camera;

    renderer->previous_view_projection = view_projection;

    renderer->has_previous_frame = true;

    renderer->frame_index = renderer->gpu->frame_index;

    // const NriColor32f clear_color = {
    //     .x = 0.025f,
    //     .y = 0.035f,
    //     .z = 0.055f,
    //     .w = 1.0f
    // };
    //
    // if (!gpu_clear_frame(renderer->gpu, command_buffer, swapchain_index, clear_color) || !gpu_end_frame(renderer->gpu, command_buffer, swapchain_index)) {
    //     return false;
    // }

    return true;
}
