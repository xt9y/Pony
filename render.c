#include "game.h"
#include "render_internal.h"

#include <SDL3_image/SDL_image.h>

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DYNAMIC_LIGHTING_TEXELS_PER_UNIT 24u
#define DYNAMIC_LIGHTING_MIN_TEXELS_PER_UNIT 0.001f
#define DYNAMIC_LIGHTING_MAX_SIZE 4096u
#define DYNAMIC_INFLUENCE_LIMIT 8u
#define DYNAMIC_TRACE_INSTANCE_LIMIT 8u
#define DYNAMIC_SURFACE_SAMPLES_PER_FRAME 2048u
#define DYNAMIC_SHADOW_SIZE 2048u
#define DYNAMIC_TIMESTAMP_BASE 8u
#define DYNAMIC_TIMESTAMP_STRIDE 4u
#define DYNAMIC_TIMING_LOG_INTERVAL 120u

typedef struct MAT4 {
    float m[16];
} MAT4;

typedef struct COLOR4 {
    float r, g, b, a;
} COLOR4;

void bake_progress(RENDERER *renderer, const char *stage, Uint32 done, Uint32 total) {
    if (!renderer || !renderer->gpu->window) return;

    renderer->bake_stage = stage;

    char title[160];

    if (total) {
        snprintf(title, sizeof(title), "Pony - B baking %s: %u/%u", stage, done, total);

        const double valid_texels = (double)renderer->lightmap_sample_count;
        const double completed_work = valid_texels * (double)done / (double)total;

        printf("frame time: %.2f ms | bake: %.2e/%.2e (%s)\n", renderer->frame_time_ms, completed_work, valid_texels, stage);
        fflush(stdout);
    } else {
        snprintf(title, sizeof(title), "Pony - B baking %s...", stage);
    }

    SDL_SetWindowTitle(renderer->gpu->window, title);
    SDL_PumpEvents();
}

static void bake_timing(const char *stage, Uint64 started) {
    const double elapsed = (double)(SDL_GetPerformanceCounter() - started) * 1000.0 / (double)SDL_GetPerformanceFrequency();

    SDL_Log("B: %s took %.2f ms", stage, elapsed);
}

static MAT4 m4_identity(void) {
    MAT4 result = {0};
    result.m[0] = result.m[5] = result.m[10] = result.m[15] = 1.0f;

    return result;
}

static MAT4 m4_mul(MAT4 a, MAT4 b) {
    MAT4 result = {0};

    for (int column = 0; column < 4; ++column) {
        for (int row = 0; row < 4; ++row) {
            result.m[column * 4 + row] =
                a.m[row] * b.m[column * 4] + a.m[4 + row] * b.m[column * 4 + 1] + a.m[8 + row] * b.m[column * 4 + 2] + a.m[12 + row] * b.m[column * 4 + 3];
        }
    }

    return result;
}

static MAT4 m4_transform(TRANSFORM transform, bool normal_matrix) {
    const float qx = transform.rotation[0];
    const float qy = transform.rotation[1];
    const float qz = transform.rotation[2];
    const float qw = transform.rotation[3];
    const float qn = sqrtf(qx * qx + qy * qy + qz * qz + qw * qw);
    const float x = qn > FLT_EPSILON ? qx / qn : 0.0f;
    const float y = qn > FLT_EPSILON ? qy / qn : 0.0f;
    const float z = qn > FLT_EPSILON ? qz / qn : 0.0f;
    const float w = qn > FLT_EPSILON ? qw / qn : 1.0f;

    float sx = transform.scale.x;
    float sy = transform.scale.y;
    float sz = transform.scale.z;

    if (normal_matrix) {
        sx = fabsf(sx) > FLT_EPSILON ? 1.0f / sx : 0.0f;
        sy = fabsf(sy) > FLT_EPSILON ? 1.0f / sy : 0.0f;
        sz = fabsf(sz) > FLT_EPSILON ? 1.0f / sz : 0.0f;
    }

    MAT4 result = m4_identity();
    result.m[0] = (1.0f - 2.0f * (y * y + z * z)) * sx;
    result.m[1] = (2.0f * (x * y + w * z)) * sx;
    result.m[2] = (2.0f * (x * z - w * y)) * sx;

    result.m[4] = (2.0f * (x * y - w * z)) * sy;
    result.m[5] = (1.0f - 2.0f * (x * x + z * z)) * sy;
    result.m[6] = (2.0f * (y * z + w * x)) * sy;

    result.m[8] = (2.0f * (x * z + w * y)) * sz;
    result.m[9] = (2.0f * (y * z - w * x)) * sz;
    result.m[10] = (1.0f - 2.0f * (x * x + y * y)) * sz;

    if (!normal_matrix) {
        result.m[12] = transform.position.x;
        result.m[13] = transform.position.y;
        result.m[14] = transform.position.z;
    }

    return result;
}

static VEC3 m4_point(MAT4 matrix, VEC3 p) {
    return v3(matrix.m[0] * p.x + matrix.m[4] * p.y + matrix.m[8] * p.z + matrix.m[12],
              matrix.m[1] * p.x + matrix.m[5] * p.y + matrix.m[9] * p.z + matrix.m[13],
              matrix.m[2] * p.x + matrix.m[6] * p.y + matrix.m[10] * p.z + matrix.m[14]);
}

static VEC3 m4_vector(MAT4 matrix, VEC3 v) {
    return v3(matrix.m[0] * v.x + matrix.m[4] * v.y + matrix.m[8] * v.z,
              matrix.m[1] * v.x + matrix.m[5] * v.y + matrix.m[9] * v.z,
              matrix.m[2] * v.x + matrix.m[6] * v.y + matrix.m[10] * v.z);
}

static MAT4 m4_inverse_transform(TRANSFORM transform) {
    const MAT4 normal = m4_transform(transform, true);
    MAT4 inverse = m4_identity();

    inverse.m[0] = normal.m[0];
    inverse.m[1] = normal.m[4];
    inverse.m[2] = normal.m[8];
    inverse.m[4] = normal.m[1];
    inverse.m[5] = normal.m[5];
    inverse.m[6] = normal.m[9];
    inverse.m[8] = normal.m[2];
    inverse.m[9] = normal.m[6];
    inverse.m[10] = normal.m[10];

    const VEC3 t = transform.position;
    inverse.m[12] = -(inverse.m[0] * t.x + inverse.m[4] * t.y + inverse.m[8] * t.z);
    inverse.m[13] = -(inverse.m[1] * t.x + inverse.m[5] * t.y + inverse.m[9] * t.z);
    inverse.m[14] = -(inverse.m[2] * t.x + inverse.m[6] * t.y + inverse.m[10] * t.z);
    return inverse;
}

static MAT4 m4_perspective(float fov_y, float aspect, float znear, float zfar) {
    const float f = 1.0f / tanf(fov_y * 0.5f);
    MAT4 result = {0};
    result.m[0] = f / aspect;
    result.m[5] = f;
    result.m[10] = zfar / (znear - zfar);
    result.m[11] = -1.0f;
    result.m[14] = (znear * zfar) / (znear - zfar);

    return result;
}

static MAT4 m4_look_at(VEC3 eye, VEC3 target, VEC3 up) {
    const VEC3 forward = v3_normalize(v3_sub(target, eye));
    const VEC3 side = v3_normalize(v3_cross(forward, up));
    const VEC3 corrected_up = v3_cross(side, forward);
    MAT4 result = m4_identity();

    result.m[0] = side.x;
    result.m[1] = corrected_up.x;
    result.m[2] = -forward.x;
    result.m[4] = side.y;
    result.m[5] = corrected_up.y;
    result.m[6] = -forward.y;
    result.m[8] = side.z;
    result.m[9] = corrected_up.z;
    result.m[10] = -forward.z;
    result.m[12] = -v3_dot(side, eye);
    result.m[13] = -v3_dot(corrected_up, eye);
    result.m[14] = v3_dot(forward, eye);

    return result;
}

static bool reserve_vertices(RENDERER *renderer, uint32_t needed) {
    if (needed <= renderer->vertex_capacity) return true;

    uint32_t capacity = renderer->vertex_capacity ? renderer->vertex_capacity : 1024u;

    while (capacity < needed) {
        if (capacity > UINT32_MAX / 2u) return false;
        capacity *= 2u;
    }

    RENDER_VERTEX *vertices = realloc(renderer->vertices, (size_t)capacity * sizeof(*vertices));

    if (!vertices) return false;

    renderer->vertices = vertices;
    renderer->vertex_capacity = capacity;

    return true;
}

static bool push_surface(RENDERER *renderer, const GLTF_VERTEX *vertex, LMAP_UV uv) {
    if (!reserve_vertices(renderer, renderer->vertex_count + 1u)) return false;

    renderer->vertices[renderer->vertex_count++] = (RENDER_VERTEX){.x = vertex->position.x,
                                                                   .y = vertex->position.y,
                                                                   .z = vertex->position.z,
                                                                   .nx = vertex->normal.x,
                                                                   .ny = vertex->normal.y,
                                                                   .nz = vertex->normal.z,
                                                                   .u = vertex->u,
                                                                   .v = vertex->v,
                                                                   .lu = uv.u,
                                                                   .lv = uv.v,
                                                                   .r = 1.0f,
                                                                   .g = 1.0f,
                                                                   .b = 1.0f,
                                                                   .a = 1.0f};

    return true;
}

static bool push_line_vertex(RENDERER *renderer, VEC3 position, COLOR4 color) {
    if (!reserve_vertices(renderer, renderer->vertex_count + 1u)) return false;

    renderer->vertices[renderer->vertex_count++] =
        (RENDER_VERTEX){.x = position.x, .y = position.y, .z = position.z, .r = color.r, .g = color.g, .b = color.b, .a = color.a};

    return true;
}

static bool add_wire_triangle(RENDERER *renderer, VEC3 a, VEC3 b, VEC3 c, COLOR4 color) {
    return push_line_vertex(renderer, a, color) && push_line_vertex(renderer, b, color) && push_line_vertex(renderer, b, color) &&
           push_line_vertex(renderer, c, color) && push_line_vertex(renderer, c, color) && push_line_vertex(renderer, a, color);
}

static void free_probe_grid(PROBE_GRID *grid) {
    if (!grid) return;

    free(grid->probes);
    memset(grid, 0, sizeof(*grid));
}

static bool make_probe_grid(const MESH *mesh, float spacing, PROBE_GRID *grid) {
    if (!mesh || !grid || spacing <= 0.0f) return false;

    memset(grid, 0, sizeof(*grid));

    const VEC3 extent = v3_sub(mesh->bounds.max, mesh->bounds.min);

    if (!isfinite(extent.x) || !isfinite(extent.y) || !isfinite(extent.z) || extent.x < 0.0f || extent.y < 0.0f || extent.z < 0.0f) return false;

    if (extent.x / spacing > 16384.0f || extent.y / spacing > 16384.0f || extent.z / spacing > 16384.0f) return false;

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

typedef struct CAMERA_UNIFORMS {
    float mvp[16];
    float view[16];
    float model[16];
    float normal_model[16];
} CAMERA_UNIFORMS;

typedef struct SKY_UNIFORMS {
    float camera_right[4];
    float camera_up[4];
    float camera_forward[4];
    float sky_zenith[4];
    float sky_horizon[4];
    float sun_direction_intensity[4];
    float sun_color_radius[4];
} SKY_UNIFORMS;

typedef struct MATERIAL_UNIFORMS {
    float base_color_factor[4];
    float emissive_metallic[4];
    float roughness_normal_ao_sun[4];
    float sun_direction[4];
    float sun_color[4];
    float camera_position[4];

    float ior_transmission_volume[4];
    float attenuation_iridescence[4];
    float iridescence_params[4];

    float camera_right_tan[4];
    float camera_up_tan[4];
    float camera_forward[4];
    float sky_zenith[4];
    float sky_horizon[4];

    float shadow_u_min[4];
    float shadow_v_min[4];
    float shadow_sun_max[4];
    float shadow_extent_bias[4];
    float shadow_texel_enabled[4];
    float dynamic_flags[4];

    float probe_origin_spacing[4];
    Uint32 probe_dims[4];
    float beam_origin[4];
    float beam_step[4];
    Uint32 beam_dims[4];

    Uint32 dynamic_influence_meta[4];
    float dynamic_influence_center_radius[DYNAMIC_INFLUENCE_LIMIT][4];
    float dynamic_influence_axis_x[DYNAMIC_INFLUENCE_LIMIT][4];
    float dynamic_influence_axis_y[DYNAMIC_INFLUENCE_LIMIT][4];
    float dynamic_influence_axis_z[DYNAMIC_INFLUENCE_LIMIT][4];
    float dynamic_influence_diffuse[DYNAMIC_INFLUENCE_LIMIT][4];
    float dynamic_influence_emissive[DYNAMIC_INFLUENCE_LIMIT][4];
} MATERIAL_UNIFORMS;

typedef struct DYNAMIC_SHADOW_UNIFORMS {
    float model[16];
    float shadow_u_min[4];
    float shadow_v_min[4];
    float shadow_sun_max[4];
    float shadow_extent[4];
} DYNAMIC_SHADOW_UNIFORMS;

typedef struct DYNAMIC_SURFACE_UNIFORMS {
    float model[16];
    float inverse_model[16];
    float normal_model[16];

    Uint32 sample_offset;
    Uint32 sample_count;
    Uint32 texture_width;
    Uint32 texture_height;

    float beam_origin[4];
    float beam_step[4];
    Uint32 beam_dims[4];

    float sun_direction_intensity[4];
    float sun_color_visibility_floor[4];
    float sky_zenith[4];
    float sky_horizon[4];
    float trace_params[4];

    Uint32 dynamic_instance_data[4];
    float dynamic_instance_inverse[DYNAMIC_TRACE_INSTANCE_LIMIT][16];
    float dynamic_instance_normal[DYNAMIC_TRACE_INSTANCE_LIMIT][16];
    Uint32 dynamic_instance_meta[DYNAMIC_TRACE_INSTANCE_LIMIT][4];
} DYNAMIC_SURFACE_UNIFORMS;

typedef struct SSAO_UNIFORMS {
    Uint32 width, height, ao_width, ao_height;
    float tan_half_fov, aspect, radius, bias;
} SSAO_UNIFORMS;

typedef struct BLOOM_UNIFORMS {
    Uint32 src_width, src_height, dst_width, dst_height;
    Uint32 phase, _pad0, _pad1, _pad2;
    float threshold, knee, strength, _pad3;
} BLOOM_UNIFORMS;

typedef struct COMPOSE_UNIFORMS {
    float exposure;
    float ao_strength;
    float bloom_strength;
    float _pad;
} COMPOSE_UNIFORMS;

typedef struct VOLUME_UNIFORMS {
    float eye_density[4], right_tan[4], up_tan[4], forward_g[4];
    float sun_intensity[4], sun_color[4], grid_origin_spacing[4];
    Uint32 grid_dims_width[4], height_debug[4];
    float beam_origin[4], beam_step[4];
    float volume_params[4], volume_radii[4], volume_filter[4];
    Uint32 volume_quality[4], volume_strides[4];

    float shadow_u_min[4];
    float shadow_v_min[4];
    float shadow_sun_max[4];
    float shadow_extent_bias[4];
    float shadow_texel_enabled[4];
} VOLUME_UNIFORMS;

typedef struct VOLUME_COMPOSE_UNIFORMS {
    Uint32 width, height, debug_view, bypass_volume;

    float volume_radii[4];
    float volume_filter[4];
    Uint32 volume_strides[4];
} VOLUME_COMPOSE_UNIFORMS;

struct DYNAMIC_LIGHTING_ALLOCATION {
    OBJECT_ID object_id;
    const LIGHTMAP *layout;
    NriTexture *texture;
    NriTexture *reference_texture;
    NriBuffer *sample_buffer;
    uint32_t dynamic_node_offset;
    uint32_t dynamic_node_count;
    uint32_t dynamic_triangle_offset;
    uint32_t dynamic_triangle_count;
    VEC3 local_center;
    VEC3 local_extents;
    float local_radius;
    VEC3 average_diffuse;
    VEC3 average_emissive;
    uint32_t transform_revision;
    uint32_t lighting_revision;
    uint32_t reference_transform_revision;
    uint32_t reference_lighting_revision;
    uint32_t pending_transform_revision;
    uint32_t pending_lighting_revision;
    uint32_t sample_cursor;
    bool cache_needs_clear;
};

typedef struct DYNAMIC_STATIC_SURFACE_GPU {
    Uint32 source_triangle;
    Uint32 _pad[3];
} DYNAMIC_STATIC_SURFACE_GPU;

struct RENDER_MATERIAL {
    GLTF_MATERIAL data;
    NriTexture *base_color;
    NriTexture *metallic_roughness;
    NriTexture *normal;
    NriTexture *occlusion;
    NriTexture *emissive;
    NriTexture *transmission;
    NriTexture *thickness;
    NriTexture *iridescence;
    NriTexture *iridescence_thickness;
};

static NriPipeline *make_surface_pipeline(RENDERER *r, NriCoreInterface *core, NriPipelineLayout *layout, const NriShaderDesc *vs, const NriShaderDesc *ps,
                                          bool transmission) {
    const NriVertexStreamDesc vb = {.bindingSlot = 0, .stepRate = NriVertexStreamStepRate_PER_VERTEX, .stride = (uint16_t)sizeof(RENDER_VERTEX)};

    const NriVertexAttributeDesc attrs[4] = {{.d3d = {.semanticName = "TEXCOORD", .semanticIndex = 0},
                                              .vk = {.location = 0},
                                              .offset = (uint32_t)offsetof(RENDER_VERTEX, x),
                                              .format = NriFormat_RGB32_SFLOAT,
                                              .streamIndex = 0},
                                             {.d3d = {.semanticName = "TEXCOORD", .semanticIndex = 1},
                                              .vk = {.location = 1},
                                              .offset = (uint32_t)offsetof(RENDER_VERTEX, nx),
                                              .format = NriFormat_RGB32_SFLOAT,
                                              .streamIndex = 0},
                                             {.d3d = {.semanticName = "TEXCOORD", .semanticIndex = 2},
                                              .vk = {.location = 2},
                                              .offset = (uint32_t)offsetof(RENDER_VERTEX, u),
                                              .format = NriFormat_RG32_SFLOAT,
                                              .streamIndex = 0},
                                             {.d3d = {.semanticName = "TEXCOORD", .semanticIndex = 3},
                                              .vk = {.location = 3},
                                              .offset = (uint32_t)offsetof(RENDER_VERTEX, lu),
                                              .format = NriFormat_RG32_SFLOAT,
                                              .streamIndex = 0}};

    const NriVertexInputDesc vertex_input = {.attributes = attrs, .attributeNum = 4, .streams = &vb, .streamNum = 1};

    const NriColorAttachmentDesc targets[2] = {{.format = NriFormat_RGBA16_SFLOAT, .colorWriteMask = NriColorWriteBits_RGBA},
                                               {.format = NriFormat_RGBA16_SFLOAT,
                                                .colorWriteMask = transmission ? NriColorWriteBits_NONE : NriColorWriteBits_RGBA}};

    const NriMultisampleDesc multisample = {.sampleMask = NRI_ALL, .sampleNum = 1};
    const NriShaderDesc shaders[2] = {*vs, *ps};

    const NriGraphicsPipelineDesc desc = {
        .pipelineLayout = layout,
        .vertexInput = &vertex_input,
        .inputAssembly = {.topology = NriTopology_TRIANGLE_LIST},
        .rasterization = {.fillMode = NriFillMode_SOLID,
                          .cullMode = transmission ? NriCullMode_BACK : NriCullMode_NONE,
                          .frontCounterClockwise = true,
                          .depthClamp = false},
        .multisample = &multisample,
        .outputMerger = {.colors = targets,
                         .colorNum = 2,
                         .depth = {.compareOp = transmission ? NriCompareOp_LESS_EQUAL : NriCompareOp_LESS, .write = !transmission},
                         .depthStencilFormat = r->depth_format},
        .shaders = shaders,
        .shaderNum = 2,
        .cache = r->gpu->pipeline_cache};

    NriPipeline *pipeline = NULL;

    if (core->CreateGraphicsPipeline(r->gpu->device, &desc, &pipeline) != NriResult_SUCCESS) return NULL;

    return pipeline;
}

static NriPipeline *make_dynamic_shadow_pipeline(RENDERER *r, NriPipelineLayout *layout, const NriShaderDesc *vs) {
    const NriVertexStreamDesc vb = {.bindingSlot = 0, .stepRate = NriVertexStreamStepRate_PER_VERTEX, .stride = (uint16_t)sizeof(RENDER_VERTEX)};
    const NriVertexAttributeDesc position = {.d3d = {.semanticName = "TEXCOORD", .semanticIndex = 0},
                                             .vk = {.location = 0},
                                             .offset = (uint32_t)offsetof(RENDER_VERTEX, x),
                                             .format = NriFormat_RGB32_SFLOAT,
                                             .streamIndex = 0};
    const NriVertexInputDesc vertex_input = {.attributes = &position, .attributeNum = 1, .streams = &vb, .streamNum = 1};
    const NriMultisampleDesc multisample = {.sampleMask = NRI_ALL, .sampleNum = 1};
    const NriGraphicsPipelineDesc desc = {
        .pipelineLayout = layout,
        .vertexInput = &vertex_input,
        .inputAssembly = {.topology = NriTopology_TRIANGLE_LIST},
        .rasterization = {.fillMode = NriFillMode_SOLID, .cullMode = NriCullMode_NONE, .frontCounterClockwise = true, .depthClamp = false},
        .multisample = &multisample,
        .outputMerger = {.depth = {.compareOp = NriCompareOp_LESS, .write = true}, .depthStencilFormat = r->depth_format},
        .shaders = vs,
        .shaderNum = 1,
        .cache = r->gpu->pipeline_cache};
    NriPipeline *pipeline = NULL;

    return r->gpu->core.CreateGraphicsPipeline(r->gpu->device, &desc, &pipeline) == NriResult_SUCCESS ? pipeline : NULL;
}

static bool create_dynamic_shadow_layout(RENDERER *r) {
    static const NriDescriptorType object[] = {NriDescriptorType_CONSTANT_BUFFER};
    const NriDescriptorType *sets[4] = {NULL, object, NULL, NULL};
    const uint8_t counts[4] = {0, 1, 0, 0};

    return gpu_create_pipeline_layout(r, &r->dynamic_shadow_layout, sets, counts, NriStageBits_VERTEX_SHADER);
}

static bool create_dynamic_surface_layout(RENDERER *r) {
    static const NriDescriptorType sources[] = {
        NriDescriptorType_STRUCTURED_BUFFER, NriDescriptorType_STRUCTURED_BUFFER, NriDescriptorType_STRUCTURED_BUFFER,
        NriDescriptorType_STRUCTURED_BUFFER, NriDescriptorType_STRUCTURED_BUFFER, NriDescriptorType_STRUCTURED_BUFFER,
        NriDescriptorType_STRUCTURED_BUFFER, NriDescriptorType_STRUCTURED_BUFFER,
        NriDescriptorType_TEXTURE, NriDescriptorType_SAMPLER,
    };

    return gpu_create_compute_layout(r, &r->dynamic_surface_layout, sources, 10, NriDescriptorType_STORAGE_TEXTURE, true);
}

static bool create_surface_layout(RENDERER *r) {
    static const NriDescriptorType camera[] = {NriDescriptorType_CONSTANT_BUFFER};

    static const NriDescriptorType material[] = {
        NriDescriptorType_TEXTURE, NriDescriptorType_TEXTURE, NriDescriptorType_TEXTURE, NriDescriptorType_TEXTURE, NriDescriptorType_TEXTURE,
        NriDescriptorType_TEXTURE, NriDescriptorType_TEXTURE, NriDescriptorType_TEXTURE, NriDescriptorType_TEXTURE, NriDescriptorType_TEXTURE,
        NriDescriptorType_TEXTURE, NriDescriptorType_TEXTURE, NriDescriptorType_STRUCTURED_BUFFER, NriDescriptorType_STRUCTURED_BUFFER,
        NriDescriptorType_SAMPLER, NriDescriptorType_SAMPLER, NriDescriptorType_SAMPLER, NriDescriptorType_SAMPLER};

    static const NriDescriptorType uniform[] = {NriDescriptorType_CONSTANT_BUFFER};

    const NriDescriptorType *sets[4] = {NULL, camera, material, uniform};
    const uint8_t counts[4] = {0, 1, 18, 1};

    return gpu_create_pipeline_layout(r, &r->surface_layout, sets, counts, NriStageBits_VERTEX_SHADER | NriStageBits_FRAGMENT_SHADER);
}

static bool create_line_layout(RENDERER *r) {
    static const NriDescriptorType camera[] = {NriDescriptorType_CONSTANT_BUFFER};
    const NriDescriptorType *sets[4] = {NULL, camera, NULL, NULL};
    const uint8_t counts[4] = {0, 1, 0, 0};

    return gpu_create_pipeline_layout(r, &r->line_layout, sets, counts, NriStageBits_VERTEX_SHADER | NriStageBits_FRAGMENT_SHADER);
}

static bool create_sky_layout(RENDERER *r) {
    static const NriDescriptorType uniform[] = {NriDescriptorType_CONSTANT_BUFFER};

    const NriDescriptorType *sets[4] = {NULL, NULL, NULL, uniform};
    const uint8_t counts[4] = {0, 0, 0, 1};

    return gpu_create_pipeline_layout(r, &r->sky_layout, sets, counts, NriStageBits_VERTEX_SHADER | NriStageBits_FRAGMENT_SHADER);
}

static bool create_ssao_layout(RENDERER *r) {
    static const NriDescriptorType src[] = {NriDescriptorType_TEXTURE, NriDescriptorType_SAMPLER};

    return gpu_create_compute_layout(r, &r->ssao_layout, src, 2, NriDescriptorType_STORAGE_TEXTURE, true);
}

static bool create_bloom_layout(RENDERER *r) {
    static const NriDescriptorType src[] = {NriDescriptorType_TEXTURE, NriDescriptorType_SAMPLER};

    return gpu_create_compute_layout(r, &r->bloom_layout, src, 2, NriDescriptorType_STORAGE_TEXTURE, true);
}

static bool create_grade_layout(RENDERER *r) {
    return gpu_create_compute_layout(r, &r->grade_layout, NULL, 0, NriDescriptorType_STORAGE_TEXTURE, false);
}

static bool create_volume_layout(RENDERER *r) {
    static const NriDescriptorType src[] = {NriDescriptorType_TEXTURE, NriDescriptorType_SAMPLER, NriDescriptorType_STRUCTURED_BUFFER,
                                            NriDescriptorType_STRUCTURED_BUFFER, NriDescriptorType_TEXTURE, NriDescriptorType_SAMPLER};

    return gpu_create_compute_layout(r, &r->volume_layout, src, 6, NriDescriptorType_STORAGE_TEXTURE, true);
}

static bool create_volume_compose_layout(RENDERER *r) {
    static const NriDescriptorType src[] = {NriDescriptorType_TEXTURE, NriDescriptorType_TEXTURE, NriDescriptorType_TEXTURE,
                                            NriDescriptorType_SAMPLER, NriDescriptorType_SAMPLER, NriDescriptorType_SAMPLER};

    return gpu_create_compute_layout(r, &r->volume_compose_layout, src, 6, NriDescriptorType_STORAGE_TEXTURE, true);
}

static bool create_compose_layout(RENDERER *r) {
    static const NriDescriptorType src[] = {NriDescriptorType_TEXTURE, NriDescriptorType_TEXTURE, NriDescriptorType_TEXTURE, NriDescriptorType_TEXTURE,
                                            NriDescriptorType_SAMPLER, NriDescriptorType_SAMPLER, NriDescriptorType_SAMPLER, NriDescriptorType_SAMPLER};

    static const NriDescriptorType uniform[] = {NriDescriptorType_CONSTANT_BUFFER};

    const NriDescriptorType *sets[4] = {NULL, NULL, src, uniform};
    const uint8_t counts[4] = {0, 0, 8, 1};

    return gpu_create_pipeline_layout(r, &r->compose_layout, sets, counts, NriStageBits_VERTEX_SHADER | NriStageBits_FRAGMENT_SHADER);
}

static bool bind_fx_resources(RENDERER *r, NriCommandBuffer *cmd, NriPipeline *pipeline, NriTexture *source, NriTexture *destination,
                              NriDescriptor *sampler_desc, const void *uniforms, uint32_t size) {
    if ((source && !gpu_transition_texture(r, cmd, source, NriAccessBits_SHADER_RESOURCE, NriLayout_SHADER_RESOURCE, NriStageBits_COMPUTE_SHADER)) ||
        !gpu_transition_texture(r, cmd, destination, NriAccessBits_SHADER_RESOURCE_STORAGE, NriLayout_SHADER_RESOURCE_STORAGE, NriStageBits_COMPUTE_SHADER))
        return false;

    NriPipelineLayout *layout = pipeline == r->fx.grade_pipeline ? r->grade_layout : pipeline == r->fx.bloom_pipeline ? r->bloom_layout : r->ssao_layout;

    NriDescriptor *dst = gpu_create_texture_view(r, destination, NriTextureView_STORAGE_TEXTURE);

    if (source) {
        NriDescriptor *src[] = {gpu_create_texture_view(r, source, NriTextureView_TEXTURE), sampler_desc};

        if (!gpu_bind_descriptor_set(r, cmd, layout, NriBindPoint_COMPUTE, 0, src, 2)) return false;
    }

    return gpu_bind_descriptor_set(r, cmd, layout, NriBindPoint_COMPUTE, 1, &dst, 1) &&
           (!size || gpu_bind_uniform_data(r, cmd, layout, NriBindPoint_COMPUTE, 2, uniforms, size));
}

static bool bind_volume_resources(RENDERER *r, NriCommandBuffer *cmd, NriTexture *normal, NriDescriptor *sampler_desc, NriBuffer *probes, NriBuffer *beams,
                                  NriTexture *output, const void *uniforms, uint32_t size) {
    NriTexture *shadow = r->dynamic_shadow_ready && r->dynamic_shadow_texture ? r->dynamic_shadow_texture : r->default_white;

    if (!gpu_transition_texture(r, cmd, normal, NriAccessBits_SHADER_RESOURCE, NriLayout_SHADER_RESOURCE, NriStageBits_COMPUTE_SHADER) ||
        !gpu_transition_texture(r, cmd, shadow, NriAccessBits_SHADER_RESOURCE, NriLayout_SHADER_RESOURCE, NriStageBits_COMPUTE_SHADER) ||
        !gpu_transition_texture(r, cmd, output, NriAccessBits_SHADER_RESOURCE_STORAGE, NriLayout_SHADER_RESOURCE_STORAGE, NriStageBits_COMPUTE_SHADER))
        return false;

    NriDescriptor *src[] = {gpu_create_texture_view(r, normal, NriTextureView_TEXTURE), sampler_desc,
                            gpu_create_buffer_view(r, probes, NriBufferView_STRUCTURED_BUFFER, sizeof(PROBE)),
                            gpu_create_buffer_view(r, beams, NriBufferView_STRUCTURED_BUFFER, sizeof(float)),
                            gpu_create_texture_view(r, shadow, NriTextureView_TEXTURE),
                            r->dynamic_shadow_sampler ? r->dynamic_shadow_sampler : sampler_desc};

    NriDescriptor *dst = gpu_create_texture_view(r, output, NriTextureView_STORAGE_TEXTURE);

    return gpu_bind_descriptor_set(r, cmd, r->volume_layout, NriBindPoint_COMPUTE, 0, src, 6) &&
           gpu_bind_descriptor_set(r, cmd, r->volume_layout, NriBindPoint_COMPUTE, 1, &dst, 1) &&
           gpu_bind_uniform_data(r, cmd, r->volume_layout, NriBindPoint_COMPUTE, 2, uniforms, size);
}

static bool bind_volume_compose_resources(RENDERER *r, NriCommandBuffer *cmd, NriTexture *hdr, NriTexture *volume, NriTexture *normal,
                                          NriDescriptor *sampler_desc, NriDescriptor *depth_sampler, NriTexture *output, const void *uniforms, uint32_t size) {
    NriTexture *sources[] = {hdr, volume, normal};

    for (uint32_t i = 0; i < 3; ++i)
        if (!gpu_transition_texture(r, cmd, sources[i], NriAccessBits_SHADER_RESOURCE, NriLayout_SHADER_RESOURCE, NriStageBits_COMPUTE_SHADER)) return false;

    if (!gpu_transition_texture(r, cmd, output, NriAccessBits_SHADER_RESOURCE_STORAGE, NriLayout_SHADER_RESOURCE_STORAGE, NriStageBits_COMPUTE_SHADER))
        return false;

    NriDescriptor *src[] = {gpu_create_texture_view(r, hdr, NriTextureView_TEXTURE),
                            gpu_create_texture_view(r, volume, NriTextureView_TEXTURE),
                            gpu_create_texture_view(r, normal, NriTextureView_TEXTURE),
                            sampler_desc,
                            sampler_desc,
                            depth_sampler};

    NriDescriptor *dst = gpu_create_texture_view(r, output, NriTextureView_STORAGE_TEXTURE);

    return gpu_bind_descriptor_set(r, cmd, r->volume_compose_layout, NriBindPoint_COMPUTE, 0, src, 6) &&
           gpu_bind_descriptor_set(r, cmd, r->volume_compose_layout, NriBindPoint_COMPUTE, 1, &dst, 1) &&
           gpu_bind_uniform_data(r, cmd, r->volume_compose_layout, NriBindPoint_COMPUTE, 2, uniforms, size);
}

static bool bind_sky_resources(RENDERER *r, NriCommandBuffer *cmd, const void *data, size_t size) {
    return gpu_bind_uniform_data(r, cmd, r->sky_layout, NriBindPoint_GRAPHICS, 3, data, size);
}

static bool bind_camera_resources(RENDERER *r, NriCommandBuffer *cmd, const void *data, size_t size) {
    return gpu_bind_uniform_data(r, cmd, r->surface_layout, NriBindPoint_GRAPHICS, 1, data, size);
}

static bool bind_surface_resources(RENDERER *r, NriCommandBuffer *cmd, const RENDER_MATERIAL *material, NriTexture *lightmap, NriTexture *scene_color,
                                   NriDescriptor *material_sampler, NriDescriptor *lightmap_sampler, NriDescriptor *scene_sampler, const void *uniforms,
                                   size_t size) {
    NriDescriptor *src[] = {gpu_create_texture_view(r, material->base_color, NriTextureView_TEXTURE),
                            gpu_create_texture_view(r, material->metallic_roughness, NriTextureView_TEXTURE),
                            gpu_create_texture_view(r, material->normal, NriTextureView_TEXTURE),
                            gpu_create_texture_view(r, material->occlusion, NriTextureView_TEXTURE),
                            gpu_create_texture_view(r, material->emissive, NriTextureView_TEXTURE),
                            gpu_create_texture_view(r, lightmap, NriTextureView_TEXTURE),
                            gpu_create_texture_view(r, material->transmission, NriTextureView_TEXTURE),
                            gpu_create_texture_view(r, material->thickness, NriTextureView_TEXTURE),
                            gpu_create_texture_view(r, material->iridescence, NriTextureView_TEXTURE),
                            gpu_create_texture_view(r, material->iridescence_thickness, NriTextureView_TEXTURE),
                            gpu_create_texture_view(r, scene_color, NriTextureView_TEXTURE),
                            gpu_create_texture_view(r, r->dynamic_shadow_ready && r->dynamic_shadow_texture ? r->dynamic_shadow_texture : r->default_white,
                                                    NriTextureView_TEXTURE),
                            gpu_create_buffer_view(r, r->beam_buffer ? r->beam_buffer : r->surface_beam_fallback_buffer,
                                                   NriBufferView_STRUCTURED_BUFFER, sizeof(float)),
                            gpu_create_buffer_view(r, r->volume_probe_buffer ? r->volume_probe_buffer : r->surface_probe_fallback_buffer,
                                                   NriBufferView_STRUCTURED_BUFFER, sizeof(PROBE)),
                            material_sampler,
                            lightmap_sampler,
                            scene_sampler,
                            r->dynamic_shadow_sampler ? r->dynamic_shadow_sampler : material_sampler};

    return gpu_bind_descriptor_set(r, cmd, r->surface_layout, NriBindPoint_GRAPHICS, 2, src, 18) &&
           gpu_bind_uniform_data(r, cmd, r->surface_layout, NriBindPoint_GRAPHICS, 3, uniforms, size);
}

static bool bind_dynamic_surface_resources(RENDERER *r, NriCommandBuffer *cmd, const DYNAMIC_LIGHTING_ALLOCATION *allocation,
                                           const DYNAMIC_SURFACE_UNIFORMS *uniforms) {
    if (!r || !cmd || !allocation || !allocation->texture || !allocation->sample_buffer || !allocation->self_node_buffer ||
        !allocation->self_triangle_buffer || !uniforms || !r->dynamic_static_node_buffer || !r->dynamic_static_triangle_buffer ||
        !r->dynamic_static_surface_buffer || !r->dynamic_static_uv_buffer || !r->beam_buffer || !r->lightmap_texture || !r->lightmap_sampler)
        return false;

    if (!gpu_transition_texture(r, cmd, r->lightmap_texture, NriAccessBits_SHADER_RESOURCE, NriLayout_SHADER_RESOURCE, NriStageBits_COMPUTE_SHADER) ||
        !gpu_transition_texture(r, cmd, allocation->texture, NriAccessBits_SHADER_RESOURCE_STORAGE, NriLayout_SHADER_RESOURCE_STORAGE,
                                NriStageBits_COMPUTE_SHADER))
        return false;

    NriDescriptor *src[] = {
        gpu_create_buffer_view(r, allocation->sample_buffer, NriBufferView_STRUCTURED_BUFFER, sizeof(LMAP_SAMPLE)),
        gpu_create_buffer_view(r, r->beam_buffer, NriBufferView_STRUCTURED_BUFFER, sizeof(float)),
        gpu_create_buffer_view(r, r->dynamic_static_node_buffer, NriBufferView_STRUCTURED_BUFFER, sizeof(BVH_NODE)),
        gpu_create_buffer_view(r, r->dynamic_static_triangle_buffer, NriBufferView_STRUCTURED_BUFFER, sizeof(BVH_TRIANGLE)),
        gpu_create_buffer_view(r, r->dynamic_static_surface_buffer, NriBufferView_STRUCTURED_BUFFER, 16u),
        gpu_create_buffer_view(r, r->dynamic_static_uv_buffer, NriBufferView_STRUCTURED_BUFFER, sizeof(LMAP_UV)),
        gpu_create_buffer_view(r, allocation->self_node_buffer, NriBufferView_STRUCTURED_BUFFER, sizeof(BVH_NODE)),
        gpu_create_buffer_view(r, allocation->self_triangle_buffer, NriBufferView_STRUCTURED_BUFFER, sizeof(BVH_TRIANGLE)),
        gpu_create_texture_view(r, r->lightmap_texture, NriTextureView_TEXTURE),
        r->lightmap_sampler,
    };
    NriDescriptor *dst = gpu_create_texture_view(r, allocation->texture, NriTextureView_STORAGE_TEXTURE);

    return gpu_bind_descriptor_set(r, cmd, r->dynamic_surface_layout, NriBindPoint_COMPUTE, 0, src, 10) &&
           gpu_bind_descriptor_set(r, cmd, r->dynamic_surface_layout, NriBindPoint_COMPUTE, 1, &dst, 1) &&
           gpu_bind_uniform_data(r, cmd, r->dynamic_surface_layout, NriBindPoint_COMPUTE, 2, uniforms, sizeof(*uniforms));
}

static bool bind_line_resources(RENDERER *r, NriCommandBuffer *cmd, const void *data, size_t size) {
    return gpu_bind_uniform_data(r, cmd, r->line_layout, NriBindPoint_GRAPHICS, 1, data, size);
}

static bool begin_surface_rendering(RENDERER *r, NriCommandBuffer *cmd, NriTexture *hdr, NriTexture *normal, NriTexture *depth, uint32_t width,
                                    uint32_t height, bool load) {
    NriDescriptor *hdr_view = gpu_create_texture_view(r, hdr, NriTextureView_COLOR_ATTACHMENT);
    NriDescriptor *normal_view = gpu_create_texture_view(r, normal, NriTextureView_COLOR_ATTACHMENT);
    NriDescriptor *depth_view = gpu_create_texture_view(r, depth, NriTextureView_DEPTH_STENCIL_ATTACHMENT);

    if (!hdr_view || !normal_view || !depth_view) return false;

    const NriAccessLayoutStage color = {NriAccessBits_COLOR_ATTACHMENT, NriLayout_COLOR_ATTACHMENT, NriStageBits_COLOR_ATTACHMENT};
    const NriAccessLayoutStage depth_state = {NriAccessBits_DEPTH_STENCIL_ATTACHMENT, NriLayout_DEPTH_STENCIL_ATTACHMENT,
                                              NriStageBits_DEPTH_STENCIL_ATTACHMENT};

    if (!gpu_texture_barrier(r, cmd, hdr, (NriAccessLayoutStage){0}, color) || !gpu_texture_barrier(r, cmd, normal, (NriAccessLayoutStage){0}, color) ||
        !gpu_texture_barrier(r, cmd, depth, (NriAccessLayoutStage){0}, depth_state))
        return false;

    const NriLoadOp load_op = load ? NriLoadOp_LOAD : NriLoadOp_CLEAR;
    const NriAttachmentDesc colors[2] = {{.descriptor = hdr_view, .loadOp = load_op, .storeOp = NriStoreOp_STORE},
                                         {.descriptor = normal_view, .loadOp = load_op, .storeOp = NriStoreOp_STORE}};

    const NriRenderingDesc desc = {
        .colors = colors,
        .colorNum = 2,
        .depth = {.descriptor = depth_view, .loadOp = load_op, .storeOp = NriStoreOp_STORE, .clearValue = {.depthStencil = {.depth = 1.0f}}}};

    r->gpu->core.CmdSetViewports(cmd, &(NriViewport){.width = (float)width, .height = (float)height, .depthMax = 1.0f}, 1);
    r->gpu->core.CmdSetScissors(cmd, &(NriRect){.width = (NriDim_t)width, .height = (NriDim_t)height}, 1);
    r->gpu->core.CmdBeginRendering(cmd, &desc);

    return true;
}

static bool ensure_dynamic_shadow_texture(RENDERER *r) {
    if (!r || !r->gpu->device) return false;
    if (r->dynamic_shadow_texture && r->dynamic_shadow_size == DYNAMIC_SHADOW_SIZE) return true;

    release_texture(r, r->dynamic_shadow_texture);
    r->dynamic_shadow_texture =
        gpu_create_texture(r, r->depth_format, NriTextureUsageBits_DEPTH_STENCIL_ATTACHMENT | NriTextureUsageBits_SHADER_RESOURCE, DYNAMIC_SHADOW_SIZE,
                           DYNAMIC_SHADOW_SIZE);
    r->dynamic_shadow_size = r->dynamic_shadow_texture ? DYNAMIC_SHADOW_SIZE : 0u;
    r->dynamic_shadow_ready = false;
    return r->dynamic_shadow_texture != NULL;
}

static bool begin_dynamic_shadow_rendering(RENDERER *r, NriCommandBuffer *cmd) {
    if (!r || !cmd || !ensure_dynamic_shadow_texture(r)) return false;

    NriDescriptor *depth_view = gpu_create_texture_view(r, r->dynamic_shadow_texture, NriTextureView_DEPTH_STENCIL_ATTACHMENT);
    if (!depth_view) return false;

    if (!gpu_transition_texture(r, cmd, r->dynamic_shadow_texture, NriAccessBits_DEPTH_STENCIL_ATTACHMENT, NriLayout_DEPTH_STENCIL_ATTACHMENT,
                                NriStageBits_DEPTH_STENCIL_ATTACHMENT))
        return false;

    const NriRenderingDesc desc = {
        .depth = {.descriptor = depth_view, .loadOp = NriLoadOp_CLEAR, .storeOp = NriStoreOp_STORE, .clearValue = {.depthStencil = {.depth = 1.0f}}}};

    r->gpu->core.CmdSetViewports(cmd, &(NriViewport){.width = (float)DYNAMIC_SHADOW_SIZE, .height = (float)DYNAMIC_SHADOW_SIZE, .depthMax = 1.0f}, 1);
    r->gpu->core.CmdSetScissors(cmd, &(NriRect){.width = DYNAMIC_SHADOW_SIZE, .height = DYNAMIC_SHADOW_SIZE}, 1);
    r->gpu->core.CmdBeginRendering(cmd, &desc);
    return true;
}

static bool snapshot_scene_color(RENDERER *r, NriCommandBuffer *cmd) {
    if (!r || !cmd || !r->fx.hdr || !r->fx.scene_color) return false;

    if (!gpu_transition_texture(r, cmd, r->fx.hdr, NriAccessBits_COPY_SOURCE, NriLayout_COPY_SOURCE, NriStageBits_COPY) ||
        !gpu_transition_texture(r, cmd, r->fx.scene_color, NriAccessBits_COPY_DESTINATION, NriLayout_COPY_DESTINATION, NriStageBits_COPY))
        return false;

    r->gpu->core.CmdCopyTexture(cmd, r->fx.scene_color, NULL, r->fx.hdr, NULL);

    return gpu_transition_texture(r, cmd, r->fx.scene_color, NriAccessBits_SHADER_RESOURCE, NriLayout_SHADER_RESOURCE, NriStageBits_FRAGMENT_SHADER);
}

static bool begin_compose_rendering(RENDERER *r, NriCommandBuffer *cmd, NriTexture *swap, NriTexture *hdr, NriTexture *ao, NriTexture *bloom, NriTexture *lut,
                                    NriDescriptor *sampler_desc, const void *uniforms, size_t size) {
    NriTexture *sources[] = {hdr, ao, bloom, lut};

    for (uint32_t i = 0; i < 4; ++i)
        if (!gpu_transition_texture(r, cmd, sources[i], NriAccessBits_SHADER_RESOURCE, NriLayout_SHADER_RESOURCE, NriStageBits_FRAGMENT_SHADER)) return false;

    NriDescriptor *src[] = {gpu_create_texture_view(r, hdr, NriTextureView_TEXTURE),
                            gpu_create_texture_view(r, ao, NriTextureView_TEXTURE),
                            gpu_create_texture_view(r, bloom, NriTextureView_TEXTURE),
                            gpu_create_texture_view(r, lut, NriTextureView_TEXTURE),
                            sampler_desc,
                            sampler_desc,
                            sampler_desc,
                            sampler_desc};

    if (!gpu_bind_descriptor_set(r, cmd, r->compose_layout, NriBindPoint_GRAPHICS, 2, src, 8) ||
        !gpu_bind_uniform_data(r, cmd, r->compose_layout, NriBindPoint_GRAPHICS, 3, uniforms, size))
        return false;

    if (!gpu_texture_barrier(
            r, cmd, swap, (NriAccessLayoutStage){.layout = NriLayout_UNDEFINED, .stages = NriStageBits_NONE},
            (NriAccessLayoutStage){.access = NriAccessBits_COLOR_ATTACHMENT, .layout = NriLayout_COLOR_ATTACHMENT, .stages = NriStageBits_COLOR_ATTACHMENT}))
        return false;

    const NriAttachmentDesc color = {
        .descriptor = gpu_swapchain_color_attachment(r, r->gpu->current_swap_index), .loadOp = NriLoadOp_CLEAR, .storeOp = NriStoreOp_STORE};

    const NriRenderingDesc desc = {.colors = &color, .colorNum = 1};

    r->gpu->core.CmdBeginRendering(cmd, &desc);

    return true;
}

static NriPipeline *make_line_pipeline(RENDERER *r, NriPipelineLayout *layout, const NriShaderDesc *vs, const NriShaderDesc *ps) {
    const NriVertexStreamDesc vb = {.bindingSlot = 0, .stepRate = NriVertexStreamStepRate_PER_VERTEX, .stride = (uint16_t)sizeof(RENDER_VERTEX)};

    const NriVertexAttributeDesc attrs[2] = {{.d3d = {.semanticName = "TEXCOORD", .semanticIndex = 0},
                                              .vk = {.location = 0},
                                              .offset = (uint32_t)offsetof(RENDER_VERTEX, x),
                                              .format = NriFormat_RGB32_SFLOAT,
                                              .streamIndex = 0},
                                             {.d3d = {.semanticName = "TEXCOORD", .semanticIndex = 1},
                                              .vk = {.location = 1},
                                              .offset = (uint32_t)offsetof(RENDER_VERTEX, r),
                                              .format = NriFormat_RGBA32_SFLOAT,
                                              .streamIndex = 0}};

    const NriVertexInputDesc vertex_input = {.attributes = attrs, .attributeNum = 2, .streams = &vb, .streamNum = 1};

    const NriColorAttachmentDesc targets[2] = {{.format = NriFormat_RGBA16_SFLOAT, .colorWriteMask = NriColorWriteBits_RGBA},
                                               {.format = NriFormat_RGBA16_SFLOAT, .colorWriteMask = NriColorWriteBits_NONE}};

    const NriMultisampleDesc multisample = {.sampleMask = NRI_ALL, .sampleNum = 1};

    const NriShaderDesc shaders[2] = {*vs, *ps};

    const NriGraphicsPipelineDesc desc = {
        .pipelineLayout = layout,
        .vertexInput = &vertex_input,
        .inputAssembly = {.topology = NriTopology_LINE_LIST},
        .rasterization = {.fillMode = NriFillMode_SOLID, .cullMode = NriCullMode_NONE, .frontCounterClockwise = true, .depthClamp = false},
        .multisample = &multisample,
        .outputMerger = {.colors = targets,
                         .colorNum = 2,
                         .depth = {.compareOp = NriCompareOp_LESS_EQUAL, .write = false},
                         .depthStencilFormat = r->depth_format},
        .shaders = shaders,
        .shaderNum = 2,
        .cache = r->gpu->pipeline_cache};

    NriPipeline *pipeline = NULL;

    if (r->gpu->core.CreateGraphicsPipeline(r->gpu->device, &desc, &pipeline) != NriResult_SUCCESS) return NULL;

    return pipeline;
}

static NriPipeline *make_sky_pipeline(RENDERER *r, NriPipelineLayout *layout, const NriShaderDesc *vs, const NriShaderDesc *ps) {
    const NriColorAttachmentDesc targets[2] = {{.format = NriFormat_RGBA16_SFLOAT, .colorWriteMask = NriColorWriteBits_RGBA},
                                               {.format = NriFormat_RGBA16_SFLOAT, .colorWriteMask = NriColorWriteBits_RGBA}};

    const NriMultisampleDesc multisample = {.sampleMask = NRI_ALL, .sampleNum = 1};

    const NriShaderDesc shaders[2] = {*vs, *ps};
    const NriGraphicsPipelineDesc desc = {
        .pipelineLayout = layout,
        .inputAssembly = {.topology = NriTopology_TRIANGLE_LIST},
        .rasterization = {.fillMode = NriFillMode_SOLID, .cullMode = NriCullMode_NONE, .frontCounterClockwise = true, .depthClamp = false},
        .multisample = &multisample,
        .outputMerger = {.colors = targets, .colorNum = 2, .depthStencilFormat = r->depth_format},
        .shaders = shaders,
        .shaderNum = 2,
        .cache = r->gpu->pipeline_cache};

    NriPipeline *pipeline = NULL;

    if (r->gpu->core.CreateGraphicsPipeline(r->gpu->device, &desc, &pipeline) != NriResult_SUCCESS) return NULL;

    return pipeline;
}

static NriTexture *pixel_texture(RENDERER *r, Uint8 red, Uint8 green, Uint8 blue, Uint8 alpha) {
    const Uint8 pixels[4] = {red, green, blue, alpha};

    NriTexture *result = gpu_create_texture(r, NriFormat_RGBA8_UNORM, NriTextureUsageBits_SHADER_RESOURCE, 1, 1);

    if (!result) return NULL;

    if (!gpu_upload_texture_data(r, result, pixels, 4, 4, NriAccessBits_SHADER_RESOURCE, NriLayout_SHADER_RESOURCE, NriStageBits_ALL)) {
        release_texture(r, result);

        return NULL;
    }

    return result;
}

static bool ensure_depth_texture(RENDERER *r, Uint32 width, Uint32 height) {
    if (r->depth_texture && r->depth_width == width && r->depth_height == height) return true;

    release_texture(r, r->depth_texture);
    r->depth_texture = gpu_create_texture(r, r->depth_format, NriTextureUsageBits_DEPTH_STENCIL_ATTACHMENT, width, height);

    if (!r->depth_texture) return false;

    r->depth_width = width;
    r->depth_height = height;

    return true;
}

static const char *image_type(const char *mime) {
    if (!mime || !*mime) return NULL;

    if (strstr(mime, "png")) return "PNG";

    if (strstr(mime, "jpeg") || strstr(mime, "jpg")) return "JPG";

    if (strstr(mime, "webp")) return "WEBP";

    if (strstr(mime, "avif")) return "AVIF";

    return NULL;
}

static NriTexture *load_image(RENDERER *r, const GLTF_IMAGE *image) {
    if (!image || !image->bytes.data || !image->bytes.size) return NULL;

    SDL_IOStream *io = SDL_IOFromConstMem(image->bytes.data, image->bytes.size);

    if (!io) return NULL;

    SDL_Surface *decoded = IMG_LoadTyped_IO(io, true, image_type(image->mime));

    if (!decoded) return NULL;

    SDL_Surface *rgba = SDL_ConvertSurface(decoded, SDL_PIXELFORMAT_RGBA32);

    SDL_DestroySurface(decoded);

    if (!rgba) return NULL;

    NriTexture *result = gpu_create_texture(r, NriFormat_RGBA8_UNORM, NriTextureUsageBits_SHADER_RESOURCE, (Uint32)rgba->w, (Uint32)rgba->h);

    if (result && !gpu_upload_texture_data(r, result, rgba->pixels, (uint32_t)rgba->pitch, (uint32_t)(rgba->pitch * rgba->h), NriAccessBits_SHADER_RESOURCE,
                                           NriLayout_SHADER_RESOURCE, NriStageBits_FRAGMENT_SHADER)) {
        release_texture(r, result);

        result = NULL;
    }

    SDL_DestroySurface(rgba);

    return result;
}

static bool load_images(RENDERER *r, const GLTF_SCENE *visual) {
    r->image_texture_count = visual->image_count;

    if (!visual->image_count) return true;

    r->image_textures = calloc(visual->image_count, sizeof(*r->image_textures));

    if (!r->image_textures) return false;

    for (uint32_t i = 0; i < visual->image_count; ++i) {
        const GLTF_IMAGE *image = &visual->images[i];

        if (!image->bytes.data || !image->bytes.size) continue;

        r->image_textures[i] = load_image(r, image);

        if (!r->image_textures[i]) {
            SDL_Log("SDL_image could not decode GLB image %u (%s): %s", i, image->mime[0] ? image->mime : "unknown", SDL_GetError());
        }
    }

    return true;
}

static NriTexture *resolve_texture(RENDERER *r, const GLTF_SCENE *visual, int32_t texture_index, NriTexture *fallback) {
    if (texture_index < 0 || (uint32_t)texture_index >= visual->texture_count) return fallback;

    const int32_t image = visual->textures[texture_index].image;

    if (image < 0 || (uint32_t)image >= r->image_texture_count || !r->image_textures[image]) return fallback;

    return r->image_textures[image];
}

static void release_dynamic_lighting(RENDERER *r) {
    if (!r) return;

    for (uint32_t i = 0; i < r->dynamic_lighting_count; ++i) {
        release_texture(r, r->dynamic_lighting[i].texture);
        release_texture(r, r->dynamic_lighting[i].reference_texture);
        release_buffer(r, r->dynamic_lighting[i].sample_buffer);
    }

    release_buffer(r, r->dynamic_object_node_buffer);
    release_buffer(r, r->dynamic_object_triangle_buffer);
    r->dynamic_object_node_buffer = NULL;
    r->dynamic_object_triangle_buffer = NULL;
    r->dynamic_object_node_count = 0u;
    r->dynamic_object_triangle_count = 0u;

    free(r->dynamic_lighting);
    r->dynamic_lighting = NULL;
    r->dynamic_lighting_count = 0u;
}

static void release_dynamic_static_transport(RENDERER *r) {
    if (!r) return;

    release_buffer(r, r->dynamic_static_node_buffer);
    release_buffer(r, r->dynamic_static_triangle_buffer);
    release_buffer(r, r->dynamic_static_surface_buffer);
    release_buffer(r, r->dynamic_static_uv_buffer);

    r->dynamic_static_node_buffer = NULL;
    r->dynamic_static_triangle_buffer = NULL;
    r->dynamic_static_surface_buffer = NULL;
    r->dynamic_static_uv_buffer = NULL;
    r->dynamic_static_node_count = 0u;
    r->dynamic_static_triangle_count = 0u;
}

static void release_reference_lighting(RENDERER *r) {
    if (!r) return;

    release_texture(r, r->reference_static_texture);
    r->reference_static_texture = NULL;
    r->reference_geometry_revision = 0u;
    r->reference_lighting_revision = 0u;
}

static DYNAMIC_LIGHTING_ALLOCATION *dynamic_lighting_find(RENDERER *r, OBJECT_ID object_id) {
    if (!r || !object_id) return NULL;

    for (uint32_t i = 0; i < r->dynamic_lighting_count; ++i)
        if (r->dynamic_lighting[i].object_id == object_id) return &r->dynamic_lighting[i];

    return NULL;
}

static const DYNAMIC_LIGHTING_ALLOCATION *dynamic_lighting_find_const(const RENDERER *r, OBJECT_ID object_id) {
    if (!r || !object_id) return NULL;

    for (uint32_t i = 0; i < r->dynamic_lighting_count; ++i)
        if (r->dynamic_lighting[i].object_id == object_id) return &r->dynamic_lighting[i];

    return NULL;
}

static void release_scene_resources(RENDERER *r) {
    if (!r || !r->gpu->device) return;

    release_dynamic_lighting(r);
    release_dynamic_static_transport(r);
    release_reference_lighting(r);

    if (r->image_textures) {
        for (uint32_t i = 0; i < r->image_texture_count; ++i) release_texture(r, r->image_textures[i]);
    }

    free(r->image_textures);
    r->image_textures = NULL;
    r->image_texture_count = 0;

    free(r->materials);
    r->materials = NULL;
    r->material_count = 0;
    r->has_transmission = false;

    free(r->transmission_draws);
    r->transmission_draws = NULL;
    r->transmission_draw_count = 0;

    release_texture(r, r->default_white);
    release_texture(r, r->default_normal);

    if (r->material_sampler) r->gpu->core.DestroyDescriptor(r->material_sampler);
    release_buffer(r, r->vertex_buffer);
    release_texture(r, r->lightmap_texture);

    if (r->lightmap_sampler) r->gpu->core.DestroyDescriptor(r->lightmap_sampler);

    r->default_white = NULL;
    r->default_normal = NULL;
    r->material_sampler = NULL;
    r->vertex_buffer = NULL;
    r->lightmap_texture = NULL;
    r->lightmap_sampler = NULL;
}

static bool upload_scene(RENDERER *r, const GLTF_SCENE *visual) {
    if (!r || !r->gpu->device || !visual || !r->vertices || !r->vertex_count) return false;

    release_scene_resources(r);

    r->default_white = pixel_texture(r, 255, 255, 255, 255);
    r->default_normal = pixel_texture(r, 128, 128, 255, 255);
    r->material_sampler = gpu_create_sampler(r, NriFilter_LINEAR, NriFilter_LINEAR, NriAddressMode_REPEAT);

    if (!r->default_white || !r->default_normal || !r->material_sampler) return false;

    const uint64_t vertex_bytes = (uint64_t)r->vertex_count * sizeof(*r->vertices);

    r->vertex_buffer = gpu_upload_buffer(r, NriBufferUsageBits_VERTEX, r->vertices, (size_t)vertex_bytes, 0);

    if (!r->vertex_buffer) return false;

    free(r->vertices);
    r->vertices = NULL;
    r->vertex_capacity = 0u;

    if (!load_images(r, visual)) return false;

    r->material_count = visual->material_count;
    r->has_transmission = false;
    r->materials = calloc(r->material_count, sizeof(*r->materials));

    if (!r->materials) return false;

    for (uint32_t i = 0; i < r->material_count; ++i) {
        RENDER_MATERIAL *m = &r->materials[i];
        m->data = visual->materials[i];
        m->base_color = resolve_texture(r, visual, m->data.base_color_texture, r->default_white);
        m->metallic_roughness = resolve_texture(r, visual, m->data.metallic_roughness_texture, r->default_white);
        m->normal = resolve_texture(r, visual, m->data.normal_texture, r->default_normal);
        m->occlusion = resolve_texture(r, visual, m->data.occlusion_texture, r->default_white);
        m->emissive = resolve_texture(r, visual, m->data.emissive_texture, r->default_white);
        m->transmission = resolve_texture(r, visual, m->data.transmission_texture, r->default_white);
        m->thickness = resolve_texture(r, visual, m->data.thickness_texture, r->default_white);
        m->iridescence = resolve_texture(r, visual, m->data.iridescence_texture, r->default_white);
        m->iridescence_thickness = resolve_texture(r, visual, m->data.iridescence_thickness_texture, r->default_white);

        r->has_transmission |= m->data.transmission_factor > 0.0f;
    }

    r->lightmap_sampler = gpu_create_sampler(r, NriFilter_LINEAR, NriFilter_LINEAR, NriAddressMode_CLAMP_TO_EDGE);
    r->lightmap_texture = pixel_texture(r, 0, 0, 0, 255);

    return r->lightmap_sampler && r->lightmap_texture;
}

static bool dispatch_one(FX_STATE *fx, NriCommandBuffer *cmd, NriPipeline *pipeline, NriTexture *source, NriTexture *destination, const void *uniforms,
                         Uint32 uniform_size, Uint32 width, Uint32 height) {
    if (!fx || !fx->owner || !cmd || !pipeline || !destination) return false;

    RENDERER *r = fx->owner;

    if (!bind_fx_resources(r, cmd, pipeline, source, destination, fx->sampler, uniforms, uniform_size)) return false;

    r->gpu->core.CmdSetPipeline(cmd, pipeline);
    r->gpu->core.CmdDispatch(cmd, &(NriDispatchDesc){.workGroupNumX = (width + 7u) / 8u, .workGroupNumY = (height + 7u) / 8u, .workGroupNumZ = 1});

    return true;
}

static void release_frame_textures(FX_STATE *fx) {
    if (!fx || !fx->owner) return;

    RENDERER *r = fx->owner;

    release_texture(r, fx->hdr);
    release_texture(r, fx->scene_color);
    release_texture(r, fx->normal_depth);
    release_texture(r, fx->ao);
    release_texture(r, fx->bloom_a);
    release_texture(r, fx->bloom_b);
    release_texture(r, fx->volume);
    release_texture(r, fx->lit);

    fx->hdr = NULL;
    fx->scene_color = NULL;
    fx->normal_depth = NULL;
    fx->ao = NULL;
    fx->bloom_a = NULL;
    fx->bloom_b = NULL;
    fx->volume = NULL;
    fx->lit = NULL;
    fx->width = fx->height = fx->ao_width = fx->ao_height = 0;
    fx->volume_ready = false;
}

static void fx_deinit(FX_STATE *fx) {
    if (!fx || !fx->owner) return;

    RENDERER *r = fx->owner;

    release_frame_textures(fx);

    release_texture(r, fx->lut);

    if (fx->sampler) r->gpu->core.DestroyDescriptor(fx->sampler);

    if (fx->depth_sampler) r->gpu->core.DestroyDescriptor(fx->depth_sampler);

    if (fx->compose_pipeline) r->gpu->core.DestroyPipeline(fx->compose_pipeline);

    if (fx->ssao_pipeline) r->gpu->core.DestroyPipeline(fx->ssao_pipeline);

    if (fx->bloom_pipeline) r->gpu->core.DestroyPipeline(fx->bloom_pipeline);

    if (fx->grade_pipeline) r->gpu->core.DestroyPipeline(fx->grade_pipeline);

    if (fx->volume_pipeline) r->gpu->core.DestroyPipeline(fx->volume_pipeline);

    if (fx->volume_compose_pipeline) r->gpu->core.DestroyPipeline(fx->volume_compose_pipeline);

    memset(fx, 0, sizeof(*fx));
}

static bool make_compose_pipeline(RENDERER *r, NriShaderDesc *vs, NriShaderDesc *ps, NriFormat swap_format, NriPipeline **out) {
    const NriColorAttachmentDesc target = {.format = swap_format, .colorWriteMask = NriColorWriteBits_RGBA};

    const NriShaderDesc shaders[2] = {*vs, *ps};
    const NriGraphicsPipelineDesc desc = {
        .pipelineLayout = r->compose_layout,
        .inputAssembly = {.topology = NriTopology_TRIANGLE_LIST},
        .rasterization = {.fillMode = NriFillMode_SOLID, .cullMode = NriCullMode_NONE, .frontCounterClockwise = true, .depthClamp = false},
        .outputMerger = {.colors = &target, .colorNum = 1},
        .shaders = shaders,
        .shaderNum = 2,
        .cache = r->gpu->pipeline_cache};

    return r->gpu->core.CreateGraphicsPipeline(r->gpu->device, &desc, out) == NriResult_SUCCESS;
}

static bool fx_init(FX_STATE *fx, RENDERER *r) {
    if (!fx || !r || !r->gpu->device) return false;

    NriShaderDesc vs = {0};
    NriShaderDesc ps = {0};
    NriCommandAllocator *allocator = NULL;
    NriCommandBuffer *cmd = NULL;

    memset(fx, 0, sizeof(*fx));
    fx->owner = r;

    fx->sampler = gpu_create_sampler(r, NriFilter_LINEAR, NriFilter_LINEAR, NriAddressMode_CLAMP_TO_EDGE);

    fx->depth_sampler = gpu_create_sampler(r, NriFilter_NEAREST, NriFilter_NEAREST, NriAddressMode_CLAMP_TO_EDGE);

    if (!fx->sampler || !fx->depth_sampler) goto fail;

    vs = gpu_load_shader("fullscreen_vs", "BUILD_FULLSCREEN_VS", NriStageBits_VERTEX_SHADER);

    ps = gpu_load_shader("compose_fs", "BUILD_COMPOSE_FS", NriStageBits_FRAGMENT_SHADER);

    fx->ssao_pipeline = gpu_compile_compute(r, r->ssao_layout, "shaders/compute.hlsl", "ssao_cs", "BUILD_SSAO_CS");

    fx->bloom_pipeline = gpu_compile_compute(r, r->bloom_layout, "shaders/compute.hlsl", "bloom_cs", "BUILD_BLOOM_CS");

    fx->grade_pipeline = gpu_compile_compute(r, r->grade_layout, "shaders/compute.hlsl", "grade_cs", "BUILD_GRADE_CS");

    fx->volume_pipeline = gpu_compile_compute(r, r->volume_layout, "shaders/vision_compute.hlsl", "volume_cs", "BUILD_VISION_VOLUME_CS");

    fx->volume_compose_pipeline =
        gpu_compile_compute(r, r->volume_compose_layout, "shaders/vision_compute.hlsl", "volume_compose_cs", "BUILD_VISION_COMPOSE_CS");

    if (!vs.bytecode || !ps.bytecode || !fx->ssao_pipeline || !fx->bloom_pipeline || !fx->grade_pipeline || !fx->volume_pipeline ||
        !fx->volume_compose_pipeline)
        goto fail;

    if (!make_compose_pipeline(r, &vs, &ps, r->gpu->swapchain_format, &fx->compose_pipeline)) goto fail;

    gpu_free_shader(&vs);
    gpu_free_shader(&ps);

    fx->lut = gpu_create_texture(r, NriFormat_RGBA16_SFLOAT, NriTextureUsageBits_SHADER_RESOURCE | NriTextureUsageBits_SHADER_RESOURCE_STORAGE, 256u, 16u);

    if (!fx->lut) goto fail;

    if (gpu_begin_commands(r, &allocator, &cmd) != NriResult_SUCCESS || !dispatch_one(fx, cmd, fx->grade_pipeline, NULL, fx->lut, NULL, 0, 256u, 16u))
        goto fail;

    const bool submitted = gpu_submit_commands(r, allocator, cmd);
    cmd = NULL;
    allocator = NULL;

    if (!submitted) goto fail;

    return true;

fail:
    gpu_free_shader(&vs);
    gpu_free_shader(&ps);
    gpu_abort_commands(r, allocator, cmd);
    fx_deinit(fx);

    return false;
}

static bool fx_ensure(FX_STATE *fx, Uint32 width, Uint32 height) {
    if (!fx || !fx->owner || !width || !height) return false;

    if (fx->hdr && fx->width == width && fx->height == height) return true;

    RENDERER *r = fx->owner;

    release_frame_textures(fx);
    fx->width = width;
    fx->height = height;
    fx->ao_width = (width + 1u) / 2u;
    fx->ao_height = (height + 1u) / 2u;

    const NriTextureUsageBits rt = NriTextureUsageBits_COLOR_ATTACHMENT | NriTextureUsageBits_SHADER_RESOURCE;

    const NriTextureUsageBits compute = NriTextureUsageBits_SHADER_RESOURCE | NriTextureUsageBits_SHADER_RESOURCE_STORAGE;

    fx->hdr = gpu_create_texture(r, NriFormat_RGBA16_SFLOAT, rt, width, height);
    fx->scene_color = gpu_create_texture(r, NriFormat_RGBA16_SFLOAT, NriTextureUsageBits_SHADER_RESOURCE, width, height);
    fx->normal_depth = gpu_create_texture(r, NriFormat_RGBA16_SFLOAT, rt, width, height);
    fx->ao = gpu_create_texture(r, NriFormat_RGBA16_SFLOAT, compute, fx->ao_width, fx->ao_height);
    fx->bloom_a = gpu_create_texture(r, NriFormat_RGBA16_SFLOAT, compute, fx->ao_width, fx->ao_height);
    fx->bloom_b = gpu_create_texture(r, NriFormat_RGBA16_SFLOAT, compute, fx->ao_width, fx->ao_height);
    fx->volume = gpu_create_texture(r, NriFormat_RGBA16_SFLOAT, compute, fx->ao_width, fx->ao_height);
    fx->lit = gpu_create_texture(r, NriFormat_RGBA16_SFLOAT, compute, width, height);

    if (!fx->hdr || !fx->scene_color || !fx->normal_depth || !fx->ao || !fx->bloom_a || !fx->bloom_b || !fx->volume || !fx->lit) {
        release_frame_textures(fx);

        return false;
    }

    return true;
}

static bool dynamic_shadow_projection(const RENDERER *r, const RENDER_FRAME *frame, float shadow_u_min[4], float shadow_v_min[4],
                                      float shadow_sun_max[4], float shadow_extent_bias[4]);

static bool fx_volume(FX_STATE *fx, NriCommandBuffer *cmd, NriBuffer *probes, NriBuffer *beams, const PROBE_GRID *grid, const BEAM_GRID *beam_grid,
                      const RENDER_FRAME *frame) {
    if (!fx || !fx->owner || !cmd || !probes || !beams || !grid || !beam_grid || !grid->probes || !fx->volume) return false;

    RENDERER *r = fx->owner;

    VOLUME_UNIFORMS u = {
        .eye_density = {frame->eye.x, frame->eye.y, frame->eye.z, 0.0f},
        .right_tan = {frame->right.x * frame->tan_half_fov * frame->aspect, frame->right.y * frame->tan_half_fov * frame->aspect,
                      frame->right.z * frame->tan_half_fov * frame->aspect, 0},
        .up_tan = {frame->up.x * frame->tan_half_fov, frame->up.y * frame->tan_half_fov, frame->up.z * frame->tan_half_fov, 0},
        .forward_g = {frame->forward.x, frame->forward.y, frame->forward.z, 0.0f},
        .sun_intensity = {frame->sun.direction.x, frame->sun.direction.y, frame->sun.direction.z, frame->sun.intensity},
        .sun_color = {frame->sun.color.x, frame->sun.color.y, frame->sun.color.z, 1.0f},
        .grid_origin_spacing = {grid->origin.x, grid->origin.y, grid->origin.z, grid->spacing},
        .grid_dims_width = {grid->count_x, grid->count_y, grid->count_z, fx->ao_width},
        .height_debug = {fx->ao_height, fx->debug_view, beam_grid->depth, 0},
        .beam_origin = {beam_grid->origin.x, beam_grid->origin.y, beam_grid->origin.z, 0},
        .beam_step = {beam_grid->step.x, beam_grid->step.y, beam_grid->step.z, 0},
        .volume_params = {frame->volumetrics.density, frame->volumetrics.anisotropy, frame->volumetrics.probe_intensity, frame->volumetrics.max_distance},
        .volume_radii = {frame->vision.center_radius, frame->vision.middle_radius, frame->vision.center_transition_width,
                         frame->vision.middle_transition_width},
        .volume_filter = {frame->vision.jitter_strength, frame->vision.volume_blur_strength, 0.0f, 0.0f},
        .volume_quality = {frame->vision.center_steps, frame->vision.middle_steps, frame->vision.peripheral_steps, 0u},
        .volume_strides = {frame->vision.center_stride, frame->vision.middle_stride, frame->vision.peripheral_stride, 0u},
        .shadow_texel_enabled = {r->dynamic_shadow_size ? 1.0f / (float)r->dynamic_shadow_size : 1.0f,
                                 r->dynamic_shadow_size ? 1.0f / (float)r->dynamic_shadow_size : 1.0f,
                                 r->dynamic_shadow_ready ? 1.0f : 0.0f, 0.0f}};

    (void)dynamic_shadow_projection(r, frame, u.shadow_u_min, u.shadow_v_min, u.shadow_sun_max, u.shadow_extent_bias);

    if (!bind_volume_resources(r, cmd, fx->normal_depth, fx->depth_sampler, probes, beams, fx->volume, &u, sizeof(u))) return false;

    r->gpu->core.CmdSetPipeline(cmd, fx->volume_pipeline);

    r->gpu->core.CmdDispatch(cmd,
                             &(NriDispatchDesc){.workGroupNumX = (fx->ao_width + 7u) / 8u, .workGroupNumY = (fx->ao_height + 7u) / 8u, .workGroupNumZ = 1});

    const VOLUME_COMPOSE_UNIFORMS compose = {
        .width = fx->width,
        .height = fx->height,
        .debug_view = fx->debug_view,
        .bypass_volume = 0u,
        .volume_radii = {frame->vision.center_radius, frame->vision.middle_radius, frame->vision.center_transition_width,
                         frame->vision.middle_transition_width},
        .volume_filter = {frame->vision.jitter_strength, frame->vision.volume_blur_strength, 0.0f, 0.0f},
        .volume_strides = {frame->vision.center_stride, frame->vision.middle_stride, frame->vision.peripheral_stride, 0u},
    };

    if (!bind_volume_compose_resources(r, cmd, fx->hdr, fx->volume, fx->normal_depth, fx->sampler, fx->depth_sampler, fx->lit, &compose, sizeof(compose)))
        return false;

    r->gpu->core.CmdSetPipeline(cmd, fx->volume_compose_pipeline);

    r->gpu->core.CmdDispatch(cmd, &(NriDispatchDesc){.workGroupNumX = (fx->width + 7u) / 8u, .workGroupNumY = (fx->height + 7u) / 8u, .workGroupNumZ = 1});

    fx->volume_ready = true;

    return true;
}

static bool run_vision_only(FX_STATE *fx, NriCommandBuffer *cmd) {
    if (!fx || !fx->owner || !cmd) return false;

    RENDERER *r = fx->owner;

    const VOLUME_COMPOSE_UNIFORMS compose = {
        .width = fx->width,
        .height = fx->height,
        .debug_view = fx->debug_view,
        .bypass_volume = 1u,
    };

    if (!bind_volume_compose_resources(r, cmd, fx->hdr, fx->volume, fx->normal_depth, fx->sampler, fx->depth_sampler, fx->lit, &compose, sizeof(compose)))
        return false;

    r->gpu->core.CmdSetPipeline(cmd, fx->volume_compose_pipeline);

    r->gpu->core.CmdDispatch(cmd, &(NriDispatchDesc){.workGroupNumX = (fx->width + 7u) / 8u, .workGroupNumY = (fx->height + 7u) / 8u, .workGroupNumZ = 1});

    return true;
}

static bool bloom_pass(FX_STATE *fx, NriCommandBuffer *cmd, NriTexture *source, NriTexture *destination, Uint32 src_width, Uint32 src_height, Uint32 phase) {
    const BLOOM_UNIFORMS u = {.src_width = src_width,
                              .src_height = src_height,
                              .dst_width = fx->ao_width,
                              .dst_height = fx->ao_height,
                              .phase = phase,
                              .threshold = 1.0f,
                              .knee = 0.55f,
                              .strength = 1.0f * 1e2};

    return dispatch_one(fx, cmd, fx->bloom_pipeline, source, destination, &u, sizeof(u), fx->ao_width, fx->ao_height);
}

static bool fx_apply_base(FX_STATE *fx, NriCommandBuffer *cmd, NriTexture *swap, float tan_half_fov, float aspect) {
    if (!fx || !fx->owner || !cmd || !swap || !fx->hdr || !fx->normal_depth) return false;

    RENDERER *r = fx->owner;

    const SSAO_UNIFORMS ao = {.width = fx->width,
                              .height = fx->height,
                              .ao_width = fx->ao_width,
                              .ao_height = fx->ao_height,
                              .tan_half_fov = tan_half_fov,
                              .aspect = aspect,
                              .radius = 0.65f,
                              .bias = 0.035f};

    NriTexture *hdr = fx->volume_ready ? fx->lit : fx->hdr;

    if (!dispatch_one(fx, cmd, fx->ssao_pipeline, fx->normal_depth, fx->ao, &ao, sizeof(ao), fx->ao_width, fx->ao_height) ||
        !bloom_pass(fx, cmd, fx->hdr, fx->bloom_a, fx->width, fx->height, 0u) ||
        !bloom_pass(fx, cmd, fx->bloom_a, fx->bloom_b, fx->ao_width, fx->ao_height, 1u) ||
        !bloom_pass(fx, cmd, fx->bloom_b, fx->bloom_a, fx->ao_width, fx->ao_height, 2u))
        return false;

    const COMPOSE_UNIFORMS u = {.exposure = 1.0f, .ao_strength = fx->debug_view >= 3u ? 0.0f : 0.62f, .bloom_strength = fx->debug_view >= 3u ? 0.0f : 0.22f};

    if (!begin_compose_rendering(r, cmd, swap, hdr, fx->ao, fx->bloom_a, fx->lut, fx->sampler, &u, sizeof(u))) return false;

    r->gpu->core.CmdSetPipeline(cmd, fx->compose_pipeline);
    r->gpu->core.CmdDraw(cmd, &(NriDrawDesc){.vertexNum = 3, .instanceNum = 1, .baseVertex = 0, .baseInstance = 0});

    r->gpu->core.CmdEndRendering(cmd);

    return true;
}

static bool fx_apply(FX_STATE *fx, NriCommandBuffer *cmd, NriTexture *swap, float tan_half_fov, float aspect) {
    if (!fx || !cmd || !swap) return false;

    const bool had_volume = fx->volume_ready;

    if (!had_volume) {
        if (!run_vision_only(fx, cmd)) return false;
        fx->volume_ready = true;
    }

    const bool ok = fx_apply_base(fx, cmd, swap, tan_half_fov, aspect);

    if (!had_volume) fx->volume_ready = false;

    return ok;
}

static bool create_pipeline_layouts(RENDERER *r) {
    return create_surface_layout(r) && create_dynamic_shadow_layout(r) && create_dynamic_surface_layout(r) && create_line_layout(r) && create_sky_layout(r) && bake_gpu_layouts_init(r) &&
           create_ssao_layout(r) && create_bloom_layout(r) && create_grade_layout(r) && create_volume_layout(r) && create_volume_compose_layout(r) &&
           create_compose_layout(r);
}

static void destroy_pipeline_layouts(RENDERER *r) {
    if (!r) return;

    bake_gpu_layouts_deinit(r);

    NriPipelineLayout **layouts[] = {&r->surface_layout, &r->dynamic_shadow_layout, &r->dynamic_surface_layout, &r->line_layout, &r->sky_layout, &r->ssao_layout,
                                     &r->bloom_layout,   &r->grade_layout,          &r->volume_layout, &r->volume_compose_layout,
                                     &r->compose_layout};

    for (uint32_t i = 0; i < sizeof(layouts) / sizeof(layouts[0]); ++i) {
        if (*layouts[i]) {
            r->gpu->core.DestroyPipelineLayout(*layouts[i]);
            *layouts[i] = NULL;
        }
    }
}

bool renderer_gpu_resources_init(RENDERER *r) {
    if (!r || !r->gpu || !r->gpu->device) return false;

    r->show_volume = true;
    r->yaw = -0.78f;
    r->pitch = 0.34f;
    r->distance = 14.0f;
    r->target = v3(0.0f, 1.0f, 0.0f);

    if (!create_pipeline_layouts(r)) {
        renderer_gpu_resources_deinit(r);
        return false;
    }

    if (r->gpu->core.GetFormatSupport(r->gpu->device, NriFormat_D32_SFLOAT) & NriFormatSupportBits_DEPTH_STENCIL_ATTACHMENT) {
        r->depth_format = NriFormat_D32_SFLOAT;
    } else if (r->gpu->core.GetFormatSupport(r->gpu->device, NriFormat_D24_UNORM_S8_UINT) & NriFormatSupportBits_DEPTH_STENCIL_ATTACHMENT) {
        r->depth_format = NriFormat_D24_UNORM_S8_UINT;
    } else {
        r->depth_format = NriFormat_D16_UNORM;
    }

    NriShaderDesc surface_vs = gpu_load_shader("surface_vs", "BUILD_SURFACE_VS", NriStageBits_VERTEX_SHADER);
    NriShaderDesc surface_ps = gpu_load_shader("surface_fs", "BUILD_SURFACE_FS", NriStageBits_FRAGMENT_SHADER);
    NriShaderDesc dynamic_shadow_vs = gpu_load_shader("dynamic_shadow_vs", "BUILD_DYNAMIC_SHADOW_VS", NriStageBits_VERTEX_SHADER);
    NriShaderDesc line_vs = gpu_load_shader("wireframe_vs", "BUILD_WIREFRAME_VS", NriStageBits_VERTEX_SHADER);
    NriShaderDesc line_ps = gpu_load_shader("wireframe_fs", "BUILD_WIREFRAME_FS", NriStageBits_FRAGMENT_SHADER);
    NriShaderDesc sky_vs = gpu_load_shader("fullscreen_vs", "BUILD_FULLSCREEN_VS", NriStageBits_VERTEX_SHADER);
    NriShaderDesc sky_ps = gpu_load_shader("sky_fs", "BUILD_SKY_FS", NriStageBits_FRAGMENT_SHADER);

    if (!surface_vs.bytecode || !surface_ps.bytecode || !dynamic_shadow_vs.bytecode || !line_vs.bytecode || !line_ps.bytecode || !sky_vs.bytecode ||
        !sky_ps.bytecode) {
        gpu_free_shader(&surface_vs);
        gpu_free_shader(&surface_ps);
        gpu_free_shader(&dynamic_shadow_vs);
        gpu_free_shader(&line_vs);
        gpu_free_shader(&line_ps);
        gpu_free_shader(&sky_vs);
        gpu_free_shader(&sky_ps);
        renderer_gpu_resources_deinit(r);
        return false;
    }

    r->solid_pipeline = make_surface_pipeline(r, &r->gpu->core, r->surface_layout, &surface_vs, &surface_ps, false);
    r->transmission_pipeline = make_surface_pipeline(r, &r->gpu->core, r->surface_layout, &surface_vs, &surface_ps, true);
    r->dynamic_shadow_pipeline = make_dynamic_shadow_pipeline(r, r->dynamic_shadow_layout, &dynamic_shadow_vs);
    r->dynamic_surface_pipeline =
        gpu_compile_compute(r, r->dynamic_surface_layout, "shaders/dynamic_surface.hlsl", "dynamic_surface_cs", "BUILD_DYNAMIC_SURFACE_CS");
    r->line_pipeline = make_line_pipeline(r, r->line_layout, &line_vs, &line_ps);
    r->sky_pipeline = make_sky_pipeline(r, r->sky_layout, &sky_vs, &sky_ps);

    gpu_free_shader(&surface_vs);
    gpu_free_shader(&surface_ps);
    gpu_free_shader(&dynamic_shadow_vs);
    gpu_free_shader(&line_vs);
    gpu_free_shader(&line_ps);
    gpu_free_shader(&sky_vs);
    gpu_free_shader(&sky_ps);

    r->dynamic_shadow_sampler = gpu_create_sampler(r, NriFilter_NEAREST, NriFilter_NEAREST, NriAddressMode_CLAMP_TO_EDGE);

    const PROBE fallback_probe = {0};
    const float fallback_beam = 1.0f;
    r->surface_probe_fallback_buffer =
        gpu_upload_buffer(r, NriBufferUsageBits_SHADER_RESOURCE, &fallback_probe, sizeof(fallback_probe), sizeof(fallback_probe));
    r->surface_beam_fallback_buffer =
        gpu_upload_buffer(r, NriBufferUsageBits_SHADER_RESOURCE, &fallback_beam, sizeof(fallback_beam), sizeof(fallback_beam));

    if (!r->solid_pipeline || !r->transmission_pipeline || !r->dynamic_shadow_pipeline || !r->dynamic_surface_pipeline || !r->dynamic_shadow_sampler ||
        !r->surface_probe_fallback_buffer || !r->surface_beam_fallback_buffer || !r->line_pipeline || !r->sky_pipeline || !fx_init(&r->fx, r)) {
        renderer_gpu_resources_deinit(r);
        return false;
    }

    SDL_Log("GPU backend: NRI");
    SDL_Log("depth format: %s", r->depth_format == NriFormat_D32_SFLOAT ? "D32_FLOAT" : r->depth_format == NriFormat_D24_UNORM_S8_UINT ? "D24S8" : "D16_UNORM");
    SDL_Log("SDL_image: %d", IMG_Version());

    return true;
}

static bool material_transmissive(const RENDER_MATERIAL *material) {
    return material && material->data.transmission_factor > 0.0f;
}

static VEC3 draw_world_center(const RENDERER *r, const DRAW_RANGE *draw) {
    if (!r || !draw || !draw->object_id || !r->scene) return draw ? draw->center : v3(0.0f, 0.0f, 0.0f);

    const OBJECT *object = scene_object_by_id_const(r->scene, draw->object_id);
    return object ? m4_point(m4_transform(object->transform, false), draw->center) : draw->center;
}

static float draw_distance_sq(const RENDERER *r, const DRAW_RANGE *draw, VEC3 eye) {
    VEC3 delta = v3_sub(draw_world_center(r, draw), eye);
    return v3_len_sq(delta);
}

static void sort_transmission_draws(RENDERER *r, VEC3 eye) {
    for (uint32_t i = 1; i < r->transmission_draw_count; ++i) {
        DRAW_RANGE value = r->transmission_draws[i];
        float distance = draw_distance_sq(r, &value, eye);
        uint32_t j = i;

        while (j > 0 && draw_distance_sq(r, &r->transmission_draws[j - 1u], eye) < distance) {
            r->transmission_draws[j] = r->transmission_draws[j - 1u];
            --j;
        }

        r->transmission_draws[j] = value;
    }
}

static bool build_transmission_draws(RENDERER *r) {
    free(r->transmission_draws);
    r->transmission_draws = NULL;
    r->transmission_draw_count = 0;

    if (!r->has_transmission) return true;

    r->transmission_draws = calloc(r->draw_count, sizeof(*r->transmission_draws));
    if (!r->transmission_draws) return false;

    for (uint32_t i = 0; i < r->draw_count; ++i) {
        const DRAW_RANGE *draw = &r->draws[i];

        if (material_transmissive(&r->materials[draw->material]))
            r->transmission_draws[r->transmission_draw_count++] = *draw;
    }

    return true;
}

static bool dynamic_shadow_projection(const RENDERER *r, const RENDER_FRAME *frame, float shadow_u_min[4], float shadow_v_min[4],
                                      float shadow_sun_max[4], float shadow_extent_bias[4]) {
    if (!r || !frame || !r->beams.width || !r->beams.height || !r->beams.depth || r->beams.step.x <= 0.0f || r->beams.step.y <= 0.0f ||
        r->beams.step.z <= 0.0f)
        return false;

    const VEC3 sun = v3_normalize(frame->sun.direction);
    VEC3 u = v3_normalize(v3_cross(v3(0.0f, 1.0f, 0.0f), sun));

    if (v3_len_sq(u) < 0.5f) u = v3_normalize(v3_cross(v3(1.0f, 0.0f, 0.0f), sun));
    if (v3_len_sq(u) < 0.5f) return false;

    const VEC3 v = v3_cross(sun, u);
    const float span_x = r->beams.step.x * (float)r->beams.width;
    const float span_y = r->beams.step.y * (float)r->beams.height;
    const float span_z = r->beams.step.z * (float)r->beams.depth;

    if (span_x <= 0.0f || span_y <= 0.0f || span_z <= 0.0f) return false;

    shadow_u_min[0] = u.x;
    shadow_u_min[1] = u.y;
    shadow_u_min[2] = u.z;
    shadow_u_min[3] = r->beams.origin.x;
    shadow_v_min[0] = v.x;
    shadow_v_min[1] = v.y;
    shadow_v_min[2] = v.z;
    shadow_v_min[3] = r->beams.origin.y;
    shadow_sun_max[0] = sun.x;
    shadow_sun_max[1] = sun.y;
    shadow_sun_max[2] = sun.z;
    shadow_sun_max[3] = r->beams.origin.z + span_z;
    shadow_extent_bias[0] = span_x;
    shadow_extent_bias[1] = span_y;
    shadow_extent_bias[2] = span_z;
    shadow_extent_bias[3] = 2.0f / (float)DYNAMIC_SHADOW_SIZE;
    return true;
}

static bool render_dynamic_shadow_map(RENDERER *r, NriCommandBuffer *cmd, const RENDER_FRAME *frame) {
    if (!r || !cmd || !frame) return false;

    r->dynamic_shadow_ready = false;

    bool has_dynamic = false;
    for (uint32_t i = 0; i < r->draw_count; ++i) {
        if (r->draws[i].object_id && !material_transmissive(&r->materials[r->draws[i].material])) {
            has_dynamic = true;
            break;
        }
    }

    if (!has_dynamic) return true;

    DYNAMIC_SHADOW_UNIFORMS base = {0};

    if (!dynamic_shadow_projection(r, frame, base.shadow_u_min, base.shadow_v_min, base.shadow_sun_max, base.shadow_extent)) return true;
    if (!begin_dynamic_shadow_rendering(r, cmd)) return false;

    const NriVertexBufferDesc vertex = {.buffer = r->vertex_buffer, .offset = 0, .stride = sizeof(RENDER_VERTEX)};
    r->gpu->core.CmdSetVertexBuffers(cmd, 0, &vertex, 1);
    r->gpu->core.CmdSetPipeline(cmd, r->dynamic_shadow_pipeline);

    for (uint32_t i = 0; i < r->draw_count; ++i) {
        const DRAW_RANGE *draw = &r->draws[i];

        if (!draw->object_id || material_transmissive(&r->materials[draw->material])) continue;

        const OBJECT *object = scene_object_by_id_const(r->scene, draw->object_id);
        if (!object || object->state != DYNAMIC) {
            r->gpu->core.CmdEndRendering(cmd);
            return false;
        }

        DYNAMIC_SHADOW_UNIFORMS uniforms = base;
        const MAT4 model = m4_transform(object->transform, false);
        memcpy(uniforms.model, model.m, sizeof(uniforms.model));

        if (!gpu_bind_uniform_data(r, cmd, r->dynamic_shadow_layout, NriBindPoint_GRAPHICS, 1, &uniforms, sizeof(uniforms))) {
            r->gpu->core.CmdEndRendering(cmd);
            return false;
        }

        r->gpu->core.CmdDraw(cmd, &(NriDrawDesc){.vertexNum = draw->count, .instanceNum = 1, .baseVertex = draw->first, .baseInstance = 0});
    }

    r->gpu->core.CmdEndRendering(cmd);

    if (!gpu_transition_texture(r, cmd, r->dynamic_shadow_texture, NriAccessBits_SHADER_RESOURCE, NriLayout_SHADER_RESOURCE, NriStageBits_FRAGMENT_SHADER))
        return false;

    r->dynamic_shadow_ready = true;
    return true;
}

static bool update_dynamic_surface_caches(RENDERER *r, NriCommandBuffer *cmd, const RENDER_FRAME *frame) {
    if (!r || !cmd || !frame) return false;
    if (r->reference_lighting_enabled || !r->has_bake || !r->dynamic_lighting_count) return true;
    if (!r->dynamic_surface_pipeline || !r->beam_buffer || !r->dynamic_static_node_buffer || !r->dynamic_static_triangle_buffer ||
        !r->dynamic_static_surface_buffer || !r->dynamic_static_uv_buffer)
        return true;

    uint32_t dirty_count = 0u;

    for (uint32_t i = 0; i < r->dynamic_lighting_count; ++i) {
        DYNAMIC_LIGHTING_ALLOCATION *allocation = &r->dynamic_lighting[i];
        const OBJECT *object = scene_object_by_id_const(r->scene, allocation->object_id);

        if (!object || object->state != DYNAMIC || object->type != MODEL || !allocation->layout || !allocation->layout->sample_count)
            return false;

        if (allocation->pending_transform_revision != object->transform_revision ||
            allocation->pending_lighting_revision != object->lighting_revision) {
            allocation->pending_transform_revision = object->transform_revision;
            allocation->pending_lighting_revision = object->lighting_revision;
            allocation->sample_cursor = 0u;
            allocation->cache_needs_clear = true;
        }

        if (allocation->cache_needs_clear || allocation->sample_cursor < allocation->layout->sample_count ||
            allocation->transform_revision != object->transform_revision || allocation->lighting_revision != object->lighting_revision)
            ++dirty_count;
    }

    uint32_t budget = DYNAMIC_SURFACE_SAMPLES_PER_FRAME;
    uint32_t dirty_left = dirty_count;

    for (uint32_t i = 0; i < r->dynamic_lighting_count && budget && dirty_left; ++i) {
        DYNAMIC_LIGHTING_ALLOCATION *allocation = &r->dynamic_lighting[i];
        const OBJECT *object = scene_object_by_id_const(r->scene, allocation->object_id);

        if (!object) return false;

        const uint32_t total = allocation->layout->sample_count;
        if (!allocation->cache_needs_clear && allocation->sample_cursor >= total &&
            allocation->transform_revision == object->transform_revision && allocation->lighting_revision == object->lighting_revision)
            continue;
        if (allocation->sample_cursor > total) allocation->sample_cursor = 0u;

        uint32_t quota = (budget + dirty_left - 1u) / dirty_left;
        uint32_t remaining = total - allocation->sample_cursor;
        uint32_t count = remaining < quota ? remaining : quota;

        const MAT4 model = m4_transform(object->transform, false);
        const MAT4 inverse_model = m4_inverse_transform(object->transform);
        const MAT4 normal_model = m4_transform(object->transform, true);
        DYNAMIC_SURFACE_UNIFORMS uniforms = {
            .sample_offset = allocation->sample_cursor,
            .sample_count = count,
            .texture_width = allocation->layout->width,
            .texture_height = allocation->layout->height,
            .beam_origin = {r->beams.origin.x, r->beams.origin.y, r->beams.origin.z, 0.0f},
            .beam_step = {r->beams.step.x, r->beams.step.y, r->beams.step.z, 0.0f},
            .beam_dims = {r->beams.width, r->beams.height, r->beams.depth, 1u},
            .sun_direction_intensity = {frame->sun.direction.x, frame->sun.direction.y, frame->sun.direction.z, frame->sun.intensity},
            .sun_color_visibility_floor = {frame->sun.color.x, frame->sun.color.y, frame->sun.color.z, 1.0f / 1024.0f},
            .sky_zenith = {frame->sky.zenith.x, frame->sky.zenith.y, frame->sky.zenith.z, frame->sky.intensity},
            .sky_horizon = {frame->sky.horizon.x, frame->sky.horizon.y, frame->sky.horizon.z, 1.0f},
            .trace_params = {fmaxf(r->scene_radius * 2.0e-5f, 1.0e-5f), 0.0f, 0.0f, 0.0f},
        };

        memcpy(uniforms.model, model.m, sizeof(uniforms.model));
        memcpy(uniforms.inverse_model, inverse_model.m, sizeof(uniforms.inverse_model));
        memcpy(uniforms.normal_model, normal_model.m, sizeof(uniforms.normal_model));

        if (allocation->cache_needs_clear) {
            DYNAMIC_SURFACE_UNIFORMS clear = uniforms;
            clear.sample_offset = 0u;
            clear.sample_count = total;
            clear.trace_params[1] = 1.0f;

            if (!bind_dynamic_surface_resources(r, cmd, allocation, &clear)) return false;

            r->gpu->core.CmdSetPipeline(cmd, r->dynamic_surface_pipeline);
            r->gpu->core.CmdDispatch(cmd, &(NriDispatchDesc){.workGroupNumX = (clear.sample_count + 63u) / 64u, .workGroupNumY = 1u, .workGroupNumZ = 1u});

            const NriAccessLayoutStage storage = {
                .access = NriAccessBits_SHADER_RESOURCE_STORAGE,
                .layout = NriLayout_SHADER_RESOURCE_STORAGE,
                .stages = NriStageBits_COMPUTE_SHADER,
            };

            if (!gpu_texture_barrier(r, cmd, allocation->texture, storage, storage)) return false;

            allocation->cache_needs_clear = false;
            allocation->transform_revision = object->transform_revision;
            allocation->lighting_revision = object->lighting_revision;
        }

        if (count) {
            uniforms.sample_offset = allocation->sample_cursor;
            uniforms.sample_count = count;

            if (!bind_dynamic_surface_resources(r, cmd, allocation, &uniforms)) return false;

            r->gpu->core.CmdSetPipeline(cmd, r->dynamic_surface_pipeline);
            r->gpu->core.CmdDispatch(cmd, &(NriDispatchDesc){.workGroupNumX = (count + 63u) / 64u, .workGroupNumY = 1u, .workGroupNumZ = 1u});

            allocation->sample_cursor += count;
            budget -= count;
        }

        if (!gpu_transition_texture(r, cmd, allocation->texture, NriAccessBits_SHADER_RESOURCE, NriLayout_SHADER_RESOURCE,
                                    NriStageBits_FRAGMENT_SHADER))
            return false;

        --dirty_left;
    }

    return true;
}

static uint32_t dynamic_influences(const RENDERER *r, MATERIAL_UNIFORMS *uniforms) {
    if (!r || !r->scene || !uniforms) return 0u;

    uint32_t count = 0u;

    for (uint32_t i = 0; i < r->dynamic_lighting_count && count < DYNAMIC_INFLUENCE_LIMIT; ++i) {
        const DYNAMIC_LIGHTING_ALLOCATION *allocation = &r->dynamic_lighting[i];
        const OBJECT *object = scene_object_by_id_const(r->scene, allocation->object_id);

        if (!object || object->state != DYNAMIC || object->type != MODEL || allocation->local_radius <= 0.0f) continue;

        const MAT4 model = m4_transform(object->transform, false);
        const VEC3 center = m4_point(model, allocation->local_center);
        const VEC3 axis_x =
            v3(model.m[0] * allocation->local_extents.x, model.m[1] * allocation->local_extents.x, model.m[2] * allocation->local_extents.x);
        const VEC3 axis_y =
            v3(model.m[4] * allocation->local_extents.y, model.m[5] * allocation->local_extents.y, model.m[6] * allocation->local_extents.y);
        const VEC3 axis_z =
            v3(model.m[8] * allocation->local_extents.z, model.m[9] * allocation->local_extents.z, model.m[10] * allocation->local_extents.z);
        const float radius = fmaxf(sqrtf(v3_len_sq(axis_x) + v3_len_sq(axis_y) + v3_len_sq(axis_z)), 1.0e-3f);

        uniforms->dynamic_influence_center_radius[count][0] = center.x;
        uniforms->dynamic_influence_center_radius[count][1] = center.y;
        uniforms->dynamic_influence_center_radius[count][2] = center.z;
        uniforms->dynamic_influence_center_radius[count][3] = radius;

        uniforms->dynamic_influence_axis_x[count][0] = axis_x.x;
        uniforms->dynamic_influence_axis_x[count][1] = axis_x.y;
        uniforms->dynamic_influence_axis_x[count][2] = axis_x.z;

        uniforms->dynamic_influence_axis_y[count][0] = axis_y.x;
        uniforms->dynamic_influence_axis_y[count][1] = axis_y.y;
        uniforms->dynamic_influence_axis_y[count][2] = axis_y.z;

        uniforms->dynamic_influence_axis_z[count][0] = axis_z.x;
        uniforms->dynamic_influence_axis_z[count][1] = axis_z.y;
        uniforms->dynamic_influence_axis_z[count][2] = axis_z.z;

        uniforms->dynamic_influence_diffuse[count][0] = allocation->average_diffuse.x;
        uniforms->dynamic_influence_diffuse[count][1] = allocation->average_diffuse.y;
        uniforms->dynamic_influence_diffuse[count][2] = allocation->average_diffuse.z;
        uniforms->dynamic_influence_diffuse[count][3] = 1.0f;

        uniforms->dynamic_influence_emissive[count][0] = allocation->average_emissive.x;
        uniforms->dynamic_influence_emissive[count][1] = allocation->average_emissive.y;
        uniforms->dynamic_influence_emissive[count][2] = allocation->average_emissive.z;
        uniforms->dynamic_influence_emissive[count][3] = 0.0f;
        ++count;
    }

    uniforms->dynamic_influence_meta[0] = count;
    return count;
}

static MATERIAL_UNIFORMS material_uniforms(const RENDERER *r, const RENDER_MATERIAL *material, const RENDER_FRAME *frame, const DRAW_RANGE *draw) {
    float dynamic_cache_valid = 0.0f;

    if (r && r->scene && draw && draw->object_id) {
        const DYNAMIC_LIGHTING_ALLOCATION *allocation = dynamic_lighting_find_const(r, draw->object_id);
        const OBJECT *object = scene_object_by_id_const(r->scene, draw->object_id);

        if (allocation && object) {
            if (r->reference_lighting_enabled) {
                if (allocation->reference_texture && allocation->reference_transform_revision == object->transform_revision &&
                    allocation->reference_lighting_revision == object->lighting_revision)
                    dynamic_cache_valid = 1.0f;
            } else if (allocation->texture && allocation->transform_revision == object->transform_revision &&
                       allocation->lighting_revision == object->lighting_revision) {
                dynamic_cache_valid = 1.0f;
            }
        }
    }

    MATERIAL_UNIFORMS result = (MATERIAL_UNIFORMS){
        .base_color_factor = {material->data.base_color[0], material->data.base_color[1], material->data.base_color[2], material->data.base_color[3]},
        .emissive_metallic = {material->data.emissive[0], material->data.emissive[1], material->data.emissive[2], material->data.metallic},
        .roughness_normal_ao_sun = {material->data.roughness, material->data.normal_scale, material->data.occlusion_strength, frame->sun.intensity},
        .sun_direction = {frame->sun.direction.x, frame->sun.direction.y, frame->sun.direction.z, frame->sun.intensity},
        .sun_color = {frame->sun.color.x, frame->sun.color.y, frame->sun.color.z, frame->sun.angular_radius},
        .camera_position = {frame->eye.x, frame->eye.y, frame->eye.z, r->debug_view == 1u ? 2.0f : (r->has_bake ? 1.0f : 0.0f)},
        .ior_transmission_volume = {material->data.ior, material->data.transmission_factor, material->data.thickness_factor, material->data.attenuation_distance},
        .attenuation_iridescence = {material->data.attenuation_color[0], material->data.attenuation_color[1], material->data.attenuation_color[2],
                                    material->data.iridescence_factor},
        .iridescence_params = {material->data.iridescence_ior, material->data.iridescence_thickness_min, material->data.iridescence_thickness_max, 0.0f},
        .camera_right_tan = {frame->right.x, frame->right.y, frame->right.z, frame->tan_half_fov * frame->aspect},
        .camera_up_tan = {frame->up.x, frame->up.y, frame->up.z, frame->tan_half_fov},
        .camera_forward = {frame->forward.x, frame->forward.y, frame->forward.z, 0.0f},
        .sky_zenith = {frame->sky.zenith.x, frame->sky.zenith.y, frame->sky.zenith.z, frame->sky.intensity},
        .sky_horizon = {frame->sky.horizon.x, frame->sky.horizon.y, frame->sky.horizon.z, 1.0f},
        .shadow_texel_enabled = {r->dynamic_shadow_size ? 1.0f / (float)r->dynamic_shadow_size : 1.0f,
                                 r->dynamic_shadow_size ? 1.0f / (float)r->dynamic_shadow_size : 1.0f,
                                 r->dynamic_shadow_ready ? 1.0f : 0.0f, 0.0f},
        .dynamic_flags = {draw && draw->object_id ? 1.0f : 0.0f, dynamic_cache_valid, r->reference_lighting_enabled ? 1.0f : 0.0f, 0.0f},
        .probe_origin_spacing = {r->volume_probes.origin.x, r->volume_probes.origin.y, r->volume_probes.origin.z, r->volume_probes.spacing},
        .probe_dims = {r->volume_probes.count_x, r->volume_probes.count_y, r->volume_probes.count_z, r->volume_probe_buffer ? 1u : 0u},
        .beam_origin = {r->beams.origin.x, r->beams.origin.y, r->beams.origin.z, 0.0f},
        .beam_step = {r->beams.step.x, r->beams.step.y, r->beams.step.z, 0.0f},
        .beam_dims = {r->beams.width, r->beams.height, r->beams.depth, r->beam_buffer ? 1u : 0u}};

    (void)dynamic_shadow_projection(r, frame, result.shadow_u_min, result.shadow_v_min, result.shadow_sun_max, result.shadow_extent_bias);
    (void)dynamic_influences(r, &result);
    return result;
}

static CAMERA_UNIFORMS camera_uniforms_for_draw(const RENDERER *r, const RENDER_FRAME *frame, const DRAW_RANGE *draw) {
    CAMERA_UNIFORMS camera = {0};
    MAT4 model = m4_identity();
    MAT4 normal_model = m4_identity();

    memcpy(camera.mvp, frame->mvp, sizeof(camera.mvp));
    memcpy(camera.view, frame->view, sizeof(camera.view));

    if (r && r->scene && draw && draw->object_id) {
        const OBJECT *object = scene_object_by_id_const(r->scene, draw->object_id);

        if (object) {
            model = m4_transform(object->transform, false);
            normal_model = m4_transform(object->transform, true);
        }
    }

    memcpy(camera.model, model.m, sizeof(camera.model));
    memcpy(camera.normal_model, normal_model.m, sizeof(camera.normal_model));
    return camera;
}

static bool draw_surface_range(RENDERER *r, NriCommandBuffer *cmd, const RENDER_FRAME *frame, const DRAW_RANGE *draw, NriTexture *scene_color,
                               NriDescriptor *scene_sampler) {
    const RENDER_MATERIAL *material = &r->materials[draw->material];
    const MATERIAL_UNIFORMS uniforms = material_uniforms(r, material, frame, draw);
    const CAMERA_UNIFORMS camera = camera_uniforms_for_draw(r, frame, draw);
    NriTexture *lighting = r->reference_lighting_enabled && r->reference_static_texture ? r->reference_static_texture : r->lightmap_texture;

    if (draw->object_id) {
        DYNAMIC_LIGHTING_ALLOCATION *allocation = dynamic_lighting_find(r, draw->object_id);

        if (uniforms.dynamic_flags[1] > 0.5f && allocation) {
            if (r->reference_lighting_enabled && allocation->reference_texture)
                lighting = allocation->reference_texture;
            else if (!r->reference_lighting_enabled && allocation->texture)
                lighting = allocation->texture;
        } else {
            lighting = r->default_white;
        }
    }

    if (!bind_camera_resources(r, cmd, &camera, sizeof(camera)) ||
        !bind_surface_resources(r, cmd, material, lighting, scene_color, r->material_sampler, r->lightmap_sampler, scene_sampler, &uniforms,
                                sizeof(uniforms)))
        return false;

    r->gpu->core.CmdDraw(cmd, &(NriDrawDesc){.vertexNum = draw->count, .instanceNum = 1, .baseVertex = draw->first, .baseInstance = 0});
    return true;
}

static bool draw_frame(RENDERER *r, const RENDER_FRAME *frame) {
    if (!r || !frame || !r->gpu->device || !r->solid_pipeline || !r->sky_pipeline || !r->vertex_buffer || !r->lightmap_texture || !r->lightmap_sampler)
        return false;

    uint32_t width = 0;
    uint32_t height = 0;

    SDL_GetWindowSizeInPixels(r->gpu->window, (int *)&width, (int *)&height);

    if (!width || !height) return true;

    if (!gpu_ensure_swapchain(r, width, height)) return false;

    if (!fx_ensure(&r->fx, width, height) || !ensure_depth_texture(r, width, height)) return false;

    uint32_t swap_index = 0;
    NriTexture *swap = NULL;
    FRAME_CONTEXT *queued_frame = NULL;
    NriCommandBuffer *cmd = NULL;

    if (!gpu_begin_render_frame(r, &queued_frame, &cmd, &swap, &swap_index)) goto failed_frame;

    const uint32_t timing_base =
        DYNAMIC_TIMESTAMP_BASE + (uint32_t)(r->gpu->frame_index % GPU_FRAME_QUEUE_DEPTH) * DYNAMIC_TIMESTAMP_STRIDE;

    if (r->gpu->frame_index >= GPU_FRAME_QUEUE_DEPTH && r->gpu->frame_index % DYNAMIC_TIMING_LOG_INTERVAL == 0u) {
        gpu_timestamp_log(r, timing_base, "dynamic surface cache");
        gpu_timestamp_log(r, timing_base + 2u, "dynamic shadow map");
    }

    if (!gpu_timestamp_begin(r, cmd, timing_base)) goto failed_frame;
    if (!update_dynamic_surface_caches(r, cmd, frame)) goto failed_frame;
    if (!gpu_timestamp_end(r, cmd, timing_base)) goto failed_frame;

    if (!gpu_timestamp_begin(r, cmd, timing_base + 2u)) goto failed_frame;
    if (!render_dynamic_shadow_map(r, cmd, frame)) goto failed_frame;
    if (!gpu_timestamp_end(r, cmd, timing_base + 2u)) goto failed_frame;

    CAMERA_UNIFORMS camera = {0};

    memcpy(camera.mvp, frame->mvp, sizeof(camera.mvp));
    memcpy(camera.view, frame->view, sizeof(camera.view));
    MAT4 identity = m4_identity();
    memcpy(camera.model, identity.m, sizeof(camera.model));
    memcpy(camera.normal_model, identity.m, sizeof(camera.normal_model));

    const SKY_UNIFORMS sky = {.camera_right = {frame->right.x * frame->tan_half_fov * frame->aspect, frame->right.y * frame->tan_half_fov * frame->aspect,
                                               frame->right.z * frame->tan_half_fov * frame->aspect, 0},
                              .camera_up = {frame->up.x * frame->tan_half_fov, frame->up.y * frame->tan_half_fov, frame->up.z * frame->tan_half_fov, 0},
                              .camera_forward = {frame->forward.x, frame->forward.y, frame->forward.z, 0},
                              .sky_zenith = {frame->sky.zenith.x, frame->sky.zenith.y, frame->sky.zenith.z, frame->sky.intensity},
                              .sky_horizon = {frame->sky.horizon.x, frame->sky.horizon.y, frame->sky.horizon.z, 1.0f},
                              .sun_direction_intensity = {frame->sun.direction.x, frame->sun.direction.y, frame->sun.direction.z, frame->sun.intensity},
                              .sun_color_radius = {frame->sun.color.x, frame->sun.color.y, frame->sun.color.z, frame->sun.angular_radius}};

    if (!begin_surface_rendering(r, cmd, r->fx.hdr, r->fx.normal_depth, r->depth_texture, width, height, false)) goto failed_frame;

    if (!bind_sky_resources(r, cmd, &sky, sizeof(sky))) goto failed_frame;
    r->gpu->core.CmdSetPipeline(cmd, r->sky_pipeline);
    r->gpu->core.CmdDraw(cmd, &(NriDrawDesc){.vertexNum = 3, .instanceNum = 1, .baseVertex = 0, .baseInstance = 0});

    const NriVertexBufferDesc vertex = {.buffer = r->vertex_buffer, .offset = 0, .stride = sizeof(RENDER_VERTEX)};
    r->gpu->core.CmdSetVertexBuffers(cmd, 0, &vertex, 1);
    r->gpu->core.CmdSetPipeline(cmd, r->solid_pipeline);

    for (uint32_t i = 0; i < r->draw_count; ++i) {
        const DRAW_RANGE *draw = &r->draws[i];

        if (material_transmissive(&r->materials[draw->material])) continue;
        if (!draw_surface_range(r, cmd, frame, draw, r->default_white, r->material_sampler)) goto failed_frame;
    }

    if (r->show_debug && r->debug_vertex_count) {
        r->gpu->core.CmdSetPipeline(cmd, r->line_pipeline);

        if (!bind_line_resources(r, cmd, camera.mvp, sizeof(camera.mvp))) goto failed_frame;

        r->gpu->core.CmdDraw(cmd, &(NriDrawDesc){.vertexNum = r->debug_vertex_count, .instanceNum = 1, .baseVertex = r->debug_vertex_start, .baseInstance = 0});
    }

    r->gpu->core.CmdEndRendering(cmd);

    if (r->has_transmission) {
        if (!snapshot_scene_color(r, cmd) ||
            !begin_surface_rendering(r, cmd, r->fx.hdr, r->fx.normal_depth, r->depth_texture, width, height, true))
            goto failed_frame;

        r->gpu->core.CmdSetVertexBuffers(cmd, 0, &vertex, 1);
        r->gpu->core.CmdSetPipeline(cmd, r->transmission_pipeline);

        sort_transmission_draws(r, frame->eye);

        for (uint32_t i = 0; i < r->transmission_draw_count; ++i) {
            const DRAW_RANGE *draw = &r->transmission_draws[i];

            if (!draw_surface_range(r, cmd, frame, draw, r->fx.scene_color, r->fx.sampler)) goto failed_frame;
        }

        r->gpu->core.CmdEndRendering(cmd);
    }

    r->fx.volume_ready = false;
    r->fx.debug_view = r->debug_view;

    static uint32_t last_logged_view = UINT32_MAX;

    if (last_logged_view != r->fx.debug_view) {
        SDL_Log("GPU debug view: %u | fog: %d | bake: %d", r->fx.debug_view, r->show_volume, r->has_bake);

        last_logged_view = r->fx.debug_view;
    }

    if (r->show_volume && r->has_bake && (r->debug_view == 0u || r->debug_view >= 3u) && r->volume_probe_buffer && r->beam_buffer &&
        !fx_volume(&r->fx, cmd, r->volume_probe_buffer, r->beam_buffer, &r->volume_probes, &r->beams, frame))
        goto failed_frame;

    if (!fx_apply(&r->fx, cmd, swap, frame->tan_half_fov, frame->aspect)) goto failed_frame;

    if (!gpu_submit_render_frame(r, queued_frame, cmd, swap_index)) return false;

    return true;

failed_frame:
    gpu_abort_render_frame(r, queued_frame);

    return false;
}

void renderer_gpu_resources_deinit(RENDERER *r) {
    if (!r) return;

    GPU *gpu = r->gpu;
    free(r->vertices);
    free(r->draws);
    free(r->transmission_draws);
    release_dynamic_lighting(r);
    release_dynamic_static_transport(r);
    release_reference_lighting(r);
    free_probe_grid(&r->volume_probes);
    beam_free(&r->beams);

    if (gpu && gpu->device) {
        if (gpu->graphics_queue) gpu->core.QueueWaitIdle(gpu->graphics_queue);

        gpu_clear_temporary(r);
        probe_wavefront_scratch_destroy(r);
        fx_deinit(&r->fx);

        if (r->image_textures) {
            for (uint32_t i = 0; i < r->image_texture_count; ++i) release_texture(r, r->image_textures[i]);
        }

        free(r->image_textures);
        free(r->materials);
        release_texture(r, r->default_white);
        release_texture(r, r->default_normal);

        if (r->material_sampler) gpu->core.DestroyDescriptor(r->material_sampler);
        if (r->dynamic_shadow_sampler) gpu->core.DestroyDescriptor(r->dynamic_shadow_sampler);

        release_buffer(r, r->vertex_buffer);
        release_bake_resources(r);
        release_buffer(r, r->volume_probe_buffer);
        release_buffer(r, r->beam_buffer);
        release_buffer(r, r->surface_probe_fallback_buffer);
        release_buffer(r, r->surface_beam_fallback_buffer);
        release_texture(r, r->depth_texture);
        release_texture(r, r->dynamic_shadow_texture);
        release_texture(r, r->lightmap_texture);
        release_texture(r, r->baked_direct_texture);

        if (r->lightmap_sampler) gpu->core.DestroyDescriptor(r->lightmap_sampler);
        if (r->sky_pipeline) gpu->core.DestroyPipeline(r->sky_pipeline);
        if (r->solid_pipeline) gpu->core.DestroyPipeline(r->solid_pipeline);
        if (r->transmission_pipeline) gpu->core.DestroyPipeline(r->transmission_pipeline);
        if (r->dynamic_shadow_pipeline) gpu->core.DestroyPipeline(r->dynamic_shadow_pipeline);
        if (r->dynamic_surface_pipeline) gpu->core.DestroyPipeline(r->dynamic_surface_pipeline);
        if (r->line_pipeline) gpu->core.DestroyPipeline(r->line_pipeline);

        destroy_pipeline_layouts(r);
    }

    memset(r, 0, sizeof(*r));
}

bool renderer_load_cached_lightmap(RENDERER *renderer, const char *path, uint64_t scene_hash, uint64_t layout_hash, uint64_t volume_hash, uint64_t beam_hash,
                                   const LIGHTMAP *lightmap) {
    if (!renderer || !renderer->gpu->device || !lightmap) return false;

    CACHED_LIGHTMAP cached = {0};

    if (!cache_read(path, scene_hash, layout_hash, volume_hash, beam_hash, &cached)) return false;

    NriTexture *replacement = NULL;
    NriTexture *direct_replacement = NULL;
    NriBuffer *volume_buffer = NULL;
    NriBuffer *beam_buffer = NULL;
    bool good = cached.width == lightmap->width && cached.height == lightmap->height;

    if (good) {
        replacement = upload_lightmap(renderer, &cached);
        good = replacement != NULL;
    }

    if (good) {
        direct_replacement = upload_direct_lightmap(renderer, &cached);
        good = direct_replacement != NULL;
    }

    if (good) {
        volume_buffer = upload_probes(renderer, &cached.volume_probes);
        good = volume_buffer != NULL;
    }

    if (good) {
        beam_buffer = upload_beams(renderer, &cached.beams);
        good = beam_buffer != NULL;
    }

    if (good) {
        NriTexture *old = renderer->lightmap_texture;
        NriTexture *old_direct = renderer->baked_direct_texture;
        NriBuffer *old_volume = renderer->volume_probe_buffer;
        NriBuffer *old_beam = renderer->beam_buffer;

        renderer->lightmap_texture = replacement;
        renderer->baked_direct_texture = direct_replacement;
        renderer->volume_probe_buffer = volume_buffer;
        renderer->beam_buffer = beam_buffer;
        renderer->lightmap_width = cached.width;
        renderer->lightmap_height = cached.height;

        free_probe_grid(&renderer->volume_probes);
        renderer->volume_probes = cached.volume_probes;
        cached.volume_probes.probes = NULL;

        beam_free(&renderer->beams);
        renderer->beams = cached.beams;
        cached.beams.cells = NULL;
        cached.beams.shadow_depth = NULL;

        renderer->has_bake = true;
        release_texture(renderer, old);
        release_texture(renderer, old_direct);
        release_buffer(renderer, old_volume);
        release_buffer(renderer, old_beam);
    } else {
        release_texture(renderer, replacement);
        release_texture(renderer, direct_replacement);
        release_buffer(renderer, volume_buffer);
        release_buffer(renderer, beam_buffer);
    }

    cache_free(&cached);

    return good;
}

bool renderer_rebake_current_scene(RENDERER *renderer, const MESH *mesh, const GLTF_SCENE *visual, const LIGHTMAP *lightmap, const struct LIGHT *light,
                                   const SKY *sky, const VOLUMETRICS_LIGHTING *volumetrics, const char *path, uint64_t scene_hash, uint64_t layout_hash,
                                   uint64_t volume_hash, uint64_t beam_hash) {
    if (!renderer || !mesh || !lightmap || !renderer->gpu->device || !light || light->type != LIGHT_DIRECTIONAL || !sky || !volumetrics) return false;

    renderer->sun = light->directional;
    renderer->sun.direction = v3_normalize(renderer->sun.direction);

    if (v3_len_sq(renderer->sun.direction) <= 0.0f) return false;

    renderer->sky = *sky;
    renderer->volumetrics = *volumetrics;

    CACHED_LIGHTMAP previous = {0};
    bool reuse = cache_read_partial(path, scene_hash, &previous);

    if (reuse && previous.layout_hash == layout_hash && previous.volume_hash == volume_hash && previous.beam_hash == beam_hash) reuse = false;

    if (!reuse) cache_free(&previous);

    const bool reuse_lightmap = reuse && previous.layout_hash == layout_hash && previous.width == lightmap->width && previous.height == lightmap->height;
    const bool reuse_volume = reuse && previous.volume_hash == volume_hash;
    const bool reuse_beams = reuse && previous.beam_hash == beam_hash;

    bake_progress(renderer, "scene geometry", 0u, 0u);

    Uint64 started = SDL_GetPerformanceCounter();
    BVH tree = {0};

    if (!bvh_build(&tree, mesh, visual)) {
        cache_free(&previous);
        return false;
    }

    bake_timing("scene geometry", started);

    PROBE_GRID volume_candidate = {0};
    BEAM_GRID beam_candidate = {0};

    bake_progress(renderer, "volume probes", 0u, 0u);
    started = SDL_GetPerformanceCounter();

    if (reuse_volume) {
        volume_candidate = previous.volume_probes;
        previous.volume_probes.probes = NULL;
        SDL_Log("B: reused cached volume probes");
    } else {
        const bool made = make_probe_grid(mesh, renderer->volumetrics.probe_spacing, &volume_candidate) &&
                          bake_probe_grid(renderer, &volume_candidate, renderer->volumetrics.probe_samples);

        if (!made) {
            free_probe_grid(&volume_candidate);
            bvh_free(&tree);
            cache_free(&previous);
            return false;
        }
    }

    bake_timing("volume probes", started);
    bake_progress(renderer, "lightmap shader", 0u, 0u);

    NriTexture *old = renderer->lightmap_texture;
    const Uint32 old_width = renderer->lightmap_width;
    const Uint32 old_height = renderer->lightmap_height;
    const bool had_bake = renderer->has_bake;

    renderer->lightmap_texture = NULL;
    started = SDL_GetPerformanceCounter();

    bool good = false;

    if (reuse_lightmap) {
        renderer->lightmap_texture = upload_lightmap(renderer, &previous);
        good = renderer->lightmap_texture && upload_bvh(renderer, &tree);

        if (good) {
            renderer->lightmap_width = previous.width;
            renderer->lightmap_height = previous.height;
            SDL_Log("B: reused cached surface lightmap");
        }
    } else {
        good = bake_lightmap(renderer, &tree, lightmap, &volume_candidate);
    }

    if (good) bake_timing(reuse_lightmap ? "cached lightmap upload submission" : "lightmap GPU submission", started);

    if (good) bake_progress(renderer, "sun visibility", 0u, 0u);

    started = SDL_GetPerformanceCounter();

    if (good && reuse_beams) {
        beam_candidate = previous.beams;
        previous.beams.cells = NULL;
        previous.beams.shadow_depth = NULL;
        SDL_Log("B: reused cached sun beams");
    } else if (good) {
        good = beam_build(&beam_candidate, mesh, &tree, renderer->sun.direction);
    }

    if (good) bake_timing("sun visibility", started);

    if (good) SDL_Log("B: compressed sun beams into %u cells", beam_candidate.count);

    NriBuffer *volume_buffer = NULL;
    NriBuffer *beam_buffer = NULL;

    if (good) {
        volume_buffer = upload_probes(renderer, &volume_candidate);
        good = volume_buffer != NULL;
    }

    if (good) {
        beam_buffer = upload_beams(renderer, &beam_candidate);
        good = beam_buffer != NULL;
    }

    CACHED_LIGHTMAP candidate = {0};

    if (good) {
        bake_progress(renderer, "saving cache", 0u, 0u);
        started = SDL_GetPerformanceCounter();
        candidate.volume_probes = volume_candidate;
        candidate.beams = beam_candidate;

        if (reuse_lightmap) {
            candidate.pixels = previous.pixels;
            candidate.direct_pixels = previous.direct_pixels;
            candidate.width = previous.width;
            candidate.height = previous.height;
            previous.pixels = NULL;
            previous.direct_pixels = NULL;
        } else {
            good = download_lightmap(renderer, &candidate);
        }

        if (good) {
            const Uint64 write_started = SDL_GetPerformanceCounter();
            good = cache_write(path, scene_hash, layout_hash, volume_hash, beam_hash, &candidate);

            if (good) bake_timing("cache serialization total", write_started);
        }

        candidate.volume_probes.probes = NULL;
        candidate.beams.cells = NULL;
        candidate.beams.shadow_depth = NULL;

        if (good) bake_timing("readback and cache write", started);
    }

    NriTexture *direct_candidate = good ? upload_direct_lightmap(renderer, &candidate) : NULL;
    if (good && !direct_candidate) good = false;

    cache_free(&candidate);
    cache_free(&previous);
    release_bake_resources(renderer);
    bvh_free(&tree);

    if (good) {
        NriBuffer *old_volume = renderer->volume_probe_buffer;
        NriBuffer *old_beam = renderer->beam_buffer;
        NriTexture *old_direct = renderer->baked_direct_texture;

        renderer->baked_direct_texture = direct_candidate;
        renderer->volume_probe_buffer = volume_buffer;
        renderer->beam_buffer = beam_buffer;

        free_probe_grid(&renderer->volume_probes);
        renderer->volume_probes = volume_candidate;

        beam_free(&renderer->beams);
        renderer->beams = beam_candidate;

        renderer->has_bake = true;
        release_texture(renderer, old);
        release_texture(renderer, old_direct);
        release_buffer(renderer, old_volume);
        release_buffer(renderer, old_beam);
    } else {
        release_texture(renderer, direct_candidate);
        release_buffer(renderer, volume_buffer);
        release_buffer(renderer, beam_buffer);
        beam_free(&beam_candidate);
        free_probe_grid(&volume_candidate);
        release_texture(renderer, renderer->lightmap_texture);
        renderer->lightmap_texture = old;
        renderer->lightmap_width = old_width;
        renderer->lightmap_height = old_height;
        renderer->has_bake = had_bake;
    }

    return good;
}

static bool model_surface_layout(struct MODEL *model, uint32_t target_samples) {
    if (!model || !model->geometry || !model->visual || !target_samples) return false;
    if (model->surface_layout && model->surface_layout->sample_count <= target_samples) return true;

    if (!model->surface_layout) {
        model->surface_layout = calloc(1, sizeof(*model->surface_layout));
        if (!model->surface_layout) return false;
    } else {
        lmap_free(model->surface_layout);
    }

    float density = (float)DYNAMIC_LIGHTING_TEXELS_PER_UNIT;

    for (;;) {
        if (!lmap_build_density(model->surface_layout, model->geometry, density, DYNAMIC_LIGHTING_MAX_SIZE)) return false;
        if (model->surface_layout->sample_count <= target_samples || density <= DYNAMIC_LIGHTING_MIN_TEXELS_PER_UNIT) break;

        const float ratio = sqrtf((float)target_samples / (float)model->surface_layout->sample_count);
        float next = density * ratio * 0.90f;

        if (!(next < density)) next = density * 0.75f;
        if (next < DYNAMIC_LIGHTING_MIN_TEXELS_PER_UNIT) next = DYNAMIC_LIGHTING_MIN_TEXELS_PER_UNIT;

        lmap_free(model->surface_layout);
        density = next;
    }

    SDL_Log("dynamic surface layout: %u samples | %.4f texels/unit | target %u%s",
            model->surface_layout->sample_count, model->surface_layout->texel_density, target_samples,
            model->surface_layout->sample_count <= target_samples ? "" : " (chart-count limited)");
    return true;
}

static void model_lighting_summary(const struct MODEL *model, VEC3 *diffuse, VEC3 *emissive) {
    VEC3 diffuse_sum = v3(0.0f, 0.0f, 0.0f);
    VEC3 emissive_sum = v3(0.0f, 0.0f, 0.0f);
    uint32_t count = 0u;

    if (model && model->visual && model->visual->materials && model->visual->material_count) {
        const GLTF_SCENE *visual = model->visual;
        const size_t triangle_count = visual->vertex_count / 3u;

        for (size_t triangle = 0; triangle < triangle_count; ++triangle) {
            const uint32_t material_index = visual->vertices[triangle * 3u].material;

            if (material_index >= visual->material_count) continue;

            const GLTF_MATERIAL *material = &visual->materials[material_index];
            const float nonmetal = 1.0f - fminf(fmaxf(material->metallic, 0.0f), 1.0f);

            diffuse_sum = v3_add(diffuse_sum, v3(material->base_color[0] * nonmetal, material->base_color[1] * nonmetal, material->base_color[2] * nonmetal));
            emissive_sum = v3_add(emissive_sum, v3(material->emissive[0], material->emissive[1], material->emissive[2]));
            ++count;
        }
    }

    if (!count) {
        diffuse_sum = v3(0.72f, 0.72f, 0.72f);
        count = 1u;
    }

    if (diffuse) *diffuse = v3_scale(diffuse_sum, 1.0f / (float)count);
    if (emissive) *emissive = v3_scale(emissive_sum, 1.0f / (float)count);
}

static bool renderer_build_dynamic_static_transport(RENDERER *renderer, const SCENE *scene) {
    if (!renderer || !scene || !scene->lightmap || !scene->lightmap->uvs || !scene->static_geometry.faces.count ||
        !scene->static_surface_refs || scene->static_surface_ref_count != scene->static_geometry.faces.count)
        return false;

    BVH tree = {0};
    if (!bvh_build_with_surfaces(&tree, &scene->static_geometry, &scene->static_visual, scene->static_surface_refs,
                                 scene->static_surface_ref_count))
        return false;

    DYNAMIC_STATIC_SURFACE_GPU *surface_refs = calloc(tree.triangle_count, sizeof(*surface_refs));
    bool good = surface_refs != NULL;

    if (good) {
        for (uint32_t i = 0; i < tree.triangle_count; ++i)
            surface_refs[i].source_triangle = tree.surfaces[i].source_triangle;
    }

    NriBuffer *nodes = NULL;
    NriBuffer *triangles = NULL;
    NriBuffer *surfaces = NULL;
    NriBuffer *uvs = NULL;

    if (good)
        nodes = gpu_upload_buffer(renderer, NriBufferUsageBits_SHADER_RESOURCE, tree.nodes,
                                  (size_t)tree.node_count * sizeof(*tree.nodes), sizeof(BVH_NODE));
    if (nodes)
        triangles = gpu_upload_buffer(renderer, NriBufferUsageBits_SHADER_RESOURCE, tree.triangles,
                                      (size_t)tree.triangle_count * sizeof(*tree.triangles), sizeof(BVH_TRIANGLE));
    if (triangles)
        surfaces = gpu_upload_buffer(renderer, NriBufferUsageBits_SHADER_RESOURCE, surface_refs,
                                     (size_t)tree.triangle_count * sizeof(*surface_refs), sizeof(*surface_refs));

    const size_t uv_count = scene->static_geometry.faces.count * 6u;
    if (surfaces)
        uvs = gpu_upload_buffer(renderer, NriBufferUsageBits_SHADER_RESOURCE, scene->lightmap->uvs,
                                uv_count * sizeof(*scene->lightmap->uvs), sizeof(LMAP_UV));

    good = nodes && triangles && surfaces && uvs;

    if (good) {
        release_dynamic_static_transport(renderer);
        renderer->dynamic_static_node_buffer = nodes;
        renderer->dynamic_static_triangle_buffer = triangles;
        renderer->dynamic_static_surface_buffer = surfaces;
        renderer->dynamic_static_uv_buffer = uvs;
        renderer->dynamic_static_node_count = tree.node_count;
        renderer->dynamic_static_triangle_count = tree.triangle_count;
    } else {
        release_buffer(renderer, nodes);
        release_buffer(renderer, triangles);
        release_buffer(renderer, surfaces);
        release_buffer(renderer, uvs);
    }

    free(surface_refs);
    bvh_free(&tree);
    return good;
}

static bool renderer_allocate_dynamic_lighting(RENDERER *renderer, SCENE *scene) {
    uint32_t count = 0u;

    for (uint32_t i = 0; i < scene->object_count; ++i)
        if (scene->objects[i].type == MODEL && scene->objects[i].state == DYNAMIC && scene->objects[i].data) ++count;

    if (!count) return true;

    const uint32_t target_samples =
        DYNAMIC_SURFACE_SAMPLES_PER_FRAME / count ? DYNAMIC_SURFACE_SAMPLES_PER_FRAME / count : 1u;

    renderer->dynamic_lighting = calloc(count, sizeof(*renderer->dynamic_lighting));
    if (!renderer->dynamic_lighting) return false;

    for (uint32_t i = 0, out = 0; i < scene->object_count; ++i) {
        OBJECT *object = &scene->objects[i];

        if (object->type != MODEL || object->state != DYNAMIC || !object->data) continue;

        struct MODEL *model = object->data;

        if (!model_surface_layout(model, target_samples)) {
            release_dynamic_lighting(renderer);
            return false;
        }

        DYNAMIC_LIGHTING_ALLOCATION *allocation = &renderer->dynamic_lighting[out++];
        allocation->object_id = object->id;
        allocation->layout = model->surface_layout;
        allocation->local_center = model->geometry->bounds.center;
        allocation->local_extents = model->geometry->bounds.extents;
        allocation->local_radius = sqrtf(v3_len_sq(model->geometry->bounds.extents));
        model_lighting_summary(model, &allocation->average_diffuse, &allocation->average_emissive);
        allocation->transform_revision = 0u;
        allocation->lighting_revision = 0u;
        allocation->reference_transform_revision = 0u;
        allocation->reference_lighting_revision = 0u;
        allocation->pending_transform_revision = 0u;
        allocation->pending_lighting_revision = 0u;
        allocation->sample_cursor = 0u;
        allocation->cache_needs_clear = true;
        allocation->texture =
            gpu_create_texture(renderer, NriFormat_RGBA16_SFLOAT,
                               NriTextureUsageBits_SHADER_RESOURCE | NriTextureUsageBits_SHADER_RESOURCE_STORAGE,
                               allocation->layout->width, allocation->layout->height);

        if (allocation->texture &&
            !gpu_clear_texture_zero(renderer, allocation->texture, 8u, NriAccessBits_SHADER_RESOURCE, NriLayout_SHADER_RESOURCE,
                                    NriStageBits_COMPUTE_SHADER | NriStageBits_FRAGMENT_SHADER)) {
            release_texture(renderer, allocation->texture);
            allocation->texture = NULL;
        }

        allocation->sample_buffer =
            gpu_upload_buffer(renderer, NriBufferUsageBits_SHADER_RESOURCE, allocation->layout->samples,
                              (size_t)allocation->layout->sample_count * sizeof(*allocation->layout->samples), sizeof(LMAP_SAMPLE));

        BVH self_tree = {0};

        if (allocation->texture && allocation->sample_buffer && bvh_build(&self_tree, model->geometry, model->visual)) {
            allocation->self_node_buffer =
                gpu_upload_buffer(renderer, NriBufferUsageBits_SHADER_RESOURCE, self_tree.nodes,
                                  (size_t)self_tree.node_count * sizeof(*self_tree.nodes), sizeof(BVH_NODE));
            allocation->self_triangle_buffer =
                gpu_upload_buffer(renderer, NriBufferUsageBits_SHADER_RESOURCE, self_tree.triangles,
                                  (size_t)self_tree.triangle_count * sizeof(*self_tree.triangles), sizeof(BVH_TRIANGLE));
        }

        bvh_free(&self_tree);

        if (!allocation->texture || !allocation->sample_buffer || !allocation->self_node_buffer || !allocation->self_triangle_buffer) {
            renderer->dynamic_lighting_count = out;
            release_dynamic_lighting(renderer);
            return false;
        }

        renderer->dynamic_lighting_count = out;
    }

    return renderer->dynamic_lighting_count == count;
}

static bool reference_world_layout(const LIGHTMAP *local, TRANSFORM transform, LIGHTMAP *world) {
    if (!local || !world || !local->samples || !local->sample_count) return false;

    *world = *local;
    world->samples = malloc((size_t)local->sample_count * sizeof(*world->samples));

    if (!world->samples) return false;

    const MAT4 model = m4_transform(transform, false);
    const MAT4 normal_model = m4_transform(transform, true);

    for (uint32_t i = 0; i < local->sample_count; ++i) {
        const LMAP_SAMPLE *source = &local->samples[i];
        LMAP_SAMPLE *target = &world->samples[i];
        const VEC3 position = m4_point(model, v3(source->position[0], source->position[1], source->position[2]));
        VEC3 normal = m4_point(normal_model, v3(source->normal[0], source->normal[1], source->normal[2]));

        normal = v3_normalize(normal);
        *target = *source;
        target->position[0] = position.x;
        target->position[1] = position.y;
        target->position[2] = position.z;
        target->normal[0] = normal.x;
        target->normal[1] = normal.y;
        target->normal[2] = normal.z;
    }

    return true;
}

static bool reference_bake_surface(RENDERER *r, const BVH *tree, const LIGHTMAP *layout, NriTexture **out_texture) {
    if (!r || !tree || !layout || !out_texture || !r->volume_probes.probes) return false;

    NriTexture *base_texture = r->lightmap_texture;
    const uint32_t base_width = r->lightmap_width;
    const uint32_t base_height = r->lightmap_height;
    const uint32_t base_sample_count = r->lightmap_sample_count;
    const uint32_t base_trace_count = r->lightmap_trace_count;
    const uint32_t base_target_samples = r->bake_target_samples;
    const uint32_t base_min_samples = r->lightmap_min_samples;
    const VEC3 base_probe_origin = r->lightmap_probe_origin;
    const float base_probe_spacing = r->lightmap_probe_spacing;
    const uint32_t base_probe_count_x = r->lightmap_probe_count_x;
    const uint32_t base_probe_count_y = r->lightmap_probe_count_y;
    const uint32_t base_probe_count_z = r->lightmap_probe_count_z;
    const float base_epsilon = r->bake_epsilon;
    const bool base_full_transport = r->bake_full_transport;

    r->lightmap_texture = NULL;
    r->bake_full_transport = true;

    const bool good = bake_lightmap(r, tree, layout, &r->volume_probes);
    NriTexture *candidate = r->lightmap_texture;

    r->lightmap_texture = base_texture;
    release_bake_resources(r);
    r->lightmap_width = base_width;
    r->lightmap_height = base_height;
    r->lightmap_sample_count = base_sample_count;
    r->lightmap_trace_count = base_trace_count;
    r->bake_target_samples = base_target_samples;
    r->lightmap_min_samples = base_min_samples;
    r->lightmap_probe_origin = base_probe_origin;
    r->lightmap_probe_spacing = base_probe_spacing;
    r->lightmap_probe_count_x = base_probe_count_x;
    r->lightmap_probe_count_y = base_probe_count_y;
    r->lightmap_probe_count_z = base_probe_count_z;
    r->bake_epsilon = base_epsilon;
    r->bake_full_transport = base_full_transport;

    if (!good || !candidate) {
        release_texture(r, candidate);
        return false;
    }

    *out_texture = candidate;
    return true;
}

static float dynamic_half_to_float(Uint16 h) {
    const Uint32 exponent = (h >> 10u) & 31u;
    const Uint32 mantissa = h & 1023u;
    const float value = exponent == 0u ? ldexpf((float)mantissa, -24)
                                      : exponent == 31u ? INFINITY : ldexpf((float)(1024u + mantissa), (int)exponent - 25);
    return (h & 0x8000u) ? -value : value;
}

static VEC3 dynamic_rgba16f_rgb(const Uint8 *pixels, uint32_t pixel) {
    Uint16 channels[3] = {0};
    memcpy(channels, pixels + (size_t)pixel * 8u, sizeof(channels));
    return v3(dynamic_half_to_float(channels[0]), dynamic_half_to_float(channels[1]), dynamic_half_to_float(channels[2]));
}

static float dynamic_rgba16f_alpha(const Uint8 *pixels, uint32_t pixel) {
    Uint16 alpha = 0u;
    memcpy(&alpha, pixels + (size_t)pixel * 8u + 6u, sizeof(alpha));
    return dynamic_half_to_float(alpha);
}

static void log_dynamic_matte_reference_error(RENDERER *r, const DYNAMIC_LIGHTING_ALLOCATION *allocation, const OBJECT *object,
                                              NriTexture *reference_texture) {
    if (!r || !allocation || !allocation->layout || !object || !reference_texture || !allocation->texture || !object->data) return;

    if (allocation->transform_revision != object->transform_revision || allocation->lighting_revision != object->lighting_revision) {
        SDL_Log("dynamic acceptance object %u: runtime cache is not current; comparison skipped", allocation->object_id);
        return;
    }

    if (r->gpu->graphics_queue && r->gpu->core.QueueWaitIdle(r->gpu->graphics_queue) != NriResult_SUCCESS) {
        SDL_Log("dynamic acceptance object %u: could not synchronize runtime cache for comparison", allocation->object_id);
        return;
    }

    Uint8 *runtime = NULL;
    Uint8 *reference = NULL;
    const uint32_t width = allocation->layout->width;
    const uint32_t height = allocation->layout->height;

    if (!download_rgba16f_texture(r, allocation->texture, width, height, &runtime) ||
        !download_rgba16f_texture(r, reference_texture, width, height, &reference)) {
        free(runtime);
        free(reference);
        SDL_Log("dynamic acceptance object %u: cache readback failed", allocation->object_id);
        return;
    }

    const struct MODEL *model_data = object->data;
    BVH self_tree = {0};

    if (!model_data->geometry || !model_data->visual || !bvh_build(&self_tree, model_data->geometry, model_data->visual)) {
        free(runtime);
        free(reference);
        SDL_Log("dynamic acceptance object %u: self BVH build failed", allocation->object_id);
        return;
    }

    const MAT4 model = m4_transform(object->transform, false);
    const MAT4 inverse_model = m4_inverse_transform(object->transform);
    const MAT4 normal_model = m4_transform(object->transform, true);
    const VEC3 sun = v3_normalize(r->sun.direction);
    const float epsilon = fmaxf(r->scene_radius * 2.0e-5f, 1.0e-5f);
    const float visibility_floor = 1.0f / 1024.0f;
    const uint64_t pixel_count = (uint64_t)width * height;

    uint32_t compared = 0u;
    uint32_t missing = 0u;
    double absolute_sum = 0.0;
    double squared_sum = 0.0;
    double reference_squared_sum = 0.0;
    float maximum = 0.0f;

    for (uint32_t i = 0; i < allocation->layout->sample_count; ++i) {
        const LMAP_SAMPLE *sample = &allocation->layout->samples[i];
        union {
            float f;
            uint32_t u;
        } bits = {sample->position[3]};

        if (bits.u >= pixel_count) continue;

        const float alpha = dynamic_rgba16f_alpha(runtime, bits.u);

        if (!isfinite(alpha) || alpha < visibility_floor * 0.5f) {
            ++missing;
            continue;
        }

        VEC3 runtime_rgb = dynamic_rgba16f_rgb(runtime, bits.u);
        const VEC3 reference_rgb = dynamic_rgba16f_rgb(reference, bits.u);

        VEC3 position = m4_point(model, v3(sample->position[0], sample->position[1], sample->position[2]));
        VEC3 normal = m4_point(normal_model, v3(sample->normal[0], sample->normal[1], sample->normal[2]));
        normal = v3_normalize(normal);

        const float cached_visibility = fminf(fmaxf((alpha - visibility_floor) / (1.0f - visibility_floor), 0.0f), 1.0f);
        const float n_dot_l = fmaxf(v3_dot(normal, sun), 0.0f);

        const VEC3 world_origin = v3_add(position, v3_scale(normal, epsilon));
        TRACE_RAY local_shadow = {
            .origin = m4_point(inverse_model, world_origin),
            .tmin = epsilon,
            .direction = m4_vector(inverse_model, sun),
            .tmax = 1.0e20f,
        };

        const float current_visibility = trace_any(&self_tree, local_shadow) ? 0.0f : 1.0f;
        const VEC3 direct =
            v3_scale(r->sun.color, r->sun.intensity * n_dot_l * cached_visibility * (current_visibility - 1.0f));
        runtime_rgb = v3(fmaxf(runtime_rgb.x + direct.x, 0.0f), fmaxf(runtime_rgb.y + direct.y, 0.0f), fmaxf(runtime_rgb.z + direct.z, 0.0f));

        const float error[3] = {
            runtime_rgb.x - reference_rgb.x,
            runtime_rgb.y - reference_rgb.y,
            runtime_rgb.z - reference_rgb.z,
        };
        const float reference_channels[3] = {reference_rgb.x, reference_rgb.y, reference_rgb.z};

        for (uint32_t channel = 0; channel < 3u; ++channel) {
            if (!isfinite(error[channel]) || !isfinite(reference_channels[channel])) continue;

            const float absolute = fabsf(error[channel]);
            absolute_sum += absolute;
            squared_sum += (double)error[channel] * error[channel];
            reference_squared_sum += (double)reference_channels[channel] * reference_channels[channel];
            maximum = fmaxf(maximum, absolute);
        }

        ++compared;
    }

    const uint32_t total = compared + missing;
    const double channel_count = (double)compared * 3.0;
    const double mae = channel_count > 0.0 ? absolute_sum / channel_count : 0.0;
    const double rmse = channel_count > 0.0 ? sqrt(squared_sum / channel_count) : 0.0;
    const double reference_rms = channel_count > 0.0 ? sqrt(reference_squared_sum / channel_count) : 0.0;
    const double nrmse = reference_rms > 1.0e-6 ? rmse / reference_rms : 0.0;
    const double coverage = total ? (double)compared * 100.0 / (double)total : 0.0;

    SDL_Log("dynamic acceptance object %u: matte cache coverage %.1f%% | RGB MAE %.5f | RMSE %.5f | NRMSE %.2f%% | max %.5f",
            allocation->object_id, coverage, mae, rmse, nrmse * 100.0, maximum);

    bvh_free(&self_tree);
    free(runtime);
    free(reference);
}

static bool renderer_update_reference_lighting(RENDERER *r, const struct LIGHT *light) {
    if (!r || !r->reference_lighting_enabled) return true;
    if (!r->scene || !light || light->type != LIGHT_DIRECTIONAL || !r->has_bake || !r->scene->lightmap) return false;

    if (!scene_compile(r->scene)) return false;

    const SCENE *scene = r->scene;
    bool current = r->reference_static_texture && r->reference_geometry_revision == scene->geometry_revision &&
                   r->reference_lighting_revision == scene->lighting_revision;

    for (uint32_t i = 0; current && i < r->dynamic_lighting_count; ++i) {
        const DYNAMIC_LIGHTING_ALLOCATION *allocation = &r->dynamic_lighting[i];
        const OBJECT *object = scene_object_by_id_const(scene, allocation->object_id);

        current = object && allocation->reference_texture &&
                  allocation->reference_transform_revision == object->transform_revision &&
                  allocation->reference_lighting_revision == object->lighting_revision;
    }

    if (current) return true;

    r->sun = light->directional;
    r->sun.direction = v3_normalize(r->sun.direction);
    r->sky = scene->sky;
    r->volumetrics = scene->volumetrics;

    if (v3_len_sq(r->sun.direction) <= 0.0f) return false;

    Uint64 started = SDL_GetPerformanceCounter();
    BVH tree = {0};

    if (!bvh_build_with_surfaces(&tree, &scene->geometry, &scene->visual, scene->surface_refs, scene->surface_ref_count)) return false;

    NriTexture *static_candidate = NULL;
    NriTexture **dynamic_candidates = r->dynamic_lighting_count ? calloc(r->dynamic_lighting_count, sizeof(*dynamic_candidates)) : NULL;
    bool good = !r->dynamic_lighting_count || dynamic_candidates != NULL;

    if (good) good = reference_bake_surface(r, &tree, scene->lightmap, &static_candidate);

    for (uint32_t i = 0; good && i < r->dynamic_lighting_count; ++i) {
        DYNAMIC_LIGHTING_ALLOCATION *allocation = &r->dynamic_lighting[i];
        const OBJECT *object = scene_object_by_id_const(scene, allocation->object_id);

        if (!object || object->type != MODEL || object->state != DYNAMIC || !object->data || !allocation->layout) {
            good = false;
            break;
        }

        LIGHTMAP world_layout = {0};

        if (!reference_world_layout(allocation->layout, object->transform, &world_layout)) {
            good = false;
            break;
        }

        good = reference_bake_surface(r, &tree, &world_layout, &dynamic_candidates[i]);
        free(world_layout.samples);
    }

    if (good && r->dynamic_lighting_count == 1u) {
        DYNAMIC_LIGHTING_ALLOCATION *allocation = &r->dynamic_lighting[0];
        const OBJECT *object = scene_object_by_id_const(scene, allocation->object_id);

        if (object && dynamic_candidates && dynamic_candidates[0])
            log_dynamic_matte_reference_error(r, allocation, object, dynamic_candidates[0]);
    }

    bvh_free(&tree);

    if (!good) {
        release_texture(r, static_candidate);

        for (uint32_t i = 0; i < r->dynamic_lighting_count; ++i)
            release_texture(r, dynamic_candidates ? dynamic_candidates[i] : NULL);

        free(dynamic_candidates);
        return false;
    }

    release_texture(r, r->reference_static_texture);
    r->reference_static_texture = static_candidate;

    for (uint32_t i = 0; i < r->dynamic_lighting_count; ++i) {
        DYNAMIC_LIGHTING_ALLOCATION *allocation = &r->dynamic_lighting[i];
        const OBJECT *object = scene_object_by_id_const(scene, allocation->object_id);

        release_texture(r, allocation->reference_texture);
        allocation->reference_texture = dynamic_candidates[i];
        allocation->reference_transform_revision = object->transform_revision;
        allocation->reference_lighting_revision = object->lighting_revision;
    }

    free(dynamic_candidates);
    r->reference_geometry_revision = scene->geometry_revision;
    r->reference_lighting_revision = scene->lighting_revision;

    SDL_Log("dynamic reference: rebuilt full-current-scene surface lighting in %.2f ms",
            (double)(SDL_GetPerformanceCounter() - started) * 1000.0 / (double)SDL_GetPerformanceFrequency());

    return true;
}

static bool renderer_build_scene(RENDERER *renderer, SCENE *scene, const LIGHTMAP *lightmap) {
    if (!renderer || !scene || !lightmap || !scene->static_visual.vertex_count || scene->static_visual.vertex_count % 3u ||
        scene->static_visual.vertex_count / 3u != scene->static_geometry.faces.count || !scene->visual.material_count || !lightmap->uvs) {
        SDL_Log("render/static-lightmap geometry mismatch");
        return false;
    }

    const MESH *mesh = &scene->geometry;
    const GLTF_SCENE *visual = &scene->visual;
    const GLTF_SCENE *static_visual = &scene->static_visual;

    renderer->target = mesh->bounds.center;
    renderer->scene_radius = fmaxf(mesh->bounds.extents.x, fmaxf(mesh->bounds.extents.y, mesh->bounds.extents.z));

    if (renderer->scene_radius < 1.0f) renderer->scene_radius = 1.0f;

    renderer->distance = renderer->scene_radius * 2.15f;

    uint32_t draw_capacity = visual->material_count;
    uint32_t dynamic_model_count = 0u;

    for (uint32_t i = 0; i < scene->object_count; ++i) {
        const OBJECT *object = &scene->objects[i];

        if (object->type != MODEL || object->state != DYNAMIC || !object->data) continue;

        const struct MODEL *model = object->data;
        if (UINT32_MAX - draw_capacity < model->visual->material_count) return false;
        draw_capacity += model->visual->material_count;
        ++dynamic_model_count;
    }

    const uint32_t dynamic_surface_target =
        dynamic_model_count ? (DYNAMIC_SURFACE_SAMPLES_PER_FRAME / dynamic_model_count ? DYNAMIC_SURFACE_SAMPLES_PER_FRAME / dynamic_model_count : 1u)
                            : DYNAMIC_SURFACE_SAMPLES_PER_FRAME;

    free(renderer->draws);
    renderer->draws = calloc(draw_capacity ? draw_capacity : 1u, sizeof(*renderer->draws));

    if (!renderer->draws) return false;

    renderer->draw_count = 0;
    renderer->vertex_count = 0;

    const size_t static_triangle_count = static_visual->vertex_count / 3u;

    for (uint32_t material = 0; material < static_visual->material_count; ++material) {
        const uint32_t first = renderer->vertex_count;
        VEC3 center = v3(0.0f, 0.0f, 0.0f);
        uint32_t center_count = 0u;

        for (size_t triangle = 0; triangle < static_triangle_count; ++triangle) {
            const GLTF_VERTEX *vertices = &static_visual->vertices[triangle * 3u];

            if (vertices[0].material != material) continue;

            const LMAP_UV *uv = &lightmap->uvs[triangle * 6u];

            if (!push_surface(renderer, &vertices[0], uv[0]) || !push_surface(renderer, &vertices[1], uv[1]) || !push_surface(renderer, &vertices[2], uv[2]))
                return false;

            center = v3_add(center, v3_add(v3_add(vertices[0].position, vertices[1].position), vertices[2].position));
            center_count += 3u;
        }

        const uint32_t count = renderer->vertex_count - first;

        if (count) {
            renderer->draws[renderer->draw_count++] =
                (DRAW_RANGE){.first = first, .count = count, .material = material, .object_id = 0, .center = v3_scale(center, 1.0f / (float)center_count)};
        }
    }

    for (uint32_t object_index = 0; object_index < scene->object_count; ++object_index) {
        OBJECT *object = &scene->objects[object_index];

        if (object->type != MODEL || object->state != DYNAMIC || !object->data) continue;

        struct MODEL *model = object->data;

        if (!model_surface_layout(model, dynamic_surface_target)) return false;

        const LIGHTMAP *layout = model->surface_layout;
        const size_t triangle_count = model->visual->vertex_count / 3u;

        if (!layout->uvs || triangle_count != model->geometry->faces.count) return false;

        for (uint32_t local_material = 0; local_material < model->visual->material_count; ++local_material) {
            const uint32_t first = renderer->vertex_count;
            VEC3 center = v3(0.0f, 0.0f, 0.0f);
            uint32_t center_count = 0u;

            for (size_t triangle = 0; triangle < triangle_count; ++triangle) {
                const GLTF_VERTEX *vertices = &model->visual->vertices[triangle * 3u];

                if (vertices[0].material != local_material) continue;

                const LMAP_UV *uv = &layout->uvs[triangle * 6u];

                if (!push_surface(renderer, &vertices[0], uv[0]) || !push_surface(renderer, &vertices[1], uv[1]) || !push_surface(renderer, &vertices[2], uv[2]))
                    return false;

                center = v3_add(center, v3_add(v3_add(vertices[0].position, vertices[1].position), vertices[2].position));
                center_count += 3u;
            }

            const uint32_t count = renderer->vertex_count - first;

            if (count) {
                const uint32_t material = object->material_offset + local_material;

                if (material >= visual->material_count) return false;

                renderer->draws[renderer->draw_count++] =
                    (DRAW_RANGE){.first = first, .count = count, .material = material, .object_id = object->id,
                                 .center = v3_scale(center, 1.0f / (float)center_count)};
            }
        }
    }

    renderer->debug_vertex_start = renderer->vertex_count;

    const COLOR4 wire = {0.18f, 0.95f, 0.24f, 1.0f};

    for (size_t triangle = 0; triangle < static_triangle_count; ++triangle) {
        const GLTF_VERTEX *vertices = &static_visual->vertices[triangle * 3u];

        if (!add_wire_triangle(renderer, vertices[0].position, vertices[1].position, vertices[2].position, wire)) return false;
    }

    renderer->debug_vertex_count = renderer->vertex_count - renderer->debug_vertex_start;

    if (!upload_scene(renderer, visual) || !renderer_allocate_dynamic_lighting(renderer, scene) || !build_transmission_draws(renderer)) return false;

    renderer->has_bake = false;
    SDL_Log("materials: %u | draw ranges: %u | dynamic lighting allocations: %u | embedded images: %u", renderer->material_count, renderer->draw_count,
            renderer->dynamic_lighting_count, renderer->image_texture_count);

    return true;
}

static void renderer_handle_event(RENDERER *renderer, const SDL_Event *event) {
    if (!renderer || !event) return;

    switch (event->type) {
    case SDL_EVENT_MOUSE_BUTTON_DOWN:
        if (event->button.button == SDL_BUTTON_LEFT) renderer->dragging = true;
        break;

    case SDL_EVENT_MOUSE_BUTTON_UP:
        if (event->button.button == SDL_BUTTON_LEFT) renderer->dragging = false;
        break;

    case SDL_EVENT_MOUSE_MOTION:
        if (renderer->dragging) {
            renderer->yaw += event->motion.xrel * 0.0075f;
            renderer->pitch += event->motion.yrel * 0.0075f;

            if (renderer->pitch > 1.45f) renderer->pitch = 1.45f;
            if (renderer->pitch < -1.45f) renderer->pitch = -1.45f;
        }
        break;

    case SDL_EVENT_MOUSE_WHEEL:
        renderer->distance -= event->wheel.y * (renderer->distance * 0.08f);

        if (renderer->distance < renderer->scene_radius * 0.05f) renderer->distance = renderer->scene_radius * 0.05f;
        if (renderer->distance > renderer->scene_radius * 20.0f) renderer->distance = renderer->scene_radius * 20.0f;
        break;

    case SDL_EVENT_KEY_DOWN:
        if (!event->key.repeat && event->key.key == SDLK_TAB) renderer->show_debug = !renderer->show_debug;
        if (!event->key.repeat && event->key.key == SDLK_F2) {
            renderer->reference_lighting_enabled = !renderer->reference_lighting_enabled;
            SDL_Log("dynamic reference lighting: %s", renderer->reference_lighting_enabled ? "enabled" : "disabled");
        }
        if (!event->key.repeat && event->key.key == SDLK_F5) renderer->show_volume = !renderer->show_volume;

        if (!event->key.repeat && (event->key.key == SDLK_F1 || event->key.key == SDLK_F3 || event->key.key == SDLK_F4)) {
            const uint32_t view = (uint32_t)(event->key.key - SDLK_F1) + 1u;
            renderer->debug_view = renderer->debug_view == view ? 0u : view;
        }
        break;

    default:
        break;
    }
}

static bool renderer_draw(RENDERER *renderer, const struct LIGHT *light, const SKY *sky, const VOLUMETRICS_LIGHTING *volumetrics,
                          const PERIPHERAL_VISION *vision) {
    if (!renderer || !renderer->gpu->window || !light || light->type != LIGHT_DIRECTIONAL || !sky || !volumetrics || !vision) return false;

    DIRECTIONAL_LIGHT sun = light->directional;
    sun.direction = v3_normalize(sun.direction);

    if (v3_len_sq(sun.direction) <= 0.0f) return false;

    renderer->sun = sun;
    renderer->sky = *sky;

    int width = 0;
    int height = 0;

    if (!SDL_GetWindowSizeInPixels(renderer->gpu->window, &width, &height)) return false;
    if (width <= 0 || height <= 0) return true;

    const float fov = 62.0f * 3.14159265358979323846f / 180.0f;
    const float cp = cosf(renderer->pitch);
    const VEC3 eye = v3(renderer->target.x + renderer->distance * cp * cosf(renderer->yaw), renderer->target.y + renderer->distance * sinf(renderer->pitch),
                        renderer->target.z + renderer->distance * cp * sinf(renderer->yaw));
    const VEC3 forward = v3_normalize(v3_sub(renderer->target, eye));
    const VEC3 right = v3_normalize(v3_cross(forward, v3(0.0f, 1.0f, 0.0f)));
    const VEC3 up = v3_cross(right, forward);
    const float aspect = (float)width / (float)height;
    const float tan_half = tanf(fov * 0.5f);
    const float znear = fmaxf(0.02f, renderer->scene_radius * 0.005f);
    const float zfar = fmaxf(100.0f, renderer->scene_radius * 10.0f);
    const MAT4 view = m4_look_at(eye, renderer->target, v3(0.0f, 1.0f, 0.0f));
    const MAT4 projection = m4_perspective(fov, aspect, znear, zfar);
    const MAT4 mvp = m4_mul(projection, view);

    RENDER_FRAME frame = {.eye = eye,
                          .right = right,
                          .up = up,
                          .forward = forward,
                          .sun = sun,
                          .sky = *sky,
                          .volumetrics = *volumetrics,
                          .vision = *vision,
                          .tan_half_fov = tan_half,
                          .aspect = aspect};

    memcpy(frame.mvp, mvp.m, sizeof(frame.mvp));
    memcpy(frame.view, view.m, sizeof(frame.view));

    return draw_frame(renderer, &frame);
}

static struct LIGHT *scene_directional_light(const SCENE *scene) {
    if (!scene) return NULL;

    for (uint32_t i = 0; i < scene->object_count; ++i) {
        OBJECT *object = &scene->objects[i];

        if (object->type != LIGHT || !object->data) continue;

        struct LIGHT *light = object->data;

        if (light->type == LIGHT_DIRECTIONAL) return light;
    }

    return NULL;
}

bool renderer_init(RENDERER *renderer, GPU *gpu) {
    if (!renderer || !gpu || !gpu->device) return false;

    memset(renderer, 0, sizeof(*renderer));
    renderer->gpu = gpu;

    if (!renderer_gpu_resources_init(renderer)) {
        renderer->gpu = NULL;
        return false;
    }

    return true;
}

bool renderer_set_scene(RENDERER *renderer, SCENE *scene) {
    if (!renderer || !scene || !scene->lightmap) return false;
    if (!scene_compile(scene)) return false;

    if (!scene->lightmap_valid) {
        SDL_SetError("scene lightmap is stale; call scene_build_lightmap after changing models or transforms");
        return false;
    }

    renderer->scene = scene;

    if (!renderer_build_scene(renderer, scene, scene->lightmap) || !renderer_build_dynamic_static_transport(renderer, scene)) {
        renderer->scene = NULL;
        return false;
    }

    scene->radius = renderer->scene_radius;

    return true;
}

void renderer_event(RENDERER *renderer, const SDL_Event *event) {
    renderer_handle_event(renderer, event);
}

bool renderer_frame(RENDERER *renderer) {
    if (!renderer || !renderer->scene) return false;

    if (!renderer->scene->lightmap_valid) {
        SDL_SetError("static scene lightmap is stale; rebuild it before rendering");
        return false;
    }

    /*
     * DYNAMIC rendering stays instance-local. Rebuilding the flattened CPU
     * scene on every transform would turn motion into an O(scene) CPU cost.
     * The full current scene is compiled only by F2 reference mode above.
     */
    struct LIGHT *light = scene_directional_light(renderer->scene);

    if (!light) return false;
    if (!renderer_update_reference_lighting(renderer, light)) {
        SDL_SetError("dynamic reference lighting rebuild failed");
        return false;
    }

    return renderer_draw(renderer, light, &renderer->scene->sky, &renderer->scene->volumetrics, &renderer->scene->vision);
}

void renderer_deinit(RENDERER *renderer) {
    if (!renderer) return;
    renderer_gpu_resources_deinit(renderer);
}
