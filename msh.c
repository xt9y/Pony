#ifndef MSH_H
#define MSH_H

#include "sdl.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

struct Bvh;

typedef struct Vec3 {
    float x, y, z;
} VEC3;

typedef struct Aabb {
    VEC3 min;
    VEC3 max;
} AABB;

typedef struct Vertex {
    float position[3];
    float normal[3];
    float uv0[2];
} VERTEX;

typedef struct LightmapCorner {
    float u;
    float v;
} LIGHTMAP_CORNER;

typedef struct Triangle {
    VEC3 a;
    VEC3 b;
    VEC3 c;
} TRIANGLE;

typedef uint32_t MESH_HANDLE;

typedef enum Mobility {
    MOBILITY_STATIC,
    MOBILITY_DYNAMIC
} MOBILITY;

typedef struct Transform {
    float matrix[16];
} TRANSFORM;

typedef struct RenderInstance {
    MESH_HANDLE mesh;
    TRANSFORM transform;
    float world_inverse[16];
    MOBILITY mobility;
    uint32_t material_id;
    uint32_t entity_id;
    uint32_t query_mask;
    float orientation_sign;
    AABB world_bounds;
} RENDER_INSTANCE;

typedef struct MeshDraw {
    uint32_t first_index;
    uint32_t index_count;
    uint32_t material_id;
} MESH_DRAW;

typedef struct DebugTriangleVertex {
    VERTEX vertex;
    float triangle_id;
} DEBUG_TRIANGLE_VERTEX;

typedef struct Mesh {
    VERTEX *vertices;
    uint32_t vertex_count;

    uint32_t *indices;
    uint32_t index_count;

    LIGHTMAP_CORNER *lightmap_corners;
    uint32_t lightmap_corner_count;

    AABB local_bounds;

    SDL_GPUBuffer *vertex_buffer;
    SDL_GPUBuffer *index_buffer;
    SDL_GPUBuffer *triangle_debug_buffer;

    MESH_DRAW *draws;
    uint32_t draw_count;

    struct Bvh *blas;
    uint32_t geometry_revision;
    int owns_cpu;
} MESH;

static VEC3 m_v3(float x, float y, float z)
{
    return (VEC3){x, y, z};
}

static VEC3 m_min3(VEC3 a, VEC3 b)
{
    return m_v3(SDL_min(a.x, b.x), SDL_min(a.y, b.y), SDL_min(a.z, b.z));
}

static VEC3 m_max3(VEC3 a, VEC3 b)
{
    return m_v3(SDL_max(a.x, b.x), SDL_max(a.y, b.y), SDL_max(a.z, b.z));
}

static VEC3 m_sub3(VEC3 a, VEC3 b)
{
    return m_v3(a.x - b.x, a.y - b.y, a.z - b.z);
}

static VEC3 m_cross3(VEC3 a, VEC3 b)
{
    return m_v3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x);
}

static float m_dot3(VEC3 a, VEC3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

static VEC3 m_norm3(VEC3 v)
{
    float len2 = m_dot3(v, v);
    if (len2 <= 1e-20f) return m_v3(0.0f, 0.0f, 0.0f);
    float inv = 1.0f / sqrtf(len2);
    return m_v3(v.x * inv, v.y * inv, v.z * inv);
}

int m_bounds(const VERTEX *vertices, uint32_t count, AABB *out)
{
    if (!vertices || !count || !out) return 0;

    AABB b = {
        .min = {FLT_MAX, FLT_MAX, FLT_MAX},
        .max = {-FLT_MAX, -FLT_MAX, -FLT_MAX}
    };

    for (uint32_t i = 0; i < count; ++i) {
        VEC3 p = {
            vertices[i].position[0],
            vertices[i].position[1],
            vertices[i].position[2]
        };
        b.min = m_min3(b.min, p);
        b.max = m_max3(b.max, p);
    }

    *out = b;
    return 1;
}

int m_validate(
    const VERTEX *vertices,
    uint32_t vertex_count,
    const uint32_t *indices,
    uint32_t index_count,
    const LIGHTMAP_CORNER *lightmap_corners,
    uint32_t lightmap_corner_count)
{
    if (!vertices || vertex_count == 0) return 0;
    if (!indices || index_count == 0 || index_count % 3 != 0) return 0;

    for (uint32_t i = 0; i < vertex_count; ++i) {
        if (!isfinite(vertices[i].position[0]) ||
            !isfinite(vertices[i].position[1]) ||
            !isfinite(vertices[i].position[2])) {
            return 0;
        }
    }

    for (uint32_t i = 0; i < index_count; ++i) {
        if (indices[i] >= vertex_count) return 0;
    }

    if (lightmap_corner_count != 0) {
        if (!lightmap_corners || lightmap_corner_count != index_count) return 0;
        for (uint32_t i = 0; i < lightmap_corner_count; ++i) {
            float u = lightmap_corners[i].u;
            float v = lightmap_corners[i].v;
            if (!isfinite(u) || !isfinite(v)) return 0;
            if (u < 0.0f || u > 1.0f || v < 0.0f || v > 1.0f) return 0;
        }
    } else if (lightmap_corners) {
        return 0;
    }

    return 1;
}

int m_triangle(const MESH *mesh, uint32_t triangle_index, TRIANGLE *out)
{
    if (!mesh || !out || !mesh->vertices || !mesh->indices) return 0;
    uint32_t triangle_count = mesh->index_count / 3;
    if (triangle_index >= triangle_count) return 0;

    uint32_t base = triangle_index * 3;
    uint32_t ia = mesh->indices[base + 0];
    uint32_t ib = mesh->indices[base + 1];
    uint32_t ic = mesh->indices[base + 2];
    if (ia >= mesh->vertex_count || ib >= mesh->vertex_count || ic >= mesh->vertex_count) return 0;

    const VERTEX *a = &mesh->vertices[ia];
    const VERTEX *b = &mesh->vertices[ib];
    const VERTEX *c = &mesh->vertices[ic];

    out->a = m_v3(a->position[0], a->position[1], a->position[2]);
    out->b = m_v3(b->position[0], b->position[1], b->position[2]);
    out->c = m_v3(c->position[0], c->position[1], c->position[2]);
    return 1;
}

VEC3 m_triangle_normal(TRIANGLE tri)
{
    return m_norm3(m_cross3(m_sub3(tri.b, tri.a), m_sub3(tri.c, tri.a)));
}

static VEC3 m_transform_point(const float m[16], VEC3 p)
{
    return m_v3(
        m[0] * p.x + m[4] * p.y + m[8] * p.z + m[12],
        m[1] * p.x + m[5] * p.y + m[9] * p.z + m[13],
        m[2] * p.x + m[6] * p.y + m[10] * p.z + m[14]);
}

int m_world_bounds(AABB local, const float matrix[16], AABB *out)
{
    if (!matrix || !out) return 0;

    AABB world = {
        .min = {FLT_MAX, FLT_MAX, FLT_MAX},
        .max = {-FLT_MAX, -FLT_MAX, -FLT_MAX}
    };

    for (uint32_t i = 0; i < 8; ++i) {
        VEC3 corner = {
            (i & 1u) ? local.max.x : local.min.x,
            (i & 2u) ? local.max.y : local.min.y,
            (i & 4u) ? local.max.z : local.min.z
        };
        VEC3 p = m_transform_point(matrix, corner);
        world.min = m_min3(world.min, p);
        world.max = m_max3(world.max, p);
    }

    *out = world;
    return 1;
}

static const VERTEX cube_v[] = {
    { { 0.5f, -0.5f, -0.5f }, { 1.0f, 0.0f, 0.0f }, { 1.0f, 0.0f } },
    { { 0.5f, 0.5f, -0.5f }, { 1.0f, 0.0f, 0.0f }, { 1.0f, 1.0f } },
    { { 0.5f, 0.5f, 0.5f }, { 1.0f, 0.0f, 0.0f }, { 0.0f, 1.0f } },
    { { 0.5f, -0.5f, 0.5f }, { 1.0f, 0.0f, 0.0f }, { 0.0f, 0.0f } },
    { { -0.5f, -0.5f, 0.5f }, { -1.0f, 0.0f, 0.0f }, { 1.0f, 0.0f } },
    { { -0.5f, 0.5f, 0.5f }, { -1.0f, 0.0f, 0.0f }, { 1.0f, 1.0f } },
    { { -0.5f, 0.5f, -0.5f }, { -1.0f, 0.0f, 0.0f }, { 0.0f, 1.0f } },
    { { -0.5f, -0.5f, -0.5f }, { -1.0f, 0.0f, 0.0f }, { 0.0f, 0.0f } },
    { { -0.5f, 0.5f, -0.5f }, { 0.0f, 1.0f, 0.0f }, { 1.0f, 0.0f } },
    { { -0.5f, 0.5f, 0.5f }, { 0.0f, 1.0f, 0.0f }, { 1.0f, 1.0f } },
    { { 0.5f, 0.5f, 0.5f }, { 0.0f, 1.0f, 0.0f }, { 0.0f, 1.0f } },
    { { 0.5f, 0.5f, -0.5f }, { 0.0f, 1.0f, 0.0f }, { 0.0f, 0.0f } },
    { { 0.5f, -0.5f, -0.5f }, { 0.0f, -1.0f, 0.0f }, { 1.0f, 0.0f } },
    { { 0.5f, -0.5f, 0.5f }, { 0.0f, -1.0f, 0.0f }, { 1.0f, 1.0f } },
    { { -0.5f, -0.5f, 0.5f }, { 0.0f, -1.0f, 0.0f }, { 0.0f, 1.0f } },
    { { -0.5f, -0.5f, -0.5f }, { 0.0f, -1.0f, 0.0f }, { 0.0f, 0.0f } },
    { { 0.5f, -0.5f, 0.5f }, { 0.0f, 0.0f, 1.0f }, { 1.0f, 0.0f } },
    { { 0.5f, 0.5f, 0.5f }, { 0.0f, 0.0f, 1.0f }, { 1.0f, 1.0f } },
    { { -0.5f, 0.5f, 0.5f }, { 0.0f, 0.0f, 1.0f }, { 0.0f, 1.0f } },
    { { -0.5f, -0.5f, 0.5f }, { 0.0f, 0.0f, 1.0f }, { 0.0f, 0.0f } },
    { { -0.5f, -0.5f, -0.5f }, { 0.0f, 0.0f, -1.0f }, { 1.0f, 0.0f } },
    { { -0.5f, 0.5f, -0.5f }, { 0.0f, 0.0f, -1.0f }, { 1.0f, 1.0f } },
    { { 0.5f, 0.5f, -0.5f }, { 0.0f, 0.0f, -1.0f }, { 0.0f, 1.0f } },
    { { 0.5f, -0.5f, -0.5f }, { 0.0f, 0.0f, -1.0f }, { 0.0f, 0.0f } }
};

static const uint32_t cube_i[] = {
    0, 1, 2, 0, 2, 3,
    4, 5, 6, 4, 6, 7,
    8, 9, 10, 8, 10, 11,
    12, 13, 14, 12, 14, 15,
    16, 17, 18, 16, 18, 19,
    20, 21, 22, 20, 22, 23
};

static const VERTEX plane_v[] = {
    { { -0.5f, 0.0f, -0.5f }, { 0.0f, 1.0f, 0.0f }, { 1.0f, 0.0f } },
    { { -0.5f, 0.0f, 0.5f }, { 0.0f, 1.0f, 0.0f }, { 1.0f, 1.0f } },
    { { 0.5f, 0.0f, 0.5f }, { 0.0f, 1.0f, 0.0f }, { 0.0f, 1.0f } },
    { { 0.5f, 0.0f, -0.5f }, { 0.0f, 1.0f, 0.0f }, { 0.0f, 0.0f } }
};

static const VERTEX wall_v[] = {
    { { 0.5f, -0.5f, 0.0f }, { 0.0f, 0.0f, 1.0f }, { 1.0f, 0.0f } },
    { { 0.5f, 0.5f, 0.0f }, { 0.0f, 0.0f, 1.0f }, { 1.0f, 1.0f } },
    { { -0.5f, 0.5f, 0.0f }, { 0.0f, 0.0f, 1.0f }, { 0.0f, 1.0f } },
    { { -0.5f, -0.5f, 0.0f }, { 0.0f, 0.0f, 1.0f }, { 0.0f, 0.0f } }
};

static const uint32_t quad_i[] = {0, 1, 2, 0, 2, 3};

static void m_quad_lightmap(const VERTEX *v, const uint32_t *idx, LIGHTMAP_CORNER out[6])
{
    const float pad = 0.02f;
    for (uint32_t i = 0; i < 6; ++i) {
        const VERTEX *vertex = &v[idx[i]];
        out[i].u = pad + vertex->uv0[0] * (1.0f - 2.0f * pad);
        out[i].v = pad + vertex->uv0[1] * (1.0f - 2.0f * pad);
    }
}

static void m_cube_lightmap(LIGHTMAP_CORNER out[36])
{
    const float pad = 0.04f;
    for (uint32_t i = 0; i < 36; ++i) {
        uint32_t face = i / 6;
        uint32_t cell_x = face % 3;
        uint32_t cell_y = face / 3;
        const VERTEX *vertex = &cube_v[cube_i[i]];
        float u = pad + vertex->uv0[0] * (1.0f - 2.0f * pad);
        float v = pad + vertex->uv0[1] * (1.0f - 2.0f * pad);
        out[i].u = ((float)cell_x + u) / 3.0f;
        out[i].v = ((float)cell_y + v) / 2.0f;
    }
}

int m_load(SDL_GPUDevice *d, SDL_GPUBuffer *b, const void *s, uint32_t n)
{
    SDL_GPUTransferBufferCreateInfo tinfo = {
        .usage = SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD,
        .size = n
    };

    SDL_GPUTransferBuffer *t = SDL_CreateGPUTransferBuffer(d, &tinfo);
    if (!t) {
        SDL_Log("transfer buffer creation failed: %s", SDL_GetError());
        return 0;
    }

    void *map = SDL_MapGPUTransferBuffer(d, t, false);
    if (!map) {
        SDL_Log("transfer buffer map failed: %s", SDL_GetError());
        SDL_ReleaseGPUTransferBuffer(d, t);
        return 0;
    }

    SDL_memcpy(map, s, n);
    SDL_UnmapGPUTransferBuffer(d, t);

    SDL_GPUCommandBuffer *cmd = SDL_AcquireGPUCommandBuffer(d);
    if (!cmd) {
        SDL_Log("command buffer failed: %s", SDL_GetError());
        SDL_ReleaseGPUTransferBuffer(d, t);
        return 0;
    }

    SDL_GPUCopyPass *pass = SDL_BeginGPUCopyPass(cmd);
    SDL_GPUTransferBufferLocation src = {.transfer_buffer = t, .offset = 0};
    SDL_GPUBufferRegion dst = {.buffer = b, .offset = 0, .size = n};
    SDL_UploadToGPUBuffer(pass, &src, &dst, false);
    SDL_EndGPUCopyPass(pass);
    SDL_SubmitGPUCommandBuffer(cmd);
    SDL_ReleaseGPUTransferBuffer(d, t);
    return 1;
}

static void m_release_cpu(MESH *m)
{
    if (!m || !m->owns_cpu) return;
    SDL_free(m->draws);
    SDL_free(m->lightmap_corners);
    SDL_free(m->indices);
    SDL_free(m->vertices);
    m->draws = NULL;
    m->lightmap_corners = NULL;
    m->indices = NULL;
    m->vertices = NULL;
    m->owns_cpu = 0;
}

int m_make(
    SDL_GPUDevice *d,
    MESH *m,
    const VERTEX *vertices,
    uint32_t vertex_count,
    const uint32_t *indices,
    uint32_t index_count,
    const LIGHTMAP_CORNER *lightmap_corners,
    uint32_t lightmap_corner_count,
    const char *vname,
    const char *iname)
{
    if (!d || !m) return 0;
    SDL_memset(m, 0, sizeof(*m));

    if (!m_validate(vertices, vertex_count, indices, index_count, lightmap_corners, lightmap_corner_count)) {
        SDL_Log("mesh validation failed");
        return 0;
    }

    m->vertices = SDL_malloc((size_t)vertex_count * sizeof(*m->vertices));
    m->indices = SDL_malloc((size_t)index_count * sizeof(*m->indices));
    if (!m->vertices || !m->indices) goto fail;

    SDL_memcpy(m->vertices, vertices, (size_t)vertex_count * sizeof(*m->vertices));
    SDL_memcpy(m->indices, indices, (size_t)index_count * sizeof(*m->indices));
    m->vertex_count = vertex_count;
    m->index_count = index_count;
    m->owns_cpu = 1;

    if (lightmap_corner_count) {
        m->lightmap_corners = SDL_malloc((size_t)lightmap_corner_count * sizeof(*m->lightmap_corners));
        if (!m->lightmap_corners) goto fail;
        SDL_memcpy(m->lightmap_corners, lightmap_corners, (size_t)lightmap_corner_count * sizeof(*m->lightmap_corners));
        m->lightmap_corner_count = lightmap_corner_count;
    }

    if (!m_bounds(m->vertices, m->vertex_count, &m->local_bounds)) goto fail;

    m->draws = SDL_malloc(sizeof(*m->draws));
    if (!m->draws) goto fail;
    m->draws[0] = (MESH_DRAW){0, index_count, 0};
    m->draw_count = 1;

    SDL_GPUBufferCreateInfo vinfo = {
        .usage = SDL_GPU_BUFFERUSAGE_VERTEX | SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_READ,
        .size = vertex_count * sizeof(VERTEX)
    };
    m->vertex_buffer = SDL_CreateGPUBuffer(d, &vinfo);
    if (!m->vertex_buffer) {
        SDL_Log("vertex buffer creation failed: %s", SDL_GetError());
        goto fail;
    }
    SDL_SetGPUBufferName(d, m->vertex_buffer, vname);

    SDL_GPUBufferCreateInfo iinfo = {
        .usage = SDL_GPU_BUFFERUSAGE_INDEX | SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_READ,
        .size = index_count * sizeof(uint32_t)
    };
    m->index_buffer = SDL_CreateGPUBuffer(d, &iinfo);
    if (!m->index_buffer) {
        SDL_Log("index buffer creation failed: %s", SDL_GetError());
        goto fail;
    }
    SDL_SetGPUBufferName(d, m->index_buffer, iname);

    if (!m_load(d, m->vertex_buffer, m->vertices, vertex_count * sizeof(VERTEX))) goto fail;
    if (!m_load(d, m->index_buffer, m->indices, index_count * sizeof(uint32_t))) goto fail;

    DEBUG_TRIANGLE_VERTEX *debug_vertices = SDL_malloc((size_t)index_count * sizeof(*debug_vertices));
    if (!debug_vertices) goto fail;
    for (uint32_t i = 0; i < index_count; ++i) {
        debug_vertices[i].vertex = m->vertices[m->indices[i]];
        debug_vertices[i].triangle_id = (float)(i / 3u);
    }

    SDL_GPUBufferCreateInfo dinfo = {
        .usage = SDL_GPU_BUFFERUSAGE_VERTEX,
        .size = index_count * sizeof(DEBUG_TRIANGLE_VERTEX)
    };
    m->triangle_debug_buffer = SDL_CreateGPUBuffer(d, &dinfo);
    if (!m->triangle_debug_buffer) {
        SDL_free(debug_vertices);
        goto fail;
    }
    SDL_SetGPUBufferName(d, m->triangle_debug_buffer, "Mesh.DebugTriangles");
    if (!m_load(d, m->triangle_debug_buffer, debug_vertices, index_count * sizeof(DEBUG_TRIANGLE_VERTEX))) {
        SDL_free(debug_vertices);
        goto fail;
    }
    SDL_free(debug_vertices);

    m->geometry_revision = 1;
    return 1;

fail:
    if (m->triangle_debug_buffer) SDL_ReleaseGPUBuffer(d, m->triangle_debug_buffer);
    if (m->index_buffer) SDL_ReleaseGPUBuffer(d, m->index_buffer);
    if (m->vertex_buffer) SDL_ReleaseGPUBuffer(d, m->vertex_buffer);
    m->triangle_debug_buffer = NULL;
    m->index_buffer = NULL;
    m->vertex_buffer = NULL;
    m_release_cpu(m);
    return 0;
}

int m_init(SDL_GPUDevice *d, MESH *m)
{
    LIGHTMAP_CORNER corners[36];
    m_cube_lightmap(corners);
    return m_make(
        d, m,
        cube_v, (uint32_t)(sizeof(cube_v) / sizeof(cube_v[0])),
        cube_i, (uint32_t)(sizeof(cube_i) / sizeof(cube_i[0])),
        corners, 36,
        "Mesh.VB.Cube", "Mesh.IB.Cube");
}

int m_plane(SDL_GPUDevice *d, MESH *m)
{
    LIGHTMAP_CORNER corners[6];
    m_quad_lightmap(plane_v, quad_i, corners);
    return m_make(
        d, m,
        plane_v, (uint32_t)(sizeof(plane_v) / sizeof(plane_v[0])),
        quad_i, (uint32_t)(sizeof(quad_i) / sizeof(quad_i[0])),
        corners, 6,
        "Mesh.VB.Plane", "Mesh.IB.Plane");
}

int m_wall(SDL_GPUDevice *d, MESH *m)
{
    LIGHTMAP_CORNER corners[6];
    m_quad_lightmap(wall_v, quad_i, corners);
    return m_make(
        d, m,
        wall_v, (uint32_t)(sizeof(wall_v) / sizeof(wall_v[0])),
        quad_i, (uint32_t)(sizeof(quad_i) / sizeof(quad_i[0])),
        corners, 6,
        "Mesh.VB.Wall", "Mesh.IB.Wall");
}

void m_draw(SDL_GPURenderPass *p, const MESH *m)
{
    SDL_GPUBufferBinding vb = {.buffer = m->vertex_buffer, .offset = 0};
    SDL_GPUBufferBinding ib = {.buffer = m->index_buffer, .offset = 0};
    SDL_BindGPUVertexBuffers(p, 0, &vb, 1);
    SDL_BindGPUIndexBuffer(p, &ib, SDL_GPU_INDEXELEMENTSIZE_32BIT);
    SDL_DrawGPUIndexedPrimitives(p, m->index_count, 1, 0, 0, 0);
}

void m_draw_triangle_debug(SDL_GPURenderPass *p, const MESH *m)
{
    SDL_GPUBufferBinding vb = {.buffer = m->triangle_debug_buffer, .offset = 0};
    SDL_BindGPUVertexBuffers(p, 0, &vb, 1);
    SDL_DrawGPUPrimitives(p, m->index_count, 1, 0, 0);
}

void m_deinit(SDL_GPUDevice *d, MESH *m)
{
    if (!m) return;
    if (m->triangle_debug_buffer && d) SDL_ReleaseGPUBuffer(d, m->triangle_debug_buffer);
    if (m->index_buffer && d) SDL_ReleaseGPUBuffer(d, m->index_buffer);
    if (m->vertex_buffer && d) SDL_ReleaseGPUBuffer(d, m->vertex_buffer);
    m->index_buffer = NULL;
    m->vertex_buffer = NULL;
    m->triangle_debug_buffer = NULL;
    m_release_cpu(m);
    m->vertex_count = 0;
    m->index_count = 0;
    m->lightmap_corner_count = 0;
    m->draw_count = 0;
    m->blas = NULL;
}

#endif // MSH_H
