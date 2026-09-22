#ifndef BVH_H
#define BVH_H

#include "col.c"

#include <float.h>
#include <stdint.h>

#define BVH_INVALID UINT32_MAX
#define BVH_DEFAULT_LEAF_SIZE 4u

typedef struct BvhPrimitive {
    AABB bounds;
    VEC3 centroid;
    uint32_t id;
} BVH_PRIMITIVE;

typedef struct BvhNode {
    AABB bounds;
    uint32_t left;
    uint32_t right;
    uint32_t first;
    uint32_t count;
    uint32_t leaf;
} BVH_NODE;

typedef struct GpuBvhNode {
    float min_x, min_y, min_z;
    uint32_t meta;
    float max_x, max_y, max_z;
    uint32_t first;
    uint32_t count;
    uint32_t escape;
    uint32_t right;
    uint32_t pad;
} GPU_BVH_NODE;

_Static_assert(sizeof(GPU_BVH_NODE) == 48, "GPU_BVH_NODE layout mismatch");

typedef struct BvhTraceStats {
    uint32_t nodes_visited;
    uint32_t aabb_tests;
    uint32_t triangle_tests;
} BVH_TRACE_STATS;

typedef struct Bvh {
    BVH_NODE *nodes;
    uint32_t node_count;
    uint32_t node_capacity;

    uint32_t *primitive_indices;
    AABB *primitive_bounds;
    uint32_t primitive_count;
    uint32_t leaf_size;

    uint32_t source_revision;
    uint32_t geometry_revision;
    float baseline_quality;
    float current_quality;

} BVH;

typedef struct GpuMeshAccel {
    uint32_t node_offset;
    uint32_t node_count;
    uint32_t primitive_offset;
    uint32_t primitive_count;
    uint32_t vertex_offset;
    uint32_t index_offset;
    uint32_t pad0;
    uint32_t pad1;
} GPU_MESH_ACCEL;

typedef struct GpuInstance {
    float world[16];
    float inverse[16];
    uint32_t mesh_accel_index;
    uint32_t flags;
    uint32_t query_mask;
    uint32_t pad;
} GPU_INSTANCE;

typedef struct SceneGpuAccel {
    SDL_GPUBuffer *tlas_nodes;
    SDL_GPUBuffer *tlas_primitives;
    SDL_GPUBuffer *blas_nodes;
    SDL_GPUBuffer *blas_primitives;
    SDL_GPUBuffer *vertices;
    SDL_GPUBuffer *indices;
    SDL_GPUBuffer *mesh_accels;
    SDL_GPUBuffer *instances;
    uint32_t geometry_revision;
    uint32_t transform_revision;
    uint32_t mesh_count;
    uint32_t instance_count;
} SCENE_GPU_ACCEL;

_Static_assert(sizeof(GPU_MESH_ACCEL) == 32, "GPU_MESH_ACCEL layout mismatch");
_Static_assert(sizeof(GPU_INSTANCE) == 144, "GPU_INSTANCE layout mismatch");

typedef struct BvhBuildContext {
    BVH *bvh;
    BVH_PRIMITIVE *primitives;
} BVH_BUILD_CONTEXT;

static AABB bvh_empty_bounds(void)
{
    return (AABB){{FLT_MAX, FLT_MAX, FLT_MAX}, {-FLT_MAX, -FLT_MAX, -FLT_MAX}};
}

static AABB bvh_union(AABB a, AABB b)
{
    return (AABB){
        {SDL_min(a.min.x, b.min.x), SDL_min(a.min.y, b.min.y), SDL_min(a.min.z, b.min.z)},
        {SDL_max(a.max.x, b.max.x), SDL_max(a.max.y, b.max.y), SDL_max(a.max.z, b.max.z)}
    };
}

static int bvh_contains(AABB outer, AABB inner)
{
    const float e = 1e-5f;
    return
        outer.min.x <= inner.min.x + e && outer.min.y <= inner.min.y + e && outer.min.z <= inner.min.z + e &&
        outer.max.x + e >= inner.max.x && outer.max.y + e >= inner.max.y && outer.max.z + e >= inner.max.z;
}

static VEC3 bvh_centroid(AABB b)
{
    return (VEC3){
        (b.min.x + b.max.x) * 0.5f,
        (b.min.y + b.max.y) * 0.5f,
        (b.min.z + b.max.z) * 0.5f
    };
}

static float bvh_surface_area(AABB b)
{
    float x = SDL_max(0.0f, b.max.x - b.min.x);
    float y = SDL_max(0.0f, b.max.y - b.min.y);
    float z = SDL_max(0.0f, b.max.z - b.min.z);
    return 2.0f * (x * y + y * z + z * x);
}

static float bvh_centroid_axis(const BVH_BUILD_CONTEXT *ctx, uint32_t order_index, int axis)
{
    uint32_t primitive_index = ctx->bvh->primitive_indices[order_index];
    VEC3 c = ctx->primitives[primitive_index].centroid;
    return axis == 0 ? c.x : (axis == 1 ? c.y : c.z);
}

static void bvh_swap_u32(uint32_t *a, uint32_t *b)
{
    uint32_t t = *a;
    *a = *b;
    *b = t;
}

static void bvh_sort_range(BVH_BUILD_CONTEXT *ctx, int lo, int hi, int axis)
{
    while (lo < hi) {
        int i = lo;
        int j = hi;
        float pivot = bvh_centroid_axis(ctx, (uint32_t)(lo + (hi - lo) / 2), axis);
        while (i <= j) {
            while (bvh_centroid_axis(ctx, (uint32_t)i, axis) < pivot) ++i;
            while (bvh_centroid_axis(ctx, (uint32_t)j, axis) > pivot) --j;
            if (i <= j) {
                bvh_swap_u32(&ctx->bvh->primitive_indices[i], &ctx->bvh->primitive_indices[j]);
                ++i;
                --j;
            }
        }
        if (j - lo < hi - i) {
            if (lo < j) bvh_sort_range(ctx, lo, j, axis);
            lo = i;
        } else {
            if (i < hi) bvh_sort_range(ctx, i, hi, axis);
            hi = j;
        }
    }
}

static AABB bvh_range_bounds(const BVH_BUILD_CONTEXT *ctx, uint32_t first, uint32_t count)
{
    AABB b = bvh_empty_bounds();
    for (uint32_t i = 0; i < count; ++i) {
        uint32_t primitive_index = ctx->bvh->primitive_indices[first + i];
        b = bvh_union(b, ctx->primitives[primitive_index].bounds);
    }
    return b;
}

static AABB bvh_range_centroid_bounds(const BVH_BUILD_CONTEXT *ctx, uint32_t first, uint32_t count)
{
    AABB b = bvh_empty_bounds();
    for (uint32_t i = 0; i < count; ++i) {
        uint32_t primitive_index = ctx->bvh->primitive_indices[first + i];
        VEC3 c = ctx->primitives[primitive_index].centroid;
        AABB p = {c, c};
        b = bvh_union(b, p);
    }
    return b;
}

static int bvh_longest_axis(AABB b)
{
    float x = b.max.x - b.min.x;
    float y = b.max.y - b.min.y;
    float z = b.max.z - b.min.z;
    if (y > x && y >= z) return 1;
    if (z > x && z > y) return 2;
    return 0;
}

static uint32_t bvh_build_node(BVH_BUILD_CONTEXT *ctx, uint32_t first, uint32_t count)
{
    BVH *bvh = ctx->bvh;
    if (bvh->node_count >= bvh->node_capacity) return BVH_INVALID;

    uint32_t node_index = bvh->node_count++;
    BVH_NODE *node = &bvh->nodes[node_index];
    SDL_memset(node, 0, sizeof(*node));
    node->bounds = bvh_range_bounds(ctx, first, count);
    node->left = BVH_INVALID;
    node->right = BVH_INVALID;

    if (count <= bvh->leaf_size) {
        node->leaf = 1;
        node->first = first;
        node->count = count;
        return node_index;
    }

    AABB centroid_bounds = bvh_range_centroid_bounds(ctx, first, count);
    int axis = bvh_longest_axis(centroid_bounds);
    bvh_sort_range(ctx, (int)first, (int)(first + count - 1u), axis);
    uint32_t mid = first + count / 2u;
    if (mid == first || mid == first + count) mid = first + SDL_max(1u, count / 2u);

    uint32_t left = bvh_build_node(ctx, first, mid - first);
    uint32_t right = bvh_build_node(ctx, mid, first + count - mid);
    if (left == BVH_INVALID || right == BVH_INVALID) return BVH_INVALID;

    node = &bvh->nodes[node_index];
    node->left = left;
    node->right = right;
    node->leaf = 0;
    node->first = 0;
    node->count = 0;
    node->bounds = bvh_union(bvh->nodes[left].bounds, bvh->nodes[right].bounds);
    return node_index;
}

static void bvh_destroy(BVH *bvh, SDL_GPUDevice *device)
{
    if (!bvh) return;
    (void)device;
    SDL_free(bvh->primitive_bounds);
    SDL_free(bvh->primitive_indices);
    SDL_free(bvh->nodes);
    SDL_free(bvh);
}

static BVH *bvh_build_from_primitives(BVH_PRIMITIVE *primitives, uint32_t count, uint32_t leaf_size)
{
    if (!primitives || count == 0) return NULL;
    if (leaf_size == 0) leaf_size = BVH_DEFAULT_LEAF_SIZE;

    BVH *bvh = SDL_calloc(1, sizeof(*bvh));
    if (!bvh) return NULL;
    bvh->primitive_count = count;
    bvh->leaf_size = leaf_size;
    bvh->node_capacity = count * 2u;
    bvh->nodes = SDL_calloc(bvh->node_capacity, sizeof(*bvh->nodes));
    bvh->primitive_indices = SDL_malloc((size_t)count * sizeof(*bvh->primitive_indices));
    bvh->primitive_bounds = SDL_malloc((size_t)count * sizeof(*bvh->primitive_bounds));
    if (!bvh->nodes || !bvh->primitive_indices || !bvh->primitive_bounds) {
        bvh_destroy(bvh, NULL);
        return NULL;
    }

    for (uint32_t i = 0; i < count; ++i) {
        bvh->primitive_indices[i] = primitives[i].id;
        bvh->primitive_bounds[primitives[i].id] = primitives[i].bounds;
    }

    BVH_BUILD_CONTEXT ctx = {.bvh = bvh, .primitives = primitives};
    if (bvh_build_node(&ctx, 0, count) == BVH_INVALID) {
        bvh_destroy(bvh, NULL);
        return NULL;
    }
    return bvh;
}

static AABB bvh_triangle_bounds(TRIANGLE t)
{
    AABB b = {t.a, t.a};
    AABB pb = {t.b, t.b};
    AABB pc = {t.c, t.c};
    b = bvh_union(b, pb);
    b = bvh_union(b, pc);
    const float e = 1e-7f;
    if (b.max.x - b.min.x < e) { b.min.x -= e; b.max.x += e; }
    if (b.max.y - b.min.y < e) { b.min.y -= e; b.max.y += e; }
    if (b.max.z - b.min.z < e) { b.min.z -= e; b.max.z += e; }
    return b;
}

int bvh_build_mesh(MESH *mesh, uint32_t leaf_size)
{
    if (!mesh) return 0;
    if (mesh->blas) bvh_destroy((BVH *)mesh->blas, NULL);
    mesh->blas = NULL;

    uint32_t triangle_count = mesh->index_count / 3u;
    if (triangle_count == 0) return 1;

    BVH_PRIMITIVE *primitives = SDL_malloc((size_t)triangle_count * sizeof(*primitives));
    if (!primitives) return 0;

    for (uint32_t i = 0; i < triangle_count; ++i) {
        TRIANGLE tri;
        if (!m_triangle(mesh, i, &tri)) {
            SDL_free(primitives);
            return 0;
        }
        AABB bounds = bvh_triangle_bounds(tri);
        primitives[i] = (BVH_PRIMITIVE){bounds, bvh_centroid(bounds), i};
    }

    BVH *bvh = bvh_build_from_primitives(primitives, triangle_count, leaf_size);
    SDL_free(primitives);
    if (!bvh) return 0;
    bvh->geometry_revision = mesh->geometry_revision;
    mesh->blas = (struct Bvh *)bvh;
    return 1;
}

void bvh_destroy_mesh(MESH *mesh, SDL_GPUDevice *device)
{
    if (!mesh || !mesh->blas) return;
    bvh_destroy((BVH *)mesh->blas, device);
    mesh->blas = NULL;
}

static int bvh_validate_node(const BVH *bvh, uint32_t node_index)
{
    if (!bvh || node_index >= bvh->node_count) return 0;
    const BVH_NODE *node = &bvh->nodes[node_index];
    if (node->leaf) {
        if (node->count == 0 || node->first + node->count > bvh->primitive_count) return 0;
        for (uint32_t i = 0; i < node->count; ++i) {
            uint32_t id = bvh->primitive_indices[node->first + i];
            if (id >= bvh->primitive_count || !bvh_contains(node->bounds, bvh->primitive_bounds[id])) return 0;
        }
        return 1;
    }
    if (node->left >= bvh->node_count || node->right >= bvh->node_count) return 0;
    if (!bvh_contains(node->bounds, bvh->nodes[node->left].bounds)) return 0;
    if (!bvh_contains(node->bounds, bvh->nodes[node->right].bounds)) return 0;
    return bvh_validate_node(bvh, node->left) && bvh_validate_node(bvh, node->right);
}

int bvh_validate(const BVH *bvh)
{
    return bvh && bvh->node_count > 0 && bvh_validate_node(bvh, 0);
}

static void bvh_collect_debug_depth(
    const BVH *bvh,
    uint32_t node_index,
    uint32_t current_depth,
    uint32_t target_depth,
    AABB *out_bounds,
    uint32_t capacity,
    uint32_t *count)
{
    if (!bvh || node_index >= bvh->node_count || !count) return;
    const BVH_NODE *node = &bvh->nodes[node_index];
    if (current_depth == target_depth || node->leaf) {
        if (*count < capacity && out_bounds) out_bounds[*count] = node->bounds;
        ++(*count);
        return;
    }
    bvh_collect_debug_depth(bvh, node->left, current_depth + 1u, target_depth, out_bounds, capacity, count);
    bvh_collect_debug_depth(bvh, node->right, current_depth + 1u, target_depth, out_bounds, capacity, count);
}

uint32_t bvh_debug_bounds_at_depth(
    const BVH *bvh,
    uint32_t depth,
    AABB *out_bounds,
    uint32_t capacity)
{
    if (!bvh || bvh->node_count == 0) return 0;
    uint32_t count = 0;
    bvh_collect_debug_depth(bvh, 0, 0, depth, out_bounds, capacity, &count);
    return count;
}

static RAY_HIT bvh_mesh_raycast_internal(
    const MESH *mesh,
    const BVH *bvh,
    RAY ray,
    int cull_backfaces,
    float orientation_sign,
    BVH_TRACE_STATS *stats,
    int any_hit)
{
    RAY_HIT best = {0};
    best.t = ray.t_max;
    best.triangle_index = UINT32_MAX;
    best.instance_index = UINT32_MAX;
    if (!mesh || !bvh || bvh->node_count == 0) return best;

    if (stats) ++stats->aabb_tests;
    if (!ray_aabb(ray, bvh->nodes[0].bounds, NULL)) return best;

    uint32_t stack[128];
    uint32_t top = 0;
    stack[top++] = 0;

    while (top) {
        uint32_t node_index = stack[--top];
        const BVH_NODE *node = &bvh->nodes[node_index];
        if (stats) ++stats->nodes_visited;

        if (node->leaf) {
            for (uint32_t i = 0; i < node->count; ++i) {
                uint32_t tri_index = bvh->primitive_indices[node->first + i];
                TRIANGLE tri;
                if (!m_triangle(mesh, tri_index, &tri)) continue;
                if (stats) ++stats->triangle_tests;

                RAY clipped = ray;
                clipped.t_max = best.hit ? SDL_min(ray.t_max, best.t + RAY_TIE_EPSILON) : best.t;
                float t, u, v;
                if (!ray_triangle_ex(clipped, tri, cull_backfaces, orientation_sign, &t, &u, &v)) continue;

                if (!best.hit || t < best.t - RAY_TIE_EPSILON ||
                    (fabsf(t - best.t) <= RAY_TIE_EPSILON && tri_index < best.triangle_index)) {
                    best.hit = 1;
                    best.t = t;
                    best.triangle_index = tri_index;
                    best.bary_u = u;
                    best.bary_v = v;
                    best.position = c_add(ray.origin, c_mul(ray.direction, t));
                    best.geometric_normal = m_triangle_normal(tri);
                    if (any_hit) return best;
                }
            }
            continue;
        }

        RAY clipped = ray;
        clipped.t_max = best.hit ? SDL_min(ray.t_max, best.t + RAY_TIE_EPSILON) : best.t;
        float left_near = 0.0f, right_near = 0.0f;
        if (stats) stats->aabb_tests += 2;
        int hit_left = ray_aabb(clipped, bvh->nodes[node->left].bounds, &left_near);
        int hit_right = ray_aabb(clipped, bvh->nodes[node->right].bounds, &right_near);

        if (hit_left && hit_right) {
            uint32_t near_child = left_near <= right_near ? node->left : node->right;
            uint32_t far_child = left_near <= right_near ? node->right : node->left;
            if (top + 2u <= 128u) {
                stack[top++] = far_child;
                stack[top++] = near_child;
            }
        } else if (hit_left) {
            if (top < 128u) stack[top++] = node->left;
        } else if (hit_right) {
            if (top < 128u) stack[top++] = node->right;
        }
    }

    return best;
}

RAY_HIT bvh_mesh_raycast(
    const MESH *mesh,
    const BVH *bvh,
    RAY ray,
    int cull_backfaces,
    float orientation_sign,
    BVH_TRACE_STATS *stats)
{
    return bvh_mesh_raycast_internal(mesh, bvh, ray, cull_backfaces, orientation_sign, stats, 0);
}

int bvh_mesh_occluded(
    const MESH *mesh,
    const BVH *bvh,
    RAY ray,
    int cull_backfaces,
    float orientation_sign,
    BVH_TRACE_STATS *stats)
{
    return bvh_mesh_raycast_internal(mesh, bvh, ray, cull_backfaces, orientation_sign, stats, 1).hit;
}

static float bvh_quality_node(const BVH *bvh, uint32_t index)
{
    const BVH_NODE *node = &bvh->nodes[index];
    if (node->leaf) return 0.0f;
    return bvh_surface_area(node->bounds) + bvh_quality_node(bvh, node->left) + bvh_quality_node(bvh, node->right);
}

static AABB bvh_refit_node(BVH *bvh, uint32_t node_index)
{
    BVH_NODE *node = &bvh->nodes[node_index];
    if (node->leaf) {
        AABB b = bvh_empty_bounds();
        for (uint32_t i = 0; i < node->count; ++i) {
            uint32_t id = bvh->primitive_indices[node->first + i];
            b = bvh_union(b, bvh->primitive_bounds[id]);
        }
        node->bounds = b;
        return b;
    }
    AABB left = bvh_refit_node(bvh, node->left);
    AABB right = bvh_refit_node(bvh, node->right);
    node->bounds = bvh_union(left, right);
    return node->bounds;
}

static int bvh_build_tlas(SCENE *scene, uint32_t leaf_size)
{
    if (!scene) return 0;
    if (scene->tlas) {
        bvh_destroy((BVH *)scene->tlas, NULL);
        scene->tlas = NULL;
    }
    if (scene->instance_count == 0) return 1;

    BVH_PRIMITIVE *primitives = SDL_malloc((size_t)scene->instance_count * sizeof(*primitives));
    if (!primitives) return 0;
    for (uint32_t i = 0; i < scene->instance_count; ++i) {
        AABB b = scene->instances[i].world_bounds;
        primitives[i] = (BVH_PRIMITIVE){b, bvh_centroid(b), i};
    }
    BVH *tlas = bvh_build_from_primitives(primitives, scene->instance_count, leaf_size);
    SDL_free(primitives);
    if (!tlas) return 0;
    tlas->source_revision = scene->transform_revision;
    tlas->geometry_revision = scene->geometry_revision;
    tlas->baseline_quality = bvh_quality_node(tlas, 0);
    tlas->current_quality = tlas->baseline_quality;
    scene->tlas = (struct Bvh *)tlas;
    return 1;
}

static int bvh_refit_tlas(SCENE *scene)
{
    if (!scene || !scene->tlas) return 0;
    BVH *tlas = (BVH *)scene->tlas;
    if (tlas->primitive_count != scene->instance_count) return 0;
    for (uint32_t i = 0; i < scene->instance_count; ++i) tlas->primitive_bounds[i] = scene->instances[i].world_bounds;
    bvh_refit_node(tlas, 0);
    tlas->source_revision = scene->transform_revision;
    tlas->current_quality = bvh_quality_node(tlas, 0);
    return 1;
}

static int bvh_upload_raw(SDL_GPUDevice *device, SDL_GPUBuffer **buffer, uint32_t size, const void *data, const char *name)
{
    if (!device || !buffer || !data || size == 0) return 0;
    if (*buffer) {
        SDL_ReleaseGPUBuffer(device, *buffer);
        *buffer = NULL;
    }
    SDL_GPUBufferCreateInfo info = {
        .usage = SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_READ,
        .size = size
    };
    *buffer = SDL_CreateGPUBuffer(device, &info);
    if (!*buffer) return 0;
    SDL_SetGPUBufferName(device, *buffer, name);
    return m_load(device, *buffer, data, size);
}

typedef struct BvhFlattenMap {
    uint32_t *cpu_to_flat;
    uint32_t cursor;
    const BVH *bvh;
    GPU_BVH_NODE *out;
} BVH_FLATTEN_MAP;

static void bvh_flatten_copy(BVH_FLATTEN_MAP *ctx, uint32_t cpu_index)
{
    uint32_t flat_index = ctx->cursor++;
    ctx->cpu_to_flat[cpu_index] = flat_index;
    const BVH_NODE *node = &ctx->bvh->nodes[cpu_index];
    GPU_BVH_NODE *out = &ctx->out[flat_index];
    SDL_memset(out, 0, sizeof(*out));
    out->min_x = node->bounds.min.x;
    out->min_y = node->bounds.min.y;
    out->min_z = node->bounds.min.z;
    out->max_x = node->bounds.max.x;
    out->max_y = node->bounds.max.y;
    out->max_z = node->bounds.max.z;
    out->meta = node->leaf ? 1u : 0u;
    out->first = node->first;
    out->count = node->count;
    out->escape = BVH_INVALID;
    out->right = BVH_INVALID;
    if (!node->leaf) {
        bvh_flatten_copy(ctx, node->left);
        bvh_flatten_copy(ctx, node->right);
    }
}

static void bvh_flatten_assign(const BVH *bvh, uint32_t cpu_index, uint32_t escape, uint32_t *map, GPU_BVH_NODE *out)
{
    uint32_t flat_index = map[cpu_index];
    const BVH_NODE *node = &bvh->nodes[cpu_index];
    out[flat_index].escape = escape;
    if (!node->leaf) {
        uint32_t right = map[node->right];
        out[flat_index].right = right;
        bvh_flatten_assign(bvh, node->left, right, map, out);
        bvh_flatten_assign(bvh, node->right, escape, map, out);
    }
}

int bvh_flatten(const BVH *bvh, GPU_BVH_NODE **out_nodes, uint32_t *out_count)
{
    if (!bvh || !out_nodes || !out_count || bvh->node_count == 0) return 0;
    GPU_BVH_NODE *nodes = SDL_calloc(bvh->node_count, sizeof(*nodes));
    uint32_t *map = SDL_malloc((size_t)bvh->node_count * sizeof(*map));
    if (!nodes || !map) {
        SDL_free(nodes);
        SDL_free(map);
        return 0;
    }
    BVH_FLATTEN_MAP ctx = {.cpu_to_flat = map, .cursor = 0, .bvh = bvh, .out = nodes};
    bvh_flatten_copy(&ctx, 0);
    bvh_flatten_assign(bvh, 0, BVH_INVALID, map, nodes);
    SDL_free(map);
    *out_nodes = nodes;
    *out_count = bvh->node_count;
    return 1;
}

static void scene_gpu_destroy(SCENE *scene, SDL_GPUDevice *device)
{
    if (!scene || !scene->gpu_accel) return;
    SCENE_GPU_ACCEL *gpu = (SCENE_GPU_ACCEL *)scene->gpu_accel;
    if (device) {
        if (gpu->instances) SDL_ReleaseGPUBuffer(device, gpu->instances);
        if (gpu->mesh_accels) SDL_ReleaseGPUBuffer(device, gpu->mesh_accels);
        if (gpu->indices) SDL_ReleaseGPUBuffer(device, gpu->indices);
        if (gpu->vertices) SDL_ReleaseGPUBuffer(device, gpu->vertices);
        if (gpu->blas_primitives) SDL_ReleaseGPUBuffer(device, gpu->blas_primitives);
        if (gpu->blas_nodes) SDL_ReleaseGPUBuffer(device, gpu->blas_nodes);
        if (gpu->tlas_primitives) SDL_ReleaseGPUBuffer(device, gpu->tlas_primitives);
        if (gpu->tlas_nodes) SDL_ReleaseGPUBuffer(device, gpu->tlas_nodes);
    }
    SDL_free(gpu);
    scene->gpu_accel = NULL;
}

static int scene_gpu_upload_geometry(SCENE *scene, SDL_GPUDevice *device)
{
    if (!scene || !device || !scene->gpu_accel) return 0;
    SCENE_GPU_ACCEL *gpu = (SCENE_GPU_ACCEL *)scene->gpu_accel;

    uint32_t node_count = 0;
    uint32_t primitive_count = 0;
    uint32_t vertex_count = 0;
    uint32_t index_count = 0;
    for (uint32_t i = 0; i < scene->mesh_count; ++i) {
        const MESH *mesh = &scene->meshes[i];
        const BVH *blas = (const BVH *)mesh->blas;
        if (mesh->index_count && !blas) return 0;
        if (blas) {
            node_count += blas->node_count;
            primitive_count += blas->primitive_count;
        }
        vertex_count += mesh->vertex_count;
        index_count += mesh->index_count;
    }

    if (!node_count || !primitive_count || !vertex_count || !index_count || !scene->mesh_count) return 0;

    GPU_BVH_NODE *nodes = SDL_malloc((size_t)node_count * sizeof(*nodes));
    uint32_t *primitives = SDL_malloc((size_t)primitive_count * sizeof(*primitives));
    VERTEX *vertices = SDL_malloc((size_t)vertex_count * sizeof(*vertices));
    uint32_t *indices = SDL_malloc((size_t)index_count * sizeof(*indices));
    GPU_MESH_ACCEL *mesh_accels = SDL_calloc(scene->mesh_count, sizeof(*mesh_accels));
    if (!nodes || !primitives || !vertices || !indices || !mesh_accels) {
        SDL_free(nodes);
        SDL_free(primitives);
        SDL_free(vertices);
        SDL_free(indices);
        SDL_free(mesh_accels);
        return 0;
    }

    uint32_t node_offset = 0;
    uint32_t primitive_offset = 0;
    uint32_t vertex_offset = 0;
    uint32_t index_offset = 0;
    int ok = 1;

    for (uint32_t i = 0; i < scene->mesh_count && ok; ++i) {
        const MESH *mesh = &scene->meshes[i];
        const BVH *blas = (const BVH *)mesh->blas;
        GPU_MESH_ACCEL *meta = &mesh_accels[i];
        meta->node_offset = node_offset;
        meta->primitive_offset = primitive_offset;
        meta->vertex_offset = vertex_offset;
        meta->index_offset = index_offset;
        meta->node_count = blas ? blas->node_count : 0;
        meta->primitive_count = blas ? blas->primitive_count : 0;

        if (blas) {
            GPU_BVH_NODE *flat = NULL;
            uint32_t flat_count = 0;
            if (!bvh_flatten(blas, &flat, &flat_count)) {
                ok = 0;
                break;
            }
            for (uint32_t n = 0; n < flat_count; ++n) {
                GPU_BVH_NODE node = flat[n];
                if (node.escape != BVH_INVALID) node.escape += node_offset;
                if (node.right != BVH_INVALID) node.right += node_offset;
                if (node.meta & 1u) node.first += primitive_offset;
                nodes[node_offset + n] = node;
            }
            SDL_free(flat);
            SDL_memcpy(
                &primitives[primitive_offset],
                blas->primitive_indices,
                (size_t)blas->primitive_count * sizeof(*primitives));
            node_offset += blas->node_count;
            primitive_offset += blas->primitive_count;
        }

        if (mesh->vertex_count) {
            SDL_memcpy(&vertices[vertex_offset], mesh->vertices, (size_t)mesh->vertex_count * sizeof(*vertices));
        }
        if (mesh->index_count) {
            SDL_memcpy(&indices[index_offset], mesh->indices, (size_t)mesh->index_count * sizeof(*indices));
        }
        vertex_offset += mesh->vertex_count;
        index_offset += mesh->index_count;
    }

    if (ok) {
        ok = bvh_upload_raw(device, &gpu->blas_nodes, node_count * sizeof(*nodes), nodes, "Scene.BLAS.Nodes") &&
             bvh_upload_raw(device, &gpu->blas_primitives, primitive_count * sizeof(*primitives), primitives, "Scene.BLAS.Primitives") &&
             bvh_upload_raw(device, &gpu->vertices, vertex_count * sizeof(*vertices), vertices, "Scene.RayVertices") &&
             bvh_upload_raw(device, &gpu->indices, index_count * sizeof(*indices), indices, "Scene.RayIndices") &&
             bvh_upload_raw(device, &gpu->mesh_accels, scene->mesh_count * sizeof(*mesh_accels), mesh_accels, "Scene.MeshAccel");
    }

    SDL_free(nodes);
    SDL_free(primitives);
    SDL_free(vertices);
    SDL_free(indices);
    SDL_free(mesh_accels);
    if (ok) {
        gpu->geometry_revision = scene->geometry_revision;
        gpu->mesh_count = scene->mesh_count;
    }
    return ok;
}

static int scene_gpu_upload_dynamic(SCENE *scene, SDL_GPUDevice *device)
{
    if (!scene || !device || !scene->gpu_accel || !scene->tlas || !scene->instance_count) return 0;
    SCENE_GPU_ACCEL *gpu = (SCENE_GPU_ACCEL *)scene->gpu_accel;
    const BVH *tlas = (const BVH *)scene->tlas;

    GPU_BVH_NODE *nodes = NULL;
    uint32_t node_count = 0;
    GPU_INSTANCE *instances = SDL_calloc(scene->instance_count, sizeof(*instances));
    if (!instances || !bvh_flatten(tlas, &nodes, &node_count)) {
        SDL_free(instances);
        SDL_free(nodes);
        return 0;
    }

    for (uint32_t i = 0; i < scene->instance_count; ++i) {
        const RENDER_INSTANCE *src = &scene->instances[i];
        GPU_INSTANCE *dst = &instances[i];
        SDL_memcpy(dst->world, src->transform.matrix, sizeof(dst->world));
        SDL_memcpy(dst->inverse, src->world_inverse, sizeof(dst->inverse));
        dst->mesh_accel_index = src->mesh;
        dst->flags = (src->mobility == MOBILITY_DYNAMIC ? 1u : 0u) |
            (src->orientation_sign < 0.0f ? 2u : 0u);
        dst->query_mask = src->query_mask;
    }

    int ok = bvh_upload_raw(device, &gpu->tlas_nodes, node_count * sizeof(*nodes), nodes, "Scene.TLAS.Nodes") &&
             bvh_upload_raw(device, &gpu->tlas_primitives,
                 tlas->primitive_count * sizeof(*tlas->primitive_indices), tlas->primitive_indices, "Scene.TLAS.Primitives") &&
             bvh_upload_raw(device, &gpu->instances,
                 scene->instance_count * sizeof(*instances), instances, "Scene.Instances");

    SDL_free(nodes);
    SDL_free(instances);
    if (ok) {
        gpu->transform_revision = scene->transform_revision;
        gpu->instance_count = scene->instance_count;
    }
    return ok;
}

static int scene_gpu_sync(SCENE *scene, SDL_GPUDevice *device)
{
    if (!device) return 1;
    if (!scene->gpu_accel) {
        scene->gpu_accel = (struct SceneGpuAccel *)SDL_calloc(1, sizeof(SCENE_GPU_ACCEL));
        if (!scene->gpu_accel) return 0;
    }
    SCENE_GPU_ACCEL *gpu = (SCENE_GPU_ACCEL *)scene->gpu_accel;
    int geometry_dirty = gpu->geometry_revision != scene->geometry_revision || gpu->mesh_count != scene->mesh_count;
    int transform_dirty = geometry_dirty || gpu->transform_revision != scene->transform_revision || gpu->instance_count != scene->instance_count;
    if (geometry_dirty && !scene_gpu_upload_geometry(scene, device)) return 0;
    if (transform_dirty && !scene_gpu_upload_dynamic(scene, device)) return 0;
    return 1;
}

int scene_accel_update(SCENE *scene, SDL_GPUDevice *device)
{
    if (!scene || !scene->meshes) return 0;

    for (uint32_t i = 0; i < scene->mesh_count; ++i) {
        MESH *mesh = &scene->meshes[i];
        BVH *blas = (BVH *)mesh->blas;
        if (!blas || blas->geometry_revision != mesh->geometry_revision) {
            if (mesh->blas) bvh_destroy_mesh(mesh, device);
            if (!bvh_build_mesh(mesh, BVH_DEFAULT_LEAF_SIZE)) return 0;
        }
    }

    BVH *tlas = (BVH *)scene->tlas;
    if (!tlas || tlas->primitive_count != scene->instance_count || tlas->geometry_revision != scene->geometry_revision) {
        if (scene->tlas) {
            bvh_destroy((BVH *)scene->tlas, device);
            scene->tlas = NULL;
        }
        if (!bvh_build_tlas(scene, BVH_DEFAULT_LEAF_SIZE)) return 0;
        return scene_gpu_sync(scene, device);
    }

    if (tlas->source_revision != scene->transform_revision) {
        if (!bvh_refit_tlas(scene)) return 0;
        tlas = (BVH *)scene->tlas;
        if (tlas->baseline_quality > 0.0f && tlas->current_quality > tlas->baseline_quality * 2.0f) {
            bvh_destroy(tlas, device);
            scene->tlas = NULL;
            if (!bvh_build_tlas(scene, BVH_DEFAULT_LEAF_SIZE)) return 0;
            tlas = (BVH *)scene->tlas;
        }
    }
    return scene_gpu_sync(scene, device);
}

void scene_accel_destroy(SCENE *scene, SDL_GPUDevice *device)
{
    if (!scene) return;
    scene_gpu_destroy(scene, device);
    if (scene->tlas) {
        bvh_destroy((BVH *)scene->tlas, device);
        scene->tlas = NULL;
    }
    if (scene->meshes) {
        for (uint32_t i = 0; i < scene->mesh_count; ++i) bvh_destroy_mesh(&scene->meshes[i], device);
    }
}

static int bvh_tlas_filter_instance(const QUERY_FILTER *filter, const RENDER_INSTANCE *instance)
{
    if ((filter->mask & instance->query_mask) == 0) return 0;
    if (instance->mobility == MOBILITY_STATIC && !filter->include_static) return 0;
    if (instance->mobility == MOBILITY_DYNAMIC && !filter->include_dynamic) return 0;
    return 1;
}

RAY_HIT bvh_scene_raycast(const SCENE *scene, RAY ray, QUERY_FILTER filter, BVH_TRACE_STATS *stats)
{
    RAY_HIT best = {0};
    best.t = ray.t_max;
    best.triangle_index = UINT32_MAX;
    best.instance_index = UINT32_MAX;
    if (!scene || !scene->tlas) return best;
    const BVH *tlas = (const BVH *)scene->tlas;

    if (stats) ++stats->aabb_tests;
    if (!ray_aabb(ray, tlas->nodes[0].bounds, NULL)) return best;

    uint32_t stack[128];
    uint32_t top = 0;
    stack[top++] = 0;
    while (top) {
        const BVH_NODE *node = &tlas->nodes[stack[--top]];
        if (stats) ++stats->nodes_visited;
        if (node->leaf) {
            for (uint32_t i = 0; i < node->count; ++i) {
                uint32_t instance_index = tlas->primitive_indices[node->first + i];
                if (instance_index >= scene->instance_count) continue;
                const RENDER_INSTANCE *instance = &scene->instances[instance_index];
                if (instance->mesh >= scene->mesh_count || !bvh_tlas_filter_instance(&filter, instance)) continue;

                RAY broad = ray;
                broad.t_max = best.hit ? SDL_min(ray.t_max, best.t + RAY_TIE_EPSILON) : best.t;
                if (stats) ++stats->aabb_tests;
                if (!ray_aabb(broad, instance->world_bounds, NULL)) continue;

                RAY local = {
                    .origin = transform_point(instance->world_inverse, ray.origin),
                    .direction = transform_direction(instance->world_inverse, ray.direction),
                    .t_min = ray.t_min,
                    .t_max = best.hit ? SDL_min(ray.t_max, best.t + RAY_TIE_EPSILON) : best.t
                };
                const MESH *mesh = &scene->meshes[instance->mesh];
                const BVH *blas = (const BVH *)mesh->blas;
                RAY_HIT local_hit = bvh_mesh_raycast(mesh, blas, local,
                    filter.cull_backfaces, instance->orientation_sign, stats);
                if (!local_hit.hit) continue;

                VEC3 world_position = transform_point(instance->transform.matrix, local_hit.position);
                float world_t = c_len(c_sub(world_position, ray.origin));
                if (world_t < ray.t_min || world_t > best.t + RAY_TIE_EPSILON) continue;
                int better = !best.hit || world_t < best.t - RAY_TIE_EPSILON ||
                    (fabsf(world_t - best.t) <= RAY_TIE_EPSILON &&
                        (instance_index < best.instance_index ||
                         (instance_index == best.instance_index && local_hit.triangle_index < best.triangle_index)));
                if (!better) continue;
                best = local_hit;
                best.t = world_t;
                best.position = world_position;
                best.geometric_normal = transform_normal_inverse_transpose(
                    instance->world_inverse, local_hit.geometric_normal, instance->orientation_sign);
                best.instance_index = instance_index;
            }
            continue;
        }

        RAY clipped = ray;
        clipped.t_max = best.hit ? SDL_min(ray.t_max, best.t + RAY_TIE_EPSILON) : best.t;
        float ln = 0.0f, rn = 0.0f;
        if (stats) stats->aabb_tests += 2;
        int hl = ray_aabb(clipped, tlas->nodes[node->left].bounds, &ln);
        int hr = ray_aabb(clipped, tlas->nodes[node->right].bounds, &rn);
        if (hl && hr) {
            uint32_t near_child = ln <= rn ? node->left : node->right;
            uint32_t far_child = ln <= rn ? node->right : node->left;
            if (top + 2u <= 128u) {
                stack[top++] = far_child;
                stack[top++] = near_child;
            }
        } else if (hl) {
            if (top < 128u) stack[top++] = node->left;
        } else if (hr) {
            if (top < 128u) stack[top++] = node->right;
        }
    }
    return best;
}

int bvh_scene_occluded(const SCENE *scene, RAY ray, QUERY_FILTER filter, BVH_TRACE_STATS *stats)
{
    if (!scene || !scene->tlas) return 0;
    const BVH *tlas = (const BVH *)scene->tlas;
    if (stats) ++stats->aabb_tests;
    if (!ray_aabb(ray, tlas->nodes[0].bounds, NULL)) return 0;

    uint32_t stack[128];
    uint32_t top = 0;
    stack[top++] = 0;
    while (top) {
        const BVH_NODE *node = &tlas->nodes[stack[--top]];
        if (stats) ++stats->nodes_visited;
        if (node->leaf) {
            for (uint32_t i = 0; i < node->count; ++i) {
                uint32_t instance_index = tlas->primitive_indices[node->first + i];
                if (instance_index >= scene->instance_count) continue;
                const RENDER_INSTANCE *instance = &scene->instances[instance_index];
                if (instance->mesh >= scene->mesh_count || !bvh_tlas_filter_instance(&filter, instance)) continue;
                if (stats) ++stats->aabb_tests;
                if (!ray_aabb(ray, instance->world_bounds, NULL)) continue;

                RAY local = {
                    .origin = transform_point(instance->world_inverse, ray.origin),
                    .direction = transform_direction(instance->world_inverse, ray.direction),
                    .t_min = ray.t_min,
                    .t_max = ray.t_max
                };
                const MESH *mesh = &scene->meshes[instance->mesh];
                if (bvh_mesh_occluded(mesh, (const BVH *)mesh->blas, local,
                    filter.cull_backfaces, instance->orientation_sign, stats)) return 1;
            }
            continue;
        }
        if (stats) stats->aabb_tests += 2;
        float ln = 0.0f, rn = 0.0f;
        int hl = ray_aabb(ray, tlas->nodes[node->left].bounds, &ln);
        int hr = ray_aabb(ray, tlas->nodes[node->right].bounds, &rn);
        if (hl && hr) {
            uint32_t near_child = ln <= rn ? node->left : node->right;
            uint32_t far_child = ln <= rn ? node->right : node->left;
            if (top + 2u <= 128u) {
                stack[top++] = far_child;
                stack[top++] = near_child;
            }
        } else if (hl) {
            if (top < 128u) stack[top++] = node->left;
        } else if (hr) {
            if (top < 128u) stack[top++] = node->right;
        }
    }
    return 0;
}

RAY_HIT scene_raycast(const SCENE *scene, RAY ray, QUERY_FILTER filter)
{
    if (scene && scene->tlas) return bvh_scene_raycast(scene, ray, filter, NULL);
    return scene_raycast_bruteforce(scene, ray, filter);
}

int scene_occluded(const SCENE *scene, RAY ray, QUERY_FILTER filter)
{
    if (scene && scene->tlas) return bvh_scene_occluded(scene, ray, filter, NULL);
    return scene_occluded_bruteforce(scene, ray, filter);
}

typedef struct GpuRayInput {
    float origin[3];
    float t_min;
    float direction[3];
    float t_max;
} GPU_RAY_INPUT;

_Static_assert(sizeof(GPU_RAY_INPUT) == 32, "GPU_RAY_INPUT layout mismatch");

typedef struct GpuSceneHit {
    uint32_t hit;
    uint32_t triangle_index;
    uint32_t instance_index;
    float t;
} GPU_SCENE_HIT;

_Static_assert(sizeof(GPU_SCENE_HIT) == 16, "GPU_SCENE_HIT layout mismatch");

typedef struct GpuSceneWork {
    GPU_RAY_INPUT ray;
    GPU_SCENE_HIT hit;
} GPU_SCENE_WORK;

_Static_assert(sizeof(GPU_SCENE_WORK) == 48, "GPU_SCENE_WORK layout mismatch");

static const char bvh_gpu_scene_verify_cs[] =
    "struct BvhNode { float3 bmin; uint meta; float3 bmax; uint first; uint count; uint escape; uint right; uint pad; };\n"
    "struct MeshAccel { uint node_offset; uint node_count; uint primitive_offset; uint primitive_count; uint vertex_offset; uint index_offset; uint pad0; uint pad1; };\n"
    "struct Instance { float4 world0; float4 world1; float4 world2; float4 world3; float4 inv0; float4 inv1; float4 inv2; float4 inv3; uint mesh; uint flags; uint query_mask; uint pad; };\n"
    "struct RayInput { float3 origin; float t_min; float3 direction; float t_max; };\n"
    "struct SceneHit { uint hit; uint triangle_index; uint instance_index; float t; };\n"
    "StructuredBuffer<BvhNode> TlasNodes : register(t0, space0);\n"
    "StructuredBuffer<uint> TlasPrimitiveIDs : register(t1, space0);\n"
    "StructuredBuffer<BvhNode> BlasNodes : register(t2, space0);\n"
    "StructuredBuffer<uint> BlasPrimitiveIDs : register(t3, space0);\n"
    "ByteAddressBuffer Vertices : register(t4, space0);\n"
    "ByteAddressBuffer Indices : register(t5, space0);\n"
    "StructuredBuffer<MeshAccel> Meshes : register(t6, space0);\n"
    "StructuredBuffer<Instance> Instances : register(t7, space0);\n"
    "struct SceneWork { RayInput ray; SceneHit hit; };\n"
    "RWStructuredBuffer<SceneWork> Work : register(u0, space1);\n"
    "cbuffer Params : register(b0, space2) { uint RayCount; uint3 ParamsPad; };\n"
    "bool RayAabb(RayInput r, float3 mn, float3 mx, float best_t)\n"
    "{\n"
    "    float tmin = r.t_min;\n"
    "    float tmax = min(r.t_max, best_t);\n"
    "    [unroll] for (uint axis = 0; axis < 3; ++axis)\n"
    "    {\n"
    "        float o = r.origin[axis];\n"
    "        float d = r.direction[axis];\n"
    "        if (abs(d) < 1e-8)\n"
    "        {\n"
    "            if (o < mn[axis] || o > mx[axis]) return false;\n"
    "            continue;\n"
    "        }\n"
    "        float inv = 1.0 / d;\n"
    "        float a = (mn[axis] - o) * inv;\n"
    "        float b = (mx[axis] - o) * inv;\n"
    "        if (a > b) { float tmp = a; a = b; b = tmp; }\n"
    "        tmin = max(tmin, a);\n"
    "        tmax = min(tmax, b);\n"
    "        if (tmin > tmax) return false;\n"
    "    }\n"
    "    return true;\n"
    "}\n"
    "bool RayTriangle(RayInput r, float3 a, float3 b, float3 c, float best_t, out float out_t)\n"
    "{\n"
    "    out_t = best_t;\n"
    "    float3 e1 = b - a;\n"
    "    float3 e2 = c - a;\n"
    "    float3 p = cross(r.direction, e2);\n"
    "    float det = dot(e1, p);\n"
    "    if (abs(det) < 1e-7) return false;\n"
    "    float inv_det = 1.0 / det;\n"
    "    float3 s = r.origin - a;\n"
    "    float u = dot(s, p) * inv_det;\n"
    "    if (u < 0.0 || u > 1.0) return false;\n"
    "    float3 q = cross(s, e1);\n"
    "    float v = dot(r.direction, q) * inv_det;\n"
    "    if (v < 0.0 || u + v > 1.0) return false;\n"
    "    float t = dot(e2, q) * inv_det;\n"
    "    if (t < r.t_min || t > min(r.t_max, best_t)) return false;\n"
    "    out_t = t;\n"
    "    return true;\n"
    "}\n"
    "float3 TransformPointInv(Instance inst, float3 p)\n"
    "{\n"
    "    return inst.inv0.xyz * p.x + inst.inv1.xyz * p.y + inst.inv2.xyz * p.z + inst.inv3.xyz;\n"
    "}\n"
    "float3 TransformDirInv(Instance inst, float3 d)\n"
    "{\n"
    "    return inst.inv0.xyz * d.x + inst.inv1.xyz * d.y + inst.inv2.xyz * d.z;\n"
    "}\n"
    "uint LoadIndex(MeshAccel mesh, uint index) { return Indices.Load((mesh.index_offset + index) * 4u); }\n"
    "float3 LoadPosition(MeshAccel mesh, uint vertex_index) { return asfloat(Vertices.Load3((mesh.vertex_offset + vertex_index) * 32u)); }\n"
    "void TraceBlas(RayInput ray, MeshAccel mesh, inout float best_t, out uint best_triangle)\n"
    "{\n"
    "    best_triangle = 0xffffffffu;\n"
    "    if (mesh.node_count == 0u) return;\n"
    "    uint node_index = mesh.node_offset;\n"
    "    while (node_index != 0xffffffffu)\n"
    "    {\n"
    "        BvhNode node = BlasNodes[node_index];\n"
    "        if (!RayAabb(ray, node.bmin, node.bmax, best_t + 1e-5))\n"
    "        {\n"
    "            node_index = node.escape;\n"
    "            continue;\n"
    "        }\n"
    "        if ((node.meta & 1u) != 0u)\n"
    "        {\n"
    "            for (uint i = 0; i < node.count; ++i)\n"
    "            {\n"
    "                uint tri = BlasPrimitiveIDs[node.first + i];\n"
    "                uint base = tri * 3u;\n"
    "                float3 a = LoadPosition(mesh, LoadIndex(mesh, base + 0u));\n"
    "                float3 b = LoadPosition(mesh, LoadIndex(mesh, base + 1u));\n"
    "                float3 c = LoadPosition(mesh, LoadIndex(mesh, base + 2u));\n"
    "                float t;\n"
    "                if (RayTriangle(ray, a, b, c, best_t + 1e-5, t) &&\n"
    "                    (best_triangle == 0xffffffffu || t < best_t - 1e-5 || (abs(t - best_t) <= 1e-5 && tri < best_triangle)))\n"
    "                {\n"
    "                    best_t = t;\n"
    "                    best_triangle = tri;\n"
    "                }\n"
    "            }\n"
    "            node_index = node.escape;\n"
    "        }\n"
    "        else node_index = node_index + 1u;\n"
    "    }\n"
    "}\n"
    "[numthreads(64, 1, 1)]\n"
    "void main(uint3 tid : SV_DispatchThreadID)\n"
    "{\n"
    "    uint ray_index = tid.x;\n"
    "    if (ray_index >= RayCount) return;\n"
    "    RayInput world_ray = Work[ray_index].ray;\n"
    "    float best_t = world_ray.t_max;\n"
    "    uint best_triangle = 0xffffffffu;\n"
    "    uint best_instance = 0xffffffffu;\n"
    "    uint node_index = 0u;\n"
    "    while (node_index != 0xffffffffu)\n"
    "    {\n"
    "        BvhNode node = TlasNodes[node_index];\n"
    "        if (!RayAabb(world_ray, node.bmin, node.bmax, best_t + 1e-5))\n"
    "        {\n"
    "            node_index = node.escape;\n"
    "            continue;\n"
    "        }\n"
    "        if ((node.meta & 1u) != 0u)\n"
    "        {\n"
    "            for (uint i = 0; i < node.count; ++i)\n"
    "            {\n"
    "                uint instance_index = TlasPrimitiveIDs[node.first + i];\n"
    "                Instance inst = Instances[instance_index];\n"
    "                MeshAccel mesh = Meshes[inst.mesh];\n"
    "                RayInput local_ray;\n"
    "                local_ray.origin = TransformPointInv(inst, world_ray.origin);\n"
    "                local_ray.t_min = world_ray.t_min;\n"
    "                local_ray.direction = TransformDirInv(inst, world_ray.direction);\n"
    "                local_ray.t_max = best_t + 1e-5;\n"
    "                float candidate_t = best_t;\n"
    "                uint candidate_triangle;\n"
    "                TraceBlas(local_ray, mesh, candidate_t, candidate_triangle);\n"
    "                if (candidate_triangle != 0xffffffffu &&\n"
    "                    (best_instance == 0xffffffffu || candidate_t < best_t - 1e-5 ||\n"
    "                     (abs(candidate_t - best_t) <= 1e-5 &&\n"
    "                      (instance_index < best_instance || (instance_index == best_instance && candidate_triangle < best_triangle)))))\n"
    "                {\n"
    "                    best_t = candidate_t;\n"
    "                    best_triangle = candidate_triangle;\n"
    "                    best_instance = instance_index;\n"
    "                }\n"
    "            }\n"
    "            node_index = node.escape;\n"
    "        }\n"
    "        else node_index = node_index + 1u;\n"
    "    }\n"
    "    SceneHit result;\n"
    "    result.hit = best_instance != 0xffffffffu ? 1u : 0u;\n"
    "    result.triangle_index = best_triangle;\n"
    "    result.instance_index = best_instance;\n"
    "    result.t = best_t;\n"
    "    Work[ray_index].hit = result;\n"
    "}\n";

static SDL_GPUComputePipeline *bvh_compile_compute(SDL_GPUDevice *device, const char *source)
{
    SDL_ShaderCross_HLSL_Info hinfo = {
        .source = source,
        .entrypoint = "main",
        .include_dir = NULL,
        .defines = NULL,
        .shader_stage = SDL_SHADERCROSS_SHADERSTAGE_COMPUTE,
        .props = 0
    };
    size_t bytecode_size = 0;
    void *spirv = SDL_ShaderCross_CompileSPIRVFromHLSL(&hinfo, &bytecode_size);
    if (!spirv) return NULL;
    SDL_ShaderCross_ComputePipelineMetadata *meta = SDL_ShaderCross_ReflectComputeSPIRV(spirv, bytecode_size, 0);
    if (!meta) {
        SDL_free(spirv);
        return NULL;
    }
    SDL_ShaderCross_SPIRV_Info sinfo = {
        .bytecode = spirv,
        .bytecode_size = bytecode_size,
        .entrypoint = "main",
        .shader_stage = SDL_SHADERCROSS_SHADERSTAGE_COMPUTE,
        .props = 0
    };
    SDL_GPUComputePipeline *pipeline = SDL_ShaderCross_CompileComputePipelineFromSPIRV(device, &sinfo, meta, 0);
    SDL_free(meta);
    SDL_free(spirv);
    return pipeline;
}

static uint32_t bvh_verify_rng(uint32_t *state)
{
    *state = *state * 1664525u + 1013904223u;
    return *state;
}

static float bvh_verify_rand(uint32_t *state)
{
    return ((float)(bvh_verify_rng(state) & 0xffffu) / 32767.5f) - 1.0f;
}

int bvh_gpu_verify_scene(SDL_GPUDevice *device, const SCENE *scene, uint32_t ray_count)
{
    if (!device || !scene || !scene->gpu_accel || !scene->tlas || ray_count == 0) return 0;
    const SCENE_GPU_ACCEL *gpu = (const SCENE_GPU_ACCEL *)scene->gpu_accel;
    const BVH *tlas = (const BVH *)scene->tlas;
    if (!gpu->tlas_nodes || !gpu->tlas_primitives || !gpu->blas_nodes || !gpu->blas_primitives ||
        !gpu->vertices || !gpu->indices || !gpu->mesh_accels || !gpu->instances) return 0;

    GPU_SCENE_WORK *work_items = SDL_calloc(ray_count, sizeof(*work_items));
    RAY_HIT *cpu_hits = SDL_malloc((size_t)ray_count * sizeof(*cpu_hits));
    if (!work_items || !cpu_hits) {
        SDL_free(work_items);
        SDL_free(cpu_hits);
        return 0;
    }

    AABB bounds = tlas->nodes[0].bounds;
    VEC3 center = bvh_centroid(bounds);
    float ex = SDL_max(0.1f, bounds.max.x - bounds.min.x);
    float ey = SDL_max(0.1f, bounds.max.y - bounds.min.y);
    float ez = SDL_max(0.1f, bounds.max.z - bounds.min.z);
    float span = SDL_max(ex, SDL_max(ey, ez));
    uint32_t seed = 0xd00df00du;
    QUERY_FILTER filter = query_filter_all();

    for (uint32_t i = 0; i < ray_count; ++i) {
        VEC3 origin = {
            center.x + bvh_verify_rand(&seed) * ex * 1.4f,
            center.y + bvh_verify_rand(&seed) * ey * 1.4f,
            bounds.max.z + span * (0.8f + 0.8f * (bvh_verify_rand(&seed) + 1.0f))
        };
        VEC3 target = {
            center.x + bvh_verify_rand(&seed) * ex * 0.8f,
            center.y + bvh_verify_rand(&seed) * ey * 0.8f,
            center.z + bvh_verify_rand(&seed) * ez * 0.8f
        };
        RAY ray;
        if (!ray_make(origin, c_sub(target, origin), 0.0f, span * 20.0f + 20.0f, &ray)) {
            SDL_free(work_items);
            SDL_free(cpu_hits);
            return 0;
        }
        work_items[i].ray = (GPU_RAY_INPUT){
            {ray.origin.x, ray.origin.y, ray.origin.z}, ray.t_min,
            {ray.direction.x, ray.direction.y, ray.direction.z}, ray.t_max
        };
        cpu_hits[i] = bvh_scene_raycast(scene, ray, filter, NULL);
    }

    SDL_GPUComputePipeline *pipeline = bvh_compile_compute(device, bvh_gpu_scene_verify_cs);
    if (!pipeline) {
        SDL_Log("GPU scene BVH verify shader compile failed: %s", SDL_GetError());
        SDL_free(work_items);
        SDL_free(cpu_hits);
        return 0;
    }

    SDL_GPUBuffer *work_buffer = NULL;
    SDL_GPUTransferBuffer *download = NULL;
    int ok = 0;

    SDL_GPUBufferCreateInfo winfo = {
        .usage = SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_WRITE,
        .size = ray_count * sizeof(GPU_SCENE_WORK)
    };
    SDL_GPUTransferBufferCreateInfo dinfo = {
        .usage = SDL_GPU_TRANSFERBUFFERUSAGE_DOWNLOAD,
        .size = ray_count * sizeof(GPU_SCENE_WORK)
    };
    work_buffer = SDL_CreateGPUBuffer(device, &winfo);
    download = SDL_CreateGPUTransferBuffer(device, &dinfo);
    if (!work_buffer || !download ||
        !m_load(device, work_buffer, work_items, ray_count * sizeof(GPU_SCENE_WORK))) goto cleanup;

    SDL_GPUCommandBuffer *cmd = SDL_AcquireGPUCommandBuffer(device);
    if (!cmd) goto cleanup;
    uint32_t params[4] = {ray_count, 0, 0, 0};
    SDL_PushGPUComputeUniformData(cmd, 0, params, sizeof(params));
    SDL_GPUStorageBufferReadWriteBinding output_binding = {.buffer = work_buffer, .cycle = false};
    SDL_GPUComputePass *pass = SDL_BeginGPUComputePass(cmd, NULL, 0, &output_binding, 1);
    SDL_BindGPUComputePipeline(pass, pipeline);
    SDL_GPUBuffer *reads[8] = {
        gpu->tlas_nodes,
        gpu->tlas_primitives,
        gpu->blas_nodes,
        gpu->blas_primitives,
        gpu->vertices,
        gpu->indices,
        gpu->mesh_accels,
        gpu->instances
    };
    SDL_BindGPUComputeStorageBuffers(pass, 0, reads, 8);
    SDL_DispatchGPUCompute(pass, (ray_count + 63u) / 64u, 1, 1);
    SDL_EndGPUComputePass(pass);

    SDL_GPUCopyPass *copy = SDL_BeginGPUCopyPass(cmd);
    SDL_GPUBufferRegion src = {.buffer = work_buffer, .offset = 0, .size = ray_count * sizeof(GPU_SCENE_WORK)};
    SDL_GPUTransferBufferLocation dst = {.transfer_buffer = download, .offset = 0};
    SDL_DownloadFromGPUBuffer(copy, &src, &dst);
    SDL_EndGPUCopyPass(copy);

    SDL_GPUFence *fence = SDL_SubmitGPUCommandBufferAndAcquireFence(cmd);
    if (!fence) goto cleanup;
    if (!SDL_WaitForGPUFences(device, true, &fence, 1)) {
        SDL_ReleaseGPUFence(device, fence);
        goto cleanup;
    }
    SDL_ReleaseGPUFence(device, fence);

    GPU_SCENE_WORK *gpu_work = SDL_MapGPUTransferBuffer(device, download, false);
    if (!gpu_work) goto cleanup;
    ok = 1;
    for (uint32_t i = 0; i < ray_count; ++i) {
        const GPU_SCENE_HIT *gpu_hit = &gpu_work[i].hit;
        if ((gpu_hit->hit != 0u) != (cpu_hits[i].hit != 0)) {
            SDL_Log("GPU scene mismatch ray=%u hit cpu=%d gpu=%u cpu_inst=%u gpu_inst=%u cpu_tri=%u gpu_tri=%u cpu_t=%f gpu_t=%f",
                i, cpu_hits[i].hit, gpu_hit->hit,
                cpu_hits[i].instance_index, gpu_hit->instance_index,
                cpu_hits[i].triangle_index, gpu_hit->triangle_index,
                cpu_hits[i].t, gpu_hit->t);
            ok = 0;
            break;
        }
        if (!cpu_hits[i].hit) continue;
        if (gpu_hit->instance_index != cpu_hits[i].instance_index ||
            gpu_hit->triangle_index != cpu_hits[i].triangle_index ||
            fabsf(gpu_hit->t - cpu_hits[i].t) > 2e-4f) {
            SDL_Log("GPU scene mismatch ray=%u cpu_inst=%u gpu_inst=%u cpu_tri=%u gpu_tri=%u cpu_t=%f gpu_t=%f",
                i, cpu_hits[i].instance_index, gpu_hit->instance_index,
                cpu_hits[i].triangle_index, gpu_hit->triangle_index,
                cpu_hits[i].t, gpu_hit->t);
            ok = 0;
            break;
        }
    }
    SDL_UnmapGPUTransferBuffer(device, download);

cleanup:
    if (download) SDL_ReleaseGPUTransferBuffer(device, download);
    if (work_buffer) SDL_ReleaseGPUBuffer(device, work_buffer);
    SDL_ReleaseGPUComputePipeline(device, pipeline);
    SDL_free(work_items);
    SDL_free(cpu_hits);
    if (!ok) SDL_Log("GPU scene BVH parity check failed");
    return ok;
}

#endif // BVH_H
