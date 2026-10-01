#include "game.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct SDF_TRIANGLE {
    VEC3 a;
    VEC3 b;
    VEC3 c;
    VEC3 centroid;
    VEC3 min;
    VEC3 max;
    uint32_t surface_id;
} SDF_TRIANGLE;

typedef struct SDF_BVH_NODE {
    VEC3 min;
    VEC3 max;
    uint32_t first;
    uint32_t count;
    uint32_t left;
    uint32_t right;
} SDF_BVH_NODE;

typedef struct SDF_BUILD {
    SDF_TRIANGLE *triangles;
    SDF_BVH_NODE *nodes;
    uint32_t triangle_count;
    uint32_t node_count;
} SDF_BUILD;

static int sort_axis;

static float component(VEC3 v, int axis) {
    return axis == 0 ? v.x : axis == 1 ? v.y : v.z;
}

static VEC3 vmin3(VEC3 a, VEC3 b) {
    return (VEC3){fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z)};
}

static VEC3 vmax3(VEC3 a, VEC3 b) {
    return (VEC3){fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z)};
}

static VEC3 sub3(VEC3 a, VEC3 b) {
    return (VEC3){a.x - b.x, a.y - b.y, a.z - b.z};
}

static VEC3 add3(VEC3 a, VEC3 b) {
    return (VEC3){a.x + b.x, a.y + b.y, a.z + b.z};
}

static VEC3 scale3(VEC3 a, float s) {
    return (VEC3){a.x * s, a.y * s, a.z * s};
}

static float dot3(VEC3 a, VEC3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

static float length_sq3(VEC3 a) {
    return dot3(a, a);
}

static int triangle_compare(const void *lhs, const void *rhs) {
    const SDF_TRIANGLE *a = lhs;
    const SDF_TRIANGLE *b = rhs;
    const float av = component(a->centroid, sort_axis);
    const float bv = component(b->centroid, sort_axis);

    return av < bv ? -1 : av > bv ? 1 : 0;
}

static float point_aabb_distance_sq(VEC3 p, VEC3 min, VEC3 max) {
    float d = 0.0f;
    const float values[3] = {p.x, p.y, p.z};
    const float lo[3] = {min.x, min.y, min.z};
    const float hi[3] = {max.x, max.y, max.z};

    for (int i = 0; i < 3; ++i) {
        if (values[i] < lo[i]) {
            const float q = lo[i] - values[i];
            d += q * q;
        } else if (values[i] > hi[i]) {
            const float q = values[i] - hi[i];
            d += q * q;
        }
    }

    return d;
}

static float point_triangle_distance_sq(VEC3 p, const SDF_TRIANGLE *triangle) {
    const VEC3 ab = sub3(triangle->b, triangle->a);
    const VEC3 ac = sub3(triangle->c, triangle->a);
    const VEC3 ap = sub3(p, triangle->a);
    const float d1 = dot3(ab, ap);
    const float d2 = dot3(ac, ap);

    if (d1 <= 0.0f && d2 <= 0.0f) return length_sq3(ap);

    const VEC3 bp = sub3(p, triangle->b);
    const float d3 = dot3(ab, bp);
    const float d4 = dot3(ac, bp);

    if (d3 >= 0.0f && d4 <= d3) return length_sq3(bp);

    const float vc = d1 * d4 - d3 * d2;

    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
        const float v = d1 / (d1 - d3);

        return length_sq3(sub3(p, add3(triangle->a, scale3(ab, v))));
    }

    const VEC3 cp = sub3(p, triangle->c);
    const float d5 = dot3(ab, cp);
    const float d6 = dot3(ac, cp);

    if (d6 >= 0.0f && d5 <= d6) return length_sq3(cp);

    const float vb = d5 * d2 - d1 * d6;

    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
        const float w = d2 / (d2 - d6);

        return length_sq3(sub3(p, add3(triangle->a, scale3(ac, w))));
    }

    const float va = d3 * d6 - d5 * d4;

    if (va <= 0.0f && d4 - d3 >= 0.0f && d5 - d6 >= 0.0f) {
        const VEC3 bc = sub3(triangle->c, triangle->b);
        const float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));

        return length_sq3(sub3(p, add3(triangle->b, scale3(bc, w))));
    }

    const float denom = 1.0f / (va + vb + vc);
    const float v = vb * denom;
    const float w = vc * denom;
    const VEC3 q = add3(triangle->a, add3(scale3(ab, v), scale3(ac, w)));

    return length_sq3(sub3(p, q));
}

static uint32_t build_node(SDF_BUILD *build, uint32_t first, uint32_t count) {
    const uint32_t node_index = build->node_count++;
    SDF_BVH_NODE *node = &build->nodes[node_index];
    node->first = first;
    node->count = count;
    node->left = UINT32_MAX;
    node->right = UINT32_MAX;
    node->min = (VEC3){FLT_MAX, FLT_MAX, FLT_MAX};
    node->max = (VEC3){-FLT_MAX, -FLT_MAX, -FLT_MAX};

    VEC3 centroid_min = node->min;
    VEC3 centroid_max = node->max;

    for (uint32_t i = first; i < first + count; ++i) {
        node->min = vmin3(node->min, build->triangles[i].min);
        node->max = vmax3(node->max, build->triangles[i].max);

        centroid_min = vmin3(centroid_min, build->triangles[i].centroid);
        centroid_max = vmax3(centroid_max, build->triangles[i].centroid);
    }

    if (count <= 8u) return node_index;

    const VEC3 extent = sub3(centroid_max, centroid_min);
    sort_axis = extent.y > extent.x ? 1 : 0;

    if (extent.z > component(extent, sort_axis)) sort_axis = 2;

    qsort(build->triangles + first, count, sizeof(*build->triangles), triangle_compare);

    const uint32_t left_count = count / 2u;

    node->left = build_node(build, first, left_count);
    node->right = build_node(build, first + left_count, count - left_count);
    node->count = 0;

    return node_index;
}

static float nearest_distance_sq(const SDF_BUILD *build, uint32_t node_index, VEC3 p, float best, uint32_t *surface_id) {
    const SDF_BVH_NODE *node = &build->nodes[node_index];

    if (point_aabb_distance_sq(p, node->min, node->max) >= best) return best;

    if (node->count) {
        for (uint32_t i = node->first; i < node->first + node->count; ++i) {
            const SDF_TRIANGLE *triangle = &build->triangles[i];
            const float d = point_triangle_distance_sq(p, triangle);

            if (d < best) {
                best = d;
                *surface_id = triangle->surface_id;
            }
        }

        return best;
    }

    const float dl = point_aabb_distance_sq(p, build->nodes[node->left].min, build->nodes[node->left].max);
    const float dr = point_aabb_distance_sq(p, build->nodes[node->right].min, build->nodes[node->right].max);

    if (dl < dr) {
        best = nearest_distance_sq(build, node->left, p, best, surface_id);
        best = nearest_distance_sq(build, node->right, p, best, surface_id);
    } else {
        best = nearest_distance_sq(build, node->right, p, best, surface_id);
        best = nearest_distance_sq(build, node->left, p, best, surface_id);
    }

    return best;
}

bool sdf_build_volume(const MESH *mesh, uint32_t resolution, SDF_VOLUME *volume) {
    if (!mesh || !volume || resolution < 4u || !mesh->vertices.buffer || !mesh->faces.buffer || !mesh->faces.count) return false;

    memset(volume, 0, sizeof(*volume));

    if (mesh->faces.count > UINT32_MAX) return false;

    const uint32_t triangle_count = (uint32_t)mesh->faces.count;

    SDF_BUILD build = {0};

    build.triangle_count = triangle_count;
    build.triangles = malloc((size_t)triangle_count * sizeof(*build.triangles));
    build.nodes = calloc((size_t)triangle_count * 2u, sizeof(*build.nodes));

    if (!build.triangles || !build.nodes) {
        free(build.triangles);
        free(build.nodes);

        return false;
    }

    const POINT *points = mesh->vertices.buffer;
    const MESH_FACE *faces = mesh->faces.buffer;

    for (uint32_t i = 0; i < triangle_count; ++i) {
        const MESH_FACE *face = &faces[i];
        SDF_TRIANGLE *triangle = &build.triangles[i];
        triangle->a = points[face->indices[0]].p;
        triangle->b = points[face->indices[1]].p;
        triangle->c = points[face->indices[2]].p;
        triangle->centroid = scale3(add3(add3(triangle->a, triangle->b), triangle->c), 1.0f / 3.0f);
        triangle->min = vmin3(triangle->a, vmin3(triangle->b, triangle->c));
        triangle->max = vmax3(triangle->a, vmax3(triangle->b, triangle->c));
        triangle->surface_id = i;
    }

    build_node(&build, 0, triangle_count);

    AABB bounds = mesh->bounds;
    const float largest_extent = fmaxf(bounds.extents.x, fmaxf(bounds.extents.y, bounds.extents.z));
    const float padding = fmaxf(0.001f, largest_extent * 0.05f);

    bounds.min = (VEC3){bounds.min.x - padding, bounds.min.y - padding, bounds.min.z - padding};
    bounds.max = (VEC3){bounds.max.x + padding, bounds.max.y + padding, bounds.max.z + padding};
    bounds.center = scale3(add3(bounds.min, bounds.max), 0.5f);
    bounds.extents = scale3(sub3(bounds.max, bounds.min), 0.5f);

    const uint64_t voxel_count64 = (uint64_t)resolution * resolution * resolution;

    if (voxel_count64 > SIZE_MAX / sizeof(float)) {
        free(build.triangles);
        free(build.nodes);

        return false;
    }

    float *distance = malloc((size_t)voxel_count64 * sizeof(*distance));
    uint32_t *surface_id = malloc((size_t)voxel_count64 * sizeof(*surface_id));

    if (!distance || !surface_id) {
        free(distance);
        free(surface_id);
        free(build.triangles);
        free(build.nodes);

        return false;
    }

    const VEC3 size = sub3(bounds.max, bounds.min);

    for (uint32_t z = 0; z < resolution; ++z) {
        for (uint32_t y = 0; y < resolution; ++y) {
            for (uint32_t x = 0; x < resolution; ++x) {
                const VEC3 uvw = {((float)x + 0.5f) / (float)resolution, ((float)y + 0.5f) / (float)resolution, ((float)z + 0.5f) / (float)resolution};
                const VEC3 p = {bounds.min.x + size.x * uvw.x, bounds.min.y + size.y * uvw.y, bounds.min.z + size.z * uvw.z};
                const uint64_t index = (uint64_t)x + (uint64_t)resolution * ((uint64_t)y + (uint64_t)resolution * z);
                uint32_t nearest_surface = UINT32_MAX;
                distance[index] = sqrtf(nearest_distance_sq(&build, 0, p, FLT_MAX, &nearest_surface));
                surface_id[index] = nearest_surface;
            }
        }
    }

    free(build.triangles);
    free(build.nodes);

    volume->distance = distance;
    volume->surface_id = surface_id;
    volume->resolution = resolution;
    volume->bounds = bounds;

    return true;
}

void sdf_free_volume(SDF_VOLUME *volume) {
    if (!volume) return;
    free(volume->distance);
    free(volume->surface_id);
    memset(volume, 0, sizeof(*volume));
}


static VEC3 sdf_transform_point(MAT4 matrix, VEC3 point) {
    return (VEC3){
        point.x * matrix.m[0] + point.y * matrix.m[4] + point.z * matrix.m[8] + matrix.m[12],
        point.x * matrix.m[1] + point.y * matrix.m[5] + point.z * matrix.m[9] + matrix.m[13],
        point.x * matrix.m[2] + point.y * matrix.m[6] + point.z * matrix.m[10] + matrix.m[14]
    };
}

static void global_bit_set(uint8_t *bits, uint32_t index) {
    bits[index >> 3u] |= (uint8_t)(1u << (index & 7u));
}

static bool global_bit_get(const uint8_t *bits, uint32_t index) {
    return (bits[index >> 3u] & (uint8_t)(1u << (index & 7u))) != 0u;
}

static uint32_t global_grid_coord(float value, float origin, float brick_world, uint32_t dimension) {
    int64_t coordinate = (int64_t)floorf((value - origin) / brick_world);
    if (coordinate < 0) coordinate = 0;
    if ((uint64_t)coordinate >= dimension) coordinate = (int64_t)dimension - 1;
    return (uint32_t)coordinate;
}

void sdf_free_global_clipmaps(GLOBAL_SDF_DATA *data) {
    if (!data) return;
    free(data->cpu_clipmaps);
    free(data->cpu_page_table);
    free(data->cpu_bricks);
    free(data->cpu_surface_ids);
    memset(data, 0, sizeof(*data));
}

bool sdf_build_global_clipmaps(
    const SCENE *scene,
    const RADIANCE_SCENE_DATA *radiance_scene,
    const GPU_OBJECT *objects,
    uint32_t object_count,
    GLOBAL_SDF_DATA *out
) {
    static const uint32_t grid_dimensions[3] = {32u, 16u, 8u};
    enum { LEVEL_COUNT = 3, BRICK_RESOLUTION = 4, BRICK_VOXELS = 64 };
    if (!scene || !radiance_scene || !objects || !out || !radiance_scene->cpu_triangles) return false;
    (void)scene;
    memset(out, 0, sizeof(*out));

    uint32_t static_count = 0u;
    for (uint32_t i = 0u; i < radiance_scene->triangle_count; ++i) {
        const GPU_SCENE_TRIANGLE *source = &radiance_scene->cpu_triangles[i];
        if (source->meta[0] >= object_count) return false;
        if (objects[source->meta[0]].state == (uint32_t)STATIC) ++static_count;
    }
    if (!static_count) return true;
    if ((uint64_t)static_count * 2u > SIZE_MAX / sizeof(SDF_BVH_NODE)) return false;

    SDF_BUILD build = {0};
    build.triangle_count = static_count;
    build.triangles = malloc((size_t)static_count * sizeof(*build.triangles));
    build.nodes = calloc((size_t)static_count * 2u, sizeof(*build.nodes));
    if (!build.triangles || !build.nodes) goto fail;

    VEC3 scene_min = {FLT_MAX, FLT_MAX, FLT_MAX};
    VEC3 scene_max = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
    uint32_t cursor = 0u;
    for (uint32_t i = 0u; i < radiance_scene->triangle_count; ++i) {
        const GPU_SCENE_TRIANGLE *source = &radiance_scene->cpu_triangles[i];
        const uint32_t object_index = source->meta[0];
        if (objects[object_index].state != (uint32_t)STATIC) continue;
        SDF_TRIANGLE *triangle = &build.triangles[cursor++];
        const MAT4 world = objects[object_index].world;
        triangle->a = sdf_transform_point(world, (VEC3){source->p0[0], source->p0[1], source->p0[2]});
        triangle->b = sdf_transform_point(world, (VEC3){source->p1[0], source->p1[1], source->p1[2]});
        triangle->c = sdf_transform_point(world, (VEC3){source->p2[0], source->p2[1], source->p2[2]});
        triangle->centroid = scale3(add3(add3(triangle->a, triangle->b), triangle->c), 1.0f / 3.0f);
        triangle->min = vmin3(triangle->a, vmin3(triangle->b, triangle->c));
        triangle->max = vmax3(triangle->a, vmax3(triangle->b, triangle->c));
        triangle->surface_id = i;
        scene_min = vmin3(scene_min, triangle->min);
        scene_max = vmax3(scene_max, triangle->max);
    }
    if (cursor != static_count) goto fail;
    build_node(&build, 0u, static_count);

    const VEC3 raw_size = sub3(scene_max, scene_min);
    const float raw_side = fmaxf(fmaxf(raw_size.x, raw_size.y), fmaxf(raw_size.z, 1.0e-3f));
    const float cube_side = raw_side * (32.0f / 30.0f);
    const float half_extent = cube_side * 0.5f;
    const VEC3 center = scale3(add3(scene_min, scene_max), 0.5f);
    const VEC3 cube_min = {center.x - half_extent, center.y - half_extent, center.z - half_extent};

    uint32_t page_bases[LEVEL_COUNT] = {0};
    uint32_t logical_counts[LEVEL_COUNT] = {0};
    uint32_t physical_counts[LEVEL_COUNT] = {0};
    uint8_t *requested[LEVEL_COUNT] = {0};
    uint64_t page_total = 0u;

    for (uint32_t level = 0u; level < LEVEL_COUNT; ++level) {
        const uint32_t dim = grid_dimensions[level];
        const uint64_t logical64 = (uint64_t)dim * dim * dim;
        if (!logical64 || logical64 > UINT32_MAX || page_total + logical64 > UINT32_MAX) goto fail_bits;
        page_bases[level] = (uint32_t)page_total;
        logical_counts[level] = (uint32_t)logical64;
        page_total += logical64;
        const size_t bytes = ((size_t)logical_counts[level] + 7u) / 8u;
        requested[level] = calloc(bytes, 1u);
        if (!requested[level]) goto fail_bits;

        if (level == 2u) {
            for (uint32_t i = 0u; i < logical_counts[level]; ++i) global_bit_set(requested[level], i);
        } else {
            const float voxel_size = cube_side / ((float)dim * (float)BRICK_RESOLUTION);
            const float brick_world = voxel_size * (float)BRICK_RESOLUTION;
            const float influence = 2.0f * brick_world;
            for (uint32_t i = 0u; i < build.triangle_count; ++i) {
                const SDF_TRIANGLE *triangle = &build.triangles[i];
                const uint32_t min_x = global_grid_coord(triangle->min.x - influence, cube_min.x, brick_world, dim);
                const uint32_t min_y = global_grid_coord(triangle->min.y - influence, cube_min.y, brick_world, dim);
                const uint32_t min_z = global_grid_coord(triangle->min.z - influence, cube_min.z, brick_world, dim);
                const uint32_t max_x = global_grid_coord(triangle->max.x + influence, cube_min.x, brick_world, dim);
                const uint32_t max_y = global_grid_coord(triangle->max.y + influence, cube_min.y, brick_world, dim);
                const uint32_t max_z = global_grid_coord(triangle->max.z + influence, cube_min.z, brick_world, dim);
                for (uint32_t z = min_z; z <= max_z; ++z)
                    for (uint32_t y = min_y; y <= max_y; ++y)
                        for (uint32_t x = min_x; x <= max_x; ++x)
                            global_bit_set(requested[level], x + dim * (y + dim * z));
            }
        }
        for (uint32_t i = 0u; i < logical_counts[level]; ++i)
            if (global_bit_get(requested[level], i)) ++physical_counts[level];
    }

    if (page_total != 37376u) goto fail_bits;
    const uint64_t physical_total = (uint64_t)physical_counts[0] + physical_counts[1] + physical_counts[2];
    const uint64_t voxel_total = physical_total * BRICK_VOXELS;
    if (!physical_total || physical_total > 37376u || voxel_total > 2392064u || voxel_total > UINT32_MAX ||
        voxel_total > SIZE_MAX / sizeof(float) || voxel_total > SIZE_MAX / sizeof(uint32_t))
        goto fail_bits;

    out->cpu_clipmaps = calloc(LEVEL_COUNT, sizeof(*out->cpu_clipmaps));
    out->cpu_page_table = malloc((size_t)page_total * sizeof(*out->cpu_page_table));
    out->cpu_bricks = malloc((size_t)voxel_total * sizeof(*out->cpu_bricks));
    out->cpu_surface_ids = malloc((size_t)voxel_total * sizeof(*out->cpu_surface_ids));
    if (!out->cpu_clipmaps || !out->cpu_page_table || !out->cpu_bricks || !out->cpu_surface_ids) goto fail_bits;
    for (uint32_t i = 0u; i < (uint32_t)page_total; ++i) out->cpu_page_table[i] = UINT32_MAX;

    uint32_t voxel_base = 0u;
    for (uint32_t level = 0u; level < LEVEL_COUNT; ++level) {
        const uint32_t dim = grid_dimensions[level];
        const float voxel_size = cube_side / ((float)dim * (float)BRICK_RESOLUTION);
        const float brick_world = voxel_size * (float)BRICK_RESOLUTION;
        GPU_GLOBAL_SDF_CLIPMAP *clip = &out->cpu_clipmaps[level];
        clip->center_extent[0] = center.x;
        clip->center_extent[1] = center.y;
        clip->center_extent[2] = center.z;
        clip->center_extent[3] = half_extent;
        clip->voxel_brick[0] = voxel_size;
        clip->voxel_brick[1] = brick_world;
        clip->grid[0] = dim;
        clip->grid[1] = dim;
        clip->grid[2] = dim;
        clip->grid[3] = page_bases[level];
        clip->data[0] = BRICK_RESOLUTION;
        clip->data[1] = BRICK_VOXELS;
        clip->data[2] = voxel_base;

        uint32_t physical = 0u;
        const float half_diagonal = 0.8660254037844386f * voxel_size;
        for (uint32_t logical = 0u; logical < logical_counts[level]; ++logical) {
            if (!global_bit_get(requested[level], logical)) continue;
            out->cpu_page_table[page_bases[level] + logical] = physical;
            const uint32_t bx = logical % dim;
            const uint32_t by = (logical / dim) % dim;
            const uint32_t bz = logical / (dim * dim);
            for (uint32_t vz = 0u; vz < BRICK_RESOLUTION; ++vz) {
                for (uint32_t vy = 0u; vy < BRICK_RESOLUTION; ++vy) {
                    for (uint32_t vx = 0u; vx < BRICK_RESOLUTION; ++vx) {
                        const uint32_t local = vx + BRICK_RESOLUTION * (vy + BRICK_RESOLUTION * vz);
                        const uint32_t index = voxel_base + physical * BRICK_VOXELS + local;
                        const VEC3 position = {
                            cube_min.x + (float)bx * brick_world + ((float)vx + 0.5f) * voxel_size,
                            cube_min.y + (float)by * brick_world + ((float)vy + 0.5f) * voxel_size,
                            cube_min.z + (float)bz * brick_world + ((float)vz + 0.5f) * voxel_size
                        };
                        uint32_t surface_id = UINT32_MAX;
                        const float distance_sq = nearest_distance_sq(&build, 0u, position, FLT_MAX, &surface_id);
                        if (surface_id == UINT32_MAX || surface_id >= radiance_scene->triangle_count || !isfinite(distance_sq)) goto fail_bits;
                        const float distance = sqrtf(fmaxf(distance_sq, 0.0f));
                        out->cpu_bricks[index] = fmaxf(0.0f, distance - half_diagonal);
                        out->cpu_surface_ids[index] = surface_id;
                    }
                }
            }
            ++physical;
        }
        if (physical != physical_counts[level]) goto fail_bits;
        voxel_base += physical * BRICK_VOXELS;
    }

    if (voxel_base != (uint32_t)voxel_total) goto fail_bits;
    out->clip_count = LEVEL_COUNT;
    out->page_table_count = (uint32_t)page_total;
    out->physical_brick_count = (uint32_t)physical_total;
    out->voxel_count = (uint32_t)voxel_total;
    out->coarsest_voxel_size = out->cpu_clipmaps[2].voxel_brick[0];
    out->valid = true;

    for (uint32_t level = 0u; level < LEVEL_COUNT; ++level) free(requested[level]);
    free(build.triangles);
    free(build.nodes);
    return true;

fail_bits:
    for (uint32_t level = 0u; level < LEVEL_COUNT; ++level) free(requested[level]);
fail:
    free(build.triangles);
    free(build.nodes);
    sdf_free_global_clipmaps(out);
    return false;
}
