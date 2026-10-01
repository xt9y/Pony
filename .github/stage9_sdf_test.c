#include "game.h"

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static MAT4 identity_matrix(void) {
    MAT4 m = {{0}};
    m.m[0] = m.m[5] = m.m[10] = m.m[15] = 1.0f;
    return m;
}

static VEC3 vsub(VEC3 a, VEC3 b) { return (VEC3){a.x - b.x, a.y - b.y, a.z - b.z}; }
static VEC3 vadd(VEC3 a, VEC3 b) { return (VEC3){a.x + b.x, a.y + b.y, a.z + b.z}; }
static VEC3 vscale(VEC3 a, float s) { return (VEC3){a.x * s, a.y * s, a.z * s}; }
static float vdot(VEC3 a, VEC3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static float vlen2(VEC3 a) { return vdot(a, a); }

static float point_segment_distance_sq(VEC3 p, VEC3 a, VEC3 b) {
    VEC3 ab = vsub(b, a);
    float denom = vlen2(ab);
    if (denom <= 1.0e-20f) return vlen2(vsub(p, a));
    float t = vdot(vsub(p, a), ab) / denom;
    t = fmaxf(0.0f, fminf(1.0f, t));
    return vlen2(vsub(p, vadd(a, vscale(ab, t))));
}

static float point_triangle_distance_sq(VEC3 p, VEC3 a, VEC3 b, VEC3 c) {
    VEC3 ab = vsub(b, a), ac = vsub(c, a), ap = vsub(p, a);
    VEC3 n = {ab.y * ac.z - ab.z * ac.y, ab.z * ac.x - ab.x * ac.z, ab.x * ac.y - ab.y * ac.x};
    if (vlen2(n) <= 1.0e-20f) {
        return fminf(point_segment_distance_sq(p, a, b),
                     fminf(point_segment_distance_sq(p, b, c), point_segment_distance_sq(p, c, a)));
    }
    float d1 = vdot(ab, ap), d2 = vdot(ac, ap);
    if (d1 <= 0.0f && d2 <= 0.0f) return vlen2(ap);
    VEC3 bp = vsub(p, b);
    float d3 = vdot(ab, bp), d4 = vdot(ac, bp);
    if (d3 >= 0.0f && d4 <= d3) return vlen2(bp);
    float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
        float v = d1 / (d1 - d3);
        return vlen2(vsub(p, vadd(a, vscale(ab, v))));
    }
    VEC3 cp = vsub(p, c);
    float d5 = vdot(ab, cp), d6 = vdot(ac, cp);
    if (d6 >= 0.0f && d5 <= d6) return vlen2(cp);
    float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
        float w = d2 / (d2 - d6);
        return vlen2(vsub(p, vadd(a, vscale(ac, w))));
    }
    float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
        VEC3 bc = vsub(c, b);
        float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return vlen2(vsub(p, vadd(b, vscale(bc, w))));
    }
    float denom = 1.0f / (va + vb + vc);
    float v = vb * denom, w = vc * denom;
    VEC3 q = vadd(a, vadd(vscale(ab, v), vscale(ac, w)));
    return vlen2(vsub(p, q));
}

static void set_triangle(GPU_SCENE_TRIANGLE *t, uint32_t object_index, VEC3 a, VEC3 b, VEC3 c) {
    memset(t, 0, sizeof(*t));
    t->p0[0] = a.x; t->p0[1] = a.y; t->p0[2] = a.z; t->p0[3] = 1.0f;
    t->p1[0] = b.x; t->p1[1] = b.y; t->p1[2] = b.z; t->p1[3] = 1.0f;
    t->p2[0] = c.x; t->p2[1] = c.y; t->p2[2] = c.z; t->p2[3] = 1.0f;
    t->meta[0] = object_index;
    t->meta[2] = 0u;
}

static void verify_conservative(const GLOBAL_SDF_DATA *g, VEC3 a, VEC3 b, VEC3 c) {
    assert(g->clip_count == 3u);
    for (uint32_t level = 0; level < g->clip_count; ++level) {
        const GPU_GLOBAL_SDF_CLIPMAP *clip = &g->cpu_clipmaps[level];
        const uint32_t dim_x = clip->grid[0], dim_y = clip->grid[1], dim_z = clip->grid[2];
        const uint32_t page_base = clip->grid[3];
        const uint32_t brick_res = clip->data[0];
        const uint32_t brick_stride = clip->data[1];
        const uint32_t voxel_base = clip->data[2];
        assert(brick_res == 4u && brick_stride == 64u);
        VEC3 cube_min = {
            clip->center_extent[0] - clip->center_extent[3],
            clip->center_extent[1] - clip->center_extent[3],
            clip->center_extent[2] - clip->center_extent[3]
        };
        for (uint32_t z = 0; z < dim_z; ++z) {
            for (uint32_t y = 0; y < dim_y; ++y) {
                for (uint32_t x = 0; x < dim_x; ++x) {
                    uint32_t logical = x + dim_x * (y + dim_y * z);
                    uint32_t physical = g->cpu_page_table[page_base + logical];
                    if (physical == UINT32_MAX) continue;
                    for (uint32_t vz = 0; vz < 4u; ++vz) {
                        for (uint32_t vy = 0; vy < 4u; ++vy) {
                            for (uint32_t vx = 0; vx < 4u; ++vx) {
                                uint32_t local = vx + 4u * (vy + 4u * vz);
                                uint32_t index = voxel_base + physical * 64u + local;
                                assert(index < g->voxel_count);
                                VEC3 p = {
                                    cube_min.x + (float)x * clip->voxel_brick[1] + ((float)vx + 0.5f) * clip->voxel_brick[0],
                                    cube_min.y + (float)y * clip->voxel_brick[1] + ((float)vy + 0.5f) * clip->voxel_brick[0],
                                    cube_min.z + (float)z * clip->voxel_brick[1] + ((float)vz + 0.5f) * clip->voxel_brick[0]
                                };
                                float exact = sqrtf(point_triangle_distance_sq(p, a, b, c));
                                assert(g->cpu_bricks[index] <= exact + 1.0e-5f);
                                assert(g->cpu_surface_ids[index] == 0u);
                            }
                        }
                    }
                }
            }
        }
    }
}

int main(void) {
    OBJECT scene_objects[2] = {0};
    scene_objects[0].type = MODEL; scene_objects[0].state = STATIC;
    scene_objects[1].type = MODEL; scene_objects[1].state = DYNAMIC;
    SCENE scene = {.objects = scene_objects, .object_count = 2u, .object_capacity = 2u};

    GPU_OBJECT objects[2] = {0};
    objects[0].world = identity_matrix(); objects[0].state = STATIC;
    objects[1].world = identity_matrix(); objects[1].state = DYNAMIC;

    GPU_SCENE_TRIANGLE triangles[2];
    VEC3 a = {-1.0f, -1.0f, 0.0f}, b = {1.0f, -1.0f, 0.0f}, c = {0.0f, 1.0f, 0.0f};
    set_triangle(&triangles[0], 0u, a, b, c);
    set_triangle(&triangles[1], 1u, (VEC3){-1.0f,-1.0f,0.05f}, (VEC3){1.0f,-1.0f,0.05f}, (VEC3){0.0f,1.0f,0.05f});
    RADIANCE_SCENE_DATA rs = {.cpu_triangles = triangles, .triangle_count = 2u};

    GLOBAL_SDF_DATA global = {0};
    assert(sdf_build_global_clipmaps(&scene, &rs, objects, 2u, &global));
    assert(global.valid);
    assert(global.clip_count == 3u);
    assert(global.page_table_count == 37376u);
    assert(global.physical_brick_count <= 37376u);
    assert(global.voxel_count == global.physical_brick_count * 64u);
    for (uint32_t i = 0; i < global.voxel_count; ++i) {
        assert(global.cpu_surface_ids[i] == UINT32_MAX || global.cpu_surface_ids[i] < rs.triangle_count);
        assert(global.cpu_surface_ids[i] != 1u);
    }
    const GPU_GLOBAL_SDF_CLIPMAP *coarse = &global.cpu_clipmaps[2];
    for (uint32_t i = 0; i < 8u * 8u * 8u; ++i)
        assert(global.cpu_page_table[coarse->grid[3] + i] != UINT32_MAX);
    verify_conservative(&global, a, b, c);
    sdf_free_global_clipmaps(&global);
    assert(!global.valid && global.clip_count == 0u && global.cpu_clipmaps == NULL);

    objects[0].state = DYNAMIC;
    GLOBAL_SDF_DATA empty = {0};
    assert(sdf_build_global_clipmaps(&scene, &rs, objects, 2u, &empty));
    assert(!empty.valid && empty.clip_count == 0u && empty.cpu_clipmaps == NULL);
    sdf_free_global_clipmaps(&empty);

    objects[0].state = STATIC;
    GPU_SCENE_TRIANGLE tiny_triangle;
    set_triangle(&tiny_triangle, 0u, (VEC3){0,0,0}, (VEC3){1.0e-8f,0,0}, (VEC3){0,1.0e-8f,0});
    RADIANCE_SCENE_DATA tiny_rs = {.cpu_triangles = &tiny_triangle, .triangle_count = 1u};
    GLOBAL_SDF_DATA tiny = {0};
    assert(sdf_build_global_clipmaps(&scene, &tiny_rs, objects, 2u, &tiny));
    assert(tiny.valid && tiny.clip_count == 3u && tiny.voxel_count <= 2392064u);
    sdf_free_global_clipmaps(&tiny);

    puts("stage9_sdf_test: PASS");
    return 0;
}
