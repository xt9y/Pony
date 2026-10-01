#!/usr/bin/env python3
from pathlib import Path
import re
import sys


def replace_once(text, old, new, label):
    count = text.count(old)
    if count != 1:
        raise SystemExit(f'{label}: expected 1 literal match, got {count}')
    return text.replace(old, new, 1)


def function_span(text, signature):
    start = text.find(signature)
    if start < 0:
        raise SystemExit(f'function not found: {signature}')
    brace = text.find('{', start)
    if brace < 0:
        raise SystemExit(f'function brace not found: {signature}')
    depth = 0
    for i in range(brace, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                return start, i + 1
    raise SystemExit(f'function end not found: {signature}')


def replace_function(text, signature, replacement):
    start, end = function_span(text, signature)
    return text[:start] + replacement.rstrip() + '\n' + text[end:]


def insert_after_function(text, signature, addition):
    _start, end = function_span(text, signature)
    return text[:end] + '\n\n' + addition.rstrip() + '\n' + text[end:]


def write_test_interface():
    Path('tests').mkdir(exist_ok=True)
    Path('tests/stage10_world_cache.c').write_text(r'''#include "game.h"

_Static_assert(WORLD_PROBE_CAPACITY == 8192u, "world probe capacity");
_Static_assert(WORLD_PROBE_HASH_CAPACITY == 16384u, "world hash capacity");
_Static_assert(WORLD_PROBE_DIRECTION_SIZE == 4u, "world direction size");
_Static_assert(WORLD_PROBE_DIRECTION_COUNT == 16u, "world direction count");
_Static_assert(WORLD_PROBE_BANK_COUNT == 2u, "world bank count");
_Static_assert(WORLD_PROBE_UPDATES_PER_FRAME == 64u, "world updates/frame");
_Static_assert(WORLD_PROBE_HASH_PROBE_LIMIT == 8u, "world hash probe limit");
_Static_assert(sizeof(WORLD_PROBE_STATE) == 64u, "WORLD_PROBE_STATE ABI");
_Static_assert(sizeof(RADIANCE_CONSTANTS) == 256u, "RADIANCE_CONSTANTS ABI");

typedef bool (*WORLD_BUILDER)(
    const GLOBAL_SDF_DATA *, const RADIANCE_SCENE_DATA *, const GPU_OBJECT *, uint32_t,
    WORLD_PROBE_STATE *, uint32_t, uint32_t *, uint32_t *, uint32_t,
    float, float, float, float
);

static WORLD_BUILDER world_builder = sdf_build_world_probes;

int main(void) {
    return world_builder == 0;
}
''')


def task1():
    p = Path('game.h')
    s = p.read_text()
    define_marker = '#define RADIANCE_MAX_GLOBAL_SDF_CLIPMAPS 8u\n#define RADIANCE_INVALID_INDEX UINT32_MAX\n'
    defines = '''#define RADIANCE_MAX_GLOBAL_SDF_CLIPMAPS 8u
#define WORLD_PROBE_CAPACITY 8192u
#define WORLD_PROBE_HASH_CAPACITY 16384u
#define WORLD_PROBE_DIRECTION_SIZE 4u
#define WORLD_PROBE_DIRECTION_COUNT 16u
#define WORLD_PROBE_BANK_COUNT 2u
#define WORLD_PROBE_UPDATES_PER_FRAME 64u
#define WORLD_PROBE_HASH_PROBE_LIMIT 8u
#define WORLD_PROBE_SPACING 0.75f
#define WORLD_PROBE_RADIUS 1.125f
#define WORLD_PROBE_CLEARANCE 0.075f
#define WORLD_PROBE_MIN_CLEARANCE 0.0375f
#define WORLD_PROBE_BLEND 0.20f
#define RADIANCE_INVALID_INDEX UINT32_MAX
'''
    s = replace_once(s, define_marker, defines, 'stage10 constants')

    state_marker = '''typedef struct WORLD_PROBE_STATE {
    float position_radius[4];
    uint32_t identity[4];
    float statistics[4];
    uint32_t state[4];
} WORLD_PROBE_STATE;
'''
    builder = state_marker + '''
bool sdf_build_world_probes(
    const GLOBAL_SDF_DATA *global_sdf,
    const RADIANCE_SCENE_DATA *radiance_scene,
    const GPU_OBJECT *objects,
    uint32_t object_count,
    WORLD_PROBE_STATE *out_probes,
    uint32_t probe_capacity,
    uint32_t *out_probe_count,
    uint32_t *out_keys,
    uint32_t key_capacity,
    float spacing,
    float radius,
    float clearance,
    float min_clearance
);
'''
    s = replace_once(s, state_marker, builder, 'stage10 builder declaration')

    world_old = '''typedef struct RADIANCE_WORLD_RESOURCES {
    NriBuffer *probes;
    NriBuffer *radiance;
    NriBuffer *keys;
    NriBuffer *invalidation_queue;

    NriDescriptor *probes_uav;
    NriDescriptor *radiance_uav;
    NriDescriptor *keys_uav;
    NriDescriptor *invalidation_queue_uav;

    NriAccessStage state;
} RADIANCE_WORLD_RESOURCES;'''
    world_new = '''typedef struct RADIANCE_WORLD_RESOURCES {
    WORLD_PROBE_STATE *cpu_probes;
    uint32_t *cpu_keys;
    uint32_t *cpu_update_list;

    NriBuffer *probes;
    NriBuffer *radiance;
    NriBuffer *keys;
    NriBuffer *invalidation_queue;

    NriDescriptor *probes_uav;
    NriDescriptor *radiance_uav;
    NriDescriptor *keys_uav;
    NriDescriptor *invalidation_queue_uav;

    NriAccessStage state;
    uint32_t probe_count;
    uint32_t probe_capacity;
    uint32_t hash_capacity;
    uint32_t direction_count;
    uint32_t bank_count;
    uint32_t update_count;
    uint32_t update_cursor;
} RADIANCE_WORLD_RESOURCES;'''
    s = replace_once(s, world_old, world_new, 'stage10 world owner')
    s = replace_once(s, '    NriAccessStage state;\n    uint32_t ray_capacity;\n    uint32_t probe_capacity;\n} RADIANCE_WAVEFRONT;',
                     '    NriAccessStage state;\n    NriAccessStage update_list_state;\n    uint32_t ray_capacity;\n    uint32_t probe_capacity;\n} RADIANCE_WAVEFRONT;', 'stage10 update list state')
    s = replace_once(s, '    NriPipeline *wavefront_history_pipeline;\n',
                     '    NriPipeline *wavefront_history_pipeline;\n    NriPipeline *world_radiance_pipeline;\n', 'stage10 world pipeline')
    s = replace_once(s, '    uint32_t width;\n    uint32_t height;\n    bool has_previous_frame;',
                     '    uint32_t width;\n    uint32_t height;\n    uint32_t radiance_revision;\n    bool has_previous_frame;', 'stage10 revision')
    p.write_text(s)


def write_test_behavior():
    Path('tests/stage10_world_cache.c').write_text(r'''#include "game.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

_Static_assert(WORLD_PROBE_CAPACITY == 8192u, "world probe capacity");
_Static_assert(WORLD_PROBE_HASH_CAPACITY == 16384u, "world hash capacity");
_Static_assert(WORLD_PROBE_DIRECTION_COUNT == 16u, "world direction count");
_Static_assert(WORLD_PROBE_BANK_COUNT == 2u, "world bank count");
_Static_assert(WORLD_PROBE_UPDATES_PER_FRAME == 64u, "world update budget");
_Static_assert(sizeof(WORLD_PROBE_STATE) == 64u, "world probe ABI");
_Static_assert(sizeof(RADIANCE_CONSTANTS) == 256u, "radiance constants ABI");
_Static_assert((uint64_t)WORLD_PROBE_CAPACITY * WORLD_PROBE_DIRECTION_COUNT * WORLD_PROBE_BANK_COUNT == 262144u, "radiance value count");
_Static_assert((uint64_t)WORLD_PROBE_CAPACITY * WORLD_PROBE_DIRECTION_COUNT * WORLD_PROBE_BANK_COUNT * sizeof(float[4]) == 4194304u, "radiance bytes");

static uint32_t hash32(uint32_t x) {
    x ^= x >> 16u; x *= 0x7feb352du; x ^= x >> 15u; x *= 0x846ca68bu; x ^= x >> 16u; return x;
}
static uint32_t hash_combine(uint32_t a, uint32_t b) {
    return hash32(a ^ (b + 0x9e3779b9u + (a << 6u) + (a >> 2u)));
}
static uint32_t cell_key(float x, float y, float z) {
    int32_t cx = (int32_t)floorf(x / WORLD_PROBE_SPACING);
    int32_t cy = (int32_t)floorf(y / WORLD_PROBE_SPACING);
    int32_t cz = (int32_t)floorf(z / WORLD_PROBE_SPACING);
    uint32_t h = hash_combine((uint32_t)cx, (uint32_t)cy);
    return hash_combine(h, (uint32_t)cz) | 1u;
}

static MAT4 identity(void) {
    MAT4 m = {{0}};
    m.m[0] = m.m[5] = m.m[10] = m.m[15] = 1.0f;
    return m;
}

static void make_fixture(GLOBAL_SDF_DATA *g, RADIANCE_SCENE_DATA *scene, GPU_OBJECT *object) {
    memset(g, 0, sizeof(*g));
    memset(scene, 0, sizeof(*scene));
    memset(object, 0, sizeof(*object));

    scene->triangle_count = 1u;
    scene->cpu_triangles = calloc(1u, sizeof(*scene->cpu_triangles));
    assert(scene->cpu_triangles);
    GPU_SCENE_TRIANGLE *t = scene->cpu_triangles;
    t->p0[0] = -2.0f; t->p0[1] = -2.0f; t->p0[2] = 0.0f;
    t->p1[0] =  2.0f; t->p1[1] = -2.0f; t->p1[2] = 0.0f;
    t->p2[0] =  0.0f; t->p2[1] =  2.0f; t->p2[2] = 0.0f;
    t->meta[0] = 0u;
    t->meta[2] = 0u;

    object->world = identity();
    object->state = (uint32_t)STATIC;
    object->revision = 7u;

    g->clip_count = 2u;
    g->valid = true;
    g->cpu_clipmaps = calloc(2u, sizeof(*g->cpu_clipmaps));
    g->cpu_page_table = calloc(2u, sizeof(*g->cpu_page_table));
    g->cpu_bricks = calloc(128u, sizeof(*g->cpu_bricks));
    g->cpu_surface_ids = calloc(128u, sizeof(*g->cpu_surface_ids));
    assert(g->cpu_clipmaps && g->cpu_page_table && g->cpu_bricks && g->cpu_surface_ids);
    g->page_table_count = 2u;
    g->voxel_count = 128u;
    for (uint32_t level = 0; level < 2u; ++level) {
        GPU_GLOBAL_SDF_CLIPMAP *c = &g->cpu_clipmaps[level];
        c->center_extent[3] = 1.0f;
        c->voxel_brick[0] = 0.5f;
        c->voxel_brick[1] = 2.0f;
        c->grid[0] = c->grid[1] = c->grid[2] = 1u;
        c->grid[3] = level;
        c->data[0] = 4u;
        c->data[1] = 64u;
        c->data[2] = level * 64u;
        g->cpu_page_table[level] = 0u;
        for (uint32_t i = 0; i < 64u; ++i) g->cpu_surface_ids[level * 64u + i] = 0u;
    }
}

static void free_fixture(GLOBAL_SDF_DATA *g, RADIANCE_SCENE_DATA *scene) {
    free(scene->cpu_triangles);
    free(g->cpu_clipmaps); free(g->cpu_page_table); free(g->cpu_bricks); free(g->cpu_surface_ids);
}

static int reachable(const WORLD_PROBE_STATE *p, uint32_t encoded_index, const uint32_t *keys) {
    uint32_t key = p->identity[0];
    uint32_t mask = WORLD_PROBE_HASH_CAPACITY - 1u;
    for (uint32_t i = 0u; i < WORLD_PROBE_HASH_PROBE_LIMIT; ++i) {
        uint32_t slot = (key + i) & mask;
        if (keys[slot] == encoded_index) return 1;
        if (keys[slot] == 0u) return 0;
    }
    return 0;
}

int main(void) {
    GLOBAL_SDF_DATA g;
    RADIANCE_SCENE_DATA scene;
    GPU_OBJECT object;
    make_fixture(&g, &scene, &object);

    WORLD_PROBE_STATE *a = calloc(WORLD_PROBE_CAPACITY, sizeof(*a));
    WORLD_PROBE_STATE *b = calloc(WORLD_PROBE_CAPACITY, sizeof(*b));
    uint32_t *ka = calloc(WORLD_PROBE_HASH_CAPACITY, sizeof(*ka));
    uint32_t *kb = calloc(WORLD_PROBE_HASH_CAPACITY, sizeof(*kb));
    assert(a && b && ka && kb);
    uint32_t na = 0u, nb = 0u;

    assert(sdf_build_world_probes(&g, &scene, &object, 1u, a, WORLD_PROBE_CAPACITY, &na, ka, WORLD_PROBE_HASH_CAPACITY,
                                  WORLD_PROBE_SPACING, WORLD_PROBE_RADIUS, WORLD_PROBE_CLEARANCE, WORLD_PROBE_MIN_CLEARANCE));
    assert(na > 0u && na <= WORLD_PROBE_CAPACITY);
    assert(sdf_build_world_probes(&g, &scene, &object, 1u, b, WORLD_PROBE_CAPACITY, &nb, kb, WORLD_PROBE_HASH_CAPACITY,
                                  WORLD_PROBE_SPACING, WORLD_PROBE_RADIUS, WORLD_PROBE_CLEARANCE, WORLD_PROBE_MIN_CLEARANCE));
    assert(na == nb);
    assert(memcmp(a, b, (size_t)na * sizeof(*a)) == 0);
    assert(memcmp(ka, kb, WORLD_PROBE_HASH_CAPACITY * sizeof(*ka)) == 0);

    int saw_negative = 0;
    for (uint32_t i = 0u; i < na; ++i) {
        assert(isfinite(a[i].position_radius[0]) && isfinite(a[i].position_radius[1]) && isfinite(a[i].position_radius[2]));
        assert(fabsf(a[i].position_radius[3] - WORLD_PROBE_RADIUS) < 1.0e-6f);
        assert(a[i].identity[2] == 0u && object.state == (uint32_t)STATIC);
        assert(a[i].identity[0] == cell_key(a[i].position_radius[0], a[i].position_radius[1], a[i].position_radius[2]));
        assert(fabsf(a[i].position_radius[2]) + 1.0e-6f >= WORLD_PROBE_MIN_CLEARANCE);
        assert(reachable(&a[i], i + 1u, ka));
        if (a[i].position_radius[0] < 0.0f || a[i].position_radius[1] < 0.0f || a[i].position_radius[2] < 0.0f) saw_negative = 1;
    }
    assert(saw_negative);
    for (uint32_t slot = 0u; slot < WORLD_PROBE_HASH_CAPACITY; ++slot) {
        if (!ka[slot]) continue;
        uint32_t index = ka[slot] - 1u;
        assert(index < na);
        assert(reachable(&a[index], ka[slot], ka));
    }

    object.state = (uint32_t)DYNAMIC;
    memset(ka, 0, WORLD_PROBE_HASH_CAPACITY * sizeof(*ka));
    uint32_t nd = 123u;
    assert(sdf_build_world_probes(&g, &scene, &object, 1u, a, WORLD_PROBE_CAPACITY, &nd, ka, WORLD_PROBE_HASH_CAPACITY,
                                  WORLD_PROBE_SPACING, WORLD_PROBE_RADIUS, WORLD_PROBE_CLEARANCE, WORLD_PROBE_MIN_CLEARANCE));
    assert(nd == 0u);

    free(a); free(b); free(ka); free(kb); free_fixture(&g, &scene);
    puts("stage10 world cache CPU PASS");
    return 0;
}
''')


WORLD_BUILDER_CODE = r'''
typedef struct WORLD_PROBE_CANDIDATE {
    int32_t cell[3];
    VEC3 position;
    float clearance;
    uint32_t surface_id;
    uint32_t object_index;
    uint32_t revision;
} WORLD_PROBE_CANDIDATE;

static VEC3 cross3(VEC3 a, VEC3 b) {
    return (VEC3){a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

static VEC3 closest_point_triangle(VEC3 p, const SDF_TRIANGLE *triangle) {
    const VEC3 ab = sub3(triangle->b, triangle->a);
    const VEC3 ac = sub3(triangle->c, triangle->a);
    const VEC3 ap = sub3(p, triangle->a);
    const float d1 = dot3(ab, ap), d2 = dot3(ac, ap);
    if (d1 <= 0.0f && d2 <= 0.0f) return triangle->a;
    const VEC3 bp = sub3(p, triangle->b);
    const float d3 = dot3(ab, bp), d4 = dot3(ac, bp);
    if (d3 >= 0.0f && d4 <= d3) return triangle->b;
    const float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) return add3(triangle->a, scale3(ab, d1 / (d1 - d3)));
    const VEC3 cp = sub3(p, triangle->c);
    const float d5 = dot3(ab, cp), d6 = dot3(ac, cp);
    if (d6 >= 0.0f && d5 <= d6) return triangle->c;
    const float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) return add3(triangle->a, scale3(ac, d2 / (d2 - d6)));
    const float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && d4 - d3 >= 0.0f && d5 - d6 >= 0.0f) {
        const VEC3 bc = sub3(triangle->c, triangle->b);
        return add3(triangle->b, scale3(bc, (d4 - d3) / ((d4 - d3) + (d5 - d6))));
    }
    const float denom = 1.0f / (va + vb + vc);
    return add3(triangle->a, add3(scale3(ab, vb * denom), scale3(ac, vc * denom)));
}

static uint32_t world_hash32(uint32_t x) {
    x ^= x >> 16u; x *= 0x7feb352du; x ^= x >> 15u; x *= 0x846ca68bu; x ^= x >> 16u; return x;
}

static uint32_t world_hash_combine(uint32_t a, uint32_t b) {
    return world_hash32(a ^ (b + 0x9e3779b9u + (a << 6u) + (a >> 2u)));
}

static uint32_t world_cell_key(const int32_t cell[3]) {
    uint32_t h = world_hash_combine((uint32_t)cell[0], (uint32_t)cell[1]);
    return world_hash_combine(h, (uint32_t)cell[2]) | 1u;
}

static int world_candidate_compare(const void *lhs, const void *rhs) {
    const WORLD_PROBE_CANDIDATE *a = lhs, *b = rhs;
    for (int axis = 0; axis < 3; ++axis) {
        if (a->cell[axis] < b->cell[axis]) return -1;
        if (a->cell[axis] > b->cell[axis]) return 1;
    }
    if (a->clearance > b->clearance) return -1;
    if (a->clearance < b->clearance) return 1;
    if (a->surface_id < b->surface_id) return -1;
    if (a->surface_id > b->surface_id) return 1;
    return 0;
}

bool sdf_build_world_probes(
    const GLOBAL_SDF_DATA *global_sdf,
    const RADIANCE_SCENE_DATA *radiance_scene,
    const GPU_OBJECT *objects,
    uint32_t object_count,
    WORLD_PROBE_STATE *out_probes,
    uint32_t probe_capacity,
    uint32_t *out_probe_count,
    uint32_t *out_keys,
    uint32_t key_capacity,
    float spacing,
    float radius,
    float clearance,
    float min_clearance
) {
    if (!global_sdf || !radiance_scene || !objects || !out_probes || !out_probe_count || !out_keys ||
        !probe_capacity || !key_capacity || (key_capacity & (key_capacity - 1u)) || spacing <= 0.0f ||
        radius <= 0.0f || clearance <= 0.0f || min_clearance < 0.0f)
        return false;
    *out_probe_count = 0u;
    memset(out_keys, 0, (size_t)key_capacity * sizeof(*out_keys));
    if (!global_sdf->valid || !global_sdf->clip_count || !global_sdf->cpu_clipmaps || !global_sdf->cpu_page_table ||
        !global_sdf->cpu_bricks || !global_sdf->cpu_surface_ids || !radiance_scene->cpu_triangles)
        return true;

    uint32_t static_count = 0u;
    for (uint32_t i = 0u; i < radiance_scene->triangle_count; ++i) {
        uint32_t object_index = radiance_scene->cpu_triangles[i].meta[0];
        if (object_index >= object_count) return false;
        if (objects[object_index].state == (uint32_t)STATIC) ++static_count;
    }
    if (!static_count) return true;

    SDF_BUILD build = {0};
    build.triangle_count = static_count;
    build.triangles = malloc((size_t)static_count * sizeof(*build.triangles));
    build.nodes = calloc((size_t)static_count * 2u, sizeof(*build.nodes));
    if (!build.triangles || !build.nodes) goto fail;
    uint32_t cursor = 0u;
    for (uint32_t i = 0u; i < radiance_scene->triangle_count; ++i) {
        const GPU_SCENE_TRIANGLE *source = &radiance_scene->cpu_triangles[i];
        uint32_t object_index = source->meta[0];
        if (objects[object_index].state != (uint32_t)STATIC) continue;
        SDF_TRIANGLE *triangle = &build.triangles[cursor++];
        MAT4 world = objects[object_index].world;
        triangle->a = sdf_transform_point(world, (VEC3){source->p0[0], source->p0[1], source->p0[2]});
        triangle->b = sdf_transform_point(world, (VEC3){source->p1[0], source->p1[1], source->p1[2]});
        triangle->c = sdf_transform_point(world, (VEC3){source->p2[0], source->p2[1], source->p2[2]});
        triangle->centroid = scale3(add3(add3(triangle->a, triangle->b), triangle->c), 1.0f / 3.0f);
        triangle->min = vmin3(triangle->a, vmin3(triangle->b, triangle->c));
        triangle->max = vmax3(triangle->a, vmax3(triangle->b, triangle->c));
        triangle->surface_id = i;
    }
    if (cursor != static_count) goto fail;
    build_node(&build, 0u, static_count);

    size_t candidate_capacity = 4096u, candidate_count = 0u;
    WORLD_PROBE_CANDIDATE *candidates = malloc(candidate_capacity * sizeof(*candidates));
    if (!candidates) goto fail;
    uint32_t level_count = global_sdf->clip_count < 2u ? global_sdf->clip_count : 2u;
    for (uint32_t level = 0u; level < level_count; ++level) {
        const GPU_GLOBAL_SDF_CLIPMAP *clip = &global_sdf->cpu_clipmaps[level];
        uint32_t dim_x = clip->grid[0], dim_y = clip->grid[1], dim_z = clip->grid[2];
        uint32_t brick_res = clip->data[0], brick_stride = clip->data[1];
        if (!dim_x || !dim_y || !dim_z || !brick_res || !brick_stride) continue;
        float voxel = clip->voxel_brick[0], brick_world = clip->voxel_brick[1];
        float extent = clip->center_extent[3];
        VEC3 origin = {clip->center_extent[0] - extent, clip->center_extent[1] - extent, clip->center_extent[2] - extent};
        uint64_t logical_count = (uint64_t)dim_x * dim_y * dim_z;
        for (uint64_t logical = 0u; logical < logical_count; ++logical) {
            uint64_t page = (uint64_t)clip->grid[3] + logical;
            if (page >= global_sdf->page_table_count) goto fail_candidates;
            uint32_t physical = global_sdf->cpu_page_table[page];
            if (physical == UINT32_MAX) continue;
            uint32_t bx = (uint32_t)(logical % dim_x);
            uint32_t by = (uint32_t)((logical / dim_x) % dim_y);
            uint32_t bz = (uint32_t)(logical / ((uint64_t)dim_x * dim_y));
            for (uint32_t vz = 0u; vz < brick_res; ++vz)
                for (uint32_t vy = 0u; vy < brick_res; ++vy)
                    for (uint32_t vx = 0u; vx < brick_res; ++vx) {
                        uint32_t local = vx + brick_res * (vy + brick_res * vz);
                        uint64_t index64 = (uint64_t)clip->data[2] + (uint64_t)physical * brick_stride + local;
                        if (index64 >= global_sdf->voxel_count) goto fail_candidates;
                        uint32_t surface_id = global_sdf->cpu_surface_ids[index64];
                        if (surface_id == UINT32_MAX || surface_id >= radiance_scene->triangle_count) continue;
                        if (global_sdf->cpu_bricks[index64] > voxel * 1.5f) continue;
                        const GPU_SCENE_TRIANGLE *source = &radiance_scene->cpu_triangles[surface_id];
                        uint32_t object_index = source->meta[0];
                        if (object_index >= object_count || objects[object_index].state != (uint32_t)STATIC) continue;
                        SDF_TRIANGLE triangle = {0};
                        MAT4 world = objects[object_index].world;
                        triangle.a = sdf_transform_point(world, (VEC3){source->p0[0], source->p0[1], source->p0[2]});
                        triangle.b = sdf_transform_point(world, (VEC3){source->p1[0], source->p1[1], source->p1[2]});
                        triangle.c = sdf_transform_point(world, (VEC3){source->p2[0], source->p2[1], source->p2[2]});
                        VEC3 sample = {origin.x + (float)bx * brick_world + ((float)vx + 0.5f) * voxel,
                                       origin.y + (float)by * brick_world + ((float)vy + 0.5f) * voxel,
                                       origin.z + (float)bz * brick_world + ((float)vz + 0.5f) * voxel};
                        VEC3 closest = closest_point_triangle(sample, &triangle);
                        VEC3 n = cross3(sub3(triangle.b, triangle.a), sub3(triangle.c, triangle.a));
                        float n2 = length_sq3(n);
                        if (n2 <= 1.0e-12f) continue;
                        n = scale3(n, 1.0f / sqrtf(n2));
                        WORLD_PROBE_CANDIDATE candidate = {0};
                        bool accepted = false;
                        for (uint32_t side = 0u; side < 2u && !accepted; ++side) {
                            float sign = side ? -1.0f : 1.0f;
                            VEC3 pos = add3(closest, scale3(n, clearance * sign));
                            uint32_t nearest_surface = UINT32_MAX;
                            float nearest_sq = nearest_distance_sq(&build, 0u, pos, FLT_MAX, &nearest_surface);
                            float nearest = sqrtf(fmaxf(nearest_sq, 0.0f));
                            if (!isfinite(pos.x) || !isfinite(pos.y) || !isfinite(pos.z) || !isfinite(nearest) || nearest + 1.0e-6f < min_clearance) continue;
                            candidate.position = pos;
                            candidate.clearance = nearest;
                            candidate.surface_id = surface_id;
                            candidate.object_index = object_index;
                            candidate.revision = objects[object_index].revision;
                            candidate.cell[0] = (int32_t)floorf(pos.x / spacing);
                            candidate.cell[1] = (int32_t)floorf(pos.y / spacing);
                            candidate.cell[2] = (int32_t)floorf(pos.z / spacing);
                            accepted = true;
                        }
                        if (!accepted) continue;
                        if (candidate_count == candidate_capacity) {
                            if (candidate_capacity > SIZE_MAX / 2u / sizeof(*candidates)) goto fail_candidates;
                            candidate_capacity *= 2u;
                            WORLD_PROBE_CANDIDATE *grown = realloc(candidates, candidate_capacity * sizeof(*candidates));
                            if (!grown) goto fail_candidates;
                            candidates = grown;
                        }
                        candidates[candidate_count++] = candidate;
                    }
        }
    }

    qsort(candidates, candidate_count, sizeof(*candidates), world_candidate_compare);
    uint32_t mask = key_capacity - 1u;
    for (size_t i = 0u; i < candidate_count && *out_probe_count < probe_capacity;) {
        size_t j = i + 1u;
        while (j < candidate_count && candidates[j].cell[0] == candidates[i].cell[0] && candidates[j].cell[1] == candidates[i].cell[1] && candidates[j].cell[2] == candidates[i].cell[2]) ++j;
        const WORLD_PROBE_CANDIDATE *candidate = &candidates[i];
        uint32_t key = world_cell_key(candidate->cell);
        uint32_t probe_index = *out_probe_count;
        bool inserted = false;
        for (uint32_t attempt = 0u; attempt < WORLD_PROBE_HASH_PROBE_LIMIT; ++attempt) {
            uint32_t slot = (key + attempt) & mask;
            if (out_keys[slot] != 0u) continue;
            out_keys[slot] = probe_index + 1u;
            inserted = true;
            break;
        }
        if (inserted) {
            WORLD_PROBE_STATE *state = &out_probes[probe_index];
            memset(state, 0, sizeof(*state));
            state->position_radius[0] = candidate->position.x;
            state->position_radius[1] = candidate->position.y;
            state->position_radius[2] = candidate->position.z;
            state->position_radius[3] = radius;
            state->identity[0] = key;
            state->identity[1] = candidate->surface_id;
            state->identity[2] = candidate->object_index;
            state->identity[3] = candidate->revision;
            ++*out_probe_count;
        }
        i = j;
    }

    free(candidates); free(build.triangles); free(build.nodes);
    return true;

fail_candidates:
    free(candidates);
fail:
    free(build.triangles); free(build.nodes);
    *out_probe_count = 0u;
    memset(out_keys, 0, (size_t)key_capacity * sizeof(*out_keys));
    return false;
}
'''


def task2():
    p = Path('sdf.c')
    s = p.read_text()
    if 'bool sdf_build_world_probes(' in s:
        raise SystemExit('task2 already applied')
    s = s.rstrip() + '\n\n' + WORLD_BUILDER_CODE.strip() + '\n'
    p.write_text(s)


WORLD_RESOURCE_FUNCTIONS = r'''static void destroy_world_radiance_resources(RENDERER *renderer) {
    RADIANCE_WORLD_RESOURCES *w = &renderer->world_radiance;
    free(w->cpu_probes);
    free(w->cpu_keys);
    free(w->cpu_update_list);
    destroy_storage_uav(renderer, &w->probes, &w->probes_uav);
    destroy_storage_uav(renderer, &w->radiance, &w->radiance_uav);
    destroy_storage_uav(renderer, &w->keys, &w->keys_uav);
    destroy_storage_uav(renderer, &w->invalidation_queue, &w->invalidation_queue_uav);
    memset(w, 0, sizeof(*w));
}

static bool create_world_radiance_resources(RENDERER *renderer) {
    destroy_world_radiance_resources(renderer);
    RADIANCE_WORLD_RESOURCES *w = &renderer->world_radiance;
    w->probe_capacity = WORLD_PROBE_CAPACITY;
    w->hash_capacity = WORLD_PROBE_HASH_CAPACITY;
    w->direction_count = WORLD_PROBE_DIRECTION_COUNT;
    w->bank_count = WORLD_PROBE_BANK_COUNT;
    w->cpu_probes = calloc(WORLD_PROBE_CAPACITY, sizeof(*w->cpu_probes));
    w->cpu_keys = calloc(WORLD_PROBE_HASH_CAPACITY, sizeof(*w->cpu_keys));
    w->cpu_update_list = calloc(WORLD_PROBE_UPDATES_PER_FRAME, sizeof(*w->cpu_update_list));
    if (!w->cpu_probes || !w->cpu_keys || !w->cpu_update_list ||
        !create_storage_uav(renderer, (uint64_t)WORLD_PROBE_CAPACITY * sizeof(WORLD_PROBE_STATE), sizeof(WORLD_PROBE_STATE), 0, &w->probes, &w->probes_uav) ||
        !create_storage_uav(renderer, (uint64_t)WORLD_PROBE_CAPACITY * WORLD_PROBE_DIRECTION_COUNT * WORLD_PROBE_BANK_COUNT * 4u * sizeof(float), 4u * sizeof(float), 0, &w->radiance, &w->radiance_uav) ||
        !create_storage_uav(renderer, (uint64_t)WORLD_PROBE_HASH_CAPACITY * sizeof(uint32_t), sizeof(uint32_t), 0, &w->keys, &w->keys_uav) ||
        !create_storage_uav(renderer, (uint64_t)WORLD_PROBE_CAPACITY * sizeof(uint32_t), sizeof(uint32_t), 0, &w->invalidation_queue, &w->invalidation_queue_uav)) {
        destroy_world_radiance_resources(renderer);
        return false;
    }
    return true;
}'''

WORLD_RENDER_HELPERS = r'''static uint32_t radiance_hash_bytes(uint32_t hash, const void *data, size_t size) {
    const uint8_t *bytes = data;
    for (size_t i = 0u; i < size; ++i) {
        hash ^= bytes[i];
        hash *= 16777619u;
    }
    return hash;
}

static uint32_t compute_radiance_revision(const RENDERER *renderer) {
    uint32_t hash = 2166136261u;
    hash = radiance_hash_bytes(hash, renderer->cpu_lights, (size_t)renderer->light_count * sizeof(*renderer->cpu_lights));
    for (uint32_t i = 0u; i < renderer->radiance_scene.emissive_triangle_count; ++i) {
        const GPU_EMISSIVE_TRIANGLE *e = &renderer->radiance_scene.cpu_emissive_triangles[i];
        hash = radiance_hash_bytes(hash, e->meta, sizeof(e->meta));
        hash = radiance_hash_bytes(hash, e->radiance_area, sizeof(e->radiance_area));
    }
    return hash ? hash : 1u;
}

static void refresh_radiance_revision(RENDERER *renderer) {
    const uint32_t revision = compute_radiance_revision(renderer);
    if (revision == renderer->radiance_revision) return;
    renderer->radiance_revision = revision;
    renderer->radiance_constants.feature_flags[1] = revision;
    renderer->world_radiance.update_cursor = 0u;
}

static bool build_world_radiance_scene(RENDERER *renderer) {
    RADIANCE_WORLD_RESOURCES *w = &renderer->world_radiance;
    if (!w->cpu_probes || !w->cpu_keys || !w->probes || !w->radiance || !w->keys) return false;
    memset(w->cpu_probes, 0, (size_t)WORLD_PROBE_CAPACITY * sizeof(*w->cpu_probes));
    memset(w->cpu_keys, 0, (size_t)WORLD_PROBE_HASH_CAPACITY * sizeof(*w->cpu_keys));
    w->probe_count = 0u;
    if (!sdf_build_world_probes(&renderer->global_sdf, &renderer->radiance_scene, renderer->cpu_objects, renderer->gpu_object_count,
                                w->cpu_probes, WORLD_PROBE_CAPACITY, &w->probe_count, w->cpu_keys, WORLD_PROBE_HASH_CAPACITY,
                                WORLD_PROBE_SPACING, WORLD_PROBE_RADIUS, WORLD_PROBE_CLEARANCE, WORLD_PROBE_MIN_CLEARANCE))
        return false;
    w->update_cursor = 0u;
    w->update_count = 0u;
    renderer->radiance_revision = compute_radiance_revision(renderer);
    const NriAccessStage storage = {.access = NriAccessBits_SHADER_RESOURCE_STORAGE, .stages = NriStageBits_COMPUTE_SHADER};
    const uint64_t radiance_values = (uint64_t)WORLD_PROBE_CAPACITY * WORLD_PROBE_DIRECTION_COUNT * WORLD_PROBE_BANK_COUNT;
    float *zero_radiance = calloc((size_t)radiance_values, sizeof(float[4]));
    uint32_t *zero_invalidations = calloc(WORLD_PROBE_CAPACITY, sizeof(uint32_t));
    if (!zero_radiance || !zero_invalidations) { free(zero_radiance); free(zero_invalidations); return false; }
    bool ok = gpu_upload_buffer(renderer->gpu, w->probes, w->cpu_probes, storage) &&
              gpu_upload_buffer(renderer->gpu, w->keys, w->cpu_keys, storage) &&
              gpu_upload_buffer(renderer->gpu, w->radiance, zero_radiance, storage) &&
              gpu_upload_buffer(renderer->gpu, w->invalidation_queue, zero_invalidations, storage);
    free(zero_radiance); free(zero_invalidations);
    if (!ok) return false;
    w->state = storage;
    return true;
}'''


def task3():
    p = Path('render.c')
    s = p.read_text()
    start, end = function_span(s, 'static void destroy_world_radiance_resources(RENDERER *renderer)')
    _s2, end2 = function_span(s, 'static bool create_world_radiance_resources(RENDERER *renderer)')
    if end2 < end: raise SystemExit('world resource function order unexpected')
    s = s[:start] + WORLD_RESOURCE_FUNCTIONS + '\n' + s[end2:]
    s = insert_after_function(s, 'static bool create_world_radiance_resources(RENDERER *renderer)', WORLD_RENDER_HELPERS)

    old_alloc = '!create_storage_uav(renderer, (uint64_t)probe_capacity * sizeof(uint32_t), sizeof(uint32_t), 0, &w->update_list, &w->update_list_uav) ||'
    new_alloc = '!create_storage_uav(renderer, (uint64_t)(probe_capacity > WORLD_PROBE_UPDATES_PER_FRAME ? probe_capacity : WORLD_PROBE_UPDATES_PER_FRAME) * sizeof(uint32_t), sizeof(uint32_t), 0, &w->update_list, &w->update_list_uav) ||'
    s = replace_once(s, old_alloc, new_alloc, 'world update list floor')

    s = replace_once(s, '        {.buffer = w->update_list, .before = w->state, .after = state},',
                     '        {.buffer = w->update_list, .before = w->update_list_state, .after = state},', 'update list barrier state')
    s = replace_once(s, '    w->state = state;\n}\n\nstatic void build_wavefront_screen_probes',
                     '    w->state = state;\n    w->update_list_state = state;\n}\n\nstatic void build_wavefront_screen_probes', 'update list state track')

    old_counts = '''    constants.cache_counts[0] = renderer->radiance_surface_cache.capacity;
    constants.cache_counts[1] = 0u;
    constants.cache_counts[2] = 0u;
    constants.cache_counts[3] = 0u;'''
    new_counts = '''    constants.cache_counts[0] = renderer->radiance_surface_cache.capacity;
    constants.cache_counts[1] = renderer->world_radiance.probe_count;
    constants.cache_counts[2] = WORLD_PROBE_HASH_CAPACITY;
    constants.cache_counts[3] = WORLD_PROBE_CAPACITY;'''
    s = replace_once(s, old_counts, new_counts, 'world cache counts')
    marker = '    constants.probe_config[3] = 1u;\n'
    addition = marker + '''    constants.world_probe_config[0] = WORLD_PROBE_DIRECTION_SIZE;
    constants.world_probe_config[1] = WORLD_PROBE_UPDATES_PER_FRAME;
    constants.world_probe_config[2] = WORLD_PROBE_BANK_COUNT;
    constants.world_probe_config[3] = 0u;
    constants.world_probe_params[0] = WORLD_PROBE_SPACING;
    constants.world_probe_params[1] = WORLD_PROBE_BLEND;
    constants.world_probe_params[2] = WORLD_PROBE_RADIUS;
    constants.world_probe_params[3] = WORLD_PROBE_CLEARANCE;
'''
    s = replace_once(s, marker, addition, 'world probe constants')
    s = replace_once(s, '    constants.feature_flags[1] = 1u;\n',
                     '    if (renderer->world_radiance.probe_count) constants.feature_flags[0] |= RADIANCE_FEATURE_WORLD_CACHE | RADIANCE_FEATURE_MULTIBOUNCE;\n    constants.feature_flags[1] = renderer->radiance_revision ? renderer->radiance_revision : 1u;\n', 'world feature/revision')

    s = replace_once(s, '!create_global_sdf_resources(renderer) || !update_radiance_constants(renderer)',
                     '!create_global_sdf_resources(renderer) || !build_world_radiance_scene(renderer) || !update_radiance_constants(renderer)', 'world scene build')
    s = replace_once(s, 'if (!update_scene_objects(renderer) || !refresh_dynamic_grid(renderer) || !refresh_emissive_sampling(renderer)) return false;\n    update_orbit_camera(renderer);',
                     'if (!update_scene_objects(renderer) || !refresh_dynamic_grid(renderer) || !refresh_emissive_sampling(renderer)) return false;\n    refresh_radiance_revision(renderer);\n    update_orbit_camera(renderer);', 'world revision refresh')
    p.write_text(s)


WORLD_SHADER_SECTION = r'''// -----------------------------------------------------------------------------
// World radiance cache.
// -----------------------------------------------------------------------------
uint WorldProbeDirectionCount() {
    uint s = max(Radiance.world_probe_config.x, 1u);
    return s * s;
}

uint WorldProbeRadianceIndex(uint probe_index, uint2 texel, uint bank) {
    uint s = max(Radiance.world_probe_config.x, 1u);
    uint direction = texel.x + s * texel.y;
    uint bank_stride = Radiance.cache_counts.w * WorldProbeDirectionCount();
    return bank * bank_stride + probe_index * WorldProbeDirectionCount() + direction;
}

uint WorldProbeCellKey(int3 cell) {
    uint h = HashCombine(asuint(cell.x), asuint(cell.y));
    h = HashCombine(h, asuint(cell.z));
    return h | 1u;
}

uint WorldProbeKey(float3 position) {
    float cell_size = max(Radiance.world_probe_params.x, 0.25f);
    return WorldProbeCellKey(int3(floor(position / cell_size)));
}

uint FindWorldProbeInternal(float3 position, float3 normal, bool surface_filter) {
    uint count = Radiance.cache_counts.y;
    if (!FeatureEnabled(RADIANCE_FEATURE_WORLD_CACHE) || count == 0u) return INVALID_INDEX;
    uint table_capacity = max(Radiance.cache_counts.z, 1u);
    uint mask = table_capacity - 1u;
    float cell_size = max(Radiance.world_probe_params.x, 0.25f);
    int3 base_cell = int3(floor(position / cell_size));
    uint best_index = INVALID_INDEX;
    float best_distance = 3.402823466e+38f;
    for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                uint key = WorldProbeCellKey(base_cell + int3(dx, dy, dz));
                [unroll]
                for (uint i = 0u; i < MAX_CACHE_PROBES; ++i) {
                    uint slot = (key + i) & mask;
                    uint encoded = WorldProbeKeys[slot];
                    if (encoded == 0u) break;
                    uint index = encoded - 1u;
                    if (index >= count) continue;
                    WorldProbeState state = WorldProbes[index];
                    if (state.identity.x != key || (state.state.w & 1u) == 0u || state.state.y != Radiance.feature_flags.y) continue;
                    if (surface_filter && dot(state.position_radius.xyz - position, normal) <= 1.0e-4f) continue;
                    float d = distance(state.position_radius.xyz, position);
                    if (d > state.position_radius.w || d >= best_distance) continue;
                    best_distance = d;
                    best_index = index;
                }
            }
        }
    }
    return best_index;
}

uint FindWorldProbe(float3 position) {
    return FindWorldProbeInternal(position, 0.0f, false);
}

uint FindWorldProbeForSurface(float3 position, float3 normal) {
    return FindWorldProbeInternal(position, normalize(normal), true);
}

uint WorldProbeReadBank(WorldProbeState state, bool stable_previous) {
    uint bank = state.state.z & 1u;
    if (stable_previous && state.state.x == Pass.dispatch.x) bank ^= 1u;
    return bank;
}

float3 SampleWorldProbeDirectional(uint probe_index, float3 direction, bool stable_previous) {
    if (probe_index == INVALID_INDEX || probe_index >= Radiance.cache_counts.y) return 0.0f;
    WorldProbeState state = WorldProbes[probe_index];
    if ((state.state.w & 1u) == 0u || state.state.y != Radiance.feature_flags.y) return 0.0f;
    uint s = max(Radiance.world_probe_config.x, 1u);
    float2 uv = OctEncode(direction);
    uint2 texel = min((uint2)floor(uv * (float)s), uint2(s - 1u, s - 1u));
    return WorldProbeRadiance[WorldProbeRadianceIndex(probe_index, texel, WorldProbeReadBank(state, stable_previous))].rgb;
}

float3 IntegrateWorldProbeDiffuse(float3 position, float3 normal, bool stable_previous) {
    uint probe = FindWorldProbeForSurface(position, normal);
    if (probe == INVALID_INDEX) return 0.0f;
    WorldProbeState state = WorldProbes[probe];
    uint bank = WorldProbeReadBank(state, stable_previous);
    uint s = max(Radiance.world_probe_config.x, 1u);
    float3 sum = 0.0f;
    float weight_sum = 0.0f;
    for (uint d = 0u; d < WorldProbeDirectionCount(); ++d) {
        uint2 texel = uint2(d % s, d / s);
        float3 direction = OctDecode((float2(texel) + 0.5f) / (float)s);
        float weight = saturate(dot(normal, direction));
        sum += WorldProbeRadiance[WorldProbeRadianceIndex(probe, texel, bank)].rgb * weight;
        weight_sum += weight;
    }
    return weight_sum > 0.0f ? sum / weight_sum : 0.0f;
}

float3 WorldIndirectReflected(SurfaceHit hit, bool stable_previous) {
    if (!FeatureEnabled(RADIANCE_FEATURE_MULTIBOUNCE) || hit.identity.y >= Radiance.scene_counts.y) return 0.0f;
    GPUMaterial material = SceneMaterials[hit.identity.y];
    float diffuse = saturate(1.0f - material.metallic);
    float3 incoming = IntegrateWorldProbeDiffuse(hit.position_distance.xyz, normalize(hit.normal_confidence.xyz), stable_previous);
    return material.base_color.rgb * diffuse * incoming;
}

float3 WorldRadianceFallback(float3 position, float3 direction) {
    uint probe = FindWorldProbe(position);
    return probe == INVALID_INDEX ? 0.0f : SampleWorldProbeDirectional(probe, direction, false);
}
'''

WORLD_UPDATE_FUNCTION = r'''[numthreads(64, 1, 1)]
void CS_UpdateWorldRadianceCache(uint3 dispatch_id : SV_DispatchThreadID) {
    uint update_index = dispatch_id.x;
    uint update_count = min(RayCounters[3], Radiance.world_probe_config.y);
    if (update_index >= update_count) return;
    uint probe_index = RadianceUpdateList[update_index];
    if (probe_index >= Radiance.cache_counts.y) return;
    WorldProbeState state = WorldProbes[probe_index];
    uint old_bank = state.state.z & 1u;
    uint new_bank = old_bank ^ 1u;
    bool established = (state.state.w & 1u) != 0u && state.state.y == Radiance.feature_flags.y;
    uint s = max(Radiance.world_probe_config.x, 1u);
    for (uint d = 0u; d < WorldProbeDirectionCount(); ++d) {
        uint2 texel = uint2(d % s, d / s);
        float3 direction = OctDecode((float2(texel) + 0.5f) / (float)s);
        float bias = max(Radiance.trace_params.y, 1.0e-3f);
        TraceRay ray = MakeTraceRay(state.position_radius.xyz, direction, bias, Radiance.trace_params.x, TRACE_RAY_WORLD_PROBE, d, uint2(0u, 0u), 0u);
        SurfaceHit hit = TraceUnifiedRay(ray, false);
        float3 sample = FutureSkyRadiance(direction);
        if (hit.identity.w != TRACE_MISS && hit.identity.w != TRACE_INACTIVE && hit.identity.y < Radiance.scene_counts.y) {
            GPUMaterial material = SceneMaterials[hit.identity.y];
            float3 normal = normalize(hit.normal_confidence.xyz);
            float3 reflected = EvaluateSurfaceReflectedDirect(hit);
            uint seed = HashCombine(probe_index, HashCombine(d, Pass.dispatch.x));
            float3 nee = EvaluateEmissiveSampleForMaterial(hit.position_distance.xyz, normal, hit.identity.x, hit.identity.y, seed);
            reflected += material.base_color.rgb * saturate(1.0f - material.metallic) * nee;
            reflected += WorldIndirectReflected(hit, true);
            sample = material.emissive + reflected;
        }
        uint old_address = WorldProbeRadianceIndex(probe_index, texel, old_bank);
        uint new_address = WorldProbeRadianceIndex(probe_index, texel, new_bank);
        float3 old = WorldProbeRadiance[old_address].rgb;
        float blend = established ? saturate(Radiance.world_probe_params.y) : 1.0f;
        WorldProbeRadiance[new_address] = float4(lerp(old, sample, blend), 1.0f);
    }
    state.statistics.x = established ? saturate(state.statistics.x + 0.1f) : 1.0f;
    state.statistics.z = established ? state.statistics.z + 1.0f : 1.0f;
    state.state.x = Pass.dispatch.x;
    state.state.y = Radiance.feature_flags.y;
    state.state.z = new_bank;
    state.state.w = 1u;
    WorldProbes[probe_index] = state;
}'''


def task4():
    p = Path('shader.hlsl')
    s = p.read_text()
    start_marker = '// -----------------------------------------------------------------------------\n// World radiance cache.\n// -----------------------------------------------------------------------------'
    end_marker = '// -----------------------------------------------------------------------------\n// Wavefront passes.'
    a = s.find(start_marker)
    b = s.find(end_marker, a)
    if a < 0 or b < 0: raise SystemExit('world shader section markers missing')
    s = s[:a] + WORLD_SHADER_SECTION.rstrip() + '\n\n' + s[b:]
    forward = 'float3 SurfaceReflectedRadiance(SurfaceHit hit);\n'
    if 'float3 WorldIndirectReflected(SurfaceHit hit, bool stable_previous);' not in s:
        s = replace_once(s, forward, forward + 'float3 WorldIndirectReflected(SurfaceHit hit, bool stable_previous);\n', 'world indirect forward')
    reflected = r'''float3 SurfaceReflectedRadiance(SurfaceHit hit) {
    if (hit.identity.w == TRACE_MISS || hit.identity.y >= Radiance.scene_counts.y) return 0.0f;
    SurfaceCacheEntry entry;
    float3 direct;
    if (SurfaceCacheLookup(hit, entry)) direct = entry.direct_radiance.rgb;
    else direct = EvaluateSurfaceReflectedDirect(hit);
    float3 indirect = WorldIndirectReflected(hit, false);
    SurfaceCacheStore(hit, direct, indirect, 1.0f);
    return direct + indirect;
}'''
    s = replace_function(s, 'float3 SurfaceReflectedRadiance(SurfaceHit hit)', reflected)
    s = replace_once(s, '        RayRadiance[index] = float4(ReflectedDirectAtPixel(hit_pixel), 1.0f);',
                     '        RayRadiance[index] = float4(ReflectedDirectAtPixel(hit_pixel) + WorldIndirectReflected(hit, false), 1.0f);', 'screen world indirect')
    s = replace_function(s, '[numthreads(64, 1, 1)]\nvoid CS_UpdateWorldRadianceCache', WORLD_UPDATE_FUNCTION)
    old_invalid = '''        WorldProbeState state = WorldProbes[target];
        state.statistics.x *= confidence_scale;
        state.state.z = 1u;
        WorldProbes[target] = state;'''
    new_invalid = '''        WorldProbeState state = WorldProbes[target];
        state.statistics.x *= confidence_scale;
        state.state.w |= 2u;
        WorldProbes[target] = state;'''
    s = replace_once(s, old_invalid, new_invalid, 'world invalidation')
    p.write_text(s)


WORLD_SCHEDULER = r'''static void prepare_world_probe_updates(RENDERER *renderer) {
    RADIANCE_WORLD_RESOURCES *world = &renderer->world_radiance;
    world->update_count = world->probe_count < WORLD_PROBE_UPDATES_PER_FRAME ? world->probe_count : WORLD_PROBE_UPDATES_PER_FRAME;
    if (!world->probe_count) {
        world->update_cursor = 0u;
        world->update_count = 0u;
        renderer->pass_constants.range[0] = 0u;
        return;
    }
    for (uint32_t i = 0u; i < world->update_count; ++i)
        world->cpu_update_list[i] = (world->update_cursor + i) % world->probe_count;
    renderer->pass_constants.range[0] = world->update_count;
    world->update_cursor = (world->update_cursor + world->update_count) % world->probe_count;
}
'''

WORLD_BARRIER = r'''static void barrier_world_radiance(RENDERER *renderer, NriCommandBuffer *command_buffer) {
    const NriAccessStage storage = {.access = NriAccessBits_SHADER_RESOURCE_STORAGE, .stages = NriStageBits_COMPUTE_SHADER};
    NriBufferBarrierDesc barriers[] = {
        {.buffer = renderer->world_radiance.probes, .before = renderer->world_radiance.state, .after = storage},
        {.buffer = renderer->world_radiance.radiance, .before = renderer->world_radiance.state, .after = storage},
        {.buffer = renderer->world_radiance.keys, .before = renderer->world_radiance.state, .after = storage}
    };
    renderer->gpu->core.CmdBarrier(command_buffer, &(NriBarrierDesc){.buffers = barriers, .bufferNum = 3});
    renderer->world_radiance.state = storage;
}'''


def task5():
    p = Path('shader.hlsl')
    s = p.read_text()
    s = replace_once(s, '    RayCounters[3] = 0u;\n', '    RayCounters[3] = Pass.range.x;\n', 'world update reset count')
    p.write_text(s)

    p = Path('render.c')
    s = p.read_text()
    s = insert_after_function(s, 'static bool refresh_emissive_sampling(RENDERER *renderer)', WORLD_SCHEDULER)
    s = insert_after_function(s, 'static void barrier_wavefront_buffers(RENDERER *renderer, NriCommandBuffer *command_buffer, NriAccessStage state)', WORLD_BARRIER)

    s = replace_once(s,
        '           create_compute_pipeline(renderer, "build/shaders/radiance_probe_history.cs.spv", renderer->wavefront_layout, &renderer->wavefront_history_pipeline);',
        '           create_compute_pipeline(renderer, "build/shaders/radiance_probe_history.cs.spv", renderer->wavefront_layout, &renderer->wavefront_history_pipeline) &&\n           create_compute_pipeline(renderer, "build/shaders/radiance_world_cache.cs.spv", renderer->wavefront_layout, &renderer->world_radiance_pipeline);',
        'world pipeline creation')
    s = replace_once(s, '            renderer->wavefront_history_pipeline\n', '            renderer->wavefront_history_pipeline,\n            renderer->world_radiance_pipeline\n', 'world pipeline destruction')

    old_pass = '''    memset(&renderer->pass_constants, 0, sizeof(renderer->pass_constants));
    renderer->pass_constants.dispatch[0] = renderer->radiance_constants.feature_flags[1];
    renderer->pass_constants.dimensions[0] = renderer->screen_probes.width;
    renderer->pass_constants.dimensions[1] = renderer->screen_probes.height;
    renderer->pass_constants.flags[0] = renderer->probe_history_valid ? 1u : 0u;'''
    new_pass = '''    memset(&renderer->pass_constants, 0, sizeof(renderer->pass_constants));
    renderer->pass_constants.dispatch[0] = (uint32_t)(renderer->gpu->frame_index & 0x00ffffffu);
    renderer->pass_constants.dimensions[0] = renderer->screen_probes.width;
    renderer->pass_constants.dimensions[1] = renderer->screen_probes.height;
    renderer->pass_constants.flags[0] = renderer->probe_history_valid ? 1u : 0u;
    prepare_world_probe_updates(renderer);'''
    s = replace_once(s, old_pass, new_pass, 'frame stamp/scheduler')

    s = replace_once(s, '    NriStreamBufferDataDesc uploads[8];\n', '    NriStreamBufferDataDesc uploads[9];\n', 'stream upload capacity')
    grid_marker = '    const NriDataSize grid_index_data = {.data = renderer->dynamic_grid.cpu_indices, .size = (uint64_t)index_upload_count * sizeof(uint32_t)};\n'
    s = replace_once(s, grid_marker, grid_marker + '    const NriDataSize world_update_data = {.data = renderer->world_radiance.cpu_update_list, .size = (uint64_t)renderer->world_radiance.update_count * sizeof(uint32_t)};\n', 'world update data')
    marker = '''    if (upload_grid) {
        uploads[upload_count++] = (NriStreamBufferDataDesc){.dataChunks = &grid_cell_data, .dataChunkNum = 1, .placementAlignment = 16, .copyBatch = batch, .dstBuffer = renderer->dynamic_grid.cells};
        uploads[upload_count++] = (NriStreamBufferDataDesc){.dataChunks = &grid_index_data, .dataChunkNum = 1, .placementAlignment = 16, .copyBatch = batch, .dstBuffer = renderer->dynamic_grid.indices};
    }
'''
    replacement = marker + '''    const bool upload_world_updates = renderer->world_radiance.update_count > 0u;
    if (upload_world_updates)
        uploads[upload_count++] = (NriStreamBufferDataDesc){.dataChunks = &world_update_data, .dataChunkNum = 1, .placementAlignment = 16, .copyBatch = batch, .dstBuffer = renderer->wavefront.update_list};
'''
    s = replace_once(s, marker, replacement, 'world update stream')
    s = replace_once(s, '    NriBufferBarrierDesc before[8];\n', '    NriBufferBarrierDesc before[9];\n', 'before barrier capacity')
    before_grid = '''    if (upload_grid) {
        before[barrier_count++] = (NriBufferBarrierDesc){.buffer = renderer->dynamic_grid.cells, .before = renderer->dynamic_grid.state, .after = copy};
        before[barrier_count++] = (NriBufferBarrierDesc){.buffer = renderer->dynamic_grid.indices, .before = renderer->dynamic_grid.state, .after = copy};
    }
'''
    s = replace_once(s, before_grid, before_grid + '    if (upload_world_updates) before[barrier_count++] = (NriBufferBarrierDesc){.buffer = renderer->wavefront.update_list, .before = renderer->wavefront.update_list_state, .after = copy};\n', 'world update before barrier')
    s = replace_once(s, '    NriBufferBarrierDesc after[8];\n', '    NriBufferBarrierDesc after[9];\n', 'after barrier capacity')
    after_grid = '''    if (upload_grid) {
        after[barrier_count++] = (NriBufferBarrierDesc){.buffer = renderer->dynamic_grid.cells, .before = copy, .after = compute_read};
        after[barrier_count++] = (NriBufferBarrierDesc){.buffer = renderer->dynamic_grid.indices, .before = copy, .after = compute_read};
    }
'''
    s = replace_once(s, after_grid, after_grid + '    if (upload_world_updates) after[barrier_count++] = (NriBufferBarrierDesc){.buffer = renderer->wavefront.update_list, .before = copy, .after = (NriAccessStage){.access = NriAccessBits_SHADER_RESOURCE_STORAGE, .stages = NriStageBits_COMPUTE_SHADER}};\n', 'world update after barrier')
    s = replace_once(s, '''    if (upload_grid) {
        renderer->dynamic_grid.state = compute_read;
        renderer->dynamic_grid.dirty = false;
    }
    return true;''', '''    if (upload_grid) {
        renderer->dynamic_grid.state = compute_read;
        renderer->dynamic_grid.dirty = false;
    }
    if (upload_world_updates) renderer->wavefront.update_list_state = (NriAccessStage){.access = NriAccessBits_SHADER_RESOURCE_STORAGE, .stages = NriStageBits_COMPUTE_SHADER};
    return true;''', 'world update state')

    reset_dispatch = '''    renderer->gpu->core.CmdDispatch(command_buffer, &(NriDispatchDesc){.workGroupNumX = 1, .workGroupNumY = 1, .workGroupNumZ = 1});
    barrier_wavefront_buffers(renderer, command_buffer, storage);

    bind_wavefront(renderer, command_buffer, renderer->wavefront_budget_pipeline);'''
    world_dispatch = '''    renderer->gpu->core.CmdDispatch(command_buffer, &(NriDispatchDesc){.workGroupNumX = 1, .workGroupNumY = 1, .workGroupNumZ = 1});
    barrier_wavefront_buffers(renderer, command_buffer, storage);

    RADIANCE_WORLD_RESOURCES *world = &renderer->world_radiance;
    if (world->update_count && (renderer->radiance_constants.feature_flags[0] & RADIANCE_FEATURE_WORLD_CACHE)) {
        bind_wavefront(renderer, command_buffer, renderer->world_radiance_pipeline);
        renderer->gpu->core.CmdDispatch(command_buffer, &(NriDispatchDesc){.workGroupNumX = (world->update_count + 63u) / 64u, .workGroupNumY = 1, .workGroupNumZ = 1});
        barrier_world_radiance(renderer, command_buffer);
    }

    bind_wavefront(renderer, command_buffer, renderer->wavefront_budget_pipeline);'''
    s = replace_once(s, reset_dispatch, world_dispatch, 'world update dispatch order')
    p.write_text(s)


def main():
    if len(sys.argv) != 2:
        raise SystemExit('usage: stage10_apply.py test1|1|test2|2|3|4|5')
    step = sys.argv[1]
    if step == 'test1': write_test_interface()
    elif step == '1': task1()
    elif step == 'test2': write_test_behavior()
    elif step == '2': task2()
    elif step == '3': task3()
    elif step == '4': task4()
    elif step == '5': task5()
    else: raise SystemExit(f'unknown step: {step}')


if __name__ == '__main__':
    main()
