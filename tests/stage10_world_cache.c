#include "game.h"

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
    for (uint32_t repeat = 0u; repeat < 8u; ++repeat) {
        memset(b, 0, WORLD_PROBE_CAPACITY * sizeof(*b));
        memset(kb, 0, WORLD_PROBE_HASH_CAPACITY * sizeof(*kb));
        nb = 0u;
        assert(sdf_build_world_probes(&g, &scene, &object, 1u, b, WORLD_PROBE_CAPACITY, &nb, kb, WORLD_PROBE_HASH_CAPACITY,
                                      WORLD_PROBE_SPACING, WORLD_PROBE_RADIUS, WORLD_PROBE_CLEARANCE, WORLD_PROBE_MIN_CLEARANCE));
        assert(na == nb);
        assert(memcmp(a, b, (size_t)na * sizeof(*a)) == 0);
        assert(memcmp(ka, kb, WORLD_PROBE_HASH_CAPACITY * sizeof(*ka)) == 0);
    }

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
