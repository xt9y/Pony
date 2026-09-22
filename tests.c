#include <assert.h>
#include <math.h>
#include <stdio.h>

#include "msh.c"
#include "col.c"
#include "bvh.c"

static int nearf(float a, float b)
{
    return fabsf(a - b) < 1e-5f;
}

static void test_mesh_validation_rejects_bad_index(void)
{
    VERTEX vertices[4] = {0};
    uint32_t indices[3] = {0, 1, 9};

    assert(!m_validate(vertices, 4, indices, 3, NULL, 0));
}

static void test_mesh_bounds(void)
{
    VERTEX vertices[2] = {
        {{-1.0f, -2.0f, -3.0f}, {0}, {0}},
        {{ 5.0f,  4.0f,  2.0f}, {0}, {0}},
    };
    AABB bounds;

    assert(m_bounds(vertices, 2, &bounds));
    assert(nearf(bounds.min.x, -1.0f));
    assert(nearf(bounds.min.y, -2.0f));
    assert(nearf(bounds.min.z, -3.0f));
    assert(nearf(bounds.max.x,  5.0f));
    assert(nearf(bounds.max.y,  4.0f));
    assert(nearf(bounds.max.z,  2.0f));
}

static void test_triangle_access(void)
{
    VERTEX vertices[3] = {
        {{0.0f, 0.0f, 0.0f}, {0}, {0}},
        {{1.0f, 0.0f, 0.0f}, {0}, {0}},
        {{0.0f, 1.0f, 0.0f}, {0}, {0}},
    };
    uint32_t indices[3] = {0, 1, 2};
    MESH mesh = {
        .vertices = vertices,
        .vertex_count = 3,
        .indices = indices,
        .index_count = 3,
    };
    TRIANGLE tri;

    assert(m_triangle(&mesh, 0, &tri));
    assert(nearf(tri.a.x, 0.0f) && nearf(tri.a.y, 0.0f));
    assert(nearf(tri.b.x, 1.0f) && nearf(tri.b.y, 0.0f));
    assert(nearf(tri.c.x, 0.0f) && nearf(tri.c.y, 1.0f));
    assert(!m_triangle(&mesh, 1, &tri));
}

static void test_world_bounds_rotate_all_corners(void)
{
    AABB local = {
        .min = {-2.0f, -0.5f, -0.5f},
        .max = { 2.0f,  0.5f,  0.5f},
    };
    float c = 0.70710678118f;
    float matrix[16] = {
         c, 0.0f, -c, 0.0f,
         0.0f, 1.0f, 0.0f, 0.0f,
         c, 0.0f,  c, 0.0f,
         3.0f, 2.0f, -4.0f, 1.0f,
    };
    AABB world;

    assert(m_world_bounds(local, matrix, &world));

    const float expected_extent = 2.5f * c;
    assert(nearf(world.min.x, 3.0f - expected_extent));
    assert(nearf(world.max.x, 3.0f + expected_extent));
    assert(nearf(world.min.z, -4.0f - expected_extent));
    assert(nearf(world.max.z, -4.0f + expected_extent));
    assert(nearf(world.min.y, 1.5f));
    assert(nearf(world.max.y, 2.5f));
}

static void test_lightmap_corner_validation(void)
{
    VERTEX vertices[3] = {
        {{0.0f, 0.0f, 0.0f}, {0}, {0}},
        {{1.0f, 0.0f, 0.0f}, {0}, {0}},
        {{0.0f, 1.0f, 0.0f}, {0}, {0}},
    };
    uint32_t indices[3] = {0, 1, 2};
    LIGHTMAP_CORNER corners[3] = {{0, 0}, {1, 0}, {0, 1}};

    assert(m_validate(vertices, 3, indices, 3, corners, 3));
    assert(!m_validate(vertices, 3, indices, 3, corners, 2));
}

static void test_ray_aabb_front_inside_parallel(void)
{
    AABB box = {{-1, -1, -1}, {1, 1, 1}};
    RAY front;
    assert(ray_make((VEC3){0, 0, -5}, (VEC3){0, 0, 1}, 0.0f, 100.0f, &front));
    float near_t = -1.0f;
    assert(ray_aabb(front, box, &near_t));
    assert(nearf(near_t, 4.0f));

    RAY inside;
    assert(ray_make((VEC3){0, 0, 0}, (VEC3){1, 0, 0}, 0.0f, 100.0f, &inside));
    assert(ray_aabb(inside, box, &near_t));
    assert(nearf(near_t, 0.0f));

    RAY parallel;
    assert(ray_make((VEC3){2, 0, 0}, (VEC3){0, 1, 0}, 0.0f, 100.0f, &parallel));
    assert(!ray_aabb(parallel, box, NULL));
}

static void test_ray_triangle_hit_and_miss(void)
{
    TRIANGLE tri = {
        {-1, -1, 0},
        { 1, -1, 0},
        { 0,  1, 0}
    };
    RAY hit;
    assert(ray_make((VEC3){0, 0, -1}, (VEC3){0, 0, 1}, 0.0f, 100.0f, &hit));
    float t, u, v;
    assert(ray_triangle(hit, tri, &t, &u, &v));
    assert(nearf(t, 1.0f));
    assert(u >= 0.0f && v >= 0.0f && u + v <= 1.0f);

    RAY miss;
    assert(ray_make((VEC3){3, 0, -1}, (VEC3){0, 0, 1}, 0.0f, 100.0f, &miss));
    assert(!ray_triangle(miss, tri, &t, &u, &v));
}

static void test_mesh_bruteforce_closest_and_any_hit(void)
{
    VERTEX vertices[6] = {
        {{-1, -1, 0}, {0, 0, -1}, {0}},
        {{ 1, -1, 0}, {0, 0, -1}, {0}},
        {{ 0,  1, 0}, {0, 0, -1}, {0}},
        {{-1, -1, 2}, {0, 0, -1}, {0}},
        {{ 1, -1, 2}, {0, 0, -1}, {0}},
        {{ 0,  1, 2}, {0, 0, -1}, {0}},
    };
    uint32_t indices[6] = {0, 1, 2, 3, 4, 5};
    MESH mesh = {
        .vertices = vertices,
        .vertex_count = 6,
        .indices = indices,
        .index_count = 6,
    };
    RAY ray;
    assert(ray_make((VEC3){0, 0, -2}, (VEC3){0, 0, 1}, 0.0f, 100.0f, &ray));
    RAY_HIT hit = mesh_raycast_bruteforce(&mesh, ray, 0);
    assert(hit.hit);
    assert(hit.triangle_index == 0);
    assert(nearf(hit.t, 2.0f));
    assert(mesh_occluded_bruteforce(&mesh, ray, 0));
}

static void test_scene_nonuniform_transform_keeps_world_distance(void)
{
    VERTEX vertices[3] = {
        {{-1, -1, 0}, {0, 0, -1}, {0}},
        {{ 1, -1, 0}, {0, 0, -1}, {0}},
        {{ 0,  1, 0}, {0, 0, -1}, {0}},
    };
    uint32_t indices[3] = {0, 1, 2};
    MESH mesh = {
        .vertices = vertices,
        .vertex_count = 3,
        .indices = indices,
        .index_count = 3,
        .local_bounds = {{-1, -1, 0}, {1, 1, 0}},
    };
    RENDER_INSTANCE instance = {
        .mesh = 0,
        .mobility = MOBILITY_STATIC,
        .query_mask = 0xffffffffu,
    };
    float world[16] = {
        2, 0, 0, 0,
        0, 3, 0, 0,
        0, 0, 4, 0,
        0, 0, 5, 1,
    };
    SDL_memcpy(instance.transform.matrix, world, sizeof(world));
    assert(scene_instance_update(&instance, &mesh));

    SCENE scene = {
        .meshes = &mesh,
        .mesh_count = 1,
        .instances = &instance,
        .instance_count = 1,
    };
    RAY ray;
    assert(ray_make((VEC3){0, 0, 0}, (VEC3){0, 0, 1}, 0.0f, 100.0f, &ray));
    QUERY_FILTER filter = query_filter_all();
    RAY_HIT hit = scene_raycast(&scene, ray, filter);
    assert(hit.hit);
    assert(nearf(hit.t, 5.0f));
    assert(nearf(hit.position.z, 5.0f));
}

static void test_overlap_queries(void)
{
    AABB a = {{-1, -1, -1}, {1, 1, 1}};
    AABB b = {{0.5f, 0.5f, 0.5f}, {2, 2, 2}};
    AABB c = {{3, 3, 3}, {4, 4, 4}};
    assert(aabb_overlap(a, b));
    assert(!aabb_overlap(a, c));
    assert(sphere_aabb_overlap((VEC3){2, 0, 0}, 1.1f, a));
    assert(!sphere_aabb_overlap((VEC3){3, 0, 0}, 1.0f, a));
}

static uint32_t test_rng_next(uint32_t *state)
{
    *state = *state * 1664525u + 1013904223u;
    return *state;
}

static float test_rng_signed(uint32_t *state)
{
    return ((float)(test_rng_next(state) & 0xffffu) / 32767.5f) - 1.0f;
}

static void test_bruteforce_random_reference_is_deterministic(void)
{
    VERTEX vertices[4] = {
        {{-1, -1, 0}, {0, 0, 1}, {0}},
        {{ 1, -1, 0}, {0, 0, 1}, {0}},
        {{ 1,  1, 0}, {0, 0, 1}, {0}},
        {{-1,  1, 0}, {0, 0, 1}, {0}},
    };
    uint32_t indices[6] = {0, 1, 2, 0, 2, 3};
    MESH mesh = {
        .vertices = vertices,
        .vertex_count = 4,
        .indices = indices,
        .index_count = 6,
    };
    RAY_HIT first[128];

    uint32_t seed = 0x12345678u;
    for (uint32_t i = 0; i < 128; ++i) {
        VEC3 origin = {test_rng_signed(&seed) * 2.0f, test_rng_signed(&seed) * 2.0f, -2.0f};
        VEC3 direction = {test_rng_signed(&seed) * 0.2f, test_rng_signed(&seed) * 0.2f, 1.0f};
        RAY ray;
        assert(ray_make(origin, direction, 0.0f, 100.0f, &ray));
        first[i] = mesh_raycast_bruteforce(&mesh, ray, 0);
    }

    seed = 0x12345678u;
    for (uint32_t i = 0; i < 128; ++i) {
        VEC3 origin = {test_rng_signed(&seed) * 2.0f, test_rng_signed(&seed) * 2.0f, -2.0f};
        VEC3 direction = {test_rng_signed(&seed) * 0.2f, test_rng_signed(&seed) * 0.2f, 1.0f};
        RAY ray;
        assert(ray_make(origin, direction, 0.0f, 100.0f, &ray));
        RAY_HIT again = mesh_raycast_bruteforce(&mesh, ray, 0);
        assert(again.hit == first[i].hit);
        if (again.hit) {
            assert(again.triangle_index == first[i].triangle_index);
            assert(nearf(again.t, first[i].t));
            assert(nearf(again.bary_u, first[i].bary_u));
            assert(nearf(again.bary_v, first[i].bary_v));
        }
    }
}

static void test_blas_matches_bruteforce(void)
{
    VERTEX vertices[8] = {
        {{-1,-1,0},{0,0,1},{0}}, {{1,-1,0},{0,0,1},{0}},
        {{1,1,0},{0,0,1},{0}}, {{-1,1,0},{0,0,1},{0}},
        {{-1,-1,2},{0,0,1},{0}}, {{1,-1,2},{0,0,1},{0}},
        {{1,1,2},{0,0,1},{0}}, {{-1,1,2},{0,0,1},{0}},
    };
    uint32_t indices[12] = {0,1,2,0,2,3,4,5,6,4,6,7};
    MESH mesh = {
        .vertices = vertices,
        .vertex_count = 8,
        .indices = indices,
        .index_count = 12,
        .geometry_revision = 1,
    };
    assert(bvh_build_mesh(&mesh, 2));
    assert(bvh_validate(mesh.blas));

    uint32_t seed = 0x51a73u;
    for (uint32_t i = 0; i < 512; ++i) {
        VEC3 origin = {test_rng_signed(&seed) * 2.5f, test_rng_signed(&seed) * 2.5f, -3.0f};
        VEC3 direction = {test_rng_signed(&seed) * 0.25f, test_rng_signed(&seed) * 0.25f, 1.0f};
        RAY ray;
        assert(ray_make(origin, direction, 0.0f, 100.0f, &ray));
        RAY_HIT brute = mesh_raycast_bruteforce(&mesh, ray, 0);
        BVH_TRACE_STATS stats = {0};
        RAY_HIT accel = bvh_mesh_raycast(&mesh, mesh.blas, ray, 0, 1.0f, &stats);
        assert(accel.hit == brute.hit);
        if (brute.hit) {
            assert(accel.triangle_index == brute.triangle_index);
            assert(fabsf(accel.t - brute.t) < 1e-5f);
        }
        assert(bvh_mesh_occluded(&mesh, mesh.blas, ray, 0, 1.0f, NULL) == brute.hit);
    }

    bvh_destroy_mesh(&mesh, NULL);
}

static void test_tlas_matches_scene_bruteforce_and_refits(void)
{
    VERTEX vertices[3] = {
        {{-1,-1,0},{0,0,1},{0}},
        {{1,-1,0},{0,0,1},{0}},
        {{0,1,0},{0,0,1},{0}},
    };
    uint32_t indices[3] = {0,1,2};
    MESH mesh = {
        .vertices = vertices,
        .vertex_count = 3,
        .indices = indices,
        .index_count = 3,
        .local_bounds = {{-1,-1,0},{1,1,0}},
        .geometry_revision = 1,
    };
    RENDER_INSTANCE instances[2] = {
        {.mesh = 0, .mobility = MOBILITY_STATIC, .query_mask = 0xffffffffu},
        {.mesh = 0, .mobility = MOBILITY_DYNAMIC, .query_mask = 0xffffffffu},
    };
    float a[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, -2,0,5,1};
    float b[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0,  2,0,7,1};
    SDL_memcpy(instances[0].transform.matrix, a, sizeof(a));
    SDL_memcpy(instances[1].transform.matrix, b, sizeof(b));
    assert(scene_instance_update(&instances[0], &mesh));
    assert(scene_instance_update(&instances[1], &mesh));

    SCENE scene = {
        .meshes = &mesh,
        .mesh_count = 1,
        .instances = instances,
        .instance_count = 2,
        .transform_revision = 1,
        .geometry_revision = 1,
    };
    assert(scene_accel_update(&scene, NULL));
    BVH *original_blas = mesh.blas;

    RAY ray;
    assert(ray_make((VEC3){-2,0,0}, (VEC3){0,0,1}, 0, 100, &ray));
    QUERY_FILTER filter = query_filter_all();
    RAY_HIT accel = bvh_scene_raycast(&scene, ray, filter, NULL);
    assert(accel.hit && accel.instance_index == 0 && nearf(accel.t, 5.0f));

    instances[0].transform.matrix[12] = 20.0f;
    assert(scene_instance_update(&instances[0], &mesh));
    ++scene.transform_revision;
    assert(scene_accel_update(&scene, NULL));
    assert(mesh.blas == original_blas);
    accel = bvh_scene_raycast(&scene, ray, filter, NULL);
    assert(!accel.hit);

    scene_accel_destroy(&scene, NULL);
}

static void test_flattened_escape_indices(void)
{
    VERTEX vertices[6] = {
        {{-3,0,0},{0,1,0},{0}}, {{-2,0,0},{0,1,0},{0}}, {{-2.5f,1,0},{0,1,0},{0}},
        {{ 2,0,0},{0,1,0},{0}}, {{ 3,0,0},{0,1,0},{0}}, {{ 2.5f,1,0},{0,1,0},{0}},
    };
    uint32_t indices[6] = {0,1,2,3,4,5};
    MESH mesh = {.vertices=vertices,.vertex_count=6,.indices=indices,.index_count=6,.geometry_revision=1};
    assert(bvh_build_mesh(&mesh, 1));
    GPU_BVH_NODE *flat = NULL;
    uint32_t count = 0;
    assert(bvh_flatten(mesh.blas, &flat, &count));
    assert(count == mesh.blas->node_count);
    assert(flat[0].escape == UINT32_MAX);
    for (uint32_t i = 0; i < count; ++i) {
        assert(flat[i].escape == UINT32_MAX || flat[i].escape > i);
        if ((flat[i].meta & 1u) == 0) assert(flat[i].right > i);
    }
    SDL_free(flat);
    bvh_destroy_mesh(&mesh, NULL);
}

static void test_bvh_degenerate_builds(void)
{
    MESH empty = {0};
    assert(bvh_build_mesh(&empty, 4));
    assert(empty.blas == NULL);

    VERTEX vertices[3] = {
        {{0,0,0},{0,0,1},{0}},
        {{1,0,0},{0,0,1},{0}},
        {{0,1,0},{0,0,1},{0}},
    };
    uint32_t one_indices[3] = {0,1,2};
    MESH one = {
        .vertices = vertices,
        .vertex_count = 3,
        .indices = one_indices,
        .index_count = 3,
        .geometry_revision = 1,
    };
    assert(bvh_build_mesh(&one, 1));
    assert(one.blas && one.blas->node_count == 1);
    assert(bvh_validate(one.blas));
    bvh_destroy_mesh(&one, NULL);

    uint32_t same_centroid_indices[12] = {
        0,1,2, 0,1,2, 0,1,2, 0,1,2
    };
    MESH same_centroid = {
        .vertices = vertices,
        .vertex_count = 3,
        .indices = same_centroid_indices,
        .index_count = 12,
        .geometry_revision = 1,
    };
    assert(bvh_build_mesh(&same_centroid, 1));
    assert(same_centroid.blas);
    assert(bvh_validate(same_centroid.blas));

    AABB debug_bounds[8];
    assert(bvh_debug_bounds_at_depth(same_centroid.blas, 0, debug_bounds, 8) == 1);
    assert(bvh_debug_bounds_at_depth(same_centroid.blas, 1, debug_bounds, 8) == 2);
    bvh_destroy_mesh(&same_centroid, NULL);
}

int main(void)
{
    test_mesh_validation_rejects_bad_index();
    test_mesh_bounds();
    test_triangle_access();
    test_world_bounds_rotate_all_corners();
    test_lightmap_corner_validation();
    test_ray_aabb_front_inside_parallel();
    test_ray_triangle_hit_and_miss();
    test_mesh_bruteforce_closest_and_any_hit();
    test_scene_nonuniform_transform_keeps_world_distance();
    test_overlap_queries();
    test_bruteforce_random_reference_is_deterministic();
    test_blas_matches_bruteforce();
    test_tlas_matches_scene_bruteforce_and_refits();
    test_flattened_escape_indices();
    test_bvh_degenerate_builds();
    puts("geometry/query tests ok");
    return 0;
}
