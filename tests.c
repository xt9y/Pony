#include <assert.h>
#include <math.h>
#include <stdio.h>

#include "msh.c"

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

int main(void)
{
    test_mesh_validation_rejects_bad_index();
    test_mesh_bounds();
    test_triangle_access();
    test_world_bounds_rotate_all_corners();
    test_lightmap_corner_validation();
    puts("mesh tests ok");
    return 0;
}
