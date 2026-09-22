#ifndef COL_H
#define COL_H

#include "msh.c"

#include <math.h>
#include <stdint.h>

#define RAY_EPSILON 1e-4f
#define RAY_TIE_EPSILON 1e-5f

typedef struct Ray {
    VEC3 origin;
    VEC3 direction;
    float t_min;
    float t_max;
} RAY;

typedef struct RayHit {
    int hit;
    float t;
    VEC3 position;
    VEC3 geometric_normal;
    float bary_u;
    float bary_v;
    uint32_t triangle_index;
    uint32_t instance_index;
} RAY_HIT;

typedef struct QueryFilter {
    uint32_t mask;
    int include_static;
    int include_dynamic;
    int cull_backfaces;
} QUERY_FILTER;

typedef struct Scene {
    MESH *meshes;
    uint32_t mesh_count;
    RENDER_INSTANCE *instances;
    uint32_t instance_count;
    struct Bvh *tlas;
    struct SceneGpuAccel *gpu_accel;
    uint32_t transform_revision;
    uint32_t geometry_revision;
} SCENE;

static VEC3 c_add(VEC3 a, VEC3 b)
{
    return (VEC3){a.x + b.x, a.y + b.y, a.z + b.z};
}

static VEC3 c_sub(VEC3 a, VEC3 b)
{
    return (VEC3){a.x - b.x, a.y - b.y, a.z - b.z};
}

static VEC3 c_mul(VEC3 a, float s)
{
    return (VEC3){a.x * s, a.y * s, a.z * s};
}

static float c_dot(VEC3 a, VEC3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

static VEC3 c_cross(VEC3 a, VEC3 b)
{
    return (VEC3){
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}

static float c_len2(VEC3 v)
{
    return c_dot(v, v);
}

static float c_len(VEC3 v)
{
    return sqrtf(c_len2(v));
}

static VEC3 c_norm(VEC3 v)
{
    float len2 = c_len2(v);
    if (len2 <= 1e-20f) return (VEC3){0, 0, 0};
    return c_mul(v, 1.0f / sqrtf(len2));
}

static float c_component(VEC3 v, int axis)
{
    return axis == 0 ? v.x : (axis == 1 ? v.y : v.z);
}

static int c_finite3(VEC3 v)
{
    return isfinite(v.x) && isfinite(v.y) && isfinite(v.z);
}

int ray_make(VEC3 origin, VEC3 direction, float t_min, float t_max, RAY *out)
{
    if (!out || !c_finite3(origin) || !c_finite3(direction)) return 0;
    if (!isfinite(t_min) || isnan(t_max) || t_max < t_min) return 0;
    float len2 = c_len2(direction);
    if (!isfinite(len2) || len2 <= 1e-20f) return 0;

    out->origin = origin;
    out->direction = c_mul(direction, 1.0f / sqrtf(len2));
    out->t_min = t_min;
    out->t_max = t_max;
    return 1;
}

VEC3 ray_offset_origin(VEC3 position, VEC3 geometric_normal, VEC3 direction, float epsilon)
{
    VEC3 n = c_norm(geometric_normal);
    if (c_dot(n, direction) < 0.0f) n = c_mul(n, -1.0f);
    return c_add(position, c_mul(n, epsilon));
}

int ray_aabb(RAY ray, AABB box, float *out_near)
{
    float tmin = ray.t_min;
    float tmax = ray.t_max;

    for (int axis = 0; axis < 3; ++axis) {
        float o = c_component(ray.origin, axis);
        float d = c_component(ray.direction, axis);
        float mn = c_component(box.min, axis);
        float mx = c_component(box.max, axis);

        if (fabsf(d) < 1e-8f) {
            if (o < mn || o > mx) return 0;
            continue;
        }

        float inv = 1.0f / d;
        float a = (mn - o) * inv;
        float b = (mx - o) * inv;
        if (a > b) {
            float tmp = a;
            a = b;
            b = tmp;
        }

        tmin = SDL_max(tmin, a);
        tmax = SDL_min(tmax, b);
        if (tmin > tmax) return 0;
    }

    if (out_near) *out_near = tmin;
    return 1;
}

static int ray_triangle_ex(
    RAY ray,
    TRIANGLE tri,
    int cull_backfaces,
    float orientation_sign,
    float *out_t,
    float *out_u,
    float *out_v)
{
    const float eps = 1e-7f;
    VEC3 e1 = c_sub(tri.b, tri.a);
    VEC3 e2 = c_sub(tri.c, tri.a);
    VEC3 p = c_cross(ray.direction, e2);
    float det = c_dot(e1, p);

    if (cull_backfaces) {
        if (det * orientation_sign <= eps) return 0;
    } else if (fabsf(det) < eps) {
        return 0;
    }

    float inv_det = 1.0f / det;
    VEC3 s = c_sub(ray.origin, tri.a);
    float u = c_dot(s, p) * inv_det;
    if (u < 0.0f || u > 1.0f) return 0;

    VEC3 q = c_cross(s, e1);
    float v = c_dot(ray.direction, q) * inv_det;
    if (v < 0.0f || u + v > 1.0f) return 0;

    float t = c_dot(e2, q) * inv_det;
    if (t < ray.t_min || t > ray.t_max) return 0;

    if (out_t) *out_t = t;
    if (out_u) *out_u = u;
    if (out_v) *out_v = v;
    return 1;
}

int ray_triangle(RAY ray, TRIANGLE tri, float *out_t, float *out_u, float *out_v)
{
    return ray_triangle_ex(ray, tri, 0, 1.0f, out_t, out_u, out_v);
}

VEC3 mesh_shading_normal(const MESH *mesh, uint32_t triangle_index, float u, float v)
{
    if (!mesh || triangle_index >= mesh->index_count / 3u) return (VEC3){0, 0, 0};
    uint32_t base = triangle_index * 3u;
    const VERTEX *a = &mesh->vertices[mesh->indices[base + 0u]];
    const VERTEX *b = &mesh->vertices[mesh->indices[base + 1u]];
    const VERTEX *c = &mesh->vertices[mesh->indices[base + 2u]];
    float w = 1.0f - u - v;
    VEC3 n = {
        a->normal[0] * w + b->normal[0] * u + c->normal[0] * v,
        a->normal[1] * w + b->normal[1] * u + c->normal[1] * v,
        a->normal[2] * w + b->normal[2] * u + c->normal[2] * v,
    };
    return c_norm(n);
}

static RAY_HIT mesh_raycast_internal(const MESH *mesh, RAY ray, int cull_backfaces, float orientation_sign)
{
    RAY_HIT best = {0};
    best.t = ray.t_max;
    best.triangle_index = UINT32_MAX;
    best.instance_index = UINT32_MAX;

    if (!mesh || !mesh->vertices || !mesh->indices) return best;

    uint32_t triangle_count = mesh->index_count / 3u;
    for (uint32_t i = 0; i < triangle_count; ++i) {
        TRIANGLE tri;
        if (!m_triangle(mesh, i, &tri)) continue;

        RAY clipped = ray;
        clipped.t_max = best.hit ? SDL_min(ray.t_max, best.t + RAY_TIE_EPSILON) : best.t;
        float t, u, v;
        if (!ray_triangle_ex(clipped, tri, cull_backfaces, orientation_sign, &t, &u, &v)) continue;

        if (!best.hit || t < best.t - RAY_TIE_EPSILON ||
            (fabsf(t - best.t) <= RAY_TIE_EPSILON && i < best.triangle_index)) {
            best.hit = 1;
            best.t = t;
            best.triangle_index = i;
            best.bary_u = u;
            best.bary_v = v;
            best.position = c_add(ray.origin, c_mul(ray.direction, t));
            best.geometric_normal = m_triangle_normal(tri);
        }
    }

    return best;
}

RAY_HIT mesh_raycast_bruteforce(const MESH *mesh, RAY ray, int cull_backfaces)
{
    return mesh_raycast_internal(mesh, ray, cull_backfaces, 1.0f);
}

int mesh_occluded_bruteforce(const MESH *mesh, RAY ray, int cull_backfaces)
{
    if (!mesh || !mesh->vertices || !mesh->indices) return 0;
    uint32_t triangle_count = mesh->index_count / 3u;
    for (uint32_t i = 0; i < triangle_count; ++i) {
        TRIANGLE tri;
        float t, u, v;
        if (m_triangle(mesh, i, &tri) &&
            ray_triangle_ex(ray, tri, cull_backfaces, 1.0f, &t, &u, &v)) {
            return 1;
        }
    }
    return 0;
}

static float mat4_affine_determinant(const float m[16])
{
    float a00 = m[0], a01 = m[4], a02 = m[8];
    float a10 = m[1], a11 = m[5], a12 = m[9];
    float a20 = m[2], a21 = m[6], a22 = m[10];
    return
        a00 * (a11 * a22 - a12 * a21) -
        a01 * (a10 * a22 - a12 * a20) +
        a02 * (a10 * a21 - a11 * a20);
}

static int mat4_inverse_affine(const float m[16], float out[16], float *out_det)
{
    float a00 = m[0], a01 = m[4], a02 = m[8];
    float a10 = m[1], a11 = m[5], a12 = m[9];
    float a20 = m[2], a21 = m[6], a22 = m[10];

    float det = mat4_affine_determinant(m);
    if (!isfinite(det) || fabsf(det) < 1e-12f) return 0;
    float inv_det = 1.0f / det;

    float r00 = (a11 * a22 - a12 * a21) * inv_det;
    float r01 = (a02 * a21 - a01 * a22) * inv_det;
    float r02 = (a01 * a12 - a02 * a11) * inv_det;
    float r10 = (a12 * a20 - a10 * a22) * inv_det;
    float r11 = (a00 * a22 - a02 * a20) * inv_det;
    float r12 = (a02 * a10 - a00 * a12) * inv_det;
    float r20 = (a10 * a21 - a11 * a20) * inv_det;
    float r21 = (a01 * a20 - a00 * a21) * inv_det;
    float r22 = (a00 * a11 - a01 * a10) * inv_det;

    out[0] = r00; out[4] = r01; out[8] = r02;  out[3] = 0.0f;
    out[1] = r10; out[5] = r11; out[9] = r12;  out[7] = 0.0f;
    out[2] = r20; out[6] = r21; out[10] = r22; out[11] = 0.0f;
    out[15] = 1.0f;

    float tx = m[12], ty = m[13], tz = m[14];
    out[12] = -(r00 * tx + r01 * ty + r02 * tz);
    out[13] = -(r10 * tx + r11 * ty + r12 * tz);
    out[14] = -(r20 * tx + r21 * ty + r22 * tz);

    if (out_det) *out_det = det;
    return 1;
}

static VEC3 transform_point(const float m[16], VEC3 p)
{
    return (VEC3){
        m[0] * p.x + m[4] * p.y + m[8] * p.z + m[12],
        m[1] * p.x + m[5] * p.y + m[9] * p.z + m[13],
        m[2] * p.x + m[6] * p.y + m[10] * p.z + m[14]
    };
}

static VEC3 transform_direction(const float m[16], VEC3 v)
{
    return (VEC3){
        m[0] * v.x + m[4] * v.y + m[8] * v.z,
        m[1] * v.x + m[5] * v.y + m[9] * v.z,
        m[2] * v.x + m[6] * v.y + m[10] * v.z
    };
}

static VEC3 transform_normal_inverse_transpose(const float inverse[16], VEC3 n, float orientation_sign)
{
    VEC3 transformed = {
        inverse[0] * n.x + inverse[1] * n.y + inverse[2] * n.z,
        inverse[4] * n.x + inverse[5] * n.y + inverse[6] * n.z,
        inverse[8] * n.x + inverse[9] * n.y + inverse[10] * n.z
    };
    return c_mul(c_norm(transformed), orientation_sign);
}

int scene_instance_update(RENDER_INSTANCE *instance, const MESH *mesh)
{
    if (!instance || !mesh) return 0;
    float det;
    if (!mat4_inverse_affine(instance->transform.matrix, instance->world_inverse, &det)) return 0;
    instance->orientation_sign = det < 0.0f ? -1.0f : 1.0f;
    if (!instance->query_mask) instance->query_mask = 0xffffffffu;
    return m_world_bounds(mesh->local_bounds, instance->transform.matrix, &instance->world_bounds);
}

QUERY_FILTER query_filter_all(void)
{
    return (QUERY_FILTER){
        .mask = 0xffffffffu,
        .include_static = 1,
        .include_dynamic = 1,
        .cull_backfaces = 0
    };
}

static int query_filter_instance(const QUERY_FILTER *filter, const RENDER_INSTANCE *instance)
{
    if ((filter->mask & instance->query_mask) == 0) return 0;
    if (instance->mobility == MOBILITY_STATIC && !filter->include_static) return 0;
    if (instance->mobility == MOBILITY_DYNAMIC && !filter->include_dynamic) return 0;
    return 1;
}

RAY_HIT scene_raycast_bruteforce(const SCENE *scene, RAY ray, QUERY_FILTER filter)
{
    RAY_HIT best = {0};
    best.t = ray.t_max;
    best.triangle_index = UINT32_MAX;
    best.instance_index = UINT32_MAX;
    if (!scene || !scene->meshes || !scene->instances) return best;

    for (uint32_t i = 0; i < scene->instance_count; ++i) {
        const RENDER_INSTANCE *instance = &scene->instances[i];
        if (instance->mesh >= scene->mesh_count || !query_filter_instance(&filter, instance)) continue;

        RAY broad = ray;
        broad.t_max = best.hit ? SDL_min(ray.t_max, best.t + RAY_TIE_EPSILON) : best.t;
        if (!ray_aabb(broad, instance->world_bounds, NULL)) continue;

        RAY local = {
            .origin = transform_point(instance->world_inverse, ray.origin),
            .direction = transform_direction(instance->world_inverse, ray.direction),
            .t_min = ray.t_min,
            .t_max = best.hit ? SDL_min(ray.t_max, best.t + RAY_TIE_EPSILON) : best.t
        };
        if (c_len2(local.direction) <= 1e-20f) continue;

        const MESH *mesh = &scene->meshes[instance->mesh];
        RAY_HIT local_hit = mesh_raycast_internal(
            mesh, local, filter.cull_backfaces, instance->orientation_sign);
        if (!local_hit.hit) continue;

        VEC3 world_position = transform_point(instance->transform.matrix, local_hit.position);
        float world_t = c_len(c_sub(world_position, ray.origin));
        if (world_t < ray.t_min || world_t > best.t + RAY_TIE_EPSILON) continue;

        int better = !best.hit || world_t < best.t - RAY_TIE_EPSILON ||
            (fabsf(world_t - best.t) <= RAY_TIE_EPSILON &&
                (i < best.instance_index ||
                 (i == best.instance_index && local_hit.triangle_index < best.triangle_index)));
        if (!better) continue;

        best = local_hit;
        best.t = world_t;
        best.position = world_position;
        best.geometric_normal = transform_normal_inverse_transpose(
            instance->world_inverse, local_hit.geometric_normal, instance->orientation_sign);
        best.instance_index = i;
    }

    return best;
}

int scene_occluded_bruteforce(const SCENE *scene, RAY ray, QUERY_FILTER filter)
{
    if (!scene || !scene->meshes || !scene->instances) return 0;

    for (uint32_t i = 0; i < scene->instance_count; ++i) {
        const RENDER_INSTANCE *instance = &scene->instances[i];
        if (instance->mesh >= scene->mesh_count || !query_filter_instance(&filter, instance)) continue;
        if (!ray_aabb(ray, instance->world_bounds, NULL)) continue;

        RAY local = {
            .origin = transform_point(instance->world_inverse, ray.origin),
            .direction = transform_direction(instance->world_inverse, ray.direction),
            .t_min = ray.t_min,
            .t_max = ray.t_max
        };
        const MESH *mesh = &scene->meshes[instance->mesh];
        uint32_t triangle_count = mesh->index_count / 3u;
        for (uint32_t tri_index = 0; tri_index < triangle_count; ++tri_index) {
            TRIANGLE tri;
            float t, u, v;
            if (m_triangle(mesh, tri_index, &tri) &&
                ray_triangle_ex(local, tri, filter.cull_backfaces, instance->orientation_sign, &t, &u, &v)) {
                VEC3 local_pos = c_add(local.origin, c_mul(local.direction, t));
                VEC3 world_pos = transform_point(instance->transform.matrix, local_pos);
                float world_t = c_len(c_sub(world_pos, ray.origin));
                if (world_t >= ray.t_min && world_t <= ray.t_max) return 1;
            }
        }
    }
    return 0;
}

int aabb_overlap(AABB a, AABB b)
{
    return
        a.min.x <= b.max.x && a.max.x >= b.min.x &&
        a.min.y <= b.max.y && a.max.y >= b.min.y &&
        a.min.z <= b.max.z && a.max.z >= b.min.z;
}

int sphere_aabb_overlap(VEC3 center, float radius, AABB box)
{
    if (radius < 0.0f || !isfinite(radius) || !c_finite3(center)) return 0;
    float d2 = 0.0f;
    for (int axis = 0; axis < 3; ++axis) {
        float c = c_component(center, axis);
        float mn = c_component(box.min, axis);
        float mx = c_component(box.max, axis);
        float q = SDL_clamp(c, mn, mx);
        float d = c - q;
        d2 += d * d;
    }
    return d2 <= radius * radius;
}

#endif // COL_H
