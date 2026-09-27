#include "game.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define LPV_MAX_AXIS 24u
#define LPV_PROPAGATION_STEPS 6u
#define LPV_BUFFER_COUNT 3u
#define LPV_EMITTER_SAMPLES 4u
#define LPV_PROPAGATION_GAIN 0.12f
#define LPV_SURFACE_BOUNCE 0.62f
#define LPV_PI 3.14159265358979323846f
#define LPV_Y00 0.2820947918f

typedef struct LPV_SURFACE_CELL {
    VEC3 position;
    VEC3 normal;
    VEC3 albedo;
    float weight;
} LPV_SURFACE_CELL;

typedef struct LPV_STATE {
    RENDERER *owner;
    const MESH *scene;
    const GLTF_SCENE *visual;
    BVH static_bvh;

    VEC3 origin;
    float spacing;
    uint32_t count_x, count_y, count_z, count;

    LPV_SURFACE_CELL *geometry;
    unsigned char *dynamic_occupied;
    VEC3 *front;
    VEC3 *next;
    VEC3 *total;
    PROBE *probes;

    NriBuffer *buffers[LPV_BUFFER_COUNT];
    NriBuffer *beam_buffer;
    uint32_t write_index;

    NriBuffer *fallback_probe_buffer;
    NriBuffer *fallback_beam_buffer;
    PROBE_GRID fallback_probes;
    BEAM_GRID fallback_beams;
    bool fallback_has_bake;

    uint32_t frame_counter;
} LPV_STATE;

static LPV_STATE *g_lpv;

static VEC3 lpv_add(VEC3 a, VEC3 b) {
    return v3(a.x + b.x, a.y + b.y, a.z + b.z);
}

static VEC3 lpv_mul(VEC3 a, VEC3 b) {
    return v3(a.x * b.x, a.y * b.y, a.z * b.z);
}

static VEC3 lpv_scale(VEC3 a, float s) {
    return v3(a.x * s, a.y * s, a.z * s);
}

static float lpv_length(VEC3 v) {
    return sqrtf(fmaxf(v3_len_sq(v), 0.0f));
}

static float lpv_max_component(VEC3 v) {
    return fmaxf(v.x, fmaxf(v.y, v.z));
}

static VEC3 lpv_matrix_point(const float m[16], VEC3 p) {
    return v3(
        m[0] * p.x + m[4] * p.y + m[8] * p.z + m[12],
        m[1] * p.x + m[5] * p.y + m[9] * p.z + m[13],
        m[2] * p.x + m[6] * p.y + m[10] * p.z + m[14]
    );
}

static uint32_t lpv_index(const LPV_STATE *s, uint32_t x, uint32_t y, uint32_t z) {
    return x + s->count_x * (y + s->count_y * z);
}

static bool lpv_world_cell(const LPV_STATE *s, VEC3 p, uint32_t *x, uint32_t *y, uint32_t *z) {
    if (!s || s->spacing <= 0.0f) return false;

    const float fx = (p.x - s->origin.x) / s->spacing;
    const float fy = (p.y - s->origin.y) / s->spacing;
    const float fz = (p.z - s->origin.z) / s->spacing;

    if (fx < -0.5f || fy < -0.5f || fz < -0.5f || fx > (float)s->count_x - 0.5f || fy > (float)s->count_y - 0.5f || fz > (float)s->count_z - 0.5f) return false;

    *x = (uint32_t)fminf((float)s->count_x - 1.0f, fmaxf(0.0f, floorf(fx + 0.5f)));
    *y = (uint32_t)fminf((float)s->count_y - 1.0f, fmaxf(0.0f, floorf(fy + 0.5f)));
    *z = (uint32_t)fminf((float)s->count_z - 1.0f, fmaxf(0.0f, floorf(fz + 0.5f)));

    return true;
}

static VEC3 lpv_cell_position(const LPV_STATE *s, uint32_t x, uint32_t y, uint32_t z) {
    return v3(
        s->origin.x + (float)x * s->spacing,
        s->origin.y + (float)y * s->spacing,
        s->origin.z + (float)z * s->spacing
    );
}

static bool lpv_ray_box(TRACE_RAY ray, AABB bounds) {
    float lo = ray.tmin;
    float hi = ray.tmax;
    const float origin[3] = {ray.origin.x, ray.origin.y, ray.origin.z};
    const float direction[3] = {ray.direction.x, ray.direction.y, ray.direction.z};
    const float bmin[3] = {bounds.min.x, bounds.min.y, bounds.min.z};
    const float bmax[3] = {bounds.max.x, bounds.max.y, bounds.max.z};

    for (uint32_t axis = 0u; axis < 3u; ++axis) {
        if (fabsf(direction[axis]) < 1.0e-7f) {
            if (origin[axis] < bmin[axis] || origin[axis] > bmax[axis]) return false;
        } else {
            float a = (bmin[axis] - origin[axis]) / direction[axis];
            float b = (bmax[axis] - origin[axis]) / direction[axis];

            if (a > b) {
                const float temporary = a;
                a = b;
                b = temporary;
            }

            lo = fmaxf(lo, a);
            hi = fminf(hi, b);

            if (lo > hi) return false;
        }
    }

    return hi >= ray.tmin;
}

static bool lpv_ray_triangle(VEC3 origin, VEC3 direction, float maximum, VEC3 a, VEC3 b, VEC3 c) {
    const VEC3 edge1 = v3_sub(b, a);
    const VEC3 edge2 = v3_sub(c, a);
    const VEC3 p = v3_cross(direction, edge2);
    const float determinant = v3_dot(edge1, p);

    if (fabsf(determinant) < 1.0e-7f) return false;

    const float inverse = 1.0f / determinant;
    const VEC3 s = v3_sub(origin, a);
    const float u = v3_dot(s, p) * inverse;

    if (u < 0.0f || u > 1.0f) return false;

    const VEC3 q = v3_cross(s, edge1);
    const float v = v3_dot(direction, q) * inverse;

    if (v < 0.0f || u + v > 1.0f) return false;

    const float t = v3_dot(edge2, q) * inverse;

    return t > 1.0e-4f && t < maximum;
}

static bool lpv_dynamic_any(RENDERER *r, TRACE_RAY ray) {
    const uint32_t count = dynamic_instance_count(r);

    for (uint32_t instance_index = 0u; instance_index < count; ++instance_index) {
        DYNAMIC_INSTANCE_DATA instance = {0};

        if (!dynamic_instance_data(r, instance_index, &instance) || !instance.model || !instance.model->visual) continue;
        if (!lpv_ray_box(ray, instance.world_bounds)) continue;

        const VEC3 world_end = v3_add(ray.origin, v3_scale(ray.direction, ray.tmax));
        const VEC3 local_origin = lpv_matrix_point(instance.inverse_world, ray.origin);
        const VEC3 local_end = lpv_matrix_point(instance.inverse_world, world_end);
        const VEC3 local_delta = v3_sub(local_end, local_origin);
        const float local_distance = lpv_length(local_delta);

        if (local_distance <= 1.0e-5f) continue;

        const VEC3 local_direction = v3_scale(local_delta, 1.0f / local_distance);
        const GLTF_SCENE *visual = instance.model->visual;

        for (size_t vertex = 0u; vertex + 2u < visual->vertex_count; vertex += 3u) {
            if (lpv_ray_triangle(
                    local_origin,
                    local_direction,
                    local_distance - 1.0e-4f,
                    visual->vertices[vertex + 0u].position,
                    visual->vertices[vertex + 1u].position,
                    visual->vertices[vertex + 2u].position
                ))
                return true;
        }
    }

    return false;
}

static VEC3 lpv_triangle_sample(VEC3 a, VEC3 b, VEC3 c, float u, float v) {
    const float root = sqrtf(fmaxf(u, 0.0f));
    const float wa = 1.0f - root;
    const float wb = root * (1.0f - v);
    const float wc = root * v;

    return v3(
        a.x * wa + b.x * wb + c.x * wc,
        a.y * wa + b.y * wb + c.y * wc,
        a.z * wa + b.z * wb + c.z * wc
    );
}

static bool lpv_visible(RENDERER *r, LPV_STATE *s, VEC3 position, VEC3 normal, VEC3 target) {
    VEC3 delta = v3_sub(target, position);
    const float distance = lpv_length(delta);

    if (distance <= 1.0e-4f) return false;

    delta = v3_scale(delta, 1.0f / distance);
    const float epsilon = fmaxf(1.0e-4f, s->spacing * 0.002f);
    TRACE_RAY ray = {
        .origin = v3_add(position, v3_scale(normal, epsilon)),
        .tmin = epsilon,
        .direction = delta,
        .tmax = fmaxf(epsilon, distance - epsilon * 2.0f)
    };

    if (ray.tmax <= ray.tmin) return false;
    if (trace_any(&s->static_bvh, ray)) return false;
    if (lpv_dynamic_any(r, ray)) return false;

    return true;
}

static bool lpv_create_buffer(RENDERER *r, NriBuffer **out, uint64_t size, uint32_t stride) {
    const NriBufferDesc desc = {
        .size = size,
        .structureStride = stride,
        .usage = NriBufferUsageBits_SHADER_RESOURCE
    };

    return r->core.CreateCommittedBuffer(r->device, NriMemoryLocation_HOST_UPLOAD, 0.0f, &desc, out) == NriResult_SUCCESS;
}

static bool lpv_upload_buffer(RENDERER *r, NriBuffer *buffer, const void *data, uint64_t size) {
    void *mapped = r->core.MapBuffer(buffer, 0u, size);

    if (!mapped) return false;
    memcpy(mapped, data, (size_t)size);
    r->core.UnmapBuffer(buffer);

    return true;
}

static VEC3 lpv_material_albedo(const GLTF_SCENE *visual, uint32_t material) {
    if (!visual || !visual->material_count) return v3(0.8f, 0.8f, 0.8f);

    if (material >= visual->material_count) material = visual->default_material < visual->material_count ? visual->default_material : 0u;

    const GLTF_MATERIAL *m = &visual->materials[material];

    return v3(
        fmaxf(m->base_color[0], 0.0f),
        fmaxf(m->base_color[1], 0.0f),
        fmaxf(m->base_color[2], 0.0f)
    );
}

static bool lpv_build_geometry(LPV_STATE *s) {
    if (!s || !s->visual || !s->geometry) return false;

    for (size_t vertex = 0u; vertex + 2u < s->visual->vertex_count; vertex += 3u) {
        const GLTF_VERTEX *a = &s->visual->vertices[vertex + 0u];
        const GLTF_VERTEX *b = &s->visual->vertices[vertex + 1u];
        const GLTF_VERTEX *c = &s->visual->vertices[vertex + 2u];
        VEC3 normal = v3_add(v3_add(a->normal, b->normal), c->normal);

        if (v3_len_sq(normal) < 1.0e-8f) normal = v3_cross(v3_sub(b->position, a->position), v3_sub(c->position, a->position));
        normal = v3_normalize(normal);

        const VEC3 albedo = lpv_material_albedo(s->visual, a->material);
        const float edge0 = lpv_length(v3_sub(b->position, a->position));
        const float edge1 = lpv_length(v3_sub(c->position, b->position));
        const float edge2 = lpv_length(v3_sub(a->position, c->position));
        uint32_t steps = (uint32_t)ceilf(fmaxf(edge0, fmaxf(edge1, edge2)) / s->spacing * 2.0f);

        if (steps < 1u) steps = 1u;
        if (steps > 64u) steps = 64u;

        for (uint32_t i = 0u; i <= steps; ++i) {
            for (uint32_t j = 0u; j + i <= steps; ++j) {
                const float u = (float)i / (float)steps;
                const float v = (float)j / (float)steps;
                const float w = fmaxf(0.0f, 1.0f - u - v);
                const VEC3 position = v3(
                    a->position.x * w + b->position.x * u + c->position.x * v,
                    a->position.y * w + b->position.y * u + c->position.y * v,
                    a->position.z * w + b->position.z * u + c->position.z * v
                );
                uint32_t x, y, z;

                if (!lpv_world_cell(s, position, &x, &y, &z)) continue;

                LPV_SURFACE_CELL *cell = &s->geometry[lpv_index(s, x, y, z)];
                cell->position = lpv_add(cell->position, position);
                cell->normal = lpv_add(cell->normal, normal);
                cell->albedo = lpv_add(cell->albedo, albedo);
                cell->weight += 1.0f;
            }
        }
    }

    for (uint32_t i = 0u; i < s->count; ++i) {
        LPV_SURFACE_CELL *cell = &s->geometry[i];

        if (cell->weight <= 0.0f) continue;

        const float inverse = 1.0f / cell->weight;
        cell->position = lpv_scale(cell->position, inverse);
        cell->normal = v3_normalize(cell->normal);
        cell->albedo = lpv_scale(cell->albedo, inverse);
    }

    return true;
}

static void lpv_dynamic_occupancy(RENDERER *r, LPV_STATE *s) {
    memset(s->dynamic_occupied, 0, s->count);

    const uint32_t count = dynamic_instance_count(r);

    for (uint32_t instance_index = 0u; instance_index < count; ++instance_index) {
        DYNAMIC_INSTANCE_DATA instance = {0};

        if (!dynamic_instance_data(r, instance_index, &instance)) continue;

        int min_x = (int)floorf((instance.world_bounds.min.x - s->origin.x) / s->spacing) - 1;
        int min_y = (int)floorf((instance.world_bounds.min.y - s->origin.y) / s->spacing) - 1;
        int min_z = (int)floorf((instance.world_bounds.min.z - s->origin.z) / s->spacing) - 1;
        int max_x = (int)ceilf((instance.world_bounds.max.x - s->origin.x) / s->spacing) + 1;
        int max_y = (int)ceilf((instance.world_bounds.max.y - s->origin.y) / s->spacing) + 1;
        int max_z = (int)ceilf((instance.world_bounds.max.z - s->origin.z) / s->spacing) + 1;

        min_x = min_x < 0 ? 0 : min_x;
        min_y = min_y < 0 ? 0 : min_y;
        min_z = min_z < 0 ? 0 : min_z;
        max_x = max_x >= (int)s->count_x ? (int)s->count_x - 1 : max_x;
        max_y = max_y >= (int)s->count_y ? (int)s->count_y - 1 : max_y;
        max_z = max_z >= (int)s->count_z ? (int)s->count_z - 1 : max_z;

        for (int z = min_z; z <= max_z; ++z) {
            for (int y = min_y; y <= max_y; ++y) {
                for (int x = min_x; x <= max_x; ++x) {
                    const VEC3 position = lpv_cell_position(s, (uint32_t)x, (uint32_t)y, (uint32_t)z);

                    if (position.x < instance.world_bounds.min.x || position.x > instance.world_bounds.max.x ||
                        position.y < instance.world_bounds.min.y || position.y > instance.world_bounds.max.y ||
                        position.z < instance.world_bounds.min.z || position.z > instance.world_bounds.max.z)
                        continue;

                    s->dynamic_occupied[lpv_index(s, (uint32_t)x, (uint32_t)y, (uint32_t)z)] = 1u;
                }
            }
        }
    }
}

static VEC3 lpv_emissive_irradiance(RENDERER *r, LPV_STATE *s, const LPV_SURFACE_CELL *surface) {
    static const float samples[LPV_EMITTER_SAMPLES][2] = {
        {0.125f, 0.125f},
        {0.625f, 0.125f},
        {0.125f, 0.625f},
        {0.625f, 0.625f}
    };

    VEC3 irradiance = {0.0f, 0.0f, 0.0f};
    float previous_importance = 0.0f;

    for (uint32_t triangle = 0u; triangle < s->static_bvh.triangle_count; ++triangle) {
        const BVH_TRIANGLE *t = &s->static_bvh.triangles[triangle];
        const float importance = t->emissive[3];
        const bool emitter = importance > previous_importance + 1.0e-8f &&
            (t->emissive[0] > 0.0f || t->emissive[1] > 0.0f || t->emissive[2] > 0.0f);

        previous_importance = importance;

        if (!emitter) continue;

        const VEC3 a = v3(t->a[0], t->a[1], t->a[2]);
        const VEC3 b = v3(t->b[0], t->b[1], t->b[2]);
        const VEC3 c = v3(t->c[0], t->c[1], t->c[2]);
        const VEC3 edge1 = v3_sub(b, a);
        const VEC3 edge2 = v3_sub(c, a);
        const float area = 0.5f * lpv_length(v3_cross(edge1, edge2));
        const VEC3 light_normal = v3_normalize(v3(t->normal[0], t->normal[1], t->normal[2]));
        const VEC3 radiance = v3(t->emissive[0], t->emissive[1], t->emissive[2]);

        if (area <= 1.0e-8f || lpv_max_component(radiance) <= 0.0f) continue;

        for (uint32_t sample = 0u; sample < LPV_EMITTER_SAMPLES; ++sample) {
            const VEC3 light_position = lpv_triangle_sample(a, b, c, samples[sample][0], samples[sample][1]);
            VEC3 delta = v3_sub(light_position, surface->position);
            const float distance2 = v3_len_sq(delta);

            if (distance2 <= 1.0e-8f) continue;

            const float distance = sqrtf(distance2);
            delta = v3_scale(delta, 1.0f / distance);
            const float receiver_cosine = fmaxf(v3_dot(surface->normal, delta), 0.0f);
            const float light_cosine = fmaxf(v3_dot(light_normal, v3_scale(delta, -1.0f)), 0.0f);

            if (receiver_cosine <= 0.0f || light_cosine <= 0.0f) continue;
            if (!lpv_visible(r, s, surface->position, surface->normal, light_position)) continue;

            const float factor = area * receiver_cosine * light_cosine /
                (distance2 * (float)LPV_EMITTER_SAMPLES);
            irradiance = lpv_add(irradiance, lpv_scale(radiance, factor));
        }
    }

    return irradiance;
}

static VEC3 lpv_sun_irradiance(RENDERER *r, LPV_STATE *s, const LPV_SURFACE_CELL *surface, const struct LIGHT *light) {
    if (!light || light->type != LIGHT_DIRECTIONAL || light->directional.intensity <= 0.0f) return v3(0.0f, 0.0f, 0.0f);

    const VEC3 direction = v3_normalize(light->directional.direction);
    const float cosine = fmaxf(v3_dot(surface->normal, direction), 0.0f);

    if (cosine <= 0.0f) return v3(0.0f, 0.0f, 0.0f);

    const float epsilon = fmaxf(1.0e-4f, s->spacing * 0.002f);
    TRACE_RAY ray = {
        .origin = v3_add(surface->position, v3_scale(surface->normal, epsilon)),
        .tmin = epsilon,
        .direction = direction,
        .tmax = fmaxf(1.0f, r->scene_radius * 4.0f)
    };

    if (trace_any(&s->static_bvh, ray) || lpv_dynamic_any(r, ray)) return v3(0.0f, 0.0f, 0.0f);

    return lpv_scale(light->directional.color, light->directional.intensity * cosine);
}

static void lpv_inject(RENDERER *r, LPV_STATE *s, const struct LIGHT *light, const SKY *sky) {
    memset(s->front, 0, (size_t)s->count * sizeof(*s->front));
    memset(s->next, 0, (size_t)s->count * sizeof(*s->next));
    memset(s->total, 0, (size_t)s->count * sizeof(*s->total));

    for (uint32_t i = 0u; i < s->count; ++i) {
        const LPV_SURFACE_CELL *surface = &s->geometry[i];

        if (surface->weight <= 0.0f || s->dynamic_occupied[i]) continue;

        VEC3 irradiance = lpv_emissive_irradiance(r, s, surface);
        irradiance = lpv_add(irradiance, lpv_sun_irradiance(r, s, surface, light));

        if (sky && sky->intensity > 0.0f) {
            const float up = fmaxf(surface->normal.y, 0.0f);
            const VEC3 sky_color = lpv_scale(lpv_add(sky->zenith, sky->horizon), 0.5f * sky->intensity * up);
            irradiance = lpv_add(irradiance, sky_color);
        }

        const VEC3 reflected_irradiance = lpv_mul(surface->albedo, irradiance);
        s->front[i] = reflected_irradiance;
        s->total[i] = reflected_irradiance;
    }
}

static VEC3 lpv_direction(int axis, int sign) {
    if (axis == 0) return v3((float)sign, 0.0f, 0.0f);
    if (axis == 1) return v3(0.0f, (float)sign, 0.0f);
    return v3(0.0f, 0.0f, (float)sign);
}

static void lpv_propagate(LPV_STATE *s) {
    for (uint32_t pass = 0u; pass < LPV_PROPAGATION_STEPS; ++pass) {
        memset(s->next, 0, (size_t)s->count * sizeof(*s->next));

        for (uint32_t z = 0u; z < s->count_z; ++z) {
            for (uint32_t y = 0u; y < s->count_y; ++y) {
                for (uint32_t x = 0u; x < s->count_x; ++x) {
                    const uint32_t source_index = lpv_index(s, x, y, z);
                    const VEC3 energy = s->front[source_index];

                    if (lpv_max_component(energy) <= 1.0e-7f || s->dynamic_occupied[source_index]) continue;

                    const LPV_SURFACE_CELL *source_surface = &s->geometry[source_index];

                    for (int axis = 0; axis < 3; ++axis) {
                        for (int sign = -1; sign <= 1; sign += 2) {
                            const int nx = (int)x + (axis == 0 ? sign : 0);
                            const int ny = (int)y + (axis == 1 ? sign : 0);
                            const int nz = (int)z + (axis == 2 ? sign : 0);

                            if (nx < 0 || ny < 0 || nz < 0 || nx >= (int)s->count_x || ny >= (int)s->count_y || nz >= (int)s->count_z) continue;

                            const uint32_t destination_index = lpv_index(s, (uint32_t)nx, (uint32_t)ny, (uint32_t)nz);

                            if (s->dynamic_occupied[destination_index]) continue;

                            const VEC3 direction = lpv_direction(axis, sign);
                            float gain = LPV_PROPAGATION_GAIN;

                            if (source_surface->weight > 0.0f) {
                                const float hemisphere = fmaxf(v3_dot(source_surface->normal, direction), 0.0f);

                                if (hemisphere <= 0.0f) continue;
                                gain *= hemisphere;
                            }

                            VEC3 transported = lpv_scale(energy, gain);
                            const LPV_SURFACE_CELL *destination_surface = &s->geometry[destination_index];

                            if (destination_surface->weight > 0.0f) {
                                const float front = fmaxf(v3_dot(destination_surface->normal, v3_scale(direction, -1.0f)), 0.0f);

                                if (front <= 0.0f) continue;
                                transported = lpv_scale(lpv_mul(transported, destination_surface->albedo), LPV_SURFACE_BOUNCE * front);
                            }

                            s->next[destination_index] = lpv_add(s->next[destination_index], transported);
                        }
                    }
                }
            }
        }

        for (uint32_t i = 0u; i < s->count; ++i)
            s->total[i] = lpv_add(s->total[i], s->next[i]);

        VEC3 *temporary = s->front;
        s->front = s->next;
        s->next = temporary;
    }
}

static void lpv_make_probes(LPV_STATE *s) {
    const float coefficient_scale = 1.0f / (LPV_Y00 * LPV_PI);

    for (uint32_t z = 0u; z < s->count_z; ++z) {
        for (uint32_t y = 0u; y < s->count_y; ++y) {
            for (uint32_t x = 0u; x < s->count_x; ++x) {
                const uint32_t index = lpv_index(s, x, y, z);
                PROBE *probe = &s->probes[index];
                memset(probe, 0, sizeof(*probe));

                const VEC3 position = lpv_cell_position(s, x, y, z);
                probe->position[0] = position.x;
                probe->position[1] = position.y;
                probe->position[2] = position.z;
                probe->position[3] = s->dynamic_occupied[index] ? 0.0f : 1.0f;
                probe->coefficients[0][0] = s->total[index].x * coefficient_scale;
                probe->coefficients[0][1] = s->total[index].y * coefficient_scale;
                probe->coefficients[0][2] = s->total[index].z * coefficient_scale;
            }
        }
    }
}

static void lpv_shadow_bounds(RENDERER *r, LPV_STATE *s, VEC3 direction) {
    VEC3 sun = v3_normalize(direction);
    VEC3 u = v3_normalize(v3_cross(v3(0.0f, 1.0f, 0.0f), sun));

    if (v3_len_sq(u) < 0.5f) u = v3_normalize(v3_cross(v3(1.0f, 0.0f, 0.0f), sun));

    const VEC3 v = v3_cross(sun, u);
    VEC3 minimum = v3(INFINITY, INFINITY, INFINITY);
    VEC3 maximum = v3(-INFINITY, -INFINITY, -INFINITY);

    for (uint32_t corner = 0u; corner < 8u; ++corner) {
        const VEC3 p = v3(
            corner & 1u ? s->scene->bounds.max.x : s->scene->bounds.min.x,
            corner & 2u ? s->scene->bounds.max.y : s->scene->bounds.min.y,
            corner & 4u ? s->scene->bounds.max.z : s->scene->bounds.min.z
        );
        const VEC3 q = v3(v3_dot(p, u), v3_dot(p, v), v3_dot(p, sun));
        minimum.x = fminf(minimum.x, q.x);
        minimum.y = fminf(minimum.y, q.y);
        minimum.z = fminf(minimum.z, q.z);
        maximum.x = fmaxf(maximum.x, q.x);
        maximum.y = fmaxf(maximum.y, q.y);
        maximum.z = fmaxf(maximum.z, q.z);
    }

    r->beams.origin = minimum;
    r->beams.step = v3(
        fmaxf(maximum.x - minimum.x, 0.001f),
        fmaxf(maximum.y - minimum.y, 0.001f),
        fmaxf(maximum.z - minimum.z, 0.001f)
    );
    r->beams.width = 1u;
    r->beams.height = 1u;
    r->beams.depth = 1u;
    r->beams.count = 0u;
    r->beams.cells = NULL;
    r->beams.shadow_depth = NULL;
}

bool lpv_init(RENDERER *r, const MESH *scene, const GLTF_SCENE *visual) {
    if (!r || !r->device || !scene || !visual || !visual->vertex_count || g_lpv) return false;

    LPV_STATE *s = calloc(1u, sizeof(*s));

    if (!s) return false;

    s->owner = r;
    s->scene = scene;
    s->visual = visual;

    const VEC3 extent = v3_sub(scene->bounds.max, scene->bounds.min);
    const float maximum_extent = fmaxf(extent.x, fmaxf(extent.y, extent.z));

    if (!isfinite(maximum_extent) || maximum_extent <= 0.0f) goto fail;

    s->spacing = maximum_extent / (float)(LPV_MAX_AXIS - 1u);
    s->origin = scene->bounds.min;
    s->count_x = (uint32_t)ceilf(extent.x / s->spacing) + 1u;
    s->count_y = (uint32_t)ceilf(extent.y / s->spacing) + 1u;
    s->count_z = (uint32_t)ceilf(extent.z / s->spacing) + 1u;

    if (!s->count_x || !s->count_y || !s->count_z || s->count_x > LPV_MAX_AXIS || s->count_y > LPV_MAX_AXIS || s->count_z > LPV_MAX_AXIS) goto fail;

    const uint64_t count64 = (uint64_t)s->count_x * s->count_y * s->count_z;

    if (!count64 || count64 > UINT32_MAX) goto fail;
    s->count = (uint32_t)count64;

    s->geometry = calloc(s->count, sizeof(*s->geometry));
    s->dynamic_occupied = calloc(s->count, sizeof(*s->dynamic_occupied));
    s->front = calloc(s->count, sizeof(*s->front));
    s->next = calloc(s->count, sizeof(*s->next));
    s->total = calloc(s->count, sizeof(*s->total));
    s->probes = calloc(s->count, sizeof(*s->probes));

    if (!s->geometry || !s->dynamic_occupied || !s->front || !s->next || !s->total || !s->probes) goto fail;
    if (!bvh_build(&s->static_bvh, scene, visual) || !lpv_build_geometry(s)) goto fail;

    const uint64_t probe_bytes = (uint64_t)s->count * sizeof(PROBE);

    for (uint32_t i = 0u; i < LPV_BUFFER_COUNT; ++i) {
        if (!lpv_create_buffer(r, &s->buffers[i], probe_bytes, sizeof(PROBE))) goto fail;
    }

    if (!lpv_create_buffer(r, &s->beam_buffer, sizeof(float), sizeof(float))) goto fail;

    const float beam = 1.0f;
    if (!lpv_upload_buffer(r, s->beam_buffer, &beam, sizeof(beam))) goto fail;

    s->fallback_probe_buffer = r->volume_probe_buffer;
    s->fallback_beam_buffer = r->beam_buffer;
    s->fallback_probes = r->volume_probes;
    s->fallback_beams = r->beams;
    s->fallback_has_bake = r->has_bake;

    g_lpv = s;
    r->volume_probes = (PROBE_GRID){
        .origin = s->origin,
        .spacing = s->spacing,
        .count_x = s->count_x,
        .count_y = s->count_y,
        .count_z = s->count_z,
        .probes = s->probes
    };
    r->volume_probe_buffer = s->buffers[0];
    r->beam_buffer = s->beam_buffer;
    r->has_bake = true;

    SDL_Log("LPV: realtime %ux%ux%u grid | %.3f world units/cell | %u same-frame propagation passes", s->count_x, s->count_y, s->count_z, s->spacing, LPV_PROPAGATION_STEPS);

    return true;

fail:
    if (s) {
        for (uint32_t i = 0u; i < LPV_BUFFER_COUNT; ++i)
            if (s->buffers[i]) r->core.DestroyBuffer(s->buffers[i]);
        if (s->beam_buffer) r->core.DestroyBuffer(s->beam_buffer);
        bvh_free(&s->static_bvh);
        free(s->geometry);
        free(s->dynamic_occupied);
        free(s->front);
        free(s->next);
        free(s->total);
        free(s->probes);
        free(s);
    }

    return false;
}

bool lpv_update(RENDERER *r, const struct LIGHT *light, const SKY *sky) {
    if (!r || !g_lpv || g_lpv->owner != r || !light || !sky) return false;

    LPV_STATE *s = g_lpv;
    const Uint64 started = SDL_GetPerformanceCounter();

    if (!dynamic_sync(r)) return false;

    lpv_dynamic_occupancy(r, s);
    lpv_inject(r, s, light, sky);
    lpv_propagate(s);
    lpv_make_probes(s);

    const uint32_t slot = s->write_index++ % LPV_BUFFER_COUNT;
    const uint64_t probe_bytes = (uint64_t)s->count * sizeof(PROBE);

    if (!lpv_upload_buffer(r, s->buffers[slot], s->probes, probe_bytes)) return false;

    r->volume_probe_buffer = s->buffers[slot];
    r->beam_buffer = s->beam_buffer;
    r->has_bake = true;
    lpv_shadow_bounds(r, s, light->directional.direction);

    if ((++s->frame_counter % 120u) == 0u) {
        const double milliseconds = (double)(SDL_GetPerformanceCounter() - started) * 1000.0 / (double)SDL_GetPerformanceFrequency();
        SDL_Log("LPV: %.3f ms CPU update | %u cells", milliseconds, s->count);
    }

    return true;
}

void lpv_deinit(RENDERER *r) {
    if (!r || !g_lpv || g_lpv->owner != r) return;

    LPV_STATE *s = g_lpv;

    r->volume_probe_buffer = s->fallback_probe_buffer;
    r->beam_buffer = s->fallback_beam_buffer;
    r->volume_probes = s->fallback_probes;
    r->beams = s->fallback_beams;
    r->has_bake = s->fallback_has_bake;

    for (uint32_t i = 0u; i < LPV_BUFFER_COUNT; ++i)
        if (s->buffers[i]) r->core.DestroyBuffer(s->buffers[i]);
    if (s->beam_buffer) r->core.DestroyBuffer(s->beam_buffer);

    bvh_free(&s->static_bvh);
    free(s->geometry);
    free(s->dynamic_occupied);
    free(s->front);
    free(s->next);
    free(s->total);
    free(s->probes);
    free(s);
    g_lpv = NULL;
}
