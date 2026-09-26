#include "game.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct DYNAMIC_SAMPLE_REF {
    uint32_t x;
    uint32_t y;
    uint32_t z;
    uint32_t sample_index;
} DYNAMIC_SAMPLE_REF;

typedef struct DYNAMIC_CELL {
    uint32_t x;
    uint32_t y;
    uint32_t z;
    uint32_t first_sample;
    uint32_t sample_count;
    uint32_t generation;
} DYNAMIC_CELL;

typedef struct DYNAMIC_CELL_LOOKUP {
    uint32_t x;
    uint32_t y;
    uint32_t z;
    uint32_t cell_plus_one;
} DYNAMIC_CELL_LOOKUP;

typedef struct DYNAMIC_CELL_JOB {
    uint32_t cell_index;
    uint32_t generation;
    uint64_t next_sample;
} DYNAMIC_CELL_JOB;

typedef struct DYNAMIC_OBJECT_ENTRY {
    OBJECT *object;
    AABB world_bounds;
    float world[16];
    float inverse_world[16];
    float normal_world[16];
} DYNAMIC_OBJECT_ENTRY;

struct DYNAMIC_STATE {
    DYNAMIC_LIGHTING settings;
    VEC3 origin;
    float cell_size;
    uint32_t max_x;
    uint32_t max_y;
    uint32_t max_z;

    uint32_t *sample_ids;
    DYNAMIC_CELL *cells;
    uint32_t cell_count;
    DYNAMIC_CELL_LOOKUP *lookup;
    uint32_t lookup_capacity;

    DYNAMIC_CELL_JOB *jobs;
    uint32_t job_count;
    uint32_t job_capacity;

    DYNAMIC_CELL_UPDATE *updates;
    uint32_t update_count;
    uint32_t update_capacity;

    DYNAMIC_OBJECT_ENTRY *objects;
    uint32_t object_count;
    uint32_t object_capacity;
    uint32_t object_generation;
};

static bool finite_v3(VEC3 v) {
    return isfinite(v.x) && isfinite(v.y) && isfinite(v.z);
}

static bool transform_valid(const TRANSFORM *transform) {
    return transform && finite_v3(transform->position) && finite_v3(transform->rotation) && finite_v3(transform->scale) && transform->scale.x != 0.0f &&
           transform->scale.y != 0.0f && transform->scale.z != 0.0f;
}

static bool transform_equal(const TRANSFORM *a, const TRANSFORM *b) {
    return a->position.x == b->position.x && a->position.y == b->position.y && a->position.z == b->position.z && a->rotation.x == b->rotation.x &&
           a->rotation.y == b->rotation.y && a->rotation.z == b->rotation.z && a->scale.x == b->scale.x && a->scale.y == b->scale.y && a->scale.z == b->scale.z;
}

static VEC3 transform_point(const TRANSFORM *transform, VEC3 p) {
    p.x *= transform->scale.x;
    p.y *= transform->scale.y;
    p.z *= transform->scale.z;

    const float cx = cosf(transform->rotation.x);
    const float sx = sinf(transform->rotation.x);
    const float cy = cosf(transform->rotation.y);
    const float sy = sinf(transform->rotation.y);
    const float cz = cosf(transform->rotation.z);
    const float sz = sinf(transform->rotation.z);

    const VEC3 rx = {p.x, p.y * cx - p.z * sx, p.y * sx + p.z * cx};
    const VEC3 ry = {rx.x * cy + rx.z * sy, rx.y, -rx.x * sy + rx.z * cy};
    const VEC3 rz = {ry.x * cz - ry.y * sz, ry.x * sz + ry.y * cz, ry.z};

    return (VEC3){rz.x + transform->position.x, rz.y + transform->position.y, rz.z + transform->position.z};
}


static void matrix_identity(float out[16]) {
    memset(out, 0, 16u * sizeof(*out));
    out[0] = out[5] = out[10] = out[15] = 1.0f;
}

static void matrix_mul(const float a[16], const float b[16], float out[16]) {
    float result[16] = {0};
    for (uint32_t c = 0; c < 4u; ++c) {
        for (uint32_t row = 0; row < 4u; ++row) {
            result[c * 4u + row] = a[row] * b[c * 4u] + a[4u + row] * b[c * 4u + 1u] + a[8u + row] * b[c * 4u + 2u] + a[12u + row] * b[c * 4u + 3u];
        }
    }
    memcpy(out, result, sizeof(result));
}

static void matrix_translation(VEC3 t, float out[16]) {
    matrix_identity(out);
    out[12] = t.x;
    out[13] = t.y;
    out[14] = t.z;
}

static void matrix_scale(VEC3 scale, float out[16]) {
    matrix_identity(out);
    out[0] = scale.x;
    out[5] = scale.y;
    out[10] = scale.z;
}

static void matrix_rotation_x(float angle, float out[16]) {
    const float c = cosf(angle), s = sinf(angle);
    matrix_identity(out);
    out[5] = c;
    out[6] = s;
    out[9] = -s;
    out[10] = c;
}

static void matrix_rotation_y(float angle, float out[16]) {
    const float c = cosf(angle), s = sinf(angle);
    matrix_identity(out);
    out[0] = c;
    out[2] = -s;
    out[8] = s;
    out[10] = c;
}

static void matrix_rotation_z(float angle, float out[16]) {
    const float c = cosf(angle), s = sinf(angle);
    matrix_identity(out);
    out[0] = c;
    out[1] = s;
    out[4] = -s;
    out[5] = c;
}

static void matrix_transpose(const float in[16], float out[16]) {
    float result[16];
    for (uint32_t c = 0; c < 4u; ++c)
        for (uint32_t row = 0; row < 4u; ++row) result[c * 4u + row] = in[row * 4u + c];
    memcpy(out, result, sizeof(result));
}

static void transform_matrices(const TRANSFORM *transform, float world[16], float inverse_world[16], float normal_world[16]) {
    float t[16], rz[16], ry[16], rx[16], scale[16], tmp0[16], tmp1[16];
    matrix_translation(transform->position, t);
    matrix_rotation_z(transform->rotation.z, rz);
    matrix_rotation_y(transform->rotation.y, ry);
    matrix_rotation_x(transform->rotation.x, rx);
    matrix_scale(transform->scale, scale);
    matrix_mul(t, rz, tmp0);
    matrix_mul(tmp0, ry, tmp1);
    matrix_mul(tmp1, rx, tmp0);
    matrix_mul(tmp0, scale, world);

    const VEC3 inverse_scale = {1.0f / transform->scale.x, 1.0f / transform->scale.y, 1.0f / transform->scale.z};
    const VEC3 inverse_translation = {-transform->position.x, -transform->position.y, -transform->position.z};
    matrix_scale(inverse_scale, scale);
    matrix_rotation_x(-transform->rotation.x, rx);
    matrix_rotation_y(-transform->rotation.y, ry);
    matrix_rotation_z(-transform->rotation.z, rz);
    matrix_translation(inverse_translation, t);
    matrix_mul(scale, rx, tmp0);
    matrix_mul(tmp0, ry, tmp1);
    matrix_mul(tmp1, rz, tmp0);
    matrix_mul(tmp0, t, inverse_world);
    matrix_transpose(inverse_world, normal_world);
}

static bool fill_object_entry(DYNAMIC_OBJECT_ENTRY *entry, OBJECT *object) {
    if (!entry || !object || !object->data || !transform_valid(&object->transform)) return false;
    struct MODEL *model = object->data;
    if (!model->geometry || !model->visual) return false;
    const AABB world_bounds = dynamic_transform_bounds(model->geometry->bounds, &object->transform);
    if (!finite_v3(world_bounds.min) || !finite_v3(world_bounds.max)) return false;
    memset(entry, 0, sizeof(*entry));
    entry->object = object;
    entry->world_bounds = world_bounds;
    transform_matrices(&object->transform, entry->world, entry->inverse_world, entry->normal_world);
    return true;
}

static AABB bounds_from_min_max(VEC3 min, VEC3 max) {
    AABB bounds = {0};
    bounds.min = min;
    bounds.max = max;
    bounds.center = (VEC3){(min.x + max.x) * 0.5f, (min.y + max.y) * 0.5f, (min.z + max.z) * 0.5f};
    bounds.extents = (VEC3){(max.x - min.x) * 0.5f, (max.y - min.y) * 0.5f, (max.z - min.z) * 0.5f};
    return bounds;
}

static AABB dynamic_transform_bounds(AABB local, const TRANSFORM *transform) {
    VEC3 min = {INFINITY, INFINITY, INFINITY};
    VEC3 max = {-INFINITY, -INFINITY, -INFINITY};

    for (uint32_t corner = 0; corner < 8u; ++corner) {
        const VEC3 p = {
            (corner & 1u) ? local.max.x : local.min.x,
            (corner & 2u) ? local.max.y : local.min.y,
            (corner & 4u) ? local.max.z : local.min.z
        };
        const VEC3 world = transform_point(transform, p);
        min.x = fminf(min.x, world.x);
        min.y = fminf(min.y, world.y);
        min.z = fminf(min.z, world.z);
        max.x = fmaxf(max.x, world.x);
        max.y = fmaxf(max.y, world.y);
        max.z = fmaxf(max.z, world.z);
    }

    return bounds_from_min_max(min, max);
}

static AABB bounds_union(AABB a, AABB b) {
    return bounds_from_min_max(
        (VEC3){fminf(a.min.x, b.min.x), fminf(a.min.y, b.min.y), fminf(a.min.z, b.min.z)},
        (VEC3){fmaxf(a.max.x, b.max.x), fmaxf(a.max.y, b.max.y), fmaxf(a.max.z, b.max.z)}
    );
}

static AABB bounds_expand(AABB bounds, float radius) {
    const VEC3 amount = {radius, radius, radius};
    return bounds_from_min_max(
        (VEC3){bounds.min.x - amount.x, bounds.min.y - amount.y, bounds.min.z - amount.z},
        (VEC3){bounds.max.x + amount.x, bounds.max.y + amount.y, bounds.max.z + amount.z}
    );
}

static int sample_ref_compare(const void *left, const void *right) {
    const DYNAMIC_SAMPLE_REF *a = left;
    const DYNAMIC_SAMPLE_REF *b = right;

    if (a->z != b->z) return a->z < b->z ? -1 : 1;
    if (a->y != b->y) return a->y < b->y ? -1 : 1;
    if (a->x != b->x) return a->x < b->x ? -1 : 1;
    if (a->sample_index != b->sample_index) return a->sample_index < b->sample_index ? -1 : 1;
    return 0;
}

static uint32_t hash_cell(uint32_t x, uint32_t y, uint32_t z) {
    uint32_t hash = 2166136261u;
    hash = (hash ^ x) * 16777619u;
    hash = (hash ^ y) * 16777619u;
    hash = (hash ^ z) * 16777619u;
    hash ^= hash >> 16u;
    return hash;
}

static uint32_t next_power_of_two(uint32_t value) {
    if (value <= 1u) return 1u;
    if (value > (1u << 30u)) return 0u;
    value--;
    value |= value >> 1u;
    value |= value >> 2u;
    value |= value >> 4u;
    value |= value >> 8u;
    value |= value >> 16u;
    return value + 1u;
}

static uint32_t find_cell(const DYNAMIC_STATE *state, uint32_t x, uint32_t y, uint32_t z) {
    if (!state || !state->lookup || !state->lookup_capacity) return UINT32_MAX;
    const uint32_t mask = state->lookup_capacity - 1u;
    uint32_t slot = hash_cell(x, y, z) & mask;

    for (uint32_t probe = 0; probe < state->lookup_capacity; ++probe) {
        const DYNAMIC_CELL_LOOKUP *entry = &state->lookup[slot];
        if (!entry->cell_plus_one) return UINT32_MAX;
        if (entry->x == x && entry->y == y && entry->z == z) return entry->cell_plus_one - 1u;
        slot = (slot + 1u) & mask;
    }

    return UINT32_MAX;
}

static bool reserve_jobs(DYNAMIC_STATE *state, uint32_t needed) {
    if (needed <= state->job_capacity) return true;
    uint32_t capacity = state->job_capacity ? state->job_capacity : 64u;
    while (capacity < needed) {
        if (capacity > UINT32_MAX / 2u) return false;
        capacity *= 2u;
    }
    DYNAMIC_CELL_JOB *jobs = realloc(state->jobs, (size_t)capacity * sizeof(*jobs));
    if (!jobs) return false;
    state->jobs = jobs;
    state->job_capacity = capacity;
    return true;
}

static bool reserve_updates(DYNAMIC_STATE *state, uint32_t needed) {
    if (needed <= state->update_capacity) return true;
    uint32_t capacity = state->update_capacity ? state->update_capacity : 64u;
    while (capacity < needed) {
        if (capacity > UINT32_MAX / 2u) return false;
        capacity *= 2u;
    }
    DYNAMIC_CELL_UPDATE *updates = realloc(state->updates, (size_t)capacity * sizeof(*updates));
    if (!updates) return false;
    state->updates = updates;
    state->update_capacity = capacity;
    return true;
}

static bool reserve_objects(DYNAMIC_STATE *state, uint32_t needed) {
    if (needed <= state->object_capacity) return true;
    uint32_t capacity = state->object_capacity ? state->object_capacity : 8u;
    while (capacity < needed) {
        if (capacity > UINT32_MAX / 2u) return false;
        capacity *= 2u;
    }
    DYNAMIC_OBJECT_ENTRY *objects = realloc(state->objects, (size_t)capacity * sizeof(*objects));
    if (!objects) return false;
    state->objects = objects;
    state->object_capacity = capacity;
    return true;
}

static bool queue_cell(DYNAMIC_STATE *state, uint32_t cell_index) {
    if (!state || cell_index >= state->cell_count || !reserve_jobs(state, state->job_count + 1u) || !reserve_updates(state, state->update_count + 1u)) return false;

    DYNAMIC_CELL *cell = &state->cells[cell_index];
    cell->generation++;
    if (!cell->generation) cell->generation = 1u;

    state->jobs[state->job_count++] = (DYNAMIC_CELL_JOB){
        .cell_index = cell_index,
        .generation = cell->generation,
        .next_sample = 0u
    };
    state->updates[state->update_count++] = (DYNAMIC_CELL_UPDATE){
        .cell_index = cell_index,
        .generation = cell->generation
    };
    return true;
}

static bool invalidate_bounds(DYNAMIC_STATE *state, AABB bounds) {
    if (!state || !state->cell_count) return true;

    const VEC3 grid_max = {
        state->origin.x + (float)(state->max_x + 1u) * state->cell_size,
        state->origin.y + (float)(state->max_y + 1u) * state->cell_size,
        state->origin.z + (float)(state->max_z + 1u) * state->cell_size
    };

    if (bounds.max.x < state->origin.x || bounds.max.y < state->origin.y || bounds.max.z < state->origin.z || bounds.min.x > grid_max.x || bounds.min.y > grid_max.y ||
        bounds.min.z > grid_max.z)
        return true;

    int64_t min_x = (int64_t)floorf((bounds.min.x - state->origin.x) / state->cell_size);
    int64_t min_y = (int64_t)floorf((bounds.min.y - state->origin.y) / state->cell_size);
    int64_t min_z = (int64_t)floorf((bounds.min.z - state->origin.z) / state->cell_size);
    int64_t max_x = (int64_t)floorf((bounds.max.x - state->origin.x) / state->cell_size);
    int64_t max_y = (int64_t)floorf((bounds.max.y - state->origin.y) / state->cell_size);
    int64_t max_z = (int64_t)floorf((bounds.max.z - state->origin.z) / state->cell_size);

    if (min_x < 0) min_x = 0;
    if (min_y < 0) min_y = 0;
    if (min_z < 0) min_z = 0;
    if (max_x > (int64_t)state->max_x) max_x = state->max_x;
    if (max_y > (int64_t)state->max_y) max_y = state->max_y;
    if (max_z > (int64_t)state->max_z) max_z = state->max_z;
    if (max_x < min_x || max_y < min_y || max_z < min_z) return true;

    for (uint32_t z = (uint32_t)min_z;; ++z) {
        for (uint32_t y = (uint32_t)min_y;; ++y) {
            for (uint32_t x = (uint32_t)min_x;; ++x) {
                const uint32_t cell = find_cell(state, x, y, z);
                if (cell != UINT32_MAX && !queue_cell(state, cell)) return false;
                if (x == (uint32_t)max_x) break;
            }
            if (y == (uint32_t)max_y) break;
        }
        if (z == (uint32_t)max_z) break;
    }

    return true;
}

static bool build_spatial_index(DYNAMIC_STATE *state, const LIGHTMAP *lm) {
    if (!state || !lm || !lm->samples || !lm->sample_count) return false;

    VEC3 min = {INFINITY, INFINITY, INFINITY};
    VEC3 max = {-INFINITY, -INFINITY, -INFINITY};

    for (uint32_t i = 0; i < lm->sample_count; ++i) {
        const VEC3 p = {lm->samples[i].position[0], lm->samples[i].position[1], lm->samples[i].position[2]};
        if (!finite_v3(p)) return false;
        min.x = fminf(min.x, p.x);
        min.y = fminf(min.y, p.y);
        min.z = fminf(min.z, p.z);
        max.x = fmaxf(max.x, p.x);
        max.y = fmaxf(max.y, p.y);
        max.z = fmaxf(max.z, p.z);
    }

    state->origin = min;
    state->cell_size = fmaxf(0.5f, state->settings.gi_radius * 0.5f);

    const double dx = floor(((double)max.x - min.x) / state->cell_size);
    const double dy = floor(((double)max.y - min.y) / state->cell_size);
    const double dz = floor(((double)max.z - min.z) / state->cell_size);
    if (dx < 0.0 || dy < 0.0 || dz < 0.0 || dx > UINT32_MAX - 1.0 || dy > UINT32_MAX - 1.0 || dz > UINT32_MAX - 1.0) return false;
    state->max_x = (uint32_t)dx;
    state->max_y = (uint32_t)dy;
    state->max_z = (uint32_t)dz;

    DYNAMIC_SAMPLE_REF *refs = malloc((size_t)lm->sample_count * sizeof(*refs));
    state->sample_ids = malloc((size_t)lm->sample_count * sizeof(*state->sample_ids));
    if (!refs || !state->sample_ids) {
        free(refs);
        return false;
    }

    for (uint32_t i = 0; i < lm->sample_count; ++i) {
        const float *p = lm->samples[i].position;
        uint32_t x = (uint32_t)floorf((p[0] - min.x) / state->cell_size);
        uint32_t y = (uint32_t)floorf((p[1] - min.y) / state->cell_size);
        uint32_t z = (uint32_t)floorf((p[2] - min.z) / state->cell_size);
        if (x > state->max_x) x = state->max_x;
        if (y > state->max_y) y = state->max_y;
        if (z > state->max_z) z = state->max_z;
        refs[i] = (DYNAMIC_SAMPLE_REF){.x = x, .y = y, .z = z, .sample_index = i};
    }

    qsort(refs, lm->sample_count, sizeof(*refs), sample_ref_compare);

    uint32_t cell_count = 1u;
    for (uint32_t i = 1; i < lm->sample_count; ++i) {
        if (refs[i].x != refs[i - 1u].x || refs[i].y != refs[i - 1u].y || refs[i].z != refs[i - 1u].z) cell_count++;
    }

    state->cells = calloc(cell_count, sizeof(*state->cells));
    if (!state->cells) {
        free(refs);
        return false;
    }
    state->cell_count = cell_count;

    uint32_t cell_index = 0u;
    uint32_t first = 0u;
    for (uint32_t i = 0; i < lm->sample_count; ++i) {
        state->sample_ids[i] = refs[i].sample_index;
        const bool end = i + 1u == lm->sample_count || refs[i + 1u].x != refs[i].x || refs[i + 1u].y != refs[i].y || refs[i + 1u].z != refs[i].z;
        if (end) {
            state->cells[cell_index++] = (DYNAMIC_CELL){
                .x = refs[i].x,
                .y = refs[i].y,
                .z = refs[i].z,
                .first_sample = first,
                .sample_count = i + 1u - first,
                .generation = 0u
            };
            first = i + 1u;
        }
    }
    free(refs);

    const uint32_t lookup_capacity = next_power_of_two(cell_count > UINT32_MAX / 2u ? 0u : cell_count * 2u);
    if (!lookup_capacity) return false;
    state->lookup = calloc(lookup_capacity, sizeof(*state->lookup));
    if (!state->lookup) return false;
    state->lookup_capacity = lookup_capacity;

    const uint32_t mask = lookup_capacity - 1u;
    for (uint32_t i = 0; i < cell_count; ++i) {
        const DYNAMIC_CELL *cell = &state->cells[i];
        uint32_t slot = hash_cell(cell->x, cell->y, cell->z) & mask;
        while (state->lookup[slot].cell_plus_one) slot = (slot + 1u) & mask;
        state->lookup[slot] = (DYNAMIC_CELL_LOOKUP){.x = cell->x, .y = cell->y, .z = cell->z, .cell_plus_one = i + 1u};
    }

    return true;
}

void dynamic_deinit(RENDERER *r) {
    if (!r || !r->dynamic) return;
    DYNAMIC_STATE *state = r->dynamic;
    free(state->sample_ids);
    free(state->cells);
    free(state->lookup);
    free(state->jobs);
    free(state->updates);
    free(state->objects);
    free(state);
    r->dynamic = NULL;
    memset(&r->dynamic_lighting, 0, sizeof(r->dynamic_lighting));
}

bool r_dynamic_init(RENDERER *r, const MESH *static_scene, const LIGHTMAP *lm, const DYNAMIC_LIGHTING *settings) {
    if (!r || !static_scene || !lm || !settings || !settings->texels_per_frame || !settings->rays_per_texel || !settings->target_samples || !settings->shadow_map_size ||
        !isfinite(settings->gi_radius) || settings->gi_radius < 0.0f || !isfinite(settings->shadow_bias) || settings->shadow_bias < 0.0f || !finite_v3(static_scene->bounds.min) ||
        !finite_v3(static_scene->bounds.max) || static_scene->bounds.min.x > static_scene->bounds.max.x || static_scene->bounds.min.y > static_scene->bounds.max.y ||
        static_scene->bounds.min.z > static_scene->bounds.max.z)
        return false;

    if (r->dynamic_model_count && r->graphics_queue && r->core.QueueWaitIdle(r->graphics_queue) != NriResult_SUCCESS) return false;
    gpu_dynamic_deinit(r);
    dynamic_deinit(r);

    DYNAMIC_STATE *state = calloc(1, sizeof(*state));
    if (!state) return false;
    state->object_generation = 1u;
    state->settings = *settings;
    r->dynamic = state;
    r->dynamic_lighting = *settings;

    if (!build_spatial_index(state, lm)) {
        dynamic_deinit(r);
        return false;
    }

    return true;
}

bool r_add_dynamic_object(RENDERER *r, OBJECT *object) {
    if (!r || !r->dynamic || !object || object->state != DYNAMIC || object->type != MODEL || !object->data || !transform_valid(&object->transform)) return false;
    struct MODEL *model = object->data;
    if (!model->geometry || !model->visual) return false;

    DYNAMIC_STATE *state = r->dynamic;
    for (uint32_t i = 0; i < state->object_count; ++i)
        if (state->objects[i].object == object) return false;

    if (!reserve_objects(state, state->object_count + 1u) || !gpu_dynamic_register_model(r, model)) return false;

    DYNAMIC_OBJECT_ENTRY entry = {0};
    if (!fill_object_entry(&entry, object)) {
        gpu_dynamic_unregister_model(r, model);
        return false;
    }

    state->objects[state->object_count++] = entry;
    object->previous_transform = object->transform;

    if (!invalidate_bounds(state, bounds_expand(entry.world_bounds, state->settings.gi_radius))) {
        state->object_count--;
        gpu_dynamic_unregister_model(r, model);
        return false;
    }

    state->object_generation++;
    return true;
}

void r_remove_dynamic_object(RENDERER *r, OBJECT *object) {
    if (!r || !r->dynamic || !object) return;
    DYNAMIC_STATE *state = r->dynamic;

    for (uint32_t i = 0; i < state->object_count; ++i) {
        if (state->objects[i].object != object) continue;
        struct MODEL *model = object->data;
        (void)invalidate_bounds(state, bounds_expand(state->objects[i].world_bounds, state->settings.gi_radius));
        state->objects[i] = state->objects[state->object_count - 1u];
        state->object_count--;
        if (model) gpu_dynamic_unregister_model(r, model);
        state->object_generation++;
        return;
    }
}

bool dynamic_sync(RENDERER *r) {
    if (!r || !r->dynamic) return false;
    DYNAMIC_STATE *state = r->dynamic;

    for (uint32_t i = 0; i < state->object_count; ++i) {
        DYNAMIC_OBJECT_ENTRY *entry = &state->objects[i];
        OBJECT *object = entry->object;
        if (!object || !transform_valid(&object->transform)) return false;
        if (transform_equal(&object->transform, &object->previous_transform)) continue;

        DYNAMIC_OBJECT_ENTRY current = {0};
        if (!fill_object_entry(&current, object)) return false;
        const AABB swept = bounds_expand(bounds_union(entry->world_bounds, current.world_bounds), state->settings.gi_radius);
        if (!invalidate_bounds(state, swept)) return false;
        *entry = current;
        object->previous_transform = object->transform;
        state->object_generation++;
    }

    return true;
}

uint32_t dynamic_take_gi_jobs(RENDERER *r, DYNAMIC_GI_JOB *out, uint32_t capacity) {
    if (!r || !r->dynamic || !out || !capacity) return 0u;
    DYNAMIC_STATE *state = r->dynamic;
    uint32_t limit = capacity;
    if (limit > state->settings.texels_per_frame) limit = state->settings.texels_per_frame;
    uint32_t count = 0u;

    while (count < limit && state->job_count) {
        DYNAMIC_CELL_JOB *job = &state->jobs[state->job_count - 1u];
        if (job->cell_index >= state->cell_count) {
            state->job_count--;
            continue;
        }
        const DYNAMIC_CELL *cell = &state->cells[job->cell_index];
        if (job->generation != cell->generation) {
            state->job_count--;
            continue;
        }

        const uint64_t total = (uint64_t)cell->sample_count * state->settings.target_samples;
        if (job->next_sample >= total) {
            state->job_count--;
            continue;
        }

        const uint32_t local = (uint32_t)(job->next_sample % cell->sample_count);
        out[count++] = (DYNAMIC_GI_JOB){
            .sample_index = state->sample_ids[cell->first_sample + local],
            .cell_index = job->cell_index,
            .generation = job->generation,
            ._pad = 0u
        };
        job->next_sample++;
        if (job->next_sample >= total) state->job_count--;
    }

    return count;
}

uint32_t dynamic_take_cell_updates(RENDERER *r, DYNAMIC_CELL_UPDATE *out, uint32_t capacity) {
    if (!r || !r->dynamic || !out || !capacity) return 0u;
    DYNAMIC_STATE *state = r->dynamic;
    uint32_t count = 0u;

    while (count < capacity && state->update_count) {
        const DYNAMIC_CELL_UPDATE update = state->updates[--state->update_count];
        if (update.cell_index >= state->cell_count || state->cells[update.cell_index].generation != update.generation) continue;
        out[count++] = update;
    }

    return count;
}


uint32_t dynamic_instance_generation(const RENDERER *r) {
    return r && r->dynamic ? r->dynamic->object_generation : 0u;
}

uint32_t dynamic_instance_count(const RENDERER *r) {
    return r && r->dynamic ? r->dynamic->object_count : 0u;
}

bool dynamic_instance_data(RENDERER *r, uint32_t index, DYNAMIC_INSTANCE_DATA *out) {
    if (!r || !r->dynamic || !out || index >= r->dynamic->object_count) return false;
    const DYNAMIC_OBJECT_ENTRY *entry = &r->dynamic->objects[index];
    if (!entry->object || !entry->object->data) return false;
    memset(out, 0, sizeof(*out));
    out->object = entry->object;
    out->model = entry->object->data;
    out->world_bounds = entry->world_bounds;
    memcpy(out->world, entry->world, sizeof(out->world));
    memcpy(out->inverse_world, entry->inverse_world, sizeof(out->inverse_world));
    memcpy(out->normal_world, entry->normal_world, sizeof(out->normal_world));
    return true;
}
