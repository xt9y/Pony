#include "game.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define DYNAMIC_MAX_CELLS (1u << 20u)

typedef struct DYNAMIC_CELL {
    uint32_t first, count, generation;
} DYNAMIC_CELL;

typedef struct DYNAMIC_CELL_JOB {
    uint32_t cell, generation, cursor, pass;
} DYNAMIC_CELL_JOB;

typedef struct DYNAMIC_OBJECT_ENTRY {
    OBJECT *object;
    TRANSFORM transform;
    AABB bounds;
    float world[16], inverse[16], normal[16];
} DYNAMIC_OBJECT_ENTRY;

struct DYNAMIC_STATE {
    DYNAMIC_LIGHTING settings;
    const LMAP_SAMPLE *samples;
    VEC3 origin;
    float cell_size;
    uint32_t dims[3], cell_count;
    uint32_t lightmap_width, lightmap_height;
    uint32_t *sample_ids;
    DYNAMIC_CELL *cells;
    DYNAMIC_CELL_JOB *jobs;
    uint32_t job_count, job_capacity;
    DYNAMIC_CELL_UPDATE *updates;
    uint32_t update_count, update_capacity;
    DYNAMIC_OBJECT_ENTRY *objects;
    uint32_t object_count, object_capacity, object_generation;
};

static bool finite3(VEC3 v) {
    return isfinite(v.x) && isfinite(v.y) && isfinite(v.z);
}

static bool transform_valid(const TRANSFORM *t) {
    return t && finite3(t->position) && finite3(t->rotation) && finite3(t->scale) &&
           t->scale.x != 0.0f && t->scale.y != 0.0f && t->scale.z != 0.0f;
}

static bool transform_equal(const TRANSFORM *a, const TRANSFORM *b) {
    return a->position.x == b->position.x && a->position.y == b->position.y && a->position.z == b->position.z &&
           a->rotation.x == b->rotation.x && a->rotation.y == b->rotation.y && a->rotation.z == b->rotation.z &&
           a->scale.x == b->scale.x && a->scale.y == b->scale.y && a->scale.z == b->scale.z;
}

static bool grow(void **data, uint32_t *capacity, uint32_t count, size_t size, uint32_t base) {
    if (count <= *capacity) return true;
    uint32_t next = *capacity ? *capacity : base;
    while (next < count) {
        if (next > UINT32_MAX / 2u) return false;
        next *= 2u;
    }
    void *p = realloc(*data, (size_t)next * size);
    if (!p) return false;
    *data = p;
    *capacity = next;
    return true;
}

static void identity(float m[16]) {
    memset(m, 0, 16u * sizeof(*m));
    m[0] = m[5] = m[10] = m[15] = 1.0f;
}

static void mul(const float a[16], const float b[16], float out[16]) {
    float r[16] = {0};
    for (uint32_t c = 0; c < 4u; ++c)
        for (uint32_t row = 0; row < 4u; ++row)
            r[c * 4u + row] = a[row] * b[c * 4u] + a[4u + row] * b[c * 4u + 1u] +
                              a[8u + row] * b[c * 4u + 2u] + a[12u + row] * b[c * 4u + 3u];
    memcpy(out, r, sizeof(r));
}

static void matrices(const TRANSFORM *t, float world[16], float inverse[16], float normal[16]) {
    const float cx = cosf(t->rotation.x), sx = sinf(t->rotation.x);
    const float cy = cosf(t->rotation.y), sy = sinf(t->rotation.y);
    const float cz = cosf(t->rotation.z), sz = sinf(t->rotation.z);
    float tr[16], rz[16], ry[16], rx[16], s[16], a[16], b[16];

    identity(tr); tr[12] = t->position.x; tr[13] = t->position.y; tr[14] = t->position.z;
    identity(rz); rz[0] = cz; rz[1] = sz; rz[4] = -sz; rz[5] = cz;
    identity(ry); ry[0] = cy; ry[2] = -sy; ry[8] = sy; ry[10] = cy;
    identity(rx); rx[5] = cx; rx[6] = sx; rx[9] = -sx; rx[10] = cx;
    identity(s); s[0] = t->scale.x; s[5] = t->scale.y; s[10] = t->scale.z;
    mul(tr, rz, a); mul(a, ry, b); mul(b, rx, a); mul(a, s, world);

    identity(tr); tr[12] = -t->position.x; tr[13] = -t->position.y; tr[14] = -t->position.z;
    identity(rz); rz[0] = cz; rz[1] = -sz; rz[4] = sz; rz[5] = cz;
    identity(ry); ry[0] = cy; ry[2] = sy; ry[8] = -sy; ry[10] = cy;
    identity(rx); rx[5] = cx; rx[6] = -sx; rx[9] = sx; rx[10] = cx;
    identity(s); s[0] = 1.0f / t->scale.x; s[5] = 1.0f / t->scale.y; s[10] = 1.0f / t->scale.z;
    mul(s, rx, a); mul(a, ry, b); mul(b, rz, a); mul(a, tr, inverse);

    for (uint32_t c = 0; c < 4u; ++c)
        for (uint32_t row = 0; row < 4u; ++row) normal[c * 4u + row] = inverse[row * 4u + c];
}

static VEC3 matrix_point(const float m[16], VEC3 p) {
    return (VEC3){
        m[0] * p.x + m[4] * p.y + m[8] * p.z + m[12],
        m[1] * p.x + m[5] * p.y + m[9] * p.z + m[13],
        m[2] * p.x + m[6] * p.y + m[10] * p.z + m[14]
    };
}

static AABB make_bounds(VEC3 min, VEC3 max) {
    return (AABB){
        .min = min,
        .max = max,
        .center = {(min.x + max.x) * 0.5f, (min.y + max.y) * 0.5f, (min.z + max.z) * 0.5f},
        .extents = {(max.x - min.x) * 0.5f, (max.y - min.y) * 0.5f, (max.z - min.z) * 0.5f}
    };
}

static AABB transform_bounds(AABB local, const float world[16]) {
    VEC3 min = {INFINITY, INFINITY, INFINITY};
    VEC3 max = {-INFINITY, -INFINITY, -INFINITY};
    for (uint32_t i = 0; i < 8u; ++i) {
        VEC3 p = matrix_point(world, (VEC3){
            i & 1u ? local.max.x : local.min.x,
            i & 2u ? local.max.y : local.min.y,
            i & 4u ? local.max.z : local.min.z
        });
        min.x = fminf(min.x, p.x); min.y = fminf(min.y, p.y); min.z = fminf(min.z, p.z);
        max.x = fmaxf(max.x, p.x); max.y = fmaxf(max.y, p.y); max.z = fmaxf(max.z, p.z);
    }
    return make_bounds(min, max);
}

static AABB swept(AABB a, AABB b, float r) {
    return make_bounds(
        (VEC3){fminf(a.min.x, b.min.x) - r, fminf(a.min.y, b.min.y) - r, fminf(a.min.z, b.min.z) - r},
        (VEC3){fmaxf(a.max.x, b.max.x) + r, fmaxf(a.max.y, b.max.y) + r, fmaxf(a.max.z, b.max.z) + r}
    );
}

static bool fill_entry(DYNAMIC_OBJECT_ENTRY *entry, OBJECT *object) {
    if (!entry || !object || !object->data || !transform_valid(&object->transform)) return false;
    struct MODEL *model = object->data;
    if (!model->geometry || !model->visual) return false;
    entry->object = object;
    entry->transform = object->transform;
    matrices(&object->transform, entry->world, entry->inverse, entry->normal);
    entry->bounds = transform_bounds(model->geometry->bounds, entry->world);
    if (!finite3(entry->bounds.min) || !finite3(entry->bounds.max)) return false;
    return true;
}

static uint32_t cell_index(const DYNAMIC_STATE *s, VEC3 p) {
    int64_t x = (int64_t)floorf((p.x - s->origin.x) / s->cell_size);
    int64_t y = (int64_t)floorf((p.y - s->origin.y) / s->cell_size);
    int64_t z = (int64_t)floorf((p.z - s->origin.z) / s->cell_size);
    if (x < 0) x = 0; if (y < 0) y = 0; if (z < 0) z = 0;
    if (x >= s->dims[0]) x = s->dims[0] - 1u;
    if (y >= s->dims[1]) y = s->dims[1] - 1u;
    if (z >= s->dims[2]) z = s->dims[2] - 1u;
    return (uint32_t)x + s->dims[0] * ((uint32_t)y + s->dims[1] * (uint32_t)z);
}

static bool grid_dims(DYNAMIC_STATE *s, VEC3 min, VEC3 max) {
    s->cell_size = fmaxf(0.5f, s->settings.gi_radius * 0.5f);
    for (;;) {
        uint64_t total = 1u;
        const float span[3] = {max.x - min.x, max.y - min.y, max.z - min.z};
        for (uint32_t axis = 0; axis < 3u; ++axis) {
            double n = floor((double)span[axis] / s->cell_size) + 1.0;
            if (n < 1.0 || n > UINT32_MAX) return false;
            s->dims[axis] = (uint32_t)n;
            total *= s->dims[axis];
        }
        if (total <= DYNAMIC_MAX_CELLS) {
            s->cell_count = (uint32_t)total;
            return true;
        }
        s->cell_size *= 2.0f;
    }
}

static bool build_grid(DYNAMIC_STATE *s, const LIGHTMAP *lm) {
    if (!lm->samples || !lm->sample_count) return false;
    VEC3 min = {INFINITY, INFINITY, INFINITY}, max = {-INFINITY, -INFINITY, -INFINITY};
    for (uint32_t i = 0; i < lm->sample_count; ++i) {
        VEC3 p = {lm->samples[i].position[0], lm->samples[i].position[1], lm->samples[i].position[2]};
        if (!finite3(p)) return false;
        min.x = fminf(min.x, p.x); min.y = fminf(min.y, p.y); min.z = fminf(min.z, p.z);
        max.x = fmaxf(max.x, p.x); max.y = fmaxf(max.y, p.y); max.z = fmaxf(max.z, p.z);
    }

    s->origin = min;
    s->samples = lm->samples;
    s->lightmap_width = lm->width;
    s->lightmap_height = lm->height;
    if (!grid_dims(s, min, max)) return false;

    s->cells = calloc(s->cell_count, sizeof(*s->cells));
    s->sample_ids = malloc((size_t)lm->sample_count * sizeof(*s->sample_ids));
    uint32_t *cursor = calloc(s->cell_count, sizeof(*cursor));
    if (!s->cells || !s->sample_ids || !cursor) { free(cursor); return false; }

    for (uint32_t i = 0; i < lm->sample_count; ++i) {
        VEC3 p = {lm->samples[i].position[0], lm->samples[i].position[1], lm->samples[i].position[2]};
        s->cells[cell_index(s, p)].count++;
    }

    uint32_t offset = 0u;
    for (uint32_t i = 0; i < s->cell_count; ++i) {
        s->cells[i].first = offset;
        cursor[i] = offset;
        offset += s->cells[i].count;
    }

    for (uint32_t i = 0; i < lm->sample_count; ++i) {
        VEC3 p = {lm->samples[i].position[0], lm->samples[i].position[1], lm->samples[i].position[2]};
        uint32_t cell = cell_index(s, p);
        s->sample_ids[cursor[cell]++] = i;
    }

    free(cursor);
    return true;
}

static bool queue_cell(DYNAMIC_STATE *s, uint32_t cell) {
    if (!s->cells[cell].count) return true;
    if (!grow((void **)&s->jobs, &s->job_capacity, s->job_count + 1u, sizeof(*s->jobs), 64u) ||
        !grow((void **)&s->updates, &s->update_capacity, s->update_count + 1u, sizeof(*s->updates), 64u))
        return false;

    uint32_t generation = ++s->cells[cell].generation;
    if (!generation) generation = s->cells[cell].generation = 1u;
    s->jobs[s->job_count++] = (DYNAMIC_CELL_JOB){cell, generation, 0u, 0u};
    s->updates[s->update_count++] = (DYNAMIC_CELL_UPDATE){cell, generation};
    return true;
}

static bool invalidate(DYNAMIC_STATE *s, AABB b) {
    const VEC3 grid_max = {
        s->origin.x + s->dims[0] * s->cell_size,
        s->origin.y + s->dims[1] * s->cell_size,
        s->origin.z + s->dims[2] * s->cell_size
    };
    if (b.max.x < s->origin.x || b.max.y < s->origin.y || b.max.z < s->origin.z ||
        b.min.x >= grid_max.x || b.min.y >= grid_max.y || b.min.z >= grid_max.z)
        return true;

    int min_x = (int)floorf((b.min.x - s->origin.x) / s->cell_size);
    int min_y = (int)floorf((b.min.y - s->origin.y) / s->cell_size);
    int min_z = (int)floorf((b.min.z - s->origin.z) / s->cell_size);
    int max_x = (int)floorf((b.max.x - s->origin.x) / s->cell_size);
    int max_y = (int)floorf((b.max.y - s->origin.y) / s->cell_size);
    int max_z = (int)floorf((b.max.z - s->origin.z) / s->cell_size);
    if (min_x < 0) min_x = 0; if (min_y < 0) min_y = 0; if (min_z < 0) min_z = 0;
    if (max_x >= (int)s->dims[0]) max_x = (int)s->dims[0] - 1;
    if (max_y >= (int)s->dims[1]) max_y = (int)s->dims[1] - 1;
    if (max_z >= (int)s->dims[2]) max_z = (int)s->dims[2] - 1;

    for (int z = min_z; z <= max_z; ++z)
        for (int y = min_y; y <= max_y; ++y)
            for (int x = min_x; x <= max_x; ++x)
                if (!queue_cell(s, (uint32_t)x + s->dims[0] * ((uint32_t)y + s->dims[1] * (uint32_t)z)))
                    return false;
    return true;
}

static void erase_job(DYNAMIC_STATE *s, uint32_t index) {
    if (index + 1u < s->job_count)
        memmove(s->jobs + index, s->jobs + index + 1u, (size_t)(s->job_count - index - 1u) * sizeof(*s->jobs));
    s->job_count--;
}

void dynamic_deinit(RENDERER *r) {
    if (!r || !r->dynamic) return;
    DYNAMIC_STATE *s = r->dynamic;
    free(s->sample_ids);
    free(s->cells);
    free(s->jobs);
    free(s->updates);
    free(s->objects);
    free(s);
    r->dynamic = NULL;
    memset(&r->dynamic_lighting, 0, sizeof(r->dynamic_lighting));
}

bool r_dynamic_init(RENDERER *r, const MESH *static_scene, const LIGHTMAP *lm, const DYNAMIC_LIGHTING *settings) {
    if (!r || !static_scene || !lm || !settings || !settings->texels_per_frame || !settings->rays_per_texel ||
        !settings->target_samples || !settings->shadow_map_size || !isfinite(settings->gi_radius) ||
        settings->gi_radius < 0.0f || !isfinite(settings->shadow_bias) || settings->shadow_bias < 0.0f)
        return false;

    if (r->dynamic_model_count && r->graphics_queue && r->core.QueueWaitIdle(r->graphics_queue) != NriResult_SUCCESS) return false;
    gpu_dynamic_deinit(r);
    dynamic_deinit(r);

    DYNAMIC_STATE *s = calloc(1, sizeof(*s));
    if (!s) return false;
    s->settings = *settings;
    s->object_generation = 1u;
    r->dynamic = s;
    r->dynamic_lighting = *settings;

    if (!build_grid(s, lm)) {
        dynamic_deinit(r);
        return false;
    }
    return true;
}

bool r_add_dynamic_object(RENDERER *r, OBJECT *object) {
    if (!r || !r->dynamic || !object || object->state != DYNAMIC || object->type != MODEL ||
        !object->data || !transform_valid(&object->transform))
        return false;

    DYNAMIC_STATE *s = r->dynamic;
    for (uint32_t i = 0; i < s->object_count; ++i)
        if (s->objects[i].object == object) return false;

    if (!grow((void **)&s->objects, &s->object_capacity, s->object_count + 1u, sizeof(*s->objects), 8u) ||
        (!s->object_count && !gpu_dynamic_runtime_init(r)) || !gpu_dynamic_register_model(r, object->data))
        return false;

    DYNAMIC_OBJECT_ENTRY entry = {0};
    if (!fill_entry(&entry, object)) {
        gpu_dynamic_unregister_model(r, object->data);
        return false;
    }

    s->objects[s->object_count++] = entry;
    AABB expanded = swept(entry.bounds, entry.bounds, s->settings.gi_radius);
    if (!invalidate(s, expanded)) {
        s->object_count--;
        gpu_dynamic_unregister_model(r, object->data);
        return false;
    }
    s->object_generation++;
    return true;
}

void r_remove_dynamic_object(RENDERER *r, OBJECT *object) {
    if (!r || !r->dynamic || !object) return;
    DYNAMIC_STATE *s = r->dynamic;
    for (uint32_t i = 0; i < s->object_count; ++i) {
        if (s->objects[i].object != object) continue;
        AABB expanded = swept(s->objects[i].bounds, s->objects[i].bounds, s->settings.gi_radius);
        (void)invalidate(s, expanded);
        struct MODEL *model = object->data;
        s->objects[i] = s->objects[--s->object_count];
        if (model) gpu_dynamic_unregister_model(r, model);
        s->object_generation++;
        if (!s->object_count) s->job_count = 0u;
        return;
    }
}

bool dynamic_sync(RENDERER *r) {
    if (!r || !r->dynamic) return false;
    DYNAMIC_STATE *s = r->dynamic;
    for (uint32_t i = 0; i < s->object_count; ++i) {
        DYNAMIC_OBJECT_ENTRY *entry = &s->objects[i];
        OBJECT *object = entry->object;
        if (!object || !transform_valid(&object->transform)) return false;
        if (transform_equal(&object->transform, &entry->transform)) continue;

        DYNAMIC_OBJECT_ENTRY current = {0};
        if (!fill_entry(&current, object) ||
            !invalidate(s, swept(entry->bounds, current.bounds, s->settings.gi_radius)))
            return false;
        *entry = current;
        s->object_generation++;
    }
    return true;
}

uint32_t dynamic_take_gi_jobs(RENDERER *r, DYNAMIC_GI_JOB *out, uint32_t capacity) {
    if (!r || !r->dynamic || !out || !capacity) return 0u;
    DYNAMIC_STATE *s = r->dynamic;
    if (!s->object_count) { s->job_count = 0u; return 0u; }

    uint32_t limit = capacity < s->settings.texels_per_frame ? capacity : s->settings.texels_per_frame;
    uint32_t count = 0u;
    uint32_t index = s->job_count;

    while (index && count < limit) {
        uint32_t j = --index;
        DYNAMIC_CELL_JOB *job = &s->jobs[j];
        if (job->cell >= s->cell_count || job->generation != s->cells[job->cell].generation) {
            erase_job(s, j);
            if (index > s->job_count) index = s->job_count;
            continue;
        }

        DYNAMIC_CELL *cell = &s->cells[job->cell];
        while (job->cursor < cell->count && count < limit) {
            uint32_t sample_index = s->sample_ids[cell->first + job->cursor++];
            const LMAP_SAMPLE *sample = &s->samples[sample_index];
            DYNAMIC_GI_JOB *dst = &out[count++];
            memcpy(dst->position, sample->position, sizeof(dst->position));
            memcpy(dst->normal, sample->normal, sizeof(dst->normal));
            dst->cell_index = job->cell;
            dst->generation = job->generation;
            dst->flags = (job->pass == 0u ? 2u : 0u) | (job->pass == 0u && job->cursor == cell->count ? 1u : 0u);
            dst->_pad = 0u;
        }

        if (job->cursor == cell->count) {
            job->cursor = 0u;
            job->pass++;
            if (job->pass >= s->settings.target_samples) erase_job(s, j);
        }
    }
    return count;
}

uint32_t dynamic_take_cell_updates(RENDERER *r, DYNAMIC_CELL_UPDATE *out, uint32_t capacity) {
    if (!r || !r->dynamic || !out || !capacity) return 0u;
    DYNAMIC_STATE *s = r->dynamic;
    uint32_t count = 0u;
    while (count < capacity && s->update_count) {
        DYNAMIC_CELL_UPDATE update = s->updates[--s->update_count];
        if (update.cell_index < s->cell_count && s->cells[update.cell_index].generation == update.generation)
            out[count++] = update;
    }
    return count;
}

bool dynamic_grid_info(const RENDERER *r, DYNAMIC_GRID_INFO *out) {
    if (!r || !r->dynamic || !out) return false;
    const DYNAMIC_STATE *s = r->dynamic;
    *out = (DYNAMIC_GRID_INFO){
        .origin_cell = {s->origin.x, s->origin.y, s->origin.z, s->cell_size},
        .dims = {s->dims[0], s->dims[1], s->dims[2], s->cell_count},
        .lightmap = {s->lightmap_width, s->lightmap_height}
    };
    return true;
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
    out->model = entry->object->data;
    out->world_bounds = entry->bounds;
    memcpy(out->world, entry->world, sizeof(out->world));
    memcpy(out->inverse_world, entry->inverse, sizeof(out->inverse_world));
    memcpy(out->normal_world, entry->normal, sizeof(out->normal_world));
    return true;
}
