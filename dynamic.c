#include "game.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define RT_BUFFER_COUNT 3u
#define RT_MAGIC 0x50525431u
#define RT_HEADER_WORDS 16u
#define RT_NODE_WORDS 12u
#define RT_STATIC_TRIANGLE_WORDS 20u
#define RT_OBJECT_WORDS 8u
#define RT_DYNAMIC_TRIANGLE_WORDS 9u
#define RT_EMITTER_WORDS 16u

typedef struct DYNAMIC_OBJECT_ENTRY {
    OBJECT *object;
    TRANSFORM transform;
    AABB bounds;

    float world[16], inverse[16], normal[16];
} DYNAMIC_OBJECT_ENTRY;

struct DYNAMIC_STATE {
    DYNAMIC_LIGHTING settings;
    BVH static_bvh;
    DYNAMIC_OBJECT_ENTRY *objects;

    uint32_t object_count, object_capacity, object_generation;

    NriBuffer *rt_buffers[RT_BUFFER_COUNT];
    uint64_t rt_capacity;
};

static bool finite3(VEC3 v) {
    return isfinite(v.x) && isfinite(v.y) && isfinite(v.z);
}

static bool transform_valid(const TRANSFORM *t) {
    return t && finite3(t->position) && finite3(t->rotation) && finite3(t->scale) && t->scale.x != 0.0f && t->scale.y != 0.0f && t->scale.z != 0.0f;
}

static bool transform_equal(const TRANSFORM *a, const TRANSFORM *b) {
    return a->position.x == b->position.x && a->position.y == b->position.y && a->position.z == b->position.z && a->rotation.x == b->rotation.x && a->rotation.y == b->rotation.y &&
           a->rotation.z == b->rotation.z && a->scale.x == b->scale.x && a->scale.y == b->scale.y && a->scale.z == b->scale.z;
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
            r[c * 4u + row] = a[row] * b[c * 4u] + a[4u + row] * b[c * 4u + 1u] + a[8u + row] * b[c * 4u + 2u] + a[12u + row] * b[c * 4u + 3u];
    memcpy(out, r, sizeof(r));
}

static void matrices(const TRANSFORM *t, float world[16], float inverse[16], float normal[16]) {
    const float cx = cosf(t->rotation.x), sx = sinf(t->rotation.x);
    const float cy = cosf(t->rotation.y), sy = sinf(t->rotation.y);
    const float cz = cosf(t->rotation.z), sz = sinf(t->rotation.z);

    float tr[16], rz[16], ry[16], rx[16], s[16], a[16], b[16];

    identity(tr);

    tr[12] = t->position.x;
    tr[13] = t->position.y;
    tr[14] = t->position.z;

    identity(rz);

    rz[0] = cz;
    rz[1] = sz;
    rz[4] = -sz;
    rz[5] = cz;

    identity(ry);

    ry[0] = cy;
    ry[2] = -sy;
    ry[8] = sy;
    ry[10] = cy;

    identity(rx);

    rx[5] = cx;
    rx[6] = sx;
    rx[9] = -sx;
    rx[10] = cx;

    identity(s);
    s[0] = t->scale.x;
    s[5] = t->scale.y;
    s[10] = t->scale.z;
    mul(tr, rz, a);
    mul(a, ry, b);
    mul(b, rx, a);
    mul(a, s, world);

    identity(tr);

    tr[12] = -t->position.x;
    tr[13] = -t->position.y;
    tr[14] = -t->position.z;

    identity(rz);

    rz[0] = cz;
    rz[1] = -sz;
    rz[4] = sz;
    rz[5] = cz;

    identity(ry);

    ry[0] = cy;
    ry[2] = sy;
    ry[8] = -sy;
    ry[10] = cy;

    identity(rx);

    rx[5] = cx;
    rx[6] = -sx;
    rx[9] = sx;
    rx[10] = cx;

    identity(s);
    s[0] = 1.0f / t->scale.x;
    s[5] = 1.0f / t->scale.y;
    s[10] = 1.0f / t->scale.z;
    mul(s, rx, a);
    mul(a, ry, b);
    mul(b, rz, a);
    mul(a, tr, inverse);

    for (uint32_t c = 0; c < 4u; ++c)
        for (uint32_t row = 0; row < 4u; ++row)
            normal[c * 4u + row] = inverse[row * 4u + c];
}

static VEC3 matrix_point(const float m[16], VEC3 p) {
    return (VEC3){m[0] * p.x + m[4] * p.y + m[8] * p.z + m[12], m[1] * p.x + m[5] * p.y + m[9] * p.z + m[13], m[2] * p.x + m[6] * p.y + m[10] * p.z + m[14]};
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
        VEC3 p = matrix_point(world, (VEC3){i & 1u ? local.max.x : local.min.x, i & 2u ? local.max.y : local.min.y, i & 4u ? local.max.z : local.min.z});
        min.x = fminf(min.x, p.x);
        min.y = fminf(min.y, p.y);
        min.z = fminf(min.z, p.z);
        max.x = fmaxf(max.x, p.x);
        max.y = fmaxf(max.y, p.y);
        max.z = fmaxf(max.z, p.z);
    }

    return make_bounds(min, max);
}

static bool fill_entry(DYNAMIC_OBJECT_ENTRY *entry, OBJECT *object) {
    if (!entry || !object || !object->data || !transform_valid(&object->transform)) return false;

    struct MODEL *model = object->data;

    if (!model->geometry || !model->visual) return false;
    entry->object = object;
    entry->transform = object->transform;
    matrices(&object->transform, entry->world, entry->inverse, entry->normal);
    entry->bounds = transform_bounds(model->geometry->bounds, entry->world);

    return finite3(entry->bounds.min) && finite3(entry->bounds.max);
}

static uint32_t float_bits(float value) {
    uint32_t bits = 0u;

    memcpy(&bits, &value, sizeof(bits));

    return bits;
}

static bool rt_owned(const DYNAMIC_STATE *s, const NriBuffer *buffer) {
    if (!s || !buffer) return false;

    for (uint32_t i = 0; i < RT_BUFFER_COUNT; ++i)
        if (s->rt_buffers[i] == buffer) return true;

    return false;
}

static void rt_destroy(RENDERER *r, DYNAMIC_STATE *s) {
    if (!r || !s) return;

    if (rt_owned(s, r->dynamic_overlay_generation)) r->dynamic_overlay_generation = NULL;

    if (r->dynamic_cell_generation == r->bvh_triangle_buffer) r->dynamic_cell_generation = NULL;

    for (uint32_t i = 0; i < RT_BUFFER_COUNT; ++i) {
        if (s->rt_buffers[i]) r->core.DestroyBuffer(s->rt_buffers[i]);
        s->rt_buffers[i] = NULL;
    }

    s->rt_capacity = 0u;
}

static bool rt_ensure_capacity(RENDERER *r, DYNAMIC_STATE *s, uint64_t bytes) {
    if (!r || !s || !bytes) return false;

    bool complete = s->rt_capacity >= bytes;

    for (uint32_t i = 0; i < RT_BUFFER_COUNT; ++i)
        complete = complete && s->rt_buffers[i] != NULL;

    if (complete) return true;

    if (r->graphics_queue && r->core.QueueWaitIdle(r->graphics_queue) != NriResult_SUCCESS) return false;
    rt_destroy(r, s);

    uint64_t capacity = 4096u;

    while (capacity < bytes) {
        if (capacity > UINT64_MAX / 2u) return false;
        capacity *= 2u;
    }

    const NriBufferDesc desc = {
        .size = capacity,
        .structureStride = sizeof(uint32_t),
        .usage = NriBufferUsageBits_SHADER_RESOURCE
    };

    for (uint32_t i = 0; i < RT_BUFFER_COUNT; ++i) {
        if (r->core.CreateCommittedBuffer(r->device, NriMemoryLocation_HOST_UPLOAD, 0.0f, &desc, &s->rt_buffers[i]) != NriResult_SUCCESS) {
            rt_destroy(r, s);

            return false;
        }
    }

    s->rt_capacity = capacity;

    return true;
}

static bool dynamic_triangle_count(const DYNAMIC_STATE *s, uint32_t *out) {
    uint64_t count = 0u;

    for (uint32_t i = 0; i < s->object_count; ++i) {
        const struct MODEL *model = s->objects[i].object ? s->objects[i].object->data : NULL;

        if (!model || !model->visual || model->visual->vertex_count % 3u) return false;
        count += model->visual->vertex_count / 3u;
    }

    if (count > UINT32_MAX) return false;
    *out = (uint32_t)count;
    return true;
}

static bool static_emitter_count(const BVH *tree, uint32_t *out) {
    if (!tree || !out) return false;

    uint32_t count = 0u;
    float previous = 0.0f;

    for (uint32_t i = 0u; i < tree->triangle_count; ++i) {
        const float cumulative = tree->triangles[i].emissive[3];

        if (!isfinite(cumulative) || cumulative + 1.0e-6f < previous) return false;

        if (cumulative > previous + 1.0e-8f) ++count;

        previous = cumulative;
    }
    *out = count;
    return true;
}

static bool rt_upload(RENDERER *r) {
    if (!r || !r->dynamic || !r->device || !r->bvh_triangle_buffer) return false;

    DYNAMIC_STATE *s = r->dynamic;

    uint32_t dynamic_triangles = 0u;
    uint32_t emitter_count = 0u;

    if (!dynamic_triangle_count(s, &dynamic_triangles) || !static_emitter_count(&s->static_bvh, &emitter_count)) return false;

    const uint64_t node_words = (uint64_t)s->static_bvh.node_count * RT_NODE_WORDS;
    const uint64_t static_triangle_words = (uint64_t)s->static_bvh.triangle_count * RT_STATIC_TRIANGLE_WORDS;
    const uint64_t object_words = (uint64_t)s->object_count * RT_OBJECT_WORDS;
    const uint64_t dynamic_triangle_words = (uint64_t)dynamic_triangles * RT_DYNAMIC_TRIANGLE_WORDS;
    const uint64_t emitter_words = (uint64_t)emitter_count * RT_EMITTER_WORDS;
    const uint64_t total_words = RT_HEADER_WORDS + node_words + static_triangle_words + object_words + dynamic_triangle_words + emitter_words;

    if (!total_words || total_words > SIZE_MAX / sizeof(uint32_t)) return false;

    const uint32_t static_node_offset = RT_HEADER_WORDS;
    const uint32_t static_triangle_offset = static_node_offset + (uint32_t)node_words;
    const uint32_t object_offset = static_triangle_offset + (uint32_t)static_triangle_words;
    const uint32_t dynamic_triangle_offset = object_offset + (uint32_t)object_words;
    const uint32_t emitter_offset = dynamic_triangle_offset + (uint32_t)dynamic_triangle_words;

    uint32_t *words = calloc((size_t)total_words, sizeof(*words));

    if (!words) return false;

    words[0] = RT_MAGIC;
    words[1] = s->static_bvh.node_count;
    words[2] = s->static_bvh.triangle_count;
    words[3] = static_node_offset;
    words[4] = static_triangle_offset;
    words[5] = s->object_count;
    words[6] = object_offset;
    words[7] = dynamic_triangles;
    words[8] = dynamic_triangle_offset;
    words[9] = emitter_count;
    words[10] = float_bits(s->static_bvh.emissive_weight);
    words[11] = float_bits(fmaxf(1.0e-4f, r->scene_radius * 1.0e-5f));
    words[12] = s->object_generation;
    words[13] = emitter_offset;

    memcpy(words + static_node_offset, s->static_bvh.nodes, (size_t)s->static_bvh.node_count * sizeof(*s->static_bvh.nodes));
    memcpy(words + static_triangle_offset, s->static_bvh.triangles, (size_t)s->static_bvh.triangle_count * sizeof(*s->static_bvh.triangles));

    uint32_t triangle_cursor = 0u;

    for (uint32_t i = 0; i < s->object_count; ++i) {
        const DYNAMIC_OBJECT_ENTRY *entry = &s->objects[i];
        const struct MODEL *model = entry->object->data;
        const GLTF_SCENE *visual = model->visual;
        const uint32_t count = (uint32_t)(visual->vertex_count / 3u);
        uint32_t *object = words + object_offset + i * RT_OBJECT_WORDS;
        object[0] = float_bits(entry->bounds.min.x);
        object[1] = float_bits(entry->bounds.min.y);
        object[2] = float_bits(entry->bounds.min.z);
        object[3] = float_bits(entry->bounds.max.x);
        object[4] = float_bits(entry->bounds.max.y);
        object[5] = float_bits(entry->bounds.max.z);
        object[6] = triangle_cursor;
        object[7] = count;

        for (uint32_t t = 0u; t < count; ++t) {
            uint32_t *triangle = words + dynamic_triangle_offset + (triangle_cursor + t) * RT_DYNAMIC_TRIANGLE_WORDS;

            for (uint32_t v = 0u; v < 3u; ++v) {
                VEC3 p = matrix_point(entry->world, visual->vertices[t * 3u + v].position);
                triangle[v * 3u + 0u] = float_bits(p.x);
                triangle[v * 3u + 1u] = float_bits(p.y);
                triangle[v * 3u + 2u] = float_bits(p.z);
            }
        }

        triangle_cursor += count;
    }

    float previous_importance = 0.0f;
    uint32_t emitter_cursor = 0u;

    for (uint32_t i = 0u; i < s->static_bvh.triangle_count; ++i) {
        const BVH_TRIANGLE *triangle = &s->static_bvh.triangles[i];
        const float cumulative = triangle->emissive[3];
        const float importance = cumulative - previous_importance;
        previous_importance = cumulative;

        if (importance <= 1.0e-8f) continue;

        uint32_t *emitter = words + emitter_offset + emitter_cursor * RT_EMITTER_WORDS;

        const float values[15] = {
            triangle->a[0],
            triangle->a[1],
            triangle->a[2],
            triangle->b[0],
            triangle->b[1],
            triangle->b[2],
            triangle->c[0],
            triangle->c[1],
            triangle->c[2],
            triangle->normal[0],
            triangle->normal[1],
            triangle->normal[2],
            triangle->emissive[0],
            triangle->emissive[1],
            triangle->emissive[2]
        };

        for (uint32_t word = 0u; word < 15u; ++word)
            emitter[word] = float_bits(values[word]);
        emitter[15] = float_bits(cumulative);
        ++emitter_cursor;
    }

    if (emitter_cursor != emitter_count) {
        free(words);

        return false;
    }

    const uint64_t bytes = total_words * sizeof(uint32_t);

    if (!rt_ensure_capacity(r, s, bytes)) {
        free(words);

        return false;
    }

    const uint32_t slot = (uint32_t)(r->frame_index % RT_BUFFER_COUNT);
    void *mapped = r->core.MapBuffer(s->rt_buffers[slot], 0u, bytes);

    if (!mapped) {
        free(words);

        return false;
    }

    memcpy(mapped, words, (size_t)bytes);
    r->core.UnmapBuffer(s->rt_buffers[slot]);
    free(words);

    r->dynamic_cell_generation = r->bvh_triangle_buffer;
    r->dynamic_overlay_generation = s->rt_buffers[slot];

    return true;
}

void dynamic_deinit(RENDERER *r) {
    if (!r || !r->dynamic) return;

    DYNAMIC_STATE *s = r->dynamic;

    if (r->device && r->graphics_queue) (void)r->core.QueueWaitIdle(r->graphics_queue);
    rt_destroy(r, s);
    bvh_free(&s->static_bvh);
    free(s->objects);
    free(s);
    r->dynamic = NULL;
    memset(&r->dynamic_lighting, 0, sizeof(r->dynamic_lighting));
}

bool r_dynamic_init(RENDERER *r, const MESH *static_scene, const GLTF_SCENE *static_visual, const LIGHTMAP *lm, const DYNAMIC_LIGHTING *settings) {
    if (!r || !r->device || !static_scene || !static_visual || !lm || !settings || !settings->texels_per_frame || !settings->rays_per_texel || !settings->target_samples ||
        !settings->shadow_map_size || !isfinite(settings->gi_radius) || settings->gi_radius < 0.0f || !isfinite(settings->shadow_bias) || settings->shadow_bias < 0.0f)
        return false;

    if (r->graphics_queue && r->core.QueueWaitIdle(r->graphics_queue) != NriResult_SUCCESS) return false;
    dynamic_deinit(r);
    gpu_dynamic_deinit(r);

    DYNAMIC_STATE *s = calloc(1, sizeof(*s));

    if (!s) return false;
    s->settings = *settings;
    s->object_generation = 1u;

    if (!bvh_build(&s->static_bvh, static_scene, static_visual)) {
        free(s);

        return false;
    }

    r->dynamic = s;
    r->dynamic_lighting = *settings;
    r->dynamic_lighting.texels_per_frame = 0u;

    if (!rt_upload(r)) {
        dynamic_deinit(r);

        return false;
    }

    return true;
}

bool r_add_dynamic_object(RENDERER *r, OBJECT *object) {
    if (!r || !r->dynamic || !object || object->state != DYNAMIC || object->type != MODEL || !object->data || !transform_valid(&object->transform)) return false;

    DYNAMIC_STATE *s = r->dynamic;

    for (uint32_t i = 0; i < s->object_count; ++i)
        if (s->objects[i].object == object) return false;

    if (!grow((void **)&s->objects, &s->object_capacity, s->object_count + 1u, sizeof(*s->objects), 8u) || !gpu_dynamic_register_model(r, object->data)) return false;

    DYNAMIC_OBJECT_ENTRY entry = {0};

    if (!fill_entry(&entry, object)) {
        gpu_dynamic_unregister_model(r, object->data);

        return false;
    }

    s->objects[s->object_count++] = entry;
    s->object_generation++;

    if (!rt_upload(r)) {
        s->object_count--;
        gpu_dynamic_unregister_model(r, object->data);

        return false;
    }

    return true;
}

void r_remove_dynamic_object(RENDERER *r, OBJECT *object) {
    if (!r || !r->dynamic || !object) return;

    DYNAMIC_STATE *s = r->dynamic;

    for (uint32_t i = 0; i < s->object_count; ++i) {
        if (s->objects[i].object != object) continue;

        struct MODEL *model = object->data;

        s->objects[i] = s->objects[--s->object_count];

        if (model) gpu_dynamic_unregister_model(r, model);
        s->object_generation++;
        (void)rt_upload(r);

        return;
    }
}

bool dynamic_sync(RENDERER *r) {
    if (!r || !r->dynamic) return false;

    DYNAMIC_STATE *s = r->dynamic;
    bool changed = false;

    for (uint32_t i = 0; i < s->object_count; ++i) {
        DYNAMIC_OBJECT_ENTRY *entry = &s->objects[i];
        OBJECT *object = entry->object;

        if (!object || !transform_valid(&object->transform)) return false;

        if (transform_equal(&object->transform, &entry->transform)) continue;

        DYNAMIC_OBJECT_ENTRY current = {0};

        if (!fill_entry(&current, object)) return false;
        *entry = current;
        changed = true;
    }

    if (changed) s->object_generation++;

    return rt_upload(r);
}

bool dynamic_grid_info(const RENDERER *r, DYNAMIC_GRID_INFO *out) {
    (void)r;
    (void)out;

    return false;
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

uint32_t dynamic_take_gi_jobs(RENDERER *r, DYNAMIC_GI_JOB *out, uint32_t capacity) {
    (void)r;
    (void)out;
    (void)capacity;

    return 0u;
}

uint32_t dynamic_take_cell_updates(RENDERER *r, DYNAMIC_CELL_UPDATE *out, uint32_t capacity) {
    (void)r;
    (void)out;
    (void)capacity;

    return 0u;
}
