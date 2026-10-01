#include "game.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

VEC3 v3(float x, float y, float z) {
    return (VEC3){x, y, z};
}

VEC3 v3_add(VEC3 a, VEC3 b) {
    return v3(a.x + b.x, a.y + b.y, a.z + b.z);
}

VEC3 v3_sub(VEC3 a, VEC3 b) {
    return v3(a.x - b.x, a.y - b.y, a.z - b.z);
}

VEC3 v3_scale(VEC3 v, float s) {
    return v3(v.x * s, v.y * s, v.z * s);
}

float v3_dot(VEC3 a, VEC3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

VEC3 v3_cross(VEC3 a, VEC3 b) {
    return v3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

float v3_len_sq(VEC3 v) {
    return v3_dot(v, v);
}

VEC3 v3_normalize(VEC3 v) {
    const float length = sqrtf(v3_len_sq(v));

    return length > FLT_EPSILON ? v3_scale(v, 1.0f / length) : v3(0.0f, 0.0f, 0.0f);
}

void mesh_free(MESH *mesh) {
    if (!mesh) return;

    free(mesh->vertices.buffer);
    free(mesh->faces.buffer);
    memset(mesh, 0, sizeof(*mesh));
}

static OBJECT *scene_add_object(SCENE *scene, OBJECT_TYPE type, OBJECT_STATE state, TRANSFORM transform, void *data) {
    if (!scene || !data) return NULL;

    if (scene->object_count == scene->object_capacity) {
        uint32_t capacity = scene->object_capacity ? scene->object_capacity * 2u : 16u;

        if (capacity < scene->object_capacity) return NULL;

        OBJECT *objects = realloc(scene->objects, (size_t)capacity * sizeof(*objects));

        if (!objects) return NULL;

        scene->objects = objects;
        scene->object_capacity = capacity;
    }

    OBJECT *object = &scene->objects[scene->object_count++];
    *object = (OBJECT){.state = state, .type = type, .transform = transform, .data = data, .revision = 1u};

    return object;
}

TRANSFORM transform_identity(void) {
    return (TRANSFORM){.position = {0.0f, 0.0f, 0.0f}, .rotation = {0.0f, 0.0f, 0.0f, 1.0f}, .scale = {1.0f, 1.0f, 1.0f}};
}

OBJECT *scene_add_model(SCENE *scene, struct MODEL *model, OBJECT_STATE state, TRANSFORM transform) {
    return scene_add_object(scene, MODEL, state, transform, model);
}

OBJECT *scene_add_light(SCENE *scene, struct LIGHT *light, OBJECT_STATE state, TRANSFORM transform) {
    return scene_add_object(scene, LIGHT, state, transform, light);
}

void object_set_transform(OBJECT *object, TRANSFORM transform) {
    if (!object) return;

    object->transform = transform;
    object_mark_dirty(object);
}

void object_mark_dirty(OBJECT *object) {
    if (!object) return;

    ++object->revision;

    if (!object->revision) object->revision = 1u;
}

void scene_free(SCENE *scene) {
    if (!scene) return;

    free(scene->objects);
    scene->objects = NULL;
    scene->object_count = 0u;
    scene->object_capacity = 0u;
}
