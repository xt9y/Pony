#include "game.h"

#include <stdlib.h>
#include <string.h>


static OBJECT *scene_add_object(
    SCENE *scene,
    OBJECT_TYPE type,
    OBJECT_STATE state,
    TRANSFORM transform,
    void *data
) {
    if (!scene || !data) return NULL;

    if (scene->object_count == scene->object_capacity) {
        uint32_t capacity =
            scene->object_capacity ?
                scene->object_capacity * 2u :
                16u;

        if (capacity < scene->object_capacity) return NULL;

        OBJECT *objects = realloc(
            scene->objects,
            (size_t)capacity * sizeof(*objects)
        );

        if (!objects) return NULL;

        scene->objects = objects;
        scene->object_capacity = capacity;
    }

    OBJECT *object = &scene->objects[scene->object_count++];

    *object = (OBJECT){
        .state = state,
        .type = type,
        .transform = transform,
        .data = data,
        .revision = 1u
    };

    return object;
}


TRANSFORM transform_identity(void) {
    return (TRANSFORM){
        .position = {0.0f, 0.0f, 0.0f},
        .rotation = {0.0f, 0.0f, 0.0f, 1.0f},
        .scale = {1.0f, 1.0f, 1.0f}
    };
}


OBJECT *scene_add_model(
    SCENE *scene,
    struct MODEL *model,
    OBJECT_STATE state,
    TRANSFORM transform
) {
    return scene_add_object(
        scene,
        MODEL,
        state,
        transform,
        model
    );
}


OBJECT *scene_add_light(
    SCENE *scene,
    struct LIGHT *light,
    OBJECT_STATE state,
    TRANSFORM transform
) {
    return scene_add_object(
        scene,
        LIGHT,
        state,
        transform,
        light
    );
}


void object_set_transform(
    OBJECT *object,
    TRANSFORM transform
) {
    if (!object) return;

    object->transform = transform;
    object_mark_dirty(object);
}


void object_mark_dirty(OBJECT *object) {
    if (!object) return;

    ++object->revision;

    if (!object->revision) {
        object->revision = 1u;
    }
}


void scene_free(SCENE *scene) {
    if (!scene) return;

    free(scene->objects);
    memset(scene, 0, sizeof(*scene));
}
