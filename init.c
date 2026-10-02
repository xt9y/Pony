#include "game.h"

#include <float.h>
#include <limits.h>
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

bool model_load(MODEL_ASSET *asset, const char *path) {
    if (!asset || !path || !*path) return false;

    model_free(asset);

    if (!glb_load(&asset->document, path)) {
        SDL_SetError("could not load %s: %s", path, glb_error(&asset->document));
        model_free(asset);
        return false;
    }

    if (!glb_extract_mesh(&asset->document, &asset->geometry)) {
        SDL_SetError("could not extract mesh from %s", path);
        model_free(asset);
        return false;
    }

    if (!gltf_extract(&asset->document, &asset->visual)) {
        SDL_SetError("could not extract visual glTF from %s", path);
        model_free(asset);
        return false;
    }

    asset->model.geometry = &asset->geometry;
    asset->model.visual = &asset->visual;

    return true;
}

void model_free(MODEL_ASSET *asset) {
    if (!asset) return;

    if (asset->model.surface_layout) {
        lmap_free(asset->model.surface_layout);
        free(asset->model.surface_layout);
        asset->model.surface_layout = NULL;
    }

    gltf_free(&asset->visual);
    mesh_free(&asset->geometry);
    glb_free(&asset->document);
    memset(asset, 0, sizeof(*asset));
}

static void revision_bump(uint32_t *revision) {
    if (!revision) return;
    ++*revision;
    if (!*revision) *revision = 1u;
}

static void scene_invalidate_geometry(SCENE *scene, bool invalidates_static_bake) {
    if (!scene) return;

    scene->compiled = false;
    revision_bump(&scene->geometry_revision);

    if (invalidates_static_bake) scene->lightmap_valid = false;
}

static void scene_invalidate_lighting(SCENE *scene) {
    if (!scene) return;
    revision_bump(&scene->lighting_revision);
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

    ++scene->next_object_id;
    if (!scene->next_object_id) ++scene->next_object_id;

    *object = (OBJECT){
        .owner = scene,
        .id = scene->next_object_id,
        .state = state,
        .type = type,
        .transform = transform,
        .data = data,
        .revision = 1u,
        .transform_revision = 1u,
        .lighting_revision = 1u,
    };

    return object;
}

TRANSFORM transform_identity(void) {
    return (TRANSFORM){.position = {0.0f, 0.0f, 0.0f}, .rotation = {0.0f, 0.0f, 0.0f, 1.0f}, .scale = {1.0f, 1.0f, 1.0f}};
}

OBJECT *scene_add_model(SCENE *scene, struct MODEL *model, OBJECT_STATE state, TRANSFORM transform) {
    OBJECT *object = scene_add_object(scene, MODEL, state, transform, model);

    if (object) {
        scene_invalidate_geometry(scene, state == STATIC);
        scene_invalidate_lighting(scene);
    }

    return object;
}

OBJECT *scene_add_light(SCENE *scene, struct LIGHT *light, OBJECT_STATE state, TRANSFORM transform) {
    OBJECT *object = scene_add_object(scene, LIGHT, state, transform, light);

    if (object) scene_invalidate_lighting(scene);
    return object;
}

OBJECT *scene_object_by_id(SCENE *scene, OBJECT_ID id) {
    if (!scene || !id) return NULL;

    for (uint32_t i = 0; i < scene->object_count; ++i)
        if (scene->objects[i].id == id) return &scene->objects[i];

    return NULL;
}

const OBJECT *scene_object_by_id_const(const SCENE *scene, OBJECT_ID id) {
    if (!scene || !id) return NULL;

    for (uint32_t i = 0; i < scene->object_count; ++i)
        if (scene->objects[i].id == id) return &scene->objects[i];

    return NULL;
}

void object_set_transform(OBJECT *object, TRANSFORM transform) {
    if (!object) return;

    object->transform = transform;
    revision_bump(&object->revision);
    revision_bump(&object->transform_revision);

    if (object->type == MODEL) {
        /*
         * DYNAMIC transforms rebuild only the current CPU trace view. They do
         * not invalidate the permanent static atlas/bake.
         */
        scene_invalidate_geometry(object->owner, object->state == STATIC);
    }

    object_mark_lighting_dirty(object);
}

void object_mark_lighting_dirty(OBJECT *object) {
    if (!object) return;

    revision_bump(&object->lighting_revision);
    scene_invalidate_lighting(object->owner);
}

void object_mark_dirty(OBJECT *object) {
    if (!object) return;

    revision_bump(&object->revision);

    if (object->type == MODEL) scene_invalidate_geometry(object->owner, object->state == STATIC);
    object_mark_lighting_dirty(object);
}

static VEC3 rotate_vector(const float rotation[4], VEC3 value) {
    const VEC3 q = v3(rotation[0], rotation[1], rotation[2]);
    const float q_length_sq = v3_len_sq(q) + rotation[3] * rotation[3];

    if (q_length_sq <= FLT_EPSILON) return value;

    const float inv_length = 1.0f / sqrtf(q_length_sq);
    const VEC3 u = v3_scale(q, inv_length);
    const float w = rotation[3] * inv_length;
    const VEC3 t = v3_scale(v3_cross(u, value), 2.0f);

    return v3_add(value, v3_add(v3_scale(t, w), v3_cross(u, t)));
}

static VEC3 transform_position(TRANSFORM transform, VEC3 position) {
    VEC3 scaled = v3(position.x * transform.scale.x, position.y * transform.scale.y, position.z * transform.scale.z);

    return v3_add(rotate_vector(transform.rotation, scaled), transform.position);
}

static VEC3 transform_normal(TRANSFORM transform, VEC3 normal) {
    VEC3 scaled = {
        fabsf(transform.scale.x) > FLT_EPSILON ? normal.x / transform.scale.x : 0.0f,
        fabsf(transform.scale.y) > FLT_EPSILON ? normal.y / transform.scale.y : 0.0f,
        fabsf(transform.scale.z) > FLT_EPSILON ? normal.z / transform.scale.z : 0.0f,
    };

    return v3_normalize(rotate_vector(transform.rotation, scaled));
}

static bool add_size(size_t *total, size_t value) {
    if (!total || value > SIZE_MAX - *total) return false;

    *total += value;
    return true;
}

static bool add_u32(uint32_t *total, uint32_t value) {
    if (!total || value > UINT32_MAX - *total) return false;

    *total += value;
    return true;
}

static bool material_texture_offset(GLTF_MATERIAL *material, uint32_t offset) {
    int32_t *indices[] = {
        &material->base_color_texture,
        &material->metallic_roughness_texture,
        &material->normal_texture,
        &material->occlusion_texture,
        &material->emissive_texture,
        &material->transmission_texture,
        &material->thickness_texture,
        &material->iridescence_texture,
        &material->iridescence_thickness_texture,
    };

    for (uint32_t i = 0; i < sizeof(indices) / sizeof(indices[0]); ++i) {
        if (*indices[i] < 0) continue;
        if ((uint32_t)*indices[i] > (uint32_t)INT32_MAX - offset) return false;

        *indices[i] += (int32_t)offset;
    }

    return true;
}

static void scene_mesh_bounds(MESH *mesh) {
    if (!mesh || !mesh->vertices.count) return;

    const POINT *points = mesh->vertices.buffer;
    VEC3 minimum = points[0].p;
    VEC3 maximum = points[0].p;

    for (size_t i = 1; i < mesh->vertices.count; ++i) {
        const VEC3 point = points[i].p;

        minimum.x = fminf(minimum.x, point.x);
        minimum.y = fminf(minimum.y, point.y);
        minimum.z = fminf(minimum.z, point.z);
        maximum.x = fmaxf(maximum.x, point.x);
        maximum.y = fmaxf(maximum.y, point.y);
        maximum.z = fmaxf(maximum.z, point.z);
    }

    mesh->bounds.min = minimum;
    mesh->bounds.max = maximum;
    mesh->bounds.center = v3_scale(v3_add(minimum, maximum), 0.5f);
    mesh->bounds.extents = v3_scale(v3_sub(maximum, minimum), 0.5f);
}

static bool scene_extract_static(const SCENE *scene, const MESH *geometry, const GLTF_SCENE *visual, MESH *out_geometry, GLTF_SCENE *out_visual) {
    if (!scene || !geometry || !visual || !out_geometry || !out_visual) return false;

    size_t vertex_count = 0;
    size_t face_count = 0;
    size_t visual_vertex_count = 0;

    for (uint32_t i = 0; i < scene->object_count; ++i) {
        const OBJECT *object = &scene->objects[i];

        if (object->state != STATIC || object->type != MODEL || !object->data) continue;

        const struct MODEL *model = object->data;

        if (!add_size(&vertex_count, model->geometry->vertices.count) || !add_size(&face_count, model->geometry->faces.count) ||
            !add_size(&visual_vertex_count, model->visual->vertex_count))
            return false;
    }

    if (!vertex_count || !face_count || !visual_vertex_count || vertex_count > UINT32_MAX || face_count > UINT32_MAX) return false;

    MESH result_geometry = {
        .vertices = {.count = vertex_count, .capacity = vertex_count, .type_size = sizeof(POINT)},
        .faces = {.count = face_count, .capacity = face_count, .type_size = sizeof(MESH_FACE)},
    };

    GLTF_SCENE result_visual = {
        .vertex_count = visual_vertex_count,
        .vertex_capacity = visual_vertex_count,
        .material_count = visual->material_count,
        .texture_count = visual->texture_count,
        .image_count = visual->image_count,
        .default_material = visual->default_material,
    };

    result_geometry.vertices.buffer = calloc(vertex_count, sizeof(POINT));
    result_geometry.faces.buffer = calloc(face_count, sizeof(MESH_FACE));
    result_visual.vertices = calloc(visual_vertex_count, sizeof(*result_visual.vertices));
    result_visual.materials = calloc(visual->material_count, sizeof(*result_visual.materials));
    result_visual.textures = visual->texture_count ? calloc(visual->texture_count, sizeof(*result_visual.textures)) : NULL;
    result_visual.images = visual->image_count ? calloc(visual->image_count, sizeof(*result_visual.images)) : NULL;

    if (!result_geometry.vertices.buffer || !result_geometry.faces.buffer || !result_visual.vertices || !result_visual.materials ||
        (visual->texture_count && !result_visual.textures) || (visual->image_count && !result_visual.images))
        goto fail;

    memcpy(result_visual.materials, visual->materials, (size_t)visual->material_count * sizeof(*result_visual.materials));
    if (visual->texture_count)
        memcpy(result_visual.textures, visual->textures, (size_t)visual->texture_count * sizeof(*result_visual.textures));
    if (visual->image_count)
        memcpy(result_visual.images, visual->images, (size_t)visual->image_count * sizeof(*result_visual.images));

    POINT *dst_points = result_geometry.vertices.buffer;
    MESH_FACE *dst_faces = result_geometry.faces.buffer;
    size_t vertex_offset = 0;
    size_t face_offset = 0;
    size_t visual_offset = 0;
    const POINT *src_points = geometry->vertices.buffer;
    const MESH_FACE *src_faces = geometry->faces.buffer;

    for (uint32_t i = 0; i < scene->object_count; ++i) {
        const OBJECT *object = &scene->objects[i];

        if (object->state != STATIC || object->type != MODEL || !object->data) continue;

        const struct MODEL *model = object->data;
        const size_t object_vertices = model->geometry->vertices.count;
        const size_t object_faces = model->geometry->faces.count;
        const size_t object_visual_vertices = model->visual->vertex_count;

        memcpy(&dst_points[vertex_offset], &src_points[object->geometry_vertex_offset], object_vertices * sizeof(*dst_points));

        for (size_t face = 0; face < object_faces; ++face) {
            MESH_FACE copied = src_faces[object->geometry_face_offset + face];

            for (uint32_t corner = 0; corner < 3u; ++corner) {
                if (copied.indices[corner] < object->geometry_vertex_offset) goto fail;

                const uint32_t local = copied.indices[corner] - object->geometry_vertex_offset;

                if (local >= object_vertices || vertex_offset + local > UINT32_MAX) goto fail;
                copied.indices[corner] = (uint32_t)vertex_offset + local;
            }

            dst_faces[face_offset + face] = copied;
        }

        memcpy(&result_visual.vertices[visual_offset], &visual->vertices[object->visual_vertex_offset],
               object_visual_vertices * sizeof(*result_visual.vertices));

        vertex_offset += object_vertices;
        face_offset += object_faces;
        visual_offset += object_visual_vertices;
    }

    scene_mesh_bounds(&result_geometry);
    *out_geometry = result_geometry;
    *out_visual = result_visual;
    return true;

fail:
    mesh_free(&result_geometry);
    gltf_free(&result_visual);
    return false;
}

bool scene_compile(SCENE *scene) {
    if (!scene) return false;
    if (scene->compiled) return true;

    size_t mesh_vertex_count = 0;
    size_t mesh_face_count = 0;
    size_t visual_vertex_count = 0;
    uint32_t material_count = 0;
    uint32_t texture_count = 0;
    uint32_t image_count = 0;
    uint32_t model_count = 0;

    for (uint32_t i = 0; i < scene->object_count; ++i) {
        const OBJECT *object = &scene->objects[i];

        if (object->type != MODEL || !object->data) continue;

        const struct MODEL *model = object->data;
        if (!model->geometry || !model->visual || !model->geometry->vertices.count || !model->geometry->faces.count || !model->visual->vertex_count ||
            model->visual->vertex_count / 3u != model->geometry->faces.count || !model->visual->material_count)
            return false;

        if (!add_size(&mesh_vertex_count, model->geometry->vertices.count) || !add_size(&mesh_face_count, model->geometry->faces.count) ||
            !add_size(&visual_vertex_count, model->visual->vertex_count) || !add_u32(&material_count, model->visual->material_count) ||
            !add_u32(&texture_count, model->visual->texture_count) || !add_u32(&image_count, model->visual->image_count))
            return false;

        ++model_count;
    }

    if (!model_count || mesh_vertex_count > UINT32_MAX || mesh_face_count > UINT32_MAX) return false;

    MESH geometry = {
        .vertices = {.count = mesh_vertex_count, .capacity = mesh_vertex_count, .type_size = sizeof(POINT)},
        .faces = {.count = mesh_face_count, .capacity = mesh_face_count, .type_size = sizeof(MESH_FACE)},
    };

    GLTF_SCENE visual = {
        .vertex_count = visual_vertex_count,
        .vertex_capacity = visual_vertex_count,
        .material_count = material_count,
        .texture_count = texture_count,
        .image_count = image_count,
        .default_material = UINT32_MAX,
    };

    geometry.vertices.buffer = calloc(mesh_vertex_count, sizeof(POINT));
    geometry.faces.buffer = calloc(mesh_face_count, sizeof(MESH_FACE));
    visual.vertices = calloc(visual_vertex_count, sizeof(*visual.vertices));
    visual.materials = calloc(material_count, sizeof(*visual.materials));
    visual.textures = texture_count ? calloc(texture_count, sizeof(*visual.textures)) : NULL;
    visual.images = image_count ? calloc(image_count, sizeof(*visual.images)) : NULL;

    if (!geometry.vertices.buffer || !geometry.faces.buffer || !visual.vertices || !visual.materials || (texture_count && !visual.textures) ||
        (image_count && !visual.images))
        goto fail;

    size_t mesh_vertex_offset = 0;
    size_t mesh_face_offset = 0;
    size_t visual_vertex_offset = 0;
    uint32_t material_offset = 0;
    uint32_t texture_offset = 0;
    uint32_t image_offset = 0;

    POINT *points = geometry.vertices.buffer;
    MESH_FACE *faces = geometry.faces.buffer;

    for (uint32_t object_index = 0; object_index < scene->object_count; ++object_index) {
        OBJECT *object = &scene->objects[object_index];

        if (object->type != MODEL || !object->data) continue;

        object->geometry_vertex_offset = (uint32_t)mesh_vertex_offset;
        object->geometry_face_offset = (uint32_t)mesh_face_offset;
        object->visual_vertex_offset = (uint32_t)visual_vertex_offset;
        object->material_offset = material_offset;

        const struct MODEL *model = object->data;
        const MESH *source_geometry = model->geometry;
        const GLTF_SCENE *source_visual = model->visual;
        const POINT *source_points = source_geometry->vertices.buffer;
        const MESH_FACE *source_faces = source_geometry->faces.buffer;

        for (size_t i = 0; i < source_geometry->vertices.count; ++i)
            points[mesh_vertex_offset + i].p = transform_position(object->transform, source_points[i].p);

        for (size_t i = 0; i < source_geometry->faces.count; ++i) {
            MESH_FACE face = source_faces[i];

            for (uint32_t j = 0; j < 3u; ++j) {
                if (face.indices[j] >= source_geometry->vertices.count || mesh_vertex_offset + face.indices[j] > UINT32_MAX) goto fail;
                face.indices[j] += (uint32_t)mesh_vertex_offset;
            }

            const VEC3 a = points[face.indices[0]].p;
            const VEC3 b = points[face.indices[1]].p;
            const VEC3 c = points[face.indices[2]].p;
            face.normal = v3_normalize(v3_cross(v3_sub(b, a), v3_sub(c, a)));
            faces[mesh_face_offset + i] = face;
        }

        for (size_t i = 0; i < source_visual->vertex_count; ++i) {
            GLTF_VERTEX vertex = source_visual->vertices[i];

            if (vertex.material >= source_visual->material_count || vertex.material > UINT32_MAX - material_offset) goto fail;

            vertex.position = transform_position(object->transform, vertex.position);
            vertex.normal = transform_normal(object->transform, vertex.normal);
            vertex.material += material_offset;
            visual.vertices[visual_vertex_offset + i] = vertex;
        }

        for (uint32_t i = 0; i < source_visual->material_count; ++i) {
            GLTF_MATERIAL material = source_visual->materials[i];

            if (!material_texture_offset(&material, texture_offset)) goto fail;

            visual.materials[material_offset + i] = material;
        }

        for (uint32_t i = 0; i < source_visual->texture_count; ++i) {
            GLTF_TEXTURE texture = source_visual->textures[i];

            if (texture.image >= 0) {
                if ((uint32_t)texture.image > (uint32_t)INT32_MAX - image_offset) goto fail;
                texture.image += (int32_t)image_offset;
            }

            visual.textures[texture_offset + i] = texture;
        }

        if (source_visual->image_count)
            memcpy(&visual.images[image_offset], source_visual->images, (size_t)source_visual->image_count * sizeof(*visual.images));

        if (visual.default_material == UINT32_MAX && source_visual->default_material < source_visual->material_count)
            visual.default_material = material_offset + source_visual->default_material;

        mesh_vertex_offset += source_geometry->vertices.count;
        mesh_face_offset += source_geometry->faces.count;
        visual_vertex_offset += source_visual->vertex_count;
        material_offset += source_visual->material_count;
        texture_offset += source_visual->texture_count;
        image_offset += source_visual->image_count;
    }

    if (visual.default_material == UINT32_MAX) visual.default_material = 0u;

    scene_mesh_bounds(&geometry);

    MESH static_geometry = {0};
    GLTF_SCENE static_visual = {0};

    if (!scene_extract_static(scene, &geometry, &visual, &static_geometry, &static_visual)) goto fail;

    mesh_free(&scene->geometry);
    gltf_free(&scene->visual);
    mesh_free(&scene->static_geometry);
    gltf_free(&scene->static_visual);
    scene->geometry = geometry;
    scene->visual = visual;
    scene->static_geometry = static_geometry;
    scene->static_visual = static_visual;
    scene->compiled = true;

    return true;

fail:
    mesh_free(&geometry);
    gltf_free(&visual);
    return false;
}

bool scene_build_lightmap(SCENE *scene, uint32_t preferred_texels_per_unit, uint32_t max_size) {
    if (!scene || !scene->lightmap || !scene_compile(scene)) return false;

    scene->lightmap_valid = lmap_build(scene->lightmap, &scene->static_geometry, preferred_texels_per_unit, max_size);
    return scene->lightmap_valid;
}

uint64_t scene_content_hash(const SCENE *scene) {
    if (!scene || !scene->compiled) return 0;

    uint64_t hash = 0;
    hash = hash_bytes(hash, &scene->static_geometry.vertices.count, sizeof(scene->static_geometry.vertices.count));
    hash = hash_bytes(hash, scene->static_geometry.vertices.buffer, scene->static_geometry.vertices.count * sizeof(POINT));
    hash = hash_bytes(hash, &scene->static_geometry.faces.count, sizeof(scene->static_geometry.faces.count));
    hash = hash_bytes(hash, scene->static_geometry.faces.buffer, scene->static_geometry.faces.count * sizeof(MESH_FACE));

    hash = hash_bytes(hash, &scene->static_visual.vertex_count, sizeof(scene->static_visual.vertex_count));
    hash = hash_bytes(hash, scene->static_visual.vertices, scene->static_visual.vertex_count * sizeof(*scene->static_visual.vertices));
    hash = hash_bytes(hash, &scene->visual.material_count, sizeof(scene->visual.material_count));
    hash = hash_bytes(hash, scene->visual.materials, (size_t)scene->visual.material_count * sizeof(*scene->visual.materials));
    hash = hash_bytes(hash, &scene->visual.texture_count, sizeof(scene->visual.texture_count));

    if (scene->visual.texture_count)
        hash = hash_bytes(hash, scene->visual.textures, (size_t)scene->visual.texture_count * sizeof(*scene->visual.textures));

    hash = hash_bytes(hash, &scene->visual.image_count, sizeof(scene->visual.image_count));

    for (uint32_t i = 0; i < scene->visual.image_count; ++i) {
        const GLTF_IMAGE *image = &scene->visual.images[i];

        hash = hash_bytes(hash, image->mime, sizeof(image->mime));
        hash = hash_bytes(hash, &image->bytes.size, sizeof(image->bytes.size));

        if (image->bytes.data && image->bytes.size) hash = hash_bytes(hash, image->bytes.data, image->bytes.size);
    }

    return hash;
}

void scene_free(SCENE *scene) {
    if (!scene) return;

    mesh_free(&scene->geometry);
    gltf_free(&scene->visual);
    mesh_free(&scene->static_geometry);
    gltf_free(&scene->static_visual);
    free(scene->objects);

    scene->objects = NULL;
    scene->object_count = 0u;
    scene->object_capacity = 0u;
    scene->compiled = false;
    scene->lightmap_valid = false;
}
