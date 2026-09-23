#include "dustmite.h"

#include <SDL3_image/SDL_image.h>

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define BVH_LEAF_TRIANGLES 4u
#define BVH_EPSILON 1.0e-6f

typedef struct build_tri {
    bvh_triangle gpu;
    vec3 centroid;
    vec3 min;
    vec3 max;
} build_tri;

static float srgb_linear(float value) {

    return value <= 0.04045f ? value / 12.92f :
        powf((value + 0.055f) / 1.055f, 2.4f);
}

static vec3 texture_color(const gltf_scene *visual, SDL_Surface **images,
                          int32_t texture_index, const gltf_vertex *vertices) {

    if (texture_index < 0 || (uint32_t)texture_index >= visual->texture_count) return v3(1, 1, 1);

    const int32_t image = visual->textures[texture_index].image;
    if (image < 0 || (uint32_t)image >= visual->image_count || !images[image]) return v3(1, 1, 1);

    const SDL_Surface *surface = images[image];
    float u = (vertices[0].u + vertices[1].u + vertices[2].u) / 3.0f;
    float v = (vertices[0].v + vertices[1].v + vertices[2].v) / 3.0f;
    u -= floorf(u);
    v -= floorf(v);

    const uint32_t x = (uint32_t)(u * surface->w);
    const uint32_t y = (uint32_t)(v * surface->h);
    const unsigned char *pixel = (const unsigned char *)surface->pixels +
        (size_t)y * surface->pitch + (size_t)x * 4u;

    return v3(srgb_linear(pixel[0] / 255.0f), srgb_linear(pixel[1] / 255.0f),
              srgb_linear(pixel[2] / 255.0f));
}

static void release_images(SDL_Surface **images, uint32_t count) {

    if (!images) return;

    for (uint32_t i = 0; i < count; ++i) {
        if (images[i]) SDL_DestroySurface(images[i]);
    }

    free(images);
}

static float axis_value(vec3 v, int axis) {
    return axis == 0 ? v.x : axis == 1 ? v.y : v.z;
}

static bool reserve_nodes(bvh *tree, uint32_t count) {

    if (count <= tree->node_capacity) return true;
    uint32_t next = tree->node_capacity ? tree->node_capacity : 64u;

    while (next < count) {

        if (next > UINT32_MAX / 2u) return false;
        next *= 2u;
    }

    bvh_node *p = realloc(tree->nodes, (size_t)next * sizeof(*p));

    if (!p) return false;
    tree->nodes = p;
    tree->node_capacity = next;
    return true;
}

static uint32_t new_node(bvh *tree) {

    if (!reserve_nodes(tree, tree->node_count + 1u)) return UINT32_MAX;
    const uint32_t index = tree->node_count++;
    memset(&tree->nodes[index], 0, sizeof(tree->nodes[index]));
    return index;
}

static void range_bounds(const build_tri *tris, uint32_t first, uint32_t count, vec3 *bmin, vec3 *bmax, vec3 *cmin, vec3 *cmax) {

    *bmin = tris[first].min;
    *bmax = tris[first].max;
    *cmin = *cmax = tris[first].centroid;

    for (uint32_t i = 0; i < count; ++i) {

        const build_tri *t = &tris[first + i];

        if (t->min.x < bmin->x) bmin->x = t->min.x;
        if (t->min.y < bmin->y) bmin->y = t->min.y;
        if (t->min.z < bmin->z) bmin->z = t->min.z;
        if (t->max.x > bmax->x) bmax->x = t->max.x;
        if (t->max.y > bmax->y) bmax->y = t->max.y;
        if (t->max.z > bmax->z) bmax->z = t->max.z;
        if (t->centroid.x < cmin->x) cmin->x = t->centroid.x;
        if (t->centroid.y < cmin->y) cmin->y = t->centroid.y;
        if (t->centroid.z < cmin->z) cmin->z = t->centroid.z;
        if (t->centroid.x > cmax->x) cmax->x = t->centroid.x;
        if (t->centroid.y > cmax->y) cmax->y = t->centroid.y;
        if (t->centroid.z > cmax->z) cmax->z = t->centroid.z;
    }
}

static void swap_tri(build_tri *a, build_tri *b) {

    const build_tri t = *a;
    *a = *b;
    *b = t;
}

static uint32_t partition_range(build_tri *tris, uint32_t first, uint32_t count, int axis, float split) {

    uint32_t i = first;
    uint32_t j = first + count;
    while (i < j) {
        if (axis_value(tris[i].centroid, axis) < split) {
            ++i;
        } else {
            --j;
            swap_tri(&tris[i], &tris[j]);
        }
    }
    return i;

}

static bool build_node(bvh *tree, build_tri *tris, uint32_t node_index, uint32_t first, uint32_t count) {
    vec3 bmin, bmax, cmin, cmax;
    range_bounds(tris, first, count, &bmin, &bmax, &cmin, &cmax);

    bvh_node *node = &tree->nodes[node_index];
    node->min[0] = bmin.x;
    node->min[1] = bmin.y;
    node->min[2] = bmin.z;
    node->max[0] = bmax.x;
    node->max[1] = bmax.y;
    node->max[2] = bmax.z;

    if (count <= BVH_LEAF_TRIANGLES) {
        node->meta[2] = first;
        node->meta[3] = count;
        return true;
    }

    const vec3 extent = v3_sub(cmax, cmin);
    int axis = 0;
    if (extent.y > extent.x) axis = 1;
    if (axis_value(extent, 2) > axis_value(extent, axis)) axis = 2;

    const float lo = axis_value(cmin, axis);
    const float hi = axis_value(cmax, axis);
    uint32_t middle;

    if (hi - lo <= BVH_EPSILON) {
        middle = first + count / 2u;
    } else {
        middle = partition_range(tris, first, count, axis, (lo + hi) * 0.5f);
        if (middle == first || middle == first + count) middle = first + count / 2u;
    }

    const uint32_t left = new_node(tree);
    const uint32_t right = new_node(tree);
    if (left == UINT32_MAX || right == UINT32_MAX) return false;

    node = &tree->nodes[node_index];
    node->meta[0] = left;
    node->meta[1] = right; // temporary: replaced by skip pointer after build
    node->meta[2] = 0;
    node->meta[3] = 0;

    return build_node(tree, tris, left, first, middle - first) && build_node(tree, tris, right, middle, first + count - middle);
}

static void thread_node(bvh *tree, uint32_t node_index, uint32_t next) {

    bvh_node *node = &tree->nodes[node_index];
    if (node->meta[3] != 0u) {
        node->meta[1] = next;
        return;
    }

    const uint32_t left = node->meta[0];
    const uint32_t right = node->meta[1];
    node->meta[1] = next;
    thread_node(tree, left, right);
    thread_node(tree, right, next);
}

void bvh_free(bvh *tree) {
    if (!tree) return;
    free(tree->nodes);
    free(tree->triangles);
    memset(tree, 0, sizeof(*tree));
}

bool bvh_build(bvh *tree, const mesh *m, const gltf_scene *visual) {

    if (!tree || !m || !m->faces.count || !m->vertices.count || m->faces.count > UINT32_MAX) return false;

    bvh_free(tree);
    const uint32_t count = (uint32_t)m->faces.count;
    build_tri *build = calloc(count, sizeof(*build));
    if (!build) return false;

    SDL_Surface **images = NULL;
    if (visual && visual->image_count) {

        images = calloc(visual->image_count, sizeof(*images));
        if (!images) {
            free(build);
            return false;
        }

        for (uint32_t i = 0; i < visual->image_count; ++i) {

            const gltf_image *source = &visual->images[i];
            if (!source->bytes.data || !source->bytes.size) continue;

            SDL_IOStream *io = SDL_IOFromConstMem(source->bytes.data, source->bytes.size);
            if (!io) continue;

            SDL_Surface *decoded = IMG_Load_IO(io, true);
            if (decoded) {
                images[i] = SDL_ConvertSurface(decoded, SDL_PIXELFORMAT_RGBA32);
                SDL_DestroySurface(decoded);
            }

            if (!images[i]) SDL_Log("bake: could not decode texture %u: %s", i, SDL_GetError());
        }
    }

    const point *points = m->vertices.buffer;
    const mesh_face *faces = m->faces.buffer;
    for (uint32_t i = 0; i < count; ++i) {

        const mesh_face f = faces[i];
        if (f.indices[0] >= m->vertices.count || f.indices[1] >= m->vertices.count || f.indices[2] >= m->vertices.count) {
            free(build);
            release_images(images, visual ? visual->image_count : 0);
            return false;
        }

        const vec3 a = points[f.indices[0]].p;
        const vec3 b = points[f.indices[1]].p;
        const vec3 c = points[f.indices[2]].p;

        vec3 n = f.normal;
        if (v3_dot(n, n) <= BVH_EPSILON) n = v3_normalize(v3_cross(v3_sub(b, a), v3_sub(c, a)));

        vec3 albedo = v3(0.72f, 0.72f, 0.72f);
        if (visual && i < visual->vertex_count / 3u) {
            uint32_t material = visual->vertices[i * 3u].material;
            if (material < visual->material_count) {
                const gltf_material *mat = &visual->materials[material];
                albedo = v3(mat->base_color[0] * (1.0f - mat->metallic),
                            mat->base_color[1] * (1.0f - mat->metallic),
                            mat->base_color[2] * (1.0f - mat->metallic));
                if (images) {

                    vec3 tex = texture_color(visual, images, mat->base_color_texture,
                                             &visual->vertices[i * 3u]);
                    albedo = v3(albedo.x * tex.x, albedo.y * tex.y, albedo.z * tex.z);
                }
            }
        }
        build[i].gpu = (bvh_triangle){.a = {a.x, a.y, a.z, albedo.x},
            .b = {b.x, b.y, b.z, albedo.y}, .c = {c.x, c.y, c.z, albedo.z},
            .normal = {n.x, n.y, n.z, 0.0f}};
        build[i].centroid = v3_scale(v3_add(v3_add(a, b), c), 1.0f / 3.0f);
        build[i].min = v3(fminf(a.x, fminf(b.x, c.x)), fminf(a.y, fminf(b.y, c.y)), fminf(a.z, fminf(b.z, c.z)));
        build[i].max = v3(fmaxf(a.x, fmaxf(b.x, c.x)), fmaxf(a.y, fmaxf(b.y, c.y)), fmaxf(a.z, fmaxf(b.z, c.z)));
    }

    release_images(images, visual ? visual->image_count : 0);

    tree->triangle_count = count;
    const uint32_t root = new_node(tree);
    if (root == UINT32_MAX || !build_node(tree, build, root, 0, count)) {
        free(build);
        bvh_free(tree);
        return false;
    }

    tree->triangles = malloc((size_t)count * sizeof(*tree->triangles));
    if (!tree->triangles) {
        free(build);
        bvh_free(tree);
        return false;
    }
    for (uint32_t i = 0; i < count; ++i)
        tree->triangles[i] = build[i].gpu;
    free(build);

    thread_node(tree, 0u, UINT32_MAX);
    return true;

}
