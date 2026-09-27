#include "game.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define BEAM_SIDE 64u
#define BEAM_MIN_SLICES 16u
#define BEAM_MAX_SLICES 128u
#define BEAM_TARGET_Z_STEP 1.0f
#define BEAM_TILE 4u

static vec3 beam_u(vec3 sun) {

    return v3_normalize(v3_cross(v3(0, 1, 0), sun));

}

#if 0
/* Reference exact-visibility path. */
static bool intersects_box(vec3 p, vec3 d, const bvh_node *node) {

    float lo = 0.0f;
    float hi = 1.0e20f;
    const float point[3] = {p.x, p.y, p.z};
    const float direction[3] = {d.x, d.y, d.z};

    for (uint32_t axis = 0; axis < 3u; ++axis) {

        if (fabsf(direction[axis]) < 1.0e-7f) {
            if (point[axis] < node->min[axis] || point[axis] > node->max[axis]) return false;
        } else {
            const float a = (node->min[axis] - point[axis]) / direction[axis];
            const float b = (node->max[axis] - point[axis]) / direction[axis];
            lo = fmaxf(lo, fminf(a, b));
            hi = fminf(hi, fmaxf(a, b));
            if (lo > hi) return false;
        }
    }

    return true;
}

static bool intersects_triangle(vec3 p, vec3 d, const bvh_triangle *t) {

    const vec3 a = v3(t->a[0], t->a[1], t->a[2]);
    const vec3 b = v3(t->b[0], t->b[1], t->b[2]);
    const vec3 c = v3(t->c[0], t->c[1], t->c[2]);
    const vec3 e1 = v3_sub(b, a);
    const vec3 e2 = v3_sub(c, a);
    const vec3 q = v3_cross(d, e2);
    const float determinant = v3_dot(e1, q);

    if (fabsf(determinant) < 1.0e-7f) return false;

    const float inverse = 1.0f / determinant;
    const vec3 s = v3_sub(p, a);
    const float u = v3_dot(s, q) * inverse;
    if (u < 0.0f || u > 1.0f) return false;

    const vec3 r = v3_cross(s, e1);
    const float v = v3_dot(d, r) * inverse;
    return v >= 0.0f && u + v <= 1.0f && v3_dot(e2, r) * inverse > 0.001f;
}

static bool shaded(const bvh *tree, vec3 p, vec3 sun) {

    uint32_t node = 0;

    while (node != UINT32_MAX) {

        const bvh_node *n = &tree->nodes[node];
        if (!intersects_box(p, sun, n)) {
            node = n->meta[1];
            continue;
        }

        if (n->meta[3]) {
            for (uint32_t i = 0; i < n->meta[3]; ++i) {
                if (intersects_triangle(p, sun, &tree->triangles[n->meta[2] + i])) return true;
            }
            node = n->meta[1];
        } else {
            node = n->meta[0];
        }
    }

    return false;
}
#endif

static uint32_t beam_depth_for_span(float span) {

    uint32_t depth = (uint32_t)ceilf(fmaxf(span, 0.001f) / BEAM_TARGET_Z_STEP);
    if (depth < BEAM_MIN_SLICES) depth = BEAM_MIN_SLICES;
    if (depth > BEAM_MAX_SLICES) depth = BEAM_MAX_SLICES;
    return depth;
}

/* Orthographic depth buffer looking from the sun toward the scene. The
 * largest sun-space z is the first surface a ray from the sun encounters. */
static void raster_depth(float *depth, const beam_grid *grid, const bvh *tree,
                         vec3 u, vec3 v, vec3 sun) {

    const size_t columns = (size_t)grid->width * grid->height;
    for (size_t n = 0; n < columns; ++n) depth[n] = -INFINITY;

    for (uint32_t n = 0; n < tree->triangle_count; ++n) {
        const bvh_triangle *t = &tree->triangles[n];
        const vec3 vertices[3] = {
            v3(t->a[0], t->a[1], t->a[2]),
            v3(t->b[0], t->b[1], t->b[2]),
            v3(t->c[0], t->c[1], t->c[2])
        };
        float x[3], y[3], z[3];
        for (uint32_t i = 0; i < 3u; ++i) {
            x[i] = (v3_dot(vertices[i], u) - grid->origin.x) / grid->step.x - 0.5f;
            y[i] = (v3_dot(vertices[i], v) - grid->origin.y) / grid->step.y - 0.5f;
            z[i] = v3_dot(vertices[i], sun);
        }

        const float area = (x[1] - x[0]) * (y[2] - y[0]) -
                           (y[1] - y[0]) * (x[2] - x[0]);
        if (fabsf(area) < 1.0e-7f) continue;

        const float xmin = fminf(x[0], fminf(x[1], x[2]));
        const float xmax = fmaxf(x[0], fmaxf(x[1], x[2]));
        const float ymin = fminf(y[0], fminf(y[1], y[2]));
        const float ymax = fmaxf(y[0], fmaxf(y[1], y[2]));
        if (xmax < 0.0f || ymax < 0.0f || xmin >= grid->width || ymin >= grid->height) continue;

        const int left = (int)fmaxf(0.0f, ceilf(xmin));
        const int right = (int)fminf((float)grid->width - 1.0f, floorf(xmax));
        const int top = (int)fmaxf(0.0f, ceilf(ymin));
        const int bottom = (int)fminf((float)grid->height - 1.0f, floorf(ymax));

        for (int py = top; py <= bottom; ++py) {
            for (int px = left; px <= right; ++px) {
                const float b = ((px - x[0]) * (y[2] - y[0]) -
                                 (py - y[0]) * (x[2] - x[0])) / area;
                const float c = ((x[1] - x[0]) * (py - y[0]) -
                                 (y[1] - y[0]) * (px - x[0])) / area;
                if (b < -1.0e-6f || c < -1.0e-6f || b + c > 1.0f + 1.0e-6f) continue;

                const float hit = z[0] + b * (z[1] - z[0]) + c * (z[2] - z[0]);
                float *pixel = &depth[px + (size_t)grid->width * py];
                *pixel = fmaxf(*pixel, hit);
            }
        }
    }
}

/* Classify a coarse tile only when all of its depth pixels agree. A
 * silhouette, an opening, or a depth change falls back to exact BVH rays. */
typedef struct depth_tile_info {
    float nearest, farthest;
    uint32_t covered;
} depth_tile_info;

static depth_tile_info measure_depth_tile(const float *depth, const beam_grid *grid,
                                          uint32_t x, uint32_t y) {
    depth_tile_info info = {INFINITY, -INFINITY, 0};
    for (uint32_t j = y; j < y + BEAM_TILE; ++j) {
        for (uint32_t i = x; i < x + BEAM_TILE; ++i) {
            const float d = depth[i + (size_t)grid->width * j];
            if (d == -INFINITY) continue;
            ++info.covered;
            info.nearest = fminf(info.nearest, d);
            info.farthest = fmaxf(info.farthest, d);
        }
    }
    return info;
}

static int depth_tile(depth_tile_info info, const beam_grid *grid, uint32_t z) {
    if (!info.covered) return 1;
    if (info.covered != BEAM_TILE * BEAM_TILE ||
        info.farthest - info.nearest > grid->step.z * 0.5f) return -1;

    const float point = grid->origin.z + (z + 0.5f) * grid->step.z;
    const float guard = fmaxf(0.002f, grid->step.z * 0.01f);
    if (info.nearest - point > guard) return 0;
    if (info.farthest - point < -guard) return 1;
    return -1;
}

static bool emit(beam_grid *grid, uint32_t x, uint32_t y, uint32_t z, uint32_t side) {

    const uint64_t max_cells = (uint64_t)grid->width * grid->height * grid->depth;
    if (grid->count >= max_cells) return false;

    if (!grid->count || (grid->count >= 64u && !(grid->count & (grid->count - 1u)))) {
        const uint32_t capacity = grid->count ? grid->count * 2u : 64u;
        beam_cell *next = realloc(grid->cells, (size_t)capacity * sizeof(*next));
        if (!next) return false;
        grid->cells = next;
    }

    grid->cells[grid->count++] = (beam_cell){x, y, z, side};
    return true;
}

static bool compress(beam_grid *grid, const uint32_t *prefix,
                     uint32_t x, uint32_t y, uint32_t z, uint32_t side) {
    const size_t stride = (size_t)grid->width + 1u;
    const uint32_t visible = prefix[(y + side) * stride + x + side] -
                             prefix[y * stride + x + side] -
                             prefix[(y + side) * stride + x] + prefix[y * stride + x];

    if (!visible) return true;
    if (visible == side * side) return emit(grid, x, y, z, side);

    const uint32_t half = side / 2u;
    return half && compress(grid, prefix, x, y, z, half) &&
           compress(grid, prefix, x + half, y, z, half) &&
           compress(grid, prefix, x, y + half, z, half) &&
           compress(grid, prefix, x + half, y + half, z, half);
}

static void prefix_slice(uint32_t *prefix, const unsigned char *samples,
                         uint32_t width, uint32_t height) {
    const size_t stride = (size_t)width + 1u;
    memset(prefix, 0, stride * sizeof(*prefix));
    for (uint32_t y = 0; y < height; ++y) {
        prefix[(size_t)(y + 1u) * stride] = 0;
        uint32_t row = 0;
        for (uint32_t x = 0; x < width; ++x) {
            row += samples[x + (size_t)width * y];
            prefix[(size_t)(y + 1u) * stride + x + 1u] =
                prefix[(size_t)y * stride + x + 1u] + row;
        }
    }
}

void beam_free(beam_grid *grid) {

    if (!grid) return;

    free(grid->cells);
    free(grid->shadow_depth);
    memset(grid, 0, sizeof(*grid));
}

bool beam_build(beam_grid *grid, const mesh *scene, const bvh *tree,
                   vec3 sun_direction) {

    if (!grid || !scene || !tree || !tree->node_count) return false;
    memset(grid, 0, sizeof(*grid));

    const vec3 sun = v3_normalize(sun_direction);
    const vec3 u = beam_u(sun);
    const vec3 v = v3_cross(sun, u);
    if (v3_len_sq(u) < 0.5f) return false;

    vec3 min = v3(INFINITY, INFINITY, INFINITY);
    vec3 max = v3(-INFINITY, -INFINITY, -INFINITY);

    for (uint32_t corner = 0; corner < 8u; ++corner) {
        const vec3 p = v3(corner & 1u ? scene->bounds.max.x : scene->bounds.min.x,
                          corner & 2u ? scene->bounds.max.y : scene->bounds.min.y,
                          corner & 4u ? scene->bounds.max.z : scene->bounds.min.z);
        const vec3 q = v3(v3_dot(p, u), v3_dot(p, v), v3_dot(p, sun));
        min = v3(fminf(min.x, q.x), fminf(min.y, q.y), fminf(min.z, q.z));
        max = v3(fmaxf(max.x, q.x), fmaxf(max.y, q.y), fmaxf(max.z, q.z));
    }

    const float z_span = fmaxf(max.z - min.z, 0.001f);
    grid->width = grid->height = BEAM_SIDE;
    grid->depth = beam_depth_for_span(z_span);
    grid->origin = min;
    grid->step = v3(fmaxf((max.x - min.x) / grid->width, 0.001f),
                    fmaxf((max.y - min.y) / grid->height, 0.001f),
                    fmaxf(z_span / grid->depth, 0.001f));

    const size_t sample_count = (size_t)grid->width * grid->height * grid->depth;
    const size_t depth_count = (size_t)grid->width * grid->height;
    unsigned char *samples = calloc(sample_count, 1);
    float *depth = malloc(depth_count * sizeof(*depth));
    depth_tile_info *tiles = malloc(depth_count / (BEAM_TILE * BEAM_TILE) * sizeof(*tiles));
    uint32_t *prefix = malloc((grid->width + 1u) * (grid->height + 1u) * sizeof(*prefix));
    if (!samples || !depth || !tiles || !prefix) {
        free(samples);
        free(depth);
        free(tiles);
        free(prefix);
        return false;
    }

    SDL_Log("B: sun beam grid %ux%ux%u, %.3f world units/slice",
            grid->width, grid->height, grid->depth, grid->step.z);

    raster_depth(depth, grid, tree, u, v, sun);
    for (uint32_t y = 0; y < grid->height; y += BEAM_TILE)
        for (uint32_t x = 0; x < grid->width; x += BEAM_TILE)
            tiles[x / BEAM_TILE + (size_t)(grid->width / BEAM_TILE) * (y / BEAM_TILE)] =
                measure_depth_tile(depth, grid, x, y);
    uint32_t traced = 0;

    for (uint32_t z = 0; z < grid->depth; ++z) {
        for (uint32_t y = 0; y < grid->height; y += BEAM_TILE) {
            for (uint32_t x = 0; x < grid->width; x += BEAM_TILE) {
                const depth_tile_info info = tiles[x / BEAM_TILE +
                    (size_t)(grid->width / BEAM_TILE) * (y / BEAM_TILE)];
                const int coarse = depth_tile(info, grid, z);
                for (uint32_t j = y; j < y + BEAM_TILE; ++j) {
                    for (uint32_t i = x; i < x + BEAM_TILE; ++i) {
                        unsigned char visible;
                        if (coarse >= 0) {
                            visible = (unsigned char)coarse;
                        } else {
                            const vec3 q = v3(min.x + (i + 0.5f) * grid->step.x,
                                              min.y + (j + 0.5f) * grid->step.y,
                                              min.z + (z + 0.5f) * grid->step.z);
                            const vec3 p = v3_add(v3_add(v3_scale(u, q.x), v3_scale(v, q.y)),
                                                  v3_scale(sun, q.z));
                            const trace_ray ray = {
                                .origin = p,
                                .tmin = 0.001f,
                                .direction = sun,
                                .tmax = 1.0e20f
                            };
                            visible = !trace_any(tree, ray);
                            ++traced;
                        }
                        samples[i + (size_t)grid->width * (j + (size_t)grid->height * z)] = visible;
                    }
                }
            }
        }
    }

    SDL_Log("B: sun depth pass selected %u/%zu exact visibility rays", traced, sample_count);

    bool good = true;
    for (uint32_t z = 0; z < grid->depth && good; ++z) {
        prefix_slice(prefix, samples + (size_t)grid->width * grid->height * z,
                     grid->width, grid->height);
        good = compress(grid, prefix, 0, 0, z, grid->width);
    }

    free(samples);
    free(tiles);
    free(prefix);
    if (good) grid->shadow_depth = depth;
    else {
        free(depth);
        beam_free(grid);
    }
    return good;
}

float *beam_expand(const beam_grid *grid) {

    if (!grid || !grid->width || !grid->height || !grid->depth ||
        grid->width > 128u || grid->height > 128u || grid->depth > BEAM_MAX_SLICES ||
        (grid->count && !grid->cells)) return NULL;

    const size_t total = (size_t)grid->width * grid->height * grid->depth;
    float *samples = calloc(total, sizeof(*samples));
    if (!samples) return NULL;

    for (uint32_t n = 0; n < grid->count; ++n) {
        const beam_cell c = grid->cells[n];
        if (!c.side || c.side > grid->width || c.x > grid->width - c.side ||
            c.y > grid->height - c.side || c.z >= grid->depth) {
            free(samples);
            return NULL;
        }

        for (uint32_t y = c.y; y < c.y + c.side; ++y) {
            for (uint32_t x = c.x; x < c.x + c.side; ++x) {
                samples[x + (size_t)grid->width * (y + (size_t)grid->height * c.z)] = 1.0f;
            }
        }
    }

    return samples;
}
