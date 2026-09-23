#include "dustmite.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define LMAP_PADDING 3u
#define LMAP_MIN_DENSITY 1.0f
#define LMAP_CHART_DOT 0.984807753f


typedef struct edge_ref {
    uint32_t a, b, face;
} edge_ref;

typedef struct chart {
    vec3 normal;
    float min_u, min_v;
    float max_u, max_v;
    uint32_t x, y;
    uint32_t inner_w, inner_h;
    uint32_t rect_w, rect_h;
} chart;

static const chart *g_sort_charts;

static uint32_t next_pow2(uint32_t v) {

    if (v <= 1u) return 1u;
    --v;

    v |= v >> 1u;
    v |= v >> 2u;
    v |= v >> 4u;
    v |= v >> 8u;
    v |= v >> 16u;
    return v + 1u;
}

static uint32_t uf_find(uint32_t *parent, uint32_t x) {

    uint32_t root = x;
    while (parent[root] != root)
        root = parent[root];

    while (parent[x] != x) {
        uint32_t next = parent[x];
        parent[x] = root;
        x = next;
    }

    return root;
}

static void uf_join(uint32_t *parent, uint8_t *rank, uint32_t a, uint32_t b) {

    a = uf_find(parent, a);
    b = uf_find(parent, b);
    if (a == b) return;
    if (rank[a] < rank[b]) {

        uint32_t t = a;
        a = b;
        b = t;
    }

    parent[b] = a;
    if (rank[a] == rank[b]) ++rank[a];

}

static int edge_compare(const void *lhs, const void *rhs) {

    const edge_ref *a = lhs;
    const edge_ref *b = rhs;

    if (a->a != b->a) return a->a < b->a ? -1 : 1;
    if (a->b != b->b) return a->b < b->b ? -1 : 1;
    if (a->face != b->face) return a->face < b->face ? -1 : 1;

    return 0;
}

static int chart_order_compare(const void *lhs, const void *rhs) {

    const uint32_t a = *(const uint32_t *)lhs;
    const uint32_t b = *(const uint32_t *)rhs;

    if (g_sort_charts[a].rect_h != g_sort_charts[b].rect_h) return g_sort_charts[a].rect_h > g_sort_charts[b].rect_h ? -1 : 1;
    if (g_sort_charts[a].rect_w != g_sort_charts[b].rect_w) return g_sort_charts[a].rect_w > g_sort_charts[b].rect_w ? -1 : 1;

    return a < b ? -1 : (a > b ? 1 : 0);
}

static vec3 geometric_normal(const point *points, mesh_face face) {

    const vec3 a = points[face.indices[0]].p;
    const vec3 b = points[face.indices[1]].p;
    const vec3 c = points[face.indices[2]].p;

    return v3_normalize(v3_cross(v3_sub(b, a), v3_sub(c, a)));
}

static void project_point(vec3 normal, vec3 p, float *u, float *v) {

    const float ax = fabsf(normal.x);
    const float ay = fabsf(normal.y);
    const float az = fabsf(normal.z);

    if (ax >= ay && ax >= az) {
        *u = p.z;
        *v = p.y;
    } else if (ay >= az) {
        *u = p.x;
        *v = p.z;
    } else {
        *u = p.x;
        *v = p.y;
    }

}

static bool pack_charts(chart *charts, uint32_t chart_count, float density, uint32_t max_size, uint32_t *out_width, uint32_t *out_height) {

    uint64_t total_area = 0;
    uint32_t largest_width = 1;

    for (uint32_t i = 0; i < chart_count; ++i) {

        const float world_w = fmaxf(charts[i].max_u - charts[i].min_u, 1.0e-4f);
        const float world_h = fmaxf(charts[i].max_v - charts[i].min_v, 1.0e-4f);

        charts[i].inner_w = (uint32_t)ceilf(world_w * density) + 1u;
        charts[i].inner_h = (uint32_t)ceilf(world_h * density) + 1u;
        if (charts[i].inner_w < 2u) charts[i].inner_w = 2u;
        if (charts[i].inner_h < 2u) charts[i].inner_h = 2u;

        charts[i].rect_w = charts[i].inner_w + LMAP_PADDING * 2u;
        charts[i].rect_h = charts[i].inner_h + LMAP_PADDING * 2u;
        if (charts[i].rect_w > max_size || charts[i].rect_h > max_size) return false;
        total_area += (uint64_t)charts[i].rect_w * charts[i].rect_h;
        if (charts[i].rect_w > largest_width) largest_width = charts[i].rect_w;

    }

    uint32_t *order = malloc((size_t)chart_count * sizeof(*order));
    if (!order && chart_count) return false;
    for (uint32_t i = 0; i < chart_count; ++i)
        order[i] = i;

    g_sort_charts = charts;
    qsort(order, chart_count, sizeof(*order), chart_order_compare);
    g_sort_charts = NULL;


    uint32_t start = next_pow2(largest_width);
    const uint32_t area_side = next_pow2((uint32_t)ceil(sqrt((double)total_area)));

    if (start < area_side) start = area_side;
    if (start < 256u) start = 256u;


    bool packed = false;
    uint32_t packed_width = 0;
    uint32_t packed_height = 0;


    for (uint32_t width = start; width <= max_size; width <<= 1u) {

        uint32_t x = 0, y = 0, row_h = 0;
        bool failed = false;

        for (uint32_t oi = 0; oi < chart_count; ++oi) {

            chart *c = &charts[order[oi]];

            if (x + c->rect_w > width) {
                x = 0;
                y += row_h;
                row_h = 0;
            }

            if (y + c->rect_h > max_size) {
                failed = true;
                break;
            }

            c->x = x;
            c->y = y;
            x += c->rect_w;

            if (c->rect_h > row_h) row_h = c->rect_h;
        }

        if (!failed) {

            const uint32_t used_h = y + row_h;
            const uint32_t height = next_pow2(used_h ? used_h : 1u);
            if (height <= max_size) {
                packed = true;
                packed_width = width;
                packed_height = height;
                break;
            }

        }
        if (width > max_size / 2u) break;
    }

    free(order);
    if (!packed) return false;
    *out_width = packed_width;
    *out_height = packed_height;
    
    return true;
}

static bool barycentric(float px, float py, lmap_uv a, lmap_uv b, lmap_uv c, float *w0, float *w1, float *w2) {

    const float den = (b.v - c.v) * (a.u - c.u) + (c.u - b.u) * (a.v - c.v);
    if (fabsf(den) < 1.0e-8f) return false;

    *w0 = ((b.v - c.v) * (px - c.u) + (c.u - b.u) * (py - c.v)) / den;
    *w1 = ((c.v - a.v) * (px - c.u) + (a.u - c.u) * (py - c.v)) / den;
    *w2 = 1.0f - *w0 - *w1;

    const float eps = -1.0e-4f;
    return *w0 >= eps && *w1 >= eps && *w2 >= eps;
}

static void vertex_normals(const mesh *m, vec3 *normals) {

    const point *points = m->vertices.buffer;
    const mesh_face *faces = m->faces.buffer;
    memset(normals, 0, m->vertices.count * sizeof(*normals));

    for (size_t i = 0; i < m->faces.count; ++i) {

        const mesh_face f = faces[i];
        if (f.indices[0] >= m->vertices.count || f.indices[1] >= m->vertices.count || f.indices[2] >= m->vertices.count) continue;

        const vec3 a = points[f.indices[0]].p;
        const vec3 b = points[f.indices[1]].p;
        const vec3 c = points[f.indices[2]].p;
        const vec3 weighted = v3_cross(v3_sub(b, a), v3_sub(c, a));


        normals[f.indices[0]] = v3_add(normals[f.indices[0]], weighted);
        normals[f.indices[1]] = v3_add(normals[f.indices[1]], weighted);
        normals[f.indices[2]] = v3_add(normals[f.indices[2]], weighted);
    }

    for (size_t i = 0; i < m->vertices.count; ++i)
        normals[i] = v3_normalize(normals[i]);

}

void lmap_free(lightmap *lm) {
    if (!lm) return;

    free(lm->uvs);
    free(lm->samples);

    memset(lm, 0, sizeof(*lm));
}

bool lmap_build(lightmap *lm, const mesh *m, uint32_t preferred_texels_per_unit, uint32_t max_size) {

    if (!lm || !m || !m->faces.count || !m->vertices.count || m->faces.count > UINT32_MAX || m->vertices.count > UINT32_MAX || max_size < 256u) return false;

    lmap_free(lm);

    const uint32_t face_count = (uint32_t)m->faces.count;
    const uint32_t vertex_count = (uint32_t)m->vertices.count;
    const point *points = m->vertices.buffer;
    const mesh_face *faces = m->faces.buffer;


    uint32_t *parent = malloc((size_t)face_count * sizeof(*parent));
    uint8_t *rank = calloc(face_count, sizeof(*rank));
    edge_ref *edges = malloc((size_t)face_count * 3u * sizeof(*edges));

    uint32_t *root_chart = malloc((size_t)face_count * sizeof(*root_chart));
    uint32_t *face_chart = malloc((size_t)face_count * sizeof(*face_chart));
    chart *charts = calloc(face_count, sizeof(*charts));
    vec3 *normals = malloc((size_t)vertex_count * sizeof(*normals));

    if (!parent || !rank || !edges || !root_chart || !face_chart || !charts || !normals) goto fail;

    for (uint32_t i = 0; i < face_count; ++i) {

        parent[i] = i;
        root_chart[i] = UINT32_MAX;

        const mesh_face f = faces[i];
        uint32_t e[3][2] = {
            {f.indices[0], f.indices[1]}, 
            {f.indices[1], f.indices[2]}, 
            {f.indices[2], f.indices[0]}
        };

        for (uint32_t j = 0; j < 3u; ++j) {

            if (e[j][0] > e[j][1]) {

                uint32_t t = e[j][0];
                e[j][0] = e[j][1];
                e[j][1] = t;
            }
            edges[i * 3u + j] = (edge_ref){e[j][0], e[j][1], i};

        }

    }

    qsort(edges, (size_t)face_count * 3u, sizeof(*edges), edge_compare);
    for (size_t first = 0; first < (size_t)face_count * 3u;) {

        size_t end = first + 1u;

        while (end < (size_t)face_count * 3u && edges[end].a == edges[first].a && edges[end].b == edges[first].b)
            ++end;

        for (size_t a = first; a < end; ++a) {

            const vec3 na = geometric_normal(points, faces[edges[a].face]);
            for (size_t b = a + 1u; b < end; ++b) {

                const vec3 nb = geometric_normal(points, faces[edges[b].face]);
                if (v3_dot(na, nb) >= LMAP_CHART_DOT) uf_join(parent, rank, edges[a].face, edges[b].face);
            }

        }

        first = end;

    }


    uint32_t chart_count = 0;
    for (uint32_t i = 0; i < face_count; ++i) {

        const uint32_t root = uf_find(parent, i);
        if (root_chart[root] == UINT32_MAX) root_chart[root] = chart_count++;

        face_chart[i] = root_chart[root];
        charts[face_chart[i]].normal = v3_add(charts[face_chart[i]].normal, geometric_normal(points, faces[i]));

    }

    for (uint32_t i = 0; i < chart_count; ++i) {
        charts[i].normal = v3_normalize(charts[i].normal);
        charts[i].min_u = charts[i].min_v = INFINITY;
        charts[i].max_u = charts[i].max_v = -INFINITY;

    }

    for (uint32_t i = 0; i < face_count; ++i) {

        chart *c = &charts[face_chart[i]];
        for (uint32_t k = 0; k < 3u; ++k) {

            const uint32_t vi = faces[i].indices[k];
            if (vi >= vertex_count) goto fail;

            float u, v;
            project_point(c->normal, points[vi].p, &u, &v);

            if (u < c->min_u) c->min_u = u;
            if (v < c->min_v) c->min_v = v;
            if (u > c->max_u) c->max_u = u;
            if (v > c->max_v) c->max_v = v;

        }

    }


    float density = preferred_texels_per_unit ? (float)preferred_texels_per_unit : 16.0f;
    uint32_t width = 0, height = 0;

    while (density >= LMAP_MIN_DENSITY && !pack_charts(charts, chart_count, density, max_size, &width, &height)) {
        density *= 0.80f;
    }

    if (!width || !height) goto fail;

    lm->uvs = malloc((size_t)face_count * 3u * sizeof(*lm->uvs));
    if (!lm->uvs) goto fail;

    lm->width = width;
    lm->height = height;
    lm->padding = LMAP_PADDING;
    lm->chart_count = chart_count;
    lm->texel_density = density;


    for (uint32_t i = 0; i < face_count; ++i) {

        const chart *c = &charts[face_chart[i]];
        const float max_x = (float)(c->x + LMAP_PADDING + c->inner_w) - 0.5f;
        const float max_y = (float)(c->y + LMAP_PADDING + c->inner_h) - 0.5f;
        for (uint32_t k = 0; k < 3u; ++k) {

            float u, v;

            project_point(c->normal, points[faces[i].indices[k]].p, &u, &v);
            float px = (float)(c->x + LMAP_PADDING) + 0.5f + (u - c->min_u) * density;
            float py = (float)(c->y + LMAP_PADDING) + 0.5f + (v - c->min_v) * density;

            if (px > max_x) px = max_x;
            if (py > max_y) py = max_y;
            lm->uvs[i * 3u + k] = (lmap_uv){px / (float)width, py / (float)height};
        }
    }

    if ((uint64_t)width * height > SIZE_MAX) goto fail;
    const size_t pixel_count = (size_t)width * height;
    uint8_t *occupied = calloc(pixel_count, 1u);
    if (!occupied) goto fail;

    uint32_t sample_count = 0;
    for (uint32_t i = 0; i < face_count; ++i) {

        lmap_uv uv[3];
        for (uint32_t k = 0; k < 3u; ++k) {

            uv[k] = lm->uvs[i * 3u + k];
            uv[k].u *= width;
            uv[k].v *= height;
        }

        int min_x = (int)floorf(fminf(uv[0].u, fminf(uv[1].u, uv[2].u)));
        int min_y = (int)floorf(fminf(uv[0].v, fminf(uv[1].v, uv[2].v)));
        int max_x = (int)ceilf(fmaxf(uv[0].u, fmaxf(uv[1].u, uv[2].u)));
        int max_y = (int)ceilf(fmaxf(uv[0].v, fmaxf(uv[1].v, uv[2].v)));

        if (min_x < 0) min_x = 0;
        if (min_y < 0) min_y = 0;
        if (max_x >= (int)width) max_x = (int)width - 1;
        if (max_y >= (int)height) max_y = (int)height - 1;

        for (int y = min_y; y <= max_y; ++y) {

            for (int x = min_x; x <= max_x; ++x) {

                float w0, w1, w2;
                
                if (!barycentric((float)x + 0.5f, (float)y + 0.5f, uv[0], uv[1], uv[2], &w0, &w1, &w2)) continue;
                const size_t pixel = (size_t)y * width + (uint32_t)x;
                if (!occupied[pixel]) {
                    occupied[pixel] = 1u;
                    ++sample_count;
                
                }
            
            }
        
        }
    
    }


    lm->samples = malloc((size_t)sample_count * sizeof(*lm->samples));
    if (!lm->samples && sample_count) {
        free(occupied);
        goto fail;
    }
    lm->sample_count = sample_count;
    memset(occupied, 0, pixel_count);
    vertex_normals(m, normals);


    uint32_t out_sample = 0;
    for (uint32_t i = 0; i < face_count; ++i) {

        const mesh_face f = faces[i];
        lmap_uv uv[3];

        for (uint32_t k = 0; k < 3u; ++k) {

            uv[k] = lm->uvs[i * 3u + k];
            uv[k].u *= width;
            uv[k].v *= height;
        }


        int min_x = (int)floorf(fminf(uv[0].u, fminf(uv[1].u, uv[2].u)));
        int min_y = (int)floorf(fminf(uv[0].v, fminf(uv[1].v, uv[2].v)));
        int max_x = (int)ceilf(fmaxf(uv[0].u, fmaxf(uv[1].u, uv[2].u)));
        int max_y = (int)ceilf(fmaxf(uv[0].v, fmaxf(uv[1].v, uv[2].v)));

        if (min_x < 0) min_x = 0;
        if (min_y < 0) min_y = 0;
        if (max_x >= (int)width) max_x = (int)width - 1;
        if (max_y >= (int)height) max_y = (int)height - 1;


        const vec3 a = points[f.indices[0]].p;
        const vec3 b = points[f.indices[1]].p;
        const vec3 c = points[f.indices[2]].p;
        for (int y = min_y; y <= max_y; ++y) {

            for (int x = min_x; x <= max_x; ++x) {

                float w0, w1, w2;
                if (!barycentric((float)x + 0.5f, (float)y + 0.5f, uv[0], uv[1], uv[2], &w0, &w1, &w2)) continue;
                const size_t pixel = (size_t)y * width + (uint32_t)x;
                if (occupied[pixel]) continue;
                occupied[pixel] = 1u;


                const vec3 p = v3_add(v3_scale(a, w0), v3_add(v3_scale(b, w1), v3_scale(c, w2)));
                vec3 n = v3_add(v3_scale(normals[f.indices[0]], w0), v3_add(v3_scale(normals[f.indices[1]], w1), v3_scale(normals[f.indices[2]], w2)));
                n = v3_normalize(n);
                if (v3_len_sq(n) < 1.0e-10f) n = f.normal;


                lmap_sample *s = &lm->samples[out_sample++];
                s->position[0] = p.x;
                s->position[1] = p.y;
                s->position[2] = p.z;
                union {
                    uint32_t u;
                    float f;
                } bits = {(uint32_t)pixel};
                
                s->position[3] = bits.f;
                s->normal[0] = n.x;
                s->normal[1] = n.y;
                s->normal[2] = n.z;
                s->normal[3] = 0.0f;
            }
        }
    
    }

    free(occupied);
    free(parent);
    free(rank);
    free(edges);
    free(root_chart);
    free(face_chart);
    free(charts);
    free(normals);

    return out_sample == sample_count && sample_count > 0u;


fail:
    free(parent);
    free(rank);
    free(edges);
    free(root_chart);
    free(face_chart);
    free(charts);
    free(normals);
    lmap_free(lm);

    return false;
}
