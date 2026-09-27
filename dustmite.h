#ifndef DUSTMITE_H
#define DUSTMITE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

//// init: math + mesh

typedef struct vec3 {
    float x, y, z;
} vec3;

typedef struct vector {
    void *buffer;
    size_t count;
    size_t capacity;
    size_t type_size;
} vector;

typedef struct point {
    vec3 p;
} point;

typedef struct mesh_face {
    uint32_t indices[3];
    vec3 normal;
} mesh_face;

typedef struct aabb {
    vec3 min;
    vec3 max;
    vec3 center;
    vec3 extents;
} aabb;

typedef struct mesh {
    vector vertices; /* point */
    vector faces;    /* mesh_face */
    aabb bounds;
} mesh;

vec3 v3(float x, float y, float z);
vec3 v3_add(vec3 a, vec3 b);
vec3 v3_sub(vec3 a, vec3 b);
vec3 v3_scale(vec3 v, float s);
float v3_dot(vec3 a, vec3 b);
vec3 v3_cross(vec3 a, vec3 b);
float v3_len_sq(vec3 v);
vec3 v3_normalize(vec3 v);
void mesh_free(mesh *m);

//// glb: container + JSON tokenizer + accessors

typedef enum glb_token_type {
    GLB_TOKEN_OBJECT,
    GLB_TOKEN_ARRAY,
    GLB_TOKEN_STRING,
    GLB_TOKEN_PRIMITIVE
} glb_token_type;

typedef struct glb_token {
    uint32_t start;
    uint32_t end;
    int32_t parent;
    uint32_t children;
    glb_token_type type;
} glb_token;

typedef struct glb_doc {
    unsigned char *data;
    size_t data_size;
    const char *json;
    size_t json_size;
    const unsigned char *bin;
    size_t bin_size;
    glb_token *tokens;
    uint32_t token_count;
    uint32_t token_capacity;
    char error[192];
} glb_doc;

typedef struct glb_span {
    const unsigned char *data;
    size_t size;
} glb_span;

typedef struct glb_accessor {
    const unsigned char *data;
    size_t count;
    size_t stride;
    uint32_t component_type;
    uint32_t components;
    bool normalized;
    int token;
    int sparse_token;
} glb_accessor;

bool glb_load(glb_doc *doc, const char *path);
void glb_free(glb_doc *doc);
const char *glb_error(const glb_doc *doc);

int glb_root(const glb_doc *doc);
int glb_get(const glb_doc *doc, int object_token, const char *key);
int glb_at(const glb_doc *doc, int array_token, size_t index);
size_t glb_count(const glb_doc *doc, int token);
bool glb_string(const glb_doc *doc, int token, const char **data, size_t *length);
bool glb_number(const glb_doc *doc, int token, double *value);
bool glb_boolean(const glb_doc *doc, int token, bool *value);

bool glb_buffer_view(const glb_doc *doc, size_t index, glb_span *span,
                     size_t *stride);
bool glb_accessor_open(const glb_doc *doc, size_t index, glb_accessor *out);
bool glb_accessor_f32(const glb_accessor *accessor, size_t element,
                      uint32_t component, float *value);
bool glb_accessor_u32(const glb_accessor *accessor, size_t element,
                      uint32_t *value);

bool glb_extract_mesh(const glb_doc *doc, mesh *out);

//// gltf: visual scene (materials, textures, images)

typedef struct gltf_vertex {
    vec3 position;
    vec3 normal;
    float u, v;
    uint32_t material;
} gltf_vertex;

typedef struct gltf_material {
    float base_color[4];
    float emissive[3];
    float metallic;
    float roughness;
    float normal_scale;
    float occlusion_strength;
    int32_t base_color_texture;
    int32_t metallic_roughness_texture;
    int32_t normal_texture;
    int32_t occlusion_texture;
    int32_t emissive_texture;
} gltf_material;

typedef struct gltf_texture {
    int32_t image;
} gltf_texture;

typedef struct gltf_image {
    glb_span bytes;
    char mime[32];
} gltf_image;

typedef struct gltf_scene {
    gltf_vertex *vertices;
    size_t vertex_count;
    size_t vertex_capacity;

    gltf_material *materials;
    uint32_t material_count;
    uint32_t default_material;

    gltf_texture *textures;
    uint32_t texture_count;

    gltf_image *images;
    uint32_t image_count;
} gltf_scene;

bool gltf_extract(const glb_doc *doc, gltf_scene *scene);
void gltf_free(gltf_scene *scene);

//// bvh

typedef struct bvh_triangle {
    float a[4];
    float b[4];
    float c[4];
    float normal[4];
} bvh_triangle;

typedef struct bvh_node {
    float min[4];
    float max[4];
    uint32_t meta[4]; /* left, next, first triangle, triangle count */
} bvh_node;

typedef struct bvh {
    bvh_node *nodes;
    uint32_t node_count;
    uint32_t node_capacity;
    bvh_triangle *triangles;
    uint32_t triangle_count;
} bvh;

typedef struct dm_trace_ray {
    vec3 origin;
    float tmin;
    vec3 direction;
    float tmax;
} dm_trace_ray;

typedef struct dm_trace_hit {
    float t;
    vec3 normal;
    vec3 albedo;
    uint32_t triangle;
} dm_trace_hit;

bool dm_trace_any(const bvh *tree, dm_trace_ray ray);
bool dm_trace_closest(const bvh *tree, dm_trace_ray ray, dm_trace_hit *hit);
bool bvh_build(bvh *tree, const mesh *m, const gltf_scene *visual);
void bvh_free(bvh *tree);

//// lmap: lightmap atlas

typedef struct lmap_uv {
    float u, v;
} lmap_uv;

typedef struct lmap_sample {
    float position[4]; /* xyz + pixel index bitcast */
    float normal[4];
} lmap_sample;

typedef struct lightmap {
    uint32_t width;
    uint32_t height;
    uint32_t padding;
    uint32_t chart_count;
    float texel_density;
    lmap_uv *uvs; /* front 3, back 3 per mesh face */
    lmap_sample *samples;
    uint32_t sample_count;
} lightmap;

bool lmap_build(lightmap *lm, const mesh *m,
                uint32_t preferred_texels_per_unit, uint32_t max_size);
void lmap_free(lightmap *lm);

typedef struct dm_probe {
    float position[4]; /* xyz and validity */
    float coefficients[9][4]; /* RGB SH9; coefficient[1].w is sun visibility */
} dm_probe;

typedef struct dm_probe_grid {
    vec3 origin;
    float spacing;
    uint32_t count_x, count_y, count_z;
    dm_probe *probes;
} dm_probe_grid;

typedef struct dm_beam_cell {
    uint32_t x, y, z, side;
} dm_beam_cell;

typedef struct dm_beam_grid {
    vec3 origin;
    vec3 step; /* world-space spacing along the three sun-space axes */
    uint32_t width, height, depth;
    uint32_t count;
    dm_beam_cell *cells; /* visible quadtree squares; all other voxels are shaded */
    float *shadow_depth; /* first sun-facing surface for each x/y column */
} dm_beam_grid;

bool dm_beam_build(dm_beam_grid *grid, const mesh *scene, const bvh *tree,
                   vec3 sun_direction);
void dm_beam_free(dm_beam_grid *grid);
float *dm_beam_expand(const dm_beam_grid *grid);

//// cache: baked lighting

typedef struct dm_cached_lightmap {
    uint32_t width;
    uint32_t height;
    uint64_t layout_hash;
    uint64_t volume_hash;
    uint64_t beam_hash;
    unsigned char *pixels; /* tightly packed RGBA16F, width * height * 8 bytes */
    dm_probe_grid object_probes;
    dm_probe_grid volume_probes;
    dm_beam_grid beams;
} dm_cached_lightmap;

uint64_t dm_hash_bytes(uint64_t seed, const void *bytes, size_t size);
bool dm_cache_read(const char *path, uint64_t scene_hash, uint64_t layout_hash,
                   uint64_t volume_hash, uint64_t beam_hash,
                   dm_cached_lightmap *out);
bool dm_cache_read_partial(const char *path, uint64_t scene_hash,
                           dm_cached_lightmap *out);
bool dm_cache_write(const char *path, uint64_t scene_hash, uint64_t layout_hash,
                    uint64_t volume_hash, uint64_t beam_hash,
                    const dm_cached_lightmap *data);
void dm_cache_free(dm_cached_lightmap *data);

/* Keep existing source includes working while gpu.h owns all GPU declarations. */
#include "gpu.h"

#endif
