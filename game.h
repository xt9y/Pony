#ifndef GAME_H
#define GAME_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

//// init: math + mesh

typedef struct VEC3 {
    float x, y, z;
} VEC3;

typedef struct VECTOR {
    void *buffer;
    size_t count;
    size_t capacity;
    size_t type_size;
} VECTOR;

typedef struct POINT {
    VEC3 p;
} POINT;

typedef struct MESH_FACE {
    uint32_t indices[3];
    VEC3 normal;
} MESH_FACE;

typedef struct AABB {
    VEC3 min;
    VEC3 max;
    VEC3 center;
    VEC3 extents;
} AABB;

typedef struct MESH {
    VECTOR vertices; /* point */
    VECTOR faces;    /* mesh_face */
    AABB bounds;
} MESH;

VEC3 v3(float x, float y, float z);
VEC3 v3_add(VEC3 a, VEC3 b);
VEC3 v3_sub(VEC3 a, VEC3 b);
VEC3 v3_scale(VEC3 v, float s);
float v3_dot(VEC3 a, VEC3 b);
VEC3 v3_cross(VEC3 a, VEC3 b);
float v3_len_sq(VEC3 v);
VEC3 v3_normalize(VEC3 v);
void mesh_free(MESH *m);

//// glb: container + JSON tokenizer + accessors

typedef enum GLB_TOKEN_TYPE { GLB_TOKEN_OBJECT, GLB_TOKEN_ARRAY, GLB_TOKEN_STRING, GLB_TOKEN_PRIMITIVE } GLB_TOKEN_TYPE;

typedef struct GLB_TOKEN {
    uint32_t start;
    uint32_t end;
    int32_t parent;
    uint32_t children;
    GLB_TOKEN_TYPE type;
} GLB_TOKEN;

typedef struct GLB_DOC {
    unsigned char *data;
    size_t data_size;
    const char *json;
    size_t json_size;
    const unsigned char *bin;
    size_t bin_size;
    GLB_TOKEN *tokens;
    uint32_t token_count;
    uint32_t token_capacity;
    char error[192];
} GLB_DOC;

typedef struct GLB_SPAN {
    const unsigned char *data;
    size_t size;
} GLB_SPAN;

typedef struct GLB_ACCESSOR {
    const unsigned char *data;
    size_t count;
    size_t stride;
    uint32_t component_type;
    uint32_t components;
    bool normalized;
    int token;
    int sparse_token;
} GLB_ACCESSOR;

bool glb_load(GLB_DOC *doc, const char *path);
void glb_free(GLB_DOC *doc);
const char *glb_error(const GLB_DOC *doc);

int glb_root(const GLB_DOC *doc);
int glb_get(const GLB_DOC *doc, int object_token, const char *key);
int glb_at(const GLB_DOC *doc, int array_token, size_t index);
size_t glb_count(const GLB_DOC *doc, int token);
bool glb_string(const GLB_DOC *doc, int token, const char **data, size_t *length);
bool glb_number(const GLB_DOC *doc, int token, double *value);
bool glb_boolean(const GLB_DOC *doc, int token, bool *value);

bool glb_buffer_view(const GLB_DOC *doc, size_t index, GLB_SPAN *span, size_t *stride);
bool glb_accessor_open(const GLB_DOC *doc, size_t index, GLB_ACCESSOR *out);
bool glb_accessor_f32(const GLB_ACCESSOR *accessor, size_t element, uint32_t component, float *value);
bool glb_accessor_u32(const GLB_ACCESSOR *accessor, size_t element, uint32_t *value);

bool glb_extract_mesh(const GLB_DOC *doc, MESH *out);

//// gltf: visual scene (materials, textures, images)

typedef struct GLTF_VERTEX {
    VEC3 position;
    VEC3 normal;

    float u, v;

    uint32_t material;
} GLTF_VERTEX;

typedef struct GLTF_MATERIAL {
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
} GLTF_MATERIAL;

typedef struct GLTF_TEXTURE {
    int32_t image;
} GLTF_TEXTURE;

typedef struct GLTF_IMAGE {
    GLB_SPAN bytes;
    char mime[32];
} GLTF_IMAGE;

typedef struct GLTF_SCENE {
    GLTF_VERTEX *vertices;
    size_t vertex_count;
    size_t vertex_capacity;

    GLTF_MATERIAL *materials;
    uint32_t material_count;
    uint32_t default_material;

    GLTF_TEXTURE *textures;
    uint32_t texture_count;

    GLTF_IMAGE *images;
    uint32_t image_count;
} GLTF_SCENE;

//// object: scene entity + attached components

typedef enum OBJECT_STATE { STATIC, DYNAMIC } OBJECT_STATE;
typedef enum OBJECT_TYPE { MODEL, LIGHT } OBJECT_TYPE;

struct MODEL {
    MESH *geometry;
    GLTF_SCENE *visual;
};

typedef enum LIGHT_TYPE { LIGHT_DIRECTIONAL, LIGHT_POINT, LIGHT_SPOT } LIGHT_TYPE;

typedef struct DIRECTIONAL_LIGHT {
    VEC3 direction;
    VEC3 color;
    float intensity;
    float angular_radius;
} DIRECTIONAL_LIGHT;

typedef struct SKY {
    VEC3 zenith;
    VEC3 horizon;
    float intensity;
} SKY;

typedef struct VOLUMETRICS_LIGHTING {
    float density;
    float anisotropy;
    float probe_intensity;
    float emissive_probe_intensity;
    float max_distance;
    float center_radius;
    float middle_radius;
    float center_transition_width;
    float middle_transition_width;
    float probe_spacing;
    uint32_t center_steps;
    uint32_t middle_steps;
    uint32_t peripheral_steps;
    uint32_t center_stride;
    uint32_t middle_stride;
    uint32_t peripheral_stride;
    uint32_t probe_samples;
    uint32_t emissive_samples;
} VOLUMETRICS_LIGHTING;

typedef struct POINT_LIGHT {
    VEC3 position;
} POINT_LIGHT;

typedef struct SPOT_LIGHT {
    VEC3 position;
    VEC3 direction;
} SPOT_LIGHT;

struct LIGHT {
    LIGHT_TYPE type;

    union {
        DIRECTIONAL_LIGHT directional;
        POINT_LIGHT point;
        SPOT_LIGHT spot;
    };
};

typedef struct OBJECT {
    OBJECT_STATE state;
    OBJECT_TYPE type;
    void *data;
} OBJECT;

bool gltf_extract(const GLB_DOC *doc, GLTF_SCENE *scene);
void gltf_free(GLTF_SCENE *scene);

//// bvh

typedef struct BVH_TRIANGLE {
    float a[4];
    float b[4];
    float c[4];
    float normal[4];
    float emissive[4]; /* rgb radiance + cumulative importance */
} BVH_TRIANGLE;

typedef struct BVH_NODE {
    float min[4];
    float max[4];
    uint32_t meta[4]; /* left, next, first triangle, triangle count */
} BVH_NODE;

typedef struct BVH {
    BVH_NODE *nodes;
    uint32_t node_count;
    uint32_t node_capacity;
    BVH_TRIANGLE *triangles;
    uint32_t triangle_count;
    float emissive_weight;
} BVH;

typedef struct TRACE_RAY {
    VEC3 origin;
    float tmin;
    VEC3 direction;
    float tmax;
} TRACE_RAY;

typedef struct TRACE_HIT {
    float t;
    VEC3 normal;
    VEC3 albedo;
    uint32_t triangle;
} TRACE_HIT;

bool trace_any(const BVH *tree, TRACE_RAY ray);
bool trace_closest(const BVH *tree, TRACE_RAY ray, TRACE_HIT *hit);
bool bvh_build(BVH *tree, const MESH *m, const GLTF_SCENE *visual);
void bvh_free(BVH *tree);

//// lmap: lightmap atlas

typedef struct LMAP_UV {
    float u, v;
} LMAP_UV;

typedef struct LMAP_SAMPLE {
    float position[4]; /* xyz + pixel index bitcast */
    float normal[4];
} LMAP_SAMPLE;

typedef struct LIGHTMAP {
    uint32_t width;
    uint32_t height;
    uint32_t padding;
    uint32_t chart_count;
    float texel_density;
    LMAP_UV *uvs; /* front 3, back 3 per mesh face */
    LMAP_SAMPLE *samples;
    uint32_t sample_count;
} LIGHTMAP;

bool lmap_build(LIGHTMAP *lm, const MESH *m, uint32_t preferred_texels_per_unit, uint32_t max_size);
void lmap_free(LIGHTMAP *lm);

typedef struct PROBE {
    float position[4];        /* xyz and validity */
    float coefficients[9][4]; /* RGB SH9; coefficient[1].w is sun visibility */
} PROBE;

typedef struct PROBE_GRID {
    VEC3 origin;
    float spacing;

    uint32_t count_x, count_y, count_z;

    PROBE *probes;
} PROBE_GRID;

typedef struct BEAM_CELL {
    uint32_t x, y, z, side;
} BEAM_CELL;

typedef struct BEAM_GRID {
    VEC3 origin;
    VEC3 step; /* world-space spacing along the three sun-space axes */
    uint32_t width, height, depth;

    uint32_t count;
    BEAM_CELL *cells;    /* visible quadtree squares; all other voxels are shaded */
    float *shadow_depth; /* first sun-facing surface for each x/y column */
} BEAM_GRID;

bool beam_build(BEAM_GRID *grid, const MESH *scene, const BVH *tree, VEC3 sun_direction);
void beam_free(BEAM_GRID *grid);
float *beam_expand(const BEAM_GRID *grid);

//// cache: baked lighting

typedef struct CACHED_LIGHTMAP {
    uint32_t width;
    uint32_t height;
    uint64_t layout_hash;
    uint64_t volume_hash;
    uint64_t beam_hash;
    unsigned char *pixels; /* tightly packed RGBA16F, width * height * 8 bytes */
    PROBE_GRID object_probes;
    PROBE_GRID volume_probes;
    BEAM_GRID beams;
} CACHED_LIGHTMAP;

uint64_t hash_bytes(uint64_t seed, const void *bytes, size_t size);
bool cache_read(const char *path, uint64_t scene_hash, uint64_t layout_hash, uint64_t volume_hash, uint64_t beam_hash, CACHED_LIGHTMAP *out);
bool cache_read_partial(const char *path, uint64_t scene_hash, CACHED_LIGHTMAP *out);
bool cache_write(const char *path, uint64_t scene_hash, uint64_t layout_hash, uint64_t volume_hash, uint64_t beam_hash, const CACHED_LIGHTMAP *data);
void cache_free(CACHED_LIGHTMAP *data);

#include "gpu.h"

/* Renderer and bake orchestration. */
bool r_init(RENDERER *r, const char *title, int width, int height);
bool r_build_scene(RENDERER *r, const MESH *m, const GLTF_SCENE *visual, const LIGHTMAP *lm);
bool r_load_cached_lightmap(RENDERER *r, const char *path, uint64_t scene_hash, uint64_t layout_hash, uint64_t volume_hash, uint64_t beam_hash, const LIGHTMAP *lm);
bool r_rebake_current_scene(RENDERER *r, const MESH *m, const GLTF_SCENE *visual, const LIGHTMAP *lm, const struct LIGHT *light, const SKY *sky, const VOLUMETRICS_LIGHTING *volumetrics, const char *path, uint64_t scene_hash, uint64_t layout_hash,
                            uint64_t volume_hash, uint64_t beam_hash);
void r_event(RENDERER *r, const SDL_Event *event);
bool r_draw(RENDERER *r, const struct LIGHT *light, const SKY *sky, const VOLUMETRICS_LIGHTING *volumetrics);
void r_deinit(RENDERER *r);

void bake_progress(RENDERER *r, const char *stage, Uint32 done, Uint32 total);
bool bake_start(RENDERER *r, const MESH *scene, const GLTF_SCENE *visual, const LIGHTMAP *layout, const struct LIGHT *light, const SKY *sky, const VOLUMETRICS_LIGHTING *volumetrics, const char *path, uint64_t scene_hash, uint64_t layout_hash, uint64_t volume_hash,
                uint64_t beam_hash);
void bake_update(RENDERER *r);
void bake_cancel(RENDERER *r);
bool bake_active(RENDERER *r);
void bake_update_title(RENDERER *r);

#endif
