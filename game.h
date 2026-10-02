#ifndef GAME_H
#define GAME_H

#include "gpu.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include <SDL3/SDL.h>

typedef struct RENDERER RENDERER;
typedef struct LIGHTMAP LIGHTMAP;
typedef struct BVH_SURFACE_REF BVH_SURFACE_REF;
typedef uint64_t OBJECT_ID;

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
    VECTOR vertices;
    VECTOR faces;
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
void mesh_free(MESH *mesh);

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

    float ior;
    float transmission_factor;
    float thickness_factor;
    float attenuation_color[3];
    float attenuation_distance;

    float iridescence_factor;
    float iridescence_ior;
    float iridescence_thickness_min;
    float iridescence_thickness_max;

    int32_t base_color_texture;
    int32_t metallic_roughness_texture;
    int32_t normal_texture;
    int32_t occlusion_texture;
    int32_t emissive_texture;
    int32_t transmission_texture;
    int32_t thickness_texture;
    int32_t iridescence_texture;
    int32_t iridescence_thickness_texture;
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

bool gltf_extract(const GLB_DOC *doc, GLTF_SCENE *scene);
void gltf_free(GLTF_SCENE *scene);

typedef struct TRANSFORM {
    VEC3 position;
    float rotation[4];
    VEC3 scale;
} TRANSFORM;

typedef enum OBJECT_STATE { STATIC, DYNAMIC } OBJECT_STATE;

typedef enum OBJECT_TYPE { MODEL, LIGHT } OBJECT_TYPE;

struct MODEL {
    MESH *geometry;
    GLTF_SCENE *visual;

    /*
     * Reusable object-space surface parameterization. Multiple instances share
     * this layout, but each DYNAMIC OBJECT owns a separate lighting cache.
     */
    LIGHTMAP *surface_layout;
};

typedef struct MODEL_ASSET {
    GLB_DOC document;
    MESH geometry;
    GLTF_SCENE visual;
    struct MODEL model;
} MODEL_ASSET;

bool model_load(MODEL_ASSET *asset, const char *path);
void model_free(MODEL_ASSET *asset);

typedef enum LIGHT_TYPE { LIGHT_DIRECTIONAL, LIGHT_POINT, LIGHT_SPOT } LIGHT_TYPE;

typedef struct DIRECTIONAL_LIGHT {
    VEC3 direction;
    VEC3 color;
    float intensity;
    float angular_radius;
} DIRECTIONAL_LIGHT;

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
    float probe_spacing;
    uint32_t probe_samples;
    uint32_t emissive_samples;
} VOLUMETRICS_LIGHTING;

typedef struct PERIPHERAL_VISION {
    float center_radius;
    float middle_radius;
    float center_transition_width;
    float middle_transition_width;
    float jitter_strength;
    float volume_blur_strength;
    uint32_t center_steps;
    uint32_t middle_steps;
    uint32_t peripheral_steps;
    uint32_t center_stride;
    uint32_t middle_stride;
    uint32_t peripheral_stride;
} PERIPHERAL_VISION;

typedef struct OBJECT {
    struct SCENE *owner;
    OBJECT_ID id;
    OBJECT_STATE state;
    OBJECT_TYPE type;
    TRANSFORM transform;
    void *data;

    /* Stable identity and split invalidation domains. */
    uint32_t revision;
    uint32_t transform_revision;
    uint32_t lighting_revision;

    /* Offsets into the compiled all-object resource tables. */
    uint32_t geometry_vertex_offset;
    uint32_t geometry_face_offset;
    uint32_t visual_vertex_offset;
    uint32_t material_offset;
} OBJECT;

typedef struct SCENE {
    OBJECT *objects;
    SKY sky;
    VOLUMETRICS_LIGHTING volumetrics;
    PERIPHERAL_VISION vision;
    LIGHTMAP *lightmap;

    /*
     * geometry/visual are the current all-object CPU trace/resource view.
     * static_geometry/static_visual are the permanent bake view and never
     * include DYNAMIC models.
     */
    MESH geometry;
    GLTF_SCENE visual;
    MESH static_geometry;
    GLTF_SCENE static_visual;

    /*
     * Stable triangle ownership maps for the flattened current and static
     * views. BVH reordering copies these refs so a hit can always return to an
     * object-local surface-cache coordinate.
     */
    BVH_SURFACE_REF *surface_refs;
    BVH_SURFACE_REF *static_surface_refs;
    uint32_t surface_ref_count;
    uint32_t static_surface_ref_count;

    float radius;
    OBJECT_ID next_object_id;
    uint32_t object_count;
    uint32_t object_capacity;
    uint32_t geometry_revision;
    uint32_t lighting_revision;
    bool compiled;
    bool lightmap_valid;
} SCENE;

TRANSFORM transform_identity(void);
OBJECT *scene_add_model(SCENE *scene, struct MODEL *model, OBJECT_STATE state, TRANSFORM transform);
OBJECT *scene_add_light(SCENE *scene, struct LIGHT *light, OBJECT_STATE state, TRANSFORM transform);
OBJECT *scene_object_by_id(SCENE *scene, OBJECT_ID id);
const OBJECT *scene_object_by_id_const(const SCENE *scene, OBJECT_ID id);
void object_set_transform(OBJECT *object, TRANSFORM transform);
void object_mark_dirty(OBJECT *object);
void object_mark_lighting_dirty(OBJECT *object);
bool scene_compile(SCENE *scene);
bool scene_build_lightmap(SCENE *scene, uint32_t preferred_texels_per_unit, uint32_t max_size);
uint64_t scene_content_hash(const SCENE *scene);
void scene_free(SCENE *scene);

typedef struct BVH_TRIANGLE {
    float a[4];
    float b[4];
    float c[4];
    float normal[4];
    float emissive[4];
} BVH_TRIANGLE;

struct BVH_SURFACE_REF {
    OBJECT_ID object_id;
    uint32_t local_triangle;
    uint32_t source_triangle;
    uint32_t material;
};

typedef struct BVH_NODE {
    float min[4];
    float max[4];
    uint32_t meta[4];
} BVH_NODE;

typedef struct BVH {
    BVH_NODE *nodes;
    uint32_t node_count;
    uint32_t node_capacity;
    BVH_TRIANGLE *triangles;
    BVH_SURFACE_REF *surfaces;
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
    VEC3 emissive;
    float barycentric[3];
    uint32_t triangle;
    uint32_t source_triangle;
    uint32_t material;
} TRACE_HIT;

bool trace_any(const BVH *tree, TRACE_RAY ray);
bool trace_closest(const BVH *tree, TRACE_RAY ray, TRACE_HIT *hit);
bool bvh_hit_surface_uv(const BVH *tree, const GLTF_SCENE *visual, const TRACE_HIT *hit, float *u, float *v);
bool bvh_build_with_surfaces(BVH *tree, const MESH *mesh, const GLTF_SCENE *visual, const BVH_SURFACE_REF *surfaces, uint32_t surface_count);
bool bvh_build(BVH *tree, const MESH *mesh, const GLTF_SCENE *visual);
void bvh_free(BVH *tree);

typedef struct LMAP_UV {
    float u, v;
} LMAP_UV;

typedef struct LMAP_SAMPLE {
    float position[4];
    float normal[4];
} LMAP_SAMPLE;

struct LIGHTMAP {
    uint32_t width;
    uint32_t height;
    uint32_t padding;
    uint32_t chart_count;
    float texel_density;
    LMAP_UV *uvs;
    LMAP_SAMPLE *samples;
    uint32_t sample_count;
};

bool lmap_build(LIGHTMAP *lightmap, const MESH *mesh, uint32_t preferred_texels_per_unit, uint32_t max_size);
void lmap_free(LIGHTMAP *lightmap);

typedef struct PROBE {
    float position[4];
    float coefficients[9][4];
} PROBE;

typedef struct PROBE_GRID {
    VEC3 origin;
    float spacing;
    uint32_t count_x, count_y, count_z;
    PROBE *probes;
} PROBE_GRID;

typedef bool (*PROBE_BAKE_PROGRESS_FN)(Uint32 done, Uint32 total, Uint32 active);

typedef struct BEAM_CELL {
    uint32_t x, y, z, side;
} BEAM_CELL;

typedef struct BEAM_GRID {
    VEC3 origin;
    VEC3 step;
    uint32_t width, height, depth;
    uint32_t count;
    BEAM_CELL *cells;
    float *shadow_depth;
} BEAM_GRID;

bool beam_build(BEAM_GRID *grid, const MESH *scene, const BVH *tree, VEC3 sun_direction);
void beam_free(BEAM_GRID *grid);
float *beam_expand(const BEAM_GRID *grid);

typedef struct CACHED_LIGHTMAP {
    uint32_t width;
    uint32_t height;
    uint64_t layout_hash;
    uint64_t volume_hash;
    uint64_t beam_hash;
    unsigned char *pixels;
    PROBE_GRID object_probes;
    PROBE_GRID volume_probes;
    BEAM_GRID beams;
} CACHED_LIGHTMAP;

uint64_t hash_bytes(uint64_t seed, const void *bytes, size_t size);
bool cache_read(const char *path, uint64_t scene_hash, uint64_t layout_hash, uint64_t volume_hash, uint64_t beam_hash, CACHED_LIGHTMAP *out);
bool cache_read_partial(const char *path, uint64_t scene_hash, CACHED_LIGHTMAP *out);
bool cache_write(const char *path, uint64_t scene_hash, uint64_t layout_hash, uint64_t volume_hash, uint64_t beam_hash, const CACHED_LIGHTMAP *data);
void cache_free(CACHED_LIGHTMAP *data);

typedef struct PROBE_WAVEFRONT_SCRATCH PROBE_WAVEFRONT_SCRATCH;
typedef struct RENDER_MATERIAL RENDER_MATERIAL;
typedef struct DYNAMIC_LIGHTING_ALLOCATION DYNAMIC_LIGHTING_ALLOCATION;

typedef struct FX_STATE {
    RENDERER *owner;
    NriPipeline *compose_pipeline;
    NriPipeline *ssao_pipeline;
    NriPipeline *bloom_pipeline;
    NriPipeline *grade_pipeline;
    NriPipeline *volume_pipeline;
    NriPipeline *volume_compose_pipeline;
    NriDescriptor *sampler;
    NriDescriptor *depth_sampler;
    NriTexture *hdr;
    NriTexture *scene_color;
    NriTexture *normal_depth;
    NriTexture *ao;
    NriTexture *bloom_a;
    NriTexture *bloom_b;
    NriTexture *lut;
    NriTexture *volume;
    NriTexture *lit;
    Uint32 width, height;
    Uint32 ao_width, ao_height;
    bool volume_ready;
    uint32_t debug_view;
} FX_STATE;

typedef struct RENDER_VERTEX {
    float x, y, z;
    float nx, ny, nz;
    float u, v;
    float lu, lv;
    float r, g, b, a;
} RENDER_VERTEX;

typedef struct DRAW_RANGE {
    uint32_t first;
    uint32_t count;
    uint32_t material;
    OBJECT_ID object_id;
    VEC3 center;
} DRAW_RANGE;

typedef struct RENDER_FRAME {
    float mvp[16];
    float view[16];
    VEC3 eye;
    VEC3 right;
    VEC3 up;
    VEC3 forward;
    DIRECTIONAL_LIGHT sun;
    SKY sky;
    VOLUMETRICS_LIGHTING volumetrics;
    PERIPHERAL_VISION vision;
    float tan_half_fov;
    float aspect;
} RENDER_FRAME;

struct RENDERER {
    GPU *gpu;
    SCENE *scene;

    PROBE_WAVEFRONT_SCRATCH *probe_scratch;

    NriPipelineLayout *surface_layout;
    NriPipelineLayout *line_layout;
    NriPipelineLayout *sky_layout;
    NriPipelineLayout *bake_layout;
    NriPipelineLayout *lightmap_queue_reset_layout;
    NriPipelineLayout *lightmap_queue_args_layout;
    NriPipelineLayout *probe_layout;
    NriPipelineLayout *ssao_layout;
    NriPipelineLayout *bloom_layout;
    NriPipelineLayout *grade_layout;
    NriPipelineLayout *volume_layout;
    NriPipelineLayout *volume_compose_layout;
    NriPipelineLayout *compose_layout;
    NriPipelineLayout *current_graphics_layout;
    NriPipelineLayout *current_compute_layout;

    NriPipeline *sky_pipeline;
    NriPipeline *solid_pipeline;
    NriPipeline *transmission_pipeline;
    NriPipeline *line_pipeline;
    NriPipeline *bake_pipeline;
    NriPipeline *lightmap_queue_reset_pipeline;
    NriPipeline *lightmap_queue_args_pipeline;

    NriBuffer *vertex_buffer;
    NriBuffer *bvh_node_buffer;
    NriBuffer *bvh_triangle_buffer;
    uint32_t bvh_triangle_count;
    float bvh_emissive_weight;
    NriBuffer *lightmap_sample_buffer;
    NriBuffer *lightmap_full_sample_buffer;
    NriBuffer *lightmap_sparse_sample_buffer;
    NriBuffer *lightmap_probe_buffer;
    NriBuffer *lightmap_patch_map_buffer;
    NriBuffer *lightmap_patch_anchor_buffer;
    NriBuffer *lightmap_active_buffer[2];
    NriBuffer *lightmap_active_count[2];
    NriBuffer *lightmap_dispatch_args;
    uint64_t lightmap_active_capacity;

    NriTexture *depth_texture;
    NriTexture *lightmap_texture;
    NriTexture *lightmap_scratch;
    NriTexture *lightmap_direct;
    NriDescriptor *lightmap_sampler;
    NriDescriptor *material_sampler;
    NriFormat depth_format;
    Uint32 depth_width;
    Uint32 depth_height;

    NriTexture **image_textures;
    uint32_t image_texture_count;
    NriTexture *default_white;
    NriTexture *default_normal;

    RENDER_MATERIAL *materials;
    uint32_t material_count;
    bool has_transmission;
    DRAW_RANGE *draws;
    uint32_t draw_count;
    DRAW_RANGE *transmission_draws;
    uint32_t transmission_draw_count;

    DYNAMIC_LIGHTING_ALLOCATION *dynamic_lighting;
    uint32_t dynamic_lighting_count;

    /*
     * Slow correctness oracle for dynamic-lighting development. When enabled,
     * these caches are rebuilt from the current full scene with probe reuse
     * disabled, so stationary and moving-pose captures can be compared
     * against the same transport equations as the bake.
     */
    NriTexture *reference_static_texture;
    uint32_t reference_geometry_revision;
    uint32_t reference_lighting_revision;
    bool reference_lighting_enabled;
    bool bake_full_transport;

    RENDER_VERTEX *vertices;
    uint32_t vertex_count;
    uint32_t vertex_capacity;
    uint32_t debug_vertex_start;
    uint32_t debug_vertex_count;

    uint32_t lightmap_width;
    uint32_t lightmap_height;
    uint32_t lightmap_sample_count;
    uint32_t lightmap_trace_count;
    uint32_t bake_target_samples;
    uint32_t lightmap_min_samples;
    VEC3 lightmap_probe_origin;
    float lightmap_probe_spacing;
    uint32_t lightmap_probe_count_x;
    uint32_t lightmap_probe_count_y;
    uint32_t lightmap_probe_count_z;
    float bake_epsilon;
    bool has_bake;
    const char *bake_stage;
    PROBE_GRID volume_probes;
    NriBuffer *volume_probe_buffer;
    NriBuffer *beam_buffer;
    BEAM_GRID beams;

    FX_STATE fx;

    float yaw;
    float pitch;
    float distance;
    float scene_radius;
    double frame_time_ms;
    VEC3 target;
    DIRECTIONAL_LIGHT sun;
    SKY sky;
    VOLUMETRICS_LIGHTING volumetrics;

    bool dragging;
    bool show_debug;
    bool show_volume;
    uint32_t debug_view;
};

bool renderer_init(RENDERER *renderer, GPU *gpu);
bool renderer_set_scene(RENDERER *renderer, SCENE *scene);
void renderer_event(RENDERER *renderer, const SDL_Event *event);
bool renderer_frame(RENDERER *renderer);
void renderer_deinit(RENDERER *renderer);

bool renderer_load_cached_lightmap(RENDERER *renderer, const char *path, uint64_t scene_hash, uint64_t layout_hash, uint64_t volume_hash, uint64_t beam_hash,
                                   const LIGHTMAP *lightmap);
bool renderer_rebake_current_scene(RENDERER *renderer, const MESH *mesh, const GLTF_SCENE *visual, const LIGHTMAP *lightmap, const struct LIGHT *light,
                                   const SKY *sky, const VOLUMETRICS_LIGHTING *volumetrics, const char *path, uint64_t scene_hash, uint64_t layout_hash,
                                   uint64_t volume_hash, uint64_t beam_hash);

void bake_progress(RENDERER *renderer, const char *stage, Uint32 done, Uint32 total);
bool bake_start(RENDERER *renderer, const SCENE *scene, const char *path, uint64_t scene_hash, uint64_t layout_hash, uint64_t volume_hash, uint64_t beam_hash);
void bake_update(RENDERER *renderer);
void bake_cancel(RENDERER *renderer);
bool bake_active(RENDERER *renderer);
void bake_update_title(RENDERER *renderer);

#endif
