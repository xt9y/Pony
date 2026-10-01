#ifndef GAME_H
#define GAME_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include <SDL3/SDL.h>

#include "NRIDescs.h"
#include "gpu.h"


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
    uint32_t material;
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
void mesh_free(MESH *m);


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
};

typedef enum LIGHT_TYPE { LIGHT_DIRECTIONAL, LIGHT_POINT, LIGHT_SPOT } LIGHT_TYPE;

typedef struct DIRECTIONAL_LIGHT {
    VEC3 direction;
    VEC3 color;
    float intensity;
    float angular_radius;
} DIRECTIONAL_LIGHT;

typedef struct POINT_LIGHT {
    VEC3 color;
    float intensity;
    float range;
} POINT_LIGHT;

typedef struct SPOT_LIGHT {
    VEC3 direction;
    VEC3 color;
    float intensity;
    float range;
    float inner_angle;
    float outer_angle;
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

typedef struct OBJECT {
    OBJECT_STATE state;
    OBJECT_TYPE type;
    TRANSFORM transform;
    void *data;
    uint32_t revision;
} OBJECT;

typedef struct SCENE {
    OBJECT *objects;
    SKY sky;
    float radius;
    uint32_t object_count;
    uint32_t object_capacity;
} SCENE;

TRANSFORM transform_identity(void);
OBJECT *scene_add_model(SCENE *scene, struct MODEL *model, OBJECT_STATE state, TRANSFORM transform);
OBJECT *scene_add_light(SCENE *scene, struct LIGHT *light, OBJECT_STATE state, TRANSFORM transform);
void object_set_transform(OBJECT *object, TRANSFORM transform);
void object_mark_dirty(OBJECT *object);
void scene_free(SCENE *scene);


typedef struct SDF_VOLUME {
    float *distance;
    uint32_t *surface_id;
    uint32_t resolution;
    AABB bounds;
} SDF_VOLUME;

bool sdf_build_volume(const MESH *mesh, uint32_t resolution, SDF_VOLUME *volume);
void sdf_free_volume(SDF_VOLUME *volume);


typedef enum TRACE_HIT_TYPE { TRACE_INACTIVE = 0, TRACE_MISS = 1, TRACE_SCREEN = 2, TRACE_SDF = 3, TRACE_GLOBAL_SDF = 4, TRACE_TRIANGLE = 5 } TRACE_HIT_TYPE;

typedef struct TRACE_HIT {
    uint32_t type;
    uint32_t object_id;
    uint32_t hit_x;
    uint32_t hit_y;
    float distance;
    float confidence;
    uint32_t padding[2];
} TRACE_HIT;

typedef enum TRACE_RAY_TYPE { TRACE_RAY_INACTIVE = 0, TRACE_RAY_DIFFUSE = 1, TRACE_RAY_REFLECTION = 2, TRACE_RAY_SHADOW = 3, TRACE_RAY_WORLD_PROBE = 4 } TRACE_RAY_TYPE;

typedef struct TRACE_RAY {
    float origin_tmin[4];
    float direction_tmax[4];
    uint32_t type;
    uint32_t destination;
    uint32_t origin_pixel;
    uint32_t source_object_id;
} TRACE_RAY;

typedef struct GPU_SDF_MODEL {
    MAT4 world_to_local;
    float bounds_min[4];
    float bounds_max[4];
    uint32_t voxel_offset;
    uint32_t resolution;
    uint32_t object_id;
    uint32_t state;
    uint32_t revision;
    uint32_t padding[3];
} GPU_SDF_MODEL;

typedef struct GPU_LIGHT {
    float position_range[4];
    float direction_type[4];
    float color_intensity[4];
    float spot_angles[4];
} GPU_LIGHT;

typedef struct SURFACE_CACHE_ENTRY {
    float position[4];
    float normal[4];
    float albedo_roughness[4];
    float emissive_metallic[4];
    float direct_radiance[4];
    float indirect_radiance[4];
    uint32_t object_id;
    uint32_t revision;
    uint32_t last_frame;
    uint32_t confidence;
} SURFACE_CACHE_ENTRY;

/* Permanent Pony Radiance GPU ABI. Future stages populate these layouts without changing shader.hlsl. */
typedef struct GPU_SCENE_TRIANGLE {
    float p0[4];
    float p1[4];
    float p2[4];
    float uv01[4];
    float uv2_area[4];
    uint32_t meta[4];
} GPU_SCENE_TRIANGLE;

typedef struct GPU_EMISSIVE_TRIANGLE {
    uint32_t meta[4];
    float radiance_area[4];
    float sampling[4];
} GPU_EMISSIVE_TRIANGLE;

typedef struct RADIANCE_SCENE_DATA {
    GPU_SCENE_TRIANGLE *cpu_triangles;
    GPU_EMISSIVE_TRIANGLE *cpu_emissive_triangles;

    NriBuffer *triangles;
    NriBuffer *emissive_triangles;

    NriDescriptor *triangles_srv;
    NriDescriptor *emissive_triangles_srv;

    NriAccessStage triangles_state;
    NriAccessStage emissive_triangles_state;

    uint32_t triangle_count;
    uint32_t emissive_triangle_count;
} RADIANCE_SCENE_DATA;


typedef struct RADIANCE_SCENE_FALLBACKS {
    NriBuffer *dynamic_grid_cells;
    NriBuffer *dynamic_grid_indices;
    NriBuffer *global_sdf_clipmaps;
    NriBuffer *global_sdf_page_table;
    NriBuffer *global_sdf_bricks;
    NriBuffer *global_sdf_surface_ids;

    NriDescriptor *dynamic_grid_cells_srv;
    NriDescriptor *dynamic_grid_indices_srv;
    NriDescriptor *global_sdf_clipmaps_srv;
    NriDescriptor *global_sdf_page_table_srv;
    NriDescriptor *global_sdf_bricks_srv;
    NriDescriptor *global_sdf_surface_ids_srv;

    NriAccessStage state;
} RADIANCE_SCENE_FALLBACKS;

typedef struct GPU_DYNAMIC_GRID_CELL {
    uint32_t range_flags[4];
    float bounds_min[4];
    float bounds_max[4];
} GPU_DYNAMIC_GRID_CELL;

typedef struct GPU_GLOBAL_SDF_CLIPMAP {
    float center_extent[4];
    float voxel_brick[4];
    uint32_t grid[4];
    uint32_t data[4];
} GPU_GLOBAL_SDF_CLIPMAP;

typedef struct SURFACE_HIT {
    float position_distance[4];
    float normal_confidence[4];
    float uv_bary[4];
    uint32_t identity[4];
    uint32_t meta[4];
} SURFACE_HIT;

typedef struct SURFACE_RADIANCE_ENTRY {
    float position_distance[4];
    float normal_confidence[4];
    float albedo_roughness[4];
    float emissive_metallic[4];
    float direct_radiance[4];
    float indirect_radiance[4];
    uint32_t identity[4];
    uint32_t state[4];
} SURFACE_RADIANCE_ENTRY;

typedef struct SCREEN_PROBE_STATE {
    float position_depth[4];
    float normal_confidence[4];
    uint32_t history[4];
    float statistics[4];
} SCREEN_PROBE_STATE;

typedef struct WORLD_PROBE_STATE {
    float position_radius[4];
    uint32_t identity[4];
    float statistics[4];
    uint32_t state[4];
} WORLD_PROBE_STATE;

typedef struct RAY_BUDGET {
    uint32_t counts[4];
    float priority[4];
} RAY_BUDGET;

typedef struct RADIANCE_CONSTANTS {
    uint32_t scene_counts[4];
    uint32_t sdf_counts[4];
    uint32_t cache_counts[4];
    uint32_t probe_config[4];
    float trace_params[4];
    uint32_t trace_limits[4];
    float temporal_params[4];
    uint32_t feature_flags[4];
    float adaptive_params[4];
    float reflection_params[4];
    uint32_t dynamic_grid[4];
    float dynamic_grid_origin_cell[4];
    float global_sdf_params[4];
    uint32_t world_probe_config[4];
    float world_probe_params[4];
    uint32_t reserved[4];
} RADIANCE_CONSTANTS;

typedef struct PASS_CONSTANTS {
    uint32_t dispatch[4];
    uint32_t range[4];
    uint32_t dimensions[4];
    uint32_t flags[4];
} PASS_CONSTANTS;

_Static_assert(sizeof(GPU_SCENE_TRIANGLE) == 96u, "GPU_SCENE_TRIANGLE GPU layout changed");
_Static_assert(sizeof(GPU_EMISSIVE_TRIANGLE) == 48u, "GPU_EMISSIVE_TRIANGLE GPU layout changed");
_Static_assert(sizeof(GPU_DYNAMIC_GRID_CELL) == 48u, "GPU_DYNAMIC_GRID_CELL GPU layout changed");
_Static_assert(sizeof(GPU_GLOBAL_SDF_CLIPMAP) == 64u, "GPU_GLOBAL_SDF_CLIPMAP GPU layout changed");
_Static_assert(sizeof(SURFACE_HIT) == 80u, "SURFACE_HIT GPU layout changed");
_Static_assert(sizeof(SURFACE_RADIANCE_ENTRY) == 128u, "SURFACE_RADIANCE_ENTRY GPU layout changed");
_Static_assert(sizeof(SCREEN_PROBE_STATE) == 64u, "SCREEN_PROBE_STATE GPU layout changed");
_Static_assert(sizeof(WORLD_PROBE_STATE) == 64u, "WORLD_PROBE_STATE GPU layout changed");
_Static_assert(sizeof(RAY_BUDGET) == 32u, "RAY_BUDGET GPU layout changed");
_Static_assert(sizeof(RADIANCE_CONSTANTS) == 256u, "RADIANCE_CONSTANTS GPU layout changed");
_Static_assert(sizeof(PASS_CONSTANTS) == 64u, "PASS_CONSTANTS GPU layout changed");

typedef enum RADIANCE_FEATURE {
    RADIANCE_FEATURE_EMISSIVE = 1u << 0,
    RADIANCE_FEATURE_DYNAMIC_GRID = 1u << 1,
    RADIANCE_FEATURE_GLOBAL_SDF = 1u << 2,
    RADIANCE_FEATURE_SURFACE_CACHE = 1u << 3,
    RADIANCE_FEATURE_TEMPORAL_PROBES = 1u << 4,
    RADIANCE_FEATURE_SPATIAL_PROBES = 1u << 5,
    RADIANCE_FEATURE_WORLD_CACHE = 1u << 6,
    RADIANCE_FEATURE_MULTIBOUNCE = 1u << 7,
    RADIANCE_FEATURE_ADAPTIVE_RAYS = 1u << 8,
    RADIANCE_FEATURE_REFLECTIONS = 1u << 9
} RADIANCE_FEATURE;

typedef enum RADIANCE_DEBUG_VIEW {
    RADIANCE_DEBUG_FINAL_GI = 0,
    RADIANCE_DEBUG_ALBEDO,
    RADIANCE_DEBUG_NORMALS,
    RADIANCE_DEBUG_DEPTH,
    RADIANCE_DEBUG_ROUGHNESS,
    RADIANCE_DEBUG_METALLIC,
    RADIANCE_DEBUG_VELOCITY,
    RADIANCE_DEBUG_OBJECT_ID,
    RADIANCE_DEBUG_MATERIAL_ID,
    RADIANCE_DEBUG_PRIMITIVE_ID,
    RADIANCE_DEBUG_HZB,
    RADIANCE_DEBUG_DIRECT_RADIANCE,
    RADIANCE_DEBUG_EMISSIVE,
    RADIANCE_DEBUG_SCREEN_TRACE,
    RADIANCE_DEBUG_LOCAL_SDF,
    RADIANCE_DEBUG_GLOBAL_SDF,
    RADIANCE_DEBUG_SURFACE_CACHE,
    RADIANCE_DEBUG_SCREEN_PROBE_DIRECTIONAL,
    RADIANCE_DEBUG_SCREEN_PROBE_IRRADIANCE,
    RADIANCE_DEBUG_SCREEN_PROBE_CONFIDENCE,
    RADIANCE_DEBUG_SCREEN_PROBE_VARIANCE,
    RADIANCE_DEBUG_SCREEN_PROBE_HISTORY,
    RADIANCE_DEBUG_WORLD_RADIANCE,
    RADIANCE_DEBUG_RAY_BUDGET,
    RADIANCE_DEBUG_REFLECTIONS
} RADIANCE_DEBUG_VIEW;

typedef struct CAMERA {
    VEC3 position;
    VEC3 forward;
    VEC3 up;
    VEC3 target;
    float yaw;
    float pitch;
    float distance;
    float radius;
    bool dragging;
    float fov_y;
    float near_plane;
    float far_plane;
} CAMERA;

#define HZB_MAX_MIPS 16u
#define SDF_DEFAULT_RESOLUTION 32u
#define SURFACE_CACHE_CAPACITY 262144u
#define SCREEN_PROBE_TILE_SIZE 8u
#define SCREEN_PROBE_DIRECTION_SIZE 4u
#define SCREEN_PROBE_DIRECTION_COUNT 8u
#define RADIANCE_MAX_SCREEN_PROBE_DIRECTION_SIZE 8u
#define RADIANCE_MAX_SCREEN_PROBE_RAYS 32u
#define RADIANCE_MAX_GLOBAL_SDF_CLIPMAPS 8u
#define RADIANCE_INVALID_INDEX UINT32_MAX

typedef struct HZB {
    NriTexture *texture;
    NriDescriptor *srv;
    NriDescriptor *mip_srvs[HZB_MAX_MIPS];
    NriDescriptor *mip_uavs[HZB_MAX_MIPS];
    NriAccessLayoutStage mip_states[HZB_MAX_MIPS];
    uint32_t mip_count;
    uint32_t width;
    uint32_t height;
} HZB;

typedef struct SCREEN_TRACE {
    NriTexture *texture;
    NriDescriptor *srv;
    NriDescriptor *uav;
    NriAccessLayoutStage state;
    uint32_t width;
    uint32_t height;
} SCREEN_TRACE;

typedef struct COMPUTE_TEXTURE {
    NriTexture *texture;
    NriDescriptor *srv;
    NriDescriptor *uav;
    NriAccessLayoutStage state;
    NriFormat format;
    uint32_t width;
    uint32_t height;
} COMPUTE_TEXTURE;

typedef struct RADIANCE_PROBES {
    COMPUTE_TEXTURE current_radiance;
    COMPUTE_TEXTURE current_meta;
    COMPUTE_TEXTURE history_radiance;
    COMPUTE_TEXTURE history_meta;
    COMPUTE_TEXTURE previous_irradiance;
    COMPUTE_TEXTURE history_depth;
    COMPUTE_TEXTURE history_normal;
} RADIANCE_PROBES;

typedef struct SDF_GPU_SCENE {
    NriBuffer *models;
    NriBuffer *voxels;
    NriBuffer *surface_ids;
    NriDescriptor *models_srv;
    NriDescriptor *voxels_srv;
    NriDescriptor *surface_ids_srv;
    NriAccessStage models_state;
    NriAccessStage voxels_state;
    NriAccessStage surface_ids_state;
    GPU_SDF_MODEL *cpu_models;
    uint32_t model_count;
    uint32_t voxel_count;
    uint32_t clipmap_count;
    uint32_t clipmap_resolution;
    float clipmap_base_extent;
} SDF_GPU_SCENE;

typedef struct SURFACE_CACHE {
    NriBuffer *keys;
    NriBuffer *entries;
    NriDescriptor *keys_uav;
    NriDescriptor *entries_uav;
    NriAccessStage keys_state;
    NriAccessStage entries_state;
    uint32_t capacity;
} SURFACE_CACHE;


typedef struct RADIANCE_WAVEFRONT {
    NriBuffer *queue_a;
    NriBuffer *queue_b;
    NriBuffer *surface_hits;
    NriBuffer *counters;
    NriBuffer *dispatch_args;
    NriBuffer *budgets;
    NriBuffer *update_list;
    NriBuffer *radiance;
    NriBuffer *flags;

    NriDescriptor *queue_a_uav;
    NriDescriptor *queue_b_uav;
    NriDescriptor *surface_hits_uav;
    NriDescriptor *counters_uav;
    NriDescriptor *dispatch_args_uav;
    NriDescriptor *budgets_uav;
    NriDescriptor *update_list_uav;
    NriDescriptor *radiance_uav;
    NriDescriptor *flags_uav;

    NriAccessStage state;
    uint32_t ray_capacity;
    uint32_t probe_capacity;
} RADIANCE_WAVEFRONT;

typedef struct RADIANCE_WORLD_RESOURCES {
    NriBuffer *probes;
    NriBuffer *radiance;
    NriBuffer *keys;
    NriBuffer *invalidation_queue;

    NriDescriptor *probes_uav;
    NriDescriptor *radiance_uav;
    NriDescriptor *keys_uav;
    NriDescriptor *invalidation_queue_uav;

    NriAccessStage state;
} RADIANCE_WORLD_RESOURCES;

typedef struct RENDER_TEXTURE {
    NriTexture *texture;
    NriDescriptor *attachment;
    NriDescriptor *srv;
    NriAccessLayoutStage state;
    NriFormat format;
} RENDER_TEXTURE;

typedef struct RENDERER {
    GPU *gpu;
    SCENE *scene;
    CAMERA camera;
    CAMERA previous_camera;
    MAT4 previous_view_projection;

    NriBuffer *vertex_buffer;
    NriBuffer *material_buffer;
    NriBuffer *object_buffer;
    NriBuffer *light_buffer;
    NriBuffer *frame_buffer;

    NriBuffer *radiance_constants_buffer;
    NriBuffer *pass_constants_buffer;

    NriDescriptor *material_srv;
    NriDescriptor *object_srv;
    NriDescriptor *light_srv;
    NriDescriptor *frame_srv;

    NriDescriptor *radiance_constants_srv;
    NriDescriptor *pass_constants_srv;

    GPU_OBJECT *cpu_objects;
    GPU_LIGHT *cpu_lights;

    RADIANCE_CONSTANTS radiance_constants;
    PASS_CONSTANTS pass_constants;

    uint32_t gpu_object_count;
    uint32_t vertex_count;
    uint32_t material_count;
    uint32_t light_count;

    RENDER_TEXTURE depth;
    RENDER_TEXTURE normal_roughness;
    RENDER_TEXTURE albedo_metallic;
    RENDER_TEXTURE emissive;
    RENDER_TEXTURE velocity;
    RENDER_TEXTURE object_id;
    RENDER_TEXTURE material_id;
    RENDER_TEXTURE primitive_id;

    HZB hzb;
    SCREEN_TRACE screen_trace;
    COMPUTE_TEXTURE direct_radiance;
    COMPUTE_TEXTURE screen_probes;
    RADIANCE_PROBES probes;
    RADIANCE_WAVEFRONT wavefront;
    RADIANCE_WORLD_RESOURCES world_radiance;
    SDF_GPU_SCENE sdf;
    SURFACE_CACHE radiance_surface_cache;
    RADIANCE_SCENE_DATA radiance_scene;
    RADIANCE_SCENE_FALLBACKS radiance_fallbacks;

    NriDescriptorPool *descriptor_pool;
    NriDescriptorSet *gbuffer_set;
    NriDescriptorSet *present_set;
    NriDescriptorSet *hzb_sets[HZB_MAX_MIPS];
    NriDescriptorSet *trace_set;
    NriDescriptorSet *wavefront_trace_set;
    NriDescriptorSet *wavefront_scene_set;
    NriDescriptorSet *wavefront_queue_set;
    NriDescriptorSet *wavefront_cache_set;
    NriDescriptorSet *wavefront_probe_set;

    NriPipelineLayout *gbuffer_layout;
    NriPipelineLayout *present_layout;
    NriPipelineLayout *hzb_layout;
    NriPipelineLayout *trace_layout;
    NriPipelineLayout *wavefront_layout;

    NriPipeline *gbuffer_pipeline;
    NriPipeline *present_pipeline;
    NriPipeline *hzb_pipeline;
    NriPipeline *direct_radiance_pipeline;
    NriPipeline *wavefront_reset_pipeline;
    NriPipeline *wavefront_budget_pipeline;
    NriPipeline *wavefront_generate_pipeline;
    NriPipeline *wavefront_screen_pipeline;
    NriPipeline *wavefront_local_pipeline;
    NriPipeline *wavefront_shade_pipeline;
    NriPipeline *wavefront_temporal_pipeline;
    NriPipeline *wavefront_spatial_pipeline;
    NriPipeline *wavefront_resolve_pipeline;
    NriPipeline *emissive_pipeline;
    NriPipeline *wavefront_history_pipeline;

    uint32_t width;
    uint32_t height;
    bool has_previous_frame;
    bool probe_history_valid;

    NriAccessStage object_state;
    NriAccessStage light_state;
    NriAccessStage frame_state;

    NriAccessStage radiance_constants_state;
    NriAccessStage pass_constants_state;

    uint64_t frame_index;
} RENDERER;

bool renderer_init(RENDERER *renderer, GPU *gpu);
void renderer_deinit(RENDERER *renderer);
bool renderer_set_scene(RENDERER *renderer, SCENE *scene);
void renderer_event(RENDERER *renderer, const SDL_Event *event);
bool renderer_frame(RENDERER *renderer);

#endif
