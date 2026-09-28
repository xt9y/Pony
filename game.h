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


// MODEL LOADING

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


// SCENE

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


// RENDERER

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
    NriBuffer *frame_buffer;

    NriDescriptor *material_srv;
    NriDescriptor *object_srv;
    NriDescriptor *frame_srv;

    GPU_OBJECT *cpu_objects;

    uint32_t gpu_object_count;
    uint32_t vertex_count;
    uint32_t material_count;

    RENDER_TEXTURE depth;
    RENDER_TEXTURE normal_roughness;
    RENDER_TEXTURE albedo_metallic;
    RENDER_TEXTURE velocity;
    RENDER_TEXTURE object_id;

    HZB hzb;
    SCREEN_TRACE screen_trace;

    NriDescriptorPool *descriptor_pool;

    NriDescriptorSet *gbuffer_set;
    NriDescriptorSet *present_set;
    NriDescriptorSet *hzb_sets[HZB_MAX_MIPS];
    NriDescriptorSet *screen_trace_set;

    NriPipelineLayout *gbuffer_layout;
    NriPipelineLayout *present_layout;
    NriPipelineLayout *hzb_layout;
    NriPipelineLayout *screen_trace_layout;

    NriPipeline *gbuffer_pipeline;
    NriPipeline *present_pipeline;
    NriPipeline *hzb_pipeline;
    NriPipeline *screen_trace_pipeline;

    uint32_t width;
    uint32_t height;

    bool has_previous_frame;

    NriAccessStage object_state;
    NriAccessStage frame_state;

    uint64_t frame_index;
} RENDERER;


bool renderer_init(RENDERER *renderer, GPU *gpu);
void renderer_deinit(RENDERER *renderer);

bool renderer_set_scene(RENDERER *renderer, SCENE *scene);

void renderer_event(RENDERER *renderer, const SDL_Event *event);
bool renderer_frame(RENDERER *renderer);


#endif
