#ifndef GPU_H
#define GPU_H

#include "game.h"

#include <SDL3/SDL.h>

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#pragma clang diagnostic ignored "-Wvariadic-macro-arguments-omitted"
#pragma clang diagnostic ignored "-Wstrict-prototypes"
#endif

#include <NRI.h>
#include <Extensions/NRIDeviceCreation.h>
#include <Extensions/NRIHelper.h>
#include <Extensions/NRISwapChain.h>

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

typedef struct RENDERER RENDERER;

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

typedef struct RENDER_MATERIAL RENDER_MATERIAL;

typedef struct DRAW_RANGE {
    uint32_t first;
    uint32_t count;
    uint32_t material;
} DRAW_RANGE;

typedef struct RENDER_FRAME {
    float mvp[16];
    float view[16];
    VEC3 eye;
    VEC3 right;
    VEC3 up;
    VEC3 forward;
    VEC3 sun;
    float tan_half_fov;
    float aspect;
} RENDER_FRAME;

typedef struct SWAPCHAIN_TEXTURE SWAPCHAIN_TEXTURE;
typedef struct FRAME_CONTEXT FRAME_CONTEXT;
typedef struct UPLOAD_CONTEXT UPLOAD_CONTEXT;
typedef struct TEXTURE_STATE TEXTURE_STATE;

struct RENDERER {
    SDL_Window *window;

    NriDevice *device;

    NriCoreInterface core;
    NriHelperInterface helper;
    NriSwapChainInterface swapchain_api;
    NriQueue *graphics_queue;
    NriSwapChain *swapchain;
    NriDescriptorPool *descriptor_pool;
    NriFence *frame_fence;
    FRAME_CONTEXT *frame_contexts;
    FRAME_CONTEXT *active_frame;
    UPLOAD_CONTEXT *upload;
    SWAPCHAIN_TEXTURE *swapchain_frames;
    NriTexture **swapchain_textures;
    uint32_t swapchain_texture_count;

    uint32_t swapchain_width, swapchain_height, current_swap_index;

    uint64_t frame_index;
    NriFormat swapchain_format;
    SDL_MetalView metal_view;
    NriDescriptor **temporary_descriptors;
    NriBuffer **temporary_buffers;

    uint32_t temporary_descriptor_num, temporary_descriptor_cap;
    uint32_t temporary_buffer_num, temporary_buffer_cap;

    TEXTURE_STATE *texture_states;

    uint32_t texture_state_num, texture_state_cap;

    NriPipelineLayout *surface_layout;
    NriPipelineLayout *line_layout;
    NriPipelineLayout *sky_layout;
    NriPipelineLayout *bake_layout;
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
    NriPipeline *line_pipeline;
    NriPipeline *bake_pipeline;

    NriBuffer *vertex_buffer;
    NriBuffer *bvh_node_buffer;
    NriBuffer *bvh_triangle_buffer;
    NriBuffer *lightmap_sample_buffer;
    NriBuffer *lightmap_full_sample_buffer;
    NriBuffer *lightmap_sparse_sample_buffer;
    NriBuffer *lightmap_probe_buffer;
    NriBuffer *lightmap_patch_map_buffer;
    NriBuffer *lightmap_patch_anchor_buffer;

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
    DRAW_RANGE *draws;
    uint32_t draw_count;

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
    VEC3 sun;

    bool dragging;
    bool show_debug;
    bool show_volume;
    uint32_t debug_view;
};

bool upload_scene(RENDERER *r, const GLTF_SCENE *visual);
bool upload_bvh(RENDERER *r, const BVH *tree);
bool bake_lightmap(RENDERER *r, const BVH *tree, const LIGHTMAP *lm, const PROBE_GRID *probes);
typedef bool (*PROBE_BAKE_PROGRESS_FN)(Uint32 done, Uint32 total, Uint32 active);
bool bake_probe_grid_fast(RENDERER *r, PROBE_GRID *grid, const BVH *tree, const BEAM_GRID *beams, PROBE_BAKE_PROGRESS_FN progress);
bool bake_probe_grid(RENDERER *r, PROBE_GRID *grid, Uint32 samples);
NriTexture *upload_lightmap(RENDERER *r, const CACHED_LIGHTMAP *cached);
NriBuffer *upload_probes(RENDERER *r, const PROBE_GRID *grid);
NriBuffer *upload_beams(RENDERER *r, const BEAM_GRID *grid);
bool download_lightmap(RENDERER *r, CACHED_LIGHTMAP *out);
void release_texture(RENDERER *r, NriTexture *texture);
void release_buffer(RENDERER *r, NriBuffer *buffer);
void release_bake_resources(RENDERER *r);
bool draw_frame(RENDERER *r, const RENDER_FRAME *frame);
bool bake_worker_init(RENDERER *r);
void bake_worker_deinit(RENDERER *r);

#endif
