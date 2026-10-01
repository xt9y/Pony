#ifndef GPU_H
#define GPU_H

#include "game.h"

#include <stdbool.h>
#include <stdint.h>

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

typedef struct SWAPCHAIN_TEXTURE SWAPCHAIN_TEXTURE;
typedef struct FRAME_CONTEXT FRAME_CONTEXT;
typedef struct UPLOAD_CONTEXT UPLOAD_CONTEXT;
typedef struct TEXTURE_STATE TEXTURE_STATE;
typedef struct PROBE_WAVEFRONT_SCRATCH PROBE_WAVEFRONT_SCRATCH;
typedef struct RENDER_MATERIAL RENDER_MATERIAL;

typedef struct GPU {
    SDL_Window *window;

#if defined(__APPLE__)
    SDL_MetalView metal_view;
#endif

    NriDevice *device;
    NriCoreInterface core;
    NriHelperInterface helper;
    NriSwapChainInterface swapchain_api;

    NriQueue *graphics_queue;
    NriQueue *compute_queue;
    NriQueue *copy_queue;
    NriQueue *work_queue;

    NriSwapChain *swapchain;
    NriDescriptorPool *descriptor_pool;
    NriFence *frame_fence;
    NriFence *work_fence;

    FRAME_CONTEXT *frame_contexts;
    FRAME_CONTEXT *work_contexts;
    FRAME_CONTEXT *active_frame;
    FRAME_CONTEXT *active_work;
    uint64_t work_index;
    uint64_t work_next_fence;
    UPLOAD_CONTEXT *upload;

    SWAPCHAIN_TEXTURE *swapchain_frames;
    NriTexture **swapchain_textures;
    uint32_t swapchain_texture_count;

    NriPipelineCache *pipeline_cache;
    NriQueryPool *timestamp_pool;
    NriBuffer *timestamp_readback;
    uint32_t timestamp_query_size;
    bool timestamp_supported;

    uint32_t swapchain_width;
    uint32_t swapchain_height;
    uint32_t current_swap_index;
    uint64_t frame_index;
    NriFormat swapchain_format;

    NriDescriptor **temporary_descriptors;
    NriBuffer **temporary_buffers;
    uint32_t temporary_descriptor_num;
    uint32_t temporary_descriptor_cap;
    uint32_t temporary_buffer_num;
    uint32_t temporary_buffer_cap;

    TEXTURE_STATE *texture_states;
    uint32_t texture_state_num;
    uint32_t texture_state_cap;
} GPU;

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
    DIRECTIONAL_LIGHT sun;
    SKY sky;
    VOLUMETRICS_LIGHTING volumetrics;

    bool dragging;
    bool show_debug;
    bool show_volume;
    uint32_t debug_view;
};

bool gpu_init(GPU *gpu, const char *title, int width, int height);
void gpu_deinit(GPU *gpu);

bool upload_scene(RENDERER *renderer, const GLTF_SCENE *visual);
bool upload_bvh(RENDERER *renderer, const BVH *tree);
void release_texture(RENDERER *renderer, NriTexture *texture);
void release_buffer(RENDERER *renderer, NriBuffer *buffer);
void release_bake_resources(RENDERER *renderer);
bool draw_frame(RENDERER *renderer, const RENDER_FRAME *frame);
bool bake_worker_init(RENDERER *renderer);
void bake_worker_deinit(RENDERER *renderer);

#endif
