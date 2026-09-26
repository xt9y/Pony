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

typedef struct renderer renderer;

typedef struct fx_state {
    renderer *owner;

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
} fx_state;

typedef struct render_vertex {
    float x, y, z;
    float nx, ny, nz;
    float u, v;
    float lu, lv;
    float r, g, b, a;
} render_vertex;

typedef struct render_material render_material;

typedef struct draw_range {
    uint32_t first;
    uint32_t count;
    uint32_t material;
} draw_range;

typedef struct render_frame {
    float mvp[16];
    float view[16];
    vec3 eye;
    vec3 right;
    vec3 up;
    vec3 forward;
    vec3 sun;
    float tan_half_fov;
    float aspect;
} render_frame;

typedef struct swapchain_texture swapchain_texture;
typedef struct frame_context frame_context;
typedef struct upload_context upload_context;
typedef struct texture_state texture_state;
typedef struct probe_wavefront_scratch probe_wavefront_scratch;

struct renderer {
    SDL_Window *window;

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
    frame_context *frame_contexts;
    frame_context *work_contexts;
    frame_context *active_frame;
    frame_context *active_work;
    uint64_t work_index;
    uint64_t work_next_fence;
    upload_context *upload;
    swapchain_texture *swapchain_frames;
    NriTexture **swapchain_textures;
    uint32_t swapchain_texture_count;

    NriPipelineCache *pipeline_cache;
    NriQueryPool *timestamp_pool;
    NriBuffer *timestamp_readback;
    uint32_t timestamp_query_size;
    bool timestamp_supported;
    probe_wavefront_scratch *probe_scratch;

    uint32_t swapchain_width, swapchain_height, current_swap_index;

    uint64_t frame_index;
    NriFormat swapchain_format;
    SDL_MetalView metal_view;
    NriDescriptor **temporary_descriptors;
    NriBuffer **temporary_buffers;

    uint32_t temporary_descriptor_num, temporary_descriptor_cap;
    uint32_t temporary_buffer_num, temporary_buffer_cap;

    texture_state *texture_states;

    uint32_t texture_state_num, texture_state_cap;

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

    render_material *materials;
    uint32_t material_count;
    draw_range *draws;
    uint32_t draw_count;

    render_vertex *vertices;
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
    vec3 lightmap_probe_origin;
    float lightmap_probe_spacing;
    uint32_t lightmap_probe_count_x;
    uint32_t lightmap_probe_count_y;
    uint32_t lightmap_probe_count_z;
    float bake_epsilon;
    bool has_bake;
    const char *bake_stage;
    probe_grid volume_probes;
    NriBuffer *volume_probe_buffer;
    NriBuffer *beam_buffer;
    beam_grid beams;

    fx_state fx;

    float yaw;
    float pitch;
    float distance;
    float scene_radius;
    double frame_time_ms;
    vec3 target;

    bool dragging;
    bool show_debug;
    bool show_volume;
    uint32_t debug_view;
};

bool upload_scene(renderer *r, const gltf_scene *visual);
bool upload_bvh(renderer *r, const bvh *tree);
bool bake_lightmap(renderer *r, const bvh *tree, const lightmap *lm, const probe_grid *probes);
typedef bool (*probe_bake_progress_fn)(Uint32 done, Uint32 total, Uint32 active);
bool bake_probe_grid_fast(renderer *r, probe_grid *grid, const bvh *tree, const beam_grid *beams, probe_bake_progress_fn progress);
bool bake_probe_grid(renderer *r, probe_grid *grid, Uint32 samples);
NriTexture *upload_lightmap(renderer *r, const cached_lightmap *cached);
NriBuffer *upload_probes(renderer *r, const probe_grid *grid);
NriBuffer *upload_beams(renderer *r, const beam_grid *grid);
bool download_lightmap(renderer *r, cached_lightmap *out);
void release_texture(renderer *r, NriTexture *texture);
void release_buffer(renderer *r, NriBuffer *buffer);
void release_bake_resources(renderer *r);
bool draw_frame(renderer *r, const render_frame *frame);
bool bake_worker_init(renderer *r);
void bake_worker_deinit(renderer *r);

#endif
