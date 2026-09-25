#ifndef GPU_H
#define GPU_H

/* Uncomment to compile the SDL_GPU backend. */
/* #define DUSTMITE_GPU_SDL */

#include "dustmite.h"

#include <SDL3/SDL.h>

#if !defined(DUSTMITE_GPU_SDL) && defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#pragma clang diagnostic ignored "-Wvariadic-macro-arguments-omitted"
#pragma clang diagnostic ignored "-Wstrict-prototypes"
#endif

#ifndef DUSTMITE_GPU_SDL
#include <NRI.h>
#include <Extensions/NRIDeviceCreation.h>
#include <Extensions/NRIHelper.h>
#include <Extensions/NRISwapChain.h>
#endif

#if !defined(DUSTMITE_GPU_SDL) && defined(__clang__)
#pragma clang diagnostic pop
#endif

#define WINDOW SDL_Window
#define EVENT SDL_Event
#ifdef DUSTMITE_GPU_SDL
#define DEVICE SDL_GPUDevice
#define BUFFER SDL_GPUBuffer
#define TEXTURE SDL_GPUTexture
#define SAMPLER SDL_GPUSampler
#define GRAPHICS_PIPELINE SDL_GPUGraphicsPipeline
#define COMPUTE_PIPELINE SDL_GPUComputePipeline
#define TEXTURE_FORMAT SDL_GPUTextureFormat
#else
#define DEVICE NriDevice
#define BUFFER NriBuffer
#define TEXTURE NriTexture
#define SAMPLER NriDescriptor
#define GRAPHICS_PIPELINE NriPipeline
#define COMPUTE_PIPELINE NriPipeline
#define TEXTURE_FORMAT NriFormat
#endif

typedef struct renderer renderer;

typedef struct fx_state {
#ifdef DUSTMITE_GPU_SDL
    DEVICE *device;
#else
    renderer *owner;
#endif

    GRAPHICS_PIPELINE *compose_pipeline;
    COMPUTE_PIPELINE *ssao_pipeline;
    COMPUTE_PIPELINE *bloom_pipeline;
    COMPUTE_PIPELINE *grade_pipeline;
    COMPUTE_PIPELINE *volume_pipeline;
    COMPUTE_PIPELINE *volume_compose_pipeline;
    SAMPLER *sampler;
    SAMPLER *depth_sampler;

    TEXTURE *hdr;
    TEXTURE *normal_depth;
    TEXTURE *ao;
    TEXTURE *bloom_a;
    TEXTURE *bloom_b;
    TEXTURE *lut;
    TEXTURE *volume;
    TEXTURE *lit;

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

#ifndef DUSTMITE_GPU_SDL
typedef struct swapchain_texture {
    TEXTURE *texture;
    NriDescriptor *color_attachment;
    NriFence *acquire;
    NriFence *release;
} swapchain_texture;
typedef struct texture_state {
    NriTexture *texture;
    NriAccessLayoutStage state;
} texture_state;
#endif

struct renderer {
    WINDOW *window;
    DEVICE *device;

#ifndef DUSTMITE_GPU_SDL
    NriCoreInterface core;
    NriHelperInterface helper;
    NriSwapChainInterface swapchain_api;
    NriQueue *graphics_queue;
    NriSwapChain *swapchain;
    NriDescriptorPool *descriptor_pool;
    swapchain_texture *swapchain_frames;
    TEXTURE **swapchain_textures;
    uint32_t swapchain_texture_count;
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
    NriPipelineLayout *probe_layout;
    NriPipelineLayout *ssao_layout;
    NriPipelineLayout *bloom_layout;
    NriPipelineLayout *grade_layout;
    NriPipelineLayout *volume_layout;
    NriPipelineLayout *volume_compose_layout;
    NriPipelineLayout *compose_layout;
    NriPipelineLayout *current_graphics_layout, *current_compute_layout;
#endif

    GRAPHICS_PIPELINE *sky_pipeline;
    GRAPHICS_PIPELINE *solid_pipeline;
    GRAPHICS_PIPELINE *line_pipeline;
    COMPUTE_PIPELINE *bake_pipeline;

    BUFFER *vertex_buffer;
    BUFFER *bvh_node_buffer;
    BUFFER *bvh_triangle_buffer;
    BUFFER *lightmap_sample_buffer;

    TEXTURE *depth_texture;
    TEXTURE *lightmap_texture;
    TEXTURE *lightmap_scratch;
    SAMPLER *lightmap_sampler;
    SAMPLER *material_sampler;
    TEXTURE_FORMAT depth_format;
    Uint32 depth_width;
    Uint32 depth_height;

    TEXTURE **image_textures;
    uint32_t image_texture_count;
    TEXTURE *default_white;
    TEXTURE *default_normal;

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
    uint32_t bake_target_samples;
    float bake_epsilon;
    bool has_bake;
    const char *bake_stage;
    dm_probe_grid volume_probes;
    BUFFER *volume_probe_buffer;
    BUFFER *beam_buffer;
    dm_beam_grid beams;

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

bool r_init(renderer *r, const char *title, int width, int height);
bool r_build_scene(renderer *r, const mesh *m, const gltf_scene *visual,
                   const lightmap *lm);
bool r_load_cached_lightmap(renderer *r, const char *path, uint64_t scene_hash,
                            uint64_t layout_hash, uint64_t volume_hash,
                            uint64_t beam_hash, const lightmap *lm);
bool r_rebake_current_scene(renderer *r, const mesh *m,
                            const gltf_scene *visual, const lightmap *lm,
                            const char *path, uint64_t scene_hash,
                            uint64_t layout_hash, uint64_t volume_hash,
                            uint64_t beam_hash);
void r_event(renderer *r, const EVENT *event);
bool r_draw(renderer *r);
void r_deinit(renderer *r);

bool upload_scene(renderer *r, const gltf_scene *visual);
bool upload_bvh(renderer *r, const bvh *tree);
bool bake_lightmap(renderer *r, const bvh *tree, const lightmap *lm);
bool bake_probe_grid(renderer *r, dm_probe_grid *grid, Uint32 samples);
TEXTURE *upload_lightmap(renderer *r, const dm_cached_lightmap *cached);
BUFFER *upload_probes(renderer *r, const dm_probe_grid *grid);
BUFFER *upload_beams(renderer *r, const dm_beam_grid *grid);
bool download_lightmap(renderer *r, dm_cached_lightmap *out);
void release_texture(renderer *r, TEXTURE *texture);
void release_buffer(renderer *r, BUFFER *buffer);
void release_bake_resources(renderer *r);
bool draw_frame(renderer *r, const render_frame *frame);
bool bake_worker_init(renderer *r);
void bake_worker_deinit(renderer *r);

void bake_progress(renderer *r, const char *stage, Uint32 done, Uint32 total);

bool bake_start(renderer *r, const mesh *scene, const gltf_scene *visual,
                const lightmap *layout, const char *path,
                uint64_t scene_hash, uint64_t layout_hash,
                uint64_t volume_hash, uint64_t beam_hash);
void bake_update(renderer *r);
void bake_cancel(renderer *r);
bool bake_active(renderer *r);
void bake_update_title(renderer *r);

#endif
