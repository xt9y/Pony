#ifndef RENDER_INTERNAL_H
#define RENDER_INTERNAL_H

#include "game.h"

#include <stddef.h>

typedef struct BAKE_UNIFORMS {
    Uint32 item_count;
    Uint32 lightmap_width;
    Uint32 lightmap_height;
    Uint32 dispatch_width;
    Uint32 iteration;
    Uint32 phase;
    Uint32 max_bounces;
    Uint32 batch_count;
    float sun_direction_intensity[4];
    float sun_color_radius[4];
    float sky_zenith[4];
    float sky_horizon[4];
    float bake_params[4];
    float probe_origin_spacing[4];
    Uint32 probe_dims_mode[4];
    float emissive_data[4];
} BAKE_UNIFORMS;

NriTexture *gpu_create_texture(RENDERER *renderer, NriFormat format, NriTextureUsageBits usage, Uint32 width, Uint32 height);
bool gpu_upload_texture_data(RENDERER *renderer, NriTexture *texture, const void *data, uint32_t row_pitch, uint32_t slice_pitch, NriAccessBits access,
                             NriLayout layout, NriStageBits stages);
bool gpu_clear_texture_zero(RENDERER *renderer, NriTexture *texture, uint32_t bytes_per_texel, NriAccessBits access, NriLayout layout, NriStageBits stages);
NriResult gpu_begin_commands(RENDERER *renderer, NriCommandAllocator **allocator, NriCommandBuffer **command_buffer);
bool gpu_transition_texture(RENDERER *renderer, NriCommandBuffer *command_buffer, NriTexture *texture, NriAccessBits access, NriLayout layout,
                            NriStageBits stages);
bool gpu_submit_commands(RENDERER *renderer, NriCommandAllocator *allocator, NriCommandBuffer *command_buffer);
void gpu_abort_commands(RENDERER *renderer, NriCommandAllocator *allocator, NriCommandBuffer *command_buffer);
NriBuffer *gpu_upload_buffer(RENDERER *renderer, NriBufferUsageBits usage, const void *data, size_t bytes, uint32_t stride);
NriPipeline *gpu_compile_compute(RENDERER *renderer, NriPipelineLayout *layout, const char *path, const char *entrypoint, const char *define);
NriDescriptor *gpu_create_buffer_view(RENDERER *renderer, NriBuffer *buffer, NriBufferView type, uint32_t stride);
NriDescriptor *gpu_create_texture_view(RENDERER *renderer, NriTexture *texture, NriTextureView type);
NriDescriptor *gpu_create_sampler(RENDERER *renderer, NriFilter min_filter, NriFilter mag_filter, NriAddressMode address);
bool gpu_init_worker(GPU *gpu);
void gpu_clear_temporary(RENDERER *renderer);
void release_texture(RENDERER *renderer, NriTexture *texture);
void release_buffer(RENDERER *renderer, NriBuffer *buffer);
bool gpu_bind_descriptor_set(RENDERER *renderer, NriCommandBuffer *command_buffer, NriPipelineLayout *layout, NriBindPoint point, uint32_t set_index,
                             NriDescriptor *const *descriptors, uint32_t count);
bool gpu_bind_uniform_data(RENDERER *renderer, NriCommandBuffer *command_buffer, NriPipelineLayout *layout, NriBindPoint point, uint32_t set,
                           const void *data, size_t size);
bool gpu_create_pipeline_layout(RENDERER *renderer, NriPipelineLayout **out, const NriDescriptorType *types[4], const uint8_t counts[4],
                                NriStageBits stages);
bool gpu_create_compute_layout(RENDERER *renderer, NriPipelineLayout **out, const NriDescriptorType *sources, uint8_t source_num,
                               NriDescriptorType output_type, bool has_uniform);
bool gpu_timestamp_begin(RENDERER *renderer, NriCommandBuffer *command_buffer, uint32_t slot);
bool gpu_timestamp_end(RENDERER *renderer, NriCommandBuffer *command_buffer, uint32_t slot);
void gpu_timestamp_log(RENDERER *renderer, uint32_t slot, const char *label);
NriShaderDesc gpu_load_shader(const char *entrypoint, const char *define, NriStageBits stage);
void gpu_free_shader(NriShaderDesc *shader);
bool gpu_texture_barrier(RENDERER *renderer, NriCommandBuffer *command_buffer, NriTexture *texture, NriAccessLayoutStage before,
                         NriAccessLayoutStage after);
bool gpu_ensure_swapchain(RENDERER *renderer, uint32_t width, uint32_t height);
bool gpu_begin_render_frame(RENDERER *renderer, FRAME_CONTEXT **frame, NriCommandBuffer **command_buffer, NriTexture **swapchain_texture,
                            uint32_t *swapchain_index);
bool gpu_submit_render_frame(RENDERER *renderer, FRAME_CONTEXT *frame, NriCommandBuffer *command_buffer, uint32_t swapchain_index);
void gpu_abort_render_frame(RENDERER *renderer, FRAME_CONTEXT *frame);
NriDescriptor *gpu_swapchain_color_attachment(RENDERER *renderer, uint32_t swapchain_index);


bool renderer_gpu_resources_init(RENDERER *renderer);
void renderer_gpu_resources_deinit(RENDERER *renderer);

bool upload_bvh(RENDERER *renderer, const BVH *tree);
bool bake_lightmap(RENDERER *renderer, const BVH *tree, const LIGHTMAP *lightmap, const PROBE_GRID *probes);
bool bake_gpu_layouts_init(RENDERER *renderer);
void bake_gpu_layouts_deinit(RENDERER *renderer);
void probe_wavefront_scratch_destroy(RENDERER *renderer);
void release_bake_resources(RENDERER *renderer);
bool bake_probe_grid_fast(RENDERER *renderer, PROBE_GRID *grid, const BVH *tree, const BEAM_GRID *beams, PROBE_BAKE_PROGRESS_FN progress);
bool bake_probe_grid(RENDERER *renderer, PROBE_GRID *grid, Uint32 samples);
NriTexture *upload_lightmap(RENDERER *renderer, const CACHED_LIGHTMAP *cached);
NriTexture *upload_direct_lightmap(RENDERER *renderer, const CACHED_LIGHTMAP *cached);
NriBuffer *upload_probes(RENDERER *renderer, const PROBE_GRID *grid);
NriBuffer *upload_beams(RENDERER *renderer, const BEAM_GRID *grid);
bool download_lightmap(RENDERER *renderer, CACHED_LIGHTMAP *out);

#endif
