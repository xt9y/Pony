#ifndef RENDER_INTERNAL_H
#define RENDER_INTERNAL_H

#include "gpu.h"

bool renderer_gpu_resources_init(RENDERER *renderer);
void renderer_gpu_resources_deinit(RENDERER *renderer);

bool bake_lightmap(RENDERER *renderer, const BVH *tree, const LIGHTMAP *lightmap, const PROBE_GRID *probes);
bool bake_probe_grid_fast(RENDERER *renderer, PROBE_GRID *grid, const BVH *tree, const BEAM_GRID *beams, PROBE_BAKE_PROGRESS_FN progress);
bool bake_probe_grid(RENDERER *renderer, PROBE_GRID *grid, Uint32 samples);
NriTexture *upload_lightmap(RENDERER *renderer, const CACHED_LIGHTMAP *cached);
NriBuffer *upload_probes(RENDERER *renderer, const PROBE_GRID *grid);
NriBuffer *upload_beams(RENDERER *renderer, const BEAM_GRID *grid);
bool download_lightmap(RENDERER *renderer, CACHED_LIGHTMAP *out);

#endif
