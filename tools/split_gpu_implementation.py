#!/usr/bin/env python3
from pathlib import Path

GPU = Path("gpu.c")
BAKE = Path("bake.c")

source = GPU.read_text(encoding="utf-8")
bake = BAKE.read_text(encoding="utf-8")

start_marker = "static NriTexture *create_lightmap_texture"
end_marker = "static bool dispatch_one"

start = source.find(start_marker)
end = source.find(end_marker)
if start < 0 or end < 0 or end <= start:
    raise SystemExit("could not locate contiguous bake implementation region")

block = source[start:end].rstrip() + "\n\n"
source = source[:start] + source[end:]

insert_marker = "typedef enum BAKE_PHASE"
insert = bake.find(insert_marker)
if insert < 0:
    raise SystemExit("could not locate bake.c insertion point")

support = '''#define BAKE_TARGET_SAMPLES 128u
#define BAKE_MAX_BOUNCES 3u
#define BAKE_BATCH_SAMPLES 8u
#define BAKE_DILATION_PASSES 3u
#define PHASE_CLEAR 0u
#define PHASE_TRACE 1u
#define PHASE_FILTER 2u
#define PHASE_DILATE 3u
#define PHASE_DIRECT 4u
#define PHASE_COMBINE 5u
#define PHASE_RECONSTRUCT 6u

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

'''

bake = bake[:insert] + support + block + bake[insert:]

GPU.write_text(source, encoding="utf-8")
BAKE.write_text(bake, encoding="utf-8")
