#!/usr/bin/env python3
from pathlib import Path

# One-shot guarded migration: move Dustmite bake GPU algorithms out of gpu.c.
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

old = "static void probe_wavefront_scratch_destroy(RENDERER *r)"
new = "void probe_wavefront_scratch_destroy(RENDERER *r)"
if old not in block:
    raise SystemExit("probe scratch cleanup was not inside bake region")
block = block.replace(old, new, 1)


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

'''

bake = bake[:insert] + support + block + bake[insert:]

GPU.write_text(source, encoding="utf-8")
BAKE.write_text(bake, encoding="utf-8")
