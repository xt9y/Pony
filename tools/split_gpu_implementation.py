#!/usr/bin/env python3
from pathlib import Path
import re

GPU = Path("gpu.c")
BAKE = Path("bake.c")

source = GPU.read_text(encoding="utf-8")
bake = BAKE.read_text(encoding="utf-8")


def extract_function(text: str, marker: str):
    start = text.find(marker)
    if start < 0:
        raise SystemExit(f"missing function marker: {marker}")

    brace = text.find("{", start)
    if brace < 0:
        raise SystemExit(f"missing function body: {marker}")

    depth = 0
    end = -1
    for index in range(brace, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                end = index + 1
                break

    if end < 0:
        raise SystemExit(f"unterminated function: {marker}")

    while end < len(text) and text[end] == "\n":
        end += 1

    block = text[start:end].rstrip() + "\n\n"
    return text[:start] + text[end:], block


# Shared render/bake primitives keep generic gpu_* names.
for old, new in (
    ("create_texture_view", "gpu_create_texture_view"),
    ("sampler", "gpu_create_sampler"),
    ("clear_temporary", "gpu_clear_temporary"),
    ("create_compute_layout", "gpu_create_compute_layout"),
):
    source = re.sub(rf"\b{old}\s*\(", f"{new}(", source)

source = re.sub(r"(^|\n)static (?=[^\n]*\bgpu_create_texture_view\s*\()", r"\1", source)
source = re.sub(r"(^|\n)static (?=[^\n]*\bgpu_create_sampler\s*\()", r"\1", source)
source = re.sub(r"(^|\n)static (?=[^\n]*\bgpu_clear_temporary\s*\()", r"\1", source)
source = re.sub(r"(^|\n)static (?=[^\n]*\bgpu_create_compute_layout\s*\()", r"\1", source)

# Generic offscreen GPU infrastructure used by the bake worker.
if "bool gpu_init_worker(GPU *gpu)" not in source:
    marker = "bool gpu_init(GPU *gpu, const char *title, int width, int height) {"
    insert = source.find(marker)
    if insert < 0:
        raise SystemExit("missing gpu_init insertion point")

    worker = """bool gpu_init_worker(GPU *gpu) {
    if (!gpu) return false;

    memset(gpu, 0, sizeof(*gpu));

    NriDeviceCreationDesc device_desc = {0};
    device_desc.graphicsAPI = NriGraphicsAPI_VK;
    device_desc.vkBindingOffsets = (NriVKBindingOffsets){
        .sRegister = 0,
        .tRegister = 16,
        .bRegister = 32,
        .uRegister = 48
    };

    if (nriCreateDevice(&device_desc, &gpu->device) != NriResult_SUCCESS) {
        gpu_deinit(gpu);
        return false;
    }

    if (nriGetInterface(gpu->device, NRI_INTERFACE(NriCoreInterface), &gpu->core) != NriResult_SUCCESS ||
        nriGetInterface(gpu->device, NRI_INTERFACE(NriHelperInterface), &gpu->helper) != NriResult_SUCCESS) {
        gpu_deinit(gpu);
        return false;
    }

    RENDERER shell = {.gpu = gpu};

    if (!acquire_queues(&shell) || !create_pipeline_cache(&shell) || !create_gpu_timestamps(&shell) ||
        !create_descriptor_pool(&shell) || !create_work_contexts(&shell)) {
        gpu_deinit(gpu);
        return false;
    }

    return true;
}

"""
    source = source[:insert] + worker + source[insert:]

# Extract the remaining bake-only implementation from gpu.c.
layout_blocks = []
for marker in (
    "static bool create_bake_layout(",
    "static bool create_lightmap_queue_layouts(",
    "static bool create_probe_layout(",
):
    source, block = extract_function(source, marker)
    layout_blocks.append(block)

binding_blocks = []
for marker in (
    "bool bake_bind_resources_ex(",
    "bool bake_bind_resources(",
    "bool bake_bind_probe_resources(",
    "bool bake_read_probe_buffer(",
):
    source, block = extract_function(source, marker)
    block = re.sub(r"^(bool bake_)", r"static \1", block, count=1)
    binding_blocks.append(block)

# Remove the old worker lifecycle; bake.c gets a worker built only from generic GPU helpers.
source, _ = extract_function(source, "bool bake_worker_init(")
source, _ = extract_function(source, "void bake_worker_deinit(")

# Generic renderer layout setup delegates the bake layouts to bake.c.
source, count = re.subn(
    r"create_bake_layout\(r\)\s*&&\s*create_lightmap_queue_layouts\(r\)\s*&&\s*create_probe_layout\(r\)\s*&&\s*",
    "bake_gpu_layouts_init(r) && ",
    source,
    count=1,
)
if count != 1:
    raise SystemExit("could not replace bake layout creation")

destroy_marker = "static void destroy_pipeline_layouts(RENDERER *r) {\n    if (!r) return;\n\n"
if destroy_marker not in source:
    raise SystemExit("could not locate pipeline layout teardown")
source = source.replace(destroy_marker, destroy_marker + "    bake_gpu_layouts_deinit(r);\n\n", 1)

for field in (
    "                                     &r->bake_layout,\n",
    "                                     &r->lightmap_queue_reset_layout,\n",
    "                                     &r->lightmap_queue_args_layout,\n",
    "                                     &r->probe_layout,\n",
):
    if field not in source:
        raise SystemExit(f"missing bake layout field in teardown: {field.strip()}")
    source = source.replace(field, "", 1)

# Keep teardown order identical while crossing the private bake boundary.
if "gpu_clear_temporary(r);" not in source:
    raise SystemExit("generic transient cleanup rename failed")

layout_support = "".join(layout_blocks) + """bool bake_gpu_layouts_init(RENDERER *r) {
    return create_bake_layout(r) && create_lightmap_queue_layouts(r) && create_probe_layout(r);
}

void bake_gpu_layouts_deinit(RENDERER *r) {
    if (!r || !r->gpu) return;

    NriPipelineLayout **layouts[] = {
        &r->bake_layout,
        &r->lightmap_queue_reset_layout,
        &r->lightmap_queue_args_layout,
        &r->probe_layout,
    };

    for (uint32_t i = 0; i < sizeof(layouts) / sizeof(layouts[0]); ++i) {
        if (*layouts[i]) {
            r->gpu->core.DestroyPipelineLayout(*layouts[i]);
            *layouts[i] = NULL;
        }
    }
}

"""

binding_support = "".join(binding_blocks)

worker_support = """static void bake_worker_deinit(RENDERER *r);

static bool bake_worker_init(RENDERER *r) {
    if (!r) return false;

    memset(r, 0, sizeof(*r));
    r->gpu = calloc(1, sizeof(*r->gpu));
    if (!r->gpu) return false;

    if (!gpu_init_worker(r->gpu) || !bake_gpu_layouts_init(r)) {
        bake_worker_deinit(r);
        return false;
    }

    r->lightmap_sampler = gpu_create_sampler(r, NriFilter_LINEAR, NriFilter_LINEAR, NriAddressMode_CLAMP_TO_EDGE);
    r->lightmap_texture = gpu_create_texture(r, NriFormat_RGBA16_SFLOAT, NriTextureUsageBits_SHADER_RESOURCE, 1, 1);

    if (!r->lightmap_sampler || !r->lightmap_texture) {
        bake_worker_deinit(r);
        return false;
    }

    return true;
}

static void bake_worker_deinit(RENDERER *r) {
    if (!r) return;

    GPU *gpu = r->gpu;

    free(r->volume_probes.probes);
    memset(&r->volume_probes, 0, sizeof(r->volume_probes));
    beam_free(&r->beams);

    if (gpu && gpu->device) {
        if (gpu->graphics_queue) gpu->core.QueueWaitIdle(gpu->graphics_queue);

        gpu_clear_temporary(r);
        probe_wavefront_scratch_destroy(r);
        release_bake_resources(r);
        release_buffer(r, r->volume_probe_buffer);
        release_buffer(r, r->beam_buffer);
        release_texture(r, r->lightmap_texture);

        if (r->lightmap_sampler) gpu->core.DestroyDescriptor(r->lightmap_sampler);
        bake_gpu_layouts_deinit(r);
    }

    if (gpu) {
        gpu_deinit(gpu);
        free(gpu);
    }

    memset(r, 0, sizeof(*r));
}

"""

insert_marker = "static NriTexture *create_lightmap_texture"
insert = bake.find(insert_marker)
if insert < 0:
    raise SystemExit("could not locate bake implementation insertion point")

bake = bake[:insert] + layout_support + binding_support + worker_support + bake[insert:]

# These two helpers are now bake-private and only crossed through explicit cleanup declarations.
bake = bake.replace("void probe_wavefront_scratch_destroy(RENDERER *r)", "void probe_wavefront_scratch_destroy(RENDERER *r)", 1)
bake = bake.replace("void release_bake_resources(RENDERER *r)", "void release_bake_resources(RENDERER *r)", 1)

# Guard the intended final boundary.
for forbidden in (
    "create_bake_layout(",
    "create_lightmap_queue_layouts(",
    "create_probe_layout(",
    "bake_bind_resources_ex(",
    "bake_bind_resources(",
    "bake_bind_probe_resources(",
    "bake_read_probe_buffer(",
    "bake_worker_init(",
    "bake_worker_deinit(",
):
    if forbidden in source:
        raise SystemExit(f"bake implementation leaked in gpu.c: {forbidden}")

for required in (
    "bake_gpu_layouts_init(",
    "static bool bake_bind_resources_ex(",
    "static bool bake_worker_init(",
    "static void bake_worker_deinit(",
):
    if required not in bake:
        raise SystemExit(f"bake.c missing moved implementation: {required}")

GPU.write_text(source, encoding="utf-8")
BAKE.write_text(bake, encoding="utf-8")
