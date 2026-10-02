#!/usr/bin/env python3
from pathlib import Path
import re

GPU = Path("gpu.c")
RENDER = Path("render.c")

gpu = GPU.read_text(encoding="utf-8")
render = RENDER.read_text(encoding="utf-8")


def extract_braced(text: str, marker: str, semicolon: bool = False):
    start = text.find(marker)
    if start < 0:
        raise SystemExit(f"missing marker: {marker}")

    brace = text.find("{", start)
    if brace < 0:
        raise SystemExit(f"missing body: {marker}")

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
        raise SystemExit(f"unterminated block: {marker}")

    if semicolon:
        semi = text.find(";", end)
        if semi < 0:
            raise SystemExit(f"missing semicolon: {marker}")
        end = semi + 1

    while end < len(text) and text[end] == "\n":
        end += 1

    block = text[start:end].rstrip() + "\n\n"
    return text[:start] + text[end:], block


def extract_function(text: str, marker: str):
    return extract_braced(text, marker, False)


# Generic primitives stay in gpu.c but get explicit internal gpu_* names.
gpu = re.sub(r"\bcompile_shader\s*\(", "gpu_load_shader(", gpu)
gpu = re.sub(r"\bfree_shader\s*\(", "gpu_free_shader(", gpu)
gpu = re.sub(r"\btexture_barrier\s*\(", "gpu_texture_barrier(", gpu)

gpu = gpu.replace(
    "NriShaderDesc gpu_load_shader(const char *path, const char *entrypoint, const char *define, NriStageBits stage) {\\n    (void)path;\\n",
    "NriShaderDesc gpu_load_shader(const char *entrypoint, const char *define, NriStageBits stage) {\\n",
)

for name in ("gpu_load_shader", "gpu_free_shader", "gpu_texture_barrier"):
    gpu = re.sub(rf"(^|\n)static (?=[^\n]*\b{name}\s*\()", r"\1", gpu)

# Renderer-only compile-time baggage leaves gpu.c.
gpu = gpu.replace("#include <SDL3_image/SDL_image.h>\n\n", "")
for define in (
    "#define BAKE_TARGET_SAMPLES 128u\n",
    "#define BAKE_MAX_BOUNCES 3u\n",
    "#define BAKE_BATCH_SAMPLES 8u\n",
    "#define BAKE_DILATION_PASSES 3u\n",
    "#define PHASE_CLEAR 0u\n",
    "#define PHASE_TRACE 1u\n",
    "#define PHASE_FILTER 2u\n",
    "#define PHASE_DILATE 3u\n",
    "#define PHASE_DIRECT 4u\n",
    "#define PHASE_COMBINE 5u\n",
    "#define PHASE_RECONSTRUCT 6u\n",
):
    gpu = gpu.replace(define, "")

moved = []

for marker in (
    "typedef struct CAMERA_UNIFORMS",
    "typedef struct SKY_UNIFORMS",
    "typedef struct MATERIAL_UNIFORMS",
    "typedef struct SSAO_UNIFORMS",
    "typedef struct BLOOM_UNIFORMS",
    "typedef struct COMPOSE_UNIFORMS",
    "typedef struct VOLUME_UNIFORMS",
    "typedef struct VOLUME_COMPOSE_UNIFORMS",
    "struct RENDER_MATERIAL",
):
    gpu, block = extract_braced(gpu, marker, True)
    moved.append(block)

# Keep moved renderer functions in their original conceptual order.
functions = (
    "static void free_probe_grid(",
    "static NriPipeline *make_surface_pipeline(",
    "static bool create_surface_layout(",
    "static bool create_line_layout(",
    "static bool create_sky_layout(",
    "static bool create_ssao_layout(",
    "static bool create_bloom_layout(",
    "static bool create_grade_layout(",
    "static bool create_volume_layout(",
    "static bool create_volume_compose_layout(",
    "static bool create_compose_layout(",
    "static bool bind_fx_resources(",
    "static bool bind_volume_resources(",
    "static bool bind_volume_compose_resources(",
    "static bool bind_sky_resources(",
    "static bool bind_camera_resources(",
    "static bool bind_surface_resources(",
    "static bool bind_line_resources(",
    "static bool begin_scene_rendering(",
    "static bool begin_compose_rendering(",
    "static NriPipeline *make_line_pipeline(",
    "static NriPipeline *make_sky_pipeline(",
    "static NriTexture *pixel_texture(",
    "static bool ensure_depth_texture(",
    "static const char *image_type(",
    "static NriTexture *load_image(",
    "static bool load_images(",
    "static NriTexture *resolve_texture(",
    "static void release_scene_resources(",
    "bool upload_scene(",
    "static bool dispatch_one(",
    "static void release_frame_textures(",
    "static void fx_deinit(",
    "static bool make_compose_pipeline(",
    "static bool fx_init(",
    "static bool fx_ensure(",
    "static bool fx_volume(",
    "static bool run_vision_only(",
    "static bool bloom_pass(",
    "static bool fx_apply_base(",
    "static bool fx_apply(",
    "static bool create_pipeline_layouts(",
    "static void destroy_pipeline_layouts(",
    "bool renderer_gpu_resources_init(",
    "bool draw_frame(",
    "void renderer_gpu_resources_deinit(",
)

for marker in functions:
    gpu, block = extract_function(gpu, marker)
    moved.append(block)

renderer_code = "".join(moved)

# Renderer-local functions no longer belong to gpu.h's public surface.
renderer_code = renderer_code.replace("bool upload_scene(RENDERER *r", "static bool upload_scene(RENDERER *r", 1)
renderer_code = renderer_code.replace("bool draw_frame(RENDERER *r", "static bool draw_frame(RENDERER *r", 1)

# The render side uses a narrow generic frame boundary instead of gpu.c internals.
old_resize = """    if (!r->gpu->swapchain || width != r->gpu->swapchain_width || height != r->gpu->swapchain_height) {
        if (r->gpu->core.QueueWaitIdle(r->gpu->graphics_queue) != NriResult_SUCCESS) return false;
        destroy_swapchain(r);

        if (!create_swapchain(r, width, height)) return false;
    }

"""
if old_resize not in renderer_code:
    raise SystemExit("draw-frame resize block changed")
renderer_code = renderer_code.replace(old_resize, "    if (!gpu_ensure_swapchain(r, width, height)) return false;\n\n", 1)

old_acquire = """    uint32_t swap_index = 0;

    if (!acquire_swapchain_texture(r, &swap_index)) {
        destroy_swapchain(r);

        return false;
    }

    r->gpu->current_swap_index = swap_index;

    NriTexture *swap = r->gpu->swapchain_textures[swap_index];

    FRAME_CONTEXT *queued_frame = NULL;
    NriCommandBuffer *cmd = NULL;

    if (!begin_frame_commands(r, &queued_frame, &cmd)) goto failed_frame;
"""
new_acquire = """    uint32_t swap_index = 0;
    NriTexture *swap = NULL;
    FRAME_CONTEXT *queued_frame = NULL;
    NriCommandBuffer *cmd = NULL;

    if (!gpu_begin_render_frame(r, &queued_frame, &cmd, &swap, &swap_index)) goto failed_frame;
"""
if old_acquire not in renderer_code:
    raise SystemExit("draw-frame acquire block changed")
renderer_code = renderer_code.replace(old_acquire, new_acquire, 1)
renderer_code = renderer_code.replace("if (!submit_frame(r, queued_frame, cmd, swap_index)) return false;",
                                      "if (!gpu_submit_render_frame(r, queued_frame, cmd, swap_index)) return false;", 1)
renderer_code = renderer_code.replace("    abort_frame_commands(r, queued_frame);\n    destroy_swapchain(r);",
                                      "    gpu_abort_render_frame(r, queued_frame);", 1)

# Swapchain attachment metadata remains private to gpu.c.
old_attachment = "r->gpu->swapchain_frames[r->gpu->current_swap_index].color_attachment"
renderer_code = renderer_code.replace(old_attachment, "gpu_swapchain_color_attachment(r, r->gpu->current_swap_index)")

# Generic shader lifetime is explicit across the internal boundary.
renderer_code = renderer_code.replace("gpu_load_shader(\"shaders/vertex.hlsl\", ", "gpu_load_shader(", 10)
renderer_code = renderer_code.replace("gpu_load_shader(\"shaders/fragment.hlsl\", ", "gpu_load_shader(", 10)
# gpu_load_shader ignores source path; compute shader paths are still handled by gpu_compile_compute.
renderer_code = re.sub(r'gpu_load_shader\("([^"]+)",\s*"([^"]+)",\s*(NriStageBits_[A-Z_]+)\)',
                       r'gpu_load_shader("\1", "\2", \3)', renderer_code)

# Add generic frame wrappers next to gpu_init.
insert_marker = "bool gpu_init_worker(GPU *gpu) {"
insert = gpu.find(insert_marker)
if insert < 0:
    raise SystemExit("gpu_init_worker insertion point missing")

wrappers = """bool gpu_ensure_swapchain(RENDERER *r, uint32_t width, uint32_t height) {
    if (!r || !r->gpu || !width || !height) return false;

    if (r->gpu->swapchain && width == r->gpu->swapchain_width && height == r->gpu->swapchain_height) return true;

    if (r->gpu->graphics_queue && r->gpu->core.QueueWaitIdle(r->gpu->graphics_queue) != NriResult_SUCCESS) return false;

    destroy_swapchain(r);
    return create_swapchain(r, width, height);
}

bool gpu_begin_render_frame(RENDERER *r, FRAME_CONTEXT **frame, NriCommandBuffer **command_buffer, NriTexture **swapchain_texture,
                            uint32_t *swapchain_index) {
    if (!r || !r->gpu || !frame || !command_buffer || !swapchain_texture || !swapchain_index) return false;

    uint32_t index = 0;

    if (!acquire_swapchain_texture(r, &index)) {
        destroy_swapchain(r);
        return false;
    }

    r->gpu->current_swap_index = index;

    if (!begin_frame_commands(r, frame, command_buffer)) {
        destroy_swapchain(r);
        return false;
    }

    *swapchain_texture = r->gpu->swapchain_textures[index];
    *swapchain_index = index;
    return true;
}

bool gpu_submit_render_frame(RENDERER *r, FRAME_CONTEXT *frame, NriCommandBuffer *command_buffer, uint32_t swapchain_index) {
    return submit_frame(r, frame, command_buffer, swapchain_index);
}

void gpu_abort_render_frame(RENDERER *r, FRAME_CONTEXT *frame) {
    if (!r) return;

    abort_frame_commands(r, frame);
    destroy_swapchain(r);
}

NriDescriptor *gpu_swapchain_color_attachment(RENDERER *r, uint32_t swapchain_index) {
    if (!r || !r->gpu || !r->gpu->swapchain_frames || swapchain_index >= r->gpu->swapchain_texture_count) return NULL;
    return r->gpu->swapchain_frames[swapchain_index].color_attachment;
}

"""
gpu = gpu[:insert] + wrappers + gpu[insert:]

# Move renderer code before existing renderer orchestration.
render_insert = render.find("void bake_progress(")
if render_insert < 0:
    raise SystemExit("render insertion point missing")

if "#include <SDL3_image/SDL_image.h>" not in render:
    render = render.replace('#include "render_internal.h"\n\n', '#include "render_internal.h"\n\n#include <SDL3_image/SDL_image.h>\n\n')

render = render[:render_insert] + renderer_code + render[render_insert:]

# Remove stale public declarations now that these are render-local.
for forbidden in (
    "bool upload_scene(RENDERER *renderer, const GLTF_SCENE *visual);\n",
    "bool draw_frame(RENDERER *renderer, const RENDER_FRAME *frame);\n",
):
    gpu_header = Path("gpu.h").read_text(encoding="utf-8")
    if forbidden in gpu_header:
        gpu_header = gpu_header.replace(forbidden, "")
        Path("gpu.h").write_text(gpu_header, encoding="utf-8")

# Guard the final ownership.
for name in (
    "create_surface_layout(",
    "bind_fx_resources(",
    "ensure_depth_texture(",
    "load_images(",
    "upload_scene(",
    "fx_init(",
    "fx_apply(",
    "renderer_gpu_resources_init(",
    "draw_frame(",
    "renderer_gpu_resources_deinit(",
):
    if name in gpu:
        raise SystemExit(f"renderer implementation leaked in gpu.c: {name}")

for name in (
    "static bool create_surface_layout(",
    "static bool bind_fx_resources(",
    "static bool ensure_depth_texture(",
    "static bool load_images(",
    "static bool upload_scene(",
    "static bool fx_init(",
    "static bool fx_apply(",
    "bool renderer_gpu_resources_init(",
    "static bool draw_frame(",
    "void renderer_gpu_resources_deinit(",
):
    if name not in render:
        raise SystemExit(f"render.c missing moved implementation: {name}")

GPU.write_text(gpu, encoding="utf-8")
RENDER.write_text(render, encoding="utf-8")
