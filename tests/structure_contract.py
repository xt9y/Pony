#!/usr/bin/env python3
from pathlib import Path
import re
import subprocess

BASE = "0ae0e70364210ec302c8b076ea5513d0dd716612"
ROOT = Path(__file__).resolve().parents[1]


def text(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def require(path: str, *needles: str) -> None:
    source = text(path)
    missing = [needle for needle in needles if needle not in source]
    if missing:
        raise AssertionError(f"{path}: missing {missing}")


def forbid(path: str, *needles: str) -> None:
    source = text(path)
    present = [needle for needle in needles if needle in source]
    if present:
        raise AssertionError(f"{path}: forbidden {present}")


def forbid_symbols(path: str, *names: str) -> None:
    source = text(path)
    present = [name for name in names if re.search(rf"\b{re.escape(name)}\s*\(", source)]
    if present:
        raise AssertionError(f"{path}: forbidden symbols {present}")


def body(source: str, pattern: str, name: str) -> str:
    match = re.search(pattern, source, re.S)
    if not match:
        raise AssertionError(f"could not find {name}")
    return match.group(1)


def main() -> None:
    # User explicitly froze these two files for this restructuring pass.
    subprocess.run(
        ["git", "diff", "--exit-code", BASE, "--", "build.c", "felix-format"],
        cwd=ROOT,
        check=True,
    )

    require(
        "game.h",
        "typedef struct TRANSFORM",
        "typedef struct SCENE",
        "TRANSFORM transform_identity(void);",
        "OBJECT *scene_add_model",
        "OBJECT *scene_add_light",
        "void object_set_transform",
        "void object_mark_dirty",
        "void scene_free",
        "bool renderer_init(RENDERER *renderer, GPU *gpu);",
        "bool renderer_set_scene(RENDERER *renderer, SCENE *scene);",
        "void renderer_event(RENDERER *renderer, const SDL_Event *event);",
        "bool renderer_frame(RENDERER *renderer);",
        "void renderer_deinit(RENDERER *renderer);",
    )

    forbid(
        "gpu.h",
        '#include "game.h"',
        "RENDERER",
        "FX_STATE",
        "RENDER_VERTEX",
        "DRAW_RANGE",
        "RENDER_FRAME",
        "PROBE_WAVEFRONT_SCRATCH",
        "RENDER_MATERIAL",
        "release_texture(",
        "release_buffer(",
    )
    require(
        "game.h",
        '#include "gpu.h"',
        "typedef struct PROBE_WAVEFRONT_SCRATCH PROBE_WAVEFRONT_SCRATCH;",
        "typedef struct RENDER_MATERIAL RENDER_MATERIAL;",
        "typedef struct FX_STATE",
        "typedef struct RENDER_VERTEX",
        "typedef struct DRAW_RANGE",
        "typedef struct RENDER_FRAME",
        "struct RENDERER",
    )
    require(
        "render_internal.h",
        '#include "game.h"',
        "void release_texture(RENDERER *renderer, NriTexture *texture);",
        "void release_buffer(RENDERER *renderer, NriBuffer *buffer);",
    )
    forbid("render_internal.h", '#include "gpu.h"')

    gpu_header = text("gpu.h")
    require(
        "gpu.h",
        "typedef struct GPU",
        "bool gpu_init(GPU *gpu, const char *title, int width, int height);",
        "void gpu_deinit(GPU *gpu);",
        "GPU *gpu;",
    )
    forbid(
        "gpu.h",
        "bool bake_lightmap(",
        "bool bake_probe_grid_fast(",
        "bool bake_probe_grid(",
        "NriTexture *upload_lightmap(",
        "NriBuffer *upload_probes(",
        "NriBuffer *upload_beams(",
        "bool download_lightmap(",
        "bool upload_bvh(",
        "void release_bake_resources(",
        "bool bake_worker_init(",
        "void bake_worker_deinit(",
    )

    gpu = body(gpu_header, r"typedef struct GPU\s*\{(.*?)\}\s*GPU;", "GPU")
    renderer = body(gpu_header, r"struct RENDERER\s*\{(.*?)\n\};", "RENDERER")
    low_level = (
        "SDL_Window *window;",
        "NriDevice *device;",
        "NriCoreInterface core;",
        "NriHelperInterface helper;",
        "NriSwapChainInterface swapchain_api;",
        "NriQueue *graphics_queue;",
        "NriQueue *compute_queue;",
        "NriQueue *copy_queue;",
        "NriQueue *work_queue;",
        "NriSwapChain *swapchain;",
        "NriDescriptorPool *descriptor_pool;",
        "NriFence *frame_fence;",
        "NriFence *work_fence;",
        "FRAME_CONTEXT *frame_contexts;",
        "FRAME_CONTEXT *work_contexts;",
        "NriTexture **swapchain_textures;",
        "uint32_t swapchain_texture_count;",
        "uint64_t frame_index;",
        "NriFormat swapchain_format;",
    )
    missing_gpu = [field for field in low_level if field not in gpu]
    leaked_renderer = [field for field in low_level if field in renderer]
    if missing_gpu:
        raise AssertionError(f"GPU missing low-level ownership: {missing_gpu}")
    if leaked_renderer:
        raise AssertionError(f"RENDERER still owns low-level GPU state: {leaked_renderer}")

    forbid(
        "gpu.c",
        "static bool create_bake_layout(",
        "static bool create_lightmap_queue_layouts(",
        "static bool create_probe_layout(",
        "bool bake_bind_resources_ex(",
        "bool bake_bind_resources(",
        "bool bake_bind_probe_resources(",
        "bool bake_read_probe_buffer(",
        "bool bake_worker_init(",
        "void bake_worker_deinit(",
        "bool bake_lightmap(",
        "bool bake_probe_grid_fast(",
        "bool bake_probe_grid(",
    )
    require(
        "bake.c",
        "static bool create_bake_layout(",
        "static bool create_lightmap_queue_layouts(",
        "static bool create_probe_layout(",
        "bool bake_gpu_layouts_init(",
        "static bool bake_bind_resources_ex(",
        "static bool bake_worker_init(",
        "static void bake_worker_deinit(",
        "bool bake_lightmap(",
        "bool bake_probe_grid_fast(",
        "bool bake_probe_grid(",
    )
    require(
        "render_internal.h",
        "bool gpu_init_worker(GPU *gpu);",
        "bool upload_bvh(RENDERER *renderer, const BVH *tree);",
        "bool bake_gpu_layouts_init(RENDERER *renderer);",
        "void bake_gpu_layouts_deinit(RENDERER *renderer);",
    )

    forbid(
        "gpu.c",
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
        "static bool ensure_depth_texture(",
        "static bool load_images(",
        "static void release_scene_resources(",
        "bool upload_scene(",
        "static bool fx_init(",
        "static bool fx_ensure(",
        "static bool fx_volume(",
        "static bool fx_apply(",
        "bool renderer_gpu_resources_init(",
        "bool draw_frame(",
        "void renderer_gpu_resources_deinit(",
        "static void free_probe_grid(",
    )
    require(
        "render.c",
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
        "static bool ensure_depth_texture(",
        "static bool load_images(",
        "static void release_scene_resources(",
        "static bool upload_scene(",
        "static bool fx_init(",
        "static bool fx_ensure(",
        "static bool fx_volume(",
        "static bool fx_apply(",
        "bool renderer_gpu_resources_init(",
        "static bool draw_frame(",
        "void renderer_gpu_resources_deinit(",
    )
    forbid(
        "gpu.h",
        "bool upload_scene(",
        "bool draw_frame(",
    )

    require(
        "render.c",
        "bool renderer_init(",
        "bool renderer_set_scene(",
        "void renderer_event(",
        "bool renderer_frame(",
        "void renderer_deinit(",
    )

    forbid_symbols("main.c", "r_init", "r_build_scene", "r_event", "r_draw", "r_deinit")
    require("main.c", "GPU gpu", "RENDERER renderer", "renderer_init(", "renderer_set_scene(", "renderer_frame(")

    # build.c is frozen, therefore the existing shader/source paths stay valid.
    for path in (
        "shaders/vertex.hlsl",
        "shaders/fragment.hlsl",
        "shaders/compute.hlsl",
        "shaders/compute_base.hlsl",
        "shaders/vision_compute.hlsl",
        "shaders/lightmap_queue.hlsl",
        "shaders/probe_wavefront.hlsl",
    ):
        if not (ROOT / path).is_file():
            raise AssertionError(f"frozen build graph requires {path}")

    print("dustmite radiance-structure contract: ok")


if __name__ == "__main__":
    main()
