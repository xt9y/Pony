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
