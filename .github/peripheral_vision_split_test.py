from pathlib import Path


game = Path("game.h").read_text()
gpu_h = Path("gpu.h").read_text()
main = Path("main.c").read_text()
render = Path("render.c").read_text()
gpu = Path("gpu.c").read_text()
compute = Path("shaders/compute.hlsl").read_text()
vision = Path("shaders/vision_compute.hlsl").read_text()

checks = {
    "separate PERIPHERAL_VISION struct": "typedef struct PERIPHERAL_VISION" in game,
    "lighting struct no center radius": "center_radius" not in game.split("typedef struct VOLUMETRICS_LIGHTING", 1)[1].split("} VOLUMETRICS_LIGHTING;", 1)[0],
    "render frame has vision settings": "PERIPHERAL_VISION vision;" in gpu_h,
    "main creates vision settings": "PERIPHERAL_VISION vision =" in main,
    "draw receives vision settings": "const PERIPHERAL_VISION *vision" in render,
    "draw forwards vision settings": ".vision = *vision" in render,
    "volume uniforms include filter controls": "float volume_filter[4]" in gpu,
    "gpu uses frame vision radii": "frame->vision.center_radius" in gpu,
    "gpu uses frame vision jitter": "frame->vision.jitter_strength" in gpu,
    "volume cbuffer includes filter controls": "float4 volume_filter;" in compute,
    "compose cbuffer includes filter controls": "float4 volume_filter;" in vision,
    "jitter uses configured strength": "volume_filter.x" in vision,
    "blur uses configured strength": "volume_filter.y" in vision,
    "old four-step clamp removed": "min(max(probe_steps, 1u), 4u)" not in vision,
    "probe loop honors configured count": "for (uint i = 0u; i < probe_steps; ++i)" in vision,
}

failed = [name for name, ok in checks.items() if not ok]
if failed:
    raise SystemExit("missing: " + ", ".join(failed))

print("peripheral vision split contract: PASS")
