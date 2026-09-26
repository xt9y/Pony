from pathlib import Path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


game = Path("game.h").read_text()
main = Path("main.c").read_text()
gpu = Path("gpu.c").read_text()
vision = Path("shaders/vision_compute.hlsl").read_text()
probe = Path("shaders/probe_wavefront.hlsl").read_text()
compute = Path("shaders/compute.hlsl").read_text()
compute_base = Path("shaders/compute_base.hlsl").read_text()

require("float probe_intensity;" in game, "VOLUMETRICS_LIGHTING needs probe_intensity")
require("float emissive_probe_intensity;" in game, "VOLUMETRICS_LIGHTING needs emissive_probe_intensity")
require("float center_transition_width;" in game, "VOLUMETRICS_LIGHTING needs center_transition_width")
require("float middle_transition_width;" in game, "VOLUMETRICS_LIGHTING needs middle_transition_width")
require("indirect_intensity" not in game, "old indirect_intensity field must be removed")

require(".probe_intensity = 0.15f" in main, "main.c needs default probe_intensity")
require(".emissive_probe_intensity = 1.0f" in main, "main.c needs default emissive probe intensity")
require(".center_transition_width = 0.08f" in main, "main.c needs default center transition width")
require(".middle_transition_width = 0.08f" in main, "main.c needs default middle transition width")
require("volumetrics.emissive_probe_intensity" in main, "emissive probe intensity must participate in the volume bake hash")

require("frame->volumetrics.probe_intensity" in gpu, "runtime volume uniforms must use probe_intensity")
require("frame->volumetrics.center_transition_width" in gpu, "runtime volume uniforms must pass center transition width")
require("frame->volumetrics.middle_transition_width" in gpu, "runtime volume uniforms must pass middle transition width")
require("volumetrics.emissive_probe_intensity" in gpu, "wavefront probe uniforms must carry emissive probe intensity")
require("r->volumetrics.emissive_probe_intensity" in gpu, "legacy probe uniforms must carry emissive probe intensity")

require("float4 emissive_params;" in probe, "probe bake cbuffer needs emissive_params")
require("emissive_params.x" in probe, "wavefront emissive contribution must use emissive probe intensity")
require("emissive_data.z" in compute_base, "legacy direct emissive probe lighting must use emissive probe intensity")
require("hit.emissive * max(emissive_data.z" in compute, "legacy primary emissive hits must use emissive probe intensity")

require("float vision_transition(" in vision, "vision shader needs smooth transition helper")
require("float3 vision_quality_weights(" in vision, "vision shader needs quality crossfade weights")
require("smoothstep(" in vision, "quality transitions must use smoothstep")
require("integrate_probe_quality(" in vision, "volume ray integration must blend neighboring step tiers")
require("reconstructed_volume_stride(" in vision, "compose pass must reconstruct explicit stride tiers")
require("quality.x * reconstructed_volume_stride" in vision, "compose pass must blend center reconstruction")
require("quality.y * reconstructed_volume_stride" in vision, "compose pass must blend middle reconstruction")
require("quality.z * reconstructed_volume_stride" in vision, "compose pass must blend peripheral reconstruction")
require("eccentricity < volume_radii.x ? volume_quality.x" not in vision, "hard probe-step ring switch must be removed")

print("volumetrics controls contract: PASS")
