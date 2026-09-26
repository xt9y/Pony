from pathlib import Path
import re


def replace_once(path: str, old: str, new: str) -> None:
    p = Path(path)
    text = p.read_text()
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected 1 exact match, found {count}: {old[:80]!r}")
    p.write_text(text.replace(old, new, 1))


def replace_count(path: str, old: str, new: str, expected: int) -> None:
    p = Path(path)
    text = p.read_text()
    count = text.count(old)
    if count != expected:
        raise RuntimeError(f"{path}: expected {expected} exact matches, found {count}: {old[:80]!r}")
    p.write_text(text.replace(old, new))


def regex_once(path: str, pattern: str, repl, flags: int = 0) -> None:
    p = Path(path)
    text = p.read_text()
    new, count = re.subn(pattern, repl, text, count=1, flags=flags)
    if count != 1:
        raise RuntimeError(f"{path}: expected 1 regex match, found {count}: {pattern}")
    p.write_text(new)


# Global volumetric controls.
replace_once(
    "game.h",
    """typedef struct VOLUMETRICS_LIGHTING {
    float density;
    float anisotropy;
    float indirect_intensity;
    float max_distance;
    float center_radius;
    float middle_radius;
    float probe_spacing;
    uint32_t center_steps;
    uint32_t middle_steps;
    uint32_t peripheral_steps;
    uint32_t center_stride;
    uint32_t middle_stride;
    uint32_t peripheral_stride;
    uint32_t probe_samples;
    uint32_t emissive_samples;
} VOLUMETRICS_LIGHTING;""",
    """typedef struct VOLUMETRICS_LIGHTING {
    float density;
    float anisotropy;
    float probe_intensity;
    float emissive_probe_intensity;
    float max_distance;
    float center_radius;
    float middle_radius;
    float center_transition_width;
    float middle_transition_width;
    float probe_spacing;
    uint32_t center_steps;
    uint32_t middle_steps;
    uint32_t peripheral_steps;
    uint32_t center_stride;
    uint32_t middle_stride;
    uint32_t peripheral_stride;
    uint32_t probe_samples;
    uint32_t emissive_samples;
} VOLUMETRICS_LIGHTING;""",
)

replace_once("main.c", ".indirect_intensity = 0.15f,", ".probe_intensity = 0.15f,\n                                         .emissive_probe_intensity = 1.0f,")
replace_once("main.c", ".middle_radius = 0.82f,", ".middle_radius = 0.82f,\n                                         .center_transition_width = 0.08f,\n                                         .middle_transition_width = 0.08f,")
regex_once(
    "main.c",
    r"const float volume_bake_settings\[\] = \{volumetrics\.probe_spacing,\s*\(float\)volumetrics\.probe_samples,\s*\(float\)volumetrics\.emissive_samples,\s*9\.0f\};",
    """const float volume_bake_settings[] = {volumetrics.probe_spacing,
                                          (float)volumetrics.probe_samples,
                                          (float)volumetrics.emissive_samples,
                                          volumetrics.emissive_probe_intensity,
                                          9.0f};""",
)

# Runtime volume uniforms and bake-time emissive gain.
replace_once(
    "gpu.c",
    ".volume_params = {frame->volumetrics.density, frame->volumetrics.anisotropy, frame->volumetrics.indirect_intensity, frame->volumetrics.max_distance},",
    ".volume_params = {frame->volumetrics.density, frame->volumetrics.anisotropy, frame->volumetrics.probe_intensity, frame->volumetrics.max_distance},",
)
replace_count(
    "gpu.c",
    ".volume_radii = {frame->volumetrics.center_radius, frame->volumetrics.middle_radius, 0.0f, 0.0f},",
    ".volume_radii = {frame->volumetrics.center_radius, frame->volumetrics.middle_radius, frame->volumetrics.center_transition_width, frame->volumetrics.middle_transition_width},",
    2,
)
replace_once(
    "gpu.c",
    ".emissive_data = {r->bvh_emissive_weight, (float)r->bvh_triangle_count, 0.0f, 0.0f}};",
    ".emissive_data = {r->bvh_emissive_weight, (float)r->bvh_triangle_count, r->volumetrics.emissive_probe_intensity, 0.0f}};",
)
regex_once(
    "gpu.c",
    r"(typedef struct PROBE_WAVEFRONT_UNIFORMS \{.*?\n\s*float bake_params\[4\];)",
    lambda m: m.group(1) + "\n    float emissive_params[4];",
    re.S,
)
replace_once(
    "gpu.c",
    ".bake_params = {epsilon, 0.72f, sky.intensity, tree->emissive_weight},",
    ".bake_params = {epsilon, 0.72f, sky.intensity, tree->emissive_weight},\n                                      .emissive_params = {volumetrics.emissive_probe_intensity, 0.0f, 0.0f, 0.0f},",
)

# Fast wavefront probe bake: keep the default energy identical, but expose a
# relative emissive-only multiplier for probe/volume lighting.
replace_once("shaders/probe_wavefront.hlsl", "    float4 bake_params;\n    float4 probe_beam_origin;", "    float4 bake_params;\n    float4 emissive_params;\n    float4 probe_beam_origin;")
replace_once(
    "shaders/probe_wavefront.hlsl",
    "state.radiance = emissive_samples == 0u ? float4(hit.emissive, 0.0f) : 0.0f;",
    "state.radiance = emissive_samples == 0u ? float4(hit.emissive * max(emissive_params.x, 0.0f), 0.0f) : 0.0f;",
)
replace_once(
    "shaders/probe_wavefront.hlsl",
    "float3 contribution = max(tri.emissive.rgb, 0.0f) / pdf_solid_angle;",
    "float3 contribution = max(tri.emissive.rgb, 0.0f) * max(emissive_params.x, 0.0f) / pdf_solid_angle;",
)
replace_once(
    "shaders/probe_wavefront.hlsl",
    """    return max(tri.emissive.rgb, 0.0f) *
        (receiver_cosine * emitter_cosine / max(distance2 * pdf_area, 1.0e-8f));""",
    """    return max(tri.emissive.rgb, 0.0f) * max(emissive_params.x, 0.0f) *
        (receiver_cosine * emitter_cosine / max(distance2 * pdf_area, 1.0e-8f));""",
)

# Legacy probe path gets the same relative emissive multiplier, without
# changing surface-lightmap emissive energy.
replace_once(
    "shaders/compute.hlsl",
    """    return max(tri.emissive.rgb, 0.0f) *
        (receiver_cosine * emitter_cosine / max(distance2 * pdf_area, 1.0e-8f));""",
    """    float emissive_scale = 1.0f;
#if defined(BUILD_PROBE_CS)
    emissive_scale = max(emissive_data.z, 0.0f);
#endif
    return max(tri.emissive.rgb, 0.0f) * emissive_scale *
        (receiver_cosine * emitter_cosine / max(distance2 * pdf_area, 1.0e-8f));""",
)
replace_once(
    "shaders/compute.hlsl",
    "incoming = trace_path(position, hit.normal, seed) * (hit.albedo / PI);",
    "incoming = hit.emissive * max(emissive_data.z, 0.0f) +\n                    trace_path(position, hit.normal, seed) * (hit.albedo / PI);",
)

# Smooth foveated quality transitions.  The transition bands use the denser
# neighboring lattice and crossfade both ray-march step quality and compose
# reconstruction, so there is no radial discontinuity at the LOD radii.
vision_path = Path("shaders/vision_compute.hlsl")
vision = vision_path.read_text()
old_stride = """uint vision_stride(float eccentricity)
{
    if (eccentricity < volume_radii.x) return max(volume_strides.x, 1u);
    if (eccentricity < volume_radii.y) return max(volume_strides.y, 1u);
    return max(volume_strides.z, 1u);
}
"""
common_helpers = """float vision_transition(float eccentricity, float radius, float width)
{
    if (width <= 1.0e-5f)
        return eccentricity < radius ? 0.0f : 1.0f;
    float half_width = 0.5f * width;
    return smoothstep(radius - half_width, radius + half_width, eccentricity);
}

float3 vision_quality_weights(float eccentricity)
{
    float center_to_middle = vision_transition(eccentricity, volume_radii.x, volume_radii.z);
    float middle_to_peripheral = vision_transition(eccentricity, volume_radii.y, volume_radii.w);
    float3 weights = float3(1.0f - center_to_middle,
                            center_to_middle * (1.0f - middle_to_peripheral),
                            middle_to_peripheral);
    return weights / max(weights.x + weights.y + weights.z, 1.0e-5f);
}

uint vision_stride(float eccentricity)
{
    float3 quality = vision_quality_weights(eccentricity);
    uint stride = 0xffffffffu;
    if (quality.x > 1.0e-4f) stride = min(stride, max(volume_strides.x, 1u));
    if (quality.y > 1.0e-4f) stride = min(stride, max(volume_strides.y, 1u));
    if (quality.z > 1.0e-4f) stride = min(stride, max(volume_strides.z, 1u));
    return stride == 0xffffffffu ? 1u : stride;
}
"""
integrate_helper = """
float3 integrate_probe_quality(uint probe_steps, float3 direction,
                               float enter, float leave,
                               float3 surface_position, float3 surface_normal,
                               bool has_surface)
{
    probe_steps = min(max(probe_steps, 1u), 4u);
    float step_size = (leave - enter) / float(probe_steps);
    float probe_transmission = exp(-volume_params.x * step_size);
    float probe_remaining = 1.0f;
    float3 sum = 0.0f;

    [loop] for (uint i = 0u; i < 4u; ++i) {
        if (i >= probe_steps) break;
        float t = enter + (float(i) + 0.5f) * step_size;
        float integral = probe_remaining * (1.0f - probe_transmission);
        sum += volume_radiance(eye_density.xyz + direction * t, direction,
                               surface_position, surface_normal, has_surface) *
               integral * volume_params.z;
        probe_remaining *= probe_transmission;
    }

    return sum;
}
"""
first = vision.find(old_stride)
if first < 0:
    raise RuntimeError("vision_compute.hlsl: first vision_stride block not found")
vision = vision[:first] + common_helpers + integrate_helper + vision[first + len(old_stride):]
second = vision.find(old_stride, first + len(common_helpers) + len(integrate_helper))
if second < 0:
    raise RuntimeError("vision_compute.hlsl: second vision_stride block not found")
vision = vision[:second] + common_helpers + vision[second + len(old_stride):]

old_probe_block = """    uint probe_steps = eccentricity < volume_radii.x ? volume_quality.x :
        (eccentricity < volume_radii.y ? volume_quality.y : volume_quality.z);
    probe_steps = max(probe_steps, 1u);
    float step_size = (leave - enter) / float(probe_steps);
    float3 sum = 0.0f;
    float sun_fraction = 0.0f;
    float probe_transmission = exp(-volume_params.x * step_size);
    float probe_remaining = 1.0f;

    [loop] for (uint i = 0u; i < 4u; ++i) {
        if (i >= probe_steps) break;
        float t = enter + (float(i) + 0.5f) * step_size;
        float integral = probe_remaining * (1.0f - probe_transmission);
        sum += volume_radiance(eye_density.xyz + direction * t, direction,
                               surface_position, surface_normal, depth > 0.0f) *
               integral * volume_params.z;
        probe_remaining *= probe_transmission;
    }
"""
new_probe_block = """    float3 quality = vision_quality_weights(eccentricity);
    float3 sum = 0.0f;
    if (quality.x > 1.0e-4f)
        sum += quality.x * integrate_probe_quality(volume_quality.x, direction, enter, leave,
                                                   surface_position, surface_normal, depth > 0.0f);
    if (quality.y > 1.0e-4f)
        sum += quality.y * integrate_probe_quality(volume_quality.y, direction, enter, leave,
                                                   surface_position, surface_normal, depth > 0.0f);
    if (quality.z > 1.0e-4f)
        sum += quality.z * integrate_probe_quality(volume_quality.z, direction, enter, leave,
                                                   surface_position, surface_normal, depth > 0.0f);
    float sun_fraction = 0.0f;
"""
if vision.count(old_probe_block) != 1:
    raise RuntimeError(f"vision_compute.hlsl: expected one hard probe quality block, found {vision.count(old_probe_block)}")
vision = vision.replace(old_probe_block, new_probe_block, 1)

old_reconstruct_start = """float4 reconstructed_volume(float2 uv, float center_depth)
{
    uint volume_width, volume_height;
    Volume.GetDimensions(volume_width, volume_height);
    uint stride = vision_stride(vision_eccentricity(uv));

    float2 volume_size = float2(volume_width, volume_height);"""
new_reconstruct_start = """float4 reconstructed_volume_stride(float2 uv, float center_depth, uint stride)
{
    uint volume_width, volume_height;
    Volume.GetDimensions(volume_width, volume_height);
    stride = max(stride, 1u);

    float2 volume_size = float2(volume_width, volume_height);"""
if vision.count(old_reconstruct_start) != 1:
    raise RuntimeError("vision_compute.hlsl: reconstruction start not found")
vision = vision.replace(old_reconstruct_start, new_reconstruct_start, 1)

old_reconstruct_end = """    return weight_sum > 1.0e-5f ? fog / weight_sum : closest_fog;
}

float3 vision_hdr"""
new_reconstruct_end = """    return weight_sum > 1.0e-5f ? fog / weight_sum : closest_fog;
}

float4 reconstructed_volume(float2 uv, float center_depth)
{
    float3 quality = vision_quality_weights(vision_eccentricity(uv));
    float4 fog = 0.0f;
    if (quality.x > 1.0e-4f)
        fog += quality.x * reconstructed_volume_stride(uv, center_depth, volume_strides.x);
    if (quality.y > 1.0e-4f)
        fog += quality.y * reconstructed_volume_stride(uv, center_depth, volume_strides.y);
    if (quality.z > 1.0e-4f)
        fog += quality.z * reconstructed_volume_stride(uv, center_depth, volume_strides.z);
    return fog;
}

float3 vision_hdr"""
if vision.count(old_reconstruct_end) != 1:
    raise RuntimeError("vision_compute.hlsl: reconstruction end not found")
vision = vision.replace(old_reconstruct_end, new_reconstruct_end, 1)
vision_path.write_text(vision)

print("volumetrics controls patch applied")
