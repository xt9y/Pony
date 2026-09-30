from pathlib import Path
import re

shader = Path('shader.hlsl')
s = shader.read_text()
render = Path('render.c')
r = render.read_text()

def replace_once(text, old, new, label):
    count = text.count(old)
    if count != 1:
        raise SystemExit(f'{label}: expected 1 occurrence, got {count}')
    return text.replace(old, new, 1)

# Shader constants: correct Lambertian scale and use a small deterministic
# sample set for emissive NEE instead of a single firefly-prone sample.
s = replace_once(
    s,
    '#define PI 3.14159265358979323846f\n#define TWO_PI 6.28318530717958647692f',
    '#define PI 3.14159265358979323846f\n#define INV_PI 0.31830988618379067154f\n#define TWO_PI 6.28318530717958647692f',
    'INV_PI'
)
s = replace_once(
    s,
    '#define SCREEN_PROBE_RAY_COUNT 16u',
    '#define SCREEN_PROBE_RAY_COUNT 16u\n#define DIRECT_EMISSIVE_SAMPLE_COUNT 4u',
    'emissive sample count'
)

# Smaller shadow tmin after the origin is already offset. Keeping the old bias
# as both origin offset and tmin skipped large regions around corners.
s = replace_once(
    s,
    'TraceRay shadow = MakeTraceRay(surface_position + surface_normal * bias, L, bias, max(distance - bias * 2.0f, bias), TRACE_RAY_SHADOW, 0u, uint2(0u, 0u), source_object_id);',
    'TraceRay shadow = MakeTraceRay(surface_position + surface_normal * bias, L, 1.0e-4f, max(distance - bias, 1.0e-4f), TRACE_RAY_SHADOW, 0u, uint2(0u, 0u), source_object_id);',
    'emissive shadow bias'
)
s = replace_once(
    s,
    'float tmax = max(max_distance - bias * 2.0f, bias);\n        TraceRay shadow = MakeTraceRay(position + normal * bias, L, bias, tmax, TRACE_RAY_SHADOW, 0u, uint2(0u, 0u), hit.identity.x);',
    'float tmax = max(max_distance - bias, 1.0e-4f);\n        TraceRay shadow = MakeTraceRay(position + normal * bias, L, 1.0e-4f, tmax, TRACE_RAY_SHADOW, 0u, uint2(0u, 0u), hit.identity.x);',
    'analytic shadow bias'
)

# Physically-correct Lambertian outgoing radiance for analytic lights.
s = replace_once(
    s,
    'reflected += material.base_color.rgb * light.color_intensity.rgb * light.color_intensity.w * attenuation * ndotl;',
    'reflected += material.base_color.rgb * light.color_intensity.rgb * light.color_intensity.w * attenuation * ndotl * INV_PI;',
    'Lambertian scale'
)

# Replace the single area-light sample with a deterministic 4-sample estimate.
s = replace_once(
    s,
    'float3 emissive_direct = EvaluateEmissiveSampleForMaterial(position, normal, hit.identity.x, hit.identity.y, seed);\n    reflected += material.base_color.rgb * emissive_direct;',
    '''float3 emissive_direct = 0.0f;\n    [unroll]\n    for (uint sample_index = 0u; sample_index < DIRECT_EMISSIVE_SAMPLE_COUNT; ++sample_index) {\n        uint sample_seed = HashCombine(seed, 0x9e3779b9u * (sample_index + 1u));\n        emissive_direct += EvaluateEmissiveSampleForMaterial(position, normal, hit.identity.x, hit.identity.y, sample_seed);\n    }\n    emissive_direct /= (float)DIRECT_EMISSIVE_SAMPLE_COUNT;\n    reflected += material.base_color.rgb * emissive_direct;''',
    'emissive multi sample'
)

# Visible direct evaluation uses the same geometry-addressed seed as offscreen
# cache misses. Moving the camera therefore does not reshuffle emitter samples.
s = replace_once(
    s,
    'uint seed = HashCombine(Pass.dispatch.x, PackPixel(pixel));\n    float3 reflected = EvaluateSurfaceReflectedDirect(hit, seed);',
    'uint seed = HashCombine(SurfaceCacheKey(hit), HashCombine(Pass.dispatch.x, Radiance.feature_flags.y));\n    float3 reflected = EvaluateSurfaceReflectedDirect(hit, seed);',
    'stable direct seed'
)

# Keep the already-frozen future wavefront path consistent with the canonical
# surface-radiance path too.
s = replace_once(
    s,
    'RayRadiance[index] = float4(ReflectedDirectAtPixel(UnpackPixel(hit.meta.x)), 1.0f);',
    'RayRadiance[index] = float4(SurfaceReflectedRadiance(hit), 1.0f);',
    'future wavefront visible shading'
)

# Replace blind four-probe bilinear interpolation with a 3x3 depth/normal-aware
# filter. This both prevents wall-to-wall color leakage and averages the raw
# 16-ray probe noise without adding a new persistent temporal system.
pattern = r'''float3 LegacyInterpolateScreenProbeGI\(int2 pixel\) \{.*?\n\}'''
replacement = r'''float3 LegacyInterpolateScreenProbeGI(int2 pixel) {
    uint full_width, full_height;
    DepthTexture.GetDimensions(full_width, full_height);

    float depth = DepthTexture.Load(int3(pixel, 0));
    if (depth <= 0.0f) return 0.0f;

    float3 normal = normalize(NormalRoughnessTexture.Load(int3(pixel, 0)).xyz);
    float linear_depth = LinearizeDepth(depth);
    float2 probe_position = (float2(pixel) + 0.5f) / (float)SCREEN_PROBE_TILE_SIZE - 0.5f;
    int2 center = int2(round(probe_position));

    uint probe_width, probe_height;
    ScreenProbesTexture.GetDimensions(probe_width, probe_height);

    float3 sum = 0.0f;
    float weight_sum = 0.0f;
    float depth_scale = max(0.05f, linear_depth * 0.035f);

    [unroll]
    for (int y = -1; y <= 1; ++y) {
        [unroll]
        for (int x = -1; x <= 1; ++x) {
            int2 probe = clamp(center + int2(x, y), int2(0, 0), int2((int)probe_width - 1, (int)probe_height - 1));
            uint2 representative = min(
                uint2(probe) * SCREEN_PROBE_TILE_SIZE + uint2(SCREEN_PROBE_TILE_SIZE / 2u, SCREEN_PROBE_TILE_SIZE / 2u),
                uint2(full_width - 1u, full_height - 1u)
            );

            float probe_depth = DepthTexture.Load(int3(representative, 0));
            if (probe_depth <= 0.0f) continue;

            float3 probe_normal = normalize(NormalRoughnessTexture.Load(int3(representative, 0)).xyz);
            float normal_similarity = saturate(dot(normal, probe_normal));
            if (normal_similarity <= 0.35f) continue;

            float probe_linear_depth = LinearizeDepth(probe_depth);
            float depth_weight = exp(-abs(probe_linear_depth - linear_depth) / depth_scale);
            float2 delta = float2(probe) - probe_position;
            float spatial_weight = rcp(1.0f + dot(delta, delta));
            float normal_weight = normal_similarity * normal_similarity;
            normal_weight *= normal_weight;
            float weight = spatial_weight * depth_weight * normal_weight;

            sum += ScreenProbesTexture.Load(int3(probe, 0)).rgb * weight;
            weight_sum += weight;
        }
    }

    if (weight_sum > 1.0e-5f) return sum / weight_sum;
    return LegacyLoadScreenProbe(center);
}'''
s, count = re.subn(pattern, replacement, s, count=1, flags=re.S)
if count != 1:
    raise SystemExit(f'probe filter: expected 1 replacement, got {count}')

shader.write_text(s)

# Shrink the world-space bias by 20x. 0.10 was a huge gap for Cornell-sized
# geometry and was the source of the bright seam/edge leaks.
r = replace_once(
    r,
    'constants.trace_params[1] = 0.10f;',
    'constants.trace_params[1] = 0.005f;',
    'radiance world bias'
)
r = replace_once(
    r,
    '.trace_params = {200.0f, 0.05f, 0.10f, 0.05f},',
    '.trace_params = {200.0f, 0.05f, 0.005f, 0.05f},',
    'legacy probe bias'
)
render.write_text(r)
