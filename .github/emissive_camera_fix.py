from pathlib import Path

p = Path('shader.hlsl')
s = p.read_text()

# 1) Explicit emissive gather: do not light emissive receiver pixels with the
# same area-light sampler, and keep the estimator in irradiance/pi units to
# match the cosine-sampled legacy screen probes.
old = '''float3 EvaluateEmissiveSample(float3 surface_position, float3 surface_normal, uint source_object_id, uint2 source_pixel, uint seed) {
    if (!FeatureEnabled(RADIANCE_FEATURE_EMISSIVE) || Radiance.scene_counts.w == 0u) return 0.0f;
'''
new = '''float3 EvaluateEmissiveSample(float3 surface_position, float3 surface_normal, uint source_object_id, uint2 source_pixel, uint seed) {
    if (!FeatureEnabled(RADIANCE_FEATURE_EMISSIVE) || Radiance.scene_counts.w == 0u) return 0.0f;
    uint source_material_id = TraceMaterialId.Load(int3(source_pixel, 0));
    if (source_material_id < Radiance.scene_counts.y) {
        GPUMaterial source_material = SceneMaterials[source_material_id];
        if (dot(source_material.emissive, source_material.emissive) > 1.0e-8f) return 0.0f;
    }
'''
if s.count(old) != 1:
    raise SystemExit(f'expected one EvaluateEmissiveSample header, got {s.count(old)}')
s = s.replace(old, new, 1)

old = '    return emitted * geometry / max(pdf_area, 1.0e-8f);\n'
new = '    return emitted * geometry / max(pdf_area * PI, 1.0e-8f);\n'
if s.count(old) != 1:
    raise SystemExit(f'expected one unnormalized emissive estimator, got {s.count(old)}')
s = s.replace(old, new, 1)

# 2) Direct light visibility must not change when geometry enters/leaves the
# camera. Use the world/SDF visibility path only.
old = '''    TraceHit screen_hit = TraceScreenRay(origin, direction, source_pixel, source_object_id, normal, tmax, TraceFrame.trace_params.y, TraceFrame.trace_params.w, TraceFrame.trace_limits.x, TraceFrame.trace_limits.y);
    if (screen_hit.type == TRACE_SCREEN) return 0.0f;
'''
if s.count(old) != 1:
    raise SystemExit(f'expected one screen-space direct visibility block, got {s.count(old)}')
s = s.replace(old, '', 1)

# 3) A screen miss is not a world miss. Do not inject sky until the SDF/world
# fallback also misses.
old = '''    float3 sky = SkyRadiance(ray.direction_tmax.xyz);
    ProbeRadianceOutput[ray_pixel] = float4(sky, 1.0f);
    ScreenTraceOutput[ray_pixel] = float4(sky, 1.0f);
'''
new = '''    ProbeRadianceOutput[ray_pixel] = 0.0f;
    ScreenTraceOutput[ray_pixel] = 0.0f;
'''
if s.count(old) != 1:
    raise SystemExit(f'expected one premature screen-miss sky write, got {s.count(old)}')
s = s.replace(old, new, 1)

old = '    if (best_model == INVALID_INDEX) return;\n'
new = '''    if (best_model == INVALID_INDEX) {
        uint miss_width, miss_height;
        ProbeRadianceOutput.GetDimensions(miss_width, miss_height);
        uint2 miss_destination = uint2(ray.destination % miss_width, ray.destination / miss_width);
        float3 sky = SkyRadiance(ray.direction_tmax.xyz);
        ProbeRadianceOutput[miss_destination] = float4(sky, 1.0f);
        ScreenTraceOutput[miss_destination] = float4(sky, 1.0f);
        return;
    }
'''
if s.count(old) != 1:
    raise SystemExit(f'expected one unresolved SDF miss return, got {s.count(old)}')
s = s.replace(old, new, 1)

p.write_text(s)
