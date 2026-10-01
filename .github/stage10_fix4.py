#!/usr/bin/env python3
from pathlib import Path
import importlib.util

spec = importlib.util.spec_from_file_location('stage10_apply', '.github/stage10_apply.py')
apply = importlib.util.module_from_spec(spec)
spec.loader.exec_module(apply)

p = Path('shader.hlsl')
s = p.read_text()
start_marker = '// -----------------------------------------------------------------------------\n// World radiance cache.\n// -----------------------------------------------------------------------------'
end_marker = '// -----------------------------------------------------------------------------\n// Wavefront passes.'
a = s.find(start_marker)
b = s.find(end_marker, a)
if a < 0 or b < 0:
    raise SystemExit('world shader section markers missing')
s = s[:a] + apply.WORLD_SHADER_SECTION.rstrip() + '\n\n' + s[b:]

forward = 'float3 SurfaceReflectedRadiance(SurfaceHit hit);\n'
if 'float3 WorldIndirectReflected(SurfaceHit hit, bool stable_previous);' not in s:
    s = apply.replace_once(s, forward, forward + 'float3 WorldIndirectReflected(SurfaceHit hit, bool stable_previous);\n', 'world indirect forward')

old_reflected = '''float3 SurfaceReflectedRadiance(SurfaceHit hit) {
    if (hit.identity.w == TRACE_MISS || hit.identity.y >= Radiance.scene_counts.y) return 0.0f;
    SurfaceCacheEntry entry;
    if (SurfaceCacheLookup(hit, entry)) return entry.direct_radiance.rgb + entry.indirect_radiance.rgb;
    float3 reflected = EvaluateSurfaceReflectedDirect(hit);
    SurfaceCacheStore(hit, reflected, 0.0f, 1.0f);
    return reflected;
}'''
new_reflected = '''float3 SurfaceReflectedRadiance(SurfaceHit hit) {
    if (hit.identity.w == TRACE_MISS || hit.identity.y >= Radiance.scene_counts.y) return 0.0f;
    SurfaceCacheEntry entry;
    float3 direct;
    if (SurfaceCacheLookup(hit, entry)) direct = entry.direct_radiance.rgb;
    else direct = EvaluateSurfaceReflectedDirect(hit);
    float3 indirect = WorldIndirectReflected(hit, false);
    SurfaceCacheStore(hit, direct, indirect, 1.0f);
    return direct + indirect;
}'''
s = apply.replace_once(s, old_reflected, new_reflected, 'surface reflected body')
s = apply.replace_once(s,
    '        RayRadiance[index] = float4(ReflectedDirectAtPixel(hit_pixel), 1.0f);',
    '        RayRadiance[index] = float4(ReflectedDirectAtPixel(hit_pixel) + WorldIndirectReflected(hit, false), 1.0f);',
    'screen world indirect')
s = apply.replace_function(s, '[numthreads(64, 1, 1)]\nvoid CS_UpdateWorldRadianceCache', apply.WORLD_UPDATE_FUNCTION)
old_invalid = '''        WorldProbeState state = WorldProbes[target];
        state.statistics.x *= confidence_scale;
        state.state.z = 1u;
        WorldProbes[target] = state;'''
new_invalid = '''        WorldProbeState state = WorldProbes[target];
        state.statistics.x *= confidence_scale;
        state.state.w |= 2u;
        WorldProbes[target] = state;'''
s = apply.replace_once(s, old_invalid, new_invalid, 'world invalidation')
p.write_text(s)
