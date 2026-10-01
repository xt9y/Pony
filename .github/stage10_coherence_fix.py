#!/usr/bin/env python3
from pathlib import Path
import sys

p = Path('shader.hlsl')
s = p.read_text()

def ok():
    return (
        'uint FindWorldProbeInternal(float3 position, float3 normal, bool surface_filter, bool stable_previous)' in s and
        'if (stable_previous && state.state.x == Pass.dispatch.x && state.statistics.z <= 1.0f) continue;' in s and
        'uint FindWorldProbeForSurface(float3 position, float3 normal, bool stable_previous)' in s and
        'FindWorldProbeForSurface(position, normal, stable_previous)' in s and
        'if (stable_previous && state.state.x == Pass.dispatch.x && state.statistics.z <= 1.0f) return 0.0f;' in s
    )

if len(sys.argv) != 2 or sys.argv[1] not in {'test', 'apply'}:
    raise SystemExit('usage: stage10_coherence_fix.py test|apply')

if sys.argv[1] == 'test':
    if not ok():
        raise SystemExit('stable-previous lookup still admits first-published probes')
    print('Stage 10 stable-previous lookup PASS')
    raise SystemExit(0)

replacements = [
    ('uint FindWorldProbeInternal(float3 position, float3 normal, bool surface_filter) {',
     'uint FindWorldProbeInternal(float3 position, float3 normal, bool surface_filter, bool stable_previous) {'),
    ('                    if (state.identity.x != key || (state.state.w & 1u) == 0u || state.state.y != Radiance.feature_flags.y) continue;\n                    if (surface_filter',
     '                    if (state.identity.x != key || (state.state.w & 1u) == 0u || state.state.y != Radiance.feature_flags.y) continue;\n                    if (stable_previous && state.state.x == Pass.dispatch.x && state.statistics.z <= 1.0f) continue;\n                    if (surface_filter'),
    ('    return FindWorldProbeInternal(position, 0.0f, false);',
     '    return FindWorldProbeInternal(position, 0.0f, false, false);'),
    ('uint FindWorldProbeForSurface(float3 position, float3 normal) {\n    return FindWorldProbeInternal(position, normalize(normal), true);\n}',
     'uint FindWorldProbeForSurface(float3 position, float3 normal, bool stable_previous) {\n    return FindWorldProbeInternal(position, normalize(normal), true, stable_previous);\n}'),
    ('    if ((state.state.w & 1u) == 0u || state.state.y != Radiance.feature_flags.y) return 0.0f;\n    uint s = max(Radiance.world_probe_config.x, 1u);',
     '    if ((state.state.w & 1u) == 0u || state.state.y != Radiance.feature_flags.y) return 0.0f;\n    if (stable_previous && state.state.x == Pass.dispatch.x && state.statistics.z <= 1.0f) return 0.0f;\n    uint s = max(Radiance.world_probe_config.x, 1u);'),
    ('    uint probe = FindWorldProbeForSurface(position, normal);',
     '    uint probe = FindWorldProbeForSurface(position, normal, stable_previous);')
]
for old, new in replacements:
    count = s.count(old)
    if count != 1:
        raise SystemExit(f'expected one coherence replacement, got {count}: {old[:70]}')
    s = s.replace(old, new, 1)
p.write_text(s)
s = p.read_text()
if not ok():
    raise SystemExit('coherence fix did not establish required semantics')
