from pathlib import Path

p = Path('shader.hlsl')
s = p.read_text()
old = '            ProbeHistoryRadianceOut[destination] = float4(sum / max(weight_sum, 1.0e-6f), 1.0f);'
new = '            ProbeHistoryRadianceOut[ProbeAtlasCoord(probe, texel)] = float4(sum / max(weight_sum, 1.0e-6f), 1.0f);'
if s.count(old) != 1:
    raise SystemExit(f'expected one spatial output assignment, found {s.count(old)}')
p.write_text(s.replace(old, new, 1))
