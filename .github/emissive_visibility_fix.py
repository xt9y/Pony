from pathlib import Path

p = Path('shader.hlsl')
s = p.read_text()

old = '    SurfaceHit hit = TraceUnifiedRay(ray, true);\n'
new = '    SurfaceHit hit = TraceUnifiedRay(ray, false);\n'
if s.count(old) != 1:
    raise SystemExit(f'expected one screen-dependent unified occlusion call, got {s.count(old)}')
s = s.replace(old, new, 1)

old = '    float cos_light = saturate(dot(light_normal, -L));\n'
new = '    float cos_light = saturate(abs(dot(light_normal, -L)));\n'
if s.count(old) != 1:
    raise SystemExit(f'expected one one-sided emissive cosine, got {s.count(old)}')
s = s.replace(old, new, 1)

p.write_text(s)
