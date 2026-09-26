from pathlib import Path

p = Path('.github/volumetrics_controls_patch.py')
text = p.read_text()
old = '''replace_once(
    "shaders/compute.hlsl",
    """    return max(tri.emissive.rgb, 0.0f) *'''
new = '''replace_once(
    "shaders/compute_base.hlsl",
    """    return max(tri.emissive.rgb, 0.0f) *'''
if text.count(old) != 1:
    raise RuntimeError(f'expected one legacy direct-emissive patch target, found {text.count(old)}')
p.write_text(text.replace(old, new, 1))
print('patch helper target corrected')
