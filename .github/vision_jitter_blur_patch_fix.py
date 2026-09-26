from pathlib import Path
p = Path('.github/vision_jitter_blur_patch.py')
text = p.read_text()
old = "    if count != 1:\n        raise RuntimeError(f'expected one match, found {count}: {old[:100]!r}')\n"
new = "    if count < 1:\n        raise RuntimeError(f'expected at least one match, found {count}: {old[:100]!r}')\n"
if old not in text:
    raise RuntimeError('patch matcher target not found')
p.write_text(text.replace(old, new, 1))
print('patch matcher corrected')
