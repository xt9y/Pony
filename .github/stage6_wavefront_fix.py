from pathlib import Path
import runpy

runpy.run_path('.github/stage6_wavefront.py', run_name='__main__')

path = Path('render.c')
text = path.read_text()
old = '.value = {.color = {.f = {.x = 0.0f, .y = 0.0f, .z = 0.0f, .w = 0.0f}}},'
new = '.value = {.f = {.x = 0.0f, .y = 0.0f, .z = 0.0f, .w = 0.0f}},'
count = text.count(old)
if count != 3:
    raise SystemExit(f'clear storage initializer: expected 3 matches, found {count}')
path.write_text(text.replace(old, new))
