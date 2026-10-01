from pathlib import Path
import re

h = Path('game.h').read_text()
r = Path('render.c').read_text()
s = Path('shader.hlsl').read_text()
b = Path('build.c').read_text()

checks = {
    'dynamic grid resource type': 'typedef struct RADIANCE_DYNAMIC_GRID' in h,
    'renderer dynamic grid state': 'RADIANCE_DYNAMIC_GRID dynamic_grid;' in h,
    'dynamic wavefront pipeline handle': 'NriPipeline *wavefront_dynamic_pipeline;' in h,
    'dynamic grid builder': 'static bool rebuild_dynamic_grid(RENDERER *renderer)' in r,
    'dynamic grid destroy': 'static void destroy_dynamic_grid(RENDERER *renderer)' in r,
    'dynamic cell count constants': 'constants.sdf_counts[1] = renderer->dynamic_grid.cell_count;' in r,
    'dynamic grid x constant': 'constants->dynamic_grid[0] = grid->dimensions[0];' in r,
    'dynamic grid index count constant': 'constants->dynamic_grid[3] = grid->index_count;' in r,
    'dynamic grid feature enable': 'constants->feature_flags[0] |= RADIANCE_FEATURE_DYNAMIC_GRID;' in r,
    'dynamic descriptors': 'renderer->dynamic_grid.cells_srv' in r and 'renderer->dynamic_grid.indices_srv' in r,
    'dynamic pipeline creation': 'radiance_dynamic.cs.spv' in r,
    'dynamic shader build': 'CS_WavefrontDynamicTrace' in b,
    'shader dynamic trace': 'void CS_WavefrontDynamicTrace' in s,
    'static trace excludes dynamic models': 'FeatureEnabled(RADIANCE_FEATURE_DYNAMIC_GRID) && LocalSDFModels[i].meta.w != 0u' in s,
}

for name, ok in checks.items():
    if not ok:
        raise SystemExit(f'stage8 missing: {name}')

fallback = re.search(r'typedef struct RADIANCE_SCENE_FALLBACKS \{(.*?)\} RADIANCE_SCENE_FALLBACKS;', h, re.S)
if not fallback:
    raise SystemExit('stage8 missing fallback struct')
if 'dynamic_grid_cells' in fallback.group(1) or 'dynamic_grid_indices' in fallback.group(1):
    raise SystemExit('stage8 still stores dynamic grid as fallback junk')

wave = re.search(r'static void build_wavefront_screen_probes\(.*?\n\}', r, re.S)
if not wave:
    raise SystemExit('stage8 missing wavefront builder')
body = wave.group(0)
dynamic = body.find('wavefront_dynamic_pipeline')
local = body.find('wavefront_local_pipeline')
if dynamic < 0 or local < 0 or dynamic >= local:
    raise SystemExit('stage8 dynamic trace must run before static local trace')

frame = re.search(r'bool renderer_frame\(.*?\n\}', r, re.S)
if not frame or 'refresh_dynamic_grid(renderer)' not in frame.group(0):
    raise SystemExit('stage8 missing dynamic grid refresh in frame path')

print('stage8 dynamic grid architecture verified')
