#!/usr/bin/env python3
from pathlib import Path
import argparse
import re
import sys

p = argparse.ArgumentParser()
p.add_argument('--red-task', type=int)
p.add_argument('--task', type=int)
a = p.parse_args()

game = Path('game.h').read_text()
render = Path('render.c').read_text()
shader = Path('shader.hlsl').read_text()
build = Path('build.c').read_text()


def need(cond, msg):
    if not cond:
        raise SystemExit(msg)


def absent(cond, msg):
    if cond:
        raise SystemExit(msg)


def check_task1():
    need('typedef struct GLOBAL_SDF_DATA {' in game, 'GLOBAL_SDF_DATA missing')
    need('bool sdf_build_global_clipmaps(' in game, 'global builder declaration missing')
    need('void sdf_free_global_clipmaps(' in game, 'global free declaration missing')
    need('GLOBAL_SDF_DATA global_sdf;' in game, 'renderer global_sdf owner missing')
    need('NriPipeline *wavefront_global_pipeline;' in game, 'global pipeline handle missing')
    need('_Static_assert(sizeof(GPU_GLOBAL_SDF_CLIPMAP) == 64u' in game, 'global clip ABI assert changed')


def check_task3():
    check_task1()
    need('static void destroy_global_sdf_resources(RENDERER *renderer)' in render, 'global destroy missing')
    need('static bool create_global_sdf_resources(RENDERER *renderer)' in render, 'global create missing')
    need('create_global_sdf_resources(renderer)' in render, 'scene setup does not create global sdf')
    need('renderer->global_sdf.clipmaps_srv' in render, 'clip descriptor not bound from global_sdf')
    need('renderer->global_sdf.page_table_srv' in render, 'page descriptor not bound from global_sdf')
    need('renderer->global_sdf.bricks_srv' in render, 'brick descriptor not bound from global_sdf')
    need('renderer->global_sdf.surface_ids_srv' in render, 'surface descriptor not bound from global_sdf')
    absent('create_radiance_scene_fallbacks' in render, 'old global fallback creation remains')
    absent('destroy_radiance_scene_fallbacks' in render, 'old global fallback destruction remains')
    absent('RADIANCE_SCENE_FALLBACKS radiance_fallbacks;' in game, 'old global fallback owner remains')
    need('constants.sdf_counts[2] = renderer->global_sdf.valid ? renderer->global_sdf.clip_count : 0u;' in render,
         'global clip count not wired')
    need('constants.global_sdf_params[2] = 0.65f;' in render, 'global epsilon scale not wired')
    need('RADIANCE_FEATURE_GLOBAL_SDF' in render, 'global feature flag not wired')


def function_body(text, name):
    m = re.search(r'\b' + re.escape(name) + r'\s*\([^)]*\)\s*\{', text)
    need(m is not None, f'{name} missing')
    start = m.end()
    depth = 1
    i = start
    while i < len(text) and depth:
        if text[i] == '{': depth += 1
        elif text[i] == '}': depth -= 1
        i += 1
    need(depth == 0, f'{name} body unterminated')
    return text[start:i-1]


def check_task4():
    check_task3()
    need('create_compute_pipeline(renderer, "build/shaders/radiance_global.cs.spv", renderer->wavefront_layout, &renderer->wavefront_global_pipeline)' in render,
         'global pipeline not created')
    need('renderer->wavefront_global_pipeline,' in render, 'global pipeline not destroyed')
    body = function_body(render, 'build_wavefront_screen_probes')
    positions = [body.find(x) for x in (
        'renderer->wavefront_dynamic_pipeline',
        'renderer->wavefront_global_pipeline',
        'renderer->wavefront_local_pipeline',
        'renderer->wavefront_shade_pipeline')]
    need(all(x >= 0 for x in positions) and positions == sorted(positions), 'wavefront order is not dynamic -> global -> local -> shade')

    unified = function_body(shader, 'TraceUnifiedRay')
    pd = unified.find('TraceDynamicGrid')
    pg = unified.find('TraceGlobalSDF')
    pl = unified.find('TraceAllLocalSDFs')
    need(pd >= 0 and pg > pd and pl > pg, 'unified trace order is not dynamic -> global -> local')
    need('if (!global_resolved)' in unified, 'local fallback is not gated on global resolution')

    global_trace = function_body(shader, 'TraceGlobalSDF')
    need('RefineGlobalSDFSurface' in global_trace, 'global trace does not exact-refine candidates')
    need('GlobalSDFNormal(' not in global_trace, 'global trace still accepts SDF normal as final geometry')
    need('bool RefineGlobalSDFSurface(' in shader, 'global exact-refinement helper missing')
    refine = function_body(shader, 'RefineGlobalSDFSurface')
    need('IntersectSceneTriangle' in refine, 'global refinement does not intersect canonical triangles')
    need('TRACE_GLOBAL_SDF' in refine and 'SurfaceFromTriangle' in refine, 'global refinement does not create canonical SurfaceHit')

    gpass = function_body(shader, 'CS_WavefrontGlobalTrace')
    need('RAY_FLAG_GLOBAL_RESOLVED' in gpass, 'global wavefront resolved bit not set')
    lpass = function_body(shader, 'CS_WavefrontLocalTrace')
    need('RAY_FLAG_GLOBAL_RESOLVED' in lpass and 'TraceAllLocalSDFs' in lpass, 'local fallback does not honor global resolved bit')
    need('#define RAY_FLAG_GLOBAL_RESOLVED 0x40000000u' in shader, 'global resolved bit definition missing')
    need(build.count('CS_WavefrontGlobalTrace') == 1, 'global trace shader job must exist exactly once')


if a.red_task == 1:
    absent('typedef struct GLOBAL_SDF_DATA {' in game, 'RED task1 unexpectedly already implemented')
    absent('wavefront_global_pipeline' in game, 'RED task1 pipeline handle unexpectedly exists')
    print('RED task1 confirmed')
elif a.red_task == 3:
    check_task1()
    absent('static bool create_global_sdf_resources(RENDERER *renderer)' in render, 'RED task3 unexpectedly already implemented')
    print('RED task3 confirmed')
elif a.red_task == 4:
    check_task3()
    absent('radiance_global.cs.spv", renderer->wavefront_layout, &renderer->wavefront_global_pipeline' in render,
           'RED task4 global pipeline unexpectedly already active')
    print('RED task4 confirmed')
elif a.task == 1:
    check_task1(); print('GREEN task1')
elif a.task == 3:
    check_task3(); print('GREEN task3')
else:
    check_task4(); print('GREEN stage9 architecture')
