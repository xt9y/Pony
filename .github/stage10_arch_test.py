#!/usr/bin/env python3
from pathlib import Path
import argparse
import sys


def require(cond, message):
    if not cond:
        raise AssertionError(message)


def check_task3():
    g = Path('game.h').read_text()
    r = Path('render.c').read_text()
    require('#define WORLD_PROBE_CAPACITY 8192u' in g, 'WORLD_PROBE_CAPACITY missing')
    require('#define WORLD_PROBE_HASH_CAPACITY 16384u' in g, 'WORLD_PROBE_HASH_CAPACITY missing')
    require('#define WORLD_PROBE_DIRECTION_COUNT 16u' in g, 'WORLD_PROBE_DIRECTION_COUNT missing')
    require('WORLD_PROBE_CAPACITY * sizeof(WORLD_PROBE_STATE)' in r, 'real world probe buffer allocation missing')
    require('WORLD_PROBE_CAPACITY * WORLD_PROBE_DIRECTION_COUNT * WORLD_PROBE_BANK_COUNT' in r, 'two-bank radiance allocation missing')
    require('WORLD_PROBE_HASH_CAPACITY * sizeof(uint32_t)' in r, 'world hash allocation missing')
    require('WORLD_PROBE_UPDATES_PER_FRAME' in r and 'update_list' in r, 'world update list floor missing')
    require('constants.cache_counts[1] = renderer->world_radiance.probe_count;' in r, 'active world count missing')
    require('constants.cache_counts[2] = WORLD_PROBE_HASH_CAPACITY;' in r, 'world hash count missing')
    require('constants.cache_counts[3] = WORLD_PROBE_CAPACITY;' in r, 'world capacity constant missing')
    require('constants.world_probe_config[0] = WORLD_PROBE_DIRECTION_SIZE;' in r, 'world direction size missing')
    require('constants.world_probe_config[1] = WORLD_PROBE_UPDATES_PER_FRAME;' in r, 'world update budget missing')
    require('constants.world_probe_config[2] = WORLD_PROBE_BANK_COUNT;' in r, 'world bank count missing')
    require('constants.world_probe_params[0] = WORLD_PROBE_SPACING;' in r, 'world spacing missing')
    require('constants.world_probe_params[1] = WORLD_PROBE_BLEND;' in r, 'world blend missing')
    require('constants.world_probe_params[2] = WORLD_PROBE_RADIUS;' in r, 'world radius missing')
    require('constants.world_probe_params[3] = WORLD_PROBE_CLEARANCE;' in r, 'world clearance missing')
    require('RADIANCE_FEATURE_WORLD_CACHE | RADIANCE_FEATURE_MULTIBOUNCE' in r, 'world feature gate missing')
    require('build_world_radiance_scene(renderer)' in r, 'world cache scene build missing')
    require('compute_radiance_revision' in r and 'renderer->radiance_revision' in r, 'radiance revision missing')


def check_task4():
    s = Path('shader.hlsl').read_text()
    require('WorldProbeRadianceIndex(uint probe_index, uint2 texel, uint bank)' in s, 'banked address helper missing')
    require('Radiance.cache_counts.w * WorldProbeDirectionCount()' in s, 'fixed physical bank stride missing')
    require('state.state.x == Pass.dispatch.x' in s and 'bank ^= 1u' in s, 'stable previous-bank read missing')
    require('for (int dz = -1; dz <= 1; ++dz)' in s, '3x3x3 z lookup missing')
    require('for (int dy = -1; dy <= 1; ++dy)' in s, '3x3x3 y lookup missing')
    require('for (int dx = -1; dx <= 1; ++dx)' in s, '3x3x3 x lookup missing')
    require('MAX_CACHE_PROBES' in s and 'WorldProbeKeys[slot]' in s, 'bounded open addressing missing')
    require('FindWorldProbeForSurface' in s, 'surface-aware lookup missing')
    require('dot(state.position_radius.xyz - position, normal) <= 1.0e-4f' in s, 'surface side rejection missing')
    require('IntegrateWorldProbeDiffuse' in s, 'diffuse world integration missing')
    require('WorldProbeDirectionCount()' in s and 'saturate(dot(normal, direction))' in s, '16-dir cosine integration missing')
    require('ReflectedDirectAtPixel(hit_pixel)' in s and 'WorldIndirectReflected(hit, false)' in s, 'screen hit world indirect missing')
    require('uint new_bank = old_bank ^ 1u;' in s, 'inactive bank selection missing')
    require('state.state.z = new_bank;' in s, 'bank publish missing')
    invalidate = s[s.find('void CS_InvalidateRadiance'):]
    require('state.state.w |= 2u;' in invalidate, 'dirty invalidation bit missing')
    require('state.state.z = 1u' not in invalidate, 'invalidation still corrupts active bank')


def check_task5():
    r = Path('render.c').read_text()
    s = Path('shader.hlsl').read_text()
    require('static void prepare_world_probe_updates(RENDERER *renderer)' in r, 'round-robin scheduler missing')
    require('WORLD_PROBE_UPDATES_PER_FRAME' in r and 'world->update_cursor' in r, 'world update budget/cursor missing')
    require('renderer->pass_constants.range[0] = world->update_count;' in r, 'update count pass constant missing')
    require('renderer->pass_constants.dispatch[0] = (uint32_t)(renderer->gpu->frame_index & 0x00ffffffu);' in r, 'Pass.dispatch.x is not frame stamp')
    require('renderer->wavefront.update_list' in r and 'cpu_update_list' in r and 'StreamBufferData' in r, 'world update list streaming missing')
    require('radiance_world_cache.cs.spv' in r and '&renderer->world_radiance_pipeline' in r, 'world pipeline creation missing')
    require('renderer->world_radiance_pipeline' in r and 'world->update_count' in r and 'CmdDispatch' in r, 'world dispatch missing')
    require('RayCounters[3] = Pass.range.x;' in s, 'reset does not preserve world update count')
    require('barrier_world_radiance' in r, 'post-world-update barrier missing')
    require('world->update_cursor = (world->update_cursor + world->update_count) % world->probe_count;' in r, 'cursor advance missing')


def check_final():
    check_task3()
    check_task4()
    check_task5()
    b = Path('build.c').read_text()
    require(b.count('CS_UpdateWorldRadianceCache') == 1, 'world cache shader job must remain unique')
    r = Path('render.c').read_text()
    camera_section = r[r.find('static void update_orbit_camera'):r.find('static bool load_shader')]
    require('build_world_radiance_scene' not in camera_section, 'camera motion rebuilds world probes')
    require('WORLD_PROBE_CAPACITY' in Path('tests/stage10_world_cache.c').read_text(), 'permanent Stage 10 regression test missing')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--red-task', type=int)
    parser.add_argument('--task', type=int)
    args = parser.parse_args()
    checks = {3: check_task3, 4: check_task4, 5: check_task5}
    if args.red_task is not None:
        fn = checks[args.red_task]
        try:
            fn()
        except AssertionError as exc:
            print(f'Task {args.red_task} RED confirmed: {exc}')
            return 0
        print(f'Task {args.red_task} unexpectedly green', file=sys.stderr)
        return 1
    if args.task is not None:
        for n in sorted(k for k in checks if k <= args.task):
            checks[n]()
        print(f'Stage 10 architecture task {args.task} PASS')
        return 0
    check_final()
    print('Stage 10 architecture PASS')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
