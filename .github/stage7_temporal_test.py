from pathlib import Path

h = Path('game.h').read_text()
r = Path('render.c').read_text()
s = Path('shader.hlsl').read_text()
b = Path('build.c').read_text()

required = {
    'game.h temporal pipeline': 'NriPipeline *wavefront_temporal_pipeline;' in h,
    'game.h spatial pipeline': 'NriPipeline *wavefront_spatial_pipeline;' in h,
    'game.h history pipeline': 'NriPipeline *wavefront_history_pipeline;' in h,
    'history validity state': 'bool probe_history_valid;' in h,
    'history output binding': 'ProbeHistoryRadianceOut' in s,
    'history commit shader': 'void CS_CommitScreenProbeHistory' in s,
    'first-frame temporal gate': 'Pass.flags.x == 0u' in s,
    'linear depth temporal rejection': 'previous_linear_depth' in s and 'current_linear_depth' in s,
    'race-free spatial destination': 'ProbeHistoryRadianceOut[ProbeAtlasCoord(probe, texel)]' in s,
    'temporal feature enabled': 'RADIANCE_FEATURE_TEMPORAL_PROBES | RADIANCE_FEATURE_SPATIAL_PROBES' in r,
    'temporal parameters': 'constants.temporal_params[0] = 0.85f;' in r,
    'spatial radius': 'constants.probe_config[3] = 1u;' in r,
    'history-valid pass flag': 'renderer->pass_constants.flags[0] = renderer->probe_history_valid ? 1u : 0u;' in r,
    'temporal pipeline creation': 'radiance_probe_temporal.cs.spv' in r,
    'spatial pipeline creation': 'radiance_probe_spatial.cs.spv' in r,
    'history pipeline creation': 'radiance_probe_history.cs.spv' in r,
    'history shader build': '{"CS_CommitScreenProbeHistory", "compute", "radiance_probe_history.cs.spv", NULL}' in b,
    'history commit in frame': 'commit_probe_history(renderer, command_buffer);' in r,
}

missing = [name for name, ok in required.items() if not ok]
if missing:
    raise SystemExit('chapter 7 missing: ' + ', '.join(missing))

# The spatial pass may read ProbeCurrentRadiance, but it must not overwrite it.
start = s.index('void CS_SpatialReuseScreenProbes')
end = s.find('\n}\n', start) + 2
spatial = s[start:end]
if 'ProbeCurrentRadiance[ProbeAtlasCoord(probe, texel)] = ' in spatial:
    raise SystemExit('chapter 7 spatial pass is still in-place/racy')

# The active frame must commit history after emissive accumulation and before presentation.
frame_start = r.index('bool renderer_frame(')
frame = r[frame_start:]
order = [
    'build_wavefront_screen_probes(renderer, command_buffer);',
    'build_emissive_gather(renderer, command_buffer);',
    'commit_probe_history(renderer, command_buffer);',
    'record_present_pass(renderer, command_buffer, swapchain_index);',
]
pos = [frame.index(x) for x in order]
if pos != sorted(pos):
    raise SystemExit('chapter 7 frame ordering is wrong')

print('chapter 7 temporal/spatial reuse architecture verified')
