from pathlib import Path
p = Path('render.c')
s = p.read_text()

old = 'create_compute_pipeline(renderer, "build/shaders/screen_trace.cs.spv", renderer->trace_layout, &renderer->screen_trace_pipeline)'
new = 'create_compute_pipeline(renderer, "build/shaders/screen_trace.cs.spv", renderer->radiance_direct_layout, &renderer->screen_trace_pipeline)'
if s.count(old) != 1:
    raise SystemExit(f'screen trace pipeline layout pattern count = {s.count(old)}')
s = s.replace(old, new, 1)

old = '    bind_trace(renderer, command_buffer, renderer->screen_trace_pipeline);\n'
new = '''    const NriBufferBarrierDesc cache_barriers[] = {
        {
            .buffer = renderer->radiance_surface_cache.keys,
            .before = renderer->radiance_surface_cache.keys_state,
            .after = buffer_write
        },
        {
            .buffer = renderer->radiance_surface_cache.entries,
            .before = renderer->radiance_surface_cache.entries_state,
            .after = buffer_write
        }
    };

    renderer->gpu->core.CmdBarrier(command_buffer, &(NriBarrierDesc){
        .buffers = cache_barriers,
        .bufferNum = 2
    });
    renderer->radiance_surface_cache.keys_state = buffer_write;
    renderer->radiance_surface_cache.entries_state = buffer_write;

    bind_radiance_world(renderer, command_buffer, renderer->screen_trace_pipeline);
'''
if s.count(old) != 1:
    raise SystemExit(f'screen trace bind pattern count = {s.count(old)}')
s = s.replace(old, new, 1)

p.write_text(s)
