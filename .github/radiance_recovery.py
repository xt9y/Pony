from pathlib import Path
import re


def replace_once(text, old, new, label):
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected exactly one match, found {count}")
    return text.replace(old, new, 1)


def sub_once(text, pattern, replacement, label):
    result, count = re.subn(pattern, replacement, text, count=1, flags=re.S)
    if count != 1:
        raise SystemExit(f"{label}: expected exactly one regex match, found {count}")
    return result

shader_path = Path("shader.hlsl")
shader = shader_path.read_text()

shader = replace_once(
    shader,
    "#define SCREEN_PROBE_RAY_COUNT 16u\n#define DIRECT_EMISSIVE_SAMPLE_COUNT 4u\n",
    "#define SCREEN_PROBE_RAY_COUNT 8u\n",
    "probe ray budget",
)

# Replace the expensive per-surface direct evaluator with a purely analytic
# version. It is used only for sparse offscreen SDF hits after this stage.
pattern = r"float3 EvaluateSurfaceReflectedDirect\(SurfaceHit hit, uint seed\) \{.*?\n\}\n\nfloat3 SurfaceReflectedRadiance\(SurfaceHit hit\);"
replacement = r'''float3 EvaluateAnalyticLightDiffuse(GPULight light, float3 albedo, float3 position, float3 normal) {
    uint type = (uint)(light.direction_type.w + 0.5f);
    if (type == 3u) return 0.0f;

    float3 L;
    float attenuation = 1.0f;
    if (type == 0u) {
        L = normalize(-light.direction_type.xyz);
    } else {
        float3 to_light = light.position_range.xyz - position;
        float d = length(to_light);
        if (d <= 1.0e-5f || d >= light.position_range.w) return 0.0f;
        L = to_light / d;
        float range_term = saturate(1.0f - d / max(light.position_range.w, 1.0e-3f));
        attenuation = range_term * range_term / max(1.0f, d * d);
        if (type == 2u) {
            float cone = dot(normalize(light.direction_type.xyz), -L);
            float cone_term = saturate((cone - light.spot_angles.y) / max(light.spot_angles.x - light.spot_angles.y, 1.0e-4f));
            attenuation *= cone_term * cone_term;
        }
    }

    float ndotl = saturate(dot(normal, L));
    if (ndotl <= 0.0f) return 0.0f;
    return albedo * light.color_intensity.rgb * light.color_intensity.w * attenuation * ndotl * INV_PI;
}

float3 EvaluateSurfaceReflectedDirect(SurfaceHit hit) {
    if (hit.identity.y >= Radiance.scene_counts.y) return 0.0f;
    GPUMaterial material = SceneMaterials[hit.identity.y];
    float3 normal = normalize(hit.normal_confidence.xyz);
    if (dot(normal, normal) <= 1.0e-8f) return 0.0f;

    float3 reflected = 0.0f;
    for (uint i = 0u; i < Radiance.sdf_counts.w; ++i)
        reflected += EvaluateAnalyticLightDiffuse(SceneLights[i], material.base_color.rgb, hit.position_distance.xyz, normal);
    return reflected;
}

float3 SurfaceReflectedRadiance(SurfaceHit hit);'''
shader = sub_once(shader, pattern, replacement, "cheap surface direct")

# Delete the obsolete duplicate full-resolution direct pass and its SDF shadow
# helper. ReflectedDirectAtPixel remains the canonical screen-hit fast path.
pattern = r"// -----------------------------------------------------------------------------\n// Compatibility direct lighting\.\n// -----------------------------------------------------------------------------\nfloat DirectVisibility\(.*?\n\}\n\nfloat3 ReflectedDirectAtPixel"
replacement = r'''// -----------------------------------------------------------------------------
// Screen-space direct-radiance reuse.
// -----------------------------------------------------------------------------
float3 ReflectedDirectAtPixel'''
shader = sub_once(shader, pattern, replacement, "remove obsolete direct pass")

shader = replace_once(
    shader,
    "// Compatibility screen probes / queue.\n",
    "// Screen probes / compacted miss queue.\n",
    "screen probe comment",
)

# Only eight of the 4x4 storage slots are active rays. Keeping the fixed 4x4
# texture shape avoids descriptor/layout churn while halving real ray work.
needle = "    uint destination = ray_pixel.x + ray_pixel.y * ray_width;\n    float depth = TraceDepth.Load(int3(source_pixel, 0));\n"
replacement = "    uint destination = ray_pixel.x + ray_pixel.y * ray_width;\n    if (ray_index >= SCREEN_PROBE_RAY_COUNT)\n        return MakeTraceRay(0.0f, float3(0.0f, 1.0f, 0.0f), 0.0f, 0.0f, TRACE_RAY_INACTIVE, destination, source_pixel, 0u);\n    float depth = TraceDepth.Load(int3(source_pixel, 0));\n"
shader = replace_once(shader, needle, replacement, "inactive probe slots")

# Add a cheap screen-only hit helper. Screen hits use the already computed
# direct-radiance texture and never touch the geometry cache or world tracer.
marker = "\n[numthreads(8, 8, 1)]\nvoid CS_ScreenTrace(uint3 dispatch_id : SV_DispatchThreadID) {"
if shader.count(marker) != 1:
    raise SystemExit("screen trace marker missing or duplicated")
helper = r'''
TraceHit TraceProbeScreenHit(TraceRay ray) {
    uint2 source_pixel = UnpackPixel(ray.origin_pixel);
    float source_depth = TraceDepth.Load(int3(source_pixel, 0));
    float3 source_position = ReconstructWorldPosition(source_pixel, source_depth);
    float3 source_normal = normalize(TraceNormalRoughness.Load(int3(source_pixel, 0)).xyz);
    float3 source_view_direction = normalize(source_position - TraceFrame.camera_position.xyz);
    if (dot(source_normal, source_view_direction) > 0.0f) source_normal = -source_normal;

    return TraceScreenRay(
        ray.origin_tmin.xyz,
        ray.direction_tmax.xyz,
        source_pixel,
        ray.source_object_id,
        source_normal,
        ray.direction_tmax.w,
        TraceFrame.trace_params.z,
        TraceFrame.trace_params.w,
        TraceFrame.trace_limits.x,
        TraceFrame.trace_limits.y
    );
}
'''
shader = shader.replace(marker, "\n" + helper + marker, 1)

pattern = r"\[numthreads\(8, 8, 1\)\]\nvoid CS_ScreenTrace\(uint3 dispatch_id : SV_DispatchThreadID\) \{.*?\n\}\n\n\[numthreads\(1, 1, 1\)\]\nvoid CS_ResetTraceQueue"
replacement = r'''[numthreads(8, 8, 1)]
void CS_ScreenTrace(uint3 dispatch_id : SV_DispatchThreadID) {
    uint ray_width, ray_height;
    ProbeRadianceOutput.GetDimensions(ray_width, ray_height);
    uint2 ray_pixel = dispatch_id.xy;
    if (ray_pixel.x >= ray_width || ray_pixel.y >= ray_height) return;

    TraceRay ray = BuildDiffuseProbeRay(ray_pixel);
    uint index = ray.destination;
    if (ray.type == TRACE_RAY_INACTIVE) {
        ScreenTraceHits[index] = MakeTraceHit(TRACE_INACTIVE, 0.0f);
        ScreenTraceOutput[ray_pixel] = 0.0f;
        ProbeRadianceOutput[ray_pixel] = 0.0f;
        return;
    }

    TraceHit hit = TraceProbeScreenHit(ray);
    ScreenTraceHits[index] = hit;
    if (hit.type == TRACE_SCREEN) {
        float3 radiance = ReflectedDirectAtPixel(hit.hit_pixel);
        ProbeRadianceOutput[ray_pixel] = float4(radiance, 1.0f);
        ScreenTraceOutput[ray_pixel] = float4(radiance, 1.0f);
        return;
    }

    ProbeRadianceOutput[ray_pixel] = 0.0f;
    ScreenTraceOutput[ray_pixel] = 0.0f;
}

[numthreads(1, 1, 1)]
void CS_ResetTraceQueue'''
shader = sub_once(shader, pattern, replacement, "cheap screen trace")

# Both old call sites are converted to the cheap signature. The full-resolution
# one is then removed entirely by the CS_RadianceDirect replacement below.
old_call = "    uint seed = HashCombine(SurfaceCacheKey(hit), HashCombine(Pass.dispatch.x, Radiance.feature_flags.y));\n    float3 reflected = EvaluateSurfaceReflectedDirect(hit, seed);\n"
call_count = shader.count(old_call)
if call_count != 2:
    raise SystemExit(f"surface direct call sites: expected 2, found {call_count}")
shader = shader.replace(old_call, "    float3 reflected = EvaluateSurfaceReflectedDirect(hit);\n", 2)

# Full-resolution direct pass: raster/G-buffer only. No SDF, no surface-cache
# atomics, no emitter shadow rays. Emissive area-light NEE is probe-frequency.
pattern = r"// -----------------------------------------------------------------------------\n// Future material-aware direct-light/cache pass\.\n// -----------------------------------------------------------------------------\n\[numthreads\(8, 8, 1\)\]\nvoid CS_RadianceDirect\(uint3 dispatch_id : SV_DispatchThreadID\) \{.*?\n\}\n\n// -----------------------------------------------------------------------------\n// Runtime debug and presentation helpers\."
replacement = r'''// -----------------------------------------------------------------------------
// Cheap full-resolution direct lighting.
// World/SDF visibility is intentionally not evaluated per pixel; ray work is
// reserved for screen probes and compacted misses.
// -----------------------------------------------------------------------------
[numthreads(8, 8, 1)]
void CS_RadianceDirect(uint3 dispatch_id : SV_DispatchThreadID) {
    uint2 pixel = dispatch_id.xy;
    uint2 resolution = uint2(TraceFrame.resolution.xy);
    if (any(pixel >= resolution)) return;

    float depth = TraceDepth.Load(int3(pixel, 0));
    if (depth <= 0.0f) {
        float2 uv = (float2(pixel) + 0.5f) * TraceFrame.resolution.zw;
        float4 far_world4 = mul(float4(ScreenUVToNDC(uv), 0.0f, 1.0f), TraceFrame.inverse_view_projection);
        float3 direction = normalize(far_world4.xyz / far_world4.w - TraceFrame.camera_position.xyz);
        DirectRadianceOutput[pixel] = float4(SkyRadiance(direction), 1.0f);
        return;
    }

    float3 position = ReconstructWorldPosition(pixel, depth);
    float3 normal = normalize(TraceNormalRoughness.Load(int3(pixel, 0)).xyz);
    float3 view_direction = normalize(position - TraceFrame.camera_position.xyz);
    if (dot(normal, view_direction) > 0.0f) normal = -normal;

    float3 albedo = TraceAlbedoMetallic.Load(int3(pixel, 0)).rgb;
    float3 radiance = TraceEmissive.Load(int3(pixel, 0)).rgb;
    for (uint i = 0u; i < TraceFrame.trace_limits.z; ++i)
        radiance += EvaluateAnalyticLightDiffuse(Lights[i], albedo, position, normal);

    DirectRadianceOutput[pixel] = float4(radiance, 1.0f);
}

// -----------------------------------------------------------------------------
// Runtime debug and presentation helpers.'''
shader = sub_once(shader, pattern, replacement, "cheap full-resolution direct")

shader_path.write_text(shader)

# CPU-side ray budget and active pipeline bindings.
game_path = Path("game.h")
game = game_path.read_text()
game = replace_once(
    game,
    "#define SCREEN_PROBE_DIRECTION_COUNT (SCREEN_PROBE_DIRECTION_SIZE * SCREEN_PROBE_DIRECTION_SIZE)\n",
    "#define SCREEN_PROBE_DIRECTION_COUNT 8u\n",
    "CPU probe ray budget",
)
game_path.write_text(game)

render_path = Path("render.c")
render = render_path.read_text()

render = replace_once(
    render,
    'create_compute_pipeline(renderer, "build/shaders/radiance_direct.cs.spv", renderer->radiance_direct_layout, &renderer->direct_radiance_pipeline)',
    'create_compute_pipeline(renderer, "build/shaders/radiance_direct.cs.spv", renderer->trace_layout, &renderer->direct_radiance_pipeline)',
    "direct pipeline layout",
)
render = replace_once(
    render,
    'create_compute_pipeline(renderer, "build/shaders/screen_trace.cs.spv", renderer->radiance_direct_layout, &renderer->screen_trace_pipeline)',
    'create_compute_pipeline(renderer, "build/shaders/screen_trace.cs.spv", renderer->trace_layout, &renderer->screen_trace_pipeline)',
    "screen pipeline layout",
)

pattern = r"static void build_direct_radiance\(RENDERER \*renderer, NriCommandBuffer \*command_buffer\) \{.*?\n\}\n\nstatic void build_surface_cache"
replacement = r'''static void build_direct_radiance(RENDERER *renderer, NriCommandBuffer *command_buffer) {
    const NriAccessLayoutStage write = {
        .access = NriAccessBits_SHADER_RESOURCE_STORAGE,
        .layout = NriLayout_SHADER_RESOURCE_STORAGE,
        .stages = NriStageBits_COMPUTE_SHADER
    };

    const NriTextureBarrierDesc to_write = {
        .texture = renderer->direct_radiance.texture,
        .before = renderer->direct_radiance.state,
        .after = write,
        .mipNum = 1,
        .layerNum = 1,
        .planes = NriPlaneBits_COLOR
    };

    renderer->gpu->core.CmdBarrier(command_buffer, &(NriBarrierDesc){
        .textures = &to_write,
        .textureNum = 1
    });

    bind_trace(renderer, command_buffer, renderer->direct_radiance_pipeline);
    renderer->gpu->core.CmdDispatch(
        command_buffer,
        &(NriDispatchDesc){
            .workGroupNumX = (renderer->width + 7u) / 8u,
            .workGroupNumY = (renderer->height + 7u) / 8u,
            .workGroupNumZ = 1
        }
    );

    const NriAccessLayoutStage read = {
        .access = NriAccessBits_SHADER_RESOURCE,
        .layout = NriLayout_SHADER_RESOURCE,
        .stages = NriStageBits_COMPUTE_SHADER | NriStageBits_FRAGMENT_SHADER
    };

    const NriTextureBarrierDesc to_read = {
        .texture = renderer->direct_radiance.texture,
        .before = write,
        .after = read,
        .mipNum = 1,
        .layerNum = 1,
        .planes = NriPlaneBits_COLOR
    };

    renderer->gpu->core.CmdBarrier(command_buffer, &(NriBarrierDesc){
        .textures = &to_read,
        .textureNum = 1
    });
    renderer->direct_radiance.state = read;
}

static void build_surface_cache'''
render = sub_once(render, pattern, replacement, "direct dispatch cleanup")

pattern = r"static void build_screen_trace\(RENDERER \*renderer, NriCommandBuffer \*command_buffer\) \{.*?\n\}\n\nstatic void build_miss_queue"
replacement = r'''static void build_screen_trace(RENDERER *renderer, NriCommandBuffer *command_buffer) {
    const NriAccessLayoutStage texture_write = {
        .access = NriAccessBits_SHADER_RESOURCE_STORAGE,
        .layout = NriLayout_SHADER_RESOURCE_STORAGE,
        .stages = NriStageBits_COMPUTE_SHADER
    };

    const NriAccessStage buffer_write = {
        .access = NriAccessBits_SHADER_RESOURCE_STORAGE,
        .stages = NriStageBits_COMPUTE_SHADER
    };

    const NriTextureBarrierDesc texture_barriers[] = {
        {
            .texture = renderer->screen_trace.texture,
            .before = renderer->screen_trace.state,
            .after = texture_write,
            .mipNum = 1,
            .layerNum = 1,
            .planes = NriPlaneBits_COLOR
        },
        {
            .texture = renderer->screen_probe_radiance.texture,
            .before = renderer->screen_probe_radiance.state,
            .after = texture_write,
            .mipNum = 1,
            .layerNum = 1,
            .planes = NriPlaneBits_COLOR
        }
    };

    const NriBufferBarrierDesc buffer_barrier = {
        .buffer = renderer->trace_hits.buffer,
        .before = renderer->trace_hits.state,
        .after = buffer_write
    };

    renderer->gpu->core.CmdBarrier(command_buffer, &(NriBarrierDesc){
        .textures = texture_barriers,
        .textureNum = 2,
        .buffers = &buffer_barrier,
        .bufferNum = 1
    });

    renderer->screen_trace.state = texture_write;
    renderer->screen_probe_radiance.state = texture_write;
    renderer->trace_hits.state = buffer_write;

    bind_trace(renderer, command_buffer, renderer->screen_trace_pipeline);
    renderer->gpu->core.CmdDispatch(
        command_buffer,
        &(NriDispatchDesc){
            .workGroupNumX = (renderer->screen_probe_radiance.width + 7u) / 8u,
            .workGroupNumY = (renderer->screen_probe_radiance.height + 7u) / 8u,
            .workGroupNumZ = 1
        }
    );
}

static void build_miss_queue'''
render = sub_once(render, pattern, replacement, "screen dispatch cleanup")

render = replace_once(
    render,
    "        /* 4x4 directional samples per 8x8 probe */\n",
    "        /* 4x4 storage slots per 8x8 probe; only 8 rays are active */\n",
    "probe resource comment",
)

render = replace_once(
    render,
    "    build_screen_probes(renderer, command_buffer);\n    record_present_pass(renderer, command_buffer, swapchain_index);\n",
    "    build_screen_probes(renderer, command_buffer);\n    build_emissive_gather(renderer, command_buffer);\n    record_present_pass(renderer, command_buffer, swapchain_index);\n",
    "probe-frequency emissive gather",
)

render_path.write_text(render)

# The obsolete duplicate direct shader entry point no longer exists.
build_path = Path("build.c")
build = build_path.read_text()
build = replace_once(
    build,
    '        {"CS_DirectRadiance", "compute", "direct_radiance.cs.spv", NULL},\n',
    "",
    "remove obsolete direct shader job",
)
build_path.write_text(build)
