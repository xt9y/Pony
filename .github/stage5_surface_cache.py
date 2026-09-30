from pathlib import Path
import re


def replace_once(text, old, new, label):
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one match, got {count}")
    return text.replace(old, new, 1)


def sub_once(text, pattern, replacement, label):
    new, count = re.subn(pattern, replacement, text, count=1, flags=re.S)
    if count != 1:
        raise SystemExit(f"{label}: expected one regex match, got {count}")
    return new

# -----------------------------------------------------------------------------
# gpu.h: the fourth GPU object metadata word is the object's global triangle
# base. This keeps SV_PrimitiveID usable across multiple model objects.
# -----------------------------------------------------------------------------
p = Path("gpu.h")
s = p.read_text()
s = replace_once(
    s,
    "    uint32_t revision;\n    uint32_t state;\n    uint32_t type;\n    uint32_t padding;\n",
    "    uint32_t revision;\n    uint32_t state;\n    uint32_t type;\n    uint32_t triangle_offset;\n",
    "GPU_OBJECT triangle offset",
)
p.write_text(s)

# -----------------------------------------------------------------------------
# game.h: keep the legacy visible-pixel cache alive only for compatibility and
# add the real 128-byte geometry-addressed cache separately.
# -----------------------------------------------------------------------------
p = Path("game.h")
s = p.read_text()
s = replace_once(
    s,
    "    SURFACE_CACHE surface_cache;\n    RADIANCE_SCENE_DATA radiance_scene;\n",
    "    SURFACE_CACHE surface_cache;\n    SURFACE_CACHE radiance_surface_cache;\n    RADIANCE_SCENE_DATA radiance_scene;\n",
    "renderer permanent surface cache",
)
p.write_text(s)

# -----------------------------------------------------------------------------
# shader.hlsl
# -----------------------------------------------------------------------------
p = Path("shader.hlsl")
s = p.read_text()

s = replace_once(
    s,
    "    [[vk::location(4)]] float4 current_clip : TEXCOORD3;\n    [[vk::location(5)]] float4 previous_clip : TEXCOORD4;\n};\n",
    "    [[vk::location(4)]] float4 current_clip : TEXCOORD3;\n    [[vk::location(5)]] float4 previous_clip : TEXCOORD4;\n    [[vk::location(6)]] nointerpolation uint primitive_base : TEXCOORD5;\n};\n",
    "G-buffer primitive base output",
)
s = replace_once(
    s,
    "    output.material = object.draw.w + input.material;\n    output.object_id = object.draw.z;\n    return output;\n",
    "    output.material = object.draw.w + input.material;\n    output.object_id = object.draw.z;\n    output.primitive_base = object.meta.w;\n    return output;\n",
    "G-buffer primitive base assignment",
)
s = replace_once(
    s,
    "    output.material_id = input.material;\n    output.primitive_id = primitive_id;\n",
    "    output.material_id = input.material;\n    output.primitive_id = input.primitive_base + primitive_id;\n",
    "global primitive id",
)

# Avoid the coarse unsigned SDF immediately re-hitting the receiver itself.
s = replace_once(
    s,
    "        if (d <= epsilon) {\n            float3 world_position = ray.origin_tmin.xyz + ray.direction_tmax.xyz * t;\n",
    "        if (d <= epsilon) {\n            float near_limit = ray.origin_tmin.w + epsilon * 1.5f;\n            if (t <= near_limit) {\n                t += max(epsilon * 1.5f / direction_scale, 1.0e-4f);\n                continue;\n            }\n            float3 world_position = ray.origin_tmin.xyz + ray.direction_tmax.xyz * t;\n",
    "local SDF near-self rejection",
)

# World-surface emissive evaluation cannot depend on a framebuffer pixel.
pattern = r"float3 EvaluateEmissiveSample\(float3 surface_position, float3 surface_normal, uint source_object_id, uint2 source_pixel, uint seed\) \{.*?\n\}\n\nfloat3 SkyRadiance"
replacement = r'''float3 EvaluateEmissiveSampleForMaterial(float3 surface_position, float3 surface_normal, uint source_object_id, uint source_material_id, uint seed) {
    if (!FeatureEnabled(RADIANCE_FEATURE_EMISSIVE) || Radiance.scene_counts.w == 0u) return 0.0f;
    if (source_material_id < Radiance.scene_counts.y) {
        GPUMaterial source_material = SceneMaterials[source_material_id];
        if (dot(source_material.emissive, source_material.emissive) > 1.0e-8f) return 0.0f;
    }
    float selector = HashFloat(seed);
    uint emitter_index = SelectEmissiveTriangle(selector);
    if (emitter_index == INVALID_INDEX) return 0.0f;
    float2 sample_u = Hammersley(seed & 31u, 32u, Hash32(seed));
    float3 light_normal;
    float3 emitted;
    float pdf_area;
    uint triangle_id;
    float3 light_position = SampleEmissivePoint(emitter_index, sample_u, light_normal, emitted, pdf_area, triangle_id);
    float3 to_light = light_position - surface_position;
    float distance_sq = dot(to_light, to_light);
    if (distance_sq <= 1.0e-8f) return 0.0f;
    float distance = sqrt(distance_sq);
    float3 L = to_light / distance;
    float cos_surface = saturate(dot(surface_normal, L));
    float cos_light = saturate(abs(dot(light_normal, -L)));
    if (cos_surface <= 0.0f || cos_light <= 0.0f) return 0.0f;
    float bias = max(Radiance.trace_params.y, 1.0e-3f);
    TraceRay shadow = MakeTraceRay(surface_position + surface_normal * bias, L, bias, max(distance - bias * 2.0f, bias), TRACE_RAY_SHADOW, 0u, uint2(0u, 0u), source_object_id);
    if (TraceUnifiedOcclusion(shadow)) return 0.0f;
    float geometry = cos_surface * cos_light / max(distance_sq, 1.0e-6f);
    return emitted * geometry / max(pdf_area * PI, 1.0e-8f);
}

float3 EvaluateEmissiveSample(float3 surface_position, float3 surface_normal, uint source_object_id, uint2 source_pixel, uint seed) {
    uint source_material_id = TraceMaterialId.Load(int3(source_pixel, 0));
    return EvaluateEmissiveSampleForMaterial(surface_position, surface_normal, source_object_id, source_material_id, seed);
}

float3 SkyRadiance'''
s = sub_once(s, pattern, replacement, "pixel-independent emissive sampling")

# One direct-light evaluator is shared by visible G-buffer receivers and SDF hits.
pattern = r"float3 FutureSkyRadiance\(float3 direction\) \{.*?\n\}\n\n// -----------------------------------------------------------------------------\n// Compatibility direct lighting\."
match = re.search(pattern, s, re.S)
if not match:
    raise SystemExit("shared direct evaluator insertion: no FutureSkyRadiance block")
future_block = match.group(0)
future_func = future_block.rsplit("\n\n// -----------------------------------------------------------------------------", 1)[0]
shared = future_func + r'''

float3 EvaluateSurfaceReflectedDirect(SurfaceHit hit, uint seed) {
    if (hit.identity.y >= Radiance.scene_counts.y) return 0.0f;
    GPUMaterial material = SceneMaterials[hit.identity.y];
    float3 position = hit.position_distance.xyz;
    float3 normal = normalize(hit.normal_confidence.xyz);
    if (dot(normal, normal) <= 1.0e-8f) return 0.0f;

    float3 reflected = 0.0f;
    for (uint i = 0u; i < Radiance.sdf_counts.w; ++i) {
        GPULight light = SceneLights[i];
        uint type = (uint)(light.direction_type.w + 0.5f);
        if (type == 3u) continue;

        float3 L;
        float attenuation = 1.0f;
        float max_distance = Radiance.trace_params.x;
        if (type == 0u) {
            L = normalize(-light.direction_type.xyz);
        } else {
            float3 to_light = light.position_range.xyz - position;
            float d = length(to_light);
            if (d <= 1.0e-5f || d >= light.position_range.w) continue;
            L = to_light / d;
            max_distance = d;
            float range_term = saturate(1.0f - d / max(light.position_range.w, 1.0e-3f));
            attenuation = range_term * range_term / max(1.0f, d * d);
            if (type == 2u) {
                float cone = dot(normalize(light.direction_type.xyz), -L);
                float cone_term = saturate((cone - light.spot_angles.y) / max(light.spot_angles.x - light.spot_angles.y, 1.0e-4f));
                attenuation *= cone_term * cone_term;
            }
        }

        float ndotl = saturate(dot(normal, L));
        if (ndotl <= 0.0f) continue;
        float bias = max(Radiance.trace_params.y, 1.0e-3f);
        float tmax = max(max_distance - bias * 2.0f, bias);
        TraceRay shadow = MakeTraceRay(position + normal * bias, L, bias, tmax, TRACE_RAY_SHADOW, 0u, uint2(0u, 0u), hit.identity.x);
        if (TraceUnifiedOcclusion(shadow)) continue;
        reflected += material.base_color.rgb * light.color_intensity.rgb * light.color_intensity.w * attenuation * ndotl;
    }

    float3 emissive_direct = EvaluateEmissiveSampleForMaterial(position, normal, hit.identity.x, hit.identity.y, seed);
    reflected += material.base_color.rgb * emissive_direct;
    return reflected;
}

float3 SurfaceReflectedRadiance(SurfaceHit hit);

// -----------------------------------------------------------------------------
// Compatibility direct lighting.'''
s = s[: match.start()] + shared + s[match.end() :]

# SDF misses now use material-aware surface identity and the permanent cache.
pattern = r"\[numthreads\(64, 1, 1\)\]\nvoid CS_SDFTrace\(uint3 dispatch_id : SV_DispatchThreadID\) \{.*?\n\}\n\n\[numthreads\(8, 8, 1\)\]\nvoid CS_SurfaceCacheUpdate"
replacement = r'''[numthreads(64, 1, 1)]
void CS_SDFTrace(uint3 dispatch_id : SV_DispatchThreadID) {
    uint queue_index = dispatch_id.x;
    if (queue_index >= MissCount[0]) return;
    TraceRay ray = MissQueue[queue_index];

    uint ray_width, ray_height;
    ProbeRadianceOutput.GetDimensions(ray_width, ray_height);
    uint2 destination = uint2(ray.destination % ray_width, ray.destination / ray_width);

    SurfaceHit surface = TraceUnifiedRay(ray, false);
    if (surface.identity.w == TRACE_MISS) {
        float3 sky = FutureSkyRadiance(ray.direction_tmax.xyz);
        ProbeRadianceOutput[destination] = float4(sky, 1.0f);
        ScreenTraceOutput[destination] = float4(sky, 1.0f);
        ScreenTraceHits[ray.destination] = MakeTraceHit(TRACE_MISS, ray.direction_tmax.w);
        return;
    }

    TraceHit hit = MakeTraceHit(surface.identity.w, surface.position_distance.w);
    hit.object_id = surface.identity.x;
    hit.confidence = surface.normal_confidence.w;
    ScreenTraceHits[ray.destination] = hit;

    // Diffuse screen probes already receive explicit emitter NEE in the direct
    // pass, so gather reflected surface radiance here rather than raw emission.
    float3 radiance = SurfaceReflectedRadiance(surface);
    ProbeRadianceOutput[destination] = float4(radiance, 1.0f);
    ScreenTraceOutput[destination] = float4(radiance, 1.0f);
}

[numthreads(8, 8, 1)]
void CS_SurfaceCacheUpdate'''
s = sub_once(s, pattern, replacement, "permanent SDF surface shading")

# Permanent cache miss evaluates that exact world surface rather than returning
# zero until it happened to be visible to the camera.
pattern = r"bool SurfaceCacheLookup\(SurfaceHit hit, out SurfaceCacheEntry entry\) \{.*?\n\}\n\nfloat3 SurfaceOutgoingRadiance\(SurfaceHit hit\) \{.*?\n\}\n\nvoid SurfaceCacheStore\(SurfaceHit hit, float3 direct, float3 indirect, float confidence\) \{.*?\n\}\n"
replacement = r'''bool SurfaceCacheLookup(SurfaceHit hit, out SurfaceCacheEntry entry) {
    entry = (SurfaceCacheEntry)0;
    if (!FeatureEnabled(RADIANCE_FEATURE_SURFACE_CACHE) || Radiance.cache_counts.x == 0u || hit.identity.z == INVALID_INDEX) return false;
    uint slot = SurfaceCacheFindSlot(SurfaceCacheKey(hit), false);
    if (slot == INVALID_INDEX) return false;
    entry = SurfaceCacheEntries[slot];
    return entry.identity.x == hit.identity.x &&
           entry.identity.z == hit.identity.z &&
           entry.identity.w == hit.meta.y &&
           entry.state.x == Radiance.feature_flags.y &&
           entry.state.w != 0u;
}

void SurfaceCacheStore(SurfaceHit hit, float3 direct, float3 indirect, float confidence) {
    if (Radiance.cache_counts.x == 0u || hit.identity.z == INVALID_INDEX) return;
    uint key = SurfaceCacheKey(hit);
    uint slot = SurfaceCacheFindSlot(key, true);
    if (slot == INVALID_INDEX) return;
    SurfaceCacheEntry entry;
    entry.position_distance = hit.position_distance;
    entry.normal_confidence = float4(hit.normal_confidence.xyz, confidence);
    GPUMaterial material = hit.identity.y < Radiance.scene_counts.y ? SceneMaterials[hit.identity.y] : (GPUMaterial)0;
    entry.albedo_roughness = float4(material.base_color.rgb, material.roughness);
    entry.emissive_metallic = float4(material.emissive, material.metallic);
    entry.direct_radiance = float4(direct, 1.0f);
    entry.indirect_radiance = float4(indirect, 1.0f);
    entry.identity = uint4(hit.identity.x, hit.identity.y, hit.identity.z, hit.meta.y);
    entry.state = uint4(Radiance.feature_flags.y, Pass.dispatch.x, 0u, 1u);
    SurfaceCacheEntries[slot] = entry;
}

float3 SurfaceReflectedRadiance(SurfaceHit hit) {
    if (hit.identity.w == TRACE_MISS || hit.identity.y >= Radiance.scene_counts.y) return 0.0f;
    SurfaceCacheEntry entry;
    if (SurfaceCacheLookup(hit, entry)) return entry.direct_radiance.rgb + entry.indirect_radiance.rgb;
    uint seed = HashCombine(SurfaceCacheKey(hit), HashCombine(Pass.dispatch.x, Radiance.feature_flags.y));
    float3 reflected = EvaluateSurfaceReflectedDirect(hit, seed);
    SurfaceCacheStore(hit, reflected, 0.0f, 1.0f);
    return reflected;
}

float3 SurfaceOutgoingRadiance(SurfaceHit hit) {
    if (hit.identity.w == TRACE_MISS) return FutureSkyRadiance(normalize(hit.normal_confidence.xyz));
    if (hit.identity.y >= Radiance.scene_counts.y) return 0.0f;
    return SceneMaterials[hit.identity.y].emissive + SurfaceReflectedRadiance(hit);
}
'''
s = sub_once(s, pattern, replacement, "geometry-addressed cache behavior")

# Visible and offscreen surfaces must use exactly the same direct evaluator and
# cache semantics.
pattern = r"\[numthreads\(8, 8, 1\)\]\nvoid CS_RadianceDirect\(uint3 dispatch_id : SV_DispatchThreadID\) \{.*?\n\}\n\n// -----------------------------------------------------------------------------\n// Runtime debug and presentation helpers\."
replacement = r'''[numthreads(8, 8, 1)]
void CS_RadianceDirect(uint3 dispatch_id : SV_DispatchThreadID) {
    uint2 pixel = dispatch_id.xy;
    uint2 resolution = uint2(TraceFrame.resolution.xy);
    if (any(pixel >= resolution)) return;
    float depth = TraceDepth.Load(int3(pixel, 0));
    if (depth <= 0.0f) {
        float2 uv = (float2(pixel) + 0.5f) * TraceFrame.resolution.zw;
        float4 far_world4 = mul(float4(ScreenUVToNDC(uv), 0.0f, 1.0f), TraceFrame.inverse_view_projection);
        float3 direction = normalize(far_world4.xyz / far_world4.w - TraceFrame.camera_position.xyz);
        DirectRadianceOutput[pixel] = float4(FutureSkyRadiance(direction), 1.0f);
        return;
    }

    float3 position = ReconstructWorldPosition(pixel, depth);
    float3 normal = normalize(TraceNormalRoughness.Load(int3(pixel, 0)).xyz);
    float3 view = normalize(position - TraceFrame.camera_position.xyz);
    if (dot(normal, view) > 0.0f) normal = -normal;

    uint primitive_id = TracePrimitiveId.Load(int3(pixel, 0));
    SurfaceHit hit = SurfaceFromTriangle(primitive_id, position, normal, 0.0f, TRACE_SCREEN);
    if (hit.identity.z == INVALID_INDEX) {
        hit.position_distance = float4(position, 0.0f);
        hit.normal_confidence = float4(normal, 1.0f);
        hit.identity = uint4(TraceObjectId.Load(int3(pixel, 0)), TraceMaterialId.Load(int3(pixel, 0)), primitive_id, TRACE_SCREEN);
        hit.meta.y = Radiance.feature_flags.y;
    } else {
        uint object_index = SceneTriangles[primitive_id].meta.x;
        hit.meta.y = object_index < Radiance.scene_counts.x ? SceneObjects[object_index].meta.x : Radiance.feature_flags.y;
    }

    GPUMaterial material = hit.identity.y < Radiance.scene_counts.y ? SceneMaterials[hit.identity.y] : (GPUMaterial)0;
    uint seed = HashCombine(Pass.dispatch.x, PackPixel(pixel));
    float3 reflected = EvaluateSurfaceReflectedDirect(hit, seed);
    DirectRadianceOutput[pixel] = float4(material.emissive + reflected, 1.0f);
    SurfaceCacheStore(hit, reflected, 0.0f, 1.0f);
}

// -----------------------------------------------------------------------------
// Runtime debug and presentation helpers.'''
s = sub_once(s, pattern, replacement, "shared visible direct path")

p.write_text(s)

# -----------------------------------------------------------------------------
# render.c
# -----------------------------------------------------------------------------
p = Path("render.c")
s = p.read_text()

# A dedicated cache has the permanent 128-byte entry stride. The legacy cache
# remains bound in space3 until Chapter 6 removes that compatibility ABI.
marker = "static void destroy_radiance_scene_gpu_resources(RENDERER *renderer) {"
if marker not in s:
    raise SystemExit("radiance cache helpers: insertion marker missing")
helpers = r'''static bool clear_radiance_surface_cache(RENDERER *renderer) {
    SURFACE_CACHE *cache = &renderer->radiance_surface_cache;
    if (!cache->keys || !cache->capacity) return false;
    uint32_t *zero_keys = calloc(cache->capacity, sizeof(*zero_keys));
    if (!zero_keys) return false;
    const NriAccessStage storage = {
        .access = NriAccessBits_SHADER_RESOURCE_STORAGE,
        .stages = NriStageBits_COMPUTE_SHADER
    };
    const bool ok = gpu_upload_buffer(renderer->gpu, cache->keys, zero_keys, storage);
    free(zero_keys);
    if (!ok) return false;
    cache->keys_state = storage;
    return true;
}

static void destroy_radiance_surface_cache(RENDERER *renderer) {
    SURFACE_CACHE *cache = &renderer->radiance_surface_cache;
    if (cache->keys_uav) renderer->gpu->core.DestroyDescriptor(cache->keys_uav);
    if (cache->entries_uav) renderer->gpu->core.DestroyDescriptor(cache->entries_uav);
    if (cache->keys) gpu_destroy_buffer(renderer->gpu, cache->keys);
    if (cache->entries) gpu_destroy_buffer(renderer->gpu, cache->entries);
    memset(cache, 0, sizeof(*cache));
}

static bool create_radiance_surface_cache(RENDERER *renderer) {
    destroy_radiance_surface_cache(renderer);
    SURFACE_CACHE *cache = &renderer->radiance_surface_cache;
    cache->capacity = SURFACE_CACHE_CAPACITY;

    const NriBufferDesc keys_desc = {
        .size = (uint64_t)cache->capacity * sizeof(uint32_t),
        .structureStride = sizeof(uint32_t),
        .usage = NriBufferUsageBits_SHADER_RESOURCE_STORAGE
    };
    const NriBufferDesc entries_desc = {
        .size = (uint64_t)cache->capacity * sizeof(SURFACE_RADIANCE_ENTRY),
        .structureStride = sizeof(SURFACE_RADIANCE_ENTRY),
        .usage = NriBufferUsageBits_SHADER_RESOURCE_STORAGE
    };

    if (!gpu_create_buffer(renderer->gpu, &keys_desc, NriMemoryLocation_DEVICE, &cache->keys) ||
        !gpu_create_buffer(renderer->gpu, &entries_desc, NriMemoryLocation_DEVICE, &cache->entries))
        return false;
    if (!create_buffer_view(renderer, cache->keys, NriBufferView_STORAGE_STRUCTURED_BUFFER, keys_desc.size, sizeof(uint32_t), &cache->keys_uav) ||
        !create_buffer_view(renderer, cache->entries, NriBufferView_STORAGE_STRUCTURED_BUFFER, entries_desc.size, sizeof(SURFACE_RADIANCE_ENTRY), &cache->entries_uav))
        return false;

    return clear_radiance_surface_cache(renderer);
}

'''
s = s.replace(marker, helpers + marker, 1)

# Record the global triangle base in each object while the authoritative scene
# triangle array is being assembled.
s = replace_once(
    s,
    "        GPU_OBJECT *gpu_object = &renderer->cpu_objects[object_index];\n\n        for (uint32_t face_index = 0;",
    "        GPU_OBJECT *gpu_object = &renderer->cpu_objects[object_index];\n        gpu_object->triangle_offset = triangle_index;\n\n        for (uint32_t face_index = 0;",
    "object triangle offset build",
)

s = replace_once(
    s,
    "    constants.cache_counts[0] = 0u;\n",
    "    constants.cache_counts[0] = renderer->radiance_surface_cache.capacity;\n",
    "surface cache count",
)
s = replace_once(
    s,
    "    constants.feature_flags[0] = renderer->radiance_scene.emissive_triangle_count ? RADIANCE_FEATURE_EMISSIVE : 0u;\n",
    "    constants.feature_flags[0] = RADIANCE_FEATURE_SURFACE_CACHE;\n    if (renderer->radiance_scene.emissive_triangle_count) constants.feature_flags[0] |= RADIANCE_FEATURE_EMISSIVE;\n",
    "surface cache feature enable",
)

s = replace_once(
    s,
    "    if (!renderer || !renderer->radiance_direct_cache_set || !renderer->surface_cache.keys_uav || !renderer->surface_cache.entries_uav) return false;\n\n    const NriDescriptor *descriptors[] = {renderer->surface_cache.keys_uav, renderer->surface_cache.entries_uav};\n",
    "    if (!renderer || !renderer->radiance_direct_cache_set || !renderer->radiance_surface_cache.keys_uav || !renderer->radiance_surface_cache.entries_uav) return false;\n\n    const NriDescriptor *descriptors[] = {renderer->radiance_surface_cache.keys_uav, renderer->radiance_surface_cache.entries_uav};\n",
    "permanent cache descriptors",
)

# The SDF fallback now references spaces 3,4,6, so create it with the same
# permanent world-radiance layout used by CS_RadianceDirect.
s = replace_once(
    s,
    '           create_compute_pipeline(renderer, "build/shaders/sdf_trace.cs.spv", renderer->trace_layout, &renderer->sdf_trace_pipeline) &&\n',
    '           create_compute_pipeline(renderer, "build/shaders/sdf_trace.cs.spv", renderer->radiance_direct_layout, &renderer->sdf_trace_pipeline) &&\n',
    "SDF permanent layout",
)

# Generic binder shared by the permanent direct and SDF surface paths.
old = '''static void bind_radiance_direct(RENDERER *renderer, NriCommandBuffer *command_buffer) {
    renderer->gpu->core.CmdSetPipelineLayout(command_buffer, NriBindPoint_COMPUTE, renderer->radiance_direct_layout);
    renderer->gpu->core.CmdSetPipeline(command_buffer, renderer->direct_radiance_pipeline);
'''
new = '''static void bind_radiance_world(RENDERER *renderer, NriCommandBuffer *command_buffer, NriPipeline *pipeline) {
    renderer->gpu->core.CmdSetPipelineLayout(command_buffer, NriBindPoint_COMPUTE, renderer->radiance_direct_layout);
    renderer->gpu->core.CmdSetPipeline(command_buffer, pipeline);
'''
s = replace_once(s, old, new, "generic permanent radiance binder")
s = replace_once(
    s,
    "    bind_radiance_direct(renderer, command_buffer);\n",
    "    bind_radiance_world(renderer, command_buffer, renderer->direct_radiance_pipeline);\n",
    "direct permanent binder call",
)

# Direct lighting writes visible entries into the same persistent cache used by
# offscreen SDF hits. Add an explicit UAV dependency before dispatch.
needle = '''    renderer->gpu->core.CmdBarrier(command_buffer, &(NriBarrierDesc){
        .textures = &to_write,
        .textureNum = 1
    });
    bind_radiance_world(renderer, command_buffer, renderer->direct_radiance_pipeline);
'''
replacement = '''    const NriAccessStage cache_storage = {
        .access = NriAccessBits_SHADER_RESOURCE_STORAGE,
        .stages = NriStageBits_COMPUTE_SHADER
    };
    const NriBufferBarrierDesc cache_barriers[] = {
        {.buffer = renderer->radiance_surface_cache.keys, .before = renderer->radiance_surface_cache.keys_state, .after = cache_storage},
        {.buffer = renderer->radiance_surface_cache.entries, .before = renderer->radiance_surface_cache.entries_state, .after = cache_storage}
    };

    renderer->gpu->core.CmdBarrier(command_buffer, &(NriBarrierDesc){
        .textures = &to_write,
        .textureNum = 1,
        .buffers = cache_barriers,
        .bufferNum = 2
    });
    renderer->radiance_surface_cache.keys_state = cache_storage;
    renderer->radiance_surface_cache.entries_state = cache_storage;
    bind_radiance_world(renderer, command_buffer, renderer->direct_radiance_pipeline);
'''
s = replace_once(s, needle, replacement, "direct cache barrier")

# SDF probe misses now synchronize/use the permanent cache rather than the
# visible-pixel legacy cache.
s = replace_once(s, "            .buffer = renderer->surface_cache.keys,\n            .before = renderer->surface_cache.keys_state,\n", "            .buffer = renderer->radiance_surface_cache.keys,\n            .before = renderer->radiance_surface_cache.keys_state,\n", "SDF cache keys barrier")
s = replace_once(s, "            .buffer = renderer->surface_cache.entries,\n            .before = renderer->surface_cache.entries_state,\n", "            .buffer = renderer->radiance_surface_cache.entries,\n            .before = renderer->radiance_surface_cache.entries_state,\n", "SDF cache entries barrier")
s = replace_once(
    s,
    "    renderer->screen_probe_radiance.state = texture_storage;\n\n    bind_trace(renderer, command_buffer, renderer->sdf_trace_pipeline);\n",
    "    renderer->screen_probe_radiance.state = texture_storage;\n    renderer->radiance_surface_cache.keys_state = storage;\n    renderer->radiance_surface_cache.entries_state = storage;\n\n    bind_radiance_world(renderer, command_buffer, renderer->sdf_trace_pipeline);\n",
    "SDF permanent cache bind",
)

# Allocate both caches: the old 112-byte cache is compatibility-only; the new
# 128-byte cache is the source of truth for world-space surfaces.
s = replace_once(
    s,
    "        !create_radiance_scene_fallbacks(renderer) || !create_pipelines(renderer) || !create_surface_cache(renderer) ||\n        !update_radiance_direct_cache_descriptors(renderer) ||\n",
    "        !create_radiance_scene_fallbacks(renderer) || !create_pipelines(renderer) || !create_surface_cache(renderer) ||\n        !create_radiance_surface_cache(renderer) || !update_radiance_direct_cache_descriptors(renderer) ||\n",
    "create permanent surface cache",
)
s = replace_once(
    s,
    "        destroy_surface_cache(renderer);\n        destroy_compute_texture(renderer, &renderer->direct_radiance);\n",
    "        destroy_surface_cache(renderer);\n        destroy_radiance_surface_cache(renderer);\n        destroy_compute_texture(renderer, &renderer->direct_radiance);\n",
    "destroy permanent surface cache",
)
s = replace_once(
    s,
    "    if (!clear_surface_cache(renderer)) return false;\n",
    "    if (!clear_surface_cache(renderer) || !clear_radiance_surface_cache(renderer)) return false;\n",
    "clear permanent surface cache on scene set",
)

# The legacy visible-pixel cache is no longer part of the frame's lighting
# truth. It stays allocated only to satisfy the compatibility descriptor set.
s = replace_once(
    s,
    "    build_direct_radiance(renderer, command_buffer);\n    build_surface_cache(renderer, command_buffer);\n    build_screen_trace(renderer, command_buffer);\n",
    "    build_direct_radiance(renderer, command_buffer);\n    build_screen_trace(renderer, command_buffer);\n",
    "retire visible-only cache dispatch",
)

p.write_text(s)
