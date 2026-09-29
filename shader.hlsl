#define FAR_PLANE 10000.0f
#define NEAR_PLANE 0.05f
#define TRACE_INACTIVE 0u
#define TRACE_MISS 1u
#define TRACE_SCREEN 2u
#define TRACE_SDF 3u
#define SURFACE_CACHE_CAPACITY 65536u

struct GPUObject {
    row_major float4x4 world;
    row_major float4x4 previous_world;
    row_major float4x4 normal_world;
    uint4 draw;
    uint4 meta;
};

struct GPUMaterial {
    float4 base_color;
    float3 emissive;
    float metallic;
    float roughness;
    float3 padding;
};

struct FrameConstants {
    row_major float4x4 view_projection;
    row_major float4x4 inverse_view_projection;
    row_major float4x4 previous_view_projection;
    float4 camera_position;
    float4 resolution;
    float4 trace_params;
    uint4 trace_limits;
};

struct TraceHit {
    uint type;
    uint object_id;
    uint2 hit_pixel;
    float distance;
    float confidence;
    uint2 padding;
};

struct TraceRay {
    float4 origin_tmin;
    float4 direction_tmax;
    uint2 source_pixel;
    uint2 padding;
};

struct GPUSDFModel {
    row_major float4x4 world_to_local;
    float4 bounds_min;
    float4 bounds_max;
    uint4 meta;
};

struct GPULight {
    float4 position_range;
    float4 direction_type;
    float4 color_intensity;
    float4 spot_angles;
};

struct SurfaceCacheEntry {
    float4 position;
    float4 radiance;
    uint4 meta;
};

[[vk::binding(0, 0)]] StructuredBuffer<GPUObject> Objects : register(t0, space0);
[[vk::binding(1, 0)]] StructuredBuffer<GPUMaterial> Materials : register(t1, space0);
[[vk::binding(2, 0)]] ConstantBuffer<FrameConstants> Frame : register(b0, space0);

struct DrawConstants {
    uint object_index;
};

[[vk::push_constant]] DrawConstants Draw;

struct GBufferVSInput {
    [[vk::location(0)]] float3 position : POSITION;
    [[vk::location(1)]] float3 normal : NORMAL0;
    [[vk::location(2)]] float2 uv : TEXCOORD0;
    [[vk::location(3)]] uint material : TEXCOORD1;
};

struct GBufferVSOutput {
    float4 position : SV_Position;
    [[vk::location(0)]] float3 normal : NORMAL0;
    [[vk::location(1)]] float2 uv : TEXCOORD0;
    [[vk::location(2)]] nointerpolation uint material : TEXCOORD1;
    [[vk::location(3)]] nointerpolation uint object_id : TEXCOORD2;
    [[vk::location(4)]] float4 current_clip : TEXCOORD3;
    [[vk::location(5)]] float4 previous_clip : TEXCOORD4;
};

GBufferVSOutput VS_GBuffer(GBufferVSInput input) {
    GPUObject object = Objects[Draw.object_index];
    float4 local_position = float4(input.position, 1.0);
    float4 world_position = mul(local_position, object.world);
    float4 previous_world_position = mul(local_position, object.previous_world);

    GBufferVSOutput output;
    output.position = mul(world_position, Frame.view_projection);
    output.current_clip = output.position;
    output.previous_clip = mul(previous_world_position, Frame.previous_view_projection);
    output.normal = normalize(mul(float4(input.normal, 0.0), object.normal_world).xyz);
    output.uv = input.uv;
    output.material = object.draw.w + input.material;
    output.object_id = object.draw.z;
    return output;
}

struct GBufferOutput {
    float4 normal_roughness : SV_Target0;
    float4 albedo_metallic : SV_Target1;
    float2 velocity : SV_Target2;
    uint object_id : SV_Target3;
    float4 emissive : SV_Target4;
};

GBufferOutput PS_GBuffer(GBufferVSOutput input) {
    GPUMaterial material = Materials[input.material];
    GBufferOutput output;
    output.normal_roughness = float4(normalize(input.normal), material.roughness);
    output.albedo_metallic = float4(material.base_color.rgb, material.metallic);
    output.emissive = float4(material.emissive, 1.0);

    const float2 current_ndc = input.current_clip.xy / input.current_clip.w;
    const float2 previous_ndc = input.previous_clip.xy / input.previous_clip.w;
    output.velocity = (current_ndc - previous_ndc) * float2(0.5, -0.5);
    output.object_id = input.object_id;
    return output;
}

struct PresentVSOutput {
    float4 position : SV_Position;
};

PresentVSOutput VS_Present(uint vertex_id : SV_VertexID) {
    const float2 positions[3] = {
        float2(-1.0, -1.0),
        float2(-1.0, 3.0),
        float2(3.0, -1.0)
    };

    PresentVSOutput output;
    output.position = float4(positions[vertex_id], 0.0, 1.0);
    return output;
}

[[vk::binding(0, 1)]] Texture2D<float> DepthTexture : register(t0, space1);
[[vk::binding(1, 1)]] Texture2D<float4> NormalRoughnessTexture : register(t1, space1);
[[vk::binding(2, 1)]] Texture2D<float4> AlbedoMetallicTexture : register(t2, space1);
[[vk::binding(3, 1)]] Texture2D<float2> VelocityTexture : register(t3, space1);
[[vk::binding(4, 1)]] Texture2D<uint> ObjectIdTexture : register(t4, space1);
[[vk::binding(5, 1)]] Texture2D<float> HZBTexture : register(t5, space1);
[[vk::binding(6, 1)]] Texture2D<float4> ScreenTraceTexture : register(t6, space1);
[[vk::binding(7, 1)]] Texture2D<float4> DirectRadianceTexture : register(t7, space1);
[[vk::binding(8, 1)]] Texture2D<float4> ScreenProbesTexture : register(t8, space1);

[[vk::binding(0, 2)]] Texture2D<float> HZBSource : register(t0, space2);
[[vk::binding(1, 2)]] RWTexture2D<float> HZBOutput : register(u0, space2);

[[vk::binding(0, 3)]] Texture2D<float> TraceDepth : register(t0, space3);
[[vk::binding(1, 3)]] Texture2D<float4> TraceNormalRoughness : register(t1, space3);
[[vk::binding(2, 3)]] Texture2D<float4> TraceAlbedoMetallic : register(t2, space3);
[[vk::binding(3, 3)]] Texture2D<float4> TraceEmissive : register(t3, space3);
[[vk::binding(4, 3)]] Texture2D<float> TraceHZB : register(t4, space3);
[[vk::binding(5, 3)]] Texture2D<uint> TraceObjectId : register(t5, space3);
[[vk::binding(6, 3)]] Texture2D<float4> TraceDirectRadiance : register(t6, space3);
[[vk::binding(7, 3)]] ConstantBuffer<FrameConstants> TraceFrame : register(b0, space3);
[[vk::binding(8, 3)]] RWTexture2D<float4> ScreenTraceOutput : register(u0, space3);
[[vk::binding(9, 3)]] RWStructuredBuffer<TraceHit> ScreenTraceHits : register(u1, space3);
[[vk::binding(10, 3)]] RWStructuredBuffer<TraceRay> MissQueue : register(u2, space3);
[[vk::binding(11, 3)]] RWStructuredBuffer<uint> MissCount : register(u3, space3);
[[vk::binding(12, 3)]] RWStructuredBuffer<uint> TraceDispatchArgs : register(u4, space3);
[[vk::binding(13, 3)]] StructuredBuffer<GPUSDFModel> SDFModels : register(t7, space3);
[[vk::binding(14, 3)]] StructuredBuffer<float> SDFVoxels : register(t8, space3);
[[vk::binding(15, 3)]] StructuredBuffer<GPULight> Lights : register(t9, space3);
[[vk::binding(16, 3)]] RWTexture2D<float4> DirectRadianceOutput : register(u5, space3);
[[vk::binding(17, 3)]] RWStructuredBuffer<uint> SurfaceCacheKeys : register(u6, space3);
[[vk::binding(18, 3)]] RWStructuredBuffer<SurfaceCacheEntry> SurfaceCacheEntries : register(u7, space3);
[[vk::binding(19, 3)]] RWTexture2D<float4> ScreenProbesOutput : register(u8, space3);

float2 ScreenUVToNDC(float2 uv) {
    return float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
}

float2 NDCToScreenUV(float2 ndc) {
    return float2(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
}

float3 ReconstructWorldPosition(uint2 pixel, float depth) {
    const float2 uv = (float2(pixel) + 0.5) * TraceFrame.resolution.zw;
    const float2 ndc = ScreenUVToNDC(uv);
    const float4 world = mul(float4(ndc, depth, 1.0), TraceFrame.inverse_view_projection);
    return world.xyz / world.w;
}

float LinearizeDepth(float depth) {
    return (NEAR_PLANE * FAR_PLANE) / (depth * (FAR_PLANE - NEAR_PLANE) + NEAR_PLANE);
}

uint2 TraceLevelDimensions(uint level) {
    if (level == 0u) return uint2(TraceFrame.resolution.xy);
    uint width, height, mip_count;
    TraceHZB.GetDimensions(level - 1u, width, height, mip_count);
    return uint2(width, height);
}

float TraceLevelDepth(float2 uv, uint level, out uint2 cell) {
    const uint2 dimensions = TraceLevelDimensions(level);
    cell = min(uint2(uv * dimensions), dimensions - 1u);
    if (level == 0u) return TraceDepth.Load(int3(cell, 0));
    return TraceHZB.Load(int3(cell, level - 1u));
}

bool ProjectTracePoint(float4 clip_origin, float4 clip_direction, float distance, out float2 uv, out float depth) {
    uv = 0.0;
    depth = 0.0;
    const float4 clip = clip_origin + clip_direction * distance;
    if (clip.w <= 1.0e-5) return false;
    const float3 ndc = clip.xyz / clip.w;
    uv = NDCToScreenUV(ndc.xy);
    depth = ndc.z;
    return all(uv >= 0.0) && all(uv < 1.0) && depth >= 0.0 && depth <= 1.0;
}

float SolveTraceDistance(float value_origin, float value_direction, float w_origin, float w_direction, float target) {
    const float denominator = value_direction - target * w_direction;
    if (abs(denominator) < 1.0e-7) return 1.0e30;
    return (target * w_origin - value_origin) / denominator;
}

float TraceCellExit(float4 clip_origin, float4 clip_direction, uint2 dimensions, uint2 cell, float distance, float2 screen_direction) {
    float exit_x = 1.0e30;
    float exit_y = 1.0e30;
    const float4 current_clip = clip_origin + clip_direction * distance;

    if (abs(screen_direction.x) > 1.0e-7) {
        const float boundary = screen_direction.x > 0.0 ? (float)(cell.x + 1u) / dimensions.x : (float)cell.x / dimensions.x;
        const float target_ndc = boundary * 2.0 - 1.0;
        exit_x = distance + SolveTraceDistance(current_clip.x, clip_direction.x, current_clip.w, clip_direction.w, target_ndc);
        if (exit_x <= distance + 1.0e-5) exit_x = 1.0e30;
    }

    if (abs(screen_direction.y) > 1.0e-7) {
        const float boundary = screen_direction.y > 0.0 ? (float)(cell.y + 1u) / dimensions.y : (float)cell.y / dimensions.y;
        const float target_ndc = 1.0 - boundary * 2.0;
        exit_y = distance + SolveTraceDistance(current_clip.y, clip_direction.y, current_clip.w, clip_direction.w, target_ndc);
        if (exit_y <= distance + 1.0e-5) exit_y = 1.0e30;
    }

    return min(exit_x, exit_y);
}

uint AscendTraceLevel(float2 previous_uv, float2 next_uv, uint level, uint maximum_level) {
    while (level < maximum_level) {
        const uint parent_level = level + 1u;
        const uint2 dimensions = TraceLevelDimensions(parent_level);
        const uint2 before_cell = min(uint2(previous_uv * dimensions), dimensions - 1u);
        const uint2 after_cell = min(uint2(next_uv * dimensions), dimensions - 1u);
        if (all(before_cell == after_cell)) break;
        level = parent_level;
    }
    return level;
}

TraceHit MakeTraceHit(uint type, float distance) {
    TraceHit hit;
    hit.type = type;
    hit.object_id = 0u;
    hit.hit_pixel = uint2(~0u, ~0u);
    hit.distance = distance;
    hit.confidence = 0.0;
    hit.padding = 0u;
    return hit;
}

TraceHit TraceScreenRay(float3 origin, float3 direction, uint2 origin_pixel, float max_distance, float thickness, float skip_thickness, uint max_steps, uint max_start_level) {
    TraceHit result = MakeTraceHit(TRACE_MISS, max_distance);
    const float4 clip_origin = mul(float4(origin, 1.0), TraceFrame.view_projection);
    const float4 clip_direction = mul(float4(direction, 0.0), TraceFrame.view_projection);
    if (clip_origin.w <= 1.0e-5) return result;

    uint hzb_width, hzb_height, hzb_mip_count;
    TraceHZB.GetDimensions(0, hzb_width, hzb_height, hzb_mip_count);
    const uint maximum_level = min(hzb_mip_count, max_start_level);
    uint level = maximum_level;
    const float2 screen_direction = float2(
        clip_direction.x * clip_origin.w - clip_origin.x * clip_direction.w,
        -(clip_direction.y * clip_origin.w - clip_origin.y * clip_direction.w)
    );

    float distance = 0.0;

    [loop]
    for (uint step = 0u; step < max_steps; ++step) {
        if (distance >= max_distance) return result;

        float2 uv;
        float ray_depth;
        if (!ProjectTracePoint(clip_origin, clip_direction, distance, uv, ray_depth)) return result;

        uint2 cell;
        const float scene_depth = TraceLevelDepth(uv, level, cell);
        const uint2 dimensions = TraceLevelDimensions(level);
        float exit_distance = TraceCellExit(clip_origin, clip_direction, dimensions, cell, distance, screen_direction);
        if (exit_distance >= 1.0e29) return result;
        exit_distance = min(exit_distance, max_distance);

        float2 exit_uv;
        float exit_depth;
        if (!ProjectTracePoint(clip_origin, clip_direction, exit_distance, exit_uv, exit_depth)) return result;

        const float farther_ray_depth = min(ray_depth, exit_depth);
        const float scene_linear = LinearizeDepth(scene_depth);
        const float farther_ray_linear = LinearizeDepth(farther_ray_depth);
        const bool clear_cell = scene_depth <= 0.0 || farther_ray_linear < scene_linear - skip_thickness;

        if (clear_cell) {
            const float next_distance = exit_distance + max(1.0e-4, exit_distance * 1.0e-5);
            float2 next_uv;
            float next_depth;
            if (!ProjectTracePoint(clip_origin, clip_direction, next_distance, next_uv, next_depth)) return result;
            level = AscendTraceLevel(uv, next_uv, level, maximum_level);
            distance = next_distance;
            continue;
        }

        if (level > 0u) {
            --level;
            continue;
        }

        const float ray_linear_a = LinearizeDepth(ray_depth);
        const float ray_linear_b = LinearizeDepth(exit_depth);
        const float segment_near = min(ray_linear_a, ray_linear_b);
        const float segment_far = max(ray_linear_a, ray_linear_b);
        const bool depth_overlap = scene_linear >= segment_near - thickness && scene_linear <= segment_far + thickness;

        if (depth_overlap && !all(cell == origin_pixel)) {
            result.type = TRACE_SCREEN;
            result.object_id = TraceObjectId.Load(int3(cell, 0));
            result.hit_pixel = cell;
            result.distance = 0.5 * (distance + exit_distance);
            result.confidence = 1.0;
            return result;
        }

        const float next_distance = exit_distance + max(1.0e-4, exit_distance * 1.0e-5);
        float2 next_uv;
        float next_depth;
        if (!ProjectTracePoint(clip_origin, clip_direction, next_distance, next_uv, next_depth)) return result;
        level = AscendTraceLevel(uv, next_uv, 0u, maximum_level);
        distance = next_distance;
    }

    return result;
}

TraceRay BuildReflectionRay(uint2 pixel) {
    TraceRay ray = (TraceRay)0;
    const float depth = TraceDepth.Load(int3(pixel, 0));
    if (depth <= 0.0) return ray;

    const float3 position = ReconstructWorldPosition(pixel, depth);
    float3 normal = normalize(TraceNormalRoughness.Load(int3(pixel, 0)).xyz);
    const float3 view_direction = normalize(position - TraceFrame.camera_position.xyz);
    if (dot(normal, view_direction) > 0.0) normal = -normal;
    const float3 direction = normalize(reflect(view_direction, normal));
    const float3 origin = position + normal * TraceFrame.trace_params.z;

    ray.origin_tmin = float4(origin, 0.0);
    ray.direction_tmax = float4(direction, TraceFrame.trace_params.x);
    ray.source_pixel = pixel;
    return ray;
}

float3 SkyRadiance(float3 direction) {
    float3 result = 0.0;
    const uint light_count = TraceFrame.trace_limits.z;
    for (uint i = 0u; i < light_count; ++i) {
        const GPULight light = Lights[i];
        if ((uint)(light.direction_type.w + 0.5) != 3u) continue;
        const float t = saturate(direction.y * 0.5 + 0.5);
        result += light.color_intensity.rgb * light.color_intensity.w * lerp(0.35, 1.0, t);
    }
    return result;
}

[numthreads(8, 8, 1)]
void CS_DirectRadiance(uint3 dispatch_id : SV_DispatchThreadID) {
    const uint2 pixel = dispatch_id.xy;
    const uint2 resolution = uint2(TraceFrame.resolution.xy);
    if (any(pixel >= resolution)) return;

    const float depth = TraceDepth.Load(int3(pixel, 0));
    if (depth <= 0.0) {
        const float2 uv = (float2(pixel) + 0.5) * TraceFrame.resolution.zw;
        const float2 ndc = ScreenUVToNDC(uv);
        const float4 far_world4 = mul(float4(ndc, 0.0, 1.0), TraceFrame.inverse_view_projection);
        const float3 far_world = far_world4.xyz / far_world4.w;
        DirectRadianceOutput[pixel] = float4(SkyRadiance(normalize(far_world - TraceFrame.camera_position.xyz)), 1.0);
        return;
    }

    const float3 position = ReconstructWorldPosition(pixel, depth);
    float3 normal = normalize(TraceNormalRoughness.Load(int3(pixel, 0)).xyz);
    const float3 view_direction = normalize(position - TraceFrame.camera_position.xyz);
    if (dot(normal, view_direction) > 0.0) normal = -normal;

    const float3 albedo = TraceAlbedoMetallic.Load(int3(pixel, 0)).rgb;
    float3 radiance = TraceEmissive.Load(int3(pixel, 0)).rgb;
    const uint light_count = TraceFrame.trace_limits.z;

    for (uint i = 0u; i < light_count; ++i) {
        const GPULight light = Lights[i];
        const uint type = (uint)(light.direction_type.w + 0.5);
        const float3 color = light.color_intensity.rgb;
        const float intensity = light.color_intensity.w;

        if (type == 3u) {
            radiance += albedo * color * intensity * (0.2 + 0.8 * saturate(normal.y));
            continue;
        }

        float3 L = 0.0;
        float attenuation = 1.0;

        if (type == 0u) {
            L = normalize(-light.direction_type.xyz);
        } else {
            const float3 to_light = light.position_range.xyz - position;
            const float distance_to_light = length(to_light);
            if (distance_to_light <= 1.0e-5 || distance_to_light >= light.position_range.w) continue;
            L = to_light / distance_to_light;
            const float range_term = saturate(1.0 - distance_to_light / max(light.position_range.w, 1.0e-3));
            attenuation = range_term * range_term / max(1.0, distance_to_light * distance_to_light);

            if (type == 2u) {
                const float cone = dot(normalize(light.direction_type.xyz), -L);
                const float cone_term = saturate((cone - light.spot_angles.y) / max(light.spot_angles.x - light.spot_angles.y, 1.0e-4));
                attenuation *= cone_term * cone_term;
            }
        }

        radiance += albedo * color * (intensity * attenuation * saturate(dot(normal, L)));
    }

    DirectRadianceOutput[pixel] = float4(radiance, 1.0);
}

[numthreads(8, 8, 1)]
void CS_ScreenTrace(uint3 dispatch_id : SV_DispatchThreadID) {
    const uint2 pixel = dispatch_id.xy;
    const uint2 resolution = uint2(TraceFrame.resolution.xy);
    if (any(pixel >= resolution)) return;

    const uint index = pixel.x + resolution.x * pixel.y;
    if (TraceDepth.Load(int3(pixel, 0)) <= 0.0) {
        ScreenTraceHits[index] = MakeTraceHit(TRACE_INACTIVE, 0.0);
        ScreenTraceOutput[pixel] = float4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    const TraceRay ray = BuildReflectionRay(pixel);
    const TraceHit hit = TraceScreenRay(
        ray.origin_tmin.xyz,
        ray.direction_tmax.xyz,
        pixel,
        TraceFrame.trace_params.x,
        TraceFrame.trace_params.y,
        TraceFrame.trace_params.w,
        TraceFrame.trace_limits.x,
        TraceFrame.trace_limits.y
    );

    ScreenTraceHits[index] = hit;
    if (hit.type == TRACE_SCREEN) {
        ScreenTraceOutput[pixel] = float4(TraceDirectRadiance.Load(int3(hit.hit_pixel, 0)).rgb, 1.0);
    } else {
        ScreenTraceOutput[pixel] = float4(1.0, 0.0, 1.0, 1.0);
    }
}

[numthreads(1, 1, 1)]
void CS_ResetTraceQueue(uint3 dispatch_id : SV_DispatchThreadID) {
    MissCount[0] = 0u;
    TraceDispatchArgs[0] = 0u;
    TraceDispatchArgs[1] = 1u;
    TraceDispatchArgs[2] = 1u;
}

[numthreads(8, 8, 1)]
void CS_CompactTraceMisses(uint3 dispatch_id : SV_DispatchThreadID) {
    const uint2 pixel = dispatch_id.xy;
    const uint2 resolution = uint2(TraceFrame.resolution.xy);
    if (any(pixel >= resolution)) return;

    const uint index = pixel.x + resolution.x * pixel.y;
    if (ScreenTraceHits[index].type != TRACE_MISS) return;

    uint queue_index;
    InterlockedAdd(MissCount[0], 1u, queue_index);
    MissQueue[queue_index] = BuildReflectionRay(pixel);
}

[numthreads(1, 1, 1)]
void CS_BuildTraceDispatchArgs(uint3 dispatch_id : SV_DispatchThreadID) {
    TraceDispatchArgs[0] = (MissCount[0] + 63u) / 64u;
    TraceDispatchArgs[1] = 1u;
    TraceDispatchArgs[2] = 1u;
}

float3 TransformPoint(float3 p, row_major float4x4 matrix) {
    return mul(float4(p, 1.0), matrix).xyz;
}

float3 TransformVector(float3 v, row_major float4x4 matrix) {
    return mul(float4(v, 0.0), matrix).xyz;
}

bool IntersectAABB(float3 origin, float3 direction, float3 bounds_min, float3 bounds_max, out float t_min, out float t_max) {
    const float3 safe_direction = float3(
        abs(direction.x) < 1.0e-8 ? (direction.x < 0.0 ? -1.0e-8 : 1.0e-8) : direction.x,
        abs(direction.y) < 1.0e-8 ? (direction.y < 0.0 ? -1.0e-8 : 1.0e-8) : direction.y,
        abs(direction.z) < 1.0e-8 ? (direction.z < 0.0 ? -1.0e-8 : 1.0e-8) : direction.z
    );
    const float3 a = (bounds_min - origin) / safe_direction;
    const float3 b = (bounds_max - origin) / safe_direction;
    const float3 lo = min(a, b);
    const float3 hi = max(a, b);
    t_min = max(lo.x, max(lo.y, lo.z));
    t_max = min(hi.x, min(hi.y, hi.z));
    return t_max >= max(t_min, 0.0);
}

float SampleSDF(GPUSDFModel model, float3 local_position) {
    const float3 bounds_min = model.bounds_min.xyz;
    const float3 bounds_max = model.bounds_max.xyz;
    const float3 uvw = (local_position - bounds_min) / max(bounds_max - bounds_min, 1.0e-6.xxx);
    if (any(uvw < 0.0) || any(uvw > 1.0)) return 1.0e6;

    const uint resolution = model.meta.y;
    const float3 grid = saturate(uvw) * (float)(resolution - 1u);
    const uint3 p0 = min(uint3(floor(grid)), resolution - 1u);
    const uint3 p1 = min(p0 + 1u, resolution - 1u);
    const float3 f = frac(grid);
    const uint offset = model.meta.x;

    uint Index(uint3 p) {
        return offset + p.x + resolution * (p.y + resolution * p.z);
    }

    const float c000 = SDFVoxels[Index(uint3(p0.x, p0.y, p0.z))];
    const float c100 = SDFVoxels[Index(uint3(p1.x, p0.y, p0.z))];
    const float c010 = SDFVoxels[Index(uint3(p0.x, p1.y, p0.z))];
    const float c110 = SDFVoxels[Index(uint3(p1.x, p1.y, p0.z))];
    const float c001 = SDFVoxels[Index(uint3(p0.x, p0.y, p1.z))];
    const float c101 = SDFVoxels[Index(uint3(p1.x, p0.y, p1.z))];
    const float c011 = SDFVoxels[Index(uint3(p0.x, p1.y, p1.z))];
    const float c111 = SDFVoxels[Index(uint3(p1.x, p1.y, p1.z))];

    const float c00 = lerp(c000, c100, f.x);
    const float c10 = lerp(c010, c110, f.x);
    const float c01 = lerp(c001, c101, f.x);
    const float c11 = lerp(c011, c111, f.x);
    return lerp(lerp(c00, c10, f.y), lerp(c01, c11, f.y), f.z);
}

bool TraceSDFModel(TraceRay ray, GPUSDFModel model, float current_best, out float hit_distance) {
    const float3 local_origin = TransformPoint(ray.origin_tmin.xyz, model.world_to_local);
    const float3 local_direction = TransformVector(ray.direction_tmax.xyz, model.world_to_local);
    const float direction_scale = length(local_direction);
    if (direction_scale <= 1.0e-8) return false;

    float t_min, t_max;
    if (!IntersectAABB(local_origin, local_direction, model.bounds_min.xyz, model.bounds_max.xyz, t_min, t_max)) return false;

    float t = max(max(t_min, ray.origin_tmin.w), 0.0);
    const float limit = min(min(t_max, ray.direction_tmax.w), current_best);
    const float3 extent = model.bounds_max.xyz - model.bounds_min.xyz;
    const float cell_size = max(extent.x, max(extent.y, extent.z)) / max((float)model.meta.y, 1.0);
    const float epsilon = max(cell_size * 0.65, 1.0e-4);

    [loop]
    for (uint step = 0u; step < 96u && t <= limit; ++step) {
        const float3 local_position = local_origin + local_direction * t;
        const float distance = SampleSDF(model, local_position);
        if (distance <= epsilon) {
            hit_distance = t;
            return true;
        }
        t += max(distance / direction_scale, epsilon * 0.25 / direction_scale);
    }

    return false;
}

uint SurfaceHash(uint object_id, float3 position) {
    const int3 cell = int3(floor(position * 4.0));
    uint h = object_id * 747796405u + 2891336453u;
    h ^= asuint(cell.x) * 277803737u;
    h ^= asuint(cell.y) * 1597334677u;
    h ^= asuint(cell.z) * 3812015801u;
    h ^= h >> 16u;
    return h | 1u;
}

bool LookupSurfaceCache(uint object_id, float3 position, out float3 radiance) {
    const uint key = SurfaceHash(object_id, position);
    const uint slot = key & (SURFACE_CACHE_CAPACITY - 1u);
    if (SurfaceCacheKeys[slot] != key) return false;
    const SurfaceCacheEntry entry = SurfaceCacheEntries[slot];
    if (entry.meta.x != object_id || distance(entry.position.xyz, position) > 0.75) return false;
    radiance = entry.radiance.rgb;
    return true;
}

[numthreads(64, 1, 1)]
void CS_SDFTrace(uint3 dispatch_id : SV_DispatchThreadID) {
    const uint queue_index = dispatch_id.x;
    const uint count = MissCount[0];
    if (queue_index >= count) return;

    const TraceRay ray = MissQueue[queue_index];
    float best = ray.direction_tmax.w;
    uint best_object = 0u;
    const uint model_count = TraceFrame.trace_limits.w;

    for (uint i = 0u; i < model_count; ++i) {
        float hit_distance;
        if (TraceSDFModel(ray, SDFModels[i], best, hit_distance)) {
            best = hit_distance;
            best_object = SDFModels[i].meta.z;
        }
    }

    if (best_object == 0u) return;

    const uint2 resolution = uint2(TraceFrame.resolution.xy);
    const uint hit_index = ray.source_pixel.x + resolution.x * ray.source_pixel.y;
    TraceHit hit = MakeTraceHit(TRACE_SDF, best);
    hit.object_id = best_object;
    hit.confidence = 1.0;
    ScreenTraceHits[hit_index] = hit;

    const float3 world_position = ray.origin_tmin.xyz + ray.direction_tmax.xyz * best;
    float3 cached_radiance;
    if (LookupSurfaceCache(best_object, world_position, cached_radiance)) {
        ScreenTraceOutput[ray.source_pixel] = float4(cached_radiance, 1.0);
    } else {
        ScreenTraceOutput[ray.source_pixel] = float4(0.0, 1.0, 1.0, 1.0);
    }
}

[numthreads(8, 8, 1)]
void CS_SurfaceCacheUpdate(uint3 dispatch_id : SV_DispatchThreadID) {
    const uint2 pixel = dispatch_id.xy;
    const uint2 resolution = uint2(TraceFrame.resolution.xy);
    if (any(pixel >= resolution)) return;

    const float depth = TraceDepth.Load(int3(pixel, 0));
    if (depth <= 0.0) return;

    const uint object_id = TraceObjectId.Load(int3(pixel, 0));
    if (object_id == 0u) return;

    const float3 position = ReconstructWorldPosition(pixel, depth);
    const uint key = SurfaceHash(object_id, position);
    const uint slot = key & (SURFACE_CACHE_CAPACITY - 1u);
    uint previous;
    InterlockedExchange(SurfaceCacheKeys[slot], key, previous);

    SurfaceCacheEntry entry;
    entry.position = float4(position, 1.0);
    entry.radiance = TraceDirectRadiance.Load(int3(pixel, 0));
    entry.meta = uint4(object_id, 0u, 0u, 0u);
    SurfaceCacheEntries[slot] = entry;
}

[numthreads(8, 8, 1)]
void CS_ScreenProbes(uint3 dispatch_id : SV_DispatchThreadID) {
    uint probe_width, probe_height;
    ScreenProbesOutput.GetDimensions(probe_width, probe_height);
    const uint2 probe = dispatch_id.xy;
    if (probe.x >= probe_width || probe.y >= probe_height) return;

    const uint2 resolution = uint2(TraceFrame.resolution.xy);
    const uint2 begin = probe * 8u;
    const uint2 end = min(begin + 8u, resolution);
    const uint2 samples[4] = {
        min(begin + uint2(2u, 2u), end - 1u),
        min(begin + uint2(5u, 2u), end - 1u),
        min(begin + uint2(2u, 5u), end - 1u),
        min(begin + uint2(5u, 5u), end - 1u)
    };

    float3 radiance = 0.0;
    [unroll]
    for (uint i = 0u; i < 4u; ++i) {
        radiance += TraceDirectRadiance.Load(int3(samples[i], 0)).rgb;
    }
    ScreenProbesOutput[probe] = float4(radiance * 0.25, 1.0);
}

[numthreads(8, 8, 1)]
void CS_HZB(uint3 dispatch_id : SV_DispatchThreadID) {
    uint src_width, src_height, dst_width, dst_height;
    HZBSource.GetDimensions(src_width, src_height);
    HZBOutput.GetDimensions(dst_width, dst_height);
    const uint2 pixel = dispatch_id.xy;
    if (pixel.x >= dst_width || pixel.y >= dst_height) return;

    const uint2 src_size = uint2(src_width, src_height);
    const uint2 dst_size = uint2(dst_width, dst_height);
    const uint2 src_begin = (pixel * src_size) / dst_size;
    const uint2 src_end = ((pixel + 1u) * src_size) / dst_size;

    float depth = 0.0;
    for (uint y = src_begin.y; y < src_end.y; ++y) {
        for (uint x = src_begin.x; x < src_end.x; ++x) {
            depth = max(depth, HZBSource.Load(int3(int2(x, y), 0)));
        }
    }
    HZBOutput[pixel] = depth;
}

float4 PS_Present(PresentVSOutput input) : SV_Target0 {
    const int2 pixel = int2(input.position.xy);

#ifdef ALBEDO
    return float4(AlbedoMetallicTexture.Load(int3(pixel, 0)).rgb, 1.0);
#elif defined(NORMALS)
    return float4(NormalRoughnessTexture.Load(int3(pixel, 0)).rgb * 0.5 + 0.5, 1.0);
#elif defined(DEPTH)
    const float linear_depth = LinearizeDepth(DepthTexture.Load(int3(pixel, 0)));
    return float4((1.0 - saturate(linear_depth / 200.0)).xxx, 1.0);
#elif defined(ROUGHNESS)
    return float4(NormalRoughnessTexture.Load(int3(pixel, 0)).w.xxx, 1.0);
#elif defined(VELOCITY)
    const float2 velocity = VelocityTexture.Load(int3(pixel, 0));
    return float4(saturate(0.5 + velocity.x * 20.0), saturate(0.5 + velocity.y * 20.0), 0.5, 1.0);
#elif defined(OBJECT_ID)
    const float value = frac((float)ObjectIdTexture.Load(int3(pixel, 0)) * 0.61803398875);
    return float4(value, frac(value * 3.17), frac(value * 7.13), 1.0);
#elif defined(HZB)
    uint hzb_width, hzb_height, hzb_mip_count;
    HZBTexture.GetDimensions(0, hzb_width, hzb_height, hzb_mip_count);
    const uint mip = min(4u, hzb_mip_count);
    float hzb_depth;
    if (mip == 0u) {
        hzb_depth = DepthTexture.Load(int3(pixel, 0));
    } else {
        const uint hzb_mip = mip - 1u;
        const uint mip_width = max(1u, hzb_width >> hzb_mip);
        const uint mip_height = max(1u, hzb_height >> hzb_mip);
        const uint2 hzb_pixel = min(uint2(pixel) >> mip, uint2(mip_width - 1u, mip_height - 1u));
        hzb_depth = HZBTexture.Load(int3(int2(hzb_pixel), hzb_mip));
    }
    const float linear_depth = LinearizeDepth(hzb_depth);
    return float4((1.0 - saturate(linear_depth / 200.0)).xxx, 1.0);
#elif defined(SCREEN_TRACE)
    return float4(ScreenTraceTexture.Load(int3(pixel, 0)).rgb, 1.0);
#elif defined(DIRECT_RADIANCE)
    return float4(DirectRadianceTexture.Load(int3(pixel, 0)).rgb, 1.0);
#elif defined(SCREEN_PROBES)
    uint full_width, full_height;
    DepthTexture.GetDimensions(full_width, full_height);
    uint probe_width, probe_height;
    ScreenProbesTexture.GetDimensions(probe_width, probe_height);
    const uint2 probe = min(uint2((uint)pixel.x * probe_width / full_width, (uint)pixel.y * probe_height / full_height), uint2(probe_width - 1u, probe_height - 1u));
    return float4(ScreenProbesTexture.Load(int3(probe, 0)).rgb, 1.0);
#else
    return float4(1.0, 0.0, 1.0, 1.0);
#endif
}
