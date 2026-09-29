#define FAR_PLANE 10000.0f
#define NEAR_PLANE 0.05f
#define TRACE_INACTIVE 0u
#define TRACE_MISS 1u
#define TRACE_SCREEN 2u
#define TRACE_SDF 3u
#define SURFACE_CACHE_CAPACITY 262144u

#define TRACE_RAY_INACTIVE 0u
#define TRACE_RAY_DIFFUSE 1u
#define TRACE_RAY_REFLECTION 2u
#define TRACE_RAY_SHADOW 3u

#define SCREEN_PROBE_TILE_SIZE 8u
#define SCREEN_PROBE_DIRECTION_SIZE 4u
#define SCREEN_PROBE_RAY_COUNT 16u

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

    uint type;
    uint destination;
    uint origin_pixel;
    uint source_object_id;
};

struct GPUSDFModel {
    row_major float4x4 world_to_local;
    float4 bounds_min;
    float4 bounds_max;

    // x = voxel offset
    // y = resolution
    // z = object id
    // w = STATIC / DYNAMIC state
    uint4 meta;

    // x = object revision
    uint4 version;
};

struct GPULight {
    float4 position_range;
    float4 direction_type;
    float4 color_intensity;
    float4 spot_angles;
};

struct SurfaceCacheEntry {
    float4 position;
    float4 normal;

    float4 albedo_roughness;
    float4 emissive_metallic;

    float4 direct_radiance;
    float4 indirect_radiance;

    uint object_id;
    uint revision;
    uint last_frame;
    uint confidence;
};

[[vk::binding(0, 0)]] StructuredBuffer<GPUObject> Objects : register(t0, space0);
[[vk::binding(1, 0)]] StructuredBuffer<GPUMaterial> Materials : register(t1, space0);
[[vk::binding(2, 0)]] ConstantBuffer<FrameConstants> Frame : register(b0, space0);

struct DrawConstants { uint object_index; };
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
    GBufferVSOutput output;
    output.position = mul(world_position, Frame.view_projection);
    output.current_clip = output.position;
    output.previous_clip = mul(mul(local_position, object.previous_world), Frame.previous_view_projection);
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
    float2 current_ndc = input.current_clip.xy / input.current_clip.w;
    float2 previous_ndc = input.previous_clip.xy / input.previous_clip.w;
    output.velocity = (current_ndc - previous_ndc) * float2(0.5, -0.5);
    output.object_id = input.object_id;
    return output;
}

struct PresentVSOutput { float4 position : SV_Position; };

PresentVSOutput VS_Present(uint vertex_id : SV_VertexID) {
    const float2 positions[3] = {float2(-1.0, -1.0), float2(-1.0, 3.0), float2(3.0, -1.0)};
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
[[vk::binding(19, 3)]] RWTexture2D<float4> ProbeRadianceOutput : register(u8, space3);
[[vk::binding(20, 3)]] RWTexture2D<float4> ScreenProbesOutput : register(u9, space3);

float2 ScreenUVToNDC(float2 uv) { return float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0); }
float2 NDCToScreenUV(float2 ndc) { return float2(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5); }

float3 ReconstructWorldPosition(uint2 pixel, float depth) {
    float2 uv = (float2(pixel) + 0.5) * TraceFrame.resolution.zw;
    float4 world = mul(float4(ScreenUVToNDC(uv), depth, 1.0), TraceFrame.inverse_view_projection);
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
    uint2 dimensions = TraceLevelDimensions(level);
    cell = min(uint2(uv * dimensions), dimensions - uint2(1u, 1u));
    return level == 0u ? TraceDepth.Load(int3(cell, 0)) : TraceHZB.Load(int3(cell, level - 1u));
}

bool ProjectTracePoint(float4 clip_origin, float4 clip_direction, float distance, out float2 uv, out float depth) {
    float4 clip = clip_origin + clip_direction * distance;
    if (clip.w <= 1.0e-5) {
        uv = 0.0;
        depth = 0.0;
        return false;
    }
    float3 ndc = clip.xyz / clip.w;
    uv = NDCToScreenUV(ndc.xy);
    depth = ndc.z;
    return all(uv >= 0.0) && all(uv < 1.0) && depth >= 0.0 && depth <= 1.0;
}

float SolveTraceDistance(float value_origin, float value_direction, float w_origin, float w_direction, float target) {
    float denominator = value_direction - target * w_direction;
    return abs(denominator) < 1.0e-7 ? 1.0e30 : (target * w_origin - value_origin) / denominator;
}

float TraceCellExit(float4 clip_origin, float4 clip_direction, uint2 dimensions, uint2 cell, float distance, float2 screen_direction) {
    float exit_x = 1.0e30;
    float exit_y = 1.0e30;
    float4 current_clip = clip_origin + clip_direction * distance;
    if (abs(screen_direction.x) > 1.0e-7) {
        float boundary = screen_direction.x > 0.0 ? (float)(cell.x + 1u) / (float)dimensions.x : (float)cell.x / (float)dimensions.x;
        exit_x = distance + SolveTraceDistance(current_clip.x, clip_direction.x, current_clip.w, clip_direction.w, boundary * 2.0 - 1.0);
        if (exit_x <= distance + 1.0e-5) exit_x = 1.0e30;
    }
    if (abs(screen_direction.y) > 1.0e-7) {
        float boundary = screen_direction.y > 0.0 ? (float)(cell.y + 1u) / (float)dimensions.y : (float)cell.y / (float)dimensions.y;
        exit_y = distance + SolveTraceDistance(current_clip.y, clip_direction.y, current_clip.w, clip_direction.w, 1.0 - boundary * 2.0);
        if (exit_y <= distance + 1.0e-5) exit_y = 1.0e30;
    }
    return min(exit_x, exit_y);
}

uint AscendTraceLevel(float2 previous_uv, float2 next_uv, uint level, uint maximum_level) {
    while (level < maximum_level) {
        uint parent_level = level + 1u;
        uint2 dimensions = TraceLevelDimensions(parent_level);
        uint2 limit = dimensions - uint2(1u, 1u);
        uint2 before_cell = min(uint2(previous_uv * dimensions), limit);
        uint2 after_cell = min(uint2(next_uv * dimensions), limit);
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
    hit.padding = uint2(0u, 0u);
    return hit;
}

TraceHit TraceScreenRay(float3 origin, float3 direction, uint2 origin_pixel, float max_distance, float thickness, float skip_thickness, uint max_steps, uint max_start_level) {
    TraceHit result = MakeTraceHit(TRACE_MISS, max_distance);
    float4 clip_origin = mul(float4(origin, 1.0), TraceFrame.view_projection);
    float4 clip_direction = mul(float4(direction, 0.0), TraceFrame.view_projection);
    if (clip_origin.w <= 1.0e-5) return result;

    uint hzb_width, hzb_height, hzb_mip_count;
    TraceHZB.GetDimensions(0, hzb_width, hzb_height, hzb_mip_count);
    uint maximum_level = min(hzb_mip_count, max_start_level);
    uint level = maximum_level;
    float2 screen_direction = float2(
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
        float scene_depth = TraceLevelDepth(uv, level, cell);
        uint2 dimensions = TraceLevelDimensions(level);
        float exit_distance = TraceCellExit(clip_origin, clip_direction, dimensions, cell, distance, screen_direction);
        if (exit_distance >= 1.0e29) return result;
        exit_distance = min(exit_distance, max_distance);

        float2 exit_uv;
        float exit_depth;
        if (!ProjectTracePoint(clip_origin, clip_direction, exit_distance, exit_uv, exit_depth)) return result;

        float scene_linear = LinearizeDepth(scene_depth);
        float farther_ray_linear = LinearizeDepth(min(ray_depth, exit_depth));
        bool clear_cell = scene_depth <= 0.0 || farther_ray_linear < scene_linear - skip_thickness;

        if (clear_cell) {
            float next_distance = exit_distance + max(1.0e-4, exit_distance * 1.0e-5);
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

        float ray_linear_a = LinearizeDepth(ray_depth);
        float ray_linear_b = LinearizeDepth(exit_depth);
        float segment_near = min(ray_linear_a, ray_linear_b);
        float segment_far = max(ray_linear_a, ray_linear_b);
        bool overlap = scene_linear >= segment_near - thickness && scene_linear <= segment_far + thickness;

        if (overlap && !all(cell == origin_pixel)) {
            result.type = TRACE_SCREEN;
            result.object_id = TraceObjectId.Load(int3(cell, 0));
            result.hit_pixel = cell;
            result.distance = 0.5 * (distance + exit_distance);
            result.confidence = 1.0;
            return result;
        }

        float next_distance = exit_distance + max(1.0e-4, exit_distance * 1.0e-5);
        float2 next_uv;
        float next_depth;
        if (!ProjectTracePoint(clip_origin, clip_direction, next_distance, next_uv, next_depth)) return result;
        level = AscendTraceLevel(uv, next_uv, 0u, maximum_level);
        distance = next_distance;
    }

    return result;
}

uint PackPixel(uint2 pixel) {
    return (pixel.x & 0xffffu) |
           ((pixel.y & 0xffffu) << 16u);
}

uint2 UnpackPixel(uint packed) {
    return uint2(
        packed & 0xffffu,
        (packed >> 16u) & 0xffffu
    );
}

TraceRay MakeTraceRay(
    float3 origin,
    float3 direction,
    float tmin,
    float tmax,
    uint type,
    uint destination,
    uint2 origin_pixel,
    uint source_object_id
) {
    TraceRay ray;

    ray.origin_tmin = float4(origin, tmin);
    ray.direction_tmax = float4(normalize(direction), tmax);

    ray.type = type;
    ray.destination = destination;
    ray.origin_pixel = PackPixel(origin_pixel);
    ray.source_object_id = source_object_id;

    return ray;
}

float RadicalInverseVdC(uint bits) {
    bits = (bits << 16u) | (bits >> 16u);
    bits =
        ((bits & 0x55555555u) << 1u) |
        ((bits & 0xaaaaaaaau) >> 1u);
    bits =
        ((bits & 0x33333333u) << 2u) |
        ((bits & 0xccccccccu) >> 2u);
    bits =
        ((bits & 0x0f0f0f0fu) << 4u) |
        ((bits & 0xf0f0f0f0u) >> 4u);
    bits =
        ((bits & 0x00ff00ffu) << 8u) |
        ((bits & 0xff00ff00u) >> 8u);

    return (float)bits * 2.3283064365386963e-10;
}

uint ProbeHash(uint2 probe) {
    uint h =
        probe.x * 0x8da6b343u ^
        probe.y * 0xd8163841u;

    h ^= h >> 16u;
    h *= 0x7feb352du;
    h ^= h >> 15u;

    return h;
}

float3 CosineHemisphereDirection(
    uint ray_index,
    uint2 probe
) {
    float u1 =
        ((float)ray_index + 0.5) /
        (float)SCREEN_PROBE_RAY_COUNT;

    float u2 = RadicalInverseVdC(ray_index);

    float rotation =
        ((float)(ProbeHash(probe) & 0x00ffffffu) /
         16777216.0) *
        6.28318530718;

    float radius = sqrt(u1);
    float phi = u2 * 6.28318530718 + rotation;

    return float3(
        radius * cos(phi),
        radius * sin(phi),
        sqrt(max(0.0, 1.0 - u1))
    );
}

void BuildNormalBasis(
    float3 normal,
    out float3 tangent,
    out float3 bitangent
) {
    float3 axis =
        abs(normal.z) < 0.999
            ? float3(0.0, 0.0, 1.0)
            : float3(0.0, 1.0, 0.0);

    tangent = normalize(cross(axis, normal));
    bitangent = cross(normal, tangent);
}

TraceRay BuildDiffuseProbeRay(uint2 ray_pixel) {
    uint ray_width;
    uint ray_height;

    ProbeRadianceOutput.GetDimensions(
        ray_width,
        ray_height
    );

    uint2 probe =
        ray_pixel / SCREEN_PROBE_DIRECTION_SIZE;

    uint2 direction_cell =
        ray_pixel % SCREEN_PROBE_DIRECTION_SIZE;

    uint ray_index =
        direction_cell.x +
        direction_cell.y *
        SCREEN_PROBE_DIRECTION_SIZE;

    uint2 resolution =
        uint2(TraceFrame.resolution.xy);

    /*
     * Representative surface = center of the 8x8 tile.
     *
     * Do not add complicated placement/reprojection in this stage.
     */
    uint2 source_pixel = min(
        probe * SCREEN_PROBE_TILE_SIZE +
            uint2(
                SCREEN_PROBE_TILE_SIZE / 2u,
                SCREEN_PROBE_TILE_SIZE / 2u
            ),
        resolution - uint2(1u, 1u)
    );

    uint destination =
        ray_pixel.x +
        ray_pixel.y * ray_width;

    float depth =
        TraceDepth.Load(int3(source_pixel, 0));

    if (depth <= 0.0) {
        return MakeTraceRay(
            0.0,
            float3(0.0, 1.0, 0.0),
            0.0,
            0.0,
            TRACE_RAY_INACTIVE,
            destination,
            source_pixel,
            0u
        );
    }

    float3 position =
        ReconstructWorldPosition(
            source_pixel,
            depth
        );

    float3 normal = normalize(
        TraceNormalRoughness.Load(
            int3(source_pixel, 0)
        ).xyz
    );

    float3 view_direction = normalize(
        position - TraceFrame.camera_position.xyz
    );

    if (dot(normal, view_direction) > 0.0)
        normal = -normal;

    float3 tangent;
    float3 bitangent;

    BuildNormalBasis(
        normal,
        tangent,
        bitangent
    );

    float3 local_direction =
        CosineHemisphereDirection(
            ray_index,
            probe
        );

    float3 direction = normalize(
        tangent * local_direction.x +
        bitangent * local_direction.y +
        normal * local_direction.z
    );

    uint object_id =
        TraceObjectId.Load(
            int3(source_pixel, 0)
        );

    float bias = TraceFrame.trace_params.z;

    return MakeTraceRay(
        position + normal * bias,
        direction,
        0.0,
        TraceFrame.trace_params.x,
        TRACE_RAY_DIFFUSE,
        destination,
        source_pixel,
        object_id
    );
}

float3 SkyRadiance(float3 direction) {
    float3 result = float3(0.0, 0.0, 0.0);
    for (uint i = 0u; i < TraceFrame.trace_limits.z; ++i) {
        GPULight light = Lights[i];
        if ((uint)(light.direction_type.w + 0.5) != 3u) continue;
        float t = saturate(direction.y * 0.5 + 0.5);
        result += lerp(light.spot_angles.xyz, light.color_intensity.rgb, t) * light.color_intensity.w;
    }
    return result;
}

bool TraceSDFAny(
    TraceRay ray,
    out float hit_distance
);

float DirectVisibility(
    uint2 source_pixel,
    uint source_object_id,
    float3 position,
    float3 normal,
    float3 direction,
    float max_distance
) {
    float bias =
        max(TraceFrame.trace_params.z, 1.0e-3);

    float tmax =
        max(max_distance - bias * 2.0, 0.0);

    if (tmax <= 0.0)
        return 1.0;

    float3 origin =
        position + normal * bias * 2.0;

    TraceHit screen_hit =
        TraceScreenRay(
            origin,
            direction,
            source_pixel,
            tmax,
            TraceFrame.trace_params.y,
            TraceFrame.trace_params.w,
            TraceFrame.trace_limits.x,
            TraceFrame.trace_limits.y
        );

    if (screen_hit.type == TRACE_SCREEN)
        return 0.0;

    TraceRay shadow_ray = MakeTraceRay(
        origin,
        direction,
        0.0,
        tmax,
        TRACE_RAY_SHADOW,
        0u,
        source_pixel,
        source_object_id
    );

    float sdf_hit_distance;

    if (TraceSDFAny(
            shadow_ray,
            sdf_hit_distance
        )) {
        return 0.0;
    }

    return 1.0;
}

[numthreads(8, 8, 1)]
void CS_DirectRadiance(uint3 dispatch_id : SV_DispatchThreadID) {
    uint2 pixel = dispatch_id.xy;
    uint2 resolution = uint2(TraceFrame.resolution.xy);
    if (any(pixel >= resolution)) return;
    float depth = TraceDepth.Load(int3(pixel, 0));

    if (depth <= 0.0) {
        float2 uv = (float2(pixel) + 0.5) * TraceFrame.resolution.zw;
        float4 far_world4 = mul(float4(ScreenUVToNDC(uv), 0.0, 1.0), TraceFrame.inverse_view_projection);
        float3 direction = normalize(far_world4.xyz / far_world4.w - TraceFrame.camera_position.xyz);
        DirectRadianceOutput[pixel] = float4(SkyRadiance(direction), 1.0);
        return;
    }

    float3 position = ReconstructWorldPosition(pixel, depth);
    float3 normal = normalize(TraceNormalRoughness.Load(int3(pixel, 0)).xyz);
    float3 view_direction = normalize(position - TraceFrame.camera_position.xyz);
    if (dot(normal, view_direction) > 0.0) normal = -normal;
    float3 albedo = TraceAlbedoMetallic.Load(int3(pixel, 0)).rgb;
    float3 radiance = TraceEmissive.Load(int3(pixel, 0)).rgb;

    for (
        uint i = 0u;
        i < TraceFrame.trace_limits.z;
        ++i
    ) {
        GPULight light = Lights[i];

        uint type =
            (uint)(light.direction_type.w + 0.5);

        /*
         * Sky lighting is now supplied naturally by
         * diffuse probe rays which miss the scene.
         *
         * Do not inject fake unshadowed sky ambient here.
         */
        if (type == 3u)
            continue;

        float3 L;
        float attenuation = 1.0;
        float light_distance =
            TraceFrame.trace_params.x;

        if (type == 0u) {
            L = normalize(
                -light.direction_type.xyz
            );
        } else {
            float3 to_light =
                light.position_range.xyz -
                position;

            float d = length(to_light);

            if (d <= 1.0e-5 ||
                d >= light.position_range.w) {
                continue;
            }

            L = to_light / d;
            light_distance = d;

            float range_term = saturate(
                1.0 -
                d /
                max(light.position_range.w, 1.0e-3)
            );

            attenuation =
                range_term *
                range_term /
                max(1.0, d * d);

            if (type == 2u) {
                float cone = dot(
                    normalize(
                        light.direction_type.xyz
                    ),
                    -L
                );

                float cone_term = saturate(
                    (cone - light.spot_angles.y) /
                    max(
                        light.spot_angles.x -
                            light.spot_angles.y,
                        1.0e-4
                    )
                );

                attenuation *=
                    cone_term * cone_term;
            }
        }

        float ndotl =
            saturate(dot(normal, L));

        if (ndotl <= 0.0)
            continue;

        uint object_id =
            TraceObjectId.Load(
                int3(pixel, 0)
            );

        float visibility =
            DirectVisibility(
                pixel,
                object_id,
                position,
                normal,
                L,
                light_distance
            );

        radiance +=
            albedo *
            light.color_intensity.rgb *
            light.color_intensity.w *
            attenuation *
            ndotl *
            visibility;
    }
    DirectRadianceOutput[pixel] = float4(radiance, 1.0);
}

[numthreads(8, 8, 1)]
void CS_ScreenTrace(
    uint3 dispatch_id : SV_DispatchThreadID
) {
    uint ray_width;
    uint ray_height;

    ProbeRadianceOutput.GetDimensions(
        ray_width,
        ray_height
    );

    uint2 ray_pixel = dispatch_id.xy;

    if (ray_pixel.x >= ray_width ||
        ray_pixel.y >= ray_height) {
        return;
    }

    TraceRay ray =
        BuildDiffuseProbeRay(ray_pixel);

    uint index = ray.destination;

    if (ray.type == TRACE_RAY_INACTIVE) {
        ScreenTraceHits[index] =
            MakeTraceHit(
                TRACE_INACTIVE,
                0.0
            );

        ScreenTraceOutput[ray_pixel] =
            0.0;

        ProbeRadianceOutput[ray_pixel] =
            0.0;

        return;
    }

    uint2 origin_pixel =
        UnpackPixel(ray.origin_pixel);

    TraceHit hit =
        TraceScreenRay(
            ray.origin_tmin.xyz,
            ray.direction_tmax.xyz,
            origin_pixel,
            ray.direction_tmax.w,
            TraceFrame.trace_params.y,
            TraceFrame.trace_params.w,
            TraceFrame.trace_limits.x,
            TraceFrame.trace_limits.y
        );

    ScreenTraceHits[index] = hit;

    if (hit.type == TRACE_SCREEN) {
        float3 radiance =
            TraceDirectRadiance.Load(
                int3(hit.hit_pixel, 0)
            ).rgb;

        ProbeRadianceOutput[ray_pixel] =
            float4(radiance, 1.0);

        ScreenTraceOutput[ray_pixel] =
            float4(radiance, 1.0);

        return;
    }

    /*
     * A true scene miss sees environment radiance.
     *
     * SDF tracing may overwrite this if offscreen
     * geometry is found.
     */
    float3 sky =
        SkyRadiance(
            ray.direction_tmax.xyz
        );

    ProbeRadianceOutput[ray_pixel] =
        float4(sky, 1.0);

    /*
     * Keep magenta as the screen-trace debug view.
     */
    ScreenTraceOutput[ray_pixel] =
        float4(1.0, 0.0, 1.0, 1.0);
}

[numthreads(1, 1, 1)]
void CS_ResetTraceQueue(uint3 dispatch_id : SV_DispatchThreadID) {
    MissCount[0] = 0u;
    TraceDispatchArgs[0] = 0u;
    TraceDispatchArgs[1] = 1u;
    TraceDispatchArgs[2] = 1u;
}

[numthreads(8, 8, 1)]
void CS_CompactTraceMisses(
    uint3 dispatch_id : SV_DispatchThreadID
) {
    uint ray_width;
    uint ray_height;

    ProbeRadianceOutput.GetDimensions(
        ray_width,
        ray_height
    );

    uint2 ray_pixel = dispatch_id.xy;

    if (ray_pixel.x >= ray_width ||
        ray_pixel.y >= ray_height) {
        return;
    }

    uint index =
        ray_pixel.x +
        ray_pixel.y * ray_width;

    if (ScreenTraceHits[index].type !=
        TRACE_MISS) {
        return;
    }

    TraceRay ray =
        BuildDiffuseProbeRay(ray_pixel);

    if (ray.type == TRACE_RAY_INACTIVE)
        return;

    uint queue_index;

    InterlockedAdd(
        MissCount[0],
        1u,
        queue_index
    );

    MissQueue[queue_index] = ray;
}

[numthreads(1, 1, 1)]
void CS_BuildTraceDispatchArgs(uint3 dispatch_id : SV_DispatchThreadID) {
    TraceDispatchArgs[0] = (MissCount[0] + 63u) / 64u;
    TraceDispatchArgs[1] = 1u;
    TraceDispatchArgs[2] = 1u;
}

float3 TransformPoint(float3 p, float4x4 matrix) { return mul(float4(p, 1.0), matrix).xyz; }
float3 TransformVector(float3 v, float4x4 matrix) { return mul(float4(v, 0.0), matrix).xyz; }

bool IntersectAABB(float3 origin, float3 direction, float3 bounds_min, float3 bounds_max, out float t_min, out float t_max) {
    float3 safe_direction = float3(
        abs(direction.x) < 1.0e-8 ? (direction.x < 0.0 ? -1.0e-8 : 1.0e-8) : direction.x,
        abs(direction.y) < 1.0e-8 ? (direction.y < 0.0 ? -1.0e-8 : 1.0e-8) : direction.y,
        abs(direction.z) < 1.0e-8 ? (direction.z < 0.0 ? -1.0e-8 : 1.0e-8) : direction.z
    );
    float3 a = (bounds_min - origin) / safe_direction;
    float3 b = (bounds_max - origin) / safe_direction;
    float3 lo = min(a, b);
    float3 hi = max(a, b);
    t_min = max(lo.x, max(lo.y, lo.z));
    t_max = min(hi.x, min(hi.y, hi.z));
    return t_max >= max(t_min, 0.0);
}

uint SDFIndex(GPUSDFModel model, uint3 p) {
    uint resolution = model.meta.y;
    return model.meta.x + p.x + resolution * (p.y + resolution * p.z);
}

float SampleSDF(GPUSDFModel model, float3 local_position) {
    float3 size = max(model.bounds_max.xyz - model.bounds_min.xyz, float3(1.0e-6, 1.0e-6, 1.0e-6));
    float3 uvw = (local_position - model.bounds_min.xyz) / size;
    if (any(uvw < 0.0) || any(uvw > 1.0)) return 1.0e6;
    uint resolution = model.meta.y;
    float3 grid = saturate(uvw) * (float)(resolution - 1u);
    uint3 max_coord = uint3(resolution - 1u, resolution - 1u, resolution - 1u);
    uint3 p0 = min((uint3)floor(grid), max_coord);
    uint3 p1 = min(p0 + uint3(1u, 1u, 1u), max_coord);
    float3 f = frac(grid);
    float c000 = SDFVoxels[SDFIndex(model, uint3(p0.x, p0.y, p0.z))];
    float c100 = SDFVoxels[SDFIndex(model, uint3(p1.x, p0.y, p0.z))];
    float c010 = SDFVoxels[SDFIndex(model, uint3(p0.x, p1.y, p0.z))];
    float c110 = SDFVoxels[SDFIndex(model, uint3(p1.x, p1.y, p0.z))];
    float c001 = SDFVoxels[SDFIndex(model, uint3(p0.x, p0.y, p1.z))];
    float c101 = SDFVoxels[SDFIndex(model, uint3(p1.x, p0.y, p1.z))];
    float c011 = SDFVoxels[SDFIndex(model, uint3(p0.x, p1.y, p1.z))];
    float c111 = SDFVoxels[SDFIndex(model, uint3(p1.x, p1.y, p1.z))];
    float c00 = lerp(c000, c100, f.x);
    float c10 = lerp(c010, c110, f.x);
    float c01 = lerp(c001, c101, f.x);
    float c11 = lerp(c011, c111, f.x);
    return lerp(lerp(c00, c10, f.y), lerp(c01, c11, f.y), f.z);
}

bool TraceSDFModel(TraceRay ray, GPUSDFModel model, float current_best, out float hit_distance) {
    float3 local_origin = TransformPoint(ray.origin_tmin.xyz, model.world_to_local);
    float3 local_direction = TransformVector(ray.direction_tmax.xyz, model.world_to_local);
    float direction_scale = length(local_direction);
    if (direction_scale <= 1.0e-8) {
        hit_distance = current_best;
        return false;
    }
    float t_min, t_max;
    if (!IntersectAABB(local_origin, local_direction, model.bounds_min.xyz, model.bounds_max.xyz, t_min, t_max)) {
        hit_distance = current_best;
        return false;
    }
    float t = max(max(t_min, ray.origin_tmin.w), 0.0);
    float limit = min(min(t_max, ray.direction_tmax.w), current_best);
    float3 extent = model.bounds_max.xyz - model.bounds_min.xyz;
    float cell_size = max(extent.x, max(extent.y, extent.z)) / max((float)model.meta.y, 1.0);
    float epsilon = max(cell_size * 0.65, 1.0e-4);
    [loop]
    for (uint step = 0u; step < 96u && t <= limit; ++step) {
        float d = SampleSDF(model, local_origin + local_direction * t);
        if (d <= epsilon) {
            hit_distance = t;
            return true;
        }
        t += max(d / direction_scale, epsilon * 0.25 / direction_scale);
    }
    hit_distance = current_best;
    return false;
}

bool TraceSDFAny(
    TraceRay ray,
    out float hit_distance
) {
    float best =
        ray.direction_tmax.w;

    bool found = false;

    for (
        uint i = 0u;
        i < TraceFrame.trace_limits.w;
        ++i
    ) {
        float candidate;

        if (TraceSDFModel(
                ray,
                SDFModels[i],
                best,
                candidate
            )) {
            best = candidate;
            found = true;
        }
    }

    hit_distance = best;
    return found;
}

uint SurfaceHash(
    uint object_id,
    uint revision,
    float3 position
) {
    /*
     * 0.5-world-unit cells.
     */
    int3 cell =
        int3(floor(position * 2.0));

    uint h =
        object_id * 747796405u +
        revision * 2891336453u;

    h ^=
        asuint(cell.x) * 277803737u;

    h ^=
        asuint(cell.y) * 1597334677u;

    h ^=
        asuint(cell.z) * 3812015801u;

    h ^= h >> 16u;

    /*
     * Zero means empty slot.
     */
    return h | 1u;
}

uint SurfaceRevision(uint object_id) {
    for (
        uint i = 0u;
        i < TraceFrame.trace_limits.w;
        ++i
    ) {
        if (SDFModels[i].meta.z == object_id)
            return SDFModels[i].version.x;
    }

    return 0u;
}

uint SurfaceCacheWriteSlot(uint key) {
    const uint mask =
        SURFACE_CACHE_CAPACITY - 1u;

    [unroll]
    for (uint probe = 0u; probe < 4u; ++probe) {
        uint slot =
            (key + probe) & mask;

        uint previous;

        InterlockedCompareExchange(
            SurfaceCacheKeys[slot],
            0u,
            key,
            previous
        );

        if (previous == 0u ||
            previous == key) {
            return slot;
        }
    }

    /*
     * All four occupied:
     * replace the primary slot.
     */
    uint slot = key & mask;
    uint previous;

    InterlockedExchange(
        SurfaceCacheKeys[slot],
        key,
        previous
    );

    return slot;
}

bool LookupSurfaceCache(
    uint object_id,
    uint revision,
    float3 position,
    out float3 radiance
) {
    uint key =
        SurfaceHash(
            object_id,
            revision,
            position
        );

    const uint mask =
        SURFACE_CACHE_CAPACITY - 1u;

    [unroll]
    for (uint probe = 0u; probe < 4u; ++probe) {
        uint slot =
            (key + probe) & mask;

        uint stored_key =
            SurfaceCacheKeys[slot];

        if (stored_key == 0u)
            break;

        if (stored_key != key)
            continue;

        SurfaceCacheEntry entry =
            SurfaceCacheEntries[slot];

        if (entry.object_id != object_id)
            continue;

        if (entry.revision != revision)
            continue;

        if (distance(
                entry.position.xyz,
                position
            ) > 0.75) {
            continue;
        }

        radiance =
            entry.direct_radiance.rgb +
            entry.indirect_radiance.rgb;

        return entry.confidence != 0u;
    }

    radiance = 0.0;
    return false;
}

[numthreads(64, 1, 1)]
void CS_SDFTrace(
    uint3 dispatch_id : SV_DispatchThreadID
) {
    uint queue_index = dispatch_id.x;

    if (queue_index >= MissCount[0])
        return;

    TraceRay ray =
        MissQueue[queue_index];

    float best =
        ray.direction_tmax.w;

    uint best_model =
        0xffffffffu;

    for (
        uint i = 0u;
        i < TraceFrame.trace_limits.w;
        ++i
    ) {
        float hit_distance;

        if (TraceSDFModel(
                ray,
                SDFModels[i],
                best,
                hit_distance
            )) {
            best = hit_distance;
            best_model = i;
        }
    }

    /*
     * No SDF hit:
     * keep the sky radiance written by CS_ScreenTrace.
     */
    if (best_model == 0xffffffffu)
        return;

    GPUSDFModel model =
        SDFModels[best_model];

    TraceHit hit =
        MakeTraceHit(
            TRACE_SDF,
            best
        );

    hit.object_id = model.meta.z;
    hit.confidence = 1.0;

    ScreenTraceHits[ray.destination] = hit;

    uint ray_width;
    uint ray_height;

    ProbeRadianceOutput.GetDimensions(
        ray_width,
        ray_height
    );

    uint2 destination =
        uint2(
            ray.destination % ray_width,
            ray.destination / ray_width
        );

    float3 world_position =
        ray.origin_tmin.xyz +
        ray.direction_tmax.xyz * best;

    float3 cached_radiance;

    bool cache_hit =
        LookupSurfaceCache(
            model.meta.z,
            model.version.x,
            world_position,
            cached_radiance
        );

    /*
     * An actual geometry hit without known radiance
     * must be dark, NOT sky.
     *
     * Otherwise we'd leak environment through walls.
     */
    ProbeRadianceOutput[destination] =
        cache_hit
            ? float4(
                  cached_radiance,
                  1.0
              )
            : float4(
                  0.0,
                  0.0,
                  0.0,
                  1.0
              );

    /*
     * Cyan remains useful as the debug indication
     * for an SDF hit whose radiance cache missed.
     */
    ScreenTraceOutput[destination] =
        cache_hit
            ? float4(
                  cached_radiance,
                  1.0
              )
            : float4(
                  0.0,
                  1.0,
                  1.0,
                  1.0
              );
}

[numthreads(8, 8, 1)]
void CS_SurfaceCacheUpdate(
    uint3 dispatch_id : SV_DispatchThreadID
) {
    uint2 pixel = dispatch_id.xy;

    uint2 resolution =
        uint2(TraceFrame.resolution.xy);

    if (any(pixel >= resolution))
        return;

    float depth =
        TraceDepth.Load(int3(pixel, 0));

    if (depth <= 0.0)
        return;

    uint object_id =
        TraceObjectId.Load(int3(pixel, 0));

    if (object_id == 0u)
        return;

    uint revision =
        SurfaceRevision(object_id);

    float3 position =
        ReconstructWorldPosition(
            pixel,
            depth
        );

    float4 normal_roughness =
        TraceNormalRoughness.Load(
            int3(pixel, 0)
        );

    float4 albedo_metallic =
        TraceAlbedoMetallic.Load(
            int3(pixel, 0)
        );

    float4 emissive =
        TraceEmissive.Load(
            int3(pixel, 0)
        );

    float4 direct =
        TraceDirectRadiance.Load(
            int3(pixel, 0)
        );

    uint key =
        SurfaceHash(
            object_id,
            revision,
            position
        );

    uint slot =
        SurfaceCacheWriteSlot(key);

    SurfaceCacheEntry entry;

    entry.position =
        float4(position, 1.0);

    entry.normal =
        float4(
            normalize(normal_roughness.xyz),
            0.0
        );

    entry.albedo_roughness =
        float4(
            albedo_metallic.rgb,
            normal_roughness.w
        );

    entry.emissive_metallic =
        float4(
            emissive.rgb,
            albedo_metallic.w
        );

    entry.direct_radiance = direct;

    /*
     * Multi-bounce not part of this milestone.
     */
    entry.indirect_radiance = 0.0;

    entry.object_id = object_id;
    entry.revision = revision;

    entry.last_frame =
        (uint)TraceFrame.camera_position.w;

    entry.confidence = 1u;

    SurfaceCacheEntries[slot] = entry;
}

[numthreads(8, 8, 1)]
void CS_ScreenProbes(
    uint3 dispatch_id : SV_DispatchThreadID
) {
    uint probe_width;
    uint probe_height;

    ScreenProbesOutput.GetDimensions(
        probe_width,
        probe_height
    );

    uint2 probe = dispatch_id.xy;

    if (probe.x >= probe_width ||
        probe.y >= probe_height) {
        return;
    }

    uint2 base =
        probe * SCREEN_PROBE_DIRECTION_SIZE;

    float3 radiance = 0.0;

    [unroll]
    for (
        uint y = 0u;
        y < SCREEN_PROBE_DIRECTION_SIZE;
        ++y
    ) {
        [unroll]
        for (
            uint x = 0u;
            x < SCREEN_PROBE_DIRECTION_SIZE;
            ++x
        ) {
            radiance +=
                ProbeRadianceOutput[
                    base + uint2(x, y)
                ].rgb;
        }
    }

    /*
     * Rays are cosine-weighted, so this average is
     * exactly the quantity needed by a Lambertian
     * albedo multiplier in the present pass.
     */
    radiance /=
        (float)SCREEN_PROBE_RAY_COUNT;

    ScreenProbesOutput[probe] =
        float4(radiance, 1.0);
}

[numthreads(8, 8, 1)]
void CS_HZB(uint3 dispatch_id : SV_DispatchThreadID) {
    uint src_width, src_height, dst_width, dst_height;
    HZBSource.GetDimensions(src_width, src_height);
    HZBOutput.GetDimensions(dst_width, dst_height);
    uint2 pixel = dispatch_id.xy;
    if (pixel.x >= dst_width || pixel.y >= dst_height) return;
    uint2 src_size = uint2(src_width, src_height);
    uint2 dst_size = uint2(dst_width, dst_height);
    uint2 src_begin = (pixel * src_size) / dst_size;
    uint2 src_end = ((pixel + uint2(1u, 1u)) * src_size) / dst_size;
    float depth = 0.0;
    for (uint y = src_begin.y; y < src_end.y; ++y)
        for (uint x = src_begin.x; x < src_end.x; ++x)
            depth = max(depth, HZBSource.Load(int3(int2(x, y), 0)));
    HZBOutput[pixel] = depth;
}

float3 LoadScreenProbe(int2 probe) {
    uint width;
    uint height;

    ScreenProbesTexture.GetDimensions(
        width,
        height
    );

    probe = clamp(
        probe,
        int2(0, 0),
        int2(
            (int)width - 1,
            (int)height - 1
        )
    );

    return ScreenProbesTexture.Load(
        int3(probe, 0)
    ).rgb;
}

float3 InterpolateScreenProbeGI(
    int2 pixel
) {
    float2 probe_position =
        (float2(pixel) + 0.5) /
        (float)SCREEN_PROBE_TILE_SIZE -
        0.5;

    int2 base =
        int2(floor(probe_position));

    float2 blend =
        frac(probe_position);

    float3 p00 =
        LoadScreenProbe(base);

    float3 p10 =
        LoadScreenProbe(
            base + int2(1, 0)
        );

    float3 p01 =
        LoadScreenProbe(
            base + int2(0, 1)
        );

    float3 p11 =
        LoadScreenProbe(
            base + int2(1, 1)
        );

    return lerp(
        lerp(p00, p10, blend.x),
        lerp(p01, p11, blend.x),
        blend.y
    );
}

float4 PS_Present(PresentVSOutput input) : SV_Target0 {
    int2 pixel = int2(input.position.xy);
#ifdef ALBEDO
    return float4(AlbedoMetallicTexture.Load(int3(pixel, 0)).rgb, 1.0);
#elif defined(NORMALS)
    return float4(NormalRoughnessTexture.Load(int3(pixel, 0)).rgb * 0.5 + 0.5, 1.0);
#elif defined(DEPTH)
    float visible_depth = 1.0 - saturate(LinearizeDepth(DepthTexture.Load(int3(pixel, 0))) / 200.0);
    return float4(visible_depth, visible_depth, visible_depth, 1.0);
#elif defined(ROUGHNESS)
    float roughness = NormalRoughnessTexture.Load(int3(pixel, 0)).w;
    return float4(roughness, roughness, roughness, 1.0);
#elif defined(VELOCITY)
    float2 velocity = VelocityTexture.Load(int3(pixel, 0));
    return float4(saturate(0.5 + velocity.x * 20.0), saturate(0.5 + velocity.y * 20.0), 0.5, 1.0);
#elif defined(OBJECT_ID)
    float value = frac((float)ObjectIdTexture.Load(int3(pixel, 0)) * 0.61803398875);
    return float4(value, frac(value * 3.17), frac(value * 7.13), 1.0);
#elif defined(HZB)
    uint hzb_width, hzb_height, hzb_mip_count;
    HZBTexture.GetDimensions(0, hzb_width, hzb_height, hzb_mip_count);
    uint mip = min(4u, hzb_mip_count);
    float hzb_depth;
    if (mip == 0u) {
        hzb_depth = DepthTexture.Load(int3(pixel, 0));
    } else {
        uint hzb_mip = mip - 1u;
        uint mip_width = max(1u, hzb_width >> hzb_mip);
        uint mip_height = max(1u, hzb_height >> hzb_mip);
        uint2 hzb_pixel = min(uint2(pixel) >> mip, uint2(mip_width - 1u, mip_height - 1u));
        hzb_depth = HZBTexture.Load(int3(int2(hzb_pixel), hzb_mip));
    }
    float value = 1.0 - saturate(LinearizeDepth(hzb_depth) / 200.0);
    return float4(value, value, value, 1.0);
#elif defined(SCREEN_TRACE)
    uint full_width;
    uint full_height;
    uint trace_width;
    uint trace_height;

    DepthTexture.GetDimensions(
        full_width,
        full_height
    );

    ScreenTraceTexture.GetDimensions(
        trace_width,
        trace_height
    );

    uint2 trace_pixel = min(
        uint2(
            (uint)pixel.x * trace_width /
                full_width,
            (uint)pixel.y * trace_height /
                full_height
        ),
        uint2(
            trace_width - 1u,
            trace_height - 1u
        )
    );

    return float4(
        ScreenTraceTexture.Load(
            int3(trace_pixel, 0)
        ).rgb,
        1.0
    );
#elif defined(DIRECT_RADIANCE)
    return float4(DirectRadianceTexture.Load(int3(pixel, 0)).rgb, 1.0);
#elif defined(SCREEN_PROBES)
    uint full_width, full_height, probe_width, probe_height;
    DepthTexture.GetDimensions(full_width, full_height);
    ScreenProbesTexture.GetDimensions(probe_width, probe_height);
    uint2 probe = min(uint2((uint)pixel.x * probe_width / full_width, (uint)pixel.y * probe_height / full_height), uint2(probe_width - 1u, probe_height - 1u));
    return float4(ScreenProbesTexture.Load(int3(probe, 0)).rgb, 1.0);
#elif defined(FINAL_GI)
    float depth =
        DepthTexture.Load(
            int3(pixel, 0)
        );

    float3 direct =
        DirectRadianceTexture.Load(
            int3(pixel, 0)
        ).rgb;

    /*
     * Background direct radiance already contains sky.
     */
    if (depth <= 0.0)
        return float4(direct, 1.0);

    float4 material =
        AlbedoMetallicTexture.Load(
            int3(pixel, 0)
        );

    float3 incoming =
        InterpolateScreenProbeGI(pixel);

    float diffuse_weight =
        1.0 - material.w;

    float3 indirect =
        material.rgb *
        diffuse_weight *
        incoming;

    return float4(
        direct + indirect,
        1.0
    );
#else
    return float4(1.0, 0.0, 1.0, 1.0);
#endif
}
