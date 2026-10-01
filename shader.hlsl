#define FAR_PLANE 10000.0f
#define NEAR_PLANE 0.05f
#define INVALID_INDEX 0xffffffffu
#define PI 3.14159265358979323846f
#define INV_PI 0.31830988618379067154f
#define TWO_PI 6.28318530717958647692f

#define TRACE_INACTIVE 0u
#define TRACE_MISS 1u
#define TRACE_SCREEN 2u
#define TRACE_SDF 3u
#define TRACE_GLOBAL_SDF 4u
#define TRACE_TRIANGLE 5u

#define TRACE_RAY_INACTIVE 0u
#define TRACE_RAY_DIFFUSE 1u
#define TRACE_RAY_REFLECTION 2u
#define TRACE_RAY_SHADOW 3u
#define TRACE_RAY_WORLD_PROBE 4u

#define RADIANCE_FEATURE_EMISSIVE 0x00000001u
#define RADIANCE_FEATURE_DYNAMIC_GRID 0x00000002u
#define RADIANCE_FEATURE_GLOBAL_SDF 0x00000004u
#define RADIANCE_FEATURE_SURFACE_CACHE 0x00000008u
#define RADIANCE_FEATURE_TEMPORAL_PROBES 0x00000010u
#define RADIANCE_FEATURE_SPATIAL_PROBES 0x00000020u
#define RADIANCE_FEATURE_WORLD_CACHE 0x00000040u
#define RADIANCE_FEATURE_MULTIBOUNCE 0x00000080u
#define RADIANCE_FEATURE_ADAPTIVE_RAYS 0x00000100u
#define RADIANCE_FEATURE_REFLECTIONS 0x00000200u

#define DEBUG_FINAL_GI 0u
#define DEBUG_ALBEDO 1u
#define DEBUG_NORMALS 2u
#define DEBUG_DEPTH 3u
#define DEBUG_ROUGHNESS 4u
#define DEBUG_METALLIC 5u
#define DEBUG_VELOCITY 6u
#define DEBUG_OBJECT_ID 7u
#define DEBUG_MATERIAL_ID 8u
#define DEBUG_PRIMITIVE_ID 9u
#define DEBUG_HZB 10u
#define DEBUG_DIRECT_RADIANCE 11u
#define DEBUG_EMISSIVE 12u
#define DEBUG_SCREEN_TRACE 13u
#define DEBUG_LOCAL_SDF 14u
#define DEBUG_GLOBAL_SDF 15u
#define DEBUG_SURFACE_CACHE 16u
#define DEBUG_SCREEN_PROBE_DIRECTIONAL 17u
#define DEBUG_SCREEN_PROBE_IRRADIANCE 18u
#define DEBUG_SCREEN_PROBE_CONFIDENCE 19u
#define DEBUG_SCREEN_PROBE_VARIANCE 20u
#define DEBUG_SCREEN_PROBE_HISTORY 21u
#define DEBUG_WORLD_RADIANCE 22u
#define DEBUG_RAY_BUDGET 23u
#define DEBUG_REFLECTIONS 24u

#define SURFACE_CACHE_CAPACITY 262144u
#define SCREEN_PROBE_TILE_SIZE 8u
#define SCREEN_PROBE_DIRECTION_SIZE 4u
#define SCREEN_PROBE_RAY_COUNT 8u

#define MAX_SCREEN_PROBE_DIRECTION_SIZE 8u
#define MAX_SCREEN_PROBE_RAYS 32u
#define MAX_GLOBAL_SDF_CLIPMAPS 8u
#define MAX_CACHE_PROBES 8u
#define MAX_DYNAMIC_GRID_STEPS 256u
#define RAY_FLAG_DYNAMIC_HIT 0x80000000u
#define RAY_FLAG_GLOBAL_RESOLVED 0x40000000u

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
    uint4 meta;
    uint4 version;
};

struct GPULight {
    float4 position_range;
    float4 direction_type;
    float4 color_intensity;
    float4 spot_angles;
};

struct LegacySurfaceCacheEntry {
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

struct GPUSceneTriangle {
    float4 p0;
    float4 p1;
    float4 p2;
    float4 uv01;
    float4 uv2_area;
    uint4 meta;
};

struct GPUEmissiveTriangle {
    uint4 meta;
    float4 radiance_area;
    float4 sampling;
};

struct GPUDynamicGridCell {
    uint4 range_flags;
    float4 bounds_min;
    float4 bounds_max;
};

struct GPUGlobalSDFClipmap {
    float4 center_extent;
    float4 voxel_brick;
    uint4 grid;
    uint4 data;
};

struct SurfaceHit {
    float4 position_distance;
    float4 normal_confidence;
    float4 uv_bary;
    uint4 identity;
    uint4 meta;
};

struct SurfaceCacheEntry {
    float4 position_distance;
    float4 normal_confidence;
    float4 albedo_roughness;
    float4 emissive_metallic;
    float4 direct_radiance;
    float4 indirect_radiance;
    uint4 identity;
    uint4 state;
};

struct ScreenProbeState {
    float4 position_depth;
    float4 normal_confidence;
    uint4 history;
    float4 statistics;
};

struct WorldProbeState {
    float4 position_radius;
    uint4 identity;
    float4 statistics;
    uint4 state;
};

struct RayBudget {
    uint4 counts;
    float4 priority;
};

struct RadianceConstants {
    uint4 scene_counts;
    uint4 sdf_counts;
    uint4 cache_counts;
    uint4 probe_config;
    float4 trace_params;
    uint4 trace_limits;
    float4 temporal_params;
    uint4 feature_flags;
    float4 adaptive_params;
    float4 reflection_params;
    uint4 dynamic_grid;
    float4 dynamic_grid_origin_cell;
    float4 global_sdf_params;
    uint4 world_probe_config;
    float4 world_probe_params;
    uint4 reserved;
};

struct PassConstants {
    uint4 dispatch;
    uint4 range;
    uint4 dimensions;
    uint4 flags;
};

// -----------------------------------------------------------------------------
// Space 0: current raster scene.
// -----------------------------------------------------------------------------
[[vk::binding(0, 0)]] StructuredBuffer<GPUObject> Objects : register(t0, space0);
[[vk::binding(1, 0)]] StructuredBuffer<GPUMaterial> Materials : register(t1, space0);
[[vk::binding(2, 0)]] ConstantBuffer<FrameConstants> Frame : register(b0, space0);

struct DrawConstants { uint object_index; };
[[vk::push_constant]] DrawConstants Draw;

// -----------------------------------------------------------------------------
// Space 1: current presentation inputs.
// -----------------------------------------------------------------------------
[[vk::binding(0, 1)]] Texture2D<float> DepthTexture : register(t0, space1);
[[vk::binding(1, 1)]] Texture2D<float4> NormalRoughnessTexture : register(t1, space1);
[[vk::binding(2, 1)]] Texture2D<float4> AlbedoMetallicTexture : register(t2, space1);
[[vk::binding(3, 1)]] Texture2D<float2> VelocityTexture : register(t3, space1);
[[vk::binding(4, 1)]] Texture2D<uint> ObjectIdTexture : register(t4, space1);
[[vk::binding(5, 1)]] Texture2D<float> HZBTexture : register(t5, space1);
[[vk::binding(6, 1)]] Texture2D<float4> ScreenTraceTexture : register(t6, space1);
[[vk::binding(7, 1)]] Texture2D<float4> DirectRadianceTexture : register(t7, space1);
[[vk::binding(8, 1)]] Texture2D<float4> ScreenProbesTexture : register(t8, space1);

// -----------------------------------------------------------------------------
// Space 2: HZB build.
// -----------------------------------------------------------------------------
[[vk::binding(0, 2)]] Texture2D<float> HZBSource : register(t0, space2);
[[vk::binding(1, 2)]] RWTexture2D<float> HZBOutput : register(u0, space2);

// -----------------------------------------------------------------------------
// Space 3: compatibility radiance path used by the current C renderer.
// -----------------------------------------------------------------------------
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
[[vk::binding(17, 3)]] RWStructuredBuffer<uint> LegacySurfaceCacheKeys : register(u6, space3);
[[vk::binding(18, 3)]] RWStructuredBuffer<LegacySurfaceCacheEntry> LegacySurfaceCacheEntries : register(u7, space3);
[[vk::binding(19, 3)]] RWTexture2D<float4> ProbeRadianceOutput : register(u8, space3);
[[vk::binding(20, 3)]] RWTexture2D<float4> ScreenProbesOutput : register(u9, space3);

// -----------------------------------------------------------------------------
// Space 4: permanent ray-scene ABI.
// Future C stages allocate and bind these resources without changing HLSL.
// -----------------------------------------------------------------------------
[[vk::binding(0, 4)]] ConstantBuffer<RadianceConstants> Radiance : register(b0, space4);
[[vk::binding(1, 4)]] ConstantBuffer<PassConstants> Pass : register(b1, space4);
[[vk::binding(2, 4)]] StructuredBuffer<GPUObject> SceneObjects : register(t0, space4);
[[vk::binding(3, 4)]] StructuredBuffer<GPUMaterial> SceneMaterials : register(t1, space4);
[[vk::binding(4, 4)]] StructuredBuffer<GPUSceneTriangle> SceneTriangles : register(t2, space4);
[[vk::binding(5, 4)]] StructuredBuffer<GPUEmissiveTriangle> EmissiveTriangles : register(t3, space4);
[[vk::binding(6, 4)]] StructuredBuffer<GPUSDFModel> LocalSDFModels : register(t4, space4);
[[vk::binding(7, 4)]] StructuredBuffer<float> LocalSDFVoxels : register(t5, space4);
[[vk::binding(8, 4)]] StructuredBuffer<uint> LocalSDFSurfaceIds : register(t6, space4);
[[vk::binding(9, 4)]] StructuredBuffer<GPUDynamicGridCell> DynamicGridCells : register(t7, space4);
[[vk::binding(10, 4)]] StructuredBuffer<uint> DynamicGridIndices : register(t8, space4);
[[vk::binding(11, 4)]] StructuredBuffer<GPUGlobalSDFClipmap> GlobalSDFClipmaps : register(t9, space4);
[[vk::binding(12, 4)]] StructuredBuffer<uint> GlobalSDFPageTable : register(t10, space4);
[[vk::binding(13, 4)]] StructuredBuffer<float> GlobalSDFBricks : register(t11, space4);
[[vk::binding(14, 4)]] StructuredBuffer<uint> GlobalSDFSurfaceIds : register(t12, space4);
[[vk::binding(15, 4)]] StructuredBuffer<GPULight> SceneLights : register(t13, space4);
[[vk::binding(16, 4)]] Texture2D<uint> TraceMaterialId : register(t14, space4);
[[vk::binding(17, 4)]] Texture2D<uint> TracePrimitiveId : register(t15, space4);

// -----------------------------------------------------------------------------
// Space 5: permanent wavefront queues and adaptive scheduling.
// -----------------------------------------------------------------------------
[[vk::binding(0, 5)]] RWStructuredBuffer<TraceRay> RayQueueA : register(u0, space5);
[[vk::binding(1, 5)]] RWStructuredBuffer<TraceRay> RayQueueB : register(u1, space5);
[[vk::binding(2, 5)]] RWStructuredBuffer<SurfaceHit> RaySurfaceHits : register(u2, space5);
[[vk::binding(3, 5)]] RWStructuredBuffer<uint> RayCounters : register(u3, space5);
[[vk::binding(4, 5)]] RWStructuredBuffer<uint> RayDispatchArgs : register(u4, space5);
[[vk::binding(5, 5)]] RWStructuredBuffer<RayBudget> RayBudgets : register(u5, space5);
[[vk::binding(6, 5)]] RWStructuredBuffer<uint> RadianceUpdateList : register(u6, space5);
[[vk::binding(7, 5)]] RWStructuredBuffer<float4> RayRadiance : register(u7, space5);
[[vk::binding(8, 5)]] RWStructuredBuffer<uint> RayFlags : register(u8, space5);

// -----------------------------------------------------------------------------
// Space 6: surface/world radiance caches.
// -----------------------------------------------------------------------------
[[vk::binding(0, 6)]] RWStructuredBuffer<uint> SurfaceCacheKeys : register(u0, space6);
[[vk::binding(1, 6)]] RWStructuredBuffer<SurfaceCacheEntry> SurfaceCacheEntries : register(u1, space6);
[[vk::binding(2, 6)]] RWStructuredBuffer<WorldProbeState> WorldProbes : register(u2, space6);
[[vk::binding(3, 6)]] RWStructuredBuffer<float4> WorldProbeRadiance : register(u3, space6);
[[vk::binding(4, 6)]] RWStructuredBuffer<uint> WorldProbeKeys : register(u4, space6);
[[vk::binding(5, 6)]] RWStructuredBuffer<uint> InvalidationQueue : register(u5, space6);

// -----------------------------------------------------------------------------
// Space 7: directional screen probes and history.
// -----------------------------------------------------------------------------
[[vk::binding(0, 7)]] Texture2D<float4> ProbeHistoryRadiance : register(t0, space7);
[[vk::binding(1, 7)]] Texture2D<float4> ProbeHistoryMeta : register(t1, space7);
[[vk::binding(2, 7)]] Texture2D<float4> ProbePreviousIrradiance : register(t2, space7);
[[vk::binding(3, 7)]] Texture2D<float> ProbeHistoryDepth : register(t3, space7);
[[vk::binding(4, 7)]] Texture2D<float4> ProbeHistoryNormal : register(t4, space7);
[[vk::binding(5, 7)]] RWTexture2D<float4> ProbeCurrentRadiance : register(u0, space7);
[[vk::binding(6, 7)]] RWTexture2D<float4> ProbeCurrentMeta : register(u1, space7);
[[vk::binding(7, 7)]] RWTexture2D<float4> ProbeIrradiance : register(u2, space7);
[[vk::binding(8, 7)]] RWTexture2D<float4> ProbeHistoryRadianceOut : register(u3, space7);
[[vk::binding(9, 7)]] RWTexture2D<float4> ProbeHistoryMetaOut : register(u4, space7);
[[vk::binding(10, 7)]] RWTexture2D<float4> ProbePreviousIrradianceOut : register(u5, space7);
[[vk::binding(11, 7)]] RWTexture2D<float> ProbeHistoryDepthOut : register(u6, space7);
[[vk::binding(12, 7)]] RWTexture2D<float4> ProbeHistoryNormalOut : register(u7, space7);

// -----------------------------------------------------------------------------
// Space 8: reflections.
// -----------------------------------------------------------------------------
[[vk::binding(0, 8)]] RWTexture2D<float4> ReflectionOutput : register(u0, space8);
[[vk::binding(1, 8)]] Texture2D<float4> ReflectionHistory : register(t0, space8);
[[vk::binding(2, 8)]] RWTexture2D<float4> ReflectionMeta : register(u1, space8);

bool FeatureEnabled(uint bit) {
    return (Radiance.feature_flags.x & bit) != 0u;
}

// -----------------------------------------------------------------------------
// Shared math and sampling.
// -----------------------------------------------------------------------------
float2 ScreenUVToNDC(float2 uv) { return float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0); }
float2 NDCToScreenUV(float2 ndc) { return float2(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5); }

uint Hash32(uint x) {
    x ^= x >> 16u;
    x *= 0x7feb352du;
    x ^= x >> 15u;
    x *= 0x846ca68bu;
    x ^= x >> 16u;
    return x;
}

uint HashCombine(uint a, uint b) {
    return Hash32(a ^ (b + 0x9e3779b9u + (a << 6u) + (a >> 2u)));
}

float HashFloat(uint x) {
    return (float)(Hash32(x) & 0x00ffffffu) / 16777216.0f;
}

float RadicalInverseVdC(uint bits) {
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xaaaaaaaau) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xccccccccu) >> 2u);
    bits = ((bits & 0x0f0f0f0fu) << 4u) | ((bits & 0xf0f0f0f0u) >> 4u);
    bits = ((bits & 0x00ff00ffu) << 8u) | ((bits & 0xff00ff00u) >> 8u);
    return (float)bits * 2.3283064365386963e-10f;
}

float2 Hammersley(uint index, uint count, uint scramble) {
    float a = ((float)index + 0.5f) / max((float)count, 1.0f);
    float b = frac(RadicalInverseVdC(index ^ scramble) + HashFloat(scramble));
    return float2(a, b);
}

void BuildNormalBasis(float3 normal, out float3 tangent, out float3 bitangent) {
    float3 axis = abs(normal.z) < 0.999f ? float3(0.0f, 0.0f, 1.0f) : float3(0.0f, 1.0f, 0.0f);
    tangent = normalize(cross(axis, normal));
    bitangent = cross(normal, tangent);
}

float3 CosineHemisphere(float2 u) {
    float radius = sqrt(saturate(u.x));
    float phi = TWO_PI * u.y;
    return float3(radius * cos(phi), radius * sin(phi), sqrt(max(0.0f, 1.0f - u.x)));
}

float2 OctEncode(float3 n) {
    n /= max(abs(n.x) + abs(n.y) + abs(n.z), 1.0e-8f);
    float2 p = n.xy;
    if (n.z < 0.0f) {
        float2 sign_p = float2(p.x >= 0.0f ? 1.0f : -1.0f, p.y >= 0.0f ? 1.0f : -1.0f);
        p = (1.0f - abs(p.yx)) * sign_p;
    }
    return p * 0.5f + 0.5f;
}

float3 OctDecode(float2 e) {
    float2 f = e * 2.0f - 1.0f;
    float3 n = float3(f.x, f.y, 1.0f - abs(f.x) - abs(f.y));
    if (n.z < 0.0f) {
        float2 sign_n = float2(n.x >= 0.0f ? 1.0f : -1.0f, n.y >= 0.0f ? 1.0f : -1.0f);
        n.xy = (1.0f - abs(n.yx)) * sign_n;
    }
    return normalize(n);
}

float3 TransformPoint(float3 p, float4x4 matrix) { return mul(float4(p, 1.0f), matrix).xyz; }
float3 TransformVector(float3 v, float4x4 matrix) { return mul(float4(v, 0.0f), matrix).xyz; }

uint PackPixel(uint2 pixel) { return (pixel.x & 0xffffu) | ((pixel.y & 0xffffu) << 16u); }
uint2 UnpackPixel(uint packed) { return uint2(packed & 0xffffu, (packed >> 16u) & 0xffffu); }

float3 ReconstructWorldPosition(uint2 pixel, float depth) {
    float2 uv = (float2(pixel) + 0.5f) * TraceFrame.resolution.zw;
    float4 world = mul(float4(ScreenUVToNDC(uv), depth, 1.0f), TraceFrame.inverse_view_projection);
    return world.xyz / world.w;
}

float LinearizeDepth(float depth) {
    return (NEAR_PLANE * FAR_PLANE) / (depth * (FAR_PLANE - NEAR_PLANE) + NEAR_PLANE);
}

bool IntersectAABB(float3 origin, float3 direction, float3 bounds_min, float3 bounds_max, out float t_min, out float t_max) {
    float3 safe_direction = float3(
        abs(direction.x) < 1.0e-8f ? (direction.x < 0.0f ? -1.0e-8f : 1.0e-8f) : direction.x,
        abs(direction.y) < 1.0e-8f ? (direction.y < 0.0f ? -1.0e-8f : 1.0e-8f) : direction.y,
        abs(direction.z) < 1.0e-8f ? (direction.z < 0.0f ? -1.0e-8f : 1.0e-8f) : direction.z
    );
    float3 a = (bounds_min - origin) / safe_direction;
    float3 b = (bounds_max - origin) / safe_direction;
    float3 lo = min(a, b);
    float3 hi = max(a, b);
    t_min = max(lo.x, max(lo.y, lo.z));
    t_max = min(hi.x, min(hi.y, hi.z));
    return t_max >= max(t_min, 0.0f);
}

TraceRay MakeTraceRay(float3 origin, float3 direction, float tmin, float tmax, uint type, uint destination, uint2 origin_pixel, uint source_object_id) {
    TraceRay ray;
    ray.origin_tmin = float4(origin, tmin);
    ray.direction_tmax = float4(normalize(direction), tmax);
    ray.type = type;
    ray.destination = destination;
    ray.origin_pixel = PackPixel(origin_pixel);
    ray.source_object_id = source_object_id;
    return ray;
}

TraceHit MakeTraceHit(uint type, float distance) {
    TraceHit hit;
    hit.type = type;
    hit.object_id = 0u;
    hit.hit_pixel = uint2(INVALID_INDEX, INVALID_INDEX);
    hit.distance = distance;
    hit.confidence = 0.0f;
    hit.padding = uint2(0u, 0u);
    return hit;
}

SurfaceHit MakeSurfaceHit(uint hit_type, float distance) {
    SurfaceHit hit;
    hit.position_distance = float4(0.0f, 0.0f, 0.0f, distance);
    hit.normal_confidence = float4(0.0f, 1.0f, 0.0f, 0.0f);
    hit.uv_bary = 0.0f;
    hit.identity = uint4(INVALID_INDEX, INVALID_INDEX, INVALID_INDEX, hit_type);
    hit.meta = uint4(0u, 0u, 0u, 0u);
    return hit;
}

// -----------------------------------------------------------------------------
// Raster visibility.
// -----------------------------------------------------------------------------
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
    [[vk::location(6)]] nointerpolation uint primitive_base : TEXCOORD5;
};

GBufferVSOutput VS_GBuffer(GBufferVSInput input) {
    GPUObject object = Objects[Draw.object_index];
    float4 local_position = float4(input.position, 1.0f);
    float4 world_position = mul(local_position, object.world);
    GBufferVSOutput output;
    output.position = mul(world_position, Frame.view_projection);
    output.current_clip = output.position;
    output.previous_clip = mul(mul(local_position, object.previous_world), Frame.previous_view_projection);
    output.normal = normalize(mul(float4(input.normal, 0.0f), object.normal_world).xyz);
    output.uv = input.uv;
    output.material = object.draw.w + input.material;
    output.object_id = object.draw.z;
    output.primitive_base = object.meta.w;
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
    output.emissive = float4(material.emissive, 1.0f);
    float2 current_ndc = input.current_clip.xy / input.current_clip.w;
    float2 previous_ndc = input.previous_clip.xy / input.previous_clip.w;
    output.velocity = (current_ndc - previous_ndc) * float2(0.5f, -0.5f);
    output.object_id = input.object_id;
    return output;
}

struct GBufferFullOutput {
    float4 normal_roughness : SV_Target0;
    float4 albedo_metallic : SV_Target1;
    float2 velocity : SV_Target2;
    uint object_id : SV_Target3;
    float4 emissive : SV_Target4;
    uint material_id : SV_Target5;
    uint primitive_id : SV_Target6;
};

GBufferFullOutput PS_GBufferFull(GBufferVSOutput input, uint primitive_id : SV_PrimitiveID) {
    GPUMaterial material = Materials[input.material];
    GBufferFullOutput output;
    output.normal_roughness = float4(normalize(input.normal), material.roughness);
    output.albedo_metallic = float4(material.base_color.rgb, material.metallic);
    output.emissive = float4(material.emissive, 1.0f);
    float2 current_ndc = input.current_clip.xy / input.current_clip.w;
    float2 previous_ndc = input.previous_clip.xy / input.previous_clip.w;
    output.velocity = (current_ndc - previous_ndc) * float2(0.5f, -0.5f);
    output.object_id = input.object_id;
    output.material_id = input.material;
    output.primitive_id = input.primitive_base + primitive_id;
    return output;
}

struct PresentVSOutput { float4 position : SV_Position; };

PresentVSOutput VS_Present(uint vertex_id : SV_VertexID) {
    const float2 positions[3] = {float2(-1.0f, -1.0f), float2(-1.0f, 3.0f), float2(3.0f, -1.0f)};
    PresentVSOutput output;
    output.position = float4(positions[vertex_id], 0.0f, 1.0f);
    return output;
}

// -----------------------------------------------------------------------------
// HZB and compatibility screen tracing.
// -----------------------------------------------------------------------------
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
    if (clip.w <= 1.0e-5f) {
        uv = 0.0f;
        depth = 0.0f;
        return false;
    }
    float3 ndc = clip.xyz / clip.w;
    uv = NDCToScreenUV(ndc.xy);
    depth = ndc.z;
    return all(uv >= 0.0f) && all(uv < 1.0f) && depth >= 0.0f && depth <= 1.0f;
}

float SolveTraceDistance(float value_origin, float value_direction, float w_origin, float w_direction, float target) {
    float denominator = value_direction - target * w_direction;
    return abs(denominator) < 1.0e-7f ? 1.0e30f : (target * w_origin - value_origin) / denominator;
}

float TraceCellExit(float4 clip_origin, float4 clip_direction, uint2 dimensions, uint2 cell, float distance, float2 screen_direction) {
    float exit_x = 1.0e30f;
    float exit_y = 1.0e30f;
    float4 current_clip = clip_origin + clip_direction * distance;
    if (abs(screen_direction.x) > 1.0e-7f) {
        float boundary = screen_direction.x > 0.0f ? (float)(cell.x + 1u) / (float)dimensions.x : (float)cell.x / (float)dimensions.x;
        exit_x = distance + SolveTraceDistance(current_clip.x, clip_direction.x, current_clip.w, clip_direction.w, boundary * 2.0f - 1.0f);
        if (exit_x <= distance + 1.0e-5f) exit_x = 1.0e30f;
    }
    if (abs(screen_direction.y) > 1.0e-7f) {
        float boundary = screen_direction.y > 0.0f ? (float)(cell.y + 1u) / (float)dimensions.y : (float)cell.y / (float)dimensions.y;
        exit_y = distance + SolveTraceDistance(current_clip.y, clip_direction.y, current_clip.w, clip_direction.w, 1.0f - boundary * 2.0f);
        if (exit_y <= distance + 1.0e-5f) exit_y = 1.0e30f;
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

TraceHit TraceScreenRay(float3 origin, float3 direction, uint2 origin_pixel, uint source_object_id, float3 source_normal, float max_distance, float thickness, float skip_thickness, uint max_steps, uint max_start_level) {
    TraceHit result = MakeTraceHit(TRACE_MISS, max_distance);
    float4 clip_origin = mul(float4(origin, 1.0f), TraceFrame.view_projection);
    float4 clip_direction = mul(float4(direction, 0.0f), TraceFrame.view_projection);
    if (clip_origin.w <= 1.0e-5f) return result;

    uint hzb_width, hzb_height, hzb_mip_count;
    TraceHZB.GetDimensions(0, hzb_width, hzb_height, hzb_mip_count);
    uint maximum_level = min(hzb_mip_count, max_start_level);
    uint level = maximum_level;
    float2 screen_direction = float2(
        clip_direction.x * clip_origin.w - clip_origin.x * clip_direction.w,
        -(clip_direction.y * clip_origin.w - clip_origin.y * clip_direction.w)
    );
    float distance = 0.0f;

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
        if (exit_distance >= 1.0e29f) return result;
        exit_distance = min(exit_distance, max_distance);

        float2 exit_uv;
        float exit_depth;
        if (!ProjectTracePoint(clip_origin, clip_direction, exit_distance, exit_uv, exit_depth)) return result;

        float scene_linear = LinearizeDepth(scene_depth);
        float farther_ray_linear = LinearizeDepth(min(ray_depth, exit_depth));
        bool clear_cell = scene_depth <= 0.0f || farther_ray_linear < scene_linear - skip_thickness;
        if (clear_cell) {
            float next_distance = exit_distance + max(1.0e-4f, exit_distance * 1.0e-5f);
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
            uint hit_object_id = TraceObjectId.Load(int3(cell, 0));
            float3 hit_position = ReconstructWorldPosition(cell, scene_depth);
            float3 hit_normal = normalize(TraceNormalRoughness.Load(int3(cell, 0)).xyz);
            float3 to_hit = hit_position - origin;
            float ray_distance = dot(to_hit, direction);
            float3 closest_point = origin + direction * ray_distance;
            float off_ray_distance = length(hit_position - closest_point);
            float plane_distance = abs(dot(to_hit, source_normal));
            bool same_surface = hit_object_id == source_object_id && abs(dot(hit_normal, source_normal)) > 0.95f && plane_distance < max(TraceFrame.trace_params.z * 2.0f, skip_thickness * 4.0f);
            float ray_tolerance = max(thickness * 2.0f, ray_distance * 0.0025f);
            bool valid_world_hit = ray_distance > 0.0f && ray_distance < max_distance && off_ray_distance <= ray_tolerance;
            if (!same_surface && valid_world_hit) {
                result.type = TRACE_SCREEN;
                result.object_id = hit_object_id;
                result.hit_pixel = cell;
                result.distance = ray_distance;
                result.confidence = 1.0f;
                return result;
            }
        }

        float next_distance = exit_distance + max(1.0e-4f, exit_distance * 1.0e-5f);
        float2 next_uv;
        float next_depth;
        if (!ProjectTracePoint(clip_origin, clip_direction, next_distance, next_uv, next_depth)) return result;
        level = AscendTraceLevel(uv, next_uv, 0u, maximum_level);
        distance = next_distance;
    }
    return result;
}

SurfaceHit SurfaceFromTriangle(uint triangle_id, float3 position, float3 normal, float distance, uint hit_type);
float3 SceneTriangleFacingNormal(uint triangle_id, float3 incoming_direction);
bool IntersectSceneTriangle(TraceRay ray, uint triangle_id, float min_t, float max_t, out float hit_t, out float3 hit_normal);

SurfaceHit TraceScreenSurface(TraceRay ray) {
    uint2 origin_pixel = UnpackPixel(ray.origin_pixel);
    float source_depth = TraceDepth.Load(int3(origin_pixel, 0));
    float3 source_position = ReconstructWorldPosition(origin_pixel, source_depth);
    float3 source_normal = normalize(TraceNormalRoughness.Load(int3(origin_pixel, 0)).xyz);
    float3 source_view_direction = normalize(source_position - TraceFrame.camera_position.xyz);
    if (dot(source_normal, source_view_direction) > 0.0f) source_normal = -source_normal;

    TraceHit legacy = TraceScreenRay(
        ray.origin_tmin.xyz,
        ray.direction_tmax.xyz,
        origin_pixel,
        ray.source_object_id,
        source_normal,
        ray.direction_tmax.w,
        Radiance.trace_params.z,
        Radiance.trace_params.w,
        Radiance.trace_limits.x,
        Radiance.trace_limits.y
    );

    if (legacy.type != TRACE_SCREEN) return MakeSurfaceHit(legacy.type, legacy.distance);

    uint triangle_id = TracePrimitiveId.Load(int3(legacy.hit_pixel, 0));
    if (triangle_id >= Radiance.scene_counts.z) return MakeSurfaceHit(TRACE_MISS, ray.direction_tmax.w);

    float tolerance = max(Radiance.trace_params.z * 4.0f, 0.05f);
    float exact_t;
    float3 exact_normal;
    SurfaceHit hit;

    if (IntersectSceneTriangle(
            ray,
            triangle_id,
            max(ray.origin_tmin.w, legacy.distance - tolerance),
            min(ray.direction_tmax.w, legacy.distance + tolerance),
            exact_t,
            exact_normal)) {
        float3 exact_position = ray.origin_tmin.xyz + ray.direction_tmax.xyz * exact_t;
        hit = SurfaceFromTriangle(triangle_id, exact_position, exact_normal, exact_t, TRACE_SCREEN);
    } else {
        float depth = TraceDepth.Load(int3(legacy.hit_pixel, 0));
        float3 position = ReconstructWorldPosition(legacy.hit_pixel, depth);
        float3 normal = SceneTriangleFacingNormal(triangle_id, ray.direction_tmax.xyz);
        if (dot(normal, normal) <= 1.0e-8f) {
            normal = normalize(TraceNormalRoughness.Load(int3(legacy.hit_pixel, 0)).xyz);
            if (dot(normal, ray.direction_tmax.xyz) > 0.0f) normal = -normal;
        }
        hit = SurfaceFromTriangle(triangle_id, position, normal, legacy.distance, TRACE_SCREEN);
    }

    if (hit.identity.z == INVALID_INDEX) return MakeSurfaceHit(TRACE_MISS, ray.direction_tmax.w);

    uint object_index = SceneTriangles[triangle_id].meta.x;
    hit.meta.y = object_index < Radiance.scene_counts.x
        ? SceneObjects[object_index].meta.x
        : Radiance.feature_flags.y;
    hit.meta.z = PackPixel(legacy.hit_pixel);
    return hit;
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
    float depth = 0.0f;
    for (uint y = src_begin.y; y < src_end.y; ++y)
        for (uint x = src_begin.x; x < src_end.x; ++x)
            depth = max(depth, HZBSource.Load(int3(int2(x, y), 0)));
    HZBOutput[pixel] = depth;
}

// -----------------------------------------------------------------------------
// Compatibility local-SDF path.
// -----------------------------------------------------------------------------
uint LegacySDFIndex(GPUSDFModel model, uint3 p) {
    uint resolution = model.meta.y;
    return model.meta.x + p.x + resolution * (p.y + resolution * p.z);
}

float SampleLegacySDF(GPUSDFModel model, float3 local_position) {
    float3 size = max(model.bounds_max.xyz - model.bounds_min.xyz, float3(1.0e-6f, 1.0e-6f, 1.0e-6f));
    float3 uvw = (local_position - model.bounds_min.xyz) / size;
    if (any(uvw < 0.0f) || any(uvw > 1.0f)) return 1.0e6f;
    uint resolution = model.meta.y;
    float3 grid = saturate(uvw) * (float)(resolution - 1u);
    uint3 max_coord = uint3(resolution - 1u, resolution - 1u, resolution - 1u);
    uint3 p0 = min((uint3)floor(grid), max_coord);
    uint3 p1 = min(p0 + uint3(1u, 1u, 1u), max_coord);
    float3 f = frac(grid);
    float c000 = SDFVoxels[LegacySDFIndex(model, uint3(p0.x, p0.y, p0.z))];
    float c100 = SDFVoxels[LegacySDFIndex(model, uint3(p1.x, p0.y, p0.z))];
    float c010 = SDFVoxels[LegacySDFIndex(model, uint3(p0.x, p1.y, p0.z))];
    float c110 = SDFVoxels[LegacySDFIndex(model, uint3(p1.x, p1.y, p0.z))];
    float c001 = SDFVoxels[LegacySDFIndex(model, uint3(p0.x, p0.y, p1.z))];
    float c101 = SDFVoxels[LegacySDFIndex(model, uint3(p1.x, p0.y, p1.z))];
    float c011 = SDFVoxels[LegacySDFIndex(model, uint3(p0.x, p1.y, p1.z))];
    float c111 = SDFVoxels[LegacySDFIndex(model, uint3(p1.x, p1.y, p1.z))];
    float c00 = lerp(c000, c100, f.x);
    float c10 = lerp(c010, c110, f.x);
    float c01 = lerp(c001, c101, f.x);
    float c11 = lerp(c011, c111, f.x);
    return lerp(lerp(c00, c10, f.y), lerp(c01, c11, f.y), f.z);
}

bool TraceLegacySDFModel(TraceRay ray, GPUSDFModel model, float current_best, out float hit_distance) {
    float3 local_origin = TransformPoint(ray.origin_tmin.xyz, model.world_to_local);
    float3 local_direction = TransformVector(ray.direction_tmax.xyz, model.world_to_local);
    float direction_scale = length(local_direction);
    if (direction_scale <= 1.0e-8f) { hit_distance = current_best; return false; }
    float t_min, t_max;
    if (!IntersectAABB(local_origin, local_direction, model.bounds_min.xyz, model.bounds_max.xyz, t_min, t_max)) { hit_distance = current_best; return false; }
    float t = max(max(t_min, ray.origin_tmin.w), 0.0f);
    float limit = min(min(t_max, ray.direction_tmax.w), current_best);
    float3 extent = model.bounds_max.xyz - model.bounds_min.xyz;
    float cell_size = max(extent.x, max(extent.y, extent.z)) / max((float)model.meta.y, 1.0f);
    float epsilon = max(cell_size * 0.65f, 1.0e-4f);
    [loop]
    for (uint step = 0u; step < 96u && t <= limit; ++step) {
        float d = SampleLegacySDF(model, local_origin + local_direction * t);
        if (d <= epsilon) { hit_distance = t; return true; }
        t += max(d / direction_scale, epsilon * 0.25f / direction_scale);
    }
    hit_distance = current_best;
    return false;
}

bool TraceLegacySDFAny(TraceRay ray, out float hit_distance) {
    float best = ray.direction_tmax.w;
    bool found = false;
    for (uint i = 0u; i < TraceFrame.trace_limits.w; ++i) {
        float candidate;
        if (TraceLegacySDFModel(ray, SDFModels[i], best, candidate)) { best = candidate; found = true; }
    }
    hit_distance = best;
    return found;
}

// -----------------------------------------------------------------------------
// Permanent local SDF with surface identity.
// -----------------------------------------------------------------------------
uint FutureSDFIndex(GPUSDFModel model, uint3 p) {
    uint resolution = model.meta.y;
    return model.meta.x + p.x + resolution * (p.y + resolution * p.z);
}

float SampleLocalSDF(GPUSDFModel model, float3 local_position, out uint surface_id) {
    surface_id = INVALID_INDEX;
    float3 size = max(model.bounds_max.xyz - model.bounds_min.xyz, float3(1.0e-6f, 1.0e-6f, 1.0e-6f));
    float3 uvw = (local_position - model.bounds_min.xyz) / size;
    if (any(uvw < 0.0f) || any(uvw > 1.0f)) return 1.0e6f;
    uint resolution = model.meta.y;
    float3 grid = saturate(uvw) * (float)(resolution - 1u);
    uint3 p = min((uint3)round(grid), uint3(resolution - 1u, resolution - 1u, resolution - 1u));
    uint index = FutureSDFIndex(model, p);
    surface_id = LocalSDFSurfaceIds[index];
    return LocalSDFVoxels[index];
}

float3 BarycentricCoordinates(float3 p, float3 a, float3 b, float3 c) {
    float3 v0 = b - a;
    float3 v1 = c - a;
    float3 v2 = p - a;
    float d00 = dot(v0, v0);
    float d01 = dot(v0, v1);
    float d11 = dot(v1, v1);
    float d20 = dot(v2, v0);
    float d21 = dot(v2, v1);
    float denom = d00 * d11 - d01 * d01;
    if (abs(denom) < 1.0e-12f) return float3(1.0f, 0.0f, 0.0f);
    float v = (d11 * d20 - d01 * d21) / denom;
    float w = (d00 * d21 - d01 * d20) / denom;
    float u = 1.0f - v - w;
    float3 bary = max(float3(u, v, w), 0.0f);
    float sum = bary.x + bary.y + bary.z;
    return sum > 1.0e-8f ? bary / sum : float3(1.0f, 0.0f, 0.0f);
}

SurfaceHit SurfaceFromTriangle(uint triangle_id, float3 position, float3 normal, float distance, uint hit_type) {
    SurfaceHit hit = MakeSurfaceHit(hit_type, distance);
    if (triangle_id >= Radiance.scene_counts.z) return hit;
    GPUSceneTriangle tri = SceneTriangles[triangle_id];
    uint object_index = tri.meta.x;
    uint object_id = object_index < Radiance.scene_counts.x ? SceneObjects[object_index].draw.z : INVALID_INDEX;
    float3 a = tri.p0.xyz;
    float3 b = tri.p1.xyz;
    float3 c = tri.p2.xyz;
    if (object_index < Radiance.scene_counts.x) {
        float4x4 world = SceneObjects[object_index].world;
        a = TransformPoint(a, world);
        b = TransformPoint(b, world);
        c = TransformPoint(c, world);
    }
    float3 bary = BarycentricCoordinates(position, a, b, c);
    float2 uv0 = tri.uv01.xy;
    float2 uv1 = tri.uv01.zw;
    float2 uv2 = tri.uv2_area.xy;
    float2 uv = uv0 * bary.x + uv1 * bary.y + uv2 * bary.z;
    hit.position_distance = float4(position, distance);
    hit.normal_confidence = float4(normalize(normal), 1.0f);
    hit.uv_bary = float4(uv, bary.y, bary.z);
    hit.identity = uint4(object_id, tri.meta.y, triangle_id, hit_type);
    hit.meta.x = tri.meta.z;
    return hit;
}


float3 SceneTriangleFacingNormal(uint triangle_id, float3 incoming_direction) {
    if (triangle_id >= Radiance.scene_counts.z) return 0.0f;
    GPUSceneTriangle tri = SceneTriangles[triangle_id];
    uint object_index = tri.meta.x;
    if (object_index >= Radiance.scene_counts.x) return 0.0f;

    row_major float4x4 world = SceneObjects[object_index].world;
    float3 a = TransformPoint(tri.p0.xyz, world);
    float3 b = TransformPoint(tri.p1.xyz, world);
    float3 c = TransformPoint(tri.p2.xyz, world);
    float3 n = cross(b - a, c - a);
    float n2 = dot(n, n);
    if (n2 <= 1.0e-12f) return 0.0f;
    n *= rsqrt(n2);
    if (dot(n, incoming_direction) > 0.0f) n = -n;
    return n;
}

bool IntersectSceneTriangle(
    TraceRay ray,
    uint triangle_id,
    float min_t,
    float max_t,
    out float hit_t,
    out float3 hit_normal
) {
    hit_t = max_t;
    hit_normal = 0.0f;
    if (triangle_id >= Radiance.scene_counts.z) return false;

    GPUSceneTriangle tri = SceneTriangles[triangle_id];
    uint object_index = tri.meta.x;
    if (object_index >= Radiance.scene_counts.x) return false;

    row_major float4x4 world = SceneObjects[object_index].world;
    float3 a = TransformPoint(tri.p0.xyz, world);
    float3 b = TransformPoint(tri.p1.xyz, world);
    float3 c = TransformPoint(tri.p2.xyz, world);
    float3 e1 = b - a;
    float3 e2 = c - a;
    float3 pvec = cross(ray.direction_tmax.xyz, e2);
    float det = dot(e1, pvec);
    if (abs(det) <= 1.0e-9f) return false;

    float inv_det = rcp(det);
    float3 tvec = ray.origin_tmin.xyz - a;
    float u = dot(tvec, pvec) * inv_det;
    if (u < -1.0e-4f || u > 1.0001f) return false;

    float3 qvec = cross(tvec, e1);
    float v = dot(ray.direction_tmax.xyz, qvec) * inv_det;
    if (v < -1.0e-4f || u + v > 1.0001f) return false;

    float t = dot(e2, qvec) * inv_det;
    if (t < min_t || t > max_t) return false;

    float3 n = cross(e1, e2);
    float n2 = dot(n, n);
    if (n2 <= 1.0e-12f) return false;
    n *= rsqrt(n2);
    if (dot(n, ray.direction_tmax.xyz) > 0.0f) n = -n;

    hit_t = t;
    hit_normal = n;
    return true;
}

bool RefineLocalSDFSurface(
    GPUSDFModel model,
    TraceRay ray,
    float3 local_position,
    float sdf_t,
    float epsilon,
    float direction_scale,
    float limit,
    out SurfaceHit hit
) {
    hit = MakeSurfaceHit(TRACE_MISS, limit);

    float3 size = max(model.bounds_max.xyz - model.bounds_min.xyz, float3(1.0e-6f, 1.0e-6f, 1.0e-6f));
    float3 uvw = (local_position - model.bounds_min.xyz) / size;
    if (any(uvw < 0.0f) || any(uvw > 1.0f)) return false;

    uint resolution = model.meta.y;
    float3 grid = saturate(uvw) * (float)(resolution - 1u);
    int3 center = int3(round(grid));
    int max_coord = (int)resolution - 1;

    float window = max(epsilon * 3.0f / max(direction_scale, 1.0e-8f), 1.0e-3f);
    float min_t = max(ray.origin_tmin.w, sdf_t - window);
    float max_t = min(limit, sdf_t + window);
    if (max_t < min_t) return false;

    bool found = false;
    float best_t = max_t;
    float3 best_normal = 0.0f;
    uint best_surface = INVALID_INDEX;

    [loop]
    for (int z = -1; z <= 1; ++z) {
        [loop]
        for (int y = -1; y <= 1; ++y) {
            [loop]
            for (int x = -1; x <= 1; ++x) {
                int3 q = clamp(center + int3(x, y, z), int3(0, 0, 0), int3(max_coord, max_coord, max_coord));
                uint surface_id = LocalSDFSurfaceIds[FutureSDFIndex(model, uint3(q))];
                if (surface_id == INVALID_INDEX) continue;

                float candidate_t;
                float3 candidate_normal;
                if (IntersectSceneTriangle(ray, surface_id, min_t, best_t, candidate_t, candidate_normal)) {
                    found = true;
                    best_t = candidate_t;
                    best_normal = candidate_normal;
                    best_surface = surface_id;
                }
            }
        }
    }

    if (!found || best_surface == INVALID_INDEX) return false;

    float3 world_position = ray.origin_tmin.xyz + ray.direction_tmax.xyz * best_t;
    hit = SurfaceFromTriangle(best_surface, world_position, best_normal, best_t, TRACE_SDF);
    if (hit.identity.x == INVALID_INDEX) hit.identity.x = model.meta.z;
    hit.meta.y = model.version.x;
    return hit.identity.z != INVALID_INDEX;
}

float3 LocalSDFNormal(GPUSDFModel model, float3 p) {
    float3 extent = max(model.bounds_max.xyz - model.bounds_min.xyz, float3(1.0e-5f, 1.0e-5f, 1.0e-5f));
    float e = max(max(extent.x, max(extent.y, extent.z)) / max((float)model.meta.y, 1.0f) * 0.5f, 1.0e-4f);
    uint ignored;
    float dx = SampleLocalSDF(model, p + float3(e, 0.0f, 0.0f), ignored) - SampleLocalSDF(model, p - float3(e, 0.0f, 0.0f), ignored);
    float dy = SampleLocalSDF(model, p + float3(0.0f, e, 0.0f), ignored) - SampleLocalSDF(model, p - float3(0.0f, e, 0.0f), ignored);
    float dz = SampleLocalSDF(model, p + float3(0.0f, 0.0f, e), ignored) - SampleLocalSDF(model, p - float3(0.0f, 0.0f, e), ignored);
    return normalize(float3(dx, dy, dz));
}

bool TraceLocalSDFModel(TraceRay ray, uint model_index, float current_best, out SurfaceHit hit) {
    hit = MakeSurfaceHit(TRACE_MISS, current_best);
    if (model_index >= Radiance.sdf_counts.x) return false;
    GPUSDFModel model = LocalSDFModels[model_index];
    float3 local_origin = TransformPoint(ray.origin_tmin.xyz, model.world_to_local);
    float3 local_direction = TransformVector(ray.direction_tmax.xyz, model.world_to_local);
    float direction_scale = length(local_direction);
    if (direction_scale <= 1.0e-8f) return false;
    float t_min, t_max;
    if (!IntersectAABB(local_origin, local_direction, model.bounds_min.xyz, model.bounds_max.xyz, t_min, t_max)) return false;
    float t = max(max(t_min, ray.origin_tmin.w), 0.0f);
    float limit = min(min(t_max, ray.direction_tmax.w), current_best);
    float3 extent = model.bounds_max.xyz - model.bounds_min.xyz;
    float cell_size = max(extent.x, max(extent.y, extent.z)) / max((float)model.meta.y, 1.0f);
    float epsilon = max(cell_size * 0.65f, 1.0e-4f);
    uint max_steps = max(Radiance.trace_limits.z, 1u);
    [loop]
    for (uint step = 0u; step < max_steps && t <= limit; ++step) {
        float3 lp = local_origin + local_direction * t;
        uint surface_id;
        float d = SampleLocalSDF(model, lp, surface_id);
        if (d <= epsilon) {
            float near_limit = ray.origin_tmin.w + epsilon * 1.5f;
            if (t <= near_limit) {
                t += max(epsilon * 1.5f / direction_scale, 1.0e-4f);
                continue;
            }

            SurfaceHit refined;
            if (RefineLocalSDFSurface(model, ray, lp, t, epsilon, direction_scale, limit, refined)) {
                hit = refined;
                return true;
            }

            // The coarse voxel identified a nearby surface, but not one that the
            // world-space ray actually intersects. Keep tracing instead of
            // shading the voxel approximation.
            t += max(epsilon * 0.5f / direction_scale, 1.0e-4f);
            continue;
        }
        t += max(d / direction_scale, epsilon * 0.25f / direction_scale);
    }
    return false;
}

bool TraceAllLocalSDFs(TraceRay ray, inout SurfaceHit best_hit) {
    bool found = false;
    float best = best_hit.position_distance.w;
    for (uint i = 0u; i < Radiance.sdf_counts.x; ++i) {
        if (FeatureEnabled(RADIANCE_FEATURE_DYNAMIC_GRID) && LocalSDFModels[i].meta.w != 0u) continue;
        SurfaceHit candidate;
        if (TraceLocalSDFModel(ray, i, best, candidate) && candidate.position_distance.w < best) {
            best = candidate.position_distance.w;
            best_hit = candidate;
            found = true;
        }
    }
    return found;
}

// -----------------------------------------------------------------------------
// Dynamic object grid.
// -----------------------------------------------------------------------------
uint Flatten3D(uint3 p, uint3 dim) { return p.x + dim.x * (p.y + dim.y * p.z); }

bool DynamicGridCellAt(float3 world_position, out uint3 cell) {
    cell = uint3(0u, 0u, 0u);
    uint3 dim = Radiance.dynamic_grid.xyz;
    float cell_size = Radiance.dynamic_grid_origin_cell.w;
    if (any(dim == 0u) || cell_size <= 0.0f) return false;
    float3 f = (world_position - Radiance.dynamic_grid_origin_cell.xyz) / cell_size;
    if (any(f < 0.0f) || any(f >= float3(dim))) return false;
    cell = (uint3)floor(f);
    return true;
}

bool TraceDynamicGrid(TraceRay ray, inout SurfaceHit best_hit) {
    if (!FeatureEnabled(RADIANCE_FEATURE_DYNAMIC_GRID) || Radiance.sdf_counts.y == 0u) return false;
    uint3 dim = Radiance.dynamic_grid.xyz;
    float cell_size = Radiance.dynamic_grid_origin_cell.w;
    if (any(dim == 0u) || cell_size <= 0.0f) return false;

    float3 grid_min = Radiance.dynamic_grid_origin_cell.xyz;
    float3 grid_max = grid_min + float3(dim) * cell_size;
    float t_enter, t_exit;
    if (!IntersectAABB(ray.origin_tmin.xyz, ray.direction_tmax.xyz, grid_min, grid_max, t_enter, t_exit)) return false;

    float t = max(max(t_enter, ray.origin_tmin.w), 0.0f);
    float limit = min(min(t_exit, ray.direction_tmax.w), best_hit.position_distance.w);
    if (t > limit) return false;

    float sample_t = min(t + max(cell_size * 1.0e-5f, 1.0e-5f), limit);
    float3 position = ray.origin_tmin.xyz + ray.direction_tmax.xyz * sample_t;
    int3 cell = int3(floor((position - grid_min) / cell_size));
    cell = clamp(cell, int3(0, 0, 0), int3(dim) - int3(1, 1, 1));

    int3 step_dir = int3(
        ray.direction_tmax.x >= 0.0f ? 1 : -1,
        ray.direction_tmax.y >= 0.0f ? 1 : -1,
        ray.direction_tmax.z >= 0.0f ? 1 : -1
    );
    float3 cell_min = grid_min + float3(cell) * cell_size;
    float3 boundary = float3(
        step_dir.x > 0 ? cell_min.x + cell_size : cell_min.x,
        step_dir.y > 0 ? cell_min.y + cell_size : cell_min.y,
        step_dir.z > 0 ? cell_min.z + cell_size : cell_min.z
    );
    float3 t_axis = float3(1.0e30f, 1.0e30f, 1.0e30f);
    float3 t_delta = float3(1.0e30f, 1.0e30f, 1.0e30f);
    if (abs(ray.direction_tmax.x) > 1.0e-8f) {
        t_axis.x = (boundary.x - ray.origin_tmin.x) / ray.direction_tmax.x;
        t_delta.x = cell_size / abs(ray.direction_tmax.x);
    }
    if (abs(ray.direction_tmax.y) > 1.0e-8f) {
        t_axis.y = (boundary.y - ray.origin_tmin.y) / ray.direction_tmax.y;
        t_delta.y = cell_size / abs(ray.direction_tmax.y);
    }
    if (abs(ray.direction_tmax.z) > 1.0e-8f) {
        t_axis.z = (boundary.z - ray.origin_tmin.z) / ray.direction_tmax.z;
        t_delta.z = cell_size / abs(ray.direction_tmax.z);
    }

    bool found = false;
    uint max_steps = min(max(Radiance.trace_limits.w, 1u), MAX_DYNAMIC_GRID_STEPS);
    [loop]
    for (uint step = 0u; step < max_steps; ++step) {
        uint3 ucell = uint3(cell);
        uint flat = Flatten3D(ucell, dim);
        if (flat >= Radiance.sdf_counts.y) break;
        GPUDynamicGridCell grid_cell = DynamicGridCells[flat];
        uint offset = grid_cell.range_flags.x;
        uint count = grid_cell.range_flags.y;
        for (uint i = 0u; i < count; ++i) {
            uint index = offset + i;
            if (index >= Radiance.dynamic_grid.w) break;
            uint model_index = DynamicGridIndices[index];
            SurfaceHit candidate;
            if (TraceLocalSDFModel(ray, model_index, best_hit.position_distance.w, candidate) && candidate.position_distance.w < best_hit.position_distance.w) {
                best_hit = candidate;
                found = true;
            }
        }

        float next_t = min(t_axis.x, min(t_axis.y, t_axis.z));
        if (found && best_hit.position_distance.w <= next_t) break;
        if (next_t > limit) break;

        if (t_axis.x <= t_axis.y && t_axis.x <= t_axis.z) {
            cell.x += step_dir.x;
            if (cell.x < 0 || cell.x >= (int)dim.x) break;
            t_axis.x += t_delta.x;
        } else if (t_axis.y <= t_axis.z) {
            cell.y += step_dir.y;
            if (cell.y < 0 || cell.y >= (int)dim.y) break;
            t_axis.y += t_delta.y;
        } else {
            cell.z += step_dir.z;
            if (cell.z < 0 || cell.z >= (int)dim.z) break;
            t_axis.z += t_delta.z;
        }
    }
    return found;
}

// -----------------------------------------------------------------------------
// Sparse global SDF clipmaps.
// -----------------------------------------------------------------------------
bool GlobalClipContains(GPUGlobalSDFClipmap clip, float3 p) {
    float3 d = abs(p - clip.center_extent.xyz);
    return all(d <= clip.center_extent.www);
}

bool GlobalBrickAddress(GPUGlobalSDFClipmap clip, float3 p, out uint brick_index, out uint3 voxel, out float voxel_size) {
    brick_index = INVALID_INDEX;
    voxel = uint3(0u, 0u, 0u);
    voxel_size = clip.voxel_brick.x;
    if (!GlobalClipContains(clip, p) || voxel_size <= 0.0f) return false;
    float3 min_corner = clip.center_extent.xyz - clip.center_extent.www;
    float brick_world = clip.voxel_brick.y;
    uint brick_res = max(clip.data.x, 1u);
    uint3 grid_dim = clip.grid.xyz;
    if (brick_world <= 0.0f || any(grid_dim == 0u)) return false;
    float3 rel = p - min_corner;
    uint3 brick_coord = min((uint3)floor(rel / brick_world), grid_dim - uint3(1u, 1u, 1u));
    uint page = clip.grid.w + Flatten3D(brick_coord, grid_dim);
    brick_index = GlobalSDFPageTable[page];
    if (brick_index == INVALID_INDEX) return false;
    float3 inside = frac(rel / brick_world);
    voxel = min((uint3)floor(inside * (float)brick_res), uint3(brick_res - 1u, brick_res - 1u, brick_res - 1u));
    return true;
}

float SampleGlobalSDF(float3 p, out uint surface_id, out float voxel_size) {
    surface_id = INVALID_INDEX;
    voxel_size = max(Radiance.global_sdf_params.y, 0.01f);
    uint clip_count = min(Radiance.sdf_counts.z, MAX_GLOBAL_SDF_CLIPMAPS);
    for (uint i = 0u; i < clip_count; ++i) {
        GPUGlobalSDFClipmap clip = GlobalSDFClipmaps[i];
        uint brick_index;
        uint3 voxel;
        if (!GlobalBrickAddress(clip, p, brick_index, voxel, voxel_size)) continue;
        uint brick_res = max(clip.data.x, 1u);
        uint brick_stride = max(clip.data.y, brick_res * brick_res * brick_res);
        uint local = voxel.x + brick_res * (voxel.y + brick_res * voxel.z);
        uint index = clip.data.z + brick_index * brick_stride + local;
        surface_id = GlobalSDFSurfaceIds[index];
        return GlobalSDFBricks[index];
    }
    return max(Radiance.global_sdf_params.y, 0.25f);
}

bool RefineGlobalSDFSurface(
    TraceRay ray,
    float3 position,
    float sdf_t,
    float epsilon,
    float voxel_size,
    float limit,
    out SurfaceHit hit
) {
    hit = MakeSurfaceHit(TRACE_MISS, limit);
    float window = max(max(epsilon * 3.0f, voxel_size * 2.0f), 1.0e-3f);
    float min_t = max(ray.origin_tmin.w, sdf_t - window);
    float max_t = min(limit, sdf_t + window);
    if (max_t < min_t) return false;

    bool found = false;
    float best_t = max_t;
    float3 best_normal = 0.0f;
    uint best_surface = INVALID_INDEX;
    float step_size = max(voxel_size, 1.0e-4f);
    [loop]
    for (int z = -1; z <= 1; ++z) {
        [loop]
        for (int y = -1; y <= 1; ++y) {
            [loop]
            for (int x = -1; x <= 1; ++x) {
                float3 sample_position = position + float3((float)x, (float)y, (float)z) * step_size;
                uint surface_id;
                float sampled_voxel_size;
                SampleGlobalSDF(sample_position, surface_id, sampled_voxel_size);
                if (surface_id == INVALID_INDEX) continue;
                float candidate_t;
                float3 candidate_normal;
                if (IntersectSceneTriangle(ray, surface_id, min_t, best_t, candidate_t, candidate_normal)) {
                    found = true;
                    best_t = candidate_t;
                    best_normal = candidate_normal;
                    best_surface = surface_id;
                }
            }
        }
    }
    if (!found || best_surface == INVALID_INDEX) return false;
    float3 world_position = ray.origin_tmin.xyz + ray.direction_tmax.xyz * best_t;
    hit = SurfaceFromTriangle(best_surface, world_position, best_normal, best_t, TRACE_GLOBAL_SDF);
    GPUSceneTriangle tri = SceneTriangles[best_surface];
    uint object_index = tri.meta.x;
    hit.meta.y = object_index < Radiance.scene_counts.x ? SceneObjects[object_index].meta.x : Radiance.feature_flags.y;
    return hit.identity.z != INVALID_INDEX;
}

bool TraceGlobalSDF(TraceRay ray, inout SurfaceHit best_hit) {
    if (!FeatureEnabled(RADIANCE_FEATURE_GLOBAL_SDF) || Radiance.sdf_counts.z == 0u) return false;
    float t = max(ray.origin_tmin.w, 0.0f);
    float limit = min(ray.direction_tmax.w, best_hit.position_distance.w);
    uint max_steps = max(Radiance.trace_limits.w, 1u);
    float epsilon_scale = max(Radiance.global_sdf_params.z, 0.5f);
    [loop]
    for (uint step = 0u; step < max_steps && t <= limit; ++step) {
        float3 p = ray.origin_tmin.xyz + ray.direction_tmax.xyz * t;
        uint surface_id;
        float voxel_size;
        float d = SampleGlobalSDF(p, surface_id, voxel_size);
        float epsilon = max(voxel_size * epsilon_scale, 1.0e-3f);
        if (d <= epsilon) {
            float near_limit = ray.origin_tmin.w + epsilon * 1.5f;
            if (t <= near_limit) {
                t += max(epsilon * 1.5f, 1.0e-4f);
                continue;
            }
            SurfaceHit refined;
            if (RefineGlobalSDFSurface(ray, p, t, epsilon, voxel_size, limit, refined)) {
                best_hit = refined;
                return true;
            }
            t += max(epsilon * 0.5f, 1.0e-4f);
            continue;
        }
        t += max(d, epsilon * 0.25f);
    }
    return false;
}

// -----------------------------------------------------------------------------
// Unified ray query.
// -----------------------------------------------------------------------------
SurfaceHit TraceUnifiedRay(TraceRay ray, bool allow_screen) {
    SurfaceHit best = MakeSurfaceHit(TRACE_MISS, ray.direction_tmax.w);
    if (allow_screen) {
        SurfaceHit screen = TraceScreenSurface(ray);
        if (screen.identity.w == TRACE_SCREEN) best = screen;
    }
    TraceDynamicGrid(ray, best);
    bool global_resolved = TraceGlobalSDF(ray, best);
    if (!global_resolved) TraceAllLocalSDFs(ray, best);
    return best;
}

bool TraceUnifiedOcclusion(TraceRay ray) {
    SurfaceHit hit = TraceUnifiedRay(ray, false);
    return hit.identity.w != TRACE_MISS && hit.position_distance.w < ray.direction_tmax.w;
}

// -----------------------------------------------------------------------------
// Triangle/material evaluation and explicit emissive area lights.
// -----------------------------------------------------------------------------
void EvaluateTriangleWorld(uint triangle_id, out float3 a, out float3 b, out float3 c, out GPUSceneTriangle tri) {
    tri = SceneTriangles[triangle_id];
    uint object_index = tri.meta.x;
    row_major float4x4 world = SceneObjects[object_index].world;
    a = TransformPoint(tri.p0.xyz, world);
    b = TransformPoint(tri.p1.xyz, world);
    c = TransformPoint(tri.p2.xyz, world);
}

uint SelectEmissiveTriangle(float u) {
    uint count = Radiance.scene_counts.w;
    if (count == 0u) return INVALID_INDEX;
    uint lo = 0u;
    uint hi = count - 1u;
    [loop]
    while (lo < hi) {
        uint mid = (lo + hi) >> 1u;
        if (u <= EmissiveTriangles[mid].sampling.x) hi = mid;
        else lo = mid + 1u;
    }
    return lo;
}

float3 SampleTriangleBarycentric(float2 u, out float3 bary) {
    float su = sqrt(saturate(u.x));
    bary = float3(1.0f - su, su * (1.0f - u.y), su * u.y);
    return bary;
}

float3 SampleEmissivePoint(uint emitter_index, float2 u, out float3 normal, out float3 emitted, out float pdf_area, out uint triangle_id) {
    GPUEmissiveTriangle emitter = EmissiveTriangles[emitter_index];
    triangle_id = emitter.meta.x;
    float3 a, b, c;
    GPUSceneTriangle tri;
    EvaluateTriangleWorld(triangle_id, a, b, c, tri);
    float3 bary;
    SampleTriangleBarycentric(u, bary);
    float3 p = a * bary.x + b * bary.y + c * bary.z;
    normal = normalize(cross(b - a, c - a));
    emitted = emitter.radiance_area.rgb;
    float area = max(emitter.radiance_area.w, 1.0e-8f);
    float selection_pdf = max(emitter.sampling.y, 1.0e-8f);
    pdf_area = selection_pdf / area;
    return p;
}

float3 EvaluateEmissiveSampleForMaterial(float3 surface_position, float3 surface_normal, uint source_object_id, uint source_material_id, uint seed) {
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
    TraceRay shadow = MakeTraceRay(surface_position + surface_normal * bias, L, 1.0e-4f, max(distance - bias, 1.0e-4f), TRACE_RAY_SHADOW, 0u, uint2(0u, 0u), source_object_id);
    if (TraceUnifiedOcclusion(shadow)) return 0.0f;
    float geometry = cos_surface * cos_light / max(distance_sq, 1.0e-6f);
    return emitted * geometry / max(pdf_area * PI, 1.0e-8f);
}

float3 EvaluateEmissiveSample(float3 surface_position, float3 surface_normal, uint source_object_id, uint2 source_pixel, uint seed) {
    uint source_material_id = TraceMaterialId.Load(int3(source_pixel, 0));
    return EvaluateEmissiveSampleForMaterial(surface_position, surface_normal, source_object_id, source_material_id, seed);
}

float3 SkyRadiance(float3 direction) {
    float3 result = 0.0f;
    for (uint i = 0u; i < TraceFrame.trace_limits.z; ++i) {
        GPULight light = Lights[i];
        if ((uint)(light.direction_type.w + 0.5f) != 3u) continue;
        float t = saturate(direction.y * 0.5f + 0.5f);
        result += lerp(light.spot_angles.xyz, light.color_intensity.rgb, t) * light.color_intensity.w;
    }
    return result;
}

float3 FutureSkyRadiance(float3 direction) {
    float3 result = 0.0f;
    for (uint i = 0u; i < Radiance.sdf_counts.w; ++i) {
        GPULight light = SceneLights[i];
        if ((uint)(light.direction_type.w + 0.5f) != 3u) continue;
        float t = saturate(direction.y * 0.5f + 0.5f);
        result += lerp(light.spot_angles.xyz, light.color_intensity.rgb, t) * light.color_intensity.w;
    }
    return result;
}

float3 EvaluateAnalyticLightDiffuse(GPULight light, float3 albedo, float3 position, float3 normal) {
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

float3 SurfaceReflectedRadiance(SurfaceHit hit);
float3 WorldIndirectReflected(SurfaceHit hit, bool stable_previous);

// -----------------------------------------------------------------------------
// Screen-space direct-radiance reuse.
// -----------------------------------------------------------------------------
float3 ReflectedDirectAtPixel(uint2 pixel) {
    float3 direct = TraceDirectRadiance.Load(int3(pixel, 0)).rgb;
    float3 emissive = TraceEmissive.Load(int3(pixel, 0)).rgb;
    return max(direct - emissive, 0.0f);
}

// -----------------------------------------------------------------------------
// Screen probes / compacted miss queue.
// -----------------------------------------------------------------------------
uint LegacyProbeHash(uint2 probe) { return HashCombine(probe.x * 0x8da6b343u, probe.y * 0xd8163841u); }

float3 LegacyCosineHemisphereDirection(uint ray_index, uint2 probe) {
    float2 u = Hammersley(ray_index, SCREEN_PROBE_RAY_COUNT, LegacyProbeHash(probe));
    return CosineHemisphere(u);
}

TraceRay BuildDiffuseProbeRay(uint2 ray_pixel) {
    uint ray_width, ray_height;
    ProbeRadianceOutput.GetDimensions(ray_width, ray_height);
    uint2 probe = ray_pixel / SCREEN_PROBE_DIRECTION_SIZE;
    uint2 direction_cell = ray_pixel % SCREEN_PROBE_DIRECTION_SIZE;
    uint ray_index = direction_cell.x + direction_cell.y * SCREEN_PROBE_DIRECTION_SIZE;
    uint2 resolution = uint2(TraceFrame.resolution.xy);
    uint2 source_pixel = min(probe * SCREEN_PROBE_TILE_SIZE + uint2(SCREEN_PROBE_TILE_SIZE / 2u, SCREEN_PROBE_TILE_SIZE / 2u), resolution - uint2(1u, 1u));
    uint destination = ray_pixel.x + ray_pixel.y * ray_width;
    if (ray_index >= SCREEN_PROBE_RAY_COUNT)
        return MakeTraceRay(0.0f, float3(0.0f, 1.0f, 0.0f), 0.0f, 0.0f, TRACE_RAY_INACTIVE, destination, source_pixel, 0u);
    float depth = TraceDepth.Load(int3(source_pixel, 0));
    if (depth <= 0.0f) return MakeTraceRay(0.0f, float3(0.0f, 1.0f, 0.0f), 0.0f, 0.0f, TRACE_RAY_INACTIVE, destination, source_pixel, 0u);
    float3 position = ReconstructWorldPosition(source_pixel, depth);
    float3 normal = normalize(TraceNormalRoughness.Load(int3(source_pixel, 0)).xyz);
    float3 view_direction = normalize(position - TraceFrame.camera_position.xyz);
    if (dot(normal, view_direction) > 0.0f) normal = -normal;
    float3 tangent, bitangent;
    BuildNormalBasis(normal, tangent, bitangent);
    float3 local_direction = LegacyCosineHemisphereDirection(ray_index, probe);
    float3 direction = normalize(tangent * local_direction.x + bitangent * local_direction.y + normal * local_direction.z);
    uint object_id = TraceObjectId.Load(int3(source_pixel, 0));
    float bias = TraceFrame.trace_params.z;
    return MakeTraceRay(position + normal * bias, direction, bias * 2.0f, TraceFrame.trace_params.x, TRACE_RAY_DIFFUSE, destination, source_pixel, object_id);
}


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

[numthreads(8, 8, 1)]
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
void CS_ResetTraceQueue(uint3 dispatch_id : SV_DispatchThreadID) {
    MissCount[0] = 0u;
    TraceDispatchArgs[0] = 0u;
    TraceDispatchArgs[1] = 1u;
    TraceDispatchArgs[2] = 1u;
}

[numthreads(8, 8, 1)]
void CS_CompactTraceMisses(uint3 dispatch_id : SV_DispatchThreadID) {
    uint ray_width, ray_height;
    ProbeRadianceOutput.GetDimensions(ray_width, ray_height);
    uint2 ray_pixel = dispatch_id.xy;
    if (ray_pixel.x >= ray_width || ray_pixel.y >= ray_height) return;
    uint index = ray_pixel.x + ray_pixel.y * ray_width;
    if (ScreenTraceHits[index].type != TRACE_MISS) return;
    TraceRay ray = BuildDiffuseProbeRay(ray_pixel);
    if (ray.type == TRACE_RAY_INACTIVE) return;
    uint queue_index;
    InterlockedAdd(MissCount[0], 1u, queue_index);
    MissQueue[queue_index] = ray;
}

[numthreads(1, 1, 1)]
void CS_BuildTraceDispatchArgs(uint3 dispatch_id : SV_DispatchThreadID) {
    TraceDispatchArgs[0] = (MissCount[0] + 63u) / 64u;
    TraceDispatchArgs[1] = 1u;
    TraceDispatchArgs[2] = 1u;
}

uint LegacySurfaceModelIndex(uint object_id) {
    for (uint i = 0u; i < TraceFrame.trace_limits.w; ++i) if (SDFModels[i].meta.z == object_id) return i;
    return INVALID_INDEX;
}

uint LegacySurfaceNormalAxis(float3 normal) {
    float3 a = abs(normal);
    if (a.x >= a.y && a.x >= a.z) return 0u;
    if (a.y >= a.z) return 1u;
    return 2u;
}

uint3 LegacySurfaceCell(GPUSDFModel model, float3 local_position) {
    float3 size = max(model.bounds_max.xyz - model.bounds_min.xyz, float3(1.0e-6f, 1.0e-6f, 1.0e-6f));
    float3 uvw = saturate((local_position - model.bounds_min.xyz) / size);
    uint grid = max(model.meta.y * 2u, 8u);
    uint3 limit = uint3(grid - 1u, grid - 1u, grid - 1u);
    return min((uint3)floor(uvw * (float)grid), limit);
}

float3 LegacySDFWorldNormal(GPUSDFModel model, float3 local_position) {
    float3 size = max(model.bounds_max.xyz - model.bounds_min.xyz, float3(1.0e-6f, 1.0e-6f, 1.0e-6f));
    float e = max(max(size.x, max(size.y, size.z)) / max((float)model.meta.y, 1.0f) * 0.5f, 1.0e-4f);
    float dx = SampleLegacySDF(model, local_position + float3(e, 0.0f, 0.0f)) - SampleLegacySDF(model, local_position - float3(e, 0.0f, 0.0f));
    float dy = SampleLegacySDF(model, local_position + float3(0.0f, e, 0.0f)) - SampleLegacySDF(model, local_position - float3(0.0f, e, 0.0f));
    float dz = SampleLegacySDF(model, local_position + float3(0.0f, 0.0f, e)) - SampleLegacySDF(model, local_position - float3(0.0f, 0.0f, e));
    float3 local_normal = normalize(float3(dx, dy, dz));
    return normalize(mul(float4(local_normal, 0.0f), transpose(model.world_to_local)).xyz);
}

uint LegacySurfaceHash(uint object_id, uint revision, GPUSDFModel model, float3 local_position, float3 world_normal) {
    uint3 cell = LegacySurfaceCell(model, local_position);
    uint h = object_id * 747796405u + revision * 2891336453u + LegacySurfaceNormalAxis(world_normal) * 2246822519u;
    h ^= cell.x * 277803737u;
    h ^= cell.y * 1597334677u;
    h ^= cell.z * 3812015801u;
    return Hash32(h) | 1u;
}

uint LegacySurfaceCacheWriteSlot(uint key) {
    const uint mask = SURFACE_CACHE_CAPACITY - 1u;
    [unroll]
    for (uint probe = 0u; probe < 4u; ++probe) {
        uint slot = (key + probe) & mask;
        uint previous;
        InterlockedCompareExchange(LegacySurfaceCacheKeys[slot], 0u, key, previous);
        if (previous == 0u || previous == key) return slot;
    }
    uint slot = key & mask;
    uint previous;
    InterlockedExchange(LegacySurfaceCacheKeys[slot], key, previous);
    return slot;
}

bool LegacySurfaceCacheLookup(GPUSDFModel model, float3 world_position, out float3 radiance) {
    float3 local_position = TransformPoint(world_position, model.world_to_local);
    float3 world_normal = LegacySDFWorldNormal(model, local_position);
    uint key = LegacySurfaceHash(model.meta.z, model.version.x, model, local_position, world_normal);
    const uint mask = SURFACE_CACHE_CAPACITY - 1u;
    uint grid = max(model.meta.y * 2u, 8u);
    float3 size = max(model.bounds_max.xyz - model.bounds_min.xyz, float3(1.0e-6f, 1.0e-6f, 1.0e-6f));
    float3 cell_extent = size / (float)grid;
    [unroll]
    for (uint probe = 0u; probe < 4u; ++probe) {
        uint slot = (key + probe) & mask;
        uint stored_key = LegacySurfaceCacheKeys[slot];
        if (stored_key == 0u) break;
        if (stored_key != key) continue;
        LegacySurfaceCacheEntry entry = LegacySurfaceCacheEntries[slot];
        if (entry.object_id != model.meta.z || entry.revision != model.version.x) continue;
        if (LegacySurfaceNormalAxis(entry.normal.xyz) != LegacySurfaceNormalAxis(world_normal)) continue;
        float3 cell_delta = abs(entry.position.xyz - local_position) / max(cell_extent, float3(1.0e-6f, 1.0e-6f, 1.0e-6f));
        if (any(cell_delta > 1.5f)) continue;
        radiance = entry.direct_radiance.rgb + entry.indirect_radiance.rgb;
        return entry.confidence != 0u;
    }
    radiance = 0.0f;
    return false;
}

[numthreads(64, 1, 1)]
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
void CS_SurfaceCacheUpdate(uint3 dispatch_id : SV_DispatchThreadID) {
    uint2 pixel = dispatch_id.xy;
    uint2 resolution = uint2(TraceFrame.resolution.xy);
    if (any(pixel >= resolution)) return;
    float depth = TraceDepth.Load(int3(pixel, 0));
    if (depth <= 0.0f) return;
    uint object_id = TraceObjectId.Load(int3(pixel, 0));
    if (object_id == 0u) return;
    uint model_index = LegacySurfaceModelIndex(object_id);
    if (model_index == INVALID_INDEX) return;
    GPUSDFModel model = SDFModels[model_index];
    float3 position = ReconstructWorldPosition(pixel, depth);
    float3 local_position = TransformPoint(position, model.world_to_local);
    float4 normal_roughness = TraceNormalRoughness.Load(int3(pixel, 0));
    float3 world_normal = normalize(normal_roughness.xyz);
    float4 albedo_metallic = TraceAlbedoMetallic.Load(int3(pixel, 0));
    float4 emissive = TraceEmissive.Load(int3(pixel, 0));
    float4 direct = TraceDirectRadiance.Load(int3(pixel, 0));
    uint key = LegacySurfaceHash(object_id, model.version.x, model, local_position, world_normal);
    uint slot = LegacySurfaceCacheWriteSlot(key);
    LegacySurfaceCacheEntry entry;
    entry.position = float4(local_position, 1.0f);
    entry.normal = float4(world_normal, 0.0f);
    entry.albedo_roughness = float4(albedo_metallic.rgb, normal_roughness.w);
    entry.emissive_metallic = float4(emissive.rgb, albedo_metallic.w);
    entry.direct_radiance = float4(ReflectedDirectAtPixel(pixel), direct.w);
    entry.indirect_radiance = 0.0f;
    entry.object_id = object_id;
    entry.revision = model.version.x;
    entry.last_frame = (uint)TraceFrame.camera_position.w;
    entry.confidence = 1u;
    LegacySurfaceCacheEntries[slot] = entry;
}

[numthreads(8, 8, 1)]
void CS_ScreenProbes(uint3 dispatch_id : SV_DispatchThreadID) {
    uint probe_width, probe_height;
    ScreenProbesOutput.GetDimensions(probe_width, probe_height);
    uint2 probe = dispatch_id.xy;
    if (probe.x >= probe_width || probe.y >= probe_height) return;
    uint2 base = probe * SCREEN_PROBE_DIRECTION_SIZE;
    float3 radiance = 0.0f;
    [unroll]
    for (uint y = 0u; y < SCREEN_PROBE_DIRECTION_SIZE; ++y)
        [unroll]
        for (uint x = 0u; x < SCREEN_PROBE_DIRECTION_SIZE; ++x)
            radiance += ProbeRadianceOutput[base + uint2(x, y)].rgb;
    radiance /= (float)SCREEN_PROBE_RAY_COUNT;
    ScreenProbesOutput[probe] = float4(radiance, 1.0f);
}

// -----------------------------------------------------------------------------
// Permanent geometry-addressed surface cache.
// -----------------------------------------------------------------------------
uint SurfaceCacheKey(SurfaceHit hit) {
    uint h = HashCombine(hit.identity.x, hit.identity.z);
    h = HashCombine(h, hit.meta.y);
    uint2 q = (uint2)floor(saturate(hit.uv_bary.xy) * 4095.0f);
    h = HashCombine(h, q.x | (q.y << 12u));
    return h | 1u;
}

uint SurfaceCacheFindSlot(uint key, bool create) {
    uint capacity = Radiance.cache_counts.x;
    if (capacity == 0u) return INVALID_INDEX;
    uint mask = capacity - 1u;
    [unroll]
    for (uint probe = 0u; probe < MAX_CACHE_PROBES; ++probe) {
        uint slot = (key + probe) & mask;
        uint stored = SurfaceCacheKeys[slot];
        if (stored == key) return slot;
        if (stored == 0u) {
            if (!create) return INVALID_INDEX;
            uint previous;
            InterlockedCompareExchange(SurfaceCacheKeys[slot], 0u, key, previous);
            if (previous == 0u || previous == key) return slot;
        }
    }
    return create ? (key & mask) : INVALID_INDEX;
}

bool SurfaceCacheLookup(SurfaceHit hit, out SurfaceCacheEntry entry) {
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
    float3 direct;
    if (SurfaceCacheLookup(hit, entry)) direct = entry.direct_radiance.rgb;
    else direct = EvaluateSurfaceReflectedDirect(hit);
    float3 indirect = WorldIndirectReflected(hit, false);
    SurfaceCacheStore(hit, direct, indirect, 1.0f);
    return direct + indirect;
}

float3 SurfaceOutgoingRadiance(SurfaceHit hit) {
    if (hit.identity.w == TRACE_MISS) return FutureSkyRadiance(normalize(hit.normal_confidence.xyz));
    if (hit.identity.y >= Radiance.scene_counts.y) return 0.0f;
    return SceneMaterials[hit.identity.y].emissive + SurfaceReflectedRadiance(hit);
}

void SurfaceCacheInvalidateSlot(uint slot, float confidence_scale) {
    if (slot >= Radiance.cache_counts.x) return;
    SurfaceCacheEntry entry = SurfaceCacheEntries[slot];
    entry.normal_confidence.w *= confidence_scale;
    if (entry.normal_confidence.w <= 0.001f) entry.state.w = 0u;
    SurfaceCacheEntries[slot] = entry;
}

// -----------------------------------------------------------------------------
// Directional screen probes.
// -----------------------------------------------------------------------------
uint ProbeDirectionSize() { return clamp(Radiance.probe_config.y, 1u, MAX_SCREEN_PROBE_DIRECTION_SIZE); }
uint ProbeRayCount() { return clamp(Radiance.probe_config.z, 1u, MAX_SCREEN_PROBE_RAYS); }
uint ProbeTileSize() { return max(Radiance.probe_config.x, 1u); }

uint2 ProbeAtlasCoord(uint2 probe, uint2 direction_texel) {
    return probe * ProbeDirectionSize() + direction_texel;
}

float3 ProbeDirection(uint2 texel) {
    uint size = ProbeDirectionSize();
    float2 uv = (float2(texel) + 0.5f) / (float)size;
    return OctDecode(uv);
}

uint2 ProbeDirectionTexel(float3 direction) {
    uint size = ProbeDirectionSize();
    float2 uv = OctEncode(direction);
    return min((uint2)floor(uv * (float)size), uint2(size - 1u, size - 1u));
}

uint2 ProbeRepresentativePixel(uint2 probe) {
    uint tile = ProbeTileSize();
    uint2 resolution = uint2(TraceFrame.resolution.xy);
    return min(probe * tile + uint2(tile / 2u, tile / 2u), resolution - uint2(1u, 1u));
}

bool PlaceScreenProbe(uint2 probe, out ScreenProbeState state) {
    state = (ScreenProbeState)0;
    uint2 pixel = ProbeRepresentativePixel(probe);
    float depth = TraceDepth.Load(int3(pixel, 0));
    if (depth <= 0.0f) return false;
    float3 position = ReconstructWorldPosition(pixel, depth);
    float3 normal = normalize(TraceNormalRoughness.Load(int3(pixel, 0)).xyz);
    float3 view_direction = normalize(position - TraceFrame.camera_position.xyz);
    if (dot(normal, view_direction) > 0.0f) normal = -normal;
    state.position_depth = float4(position, depth);
    state.normal_confidence = float4(normal, 1.0f);
    state.history = uint4(0u, Radiance.feature_flags.y, PackPixel(pixel), TraceObjectId.Load(int3(pixel, 0)));
    state.statistics = 0.0f;
    return true;
}

TraceRay BuildFutureProbeRay(uint2 probe, uint ray_index, uint destination) {
    ScreenProbeState state;
    if (!PlaceScreenProbe(probe, state)) return MakeTraceRay(0.0f, float3(0.0f, 1.0f, 0.0f), 0.0f, 0.0f, TRACE_RAY_INACTIVE, destination, uint2(0u, 0u), 0u);
    uint scramble = HashCombine(probe.x, HashCombine(probe.y, Pass.dispatch.x));
    float2 u = Hammersley(ray_index, ProbeRayCount(), scramble);
    float3 local = CosineHemisphere(u);
    float3 tangent, bitangent;
    BuildNormalBasis(state.normal_confidence.xyz, tangent, bitangent);
    float3 direction = normalize(tangent * local.x + bitangent * local.y + state.normal_confidence.xyz * local.z);
    float bias = max(Radiance.trace_params.y, 1.0e-3f);
    uint2 pixel = UnpackPixel(state.history.z);
    return MakeTraceRay(state.position_depth.xyz + state.normal_confidence.xyz * bias, direction, bias, Radiance.trace_params.x, TRACE_RAY_DIFFUSE, destination, pixel, state.history.w);
}

float3 ResolvedProbeRadiance(uint2 probe, uint2 texel) {
    uint2 coord = ProbeAtlasCoord(probe, texel);
    if (FeatureEnabled(RADIANCE_FEATURE_SPATIAL_PROBES))
        return ProbeHistoryRadiance.Load(int3(coord, 0)).rgb;
    return ProbeCurrentRadiance[coord].rgb;
}

float3 IntegrateProbeIrradiance(uint2 probe, float3 normal) {
    uint size = ProbeDirectionSize();
    float3 sum = 0.0f;
    float weight_sum = 0.0f;
    for (uint y = 0u; y < size; ++y) {
        for (uint x = 0u; x < size; ++x) {
            uint2 texel = uint2(x, y);
            float3 direction = ProbeDirection(texel);
            float weight = saturate(dot(normal, direction));
            sum += ResolvedProbeRadiance(probe, texel) * weight;
            weight_sum += weight;
        }
    }
    return weight_sum > 0.0f ? sum / weight_sum : 0.0f;
}

// -----------------------------------------------------------------------------
// World radiance cache.
// -----------------------------------------------------------------------------
uint WorldProbeDirectionCount() {
    uint s = max(Radiance.world_probe_config.x, 1u);
    return s * s;
}

uint WorldProbeRadianceIndex(uint probe_index, uint2 texel, uint bank) {
    uint s = max(Radiance.world_probe_config.x, 1u);
    uint direction = texel.x + s * texel.y;
    uint bank_stride = Radiance.cache_counts.w * WorldProbeDirectionCount();
    return bank * bank_stride + probe_index * WorldProbeDirectionCount() + direction;
}

uint WorldProbeCellKey(int3 cell) {
    uint h = HashCombine(asuint(cell.x), asuint(cell.y));
    h = HashCombine(h, asuint(cell.z));
    return h | 1u;
}

uint WorldProbeKey(float3 position) {
    float cell_size = max(Radiance.world_probe_params.x, 0.25f);
    return WorldProbeCellKey(int3(floor(position / cell_size)));
}

uint FindWorldProbeInternal(float3 position, float3 normal, bool surface_filter, bool stable_previous) {
    uint count = Radiance.cache_counts.y;
    if (!FeatureEnabled(RADIANCE_FEATURE_WORLD_CACHE) || count == 0u) return INVALID_INDEX;
    uint table_capacity = max(Radiance.cache_counts.z, 1u);
    uint mask = table_capacity - 1u;
    float cell_size = max(Radiance.world_probe_params.x, 0.25f);
    int3 base_cell = int3(floor(position / cell_size));
    uint best_index = INVALID_INDEX;
    float best_distance = 3.402823466e+38f;
    for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                uint key = WorldProbeCellKey(base_cell + int3(dx, dy, dz));
                [unroll]
                for (uint i = 0u; i < MAX_CACHE_PROBES; ++i) {
                    uint slot = (key + i) & mask;
                    uint encoded = WorldProbeKeys[slot];
                    if (encoded == 0u) break;
                    uint index = encoded - 1u;
                    if (index >= count) continue;
                    WorldProbeState state = WorldProbes[index];
                    if (state.identity.x != key || (state.state.w & 1u) == 0u || state.state.y != Radiance.feature_flags.y) continue;
                    if (stable_previous && state.state.x == Pass.dispatch.x && state.statistics.z <= 1.0f) continue;
                    if (surface_filter && dot(state.position_radius.xyz - position, normal) <= 1.0e-4f) continue;
                    float d = distance(state.position_radius.xyz, position);
                    if (d > state.position_radius.w || d >= best_distance) continue;
                    best_distance = d;
                    best_index = index;
                }
            }
        }
    }
    return best_index;
}

uint FindWorldProbe(float3 position) {
    return FindWorldProbeInternal(position, 0.0f, false, false);
}

uint FindWorldProbeForSurface(float3 position, float3 normal, bool stable_previous) {
    return FindWorldProbeInternal(position, normalize(normal), true, stable_previous);
}

uint WorldProbeReadBank(WorldProbeState state, bool stable_previous) {
    uint bank = state.state.z & 1u;
    if (stable_previous && state.state.x == Pass.dispatch.x) bank ^= 1u;
    return bank;
}

float3 SampleWorldProbeDirectional(uint probe_index, float3 direction, bool stable_previous) {
    if (probe_index == INVALID_INDEX || probe_index >= Radiance.cache_counts.y) return 0.0f;
    WorldProbeState state = WorldProbes[probe_index];
    if ((state.state.w & 1u) == 0u || state.state.y != Radiance.feature_flags.y) return 0.0f;
    if (stable_previous && state.state.x == Pass.dispatch.x && state.statistics.z <= 1.0f) return 0.0f;
    uint s = max(Radiance.world_probe_config.x, 1u);
    float2 uv = OctEncode(direction);
    uint2 texel = min((uint2)floor(uv * (float)s), uint2(s - 1u, s - 1u));
    return WorldProbeRadiance[WorldProbeRadianceIndex(probe_index, texel, WorldProbeReadBank(state, stable_previous))].rgb;
}

float3 IntegrateWorldProbeDiffuse(float3 position, float3 normal, bool stable_previous) {
    uint probe = FindWorldProbeForSurface(position, normal, stable_previous);
    if (probe == INVALID_INDEX) return 0.0f;
    WorldProbeState state = WorldProbes[probe];
    uint bank = WorldProbeReadBank(state, stable_previous);
    uint s = max(Radiance.world_probe_config.x, 1u);
    float3 sum = 0.0f;
    float weight_sum = 0.0f;
    for (uint d = 0u; d < WorldProbeDirectionCount(); ++d) {
        uint2 texel = uint2(d % s, d / s);
        float3 direction = OctDecode((float2(texel) + 0.5f) / (float)s);
        float weight = saturate(dot(normal, direction));
        sum += WorldProbeRadiance[WorldProbeRadianceIndex(probe, texel, bank)].rgb * weight;
        weight_sum += weight;
    }
    return weight_sum > 0.0f ? sum / weight_sum : 0.0f;
}

float3 WorldIndirectReflected(SurfaceHit hit, bool stable_previous) {
    if (!FeatureEnabled(RADIANCE_FEATURE_MULTIBOUNCE) || hit.identity.y >= Radiance.scene_counts.y) return 0.0f;
    GPUMaterial material = SceneMaterials[hit.identity.y];
    float diffuse = saturate(1.0f - material.metallic);
    float3 incoming = IntegrateWorldProbeDiffuse(hit.position_distance.xyz, normalize(hit.normal_confidence.xyz), stable_previous);
    return material.base_color.rgb * diffuse * incoming;
}

float3 WorldRadianceFallback(float3 position, float3 direction) {
    uint probe = FindWorldProbe(position);
    return probe == INVALID_INDEX ? 0.0f : SampleWorldProbeDirectional(probe, direction, false);
}

// -----------------------------------------------------------------------------
// Wavefront passes.
// RayCounters layout: [0]=A count, [1]=B count, [2]=hit count, [3]=updates.
// -----------------------------------------------------------------------------
[numthreads(1, 1, 1)]
void CS_ResetWavefront(uint3 dispatch_id : SV_DispatchThreadID) {
    RayCounters[0] = 0u;
    RayCounters[1] = 0u;
    RayCounters[2] = 0u;
    RayCounters[3] = Pass.range.x;
    RayDispatchArgs[0] = 0u;
    RayDispatchArgs[1] = 1u;
    RayDispatchArgs[2] = 1u;
}

[numthreads(8, 8, 1)]
void CS_ClassifyRayBudgets(uint3 dispatch_id : SV_DispatchThreadID) {
    uint2 probe = dispatch_id.xy;
    uint probe_width = Pass.dimensions.x;
    uint probe_height = Pass.dimensions.y;
    if (probe.x >= probe_width || probe.y >= probe_height) return;
    uint index = probe.x + probe.y * probe_width;
    float4 meta = ProbeCurrentMeta[probe];
    float confidence = saturate(meta.x);
    float variance = max(meta.y, 0.0f);
    float age = max(meta.z, 0.0f);
    uint min_rays = (uint)max(Radiance.adaptive_params.x, 0.0f);
    uint max_rays = (uint)max(Radiance.adaptive_params.y, (float)min_rays);
    uint fixed_rays = ProbeRayCount();
    uint rays = fixed_rays;
    if (FeatureEnabled(RADIANCE_FEATURE_ADAPTIVE_RAYS)) {
        float urgency = saturate(variance * Radiance.adaptive_params.z + (1.0f - confidence) + age * Radiance.adaptive_params.w);
        rays = (uint)round(lerp((float)min_rays, (float)max_rays, urgency));
    }
    RayBudget budget;
    budget.counts = uint4(rays, 0u, 0u, 0u);
    budget.priority = float4(variance, confidence, age, 0.0f);
    RayBudgets[index] = budget;
}

[numthreads(8, 8, 1)]
void CS_GenerateProbeRays(uint3 dispatch_id : SV_DispatchThreadID) {
    uint2 probe = dispatch_id.xy;
    uint probe_width = Pass.dimensions.x;
    uint probe_height = Pass.dimensions.y;
    if (probe.x >= probe_width || probe.y >= probe_height) return;
    uint probe_index = probe.x + probe.y * probe_width;
    uint count = min(RayBudgets[probe_index].counts.x, MAX_SCREEN_PROBE_RAYS);
    for (uint i = 0u; i < count; ++i) {
        uint queue_index;
        InterlockedAdd(RayCounters[0], 1u, queue_index);
        if (queue_index >= Radiance.feature_flags.z) continue;
        TraceRay ray = BuildFutureProbeRay(probe, i, queue_index);
        RayQueueA[queue_index] = ray;
        RayFlags[queue_index] = (probe_index & 0x000fffffu) | ((i & 0x3fu) << 20u);
    }
}

[numthreads(64, 1, 1)]
void CS_WavefrontScreenTrace(uint3 dispatch_id : SV_DispatchThreadID) {
    uint index = dispatch_id.x;
    if (index >= RayCounters[0]) return;
    TraceRay ray = RayQueueA[index];
    if (ray.type == TRACE_RAY_INACTIVE) return;
    SurfaceHit hit = TraceScreenSurface(ray);
    RaySurfaceHits[index] = hit;
    if (hit.identity.w == TRACE_SCREEN) {
        RaySurfaceHits[index] = hit;
        uint2 hit_pixel = UnpackPixel(hit.meta.z);
        RayRadiance[index] = float4(ReflectedDirectAtPixel(hit_pixel) + WorldIndirectReflected(hit, false), 1.0f);
        return;
    }
    uint out_index;
    InterlockedAdd(RayCounters[1], 1u, out_index);
    if (out_index < Radiance.feature_flags.z) {
        ray.destination = index;
        RayQueueB[out_index] = ray;
    }
}

[numthreads(64, 1, 1)]
void CS_WavefrontDynamicTrace(uint3 dispatch_id : SV_DispatchThreadID) {
    uint index = dispatch_id.x;
    if (index >= RayCounters[1]) return;
    TraceRay ray = RayQueueB[index];
    SurfaceHit hit = MakeSurfaceHit(TRACE_MISS, ray.direction_tmax.w);
    if (TraceDynamicGrid(ray, hit)) {
        uint original = ray.destination;
        RaySurfaceHits[original] = hit;
        RayFlags[original] |= RAY_FLAG_DYNAMIC_HIT;
    }
}

[numthreads(64, 1, 1)]
void CS_WavefrontLocalTrace(uint3 dispatch_id : SV_DispatchThreadID) {
    uint index = dispatch_id.x;
    if (index >= RayCounters[1]) return;
    TraceRay ray = RayQueueB[index];
    uint original = ray.destination;
    SurfaceHit hit = (RayFlags[original] & RAY_FLAG_DYNAMIC_HIT) != 0u ? RaySurfaceHits[original] : MakeSurfaceHit(TRACE_MISS, ray.direction_tmax.w);
    if ((RayFlags[original] & RAY_FLAG_GLOBAL_RESOLVED) == 0u) TraceAllLocalSDFs(ray, hit);
    RaySurfaceHits[original] = hit;
}

[numthreads(64, 1, 1)]
void CS_WavefrontGlobalTrace(uint3 dispatch_id : SV_DispatchThreadID) {
    uint index = dispatch_id.x;
    if (index >= RayCounters[1]) return;
    TraceRay ray = RayQueueB[index];
    uint original = ray.destination;
    SurfaceHit hit = (RayFlags[original] & RAY_FLAG_DYNAMIC_HIT) != 0u
        ? RaySurfaceHits[original]
        : MakeSurfaceHit(TRACE_MISS, ray.direction_tmax.w);
    if (TraceGlobalSDF(ray, hit)) RayFlags[original] |= RAY_FLAG_GLOBAL_RESOLVED;
    RaySurfaceHits[original] = hit;
}

[numthreads(64, 1, 1)]
void CS_ShadeRayHits(uint3 dispatch_id : SV_DispatchThreadID) {
    uint index = dispatch_id.x;
    if (index >= RayCounters[0]) return;
    TraceRay ray = RayQueueA[index];
    SurfaceHit hit = RaySurfaceHits[index];
    float3 radiance;
    if (hit.identity.w == TRACE_SCREEN) radiance = RayRadiance[index].rgb;
    else if (hit.identity.w == TRACE_MISS || hit.identity.w == TRACE_INACTIVE) radiance = FutureSkyRadiance(ray.direction_tmax.xyz);
    else if (ray.type == TRACE_RAY_DIFFUSE) radiance = SurfaceReflectedRadiance(hit);
    else radiance = SurfaceOutgoingRadiance(hit);
    RayRadiance[index] = float4(radiance, 1.0f);
    uint packed_flags = RayFlags[index];
    uint probe_index = packed_flags & 0x000fffffu;
    uint probe_width = Pass.dimensions.x;
    uint2 probe = uint2(probe_index % probe_width, probe_index / probe_width);
    float3 direction = ray.direction_tmax.xyz;
    uint2 texel = ProbeDirectionTexel(direction);
    ProbeCurrentRadiance[ProbeAtlasCoord(probe, texel)] = float4(radiance, 1.0f);
}

[numthreads(8, 8, 1)]
void CS_EmissiveGather(uint3 dispatch_id : SV_DispatchThreadID) {
    uint2 probe = dispatch_id.xy;
    if (probe.x >= Pass.dimensions.x || probe.y >= Pass.dimensions.y) return;
    ScreenProbeState state;
    if (!PlaceScreenProbe(probe, state)) return;
    uint seed = HashCombine(probe.x + probe.y * Pass.dimensions.x, Pass.dispatch.x);
    uint2 source_pixel = UnpackPixel(state.history.z);
    float3 contribution = EvaluateEmissiveSample(state.position_depth.xyz, state.normal_confidence.xyz, state.history.w, source_pixel, seed);
    float4 current = ProbeIrradiance[probe];
    ProbeIrradiance[probe] = float4(current.rgb + contribution, 1.0f);
}

[numthreads(8, 8, 1)]
void CS_ResolveDirectionalProbes(uint3 dispatch_id : SV_DispatchThreadID) {
    uint2 probe = dispatch_id.xy;
    if (probe.x >= Pass.dimensions.x || probe.y >= Pass.dimensions.y) return;
    ScreenProbeState state;
    if (!PlaceScreenProbe(probe, state)) {
        ProbeIrradiance[probe] = 0.0f;
        ProbeCurrentMeta[probe] = 0.0f;
        return;
    }
    float3 irradiance = IntegrateProbeIrradiance(probe, state.normal_confidence.xyz);
    float variance = 0.0f;
    if (FeatureEnabled(RADIANCE_FEATURE_TEMPORAL_PROBES) && Pass.flags.x != 0u) {
        float previous_luma = dot(ProbePreviousIrradiance.Load(int3(probe, 0)).rgb, float3(0.2126f, 0.7152f, 0.0722f));
        float current_luma = dot(irradiance, float3(0.2126f, 0.7152f, 0.0722f));
        variance = abs(current_luma - previous_luma);
    }
    ProbeIrradiance[probe] = float4(irradiance, 1.0f);
    ProbeCurrentMeta[probe] = float4(state.normal_confidence.w, variance, 0.0f, (float)Radiance.feature_flags.y);
}

// -----------------------------------------------------------------------------
// Temporal/spatial screen-probe reuse.
// -----------------------------------------------------------------------------
[numthreads(8, 8, 1)]
void CS_ReprojectScreenProbes(uint3 dispatch_id : SV_DispatchThreadID) {
    uint2 probe = dispatch_id.xy;
    if (probe.x >= Pass.dimensions.x || probe.y >= Pass.dimensions.y) return;
    if (!FeatureEnabled(RADIANCE_FEATURE_TEMPORAL_PROBES) || Pass.flags.x == 0u) return;

    uint2 pixel = ProbeRepresentativePixel(probe);
    float depth = TraceDepth.Load(int3(pixel, 0));
    if (depth <= 0.0f) return;

    float2 velocity = VelocityTexture.Load(int3(pixel, 0));
    float2 previous_pixel = float2(pixel) - velocity * TraceFrame.resolution.xy;
    int2 previous_probe = int2(floor(previous_pixel / (float)ProbeTileSize()));
    if (any(previous_probe < int2(0, 0)) || previous_probe.x >= (int)Pass.dimensions.x || previous_probe.y >= (int)Pass.dimensions.y) return;

    uint2 pp = uint2(previous_probe);
    float previous_depth = ProbeHistoryDepth.Load(int3(pp, 0));
    if (previous_depth <= 0.0f) return;

    float3 previous_normal = ProbeHistoryNormal.Load(int3(pp, 0)).xyz;
    float previous_normal_length = dot(previous_normal, previous_normal);
    if (previous_normal_length <= 1.0e-8f) return;
    previous_normal *= rsqrt(previous_normal_length);

    float3 current_normal = normalize(TraceNormalRoughness.Load(int3(pixel, 0)).xyz);
    float previous_linear_depth = LinearizeDepth(previous_depth);
    float current_linear_depth = LinearizeDepth(depth);
    float relative_depth_error = abs(previous_linear_depth - current_linear_depth) / max(current_linear_depth, 1.0e-3f);
    float depth_tolerance = max(Radiance.temporal_params.z, 1.0e-4f);
    if (relative_depth_error > depth_tolerance || dot(previous_normal, current_normal) < Radiance.temporal_params.w) return;

    float history_weight = saturate(Radiance.temporal_params.x);
    uint size = ProbeDirectionSize();
    for (uint y = 0u; y < size; ++y) {
        for (uint x = 0u; x < size; ++x) {
            uint2 texel = uint2(x, y);
            float3 history = ProbeHistoryRadiance.Load(int3(ProbeAtlasCoord(pp, texel), 0)).rgb;
            uint2 dst = ProbeAtlasCoord(probe, texel);
            float3 current = ProbeCurrentRadiance[dst].rgb;
            ProbeCurrentRadiance[dst] = float4(lerp(current, history, history_weight), 1.0f);
        }
    }

}

[numthreads(8, 8, 1)]
void CS_SpatialReuseScreenProbes(uint3 dispatch_id : SV_DispatchThreadID) {
    uint2 probe = dispatch_id.xy;
    if (probe.x >= Pass.dimensions.x || probe.y >= Pass.dimensions.y) return;

    uint size = ProbeDirectionSize();
    ScreenProbeState center;
    bool valid_center = PlaceScreenProbe(probe, center);
    uint radius = min(Radiance.probe_config.w, 2u);

    for (uint y = 0u; y < size; ++y) {
        for (uint x = 0u; x < size; ++x) {
            uint2 texel = uint2(x, y);
            uint2 destination = ProbeAtlasCoord(probe, texel);

            if (!valid_center) {
                ProbeHistoryRadianceOut[destination] = 0.0f;
                continue;
            }

            float3 sum = ProbeCurrentRadiance[destination].rgb;
            float weight_sum = 1.0f;

            if (FeatureEnabled(RADIANCE_FEATURE_SPATIAL_PROBES)) {
                for (int oy = -(int)radius; oy <= (int)radius; ++oy) {
                    for (int ox = -(int)radius; ox <= (int)radius; ++ox) {
                        if (ox == 0 && oy == 0) continue;
                        int2 np = int2(probe) + int2(ox, oy);
                        if (np.x < 0 || np.y < 0 || np.x >= (int)Pass.dimensions.x || np.y >= (int)Pass.dimensions.y) continue;

                        ScreenProbeState neighbor;
                        if (!PlaceScreenProbe(uint2(np), neighbor)) continue;
                        float normal_weight = saturate(dot(center.normal_confidence.xyz, neighbor.normal_confidence.xyz));
                        if (normal_weight <= 0.35f) continue;
                        float distance_weight = rcp(1.0f + length(center.position_depth.xyz - neighbor.position_depth.xyz));
                        float weight = normal_weight * normal_weight * distance_weight;
                        sum += ProbeCurrentRadiance[ProbeAtlasCoord(uint2(np), texel)].rgb * weight;
                        weight_sum += weight;
                    }
                }
            }

            ProbeHistoryRadianceOut[ProbeAtlasCoord(probe, texel)] = float4(sum / max(weight_sum, 1.0e-6f), 1.0f);
        }
    }
}

[numthreads(8, 8, 1)]
void CS_CommitScreenProbeHistory(uint3 dispatch_id : SV_DispatchThreadID) {
    uint2 probe = dispatch_id.xy;
    if (probe.x >= Pass.dimensions.x || probe.y >= Pass.dimensions.y) return;

    ScreenProbeState state;
    if (!PlaceScreenProbe(probe, state)) {
        ProbeHistoryMetaOut[probe] = 0.0f;
        ProbePreviousIrradianceOut[probe] = 0.0f;
        ProbeHistoryDepthOut[probe] = 0.0f;
        ProbeHistoryNormalOut[probe] = 0.0f;
        return;
    }

    ProbeHistoryMetaOut[probe] = ProbeCurrentMeta[probe];
    ProbePreviousIrradianceOut[probe] = ProbeIrradiance[probe];
    ProbeHistoryDepthOut[probe] = state.position_depth.w;
    ProbeHistoryNormalOut[probe] = float4(state.normal_confidence.xyz, 1.0f);
}

// -----------------------------------------------------------------------------
// World radiance-cache update and iterative multi-bounce.
// -----------------------------------------------------------------------------
[numthreads(64, 1, 1)]
void CS_UpdateWorldRadianceCache(uint3 dispatch_id : SV_DispatchThreadID) {
    uint update_index = dispatch_id.x;
    uint update_count = min(RayCounters[3], Radiance.world_probe_config.y);
    if (update_index >= update_count) return;
    uint probe_index = RadianceUpdateList[update_index];
    if (probe_index >= Radiance.cache_counts.y) return;
    WorldProbeState state = WorldProbes[probe_index];
    uint old_bank = state.state.z & 1u;
    uint new_bank = old_bank ^ 1u;
    bool established = (state.state.w & 1u) != 0u && state.state.y == Radiance.feature_flags.y;
    uint s = max(Radiance.world_probe_config.x, 1u);
    for (uint d = 0u; d < WorldProbeDirectionCount(); ++d) {
        uint2 texel = uint2(d % s, d / s);
        float3 direction = OctDecode((float2(texel) + 0.5f) / (float)s);
        float bias = max(Radiance.trace_params.y, 1.0e-3f);
        TraceRay ray = MakeTraceRay(state.position_radius.xyz, direction, bias, Radiance.trace_params.x, TRACE_RAY_WORLD_PROBE, d, uint2(0u, 0u), 0u);
        SurfaceHit hit = TraceUnifiedRay(ray, false);
        float3 sample = FutureSkyRadiance(direction);
        if (hit.identity.w != TRACE_MISS && hit.identity.w != TRACE_INACTIVE && hit.identity.y < Radiance.scene_counts.y) {
            GPUMaterial material = SceneMaterials[hit.identity.y];
            float3 normal = normalize(hit.normal_confidence.xyz);
            float3 reflected = EvaluateSurfaceReflectedDirect(hit);
            uint seed = HashCombine(probe_index, HashCombine(d, Pass.dispatch.x));
            float3 nee = EvaluateEmissiveSampleForMaterial(hit.position_distance.xyz, normal, hit.identity.x, hit.identity.y, seed);
            reflected += material.base_color.rgb * saturate(1.0f - material.metallic) * nee;
            reflected += WorldIndirectReflected(hit, true);
            sample = material.emissive + reflected;
        }
        uint old_address = WorldProbeRadianceIndex(probe_index, texel, old_bank);
        uint new_address = WorldProbeRadianceIndex(probe_index, texel, new_bank);
        float3 old = WorldProbeRadiance[old_address].rgb;
        float blend = established ? saturate(Radiance.world_probe_params.y) : 1.0f;
        WorldProbeRadiance[new_address] = float4(lerp(old, sample, blend), 1.0f);
    }
    state.statistics.x = established ? saturate(state.statistics.x + 0.1f) : 1.0f;
    state.statistics.z = established ? state.statistics.z + 1.0f : 1.0f;
    state.state.x = Pass.dispatch.x;
    state.state.y = Radiance.feature_flags.y;
    state.state.z = new_bank;
    state.state.w = 1u;
    WorldProbes[probe_index] = state;
}


[numthreads(64, 1, 1)]
void CS_InvalidateRadiance(uint3 dispatch_id : SV_DispatchThreadID) {
    uint index = dispatch_id.x;
    uint count = Pass.range.x;
    if (index >= count) return;
    uint encoded = InvalidationQueue[Pass.range.y + index];
    uint kind = encoded >> 30u;
    uint target = encoded & 0x3fffffffu;
    float confidence_scale = saturate(Radiance.temporal_params.y);
    if (kind == 0u) {
        SurfaceCacheInvalidateSlot(target, confidence_scale);
    } else if (kind == 1u && target < Radiance.cache_counts.y) {
        WorldProbeState state = WorldProbes[target];
        state.statistics.x *= confidence_scale;
        state.state.w |= 2u;
        WorldProbes[target] = state;
    }
}

// -----------------------------------------------------------------------------
// Reflection hierarchy.
// -----------------------------------------------------------------------------
float3 SampleDirectionalProbeForPixel(uint2 pixel, float3 direction) {
    uint2 probe = pixel / ProbeTileSize();
    uint2 texel = ProbeDirectionTexel(direction);
    return ProbeCurrentRadiance[ProbeAtlasCoord(probe, texel)].rgb;
}

[numthreads(8, 8, 1)]
void CS_ReflectionTrace(uint3 dispatch_id : SV_DispatchThreadID) {
    uint2 pixel = dispatch_id.xy;
    uint2 resolution = uint2(TraceFrame.resolution.xy);
    if (any(pixel >= resolution)) return;
    float depth = TraceDepth.Load(int3(pixel, 0));
    if (depth <= 0.0f || !FeatureEnabled(RADIANCE_FEATURE_REFLECTIONS)) {
        ReflectionOutput[pixel] = 0.0f;
        ReflectionMeta[pixel] = 0.0f;
        return;
    }
    float4 nr = TraceNormalRoughness.Load(int3(pixel, 0));
    float roughness = nr.w;
    float3 position = ReconstructWorldPosition(pixel, depth);
    float3 normal = normalize(nr.xyz);
    float3 view = normalize(position - TraceFrame.camera_position.xyz);
    if (dot(normal, view) > 0.0f) normal = -normal;
    float3 reflected = normalize(reflect(view, normal));
    float rough_threshold = Radiance.reflection_params.x;
    float smooth_threshold = Radiance.reflection_params.y;
    float3 result = 0.0f;
    uint mode = 0u;
    if (roughness >= rough_threshold) {
        result = SampleDirectionalProbeForPixel(pixel, reflected);
        mode = 1u;
    } else if (roughness >= smooth_threshold) {
        result = WorldRadianceFallback(position, reflected);
        if (all(result == 0.0f)) result = SampleDirectionalProbeForPixel(pixel, reflected);
        mode = 2u;
    } else {
        float bias = max(Radiance.trace_params.y, 1.0e-3f);
        TraceRay ray = MakeTraceRay(position + normal * bias, reflected, bias, Radiance.trace_params.x, TRACE_RAY_REFLECTION, 0u, pixel, TraceObjectId.Load(int3(pixel, 0)));
        SurfaceHit hit = TraceUnifiedRay(ray, true);
        result = hit.identity.w == TRACE_MISS ? FutureSkyRadiance(reflected) : SurfaceOutgoingRadiance(hit);
        mode = 3u;
    }
    ReflectionOutput[pixel] = float4(result, 1.0f);
    ReflectionMeta[pixel] = float4((float)mode, roughness, 1.0f, 0.0f);
}

// -----------------------------------------------------------------------------
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
// Runtime debug and presentation helpers.
// -----------------------------------------------------------------------------
float3 LegacyLoadScreenProbe(int2 probe) {
    uint width, height;
    ScreenProbesTexture.GetDimensions(width, height);
    probe = clamp(probe, int2(0, 0), int2((int)width - 1, (int)height - 1));
    return ScreenProbesTexture.Load(int3(probe, 0)).rgb;
}

float3 LegacyInterpolateScreenProbeGI(int2 pixel) {
    uint full_width, full_height;
    DepthTexture.GetDimensions(full_width, full_height);

    float depth = DepthTexture.Load(int3(pixel, 0));
    if (depth <= 0.0f) return 0.0f;

    float3 normal = normalize(NormalRoughnessTexture.Load(int3(pixel, 0)).xyz);
    float linear_depth = LinearizeDepth(depth);
    float2 probe_position = (float2(pixel) + 0.5f) / (float)SCREEN_PROBE_TILE_SIZE - 0.5f;
    int2 center = int2(round(probe_position));

    uint probe_width, probe_height;
    ScreenProbesTexture.GetDimensions(probe_width, probe_height);

    float3 sum = 0.0f;
    float weight_sum = 0.0f;
    float depth_scale = max(0.05f, linear_depth * 0.035f);

    [unroll]
    for (int y = -1; y <= 1; ++y) {
        [unroll]
        for (int x = -1; x <= 1; ++x) {
            int2 probe = clamp(center + int2(x, y), int2(0, 0), int2((int)probe_width - 1, (int)probe_height - 1));
            uint2 representative = min(
                uint2(probe) * SCREEN_PROBE_TILE_SIZE + uint2(SCREEN_PROBE_TILE_SIZE / 2u, SCREEN_PROBE_TILE_SIZE / 2u),
                uint2(full_width - 1u, full_height - 1u)
            );

            float probe_depth = DepthTexture.Load(int3(representative, 0));
            if (probe_depth <= 0.0f) continue;

            float3 probe_normal = normalize(NormalRoughnessTexture.Load(int3(representative, 0)).xyz);
            float normal_similarity = saturate(dot(normal, probe_normal));
            if (normal_similarity <= 0.35f) continue;

            float probe_linear_depth = LinearizeDepth(probe_depth);
            float depth_weight = exp(-abs(probe_linear_depth - linear_depth) / depth_scale);
            float2 delta = float2(probe) - probe_position;
            float spatial_weight = rcp(1.0f + dot(delta, delta));
            float normal_weight = normal_similarity * normal_similarity;
            normal_weight *= normal_weight;
            float weight = spatial_weight * depth_weight * normal_weight;

            sum += ScreenProbesTexture.Load(int3(probe, 0)).rgb * weight;
            weight_sum += weight;
        }
    }

    if (weight_sum > 1.0e-5f) return sum / weight_sum;
    return LegacyLoadScreenProbe(center);
}

float3 PresentHZB(int2 pixel) {
    uint hzb_width, hzb_height, hzb_mip_count;
    HZBTexture.GetDimensions(0, hzb_width, hzb_height, hzb_mip_count);
    uint mip = min(4u, hzb_mip_count);
    float hzb_depth;
    if (mip == 0u) hzb_depth = DepthTexture.Load(int3(pixel, 0));
    else {
        uint hzb_mip = mip - 1u;
        uint mip_width = max(1u, hzb_width >> hzb_mip);
        uint mip_height = max(1u, hzb_height >> hzb_mip);
        uint2 hzb_pixel = min(uint2(pixel) >> mip, uint2(mip_width - 1u, mip_height - 1u));
        hzb_depth = HZBTexture.Load(int3(int2(hzb_pixel), hzb_mip));
    }
    float value = 1.0f - saturate(LinearizeDepth(hzb_depth) / 200.0f);
    return value.xxx;
}

float3 PresentScreenTrace(int2 pixel) {
    uint full_width, full_height, trace_width, trace_height;
    DepthTexture.GetDimensions(full_width, full_height);
    ScreenTraceTexture.GetDimensions(trace_width, trace_height);
    uint2 trace_pixel = min(uint2((uint)pixel.x * trace_width / full_width, (uint)pixel.y * trace_height / full_height), uint2(trace_width - 1u, trace_height - 1u));
    return ScreenTraceTexture.Load(int3(trace_pixel, 0)).rgb;
}

float3 PresentLegacyScreenProbes(int2 pixel) {
    uint full_width, full_height, probe_width, probe_height;
    DepthTexture.GetDimensions(full_width, full_height);
    ScreenProbesTexture.GetDimensions(probe_width, probe_height);
    uint2 probe = min(uint2((uint)pixel.x * probe_width / full_width, (uint)pixel.y * probe_height / full_height), uint2(probe_width - 1u, probe_height - 1u));
    return ScreenProbesTexture.Load(int3(probe, 0)).rgb;
}

float3 PresentFinalGI(int2 pixel) {
    float depth = DepthTexture.Load(int3(pixel, 0));
    float3 direct = DirectRadianceTexture.Load(int3(pixel, 0)).rgb;
    if (depth <= 0.0f) return direct;
    float4 material = AlbedoMetallicTexture.Load(int3(pixel, 0));
    float3 incoming = LegacyInterpolateScreenProbeGI(pixel);
    float diffuse_weight = 1.0f - material.w;
    return direct + material.rgb * diffuse_weight * incoming;
}

float4 PS_Present(PresentVSOutput input) : SV_Target0 {
    int2 pixel = int2(input.position.xy);
#ifdef ALBEDO
    return float4(AlbedoMetallicTexture.Load(int3(pixel, 0)).rgb, 1.0f);
#elif defined(NORMALS)
    return float4(NormalRoughnessTexture.Load(int3(pixel, 0)).rgb * 0.5f + 0.5f, 1.0f);
#elif defined(DEPTH)
    float visible_depth = 1.0f - saturate(LinearizeDepth(DepthTexture.Load(int3(pixel, 0))) / 200.0f);
    return float4(visible_depth.xxx, 1.0f);
#elif defined(ROUGHNESS)
    float roughness = NormalRoughnessTexture.Load(int3(pixel, 0)).w;
    return float4(roughness.xxx, 1.0f);
#elif defined(VELOCITY)
    float2 velocity = VelocityTexture.Load(int3(pixel, 0));
    return float4(saturate(0.5f + velocity.x * 20.0f), saturate(0.5f + velocity.y * 20.0f), 0.5f, 1.0f);
#elif defined(OBJECT_ID)
    float value = frac((float)ObjectIdTexture.Load(int3(pixel, 0)) * 0.61803398875f);
    return float4(value, frac(value * 3.17f), frac(value * 7.13f), 1.0f);
#elif defined(HZB)
    return float4(PresentHZB(pixel), 1.0f);
#elif defined(SCREEN_TRACE)
    return float4(PresentScreenTrace(pixel), 1.0f);
#elif defined(DIRECT_RADIANCE)
    return float4(DirectRadianceTexture.Load(int3(pixel, 0)).rgb, 1.0f);
#elif defined(SCREEN_PROBES)
    return float4(PresentLegacyScreenProbes(pixel), 1.0f);
#elif defined(FINAL_GI)
    return float4(PresentFinalGI(pixel), 1.0f);
#else
    return float4(PresentFinalGI(pixel), 1.0f);
#endif
}

float4 PS_PresentRuntime(PresentVSOutput input) : SV_Target0 {
    int2 pixel = int2(input.position.xy);
    uint mode = Radiance.reserved.x;
    if (mode == DEBUG_ALBEDO) return float4(AlbedoMetallicTexture.Load(int3(pixel, 0)).rgb, 1.0f);
    if (mode == DEBUG_NORMALS) return float4(NormalRoughnessTexture.Load(int3(pixel, 0)).rgb * 0.5f + 0.5f, 1.0f);
    if (mode == DEBUG_DEPTH) {
        float d = 1.0f - saturate(LinearizeDepth(DepthTexture.Load(int3(pixel, 0))) / 200.0f);
        return float4(d.xxx, 1.0f);
    }
    if (mode == DEBUG_ROUGHNESS) { float v = NormalRoughnessTexture.Load(int3(pixel, 0)).w; return float4(v.xxx, 1.0f); }
    if (mode == DEBUG_METALLIC) { float v = AlbedoMetallicTexture.Load(int3(pixel, 0)).w; return float4(v.xxx, 1.0f); }
    if (mode == DEBUG_VELOCITY) { float2 v = VelocityTexture.Load(int3(pixel, 0)); return float4(saturate(0.5f + v.x * 20.0f), saturate(0.5f + v.y * 20.0f), 0.5f, 1.0f); }
    if (mode == DEBUG_OBJECT_ID) { float v = frac((float)ObjectIdTexture.Load(int3(pixel, 0)) * 0.61803398875f); return float4(v, frac(v * 3.17f), frac(v * 7.13f), 1.0f); }
    if (mode == DEBUG_MATERIAL_ID) { float v = frac((float)TraceMaterialId.Load(int3(pixel, 0)) * 0.61803398875f); return float4(v, frac(v * 2.31f), frac(v * 5.73f), 1.0f); }
    if (mode == DEBUG_PRIMITIVE_ID) { float v = frac((float)TracePrimitiveId.Load(int3(pixel, 0)) * 0.38196601125f); return float4(frac(v * 4.11f), v, frac(v * 8.17f), 1.0f); }
    if (mode == DEBUG_HZB) return float4(PresentHZB(pixel), 1.0f);
    if (mode == DEBUG_DIRECT_RADIANCE) return float4(DirectRadianceTexture.Load(int3(pixel, 0)).rgb, 1.0f);
    if (mode == DEBUG_EMISSIVE) return TraceEmissive.Load(int3(pixel, 0));
    if (mode == DEBUG_SCREEN_TRACE) return float4(PresentScreenTrace(pixel), 1.0f);
    if (mode == DEBUG_SCREEN_PROBE_IRRADIANCE) {
        uint2 probe = uint2(pixel) / ProbeTileSize();
        return ProbeIrradiance[probe];
    }
    if (mode == DEBUG_SCREEN_PROBE_CONFIDENCE || mode == DEBUG_SCREEN_PROBE_VARIANCE || mode == DEBUG_SCREEN_PROBE_HISTORY) {
        uint2 probe = uint2(pixel) / ProbeTileSize();
        float4 m = ProbeCurrentMeta[probe];
        float v = mode == DEBUG_SCREEN_PROBE_CONFIDENCE ? m.x : (mode == DEBUG_SCREEN_PROBE_VARIANCE ? m.y : m.z / max(Radiance.temporal_params.z, 1.0f));
        return float4(v.xxx, 1.0f);
    }
    if (mode == DEBUG_REFLECTIONS) return ReflectionOutput[uint2(pixel)];
    return float4(PresentFinalGI(pixel), 1.0f);
}
