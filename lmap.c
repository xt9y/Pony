#ifndef LMAP_C
#define LMAP_C

#include "sdl.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>

typedef enum BakeStage {
    BAKE_STAGE_SURFACE_LOOKUP,
    BAKE_STAGE_TRACE,
    BAKE_STAGE_DILATE,
    BAKE_STAGE_COMPLETE
} BAKE_STAGE;

typedef struct LightmapRasterVertex {
    float uv[2];
    float bary[3];
    float triangle_id;
} LIGHTMAP_RASTER_VERTEX;

typedef struct LightmapRuntimeVertex {
    float position[3];
    float normal[3];
    float uv0[2];
    float lightmap_uv[2];
} LIGHTMAP_RUNTIME_VERTEX;

typedef struct LightmapBakeUniform {
    uint32_t width;
    uint32_t height;
    uint32_t sample_index;
    uint32_t target_instance;
    float sun_direction[3];
    float sun_intensity;
    float sun_color[3];
    float bounce_strength;
    float environment[3];
    float ray_epsilon;
} LIGHTMAP_BAKE_UNIFORM;

typedef struct LightmapSizeUniform {
    uint32_t width;
    uint32_t height;
    uint32_t pad0;
    uint32_t pad1;
} LIGHTMAP_SIZE_UNIFORM;

typedef struct LightmapBake {
    BAKE_STAGE stage;
    uint32_t width;
    uint32_t height;
    uint32_t target_instance;
    uint32_t sample_index;
    uint32_t target_samples;
    uint32_t dilation_iteration;
    uint32_t target_dilation_iterations;
    uint64_t source_signature;

    float sun_direction[3];
    float sun_intensity;
    float sun_color[3];
    float bounce_strength;
    float environment[3];
    float ray_epsilon;

    SDL_GPUTextureFormat surface_format;
    SDL_GPUTextureFormat lightmap_format;
    SDL_GPUTexture *surface;
    SDL_GPUTexture *lightmap_a;
    SDL_GPUTexture *lightmap_b;
    SDL_GPUTexture *result;
    SDL_GPUBuffer *accumulation;
    SDL_GPUBuffer *raster_vertices;
    SDL_GPUBuffer *runtime_vertices;
    uint32_t raster_vertex_count;
    uint32_t runtime_vertex_count;

    SDL_GPUSampler *point_sampler;
    SDL_GPUSampler *linear_sampler;

    SDL_GPUShader *surface_vs;
    SDL_GPUShader *surface_fs;
    SDL_GPUGraphicsPipeline *surface_pipeline;

    SDL_GPUShader *runtime_vs;
    SDL_GPUShader *runtime_fs;
    SDL_GPUGraphicsPipeline *runtime_pipeline;

    SDL_GPUComputePipeline *trace_pipeline;
    SDL_GPUComputePipeline *finalize_pipeline;
    SDL_GPUComputePipeline *dilate_pipeline;
} LIGHTMAP_BAKE;

_Static_assert(sizeof(LIGHTMAP_RASTER_VERTEX) == 24, "LIGHTMAP_RASTER_VERTEX layout mismatch");
_Static_assert(sizeof(LIGHTMAP_RUNTIME_VERTEX) == 40, "LIGHTMAP_RUNTIME_VERTEX layout mismatch");
_Static_assert(sizeof(LIGHTMAP_BAKE_UNIFORM) == 64, "LIGHTMAP_BAKE_UNIFORM layout mismatch");
_Static_assert(sizeof(LIGHTMAP_SIZE_UNIFORM) == 16, "LIGHTMAP_SIZE_UNIFORM layout mismatch");

static const char lm_surface_vs_src[] =
    "struct VSIn { float2 uv : TEXCOORD0; float3 bary : TEXCOORD1; float triangle_id : TEXCOORD2; };\n"
    "struct VSOut { float4 position : SV_Position; float3 bary : TEXCOORD0; nointerpolation float triangle_id : TEXCOORD1; };\n"
    "VSOut main(VSIn i)\n"
    "{\n"
    "    VSOut o;\n"
    "    o.position = float4(i.uv.x * 2.0 - 1.0, 1.0 - i.uv.y * 2.0, 0.0, 1.0);\n"
    "    o.bary = i.bary;\n"
    "    o.triangle_id = i.triangle_id;\n"
    "    return o;\n"
    "}\n";

static const char lm_surface_fs_src[] =
    "struct FSIn { float4 position : SV_Position; float3 bary : TEXCOORD0; nointerpolation float triangle_id : TEXCOORD1; };\n"
    "float4 main(FSIn i) : SV_Target\n"
    "{\n"
    "    return float4(i.bary.y, i.bary.z, i.triangle_id + 1.0, 1.0);\n"
    "}\n";

static const char lm_runtime_vs_src[] =
    "cbuffer Camera : register(b0, space1) { float4x4 view_proj; };\n"
    "struct VSIn { float3 position : TEXCOORD0; float3 normal : TEXCOORD1; float2 uv0 : TEXCOORD2; float2 lightmap_uv : TEXCOORD3; };\n"
    "struct VSOut { float4 position : SV_Position; float3 normal : TEXCOORD0; float2 lightmap_uv : TEXCOORD1; };\n"
    "VSOut main(VSIn i)\n"
    "{\n"
    "    VSOut o;\n"
    "    o.position = mul(view_proj, float4(i.position, 1.0));\n"
    "    o.normal = i.normal;\n"
    "    o.lightmap_uv = i.lightmap_uv;\n"
    "    return o;\n"
    "}\n";

static const char lm_runtime_fs_src[] =
    "Texture2D<float4> Lightmap : register(t0, space2);\n"
    "SamplerState LightmapSampler : register(s0, space2);\n"
    "struct FSIn { float4 position : SV_Position; float3 normal : TEXCOORD0; float2 lightmap_uv : TEXCOORD1; };\n"
    "float4 main(FSIn i) : SV_Target\n"
    "{\n"
    "    float3 baked = Lightmap.Sample(LightmapSampler, i.lightmap_uv).rgb;\n"
    "    return float4(baked, 1.0);\n"
    "}\n";

static const char lm_trace_cs[] =
    "struct BvhNode { float3 bmin; uint meta; float3 bmax; uint first; uint count; uint escape; uint right; uint pad; };\n"
    "struct MeshAccel { uint node_offset; uint node_count; uint primitive_offset; uint primitive_count; uint vertex_offset; uint index_offset; uint pad0; uint pad1; };\n"
    "struct Instance { float4 world0; float4 world1; float4 world2; float4 world3; float4 inv0; float4 inv1; float4 inv2; float4 inv3; uint mesh; uint flags; uint query_mask; uint pad; };\n"
    "struct Ray { float3 origin; float t_min; float3 direction; float t_max; };\n"
    "struct Hit { uint hit; uint triangle_id; uint instance; float t; float u; float v; float2 pad; };\n"
    "Texture2D<float4> Surface : register(t0, space0);\n"
    "SamplerState SurfaceSampler : register(s0, space0);\n"
    "StructuredBuffer<BvhNode> TlasNodes : register(t1, space0);\n"
    "StructuredBuffer<uint> TlasPrimitiveIDs : register(t2, space0);\n"
    "StructuredBuffer<BvhNode> BlasNodes : register(t3, space0);\n"
    "StructuredBuffer<uint> BlasPrimitiveIDs : register(t4, space0);\n"
    "ByteAddressBuffer Vertices : register(t5, space0);\n"
    "ByteAddressBuffer Indices : register(t6, space0);\n"
    "StructuredBuffer<MeshAccel> Meshes : register(t7, space0);\n"
    "StructuredBuffer<Instance> Instances : register(t8, space0);\n"
    "RWStructuredBuffer<float4> Accumulation : register(u0, space1);\n"
    "cbuffer Params : register(b0, space2)\n"
    "{\n"
    "    uint Width; uint Height; uint SampleIndex; uint TargetInstance;\n"
    "    float3 SunDirection; float SunIntensity;\n"
    "    float3 SunColor; float BounceStrength;\n"
    "    float3 Environment; float RayEpsilon;\n"
    "};\n"
    "uint LoadIndex(MeshAccel mesh, uint index) { return Indices.Load((mesh.index_offset + index) * 4u); }\n"
    "float3 LoadPosition(MeshAccel mesh, uint vertex_index) { return asfloat(Vertices.Load3((mesh.vertex_offset + vertex_index) * 32u)); }\n"
    "float3 LoadNormal(MeshAccel mesh, uint vertex_index) { return asfloat(Vertices.Load3((mesh.vertex_offset + vertex_index) * 32u + 12u)); }\n"
    "float3 TransformPoint(Instance inst, float3 p) { return inst.world0.xyz * p.x + inst.world1.xyz * p.y + inst.world2.xyz * p.z + inst.world3.xyz; }\n"
    "float3 TransformPointInv(Instance inst, float3 p) { return inst.inv0.xyz * p.x + inst.inv1.xyz * p.y + inst.inv2.xyz * p.z + inst.inv3.xyz; }\n"
    "float3 TransformDirInv(Instance inst, float3 d) { return inst.inv0.xyz * d.x + inst.inv1.xyz * d.y + inst.inv2.xyz * d.z; }\n"
    "float3 TransformNormal(Instance inst, float3 n)\n"
    "{\n"
    "    float3 r = float3(dot(inst.inv0.xyz, n), dot(inst.inv1.xyz, n), dot(inst.inv2.xyz, n));\n"
    "    if ((inst.flags & 2u) != 0u) r = -r;\n"
    "    return normalize(r);\n"
    "}\n"
    "bool RayAabb(Ray r, float3 mn, float3 mx, float best_t)\n"
    "{\n"
    "    float tmin = r.t_min; float tmax = min(r.t_max, best_t);\n"
    "    [unroll] for (uint axis = 0; axis < 3; ++axis)\n"
    "    {\n"
    "        float o = r.origin[axis]; float d = r.direction[axis];\n"
    "        if (abs(d) < 1e-8) { if (o < mn[axis] || o > mx[axis]) return false; continue; }\n"
    "        float inv = 1.0 / d; float a = (mn[axis] - o) * inv; float b = (mx[axis] - o) * inv;\n"
    "        if (a > b) { float tmp = a; a = b; b = tmp; }\n"
    "        tmin = max(tmin, a); tmax = min(tmax, b); if (tmin > tmax) return false;\n"
    "    }\n"
    "    return true;\n"
    "}\n"
    "bool RayTriangle(Ray r, float3 a, float3 b, float3 c, float best_t, out float out_t, out float out_u, out float out_v)\n"
    "{\n"
    "    out_t = best_t; out_u = 0.0; out_v = 0.0;\n"
    "    float3 e1 = b - a; float3 e2 = c - a; float3 p = cross(r.direction, e2); float det = dot(e1, p);\n"
    "    if (abs(det) < 1e-7) return false;\n"
    "    float inv_det = 1.0 / det; float3 s = r.origin - a; float u = dot(s, p) * inv_det;\n"
    "    if (u < 0.0 || u > 1.0) return false;\n"
    "    float3 q = cross(s, e1); float v = dot(r.direction, q) * inv_det;\n"
    "    if (v < 0.0 || u + v > 1.0) return false;\n"
    "    float t = dot(e2, q) * inv_det; if (t < r.t_min || t > min(r.t_max, best_t)) return false;\n"
    "    out_t = t; out_u = u; out_v = v; return true;\n"
    "}\n"
    "bool TraceBlas(Ray ray, MeshAccel mesh, inout float best_t, inout uint best_triangle, out float best_u, out float best_v, bool any_hit)\n"
    "{\n"
    "    best_u = 0.0; best_v = 0.0; if (mesh.node_count == 0u) return false;\n"
    "    uint node_index = mesh.node_offset;\n"
    "    while (node_index != 0xffffffffu)\n"
    "    {\n"
    "        BvhNode node = BlasNodes[node_index];\n"
    "        if (!RayAabb(ray, node.bmin, node.bmax, best_t + 1e-5)) { node_index = node.escape; continue; }\n"
    "        if ((node.meta & 1u) != 0u)\n"
    "        {\n"
    "            for (uint i = 0; i < node.count; ++i)\n"
    "            {\n"
    "                uint tri = BlasPrimitiveIDs[node.first + i]; uint base = tri * 3u;\n"
    "                float3 a = LoadPosition(mesh, LoadIndex(mesh, base + 0u));\n"
    "                float3 b = LoadPosition(mesh, LoadIndex(mesh, base + 1u));\n"
    "                float3 c = LoadPosition(mesh, LoadIndex(mesh, base + 2u));\n"
    "                float t, u, v;\n"
    "                if (RayTriangle(ray, a, b, c, best_t + 1e-5, t, u, v))\n"
    "                {\n"
    "                    if (any_hit) { best_t = t; best_triangle = tri; best_u = u; best_v = v; return true; }\n"
    "                    if (best_triangle == 0xffffffffu || t < best_t - 1e-5 || (abs(t - best_t) <= 1e-5 && tri < best_triangle))\n"
    "                    { best_t = t; best_triangle = tri; best_u = u; best_v = v; }\n"
    "                }\n"
    "            }\n"
    "            node_index = node.escape;\n"
    "        } else node_index = node_index + 1u;\n"
    "    }\n"
    "    return best_triangle != 0xffffffffu;\n"
    "}\n"
    "Hit TraceScene(Ray world_ray, bool any_hit)\n"
    "{\n"
    "    Hit best; best.hit = 0u; best.triangle_id = 0xffffffffu; best.instance = 0xffffffffu; best.t = world_ray.t_max; best.u = 0.0; best.v = 0.0; best.pad = 0.0;\n"
    "    uint node_index = 0u;\n"
    "    while (node_index != 0xffffffffu)\n"
    "    {\n"
    "        BvhNode node = TlasNodes[node_index];\n"
    "        if (!RayAabb(world_ray, node.bmin, node.bmax, best.t + 1e-5)) { node_index = node.escape; continue; }\n"
    "        if ((node.meta & 1u) != 0u)\n"
    "        {\n"
    "            for (uint i = 0; i < node.count; ++i)\n"
    "            {\n"
    "                uint instance_index = TlasPrimitiveIDs[node.first + i]; Instance inst = Instances[instance_index];\n"
    "                if ((inst.flags & 1u) != 0u) continue;\n"
    "                MeshAccel mesh = Meshes[inst.mesh];\n"
    "                Ray local_ray; local_ray.origin = TransformPointInv(inst, world_ray.origin); local_ray.direction = TransformDirInv(inst, world_ray.direction); local_ray.t_min = world_ray.t_min; local_ray.t_max = best.t + 1e-5;\n"
    "                float candidate_t = best.t; uint candidate_triangle = 0xffffffffu; float candidate_u, candidate_v;\n"
    "                if (TraceBlas(local_ray, mesh, candidate_t, candidate_triangle, candidate_u, candidate_v, any_hit))\n"
    "                {\n"
    "                    if (any_hit) { best.hit = 1u; best.triangle_id = candidate_triangle; best.instance = instance_index; best.t = candidate_t; best.u = candidate_u; best.v = candidate_v; return best; }\n"
    "                    if (best.hit == 0u || candidate_t < best.t - 1e-5 || (abs(candidate_t - best.t) <= 1e-5 && (instance_index < best.instance || (instance_index == best.instance && candidate_triangle < best.triangle_id))))\n"
    "                    { best.hit = 1u; best.triangle_id = candidate_triangle; best.instance = instance_index; best.t = candidate_t; best.u = candidate_u; best.v = candidate_v; }\n"
    "                }\n"
    "            }\n"
    "            node_index = node.escape;\n"
    "        } else node_index = node_index + 1u;\n"
    "    }\n"
    "    return best;\n"
    "}\n"
    "void Reconstruct(uint instance_index, uint triangle_id, float u, float v, out float3 world_pos, out float3 shading_normal, out float3 geometric_normal)\n"
    "{\n"
    "    Instance inst = Instances[instance_index]; MeshAccel mesh = Meshes[inst.mesh]; uint base = triangle_id * 3u;\n"
    "    uint ia = LoadIndex(mesh, base + 0u); uint ib = LoadIndex(mesh, base + 1u); uint ic = LoadIndex(mesh, base + 2u);\n"
    "    float3 a = LoadPosition(mesh, ia); float3 b = LoadPosition(mesh, ib); float3 c = LoadPosition(mesh, ic);\n"
    "    float3 na = LoadNormal(mesh, ia); float3 nb = LoadNormal(mesh, ib); float3 nc = LoadNormal(mesh, ic);\n"
    "    float w = 1.0 - u - v; float3 local_pos = a * w + b * u + c * v;\n"
    "    world_pos = TransformPoint(inst, local_pos);\n"
    "    shading_normal = TransformNormal(inst, normalize(na * w + nb * u + nc * v));\n"
    "    geometric_normal = TransformNormal(inst, normalize(cross(b - a, c - a)));\n"
    "}\n"
    "float3 OffsetOrigin(float3 p, float3 n, float3 d) { if (dot(n, d) < 0.0) n = -n; return p + n * RayEpsilon; }\n"
    "float3 DirectLight(float3 p, float3 n, float3 gn)\n"
    "{\n"
    "    float3 L = normalize(-SunDirection); float ndotl = saturate(dot(n, L)); if (ndotl <= 0.0) return 0.0;\n"
    "    Ray shadow; shadow.origin = OffsetOrigin(p, gn, L); shadow.direction = L; shadow.t_min = 0.0; shadow.t_max = 10000.0;\n"
    "    if (TraceScene(shadow, true).hit != 0u) return 0.0;\n"
    "    return SunColor * SunIntensity * ndotl;\n"
    "}\n"
    "uint Hash(uint x) { x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16; return x; }\n"
    "float Rand(inout uint state) { state = Hash(state + 0x9e3779b9u); return (float)(state & 0x00ffffffu) / 16777216.0; }\n"
    "float3 CosineHemisphere(float3 n, float u1, float u2)\n"
    "{\n"
    "    float r = sqrt(u1); float phi = 6.28318530718 * u2; float x = r * cos(phi); float y = r * sin(phi); float z = sqrt(max(0.0, 1.0 - u1));\n"
    "    float3 helper = abs(n.z) < 0.999 ? float3(0,0,1) : float3(0,1,0); float3 t = normalize(cross(helper, n)); float3 b = cross(n, t);\n"
    "    return normalize(t * x + b * y + n * z);\n"
    "}\n"
    "[numthreads(8, 8, 1)]\n"
    "void main(uint3 tid : SV_DispatchThreadID)\n"
    "{\n"
    "    if (tid.x >= Width || tid.y >= Height) return;\n"
    "    float2 uv = (float2(tid.xy) + 0.5) / float2(Width, Height); float4 surface = Surface.SampleLevel(SurfaceSampler, uv, 0.0);\n"
    "    if (surface.w < 0.5 || surface.z < 0.5) return;\n"
    "    uint triangle_id = (uint)(surface.z + 0.5) - 1u; float u = surface.x; float v = surface.y;\n"
    "    float3 p, n, gn; Reconstruct(TargetInstance, triangle_id, u, v, p, n, gn);\n"
    "    float3 sample = DirectLight(p, n, gn);\n"
    "    uint state = Hash(tid.x + tid.y * Width + SampleIndex * 0x9e3779b9u); float3 bounce_dir = CosineHemisphere(n, Rand(state), Rand(state));\n"
    "    Ray bounce; bounce.origin = OffsetOrigin(p, gn, bounce_dir); bounce.direction = bounce_dir; bounce.t_min = 0.0; bounce.t_max = 10000.0;\n"
    "    Hit bounce_hit = TraceScene(bounce, false);\n"
    "    if (bounce_hit.hit != 0u)\n"
    "    {\n"
    "        float3 q, qn, qgn; Reconstruct(bounce_hit.instance, bounce_hit.triangle_id, bounce_hit.u, bounce_hit.v, q, qn, qgn);\n"
    "        sample += DirectLight(q, qn, qgn) * BounceStrength;\n"
    "    } else sample += Environment * BounceStrength;\n"
    "    uint index = tid.y * Width + tid.x; float4 old = Accumulation[index]; Accumulation[index] = float4(old.rgb + sample, old.a + 1.0);\n"
    "}\n";

static const char lm_finalize_cs[] =
    "Texture2D<float4> Surface : register(t0, space0); SamplerState SurfaceSampler : register(s0, space0);\n"
    "StructuredBuffer<float4> Accumulation : register(t1, space0);\n"
    "RWTexture2D<float4> Output : register(u0, space1);\n"
    "cbuffer Params : register(b0, space2) { uint Width; uint Height; uint2 Pad; };\n"
    "[numthreads(8,8,1)] void main(uint3 tid : SV_DispatchThreadID)\n"
    "{\n"
    "    if (tid.x >= Width || tid.y >= Height) return; uint index = tid.y * Width + tid.x;\n"
    "    float2 uv = (float2(tid.xy) + 0.5) / float2(Width, Height); float4 s = Surface.SampleLevel(SurfaceSampler, uv, 0.0);\n"
    "    if (s.w < 0.5) { Output[tid.xy] = 0.0; return; }\n"
    "    float4 a = Accumulation[index]; Output[tid.xy] = float4(a.rgb / max(a.a, 1.0), 1.0);\n"
    "}\n";

static const char lm_dilate_cs[] =
    "Texture2D<float4> Source : register(t0, space0); SamplerState SourceSampler : register(s0, space0);\n"
    "RWTexture2D<float4> Output : register(u0, space1);\n"
    "cbuffer Params : register(b0, space2) { uint Width; uint Height; uint2 Pad; };\n"
    "float4 ReadPixel(int2 p) { p = clamp(p, int2(0,0), int2((int)Width - 1, (int)Height - 1)); float2 uv = (float2(p) + 0.5) / float2(Width, Height); return Source.SampleLevel(SourceSampler, uv, 0.0); }\n"
    "[numthreads(8,8,1)] void main(uint3 tid : SV_DispatchThreadID)\n"
    "{\n"
    "    if (tid.x >= Width || tid.y >= Height) return; int2 p = int2(tid.xy); float4 c = ReadPixel(p); if (c.a > 0.5) { Output[tid.xy] = c; return; }\n"
    "    float4 sum = 0.0; float count = 0.0; int2 o[4] = { int2(-1,0), int2(1,0), int2(0,-1), int2(0,1) };\n"
    "    [unroll] for (uint i = 0; i < 4; ++i) { float4 n = ReadPixel(p + o[i]); if (n.a > 0.5) { sum += n; count += 1.0; } }\n"
    "    Output[tid.xy] = count > 0.0 ? float4(sum.rgb / count, 1.0) : 0.0;\n"
    "}\n";

static uint64_t lm_hash_bytes(uint64_t h, const void *data, size_t size)
{
    const uint8_t *bytes = (const uint8_t *)data;
    for (size_t i = 0; i < size; ++i) {
        h ^= bytes[i];
        h *= 1099511628211ull;
    }
    return h;
}

static uint64_t lm_signature(const SCENE *scene, const LIGHTMAP_BAKE *bake)
{
    if (!scene || !bake || bake->target_instance >= scene->instance_count) return 0;
    const RENDER_INSTANCE *instance = &scene->instances[bake->target_instance];
    if (instance->mesh >= scene->mesh_count) return 0;
    static const uint32_t algorithm_version = 1;
    uint64_t h = 1469598103934665603ull;
    h = lm_hash_bytes(h, &algorithm_version, sizeof(algorithm_version));
    h = lm_hash_bytes(h, &bake->target_instance, sizeof(bake->target_instance));
    h = lm_hash_bytes(h, &bake->width, sizeof(bake->width));
    h = lm_hash_bytes(h, &bake->height, sizeof(bake->height));
    h = lm_hash_bytes(h, &bake->target_samples, sizeof(bake->target_samples));
    h = lm_hash_bytes(h, &bake->target_dilation_iterations, sizeof(bake->target_dilation_iterations));
    h = lm_hash_bytes(h, bake->sun_direction, sizeof(bake->sun_direction));
    h = lm_hash_bytes(h, &bake->sun_intensity, sizeof(bake->sun_intensity));
    h = lm_hash_bytes(h, bake->sun_color, sizeof(bake->sun_color));
    h = lm_hash_bytes(h, &bake->bounce_strength, sizeof(bake->bounce_strength));
    h = lm_hash_bytes(h, bake->environment, sizeof(bake->environment));
    h = lm_hash_bytes(h, &bake->ray_epsilon, sizeof(bake->ray_epsilon));

    for (uint32_t i = 0; i < scene->instance_count; ++i) {
        const RENDER_INSTANCE *static_instance = &scene->instances[i];
        if (static_instance->mobility != MOBILITY_STATIC || static_instance->mesh >= scene->mesh_count) continue;
        const MESH *mesh = &scene->meshes[static_instance->mesh];
        h = lm_hash_bytes(h, &i, sizeof(i));
        h = lm_hash_bytes(h, &static_instance->mesh, sizeof(static_instance->mesh));
        h = lm_hash_bytes(h, &mesh->geometry_revision, sizeof(mesh->geometry_revision));
        h = lm_hash_bytes(h, static_instance->transform.matrix, sizeof(static_instance->transform.matrix));
    }
    return h;
}

static SDL_GPUTextureFormat lm_pick_lightmap_format(SDL_GPUDevice *device)
{
    SDL_GPUTextureUsageFlags usage = SDL_GPU_TEXTUREUSAGE_SAMPLER | SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_WRITE;
    if (SDL_GPUTextureSupportsFormat(device, SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT, SDL_GPU_TEXTURETYPE_2D, usage)) {
        return SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT;
    }
    if (SDL_GPUTextureSupportsFormat(device, SDL_GPU_TEXTUREFORMAT_R32G32B32A32_FLOAT, SDL_GPU_TEXTURETYPE_2D, usage)) {
        return SDL_GPU_TEXTUREFORMAT_R32G32B32A32_FLOAT;
    }
    if (SDL_GPUTextureSupportsFormat(device, SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM, SDL_GPU_TEXTURETYPE_2D, usage)) {
        return SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM;
    }
    return SDL_GPU_TEXTUREFORMAT_INVALID;
}

static SDL_GPUTexture *lm_texture(SDL_GPUDevice *device, SDL_GPUTextureFormat format, SDL_GPUTextureUsageFlags usage, uint32_t w, uint32_t h, const char *name)
{
    SDL_GPUTextureCreateInfo info = {
        .type = SDL_GPU_TEXTURETYPE_2D,
        .format = format,
        .usage = usage,
        .width = w,
        .height = h,
        .layer_count_or_depth = 1,
        .num_levels = 1,
        .sample_count = SDL_GPU_SAMPLECOUNT_1
    };
    SDL_GPUTexture *texture = SDL_CreateGPUTexture(device, &info);
    if (texture) SDL_SetGPUTextureName(device, texture, name);
    return texture;
}

static int lm_make_vertex_buffers(SDL_GPUDevice *device, const MESH *mesh, LIGHTMAP_BAKE *bake)
{
    if (!device || !mesh || !bake || mesh->lightmap_corner_count != mesh->index_count) return 0;
    uint32_t count = mesh->index_count;
    LIGHTMAP_RASTER_VERTEX *raster = SDL_malloc((size_t)count * sizeof(*raster));
    LIGHTMAP_RUNTIME_VERTEX *runtime = SDL_malloc((size_t)count * sizeof(*runtime));
    if (!raster || !runtime) {
        SDL_free(raster);
        SDL_free(runtime);
        return 0;
    }

    for (uint32_t i = 0; i < count; ++i) {
        uint32_t triangle = i / 3u;
        uint32_t corner = i % 3u;
        uint32_t vertex_index = mesh->indices[i];
        const VERTEX *v = &mesh->vertices[vertex_index];
        raster[i].uv[0] = mesh->lightmap_corners[i].u;
        raster[i].uv[1] = mesh->lightmap_corners[i].v;
        raster[i].bary[0] = corner == 0 ? 1.0f : 0.0f;
        raster[i].bary[1] = corner == 1 ? 1.0f : 0.0f;
        raster[i].bary[2] = corner == 2 ? 1.0f : 0.0f;
        raster[i].triangle_id = (float)triangle;
        SDL_memcpy(runtime[i].position, v->position, sizeof(runtime[i].position));
        SDL_memcpy(runtime[i].normal, v->normal, sizeof(runtime[i].normal));
        SDL_memcpy(runtime[i].uv0, v->uv0, sizeof(runtime[i].uv0));
        runtime[i].lightmap_uv[0] = mesh->lightmap_corners[i].u;
        runtime[i].lightmap_uv[1] = mesh->lightmap_corners[i].v;
    }

    SDL_GPUBufferCreateInfo raster_info = {
        .usage = SDL_GPU_BUFFERUSAGE_VERTEX,
        .size = count * sizeof(*raster)
    };
    SDL_GPUBufferCreateInfo runtime_info = {
        .usage = SDL_GPU_BUFFERUSAGE_VERTEX,
        .size = count * sizeof(*runtime)
    };
    bake->raster_vertices = SDL_CreateGPUBuffer(device, &raster_info);
    bake->runtime_vertices = SDL_CreateGPUBuffer(device, &runtime_info);
    int ok = bake->raster_vertices && bake->runtime_vertices &&
        m_load(device, bake->raster_vertices, raster, count * sizeof(*raster)) &&
        m_load(device, bake->runtime_vertices, runtime, count * sizeof(*runtime));
    SDL_free(raster);
    SDL_free(runtime);
    if (!ok) return 0;
    bake->raster_vertex_count = count;
    bake->runtime_vertex_count = count;
    SDL_SetGPUBufferName(device, bake->raster_vertices, "Lightmap.RasterVertices");
    SDL_SetGPUBufferName(device, bake->runtime_vertices, "Lightmap.RuntimeVertices");
    return 1;
}

static SDL_GPUGraphicsPipeline *lm_surface_pipeline(RENDERER *r, LIGHTMAP_BAKE *bake)
{
    SDL_GPUVertexBufferDescription vb = {
        .slot = 0,
        .pitch = sizeof(LIGHTMAP_RASTER_VERTEX),
        .input_rate = SDL_GPU_VERTEXINPUTRATE_VERTEX,
        .instance_step_rate = 0
    };
    SDL_GPUVertexAttribute attrs[3] = {
        {.location = 0, .buffer_slot = 0, .format = SDL_GPU_VERTEXELEMENTFORMAT_FLOAT2, .offset = offsetof(LIGHTMAP_RASTER_VERTEX, uv)},
        {.location = 1, .buffer_slot = 0, .format = SDL_GPU_VERTEXELEMENTFORMAT_FLOAT3, .offset = offsetof(LIGHTMAP_RASTER_VERTEX, bary)},
        {.location = 2, .buffer_slot = 0, .format = SDL_GPU_VERTEXELEMENTFORMAT_FLOAT, .offset = offsetof(LIGHTMAP_RASTER_VERTEX, triangle_id)},
    };
    SDL_GPUColorTargetDescription color = {.format = bake->surface_format};
    SDL_GPUGraphicsPipelineCreateInfo info = {
        .vertex_shader = bake->surface_vs,
        .fragment_shader = bake->surface_fs,
        .vertex_input_state = {
            .vertex_buffer_descriptions = &vb,
            .num_vertex_buffers = 1,
            .vertex_attributes = attrs,
            .num_vertex_attributes = 3
        },
        .primitive_type = SDL_GPU_PRIMITIVETYPE_TRIANGLELIST,
        .rasterizer_state = {
            .fill_mode = SDL_GPU_FILLMODE_FILL,
            .cull_mode = SDL_GPU_CULLMODE_NONE,
            .front_face = SDL_GPU_FRONTFACE_COUNTER_CLOCKWISE
        },
        .multisample_state = {.sample_count = SDL_GPU_SAMPLECOUNT_1},
        .target_info = {.color_target_descriptions = &color, .num_color_targets = 1}
    };
    return SDL_CreateGPUGraphicsPipeline(r->device, &info);
}

static SDL_GPUGraphicsPipeline *lm_runtime_pipeline(RENDERER *r, LIGHTMAP_BAKE *bake)
{
    SDL_GPUVertexBufferDescription vb = {
        .slot = 0,
        .pitch = sizeof(LIGHTMAP_RUNTIME_VERTEX),
        .input_rate = SDL_GPU_VERTEXINPUTRATE_VERTEX,
        .instance_step_rate = 0
    };
    SDL_GPUVertexAttribute attrs[4] = {
        {.location = 0, .buffer_slot = 0, .format = SDL_GPU_VERTEXELEMENTFORMAT_FLOAT3, .offset = offsetof(LIGHTMAP_RUNTIME_VERTEX, position)},
        {.location = 1, .buffer_slot = 0, .format = SDL_GPU_VERTEXELEMENTFORMAT_FLOAT3, .offset = offsetof(LIGHTMAP_RUNTIME_VERTEX, normal)},
        {.location = 2, .buffer_slot = 0, .format = SDL_GPU_VERTEXELEMENTFORMAT_FLOAT2, .offset = offsetof(LIGHTMAP_RUNTIME_VERTEX, uv0)},
        {.location = 3, .buffer_slot = 0, .format = SDL_GPU_VERTEXELEMENTFORMAT_FLOAT2, .offset = offsetof(LIGHTMAP_RUNTIME_VERTEX, lightmap_uv)},
    };
    SDL_GPUColorTargetDescription color = {.format = r->swp_format};
    SDL_GPUGraphicsPipelineCreateInfo info = {
        .vertex_shader = bake->runtime_vs,
        .fragment_shader = bake->runtime_fs,
        .vertex_input_state = {
            .vertex_buffer_descriptions = &vb,
            .num_vertex_buffers = 1,
            .vertex_attributes = attrs,
            .num_vertex_attributes = 4
        },
        .primitive_type = SDL_GPU_PRIMITIVETYPE_TRIANGLELIST,
        .rasterizer_state = {
            .fill_mode = SDL_GPU_FILLMODE_FILL,
            .cull_mode = SDL_GPU_CULLMODE_BACK,
            .front_face = SDL_GPU_FRONTFACE_COUNTER_CLOCKWISE
        },
        .multisample_state = {.sample_count = SDL_GPU_SAMPLECOUNT_1},
        .depth_stencil_state = {
            .compare_op = SDL_GPU_COMPAREOP_LESS,
            .enable_depth_test = true,
            .enable_depth_write = true
        },
        .target_info = {
            .color_target_descriptions = &color,
            .num_color_targets = 1,
            .depth_stencil_format = r->depth_format,
            .has_depth_stencil_target = true
        }
    };
    return SDL_CreateGPUGraphicsPipeline(r->device, &info);
}

static int lm_clear_accumulation(SDL_GPUDevice *device, LIGHTMAP_BAKE *bake)
{
    size_t bytes = (size_t)bake->width * bake->height * sizeof(float) * 4u;
    void *zero = SDL_calloc(1, bytes);
    if (!zero) return 0;
    int ok = m_load(device, bake->accumulation, zero, (uint32_t)bytes);
    SDL_free(zero);
    return ok;
}

int lm_init(RENDERER *r, const SCENE *scene, uint32_t target_instance, uint32_t width, uint32_t height, uint32_t samples, LIGHTMAP_BAKE *bake)
{
    if (!r || !r->device || !scene || !bake || target_instance >= scene->instance_count || width == 0 || height == 0 || samples == 0) return 0;
    SDL_memset(bake, 0, sizeof(*bake));
    const RENDER_INSTANCE *instance = &scene->instances[target_instance];
    if (instance->mobility != MOBILITY_STATIC || instance->mesh >= scene->mesh_count) return 0;
    const MESH *mesh = &scene->meshes[instance->mesh];
    if (!mesh->lightmap_corners || mesh->lightmap_corner_count != mesh->index_count) return 0;

    bake->stage = BAKE_STAGE_SURFACE_LOOKUP;
    bake->width = width;
    bake->height = height;
    bake->target_instance = target_instance;
    bake->target_samples = samples;
    bake->target_dilation_iterations = 4;
    bake->sun_direction[0] = 0.35f;
    bake->sun_direction[1] = -1.0f;
    bake->sun_direction[2] = 0.2f;
    bake->sun_intensity = 2.0f;
    bake->sun_color[0] = 1.0f;
    bake->sun_color[1] = 0.95f;
    bake->sun_color[2] = 0.85f;
    bake->bounce_strength = 0.6f;
    bake->environment[0] = 0.03f;
    bake->environment[1] = 0.04f;
    bake->environment[2] = 0.06f;
    bake->ray_epsilon = 1e-4f;
    bake->surface_format = SDL_GPU_TEXTUREFORMAT_R32G32B32A32_FLOAT;
    bake->lightmap_format = lm_pick_lightmap_format(r->device);
    if (bake->lightmap_format == SDL_GPU_TEXTUREFORMAT_INVALID) return 0;

    SDL_GPUTextureUsageFlags surface_usage = SDL_GPU_TEXTUREUSAGE_COLOR_TARGET | SDL_GPU_TEXTUREUSAGE_SAMPLER;
    if (!SDL_GPUTextureSupportsFormat(r->device, bake->surface_format, SDL_GPU_TEXTURETYPE_2D, surface_usage)) return 0;
    bake->surface = lm_texture(r->device, bake->surface_format, surface_usage, width, height, "Lightmap.Surface");
    SDL_GPUTextureUsageFlags lightmap_usage = SDL_GPU_TEXTUREUSAGE_SAMPLER | SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_WRITE;
    bake->lightmap_a = lm_texture(r->device, bake->lightmap_format, lightmap_usage, width, height, "Lightmap.A");
    bake->lightmap_b = lm_texture(r->device, bake->lightmap_format, lightmap_usage, width, height, "Lightmap.B");
    if (!bake->surface || !bake->lightmap_a || !bake->lightmap_b) return 0;

    SDL_GPUBufferCreateInfo accumulation_info = {
        .usage = SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_READ | SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_WRITE,
        .size = width * height * sizeof(float) * 4u
    };
    bake->accumulation = SDL_CreateGPUBuffer(r->device, &accumulation_info);
    if (!bake->accumulation || !lm_clear_accumulation(r->device, bake)) return 0;
    SDL_SetGPUBufferName(r->device, bake->accumulation, "Lightmap.Accumulation");

    SDL_GPUSamplerCreateInfo point_info = {
        .min_filter = SDL_GPU_FILTER_NEAREST,
        .mag_filter = SDL_GPU_FILTER_NEAREST,
        .mipmap_mode = SDL_GPU_SAMPLERMIPMAPMODE_NEAREST,
        .address_mode_u = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE,
        .address_mode_v = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE,
        .address_mode_w = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE,
        .min_lod = 0.0f,
        .max_lod = 0.0f
    };
    SDL_GPUSamplerCreateInfo linear_info = point_info;
    linear_info.min_filter = SDL_GPU_FILTER_LINEAR;
    linear_info.mag_filter = SDL_GPU_FILTER_LINEAR;
    bake->point_sampler = SDL_CreateGPUSampler(r->device, &point_info);
    bake->linear_sampler = SDL_CreateGPUSampler(r->device, &linear_info);
    if (!bake->point_sampler || !bake->linear_sampler) return 0;

    if (!lm_make_vertex_buffers(r->device, mesh, bake)) return 0;
    bake->surface_vs = s_gfx(r->device, lm_surface_vs_src, SDL_SHADERCROSS_SHADERSTAGE_VERTEX);
    bake->surface_fs = s_gfx(r->device, lm_surface_fs_src, SDL_SHADERCROSS_SHADERSTAGE_FRAGMENT);
    bake->runtime_vs = s_gfx(r->device, lm_runtime_vs_src, SDL_SHADERCROSS_SHADERSTAGE_VERTEX);
    bake->runtime_fs = s_gfx(r->device, lm_runtime_fs_src, SDL_SHADERCROSS_SHADERSTAGE_FRAGMENT);
    if (!bake->surface_vs || !bake->surface_fs || !bake->runtime_vs || !bake->runtime_fs) return 0;
    bake->surface_pipeline = lm_surface_pipeline(r, bake);
    bake->runtime_pipeline = lm_runtime_pipeline(r, bake);
    bake->trace_pipeline = bvh_compile_compute(r->device, lm_trace_cs);
    bake->finalize_pipeline = bvh_compile_compute(r->device, lm_finalize_cs);
    bake->dilate_pipeline = bvh_compile_compute(r->device, lm_dilate_cs);
    if (!bake->surface_pipeline || !bake->runtime_pipeline || !bake->trace_pipeline || !bake->finalize_pipeline || !bake->dilate_pipeline) {
        SDL_Log("lightmap pipeline creation failed: %s", SDL_GetError());
        return 0;
    }
    bake->source_signature = lm_signature(scene, bake);
    return 1;
}

static int lm_submit_surface(RENDERER *r, LIGHTMAP_BAKE *bake)
{
    SDL_GPUCommandBuffer *cmd = SDL_AcquireGPUCommandBuffer(r->device);
    if (!cmd) return 0;
    SDL_GPUColorTargetInfo color = {
        .texture = bake->surface,
        .clear_color = {0,0,0,0},
        .load_op = SDL_GPU_LOADOP_CLEAR,
        .store_op = SDL_GPU_STOREOP_STORE
    };
    SDL_GPURenderPass *pass = SDL_BeginGPURenderPass(cmd, &color, 1, NULL);
    SDL_BindGPUGraphicsPipeline(pass, bake->surface_pipeline);
    SDL_GPUBufferBinding vb = {.buffer = bake->raster_vertices, .offset = 0};
    SDL_BindGPUVertexBuffers(pass, 0, &vb, 1);
    SDL_DrawGPUPrimitives(pass, bake->raster_vertex_count, 1, 0, 0);
    SDL_EndGPURenderPass(pass);
    return SDL_SubmitGPUCommandBuffer(cmd);
}

static int lm_submit_trace(RENDERER *r, const SCENE *scene, LIGHTMAP_BAKE *bake)
{
    const SCENE_GPU_ACCEL *gpu = (const SCENE_GPU_ACCEL *)scene->gpu_accel;
    if (!gpu) return 0;
    SDL_GPUCommandBuffer *cmd = SDL_AcquireGPUCommandBuffer(r->device);
    if (!cmd) return 0;
    LIGHTMAP_BAKE_UNIFORM uniform = {
        .width = bake->width,
        .height = bake->height,
        .sample_index = bake->sample_index,
        .target_instance = bake->target_instance,
        .sun_intensity = bake->sun_intensity,
        .bounce_strength = bake->bounce_strength,
        .ray_epsilon = bake->ray_epsilon,
    };
    SDL_memcpy(uniform.sun_direction, bake->sun_direction, sizeof(uniform.sun_direction));
    SDL_memcpy(uniform.sun_color, bake->sun_color, sizeof(uniform.sun_color));
    SDL_memcpy(uniform.environment, bake->environment, sizeof(uniform.environment));
    SDL_PushGPUComputeUniformData(cmd, 0, &uniform, sizeof(uniform));

    SDL_GPUStorageBufferReadWriteBinding rw = {.buffer = bake->accumulation, .cycle = false};
    SDL_GPUComputePass *pass = SDL_BeginGPUComputePass(cmd, NULL, 0, &rw, 1);
    SDL_BindGPUComputePipeline(pass, bake->trace_pipeline);
    SDL_GPUTextureSamplerBinding surface_binding = {.texture = bake->surface, .sampler = bake->point_sampler};
    SDL_BindGPUComputeSamplers(pass, 0, &surface_binding, 1);
    SDL_GPUBuffer *reads[8] = {
        gpu->tlas_nodes,
        gpu->tlas_primitives,
        gpu->blas_nodes,
        gpu->blas_primitives,
        gpu->vertices,
        gpu->indices,
        gpu->mesh_accels,
        gpu->instances
    };
    SDL_BindGPUComputeStorageBuffers(pass, 0, reads, 8);
    SDL_DispatchGPUCompute(pass, (bake->width + 7u) / 8u, (bake->height + 7u) / 8u, 1);
    SDL_EndGPUComputePass(pass);
    return SDL_SubmitGPUCommandBuffer(cmd);
}

static int lm_submit_finalize(RENDERER *r, LIGHTMAP_BAKE *bake)
{
    SDL_GPUCommandBuffer *cmd = SDL_AcquireGPUCommandBuffer(r->device);
    if (!cmd) return 0;
    LIGHTMAP_SIZE_UNIFORM uniform = {.width = bake->width, .height = bake->height};
    SDL_PushGPUComputeUniformData(cmd, 0, &uniform, sizeof(uniform));
    SDL_GPUStorageTextureReadWriteBinding output = {.texture = bake->lightmap_a, .mip_level = 0, .layer = 0, .cycle = false};
    SDL_GPUComputePass *pass = SDL_BeginGPUComputePass(cmd, &output, 1, NULL, 0);
    SDL_BindGPUComputePipeline(pass, bake->finalize_pipeline);
    SDL_GPUTextureSamplerBinding surface_binding = {.texture = bake->surface, .sampler = bake->point_sampler};
    SDL_BindGPUComputeSamplers(pass, 0, &surface_binding, 1);
    SDL_GPUBuffer *accum = bake->accumulation;
    SDL_BindGPUComputeStorageBuffers(pass, 0, &accum, 1);
    SDL_DispatchGPUCompute(pass, (bake->width + 7u) / 8u, (bake->height + 7u) / 8u, 1);
    SDL_EndGPUComputePass(pass);
    bake->result = bake->lightmap_a;
    return SDL_SubmitGPUCommandBuffer(cmd);
}

static int lm_submit_dilate(RENDERER *r, LIGHTMAP_BAKE *bake)
{
    SDL_GPUTexture *source = bake->result;
    SDL_GPUTexture *dest = source == bake->lightmap_a ? bake->lightmap_b : bake->lightmap_a;
    SDL_GPUCommandBuffer *cmd = SDL_AcquireGPUCommandBuffer(r->device);
    if (!cmd) return 0;
    LIGHTMAP_SIZE_UNIFORM uniform = {.width = bake->width, .height = bake->height};
    SDL_PushGPUComputeUniformData(cmd, 0, &uniform, sizeof(uniform));
    SDL_GPUStorageTextureReadWriteBinding output = {.texture = dest, .mip_level = 0, .layer = 0, .cycle = false};
    SDL_GPUComputePass *pass = SDL_BeginGPUComputePass(cmd, &output, 1, NULL, 0);
    SDL_BindGPUComputePipeline(pass, bake->dilate_pipeline);
    SDL_GPUTextureSamplerBinding source_binding = {.texture = source, .sampler = bake->point_sampler};
    SDL_BindGPUComputeSamplers(pass, 0, &source_binding, 1);
    SDL_DispatchGPUCompute(pass, (bake->width + 7u) / 8u, (bake->height + 7u) / 8u, 1);
    SDL_EndGPUComputePass(pass);
    bake->result = dest;
    return SDL_SubmitGPUCommandBuffer(cmd);
}

static int lm_restart(RENDERER *r, const SCENE *scene, LIGHTMAP_BAKE *bake)
{
    if (!lm_clear_accumulation(r->device, bake)) return 0;
    bake->stage = BAKE_STAGE_SURFACE_LOOKUP;
    bake->sample_index = 0;
    bake->dilation_iteration = 0;
    bake->result = NULL;
    bake->source_signature = lm_signature(scene, bake);
    return 1;
}

int lm_tick(RENDERER *r, const SCENE *scene, LIGHTMAP_BAKE *bake)
{
    if (!r || !scene || !bake) return 0;
    uint64_t signature = lm_signature(scene, bake);
    if (!signature) return 0;
    if (signature != bake->source_signature && !lm_restart(r, scene, bake)) return 0;

    switch (bake->stage) {
        case BAKE_STAGE_SURFACE_LOOKUP:
            if (!lm_submit_surface(r, bake)) return 0;
            bake->stage = BAKE_STAGE_TRACE;
            return 1;
        case BAKE_STAGE_TRACE:
            if (!lm_submit_trace(r, scene, bake)) return 0;
            ++bake->sample_index;
            if (bake->sample_index >= bake->target_samples) {
                if (!lm_submit_finalize(r, bake)) return 0;
                bake->stage = BAKE_STAGE_DILATE;
            }
            return 1;
        case BAKE_STAGE_DILATE:
            if (bake->dilation_iteration < bake->target_dilation_iterations) {
                if (!lm_submit_dilate(r, bake)) return 0;
                ++bake->dilation_iteration;
            }
            if (bake->dilation_iteration >= bake->target_dilation_iterations) bake->stage = BAKE_STAGE_COMPLETE;
            return 1;
        case BAKE_STAGE_COMPLETE:
            return 1;
        default:
            return 0;
    }
}

int lm_complete(const LIGHTMAP_BAKE *bake)
{
    return bake && bake->stage == BAKE_STAGE_COMPLETE && bake->result != NULL;
}

int lm_validate_nonempty(SDL_GPUDevice *device, const LIGHTMAP_BAKE *bake)
{
    if (!device || !lm_complete(bake) || !bake->accumulation) return 0;
    uint64_t texel_count = (uint64_t)bake->width * bake->height;
    uint64_t byte_count = texel_count * sizeof(float) * 4u;
    if (texel_count == 0 || byte_count > UINT32_MAX) return 0;

    SDL_GPUTransferBufferCreateInfo info = {
        .usage = SDL_GPU_TRANSFERBUFFERUSAGE_DOWNLOAD,
        .size = (uint32_t)byte_count
    };
    SDL_GPUTransferBuffer *download = SDL_CreateGPUTransferBuffer(device, &info);
    if (!download) return 0;

    SDL_GPUCommandBuffer *cmd = SDL_AcquireGPUCommandBuffer(device);
    if (!cmd) {
        SDL_ReleaseGPUTransferBuffer(device, download);
        return 0;
    }
    SDL_GPUCopyPass *copy = SDL_BeginGPUCopyPass(cmd);
    SDL_GPUBufferRegion src = {
        .buffer = bake->accumulation,
        .offset = 0,
        .size = (uint32_t)byte_count
    };
    SDL_GPUTransferBufferLocation dst = {.transfer_buffer = download, .offset = 0};
    SDL_DownloadFromGPUBuffer(copy, &src, &dst);
    SDL_EndGPUCopyPass(copy);

    SDL_GPUFence *fence = SDL_SubmitGPUCommandBufferAndAcquireFence(cmd);
    if (!fence) {
        SDL_ReleaseGPUTransferBuffer(device, download);
        return 0;
    }
    int ok = SDL_WaitForGPUFences(device, true, &fence, 1);
    SDL_ReleaseGPUFence(device, fence);
    if (!ok) {
        SDL_ReleaseGPUTransferBuffer(device, download);
        return 0;
    }

    const float *data = SDL_MapGPUTransferBuffer(device, download, false);
    if (!data) {
        SDL_ReleaseGPUTransferBuffer(device, download);
        return 0;
    }

    uint32_t valid_texels = 0;
    uint32_t lit_texels = 0;
    ok = 1;
    for (uint64_t i = 0; i < texel_count; ++i) {
        const float *v = &data[i * 4u];
        if (!isfinite(v[0]) || !isfinite(v[1]) || !isfinite(v[2]) || !isfinite(v[3])) {
            ok = 0;
            break;
        }
        if (v[3] <= 0.5f) continue;
        ++valid_texels;
        if (fabsf(v[3] - (float)bake->target_samples) > 0.5f) {
            ok = 0;
            break;
        }
        if (v[0] > 1e-6f || v[1] > 1e-6f || v[2] > 1e-6f) ++lit_texels;
    }
    SDL_UnmapGPUTransferBuffer(device, download);
    SDL_ReleaseGPUTransferBuffer(device, download);
    SDL_Log("lightmap validation: valid=%u lit=%u samples=%u", valid_texels, lit_texels, bake->target_samples);
    return ok && valid_texels > 0 && lit_texels > 0;
}

void lm_draw_runtime(
    SDL_GPUCommandBuffer *cmd,
    SDL_GPURenderPass *pass,
    LIGHTMAP_BAKE *bake,
    const float mvp[16])
{
    if (!cmd || !pass || !bake || !lm_complete(bake)) return;
    SDL_BindGPUGraphicsPipeline(pass, bake->runtime_pipeline);
    SDL_GPUBufferBinding vb = {.buffer = bake->runtime_vertices, .offset = 0};
    SDL_BindGPUVertexBuffers(pass, 0, &vb, 1);
    SDL_GPUTextureSamplerBinding binding = {.texture = bake->result, .sampler = bake->linear_sampler};
    SDL_BindGPUFragmentSamplers(pass, 0, &binding, 1);
    SDL_PushGPUVertexUniformData(cmd, 0, mvp, sizeof(float) * 16u);
    SDL_DrawGPUPrimitives(pass, bake->runtime_vertex_count, 1, 0, 0);
}

void lm_deinit(SDL_GPUDevice *device, LIGHTMAP_BAKE *bake)
{
    if (!device || !bake) return;
    if (bake->dilate_pipeline) SDL_ReleaseGPUComputePipeline(device, bake->dilate_pipeline);
    if (bake->finalize_pipeline) SDL_ReleaseGPUComputePipeline(device, bake->finalize_pipeline);
    if (bake->trace_pipeline) SDL_ReleaseGPUComputePipeline(device, bake->trace_pipeline);
    if (bake->runtime_pipeline) SDL_ReleaseGPUGraphicsPipeline(device, bake->runtime_pipeline);
    if (bake->surface_pipeline) SDL_ReleaseGPUGraphicsPipeline(device, bake->surface_pipeline);
    if (bake->runtime_fs) SDL_ReleaseGPUShader(device, bake->runtime_fs);
    if (bake->runtime_vs) SDL_ReleaseGPUShader(device, bake->runtime_vs);
    if (bake->surface_fs) SDL_ReleaseGPUShader(device, bake->surface_fs);
    if (bake->surface_vs) SDL_ReleaseGPUShader(device, bake->surface_vs);
    if (bake->linear_sampler) SDL_ReleaseGPUSampler(device, bake->linear_sampler);
    if (bake->point_sampler) SDL_ReleaseGPUSampler(device, bake->point_sampler);
    if (bake->runtime_vertices) SDL_ReleaseGPUBuffer(device, bake->runtime_vertices);
    if (bake->raster_vertices) SDL_ReleaseGPUBuffer(device, bake->raster_vertices);
    if (bake->accumulation) SDL_ReleaseGPUBuffer(device, bake->accumulation);
    if (bake->lightmap_b) SDL_ReleaseGPUTexture(device, bake->lightmap_b);
    if (bake->lightmap_a) SDL_ReleaseGPUTexture(device, bake->lightmap_a);
    if (bake->surface) SDL_ReleaseGPUTexture(device, bake->surface);
    SDL_memset(bake, 0, sizeof(*bake));
}

#endif // LMAP_C
