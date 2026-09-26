#ifndef GPU_BIND_S
#define GPU_BIND_S(n,s) [[vk::binding(n, s)]]
#define GPU_BIND_T(n,s) [[vk::binding(n + 16, s)]]
#define GPU_BIND_B(n,s) [[vk::binding(n + 32, s)]]
#define GPU_BIND_U(n,s) [[vk::binding(n + 48, s)]]
#endif

#if defined(BUILD_DYNAMIC_TRACE_CS)
struct DynamicBvhNode { float4 bmin; float4 bmax; uint4 meta; };
struct DynamicBvhTriangle { float4 a; float4 b; float4 c; float4 normal; float4 emissive; };
struct DynamicInstance {
    float4x4 world;
    float4x4 inverse_world;
    float4x4 normal_world;
    float4 bounds_min;
    float4 bounds_max;
    uint4 indices;
};
struct DynamicTraceRay { float3 origin; float tmin; float3 direction; float tmax; };
struct DynamicTraceHit { float t; float3 normal; float3 albedo; uint triangle_index; float3 emissive; uint instance_index; };

GPU_BIND_T(0, 0) StructuredBuffer<DynamicInstance> DynamicInstances : register(t0, space0);
GPU_BIND_T(1, 0) StructuredBuffer<DynamicBvhNode> DynamicNodes : register(t1, space0);
GPU_BIND_T(2, 0) StructuredBuffer<DynamicBvhTriangle> DynamicTriangles : register(t2, space0);
GPU_BIND_U(0, 1) RWStructuredBuffer<uint> DynamicTraceOutput : register(u0, space1);
GPU_BIND_B(0, 2) cbuffer DynamicTraceData : register(b0, space2) { uint dynamic_instance_count; uint3 _dynamic_pad; };

static const uint DYNAMIC_INVALID_NODE = 0xffffffffu;

bool dynamic_trace_box(DynamicTraceRay ray, float3 bmin, float3 bmax, float max_t)
{
    float tmin = ray.tmin;
    float tmax = min(ray.tmax, max_t);
    [unroll] for (uint axis = 0u; axis < 3u; ++axis) {
        float d = ray.direction[axis];
        if (abs(d) < 1.0e-7f) {
            if (ray.origin[axis] < bmin[axis] || ray.origin[axis] > bmax[axis]) return false;
        } else {
            float inv = 1.0f / d;
            float a = (bmin[axis] - ray.origin[axis]) * inv;
            float b = (bmax[axis] - ray.origin[axis]) * inv;
            if (a > b) { float temp = a; a = b; b = temp; }
            tmin = max(tmin, a);
            tmax = min(tmax, b);
            if (tmin > tmax) return false;
        }
    }
    return tmax >= ray.tmin;
}

bool dynamic_trace_triangle(DynamicTraceRay ray, DynamicBvhTriangle tri, float max_t, out float hit_t)
{
    float3 a = tri.a.xyz;
    float3 e1 = tri.b.xyz - a;
    float3 e2 = tri.c.xyz - a;
    float3 p = cross(ray.direction, e2);
    float det = dot(e1, p);
    if (abs(det) < 1.0e-7f) { hit_t = 0.0f; return false; }
    float inv_det = 1.0f / det;
    float3 s = ray.origin - a;
    float u = dot(s, p) * inv_det;
    if (u < 0.0f || u > 1.0f) { hit_t = 0.0f; return false; }
    float3 q = cross(s, e1);
    float v = dot(ray.direction, q) * inv_det;
    if (v < 0.0f || u + v > 1.0f) { hit_t = 0.0f; return false; }
    float t = dot(e2, q) * inv_det;
    if (t <= ray.tmin || t >= min(ray.tmax, max_t)) { hit_t = 0.0f; return false; }
    hit_t = t;
    return true;
}

bool dynamic_trace_instance(uint instance_index, DynamicTraceRay world_ray, float max_t, out DynamicTraceHit hit)
{
    DynamicInstance instance = DynamicInstances[instance_index];
    if (!dynamic_trace_box(world_ray, instance.bounds_min.xyz, instance.bounds_max.xyz, max_t)) return false;

    DynamicTraceRay ray;
    ray.origin = mul(instance.inverse_world, float4(world_ray.origin, 1.0f)).xyz;
    ray.direction = mul((float3x3)instance.inverse_world, world_ray.direction);
    ray.tmin = world_ray.tmin;
    ray.tmax = min(world_ray.tmax, max_t);

    uint node_index = instance.indices.x;
    float closest = ray.tmax;
    bool found = false;
    while (node_index != DYNAMIC_INVALID_NODE) {
        DynamicBvhNode node = DynamicNodes[node_index];
        if (!dynamic_trace_box(ray, node.bmin.xyz, node.bmax.xyz, closest)) { node_index = node.meta.y; continue; }
        if (node.meta.w != 0u) {
            for (uint i = 0u; i < node.meta.w; ++i) {
                uint triangle_index = node.meta.z + i;
                float t;
                if (!dynamic_trace_triangle(ray, DynamicTriangles[triangle_index], closest, t)) continue;
                closest = t;
                DynamicBvhTriangle tri = DynamicTriangles[triangle_index];
                float3 normal = normalize(mul((float3x3)instance.normal_world, tri.normal.xyz));
                if (dot(normal, world_ray.direction) > 0.0f) normal = -normal;
                hit.t = t;
                hit.normal = normal;
                hit.albedo = saturate(float3(tri.a.w, tri.b.w, tri.c.w));
                hit.triangle_index = triangle_index;
                hit.emissive = max(tri.emissive.rgb, 0.0f);
                hit.instance_index = instance_index;
                found = true;
            }
            node_index = node.meta.y;
        } else {
            node_index = node.meta.x;
        }
    }
    return found;
}

bool dynamic_trace_any(DynamicTraceRay ray)
{
    DynamicTraceHit hit;
    for (uint i = 0u; i < dynamic_instance_count; ++i)
        if (dynamic_trace_instance(i, ray, ray.tmax, hit)) return true;
    return false;
}

bool dynamic_trace_closest(DynamicTraceRay ray, out DynamicTraceHit hit)
{
    bool found = false;
    float closest = ray.tmax;
    DynamicTraceHit candidate;
    for (uint i = 0u; i < dynamic_instance_count; ++i) {
        if (!dynamic_trace_instance(i, ray, closest, candidate)) continue;
        closest = candidate.t;
        hit = candidate;
        found = true;
    }
    return found;
}

[numthreads(1, 1, 1)]
void dynamic_trace_cs(uint3 id : SV_DispatchThreadID)
{
    if (id.x != 0u) return;
    DynamicTraceRay ray;
    ray.origin = float3(0.0f, 0.0f, 0.0f);
    ray.direction = float3(0.0f, 0.0f, 1.0f);
    ray.tmin = 1.0e-4f;
    ray.tmax = 1.0e20f;
    DynamicTraceHit hit;
    DynamicTraceOutput[0] = dynamic_trace_closest(ray, hit) ? asuint(hit.t) : DYNAMIC_INVALID_NODE;
}
#endif
