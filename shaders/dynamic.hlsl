#ifndef GPU_BIND_S
#define GPU_BIND_S(n,s) [[vk::binding(n, s)]]
#define GPU_BIND_T(n,s) [[vk::binding(n + 16, s)]]
#define GPU_BIND_B(n,s) [[vk::binding(n + 32, s)]]
#define GPU_BIND_U(n,s) [[vk::binding(n + 48, s)]]
#define GPU_STORAGE_RGBA16F [[vk::image_format("rgba16f")]]
#endif

#if defined(BUILD_DYNAMIC_SHADOW_VS)

GPU_BIND_B(0, 1) cbuffer DynamicShadowData : register(b0, space1)
{
    float4x4 shadow_model;
    float4 shadow_u_min;
    float4 shadow_v_min;
    float4 shadow_sun_max;
    float4 shadow_extent;
};

struct DynamicShadowInput { float3 position : TEXCOORD0; };

float4 dynamic_shadow_vs(DynamicShadowInput input) : SV_Position
{
    float3 world = mul(shadow_model, float4(input.position, 1.0f)).xyz;
    float2 uv = (float2(dot(world, shadow_u_min.xyz), dot(world, shadow_v_min.xyz)) -
                 float2(shadow_u_min.w, shadow_v_min.w)) /
                max(shadow_extent.xy, float2(1.0e-6f, 1.0e-6f));
    float depth = (shadow_sun_max.w - dot(world, shadow_sun_max.xyz)) /
                  max(shadow_extent.z, 1.0e-6f);
    return float4(uv * 2.0f - 1.0f, saturate(depth), 1.0f);
}

#elif defined(BUILD_DYNAMIC_CELL_CS)

struct CellUpdate { uint cell; uint generation; };
GPU_BIND_T(0, 0) StructuredBuffer<CellUpdate> Updates : register(t0, space0);
GPU_BIND_U(0, 1) RWStructuredBuffer<uint> CellGenerations : register(u0, space1);
GPU_BIND_B(0, 2) cbuffer CellData : register(b0, space2) { uint update_count; uint3 _cell_pad; };

[numthreads(64, 1, 1)]
void dynamic_cell_cs(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= update_count) return;
    CellUpdate update = Updates[id.x];
    CellGenerations[update.cell] = update.generation;
}

#elif defined(BUILD_DYNAMIC_GI_CS)

struct BvhNode { float4 bmin; float4 bmax; uint4 meta; };
struct BvhTriangle { float4 a; float4 b; float4 c; float4 normal; float4 emissive; };
struct DynamicInstance {
    float4x4 world;
    float4x4 inverse_world;
    float4x4 normal_world;
    float4 bounds_min;
    float4 bounds_max;
    uint4 indices;
};
struct Probe { float4 position; float4 coefficient[9]; };
struct GiJob {
    float4 position;
    float4 normal;
    uint cell;
    uint generation;
    uint flags;
    uint _pad;
};
struct Ray { float3 origin; float tmin; float3 direction; float tmax; };
struct Hit { float t; float3 normal; float3 albedo; float3 emissive; uint dynamic; };

GPU_BIND_T(0, 0) StructuredBuffer<GiJob> Jobs : register(t0, space0);
GPU_BIND_T(1, 0) StructuredBuffer<BvhNode> StaticNodes : register(t1, space0);
GPU_BIND_T(2, 0) StructuredBuffer<BvhTriangle> StaticTriangles : register(t2, space0);
GPU_BIND_T(3, 0) StructuredBuffer<DynamicInstance> DynamicInstances : register(t3, space0);
GPU_BIND_T(4, 0) StructuredBuffer<BvhNode> DynamicNodes : register(t4, space0);
GPU_BIND_T(5, 0) StructuredBuffer<BvhTriangle> DynamicTriangles : register(t5, space0);
GPU_BIND_T(6, 0) StructuredBuffer<Probe> Probes : register(t6, space0);
GPU_BIND_T(7, 0) StructuredBuffer<uint> CellGenerations : register(t7, space0);
GPU_BIND_U(0, 1) GPU_STORAGE_RGBA16F RWTexture2D<float4> Overlay : register(u0, space1);
GPU_BIND_U(1, 1) RWStructuredBuffer<uint> OverlayGenerations : register(u1, space1);

GPU_BIND_B(0, 2) cbuffer GiData : register(b0, space2)
{
    uint job_count;
    uint dynamic_instance_count;
    uint lightmap_width;
    uint lightmap_height;
    uint rays_per_texel;
    uint frame_index;
    uint2 _gi_pad;
    float4 sky_zenith;
    float4 sky_horizon;
    float4 sun_direction_intensity;
    float4 sun_color_epsilon;
    float4 probe_origin_spacing;
    uint4 probe_dims;
};

static const uint INVALID_NODE = 0xffffffffu;
static const float PI = 3.14159265358979323846f;

uint hash_u32(uint x)
{
    x ^= x >> 16u;
    x *= 0x7feb352du;
    x ^= x >> 15u;
    x *= 0x846ca68bu;
    return x ^ (x >> 16u);
}

float random01(inout uint seed)
{
    seed = hash_u32(seed + 0x9e3779b9u);
    return (float)(seed & 0x00ffffffu) * (1.0f / 16777216.0f);
}

float3 cosine_direction(float3 n, inout uint seed)
{
    float r = sqrt(random01(seed));
    float phi = 2.0f * PI * random01(seed);
    float3 up = abs(n.z) < 0.999f ? float3(0,0,1) : float3(0,1,0);
    float3 t = normalize(cross(up, n));
    float3 b = cross(n, t);
    return normalize(t * (r * cos(phi)) + b * (r * sin(phi)) + n * sqrt(max(0.0f, 1.0f - r * r)));
}

bool hit_box(Ray ray, float3 bmin, float3 bmax, float max_t)
{
    float lo = ray.tmin, hi = min(ray.tmax, max_t);
    [unroll] for (uint a = 0u; a < 3u; ++a) {
        float d = ray.direction[a];
        if (abs(d) < 1.0e-7f) {
            if (ray.origin[a] < bmin[a] || ray.origin[a] > bmax[a]) return false;
        } else {
            float x = (bmin[a] - ray.origin[a]) / d;
            float y = (bmax[a] - ray.origin[a]) / d;
            if (x > y) { float q = x; x = y; y = q; }
            lo = max(lo, x); hi = min(hi, y);
            if (lo > hi) return false;
        }
    }
    return hi >= ray.tmin;
}

bool hit_triangle(Ray ray, BvhTriangle tri, float max_t, out float t)
{
    float3 e1 = tri.b.xyz - tri.a.xyz;
    float3 e2 = tri.c.xyz - tri.a.xyz;
    float3 p = cross(ray.direction, e2);
    float det = dot(e1, p);
    if (abs(det) < 1.0e-7f) { t = 0.0f; return false; }
    float inv = 1.0f / det;
    float3 s = ray.origin - tri.a.xyz;
    float u = dot(s, p) * inv;
    float3 q = cross(s, e1);
    float v = dot(ray.direction, q) * inv;
    t = dot(e2, q) * inv;
    return u >= 0.0f && v >= 0.0f && u + v <= 1.0f &&
           t > ray.tmin && t < min(ray.tmax, max_t);
}

bool static_closest(Ray ray, inout Hit hit)
{
    uint node = 0u;
    float closest = hit.t;
    bool found = false;
    while (node != INVALID_NODE) {
        BvhNode n = StaticNodes[node];
        if (!hit_box(ray, n.bmin.xyz, n.bmax.xyz, closest)) { node = n.meta.y; continue; }
        if (n.meta.w) {
            for (uint i = 0u; i < n.meta.w; ++i) {
                uint ti = n.meta.z + i;
                float t;
                if (!hit_triangle(ray, StaticTriangles[ti], closest, t)) continue;
                BvhTriangle tri = StaticTriangles[ti];
                float3 normal = normalize(tri.normal.xyz);
                if (dot(normal, ray.direction) > 0.0f) normal = -normal;
                closest = hit.t = t;
                hit.normal = normal;
                hit.albedo = saturate(float3(tri.a.w, tri.b.w, tri.c.w));
                hit.emissive = max(tri.emissive.rgb, 0.0f);
                hit.dynamic = 0u;
                found = true;
            }
            node = n.meta.y;
        } else node = n.meta.x;
    }
    return found;
}

bool dynamic_closest(Ray world_ray, inout Hit hit)
{
    bool found = false;
    for (uint instance_index = 0u; instance_index < dynamic_instance_count; ++instance_index) {
        DynamicInstance instance = DynamicInstances[instance_index];
        if (!hit_box(world_ray, instance.bounds_min.xyz, instance.bounds_max.xyz, hit.t)) continue;

        Ray ray;
        ray.origin = mul(instance.inverse_world, float4(world_ray.origin, 1.0f)).xyz;
        ray.direction = mul((float3x3)instance.inverse_world, world_ray.direction);
        ray.tmin = world_ray.tmin;
        ray.tmax = min(world_ray.tmax, hit.t);

        uint node = instance.indices.x;
        float closest = hit.t;
        while (node != INVALID_NODE) {
            BvhNode n = DynamicNodes[node];
            if (!hit_box(ray, n.bmin.xyz, n.bmax.xyz, closest)) { node = n.meta.y; continue; }
            if (n.meta.w) {
                for (uint i = 0u; i < n.meta.w; ++i) {
                    uint ti = n.meta.z + i;
                    float t;
                    if (!hit_triangle(ray, DynamicTriangles[ti], closest, t)) continue;
                    BvhTriangle tri = DynamicTriangles[ti];
                    float3 normal = normalize(mul((float3x3)instance.normal_world, tri.normal.xyz));
                    if (dot(normal, world_ray.direction) > 0.0f) normal = -normal;
                    closest = hit.t = t;
                    hit.normal = normal;
                    hit.albedo = saturate(float3(tri.a.w, tri.b.w, tri.c.w));
                    hit.emissive = max(tri.emissive.rgb, 0.0f);
                    hit.dynamic = 1u;
                    found = true;
                }
                node = n.meta.y;
            } else node = n.meta.x;
        }
    }
    return found;
}

bool scene_closest(Ray ray, out Hit hit)
{
    hit.t = ray.tmax;
    hit.normal = 0.0f;
    hit.albedo = 0.0f;
    hit.emissive = 0.0f;
    hit.dynamic = 0u;
    bool found = static_closest(ray, hit);
    found = dynamic_closest(ray, hit) || found;
    return found;
}

float3 probe_value(Probe probe, float3 n)
{
    float nx = n.x, ny = n.y, nz = n.z;
    float3 e = probe.coefficient[0].rgb * (0.2820947918f * PI);
    e += (probe.coefficient[1].rgb * (0.4886025119f * ny) +
          probe.coefficient[2].rgb * (0.4886025119f * nz) +
          probe.coefficient[3].rgb * (0.4886025119f * nx)) * (2.0f * PI / 3.0f);
    e += (probe.coefficient[4].rgb * (1.0925484306f * nx * ny) +
          probe.coefficient[5].rgb * (1.0925484306f * ny * nz) +
          probe.coefficient[6].rgb * (0.3153915653f * (3.0f * nz * nz - 1.0f)) +
          probe.coefficient[7].rgb * (1.0925484306f * nx * nz) +
          probe.coefficient[8].rgb * (0.5462742153f * (nx * nx - ny * ny))) * (PI * 0.25f);
    return max(e, 0.0f);
}

float3 probe_irradiance(float3 p, float3 n)
{
    if (any(probe_dims.xyz == 0u) || probe_origin_spacing.w <= 0.0f) return 0.12f.xxx;
    n = normalize(n);
    float3 coord = clamp((p - probe_origin_spacing.xyz) / probe_origin_spacing.w,
                         0.0f, float3(probe_dims.xyz) - 1.0f);
    uint3 base = uint3(floor(coord));
    float3 f = frac(coord);
    float3 sum = 0.0f;
    float weight_sum = 0.0f;

    [unroll] for (uint z = 0u; z < 2u; ++z)
    [unroll] for (uint y = 0u; y < 2u; ++y)
    [unroll] for (uint x = 0u; x < 2u; ++x) {
        uint3 c = min(base + uint3(x,y,z), probe_dims.xyz - 1u);
        float3 aw = lerp(1.0f - f, f, float3(x,y,z));
        float w = aw.x * aw.y * aw.z;
        Probe probe = Probes[c.x + probe_dims.x * (c.y + probe_dims.y * c.z)];
        w *= saturate(probe.position.w);
        if (w <= 0.0f) continue;
        sum += probe_value(probe, n) * w;
        weight_sum += w;
    }

    if (weight_sum > 0.0f) return sum / weight_sum;

    float best_distance2 = 1.0e30f;
    uint best_index = 0u;
    bool found = false;
    int3 center = int3(floor(coord + 0.5f));
    [unroll] for (int z = -1; z <= 1; ++z)
    [unroll] for (int y = -1; y <= 1; ++y)
    [unroll] for (int x = -1; x <= 1; ++x) {
        int3 c = center + int3(x, y, z);
        if (any(c < 0) || any(c >= int3(probe_dims.xyz))) continue;
        uint index = (uint)c.x + probe_dims.x * ((uint)c.y + probe_dims.y * (uint)c.z);
        Probe probe = Probes[index];
        if (probe.position.w <= 0.0f) continue;
        float3 delta = probe.position.xyz - p;
        float distance2 = dot(delta, delta);
        if (distance2 < best_distance2) {
            best_distance2 = distance2;
            best_index = index;
            found = true;
        }
    }

    return found ? probe_value(Probes[best_index], n) : 0.12f.xxx;
}

float3 sky(float3 d)
{
    float t = pow(saturate(d.y * 0.5f + 0.5f), 0.35f);
    return lerp(sky_horizon.rgb, sky_zenith.rgb, t) * sky_zenith.w;
}

float3 sun(float3 p, float3 n)
{
    float3 l = normalize(sun_direction_intensity.xyz);
    float ndotl = saturate(dot(n, l));
    if (ndotl <= 0.0f) return 0.0f;
    Ray ray;
    ray.origin = p + n * sun_color_epsilon.w;
    ray.tmin = sun_color_epsilon.w;
    ray.direction = l;
    ray.tmax = 1.0e20f;
    Hit blocker;
    if (scene_closest(ray, blocker)) return 0.0f;
    return sun_color_epsilon.rgb * (sun_direction_intensity.w * ndotl);
}

float3 static_emissive(float3 p, float3 n, inout uint seed)
{
    uint triangle_count = 0u, triangle_stride = 0u;
    StaticTriangles.GetDimensions(triangle_count, triangle_stride);
    if (triangle_count == 0u) return 0.0f;

    float total_weight = StaticTriangles[triangle_count - 1u].emissive.w;
    if (total_weight <= 0.0f) return 0.0f;

    float target = random01(seed) * total_weight;
    uint lo = 0u, hi = triangle_count;
    while (lo < hi) {
        uint mid = lo + (hi - lo) / 2u;
        if (StaticTriangles[mid].emissive.w > target) hi = mid;
        else lo = mid + 1u;
    }
    if (lo >= triangle_count) return 0.0f;

    BvhTriangle tri = StaticTriangles[lo];
    float previous = lo == 0u ? 0.0f : StaticTriangles[lo - 1u].emissive.w;
    float triangle_weight = tri.emissive.w - previous;
    if (triangle_weight <= 0.0f) return 0.0f;

    float3 edge1 = tri.b.xyz - tri.a.xyz;
    float3 edge2 = tri.c.xyz - tri.a.xyz;
    float area = 0.5f * length(cross(edge1, edge2));
    if (area <= 1.0e-10f) return 0.0f;

    float root = sqrt(random01(seed));
    float bary = random01(seed);
    float3 light_position = tri.a.xyz + edge1 * (root * (1.0f - bary)) + edge2 * (root * bary);
    float3 delta = light_position - p;
    float distance2 = dot(delta, delta);
    float epsilon = sun_color_epsilon.w;
    if (distance2 <= epsilon * epsilon) return 0.0f;

    float distance = sqrt(distance2);
    float3 direction = delta / distance;
    float receiver_cosine = saturate(dot(n, direction));
    float emitter_cosine = saturate(dot(normalize(tri.normal.xyz), -direction));
    if (receiver_cosine <= 0.0f || emitter_cosine <= 0.0f) return 0.0f;

    Ray shadow;
    shadow.origin = p + n * epsilon;
    shadow.tmin = epsilon;
    shadow.direction = direction;
    shadow.tmax = max(epsilon, distance - 2.0f * epsilon);
    Hit blocker;
    if (shadow.tmax > shadow.tmin && scene_closest(shadow, blocker)) return 0.0f;

    float pdf_area = (triangle_weight / total_weight) / area;
    if (pdf_area <= 1.0e-12f) return 0.0f;

    return max(tri.emissive.rgb, 0.0f) *
           (receiver_cosine * emitter_cosine /
            max(PI * distance2 * pdf_area, 1.0e-8f));
}

float3 trace_indirect(float3 p, float3 n, inout uint seed)
{
    float3 d = cosine_direction(n, seed);
    Ray ray;
    ray.origin = p + n * sun_color_epsilon.w;
    ray.tmin = sun_color_epsilon.w;
    ray.direction = d;
    ray.tmax = 1.0e20f;
    Hit hit;
    if (!scene_closest(ray, hit)) return sky(d);
    float3 hp = ray.origin + d * hit.t;
    float3 emitted = hit.dynamic != 0u ? hit.emissive : 0.0f;
    return emitted + hit.albedo * (probe_irradiance(hp, hit.normal) / PI + sun(hp, hit.normal));
}

[numthreads(64, 1, 1)]
void dynamic_gi_cs(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= job_count) return;
    GiJob job = Jobs[id.x];
    if (CellGenerations[job.cell] != job.generation) return;

    uint pixel = asuint(job.position.w);
    uint pixels = lightmap_width * lightmap_height;
    if (pixel >= pixels) return;
    uint2 xy = uint2(pixel % lightmap_width, pixel / lightmap_width);

    float3 n = normalize(job.normal.xyz);
    uint seed = hash_u32(pixel ^ (job.generation * 0x9e3779b9u) ^ ((frame_index + 1u) * 0x85ebca6bu));
    float3 radiance = 0.0f;
    [loop] for (uint i = 0u; i < rays_per_texel; ++i) {
        radiance += trace_indirect(job.position.xyz, n, seed);
        radiance += static_emissive(job.position.xyz, n, seed);
    }
    radiance /= max((float)rays_per_texel, 1.0f);

    bool first_sweep = (job.flags & 2u) != 0u;
    bool valid = !first_sweep && OverlayGenerations[job.cell] == job.generation;
    float4 old = valid ? Overlay[xy] : 0.0f;
    float count = valid ? old.a : 0.0f;
    float next = count + 1.0f;
    Overlay[xy] = float4((old.rgb * count + max(radiance, 0.0f)) / next, next);

    if ((job.flags & 1u) != 0u && CellGenerations[job.cell] == job.generation)
        OverlayGenerations[job.cell] = job.generation;
}

#endif
