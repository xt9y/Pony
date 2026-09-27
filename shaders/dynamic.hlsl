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
    uv.y = 1.0f - uv.y;
    return float4(uv * 2.0f - 1.0f, saturate(depth), 1.0f);
}

#elif defined(BUILD_DYNAMIC_CELL_CS)

struct CellUpdate { uint cell; uint generation; };
GPU_BIND_T(0, 0) StructuredBuffer<CellUpdate> Updates : register(t0, space0);
GPU_BIND_U(0, 1) RWStructuredBuffer<uint> CellGenerations : register(u0, space1);
GPU_BIND_B(0, 2) cbuffer CellData : register(b0, space2)
{
    uint update_count;
    uint3 _cell_pad;
};

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
struct DynamicInstance
{
    float4x4 world;
    float4x4 inverse_world;
    float4x4 normal_world;
    float4 bounds_min;
    float4 bounds_max;
    uint4 indices;
};
struct Probe { float4 position; float4 coefficient[9]; };
struct GiJob
{
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
static const uint EMITTER_VISIBILITY_SAMPLES = 1u;

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

bool hit_box(Ray ray, float3 bmin, float3 bmax, float max_t)
{
    float lo = ray.tmin;
    float hi = min(ray.tmax, max_t);
    [unroll] for (uint axis = 0u; axis < 3u; ++axis)
    {
        float direction = ray.direction[axis];
        if (abs(direction) < 1.0e-7f)
        {
            if (ray.origin[axis] < bmin[axis] || ray.origin[axis] > bmax[axis]) return false;
        }
        else
        {
            float a = (bmin[axis] - ray.origin[axis]) / direction;
            float b = (bmax[axis] - ray.origin[axis]) / direction;
            if (a > b) { float tmp = a; a = b; b = tmp; }
            lo = max(lo, a);
            hi = min(hi, b);
            if (lo > hi) return false;
        }
    }
    return hi >= ray.tmin;
}

bool hit_triangle(Ray ray, BvhTriangle tri, float max_t, out float t)
{
    float3 edge1 = tri.b.xyz - tri.a.xyz;
    float3 edge2 = tri.c.xyz - tri.a.xyz;
    float3 p = cross(ray.direction, edge2);
    float determinant = dot(edge1, p);
    if (abs(determinant) < 1.0e-7f) { t = 0.0f; return false; }

    float inverse = 1.0f / determinant;
    float3 s = ray.origin - tri.a.xyz;
    float u = dot(s, p) * inverse;
    float3 q = cross(s, edge1);
    float v = dot(ray.direction, q) * inverse;
    t = dot(edge2, q) * inverse;
    return u >= 0.0f && v >= 0.0f && u + v <= 1.0f &&
           t > ray.tmin && t < min(ray.tmax, max_t);
}

bool static_closest(Ray ray, inout Hit hit)
{
    uint node = 0u;
    float closest = hit.t;
    bool found = false;

    while (node != INVALID_NODE)
    {
        BvhNode bvh = StaticNodes[node];
        if (!hit_box(ray, bvh.bmin.xyz, bvh.bmax.xyz, closest))
        {
            node = bvh.meta.y;
            continue;
        }

        if (bvh.meta.w != 0u)
        {
            for (uint i = 0u; i < bvh.meta.w; ++i)
            {
                uint triangle_index = bvh.meta.z + i;
                float t;
                if (!hit_triangle(ray, StaticTriangles[triangle_index], closest, t)) continue;
                BvhTriangle tri = StaticTriangles[triangle_index];
                float3 normal = normalize(tri.normal.xyz);
                if (dot(normal, ray.direction) > 0.0f) normal = -normal;
                closest = hit.t = t;
                hit.normal = normal;
                hit.albedo = saturate(float3(tri.a.w, tri.b.w, tri.c.w));
                hit.emissive = max(tri.emissive.rgb, 0.0f);
                hit.dynamic = 0u;
                found = true;
            }
            node = bvh.meta.y;
        }
        else
        {
            node = bvh.meta.x;
        }
    }

    return found;
}

bool dynamic_closest(Ray world_ray, inout Hit hit)
{
    bool found = false;

    for (uint instance_index = 0u; instance_index < dynamic_instance_count; ++instance_index)
    {
        DynamicInstance instance = DynamicInstances[instance_index];
        if (!hit_box(world_ray, instance.bounds_min.xyz, instance.bounds_max.xyz, hit.t)) continue;

        Ray ray;
        ray.origin = mul(instance.inverse_world, float4(world_ray.origin, 1.0f)).xyz;
        ray.direction = mul((float3x3)instance.inverse_world, world_ray.direction);
        ray.tmin = world_ray.tmin;
        ray.tmax = min(world_ray.tmax, hit.t);

        uint node = instance.indices.x;
        float closest = hit.t;
        while (node != INVALID_NODE)
        {
            BvhNode bvh = DynamicNodes[node];
            if (!hit_box(ray, bvh.bmin.xyz, bvh.bmax.xyz, closest))
            {
                node = bvh.meta.y;
                continue;
            }

            if (bvh.meta.w != 0u)
            {
                for (uint i = 0u; i < bvh.meta.w; ++i)
                {
                    uint triangle_index = bvh.meta.z + i;
                    float t;
                    if (!hit_triangle(ray, DynamicTriangles[triangle_index], closest, t)) continue;
                    BvhTriangle tri = DynamicTriangles[triangle_index];
                    float3 normal = normalize(mul((float3x3)instance.normal_world, tri.normal.xyz));
                    if (dot(normal, world_ray.direction) > 0.0f) normal = -normal;
                    closest = hit.t = t;
                    hit.normal = normal;
                    hit.albedo = saturate(float3(tri.a.w, tri.b.w, tri.c.w));
                    hit.emissive = max(tri.emissive.rgb, 0.0f);
                    hit.dynamic = 1u;
                    found = true;
                }
                node = bvh.meta.y;
            }
            else
            {
                node = bvh.meta.x;
            }
        }
    }

    return found;
}

bool static_scene_closest(Ray ray, out Hit hit)
{
    hit.t = ray.tmax;
    hit.normal = 0.0f;
    hit.albedo = 0.0f;
    hit.emissive = 0.0f;
    hit.dynamic = 0u;
    return static_closest(ray, hit);
}

void emissive_visibility_sample(float3 position, float3 normal, uint seed,
                                out float static_visible, out float dynamic_visible)
{
    static_visible = 0.0f;
    dynamic_visible = 0.0f;

    uint triangle_count = 0u;
    uint triangle_stride = 0u;
    StaticTriangles.GetDimensions(triangle_count, triangle_stride);
    if (triangle_count == 0u) return;

    float total_weight = StaticTriangles[triangle_count - 1u].emissive.w;
    if (total_weight <= 0.0f) return;

    float target = random01(seed) * total_weight;
    uint lo = 0u;
    uint hi = triangle_count;
    while (lo < hi)
    {
        uint mid = lo + (hi - lo) / 2u;
        if (StaticTriangles[mid].emissive.w > target) hi = mid;
        else lo = mid + 1u;
    }
    if (lo >= triangle_count) return;

    BvhTriangle tri = StaticTriangles[lo];
    float previous = lo == 0u ? 0.0f : StaticTriangles[lo - 1u].emissive.w;
    if (tri.emissive.w <= previous) return;

    float root = sqrt(random01(seed));
    float barycentric = random01(seed);
    float3 light_position = tri.a.xyz * (1.0f - root) +
        tri.b.xyz * (root * (1.0f - barycentric)) +
        tri.c.xyz * (root * barycentric);

    float3 delta = light_position - position;
    float distance2 = dot(delta, delta);
    float epsilon = sun_color_epsilon.w;
    if (distance2 <= epsilon * epsilon) return;

    float distance = sqrt(distance2);
    float3 direction = delta / distance;
    if (dot(normal, direction) <= 0.0f ||
        dot(normalize(tri.normal.xyz), -direction) <= 0.0f)
        return;

    Ray shadow;
    shadow.origin = position + normal * epsilon;
    shadow.tmin = epsilon;
    shadow.direction = direction;
    shadow.tmax = max(epsilon, distance - 2.0f * epsilon);
    if (shadow.tmax <= shadow.tmin) return;

    Hit blocker;
    if (static_scene_closest(shadow, blocker)) return;
    static_visible = 1.0f;

    blocker.t = shadow.tmax;
    blocker.normal = 0.0f;
    blocker.albedo = 0.0f;
    blocker.emissive = 0.0f;
    blocker.dynamic = 0u;
    if (!dynamic_closest(shadow, blocker)) dynamic_visible = 1.0f;
}

[numthreads(64, 1, 1)]
void dynamic_gi_cs(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= job_count) return;

    GiJob job = Jobs[id.x];
    if (CellGenerations[job.cell] != job.generation) return;

    uint pixel = asuint(job.position.w);
    uint pixel_count = lightmap_width * lightmap_height;
    if (pixel >= pixel_count) return;
    uint2 xy = uint2(pixel % lightmap_width, pixel / lightmap_width);

    bool first_sweep = (job.flags & 2u) != 0u;
    bool valid = !first_sweep && OverlayGenerations[job.cell] == job.generation;
    float4 old = valid ? Overlay[xy] : 0.0f;
    uint sample_base = valid ? (uint)old.a : 0u;

    float dynamic_visible = 0.0f;
    float static_visible = 0.0f;
    float3 normal = normalize(job.normal.xyz);

    [unroll] for (uint sample_index = 0u; sample_index < EMITTER_VISIBILITY_SAMPLES; ++sample_index)
    {
        uint global_sample = sample_base + sample_index;
        uint seed = hash_u32((global_sample + 1u) * 0x9e3779b9u);
        float static_sample;
        float dynamic_sample;
        emissive_visibility_sample(job.position.xyz, normal, seed,
                                   static_sample, dynamic_sample);
        static_visible += static_sample;
        dynamic_visible += dynamic_sample;
    }

    Overlay[xy] = float4(old.r + dynamic_visible,
                         old.g + static_visible,
                         0.0f,
                         old.a + (float)EMITTER_VISIBILITY_SAMPLES);

    if ((job.flags & 1u) != 0u && CellGenerations[job.cell] == job.generation)
        OverlayGenerations[job.cell] = job.generation;
}

#endif