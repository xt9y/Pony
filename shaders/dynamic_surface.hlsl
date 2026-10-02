#ifndef GPU_BIND_S
#define GPU_BIND_S(n, s) [[vk::binding(n, s)]]
#define GPU_BIND_T(n, s) [[vk::binding(n + 16, s)]]
#define GPU_BIND_B(n, s) [[vk::binding(n + 32, s)]]
#define GPU_BIND_U(n, s) [[vk::binding(n + 48, s)]]
#define GPU_STORAGE_RGBA16F [[vk::image_format("rgba16f")]]
#endif

#if defined(BUILD_DYNAMIC_SURFACE_CS)

static const float PI = 3.14159265358979323846f;
static const uint INVALID_NODE = 0xffffffffu;
static const uint DYNAMIC_RAYS_PER_SAMPLE = 8u;

struct SurfaceSample {
    float4 position;
    float4 normal;
};

struct BvhNode {
    float4 bmin;
    float4 bmax;
    uint4 meta;
};

struct BvhTriangle {
    float4 a;
    float4 b;
    float4 c;
    float4 normal;
    float4 emissive;
};

struct StaticSurfaceRef {
    uint source_triangle;
    uint3 _pad;
};

struct TraceRay {
    float3 origin;
    float tmin;
    float3 direction;
    float tmax;
};

GPU_BIND_T(0, 0) StructuredBuffer<SurfaceSample> Samples : register(t0, space0);
GPU_BIND_T(1, 0) StructuredBuffer<float> Beams : register(t1, space0);
GPU_BIND_T(2, 0) StructuredBuffer<BvhNode> Nodes : register(t2, space0);
GPU_BIND_T(3, 0) StructuredBuffer<BvhTriangle> Triangles : register(t3, space0);
GPU_BIND_T(4, 0) StructuredBuffer<StaticSurfaceRef> SurfaceRefs : register(t4, space0);
GPU_BIND_T(5, 0) StructuredBuffer<float2> StaticUVs : register(t5, space0);
GPU_BIND_T(6, 0) Texture2D<float4> StaticLightmap : register(t6, space0);
GPU_BIND_S(0, 0) SamplerState StaticLightmapSampler : register(s0, space0);
GPU_BIND_U(0, 1) GPU_STORAGE_RGBA16F RWTexture2D<float4> Output : register(u0, space1);

GPU_BIND_B(0, 2) cbuffer DynamicSurfaceData : register(b0, space2) {
    float4x4 model;
    float4x4 normal_model;

    uint sample_offset;
    uint sample_count;
    uint texture_width;
    uint texture_height;

    float4 beam_origin;
    float4 beam_step;
    uint4 beam_dims;

    float4 sun_direction_intensity;
    float4 sun_color_visibility_floor;
    float4 sky_zenith;
    float4 sky_horizon;
    float4 trace_params;
};

uint hash_u32(uint x) {
    x ^= x >> 16u;
    x *= 0x7feb352du;
    x ^= x >> 15u;
    x *= 0x846ca68bu;
    return x ^ (x >> 16u);
}

float random01(inout uint seed) {
    seed = hash_u32(seed + 0x9e3779b9u);
    return (float)(seed & 0x00ffffffu) * (1.0f / 16777216.0f);
}

void basis(float3 normal, out float3 tangent, out float3 bitangent) {
    float3 helper = abs(normal.y) < 0.999f ? float3(0.0f, 1.0f, 0.0f) : float3(1.0f, 0.0f, 0.0f);
    tangent = normalize(cross(helper, normal));
    bitangent = cross(normal, tangent);
}

float3 cosine_hemisphere(float3 normal, inout uint seed) {
    float u1 = random01(seed);
    float u2 = random01(seed);
    float radius = sqrt(u1);
    float phi = 2.0f * PI * u2;
    float3 local = float3(radius * cos(phi), sqrt(max(0.0f, 1.0f - u1)), radius * sin(phi));
    float3 tangent, bitangent;
    basis(normal, tangent, bitangent);
    return normalize(tangent * local.x + normal * local.y + bitangent * local.z);
}

float3 sky_radiance(float3 direction) {
    float t = saturate(normalize(direction).y * 0.5f + 0.5f);
    t = pow(t, 0.35f);
    return lerp(sky_horizon.rgb, sky_zenith.rgb, t) * sky_zenith.w;
}

bool trace_box(TraceRay ray, BvhNode node, float max_t) {
    float lo = ray.tmin;
    float hi = min(ray.tmax, max_t);

    [unroll] for (uint axis = 0u; axis < 3u; ++axis) {
        float direction = ray.direction[axis];

        if (abs(direction) < 1.0e-7f) {
            if (ray.origin[axis] < node.bmin[axis] || ray.origin[axis] > node.bmax[axis]) return false;
        } else {
            float a = (node.bmin[axis] - ray.origin[axis]) / direction;
            float b = (node.bmax[axis] - ray.origin[axis]) / direction;

            if (a > b) {
                float tmp = a;
                a = b;
                b = tmp;
            }

            lo = max(lo, a);
            hi = min(hi, b);
            if (lo > hi) return false;
        }
    }

    return hi >= ray.tmin;
}

bool trace_triangle(TraceRay ray, BvhTriangle tri, float max_t, out float hit_t, out float hit_u, out float hit_v) {
    float3 edge1 = tri.b.xyz - tri.a.xyz;
    float3 edge2 = tri.c.xyz - tri.a.xyz;
    float3 p = cross(ray.direction, edge2);
    float determinant = dot(edge1, p);

    if (abs(determinant) < 1.0e-7f) {
        hit_t = 0.0f;
        hit_u = 0.0f;
        hit_v = 0.0f;
        return false;
    }

    float inverse = 1.0f / determinant;
    float3 s = ray.origin - tri.a.xyz;
    float u = dot(s, p) * inverse;

    if (u < 0.0f || u > 1.0f) {
        hit_t = 0.0f;
        hit_u = 0.0f;
        hit_v = 0.0f;
        return false;
    }

    float3 q = cross(s, edge1);
    float v = dot(ray.direction, q) * inverse;

    if (v < 0.0f || u + v > 1.0f) {
        hit_t = 0.0f;
        hit_u = 0.0f;
        hit_v = 0.0f;
        return false;
    }

    float t = dot(edge2, q) * inverse;
    if (t <= ray.tmin || t >= min(ray.tmax, max_t)) {
        hit_t = 0.0f;
        hit_u = 0.0f;
        hit_v = 0.0f;
        return false;
    }

    hit_t = t;
    hit_u = u;
    hit_v = v;
    return true;
}

bool static_closest(TraceRay ray, out uint triangle_index, out float3 barycentric, out bool back_face) {
    uint node_index = 0u;
    float closest = ray.tmax;
    bool found = false;
    triangle_index = INVALID_NODE;
    barycentric = 0.0f;
    back_face = false;

    while (node_index != INVALID_NODE) {
        BvhNode node = Nodes[node_index];

        if (!trace_box(ray, node, closest)) {
            node_index = node.meta.y;
            continue;
        }

        if (node.meta.w != 0u) {
            for (uint i = 0u; i < node.meta.w; ++i) {
                uint candidate = node.meta.z + i;
                BvhTriangle tri = Triangles[candidate];
                if (tri.normal.w >= 0.999f) continue;

                float t, u, v;
                if (!trace_triangle(ray, tri, closest, t, u, v)) continue;

                closest = t;
                triangle_index = candidate;
                barycentric = float3(1.0f - u - v, u, v);
                back_face = dot(normalize(tri.normal.xyz), ray.direction) > 0.0f;
                found = true;
            }

            node_index = node.meta.y;
        } else {
            node_index = node.meta.x;
        }
    }

    return found;
}

float2 static_lightmap_uv(uint triangle_index, float3 barycentric, bool back_face) {
    uint source_triangle = SurfaceRefs[triangle_index].source_triangle;
    uint first = source_triangle * 6u + (back_face ? 3u : 0u);
    return StaticUVs[first] * barycentric.x + StaticUVs[first + 1u] * barycentric.y + StaticUVs[first + 2u] * barycentric.z;
}

float3 static_outgoing(uint triangle_index, float3 barycentric, bool back_face) {
    BvhTriangle tri = Triangles[triangle_index];
    float2 uv = static_lightmap_uv(triangle_index, barycentric, back_face);
    float3 lighting = max(StaticLightmap.SampleLevel(StaticLightmapSampler, uv, 0.0f).rgb, 0.0f);
    float3 albedo = max(float3(tri.a.w, tri.b.w, tri.c.w), 0.0f);
    float3 emissive = max(tri.emissive.rgb, 0.0f);
    return albedo * lighting + emissive;
}

float3 trace_static_indirect(float3 position, float3 normal, uint sample_id) {
    float3 sum = 0.0f;
    float epsilon = max(trace_params.x, 1.0e-5f);

    [loop] for (uint ray_index = 0u; ray_index < DYNAMIC_RAYS_PER_SAMPLE; ++ray_index) {
        uint seed = hash_u32((sample_id + 1u) * 0x9e3779b9u ^ (ray_index + 1u) * 0x85ebca6bu);
        float3 direction = cosine_hemisphere(normal, seed);
        TraceRay ray;
        ray.origin = position + normal * epsilon;
        ray.tmin = epsilon;
        ray.direction = direction;
        ray.tmax = 1.0e20f;

        uint triangle_index;
        float3 barycentric;
        bool back_face;

        if (static_closest(ray, triangle_index, barycentric, back_face))
            sum += static_outgoing(triangle_index, barycentric, back_face);
        else
            sum += sky_radiance(direction);
    }

    return sum / (float)DYNAMIC_RAYS_PER_SAMPLE;
}

float beam_visibility(float3 position) {
    if (beam_dims.w == 0u || beam_dims.x == 0u || beam_dims.y == 0u || beam_dims.z == 0u ||
        beam_step.x <= 0.0f || beam_step.y <= 0.0f || beam_step.z <= 0.0f)
        return 1.0f;

    float3 sun = normalize(sun_direction_intensity.xyz);
    float3 helper = abs(sun.y) < 0.999f ? float3(0.0f, 1.0f, 0.0f) : float3(1.0f, 0.0f, 0.0f);
    float3 u = normalize(cross(helper, sun));
    float3 v = cross(sun, u);
    float3 q = float3(dot(position, u), dot(position, v), dot(position, sun));
    float3 coord = (q - beam_origin.xyz) / beam_step.xyz - 0.5f;

    if (any(coord < -0.5f) || any(coord > float3(beam_dims.xyz) - 0.5f)) return 1.0f;

    int3 base = int3(floor(coord));
    float3 fraction = frac(coord);
    float visibility = 0.0f;
    float total = 0.0f;

    [unroll] for (uint z = 0u; z < 2u; ++z)
    [unroll] for (uint y = 0u; y < 2u; ++y)
    [unroll] for (uint x = 0u; x < 2u; ++x) {
        int3 cell = base + int3(x, y, z);
        if (any(cell < 0) || any(cell >= int3(beam_dims.xyz))) continue;

        float3 axis_weight = lerp(1.0f - fraction, fraction, float3(x, y, z));
        float weight = axis_weight.x * axis_weight.y * axis_weight.z;
        uint index = (uint)cell.x + beam_dims.x * ((uint)cell.y + beam_dims.y * (uint)cell.z);
        visibility += Beams[index] * weight;
        total += weight;
    }

    return total > 0.0f ? saturate(visibility / total) : 1.0f;
}

[numthreads(64, 1, 1)]
void dynamic_surface_cs(uint3 id : SV_DispatchThreadID) {
    if (id.x >= sample_count) return;

    uint sample_id = sample_offset + id.x;
    SurfaceSample sample = Samples[sample_id];
    uint pixel = asuint(sample.position.w);
    uint pixel_count = texture_width * texture_height;
    if (pixel >= pixel_count) return;

    float3 world_position = mul(model, float4(sample.position.xyz, 1.0f)).xyz;
    float3 world_normal = normalize(mul((float3x3)normal_model, sample.normal.xyz));

    float3 indirect = trace_static_indirect(world_position, world_normal, sample_id);
    float visibility = beam_visibility(world_position);
    float n_dot_l = saturate(dot(world_normal, normalize(sun_direction_intensity.xyz)));
    float3 direct = sun_color_visibility_floor.rgb * (sun_direction_intensity.w * n_dot_l * visibility);
    float visibility_floor = max(sun_color_visibility_floor.w, 1.0f / 65504.0f);
    float encoded_visibility = visibility_floor + visibility * (1.0f - visibility_floor);

    Output[uint2(pixel % texture_width, pixel / texture_width)] = float4(max(indirect + direct, 0.0f), encoded_visibility);
}

#endif
