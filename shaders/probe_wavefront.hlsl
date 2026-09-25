#if defined(BUILD_PROBE_PREP_CS) || defined(BUILD_PROBE_RESET_CS) || \
      defined(BUILD_PROBE_VALIDATE_CS) || defined(BUILD_PROBE_PRIMARY_CS) || \
      defined(BUILD_PROBE_BOUNCE_CS) || defined(BUILD_PROBE_REDUCE_CS) || \
      defined(BUILD_PROBE_EMISSIVE_CS)

static const float PROBE_PI = 3.14159265358979323846f;
static const float PROBE_SH0 = 0.2820947918f;
static const uint PROBE_INVALID = 0xffffffffu;
static const uint PROBE_LEAF_BIT = 0x80000000u;
static const uint PROBE_FIRST_MASK = 0x07ffffffu;
static const uint PROBE_BLOCK_SAMPLES = 128u;

struct SourceBvhNode
{
    float4 bmin;
    float4 bmax;
    uint4 meta;
};

struct SourceBvhTriangle
{
    float4 a;
    float4 b;
    float4 c;
    float4 normal;
    float4 emissive;
};

struct PackedProbeNode
{
    float3 bmin;
    uint meta0;
    float3 bmax;
    uint meta1;
};

struct PackedProbeTriangle
{
    float4 a;
    float4 edge1;
    float4 edge2;
    float4 normal;
    float4 emissive;
};

struct ProbeTraceRay
{
    float3 origin;
    float tmin;
    float3 direction;
    float tmax;
    float3 inv_direction;
    float _pad;
};

struct ProbeTraceHit
{
    float t;
    float3 normal;
    float3 albedo;
    uint triangle_index;
    float3 emissive;
};

struct ProbeRayState
{
    float4 position;
    float4 normal;
    float4 throughput;
    float4 radiance;
    uint4 meta;
};

struct ProbeAccum
{
    float4 radiance_count;
    float4 stats;
};

GPU_BIND_B(0, 2) cbuffer ProbeBakeData : register(b0, space2)
{
    uint probe_count;
    uint sample_offset;
    uint samples_per_block;
    uint total_samples;
    uint node_count;
    uint triangle_count;
    uint bounce_index;
    uint max_bounces;
    uint beam_depth;
    uint emissive_samples;
    uint _probe_pad1;
    uint _probe_pad2;
    float4 sun_direction_intensity;
    float4 sun_color_radius;
    float4 sky_zenith;
    float4 sky_horizon;
    float4 bake_params;
    float4 probe_beam_origin;
    float4 probe_beam_step;
};

uint probe_hash(uint x)
{
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

float probe_random(inout uint state)
{
    state = probe_hash(state + 0x9e3779b9u);
    return (state & 0x00ffffffu) / 16777216.0f;
}

float3 probe_sky(float3 direction)
{
    float t = saturate(direction.y * 0.5f + 0.5f);
    t = pow(t, 0.35f);
    return lerp(sky_horizon.rgb, sky_zenith.rgb, t) * bake_params.z;
}

void probe_frame(float3 normal, out float3 tangent, out float3 bitangent)
{
    float3 helper = abs(normal.y) < 0.999f
        ? float3(0.0f, 1.0f, 0.0f)
        : float3(1.0f, 0.0f, 0.0f);
    tangent = normalize(cross(helper, normal));
    bitangent = cross(normal, tangent);
}

float3 probe_cosine_hemisphere(float3 normal, inout uint seed)
{
    float u1 = probe_random(seed);
    float u2 = probe_random(seed);
    float r = sqrt(u1);
    float phi = 2.0f * PROBE_PI * u2;
    float3 local = float3(r * cos(phi), sqrt(max(0.0f, 1.0f - u1)), r * sin(phi));
    float3 tangent, bitangent;
    probe_frame(normal, tangent, bitangent);
    return normalize(tangent * local.x + normal * local.y + bitangent * local.z);
}

float3 probe_sample_sun(float3 direction, float radius, inout uint seed)
{
    float3 tangent, bitangent;
    probe_frame(direction, tangent, bitangent);
    float r = sqrt(probe_random(seed)) * tan(radius);
    float phi = 2.0f * PROBE_PI * probe_random(seed);
    return normalize(direction + tangent * (cos(phi) * r) + bitangent * (sin(phi) * r));
}

float2 probe_r2(uint probe_index, uint sample_index)
{
    uint h0 = probe_hash(probe_index * 0x9e3779b9u + 0x68bc21ebu);
    uint h1 = probe_hash(probe_index * 0x85ebca6bu + 0x02e5be93u);
    float2 rotation = float2(h0 & 0x00ffffffu, h1 & 0x00ffffffu) / 16777216.0f;
    float n = (float)sample_index + 1.0f;
    return frac(rotation + 0.5f + n * float2(0.7548776662466927f, 0.5698402909980532f));
}

float3 probe_sphere_direction(uint probe_index, uint sample_index)
{
    float2 xi = probe_r2(probe_index, sample_index);
    float z = 1.0f - 2.0f * xi.x;
    float phi = 2.0f * PROBE_PI * xi.y;
    float radius = sqrt(max(0.0f, 1.0f - z * z));
    return float3(radius * cos(phi), radius * sin(phi), z);
}

void probe_sh_basis(float3 direction, out float basis[9])
{
    direction = normalize(direction);
    float x = direction.x, y = direction.y, z = direction.z;
    basis[0] = 0.2820947918f;
    basis[1] = 0.4886025119f * y;
    basis[2] = 0.4886025119f * z;
    basis[3] = 0.4886025119f * x;
    basis[4] = 1.0925484306f * x * y;
    basis[5] = 1.0925484306f * y * z;
    basis[6] = 0.3153915653f * (3.0f * z * z - 1.0f);
    basis[7] = 1.0925484306f * x * z;
    basis[8] = 0.5462742153f * (x * x - y * y);
}

float3 probe_safe_inverse(float3 direction)
{
    return float3(
        abs(direction.x) < 1.0e-8f ? 0.0f : rcp(direction.x),
        abs(direction.y) < 1.0e-8f ? 0.0f : rcp(direction.y),
        abs(direction.z) < 1.0e-8f ? 0.0f : rcp(direction.z));
}

ProbeTraceRay probe_make_ray(float3 origin, float3 direction, float tmin, float tmax)
{
    ProbeTraceRay ray;
    ray.origin = origin;
    ray.tmin = tmin;
    ray.direction = direction;
    ray.tmax = tmax;
    ray.inv_direction = probe_safe_inverse(direction);
    ray._pad = 0.0f;
    return ray;
}

bool probe_box_near(ProbeTraceRay ray, PackedProbeNode node, float max_t, out float near_t)
{
    float lo = ray.tmin;
    float hi = min(ray.tmax, max_t);
    if (hi < lo) {
        near_t = lo;
        return false;
    }

    [unroll] for (uint axis = 0u; axis < 3u; ++axis) {
        float d = ray.direction[axis];
        if (abs(d) < 1.0e-7f) {
            if (ray.origin[axis] < node.bmin[axis] || ray.origin[axis] > node.bmax[axis]) {
                near_t = lo;
                return false;
            }
            continue;
        }

        float a = (node.bmin[axis] - ray.origin[axis]) * ray.inv_direction[axis];
        float b = (node.bmax[axis] - ray.origin[axis]) * ray.inv_direction[axis];
        lo = max(lo, min(a, b));
        hi = min(hi, max(a, b));
        if (lo > hi) {
            near_t = lo;
            return false;
        }
    }

    near_t = lo;
    return hi >= ray.tmin;
}

bool probe_triangle_hit(ProbeTraceRay ray, PackedProbeTriangle tri,
                        float max_t, out float hit_t)
{
    float3 p = cross(ray.direction, tri.edge2.xyz);
    float det = dot(tri.edge1.xyz, p);
    if (abs(det) < 1.0e-7f) {
        hit_t = 0.0f;
        return false;
    }

    float inv_det = rcp(det);
    float3 s = ray.origin - tri.a.xyz;
    float u = dot(s, p) * inv_det;
    if (u < 0.0f || u > 1.0f) {
        hit_t = 0.0f;
        return false;
    }

    float3 q = cross(s, tri.edge1.xyz);
    float v = dot(ray.direction, q) * inv_det;
    if (v < 0.0f || u + v > 1.0f) {
        hit_t = 0.0f;
        return false;
    }

    float t = dot(tri.edge2.xyz, q) * inv_det;
    if (t <= ray.tmin || t >= min(ray.tmax, max_t)) {
        hit_t = 0.0f;
        return false;
    }

    hit_t = t;
    return true;
}

#if defined(BUILD_PROBE_PREP_CS)
GPU_BIND_T(0, 0) StructuredBuffer<SourceBvhNode> ProbeSourceNodes : register(t0, space0);
GPU_BIND_T(1, 0) StructuredBuffer<SourceBvhTriangle> ProbeSourceTriangles : register(t1, space0);
GPU_BIND_U(0, 1) RWStructuredBuffer<PackedProbeNode> ProbePackedNodesOut : register(u0, space1);
GPU_BIND_U(1, 1) RWStructuredBuffer<PackedProbeTriangle> ProbePackedTrianglesOut : register(u1, space1);

[numthreads(64, 1, 1)]
void probe_prepare_cs(uint3 id : SV_DispatchThreadID)
{
    uint index = id.x;
    if (index < node_count) {
        SourceBvhNode source = ProbeSourceNodes[index];
        PackedProbeNode packed;
        packed.bmin = source.bmin.xyz;
        packed.bmax = source.bmax.xyz;
        if (source.meta.w != 0u) {
            packed.meta0 = PROBE_LEAF_BIT |
                ((source.meta.w & 0x0fu) << 27u) |
                (source.meta.z & PROBE_FIRST_MASK);
            packed.meta1 = source.meta.y;
        } else {
            packed.meta0 = source.meta.x;
            packed.meta1 = source.meta.y;
        }
        ProbePackedNodesOut[index] = packed;
    }

    if (index < triangle_count) {
        SourceBvhTriangle source = ProbeSourceTriangles[index];
        PackedProbeTriangle packed;
        packed.a = source.a;
        packed.edge1 = float4(source.b.xyz - source.a.xyz, source.b.w);
        packed.edge2 = float4(source.c.xyz - source.a.xyz, source.c.w);
        packed.normal = source.normal;
        packed.emissive = source.emissive;
        ProbePackedTrianglesOut[index] = packed;
    }
}
#endif

#if defined(BUILD_PROBE_VALIDATE_CS)
GPU_BIND_T(0, 0) StructuredBuffer<float4> ProbePositionsFast : register(t0, space0);
GPU_BIND_T(1, 0) StructuredBuffer<PackedProbeNode> ProbeNodesValidate : register(t1, space0);
GPU_BIND_T(2, 0) StructuredBuffer<PackedProbeTriangle> ProbeTrianglesValidate : register(t2, space0);
GPU_BIND_U(0, 1) RWStructuredBuffer<ProbeAccum> ProbeAccumsValidate : register(u0, space1);
#define PROBE_NODES ProbeNodesValidate
#define PROBE_TRIANGLES ProbeTrianglesValidate
#elif defined(BUILD_PROBE_PRIMARY_CS)
GPU_BIND_T(0, 0) StructuredBuffer<float4> ProbePositionsFast : register(t0, space0);
GPU_BIND_T(1, 0) StructuredBuffer<ProbeAccum> ProbeAccumsPrimary : register(t1, space0);
GPU_BIND_T(2, 0) StructuredBuffer<PackedProbeNode> ProbeNodesPrimary : register(t2, space0);
GPU_BIND_T(3, 0) StructuredBuffer<PackedProbeTriangle> ProbeTrianglesPrimary : register(t3, space0);
GPU_BIND_U(0, 1) RWStructuredBuffer<float4> ProbeSampleResultsPrimary : register(u0, space1);
GPU_BIND_U(1, 1) RWStructuredBuffer<ProbeRayState> ProbeStatesPrimary : register(u1, space1);
GPU_BIND_U(2, 1) RWStructuredBuffer<uint> ProbeCountersPrimary : register(u2, space1);
#define PROBE_NODES ProbeNodesPrimary
#define PROBE_TRIANGLES ProbeTrianglesPrimary
#elif defined(BUILD_PROBE_BOUNCE_CS)
GPU_BIND_T(0, 0) StructuredBuffer<PackedProbeNode> ProbeNodesBounce : register(t0, space0);
GPU_BIND_T(1, 0) StructuredBuffer<PackedProbeTriangle> ProbeTrianglesBounce : register(t1, space0);
GPU_BIND_T(2, 0) StructuredBuffer<ProbeRayState> ProbeStatesIn : register(t2, space0);
GPU_BIND_T(3, 0) StructuredBuffer<float> ProbeSunBeams : register(t3, space0);
GPU_BIND_U(0, 1) RWStructuredBuffer<float4> ProbeSampleResultsBounce : register(u0, space1);
GPU_BIND_U(1, 1) RWStructuredBuffer<ProbeRayState> ProbeStatesOut : register(u1, space1);
GPU_BIND_U(2, 1) RWStructuredBuffer<uint> ProbeCountersBounce : register(u2, space1);
#define PROBE_NODES ProbeNodesBounce
#define PROBE_TRIANGLES ProbeTrianglesBounce
#elif defined(BUILD_PROBE_EMISSIVE_CS)
GPU_BIND_T(0, 0) StructuredBuffer<float4> ProbePositionsEmissive : register(t0, space0);
GPU_BIND_T(1, 0) StructuredBuffer<PackedProbeNode> ProbeNodesEmissive : register(t1, space0);
GPU_BIND_T(2, 0) StructuredBuffer<PackedProbeTriangle> ProbeTrianglesEmissive : register(t2, space0);
GPU_BIND_U(0, 1) RWStructuredBuffer<float4> ProbeCoefficientsEmissive : register(u0, space1);
#define PROBE_NODES ProbeNodesEmissive
#define PROBE_TRIANGLES ProbeTrianglesEmissive
#endif

#if defined(BUILD_PROBE_VALIDATE_CS) || defined(BUILD_PROBE_PRIMARY_CS) || defined(BUILD_PROBE_BOUNCE_CS) || defined(BUILD_PROBE_EMISSIVE_CS)
bool probe_trace_any(ProbeTraceRay ray)
{
    uint node_index = 0u;
    while (node_index != PROBE_INVALID) {
        PackedProbeNode node = PROBE_NODES[node_index];
        float near_t;
        if (!probe_box_near(ray, node, ray.tmax, near_t)) {
            node_index = node.meta1;
            continue;
        }

        if ((node.meta0 & PROBE_LEAF_BIT) != 0u) {
            uint first = node.meta0 & PROBE_FIRST_MASK;
            uint count = (node.meta0 >> 27u) & 0x0fu;
            for (uint i = 0u; i < count; ++i) {
                float t;
                if (probe_triangle_hit(ray, PROBE_TRIANGLES[first + i], ray.tmax, t))
                    return true;
            }
            node_index = node.meta1;
        } else {
            node_index = node.meta0;
        }
    }
    return false;
}

bool probe_trace_closest_threaded(ProbeTraceRay ray, out ProbeTraceHit hit)
{
    uint node_index = 0u;
    float closest = ray.tmax;
    bool found = false;
    hit.t = ray.tmax;
    hit.normal = 0.0f;
    hit.albedo = 0.0f;
    hit.triangle_index = PROBE_INVALID;
    hit.emissive = 0.0f;

    while (node_index != PROBE_INVALID) {
        PackedProbeNode node = PROBE_NODES[node_index];
        float near_t;
        if (!probe_box_near(ray, node, closest, near_t)) {
            node_index = node.meta1;
            continue;
        }

        if ((node.meta0 & PROBE_LEAF_BIT) != 0u) {
            uint first = node.meta0 & PROBE_FIRST_MASK;
            uint count = (node.meta0 >> 27u) & 0x0fu;
            for (uint i = 0u; i < count; ++i) {
                uint triangle_index = first + i;
                float t;
                PackedProbeTriangle tri = PROBE_TRIANGLES[triangle_index];
                if (!probe_triangle_hit(ray, tri, closest, t))
                    continue;
                closest = t;
                float3 normal = normalize(tri.normal.xyz);
                if (dot(normal, ray.direction) > 0.0f)
                    normal = -normal;
                hit.t = t;
                hit.normal = normal;
                hit.albedo = saturate(float3(tri.a.w, tri.edge1.w, tri.edge2.w));
                hit.triangle_index = triangle_index;
                hit.emissive = max(tri.emissive.rgb, 0.0f);
                found = true;
            }
            node_index = node.meta1;
        } else {
            node_index = node.meta0;
        }
    }
    return found;
}

bool probe_trace_closest(ProbeTraceRay ray, out ProbeTraceHit hit)
{
    uint stack[32];
    uint stack_size = 0u;
    uint node_index = 0u;
    float closest = ray.tmax;
    bool found = false;
    hit.t = ray.tmax;
    hit.normal = 0.0f;
    hit.albedo = 0.0f;
    hit.triangle_index = PROBE_INVALID;
    hit.emissive = 0.0f;

    while (node_index != PROBE_INVALID) {
        PackedProbeNode node = PROBE_NODES[node_index];
        float near_t;
        if (!probe_box_near(ray, node, closest, near_t)) {
            node_index = stack_size ? stack[--stack_size] : PROBE_INVALID;
            continue;
        }

        if ((node.meta0 & PROBE_LEAF_BIT) != 0u) {
            uint first = node.meta0 & PROBE_FIRST_MASK;
            uint count = (node.meta0 >> 27u) & 0x0fu;
            for (uint i = 0u; i < count; ++i) {
                uint triangle_index = first + i;
                float t;
                PackedProbeTriangle tri = PROBE_TRIANGLES[triangle_index];
                if (!probe_triangle_hit(ray, tri, closest, t))
                    continue;
                closest = t;
                float3 normal = normalize(tri.normal.xyz);
                if (dot(normal, ray.direction) > 0.0f)
                    normal = -normal;
                hit.t = t;
                hit.normal = normal;
                hit.albedo = saturate(float3(tri.a.w, tri.edge1.w, tri.edge2.w));
                hit.triangle_index = triangle_index;
                hit.emissive = max(tri.emissive.rgb, 0.0f);
                found = true;
            }
            node_index = stack_size ? stack[--stack_size] : PROBE_INVALID;
            continue;
        }

        uint left = node.meta0;
        uint right = left + 1u;
        float left_near, right_near;
        bool hit_left = probe_box_near(ray, PROBE_NODES[left], closest, left_near);
        bool hit_right = probe_box_near(ray, PROBE_NODES[right], closest, right_near);
        if (hit_left && hit_right) {
            uint near_child = left_near <= right_near ? left : right;
            uint far_child = left_near <= right_near ? right : left;
            if (stack_size >= 32u)
                return probe_trace_closest_threaded(ray, hit);
            stack[stack_size++] = far_child;
            node_index = near_child;
        } else if (hit_left) {
            node_index = left;
        } else if (hit_right) {
            node_index = right;
        } else {
            node_index = stack_size ? stack[--stack_size] : PROBE_INVALID;
        }
    }

    return found;
}
#endif


#if defined(BUILD_PROBE_EMISSIVE_CS)
[numthreads(64, 1, 1)]
void probe_emissive_cs(uint3 id : SV_DispatchThreadID)
{
    uint probe_index = id.x;
    if (probe_index >= probe_count || emissive_samples == 0u || bake_params.w <= 0.0f)
        return;
    if (ProbeCoefficientsEmissive[probe_index * 9u].w <= 0.0f)
        return;

    float3 position = ProbePositionsEmissive[probe_index].xyz;
    float3 sums[9];
    [unroll] for (uint coefficient = 0u; coefficient < 9u; ++coefficient)
        sums[coefficient] = 0.0f;

    uint seed = probe_hash(probe_index * 0x9e3779b9u + 0x6a09e667u);
    [loop] for (uint sample = 0u; sample < emissive_samples; ++sample) {
        float target = probe_random(seed) * bake_params.w;
        uint lo = 0u, hi = triangle_count;
        while (lo < hi) {
            uint mid = lo + (hi - lo) / 2u;
            if (ProbeTrianglesEmissive[mid].emissive.w > target)
                hi = mid;
            else
                lo = mid + 1u;
        }
        if (lo >= triangle_count)
            continue;

        PackedProbeTriangle tri = ProbeTrianglesEmissive[lo];
        float previous = lo == 0u ? 0.0f : ProbeTrianglesEmissive[lo - 1u].emissive.w;
        float triangle_weight = tri.emissive.w - previous;
        float area = 0.5f * length(cross(tri.edge1.xyz, tri.edge2.xyz));
        if (triangle_weight <= 0.0f || area <= 1.0e-10f)
            continue;

        float root = sqrt(probe_random(seed));
        float bary = probe_random(seed);
        float3 light_position = tri.a.xyz + tri.edge1.xyz * (root * (1.0f - bary)) +
            tri.edge2.xyz * (root * bary);
        float3 delta = light_position - position;
        float distance2 = dot(delta, delta);
        if (distance2 <= bake_params.x * bake_params.x)
            continue;
        float distance = sqrt(distance2);
        float3 direction = delta / distance;
        float emitter_cosine = saturate(dot(normalize(tri.normal.xyz), -direction));
        if (emitter_cosine <= 0.0f)
            continue;

        ProbeTraceRay shadow = probe_make_ray(position + direction * bake_params.x,
            direction, bake_params.x, max(bake_params.x, distance - 2.0f * bake_params.x));
        if (probe_trace_any(shadow))
            continue;

        float pdf_area = (triangle_weight / bake_params.w) / area;
        float pdf_solid_angle = pdf_area * distance2 / emitter_cosine;
        if (pdf_solid_angle <= 1.0e-12f)
            continue;

        float3 contribution = max(tri.emissive.rgb, 0.0f) / pdf_solid_angle;
        float basis[9];
        probe_sh_basis(direction, basis);
        [unroll] for (uint coefficient = 0u; coefficient < 9u; ++coefficient)
            sums[coefficient] += contribution * basis[coefficient];
    }

    float inv_samples = rcp((float)emissive_samples);
    [unroll] for (uint coefficient = 0u; coefficient < 9u; ++coefficient) {
        uint index = probe_index * 9u + coefficient;
        float4 value = ProbeCoefficientsEmissive[index];
        value.rgb += sums[coefficient] * inv_samples;
        ProbeCoefficientsEmissive[index] = value;
    }
}
#endif

#if defined(PROBE_NODES)
#undef PROBE_NODES
#undef PROBE_TRIANGLES
#endif

#if defined(BUILD_PROBE_RESET_CS)
GPU_BIND_U(0, 1) RWStructuredBuffer<uint> ProbeCountersReset : register(u0, space1);
GPU_BIND_U(1, 1) RWStructuredBuffer<ProbeAccum> ProbeAccumsReset : register(u1, space1);
GPU_BIND_U(2, 1) RWStructuredBuffer<float4> ProbeCoefficientsReset : register(u2, space1);

[numthreads(64, 1, 1)]
void probe_reset_cs(uint3 id : SV_DispatchThreadID)
{
    if (id.x < 4u)
        ProbeCountersReset[id.x] = 0u;
    if (sample_offset == 0u && id.x < probe_count) {
        ProbeAccum zero_accum;
        zero_accum.radiance_count = 0.0f;
        zero_accum.stats = 0.0f;
        ProbeAccumsReset[id.x] = zero_accum;
        [unroll] for (uint coefficient = 0u; coefficient < 9u; ++coefficient)
            ProbeCoefficientsReset[id.x * 9u + coefficient] = 0.0f;
    }
}
#endif

#if defined(BUILD_PROBE_VALIDATE_CS)
[numthreads(64, 1, 1)]
void probe_validate_cs(uint3 id : SV_DispatchThreadID)
{
    uint probe_index = id.x;
    if (probe_index >= probe_count)
        return;

    float4 input = ProbePositionsFast[probe_index];
    bool valid = input.w > 0.0f;
    const float3 axes[6] = {
        float3(1,0,0), float3(-1,0,0), float3(0,1,0),
        float3(0,-1,0), float3(0,0,1), float3(0,0,-1)
    };
    for (uint axis = 0u; axis < 6u && valid; ++axis) {
        ProbeTraceRay ray = probe_make_ray(input.xyz, axes[axis], bake_params.x, 0.15f);
        if (probe_trace_any(ray))
            valid = false;
    }

    ProbeAccum accum = ProbeAccumsValidate[probe_index];
    accum.stats.w = valid ? 1.0f : 0.0f;
    ProbeAccumsValidate[probe_index] = accum;
}
#endif

#if defined(BUILD_PROBE_PRIMARY_CS)
[numthreads(64, 1, 1)]
void probe_primary_cs(uint3 id : SV_DispatchThreadID)
{
    uint total = probe_count * samples_per_block;
    uint flat = id.x;
    if (flat >= total)
        return;

    uint probe_index = flat / samples_per_block;
    uint lane = flat - probe_index * samples_per_block;
    uint result_index = probe_index * PROBE_BLOCK_SAMPLES + lane;
    ProbeSampleResultsPrimary[result_index] = 0.0f;

    ProbeAccum accum = ProbeAccumsPrimary[probe_index];
    if (accum.stats.w <= 0.0f)
        return;

    uint sample_index = sample_offset + lane;
    float3 direction = probe_sphere_direction(probe_index, sample_index);
    float3 origin = ProbePositionsFast[probe_index].xyz + direction * bake_params.x;
    ProbeTraceRay ray = probe_make_ray(origin, direction, bake_params.x, 1.0e20f);
    ProbeTraceHit hit;
    if (!probe_trace_closest(ray, hit)) {
        ProbeSampleResultsPrimary[result_index] = float4(probe_sky(direction), 1.0f);
        return;
    }

    uint seed = probe_hash(probe_index * 9781u + sample_index * 6271u + 0x51f2e91du);
    ProbeRayState state;
    state.position = float4(ray.origin + ray.direction * hit.t, 0.0f);
    state.normal = float4(hit.normal, 0.0f);
    state.throughput = float4(hit.albedo / PROBE_PI, 0.0f);
    state.radiance = emissive_samples == 0u ? float4(hit.emissive, 0.0f) : 0.0f;
    state.meta = uint4(result_index, seed, 0u, 0u);

    uint slot;
    InterlockedAdd(ProbeCountersPrimary[0], 1u, slot);
    ProbeStatesPrimary[slot] = state;
}
#endif

#if defined(BUILD_PROBE_BOUNCE_CS)
bool probe_sun_hint(float3 position, out bool visible)
{
    visible = false;
    if (beam_depth < 16u || beam_depth > 128u)
        return false;

    float3 sun = normalize(sun_direction_intensity.xyz);
    float3 u = normalize(cross(float3(0, 1, 0), sun));
    if (dot(u, u) < 0.5f)
        return false;
    float3 v = cross(sun, u);
    float3 q = float3(dot(position, u), dot(position, v), dot(position, sun));
    float3 coord = (q - probe_beam_origin.xyz) / probe_beam_step.xyz;
    int x = (int)floor(coord.x);
    int y = (int)floor(coord.y);
    int z = (int)floor(coord.z);
    if (x < 1 || x >= 63 || y < 1 || y >= 63 || z < 0 || z >= (int)beam_depth)
        return false;

    float first_visibility = ProbeSunBeams[(uint)x + 64u * ((uint)y + 64u * (uint)z)];
    bool lit = first_visibility > 0.5f;
    float nearest_blocker_delta = 1.0e20f;
    uint shadow_base = 64u * 64u * beam_depth;
    [unroll] for (int oy = -1; oy <= 1; ++oy) {
        [unroll] for (int ox = -1; ox <= 1; ++ox) {
            uint sx = (uint)(x + ox);
            uint sy = (uint)(y + oy);
            float cell_visibility = ProbeSunBeams[sx + 64u * (sy + 64u * (uint)z)];
            if ((cell_visibility > 0.5f) != lit)
                return false;
            float blocker = ProbeSunBeams[shadow_base + sx + 64u * sy];
            if (blocker > -1.0e20f)
                nearest_blocker_delta = min(nearest_blocker_delta, abs(q.z - blocker));
        }
    }

    float guard = max(0.05f, probe_beam_step.z * 1.5f);
    if (nearest_blocker_delta < guard)
        return false;

    visible = lit;
    return true;
}

float3 probe_direct_sun(float3 position, float3 normal, inout uint seed)
{
    float3 center = normalize(sun_direction_intensity.xyz);
    float3 direction = probe_sample_sun(center, sun_color_radius.w, seed);
    float n_dot_l = saturate(dot(normal, direction));
    if (n_dot_l <= 0.0f)
        return 0.0f;

    bool visible;
    if (probe_sun_hint(position, visible)) {
        if (!visible)
            return 0.0f;
    } else {
        ProbeTraceRay ray = probe_make_ray(position + normal * bake_params.x,
                                           direction, bake_params.x, 1.0e20f);
        if (probe_trace_any(ray))
            return 0.0f;
    }

    return sun_color_radius.rgb * (sun_direction_intensity.w * n_dot_l);
}
float3 probe_direct_emissive(float3 position, float3 normal, inout uint seed)
{
    const float total_weight = bake_params.w;
    if (total_weight <= 0.0f || triangle_count == 0u)
        return 0.0f;

    const float target = probe_random(seed) * total_weight;
    uint lo = 0u;
    uint hi = triangle_count;
    while (lo < hi)
    {
        uint mid = lo + (hi - lo) / 2u;
        if (ProbeTrianglesBounce[mid].emissive.w > target)
            hi = mid;
        else
            lo = mid + 1u;
    }
    if (lo >= triangle_count)
        return 0.0f;

    PackedProbeTriangle tri = ProbeTrianglesBounce[lo];
    const float previous = lo == 0u ? 0.0f : ProbeTrianglesBounce[lo - 1u].emissive.w;
    const float triangle_weight = tri.emissive.w - previous;
    if (triangle_weight <= 0.0f)
        return 0.0f;

    const float area = 0.5f * length(cross(tri.edge1.xyz, tri.edge2.xyz));
    if (area <= 1.0e-10f)
        return 0.0f;

    const float root = sqrt(probe_random(seed));
    const float bary = probe_random(seed);
    const float3 light_position = tri.a.xyz + tri.edge1.xyz * (root * (1.0f - bary)) +
        tri.edge2.xyz * (root * bary);
    const float3 delta = light_position - position;
    const float distance2 = dot(delta, delta);
    if (distance2 <= bake_params.x * bake_params.x)
        return 0.0f;

    const float distance = sqrt(distance2);
    const float3 direction = delta / distance;
    const float receiver_cosine = saturate(dot(normal, direction));
    const float emitter_cosine = saturate(dot(normalize(tri.normal.xyz), -direction));
    if (receiver_cosine <= 0.0f || emitter_cosine <= 0.0f)
        return 0.0f;

    ProbeTraceRay shadow = probe_make_ray(position + normal * bake_params.x,
        direction, bake_params.x, max(bake_params.x, distance - 2.0f * bake_params.x));
    if (probe_trace_any(shadow))
        return 0.0f;

    const float pdf_area = (triangle_weight / total_weight) / area;
    if (pdf_area <= 1.0e-12f)
        return 0.0f;

    return max(tri.emissive.rgb, 0.0f) *
        (receiver_cosine * emitter_cosine / max(distance2 * pdf_area, 1.0e-8f));
}

[numthreads(64, 1, 1)]
void probe_bounce_cs(uint3 id : SV_DispatchThreadID)
{
    uint active_count = ProbeCountersBounce[bounce_index];
    uint index = id.x;
    if (index >= active_count)
        return;

    ProbeRayState state = ProbeStatesIn[index];
    uint result_index = state.meta.x;
    uint seed = state.meta.y;
    float3 position = state.position.xyz;
    float3 normal = state.normal.xyz;
    float3 throughput = state.throughput.rgb;
    float3 radiance = state.radiance.rgb;

    radiance += throughput * probe_direct_sun(position, normal, seed);
    radiance += throughput * probe_direct_emissive(position, normal, seed);

    float3 direction = probe_cosine_hemisphere(normal, seed);
    ProbeTraceRay ray = probe_make_ray(position + normal * bake_params.x,
                                       direction, bake_params.x, 1.0e20f);
    ProbeTraceHit hit;
    if (!probe_trace_closest(ray, hit)) {
        radiance += throughput * probe_sky(direction);
        ProbeSampleResultsBounce[result_index] = float4(radiance, 1.0f);
        return;
    }

    if (bounce_index + 1u >= max_bounces) {
        ProbeSampleResultsBounce[result_index] = float4(radiance, 1.0f);
        return;
    }

    throughput *= hit.albedo;
    state.position = float4(ray.origin + ray.direction * hit.t, 0.0f);
    state.normal = float4(hit.normal, 0.0f);
    state.meta.y = seed;
    state.throughput = float4(throughput, 0.0f);
    state.radiance = float4(radiance, 0.0f);

    uint slot;
    InterlockedAdd(ProbeCountersBounce[bounce_index + 1u], 1u, slot);
    ProbeStatesOut[slot] = state;
}
#endif

#if defined(BUILD_PROBE_REDUCE_CS)
GPU_BIND_T(0, 0) StructuredBuffer<float4> ProbeSampleResultsReduce : register(t0, space0);
GPU_BIND_U(0, 1) RWStructuredBuffer<ProbeAccum> ProbeAccumsReduce : register(u0, space1);
GPU_BIND_U(1, 1) RWStructuredBuffer<float4> ProbeCoefficientsReduce : register(u1, space1);
GPU_BIND_U(2, 1) RWStructuredBuffer<uint> ProbeCountersReduce : register(u2, space1);

groupshared float3 ProbeReduceRadiance[PROBE_BLOCK_SAMPLES];
groupshared float ProbeReduceLuma[PROBE_BLOCK_SAMPLES];
groupshared float ProbeReduceLuma2[PROBE_BLOCK_SAMPLES];

[numthreads(PROBE_BLOCK_SAMPLES, 1, 1)]
void probe_reduce_cs(uint3 group_id : SV_GroupID, uint3 local_id : SV_GroupThreadID)
{
    uint probe_index = group_id.x;
    uint lane = local_id.x;
    if (probe_index >= probe_count)
        return;

    float3 radiance = 0.0f;
    float luma = 0.0f;
    float basis[9];
    [unroll] for (uint coefficient = 0u; coefficient < 9u; ++coefficient)
        basis[coefficient] = 0.0f;
    if (lane < samples_per_block) {
        radiance = ProbeSampleResultsReduce[probe_index * PROBE_BLOCK_SAMPLES + lane].rgb;
        luma = dot(radiance, float3(0.2126f, 0.7152f, 0.0722f));
        probe_sh_basis(probe_sphere_direction(probe_index, sample_offset + lane), basis);
    }

    ProbeReduceRadiance[lane] = radiance;
    ProbeReduceLuma[lane] = luma;
    ProbeReduceLuma2[lane] = luma * luma;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = PROBE_BLOCK_SAMPLES / 2u; stride > 0u; stride >>= 1u) {
        if (lane < stride) {
            ProbeReduceRadiance[lane] += ProbeReduceRadiance[lane + stride];
            ProbeReduceLuma[lane] += ProbeReduceLuma[lane + stride];
            ProbeReduceLuma2[lane] += ProbeReduceLuma2[lane + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    ProbeAccum accum = ProbeAccumsReduce[probe_index];
    bool valid = accum.stats.w > 0.0f;
    if (lane == 0u && valid) {
        accum.radiance_count.rgb += ProbeReduceRadiance[0];
        accum.radiance_count.w += (float)samples_per_block;
        accum.stats.x += ProbeReduceLuma[0];
        accum.stats.y += ProbeReduceLuma2[0];

        float n = max(accum.radiance_count.w, 1.0f);
        float mean = accum.stats.x / n;
        float variance = max(accum.stats.y / n - mean * mean, 0.0f);
        float stderr = sqrt(variance / n);
        bool checkpoint = (uint)n >= 256u && (uint)n < total_samples;
        bool stable = checkpoint && stderr < 0.01f + 0.025f * abs(mean);
        accum.stats.z = stable ? accum.stats.z + 1.0f : 0.0f;
        bool active = (uint)n < total_samples && accum.stats.z < 2.0f;
        ProbeAccumsReduce[probe_index] = accum;
        if (active) {
            uint ignored;
            InterlockedAdd(ProbeCountersReduce[3], 1u, ignored);
        }
    }
    GroupMemoryBarrierWithGroupSync();

    float n = max(ProbeAccumsReduce[probe_index].radiance_count.w, 1.0f);
    float previous_n = max(n - (float)samples_per_block, 0.0f);
    [unroll] for (uint coefficient = 0u; coefficient < 9u; ++coefficient) {
        ProbeReduceRadiance[lane] = lane < samples_per_block ? radiance * basis[coefficient] : 0.0f;
        GroupMemoryBarrierWithGroupSync();
        for (uint stride = PROBE_BLOCK_SAMPLES / 2u; stride > 0u; stride >>= 1u) {
            if (lane < stride)
                ProbeReduceRadiance[lane] += ProbeReduceRadiance[lane + stride];
            GroupMemoryBarrierWithGroupSync();
        }
        if (lane == 0u && valid) {
            uint index = probe_index * 9u + coefficient;
            float4 old_value = ProbeCoefficientsReduce[index];
            float3 old_sum = old_value.rgb * (previous_n / (4.0f * PROBE_PI));
            float3 coefficient_value = (old_sum + ProbeReduceRadiance[0]) * (4.0f * PROBE_PI / n);
            float metadata = old_value.w;
            if (coefficient == 0u)
                metadata = 1.0f;
            if (coefficient == 2u)
                metadata = n;
            ProbeCoefficientsReduce[index] = float4(coefficient_value, metadata);
        }
        GroupMemoryBarrierWithGroupSync();
    }
}
#endif

#endif