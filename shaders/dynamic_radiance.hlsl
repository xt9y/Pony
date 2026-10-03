#ifndef GPU_BIND_S
#define GPU_BIND_S(n, s) [[vk::binding(n, s)]]
#define GPU_BIND_T(n, s) [[vk::binding(n + 16, s)]]
#define GPU_BIND_B(n, s) [[vk::binding(n + 32, s)]]
#define GPU_BIND_U(n, s) [[vk::binding(n + 48, s)]]
#endif

#if defined(BUILD_DYNAMIC_RADIANCE_CS)

static const uint INVALID_NODE = 0xffffffffu;
static const float PI = 3.14159265358979323846f;

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

struct TraceRay {
    float3 origin;
    float tmin;
    float3 direction;
    float tmax;
};

GPU_BIND_T(0, 0) StructuredBuffer<BvhNode> Nodes : register(t0, space0);
GPU_BIND_T(1, 0) StructuredBuffer<BvhTriangle> Triangles : register(t1, space0);
GPU_BIND_U(0, 1) RWStructuredBuffer<float4> Coefficients : register(u0, space1);

GPU_BIND_B(0, 2) cbuffer DynamicRadianceData : register(b0, space2) {
    uint4 bvh_meta;
    uint4 field_meta;
    float4 field_origin_spacing;
    float4 field_params;
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

void sh_basis(float3 direction, out float basis[9]) {
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

bool trace_box(TraceRay ray, BvhNode node) {
    float lo = ray.tmin;
    float hi = ray.tmax;

    [unroll] for (uint axis = 0u; axis < 3u; ++axis) {
        float d = ray.direction[axis];

        if (abs(d) < 1.0e-7f) {
            if (ray.origin[axis] < node.bmin[axis] || ray.origin[axis] > node.bmax[axis]) return false;
            continue;
        }

        float a = (node.bmin[axis] - ray.origin[axis]) / d;
        float b = (node.bmax[axis] - ray.origin[axis]) / d;
        lo = max(lo, min(a, b));
        hi = min(hi, max(a, b));
        if (lo > hi) return false;
    }

    return hi >= ray.tmin;
}

bool trace_triangle(TraceRay ray, BvhTriangle tri) {
    float3 edge1 = tri.b.xyz - tri.a.xyz;
    float3 edge2 = tri.c.xyz - tri.a.xyz;
    float3 p = cross(ray.direction, edge2);
    float determinant = dot(edge1, p);
    if (abs(determinant) < 1.0e-7f) return false;

    float inverse = rcp(determinant);
    float3 s = ray.origin - tri.a.xyz;
    float u = dot(s, p) * inverse;
    if (u < 0.0f || u > 1.0f) return false;

    float3 q = cross(s, edge1);
    float v = dot(ray.direction, q) * inverse;
    if (v < 0.0f || u + v > 1.0f) return false;

    float t = dot(edge2, q) * inverse;
    return t > ray.tmin && t < ray.tmax;
}

bool trace_any(TraceRay ray) {
    uint node_index = bvh_meta.x;
    uint node_end = bvh_meta.x + bvh_meta.y;
    uint triangle_begin = bvh_meta.z;
    uint triangle_end = bvh_meta.z + bvh_meta.w;

    while (node_index != INVALID_NODE && node_index >= bvh_meta.x && node_index < node_end) {
        BvhNode node = Nodes[node_index];

        if (!trace_box(ray, node)) {
            node_index = node.meta.y;
            continue;
        }

        if (node.meta.w != 0u) {
            for (uint i = 0u; i < node.meta.w; ++i) {
                uint triangle_index = node.meta.z + i;
                if (triangle_index < triangle_begin || triangle_index >= triangle_end) continue;

                BvhTriangle tri = Triangles[triangle_index];
                if (tri.normal.w >= 0.999f) continue;
                if (trace_triangle(ray, tri)) return true;
            }

            node_index = node.meta.y;
        } else {
            node_index = node.meta.x;
        }
    }

    return false;
}

[numthreads(64, 1, 1)]
void dynamic_radiance_cs(uint3 id : SV_DispatchThreadID) {
    uint probe_count = field_meta.y * field_meta.z * field_meta.w;
    uint probe_index = id.x;
    if (probe_index >= probe_count) return;

    uint x = probe_index % field_meta.y;
    uint yz = probe_index / field_meta.y;
    uint y = yz % field_meta.z;
    uint z = yz / field_meta.z;
    float3 position = field_origin_spacing.xyz + float3(x, y, z) * field_origin_spacing.w;

    float3 sums[9];
    [unroll] for (uint coefficient = 0u; coefficient < 9u; ++coefficient) sums[coefficient] = 0.0f;

    uint sample_count = max((uint)field_params.w, 1u);
    float emissive_weight = max(field_params.y, 0.0f);
    float epsilon = max(field_params.x, 1.0e-5f);
    uint seed = hash_u32((field_meta.x + probe_index) * 0x9e3779b9u + 0x6a09e667u);

    if (emissive_weight > 0.0f) {
        [loop] for (uint sample = 0u; sample < sample_count; ++sample) {
            float target = random01(seed) * emissive_weight;
            uint lo = bvh_meta.z;
            uint hi = bvh_meta.z + bvh_meta.w;

            while (lo < hi) {
                uint mid = lo + (hi - lo) / 2u;
                if (Triangles[mid].emissive.w > target)
                    hi = mid;
                else
                    lo = mid + 1u;
            }

            if (lo >= bvh_meta.z + bvh_meta.w) continue;

            BvhTriangle tri = Triangles[lo];
            float previous = lo == bvh_meta.z ? 0.0f : Triangles[lo - 1u].emissive.w;
            float triangle_weight = tri.emissive.w - previous;
            float3 edge1 = tri.b.xyz - tri.a.xyz;
            float3 edge2 = tri.c.xyz - tri.a.xyz;
            float area = 0.5f * length(cross(edge1, edge2));
            if (triangle_weight <= 0.0f || area <= 1.0e-10f) continue;

            float root = sqrt(random01(seed));
            float bary = random01(seed);
            float3 light_position = tri.a.xyz + edge1 * (root * (1.0f - bary)) + edge2 * (root * bary);
            float3 delta = light_position - position;
            float distance2 = dot(delta, delta);
            if (distance2 <= epsilon * epsilon) continue;

            float distance = sqrt(distance2);
            float3 direction = delta / distance;
            float3 light_normal = normalize(cross(edge1, edge2));
            float emitter_cosine = abs(dot(light_normal, -direction));
            if (emitter_cosine <= 0.0f) continue;

            TraceRay shadow;
            shadow.origin = position + direction * epsilon;
            shadow.tmin = epsilon;
            shadow.direction = direction;
            shadow.tmax = max(epsilon, distance - 2.0f * epsilon);
            if (trace_any(shadow)) continue;

            float pdf_area = (triangle_weight / emissive_weight) / area;
            float pdf_solid_angle = pdf_area * distance2 / emitter_cosine;
            if (pdf_solid_angle <= 1.0e-12f) continue;

            float3 contribution = max(tri.emissive.rgb, 0.0f) * max(field_params.z, 0.0f) / pdf_solid_angle;
            float basis[9];
            sh_basis(direction, basis);

            [unroll] for (uint coefficient = 0u; coefficient < 9u; ++coefficient)
                sums[coefficient] += contribution * basis[coefficient];
        }
    }

    float inverse_samples = rcp((float)sample_count);
    uint coefficient_base = (field_meta.x + probe_index) * 9u;

    [unroll] for (uint coefficient = 0u; coefficient < 9u; ++coefficient)
        Coefficients[coefficient_base + coefficient] = float4(sums[coefficient] * inverse_samples, 0.0f);
}

#endif
