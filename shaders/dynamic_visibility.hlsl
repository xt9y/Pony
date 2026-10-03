#ifndef GPU_BIND_T
#define GPU_BIND_T(n, s) [[vk::binding(n + 16, s)]]
#define GPU_BIND_B(n, s) [[vk::binding(n + 32, s)]]
#define GPU_BIND_U(n, s) [[vk::binding(n + 48, s)]]
#endif

#if defined(BUILD_DYNAMIC_VISIBILITY_CS)

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
GPU_BIND_U(0, 1) RWStructuredBuffer<float> Visibility : register(u0, space1);

GPU_BIND_B(0, 2) cbuffer DynamicVisibilityData : register(b0, space2) {
    uint4 bvh_meta;
    uint4 field_meta;
    float4 field_origin_spacing;
    float4 emitter_center_radius;
    float4 params;
    float4x4 model;
};

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
    float3 relative = ray.origin - tri.a.xyz;
    float u = dot(relative, p) * inverse;
    if (u < 0.0f || u > 1.0f) return false;

    float3 q = cross(relative, edge1);
    float v = dot(ray.direction, q) * inverse;
    if (v < 0.0f || u + v > 1.0f) return false;

    float t = dot(edge2, q) * inverse;
    return t > ray.tmin && t < ray.tmax;
}

bool trace_any(TraceRay ray) {
    uint node_index = 0u;

    while (node_index != 0xffffffffu && node_index < bvh_meta.x) {
        BvhNode node = Nodes[node_index];

        if (!trace_box(ray, node)) {
            node_index = node.meta.y;
            continue;
        }

        if (node.meta.w != 0u) {
            [loop] for (uint i = 0u; i < node.meta.w; ++i) {
                uint triangle_index = node.meta.z + i;
                if (triangle_index >= bvh_meta.y) continue;

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
void dynamic_visibility_cs(uint3 id : SV_DispatchThreadID) {
    uint probe_count = field_meta.y * field_meta.z * field_meta.w;
    uint probe_index = id.x;
    if (probe_index >= probe_count) return;

    uint x = probe_index % field_meta.y;
    uint yz = probe_index / field_meta.y;
    uint y = yz % field_meta.z;
    uint z = yz / field_meta.z;

    float3 local_position =
        field_origin_spacing.xyz + float3(x, y, z) * field_origin_spacing.w;
    float3 world_position = mul(model, float4(local_position, 1.0f)).xyz;
    float3 emitter_center = mul(model, float4(emitter_center_radius.xyz, 1.0f)).xyz;

    float3 delta = emitter_center - world_position;
    float distance2 = dot(delta, delta);
    float epsilon = max(params.x, 1.0e-5f);
    float emitter_radius = max(emitter_center_radius.w, 0.0f);
    float visibility = 1.0f;

    if (distance2 > epsilon * epsilon && bvh_meta.x != 0u) {
        float distance = sqrt(distance2);
        float3 direction = delta / distance;

        /*
         * Visibility is to the emissive region, not to its center. The center
         * of a wall/ceiling-mounted area emitter may lie on or behind static
         * mounting geometry; tracing all the way there incorrectly marks the
         * receiver as occluded by the mounting surface itself.
         *
         * Stop at the near surface of the emissive bounding sphere. The
         * object-local radiance field already accounts for self-occlusion by
         * the dynamic object's own geometry.
         */
        float target_distance = max(epsilon, distance - emitter_radius);

        TraceRay ray;
        ray.origin = world_position + direction * epsilon;
        ray.tmin = epsilon;
        ray.direction = direction;
        ray.tmax = max(epsilon, target_distance - epsilon);

        visibility = target_distance <= epsilon * 2.0f || !trace_any(ray) ? 1.0f : 0.0f;
    }

    Visibility[field_meta.x + probe_index] = visibility;
}

#endif
