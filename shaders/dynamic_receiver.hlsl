#ifndef GPU_BIND_S
#define GPU_BIND_S(n, s) [[vk::binding(n, s)]]
#define GPU_BIND_T(n, s) [[vk::binding(n + 16, s)]]
#define GPU_BIND_B(n, s) [[vk::binding(n + 32, s)]]
#define GPU_BIND_U(n, s) [[vk::binding(n + 48, s)]]
#define GPU_STORAGE_RGBA16F [[vk::image_format("rgba16f")]]
#endif

#if defined(BUILD_DYNAMIC_RECEIVER_CS)

static const uint INVALID_NODE = 0xffffffffu;
static const uint PHASE_DIRECT = 0u;
static const uint PHASE_FILTER = 1u;
static const uint DYNAMIC_INSTANCE_LIMIT = 8u;
static const uint EMISSIVE_SAMPLES = 8u;

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

struct TraceRay {
    float3 origin;
    float tmin;
    float3 direction;
    float tmax;
};

GPU_BIND_T(0, 0) StructuredBuffer<SurfaceSample> Samples : register(t0, space0);
GPU_BIND_T(1, 0) StructuredBuffer<BvhNode> StaticNodes : register(t1, space0);
GPU_BIND_T(2, 0) StructuredBuffer<BvhTriangle> StaticTriangles : register(t2, space0);
GPU_BIND_T(3, 0) StructuredBuffer<BvhNode> DynamicNodes : register(t3, space0);
GPU_BIND_T(4, 0) StructuredBuffer<BvhTriangle> DynamicTriangles : register(t4, space0);
GPU_BIND_T(5, 0) Texture2D<float4> Source : register(t5, space0);
GPU_BIND_S(0, 0) SamplerState SourceSampler : register(s0, space0);
GPU_BIND_U(0, 1) GPU_STORAGE_RGBA16F RWTexture2D<float4> Output : register(u0, space1);

GPU_BIND_B(0, 2) cbuffer DynamicReceiverData : register(b0, space2) {
    uint4 dispatch_data;
    uint4 dynamic_instance_data;
    float4 receiver_params;
    float4 temporal_params;
    float4x4 view_projection;

    float4x4 dynamic_instance_model[DYNAMIC_INSTANCE_LIMIT];
    float4x4 dynamic_instance_inverse[DYNAMIC_INSTANCE_LIMIT];
    uint4 dynamic_instance_meta[DYNAMIC_INSTANCE_LIMIT];
    float4 dynamic_instance_emissive[DYNAMIC_INSTANCE_LIMIT];
    float4 dynamic_instance_center_radius[DYNAMIC_INSTANCE_LIMIT];
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

uint generation_age(float stored_alpha, uint current_generation) {
    uint stored = (uint)max(stored_alpha + 0.5f, 0.0f);
    if (stored == 0u || stored > 1024u || current_generation == 0u || current_generation > 1024u) return 0xffffffffu;
    return current_generation >= stored ? current_generation - stored : current_generation + 1024u - stored;
}

bool triangle_is_transmissive(BvhTriangle tri) {
    return tri.normal.w >= 0.999f;
}

bool trace_box(TraceRay ray, BvhNode node) {
    float lo = ray.tmin;
    float hi = ray.tmax;

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

bool trace_triangle(TraceRay ray, BvhTriangle tri) {
    float3 edge1 = tri.b.xyz - tri.a.xyz;
    float3 edge2 = tri.c.xyz - tri.a.xyz;
    float3 p = cross(ray.direction, edge2);
    float determinant = dot(edge1, p);
    if (abs(determinant) < 1.0e-7f) return false;

    float inverse = 1.0f / determinant;
    float3 s = ray.origin - tri.a.xyz;
    float u = dot(s, p) * inverse;
    if (u < 0.0f || u > 1.0f) return false;

    float3 q = cross(s, edge1);
    float v = dot(ray.direction, q) * inverse;
    if (v < 0.0f || u + v > 1.0f) return false;

    float t = dot(edge2, q) * inverse;
    return t > ray.tmin && t < ray.tmax;
}

bool static_any(TraceRay ray) {
    uint node_index = 0u;

    while (node_index != INVALID_NODE) {
        BvhNode node = StaticNodes[node_index];

        if (!trace_box(ray, node)) {
            node_index = node.meta.y;
            continue;
        }

        if (node.meta.w != 0u) {
            for (uint i = 0u; i < node.meta.w; ++i) {
                BvhTriangle tri = StaticTriangles[node.meta.z + i];
                if (triangle_is_transmissive(tri)) continue;
                if (trace_triangle(ray, tri)) return true;
            }

            node_index = node.meta.y;
        } else {
            node_index = node.meta.x;
        }
    }

    return false;
}

bool dynamic_instance_any(TraceRay world_ray, uint instance_index) {
    if (instance_index >= min(dynamic_instance_data.x, DYNAMIC_INSTANCE_LIMIT)) return false;

    uint4 meta = dynamic_instance_meta[instance_index];
    if (meta.y == 0u || meta.w == 0u) return false;

    TraceRay ray;
    ray.origin = mul(dynamic_instance_inverse[instance_index], float4(world_ray.origin, 1.0f)).xyz;
    ray.tmin = world_ray.tmin;
    ray.direction = mul((float3x3)dynamic_instance_inverse[instance_index], world_ray.direction);
    ray.tmax = world_ray.tmax;

    uint node_index = meta.x;
    uint node_end = meta.x + meta.y;
    uint triangle_begin = meta.z;
    uint triangle_end = meta.z + meta.w;

    while (node_index != INVALID_NODE && node_index >= meta.x && node_index < node_end) {
        BvhNode node = DynamicNodes[node_index];

        if (!trace_box(ray, node)) {
            node_index = node.meta.y;
            continue;
        }

        if (node.meta.w != 0u) {
            for (uint i = 0u; i < node.meta.w; ++i) {
                uint candidate = node.meta.z + i;
                if (candidate < triangle_begin || candidate >= triangle_end) continue;

                BvhTriangle tri = DynamicTriangles[candidate];
                if (triangle_is_transmissive(tri)) continue;
                if (trace_triangle(ray, tri)) return true;
            }

            node_index = node.meta.y;
        } else {
            node_index = node.meta.x;
        }
    }

    return false;
}

bool dynamic_any(TraceRay ray) {
    uint count = min(dynamic_instance_data.x, DYNAMIC_INSTANCE_LIMIT);

    [loop] for (uint i = 0u; i < count; ++i)
        if (dynamic_instance_any(ray, i)) return true;

    return false;
}

bool receiver_relevant(float3 position) {
    float4 clip = mul(view_projection, float4(position, 1.0f));
    if (clip.w <= 1.0e-5f) return false;

    float2 ndc = clip.xy / clip.w;
    const float screen_margin = 1.25f;
    if (abs(ndc.x) > screen_margin || abs(ndc.y) > screen_margin) return false;

    uint count = min(dynamic_instance_data.x, DYNAMIC_INSTANCE_LIMIT);
    const float irradiance_floor = max(receiver_params.y, 1.0e-5f);

    [loop] for (uint i = 0u; i < count; ++i) {
        float previous = i == 0u ? 0.0f : dynamic_instance_emissive[i - 1u].y;
        float world_weight = dynamic_instance_emissive[i].y - previous;
        if (world_weight <= 0.0f) continue;

        float radius = max(dynamic_instance_center_radius[i].w, 1.0e-3f);
        float distance_to_surface = max(length(position - dynamic_instance_center_radius[i].xyz) - radius, 0.0f);

        /*
         * Conservative upper bound for an area emitter: irradiance falls
         * with area*luminance / r^2. Contributions below the linear-light
         * floor are visually lost after the shared material/ACES path.
         */
        float influence = sqrt(world_weight / max(3.14159265358979323846f * irradiance_floor, 1.0e-8f));
        influence = max(influence, radius * 1.5f);

        if (distance_to_surface <= influence) return true;
    }

    return false;
}

float3 direct_emissive_target(float3 position, float3 normal, inout uint seed, float target01) {
    const float epsilon = receiver_params.x;
    const float total_weight = receiver_params.z;
    const uint instance_count = min(dynamic_instance_data.x, DYNAMIC_INSTANCE_LIMIT);
    if (total_weight <= 0.0f || instance_count == 0u) return 0.0f;

    const float target = saturate(target01) * total_weight;
    uint instance_index = INVALID_NODE;
    float previous_instance_weight = 0.0f;

    [loop] for (uint i = 0u; i < instance_count; ++i) {
        const float cumulative = dynamic_instance_emissive[i].y;

        if (cumulative > target) {
            instance_index = i;
            previous_instance_weight = i == 0u ? 0.0f : dynamic_instance_emissive[i - 1u].y;
            break;
        }
    }

    if (instance_index == INVALID_NODE) return 0.0f;

    const uint4 meta = dynamic_instance_meta[instance_index];
    const float local_total = dynamic_instance_emissive[instance_index].x;
    const float area_scale = max(dynamic_instance_emissive[instance_index].z, 1.0e-8f);
    const float instance_weight = dynamic_instance_emissive[instance_index].y - previous_instance_weight;
    if (meta.w == 0u || local_total <= 0.0f || instance_weight <= 0.0f) return 0.0f;

    const float within_instance = saturate((target - previous_instance_weight) / instance_weight);
    const float local_target = within_instance * local_total;
    const uint triangle_begin = meta.z;
    const uint triangle_end = meta.z + meta.w;

    uint lo = triangle_begin;
    uint hi = triangle_end;

    while (lo < hi) {
        uint mid = lo + (hi - lo) / 2u;

        if (DynamicTriangles[mid].emissive.w > local_target)
            hi = mid;
        else
            lo = mid + 1u;
    }

    if (lo >= triangle_end) return 0.0f;

    BvhTriangle tri = DynamicTriangles[lo];
    const float previous_triangle = lo == triangle_begin ? 0.0f : DynamicTriangles[lo - 1u].emissive.w;
    const float triangle_weight = tri.emissive.w - previous_triangle;
    if (triangle_weight <= 0.0f) return 0.0f;

    const float root = sqrt(random01(seed));
    const float bary = random01(seed);
    const float3 local_position =
        tri.a.xyz * (1.0f - root) + tri.b.xyz * (root * (1.0f - bary)) + tri.c.xyz * (root * bary);

    const float3 world_a = mul(dynamic_instance_model[instance_index], float4(tri.a.xyz, 1.0f)).xyz;
    const float3 world_b = mul(dynamic_instance_model[instance_index], float4(tri.b.xyz, 1.0f)).xyz;
    const float3 world_c = mul(dynamic_instance_model[instance_index], float4(tri.c.xyz, 1.0f)).xyz;
    const float3 light_position = mul(dynamic_instance_model[instance_index], float4(local_position, 1.0f)).xyz;

    const float3 world_cross = cross(world_b - world_a, world_c - world_a);
    const float world_area = 0.5f * length(world_cross);
    if (world_area <= 1.0e-10f) return 0.0f;

    const float3 delta = light_position - position;
    const float distance2 = dot(delta, delta);
    if (distance2 <= epsilon * epsilon) return 0.0f;

    const float distance = sqrt(distance2);
    const float3 direction = delta / distance;
    const float receiver_cosine = saturate(dot(normal, direction));
    const float emitter_cosine = abs(dot(normalize(world_cross), -direction));
    if (receiver_cosine <= 0.0f || emitter_cosine <= 0.0f) return 0.0f;

    TraceRay shadow;
    shadow.origin = position + normalize(normal) * epsilon;
    shadow.tmin = epsilon;
    shadow.direction = direction;
    shadow.tmax = max(epsilon, distance - 2.0f * epsilon);

    if (dynamic_any(shadow) || static_any(shadow)) return 0.0f;

    const float triangle_probability = (triangle_weight * area_scale) / total_weight;
    const float pdf_area = triangle_probability / world_area;
    if (pdf_area <= 1.0e-12f) return 0.0f;

    return max(tri.emissive.rgb, 0.0f) *
           (receiver_cosine * emitter_cosine / max(distance2 * pdf_area, 1.0e-8f));
}

float3 direct_emissive_stratified(float3 position, float3 normal, inout uint seed, uint sample_index, uint sample_count) {
    if (sample_count == 0u) return 0.0f;

    const float target01 = ((float)sample_index + random01(seed)) / (float)sample_count;
    return direct_emissive_target(position, normal, seed, target01);
}

float4 source_pixel(int2 p) {
    return Source.Load(int3(p, 0));
}

float4 filtered_pixel(int2 p) {
    const uint generation = (uint)receiver_params.w;
    float4 center = source_pixel(p);
    if (generation_age(center.a, generation) > 1u) return 0.0f;

    float3 sum = 0.0f;
    float alpha_sum = 0.0f;
    float total = 0.0f;

    [unroll] for (int y = -1; y <= 1; ++y) {
        [unroll] for (int x = -1; x <= 1; ++x) {
            int2 q = clamp(p + int2(x, y), int2(0, 0), int2((int)dispatch_data.z - 1, (int)dispatch_data.w - 1));
            float4 c = source_pixel(q);
            if (generation_age(c.a, generation) > 2u) continue;

            float difference = length(c.rgb - center.rgb);
            float weight = 1.0f / (1.0f + difference * 4.0f);
            sum += c.rgb * weight;
            alpha_sum += c.a * weight;
            total += weight;
        }
    }

    return total > 0.0f ? float4(sum / total, alpha_sum / total) : center;
}

[numthreads(64, 1, 1)]
void dynamic_receiver_cs(uint3 id : SV_DispatchThreadID) {
    const uint phase = dispatch_data.x;
    const uint item_count = dispatch_data.y;
    if (id.x >= item_count) return;

    if (phase == PHASE_DIRECT) {
        SurfaceSample sample = Samples[dynamic_instance_data.y + id.x];
        uint pixel = asuint(sample.position.w);
        uint pixel_count = dispatch_data.z * dispatch_data.w;
        if (pixel >= pixel_count) return;

        float3 normal = normalize(sample.normal.xyz);

        if (!receiver_relevant(sample.position.xyz) || receiver_params.z <= 0.0f) {
            Output[uint2(pixel % dispatch_data.z, pixel / dispatch_data.z)] = float4(0.0f, 0.0f, 0.0f, receiver_params.w);
            return;
        }

        uint generation = (uint)receiver_params.w;
        uint seed = hash_u32(pixel ^ 0xb5297a4du ^ hash_u32(generation * 0x68bc21ebu));
        float3 emissive_sum = 0.0f;

        [loop] for (uint i = 0u; i < EMISSIVE_SAMPLES; ++i)
            emissive_sum += direct_emissive_stratified(sample.position.xyz, normal, seed, i, EMISSIVE_SAMPLES);

        float3 current = emissive_sum / (float)EMISSIVE_SAMPLES;
        float4 history = Source.Load(int3(uint2(pixel % dispatch_data.z, pixel / dispatch_data.z), 0));
        uint history_age = generation_age(history.a, generation);
        float age_weight = history_age <= 12u ? 1.0f - (float)history_age / 13.0f : 0.0f;
        float history_weight = saturate(temporal_params.x) * age_weight;
        float3 accumulated = lerp(current, max(history.rgb, 0.0f), history_weight);

        Output[uint2(pixel % dispatch_data.z, pixel / dispatch_data.z)] =
            float4(accumulated, receiver_params.w);
        return;
    }

    if (phase == PHASE_FILTER) {
        SurfaceSample sample = Samples[dynamic_instance_data.y + id.x];
        uint pixel = asuint(sample.position.w);
        uint pixel_count = dispatch_data.z * dispatch_data.w;
        if (pixel >= pixel_count) return;

        uint2 p = uint2(pixel % dispatch_data.z, pixel / dispatch_data.z);
        Output[p] = filtered_pixel(int2(p));
    }
}

#endif
