#ifndef GPU_BIND_S
#define GPU_BIND_S(n, s) [[vk::binding(n, s)]]
#define GPU_BIND_T(n, s) [[vk::binding(n + 16, s)]]
#define GPU_BIND_B(n, s) [[vk::binding(n + 32, s)]]
#define GPU_BIND_U(n, s) [[vk::binding(n + 48, s)]]
#define GPU_STORAGE_RGBA16F [[vk::image_format("rgba16f")]]
#endif

#if defined(BUILD_DYNAMIC_SURFACE_CS)

static const float PI = 3.14159265358979323846f;

struct SurfaceSample {
    float4 position;
    float4 normal;
};

struct SurfaceProbe {
    float4 position;
    float4 coefficient[9];
};

GPU_BIND_T(0, 0) StructuredBuffer<SurfaceSample> Samples : register(t0, space0);
GPU_BIND_T(1, 0) StructuredBuffer<SurfaceProbe> Probes : register(t1, space0);
GPU_BIND_T(2, 0) StructuredBuffer<float> Beams : register(t2, space0);
GPU_BIND_U(0, 1) GPU_STORAGE_RGBA16F RWTexture2D<float4> Output : register(u0, space1);

GPU_BIND_B(0, 2) cbuffer DynamicSurfaceData : register(b0, space2) {
    float4x4 model;
    float4x4 normal_model;

    uint sample_offset;
    uint sample_count;
    uint texture_width;
    uint texture_height;

    float4 probe_origin_spacing;
    uint4 probe_dims;
    float4 beam_origin;
    float4 beam_step;
    uint4 beam_dims;

    float4 sun_direction_intensity;
    float4 sun_color_visibility_floor;
};

float3 probe_value(SurfaceProbe probe, float3 normal) {
    const float nx = normal.x, ny = normal.y, nz = normal.z;
    float3 irradiance = probe.coefficient[0].rgb * (0.2820947918f * PI);
    irradiance += (probe.coefficient[1].rgb * (0.4886025119f * ny) +
                   probe.coefficient[2].rgb * (0.4886025119f * nz) +
                   probe.coefficient[3].rgb * (0.4886025119f * nx)) * (2.0f * PI / 3.0f);
    irradiance += (probe.coefficient[4].rgb * (1.0925484306f * nx * ny) +
                   probe.coefficient[5].rgb * (1.0925484306f * ny * nz) +
                   probe.coefficient[6].rgb * (0.3153915653f * (3.0f * nz * nz - 1.0f)) +
                   probe.coefficient[7].rgb * (1.0925484306f * nx * nz) +
                   probe.coefficient[8].rgb * (0.5462742153f * (nx * nx - ny * ny))) * (PI * 0.25f);
    return max(irradiance, 0.0f);
}

float3 probe_irradiance(float3 position, float3 normal) {
    if (probe_dims.w == 0u || probe_dims.x == 0u || probe_dims.y == 0u || probe_dims.z == 0u || probe_origin_spacing.w <= 0.0f)
        return float3(0.12f, 0.12f, 0.12f) * PI;

    float3 coord = clamp((position - probe_origin_spacing.xyz) / probe_origin_spacing.w, 0.0f, float3(probe_dims.xyz) - 1.0f);
    uint3 base = uint3(floor(coord));
    float3 fraction = frac(coord);
    float3 sum = 0.0f;
    float weight_sum = 0.0f;

    [unroll] for (uint z = 0u; z < 2u; ++z)
    [unroll] for (uint y = 0u; y < 2u; ++y)
    [unroll] for (uint x = 0u; x < 2u; ++x) {
        uint3 cell = min(base + uint3(x, y, z), probe_dims.xyz - 1u);
        float3 axis_weight = lerp(1.0f - fraction, fraction, float3(x, y, z));
        float weight = axis_weight.x * axis_weight.y * axis_weight.z;
        SurfaceProbe probe = Probes[cell.x + probe_dims.x * (cell.y + probe_dims.y * cell.z)];
        weight *= saturate(probe.position.w);

        if (weight <= 0.0f) continue;
        sum += probe_value(probe, normal) * weight;
        weight_sum += weight;
    }

    return weight_sum > 0.0f ? sum / weight_sum : float3(0.12f, 0.12f, 0.12f) * PI;
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

    SurfaceSample sample = Samples[sample_offset + id.x];
    uint pixel = asuint(sample.position.w);
    uint pixel_count = texture_width * texture_height;
    if (pixel >= pixel_count) return;

    float3 world_position = mul(model, float4(sample.position.xyz, 1.0f)).xyz;
    float3 world_normal = normalize(mul((float3x3)normal_model, sample.normal.xyz));
    float visibility = beam_visibility(world_position);
    float3 indirect = probe_irradiance(world_position, world_normal) / PI;
    float n_dot_l = saturate(dot(world_normal, normalize(sun_direction_intensity.xyz)));
    float3 direct = sun_color_visibility_floor.rgb * (sun_direction_intensity.w * n_dot_l * visibility);
    float visibility_floor = max(sun_color_visibility_floor.w, 1.0f / 65504.0f);
    float encoded_visibility = visibility_floor + visibility * (1.0f - visibility_floor);

    Output[uint2(pixel % texture_width, pixel / texture_width)] = float4(max(indirect + direct, 0.0f), encoded_visibility);
}

#endif
