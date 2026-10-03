#ifndef GPU_BIND_S
#define GPU_BIND_S(n, s) [[vk::binding(n, s)]]
#define GPU_BIND_T(n, s) [[vk::binding(n + 16, s)]]
#define GPU_BIND_B(n, s) [[vk::binding(n + 32, s)]]
#define GPU_BIND_U(n, s) [[vk::binding(n + 48, s)]]
#define GPU_STORAGE_RGBA16F [[vk::image_format("rgba16f")]]
#endif

#include "transport.hlsl"

#if defined(BUILD_SURFACE_FS)
GPU_BIND_T(0, 2) Texture2D<float4> BaseColor : register(t0, space2);
GPU_BIND_T(1, 2) Texture2D<float4> MetallicRoughness : register(t1, space2);
GPU_BIND_T(2, 2) Texture2D<float4> NormalMap : register(t2, space2);
GPU_BIND_T(3, 2) Texture2D<float4> Occlusion : register(t3, space2);
GPU_BIND_T(4, 2) Texture2D<float4> Emissive : register(t4, space2);
GPU_BIND_T(5, 2) Texture2D<float4> Lightmap : register(t5, space2);
GPU_BIND_T(6, 2) Texture2D<float4> Transmission : register(t6, space2);
GPU_BIND_T(7, 2) Texture2D<float4> Thickness : register(t7, space2);
GPU_BIND_T(8, 2) Texture2D<float4> Iridescence : register(t8, space2);
GPU_BIND_T(9, 2) Texture2D<float4> IridescenceThickness : register(t9, space2);
GPU_BIND_T(10, 2) Texture2D<float4> SceneColor : register(t10, space2);
GPU_BIND_T(11, 2) Texture2D<float> DynamicShadow : register(t11, space2);
struct SurfaceProbe {
    float4 position;
    float4 coefficient[9];
};

GPU_BIND_T(12, 2) StructuredBuffer<float> SurfaceBeams : register(t12, space2);
GPU_BIND_T(13, 2) StructuredBuffer<SurfaceProbe> SurfaceProbes : register(t13, space2);
GPU_BIND_T(14, 2) Texture3D<float4> DynamicRadiance0 : register(t14, space2);
GPU_BIND_T(15, 2) Texture3D<float4> DynamicRadiance1 : register(t15, space2);
GPU_BIND_T(16, 2) Texture3D<float4> DynamicRadiance2 : register(t16, space2);
GPU_BIND_T(17, 2) Texture3D<float4> DynamicRadiance3 : register(t17, space2);
GPU_BIND_T(18, 2) Texture3D<float4> DynamicRadiance4 : register(t18, space2);
GPU_BIND_T(19, 2) Texture3D<float4> DynamicRadiance5 : register(t19, space2);
GPU_BIND_T(20, 2) Texture3D<float4> DynamicRadiance6 : register(t20, space2);
GPU_BIND_T(21, 2) Texture3D<float> DynamicRadianceVisibility : register(t21, space2);

GPU_BIND_S(0, 2) SamplerState MaterialSampler : register(s0, space2);
GPU_BIND_S(1, 2) SamplerState LightmapSampler : register(s1, space2);
GPU_BIND_S(2, 2) SamplerState SceneSampler : register(s2, space2);
GPU_BIND_S(3, 2) SamplerState DynamicShadowSampler : register(s3, space2);
GPU_BIND_S(4, 2) SamplerState DynamicRadianceSampler : register(s4, space2);

GPU_BIND_B(0, 3) cbuffer MaterialData : register(b0, space3) {
    float4 base_color_factor;
    float4 emissive_metallic;
    float4 roughness_normal_ao_sun;
    float4 sun_direction;
    float4 sun_color;
    float4 camera_position;

    float4 ior_transmission_volume;
    float4 attenuation_iridescence;
    float4 iridescence_params;

    float4 camera_right_tan;
    float4 camera_up_tan;
    float4 camera_forward;
    float4 sky_zenith;
    float4 sky_horizon;

    float4 shadow_u_min;
    float4 shadow_v_min;
    float4 shadow_sun_max;
    float4 shadow_extent_bias;
    float4 shadow_texel_enabled;
    float4 dynamic_flags;

    float4 probe_origin_spacing;
    uint4 probe_dims;
    float4 beam_origin;
    float4 beam_step;
    uint4 beam_dims;

    uint4 dynamic_influence_meta;
    float4 dynamic_influence_center_radius[8];
    float4 dynamic_influence_axis_x[8];
    float4 dynamic_influence_axis_y[8];
    float4 dynamic_influence_axis_z[8];
    float4 dynamic_influence_diffuse[8];
    float4 dynamic_influence_emissive[8];

    float4 dynamic_radiance_origin_spacing[8];
    uint4 dynamic_radiance_dims_offset[8];
    float4 dynamic_visibility_origin_spacing[8];
    uint4 dynamic_visibility_dims_offset[8];
    float4x4 dynamic_radiance_inverse[8];
    float4x4 dynamic_radiance_model[8];
};

struct SurfaceInput {
    float4 position : SV_Position;
    float3 world_position : TEXCOORD0;
    float3 world_normal : TEXCOORD1;
    float2 uv : TEXCOORD2;
    float2 lightmap_uv : TEXCOORD3;
    float3 view_normal : TEXCOORD4;
    float view_depth : TEXCOORD5;
    float2 back_lightmap_uv : TEXCOORD6;
};

struct SurfaceOutput {
    float4 hdr : SV_Target0;
    float4 normal_depth : SV_Target1;
};

static const float PI = 3.14159265358979323846f;

float distribution_ggx(float n_dot_h, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float d = n_dot_h * n_dot_h * (a2 - 1.0f) + 1.0f;
    return a2 / max(PI * d * d, 1.0e-5f);
}

float geometry_schlick(float n_dot_v, float roughness) {
    float r = roughness + 1.0f;
    float k = (r * r) * 0.125f;
    return n_dot_v / max(n_dot_v * (1.0f - k) + k, 1.0e-5f);
}

float3 fresnel_schlick(float cos_theta, float3 f0) {
    return f0 + (1.0f - f0) * pow(1.0f - saturate(cos_theta), 5.0f);
}

float dielectric_f0(float ior) {
    float r = (ior - 1.0f) / max(ior + 1.0f, 1.0e-4f);
    return r * r;
}

float fresnel_schlick_scalar(float f0, float cos_theta) {
    return f0 + (1.0f - f0) * pow(1.0f - saturate(cos_theta), 5.0f);
}

float3 fresnel0_to_ior(float3 f0) {
    float3 root = sqrt(saturate(f0));
    return (1.0f + root) / max(1.0f - root, 1.0e-4f);
}

float3 iridescence_sensitivity(float opd, float3 shift) {
    float phase = 2.0f * PI * opd * 1.0e-9f;
    float phase_sq = phase * phase;

    float3 val = float3(5.4856e-13f, 4.4201e-13f, 5.2481e-13f);
    float3 pos = float3(1.6810e+06f, 1.7953e+06f, 2.2084e+06f);
    float3 var = float3(4.3278e+09f, 9.3046e+09f, 6.6121e+09f);

    float3 xyz = val * sqrt(2.0f * PI * var) * cos(pos * phase + shift) * exp(-phase_sq * var);
    xyz.x += 9.7470e-14f * sqrt(2.0f * PI * 4.5282e+09f) * cos(2.2399e+06f * phase + shift.x) * exp(-4.5282e+09f * phase_sq);
    xyz /= 1.0685e-7f;

    return float3(3.2404542f * xyz.x - 1.5371385f * xyz.y - 0.4985314f * xyz.z,
                  -0.9692660f * xyz.x + 1.8760108f * xyz.y + 0.0415560f * xyz.z,
                  0.0556434f * xyz.x - 0.2040259f * xyz.y + 1.0572252f * xyz.z);
}

float3 thin_film_fresnel(float cos_theta, float3 base_f0, float film_ior, float thickness_nm) {
    const float outside_ior = 1.0f;

    film_ior = max(film_ior, 1.001f);
    cos_theta = saturate(cos_theta);

    float interface_f0 = dielectric_f0(film_ior / outside_ior);
    float r12 = fresnel_schlick_scalar(interface_f0, cos_theta);
    float t121 = 1.0f - r12;

    float sin_theta2_sq = (outside_ior * outside_ior / (film_ior * film_ior)) * (1.0f - cos_theta * cos_theta);
    float cos_theta2_sq = 1.0f - sin_theta2_sq;
    if (cos_theta2_sq < 0.0f) return 1.0f.xxx;

    float cos_theta2 = sqrt(cos_theta2_sq);
    float3 base_ior = fresnel0_to_ior(min(base_f0 + 0.0001f, 0.9999f));

    float3 interface_ratio = (base_ior - film_ior) / max(base_ior + film_ior, 1.0e-4f);
    float3 r1 = interface_ratio * interface_ratio;
    float3 r23 = r1 + (1.0f - r1) * pow(1.0f - cos_theta2, 5.0f);

    float opd = 2.0f * film_ior * max(thickness_nm, 0.0f) * cos_theta2;
    float phi12 = film_ior < outside_ior ? PI : 0.0f;
    float phi21 = PI - phi12;
    float3 phi23 = PI * (1.0f - step(film_ior.xxx, base_ior));
    float3 phi = phi21.xxx + phi23;

    float3 r123 = clamp(r12 * r23, 1.0e-5f, 0.9999f);
    float3 root_r123 = sqrt(r123);
    float3 rs = (t121 * t121) * r23 / max(1.0f - r123, 1.0e-4f);

    float3 result = r12 + rs;
    float3 cm = rs - t121;

    [unroll]
    for (int order = 1; order <= 2; ++order) {
        cm *= root_r123;
        result += cm * (2.0f * iridescence_sensitivity((float)order * opd, (float)order * phi));
    }

    return max(result, 0.0f);
}

float3 mapped_normal(SurfaceInput input, float scale) {
    float3 n = normalize(input.world_normal);
    float3 dpdx = ddx(input.world_position);
    float3 dpdy = ddy(input.world_position);
    float2 duvdx = ddx(input.uv);
    float2 duvdy = ddy(input.uv);
    float determinant = duvdx.x * duvdy.y - duvdx.y * duvdy.x;

    if (abs(determinant) < 1.0e-7f) return n;

    float inv = 1.0f / determinant;
    float3 t = normalize((dpdx * duvdy.y - dpdy * duvdx.y) * inv);
    t = normalize(t - n * dot(n, t));
    float3 b = normalize(cross(n, t) * (determinant < 0.0f ? -1.0f : 1.0f));

    float3 sample_normal = NormalMap.Sample(MaterialSampler, input.uv).xyz * 2.0f - 1.0f;
    sample_normal.xy *= scale;
    return normalize(t * sample_normal.x + b * sample_normal.y + n * sample_normal.z);
}

float3 material_sky_radiance(float3 direction) {
    direction = normalize(direction);

    float t = saturate(direction.y * 0.5f + 0.5f);
    t = pow(t, 0.35f);

    float3 sky = lerp(sky_horizon.rgb, sky_zenith.rgb, t) * sky_zenith.w;
    float3 sun_dir = normalize(sun_direction.xyz);
    float radius = max(sun_color.w, 0.0001f);
    float sun_cos = cos(radius);
    float halo_cos = cos(radius * 8.0f);
    float d = dot(direction, sun_dir);
    float disc = smoothstep(sun_cos, 1.0f, d);
    float halo = smoothstep(halo_cos, sun_cos, d) * 0.08f;

    return sky + sun_color.rgb * sun_direction.w * (disc + halo);
}

float dynamic_shadow_visibility(float3 position) {
    if (shadow_texel_enabled.z < 0.5f || any(shadow_extent_bias.xyz <= 0.0f)) return 1.0f;

    float sx = dot(position, shadow_u_min.xyz);
    float sy = dot(position, shadow_v_min.xyz);
    float sz = dot(position, shadow_sun_max.xyz);
    float2 uv = (float2(sx, sy) - float2(shadow_u_min.w, shadow_v_min.w)) / shadow_extent_bias.xy;
    float depth = (shadow_sun_max.w - sz) / shadow_extent_bias.z;

    if (any(uv < 0.0f) || any(uv > 1.0f) || depth < 0.0f || depth > 1.0f) return 1.0f;

    float visibility = 0.0f;
    [unroll] for (int y = -1; y <= 1; ++y) {
        [unroll] for (int x = -1; x <= 1; ++x) {
            float2 sample_uv = saturate(uv + float2((float)x, (float)y) * shadow_texel_enabled.xy);
            float blocker = DynamicShadow.SampleLevel(DynamicShadowSampler, sample_uv, 0.0f);
            visibility += depth <= blocker + shadow_extent_bias.w ? 1.0f : 0.0f;
        }
    }
    return visibility / 9.0f;
}

float3 surface_probe_value(SurfaceProbe probe, float3 normal) {
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

float3 surface_probe_irradiance(float3 position, float3 normal) {
    if (probe_dims.w == 0u || probe_dims.x == 0u || probe_dims.y == 0u || probe_dims.z == 0u || probe_origin_spacing.w <= 0.0f)
        return float3(0.12f, 0.12f, 0.12f) * PI;

    normal = normalize(normal);
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
        SurfaceProbe probe = SurfaceProbes[cell.x + probe_dims.x * (cell.y + probe_dims.y * cell.z)];

        weight *= saturate(probe.position.w);
        if (weight <= 0.0f) continue;

        sum += surface_probe_value(probe, normal) * weight;
        weight_sum += weight;
    }

    if (weight_sum > 0.0f) return sum / weight_sum;

    float best_distance2 = 1.0e30f;
    uint best_index = 0u;
    bool found = false;
    int3 center = int3(floor(coord + 0.5f));

    [unroll] for (int z = -1; z <= 1; ++z)
    [unroll] for (int y = -1; y <= 1; ++y)
    [unroll] for (int x = -1; x <= 1; ++x) {
        int3 cell = center + int3(x, y, z);
        if (any(cell < 0) || any(cell >= int3(probe_dims.xyz))) continue;

        uint index = (uint)cell.x + probe_dims.x * ((uint)cell.y + probe_dims.y * (uint)cell.z);
        SurfaceProbe probe = SurfaceProbes[index];
        if (probe.position.w <= 0.0f) continue;

        float3 delta = probe.position.xyz - position;
        float distance2 = dot(delta, delta);
        if (distance2 < best_distance2) {
            best_distance2 = distance2;
            best_index = index;
            found = true;
        }
    }

    return found ? surface_probe_value(SurfaceProbes[best_index], normal) : float3(0.12f, 0.12f, 0.12f) * PI;
}

float static_beam_visibility(float3 position) {
    if (beam_dims.w == 0u || beam_dims.x == 0u || beam_dims.y == 0u || beam_dims.z == 0u ||
        beam_step.x <= 0.0f || beam_step.y <= 0.0f || beam_step.z <= 0.0f)
        return 1.0f;

    float3 sun = normalize(sun_direction.xyz);
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

        visibility += SurfaceBeams[index] * weight;
        total += weight;
    }

    return total > 0.0f ? saturate(visibility / total) : 1.0f;
}

float4 reference_dynamic_lightmap_sample(float2 uv) {
    uint width, height;
    Lightmap.GetDimensions(width, height);

    if (width == 0u || height == 0u) return 0.0f;

    float2 texel_position = uv * float2(width, height) - 0.5f;
    int2 base = int2(floor(texel_position));
    float2 fraction = frac(texel_position);
    float4 sum = 0.0f;
    float weight_sum = 0.0f;
    const float validity_floor = (1.0f / 1024.0f) * 0.5f;

    [unroll] for (uint y = 0u; y < 2u; ++y)
    [unroll] for (uint x = 0u; x < 2u; ++x) {
        int2 pixel = clamp(base + int2(x, y), int2(0, 0), int2((int)width - 1, (int)height - 1));
        float2 axis_weight = lerp(1.0f - fraction, fraction, float2(x, y));
        float weight = axis_weight.x * axis_weight.y;
        float4 sample = Lightmap.Load(int3(pixel, 0));

        if (sample.a < validity_floor) continue;

        sum += sample * weight;
        weight_sum += weight;
    }

    return weight_sum > 1.0e-6f ? sum / weight_sum : 0.0f;
}

struct DynamicIrradianceWeights {
    float4 low;
    float4 high;
    float last;
};

DynamicIrradianceWeights dynamic_radiance_irradiance_weights(float3 normal) {
    normal = normalize(normal);
    const float nx = normal.x, ny = normal.y, nz = normal.z;
    const float l1 = 2.0f * PI / 3.0f;
    const float l2 = PI * 0.25f;

    DynamicIrradianceWeights weights;
    weights.low = float4(
        0.2820947918f * PI,
        0.4886025119f * ny * l1,
        0.4886025119f * nz * l1,
        0.4886025119f * nx * l1);
    weights.high = float4(
        1.0925484306f * nx * ny * l2,
        1.0925484306f * ny * nz * l2,
        0.3153915653f * (3.0f * nz * nz - 1.0f) * l2,
        1.0925484306f * nx * nz * l2);
    weights.last = 0.5462742153f * (nx * nx - ny * ny) * l2;
    return weights;
}

float3 dynamic_radiance_filtered(float3 uvw, DynamicIrradianceWeights weights) {
    float4 p0 = DynamicRadiance0.SampleLevel(DynamicRadianceSampler, uvw, 0.0f);
    float4 p1 = DynamicRadiance1.SampleLevel(DynamicRadianceSampler, uvw, 0.0f);
    float4 p2 = DynamicRadiance2.SampleLevel(DynamicRadianceSampler, uvw, 0.0f);
    float4 p3 = DynamicRadiance3.SampleLevel(DynamicRadianceSampler, uvw, 0.0f);
    float4 p4 = DynamicRadiance4.SampleLevel(DynamicRadianceSampler, uvw, 0.0f);
    float4 p5 = DynamicRadiance5.SampleLevel(DynamicRadianceSampler, uvw, 0.0f);
    float4 p6 = DynamicRadiance6.SampleLevel(DynamicRadianceSampler, uvw, 0.0f);

    float3 c0 = p0.xyz;
    float3 c1 = float3(p0.w, p1.x, p1.y);
    float3 c2 = float3(p1.z, p1.w, p2.x);
    float3 c3 = p2.yzw;
    float3 c4 = p3.xyz;
    float3 c5 = float3(p3.w, p4.x, p4.y);
    float3 c6 = float3(p4.z, p4.w, p5.x);
    float3 c7 = p5.yzw;
    float3 c8 = p6.xyz;

    float3 irradiance =
        c0 * weights.low.x + c1 * weights.low.y + c2 * weights.low.z + c3 * weights.low.w +
        c4 * weights.high.x + c5 * weights.high.y + c6 * weights.high.z + c7 * weights.high.w +
        c8 * weights.last;
    return max(irradiance, 0.0f);
}

float3 dynamic_field_uvw(uint4 meta, float3 coord, uint field_count) {
    float3 texture_size = float3(meta.x, meta.y, max(meta.z * field_count, 1u));
    return (float3(coord.x, coord.y, coord.z + (float)meta.w) + 0.5f) / texture_size;
}

float dynamic_radiance_static_visibility(uint index, float3 local_position) {
    uint4 meta = dynamic_visibility_dims_offset[index];
    float spacing = dynamic_visibility_origin_spacing[index].w;

    if (any(meta.xyz < 2u) || spacing <= 0.0f) return 1.0f;

    float3 coord = (local_position - dynamic_visibility_origin_spacing[index].xyz) / spacing;
    float3 maximum = float3(meta.xyz - 1u);
    if (any(coord < 0.0f) || any(coord > maximum)) return 1.0f;

    uint field_count = max(dynamic_influence_meta.w, 1u);
    float3 uvw = dynamic_field_uvw(meta, coord, field_count);
    return saturate(DynamicRadianceVisibility.SampleLevel(DynamicRadianceSampler, uvw, 0.0f));
}

float3 dynamic_radiance_field(uint index, float3 world_position, float3 world_normal) {
    uint4 meta = dynamic_radiance_dims_offset[index];
    float spacing = dynamic_radiance_origin_spacing[index].w;
    if (any(meta.xyz < 2u) || spacing <= 0.0f) return 0.0f;

    float3 local_position = mul(dynamic_radiance_inverse[index], float4(world_position, 1.0f)).xyz;
    float3 coord = (local_position - dynamic_radiance_origin_spacing[index].xyz) / spacing;
    float3 maximum = float3(meta.xyz - 1u);
    if (any(coord < 0.0f) || any(coord > maximum)) return 0.0f;

    float3 local_normal = normalize(mul(transpose((float3x3)dynamic_radiance_model[index]), world_normal));
    DynamicIrradianceWeights weights = dynamic_radiance_irradiance_weights(local_normal);
    uint field_count = max(dynamic_influence_meta.w, 1u);
    float3 uvw = dynamic_field_uvw(meta, coord, field_count);
    float3 irradiance = dynamic_radiance_filtered(uvw, weights);

    float3 edge_distance = min(coord, maximum - coord);
    float edge = min(edge_distance.x, min(edge_distance.y, edge_distance.z));
    float fade = smoothstep(0.0f, 2.0f, edge);
    float visibility = dynamic_radiance_static_visibility(index, local_position);

    return irradiance * (fade * visibility / PI);
}

float3 dynamic_radiance_lighting(float3 world_position, float3 world_normal) {
    if (dynamic_influence_meta.z == 0u) return 0.0f;

    uint count = min(dynamic_influence_meta.x, 8u);
    float3 result = 0.0f;

    [loop] for (uint i = 0u; i < count; ++i) {
        if (i == dynamic_influence_meta.y) continue;
        result += dynamic_radiance_field(i, world_position, world_normal);
    }

    return result;
}

bool dynamic_proxy_reflection_hit(float3 origin, float3 direction, uint index, out float hit_t, out float3 hit_normal) {
    float3 center = dynamic_influence_center_radius[index].xyz;
    float3 axis_x = dynamic_influence_axis_x[index].xyz;
    float3 axis_y = dynamic_influence_axis_y[index].xyz;
    float3 axis_z = dynamic_influence_axis_z[index].xyz;

    float extent_x = length(axis_x);
    float extent_y = length(axis_y);
    float extent_z = length(axis_z);

    hit_t = 0.0f;
    hit_normal = 0.0f;

    if (extent_x <= 1.0e-5f || extent_y <= 1.0e-5f || extent_z <= 1.0e-5f) return false;

    float3 basis_x = axis_x / extent_x;
    float3 basis_y = axis_y / extent_y;
    float3 basis_z = axis_z / extent_z;
    float3 relative = origin - center;
    float3 local_origin = float3(dot(relative, basis_x), dot(relative, basis_y), dot(relative, basis_z));
    float3 local_direction = float3(dot(direction, basis_x), dot(direction, basis_y), dot(direction, basis_z));
    float3 extents = float3(extent_x, extent_y, extent_z);

    /* Skip the proxy containing the shading point, which is normally self. */
    if (all(abs(local_origin) <= extents * 1.001f)) return false;

    float t_min = 1.0e-4f;
    float t_max = 1.0e20f;

    [unroll] for (uint axis = 0u; axis < 3u; ++axis) {
        float d = local_direction[axis];

        if (abs(d) < 1.0e-7f) {
            if (abs(local_origin[axis]) > extents[axis]) return false;
            continue;
        }

        float a = (-extents[axis] - local_origin[axis]) / d;
        float b = ( extents[axis] - local_origin[axis]) / d;
        if (a > b) {
            float temporary = a;
            a = b;
            b = temporary;
        }

        t_min = max(t_min, a);
        t_max = min(t_max, b);
        if (t_min > t_max) return false;
    }

    if (t_max <= 1.0e-4f) return false;
    hit_t = t_min > 1.0e-4f ? t_min : t_max;
    if (hit_t <= 1.0e-4f) return false;

    float3 local_hit = local_origin + local_direction * hit_t;
    float3 normalized_hit = local_hit / max(extents, float3(1.0e-5f, 1.0e-5f, 1.0e-5f));
    float3 absolute_hit = abs(normalized_hit);

    if (absolute_hit.x >= absolute_hit.y && absolute_hit.x >= absolute_hit.z)
        hit_normal = basis_x * (normalized_hit.x >= 0.0f ? 1.0f : -1.0f);
    else if (absolute_hit.y >= absolute_hit.z)
        hit_normal = basis_y * (normalized_hit.y >= 0.0f ? 1.0f : -1.0f);
    else
        hit_normal = basis_z * (normalized_hit.z >= 0.0f ? 1.0f : -1.0f);

    if (dot(hit_normal, direction) > 0.0f) hit_normal = -hit_normal;
    return true;
}

float3 dynamic_reflection_radiance(float3 origin, float3 direction, float roughness, out float blend_weight) {
    uint count = min(dynamic_influence_meta.x, 8u);
    float closest = 1.0e20f;
    uint best = 0u;
    float3 best_normal = 0.0f;
    bool found = false;

    direction = normalize(direction);

    [loop] for (uint i = 0u; i < count; ++i) {
        if (i == dynamic_influence_meta.y) continue;

        float hit_t;
        float3 hit_normal;

        if (!dynamic_proxy_reflection_hit(origin, direction, i, hit_t, hit_normal) || hit_t >= closest) continue;

        closest = hit_t;
        best = i;
        best_normal = hit_normal;
        found = true;
    }

    if (!found) {
        blend_weight = 0.0f;
        return 0.0f;
    }

    float3 hit_position = origin + direction * closest;
    float3 indirect = surface_probe_irradiance(hit_position, best_normal) / PI;
    float3 sun = normalize(sun_direction.xyz);
    float sun_visibility = static_beam_visibility(hit_position);
    float n_dot_l = saturate(dot(best_normal, sun));
    float3 direct = sun_color.rgb * roughness_normal_ao_sun.w * n_dot_l * sun_visibility;

    float3 outgoing =
        dynamic_influence_diffuse[best].rgb * max(indirect + direct, 0.0f) +
        max(dynamic_influence_emissive[best].rgb, 0.0f);

    /*
     * This is deliberately a bounded glossy hook, not a second lighting path.
     * Rough surfaces lean back toward the environment because the OBB hit is
     * only a single-direction proxy rather than a filtered reflection field.
     */
    blend_weight = saturate(1.0f - roughness * roughness);
    return outgoing;
}

float3 environment_radiance(float3 direction, float roughness) {
    direction = normalize(direction);

    float spread = roughness * roughness * 0.55f;
    if (spread < 0.005f) return material_sky_radiance(direction);

    float3 axis = abs(direction.y) < 0.99f ? float3(0.0f, 1.0f, 0.0f) : float3(1.0f, 0.0f, 0.0f);
    float3 tangent = normalize(cross(axis, direction));
    float3 bitangent = normalize(cross(direction, tangent));

    float3 result = material_sky_radiance(direction) * 4.0f;
    result += material_sky_radiance(normalize(direction + tangent * spread));
    result += material_sky_radiance(normalize(direction - tangent * spread));
    result += material_sky_radiance(normalize(direction + bitangent * spread));
    result += material_sky_radiance(normalize(direction - bitangent * spread));

    return result * 0.125f;
}

float3 volume_attenuation(float3 color, float distance, float attenuation_distance) {
    if (distance <= 0.0f || attenuation_distance <= 0.0f) return 1.0f.xxx;

    float3 safe_color = max(color, 1.0e-4f);
    return exp(log(safe_color) * (distance / attenuation_distance));
}

float2 refracted_scene_uv(SurfaceInput input, float3 direction, float thickness) {
    uint width, height;
    SceneColor.GetDimensions(width, height);

    float2 uv = input.position.xy / max(float2(width, height), 1.0f.xx);
    float forward = max(dot(direction, normalize(camera_forward.xyz)), 0.1f);
    float x = dot(direction, normalize(camera_right_tan.xyz)) / max(camera_right_tan.w * forward, 1.0e-3f);
    float y = dot(direction, normalize(camera_up_tan.xyz)) / max(camera_up_tan.w * forward, 1.0e-3f);
    float scale = 0.5f * max(thickness, 0.0f) / max(input.view_depth, 0.1f);

    return saturate(uv + float2(x, -y) * scale);
}

float3 sample_transmitted_scene(float2 uv, float roughness) {
    uint width, height;
    SceneColor.GetDimensions(width, height);

    float radius = roughness * roughness * 6.0f;
    float3 center = SceneColor.SampleLevel(SceneSampler, uv, 0.0f).rgb;
    if (radius < 0.25f) return center;

    float2 texel = radius / max(float2(width, height), 1.0f.xx);
    float3 result = center * 4.0f;
    result += SceneColor.SampleLevel(SceneSampler, uv + float2(texel.x, 0.0f), 0.0f).rgb;
    result += SceneColor.SampleLevel(SceneSampler, uv - float2(texel.x, 0.0f), 0.0f).rgb;
    result += SceneColor.SampleLevel(SceneSampler, uv + float2(0.0f, texel.y), 0.0f).rgb;
    result += SceneColor.SampleLevel(SceneSampler, uv - float2(0.0f, texel.y), 0.0f).rgb;

    return result * 0.125f;
}

SurfaceOutput surface_fs(SurfaceInput input, bool front_face : SV_IsFrontFace) {
    SurfaceOutput output;

    float4 base_sample = BaseColor.Sample(MaterialSampler, input.uv);
    float3 base = transport_srgb_to_linear(base_sample.rgb) * base_color_factor.rgb;

    float4 mr = MetallicRoughness.Sample(MaterialSampler, input.uv);
    float metallic = saturate(emissive_metallic.w * mr.b);
    float roughness = clamp(roughness_normal_ao_sun.x * mr.g, 0.045f, 1.0f);

    float ao_sample = Occlusion.Sample(MaterialSampler, input.uv).r;
    float material_ao = lerp(1.0f, ao_sample, saturate(roughness_normal_ao_sun.z));
    float3 emissive = transport_srgb_to_linear(Emissive.Sample(MaterialSampler, input.uv).rgb) * emissive_metallic.rgb;

    float3 n = mapped_normal(input, roughness_normal_ao_sun.y);
    if (!front_face) n = -n;

    float3 v = normalize(camera_position.xyz - input.world_position);
    float3 l = normalize(sun_direction.xyz);
    float3 h = normalize(v + l);
    float n_dot_v = max(dot(n, v), 0.001f);
    float n_dot_l = max(dot(n, l), 0.0f);
    float n_dot_h = max(dot(n, h), 0.0f);
    float v_dot_h = max(dot(v, h), 0.0f);

    float ior = max(ior_transmission_volume.x, 1.0f);
    float transmission = saturate(ior_transmission_volume.y * Transmission.Sample(MaterialSampler, input.uv).r);
    float volume_thickness = max(ior_transmission_volume.z, 0.0f) * Thickness.Sample(MaterialSampler, input.uv).g;
    float iridescence = saturate(attenuation_iridescence.w * Iridescence.Sample(MaterialSampler, input.uv).r);
    float iridescence_thickness =
        lerp(iridescence_params.y, iridescence_params.z, IridescenceThickness.Sample(MaterialSampler, input.uv).g);

    float dielectric = dielectric_f0(ior);
    float3 f0 = lerp(dielectric.xxx, base, metallic);

    float3 f = fresnel_schlick(v_dot_h, f0);
    float3 view_f = fresnel_schlick(n_dot_v, f0);

    if (iridescence > 0.0f) {
        float3 film_f = thin_film_fresnel(v_dot_h, f0, iridescence_params.x, iridescence_thickness);
        float3 film_view_f = thin_film_fresnel(n_dot_v, f0, iridescence_params.x, iridescence_thickness);
        f = lerp(f, film_f, iridescence);
        view_f = lerp(view_f, film_view_f, iridescence);
    }

    float d = distribution_ggx(n_dot_h, roughness);
    float g = geometry_schlick(n_dot_v, roughness) * geometry_schlick(n_dot_l, roughness);
    float3 specular = d * g * f / max(4.0f * n_dot_v * max(n_dot_l, 0.001f), 1.0e-4f);

    float is_dynamic = saturate(dynamic_flags.x);
    float dynamic_reference_valid = saturate(dynamic_flags.y);
    float reference_mode = saturate(dynamic_flags.z);
    float2 lighting_uv = front_face ? input.lightmap_uv : input.back_lightmap_uv;
    float4 baked_sample = float4(0.12f, 0.12f, 0.12f, 1.0f);

    if (camera_position.w > 0.5f) {
        baked_sample = is_dynamic > 0.5f && dynamic_reference_valid > 0.5f
                           ? reference_dynamic_lightmap_sample(lighting_uv)
                           : Lightmap.Sample(LightmapSampler, lighting_uv);
    }

    const float visibility_floor = 1.0f / 1024.0f;
    if (is_dynamic > 0.5f && baked_sample.a < visibility_floor * 0.5f) dynamic_reference_valid = 0.0f;

    float cached_sun_visibility = camera_position.w > 0.5f
                                      ? saturate((baked_sample.a - visibility_floor) / (1.0f - visibility_floor))
                                      : 1.0f;

    float3 geometric_normal = normalize(input.world_normal) * (front_face ? 1.0f : -1.0f);
    float geometric_n_dot_l = max(dot(geometric_normal, l), 0.0f);

    if (is_dynamic > 0.5f && dynamic_reference_valid < 0.5f) {
        baked_sample.rgb = surface_probe_irradiance(input.world_position, geometric_normal) / PI;
        cached_sun_visibility = static_beam_visibility(input.world_position);
    }

    float dynamic_visibility = dynamic_shadow_visibility(input.world_position);
    float sun_visibility = reference_mode > 0.5f ? cached_sun_visibility : cached_sun_visibility * dynamic_visibility;
    float3 baked = max(baked_sample.rgb, 0.0f);
    float3 static_direct = sun_color.rgb * roughness_normal_ao_sun.w * geometric_n_dot_l;

    if (is_dynamic > 0.5f && dynamic_reference_valid < 0.5f) {
        baked += static_direct * sun_visibility;
    } else if (reference_mode < 0.5f && camera_position.w > 0.5f) {
        if (is_dynamic > 0.5f)
            baked = max(baked + static_direct * (sun_visibility - cached_sun_visibility), 0.0f);
        else
            baked = max(baked + static_direct * cached_sun_visibility * (dynamic_visibility - 1.0f), 0.0f);
    }

    float3 dynamic_correction = 0.0f;

    if (reference_mode < 0.5f && camera_position.w > 0.5f) {
        dynamic_correction = dynamic_radiance_lighting(input.world_position, geometric_normal);
        baked = max(baked + dynamic_correction, 0.0f);
    }

    if (dynamic_flags.w > 4.5f && dynamic_flags.w < 5.5f) {
        float3 receive_debug = is_dynamic > 0.5f
                                 ? (dynamic_reference_valid > 0.5f ? float3(0.05f, 0.25f, 1.0f) : float3(0.05f, 1.0f, 0.05f))
                                 : float3(0.04f, 0.04f, 0.04f);

        if (reference_mode > 0.5f) receive_debug = lerp(receive_debug, float3(0.05f, 0.25f, 1.0f), 0.55f);

        output.hdr = float4(receive_debug, 1.0f);
        output.normal_depth = float4(normalize(input.view_normal) * (front_face ? 0.5f : -0.5f) + 0.5f, max(input.view_depth, 0.0f));
        return output;
    }

    if (dynamic_flags.w > 5.5f && dynamic_flags.w < 6.5f) {
        float positive = length(max(dynamic_correction, 0.0f));
        float3 correction_debug = saturate(float3(0.0f, positive, 0.0f) * 6.0f);

        output.hdr = float4(correction_debug, 1.0f);
        output.normal_depth = float4(normalize(input.view_normal) * (front_face ? 0.5f : -0.5f) + 0.5f, max(input.view_depth, 0.0f));
        return output;
    }

    if (camera_position.w > 1.5f) {
        output.hdr = float4(baked, 1.0f);
        output.normal_depth = float4(normalize(input.view_normal) * (front_face ? 0.5f : -0.5f) + 0.5f, max(input.view_depth, 0.0f));
        return output;
    }

    float baked_luma = dot(baked, float3(0.2126f, 0.7152f, 0.0722f));
    float3 direct_specular = specular * sun_color.rgb * roughness_normal_ao_sun.w * n_dot_l * sun_visibility;

    float3 legacy_environment = f0 * (0.025f + 0.10f * (1.0f - roughness)) * material_ao *
                                (camera_position.w > 0.5f ? saturate(baked_luma * 2.0f) : 1.0f);

    float3 reflection_direction = reflect(-v, n);
    float3 reflected = environment_radiance(reflection_direction, roughness);
    float dynamic_reflection_weight = 0.0f;
    float3 dynamic_reflection = dynamic_reflection_radiance(input.world_position + n * 1.0e-3f, reflection_direction, roughness,
                                                            dynamic_reflection_weight);
    reflected = lerp(reflected, dynamic_reflection, dynamic_reflection_weight);
    float3 physical_environment = reflected * view_f * material_ao * (1.0f - 0.35f * roughness);
    float advanced_weight = max(iridescence, transmission);
    float3 environment_specular = lerp(legacy_environment, physical_environment, advanced_weight);

    float transmission_weight = transmission * (1.0f - metallic);
    float iridescence_energy = lerp(1.0f, 1.0f - max(view_f.r, max(view_f.g, view_f.b)), iridescence);
    float3 diffuse_albedo = transport_diffuse_albedo(base, metallic);
    float3 diffuse = transport_diffuse_response(diffuse_albedo, baked) * material_ao * (1.0f - transmission_weight) * iridescence_energy;
    float3 transmitted = 0.0f.xxx;

    if (transmission_weight > 0.0f) {
        float3 refracted = refract(-v, n, 1.0f / ior);
        if (dot(refracted, refracted) < 1.0e-6f) refracted = reflect(-v, n);
        refracted = normalize(refracted);

        float path_length = volume_thickness / max(abs(dot(n, refracted)), 0.1f);
        float3 attenuation = volume_attenuation(attenuation_iridescence.rgb, path_length, ior_transmission_volume.w);
        float2 scene_uv = refracted_scene_uv(input, refracted, volume_thickness);
        float3 scene = max(sample_transmitted_scene(scene_uv, roughness), 0.0f);

        transmitted = scene * attenuation * base * transmission_weight * (1.0f - view_f);
    }

    output.hdr = float4(max(diffuse + direct_specular + environment_specular + transmitted + transport_visible_emissive(emissive), 0.0f),
                        base_sample.a * base_color_factor.a);
    output.normal_depth = float4(normalize(input.view_normal) * (front_face ? 0.5f : -0.5f) + 0.5f, max(input.view_depth, 0.0f));

    return output;
}
#elif defined(BUILD_SKY_FS)
GPU_BIND_B(0, 3) cbuffer SkyData : register(b0, space3) {
    float4 camera_right;
    float4 camera_up;
    float4 camera_forward;
    float4 sky_zenith;
    float4 sky_horizon;
    float4 sun_direction_intensity;
    float4 sun_color_radius;
};

struct SkyInput {
    float4 position : SV_Position;
    float2 ndc : TEXCOORD0;
};

struct SkyOutput {
    float4 hdr : SV_Target0;
    float4 normal_depth : SV_Target1;
};

float3 sky_radiance(float3 direction) {
    float t = saturate(direction.y * 0.5f + 0.5f);
    t = pow(t, 0.35f);
    float3 sky = lerp(sky_horizon.rgb, sky_zenith.rgb, t) * sky_zenith.w;

    float3 sun_dir = normalize(sun_direction_intensity.xyz);
    float sun_cos = cos(sun_color_radius.w);
    float halo_cos = cos(sun_color_radius.w * 8.0f);
    float d = dot(direction, sun_dir);
    float disc = smoothstep(sun_cos, 1.0f, d);
    float halo = smoothstep(halo_cos, sun_cos, d) * 0.08f;
    return sky + sun_color_radius.rgb * sun_direction_intensity.w * (disc + halo);
}

SkyOutput sky_fs(SkyInput input) {
    float3 direction = normalize(camera_forward.xyz + camera_right.xyz * input.ndc.x + camera_up.xyz * input.ndc.y);
    SkyOutput output;
    output.hdr = float4(sky_radiance(direction), 1.0f);
    output.normal_depth = float4(0.0f, 0.0f, 0.0f, 0.0f);
    return output;
}
#elif defined(BUILD_COMPOSE_FS)
GPU_BIND_T(0, 2) Texture2D<float4> Hdr : register(t0, space2);
GPU_BIND_S(0, 2) SamplerState HdrSampler : register(s0, space2);
GPU_BIND_T(1, 2) Texture2D<float4> Ao : register(t1, space2);
GPU_BIND_S(1, 2) SamplerState AoSampler : register(s1, space2);
GPU_BIND_T(2, 2) Texture2D<float4> Bloom : register(t2, space2);
GPU_BIND_S(2, 2) SamplerState BloomSampler : register(s2, space2);
GPU_BIND_T(3, 2) Texture2D<float4> Lut : register(t3, space2);
GPU_BIND_S(3, 2) SamplerState LutSampler : register(s3, space2);

GPU_BIND_B(0, 3) cbuffer ComposeData : register(b0, space3) {
    float exposure;
    float ao_strength;
    float bloom_strength;
    float _pad;
};

struct ComposeInput {
    float4 position : SV_Position;
    float2 ndc : TEXCOORD0;
};

float3 aces(float3 x) {
    const float a = 2.51f;
    const float b = 0.03f;
    const float c = 2.43f;
    const float d = 0.59f;
    const float e = 0.14f;
    return saturate((x * (a * x + b)) / (x * (c * x + d) + e));
}

float3 sample_lut(float3 color) {
    const float size = 16.0f;
    color = saturate(color);
    float blue = color.b * (size - 1.0f);
    float b0 = floor(blue);
    float b1 = min(b0 + 1.0f, size - 1.0f);
    float r = color.r * (size - 1.0f);
    float g = color.g * (size - 1.0f);

    float2 uv0 = float2((b0 * size + r + 0.5f) / (size * size), (g + 0.5f) / size);
    float2 uv1 = float2((b1 * size + r + 0.5f) / (size * size), (g + 0.5f) / size);
    float3 a = Lut.SampleLevel(LutSampler, uv0, 0.0f).rgb;
    float3 b = Lut.SampleLevel(LutSampler, uv1, 0.0f).rgb;
    return lerp(a, b, frac(blue));
}

float4 compose_fs(ComposeInput input) : SV_Target0 {
    uint width, height;
    Hdr.GetDimensions(width, height);
    float2 uv = input.position.xy / float2(width, height);

    if (exposure < 0.0f) return float4(saturate(Hdr.SampleLevel(HdrSampler, uv, 0.0f).rgb), 1.0f);

    float3 hdr = Hdr.SampleLevel(HdrSampler, uv, 0.0f).rgb * exposure;
    float ao = Ao.SampleLevel(AoSampler, uv, 0.0f).r;
    float3 bloom = Bloom.SampleLevel(BloomSampler, uv, 0.0f).rgb;

    hdr *= lerp(1.0f, ao, ao_strength);
    hdr += bloom * bloom_strength;

    float3 mapped = aces(max(hdr, 0.0f));
    mapped = sample_lut(mapped);
    mapped = pow(saturate(mapped), 1.0f / 2.2f);
    return float4(mapped, 1.0f);
}
#elif defined(BUILD_WIREFRAME_FS)
struct WireframeInput {
    float4 position : SV_Position;
    float4 color : TEXCOORD0;
};

float4 wireframe_fs(WireframeInput input) : SV_Target0 {
    return input.color;
}
#endif
