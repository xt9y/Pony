#ifndef GPU_BIND_S
#define GPU_BIND_S(n,s) [[vk::binding(n, s)]]
#define GPU_BIND_T(n,s) [[vk::binding(n + 16, s)]]
#define GPU_BIND_B(n,s) [[vk::binding(n + 32, s)]]
#define GPU_BIND_U(n,s) [[vk::binding(n + 48, s)]]
#define GPU_STORAGE_RGBA16F [[vk::image_format("rgba16f")]]
#endif

#if defined(BUILD_SURFACE_FS)
GPU_BIND_T(0, 2) Texture2D<float4> BaseColor : register(t0, space2);
GPU_BIND_S(0, 2) SamplerState BaseColorSampler : register(s0, space2);
GPU_BIND_T(1, 2) Texture2D<float4> MetallicRoughness : register(t1, space2);
GPU_BIND_S(1, 2) SamplerState MetallicRoughnessSampler : register(s1, space2);
GPU_BIND_T(2, 2) Texture2D<float4> NormalMap : register(t2, space2);
GPU_BIND_S(2, 2) SamplerState NormalMapSampler : register(s2, space2);
GPU_BIND_T(3, 2) Texture2D<float4> Occlusion : register(t3, space2);
GPU_BIND_S(3, 2) SamplerState OcclusionSampler : register(s3, space2);
GPU_BIND_T(4, 2) Texture2D<float4> Emissive : register(t4, space2);
GPU_BIND_S(4, 2) SamplerState EmissiveSampler : register(s4, space2);
GPU_BIND_T(5, 2) Texture2D<float4> IndirectLightmap : register(t5, space2);
GPU_BIND_S(5, 2) SamplerState IndirectLightmapSampler : register(s5, space2);
GPU_BIND_T(6, 2) Texture2D<float4> DirectSunLightmap : register(t6, space2);
GPU_BIND_S(6, 2) SamplerState DirectSunLightmapSampler : register(s6, space2);
GPU_BIND_T(7, 2) Texture2D<float> DynamicShadow : register(t7, space2);
GPU_BIND_S(7, 2) SamplerState DynamicShadowSampler : register(s7, space2);
struct SurfaceProbe { float4 position; float4 coefficient[9]; };
GPU_BIND_T(8, 2) StructuredBuffer<SurfaceProbe> SurfaceProbes : register(t8, space2);
GPU_BIND_T(9, 2) StructuredBuffer<float> SurfaceBeams : register(t9, space2);
GPU_BIND_T(10, 2) Texture2D<float4> DynamicOverlay : register(t10, space2);
GPU_BIND_T(11, 2) StructuredBuffer<uint> CellGenerations : register(t11, space2);
GPU_BIND_T(12, 2) StructuredBuffer<uint> OverlayGenerations : register(t12, space2);

GPU_BIND_B(0, 1) cbuffer Camera : register(b0, space1)
{
    float4x4 mvp;
    float4x4 view;
    float4x4 model;
    float4x4 normal_model;
    uint object_dynamic;
    uint3 _camera_pad;
};

GPU_BIND_B(0, 3) cbuffer MaterialData : register(b0, space3)
{
    float4 base_color_factor;
    float4 emissive_metallic;
    float4 roughness_normal_ao_sun;
    float4 sun_direction;
    float4 sun_color;
    float4 camera_position;
    float4 probe_origin_spacing;
    uint4 probe_dims;
    float4 beam_origin;
    float4 beam_step;
    uint4 beam_dims;
    float4 shadow_u_min;
    float4 shadow_v_min;
    float4 shadow_sun_max;
    float4 shadow_extent_bias;
    float4 shadow_texel_enabled;
    float4 dynamic_grid_origin_cell;
    uint4 dynamic_grid_dims_target;
    // float4 camera_forward; // Fragment depth diagnostic.
};

struct SurfaceInput
{
    float4 position : SV_Position;
    float3 world_position : TEXCOORD0;
    float3 world_normal : TEXCOORD1;
    float2 uv : TEXCOORD2;
    float2 lightmap_uv : TEXCOORD3;
    float3 view_normal : TEXCOORD4;
    float view_depth : TEXCOORD5;
    float2 back_lightmap_uv : TEXCOORD6;
};

struct SurfaceOutput
{
    float4 hdr : SV_Target0;
    float4 normal_depth : SV_Target1;
};

static const float PI = 3.14159265358979323846f;

float3 srgb_to_linear(float3 c)
{
    float3 low = c / 12.92f;
    float3 high = pow((c + 0.055f) / 1.055f, 2.4f);
    return lerp(high, low, step(c, float3(0.04045f, 0.04045f, 0.04045f)));
}

float distribution_ggx(float n_dot_h, float roughness)
{
    float a = roughness * roughness;
    float a2 = a * a;
    float d = n_dot_h * n_dot_h * (a2 - 1.0f) + 1.0f;
    return a2 / max(PI * d * d, 1.0e-5f);
}

float geometry_schlick(float n_dot_v, float roughness)
{
    float r = roughness + 1.0f;
    float k = (r * r) * 0.125f;
    return n_dot_v / max(n_dot_v * (1.0f - k) + k, 1.0e-5f);
}

float3 fresnel_schlick(float cos_theta, float3 f0)
{
    return f0 + (1.0f - f0) * pow(1.0f - saturate(cos_theta), 5.0f);
}

float3 surface_probe_value(SurfaceProbe probe, float3 normal)
{
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

float3 surface_probe_irradiance(float3 position, float3 normal)
{
    if (probe_dims.x == 0u || probe_dims.y == 0u || probe_dims.z == 0u || probe_origin_spacing.w <= 0.0f)
        return float3(0.12f, 0.12f, 0.12f);

    normal = normalize(normal);
    float3 coord = clamp((position - probe_origin_spacing.xyz) / probe_origin_spacing.w, 0.0f, float3(probe_dims.xyz) - 1.0f);
    uint3 base = uint3(floor(coord));
    float3 fraction = frac(coord);
    float3 sum = 0.0f;
    float weight_sum = 0.0f;

    [unroll] for (uint z = 0u; z < 2u; ++z)
    [unroll] for (uint y = 0u; y < 2u; ++y)
    [unroll] for (uint x = 0u; x < 2u; ++x)
    {
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
    [unroll] for (int x = -1; x <= 1; ++x)
    {
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

    return found ? surface_probe_value(SurfaceProbes[best_index], normal) : float3(0.12f, 0.12f, 0.12f);
}

float static_beam_visibility(float3 position)
{
    if (beam_dims.w == 0u || beam_dims.x == 0u || beam_dims.y == 0u || beam_dims.z == 0u || beam_step.x <= 0.0f || beam_step.y <= 0.0f)
        return 1.0f;

    float3 q = float3(dot(position, shadow_u_min.xyz), dot(position, shadow_v_min.xyz), dot(position, shadow_sun_max.xyz));
    float2 coord = (q.xy - beam_origin.xy) / beam_step.xy - 0.5f;
    int2 base = int2(floor(coord));
    float2 fraction = frac(coord);
    float visibility = 0.0f;
    uint blocker_offset = beam_dims.x * beam_dims.y * beam_dims.z;
    float guard = max(0.002f, min(0.02f, min(beam_step.x, beam_step.y) * 0.01f));

    [unroll] for (uint y = 0u; y < 2u; ++y)
    [unroll] for (uint x = 0u; x < 2u; ++x)
    {
        int2 cell = base + int2(x, y);
        float2 axis_weight = lerp(1.0f - fraction, fraction, float2(x, y));
        float weight = axis_weight.x * axis_weight.y;
        float sample_visibility = 1.0f;
        if (all(cell >= 0) && cell.x < (int)beam_dims.x && cell.y < (int)beam_dims.y) {
            float blocker = SurfaceBeams[blocker_offset + (uint)cell.x + beam_dims.x * (uint)cell.y];
            if (blocker > -1.0e20f) sample_visibility = q.z >= blocker - guard ? 1.0f : 0.0f;
        }
        visibility += weight * sample_visibility;
    }
    return saturate(visibility);
}

float dynamic_shadow_visibility(float3 position)
{
    if (shadow_texel_enabled.z < 0.5f || any(shadow_extent_bias.xyz <= 0.0f)) return 1.0f;
    float sx = dot(position, shadow_u_min.xyz);
    float sy = dot(position, shadow_v_min.xyz);
    float sz = dot(position, shadow_sun_max.xyz);
    float2 uv = (float2(sx, sy) - float2(shadow_u_min.w, shadow_v_min.w)) / shadow_extent_bias.xy;
    float depth = (shadow_sun_max.w - sz) / shadow_extent_bias.z;
    if (any(uv < 0.0f) || any(uv > 1.0f) || depth < 0.0f || depth > 1.0f) return 1.0f;

    float visibility = 0.0f;
    [unroll] for (int y = -1; y <= 1; ++y)
    [unroll] for (int x = -1; x <= 1; ++x)
    {
        float2 sample_uv = saturate(uv + float2((float)x, (float)y) * shadow_texel_enabled.xy);
        float blocker = DynamicShadow.SampleLevel(DynamicShadowSampler, sample_uv, 0.0f);
        visibility += depth <= blocker + shadow_extent_bias.w ? 1.0f : 0.0f;
    }
    return visibility / 9.0f;
}

float3 runtime_indirect(float3 world, float2 uv, float3 baked)
{
    if (dynamic_grid_dims_target.w == 0u || dynamic_grid_origin_cell.w <= 0.0f || any(dynamic_grid_dims_target.xyz == 0u)) return baked;
    int3 cell = int3(floor((world - dynamic_grid_origin_cell.xyz) / dynamic_grid_origin_cell.w));
    if (any(cell < 0) || any(cell >= int3(dynamic_grid_dims_target.xyz))) return baked;
    uint index = (uint)cell.x + dynamic_grid_dims_target.x * ((uint)cell.y + dynamic_grid_dims_target.y * (uint)cell.z);
    uint generation = CellGenerations[index];
    if (generation == 0u || OverlayGenerations[index] != generation) return baked;
    float4 overlay = DynamicOverlay.SampleLevel(IndirectLightmapSampler, uv, 0.0f);
    float confidence_samples = min((float)dynamic_grid_dims_target.w, 4.0f);
    float confidence = saturate(overlay.a / max(confidence_samples, 1.0f));
    return lerp(baked, max(overlay.rgb, 0.0f), confidence);
}

float3 mapped_normal(SurfaceInput input, float scale)
{
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

    float3 sample_normal = NormalMap.Sample(NormalMapSampler, input.uv).xyz * 2.0f - 1.0f;
    sample_normal.xy *= scale;
    return normalize(t * sample_normal.x + b * sample_normal.y + n * sample_normal.z);
}

SurfaceOutput surface_fs(SurfaceInput input, bool front_face : SV_IsFrontFace)
{
    SurfaceOutput output;

    // float3 relative = input.world_position - camera_position.xyz;

    // float error = abs(input.view_depth - dot(relative, camera_forward.xyz));
    // float error = abs(input.view_depth - dot(input.world_position - camera_position.xyz, camera_forward.xyz));
    // float error = abs(input.view_depth - dot(input.world_position - camera_position.xyz, camera_forward.xyz));
    // output.hdr = float4(0, 0, 0, 1);
    // output.normal_depth = float4(normalize(input.view_normal) * 0.5f + 0.5f, max(input.view_depth, 0.0f));
    // output.normal_depth = float4(saturate(error * 2.0f), 0, 0, max(input.view_depth, 0.0f));
    // return output;

    // float error = abs(input.view_depth - input.world_position.z);
    // output.normal_depth = float4(saturate(error * 2.0f), 0, 0, 1.0f);

    // float raster_depth = 1.0f / max(input.position.w, 1.0e-6f);
    // float error = abs(input.view_depth - raster_depth);

    // output.hdr = float4(0, 0, 0, 1);
    // output.normal_depth = float4(camera_forward.xyz, input.view_depth);
    // output.normal_depth = float4(saturate(error * 2.0f), 0, 0, input.view_depth);
    // output.normal_depth = float4(camera_position.xyz, input.view_depth);
    // return output;

    float4 base_sample = BaseColor.Sample(BaseColorSampler, input.uv);
    float3 base = srgb_to_linear(base_sample.rgb) * base_color_factor.rgb;
    // output.hdr = float4(base, 1.0f);
    // output.normal_depth = float4(
    //      normalize(input.view_normal) * 0.5f + 0.5f, 
    //      input.view_depth
    // );
    // return output;

    float4 mr = MetallicRoughness.Sample(MetallicRoughnessSampler, input.uv);
    float metallic = saturate(emissive_metallic.w * mr.b);
    // output.hdr = float4(metallic.xxx, 1.0f);
    // output.normal_depth = float4(
    //      normalize(input.view_normal) * 0.5f + 0.5f, 
    //      input.view_depth
    // );
    // return output;

    float roughness = clamp(roughness_normal_ao_sun.x * mr.g, 0.045f, 1.0f);
    // output.hdr = float4(roughness.xxx, 1.0f);
    // output.normal_depth = float4(
    //      normalize(input.view_normal) * 0.5f + 0.5f, 
    //      input.view_depth
    // );
    // return output;

    float ao_sample = Occlusion.Sample(OcclusionSampler, input.uv).r;
    float material_ao = lerp(1.0f, ao_sample, saturate(roughness_normal_ao_sun.z));
    // output.hdr = float4(material_ao.xxx, 1.0f);
    // output.normal_depth = float4(
    //      normalize(input.view_normal) * 0.5f + 0.5f, 
    //      input.view_depth
    // );
    // return output;

    float3 emissive = srgb_to_linear(Emissive.Sample(EmissiveSampler, input.uv).rgb) *
                      emissive_metallic.rgb;
    // output.hdr = float4(emissive, 1.0f);
    // output.normal_depth = float4(
    //      normalize(input.view_normal) * 0.5f + 0.5f, 
    //      input.view_depth
    // );
    // return output;
    
    float3 n = mapped_normal(input, roughness_normal_ao_sun.y);
    if (!front_face) n = -n;
    // output.hdr = float4(n * 0.5f + 0.5f, 1.0f);
    // output.normal_depth = float4(
    //      normalize(input.view_normal) * 0.5f + 0.5f, 
    //      input.view_depth
    // );
    // return output;
 
    float3 v = normalize(camera_position.xyz - input.world_position);
    float3 l = normalize(sun_direction.xyz);
    float3 h = normalize(v + l);
    float n_dot_v = max(dot(n, v), 0.001f);
    float n_dot_l = max(dot(n, l), 0.0f);
    float n_dot_h = max(dot(n, h), 0.0f);
    float v_dot_h = max(dot(v, h), 0.0f);

    float3 f0 = lerp(float3(0.04f, 0.04f, 0.04f), base, metallic);
    float3 f = fresnel_schlick(v_dot_h, f0);
    float d = distribution_ggx(n_dot_h, roughness);
    float g = geometry_schlick(n_dot_v, roughness) * geometry_schlick(n_dot_l, roughness);
    float3 specular = d * g * f / max(4.0f * n_dot_v * max(n_dot_l, 0.001f), 1.0e-4f);

    float2 baked_uv = front_face ? input.lightmap_uv : input.back_lightmap_uv;
    float static_visibility = static_beam_visibility(input.world_position);
    float moving_visibility = dynamic_shadow_visibility(input.world_position);
    float sun_visibility = static_visibility * moving_visibility;
    float3 base_indirect = camera_position.w > 0.5f ? max(IndirectLightmap.Sample(IndirectLightmapSampler, baked_uv).rgb, 0.0f) : float3(0.12f, 0.12f, 0.12f);
    float3 indirect = object_dynamic != 0u ? surface_probe_irradiance(input.world_position, n) / PI : runtime_indirect(input.world_position, baked_uv, base_indirect);
    float3 direct_sun = object_dynamic != 0u
        ? sun_color.rgb * (roughness_normal_ao_sun.w * n_dot_l * sun_visibility)
        : (camera_position.w > 0.5f ? max(DirectSunLightmap.Sample(DirectSunLightmapSampler, baked_uv).rgb, 0.0f) * moving_visibility : float3(0.0f, 0.0f, 0.0f));
    float3 baked = indirect + direct_sun;
    if (camera_position.w > 1.5f) {
        output.hdr=float4(baked,1.0f);
        output.normal_depth=float4(normalize(input.view_normal) * (front_face ? 0.5f : -0.5f) + 0.5f,
                                    max(input.view_depth, 0.0f));
        return output;
    }
    float baked_luma = dot(baked, float3(0.2126f, 0.7152f, 0.0722f));
    float3 direct_specular = specular * sun_color.rgb * roughness_normal_ao_sun.w *
                             n_dot_l * sun_visibility;
    // output.hdr = float4(direct_specular, 1.0f);
    // output.normal_depth = float4(
    //      normalize(input.view_normal) * 0.5f + 0.5f, 
    //      input.view_depth
    // );
    // return output;

    float3 environment_specular = f0 * (0.025f + 0.10f * (1.0f - roughness)) *
                                  material_ao * (camera_position.w > 0.5f
                                  ? saturate(baked_luma * 2.0f) : 1.0f);
    // output.hdr = float4(environment_specular, 1.0f);
    // output.normal_depth = float4(
    //      normalize(input.view_normal) * 0.5f + 0.5f, 
    //      input.view_depth
    // );
    // return output;

    float3 diffuse = base * baked * material_ao * (1.0f - metallic);
    output.hdr = float4(max(diffuse + direct_specular + environment_specular + emissive, 0.0f),
                        base_sample.a * base_color_factor.a);

    // output.normal_depth = float4(input.world_position, input.view_depth);

    // output.normal_depth = float4(input.world_position - camera_position.xyz,
    //                              input.view_depth);

    output.normal_depth = float4(normalize(input.view_normal) * (front_face ? 0.5f : -0.5f) + 0.5f,
                                 max(input.view_depth, 0.0f));


    return output;
}
#elif defined(BUILD_SKY_FS)
GPU_BIND_B(0, 3) cbuffer SkyData : register(b0, space3)
{
    float4 camera_right;
    float4 camera_up;
    float4 camera_forward;
    float4 sky_zenith;
    float4 sky_horizon;
    float4 sun_direction_intensity;
    float4 sun_color_radius;
};

struct SkyInput
{
    float4 position : SV_Position;
    float2 ndc : TEXCOORD0;
};

struct SkyOutput
{
    float4 hdr : SV_Target0;
    float4 normal_depth : SV_Target1;
};

float3 sky_radiance(float3 direction)
{
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

SkyOutput sky_fs(SkyInput input)
{
    float3 direction = normalize(camera_forward.xyz +
                                 camera_right.xyz * input.ndc.x +
                                 camera_up.xyz * input.ndc.y);
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

GPU_BIND_B(0, 3) cbuffer ComposeData : register(b0, space3)
{
    float exposure;
    float ao_strength;
    float bloom_strength;
    float _pad;
};

struct ComposeInput
{
    float4 position : SV_Position;
    float2 ndc : TEXCOORD0;
};

float3 aces(float3 x)
{
    const float a = 2.51f;
    const float b = 0.03f;
    const float c = 2.43f;
    const float d = 0.59f;
    const float e = 0.14f;
    return saturate((x * (a * x + b)) / (x * (c * x + d) + e));
}

float3 sample_lut(float3 color)
{
    const float size = 16.0f;
    color = saturate(color);
    float blue = color.b * (size - 1.0f);
    float b0 = floor(blue);
    float b1 = min(b0 + 1.0f, size - 1.0f);
    float r = color.r * (size - 1.0f);
    float g = color.g * (size - 1.0f);

    float2 uv0 = float2((b0 * size + r + 0.5f) / (size * size),
                        (g + 0.5f) / size);
    float2 uv1 = float2((b1 * size + r + 0.5f) / (size * size),
                        (g + 0.5f) / size);
    float3 a = Lut.SampleLevel(LutSampler, uv0, 0.0f).rgb;
    float3 b = Lut.SampleLevel(LutSampler, uv1, 0.0f).rgb;
    return lerp(a, b, frac(blue));
}

float4 compose_fs(ComposeInput input) : SV_Target0
{
    uint width, height;
    Hdr.GetDimensions(width, height);
    float2 uv = input.position.xy / float2(width, height);
    

    if (exposure < 0.0f) 
        return float4(saturate(Hdr.SampleLevel(HdrSampler, uv, 0.0f).rgb), 1.0f);


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
struct WireframeInput
{
    float4 position : SV_Position;
    float4 color : TEXCOORD0;
};

float4 wireframe_fs(WireframeInput input) : SV_Target0
{
    return input.color;
}
#endif
