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
GPU_BIND_T(5, 2) Texture2D<float4> Lightmap : register(t5, space2);
GPU_BIND_S(5, 2) SamplerState LightmapSampler : register(s5, space2);

GPU_BIND_B(0, 3) cbuffer MaterialData : register(b0, space3)
{
    float4 base_color_factor;
    float4 emissive_metallic;
    float4 roughness_normal_ao_sun;
    float4 sun_direction;
    float4 sun_color;
    float4 camera_position;
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

    float3 baked = camera_position.w > 0.5f
        ? max(Lightmap.Sample(LightmapSampler,
                              front_face ? input.lightmap_uv : input.back_lightmap_uv).rgb, 0.0f)
        : float3(0.12f, 0.12f, 0.12f);
    if (camera_position.w > 1.5f) {
        output.hdr=float4(baked,1.0f);
        output.normal_depth=float4(normalize(input.view_normal) * (front_face ? 0.5f : -0.5f) + 0.5f,
                                    max(input.view_depth, 0.0f));
        return output;
    }
    float baked_luma = dot(baked, float3(0.2126f, 0.7152f, 0.0722f));
    float sun_visibility = camera_position.w > 0.5f
        ? saturate(baked_luma * 0.55f) : 1.0f;
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
