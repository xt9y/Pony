#if !defined(BUILD_SURFACE_FS)
#error BUILD_SURFACE_FS required
#endif

#define surface_fs legacy_surface_fs
#include "fragment.hlsl"
#undef surface_fs

SurfaceOutput surface_fs(SurfaceInput input, bool front_face : SV_IsFrontFace)
{
    SurfaceOutput output;

    float4 base_sample = BaseColor.Sample(BaseColorSampler, input.uv);
    float3 base = srgb_to_linear(base_sample.rgb) * base_color_factor.rgb;
    float4 metallic_roughness = MetallicRoughness.Sample(MetallicRoughnessSampler, input.uv);
    float metallic = saturate(emissive_metallic.w * metallic_roughness.b);
    float roughness = clamp(roughness_normal_ao_sun.x * metallic_roughness.g, 0.045f, 1.0f);
    float ao_sample = Occlusion.Sample(OcclusionSampler, input.uv).r;
    float material_ao = lerp(1.0f, ao_sample, saturate(roughness_normal_ao_sun.z));
    float3 emissive = srgb_to_linear(Emissive.Sample(EmissiveSampler, input.uv).rgb) *
                      emissive_metallic.rgb;

    float3 n = mapped_normal(input, roughness_normal_ao_sun.y);
    if (!front_face) n = -n;

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
    float3 specular = d * g * f /
        max(4.0f * n_dot_v * max(n_dot_l, 0.001f), 1.0e-4f);

    float moving_visibility = dynamic_shadow_visibility(input.world_position);

    // Dynamic objects occupy LPV cells to stop transport through their volume.
    // Sampling exactly on their surface therefore aliases against invalid cells
    // as they move between voxels. Sample just outside the geometric surface so
    // the object reads incident light rather than the blocked interior field.
    float3 probe_position = input.world_position;
    if (object_dynamic != 0u)
    {
        float3 geometric_normal = normalize(input.world_normal);
        if (!front_face) geometric_normal = -geometric_normal;
        probe_position += geometric_normal * max(probe_origin_spacing.w * 0.60f, 0.01f);
    }

    // Emissive geometry is injected into the LPV with software visibility
    // against both static and dynamic geometry. Do not add the analytic emitter
    // path here: its old directional shadow visibility was not emitter visibility
    // and kept the room lit even when a moving object covered the area light.
    float3 indirect = surface_probe_irradiance(probe_position, n) / PI;

    float3 direct_sun = sun_color.rgb *
        (roughness_normal_ao_sun.w * n_dot_l * moving_visibility);

    float3 lighting = indirect + direct_sun;
    if (camera_position.w > 1.5f)
    {
        output.hdr = float4(lighting, 1.0f);
        output.normal_depth = float4(
            normalize(input.view_normal) * (front_face ? 0.5f : -0.5f) + 0.5f,
            max(input.view_depth, 0.0f));
        return output;
    }

    float lighting_luma = dot(lighting, float3(0.2126f, 0.7152f, 0.0722f));
    float3 direct_specular = specular * sun_color.rgb * roughness_normal_ao_sun.w *
                             n_dot_l * moving_visibility;
    float3 environment_specular = f0 * (0.025f + 0.10f * (1.0f - roughness)) *
        material_ao * saturate(lighting_luma * 2.0f);

    float3 diffuse = base * lighting * material_ao * (1.0f - metallic);
    output.hdr = float4(max(diffuse + direct_specular + environment_specular + emissive, 0.0f),
                        base_sample.a * base_color_factor.a);
    output.normal_depth = float4(
        normalize(input.view_normal) * (front_face ? 0.5f : -0.5f) + 0.5f,
        max(input.view_depth, 0.0f));
    return output;
}
