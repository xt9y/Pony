#ifndef GPU_BIND_S
#define GPU_BIND_S(n,s) [[vk::binding(n, s)]]
#define GPU_BIND_T(n,s) [[vk::binding(n + 16, s)]]
#define GPU_BIND_B(n,s) [[vk::binding(n + 32, s)]]
#define GPU_BIND_U(n,s) [[vk::binding(n + 48, s)]]
#define GPU_STORAGE_RGBA16F [[vk::image_format("rgba16f")]]
#endif

#if defined(BUILD_VISION_VOLUME_CS)

/* Reuse Pony's current volumetric helpers and resource layout verbatim,
 * but replace only the entry point.  This keeps the beam/probe integration
 * logic in one place while allowing the expensive camera rays to be sampled
 * at a lower spatial rate in peripheral vision. */
#define BUILD_VOLUME_CS
#define volume_cs pony_dense_volume_cs
#include "compute.hlsl"
#undef volume_cs
#undef BUILD_VOLUME_CS

float vision_eccentricity(float2 uv)
{
    return saturate(length(uv * 2.0f - 1.0f));
}

float vision_jitter(uint2 pixel_id)
{
    uint h = pixel_id.x * 0x8da6b343u + pixel_id.y * 0xd8163841u;
    h ^= h >> 16u;
    h *= 0x7feb352du;
    h ^= h >> 15u;
    h *= 0x846ca68bu;
    h ^= h >> 16u;
    return (float)(h & 0x00ffffffu) / 16777216.0f;
}

float vision_transition(float eccentricity, float radius, float width)
{
    if (width <= 1.0e-5f)
        return eccentricity < radius ? 0.0f : 1.0f;
    float half_width = 0.5f * width;
    return smoothstep(radius - half_width, radius + half_width, eccentricity);
}

float3 vision_quality_weights(float eccentricity)
{
    float center_to_middle = vision_transition(eccentricity, volume_radii.x, volume_radii.z);
    float middle_to_peripheral = vision_transition(eccentricity, volume_radii.y, volume_radii.w);
    float3 weights = float3(1.0f - center_to_middle,
                            center_to_middle * (1.0f - middle_to_peripheral),
                            middle_to_peripheral);
    return weights / max(weights.x + weights.y + weights.z, 1.0e-5f);
}

uint vision_stride(float eccentricity)
{
    float3 quality = vision_quality_weights(eccentricity);
    uint stride = 0xffffffffu;
    if (quality.x > 1.0e-4f) stride = min(stride, max(volume_strides.x, 1u));
    if (quality.y > 1.0e-4f) stride = min(stride, max(volume_strides.y, 1u));
    if (quality.z > 1.0e-4f) stride = min(stride, max(volume_strides.z, 1u));
    return stride == 0xffffffffu ? 1u : stride;
}

float3 integrate_probe_quality(uint probe_steps, uint2 pixel_id, float3 direction,
                               float enter, float leave,
                               float3 surface_position, float3 surface_normal,
                               bool has_surface)
{
    probe_steps = max(probe_steps, 1u);
    float step_size = (leave - enter) / float(probe_steps);
    float probe_transmission = exp(-volume_params.x * step_size);
    float probe_remaining = 1.0f;
    float3 sum = 0.0f;
    float jitter = vision_jitter(pixel_id);
    float jitter_strength = max(volume_filter.x, 0.0f);

    [loop] for (uint i = 0u; i < probe_steps; ++i) {
        float t = enter + (float(i) + 0.5f + (jitter - 0.5f) * jitter_strength) * step_size;
        t = clamp(t, enter + 1.0e-5f, leave - 1.0e-5f);
        float integral = probe_remaining * (1.0f - probe_transmission);
        sum += volume_radiance(eye_density.xyz + direction * t, direction,
                               surface_position, surface_normal, has_surface) *
               integral * volume_params.z;
        probe_remaining *= probe_transmission;
    }

    return sum;
}

[numthreads(8, 8, 1)]
void volume_cs(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= grid_dims_width.w || id.y >= height_debug.x) return;

    float2 uv = (float2(id.xy) + 0.5f) /
        float2(grid_dims_width.w, height_debug.x);
    float eccentricity = vision_eccentricity(uv);
    uint stride = vision_stride(eccentricity);

    /* Every thread still writes a deterministic value, but only lattice
     * anchors perform the expensive beam traversal.  Alpha < 0 marks holes;
     * the compose pass reconstructs them from traced anchors. */
    if ((id.x % stride) != 0u || (id.y % stride) != 0u) {
        Output[id.xy] = float4(0.0f, 0.0f, 0.0f, -1.0f);
        return;
    }

    float2 ndc = uv * 2.0f - 1.0f;
    float3 direction = normalize(forward_g.xyz + right_tan.xyz * ndc.x -
                                  up_tan.xyz * ndc.y);
    float3 minimum = grid_origin_spacing.xyz;
    float3 maximum = minimum + grid_origin_spacing.w *
        float3(grid_dims_width.xyz - 1u);
    float enter = 0.0f;
    float leave = volume_params.w;

    [unroll] for (uint axis = 0; axis < 3u; ++axis) {
        if (abs(direction[axis]) < 1.0e-6f) {
            if (eye_density[axis] < minimum[axis] ||
                eye_density[axis] > maximum[axis]) {
                Output[id.xy] = float4(0, 0, 0, 1);
                return;
            }
        } else {
            float a = (minimum[axis] - eye_density[axis]) / direction[axis];
            float b = (maximum[axis] - eye_density[axis]) / direction[axis];
            enter = max(enter, min(a, b));
            leave = min(leave, max(a, b));
        }
    }

    float4 surface = NormalDepth.SampleLevel(DepthSampler, uv, 0.0f);
    float depth = surface.w;
    if (depth > 0.0f)
        leave = min(leave, depth / max(dot(direction, forward_g.xyz), 0.01f));
    if (leave <= enter) {
        Output[id.xy] = float4(0, 0, 0, 1);
        return;
    }

    float3 surface_position = eye_density.xyz + direction *
        (depth / max(dot(direction, forward_g.xyz), 0.01f));
    float3 view_normal = normalize(surface.xyz * 2.0f - 1.0f);
    float3 surface_normal = normalize(normalize(right_tan.xyz) * view_normal.x +
                                      normalize(up_tan.xyz) * view_normal.y -
                                      forward_g.xyz * view_normal.z);
    if (dot(eye_density.xyz - surface_position, surface_normal) < 0.0f)
        surface_normal = -surface_normal;

    float3 quality = vision_quality_weights(eccentricity);
    float3 sum = 0.0f;
    if (quality.x > 1.0e-4f)
        sum += quality.x * integrate_probe_quality(volume_quality.x, id.xy, direction, enter, leave,
                                                   surface_position, surface_normal, depth > 0.0f);
    if (quality.y > 1.0e-4f)
        sum += quality.y * integrate_probe_quality(volume_quality.y, id.xy, direction, enter, leave,
                                                   surface_position, surface_normal, depth > 0.0f);
    if (quality.z > 1.0e-4f)
        sum += quality.z * integrate_probe_quality(volume_quality.z, id.xy, direction, enter, leave,
                                                   surface_position, surface_normal, depth > 0.0f);
    float sun_fraction = 0.0f;

    float g = volume_params.y;
    float cosine = dot(direction, normalize(sun_intensity.xyz));
    float hg = (1.0f - g * g) / (12.5663706144f *
        pow(max(1.0f + g * g - 2.0f * g * cosine, 0.001f), 1.5f));
    float sun_integral = integrate_sun_grid(eye_density.xyz, direction, enter, leave,
                                            volume_params.x, sun_fraction);
    sum += sun_integral * sun_intensity.w * hg * sun_color.rgb;
    Output[id.xy] = float4(sum, exp(-volume_params.x * (leave - enter)));
}

#elif defined(BUILD_VISION_COMPOSE_CS)

GPU_BIND_T(0, 0) Texture2D<float4> Hdr : register(t0, space0);
GPU_BIND_S(0, 0) SamplerState HdrSampler : register(s0, space0);
GPU_BIND_T(1, 0) Texture2D<float4> Volume : register(t1, space0);
GPU_BIND_S(1, 0) SamplerState VolumeSampler : register(s1, space0);
GPU_BIND_T(2, 0) Texture2D<float4> NormalDepth : register(t2, space0);
GPU_BIND_S(2, 0) SamplerState DepthSampler : register(s2, space0);
GPU_BIND_U(0, 1) GPU_STORAGE_RGBA16F RWTexture2D<float4> Output : register(u0, space1);

GPU_BIND_B(0, 2) cbuffer VolumeComposeData : register(b0, space2)
{
    uint width;
    uint height;
    uint debug_view;
    uint bypass_volume;
    float4 volume_radii;
    float4 volume_filter;
    uint4 volume_strides;
};

float vision_eccentricity(float2 uv)
{
    return saturate(length(uv * 2.0f - 1.0f));
}

float vision_transition(float eccentricity, float radius, float width)
{
    if (width <= 1.0e-5f)
        return eccentricity < radius ? 0.0f : 1.0f;
    float half_width = 0.5f * width;
    return smoothstep(radius - half_width, radius + half_width, eccentricity);
}

float3 vision_quality_weights(float eccentricity)
{
    float center_to_middle = vision_transition(eccentricity, volume_radii.x, volume_radii.z);
    float middle_to_peripheral = vision_transition(eccentricity, volume_radii.y, volume_radii.w);
    float3 weights = float3(1.0f - center_to_middle,
                            center_to_middle * (1.0f - middle_to_peripheral),
                            middle_to_peripheral);
    return weights / max(weights.x + weights.y + weights.z, 1.0e-5f);
}

uint vision_stride(float eccentricity)
{
    float3 quality = vision_quality_weights(eccentricity);
    uint stride = 0xffffffffu;
    if (quality.x > 1.0e-4f) stride = min(stride, max(volume_strides.x, 1u));
    if (quality.y > 1.0e-4f) stride = min(stride, max(volume_strides.y, 1u));
    if (quality.z > 1.0e-4f) stride = min(stride, max(volume_strides.z, 1u));
    return stride == 0xffffffffu ? 1u : stride;
}

float depth_similarity(float center_depth, float sample_depth)
{
    if (center_depth <= 0.0f || sample_depth <= 0.0f)
        return (center_depth <= 0.0f && sample_depth <= 0.0f) ? 1.0f : 0.0f;
    return exp(-abs(center_depth - sample_depth) /
               max(0.025f, center_depth * 0.025f));
}

float4 reconstructed_volume_stride(float2 uv, float center_depth, uint stride)
{
    uint volume_width, volume_height;
    Volume.GetDimensions(volume_width, volume_height);
    stride = max(stride, 1u);

    float2 volume_size = float2(volume_width, volume_height);
    float2 position = uv * volume_size - 0.5f;
    float2 coarse = position / float(stride);
    int2 base = int2(floor(coarse)) * int(stride);
    float2 fraction = frac(coarse);

    int2 max_anchor = int2(
        int(((volume_width - 1u) / stride) * stride),
        int(((volume_height - 1u) / stride) * stride));

    float4 fog = 0.0f;
    float weight_sum = 0.0f;
    float best_difference = 1.0e30f;
    float4 closest_fog = float4(0, 0, 0, 1);

    [unroll] for (uint y = 0u; y < 2u; ++y)
    [unroll] for (uint x = 0u; x < 2u; ++x) {
        int2 pixel = clamp(base + int2(x, y) * int(stride),
                           int2(0, 0), max_anchor);
        float2 sample_uv = (float2(pixel) + 0.5f) / volume_size;
        float sample_depth = NormalDepth.SampleLevel(DepthSampler, sample_uv, 0.0f).w;
        float2 axis_weight = lerp(1.0f - fraction, fraction, float2(x, y));
        float weight = axis_weight.x * axis_weight.y *
                       depth_similarity(center_depth, sample_depth);
        float4 sample_fog = Volume.SampleLevel(VolumeSampler, sample_uv, 0.0f);

        if (sample_fog.a < 0.0f) continue;

        float difference = (center_depth <= 0.0f) == (sample_depth <= 0.0f)
            ? abs(center_depth - sample_depth) : 1.0e20f;
        if (difference < best_difference) {
            best_difference = difference;
            closest_fog = sample_fog;
        }

        fog += sample_fog * weight;
        weight_sum += weight;
    }

    return weight_sum > 1.0e-5f ? fog / weight_sum : closest_fog;
}

float4 reconstructed_volume(float2 uv, float center_depth)
{
    float3 quality = vision_quality_weights(vision_eccentricity(uv));
    float4 fog = 0.0f;
    if (quality.x > 1.0e-4f)
        fog += quality.x * reconstructed_volume_stride(uv, center_depth, volume_strides.x);
    if (quality.y > 1.0e-4f)
        fog += quality.y * reconstructed_volume_stride(uv, center_depth, volume_strides.y);
    if (quality.z > 1.0e-4f)
        fog += quality.z * reconstructed_volume_stride(uv, center_depth, volume_strides.z);
    return fog;
}

float4 edge_aware_volume_blur(float2 uv, float center_depth, float4 center_fog)
{
    float blur_strength = max(volume_filter.y, 0.0f);
    if (blur_strength <= 1.0e-5f) return center_fog;

    int2 offsets[5] = {
        int2(0, 0), int2(1, 0), int2(-1, 0), int2(0, 1), int2(0, -1)
    };
    float4 sum = center_fog * 1.5f;
    float weight_sum = 1.5f;
    float2 texel = 1.0f / float2(width, height);

    [unroll] for (uint i = 1u; i < 5u; ++i) {
        float2 sample_uv = saturate(uv + float2(offsets[i]) * texel);
        float sample_depth = NormalDepth.SampleLevel(DepthSampler, sample_uv, 0.0f).w;
        float weight = blur_strength * depth_similarity(center_depth, sample_depth);
        if (weight <= 1.0e-4f) continue;
        float4 sample_fog = reconstructed_volume(sample_uv, sample_depth);
        sum += sample_fog * weight;
        weight_sum += weight;
    }

    return sum / max(weight_sum, 1.0e-5f);
}

float3 vision_hdr(float2 uv, float center_depth, float eccentricity)
{
    float3 center = Hdr.SampleLevel(HdrSampler, uv, 0.0f).rgb;
    if (debug_view >= 3u) return center;

    float focus_depth = NormalDepth.SampleLevel(
        DepthSampler, float2(0.5f, 0.5f), 0.0f).w;
    if (focus_depth <= 0.0f) focus_depth = 10000.0f;

    float effective_depth = center_depth > 0.0f ? center_depth : 10000.0f;
    float relative_defocus = abs(effective_depth - focus_depth) /
                             max(effective_depth, 0.05f);
    float dof_radius = min(7.0f, relative_defocus * 4.5f);
    float peripheral_radius = smoothstep(0.48f, 0.95f, eccentricity) * 3.25f;
    float radius = max(dof_radius, peripheral_radius);

    if (radius < 0.55f) return center;

    float dof_mix = saturate(dof_radius / 7.0f);
    float3 sum = center * 1.5f;
    float weight_sum = 1.5f;

    [unroll] for (uint i = 0u; i < 6u; ++i) {
        float angle = (float(i) + 0.5f) * 1.0471975512f;
        float ring = (i & 1u) != 0u ? 1.0f : 0.62f;
        float2 offset = float2(cos(angle), sin(angle)) * radius * ring /
                        float2(width, height);
        float2 sample_uv = saturate(uv + offset);
        float sample_depth = NormalDepth.SampleLevel(DepthSampler, sample_uv, 0.0f).w;

        float depth_guard;
        if (center_depth <= 0.0f || sample_depth <= 0.0f) {
            depth_guard = (center_depth <= 0.0f && sample_depth <= 0.0f) ? 1.0f : 0.08f;
        } else {
            float relative_delta = abs(center_depth - sample_depth) /
                                   max(max(center_depth, sample_depth), 0.05f);
            depth_guard = exp(-relative_delta * 14.0f);
        }

        /* Peripheral acuity blur stays edge-aware.  True depth defocus is
         * allowed to mix more aggressively across depth discontinuities. */
        float weight = lerp(depth_guard, 1.0f, dof_mix * 0.65f);
        sum += Hdr.SampleLevel(HdrSampler, sample_uv, 0.0f).rgb * weight;
        weight_sum += weight;
    }

    return sum / max(weight_sum, 1.0e-4f);
}

[numthreads(8, 8, 1)]
void volume_compose_cs(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= width || id.y >= height) return;

    float2 uv = (float2(id.xy) + 0.5f) / float2(width, height);
    float center_depth = NormalDepth.SampleLevel(DepthSampler, uv, 0.0f).w;
    float eccentricity = vision_eccentricity(uv);
    float3 source = vision_hdr(uv, center_depth, eccentricity);

    float4 fog = bypass_volume != 0u
        ? float4(0.0f, 0.0f, 0.0f, 1.0f)
        : reconstructed_volume(uv, center_depth);
    if (bypass_volume == 0u)
        fog = edge_aware_volume_blur(uv, center_depth, fog);

    if (debug_view >= 3u && bypass_volume == 0u) {
        Output[id.xy] = float4(fog.rgb, 1.0f);
        return;
    }

    Output[id.xy] = float4(source * fog.a + fog.rgb, 1.0f);
}

#endif
