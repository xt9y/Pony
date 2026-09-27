#if defined(BUILD_VOLUME_CS)
GPU_BIND_T(0, 0) Texture2D<float4> NormalDepth : register(t0, space0);
GPU_BIND_S(0, 0) SamplerState DepthSampler : register(s0, space0);
struct VolumeProbe { float4 position; float4 coefficient[9]; };
GPU_BIND_T(1, 0) StructuredBuffer<VolumeProbe> VolumeProbes : register(t1, space0);
GPU_BIND_T(2, 0) StructuredBuffer<float> SunBeams : register(t2, space0);
GPU_BIND_U(0, 1) GPU_STORAGE_RGBA16F RWTexture2D<float4> Output : register(u0, space1);
GPU_BIND_B(0, 2) cbuffer VolumeData : register(b0, space2)
{
    float4 eye_density;
    float4 right_tan;
    float4 up_tan;
    float4 forward_g;
    float4 sun_intensity;
    float4 grid_origin_spacing;
    uint4 grid_dims_width;
    uint4 height_debug;
    float4 beam_origin;
    float4 beam_step;
};

float integrate_beam_interval(float3 ray_origin, float3 ray_direction,
                              float t0, float t1, float medium_enter,
                              float density, int layer,
                              out float lit_distance)
{
    lit_distance = 0.0f;
    if (layer < 0 || layer >= 16 || t1 <= t0)
        return 0.0f;

    float midpoint = 0.5f * (t0 + t1);
    float3 sun_position = ray_origin + ray_direction * midpoint;
    float2 cell = (sun_position.xy - beam_origin.xy) / beam_step.xy - 0.5f;
    int2 base = int2(floor(cell));
    float2 fraction = frac(cell);

    // Preserve the previous weak-visibility suppression at beam edges, but
    // compute it without the roof blocker. The blocker is integrated exactly
    // below instead of being sampled once at the cell midpoint.
    float raw_visibility = 0.0f;
    [unroll] for (uint y = 0; y < 2u; ++y)
    [unroll] for (uint x = 0; x < 2u; ++x)
    {
        int2 index = base + int2(x, y);
        if (any(index < 0) || index.x >= 64 || index.y >= 64)
            continue;

        float2 axis_weight = lerp(1.0f - fraction, fraction, float2(x, y));
        raw_visibility += SunBeams[index.x + 64 * (index.y + 64 * layer)] *
                          axis_weight.x * axis_weight.y;
    }

    float visibility_shape = smoothstep(0.35f, 0.85f, raw_visibility);
    if (visibility_shape <= 0.0f)
        return 0.0f;

    float integrated = 0.0f;
    [unroll] for (uint y = 0; y < 2u; ++y)
    [unroll] for (uint x = 0; x < 2u; ++x)
    {
        int2 index = base + int2(x, y);
        if (any(index < 0) || index.x >= 64 || index.y >= 64)
            continue;

        float2 axis_weight = lerp(1.0f - fraction, fraction, float2(x, y));
        float visibility = SunBeams[index.x + 64 * (index.y + 64 * layer)] *
                           axis_weight.x * axis_weight.y * visibility_shape;
        if (visibility <= 0.0f)
            continue;

        float a = t0;
        float b = t1;
        float blocker = SunBeams[64 * 64 * 16 + index.x + 64 * index.y];

        // sun-space +Z points toward the sun. For each bilinear beam column,
        // split this camera-ray interval exactly where it crosses the first
        // sun-facing surface. This prevents an entire coarse Z cell from being
        // treated as lit merely because its midpoint lies above the roof.
        if (blocker >= -1.0e20f)
        {
            if (abs(ray_direction.z) < 1.0e-7f)
            {
                if (sun_position.z < blocker)
                    continue;
            }
            else
            {
                float blocker_t = (blocker - ray_origin.z) / ray_direction.z;
                if (ray_direction.z > 0.0f)
                    a = max(a, blocker_t);
                else
                    b = min(b, blocker_t);
            }
        }

        a = max(a, t0);
        b = min(b, t1);
        if (b <= a)
            continue;

        float transmission_a = exp(-density * max(a - medium_enter, 0.0f));
        float transmission_b = exp(-density * max(b - medium_enter, 0.0f));
        integrated += visibility * (transmission_a - transmission_b);
        lit_distance += visibility * (b - a);
    }

    return integrated;
}

float integrate_sun_grid(float3 world_origin, float3 world_direction,
                         float medium_enter, float medium_leave, float density,
                         out float lit_fraction)
{
    lit_fraction = 0.0f;

    float3 sun = normalize(sun_intensity.xyz);
    float3 u = normalize(cross(float3(0, 1, 0), sun));
    float3 v = cross(sun, u);
    float3 ray_origin = float3(dot(world_origin, u), dot(world_origin, v), dot(world_origin, sun));
    float3 ray_direction = float3(dot(world_direction, u), dot(world_direction, v), dot(world_direction, sun));

    float3 bounds_min = beam_origin.xyz;
    float3 bounds_max = bounds_min + beam_step.xyz * float3(64.0f, 64.0f, 16.0f);
    float grid_enter = medium_enter;
    float grid_leave = medium_leave;

    [unroll] for (uint axis = 0; axis < 3u; ++axis)
    {
        if (abs(ray_direction[axis]) < 1.0e-7f)
        {
            if (ray_origin[axis] < bounds_min[axis] || ray_origin[axis] > bounds_max[axis])
                return 0.0f;
        }
        else
        {
            float a = (bounds_min[axis] - ray_origin[axis]) / ray_direction[axis];
            float b = (bounds_max[axis] - ray_origin[axis]) / ray_direction[axis];
            grid_enter = max(grid_enter, min(a, b));
            grid_leave = min(grid_leave, max(a, b));
        }
    }

    if (grid_leave <= grid_enter)
        return 0.0f;

    float epsilon = max(1.0e-5f, min(beam_step.x, min(beam_step.y, beam_step.z)) * 1.0e-4f);
    float inside_t = min(grid_leave, grid_enter + epsilon);
    float3 inside = ray_origin + ray_direction * inside_t;
    int3 cell = int3(floor((inside - bounds_min) / beam_step.xyz));
    cell = clamp(cell, int3(0, 0, 0), int3(63, 63, 15));

    int3 step_direction = int3(
        ray_direction.x > 1.0e-7f ? 1 : (ray_direction.x < -1.0e-7f ? -1 : 0),
        ray_direction.y > 1.0e-7f ? 1 : (ray_direction.y < -1.0e-7f ? -1 : 0),
        ray_direction.z > 1.0e-7f ? 1 : (ray_direction.z < -1.0e-7f ? -1 : 0));

    const float huge = 1.0e30f;
    float3 t_delta = float3(
        step_direction.x == 0 ? huge : beam_step.x / abs(ray_direction.x),
        step_direction.y == 0 ? huge : beam_step.y / abs(ray_direction.y),
        step_direction.z == 0 ? huge : beam_step.z / abs(ray_direction.z));

    float3 next_boundary = float3(
        bounds_min.x + (step_direction.x > 0 ? float(cell.x + 1) : float(cell.x)) * beam_step.x,
        bounds_min.y + (step_direction.y > 0 ? float(cell.y + 1) : float(cell.y)) * beam_step.y,
        bounds_min.z + (step_direction.z > 0 ? float(cell.z + 1) : float(cell.z)) * beam_step.z);
    float3 t_max = float3(
        step_direction.x == 0 ? huge : (next_boundary.x - ray_origin.x) / ray_direction.x,
        step_direction.y == 0 ? huge : (next_boundary.y - ray_origin.y) / ray_direction.y,
        step_direction.z == 0 ? huge : (next_boundary.z - ray_origin.z) / ray_direction.z);

    float integrated = 0.0f;
    float lit_distance = 0.0f;
    float t = grid_enter;

    [loop] for (uint step_index = 0; step_index < 160u; ++step_index)
    {
        if (any(cell < 0) || cell.x >= 64 || cell.y >= 64 || cell.z >= 16)
            break;

        float t_next = min(grid_leave, min(t_max.x, min(t_max.y, t_max.z)));
        t_next = max(t_next, t);

        if (t_next > t + 1.0e-7f)
        {
            float interval_lit_distance = 0.0f;
            integrated += integrate_beam_interval(ray_origin, ray_direction,
                                                  t, t_next, medium_enter,
                                                  density, cell.z,
                                                  interval_lit_distance);
            lit_distance += interval_lit_distance;
        }

        if (t_next >= grid_leave - 1.0e-7f)
            break;

        float boundary_epsilon = 1.0e-6f;
        if (t_max.x <= t_next + boundary_epsilon)
        {
            cell.x += step_direction.x;
            t_max.x += t_delta.x;
        }
        if (t_max.y <= t_next + boundary_epsilon)
        {
            cell.y += step_direction.y;
            t_max.y += t_delta.y;
        }
        if (t_max.z <= t_next + boundary_epsilon)
        {
            cell.z += step_direction.z;
            t_max.z += t_delta.z;
        }
        t = t_next;
    }

    lit_fraction = lit_distance / max(medium_leave - medium_enter, 1.0e-6f);
    return integrated;
}

float3 volume_radiance(float3 p)
{
    float3 coord = clamp((p - grid_origin_spacing.xyz) / grid_origin_spacing.w,
                         0.0f, float3(grid_dims_width.xyz) - 1.0f);
    uint3 base = uint3(floor(coord));
    float3 fraction = frac(coord);
    float3 radiance = 0.0f;
    [unroll] for (uint z = 0; z < 2u; ++z)
    [unroll] for (uint y = 0; y < 2u; ++y)
    [unroll] for (uint x = 0; x < 2u; ++x)
    {
        uint3 cell = min(base + uint3(x,y,z), grid_dims_width.xyz - 1u);
        float3 w = lerp(1.0f - fraction, fraction, float3(x,y,z));
        float weight = w.x*w.y*w.z;
        VolumeProbe probe = VolumeProbes[cell.x + grid_dims_width.x *
            (cell.y + grid_dims_width.y * cell.z)];
        weight *= probe.position.w;
        float3 indirect = probe.coefficient[0].rgb * 0.2820947918f;
        radiance += max(indirect, 0.0f) * weight;
    }
    return radiance;
}

[numthreads(8, 8, 1)]
void volume_cs(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= grid_dims_width.w || id.y >= height_debug.x) return;
    float2 uv = (float2(id.xy) + 0.5f) /
        float2(grid_dims_width.w, height_debug.x);
    float2 ndc = uv * 2.0f - 1.0f;
    float3 direction = normalize(forward_g.xyz + right_tan.xyz * ndc.x -
                                  up_tan.xyz * ndc.y);
    float3 minimum = grid_origin_spacing.xyz;
    float3 maximum = minimum + grid_origin_spacing.w *
        float3(grid_dims_width.xyz - 1u);
    float enter = 0.0f, leave = 10000.0f;
    [unroll] for (uint axis = 0; axis < 3u; ++axis) {
        if (abs(direction[axis]) < 1.0e-6f) {
            if (eye_density[axis] < minimum[axis] ||
                eye_density[axis] > maximum[axis]) { Output[id.xy] = float4(0,0,0,1); return; }
        } else {
            float a = (minimum[axis] - eye_density[axis]) / direction[axis];
            float b = (maximum[axis] - eye_density[axis]) / direction[axis];
            enter = max(enter, min(a,b));
            leave = min(leave, max(a,b));
        }
    }
    float depth = NormalDepth.SampleLevel(DepthSampler, uv, 0.0f).w;
    if (depth > 0.0f) leave = min(leave, depth / max(dot(direction,forward_g.xyz),0.01f));
    if (leave <= enter) { Output[id.xy] = float4(0,0,0,1); return; }
    float step_size = (leave-enter) * 0.25f;
    float3 sum = 0.0f;
    float sun_fraction = 0.0f;
    float probe_transmission = exp(-eye_density.w * step_size);
    float probe_remaining = 1.0f;
    [unroll] for (uint i = 0; i < 4u; ++i) {
        float t = enter + (float(i) + 0.5f) * step_size;
        float integral = probe_remaining * (1.0f - probe_transmission);
        sum += volume_radiance(eye_density.xyz + direction * t) * integral * 0.15f;
        probe_remaining *= probe_transmission;
    }
    float g = forward_g.w;
    float cosine = dot(direction, normalize(sun_intensity.xyz));
    float hg = (1.0f - g*g) / (12.5663706144f *
        pow(max(1.0f + g*g - 2.0f*g*cosine, 0.001f), 1.5f));
    float sun_integral = integrate_sun_grid(eye_density.xyz, direction, enter, leave,
                                            eye_density.w, sun_fraction);
    sum += sun_integral * sun_intensity.w * hg * float3(1.0f, 0.94f, 0.84f);
    if (height_debug.y == 3u) {
        Output[id.xy] = float4(sun_fraction, sun_fraction, sun_fraction, 1.0f);
        return;
    }
    if (height_debug.y == 4u) {
        Output[id.xy] = float4(sum, 1.0f);
        return;
    }
    Output[id.xy] = float4(sum, exp(-eye_density.w * (leave-enter)));
}
#elif defined(BUILD_VOLUME_COMPOSE_CS)
GPU_BIND_T(0, 0) Texture2D<float4> Hdr : register(t0, space0);
GPU_BIND_S(0, 0) SamplerState HdrSampler : register(s0, space0);
GPU_BIND_T(1, 0) Texture2D<float4> Volume : register(t1, space0);
GPU_BIND_S(1, 0) SamplerState VolumeSampler : register(s1, space0);
GPU_BIND_T(2, 0) Texture2D<float4> NormalDepth : register(t2, space0);
GPU_BIND_S(2, 0) SamplerState DepthSampler : register(s2, space0);
GPU_BIND_U(0, 1) GPU_STORAGE_RGBA16F RWTexture2D<float4> Output : register(u0, space1);
GPU_BIND_B(0, 2) cbuffer VolumeComposeData : register(b0, space2) {
    uint width; uint height; uint debug_view; uint _pad1;
};
[numthreads(8,8,1)]
void volume_compose_cs(uint3 id : SV_DispatchThreadID) {
    if (id.x >= width || id.y >= height) return;
    float2 uv = (float2(id.xy)+0.5f)/float2(width,height);
    float4 source = Hdr.SampleLevel(HdrSampler, uv, 0.0f);
    uint volume_width, volume_height;
    Volume.GetDimensions(volume_width,volume_height);
    float2 volume_size=float2(volume_width,volume_height);
    float2 position=uv*volume_size-0.5f;
    int2 base=int2(floor(position));
    float2 fraction=frac(position);
    float center_depth=NormalDepth.SampleLevel(DepthSampler,uv,0.0f).w;
    float4 fog=0.0f;
    float weight_sum=0.0f;
    float best_difference=1.0e30f;
    float4 closest_fog=float4(0,0,0,1);
    [unroll] for (uint y=0; y<2u; ++y)
    [unroll] for (uint x=0; x<2u; ++x) {
        int2 pixel=clamp(base+int2(x,y),int2(0,0),int2(volume_width-1,volume_height-1));
        float2 sample_uv=(float2(pixel)+0.5f)/volume_size;
        float sample_depth=NormalDepth.SampleLevel(DepthSampler,sample_uv,0.0f).w;
        float2 axis_weight=lerp(1.0f-fraction,fraction,float2(x,y));
        float weight=axis_weight.x*axis_weight.y;
        if (center_depth<=0.0f || sample_depth<=0.0f)
            weight*=center_depth<=0.0f && sample_depth<=0.0f ? 1.0f : 0.0f;
        else
            weight*=exp(-abs(center_depth-sample_depth)/max(0.02f,center_depth*0.02f));
        float4 sample_fog=Volume.SampleLevel(VolumeSampler,sample_uv,0.0f);
        float difference=(center_depth<=0.0f)==(sample_depth<=0.0f) ?
            abs(center_depth-sample_depth) : 1.0e20f;
        if (difference<best_difference) {
            best_difference=difference;
            closest_fog=sample_fog;
        }
        fog+=sample_fog*weight;
        weight_sum+=weight;
    }
    fog=weight_sum>1.0e-5f ? fog/weight_sum : closest_fog;
    Output[id.xy] = debug_view >= 3u ? float4(fog.rgb,1.0f) :
        float4(source.rgb*fog.a + fog.rgb, source.a);
}
#elif defined(BUILD_SSAO_CS)
GPU_BIND_T(0, 0) Texture2D<float4> NormalDepth : register(t0, space0);
GPU_BIND_S(0, 0) SamplerState NormalDepthSampler : register(s0, space0);
GPU_BIND_U(0, 1) GPU_STORAGE_RGBA16F RWTexture2D<float4> Output : register(u0, space1);

GPU_BIND_B(0, 2) cbuffer SsaoData : register(b0, space2)
{
    uint width;
    uint height;
    uint ao_width;
    uint ao_height;
    float tan_half_fov;
    float aspect;
    float radius;
    float bias;
};

float hash12(float2 p)
{
    float3 p3 = frac(float3(p.xyx) * 0.1031f);
    p3 += dot(p3, p3.yzx + 33.33f);
    return frac((p3.x + p3.y) * p3.z);
}

[numthreads(8, 8, 1)]
void ssao_cs(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= ao_width || id.y >= ao_height) return;

    float2 uv = (float2(id.xy) + 0.5f) / float2(ao_width, ao_height);
    float4 center = NormalDepth.SampleLevel(NormalDepthSampler, uv, 0.0f);
    float depth = center.w;
    if (depth <= 0.0f) {
        Output[id.xy] = float4(1.0f, 1.0f, 1.0f, 1.0f);
        return;
    }

    float3 normal = normalize(center.xyz * 2.0f - 1.0f);
    float pixel_radius = radius / max(depth * tan_half_fov, 1.0e-4f) *
                         (float(height) * 0.5f);
    pixel_radius = clamp(pixel_radius, 2.0f, 28.0f);

    float rotation = hash12(float2(id.xy)) * 6.28318530718f;
    float occlusion = 0.0f;
    float weight = 0.0f;

    float eccentricity = saturate(length(uv * 2.0f - 1.0f));
    uint sample_count = eccentricity < 0.50f ? 8u : eccentricity < 0.82f ? 6u : 4u;
    for (uint i = 0; i < sample_count; ++i) {
        float angle = rotation + (float(i) + 0.5f) * 2.39996322973f;
        float scale = (float(i) + 1.0f) / (float)sample_count;
        float2 offset_px = float2(cos(angle), sin(angle)) * pixel_radius * scale;
        float2 sample_uv = uv + offset_px / float2(width, height);
        if (any(sample_uv <= 0.0f) || any(sample_uv >= 1.0f)) continue;

        float4 sample_data = NormalDepth.SampleLevel(NormalDepthSampler, sample_uv, 0.0f);
        if (sample_data.w <= 0.0f) continue;

        float2 ndc_delta = offset_px / float2(width, height) * 2.0f;
        float3 delta = float3(ndc_delta.x * depth * tan_half_fov * aspect,
                              -ndc_delta.y * depth * tan_half_fov,
                              depth - sample_data.w);
        float distance_to_sample = length(delta);
        if (distance_to_sample <= 1.0e-5f || distance_to_sample > radius) continue;

        float alignment = dot(normal, delta / distance_to_sample);
        float range_weight = 1.0f - distance_to_sample / radius;
        occlusion += smoothstep(bias, 0.35f, alignment) * range_weight;
        weight += range_weight;
    }

    float ao = 1.0f - (weight > 0.0f ? occlusion / weight : 0.0f);
    ao = saturate(pow(ao, 1.25f));
    Output[id.xy] = float4(ao, ao, ao, 1.0f);
}
#elif defined(BUILD_BLOOM_CS)
GPU_BIND_T(0, 0) Texture2D<float4> Source : register(t0, space0);
GPU_BIND_S(0, 0) SamplerState SourceSampler : register(s0, space0);
GPU_BIND_U(0, 1) GPU_STORAGE_RGBA16F RWTexture2D<float4> Output : register(u0, space1);

GPU_BIND_B(0, 2) cbuffer BloomData : register(b0, space2)
{
    uint src_width;
    uint src_height;
    uint dst_width;
    uint dst_height;
    uint phase;
    uint _pad0;
    uint _pad1;
    uint _pad2;
    float threshold;
    float knee;
    float strength;
    float _pad3;
};

float3 extract_bloom(float3 color)
{
    float brightness = max(color.r, max(color.g, color.b));
    float soft = saturate((brightness - threshold + knee) / max(2.0f * knee, 1.0e-4f));
    soft = soft * soft * (3.0f - 2.0f * soft);
    float contribution = max(brightness - threshold, 0.0f) + soft * knee;
    return color * (contribution / max(brightness, 1.0e-4f));
}

[numthreads(8, 8, 1)]
void bloom_cs(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= dst_width || id.y >= dst_height) return;
    float2 uv = (float2(id.xy) + 0.5f) / float2(dst_width, dst_height);
    float2 texel = 1.0f / float2(src_width, src_height);

    float3 color = 0.0f;
    if (phase == 0u) {
        color += Source.SampleLevel(SourceSampler, uv + texel * float2(-0.5f, -0.5f), 0.0f).rgb;
        color += Source.SampleLevel(SourceSampler, uv + texel * float2( 0.5f, -0.5f), 0.0f).rgb;
        color += Source.SampleLevel(SourceSampler, uv + texel * float2(-0.5f,  0.5f), 0.0f).rgb;
        color += Source.SampleLevel(SourceSampler, uv + texel * float2( 0.5f,  0.5f), 0.0f).rgb;
        color = extract_bloom(color * 0.25f) * strength;
    } else {
        float2 axis = phase == 1u ? float2(texel.x, 0.0f) : float2(0.0f, texel.y);
        color += Source.SampleLevel(SourceSampler, uv, 0.0f).rgb * 0.2270270270f;
        color += Source.SampleLevel(SourceSampler, uv + axis * 1.3846153846f, 0.0f).rgb * 0.3162162162f;
        color += Source.SampleLevel(SourceSampler, uv - axis * 1.3846153846f, 0.0f).rgb * 0.3162162162f;
        color += Source.SampleLevel(SourceSampler, uv + axis * 3.2307692308f, 0.0f).rgb * 0.0702702703f;
        color += Source.SampleLevel(SourceSampler, uv - axis * 3.2307692308f, 0.0f).rgb * 0.0702702703f;
    }
    Output[id.xy] = float4(color, 1.0f);
}
#elif defined(BUILD_GRADE_CS)
GPU_BIND_U(0, 1) GPU_STORAGE_RGBA16F RWTexture2D<float4> Output : register(u0, space1);

float3 grade(float3 c)
{
    float luma = dot(c, float3(0.2126f, 0.7152f, 0.0722f));
    c = lerp(luma.xxx, c, 0.92f);

    float shadows = 1.0f - smoothstep(0.12f, 0.55f, luma);
    float highlights = smoothstep(0.52f, 0.95f, luma);
    c += shadows * float3(-0.018f, 0.006f, 0.028f);
    c += highlights * float3(0.028f, 0.010f, -0.012f);

    c = (c - 0.5f) * 1.055f + 0.5f;
    c.r *= 0.985f;
    c.b *= 1.015f;
    return saturate(c);
}

[numthreads(8, 8, 1)]
void grade_cs(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= 256u || id.y >= 16u) return;
    const uint size = 16u;
    uint r = id.x % size;
    uint b = id.x / size;
    uint g = id.y;
    float3 color = float3(r, g, b) / float(size - 1u);
    Output[id.xy] = float4(grade(color), 1.0f);
}
#elif (defined(BUILD_LIGHTMAP_CS) || defined(BUILD_LIGHTMAP_WAVE_CS)) || defined(BUILD_PROBE_CS)
struct BvhNode
{
    float4 bmin;
    float4 bmax;
    uint4 meta;
};

struct BvhTriangle
{
    float4 a;
    float4 b;
    float4 c;
    float4 normal;
};

struct TraceRay
{
    float3 origin;
    float tmin;
    float3 direction;
    float tmax;
};

struct TraceHit
{
    float t;
    float3 normal;
    float3 albedo;
    uint triangle_index;
};

struct BakeSample
{
    float4 position;
    float4 normal;
};

#if (defined(BUILD_LIGHTMAP_CS) || defined(BUILD_LIGHTMAP_WAVE_CS))
GPU_BIND_T(2, 0) StructuredBuffer<BvhNode> Nodes : register(t2, space0);
GPU_BIND_T(3, 0) StructuredBuffer<BvhTriangle> Triangles : register(t3, space0);
#else
GPU_BIND_T(1, 0) StructuredBuffer<BvhNode> Nodes : register(t1, space0);
GPU_BIND_T(2, 0) StructuredBuffer<BvhTriangle> Triangles : register(t2, space0);
#endif
#if (defined(BUILD_LIGHTMAP_CS) || defined(BUILD_LIGHTMAP_WAVE_CS))
GPU_BIND_T(0, 0) Texture2D<float4> Source : register(t0, space0);
GPU_BIND_S(0, 0) SamplerState SourceSampler : register(s0, space0);
GPU_BIND_T(1, 0) Texture2D<float4> Direct : register(t1, space0);
GPU_BIND_S(1, 0) SamplerState DirectSampler : register(s1, space0);
struct BakeProbe { float4 position; float4 coefficient[9]; };
GPU_BIND_T(5, 0) StructuredBuffer<BakeProbe> BakeProbes : register(t5, space0);
GPU_BIND_T(6, 0) StructuredBuffer<uint> PatchMap : register(t6, space0);
GPU_BIND_T(7, 0) StructuredBuffer<uint4> PatchAnchors : register(t7, space0);
GPU_BIND_T(4, 0) StructuredBuffer<BakeSample> Samples : register(t4, space0);
GPU_BIND_T(8, 0) StructuredBuffer<uint> ActiveIndices : register(t8, space0);
GPU_BIND_T(9, 0) StructuredBuffer<uint> ActiveCount : register(t9, space0);
GPU_BIND_U(0, 1) GPU_STORAGE_RGBA16F RWTexture2D<float4> Output : register(u0, space1);
GPU_BIND_U(1, 1) RWStructuredBuffer<uint> ActiveOut : register(u1, space1);
GPU_BIND_U(2, 1) RWStructuredBuffer<uint> ActiveOutCount : register(u2, space1);
#else
GPU_BIND_T(0, 0) StructuredBuffer<float4> ProbePositions : register(t0, space0);
GPU_BIND_U(0, 1) RWStructuredBuffer<float4> ProbeCoefficients : register(u0, space1);
#endif

GPU_BIND_B(0, 2) cbuffer BakeData : register(b0, space2)
{
    uint item_count;
    uint lightmap_width;
    uint lightmap_height;
    uint dispatch_width;

    uint iteration;
    uint phase;
    uint max_bounces;
    uint batch_count;

    float4 sun_direction_intensity;
    float4 sun_color_radius;
    float4 sky_zenith;
    float4 sky_horizon;
    float4 bake_params;
    float4 probe_origin_spacing;
    uint4 probe_dims_mode;
};

static const float PI = 3.14159265358979323846f;
static const uint INVALID_NODE = 0xffffffffu;
static const uint PHASE_CLEAR = 0;
static const uint PHASE_TRACE = 1;
static const uint PHASE_FILTER = 2;
static const uint PHASE_DILATE = 3;
static const uint PHASE_DIRECT = 4;
static const uint PHASE_COMBINE = 5;
static const uint PHASE_RECONSTRUCT = 6;

uint hash_u32(uint x)
{
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

float random01(inout uint state)
{
    state = hash_u32(state + 0x9e3779b9u);
    return (state & 0x00ffffffu) / 16777216.0f;
}

float3 sky_radiance(float3 direction)
{
    float t = saturate(direction.y * 0.5f + 0.5f);
    t = pow(t, 0.35f);
    return lerp(sky_horizon.rgb, sky_zenith.rgb, t) * bake_params.z;
}

#if (defined(BUILD_LIGHTMAP_CS) || defined(BUILD_LIGHTMAP_WAVE_CS))
float4 source_pixel(int2 p)
{
    float2 uv = (float2(p) + 0.5f) / float2(lightmap_width, lightmap_height);
    return Source.SampleLevel(SourceSampler, uv, 0.0f);
}
#endif

void basis(float3 normal, out float3 tangent, out float3 bitangent)
{
    float3 helper = abs(normal.y) < 0.999f
        ? float3(0.0f, 1.0f, 0.0f)
        : float3(1.0f, 0.0f, 0.0f);
    tangent = normalize(cross(helper, normal));
    bitangent = cross(normal, tangent);
}

float3 cosine_hemisphere(float3 normal, inout uint seed)
{
    float u1 = random01(seed);
    float u2 = random01(seed);
    float r = sqrt(u1);
    float phi = 2.0f * PI * u2;
    float3 local = float3(r * cos(phi), sqrt(max(0.0f, 1.0f - u1)), r * sin(phi));
    float3 tangent, bitangent;
    basis(normal, tangent, bitangent);
    return normalize(tangent * local.x + normal * local.y + bitangent * local.z);
}

float3 sample_sun(float3 direction, float radius, inout uint seed)
{
    float3 tangent, bitangent;
    basis(direction, tangent, bitangent);
    float r = sqrt(random01(seed)) * tan(radius);
    float phi = 2.0f * PI * random01(seed);
    return normalize(direction + tangent * (cos(phi) * r) + bitangent * (sin(phi) * r));
}

TraceRay make_trace_ray(float3 origin, float3 direction, float tmin, float tmax)
{
    TraceRay ray;
    ray.origin = origin;
    ray.tmin = tmin;
    ray.direction = direction;
    ray.tmax = tmax;
    return ray;
}

bool trace_ray_box(TraceRay ray, BvhNode node, float max_t)
{
    float tmin = ray.tmin;
    float tmax = min(ray.tmax, max_t);
    if (tmax < tmin)
        return false;

    [unroll]
    for (uint axis = 0; axis < 3; ++axis)
    {
        float o = ray.origin[axis];
        float d = ray.direction[axis];
        if (abs(d) < 1.0e-7f)
        {
            if (o < node.bmin[axis] || o > node.bmax[axis])
                return false;
            continue;
        }

        float inv = 1.0f / d;
        float a = (node.bmin[axis] - o) * inv;
        float b = (node.bmax[axis] - o) * inv;
        if (a > b)
        {
            float tmp = a;
            a = b;
            b = tmp;
        }
        tmin = max(tmin, a);
        tmax = min(tmax, b);
        if (tmin > tmax)
            return false;
    }

    return tmax >= ray.tmin;
}

bool trace_ray_triangle(TraceRay ray, BvhTriangle tri, float max_t, out float hit_t)
{
    float3 a = tri.a.xyz;
    float3 e1 = tri.b.xyz - a;
    float3 e2 = tri.c.xyz - a;
    float3 p = cross(ray.direction, e2);
    float det = dot(e1, p);
    if (abs(det) < 1.0e-7f)
    {
        hit_t = 0.0f;
        return false;
    }

    float inv_det = 1.0f / det;
    float3 s = ray.origin - a;
    float u = dot(s, p) * inv_det;
    if (u < 0.0f || u > 1.0f)
    {
        hit_t = 0.0f;
        return false;
    }

    float3 q = cross(s, e1);
    float v = dot(ray.direction, q) * inv_det;
    if (v < 0.0f || u + v > 1.0f)
    {
        hit_t = 0.0f;
        return false;
    }

    float t = dot(e2, q) * inv_det;
    if (t <= ray.tmin || t >= min(ray.tmax, max_t))
    {
        hit_t = 0.0f;
        return false;
    }

    hit_t = t;
    return true;
}

bool trace_any(TraceRay ray)
{
    uint node_index = 0u;
    while (node_index != INVALID_NODE)
    {
        BvhNode node = Nodes[node_index];
        if (!trace_ray_box(ray, node, ray.tmax))
        {
            node_index = node.meta.y;
            continue;
        }

        uint count = node.meta.w;
        if (count != 0u)
        {
            uint first = node.meta.z;
            for (uint i = 0; i < count; ++i)
            {
                float t;
                if (trace_ray_triangle(ray, Triangles[first + i], ray.tmax, t))
                    return true;
            }
            node_index = node.meta.y;
            continue;
        }

        node_index = node.meta.x;
    }
    return false;
}

bool trace_closest(TraceRay ray, out TraceHit hit)
{
    uint node_index = 0u;
    float closest = ray.tmax;
    bool found = false;
    hit.t = ray.tmax;
    hit.normal = 0.0f;
    hit.albedo = 0.0f;
    hit.triangle_index = INVALID_NODE;

    while (node_index != INVALID_NODE)
    {
        BvhNode node = Nodes[node_index];
        if (!trace_ray_box(ray, node, closest))
        {
            node_index = node.meta.y;
            continue;
        }

        uint count = node.meta.w;
        if (count != 0u)
        {
            uint first = node.meta.z;
            for (uint i = 0; i < count; ++i)
            {
                uint tri_index = first + i;
                float t;
                if (!trace_ray_triangle(ray, Triangles[tri_index], closest, t))
                    continue;

                closest = t;
                float3 normal = normalize(Triangles[tri_index].normal.xyz);
                if (dot(normal, ray.direction) > 0.0f)
                    normal = -normal;

                hit.t = t;
                hit.normal = normal;
                hit.albedo = saturate(float3(Triangles[tri_index].a.w,
                                             Triangles[tri_index].b.w,
                                             Triangles[tri_index].c.w));
                hit.triangle_index = tri_index;
                found = true;
            }
            node_index = node.meta.y;
            continue;
        }

        node_index = node.meta.x;
    }

    return found;
}

float3 direct_sun(float3 position, float3 normal, inout uint seed)
{
    float3 center = normalize(sun_direction_intensity.xyz);
    float3 direction = sample_sun(center, sun_color_radius.w, seed);
    float n_dot_l = saturate(dot(normal, direction));
    if (n_dot_l <= 0.0f)
        return 0.0f;

    TraceRay ray = make_trace_ray(position + normal * bake_params.x,
                                  direction, bake_params.x, 1.0e20f);
    if (trace_any(ray))
        return 0.0f;

    return sun_color_radius.rgb * (sun_direction_intensity.w * n_dot_l);
}

#if (defined(BUILD_LIGHTMAP_CS) || defined(BUILD_LIGHTMAP_WAVE_CS))
float3 bake_probe_irradiance(float3 position, float3 normal,
                             out float validity)
{
    validity = 0.0f;
    if (probe_dims_mode.x == 0u || probe_origin_spacing.w <= 0.0f)
        return 0.0f;
    float3 coord = clamp((position + normal * (0.3f * probe_origin_spacing.w) -
                          probe_origin_spacing.xyz) / probe_origin_spacing.w,
                          0.0f, float3(probe_dims_mode.xyz) - 1.0f);
    uint3 base = uint3(floor(coord));
    float3 f = frac(coord);
    float3 sum = 0.0f;
    float weight_sum = 0.0f;
    [unroll] for (uint z = 0u; z < 2u; ++z)
    [unroll] for (uint y = 0u; y < 2u; ++y)
    [unroll] for (uint x = 0u; x < 2u; ++x)
    {
        uint3 cell = min(base + uint3(x, y, z), probe_dims_mode.xyz - 1u);
        float3 w = lerp(1.0f - f, f, float3(x, y, z));
        BakeProbe probe = BakeProbes[cell.x + probe_dims_mode.x *
                                        (cell.y + probe_dims_mode.y * cell.z)];
        float weight = w.x * w.y * w.z * saturate(probe.position.w);
        sum += max(probe.coefficient[0].rgb * 0.2820947918f, 0.0f) * weight;
        weight_sum += weight;
    }
    validity = weight_sum;
    return weight_sum > 0.0f ? sum / weight_sum : 0.0f;
}
#endif

float3 trace_path_core(float3 position, float3 normal, inout uint seed,
                       bool include_primary_sun)
{
    float3 radiance = 0.0f;
    float3 throughput = 1.0f;

    for (uint bounce = 0; bounce < max_bounces; ++bounce)
    {
        if (bounce != 0u || include_primary_sun)
            radiance += throughput * direct_sun(position, normal, seed);

        float3 direction = cosine_hemisphere(normal, seed);
        TraceRay ray = make_trace_ray(position + normal * bake_params.x,
                                      direction, bake_params.x, 1.0e20f);
        TraceHit hit;

        if (!trace_closest(ray, hit))
        {
            radiance += throughput * sky_radiance(direction);
            break;
        }

        throughput *= hit.albedo;
        position = ray.origin + ray.direction * hit.t;
        normal = hit.normal;
#if (defined(BUILD_LIGHTMAP_CS) || defined(BUILD_LIGHTMAP_WAVE_CS))
        // Only reuse probes at secondary hits; primary reuse leaks across walls.
        {
            float validity;
            float3 cached = bake_probe_irradiance(position, normal, validity);
            if (validity >= 0.25f)
            {
                radiance += throughput * cached;
                break;
            }
        }
#endif
    }

    return radiance;
}

float3 trace_path(float3 position, float3 normal, inout uint seed)
{
    return trace_path_core(position, normal, seed, true);
}

#if (defined(BUILD_LIGHTMAP_CS) || defined(BUILD_LIGHTMAP_WAVE_CS))
float4 filtered_pixel(int2 p)
{
    float4 center = source_pixel(p);
    if (center.a == 0.0f) return 0.0f;

    float3 sum = 0.0f;
    float total = 0.0f;
    [unroll]
    for (int y = -1; y <= 1; ++y)
    {
        [unroll]
        for (int x = -1; x <= 1; ++x)
        {
            int2 q = clamp(p + int2(x, y), int2(0, 0),
                           int2((int)lightmap_width - 1, (int)lightmap_height - 1));
            float4 c = source_pixel(q);
            if (c.a == 0.0f) continue;
            float difference = length(c.rgb - center.rgb);
            float weight = 1.0f / (1.0f + difference * 4.0f);
            sum += c.rgb * weight;
            total += weight;
        }
    }
    return float4(total > 0.0f ? sum / total : center.rgb, 1.0f);
}

float4 dilated_pixel(int2 p)
{
    float4 center = source_pixel(p);
    if (center.a != 0.0f) return float4(center.rgb, 1.0f);

    float3 sum = 0.0f;
    float count = 0.0f;
    [unroll]
    for (int y = -1; y <= 1; ++y)
    {
        [unroll]
        for (int x = -1; x <= 1; ++x)
        {
            int2 q = clamp(p + int2(x, y), int2(0, 0),
                           int2((int)lightmap_width - 1, (int)lightmap_height - 1));
            float4 c = source_pixel(q);
            if (c.a == 0.0f) continue;
            sum += c.rgb;
            count += 1.0f;
        }
    }
    return count > 0.0f ? float4(sum / count, 1.0f) : 0.0f;
}

[numthreads(64, 1, 1)]
void lightmap_cs(uint3 dispatch_id : SV_DispatchThreadID)
{
    uint dispatch_index = dispatch_id.x + dispatch_id.y * dispatch_width;
    uint index = dispatch_index;
    if (phase == PHASE_TRACE && probe_dims_mode.w != 0u)
    {
        uint active_count = ActiveCount[0];
        if (dispatch_index >= active_count)
            return;
        index = ActiveIndices[dispatch_index];
    }
    else if (index >= item_count)
        return;

    if (phase == PHASE_CLEAR)
    {
        uint2 p = uint2(index % lightmap_width, index / lightmap_width);
        Output[p] = 0.0f;
        return;
    }

    if (phase == PHASE_DIRECT)
    {
        BakeSample sample = Samples[index];
        uint pixel = asuint(sample.position.w);
        float3 normal = normalize(sample.normal.xyz);
        uint seed = hash_u32(pixel ^ 0x4f03d2b1u);
        float3 a = direct_sun(sample.position.xyz, normal, seed);
        float3 b = direct_sun(sample.position.xyz, normal, seed);
        float3 c = direct_sun(sample.position.xyz, normal, seed);
        float3 d = direct_sun(sample.position.xyz, normal, seed);
        float3 sum = a + b + c + d;
        uint count = 4u;
        float3 mean = sum * 0.25f;
        if (length(a - mean) + length(b - mean) +
            length(c - mean) + length(d - mean) > 0.02f)
        {
            for (uint i = 0u; i < 12u; ++i)
                sum += direct_sun(sample.position.xyz, normal, seed);
            count = 16u;
        }
        Output[uint2(pixel % lightmap_width, pixel / lightmap_width)] =
            float4(sum / (float)count, 1.0f);
        return;
    }

    if (phase == PHASE_COMBINE)
    {
        uint2 p = uint2(index % lightmap_width, index / lightmap_width);
        float4 indirect = source_pixel(int2(p));
        float4 direct = Direct.SampleLevel(DirectSampler,
            (float2(p) + 0.5f) / float2(lightmap_width, lightmap_height), 0.0f);
        Output[p] = indirect.a != 0.0f ? float4(indirect.rgb + direct.rgb, 1.0f) : 0.0f;
        return;
    }

    if (phase == PHASE_RECONSTRUCT)
    {
        uint pixel = asuint(Samples[index].position.w);
        uint2 p = uint2(pixel % lightmap_width, pixel / lightmap_width);
        uint patch = PatchMap[index];
        float3 color;
        if (patch == 0xffffffffu)
            color = source_pixel(int2(p)).rgb;
        else
        {
            uint4 anchors = PatchAnchors[patch];
            float2 t = float2(p.x % 4u, p.y % 4u) / 3.0f;
            float3 c00 = source_pixel(int2(anchors.x % lightmap_width,
                                           anchors.x / lightmap_width)).rgb;
            float3 c10 = source_pixel(int2(anchors.y % lightmap_width,
                                           anchors.y / lightmap_width)).rgb;
            float3 c01 = source_pixel(int2(anchors.z % lightmap_width,
                                           anchors.z / lightmap_width)).rgb;
            float3 c11 = source_pixel(int2(anchors.w % lightmap_width,
                                           anchors.w / lightmap_width)).rgb;
            color = lerp(lerp(c00, c10, t.x), lerp(c01, c11, t.x), t.y);
        }
        Output[p] = float4(color, 1.0f);
        return;
    }

    if (phase == PHASE_TRACE)
    {
        BakeSample sample = Samples[index];
        float3 position = sample.position.xyz;
        float3 normal = normalize(sample.normal.xyz);
        uint pixel = asuint(sample.position.w);
        uint2 pixel_xy = uint2(pixel % lightmap_width, pixel / lightmap_width);
        float4 previous = iteration == 0u ? 0.0f : source_pixel(int2(pixel_xy));
        for (uint offset = 0u; offset < batch_count && previous.a >= 0.0f; ++offset)
        {
            uint current = iteration + offset;
            uint seed = hash_u32(pixel ^ hash_u32(current + 0x51f2e91du));
            float3 value = trace_path_core(position, normal, seed, false);
            float n = (float)current;
            float3 mean = (previous.rgb * n + value) / (n + 1.0f);
            float old_luma = dot(previous.rgb, float3(0.2126f, 0.7152f, 0.0722f));
            float sample_luma = dot(value, float3(0.2126f, 0.7152f, 0.0722f));
            float new_luma = dot(mean, float3(0.2126f, 0.7152f, 0.0722f));
            float m2 = current == 0u ? 0.0f : previous.a;
            m2 += (sample_luma - old_luma) * (sample_luma - new_luma);
            float stderr = sqrt(max(m2, 0.0f) / max(n * (n + 1.0f), 1.0f));
            // Use a tighter error bound before 64 samples.
            float threshold = 0.01f + 0.025f * abs(new_luma);
            if (current + 1u < 64u) threshold *= 0.5f;
            bool converged = current + 1u >= (uint)bake_params.w &&
                ((current + 1u) % 8u == 0u) &&
                stderr < threshold;
            // Retain the sample count in negative alpha to mark convergence.
            previous = float4(mean, converged ? -(float)(current + 1u) : max(m2, 1.0e-6f));
        }
        if (previous.a >= 0.0f)
        {
            uint slot = 0u;
#if defined(BUILD_LIGHTMAP_WAVE_CS)
            uint wave_offset = WavePrefixCountBits(true);
            uint wave_count = WaveActiveCountBits(true);
            uint wave_base = 0u;
            if (WaveIsFirstLane())
                InterlockedAdd(ActiveOutCount[0], wave_count, wave_base);
            wave_base = WaveReadLaneFirst(wave_base);
            slot = wave_base + wave_offset;
#else
            InterlockedAdd(ActiveOutCount[0], 1u, slot);
#endif
            ActiveOut[slot] = index;
        }

        Output[pixel_xy] = previous;
        return;
    }

    uint2 pixel_xy = uint2(index % lightmap_width, index / lightmap_width);
    int2 p = int2(pixel_xy);
    if (phase == PHASE_FILTER)
        Output[pixel_xy] = filtered_pixel(p);
    else if (phase == PHASE_DILATE)
        Output[pixel_xy] = dilated_pixel(p);
}
#else
float3 uniform_sphere(inout uint seed)
{
    float z = 1.0f - 2.0f * random01(seed);
    float phi = 2.0f * PI * random01(seed);
    float radius = sqrt(max(0.0f, 1.0f - z*z));
    return float3(radius*cos(phi), radius*sin(phi), z);
}

void probe_basis(float3 d, out float basis_values[9])
{
    float x = d.x, y = d.y, z = d.z;
    basis_values[0] = 0.2820947918f;
    basis_values[1] = 0.4886025119f*y;
    basis_values[2] = 0.4886025119f*z;
    basis_values[3] = 0.4886025119f*x;
    basis_values[4] = 1.0925484306f*x*y;
    basis_values[5] = 1.0925484306f*y*z;
    basis_values[6] = 0.3153915653f*(3.0f*z*z-1.0f);
    basis_values[7] = 1.0925484306f*x*z;
    basis_values[8] = 0.5462742153f*(x*x-y*y);
}

groupshared float3 ProbePartial[64][9];
groupshared float ProbeValid;
groupshared float4 ProbeSHMeans[64];
groupshared float4 ProbeSHSquares[64];
groupshared uint ProbeSampleCount;
groupshared uint ProbeContinue;

[numthreads(64, 1, 1)]
void probe_cs(uint3 group_id : SV_GroupID, uint3 local_id : SV_GroupThreadID)
{
    uint probe_index = group_id.x;
    uint lane = local_id.x;
    float4 input = ProbePositions[probe_index];
    if (lane == 0u) {
        ProbeValid = input.w;
        ProbeSampleCount = item_count;
        ProbeContinue = 1u;
        float3 axes[6] = {
            float3(1,0,0), float3(-1,0,0), float3(0,1,0),
            float3(0,-1,0), float3(0,0,1), float3(0,0,-1)
        };
        for (uint axis = 0; axis < 6u && ProbeValid > 0.0f; ++axis) {
            TraceRay ray = make_trace_ray(input.xyz, axes[axis], bake_params.x, 0.15f);
            TraceHit hit;
            if (trace_closest(ray, hit))
                ProbeValid = 0.0f;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    float3 partial[9];
    [unroll] for (uint j = 0; j < 9; ++j) partial[j] = 0.0f;
    float4 sh_mean = 0.0f;
    float4 sh_square = 0.0f;
    uint seed = hash_u32(probe_index * 9781u + lane * 6271u + iteration * 13007u);

    if (ProbeValid > 0.0f) {
        for (uint block = 0u; block < item_count && ProbeContinue != 0u; block += 64u) {
            uint sample_index = block + lane;
            if (sample_index < item_count) {
                float3 d = uniform_sphere(seed);
                TraceRay ray = make_trace_ray(input.xyz + d * bake_params.x,
                                              d, bake_params.x, 1.0e20f);
                TraceHit hit;
                float3 incoming;
                if (trace_closest(ray, hit)) {
                    float3 position = ray.origin + ray.direction * hit.t;
                    incoming = trace_path(position, hit.normal, seed) * (hit.albedo / PI);
                } else {
                    incoming = sky_radiance(d);
                }
                float sh[9];
                probe_basis(d, sh);
                [unroll] for (uint j = 0; j < 9; ++j)
                    partial[j] += incoming * sh[j];
                float luma = dot(incoming, float3(0.2126f, 0.7152f, 0.0722f));
                float4 sh_value = luma * float4(sh[0], sh[1], sh[2], sh[3]);
                sh_mean += sh_value;
                sh_square += sh_value * sh_value;
            }
            if ((block + 64u) % 128u == 0u && block + 64u >= 512u &&
                block + 64u < item_count) {
                ProbeSHMeans[lane] = sh_mean;
                ProbeSHSquares[lane] = sh_square;
                GroupMemoryBarrierWithGroupSync();
                if (lane == 0u) {
                    float4 total_mean = 0.0f;
                    float4 total_square = 0.0f;
                    for (uint i = 0u; i < 64u; ++i) {
                        total_mean += ProbeSHMeans[i];
                        total_square += ProbeSHSquares[i];
                    }
                    float samples = (float)(block + 64u);
                    float4 mean = total_mean / samples;
                    float4 variance = max(total_square / samples - mean * mean, 0.0f);
                    float4 stderr = sqrt(variance / samples);
                    float worst = max(max(stderr.x, stderr.y), max(stderr.z, stderr.w));
                    if (worst < 0.01f + 0.025f * abs(mean.x)) {
                        ProbeSampleCount = block + 64u;
                        ProbeContinue = 0u;
                    }
                }
                GroupMemoryBarrierWithGroupSync();
            }
        }
    }
    [unroll] for (uint j = 0; j < 9; ++j)
        ProbePartial[lane][j] = partial[j];
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = 32u; stride > 0u; stride >>= 1u) {
        if (lane < stride) {
            [unroll] for (uint j = 0; j < 9; ++j)
                ProbePartial[lane][j] += ProbePartial[lane + stride][j];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (lane < 9u) {
        float scale = 4.0f * PI / max((float)ProbeSampleCount, 1.0f);
        float3 sun = normalize(sun_direction_intensity.xyz);
        TraceRay sun_ray = make_trace_ray(input.xyz + sun * bake_params.x,
                                          sun, bake_params.x, 1.0e20f);
        float sun_visible = lane == 1u && ProbeValid > 0.0f &&
            !trace_any(sun_ray) ? 1.0f : 0.0f;
        ProbeCoefficients[probe_index * 9u + lane] =
            float4(ProbePartial[0][lane] * scale,
                   lane == 0u ? ProbeValid : sun_visible);
    }
}
#endif
#endif
