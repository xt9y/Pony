from pathlib import Path

path = Path('shaders/vision_compute.hlsl')
text = path.read_text()

def replace_once(old, new):
    global text
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f'expected one match, found {count}: {old[:100]!r}')
    text = text.replace(old, new, 1)

replace_once(
'''float vision_eccentricity(float2 uv)
{
    return saturate(length(uv * 2.0f - 1.0f));
}

float vision_transition(float eccentricity, float radius, float width)
''',
'''float vision_eccentricity(float2 uv)
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
''')

replace_once(
'''float3 integrate_probe_quality(uint probe_steps, float3 direction,
                               float enter, float leave,
                               float3 surface_position, float3 surface_normal,
                               bool has_surface)
{
    probe_steps = min(max(probe_steps, 1u), 4u);
    float step_size = (leave - enter) / float(probe_steps);
    float probe_transmission = exp(-volume_params.x * step_size);
    float probe_remaining = 1.0f;
    float3 sum = 0.0f;

    [loop] for (uint i = 0u; i < 4u; ++i) {
        if (i >= probe_steps) break;
        float t = enter + (float(i) + 0.5f) * step_size;
''',
'''float3 integrate_probe_quality(uint probe_steps, uint2 pixel_id, float3 direction,
                               float enter, float leave,
                               float3 surface_position, float3 surface_normal,
                               bool has_surface)
{
    probe_steps = min(max(probe_steps, 1u), 4u);
    float step_size = (leave - enter) / float(probe_steps);
    float probe_transmission = exp(-volume_params.x * step_size);
    float probe_remaining = 1.0f;
    float3 sum = 0.0f;
    float jitter = vision_jitter(pixel_id);

    [loop] for (uint i = 0u; i < 4u; ++i) {
        if (i >= probe_steps) break;
        float t = enter + (float(i) + 0.5f + (jitter - 0.5f) * 0.65f) * step_size;
        t = clamp(t, enter + 1.0e-5f, leave - 1.0e-5f);
''')

text = text.replace(
'integrate_probe_quality(volume_quality.x, direction, enter, leave,',
'integrate_probe_quality(volume_quality.x, id.xy, direction, enter, leave,', 1)
text = text.replace(
'integrate_probe_quality(volume_quality.y, direction, enter, leave,',
'integrate_probe_quality(volume_quality.y, id.xy, direction, enter, leave,', 1)
text = text.replace(
'integrate_probe_quality(volume_quality.z, direction, enter, leave,',
'integrate_probe_quality(volume_quality.z, id.xy, direction, enter, leave,', 1)

replace_once(
'''float4 reconstructed_volume(float2 uv, float center_depth)
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

float3 vision_hdr''',
'''float4 reconstructed_volume(float2 uv, float center_depth)
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
    int2 offsets[5] = {
        int2(0, 0), int2(1, 0), int2(-1, 0), int2(0, 1), int2(0, -1)
    };
    float4 sum = center_fog * 1.5f;
    float weight_sum = 1.5f;
    float2 texel = 1.0f / float2(width, height);

    [unroll] for (uint i = 1u; i < 5u; ++i) {
        float2 sample_uv = saturate(uv + float2(offsets[i]) * texel);
        float sample_depth = NormalDepth.SampleLevel(DepthSampler, sample_uv, 0.0f).w;
        float weight = 0.35f * depth_similarity(center_depth, sample_depth);
        if (weight <= 1.0e-4f) continue;
        float4 sample_fog = reconstructed_volume(sample_uv, sample_depth);
        sum += sample_fog * weight;
        weight_sum += weight;
    }

    return sum / max(weight_sum, 1.0e-5f);
}

float3 vision_hdr''')

replace_once(
'''    float4 fog = bypass_volume != 0u
        ? float4(0.0f, 0.0f, 0.0f, 1.0f)
        : reconstructed_volume(uv, center_depth);

    if (debug_view >= 3u && bypass_volume == 0u) {
''',
'''    float4 fog = bypass_volume != 0u
        ? float4(0.0f, 0.0f, 0.0f, 1.0f)
        : reconstructed_volume(uv, center_depth);
    if (bypass_volume == 0u)
        fog = edge_aware_volume_blur(uv, center_depth, fog);

    if (debug_view >= 3u && bypass_volume == 0u) {
''')

path.write_text(text)
print('vision jitter/blur patch applied')
