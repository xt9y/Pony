from pathlib import Path


def patch(path, replacements):
    p = Path(path)
    text = p.read_text()
    for old, new in replacements:
        count = text.count(old)
        if count != 1:
            raise RuntimeError(f"{path}: expected one match, found {count}: {old[:120]!r}")
        text = text.replace(old, new, 1)
    p.write_text(text)


patch("game.h", [
    (
'''typedef struct VOLUMETRICS_LIGHTING {
    float density;
    float anisotropy;
    float probe_intensity;
    float emissive_probe_intensity;
    float max_distance;
    float center_radius;
    float middle_radius;
    float center_transition_width;
    float middle_transition_width;
    float probe_spacing;
    uint32_t center_steps;
    uint32_t middle_steps;
    uint32_t peripheral_steps;
    uint32_t center_stride;
    uint32_t middle_stride;
    uint32_t peripheral_stride;
    uint32_t probe_samples;
    uint32_t emissive_samples;
} VOLUMETRICS_LIGHTING;
''',
'''typedef struct VOLUMETRICS_LIGHTING {
    float density;
    float anisotropy;
    float probe_intensity;
    float emissive_probe_intensity;
    float max_distance;
    float probe_spacing;
    uint32_t probe_samples;
    uint32_t emissive_samples;
} VOLUMETRICS_LIGHTING;

typedef struct PERIPHERAL_VISION {
    float center_radius;
    float middle_radius;
    float center_transition_width;
    float middle_transition_width;
    float jitter_strength;
    float volume_blur_strength;
    uint32_t center_steps;
    uint32_t middle_steps;
    uint32_t peripheral_steps;
    uint32_t center_stride;
    uint32_t middle_stride;
    uint32_t peripheral_stride;
} PERIPHERAL_VISION;
'''),
    (
'''bool r_draw(RENDERER *r, const struct LIGHT *light, const SKY *sky, const VOLUMETRICS_LIGHTING *volumetrics);''',
'''bool r_draw(RENDERER *r, const struct LIGHT *light, const SKY *sky, const VOLUMETRICS_LIGHTING *volumetrics, const PERIPHERAL_VISION *vision);''')
])

patch("gpu.h", [
    (
'''    SKY sky;
    VOLUMETRICS_LIGHTING volumetrics;
    float tan_half_fov;''',
'''    SKY sky;
    VOLUMETRICS_LIGHTING volumetrics;
    PERIPHERAL_VISION vision;
    float tan_half_fov;''')
])

patch("main.c", [
    (
'''    VOLUMETRICS_LIGHTING volumetrics = {.density = 0.045f,
                                         .anisotropy = 0.55f,
                                         .probe_intensity = 0.15f,
                                         .emissive_probe_intensity = 1.0f,
                                         .max_distance = 10000.0f,
                                         .center_radius = 0.50f,
                                         .middle_radius = 0.82f,
                                         .center_transition_width = 0.08f,
                                         .middle_transition_width = 0.08f,
                                         .probe_spacing = 4.0f,
                                         .center_steps = 4u,
                                         .middle_steps = 3u,
                                         .peripheral_steps = 2u,
                                         .center_stride = 1u,
                                         .middle_stride = 2u,
                                         .peripheral_stride = 4u,
                                         .probe_samples = 1024u,
                                         .emissive_samples = 128u};''',
'''    VOLUMETRICS_LIGHTING volumetrics = {.density = 0.045f,
                                         .anisotropy = 0.55f,
                                         .probe_intensity = 0.15f,
                                         .emissive_probe_intensity = 1.0f,
                                         .max_distance = 10000.0f,
                                         .probe_spacing = 4.0f,
                                         .probe_samples = 1024u,
                                         .emissive_samples = 128u};
    PERIPHERAL_VISION vision = {.center_radius = 0.50f,
                                .middle_radius = 0.82f,
                                .center_transition_width = 0.08f,
                                .middle_transition_width = 0.08f,
                                .jitter_strength = 0.65f,
                                .volume_blur_strength = 0.35f,
                                .center_steps = 4u,
                                .middle_steps = 2u,
                                .peripheral_steps = 1u,
                                .center_stride = 1u,
                                .middle_stride = 8u,
                                .peripheral_stride = 24u};'''),
    (
'''            if (!r_draw(&r, light_data, &sky, &volumetrics)) {''',
'''            if (!r_draw(&r, light_data, &sky, &volumetrics, &vision)) {''')
])

patch("render.c", [
    (
'''bool r_draw(RENDERER *r, const struct LIGHT *light, const SKY *sky, const VOLUMETRICS_LIGHTING *volumetrics) {
    if (!r || !r->window || !light || light->type != LIGHT_DIRECTIONAL || !sky || !volumetrics) return false;''',
'''bool r_draw(RENDERER *r, const struct LIGHT *light, const SKY *sky, const VOLUMETRICS_LIGHTING *volumetrics, const PERIPHERAL_VISION *vision) {
    if (!r || !r->window || !light || light->type != LIGHT_DIRECTIONAL || !sky || !volumetrics || !vision) return false;'''),
    (
'''    RENDER_FRAME frame = {.eye = eye, .right = right, .up = up, .forward = forward, .sun = sun, .sky = *sky, .volumetrics = *volumetrics, .tan_half_fov = tan_half, .aspect = aspect};''',
'''    RENDER_FRAME frame = {.eye = eye, .right = right, .up = up, .forward = forward, .sun = sun, .sky = *sky, .volumetrics = *volumetrics, .vision = *vision, .tan_half_fov = tan_half, .aspect = aspect};''')
])

patch("gpu.c", [
    (
'''    float volume_params[4], volume_radii[4];
    Uint32 volume_quality[4], volume_strides[4];''',
'''    float volume_params[4], volume_radii[4], volume_filter[4];
    Uint32 volume_quality[4], volume_strides[4];'''),
    (
'''    float volume_radii[4];
    Uint32 volume_strides[4];''',
'''    float volume_radii[4];
    float volume_filter[4];
    Uint32 volume_strides[4];'''),
    (
'''                               .volume_params = {frame->volumetrics.density, frame->volumetrics.anisotropy, frame->volumetrics.probe_intensity, frame->volumetrics.max_distance},
                               .volume_radii = {frame->volumetrics.center_radius, frame->volumetrics.middle_radius, frame->volumetrics.center_transition_width, frame->volumetrics.middle_transition_width},
                               .volume_quality = {frame->volumetrics.center_steps, frame->volumetrics.middle_steps, frame->volumetrics.peripheral_steps, 0u},
                               .volume_strides = {frame->volumetrics.center_stride, frame->volumetrics.middle_stride, frame->volumetrics.peripheral_stride, 0u}};''',
'''                               .volume_params = {frame->volumetrics.density, frame->volumetrics.anisotropy, frame->volumetrics.probe_intensity, frame->volumetrics.max_distance},
                               .volume_radii = {frame->vision.center_radius, frame->vision.middle_radius, frame->vision.center_transition_width, frame->vision.middle_transition_width},
                               .volume_filter = {frame->vision.jitter_strength, frame->vision.volume_blur_strength, 0.0f, 0.0f},
                               .volume_quality = {frame->vision.center_steps, frame->vision.middle_steps, frame->vision.peripheral_steps, 0u},
                               .volume_strides = {frame->vision.center_stride, frame->vision.middle_stride, frame->vision.peripheral_stride, 0u}};'''),
    (
'''        .volume_radii = {frame->volumetrics.center_radius, frame->volumetrics.middle_radius, frame->volumetrics.center_transition_width, frame->volumetrics.middle_transition_width},
        .volume_strides = {frame->volumetrics.center_stride, frame->volumetrics.middle_stride, frame->volumetrics.peripheral_stride, 0u},''',
'''        .volume_radii = {frame->vision.center_radius, frame->vision.middle_radius, frame->vision.center_transition_width, frame->vision.middle_transition_width},
        .volume_filter = {frame->vision.jitter_strength, frame->vision.volume_blur_strength, 0.0f, 0.0f},
        .volume_strides = {frame->vision.center_stride, frame->vision.middle_stride, frame->vision.peripheral_stride, 0u},''')
])

patch("shaders/compute.hlsl", [
    (
'''    float4 volume_params;
    float4 volume_radii;
    uint4 volume_quality;''',
'''    float4 volume_params;
    float4 volume_radii;
    float4 volume_filter;
    uint4 volume_quality;''')
])

patch("shaders/vision_compute.hlsl", [
    (
'''    probe_steps = min(max(probe_steps, 1u), 4u);
    float step_size = (leave - enter) / float(probe_steps);''',
'''    probe_steps = max(probe_steps, 1u);
    float step_size = (leave - enter) / float(probe_steps);'''),
    (
'''    float jitter = vision_jitter(pixel_id);

    [loop] for (uint i = 0u; i < 4u; ++i) {
        if (i >= probe_steps) break;
        float t = enter + (float(i) + 0.5f + (jitter - 0.5f) * 0.65f) * step_size;''',
'''    float jitter = vision_jitter(pixel_id);
    float jitter_strength = max(volume_filter.x, 0.0f);

    [loop] for (uint i = 0u; i < probe_steps; ++i) {
        float t = enter + (float(i) + 0.5f + (jitter - 0.5f) * jitter_strength) * step_size;'''),
    (
'''    float4 volume_radii;
    uint4 volume_strides;''',
'''    float4 volume_radii;
    float4 volume_filter;
    uint4 volume_strides;'''),
    (
'''float4 edge_aware_volume_blur(float2 uv, float center_depth, float4 center_fog)
{
    int2 offsets[5] = {''',
'''float4 edge_aware_volume_blur(float2 uv, float center_depth, float4 center_fog)
{
    float blur_strength = max(volume_filter.y, 0.0f);
    if (blur_strength <= 1.0e-5f) return center_fog;

    int2 offsets[5] = {'''),
    (
'''        float weight = 0.35f * depth_similarity(center_depth, sample_depth);''',
'''        float weight = blur_strength * depth_similarity(center_depth, sample_depth);''')
])

print("peripheral vision split patch applied")
