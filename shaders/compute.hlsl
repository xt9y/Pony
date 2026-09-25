#if defined(BUILD_VOLUME_CS)
Texture2D<float4> NormalDepth : register(t0, space0);
SamplerState DepthSampler : register(s0, space0);
struct VolumeProbe { float4 position; float4 coefficient[9]; };
StructuredBuffer<VolumeProbe> VolumeProbes : register(t1, space0);
StructuredBuffer<float> SunBeams : register(t2, space0);
RWTexture2D<float4> Output : register(u0, space1);
cbuffer VolumeData : register(b0, space2)
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

static const uint BEAM_WIDTH = 64u;
static const uint BEAM_HEIGHT = 64u;

uint beam_shadow_index(uint x, uint y, uint beam_depth)
{
    return BEAM_WIDTH * BEAM_HEIGHT * beam_depth + x + BEAM_WIDTH * y;
}

float integrate_beam_interval(float3 ray_origin, float3 ray_direction,
                              float t0, float t1, float medium_enter,
                              float density, uint beam_depth,
                              out float lit_distance)
{
    lit_distance = 0.0f;
    if (t1 <= t0)
        return 0.0f;

    float midpoint = 0.5f * (t0 + t1);
    float3 sun_position = ray_origin + ray_direction * midpoint;
    float2 cell = (sun_position.xy - beam_origin.xy) / beam_step.xy - 0.5f;
    int2 base = int2(floor(cell));
    float2 fraction = frac(cell);
    float blocker_guard = max(0.002f, min(0.02f, min(beam_step.x, beam_step.y) * 0.01f));

    float z0 = ray_origin.z + ray_direction.z * t0;
    float z1 = ray_origin.z + ray_direction.z * t1;
    float interval_max_z = max(z0, z1);

    // Directional-sun visibility is a 2D height field: every x/y column is
    // lit until its first sun-facing blocker and shadowed behind it. Shape
    // beam edges from that field only; never from baked Z slices.
    float raw_visibility = 0.0f;
    [unroll] for (uint y = 0; y < 2u; ++y)
    [unroll] for (uint x = 0; x < 2u; ++x)
    {
        int2 index = base + int2(x, y);
        if (any(index < 0) || index.x >= (int)BEAM_WIDTH || index.y >= (int)BEAM_HEIGHT)
            continue;

        float2 axis_weight = lerp(1.0f - fraction, fraction, float2(x, y));
        float weight = axis_weight.x * axis_weight.y;
        float blocker = SunBeams[beam_shadow_index((uint)index.x, (uint)index.y, beam_depth)];
        bool interval_has_light = blocker >= -1.0e20f && interval_max_z > blocker + blocker_guard;
        // bool interval_has_light = blocker < -1.0e20f || interval_max_z > blocker + blocker_guard;
        raw_visibility += interval_has_light ? weight : 0.0f;
    }

    float visibility_shape = smoothstep(0.35f, 0.85f, raw_visibility);
    if (visibility_shape <= 0.0f)
        return 0.0f;

    float integrated = 0.0f;
    [unroll] for (uint y = 0; y < 2u; ++y)
    [unroll] for (uint x = 0; x < 2u; ++x)
    {
        int2 index = base + int2(x, y);
        if (any(index < 0) || index.x >= (int)BEAM_WIDTH || index.y >= (int)BEAM_HEIGHT)
            continue;

        float2 axis_weight = lerp(1.0f - fraction, fraction, float2(x, y));
        float visibility = axis_weight.x * axis_weight.y * visibility_shape;
        if (visibility <= 0.0f)
            continue;

        float a = t0;
        float b = t1;
        float blocker = SunBeams[beam_shadow_index((uint)index.x, (uint)index.y, beam_depth)];
        if (blocker < -1.0e20f) continue;

        if (blocker >= -1.0e20f)
        {
            blocker += blocker_guard;
            if (abs(ray_direction.z) < 1.0e-7f)
            {
                if (sun_position.z <= blocker)
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

    const uint beam_depth = height_debug.z;
    if (beam_depth < 16u || beam_depth > 128u)
        return 0.0f;

    float3 sun = normalize(sun_intensity.xyz);
    float3 u = normalize(cross(float3(0, 1, 0), sun));
    float3 v = cross(sun, u);
    float3 ray_origin = float3(dot(world_origin, u), dot(world_origin, v), dot(world_origin, sun));
    float3 ray_direction = float3(dot(world_direction, u), dot(world_direction, v), dot(world_direction, sun));

    float3 bounds_min = beam_origin.xyz;
    float3 bounds_max = bounds_min + beam_step.xyz *
        float3((float)BEAM_WIDTH, (float)BEAM_HEIGHT, (float)beam_depth);
    float grid_enter = medium_enter;
    float grid_leave = medium_leave;

    // Clip once against the complete sun-space prism. Z only limits the valid
    // baked region; it must not subdivide the integration into artificial
    // planes.
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

    // Traverse only x/y boundaries of the blocker map. Boundaries are placed
    // where the bilinear base texel changes (half a texel from sample centers),
    // so every interval uses one continuous bilinear neighborhood.
    float epsilon = max(1.0e-5f, min(beam_step.x, beam_step.y) * 1.0e-4f);
    float inside_t = min(grid_leave, grid_enter + epsilon);
    float2 inside = ray_origin.xy + ray_direction.xy * inside_t;
    int2 cell = int2(floor((inside - bounds_min.xy) / beam_step.xy - 0.5f));
    cell = clamp(cell, int2(-1, -1), int2((int)BEAM_WIDTH - 1, (int)BEAM_HEIGHT - 1));

    int2 step_direction = int2(
        ray_direction.x > 1.0e-7f ? 1 : (ray_direction.x < -1.0e-7f ? -1 : 0),
        ray_direction.y > 1.0e-7f ? 1 : (ray_direction.y < -1.0e-7f ? -1 : 0));

    const float huge = 1.0e30f;
    float2 t_delta = float2(
        step_direction.x == 0 ? huge : beam_step.x / abs(ray_direction.x),
        step_direction.y == 0 ? huge : beam_step.y / abs(ray_direction.y));

    float2 next_boundary = float2(
        bounds_min.x + (step_direction.x > 0 ? float(cell.x) + 1.5f : float(cell.x) + 0.5f) * beam_step.x,
        bounds_min.y + (step_direction.y > 0 ? float(cell.y) + 1.5f : float(cell.y) + 0.5f) * beam_step.y);
    float2 t_max = float2(
        step_direction.x == 0 ? huge : (next_boundary.x - ray_origin.x) / ray_direction.x,
        step_direction.y == 0 ? huge : (next_boundary.y - ray_origin.y) / ray_direction.y);

    float integrated = 0.0f;
    float lit_distance = 0.0f;
    float t = grid_enter;

    [loop] for (uint step_index = 0; step_index < 160u; ++step_index)
    {
        if (cell.x < -1 || cell.x >= (int)BEAM_WIDTH ||
            cell.y < -1 || cell.y >= (int)BEAM_HEIGHT)
            break;

        float t_next = min(grid_leave, min(t_max.x, t_max.y));
        t_next = max(t_next, t);

        if (t_next > t + 1.0e-7f)
        {
            float interval_lit_distance = 0.0f;
            integrated += integrate_beam_interval(ray_origin, ray_direction,
                                                  t, t_next, medium_enter,
                                                  density, beam_depth,
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
        t = t_next;
    }

    lit_fraction = lit_distance / max(medium_leave - medium_enter, 1.0e-6f);
    return integrated;
}

float3 volume_radiance(float3 p, float3 surface_position,
                       float3 surface_normal, bool has_surface)
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
        // Do not interpolate light from a probe behind the visible surface.
        // Probes embedded in thick walls or roofs can otherwise brighten fog
        // in front of those surfaces even though their own validity is true.
        if (has_surface && dot(probe.position.xyz - surface_position,
                               surface_normal) < -0.01f)
            continue;
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

    // [unroll] for (uint axis = 0; axis < 3u; ++axis) {
    //
    //     if (abs(direction[axis]) < 1.0e-6f) {
    //         if (eye_density[axis] < minimum[axis] ||
    //             eye_density[axis] > maximum[axis]) { Output[id.xy] = float4(0,0,0,1); return; }
    //     } else {
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


    float4 surface = NormalDepth.SampleLevel(DepthSampler, uv, 0.0f);
    float depth = surface.w;

    // if (height_debug.y == 3u) {
    //     if (depth <= 0.0f) {
    //         Output[id.xy] = float4(0, 0, 0, 1);
    //         return;
    //     }
    //
    //     float distance = depth / max(dot(direction, forward_g.xyz), 0.01f);
    //     float3 reconstructed = eye_density.xyz + direction * distance;
    //     float error = length(reconstructed - surface.xyz);
    //
    //     Output[id.xy] = error < 0.2f
    //         ? float4(0, 1, 0, 1)   // camera ray reaches the actual surface
    //         : float4(1, 0, 0, 1);  // depth/ray reconstruction disagrees
    //     return;
    // }

    // if (height_debug.y == 3u) {
    //     Output[id.xy] = depth > 0.0f
    //         ? float4(surface.r, 0, 0, 1)
    //         : float4(0, 0, 0, 1);
    //     return;
    // } 
    // if (height_debug.y == 3u) {
    //     if (depth <= 0.0f) {
    //         Output[id.xy] = float4(0, 0, 0, 1);
    //         return;
    //     }
    //     float mismatch = length(surface.xyz - eye_density.xyz);
    //     Output[id.xy] = float4(saturate(mismatch * 5.0f), 0, 0, 1);
    //     return;
    // }

    // if (height_debug.y == 3u) {
    //     float mismatch = length(surface.xyz - forward_g.xyz);
    //     Output[id.xy] = float4(saturate(mismatch * 5.0f), 0, 0, 1);
    //     // Output[id.xy] = float4(surface.rgb, 1.0f);
    //     return;
    // }


    // if (height_debug.y == 3u) {
    //     if (depth <= 0.0f) {
    //         Output[id.xy] = float4(0, 0, 0, 1);
    //         return;
    //     }
    //
    //
    //     // float3 actual = surface.xyz - eye_density.xyz;
    //
    //     float3 actual = surface.xyz;
    //
    //
    //     float depth_error = abs(depth - dot(actual, forward_g.xyz));
    //     float direction_error = length(direction - normalize(actual));
    //
    //     Output[id.xy] = float4(
    //         saturate(depth_error * 2.0f),      // red: view-depth disagreement
    //         0.0f,
    //         saturate(direction_error * 25.0f), // blue: camera-ray disagreement
    //         1.0f);
    //     return;
    // }


    // if (height_debug.y == 3u) {
    //
    //     if (depth <= 0.0f) {
    //         Output[id.xy] = float4(0, 0, 0, 1);
    //         return;
    //     }
    //
    //     float surface_t = depth / max(dot(direction, forward_g.xyz), 0.01f);
    //     float3 p = eye_density.xyz + direction * max(surface_t - 0.02f, 0.0f);
    //     float3 sun = normalize(sun_intensity.xyz);
    //     float3 u = normalize(cross(float3(0, 1, 0), sun));
    //     float3 v = cross(sun, u);
    //
    //     float2 xy = float2(dot(p, u), dot(p, v));
    //     int2 pixel = clamp(
    //         int2(floor((xy - beam_origin.xy) / beam_step.xy)),
    //         int2(0, 0), int2(63, 63));
    //
    //     float blocker = SunBeams[64u * 64u * height_debug.z +
    //                              (uint)pixel.x + 64u * (uint)pixel.y];
    //
    //     // Magenta: no blocker. Red: blocker says ceiling is sunlit.
    //     // Green: ceiling is correctly shadowed at its own position.
    //     Output[id.xy] = blocker < -1.0e20f ? float4(1, 0, 1, 1) :
    //         dot(p, sun) > blocker + 0.02f ? float4(1, 0, 0, 1) :
    //                                         float4(0, 1, 0, 1);
    //     return;
    // }

    // if (height_debug.y == 3u) {
    //     Output[id.xy] = depth > 0.0f
    //         ? float4(0.0f, 1.0f, 0.0f, 1.0f)  // surface depth exists
    //         : float4(1.0f, 0.0f, 0.0f, 1.0f); // shader sees sky/no surface
    //     return;
    // }


    if (depth > 0.0f) leave = min(leave, depth / max(dot(direction,forward_g.xyz),0.01f));
    if (leave <= enter) { Output[id.xy] = float4(0,0,0,1); return; }
    float3 surface_position = eye_density.xyz + direction *
        (depth / max(dot(direction, forward_g.xyz), 0.01f));
    float3 view_normal = normalize(surface.xyz * 2.0f - 1.0f);
    float3 surface_normal = normalize(normalize(right_tan.xyz) * view_normal.x +
                                      normalize(up_tan.xyz) * view_normal.y -
                                      forward_g.xyz * view_normal.z);
    if (dot(eye_density.xyz - surface_position, surface_normal) < 0.0f)
        surface_normal = -surface_normal;
    float step_size = (leave-enter) * 0.25f;
    float3 sum = 0.0f;
    float sun_fraction = 0.0f;
    float probe_transmission = exp(-eye_density.w * step_size);
    float probe_remaining = 1.0f;
    [unroll] for (uint i = 0; i < 4u; ++i) {
        float t = enter + (float(i) + 0.5f) * step_size;
        float integral = probe_remaining * (1.0f - probe_transmission);
        sum += volume_radiance(eye_density.xyz + direction * t,
                               surface_position, surface_normal, depth > 0.0f) *
               integral * 0.15f;
        probe_remaining *= probe_transmission;
    }
    float g = forward_g.w;
    float cosine = dot(direction, normalize(sun_intensity.xyz));
    float hg = (1.0f - g*g) / (12.5663706144f *
        pow(max(1.0f + g*g - 2.0f*g*cosine, 0.001f), 1.5f));
    float sun_integral = integrate_sun_grid(eye_density.xyz, direction, enter, leave,
                                            eye_density.w, sun_fraction);
    sum += sun_integral * sun_intensity.w * hg * float3(1.0f, 0.94f, 0.84f);
    // float T = exp(-eye_density.w * (leave - enter));
    // Output[id.xy] = float4(T.xxx, 1.0f);
    // return;
    Output[id.xy] = float4(sum, exp(-eye_density.w * (leave-enter)));
}
#else
#include "compute_base.hlsl"
#endif
