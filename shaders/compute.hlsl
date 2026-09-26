#ifndef GPU_BIND_S
#define GPU_BIND_S(n,s) [[vk::binding(n, s)]]
#define GPU_BIND_T(n,s) [[vk::binding(n + 16, s)]]
#define GPU_BIND_B(n,s) [[vk::binding(n + 32, s)]]
#define GPU_BIND_U(n,s) [[vk::binding(n + 48, s)]]
#define GPU_STORAGE_RGBA16F [[vk::image_format("rgba16f")]]
#endif

#if defined(BUILD_PROBE_CS)
#define probe_cs probe_cs_base
#include "compute_base.hlsl"
#undef probe_cs

groupshared uint ProbeStableChecks;

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
        ProbeStableChecks = 0u;
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
                    incoming = hit.emissive * max(emissive_data.z, 0.0f) +
                    trace_path(position, hit.normal, seed) * (hit.albedo / PI);
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

            if ((block + 64u) % 128u == 0u && block + 64u >= 256u &&
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
                    float sample_count = (float)(block + 64u);
                    float4 mean = total_mean / sample_count;
                    float4 variance = max(total_square / sample_count - mean * mean, 0.0f);
                    float4 stderr = sqrt(variance / sample_count);
                    float worst = max(max(stderr.x, stderr.y), max(stderr.z, stderr.w));
                    bool stable = worst < 0.01f + 0.025f * abs(mean.x);
                    ProbeStableChecks = stable ? ProbeStableChecks + 1u : 0u;
                    if (ProbeStableChecks >= 2u) {
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
        float metadata = lane == 0u ? ProbeValid :
                         lane == 1u ? sun_visible :
                         lane == 2u ? (float)ProbeSampleCount : 0.0f;
        ProbeCoefficients[probe_index * 9u + lane] =
            float4(ProbePartial[0][lane] * scale, metadata);
    }
}
#else
#include "compute_base.hlsl"
#endif