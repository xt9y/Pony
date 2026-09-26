#ifndef GPU_BIND_S
#define GPU_BIND_S(n,s) [[vk::binding(n, s)]]
#define GPU_BIND_T(n,s) [[vk::binding(n + 16, s)]]
#define GPU_BIND_B(n,s) [[vk::binding(n + 32, s)]]
#define GPU_BIND_U(n,s) [[vk::binding(n + 48, s)]]
#define GPU_STORAGE_RGBA16F [[vk::image_format("rgba16f")]]
#endif

#if defined(BUILD_LIGHTMAP_QUEUE_RESET_CS)
GPU_BIND_U(0, 1) RWStructuredBuffer<uint> QueueCount : register(u0, space1);

[numthreads(1, 1, 1)]
void lightmap_queue_reset_cs(uint3 id : SV_DispatchThreadID)
{
    if (id.x == 0u)
        QueueCount[0] = 0u;
}
#elif defined(BUILD_LIGHTMAP_QUEUE_ARGS_CS)
GPU_BIND_T(0, 0) StructuredBuffer<uint> QueueCount : register(t0, space0);
GPU_BIND_U(0, 1) RWStructuredBuffer<uint> DispatchArgs : register(u0, space1);
GPU_BIND_B(0, 2) cbuffer LightmapQueueData : register(b0, space2)
{
    uint dispatch_width;
    uint _pad0;
    uint _pad1;
    uint _pad2;
};

[numthreads(1, 1, 1)]
void lightmap_queue_args_cs(uint3 id : SV_DispatchThreadID)
{
    if (id.x != 0u)
        return;

    uint width = max(dispatch_width, 64u);
    uint groups_x = max(width / 64u, 1u);
    uint count = QueueCount[0];
    uint groups_y = max((count + width - 1u) / width, 1u);

    DispatchArgs[0] = groups_x;
    DispatchArgs[1] = groups_y;
    DispatchArgs[2] = 1u;
}
#endif
