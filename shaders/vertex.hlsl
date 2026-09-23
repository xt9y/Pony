#if defined(BUILD_SURFACE_VS)
cbuffer Camera : register(b0, space1)
{
    float4x4 mvp;
    float4x4 view;
};

struct SurfaceInput
{
    float3 position : TEXCOORD0;
    float3 normal : TEXCOORD1;
    float2 uv : TEXCOORD2;
    float2 lightmap_uv : TEXCOORD3;
};

struct SurfaceOutput
{
    float4 position : SV_Position;
    float3 world_position : TEXCOORD0;
    float3 world_normal : TEXCOORD1;
    float2 uv : TEXCOORD2;
    float2 lightmap_uv : TEXCOORD3;
    float3 view_normal : TEXCOORD4;
    float view_depth : TEXCOORD5;
};

SurfaceOutput surface_vs(SurfaceInput input)
{
    SurfaceOutput output;
    float4 world = float4(input.position, 1.0f);
    float4 view_position = mul(view, world);
    output.position = mul(mvp, world);
    output.world_position = input.position;
    output.world_normal = normalize(input.normal);
    output.uv = input.uv;
    output.lightmap_uv = input.lightmap_uv;
    output.view_normal = normalize(mul((float3x3)view, input.normal));
    // output.view_depth = max(-view_position.z, 0.0f);
    output.view_depth = max(output.position.w, 0.0f);
    return output;
}
#elif defined(BUILD_FULLSCREEN_VS)
struct FullscreenOutput
{
    float4 position : SV_Position;
    float2 ndc : TEXCOORD0;
};

FullscreenOutput fullscreen_vs(uint vertex_id : SV_VertexID)
{
    float2 p;
    if (vertex_id == 0u) p = float2(-1.0f, -1.0f);
    else if (vertex_id == 1u) p = float2(-1.0f, 3.0f);
    else p = float2(3.0f, -1.0f);

    FullscreenOutput output;
    output.position = float4(p, 0.0f, 1.0f);
    output.ndc = p;
    return output;
}
#elif defined(BUILD_WIREFRAME_VS)
cbuffer Camera : register(b0, space1)
{
    float4x4 mvp;
};

struct WireframeInput
{
    float3 position : TEXCOORD0;
    float4 color : TEXCOORD1;
};

struct WireframeOutput
{
    float4 position : SV_Position;
    float4 color : TEXCOORD0;
};

WireframeOutput wireframe_vs(WireframeInput input)
{
    WireframeOutput output;
    output.position = mul(mvp, float4(input.position, 1.0f));
    output.color = input.color;
    return output;
}
#endif
