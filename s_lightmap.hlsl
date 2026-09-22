RWTexture2D<float4> Lightmap : register(u0, space1);
cbuffer Bake : register(b0, space2)
{
    uint Width;
    uint Height;
    float Exposure;
    float Padding;
};
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= Width || tid.y >= Height) return;
    float2 uv = (float2(tid.xy) + 0.5) / float2(Width, Height);
    float2 p = uv * 2.0 - 1.0;
    float vignette = 1.0 - 0.20 * saturate(dot(p, p));
    float warm = smoothstep(0.0, 1.0, uv.y);
    float3 low = float3(0.42, 0.48, 0.62);
    float3 high = float3(0.92, 0.78, 0.56);
    float3 gi = lerp(low, high, warm) * vignette * Exposure;
    Lightmap[tid.xy] = float4(gi, 1.0);
}
