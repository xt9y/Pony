Texture2D<float4> Lightmap : register(t0, space2);
SamplerState LightmapSampler : register(s0, space2);
struct FSIn
{
    float4 position : SV_Position;
    float3 normal : TEXCOORD0;
    float2 uv : TEXCOORD1;
    float2 lightmap_uv : TEXCOORD2;
    float3 albedo : TEXCOORD3;
};
float4 main(FSIn i) : SV_Target
{
    float3 n = normalize(i.normal);
    float3 sun_dir = normalize(float3(0.45, 0.85, 0.30));
    float direct = saturate(dot(n, sun_dir));
    float3 baked = Lightmap.Sample(LightmapSampler, i.lightmap_uv).rgb;
    float checker = fmod(floor(i.uv.x * 8.0) + floor(i.uv.y * 8.0), 2.0);
    float surface = lerp(0.82, 1.0, checker);
    float3 lighting = baked + direct * float3(0.45, 0.42, 0.36);
    return float4(i.albedo * surface * lighting, 1.0);
}
