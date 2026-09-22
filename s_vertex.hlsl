cbuffer Draw : register(b0, space1)
{
    float4x4 MVP;
    float4x4 Model;
    float4 Albedo;
};
struct VSIn
{
    float3 position : TEXCOORD0;
    float3 normal : TEXCOORD1;
    float2 uv : TEXCOORD2;
    float2 lightmap_uv : TEXCOORD3;
};
struct VSOut
{
    float4 position : SV_Position;
    float3 normal : TEXCOORD0;
    float2 uv : TEXCOORD1;
    float2 lightmap_uv : TEXCOORD2;
    float3 albedo : TEXCOORD3;
};
VSOut main(VSIn i)
{
    VSOut o;
    o.position = mul(MVP, float4(i.position, 1.0));
    o.normal = normalize(mul((float3x3)Model, i.normal));
    o.uv = i.uv;
    o.lightmap_uv = i.lightmap_uv;
    o.albedo = Albedo.rgb;
    return o;
}
