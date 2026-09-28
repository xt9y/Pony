struct GPUObject {

    row_major float4x4 world;
    row_major float4x4 previous_world;
    row_major float4x4 normal_world;

    uint4 draw;
    uint4 meta;
};

struct GPUMaterial {

    float4 base_color;

    float3 emissive;
    float metallic;

    float roughness;
    float3 padding;
};

struct FrameConstants {

    row_major float4x4 view_projection;
    row_major float4x4 previous_view_projection;

    float4 camera_position;
    float4 resolution;
};

StructuredBuffer<GPUObject> Objects : register(t0, space0);

StructuredBuffer<GPUMaterial> Materials : register(t1, space0);

ConstantBuffer<FrameConstants> Frame : register(b0, space0);

struct DrawConstants {
    uint object_index;
};

[[vk::push_constant]]
DrawConstants Draw;


struct GBufferVSInput {

    [[vk::location(0)]]
    float3 position : POSITION;

    [[vk::location(1)]]
    float3 normal : NORMAL;

    [[vk::location(2)]]
    float2 uv : TEXCOORD0;

    [[vk::location(3)]]
    uint material : TEXCOORD1;
};


struct GBufferVSOutput {

    float4 position : SV_Position;

    [[vk::location(0)]]
    float3 normal : NORMAL0;

    [[vk::location(1)]]
    float2 uv : TEXCOORD0;

    [[vk::location(2)]]
    nointerpolation uint material : TEXCOORD1;

    [[vk::location(3)]]
    nointerpolation uint object_id : TEXCOORD2;

    [[vk::location(4)]]
    float4 current_clip : TEXCOORD3;

    [[vk::location(5)]]
    float4 previous_clip : TEXCOORD4;
};


GBufferVSOutput VS_GBuffer(GBufferVSInput input) {
    
    GPUObject object = Objects[Draw.object_index];

    float4 local_position = float4(input.position, 1.0);

    float4 world_position = mul(local_position, object.world);

    float4 previous_world_position = mul(local_position, object.previous_world);

    GBufferVSOutput output;

    output.position = mul(world_position, Frame.view_projection);

    output.current_clip =
        output.position;

    output.previous_clip = mul(previous_world_position, Frame.previous_view_projection);

    output.normal = normalize(mul(float4(input.normal, 0.0), object.normal_world).xyz);

    output.uv = input.uv;

    output.material = object.draw.w + input.material;

    output.object_id = object.draw.z;

    return output;
}


struct GBufferOutput {

    float4 normal_roughness : SV_Target0;

    float4 albedo_metallic : SV_Target1;

    float2 velocity : SV_Target2;

    uint object_id : SV_Target3;
};


GBufferOutput PS_GBuffer(GBufferVSOutput input) {
    
    GPUMaterial material = Materials[input.material];

    GBufferOutput output;

    output.normal_roughness =float4( normalize(input.normal), material.roughness);

    output.albedo_metallic = float4(material.base_color.rgb, material.metallic);

    const float2 current_ndc = input.current_clip.xy / input.current_clip.w;

    const float2 previous_ndc = input.previous_clip.xy / input.previous_clip.w;

    output.velocity = (current_ndc - previous_ndc) * float2(0.5, -0.5);

    output.object_id = input.object_id;

    return output;
}


struct PresentVSOutput {

    float4 position : SV_Position;
};


PresentVSOutput VS_Present(uint vertex_id : SV_VertexID) {

    const float2 positions[3] = {
        float2(-1.0, -1.0),
        float2(-1.0,  3.0),
        float2( 3.0, -1.0)
    };

    PresentVSOutput output;

    output.position = float4(positions[vertex_id], 0.0, 1.0);

    return output;
}


Texture2D<float4> PresentTexture : register(t0, space0);


float4 PS_Present(PresentVSOutput input) : SV_Target0 {

    const int2 pixel = int2(input.position.xy);

    const float3 color = PresentTexture.Load(int3(pixel, 0)).rgb;

    return float4(color, 1.0);
}

