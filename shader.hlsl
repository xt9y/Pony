#define FAR_PLANE 10000.0f
#define NEAR_PLANE 0.05f

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

[[vk::binding(0, 0)]]
StructuredBuffer<GPUObject> Objects : register(t0, space0);

[[vk::binding(1, 0)]]
StructuredBuffer<GPUMaterial> Materials : register(t1, space0);

[[vk::binding(2, 0)]]
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

    output.current_clip = output.position;

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

    output.normal_roughness = float4(normalize(input.normal), material.roughness);

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
        float2(-1.0, 3.0),
        float2(3.0, -1.0)
    };

    PresentVSOutput output;

    output.position = float4(positions[vertex_id], 0.0, 1.0);

    return output;
}


[[vk::binding(0, 1)]]
Texture2D<float> DepthTexture : register(t0, space1);

[[vk::binding(1, 1)]]
Texture2D<float4> NormalRoughnessTexture : register(t1, space1);

[[vk::binding(2, 1)]]
Texture2D<float4> AlbedoMetallicTexture : register(t2, space1);

[[vk::binding(3, 1)]]
Texture2D<float2> VelocityTexture : register(t3, space1);

[[vk::binding(4, 1)]]
Texture2D<uint> ObjectIdTexture : register(t4, space1);


[[vk::binding(5, 1)]]
Texture2D<float> HZBTexture : register(t5, space1);

[[vk::binding(0, 2)]]
Texture2D<float> HZBSource : register(t0, space2);

[[vk::binding(1, 2)]]
RWTexture2D<float> HZBOutput : register(u0, space2);

[numthreads(8, 8, 1)]
void CS_HZB(uint3 dispatch_id : SV_DispatchThreadID) {

    uint src_width;
    uint src_height;

    uint dst_width;
    uint dst_height;

    HZBSource.GetDimensions(src_width, src_height);
    HZBOutput.GetDimensions(dst_width, dst_height);

    const uint2 pixel = dispatch_id.xy;

    if (pixel.x >= dst_width || pixel.y >= dst_height) {
        return;
    }

    // HZB mip 0 is an exact copy of the raster depth buffer.
    // if (src_width == dst_width && src_height == dst_height) {
    //     HZBOutput[pixel] = HZBSource.Load(int3(pixel, 0));
    //     return;
    // }

    const uint2 src_size = uint2(src_width, src_height);
    const uint2 dst_size = uint2(dst_width, dst_height);

    const uint2 src_begin = (pixel * src_size) / dst_size;
    const uint2 src_end = ((pixel + uint2(1, 1)) * src_size) / dst_size;

    float depth = 0.0;

    for (uint y = src_begin.y; y < src_end.y; ++y) {

        for (uint x = src_begin.x; x < src_end.x; ++x) {

            depth = max(depth, HZBSource.Load(int3(int2(x, y), 0)));
        }
    }

    HZBOutput[pixel] = depth;
}

float4 PS_Present(PresentVSOutput input) : SV_Target0 {

    const int2 pixel = int2(input.position.xy);

    const float near_plane = NEAR_PLANE;
    const float far_plane = FAR_PLANE;

    #ifdef ALBEDO
        return float4(
            AlbedoMetallicTexture.Load(int3(pixel, 0)).rgb, 
            1.0
        );

    #elif defined(NORMALS)
        return float4(
            (NormalRoughnessTexture.Load(int3(pixel, 0)).rgb) * 0.5 + 0.5,
            1.0
        );
        
    #elif defined(DEPTH)
        const float linear_depth = (near_plane * far_plane) / (DepthTexture.Load(int3(pixel, 0)) * (far_plane - near_plane) + near_plane);
        const float visible_depth = 1.0 - saturate(linear_depth / 200.0);

        return float4(
            visible_depth.xxx,
            1.0
        );

    #elif defined(ROUGHNESS)
        return float4(
            (NormalRoughnessTexture.Load(int3(pixel, 0)).w).xxx, 
            1.0
        );

    #elif defined(VELOCITY)
        return float4(
            saturate(0.5 + VelocityTexture.Load(int3(pixel, 0)) * 20).x,
            saturate(0.5 + VelocityTexture.Load(int3(pixel, 0)) * 20).y,
            0.5,
            1.0
        );

    #elif defined(OBJECT_ID)
        const float value = frac((float)ObjectIdTexture.Load(int3(pixel, 0)) * 0.61803398875);
        return float4(
            value,
            frac(value * 3.17),
            frac(value * 7.13),
            1.0
        );

    #elif defined(HZB)
        // const float depth_a = DepthTexture.Load(int3(pixel, 0));
        // const float depth_b = HZBTexture.Load(int3(pixel, 0));
        // const float diff = saturate(abs(depth_a - depth_b) * 50000.0);
        // return float4(diff.xxx, 1.0);

        uint hzb_width;
        uint hzb_height;
        uint hzb_mip_count;

        HZBTexture.GetDimensions(0, hzb_width, hzb_height, hzb_mip_count);

        const uint mip = 0u;

        float hzb_depth;

        if (mip == 0u) {
            hzb_depth = DepthTexture.Load(int3(pixel, 0));
        } else {
            const uint hzb_mip = mip - 1u;

            uint hzb_width;
            uint hzb_height;
            uint hzb_mip_count;

            HZBTexture.GetDimensions(0, hzb_width, hzb_height, hzb_mip_count);

            const uint mip_width = max(1u, hzb_width >> hzb_mip);
            const uint mip_height = max(1u, hzb_height >> hzb_mip);

            const uint2 hzb_pixel = min(
                uint2(pixel) >> mip,
                uint2(mip_width - 1u, mip_height - 1u)
            );

            hzb_depth = HZBTexture.Load(int3(int2(hzb_pixel), hzb_mip));
        }

        const float linear_depth =
            (near_plane * far_plane) /
            (hzb_depth * (far_plane - near_plane) + near_plane);

        return float4(
            1.0 - saturate(linear_depth / 200.0).xxx,
            1.0
        );

    #else


    return float4(
        1.0,
        0.0,
        1.0,
        1.0
    );

    #endif
}


