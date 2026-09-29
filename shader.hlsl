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
    row_major float4x4 inverse_view_projection;
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


[[vk::binding(6, 1)]]
Texture2D<float4> ScreenTraceTexture : register(t6, space1);

[[vk::binding(0, 3)]]
Texture2D<float> TraceDepth : register(t0, space3);

[[vk::binding(1, 3)]]
Texture2D<float4> TraceNormalRoughness : register(t1, space3);

[[vk::binding(2, 3)]]
Texture2D<float4> TraceAlbedoMetallic : register(t2, space3);

[[vk::binding(3, 3)]]
Texture2D<float> TraceHZB : register(t3, space3);

[[vk::binding(4, 3)]]
ConstantBuffer<FrameConstants> TraceFrame : register(b0, space3);

[[vk::binding(5, 3)]]
RWTexture2D<float4> ScreenTraceOutput : register(u0, space3);



float2 ScreenUVToNDC(float2 uv) {

    return float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
}

float2 NDCToScreenUV(float2 ndc) {

    return float2(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
}

float3 ReconstructWorldPosition(uint2 pixel, float depth) {

    const float2 uv = (float2(pixel) + 0.5) * TraceFrame.resolution.zw;
    const float2 ndc = ScreenUVToNDC(uv);

    float4 world = mul(float4(ndc, depth, 1.0), TraceFrame.inverse_view_projection);

    return world.xyz / world.w;
}

uint2 TraceLevelDimensions(uint level) {

    if (level == 0) return uint2(TraceFrame.resolution.xy);

    uint width;
    uint height;
    uint mip_count;

    TraceHZB.GetDimensions(level - 1, width, height, mip_count);

    return uint2(width, height);
}

float TraceLevelDepth(float2 uv, uint level, out uint2 cell) {

    const uint2 dimensions = TraceLevelDimensions(level);
    cell = min( uint2(uv * dimensions), dimensions - 1);

    if (level == 0) return TraceDepth.Load(int3(cell, 0));

    return TraceHZB.Load(int3(cell, level - 1));
}

bool ProjectTracePoint(float4 clip_origin, float4 clip_direction, float distance, out float2 uv, out float depth) {

    uv = float2(0.0, 0.0);
    depth = 0.0;
    const float4 clip = clip_origin + clip_direction * distance;

    if (clip.w <= 1.0e-5) return false;

    const float3 ndc = clip.xyz / clip.w;

    uv = NDCToScreenUV(ndc.xy);
    depth = ndc.z;

    return
        all(uv >= 0.0) &&
        all(uv < 1.0) &&
        depth >= 0.0 &&
        depth <= 1.0;
}

float SolveTraceDistance(float value_origin, float value_direction, float w_origin, float w_direction, float target) {

    const float denominator = value_direction - target * w_direction;

    if (abs(denominator) < 1.0e-7) return 1.0e30;

    return(target * w_origin - value_origin) / denominator;
}

float TraceCellExit(float4 clip_origin, float4 clip_direction, float2 uv, uint2 dimensions, uint2 cell, float distance, float2 screen_direction) {

    float exit_x = 1.0e30;
    float exit_y = 1.0e30;

    const float4 current_clip = clip_origin + clip_direction * distance;

    if (abs(screen_direction.x) > 1.0e-7) {

        const float boundary = screen_direction.x > 0.0 ? (float)(cell.x + 1u) / (float)dimensions.x : (float)cell.x / (float)dimensions.x;
        const float target_ndc = boundary * 2.0 - 1.0;
        exit_x = distance + SolveTraceDistance(current_clip.x, clip_direction.x, current_clip.w, clip_direction.w, target_ndc);

        if (exit_x <= distance + 1.0e-5) exit_x = 1.0e30;
    }

    if (abs(screen_direction.y) > 1.0e-7) {

        const float boundary = screen_direction.y > 0.0 ? (float)(cell.y +1u) / (float)dimensions.y : (float)(cell.y) / (float)dimensions.y;
        const float target_ndc = 1.0 - boundary * 2.0;

        exit_y = distance + SolveTraceDistance(current_clip.y, clip_direction.y, current_clip.w, clip_direction.w, target_ndc);

        if (exit_y <= distance + 1.0e-5) exit_y = 1.0e30;
    }

    return min(exit_x, exit_y);
}

float SolveDepthCrossing(float4 clip_origin, float4 clip_direction, float target_depth) {

    return SolveTraceDistance(
        clip_origin.z,
        clip_direction.z,
        clip_origin.w,
        clip_direction.w,
        target_depth
    );
}

float LinearizeDepth(float depth) {
    return
        (NEAR_PLANE * FAR_PLANE) /
        (depth *
            (FAR_PLANE - NEAR_PLANE) +
            NEAR_PLANE
        );
}

bool TraceScreenRay(float3 origin, float3 direction, uint2 origin_pixel, out uint2 hit_pixel) {

    hit_pixel = uint2(0, 0);
    static const float MAX_DISTANCE = 200.0;
    static const uint MAX_STEPS = 128;
    static const uint MAX_START_LEVEL = 5;

    const float4 clip_origin = mul(
        float4(origin, 1.0),
        TraceFrame.view_projection
    );

    const float4 clip_direction = mul(
        float4(direction, 0.0),
        TraceFrame.view_projection
    );

    if (clip_origin.w <= 1.0e-5) return false;

    uint hzb_width;
    uint hzb_height;
    uint hzb_mip_count;

    TraceHZB.GetDimensions(0, hzb_width, hzb_height, hzb_mip_count);

    const uint maximum_level = min(hzb_mip_count, MAX_START_LEVEL);
    uint level = maximum_level;

    /*
     * Sign of projected movement.
     * A projected 3D line remains a line, although
     * its parameterization is perspective nonlinear.
     */
    const float2 screen_direction = float2(
        clip_direction.x * clip_origin.w - clip_origin.x * clip_direction.w,
        -(clip_direction.y * clip_origin.w - clip_origin.y * clip_direction.w)
    );

    float distance = 0.0;

    [loop]
    for (uint step = 0; step < MAX_STEPS; ++step) {

        if (distance >= MAX_DISTANCE) return false;

        float2 uv;
        float ray_depth;

        if (!ProjectTracePoint(clip_origin, clip_direction, distance, uv, ray_depth)) {
            return false;
        }

        uint2 cell;
        const float scene_depth = TraceLevelDepth(uv, level, cell);
        const uint2 dimensions = TraceLevelDimensions(level);
        float exit_distance = TraceCellExit(clip_origin, clip_direction, uv, dimensions, cell, distance, screen_direction);
        // float exit_distance = distance + 0.1;
        if (exit_distance >= 1.0e29) return false;
        exit_distance = min(exit_distance, MAX_DISTANCE);

        float2 exit_uv;
        float exit_depth;

        if (!ProjectTracePoint(clip_origin, clip_direction, exit_distance, exit_uv, exit_depth)) {
            return false;
        }

        /*
         * Reverse Z:
         *
         * larger depth = nearer.
         *
         * The HZB stores max(), therefore scene_depth
         * is the nearest surface represented by the cell.
         *
         * If even the farther end of this ray segment is
         * still nearer than that value, the complete cell
         * can safely be skipped.
         */
        // const float farther_ray_depth = min(ray_depth, exit_depth);
        // const bool clear_cell = scene_depth <= 0.0 || farther_ray_depth > scene_depth + 1.0e-5;

        const float farther_ray_depth =
            min(ray_depth, exit_depth);

        const float scene_linear =
            LinearizeDepth(scene_depth);

        const float farther_ray_linear =
            LinearizeDepth(farther_ray_depth);

        const float skip_thickness =
            0.05;

        const bool clear_cell =
            scene_depth <= 0.0 ||
            farther_ray_linear <
            scene_linear - skip_thickness;

        // if (clear_cell) {
        //     distance = exit_distance + max(1.0e-4, exit_distance * 1.0e-5);
        //     // distance = exit_distance + 0.01;
        //     // level = min(level + 1u, maximum_level);
        //     continue;
        // }

        if (clear_cell) {
            distance = exit_distance + max(1.0e-4, exit_distance * 1.0e-5);
            continue;
        }

        if (level > 0u) {
            --level;
            continue;
        }

        // Full-resolution candidate.
        // const float hit_distance = SolveDepthCrossing(clip_origin, clip_direction, scene_depth);
        //
        // if (hit_distance >= distance - 1.0e-4 && hit_distance <= exit_distance + 1.0e-4) {
        //
        //     const float3 surface_position = ReconstructWorldPosition(cell, scene_depth);
        //     const float3 ray_position = origin + direction * hit_distance;
        //     const float camera_distance = length(surface_position - TraceFrame.camera_position.xyz);
        //     const float thickness = max(0.03, camera_distance * 0.002);
        //     const bool self_hit = all(cell == origin_pixel) && hit_distance < 0.05;
        //
        //     if (!self_hit && length(surface_position - ray_position) <= thickness) {
        //         hit_pixel = cell;
        //         return true;
        //     }
        // }

        // const float hit_distance =
        //     SolveDepthCrossing(
        //         clip_origin,
        //         clip_direction,
        //         scene_depth
        //     );
        //
        // if (hit_distance >= distance - 1.0e-4 &&
        //     hit_distance <= exit_distance + 1.0e-4) {
        //
        //     const bool self_hit =
        //         all(cell == origin_pixel) &&
        //         hit_distance < 0.05;
        //
        //     if (!self_hit) {
        //         hit_pixel = cell;
        //         return true;
        //     }
        // }

        // if (!all(cell == origin_pixel)) {
        //     hit_pixel = cell;
        //     return true;
        // }

        // const float segment_near =
        //     max(ray_depth, exit_depth);
        //
        // const float segment_far =
        //     min(ray_depth, exit_depth);
        //
        // const float thickness = 1.0e-4;
        //
        // const bool depth_overlap =
        //     scene_depth <= segment_near + thickness &&
        //     scene_depth >= segment_far - thickness;
        //
        // if (depth_overlap &&
        //     !all(cell == origin_pixel)) {
        //
        //     hit_pixel = cell;
        //     return true;
        // }


        const float ray_linear_a =
            LinearizeDepth(ray_depth);

        const float ray_linear_b =
            LinearizeDepth(exit_depth);

        const float segment_near =
            min(ray_linear_a, ray_linear_b);

        const float segment_far =
            max(ray_linear_a, ray_linear_b);

        // const float thickness =
        //     max(
        //         0.03,
        //         scene_linear * 0.002
        //     );

        const float thickness = 0.05;

        // const float thickness =
        //     clamp(
        //         scene_linear * 0.01,
        //         0.01,
        //         0.01
        //     );

        const bool depth_overlap =
            scene_linear >= segment_near - thickness &&
            scene_linear <= segment_far + thickness;

        // if (depth_overlap &&
        //     !all(cell == origin_pixel)) {
        //
        //     hit_pixel = cell;
        //     return true;
        // }

        if (depth_overlap &&
            !all(cell == origin_pixel)) {

            hit_pixel = cell;
            return true;
        }

        distance = exit_distance + max(1.0e-4, exit_distance * 1.0e-5);
        // distance = exit_distance + 0.01;

        level = min(1u, maximum_level);
    }

    return false;
}

[numthreads(8, 8, 1)]
void CS_ScreenTrace(uint3 dispatch_id : SV_DispatchThreadID) {

    const uint2 pixel = dispatch_id.xy;
    const uint2 resolution = uint2(TraceFrame.resolution.xy);

    if (pixel.x >= resolution.x || pixel.y >= resolution.y) return;

    const float depth = TraceDepth.Load(int3(pixel, 0));

    // Background doesn't launch a ray.
    if (depth <= 0.0) {
        ScreenTraceOutput[pixel] = float4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    const float3 position = ReconstructWorldPosition(pixel, depth);
    float3 normal = normalize(TraceNormalRoughness.Load(int3(pixel, 0)).xyz);
    const float3 view_direction = normalize(position - TraceFrame.camera_position.xyz);

    // Rasterization is two-sided right now.
    if (dot(normal, view_direction) > 0.0) normal = -normal;

    const float3 reflection = normalize(reflect(view_direction, normal));

    // ScreenTraceOutput[pixel] = float4(reflection * 0.5 + 0.5, 1.0);
    // return;

    const float camera_distance = length(position - TraceFrame.camera_position.xyz);
    // const float bias = max(0.02, camera_distance * 0.001);
    const float bias = 0.10;
    // const float3 origin = position + normal * bias + reflection * bias;
    const float3 origin = position + normal * bias;
    uint2 hit_pixel;

    if (TraceScreenRay(origin, reflection, pixel, hit_pixel)) {
        
        ScreenTraceOutput[pixel] = float4(TraceAlbedoMetallic.Load(int3(hit_pixel, 0)).rgb, 1.0);
        return;

        // const float2 hit_uv =
        //     (float2(hit_pixel) + 0.5) *
        //     TraceFrame.resolution.zw;
        //
        // ScreenTraceOutput[pixel] =
        //     float4(
        //         hit_uv.x,
        //         hit_uv.y,
        //         0.0,
        //         1.0
        //     );
        //
        // return;
    }

    ScreenTraceOutput[pixel] = float4(1.0, 0.0, 1.0, 1.0);
}

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

        const uint mip = min(4u, hzb_mip_count);

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
        
    #elif defined(SCREEN_TRACE)
        return float4(ScreenTraceTexture.Load(int3(pixel, 0)).rgb, 1.0);

    #else


    return float4(
        1.0,
        0.0,
        1.0,
        1.0
    );

    #endif
}


