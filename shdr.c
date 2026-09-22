#ifndef SHDR_H
#define SHDR_H

#include "sdl.h"

#include <stddef.h>

static const char vs_src[] =
    "cbuffer Camera : register(b0, space1)\n"
    "{\n"
    "    float4x4 view_proj;\n"
    "};\n"
    "struct VSIn\n"
    "{\n"
    "    float3 position : TEXCOORD0;\n"
    "    float3 normal : TEXCOORD1;\n"
    "    float2 uv : TEXCOORD2;\n"
    "};\n"
    "struct VSOut\n"
    "{\n"
    "    float4 position : SV_Position;\n"
    "    float3 normal : TEXCOORD0;\n"
    "    float2 uv : TEXCOORD1;\n"
    "};\n"
    "VSOut main(VSIn i)\n"
    "{\n"
    "    VSOut o;\n"
    "    o.position = mul(view_proj, float4(i.position, 1.0));\n"
    "    o.normal = i.normal;\n"
    "    o.uv = i.uv;\n"
    "    return o;\n"
    "}\n";

static const char fs_normal_src[] =
    "struct FSIn\n"
    "{\n"
    "    float4 position : SV_Position;\n"
    "    float3 normal : TEXCOORD0;\n"
    "    float2 uv : TEXCOORD1;\n"
    "};\n"
    "float4 main(FSIn i) : SV_Target\n"
    "{\n"
    "    return float4(normalize(i.normal) * 0.5 + 0.5, 1.0);\n"
    "}\n";

static const char fs_uv_src[] =
    "struct FSIn\n"
    "{\n"
    "    float4 position : SV_Position;\n"
    "    float3 normal : TEXCOORD0;\n"
    "    float2 uv : TEXCOORD1;\n"
    "};\n"
    "float4 main(FSIn i) : SV_Target\n"
    "{\n"
    "    float2 cell = floor(i.uv * 12.0);\n"
    "    float checker = fmod(cell.x + cell.y, 2.0);\n"
    "    float3 a = float3(0.08, 0.08, 0.08);\n"
    "    float3 b = float3(0.85, 0.85, 0.85);\n"
    "    return float4(lerp(a, b, checker), 1.0);\n"
    "}\n";

static const char fs_triangle_src[] =
    "struct FSIn\n"
    "{\n"
    "    float4 position : SV_Position;\n"
    "    nointerpolation float triangle_id : TEXCOORD0;\n"
    "};\n"
    "uint hash_u32(uint x)\n"
    "{\n"
    "    x ^= x >> 16;\n"
    "    x *= 0x7feb352du;\n"
    "    x ^= x >> 15;\n"
    "    x *= 0x846ca68bu;\n"
    "    x ^= x >> 16;\n"
    "    return x;\n"
    "}\n"
    "float4 main(FSIn i) : SV_Target\n"
    "{\n"
    "    uint h = hash_u32((uint)i.triangle_id + 1u);\n"
    "    float3 c = float3((h & 255u), ((h >> 8) & 255u), ((h >> 16) & 255u)) / 255.0;\n"
    "    return float4(c, 1.0);\n"
    "}\n";

static const char triangle_vs_src[] =
    "cbuffer Camera : register(b0, space1)\n"
    "{\n"
    "    float4x4 view_proj;\n"
    "};\n"
    "struct VSIn\n"
    "{\n"
    "    float3 position : TEXCOORD0;\n"
    "    float3 normal : TEXCOORD1;\n"
    "    float2 uv : TEXCOORD2;\n"
    "    float triangle_id : TEXCOORD3;\n"
    "};\n"
    "struct VSOut\n"
    "{\n"
    "    float4 position : SV_Position;\n"
    "    nointerpolation float triangle_id : TEXCOORD0;\n"
    "};\n"
    "VSOut main(VSIn i)\n"
    "{\n"
    "    VSOut o;\n"
    "    o.position = mul(view_proj, float4(i.position, 1.0));\n"
    "    o.triangle_id = i.triangle_id;\n"
    "    return o;\n"
    "}\n";

static const char bounds_vs_src[] =
    "cbuffer Camera : register(b0, space1)\n"
    "{\n"
    "    float4x4 view_proj;\n"
    "};\n"
    "static const float3 kLines[24] = {\n"
    "    float3(-0.5,-0.5,-0.5), float3( 0.5,-0.5,-0.5),\n"
    "    float3( 0.5,-0.5,-0.5), float3( 0.5, 0.5,-0.5),\n"
    "    float3( 0.5, 0.5,-0.5), float3(-0.5, 0.5,-0.5),\n"
    "    float3(-0.5, 0.5,-0.5), float3(-0.5,-0.5,-0.5),\n"
    "    float3(-0.5,-0.5, 0.5), float3( 0.5,-0.5, 0.5),\n"
    "    float3( 0.5,-0.5, 0.5), float3( 0.5, 0.5, 0.5),\n"
    "    float3( 0.5, 0.5, 0.5), float3(-0.5, 0.5, 0.5),\n"
    "    float3(-0.5, 0.5, 0.5), float3(-0.5,-0.5, 0.5),\n"
    "    float3(-0.5,-0.5,-0.5), float3(-0.5,-0.5, 0.5),\n"
    "    float3( 0.5,-0.5,-0.5), float3( 0.5,-0.5, 0.5),\n"
    "    float3( 0.5, 0.5,-0.5), float3( 0.5, 0.5, 0.5),\n"
    "    float3(-0.5, 0.5,-0.5), float3(-0.5, 0.5, 0.5)\n"
    "};\n"
    "float4 main(uint vertex_id : SV_VertexID) : SV_Position\n"
    "{\n"
    "    return mul(view_proj, float4(kLines[vertex_id], 1.0));\n"
    "}\n";

static const char bounds_fs_src[] =
    "float4 main() : SV_Target\n"
    "{\n"
    "    return float4(0.1, 1.0, 0.2, 1.0);\n"
    "}\n";

static const char cs_src[] =
    "RWStructuredBuffer<uint> Output : register(u0, space1);\n"
    "[numthreads(64, 1, 1)]\n"
    "void main(uint3 tid : SV_DispatchThreadID)\n"
    "{\n"
    "    if (tid.x < 1024)\n"
    "        Output[tid.x] = tid.x * 2;\n"
    "}\n";

typedef struct Pipe {
    SDL_GPUShader *vs;
    SDL_GPUShader *fs;
    SDL_GPUShader *fs_uv;
    SDL_GPUShader *fs_triangle;
    SDL_GPUShader *triangle_vs;
    SDL_GPUShader *bounds_vs;
    SDL_GPUShader *bounds_fs;

    SDL_GPUGraphicsPipeline *gfx;
    SDL_GPUGraphicsPipeline *uv;
    SDL_GPUGraphicsPipeline *triangle;
    SDL_GPUGraphicsPipeline *bounds;

    SDL_GPUComputePipeline *cmp;
} PIPE;

SDL_GPUShader *s_gfx(SDL_GPUDevice *d, const char *src, SDL_ShaderCross_ShaderStage stage)
{
    SDL_ShaderCross_HLSL_Info hinfo = {
        .source = src,
        .entrypoint = "main",
        .include_dir = NULL,
        .defines = NULL,
        .shader_stage = stage,
        .props = 0
    };

    size_t n = 0;
    void *spirv = SDL_ShaderCross_CompileSPIRVFromHLSL(&hinfo, &n);
    if (!spirv) {
        SDL_Log("hlsl to spirv failed: %s", SDL_GetError());
        return NULL;
    }

    SDL_ShaderCross_GraphicsShaderMetadata *meta = SDL_ShaderCross_ReflectGraphicsSPIRV(spirv, n, 0);
    if (!meta) {
        SDL_Log("graphics reflect failed: %s", SDL_GetError());
        SDL_free(spirv);
        return NULL;
    }

    SDL_ShaderCross_SPIRV_Info sinfo = {
        .bytecode = spirv,
        .bytecode_size = n,
        .entrypoint = "main",
        .shader_stage = stage,
        .props = 0
    };

    SDL_GPUShader *s = SDL_ShaderCross_CompileGraphicsShaderFromSPIRV(d, &sinfo, &meta->resource_info, 0);
    if (!s) SDL_Log("graphics shader compile failed: %s", SDL_GetError());

    SDL_free(meta);
    SDL_free(spirv);
    return s;
}

SDL_GPUComputePipeline *s_cmp(SDL_GPUDevice *d, const char *src)
{
    SDL_ShaderCross_HLSL_Info hinfo = {
        .source = src,
        .entrypoint = "main",
        .include_dir = NULL,
        .defines = NULL,
        .shader_stage = SDL_SHADERCROSS_SHADERSTAGE_COMPUTE,
        .props = 0
    };

    size_t n = 0;
    void *spirv = SDL_ShaderCross_CompileSPIRVFromHLSL(&hinfo, &n);
    if (!spirv) {
        SDL_Log("hlsl to spirv failed: %s", SDL_GetError());
        return NULL;
    }

    SDL_ShaderCross_ComputePipelineMetadata *meta = SDL_ShaderCross_ReflectComputeSPIRV(spirv, n, 0);
    if (!meta) {
        SDL_Log("compute reflect failed: %s", SDL_GetError());
        SDL_free(spirv);
        return NULL;
    }

    SDL_ShaderCross_SPIRV_Info sinfo = {
        .bytecode = spirv,
        .bytecode_size = n,
        .entrypoint = "main",
        .shader_stage = SDL_SHADERCROSS_SHADERSTAGE_COMPUTE,
        .props = 0
    };

    SDL_GPUComputePipeline *p = SDL_ShaderCross_CompileComputePipelineFromSPIRV(d, &sinfo, meta, 0);
    if (!p) SDL_Log("compute pipeline compile failed: %s", SDL_GetError());

    SDL_free(meta);
    SDL_free(spirv);
    return p;
}

static SDL_GPUGraphicsPipeline *p_mesh_pipeline(RENDERER *r, SDL_GPUShader *vs, SDL_GPUShader *fs)
{
    SDL_GPUVertexBufferDescription vb = {
        .slot = 0,
        .pitch = sizeof(VERTEX),
        .input_rate = SDL_GPU_VERTEXINPUTRATE_VERTEX,
        .instance_step_rate = 0
    };

    SDL_GPUVertexAttribute attrs[3] = {
        { .location = 0, .buffer_slot = 0, .format = SDL_GPU_VERTEXELEMENTFORMAT_FLOAT3, .offset = offsetof(VERTEX, position) },
        { .location = 1, .buffer_slot = 0, .format = SDL_GPU_VERTEXELEMENTFORMAT_FLOAT3, .offset = offsetof(VERTEX, normal) },
        { .location = 2, .buffer_slot = 0, .format = SDL_GPU_VERTEXELEMENTFORMAT_FLOAT2, .offset = offsetof(VERTEX, uv0) }
    };

    SDL_GPUColorTargetDescription cdesc = {.format = r->swp_format};

    SDL_GPUGraphicsPipelineCreateInfo info = {
        .vertex_shader = vs,
        .fragment_shader = fs,
        .vertex_input_state = {
            .vertex_buffer_descriptions = &vb,
            .num_vertex_buffers = 1,
            .vertex_attributes = attrs,
            .num_vertex_attributes = 3
        },
        .primitive_type = SDL_GPU_PRIMITIVETYPE_TRIANGLELIST,
        .rasterizer_state = {
            .fill_mode = SDL_GPU_FILLMODE_FILL,
            .cull_mode = SDL_GPU_CULLMODE_BACK,
            .front_face = SDL_GPU_FRONTFACE_COUNTER_CLOCKWISE
        },
        .multisample_state = {.sample_count = SDL_GPU_SAMPLECOUNT_1},
        .depth_stencil_state = {
            .compare_op = SDL_GPU_COMPAREOP_LESS,
            .enable_depth_test = true,
            .enable_depth_write = true
        },
        .target_info = {
            .color_target_descriptions = &cdesc,
            .num_color_targets = 1,
            .depth_stencil_format = r->depth_format,
            .has_depth_stencil_target = true
        }
    };

    return SDL_CreateGPUGraphicsPipeline(r->device, &info);
}

static SDL_GPUGraphicsPipeline *p_bounds_pipeline(RENDERER *r, SDL_GPUShader *vs, SDL_GPUShader *fs)
{
    SDL_GPUColorTargetDescription cdesc = {.format = r->swp_format};
    SDL_GPUGraphicsPipelineCreateInfo info = {
        .vertex_shader = vs,
        .fragment_shader = fs,
        .primitive_type = SDL_GPU_PRIMITIVETYPE_LINELIST,
        .rasterizer_state = {
            .fill_mode = SDL_GPU_FILLMODE_FILL,
            .cull_mode = SDL_GPU_CULLMODE_NONE,
            .front_face = SDL_GPU_FRONTFACE_COUNTER_CLOCKWISE
        },
        .multisample_state = {.sample_count = SDL_GPU_SAMPLECOUNT_1},
        .depth_stencil_state = {
            .compare_op = SDL_GPU_COMPAREOP_ALWAYS,
            .enable_depth_test = false,
            .enable_depth_write = false
        },
        .target_info = {
            .color_target_descriptions = &cdesc,
            .num_color_targets = 1,
            .depth_stencil_format = r->depth_format,
            .has_depth_stencil_target = true
        }
    };
    return SDL_CreateGPUGraphicsPipeline(r->device, &info);
}

static SDL_GPUGraphicsPipeline *p_triangle_pipeline(RENDERER *r, SDL_GPUShader *vs, SDL_GPUShader *fs)
{
    SDL_GPUVertexBufferDescription vb = {
        .slot = 0,
        .pitch = sizeof(DEBUG_TRIANGLE_VERTEX),
        .input_rate = SDL_GPU_VERTEXINPUTRATE_VERTEX,
        .instance_step_rate = 0
    };
    SDL_GPUVertexAttribute attrs[4] = {
        { .location = 0, .buffer_slot = 0, .format = SDL_GPU_VERTEXELEMENTFORMAT_FLOAT3, .offset = offsetof(DEBUG_TRIANGLE_VERTEX, vertex.position) },
        { .location = 1, .buffer_slot = 0, .format = SDL_GPU_VERTEXELEMENTFORMAT_FLOAT3, .offset = offsetof(DEBUG_TRIANGLE_VERTEX, vertex.normal) },
        { .location = 2, .buffer_slot = 0, .format = SDL_GPU_VERTEXELEMENTFORMAT_FLOAT2, .offset = offsetof(DEBUG_TRIANGLE_VERTEX, vertex.uv0) },
        { .location = 3, .buffer_slot = 0, .format = SDL_GPU_VERTEXELEMENTFORMAT_FLOAT, .offset = offsetof(DEBUG_TRIANGLE_VERTEX, triangle_id) }
    };
    SDL_GPUColorTargetDescription cdesc = {.format = r->swp_format};
    SDL_GPUGraphicsPipelineCreateInfo info = {
        .vertex_shader = vs,
        .fragment_shader = fs,
        .vertex_input_state = {
            .vertex_buffer_descriptions = &vb,
            .num_vertex_buffers = 1,
            .vertex_attributes = attrs,
            .num_vertex_attributes = 4
        },
        .primitive_type = SDL_GPU_PRIMITIVETYPE_TRIANGLELIST,
        .rasterizer_state = {
            .fill_mode = SDL_GPU_FILLMODE_FILL,
            .cull_mode = SDL_GPU_CULLMODE_BACK,
            .front_face = SDL_GPU_FRONTFACE_COUNTER_CLOCKWISE
        },
        .multisample_state = {.sample_count = SDL_GPU_SAMPLECOUNT_1},
        .depth_stencil_state = {
            .compare_op = SDL_GPU_COMPAREOP_LESS,
            .enable_depth_test = true,
            .enable_depth_write = true
        },
        .target_info = {
            .color_target_descriptions = &cdesc,
            .num_color_targets = 1,
            .depth_stencil_format = r->depth_format,
            .has_depth_stencil_target = true
        }
    };
    return SDL_CreateGPUGraphicsPipeline(r->device, &info);
}

int p_init(RENDERER *r, PIPE *p)
{
    SDL_memset(p, 0, sizeof(*p));

    p->vs = s_gfx(r->device, vs_src, SDL_SHADERCROSS_SHADERSTAGE_VERTEX);
    p->fs = s_gfx(r->device, fs_normal_src, SDL_SHADERCROSS_SHADERSTAGE_FRAGMENT);
    p->fs_uv = s_gfx(r->device, fs_uv_src, SDL_SHADERCROSS_SHADERSTAGE_FRAGMENT);
    p->fs_triangle = s_gfx(r->device, fs_triangle_src, SDL_SHADERCROSS_SHADERSTAGE_FRAGMENT);
    p->triangle_vs = s_gfx(r->device, triangle_vs_src, SDL_SHADERCROSS_SHADERSTAGE_VERTEX);
    p->bounds_vs = s_gfx(r->device, bounds_vs_src, SDL_SHADERCROSS_SHADERSTAGE_VERTEX);
    p->bounds_fs = s_gfx(r->device, bounds_fs_src, SDL_SHADERCROSS_SHADERSTAGE_FRAGMENT);
    if (!p->vs || !p->fs || !p->fs_uv || !p->fs_triangle || !p->triangle_vs || !p->bounds_vs || !p->bounds_fs) return 0;

    p->gfx = p_mesh_pipeline(r, p->vs, p->fs);
    p->uv = p_mesh_pipeline(r, p->vs, p->fs_uv);
    p->triangle = p_triangle_pipeline(r, p->triangle_vs, p->fs_triangle);
    p->bounds = p_bounds_pipeline(r, p->bounds_vs, p->bounds_fs);
    if (!p->gfx || !p->uv || !p->triangle || !p->bounds) {
        SDL_Log("graphics pipeline creation failed: %s", SDL_GetError());
        return 0;
    }

    p->cmp = s_cmp(r->device, cs_src);
    return p->cmp != NULL;
}

void p_deinit(SDL_GPUDevice *d, PIPE *p)
{
    if (p->cmp) SDL_ReleaseGPUComputePipeline(d, p->cmp);
    if (p->bounds) SDL_ReleaseGPUGraphicsPipeline(d, p->bounds);
    if (p->triangle) SDL_ReleaseGPUGraphicsPipeline(d, p->triangle);
    if (p->uv) SDL_ReleaseGPUGraphicsPipeline(d, p->uv);
    if (p->gfx) SDL_ReleaseGPUGraphicsPipeline(d, p->gfx);
    if (p->bounds_fs) SDL_ReleaseGPUShader(d, p->bounds_fs);
    if (p->bounds_vs) SDL_ReleaseGPUShader(d, p->bounds_vs);
    if (p->fs_triangle) SDL_ReleaseGPUShader(d, p->fs_triangle);
    if (p->triangle_vs) SDL_ReleaseGPUShader(d, p->triangle_vs);
    if (p->fs_uv) SDL_ReleaseGPUShader(d, p->fs_uv);
    if (p->fs) SDL_ReleaseGPUShader(d, p->fs);
    if (p->vs) SDL_ReleaseGPUShader(d, p->vs);
    SDL_memset(p, 0, sizeof(*p));
}

#endif // SHDR_H
