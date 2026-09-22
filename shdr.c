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

static const char fs_src[] =
    "struct FSIn\n"
    "{\n"
    "    float4 position : SV_Position;\n"
    "    float3 normal : TEXCOORD0;\n"
    "    float2 uv : TEXCOORD1;\n"
    "};\n"
    "float4 main(FSIn i) : SV_Target\n"
    "{\n"
    "    float3 c = i.normal * 0.5 + 0.5;\n"
    "    return float4(c, 1.0);\n"
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
    SDL_GPUGraphicsPipeline *gfx;
    SDL_GPUComputePipeline *cmp;
} PIPE;

SDL_GPUShader *s_gfx(SDL_GPUDevice *d, const char *src, SDL_ShaderCross_ShaderStage stage) {
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
    if (!s) {
        SDL_Log("graphics shader compile failed: %s", SDL_GetError());
    }

    SDL_free(meta);
    SDL_free(spirv);
    return s;
}

SDL_GPUComputePipeline *s_cmp(SDL_GPUDevice *d, const char *src) {
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
    if (!p) {
        SDL_Log("compute pipeline compile failed: %s", SDL_GetError());
    }

    SDL_free(meta);
    SDL_free(spirv);
    return p;
}

int p_init(RENDERER *r, PIPE *p) {
    p->vs = NULL;
    p->fs = NULL;
    p->gfx = NULL;
    p->cmp = NULL;

    p->vs = s_gfx(r->device, vs_src, SDL_SHADERCROSS_SHADERSTAGE_VERTEX);
    if (!p->vs) {
        return 0;
    }

    p->fs = s_gfx(r->device, fs_src, SDL_SHADERCROSS_SHADERSTAGE_FRAGMENT);
    if (!p->fs) {
        return 0;
    }

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

    SDL_GPUColorTargetDescription cdesc = {
        .format = r->swp_format
    };

    SDL_GPUGraphicsPipelineCreateInfo info = {
        .vertex_shader = p->vs,
        .fragment_shader = p->fs,
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
        .multisample_state = {
            .sample_count = SDL_GPU_SAMPLECOUNT_1
        },
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

    p->gfx = SDL_CreateGPUGraphicsPipeline(r->device, &info);
    if (!p->gfx) {
        SDL_Log("graphics pipeline creation failed: %s", SDL_GetError());
        return 0;
    }

    p->cmp = s_cmp(r->device, cs_src);
    if (!p->cmp) {
        return 0;
    }

    return 1;
}

void p_deinit(SDL_GPUDevice *d, PIPE *p) {
    if (p->cmp) {
        SDL_ReleaseGPUComputePipeline(d, p->cmp);
        p->cmp = NULL;
    }
    if (p->gfx) {
        SDL_ReleaseGPUGraphicsPipeline(d, p->gfx);
        p->gfx = NULL;
    }
    if (p->fs) {
        SDL_ReleaseGPUShader(d, p->fs);
        p->fs = NULL;
    }
    if (p->vs) {
        SDL_ReleaseGPUShader(d, p->vs);
        p->vs = NULL;
    }
}

#endif // SHDR_H
