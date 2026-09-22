#ifndef REN_H
#define REN_H

#include "sdl.h"

#include <math.h>

typedef struct Renderer {
    SDL_Window *win;
    SDL_GPUDevice *device;
    SDL_GPUTextureFormat swp_format;
    SDL_GPUTexture *depth;
    SDL_GPUTextureFormat depth_format;
    uint32_t w;
    uint32_t h;
} RENDERER;

typedef struct RenderTarget {
    SDL_GPUTexture *color;
    SDL_GPUTexture *depth;
    uint32_t w;
    uint32_t h;
} RTGT;

int r_depth(RENDERER *r, uint32_t w, uint32_t h) {
    if (r->depth) {
        SDL_ReleaseGPUTexture(r->device, r->depth);
        r->depth = NULL;
    }

    SDL_GPUTextureCreateInfo info = {
        .type = SDL_GPU_TEXTURETYPE_2D,
        .format = r->depth_format,
        .usage = SDL_GPU_TEXTUREUSAGE_DEPTH_STENCIL_TARGET,
        .width = w,
        .height = h,
        .layer_count_or_depth = 1,
        .num_levels = 1,
        .sample_count = SDL_GPU_SAMPLECOUNT_1
    };

    r->depth = SDL_CreateGPUTexture(r->device, &info);
    if (!r->depth) {
        SDL_Log("depth texture creation failed: %s", SDL_GetError());
        return 0;
    }

    SDL_SetGPUTextureName(r->device, r->depth, "Main.Depth");
    return 1;
}

int r_init(RENDERER *r) {
    r->win = NULL;
    r->device = NULL;
    r->depth = NULL;

    if (!SDL_Init(SDL_INIT_VIDEO)) {
        SDL_Log("SDL_Init failed: %s", SDL_GetError());
        return 0;
    }

    r->win = SDL_CreateWindow("Untitled", r->w, r->h, SDL_WINDOW_RESIZABLE);
    if (!r->win) {
        SDL_Log("window creation failed: %s", SDL_GetError());
        return 0;
    }

    if (!SDL_ShaderCross_Init()) {
        SDL_Log("shadercross init failed: %s", SDL_GetError());
        return 0;
    }

    SDL_GPUShaderFormat formats = SDL_ShaderCross_GetHLSLShaderFormats();

    r->device = SDL_CreateGPUDevice(formats, true, NULL);
    if (!r->device) {
        SDL_Log("GPU device creation failed: %s", SDL_GetError());
        return 0;
    }

    if (!SDL_ClaimWindowForGPUDevice(r->device, r->win)) {
        SDL_Log("claim window failed: %s", SDL_GetError());
        return 0;
    }

    r->swp_format = SDL_GetGPUSwapchainTextureFormat(r->device, r->win);

    r->depth_format = SDL_GPU_TEXTUREFORMAT_D16_UNORM;
    if (SDL_GPUTextureSupportsFormat(r->device, SDL_GPU_TEXTUREFORMAT_D32_FLOAT, SDL_GPU_TEXTURETYPE_2D, SDL_GPU_TEXTUREUSAGE_DEPTH_STENCIL_TARGET)) {
        r->depth_format = SDL_GPU_TEXTUREFORMAT_D32_FLOAT;
    }

    if (!r_depth(r, r->w, r->h)) {
        return 0;
    }

    return 1;
}

int r_resize(RENDERER *r, uint32_t w, uint32_t h) {
    if (w == r->w && h == r->h) {
        return 1;
    }

    r->w = w;
    r->h = h;
    return r_depth(r, w, h);
}

int rt_init(RENDERER *r, RTGT *t, uint32_t w, uint32_t h) {
    t->color = NULL;
    t->depth = NULL;
    t->w = w;
    t->h = h;

    SDL_GPUTextureCreateInfo cinfo = {
        .type = SDL_GPU_TEXTURETYPE_2D,
        .format = SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM,
        .usage = SDL_GPU_TEXTUREUSAGE_COLOR_TARGET | SDL_GPU_TEXTUREUSAGE_SAMPLER,
        .width = w,
        .height = h,
        .layer_count_or_depth = 1,
        .num_levels = 1,
        .sample_count = SDL_GPU_SAMPLECOUNT_1
    };

    t->color = SDL_CreateGPUTexture(r->device, &cinfo);
    if (!t->color) {
        SDL_Log("offscreen color creation failed: %s", SDL_GetError());
        return 0;
    }

    SDL_GPUTextureCreateInfo dinfo = {
        .type = SDL_GPU_TEXTURETYPE_2D,
        .format = r->depth_format,
        .usage = SDL_GPU_TEXTUREUSAGE_DEPTH_STENCIL_TARGET,
        .width = w,
        .height = h,
        .layer_count_or_depth = 1,
        .num_levels = 1,
        .sample_count = SDL_GPU_SAMPLECOUNT_1
    };

    t->depth = SDL_CreateGPUTexture(r->device, &dinfo);
    if (!t->depth) {
        SDL_Log("offscreen depth creation failed: %s", SDL_GetError());
        SDL_ReleaseGPUTexture(r->device, t->color);
        t->color = NULL;
        return 0;
    }

    SDL_SetGPUTextureName(r->device, t->color, "RTGT.Color");
    SDL_SetGPUTextureName(r->device, t->depth, "RTGT.Depth");
    return 1;
}

void rt_deinit(RENDERER *r, RTGT *t) {
    if (t->depth) {
        SDL_ReleaseGPUTexture(r->device, t->depth);
        t->depth = NULL;
    }
    if (t->color) {
        SDL_ReleaseGPUTexture(r->device, t->color);
        t->color = NULL;
    }
}

void r_deinit(RENDERER *r) {
    if (r->device) {
        if (r->depth) {
            SDL_ReleaseGPUTexture(r->device, r->depth);
            r->depth = NULL;
        }
        if (r->win) {
            SDL_ReleaseWindowFromGPUDevice(r->device, r->win);
        }
        SDL_DestroyGPUDevice(r->device);
        r->device = NULL;
    }
    SDL_ShaderCross_Quit();
    if (r->win) {
        SDL_DestroyWindow(r->win);
        r->win = NULL;
    }
    SDL_Quit();
}

void r_ident(float *m) {
    for (int i = 0; i < 16; i++) {
        m[i] = 0.0f;
    }
    m[0] = 1.0f;
    m[5] = 1.0f;
    m[10] = 1.0f;
    m[15] = 1.0f;
}

void r_mul(float *o, const float *a, const float *b) {
    float t[16];
    for (int c = 0; c < 4; c++) {
        for (int r = 0; r < 4; r++) {
            t[c * 4 + r] = a[r] * b[c * 4] + a[4 + r] * b[c * 4 + 1] + a[8 + r] * b[c * 4 + 2] + a[12 + r] * b[c * 4 + 3];
        }
    }
    for (int i = 0; i < 16; i++) {
        o[i] = t[i];
    }
}

void r_persp(float *m, float fov, float aspect, float n, float f) {
    float t = 1.0f / tanf(fov * 0.5f);
    for (int i = 0; i < 16; i++) {
        m[i] = 0.0f;
    }
    m[0] = t / aspect;
    m[5] = t;
    m[10] = f / (n - f);
    m[11] = -1.0f;
    m[14] = -(f * n) / (f - n);
}

void r_view(float *m, float x, float y, float z) {
    r_ident(m);
    m[12] = -x;
    m[13] = -y;
    m[14] = -z;
}

void r_look(float *m, float *e, float *c, float *u) {
    float f[3] = { c[0] - e[0], c[1] - e[1], c[2] - e[2] };
    float n = sqrtf(f[0] * f[0] + f[1] * f[1] + f[2] * f[2]);
    f[0] /= n;
    f[1] /= n;
    f[2] /= n;

    float s[3] = { f[1] * u[2] - f[2] * u[1], f[2] * u[0] - f[0] * u[2], f[0] * u[1] - f[1] * u[0] };
    n = sqrtf(s[0] * s[0] + s[1] * s[1] + s[2] * s[2]);
    s[0] /= n;
    s[1] /= n;
    s[2] /= n;

    float v[3] = { s[1] * f[2] - s[2] * f[1], s[2] * f[0] - s[0] * f[2], s[0] * f[1] - s[1] * f[0] };

    r_ident(m);
    m[0] = s[0];
    m[1] = v[0];
    m[2] = -f[0];
    m[4] = s[1];
    m[5] = v[1];
    m[6] = -f[1];
    m[8] = s[2];
    m[9] = v[2];
    m[10] = -f[2];
    m[12] = -(s[0] * e[0] + s[1] * e[1] + s[2] * e[2]);
    m[13] = -(v[0] * e[0] + v[1] * e[1] + v[2] * e[2]);
    m[14] = f[0] * e[0] + f[1] * e[1] + f[2] * e[2];
}

void r_xform(float *m, float x, float y, float z, float a, float sx, float sy, float sz) {
    float c = cosf(a);
    float s = sinf(a);
    r_ident(m);
    m[0] = c * sx;
    m[2] = -s * sx;
    m[5] = sy;
    m[8] = s * sz;
    m[10] = c * sz;
    m[12] = x;
    m[13] = y;
    m[14] = z;
}

#endif // REN_H
