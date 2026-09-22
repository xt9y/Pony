#ifndef CMP_H
#define CMP_H

#include "sdl.h"

typedef struct Compute {
    SDL_GPUBuffer *buf;
} CMP;

int c_init(SDL_GPUDevice *d, CMP *c) {
    c->buf = NULL;

    SDL_GPUBufferCreateInfo info = {
        .usage = SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_WRITE,
        .size = 1024 * sizeof(uint32_t)
    };

    c->buf = SDL_CreateGPUBuffer(d, &info);
    if (!c->buf) {
        SDL_Log("compute buffer creation failed: %s", SDL_GetError());
        return 0;
    }

    SDL_SetGPUBufferName(d, c->buf, "Compute.Proof");
    return 1;
}

int c_run(SDL_GPUDevice *d, CMP *c, SDL_GPUComputePipeline *p) {
    SDL_GPUTransferBufferCreateInfo tinfo = {
        .usage = SDL_GPU_TRANSFERBUFFERUSAGE_DOWNLOAD,
        .size = 1024 * sizeof(uint32_t)
    };

    SDL_GPUTransferBuffer *t = SDL_CreateGPUTransferBuffer(d, &tinfo);
    if (!t) {
        SDL_Log("compute download buffer creation failed: %s", SDL_GetError());
        return 0;
    }

    SDL_GPUCommandBuffer *cmd = SDL_AcquireGPUCommandBuffer(d);
    if (!cmd) {
        SDL_Log("command buffer failed: %s", SDL_GetError());
        SDL_ReleaseGPUTransferBuffer(d, t);
        return 0;
    }

    SDL_GPUStorageBufferReadWriteBinding b = {
        .buffer = c->buf,
        .cycle = false
    };

    SDL_GPUComputePass *cpass = SDL_BeginGPUComputePass(cmd, NULL, 0, &b, 1);
    SDL_BindGPUComputePipeline(cpass, p);
    SDL_DispatchGPUCompute(cpass, 16, 1, 1);
    SDL_EndGPUComputePass(cpass);

    SDL_GPUCopyPass *cp = SDL_BeginGPUCopyPass(cmd);

    SDL_GPUBufferRegion src = {
        .buffer = c->buf,
        .offset = 0,
        .size = 1024 * sizeof(uint32_t)
    };

    SDL_GPUTransferBufferLocation dst = {
        .transfer_buffer = t,
        .offset = 0
    };

    SDL_DownloadFromGPUBuffer(cp, &src, &dst);
    SDL_EndGPUCopyPass(cp);

    SDL_GPUFence *f = SDL_SubmitGPUCommandBufferAndAcquireFence(cmd);
    if (!f) {
        SDL_Log("compute submit failed: %s", SDL_GetError());
        SDL_ReleaseGPUTransferBuffer(d, t);
        return 0;
    }

    if (!SDL_WaitForGPUFences(d, true, &f, 1)) {
        SDL_Log("compute fence wait failed: %s", SDL_GetError());
        SDL_ReleaseGPUFence(d, f);
        SDL_ReleaseGPUTransferBuffer(d, t);
        return 0;
    }
    SDL_ReleaseGPUFence(d, f);

    uint32_t *v = SDL_MapGPUTransferBuffer(d, t, false);
    if (!v) {
        SDL_Log("compute download map failed: %s", SDL_GetError());
        SDL_ReleaseGPUTransferBuffer(d, t);
        return 0;
    }

    int ok = v[0] == 0 && v[1] == 2 && v[511] == 1022 && v[1023] == 2046;
    SDL_UnmapGPUTransferBuffer(d, t);
    SDL_ReleaseGPUTransferBuffer(d, t);

    if (!ok) {
        SDL_Log("compute proof mismatch");
        return 0;
    }

    SDL_Log("compute proof ok");
    return 1;
}

void c_deinit(SDL_GPUDevice *d, CMP *c) {
    if (c->buf) {
        SDL_ReleaseGPUBuffer(d, c->buf);
        c->buf = NULL;
    }
}

#endif // CMP_H
