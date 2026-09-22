#ifndef MSH_H
#define MSH_H

#include "sdl.h"

typedef struct Vertex {
    float position[3];
    float normal[3];
    float uv0[2];
} VERTEX;

typedef struct Mesh {
    SDL_GPUBuffer *vb;
    SDL_GPUBuffer *ib;
    uint32_t vcount;
    uint32_t icount;
} MESH;

static const VERTEX cube_v[] = {
    { { 0.5f, -0.5f, -0.5f }, { 1.0f, 0.0f, 0.0f }, { 1.0f, 0.0f } },
    { { 0.5f, 0.5f, -0.5f }, { 1.0f, 0.0f, 0.0f }, { 1.0f, 1.0f } },
    { { 0.5f, 0.5f, 0.5f }, { 1.0f, 0.0f, 0.0f }, { 0.0f, 1.0f } },
    { { 0.5f, -0.5f, 0.5f }, { 1.0f, 0.0f, 0.0f }, { 0.0f, 0.0f } },
    { { -0.5f, -0.5f, 0.5f }, { -1.0f, 0.0f, 0.0f }, { 1.0f, 0.0f } },
    { { -0.5f, 0.5f, 0.5f }, { -1.0f, 0.0f, 0.0f }, { 1.0f, 1.0f } },
    { { -0.5f, 0.5f, -0.5f }, { -1.0f, 0.0f, 0.0f }, { 0.0f, 1.0f } },
    { { -0.5f, -0.5f, -0.5f }, { -1.0f, 0.0f, 0.0f }, { 0.0f, 0.0f } },
    { { -0.5f, 0.5f, -0.5f }, { 0.0f, 1.0f, 0.0f }, { 1.0f, 0.0f } },
    { { -0.5f, 0.5f, 0.5f }, { 0.0f, 1.0f, 0.0f }, { 1.0f, 1.0f } },
    { { 0.5f, 0.5f, 0.5f }, { 0.0f, 1.0f, 0.0f }, { 0.0f, 1.0f } },
    { { 0.5f, 0.5f, -0.5f }, { 0.0f, 1.0f, 0.0f }, { 0.0f, 0.0f } },
    { { 0.5f, -0.5f, -0.5f }, { 0.0f, -1.0f, 0.0f }, { 1.0f, 0.0f } },
    { { 0.5f, -0.5f, 0.5f }, { 0.0f, -1.0f, 0.0f }, { 1.0f, 1.0f } },
    { { -0.5f, -0.5f, 0.5f }, { 0.0f, -1.0f, 0.0f }, { 0.0f, 1.0f } },
    { { -0.5f, -0.5f, -0.5f }, { 0.0f, -1.0f, 0.0f }, { 0.0f, 0.0f } },
    { { 0.5f, -0.5f, 0.5f }, { 0.0f, 0.0f, 1.0f }, { 1.0f, 0.0f } },
    { { 0.5f, 0.5f, 0.5f }, { 0.0f, 0.0f, 1.0f }, { 1.0f, 1.0f } },
    { { -0.5f, 0.5f, 0.5f }, { 0.0f, 0.0f, 1.0f }, { 0.0f, 1.0f } },
    { { -0.5f, -0.5f, 0.5f }, { 0.0f, 0.0f, 1.0f }, { 0.0f, 0.0f } },
    { { -0.5f, -0.5f, -0.5f }, { 0.0f, 0.0f, -1.0f }, { 1.0f, 0.0f } },
    { { -0.5f, 0.5f, -0.5f }, { 0.0f, 0.0f, -1.0f }, { 1.0f, 1.0f } },
    { { 0.5f, 0.5f, -0.5f }, { 0.0f, 0.0f, -1.0f }, { 0.0f, 1.0f } },
    { { 0.5f, -0.5f, -0.5f }, { 0.0f, 0.0f, -1.0f }, { 0.0f, 0.0f } }
};

static const uint32_t cube_i[] = {
    0, 1, 2, 0, 2, 3,
    4, 5, 6, 4, 6, 7,
    8, 9, 10, 8, 10, 11,
    12, 13, 14, 12, 14, 15,
    16, 17, 18, 16, 18, 19,
    20, 21, 22, 20, 22, 23
};

static const VERTEX plane_v[] = {
    { { -0.5f, 0.0f, -0.5f }, { 0.0f, 1.0f, 0.0f }, { 1.0f, 0.0f } },
    { { -0.5f, 0.0f, 0.5f }, { 0.0f, 1.0f, 0.0f }, { 1.0f, 1.0f } },
    { { 0.5f, 0.0f, 0.5f }, { 0.0f, 1.0f, 0.0f }, { 0.0f, 1.0f } },
    { { 0.5f, 0.0f, -0.5f }, { 0.0f, 1.0f, 0.0f }, { 0.0f, 0.0f } }
};

static const VERTEX wall_v[] = {
    { { 0.5f, -0.5f, 0.0f }, { 0.0f, 0.0f, 1.0f }, { 1.0f, 0.0f } },
    { { 0.5f, 0.5f, 0.0f }, { 0.0f, 0.0f, 1.0f }, { 1.0f, 1.0f } },
    { { -0.5f, 0.5f, 0.0f }, { 0.0f, 0.0f, 1.0f }, { 0.0f, 1.0f } },
    { { -0.5f, -0.5f, 0.0f }, { 0.0f, 0.0f, 1.0f }, { 0.0f, 0.0f } }
};

static const uint32_t quad_i[] = {
    0, 1, 2, 0, 2, 3
};

int m_load(SDL_GPUDevice *d, SDL_GPUBuffer *b, void *s, uint32_t n) {
    SDL_GPUTransferBufferCreateInfo tinfo = {
        .usage = SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD,
        .size = n
    };

    SDL_GPUTransferBuffer *t = SDL_CreateGPUTransferBuffer(d, &tinfo);
    if (!t) {
        SDL_Log("transfer buffer creation failed: %s", SDL_GetError());
        return 0;
    }

    void *map = SDL_MapGPUTransferBuffer(d, t, false);
    if (!map) {
        SDL_Log("transfer buffer map failed: %s", SDL_GetError());
        SDL_ReleaseGPUTransferBuffer(d, t);
        return 0;
    }

    SDL_memcpy(map, s, n);
    SDL_UnmapGPUTransferBuffer(d, t);

    SDL_GPUCommandBuffer *cmd = SDL_AcquireGPUCommandBuffer(d);
    if (!cmd) {
        SDL_Log("command buffer failed: %s", SDL_GetError());
        SDL_ReleaseGPUTransferBuffer(d, t);
        return 0;
    }

    SDL_GPUCopyPass *pass = SDL_BeginGPUCopyPass(cmd);

    SDL_GPUTransferBufferLocation src = {
        .transfer_buffer = t,
        .offset = 0
    };

    SDL_GPUBufferRegion dst = {
        .buffer = b,
        .offset = 0,
        .size = n
    };

    SDL_UploadToGPUBuffer(pass, &src, &dst, false);
    SDL_EndGPUCopyPass(pass);
    SDL_SubmitGPUCommandBuffer(cmd);
    SDL_ReleaseGPUTransferBuffer(d, t);
    return 1;
}

int m_make(SDL_GPUDevice *d, MESH *m, void *v, uint32_t vn, void *idx, uint32_t in, const char *vname, const char *iname) {
    m->vb = NULL;
    m->ib = NULL;
    m->vcount = vn;
    m->icount = in;

    SDL_GPUBufferCreateInfo vinfo = {
        .usage = SDL_GPU_BUFFERUSAGE_VERTEX,
        .size = vn * sizeof(VERTEX)
    };

    m->vb = SDL_CreateGPUBuffer(d, &vinfo);
    if (!m->vb) {
        SDL_Log("vertex buffer creation failed: %s", SDL_GetError());
        return 0;
    }
    SDL_SetGPUBufferName(d, m->vb, vname);

    SDL_GPUBufferCreateInfo iinfo = {
        .usage = SDL_GPU_BUFFERUSAGE_INDEX,
        .size = in * sizeof(uint32_t)
    };

    m->ib = SDL_CreateGPUBuffer(d, &iinfo);
    if (!m->ib) {
        SDL_Log("index buffer creation failed: %s", SDL_GetError());
        return 0;
    }
    SDL_SetGPUBufferName(d, m->ib, iname);

    if (!m_load(d, m->vb, v, vn * sizeof(VERTEX))) return 0;
    if (!m_load(d, m->ib, idx, in * sizeof(uint32_t))) return 0;
    return 1;
}

int m_init(SDL_GPUDevice *d, MESH *m) {
    return m_make(d, m, (void *)cube_v, sizeof(cube_v) / sizeof(cube_v[0]), (void *)cube_i, sizeof(cube_i) / sizeof(cube_i[0]), "Mesh.VB.Cube", "Mesh.IB.Cube");
}

int m_plane(SDL_GPUDevice *d, MESH *m) {
    return m_make(d, m, (void *)plane_v, sizeof(plane_v) / sizeof(plane_v[0]), (void *)quad_i, sizeof(quad_i) / sizeof(quad_i[0]), "Mesh.VB.Plane", "Mesh.IB.Plane");
}

int m_wall(SDL_GPUDevice *d, MESH *m) {
    return m_make(d, m, (void *)wall_v, sizeof(wall_v) / sizeof(wall_v[0]), (void *)quad_i, sizeof(quad_i) / sizeof(quad_i[0]), "Mesh.VB.Wall", "Mesh.IB.Wall");
}

void m_draw(SDL_GPURenderPass *p, MESH *m) {
    SDL_GPUBufferBinding vb = {
        .buffer = m->vb,
        .offset = 0
    };

    SDL_GPUBufferBinding ib = {
        .buffer = m->ib,
        .offset = 0
    };

    SDL_BindGPUVertexBuffers(p, 0, &vb, 1);
    SDL_BindGPUIndexBuffer(p, &ib, SDL_GPU_INDEXELEMENTSIZE_32BIT);
    SDL_DrawGPUIndexedPrimitives(p, m->icount, 1, 0, 0, 0);
}

void m_deinit(SDL_GPUDevice *d, MESH *m) {
    if (m->ib) {
        SDL_ReleaseGPUBuffer(d, m->ib);
        m->ib = NULL;
    }
    if (m->vb) {
        SDL_ReleaseGPUBuffer(d, m->vb);
        m->vb = NULL;
    }
}

#endif // MSH_H
