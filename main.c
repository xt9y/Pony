#include <stdio.h>
#include <stdlib.h>

#define CHECK(expr, ...)                                    \
    do {                                                    \
        if (!(expr)) {                                      \
            fprintf(stderr, "CHECK failed: %s", #expr);    \
            fprintf(stderr, ": " __VA_ARGS__);             \
            fprintf(stderr, "\n");                         \
            ret = -1;                                       \
            goto deinit;                                    \
        }                                                   \
    } while (0)

#include "ren.c"
#include "msh.c"
#include "shdr.c"
#include "cmp.c"
#include "inpt.c"

enum {
    MESH_CUBE,
    MESH_FLOOR,
    MESH_WALL,
    MESH_COUNT
};

static int setup_instance(
    RENDER_INSTANCE *instance,
    const MESH *mesh,
    MESH_HANDLE handle,
    uint32_t entity_id,
    MOBILITY mobility,
    float x, float y, float z,
    float angle,
    float sx, float sy, float sz)
{
    instance->mesh = handle;
    instance->mobility = mobility;
    instance->material_id = 0;
    instance->entity_id = entity_id;
    r_xform(instance->transform.matrix, x, y, z, angle, sx, sy, sz);
    return m_world_bounds(mesh->local_bounds, instance->transform.matrix, &instance->world_bounds);
}

int main(void)
{
    RENDERER renderer = {.w = 1270, .h = 750};
    MESH meshes[MESH_COUNT] = {0};
    RENDER_INSTANCE instances[6] = {0};
    PIPE pipe = {0};
    CMP cmp = {0};
    int ret = 0;

    CHECK(r_init(&renderer), "renderer init");
    CHECK(m_init(renderer.device, &meshes[MESH_CUBE]), "cube mesh");
    CHECK(m_plane(renderer.device, &meshes[MESH_FLOOR]), "floor mesh");
    CHECK(m_wall(renderer.device, &meshes[MESH_WALL]), "wall mesh");
    CHECK(p_init(&renderer, &pipe), "pipelines");
    CHECK(c_init(renderer.device, &cmp), "compute buffer");
    CHECK(c_run(renderer.device, &cmp, pipe.cmp), "compute proof");

    CHECK(setup_instance(&instances[0], &meshes[MESH_FLOOR], MESH_FLOOR, 0, MOBILITY_STATIC,
        0.0f, -1.0f, 0.0f, 0.0f, 12.0f, 1.0f, 12.0f), "floor instance");
    CHECK(setup_instance(&instances[1], &meshes[MESH_WALL], MESH_WALL, 1, MOBILITY_STATIC,
        0.0f, 2.0f, -3.0f, 0.0f, 12.0f, 6.0f, 1.0f), "wall instance");
    CHECK(setup_instance(&instances[2], &meshes[MESH_CUBE], MESH_CUBE, 2, MOBILITY_STATIC,
        -1.2f, -0.5f, 0.5f, 0.3f, 1.0f, 1.0f, 1.0f), "cube instance 0");
    CHECK(setup_instance(&instances[3], &meshes[MESH_CUBE], MESH_CUBE, 3, MOBILITY_STATIC,
        1.0f, -0.5f, -0.8f, -0.4f, 1.0f, 1.0f, 1.0f), "cube instance 1");
    CHECK(setup_instance(&instances[4], &meshes[MESH_CUBE], MESH_CUBE, 4, MOBILITY_STATIC,
        -1.2f, 0.25f, 0.5f, 0.3f, 0.5f, 0.5f, 0.5f), "cube instance 2");
    CHECK(setup_instance(&instances[5], &meshes[MESH_CUBE], MESH_CUBE, 5, MOBILITY_STATIC,
        2.2f, 0.1f, 1.0f, 0.0f, 0.6f, 2.2f, 0.6f), "cube instance 3");

    if (getenv("UNTITLED_SMOKE")) goto deinit;

    INPUT input = {0};
    SDL_SetWindowRelativeMouseMode(renderer.win, true);

    float campos[3] = {0.0f, 0.6f, 4.5f};
    float yaw = 0.0f;
    float pitch = 0.0f;
    uint64_t last = SDL_GetTicks();

    float proj[16];
    float view[16];
    float model[16];
    float tmp[16];
    float mvp[16];

    while (i_poll(&input, renderer.win)) {
        uint64_t now = SDL_GetTicks();
        float dt = (float)(now - last) / 1000.0f;
        last = now;
        if (dt > 0.05f) dt = 0.05f;

        yaw -= input.dx * 0.0025f;
        pitch -= input.dy * 0.0025f;
        if (pitch > 1.55f) pitch = 1.55f;
        if (pitch < -1.55f) pitch = -1.55f;

        float cp = cosf(pitch);
        float fwd[3] = {-sinf(yaw) * cp, sinf(pitch), -cosf(yaw) * cp};
        float right[3] = {cosf(yaw), 0.0f, -sinf(yaw)};

        float mx = (float)(input.right - input.left);
        float mz = (float)(input.fwd - input.back);
        float ml = sqrtf(mx * mx + mz * mz);
        if (ml > 1.0f) {
            mx /= ml;
            mz /= ml;
        }

        float speed = input.run ? 12.0f : 3.0f;
        campos[0] += (fwd[0] * mz + right[0] * mx) * speed * dt;
        campos[1] += fwd[1] * mz * speed * dt;
        campos[2] += (fwd[2] * mz + right[2] * mx) * speed * dt;

        float center[3] = {campos[0] + fwd[0], campos[1] + fwd[1], campos[2] + fwd[2]};
        float up[3] = {0.0f, 1.0f, 0.0f};
        r_look(view, campos, center, up);

        SDL_GPUCommandBuffer *cmd = SDL_AcquireGPUCommandBuffer(renderer.device);
        CHECK(cmd, "acquire command buffer");

        SDL_GPUTexture *swp = NULL;
        uint32_t w = 0;
        uint32_t h = 0;
        CHECK(SDL_WaitAndAcquireGPUSwapchainTexture(cmd, renderer.win, &swp, &w, &h), "swapchain texture");

        if (swp) {
            if (!r_resize(&renderer, w, h)) {
                ret = -1;
                break;
            }

            r_persp(proj, 1.0472f, (float)w / (float)h, 0.1f, 100.0f);

            SDL_GPUColorTargetInfo color = {
                .texture = swp,
                .clear_color = {0.02f, 0.02f, 0.03f, 1.0f},
                .load_op = SDL_GPU_LOADOP_CLEAR,
                .store_op = SDL_GPU_STOREOP_STORE
            };
            SDL_GPUDepthStencilTargetInfo ds = {
                .texture = renderer.depth,
                .clear_depth = 1.0f,
                .load_op = SDL_GPU_LOADOP_CLEAR,
                .store_op = SDL_GPU_STOREOP_DONT_CARE,
                .stencil_load_op = SDL_GPU_LOADOP_DONT_CARE,
                .stencil_store_op = SDL_GPU_STOREOP_DONT_CARE
            };

            SDL_GPURenderPass *pass = SDL_BeginGPURenderPass(cmd, &color, 1, &ds);
            SDL_GPUViewport vp = {
                .x = 0.0f, .y = 0.0f, .w = (float)w, .h = (float)h,
                .min_depth = 0.0f, .max_depth = 1.0f
            };
            SDL_SetGPUViewport(pass, &vp);
            SDL_Rect sc = {.x = 0, .y = 0, .w = (int)w, .h = (int)h};
            SDL_SetGPUScissor(pass, &sc);

            SDL_GPUGraphicsPipeline *mesh_pipeline = pipe.gfx;
            if (input.debug_mode == 1) mesh_pipeline = pipe.uv;
            if (input.debug_mode == 2) mesh_pipeline = pipe.triangle;
            SDL_BindGPUGraphicsPipeline(pass, mesh_pipeline);

            for (uint32_t i = 0; i < 6; ++i) {
                const RENDER_INSTANCE *instance = &instances[i];
                r_mul(tmp, view, instance->transform.matrix);
                r_mul(mvp, proj, tmp);
                SDL_PushGPUVertexUniformData(cmd, 0, mvp, sizeof(mvp));
                if (input.debug_mode == 2) {
                    m_draw_triangle_debug(pass, &meshes[instance->mesh]);
                } else {
                    m_draw(pass, &meshes[instance->mesh]);
                }
            }

            if (input.debug_mode == 3) {
                SDL_BindGPUGraphicsPipeline(pass, pipe.bounds);
                for (uint32_t i = 0; i < 6; ++i) {
                    const AABB b = instances[i].world_bounds;
                    float cx = (b.min.x + b.max.x) * 0.5f;
                    float cy = (b.min.y + b.max.y) * 0.5f;
                    float cz = (b.min.z + b.max.z) * 0.5f;
                    float sx = b.max.x - b.min.x;
                    float sy = b.max.y - b.min.y;
                    float sz = b.max.z - b.min.z;
                    r_xform(model, cx, cy, cz, 0.0f, sx, sy, sz);
                    r_mul(tmp, view, model);
                    r_mul(mvp, proj, tmp);
                    SDL_PushGPUVertexUniformData(cmd, 0, mvp, sizeof(mvp));
                    SDL_DrawGPUPrimitives(pass, 24, 1, 0, 0);
                }
            }

            SDL_EndGPURenderPass(pass);
        }

        SDL_SubmitGPUCommandBuffer(cmd);
    }

deinit:
    if (renderer.device) {
        c_deinit(renderer.device, &cmp);
        p_deinit(renderer.device, &pipe);
        for (uint32_t i = 0; i < MESH_COUNT; ++i) {
            m_deinit(renderer.device, &meshes[i]);
        }
    }
    r_deinit(&renderer);
    return ret;
}
