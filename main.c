#include <stdio.h>

#define CHECK(expr, ...)                                    \
    do {                                                    \
        if (!(expr)) {                                      \
            fprintf(stderr, "CHECK failed: %s", #expr);     \
            fprintf(stderr, ": " __VA_ARGS__);              \
            fprintf(stderr, "\n");                          \
            ret = -1;                                       \
            goto deinit;                                    \
        }                                                   \
    } while (0)

#include "ren.c"
#include "msh.c"
#include "shdr.c"
#include "cmp.c"
#include "inpt.c"

int main() {
    RENDERER renderer = {
        .w = 1270, .h = 750
    };

    MESH cube = { 0 };
    MESH floor = { 0 };
    MESH wall = { 0 };
    PIPE pipe = { 0 };
    CMP cmp = { 0 };

    int ret = 0;

    CHECK(r_init(&renderer));
    CHECK(m_init(renderer.device, &cube));
    CHECK(m_plane(renderer.device, &floor));
    CHECK(m_wall(renderer.device, &wall));
    CHECK(p_init(&renderer, &pipe));
    CHECK(c_init(renderer.device, &cmp));
    CHECK(c_run(renderer.device, &cmp, pipe.cmp));

    INPUT input = { 0 };

    SDL_SetWindowRelativeMouseMode(renderer.win, true);

    float campos[3] = { 0.0f, 0.6f, 4.5f };
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
        if (pitch > 1.55f)  pitch = 1.55f;
        if (pitch < -1.55f) pitch = -1.55f;

        float cp = cosf(pitch);
        float fwd[3] = { -sinf(yaw) * cp, sinf(pitch), -cosf(yaw) * cp };
        float right[3] = { cosf(yaw), 0.0f, -sinf(yaw) };

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

        float center[3] = { campos[0] + fwd[0], campos[1] + fwd[1], campos[2] + fwd[2] };
        float up[3] = { 0.0f, 1.0f, 0.0f };
        r_look(view, campos, center, up);

        SDL_GPUCommandBuffer *cmd = SDL_AcquireGPUCommandBuffer(renderer.device);
        CHECK(cmd);

        SDL_GPUTexture *swp = NULL;
        uint32_t w = 0;
        uint32_t h = 0;

        CHECK(SDL_WaitAndAcquireGPUSwapchainTexture(cmd, renderer.win, &swp, &w, &h));

        if (swp) {
            if (!r_resize(&renderer, w, h)) {
                ret = -1;
                break;
            }

            r_persp(proj, 1.0472f, (float)w / (float)h, 0.1f, 100.0f);

            SDL_GPUColorTargetInfo color = {
                .texture = swp,
                .clear_color = { 0.02f, 0.02f, 0.03f, 1.0f },
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
                .x = 0.0f,
                .y = 0.0f,
                .w = (float)w,
                .h = (float)h,
                .min_depth = 0.0f,
                .max_depth = 1.0f
            };
            SDL_SetGPUViewport(pass, &vp);

            SDL_Rect sc = {
                .x = 0,
                .y = 0,
                .w = (int)w,
                .h = (int)h
            };
            SDL_SetGPUScissor(pass, &sc);

            SDL_BindGPUGraphicsPipeline(pass, pipe.gfx);

            r_xform(model, 0.0f, -1.0f, 0.0f, 0.0f, 12.0f, 1.0f, 12.0f);
            r_mul(tmp, view, model);
            r_mul(mvp, proj, tmp);
            SDL_PushGPUVertexUniformData(cmd, 0, mvp, sizeof(mvp));
            m_draw(pass, &floor);

            r_xform(model, 0.0f, 2.0f, -3.0f, 0.0f, 12.0f, 6.0f, 1.0f);
            r_mul(tmp, view, model);
            r_mul(mvp, proj, tmp);
            SDL_PushGPUVertexUniformData(cmd, 0, mvp, sizeof(mvp));
            m_draw(pass, &wall);

            r_xform(model, -1.2f, -0.5f, 0.5f, 0.3f, 1.0f, 1.0f, 1.0f);
            r_mul(tmp, view, model);
            r_mul(mvp, proj, tmp);
            SDL_PushGPUVertexUniformData(cmd, 0, mvp, sizeof(mvp));
            m_draw(pass, &cube);

            r_xform(model, 1.0f, -0.5f, -0.8f, -0.4f, 1.0f, 1.0f, 1.0f);
            r_mul(tmp, view, model);
            r_mul(mvp, proj, tmp);
            SDL_PushGPUVertexUniformData(cmd, 0, mvp, sizeof(mvp));
            m_draw(pass, &cube);

            r_xform(model, -1.2f, 0.25f, 0.5f, 0.3f, 0.5f, 0.5f, 0.5f);
            r_mul(tmp, view, model);
            r_mul(mvp, proj, tmp);
            SDL_PushGPUVertexUniformData(cmd, 0, mvp, sizeof(mvp));
            m_draw(pass, &cube);

            r_xform(model, 2.2f, 0.1f, 1.0f, 0.0f, 0.6f, 2.2f, 0.6f);
            r_mul(tmp, view, model);
            r_mul(mvp, proj, tmp);
            SDL_PushGPUVertexUniformData(cmd, 0, mvp, sizeof(mvp));
            m_draw(pass, &cube);

            SDL_EndGPURenderPass(pass);
        }

        SDL_SubmitGPUCommandBuffer(cmd);
    }

deinit:
    if (renderer.device) {
        c_deinit(renderer.device, &cmp);
        p_deinit(renderer.device, &pipe);
        m_deinit(renderer.device, &cube);
        m_deinit(renderer.device, &floor);
        m_deinit(renderer.device, &wall);
    }
    r_deinit(&renderer);
    return ret;
}
