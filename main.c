#include "game.h"

#include <SDL3/SDL.h>

#include <stdio.h>

int main(int argc, char **argv) {
    const char *model_path = argc > 1 ? argv[1] : "concrete_temple.glb";

    if (!SDL_Init(SDL_INIT_VIDEO)) {
        fprintf(stderr, "Could not initialize SDL: %s\n", SDL_GetError());
        return 1;
    }

    GLB_DOC document = {0};
    MESH geometry = {0};
    GLTF_SCENE visual = {0};

    if (!glb_load(&document, model_path)) {
        fprintf(stderr, "Could not load %s: %s\n", model_path, glb_error(&document));
        SDL_Quit();
        return 1;
    }

    if (!glb_extract_mesh(&document, &geometry)) {
        fprintf(stderr, "Could not extract mesh from %s\n", model_path);
        glb_free(&document);
        SDL_Quit();
        return 1;
    }

    if (!gltf_extract(&document, &visual)) {
        fprintf(stderr, "Could not extract glTF scene from %s\n", model_path);
        mesh_free(&geometry);
        glb_free(&document);
        SDL_Quit();
        return 1;
    }

    struct MODEL model = {
        .geometry = &geometry,
        .visual = &visual
    };

    struct LIGHT sun = {
        .type = LIGHT_DIRECTIONAL,
        .directional = {
            .direction = {0.38f, -0.72f, 0.32f},
            .color = {1.0f, 0.95f, 0.86f},
            .intensity = 2.5f,
            .angular_radius = 0.01f
        }
    };

    SCENE scene = {
        .sky = {
            .zenith = {0.22f, 0.42f, 0.78f},
            .horizon = {0.68f, 0.76f, 0.88f},
            .intensity = 0.35f
        }
    };

    if (!scene_add_model(&scene, &model, STATIC, transform_identity()) ||
        !scene_add_light(&scene, &sun, STATIC, transform_identity())) {
        fprintf(stderr, "Could not create scene\n");
        gltf_free(&visual);
        mesh_free(&geometry);
        glb_free(&document);
        SDL_Quit();
        return 1;
    }

    GPU gpu = {0};
    RENDERER renderer = {0};

    if (!gpu_init(&gpu, "Pony Radiance", 1280, 720)) {
        fprintf(stderr, "Could not initialize GPU: %s\n", SDL_GetError());
        scene_free(&scene);
        gltf_free(&visual);
        mesh_free(&geometry);
        glb_free(&document);
        SDL_Quit();
        return 1;
    }

    if (!renderer_init(&renderer, &gpu) || !renderer_set_scene(&renderer, &scene)) {
        fprintf(stderr, "Could not initialize renderer\n");
        renderer_deinit(&renderer);
        gpu_deinit(&gpu);
        scene_free(&scene);
        gltf_free(&visual);
        mesh_free(&geometry);
        glb_free(&document);
        SDL_Quit();
        return 1;
    }

    printf(
        "%s | %zu vertices | %zu triangles | %u materials | %u images\n",
        model_path,
        visual.vertex_count,
        visual.vertex_count / 3u,
        visual.material_count,
        visual.image_count
    );

    bool running = true;
    while (running) {
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_EVENT_QUIT) running = false;
            if (event.type == SDL_EVENT_KEY_DOWN && !event.key.repeat && event.key.key == SDLK_ESCAPE) running = false;
            renderer_event(&renderer, &event);
        }
        if (!running) break;
        if (!renderer_frame(&renderer)) {
            fprintf(stderr, "Renderer frame failed\n");
            running = false;
        }
        SDL_Delay(1);
    }

    renderer_deinit(&renderer);
    gpu_deinit(&gpu);
    scene_free(&scene);
    gltf_free(&visual);
    mesh_free(&geometry);
    glb_free(&document);
    SDL_Quit();
    return 0;
}
