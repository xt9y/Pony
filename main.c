#include "game.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define LIGHTMAP_TEXELS_PER_UNIT 24u
#define LIGHTMAP_MAX_SIZE 4096u
#define BAKE_IDLE_GRACE_MS 180u
#define BAKE_IDLE_RENDER_MS 200u
#define BAKE_IDLE_SLEEP_MS 2u

static double elapsed_ms(Uint64 begin) {
    return (double)(SDL_GetPerformanceCounter() - begin) * 1000.0 / (double)SDL_GetPerformanceFrequency();
}

static bool input_event(const SDL_Event *event) {
    if (!event) return false;

    switch (event->type) {
    case SDL_EVENT_KEY_DOWN:
    case SDL_EVENT_KEY_UP:
    case SDL_EVENT_MOUSE_BUTTON_DOWN:
    case SDL_EVENT_MOUSE_BUTTON_UP:
    case SDL_EVENT_MOUSE_MOTION:
    case SDL_EVENT_MOUSE_WHEEL:
        return true;
    default:
        return false;
    }
}

static char *bake_cache_path(const char *model_path) {
    const char *filename = strrchr(model_path, '/');
    const char *windows_filename = strrchr(model_path, '\\');

    if (windows_filename && (!filename || windows_filename > filename)) filename = windows_filename;

    const char *extension = strrchr(model_path, '.');
    const size_t stem_length = extension && (!filename || extension > filename) ? (size_t)(extension - model_path) : strlen(model_path);
    const char suffix[] = ".baked";
    char *path = malloc(stem_length + sizeof(suffix));

    if (!path) return NULL;

    memcpy(path, model_path, stem_length);
    memcpy(path + stem_length, suffix, sizeof(suffix));

    return path;
}

int main(int argc, char **argv) {
    const char *model_path = argc > 1 ? argv[1] : "concrete_temple.glb";
    const char *filename = strrchr(model_path, '/');
    const char *windows_filename = strrchr(model_path, '\\');

    if (windows_filename && (!filename || windows_filename > filename)) filename = windows_filename;

    char *bake_path = bake_cache_path(model_path);

    if (!bake_path) return 1;

    char title[512];
    snprintf(title, sizeof(title), "INIT | %s", filename ? filename + 1 : model_path);

    if (!SDL_Init(SDL_INIT_VIDEO)) {
        fprintf(stderr, "Could not initialize SDL: %s\n", SDL_GetError());
        free(bake_path);
        return 1;
    }

    const Uint64 load_begin = SDL_GetPerformanceCounter();
    GLB_DOC document = {0};
    MESH geometry = {0};
    GLTF_SCENE visual = {0};
    LIGHTMAP lightmap = {0};

    struct MODEL model = {.geometry = &geometry, .visual = &visual};

    struct LIGHT sun = {.type = LIGHT_DIRECTIONAL,
                        .directional = {.direction = {0.38f, 0.30f, 0.32f}, .color = {1.00f, 0.94f, 0.84f}, .intensity = 1.0f, .angular_radius = 0.00465f}};

    SCENE scene = {.sky = {.zenith = {0.22f, 0.42f, 0.78f}, .horizon = {0.68f, 0.76f, 0.88f}, .intensity = 1.0f},
                   .volumetrics = {.density = 0.045f,
                                   .anisotropy = 0.55f,
                                   .probe_intensity = 0.15f,
                                   .emissive_probe_intensity = 1.0f,
                                   .max_distance = 10000.0f,
                                   .probe_spacing = 4.0f,
                                   .probe_samples = 1024u,
                                   .emissive_samples = 128u},
                   .vision = {.center_radius = 0.50f,
                              .middle_radius = 0.82f,
                              .center_transition_width = 0.08f,
                              .middle_transition_width = 0.08f,
                              .jitter_strength = 0.65f,
                              .volume_blur_strength = 0.35f,
                              .center_steps = 4u,
                              .middle_steps = 2u,
                              .peripheral_steps = 1u,
                              .center_stride = 1u,
                              .middle_stride = 8u,
                              .peripheral_stride = 24u},
                   .lightmap = &lightmap};

    const char *startup_stage = NULL;
    const char *startup_detail = NULL;

    if (!glb_load(&document, model_path)) {
        startup_stage = "GLB load";
        startup_detail = glb_error(&document);
    } else if (!glb_extract_mesh(&document, model.geometry)) {
        startup_stage = "mesh extraction";
    } else if (!gltf_extract(&document, model.visual)) {
        startup_stage = "visual glTF extraction";
    } else if (!scene_add_model(&scene, &model, STATIC, transform_identity()) || !scene_add_light(&scene, &sun, STATIC, transform_identity())) {
        startup_stage = "scene composition";
    }

    const double load_ms = elapsed_ms(load_begin);
    const Uint64 atlas_begin = SDL_GetPerformanceCounter();

    if (!startup_stage && !lmap_build(&lightmap, model.geometry, LIGHTMAP_TEXELS_PER_UNIT, LIGHTMAP_MAX_SIZE)) startup_stage = "lightmap atlas generation";

    const double atlas_ms = elapsed_ms(atlas_begin);

    GPU gpu = {0};
    RENDERER renderer = {0};

    if (!startup_stage && !gpu_init(&gpu, title, 1280, 720)) {
        startup_stage = "GPU configuration";
        startup_detail = SDL_GetError();
    }

    if (!startup_stage && !renderer_init(&renderer, &gpu)) {
        startup_stage = "renderer initialization";
        startup_detail = SDL_GetError();
    }

    if (!startup_stage && !renderer_set_scene(&renderer, &scene)) {
        startup_stage = "GPU scene upload";
        startup_detail = SDL_GetError();
    }

    if (startup_stage) {
        fprintf(stderr, "Startup failed at %s", startup_stage);
        if (startup_detail && *startup_detail) fprintf(stderr, ": %s", startup_detail);
        fputc('\n', stderr);

        bake_cancel(&renderer);
        renderer_deinit(&renderer);
        gpu_deinit(&gpu);
        scene_free(&scene);
        lmap_free(&lightmap);
        gltf_free(model.visual);
        mesh_free(model.geometry);
        glb_free(&document);
        SDL_Quit();
        free(bake_path);
        return 1;
    }

    const uint64_t scene_hash = hash_bytes(0, document.data, document.data_size);
    uint64_t layout_hash = hash_bytes(0, &lightmap.width, sizeof(lightmap.width));
    layout_hash = hash_bytes(layout_hash, &lightmap.height, sizeof(lightmap.height));
    layout_hash = hash_bytes(layout_hash, lightmap.uvs, geometry.faces.count * 6u * sizeof(*lightmap.uvs));

    const uint32_t bake_settings[] = {
        LIGHTMAP_TEXELS_PER_UNIT, LIGHTMAP_MAX_SIZE, 128u, 3u, 4u, 32u, 2u, 32u, 8u, 50u, 75u, 1u, 4u, 16u, 2u, 1u, 1u, 4u, 995u, 25u, 60u, 100u};

    layout_hash = hash_bytes(layout_hash, bake_settings, sizeof(bake_settings));

    const float lighting_settings[] = {sun.directional.direction.x, sun.directional.direction.y, sun.directional.direction.z, sun.directional.intensity,
                                       sun.directional.color.x,     sun.directional.color.y,     sun.directional.color.z,     sun.directional.angular_radius,
                                       scene.sky.zenith.x,          scene.sky.zenith.y,          scene.sky.zenith.z,          scene.sky.horizon.x,
                                       scene.sky.horizon.y,         scene.sky.horizon.z,         scene.sky.intensity};

    layout_hash = hash_bytes(layout_hash, lighting_settings, sizeof(lighting_settings));

    const float volume_bake_settings[] = {scene.volumetrics.probe_spacing, (float)scene.volumetrics.probe_samples, (float)scene.volumetrics.emissive_samples,
                                          scene.volumetrics.emissive_probe_intensity, 9.0f};

    uint64_t volume_hash = hash_bytes(scene_hash, volume_bake_settings, sizeof(volume_bake_settings));
    volume_hash = hash_bytes(volume_hash, lighting_settings, sizeof(lighting_settings));

    const uint32_t beam_settings[] = {64u, 16u, 5u};
    uint64_t beam_hash = hash_bytes(scene_hash, beam_settings, sizeof(beam_settings));
    beam_hash = hash_bytes(beam_hash, lighting_settings, 3u * sizeof(float));

    const bool cached = renderer_load_cached_lightmap(&renderer, bake_path, scene_hash, layout_hash, volume_hash, beam_hash, &lightmap);

    SDL_SetWindowTitle(gpu.window, cached ? "READY" : "UNBAKED");

    printf("%s: %.2f ms load | %zu vertices | %zu triangles | %.2f MiB BIN\n", model_path, load_ms, geometry.vertices.count, geometry.faces.count,
           (double)document.bin_size / (1024.0 * 1024.0));
    printf("Visual: %zu vertices | %u materials | %u textures | %u images\n", visual.vertex_count, visual.material_count, visual.texture_count,
           visual.image_count);
    printf("Lightmap: %.2f ms atlas | %ux%u | %u charts | %.2f texels/unit | %u valid texels\n", atlas_ms, lightmap.width, lightmap.height,
           lightmap.chart_count, lightmap.texel_density, lightmap.sample_count);
    printf("Lighting: %s. Press B to rebake this scene in the renderer.\n", cached ? "loaded saved bake" : "unbaked fallback");
    printf("Runtime: PBR + sun beams + volume probes -> HDR -> bloom -> ACES + GPU LUT\n");
    printf("LMB drag: orbit | wheel: zoom | B: rebake | F5: fog on/off | Tab: wireframe | F11: fullscreen | Esc: quit\n");

    bool running = true;
    Uint64 last_frame_print = SDL_GetTicks();
    Uint64 last_input = SDL_GetTicks();
    Uint64 last_render = 0u;

    while (running) {
        SDL_Event event;

        while (SDL_PollEvent(&event)) {
            if (input_event(&event)) last_input = SDL_GetTicks();
            if (event.type == SDL_EVENT_QUIT) running = false;

            if (event.type == SDL_EVENT_KEY_DOWN && !event.key.repeat) {
                if (event.key.key == SDLK_ESCAPE) running = false;

                if (event.key.scancode == SDL_SCANCODE_B || event.key.key == SDLK_B) {
                    SDL_ClearError();

                    if (!bake_start(&renderer, model.geometry, model.visual, &lightmap, &sun, &scene.sky, &scene.volumetrics, bake_path, scene_hash,
                                    layout_hash, volume_hash, beam_hash)) {
                        SDL_Log("B: could not start rebake: %s", *SDL_GetError() ? SDL_GetError() : "unknown error");
                    }
                }
            }

            renderer_event(&renderer, &event);
        }

        if (!running) break;

        const Uint64 now = SDL_GetTicks();
        const bool idle_bake = bake_active(&renderer) && now - last_input >= BAKE_IDLE_GRACE_MS;
        const bool render_due = !idle_bake || now - last_render >= BAKE_IDLE_RENDER_MS;

        if (render_due) {
            const Uint64 frame_begin = SDL_GetPerformanceCounter();

            if (!renderer_frame(&renderer)) {
                SDL_Log("draw failed: %s", SDL_GetError());
                running = false;
            }

            renderer.frame_time_ms = elapsed_ms(frame_begin);
            last_render = now;
        } else {
            SDL_Delay(BAKE_IDLE_SLEEP_MS);
        }

        bake_update(&renderer);
        bake_update_title(&renderer);

        const Uint64 tick = SDL_GetTicks();

        if (tick - last_frame_print >= 1000u) {
            printf("frame time: %.2f ms\n", renderer.frame_time_ms);
            fflush(stdout);
            last_frame_print = tick;
        }
    }

    bake_cancel(&renderer);
    renderer_deinit(&renderer);
    gpu_deinit(&gpu);
    scene_free(&scene);
    gltf_free(model.visual);
    lmap_free(&lightmap);
    mesh_free(model.geometry);
    glb_free(&document);
    SDL_Quit();
    free(bake_path);

    return 0;
}
