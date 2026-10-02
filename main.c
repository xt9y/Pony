#include "game.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define LIGHTMAP_TEXELS_PER_UNIT 24u
#define LIGHTMAP_MAX_SIZE 4096u
#define BAKE_IDLE_GRACE_MS 180u
#define BAKE_IDLE_RENDER_MS 200u
#define BAKE_IDLE_SLEEP_MS 2u
#define DYNAMIC_Z_AMPLITUDE 1.0f
#define DYNAMIC_Z_PERIOD_SECONDS 4.0f
#define BASE_MODEL_PATH "hospital_hallway.glb"

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

static void free_models(MODEL_ASSET *models, int count) {
    if (!models) return;

    for (int i = 0; i < count; ++i) model_free(&models[i]);

    free(models);
}

static int extra_model_start(int argc, char **argv) {
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--") == 0) return i + 1;
    }

    return argc > 1 ? 1 : argc;
}

static bool load_scene_model(SCENE *scene, MODEL_ASSET *asset, const char *path, OBJECT_STATE state, TRANSFORM transform, size_t *total_bin_size) {
    if (!scene || !asset || !path) return false;
    if (!model_load(asset, path)) return false;

    if (!scene_add_model(scene, &asset->model, state, transform)) {
        SDL_SetError("could not add %s to scene", path);
        return false;
    }

    if (total_bin_size) *total_bin_size += asset->document.bin_size;

    return true;
}

static void transform_rotate_y(TRANSFORM *transform, float degrees)
{
    float radians = degrees * 3.14159265359f / 180.0f;
    float half = radians * 0.5f;

    transform->rotation[0] = 0.0f;
    transform->rotation[1] = sinf(half);
    transform->rotation[2] = 0.0f;
    transform->rotation[3] = cosf(half);
}

int main(int argc, char **argv) {
    const int extra_start = extra_model_start(argc, argv);
    const int extra_model_count = argc - extra_start;
    const int model_count = 1 + extra_model_count;
    const char *filename = strrchr(BASE_MODEL_PATH, '/');
    const char *windows_filename = strrchr(BASE_MODEL_PATH, '\\');

    if (windows_filename && (!filename || windows_filename > filename)) filename = windows_filename;

    char *bake_path = bake_cache_path(BASE_MODEL_PATH);

    if (!bake_path) return 1;

    char title[512];

    if (!extra_model_count) {
        snprintf(title, sizeof(title), "INIT | %s", filename ? filename + 1 : BASE_MODEL_PATH);
    } else {
        snprintf(title, sizeof(title), "INIT | %s + %d", filename ? filename + 1 : BASE_MODEL_PATH, extra_model_count);
    }

    if (!SDL_Init(SDL_INIT_VIDEO)) {
        fprintf(stderr, "Could not initialize SDL: %s\n", SDL_GetError());
        free(bake_path);
        return 1;
    }

    MODEL_ASSET *models = calloc((size_t)model_count, sizeof(*models));

    if (!models) {
        SDL_Quit();
        free(bake_path);
        return 1;
    }

    LIGHTMAP lightmap = {0};

    struct LIGHT sun = {.type = LIGHT_DIRECTIONAL,
                        .directional = {.direction = {0.38f, 0.30f, 0.32f}, .color = {1.00f, 0.94f, 0.84f}, .intensity = 0.0f, .angular_radius = 0.00465f}};

    SCENE scene = {.sky = {.zenith = {0.22f, 0.42f, 0.78f}, .horizon = {0.68f, 0.76f, 0.88f}, .intensity = 0.0f},
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

    const Uint64 load_begin = SDL_GetPerformanceCounter();
    const char *startup_stage = NULL;
    const char *startup_detail = NULL;
    size_t total_bin_size = 0;

    TRANSFORM barn_lamp = {
        .position = {-4.0f, 5.0f, -2.0f},
        .scale = {13.0f, 13.0f, 13.0f},
        .rotation = {0.0f, 0.0f, 0.0f}
    };
    
    transform_rotate_y(&barn_lamp, 90.0f);

    TRANSFORM cornell = {
        .position = {0.0f, 0.0f, 0.0f},
        .scale = {1.0f, 1.0f, 1.0f},
        .rotation = {0.0f, 0.0f, 0.0f}
    };
    
    transform_rotate_y(&cornell, -90.0f);
    
    TRANSFORM transform = barn_lamp;
    // TRANSFORM transform = cornell;

    if (!load_scene_model(&scene, &models[0], BASE_MODEL_PATH, STATIC, transform_identity(), &total_bin_size)) {
        startup_stage = "base model load";
        startup_detail = SDL_GetError();
    }

    for (int i = 0; i < extra_model_count && !startup_stage; ++i) {
        const char *path = argv[extra_start + i];
        OBJECT_STATE state = DYNAMIC;

        if (strncmp(path, "static:", 7u) == 0) {
            state = STATIC;
            path += 7u;
        } else if (strncmp(path, "dynamic:", 8u) == 0) {
            path += 8u;
        }

        SDL_Log("loading extra model %d/%d: %s%s", i + 1, extra_model_count, state == DYNAMIC ? "DYNAMIC " : "STATIC ", path);

        if (!load_scene_model(&scene, &models[i + 1], path, state, transform, &total_bin_size)) {
            startup_stage = "extra model load";
            startup_detail = SDL_GetError();
        }
    }

    OBJECT_ID animated_dynamic_id = 0u;

    for (uint32_t i = 0; i < scene.object_count; ++i) {
        if (scene.objects[i].type != MODEL || scene.objects[i].state != DYNAMIC) continue;

        animated_dynamic_id = scene.objects[i].id;

        TRANSFORM animated_start = scene.objects[i].transform;
        animated_start.position.z = 0.0f;
        object_set_transform(&scene.objects[i], animated_start);
        break;
    }

    if (!startup_stage && !scene_add_light(&scene, &sun, STATIC, transform_identity())) startup_stage = "scene composition";

    const double load_ms = elapsed_ms(load_begin);
    const Uint64 atlas_begin = SDL_GetPerformanceCounter();

    if (!startup_stage && !scene_build_lightmap(&scene, LIGHTMAP_TEXELS_PER_UNIT, LIGHTMAP_MAX_SIZE)) startup_stage = "lightmap atlas generation";

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
        free_models(models, model_count);
        SDL_Quit();
        free(bake_path);
        return 1;
    }

    const uint64_t scene_hash = scene_content_hash(&scene);
    uint64_t layout_hash = hash_bytes(0, &lightmap.width, sizeof(lightmap.width));
    layout_hash = hash_bytes(layout_hash, &lightmap.height, sizeof(lightmap.height));
    layout_hash = hash_bytes(layout_hash, lightmap.uvs, scene.static_geometry.faces.count * 6u * sizeof(*lightmap.uvs));

    const uint32_t bake_settings[] = {
        LIGHTMAP_TEXELS_PER_UNIT, LIGHTMAP_MAX_SIZE, 128u, 3u, 4u, 32u, 2u, 32u, 8u, 50u, 75u, 1u, 4u, 16u, 2u, 1u, 1u, 4u, 995u, 25u, 60u, 100u, 32u};

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

    printf("Scene: %s + %d extra model%s | %.2f ms load | %zu vertices | %zu triangles | %.2f MiB BIN\n", BASE_MODEL_PATH, extra_model_count,
           extra_model_count == 1 ? "" : "s", load_ms, scene.geometry.vertices.count, scene.geometry.faces.count,
           (double)total_bin_size / (1024.0 * 1024.0));

    for (int i = 0; i < extra_model_count; ++i) printf("Extra model %d: %s\n", i + 1, argv[extra_start + i]);
    printf("Visual: %zu vertices | %u materials | %u textures | %u images\n", scene.visual.vertex_count, scene.visual.material_count,
           scene.visual.texture_count, scene.visual.image_count);
    printf("Lightmap: %.2f ms atlas | %ux%u | %u charts | %.2f texels/unit | %u valid texels\n", atlas_ms, lightmap.width, lightmap.height,
           lightmap.chart_count, lightmap.texel_density, lightmap.sample_count);
    printf("Lighting: %s. Press B to rebake this scene in the renderer.\n", cached ? "loaded saved bake" : "unbaked fallback");
    printf("Runtime: PBR + sun beams + volume probes -> HDR -> bloom -> ACES + GPU LUT\n");
    printf("Extra models are DYNAMIC by default; prefix static: to bake one permanently | first dynamic model auto-oscillates Z from -1 to +1 | left/right: move first dynamic model on X | LMB drag: orbit | wheel: zoom | B: rebake | F2: reference lighting | F5: fog on/off | F6: cache validity | F7: signed correction | Tab: wireframe | F11: fullscreen | Esc: quit\n");

    bool running = true;
    Uint64 last_frame_print = SDL_GetTicks();
    Uint64 last_input = SDL_GetTicks();
    Uint64 last_render = 0u;
    const Uint64 dynamic_animation_start = SDL_GetTicks();

    while (running) {
        SDL_Event event;

        while (SDL_PollEvent(&event)) {
            if (input_event(&event)) last_input = SDL_GetTicks();
            if (event.type == SDL_EVENT_QUIT) running = false;

            if (event.type == SDL_EVENT_KEY_DOWN && !event.key.repeat) {
                if (event.key.key == SDLK_ESCAPE) running = false;

                if (event.key.scancode == SDL_SCANCODE_B || event.key.key == SDLK_B) {
                    SDL_ClearError();

                    if (!bake_start(&renderer, &scene, bake_path, scene_hash, layout_hash, volume_hash, beam_hash)) {
                        SDL_Log("B: could not start rebake: %s", *SDL_GetError() ? SDL_GetError() : "unknown error");
                    }
                }

                OBJECT *dynamic = NULL;

                for (uint32_t i = 0; i < scene.object_count; ++i) {
                    if (scene.objects[i].type == MODEL && scene.objects[i].state == DYNAMIC) {
                        dynamic = &scene.objects[i];
                        break;
                    }
                }

                if (dynamic) {
                    TRANSFORM moved = dynamic->transform;
                    bool changed = true;

                    if (event.key.key == SDLK_LEFT)
                        moved.position.x -= 0.25f;
                    else if (event.key.key == SDLK_RIGHT)
                        moved.position.x += 0.25f;
                    else
                        changed = false;

                    if (changed) object_set_transform(dynamic, moved);
                }
            }

            renderer_event(&renderer, &event);
        }

        if (!running) break;

        const Uint64 now = SDL_GetTicks();
        const bool idle_bake = bake_active(&renderer) && now - last_input >= BAKE_IDLE_GRACE_MS;
        const bool render_due = !idle_bake || now - last_render >= BAKE_IDLE_RENDER_MS;

        if (render_due) {
            if (animated_dynamic_id) {
                OBJECT *animated = scene_object_by_id(&scene, animated_dynamic_id);

                if (animated && animated->state == DYNAMIC && animated->type == MODEL) {
                    const float seconds = (float)(now - dynamic_animation_start) * 0.001f;
                    const float phase = seconds * (6.28318530717958647692f / DYNAMIC_Z_PERIOD_SECONDS);
                    const float z = sinf(phase) * DYNAMIC_Z_AMPLITUDE;

                    if (fabsf(animated->transform.position.z - z) > 1.0e-5f) {
                        TRANSFORM moved = animated->transform;
                        moved.position.z = z;
                        object_set_transform(animated, moved);
                    }
                }
            }

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
    lmap_free(&lightmap);
    free_models(models, model_count);
    SDL_Quit();
    free(bake_path);

    return 0;
}
