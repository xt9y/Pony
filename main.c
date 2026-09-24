#include "dustmite.h"
#include "cache.h"

#include <SDL3/SDL.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define LIGHTMAP_TEXELS_PER_UNIT 24u
#define LIGHTMAP_MAX_SIZE 4096u

static double elapsed_ms(Uint64 begin) {

    return (double)(SDL_GetPerformanceCounter() - begin) * 1000.0 /
           (double)SDL_GetPerformanceFrequency();

}

static void print_required_extensions(const glb_doc *doc) {

    const int extensions = glb_get(doc, glb_root(doc), "extensionsRequired");

    for (size_t i = 0; i < glb_count(doc, extensions); ++i) {

        const char *name = NULL;
        size_t length = 0;
        if (glb_string(doc, glb_at(doc, extensions, i), &name, &length)) {
            fprintf(stderr, "GLB requires extension: %.*s\n", (int)length, name);
        }
    }
}

int main(int argc, char **argv) {

    const char *model_path = argc > 1 ? argv[1] : "concrete_temple.glb";
    const char *filename = strrchr(model_path, '/');
    const char *windows_filename = strrchr(model_path, '\\');

    if (windows_filename && (!filename || windows_filename > filename)) filename = windows_filename;
    const char *extension = strrchr(model_path, '.');
    size_t stem_length = extension && (!filename || extension > filename) ? (size_t)(extension - model_path) : strlen(model_path);
    char *bake_path = malloc(stem_length + sizeof(".baked"));
    if (!bake_path) return 1;

    memcpy(bake_path, model_path, stem_length);
    memcpy(bake_path + stem_length, ".baked", sizeof(".baked"));

    char title[512];
    snprintf(title, sizeof(title), "Dustmite - %s", filename ? filename + 1 : model_path);

    if (!SDL_Init(SDL_INIT_VIDEO)) {
        fprintf(stderr, "Could not initialize SDL: %s\n", SDL_GetError());
        free(bake_path);
        return 1;
    }

    const Uint64 load_begin = SDL_GetPerformanceCounter();
    glb_doc model = {0};
    mesh scene = {0};
    gltf_scene visual = {0};
    lightmap lm = {0};
    renderer r = {0};

    const bool loaded = glb_load(&model, model_path);
    if (loaded) print_required_extensions(&model);

    const bool extracted = loaded && glb_extract_mesh(&model, &scene) &&
                           gltf_extract(&model, &visual);
    const double load_ms = elapsed_ms(load_begin);

    const Uint64 atlas_begin = SDL_GetPerformanceCounter();
    const bool atlas_ready = extracted &&
        lmap_build(&lm, &scene, LIGHTMAP_TEXELS_PER_UNIT, LIGHTMAP_MAX_SIZE);
    const double atlas_ms = elapsed_ms(atlas_begin);

    if (!atlas_ready || !r_init(&r, title, 1280, 720) ||
        !r_build_scene(&r, &scene, &visual, &lm)) {
        fprintf(stderr, "Could not initialize renderer or scene: %s %s\n",
                glb_error(&model), SDL_GetError());
        r_deinit(&r);
        lmap_free(&lm);
        gltf_free(&visual);
        mesh_free(&scene);
        glb_free(&model);
        SDL_Quit();
        free(bake_path);
        return 1;
    }

    const uint64_t scene_hash = dm_hash_bytes(0, model.data, model.data_size);
    uint64_t layout_hash = dm_hash_bytes(0, &lm.width, sizeof(lm.width));
    layout_hash = dm_hash_bytes(layout_hash, &lm.height, sizeof(lm.height));
    layout_hash = dm_hash_bytes(layout_hash, lm.uvs,
                                scene.faces.count * 6u * sizeof(*lm.uvs));

    const uint32_t bake_settings[] = {
        LIGHTMAP_TEXELS_PER_UNIT, LIGHTMAP_MAX_SIZE, 128u, 3u,
        4u /* separate front and back irradiance */
    };
    layout_hash = dm_hash_bytes(layout_hash, bake_settings, sizeof(bake_settings));

    const float lighting_settings[] = {
        0.38f, 0.30f, 0.32f, 2.4f, 1.0f, 0.94f, 0.84f, 0.00465f,
        0.22f, 0.42f, 0.78f, 0.68f, 0.76f, 0.88f
    };
    layout_hash = dm_hash_bytes(layout_hash, lighting_settings, sizeof(lighting_settings));

    const uint32_t volume_settings[] = {1024u, 4u, 3u};
    uint64_t volume_hash = dm_hash_bytes(scene_hash, volume_settings, sizeof(volume_settings));
    volume_hash = dm_hash_bytes(volume_hash, lighting_settings, sizeof(lighting_settings));
    const uint32_t beam_settings[] = {64u, 16u, 5u};
    uint64_t beam_hash = dm_hash_bytes(scene_hash, beam_settings, sizeof(beam_settings));
    beam_hash = dm_hash_bytes(beam_hash, lighting_settings, 3u * sizeof(float));

    const bool cached = r_load_cached_lightmap(&r, bake_path, scene_hash, layout_hash,
                                                volume_hash, beam_hash, &lm);
    SDL_SetWindowTitle(r.window, cached ? "Dustmite - baked lighting loaded (B to rebake)" :
                       "Dustmite - unbaked scene (press B to bake)");

    printf("%s: %.2f ms load | %zu vertices | %zu triangles | %.2f MiB BIN\n",
           model_path, load_ms, scene.vertices.count, scene.faces.count,
           (double)model.bin_size / (1024.0 * 1024.0));
    printf("Visual: %zu vertices | %u materials | %u textures | %u images\n",
           visual.vertex_count, visual.material_count, visual.texture_count, visual.image_count);
    printf("Lightmap: %.2f ms atlas | %ux%u | %u charts | %.2f texels/unit | %u valid texels\n",
           atlas_ms, lm.width, lm.height, lm.chart_count, lm.texel_density, lm.sample_count);
    printf("Lighting: %s. Press B to rebake this scene in the renderer.\n",
           cached ? "loaded saved bake" : "unbaked fallback");
    printf("Runtime: PBR + sun beams + volume probes -> HDR -> bloom -> ACES + GPU LUT\n");
    printf("LMB drag: orbit | wheel: zoom | B: rebake | F5: fog on/off | Tab: wireframe | F11: fullscreen | Esc: quit\n");

    bool running = true;
    Uint64 last_frame_print = SDL_GetTicks();

    while (running) {

        SDL_Event event;

        while (SDL_PollEvent(&event)) {

            if (event.type == SDL_EVENT_QUIT) running = false;

            if (event.type == SDL_EVENT_KEY_DOWN && !event.key.repeat) {

                if (event.key.key == SDLK_ESCAPE) running = false;

                if (event.key.scancode == SDL_SCANCODE_B || event.key.key == SDLK_B) {

                    SDL_ClearError();
                    SDL_SetWindowTitle(r.window, "Dustmite - preparing bake...");
                    SDL_PumpEvents();
                    SDL_Log("B: rebaking current scene (this may take a while)");

                    const Uint64 begin = SDL_GetPerformanceCounter();
                    const bool good = r_rebake_current_scene(&r, &scene, &visual, &lm,
                                                               bake_path, scene_hash, layout_hash,
                                                               volume_hash, beam_hash);

                    SDL_Log("B: %s after %.2f ms%s%s%s%s",
                            good ? "bake saved" : "bake failed; previous lighting retained",
                            elapsed_ms(begin), good ? "" : " at ",
                            good ? "" : (r.bake_stage ? r.bake_stage : "unknown stage"),
                            good ? "" : ": ",
                            good ? "" : (*SDL_GetError() ? SDL_GetError() : "see the failing stage"));
                    // SDL_Log("draw %u: material=%u first=%u count=%u", i, draw->material, draw->first, draw->count);
                }
            }

            r_event(&r, &event);
        }

        const Uint64 frame_begin = SDL_GetPerformanceCounter();
        r_draw(&r);
        r.frame_time_ms = elapsed_ms(frame_begin);

        const Uint64 now = SDL_GetTicks();
        if (now - last_frame_print >= 1000u) {
            printf("frame time: %.2f ms\n", r.frame_time_ms);
            fflush(stdout);
            last_frame_print = now;
        }
    }

    r_deinit(&r);
    gltf_free(&visual);
    lmap_free(&lm);
    mesh_free(&scene);
    glb_free(&model);
    SDL_Quit();
    free(bake_path);
    return 0;
}
