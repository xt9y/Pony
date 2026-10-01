#include "game.h"
#include "render_internal.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct MAT4 {
    float m[16];
} MAT4;

typedef struct COLOR4 {
    float r, g, b, a;
} COLOR4;

void bake_progress(RENDERER *renderer, const char *stage, Uint32 done, Uint32 total) {
    if (!renderer || !renderer->window) return;

    renderer->bake_stage = stage;

    char title[160];

    if (total) {
        snprintf(title, sizeof(title), "Pony - B baking %s: %u/%u", stage, done, total);

        const double valid_texels = (double)renderer->lightmap_sample_count;
        const double completed_work = valid_texels * (double)done / (double)total;

        printf("frame time: %.2f ms | bake: %.2e/%.2e (%s)\n", renderer->frame_time_ms, completed_work, valid_texels, stage);
        fflush(stdout);
    } else {
        snprintf(title, sizeof(title), "Pony - B baking %s...", stage);
    }

    SDL_SetWindowTitle(renderer->window, title);
    SDL_PumpEvents();
}

static void bake_timing(const char *stage, Uint64 started) {
    const double elapsed = (double)(SDL_GetPerformanceCounter() - started) * 1000.0 / (double)SDL_GetPerformanceFrequency();

    SDL_Log("B: %s took %.2f ms", stage, elapsed);
}

static MAT4 m4_identity(void) {
    MAT4 result = {0};
    result.m[0] = result.m[5] = result.m[10] = result.m[15] = 1.0f;

    return result;
}

static MAT4 m4_mul(MAT4 a, MAT4 b) {
    MAT4 result = {0};

    for (int column = 0; column < 4; ++column) {
        for (int row = 0; row < 4; ++row) {
            result.m[column * 4 + row] =
                a.m[row] * b.m[column * 4] + a.m[4 + row] * b.m[column * 4 + 1] + a.m[8 + row] * b.m[column * 4 + 2] + a.m[12 + row] * b.m[column * 4 + 3];
        }
    }

    return result;
}

static MAT4 m4_perspective(float fov_y, float aspect, float znear, float zfar) {
    const float f = 1.0f / tanf(fov_y * 0.5f);
    MAT4 result = {0};
    result.m[0] = f / aspect;
    result.m[5] = f;
    result.m[10] = zfar / (znear - zfar);
    result.m[11] = -1.0f;
    result.m[14] = (znear * zfar) / (znear - zfar);

    return result;
}

static MAT4 m4_look_at(VEC3 eye, VEC3 target, VEC3 up) {
    const VEC3 forward = v3_normalize(v3_sub(target, eye));
    const VEC3 side = v3_normalize(v3_cross(forward, up));
    const VEC3 corrected_up = v3_cross(side, forward);
    MAT4 result = m4_identity();

    result.m[0] = side.x;
    result.m[1] = corrected_up.x;
    result.m[2] = -forward.x;
    result.m[4] = side.y;
    result.m[5] = corrected_up.y;
    result.m[6] = -forward.y;
    result.m[8] = side.z;
    result.m[9] = corrected_up.z;
    result.m[10] = -forward.z;
    result.m[12] = -v3_dot(side, eye);
    result.m[13] = -v3_dot(corrected_up, eye);
    result.m[14] = v3_dot(forward, eye);

    return result;
}

static bool reserve_vertices(RENDERER *renderer, uint32_t needed) {
    if (needed <= renderer->vertex_capacity) return true;

    uint32_t capacity = renderer->vertex_capacity ? renderer->vertex_capacity : 1024u;

    while (capacity < needed) {
        if (capacity > UINT32_MAX / 2u) return false;
        capacity *= 2u;
    }

    RENDER_VERTEX *vertices = realloc(renderer->vertices, (size_t)capacity * sizeof(*vertices));

    if (!vertices) return false;

    renderer->vertices = vertices;
    renderer->vertex_capacity = capacity;

    return true;
}

static bool push_surface(RENDERER *renderer, const GLTF_VERTEX *vertex, LMAP_UV uv) {
    if (!reserve_vertices(renderer, renderer->vertex_count + 1u)) return false;

    renderer->vertices[renderer->vertex_count++] = (RENDER_VERTEX){
        .x = vertex->position.x,
        .y = vertex->position.y,
        .z = vertex->position.z,
        .nx = vertex->normal.x,
        .ny = vertex->normal.y,
        .nz = vertex->normal.z,
        .u = vertex->u,
        .v = vertex->v,
        .lu = uv.u,
        .lv = uv.v,
        .r = 1.0f,
        .g = 1.0f,
        .b = 1.0f,
        .a = 1.0f
    };

    return true;
}

static bool push_line_vertex(RENDERER *renderer, VEC3 position, COLOR4 color) {
    if (!reserve_vertices(renderer, renderer->vertex_count + 1u)) return false;

    renderer->vertices[renderer->vertex_count++] = (RENDER_VERTEX){
        .x = position.x,
        .y = position.y,
        .z = position.z,
        .r = color.r,
        .g = color.g,
        .b = color.b,
        .a = color.a
    };

    return true;
}

static bool add_wire_triangle(RENDERER *renderer, VEC3 a, VEC3 b, VEC3 c, COLOR4 color) {
    return push_line_vertex(renderer, a, color) && push_line_vertex(renderer, b, color) && push_line_vertex(renderer, b, color) &&
           push_line_vertex(renderer, c, color) && push_line_vertex(renderer, c, color) && push_line_vertex(renderer, a, color);
}

static void free_probe_grid(PROBE_GRID *grid) {
    if (!grid) return;

    free(grid->probes);
    memset(grid, 0, sizeof(*grid));
}

static bool make_probe_grid(const MESH *mesh, float spacing, PROBE_GRID *grid) {
    if (!mesh || !grid || spacing <= 0.0f) return false;

    memset(grid, 0, sizeof(*grid));

    const VEC3 extent = v3_sub(mesh->bounds.max, mesh->bounds.min);

    if (!isfinite(extent.x) || !isfinite(extent.y) || !isfinite(extent.z) || extent.x < 0.0f || extent.y < 0.0f || extent.z < 0.0f) return false;

    if (extent.x / spacing > 16384.0f || extent.y / spacing > 16384.0f || extent.z / spacing > 16384.0f) return false;

    grid->count_x = (uint32_t)ceilf(extent.x / spacing) + 1u;
    grid->count_y = (uint32_t)ceilf(extent.y / spacing) + 1u;
    grid->count_z = (uint32_t)ceilf(extent.z / spacing) + 1u;

    const uint64_t count = (uint64_t)grid->count_x * grid->count_y * grid->count_z;

    if (!count || count > 16384u) return false;

    grid->origin = mesh->bounds.min;
    grid->spacing = spacing;
    grid->probes = calloc((size_t)count, sizeof(*grid->probes));

    if (!grid->probes) return false;

    for (uint32_t z = 0; z < grid->count_z; ++z) {
        for (uint32_t y = 0; y < grid->count_y; ++y) {
            for (uint32_t x = 0; x < grid->count_x; ++x) {
                const size_t index = x + (size_t)grid->count_x * (y + (size_t)grid->count_y * z);
                PROBE *probe = &grid->probes[index];

                probe->position[0] = grid->origin.x + x * spacing;
                probe->position[1] = grid->origin.y + y * spacing;
                probe->position[2] = grid->origin.z + z * spacing;
                probe->position[3] = 1.0f;
            }
        }
    }

    return true;
}

bool renderer_load_cached_lightmap(
    RENDERER *renderer,
    const char *path,
    uint64_t scene_hash,
    uint64_t layout_hash,
    uint64_t volume_hash,
    uint64_t beam_hash,
    const LIGHTMAP *lightmap
) {
    if (!renderer || !renderer->device || !lightmap) return false;

    CACHED_LIGHTMAP cached = {0};

    if (!cache_read(path, scene_hash, layout_hash, volume_hash, beam_hash, &cached)) return false;

    NriTexture *replacement = NULL;
    NriBuffer *volume_buffer = NULL;
    NriBuffer *beam_buffer = NULL;
    bool good = cached.width == lightmap->width && cached.height == lightmap->height;

    if (good) {
        replacement = upload_lightmap(renderer, &cached);
        good = replacement != NULL;
    }

    if (good) {
        volume_buffer = upload_probes(renderer, &cached.volume_probes);
        good = volume_buffer != NULL;
    }

    if (good) {
        beam_buffer = upload_beams(renderer, &cached.beams);
        good = beam_buffer != NULL;
    }

    if (good) {
        NriTexture *old = renderer->lightmap_texture;
        NriBuffer *old_volume = renderer->volume_probe_buffer;
        NriBuffer *old_beam = renderer->beam_buffer;

        renderer->lightmap_texture = replacement;
        renderer->volume_probe_buffer = volume_buffer;
        renderer->beam_buffer = beam_buffer;
        renderer->lightmap_width = cached.width;
        renderer->lightmap_height = cached.height;

        free_probe_grid(&renderer->volume_probes);
        renderer->volume_probes = cached.volume_probes;
        cached.volume_probes.probes = NULL;

        beam_free(&renderer->beams);
        renderer->beams = cached.beams;
        cached.beams.cells = NULL;
        cached.beams.shadow_depth = NULL;

        renderer->has_bake = true;
        release_texture(renderer, old);
        release_buffer(renderer, old_volume);
        release_buffer(renderer, old_beam);
    } else {
        release_texture(renderer, replacement);
        release_buffer(renderer, volume_buffer);
        release_buffer(renderer, beam_buffer);
    }

    cache_free(&cached);

    return good;
}

bool renderer_rebake_current_scene(
    RENDERER *renderer,
    const MESH *mesh,
    const GLTF_SCENE *visual,
    const LIGHTMAP *lightmap,
    const struct LIGHT *light,
    const SKY *sky,
    const VOLUMETRICS_LIGHTING *volumetrics,
    const char *path,
    uint64_t scene_hash,
    uint64_t layout_hash,
    uint64_t volume_hash,
    uint64_t beam_hash
) {
    if (!renderer || !mesh || !lightmap || !renderer->device || !light || light->type != LIGHT_DIRECTIONAL || !sky || !volumetrics) return false;

    renderer->sun = light->directional;
    renderer->sun.direction = v3_normalize(renderer->sun.direction);

    if (v3_len_sq(renderer->sun.direction) <= 0.0f) return false;

    renderer->sky = *sky;
    renderer->volumetrics = *volumetrics;

    CACHED_LIGHTMAP previous = {0};
    bool reuse = cache_read_partial(path, scene_hash, &previous);

    if (reuse && previous.layout_hash == layout_hash && previous.volume_hash == volume_hash && previous.beam_hash == beam_hash) reuse = false;

    if (!reuse) cache_free(&previous);

    const bool reuse_lightmap = reuse && previous.layout_hash == layout_hash && previous.width == lightmap->width && previous.height == lightmap->height;
    const bool reuse_volume = reuse && previous.volume_hash == volume_hash;
    const bool reuse_beams = reuse && previous.beam_hash == beam_hash;

    bake_progress(renderer, "scene geometry", 0u, 0u);

    Uint64 started = SDL_GetPerformanceCounter();
    BVH tree = {0};

    if (!bvh_build(&tree, mesh, visual)) {
        cache_free(&previous);
        return false;
    }

    bake_timing("scene geometry", started);

    PROBE_GRID volume_candidate = {0};
    BEAM_GRID beam_candidate = {0};

    bake_progress(renderer, "volume probes", 0u, 0u);
    started = SDL_GetPerformanceCounter();

    if (reuse_volume) {
        volume_candidate = previous.volume_probes;
        previous.volume_probes.probes = NULL;
        SDL_Log("B: reused cached volume probes");
    } else {
        const bool made = make_probe_grid(mesh, renderer->volumetrics.probe_spacing, &volume_candidate) &&
                          bake_probe_grid(renderer, &volume_candidate, renderer->volumetrics.probe_samples);

        if (!made) {
            free_probe_grid(&volume_candidate);
            bvh_free(&tree);
            cache_free(&previous);
            return false;
        }
    }

    bake_timing("volume probes", started);
    bake_progress(renderer, "lightmap shader", 0u, 0u);

    NriTexture *old = renderer->lightmap_texture;
    const Uint32 old_width = renderer->lightmap_width;
    const Uint32 old_height = renderer->lightmap_height;
    const bool had_bake = renderer->has_bake;

    renderer->lightmap_texture = NULL;
    started = SDL_GetPerformanceCounter();

    bool good = false;

    if (reuse_lightmap) {
        renderer->lightmap_texture = upload_lightmap(renderer, &previous);
        good = renderer->lightmap_texture && upload_bvh(renderer, &tree);

        if (good) {
            renderer->lightmap_width = previous.width;
            renderer->lightmap_height = previous.height;
            SDL_Log("B: reused cached surface lightmap");
        }
    } else {
        good = bake_lightmap(renderer, &tree, lightmap, &volume_candidate);
    }

    if (good) bake_timing(reuse_lightmap ? "cached lightmap upload submission" : "lightmap GPU submission", started);

    if (good) bake_progress(renderer, "sun visibility", 0u, 0u);

    started = SDL_GetPerformanceCounter();

    if (good && reuse_beams) {
        beam_candidate = previous.beams;
        previous.beams.cells = NULL;
        previous.beams.shadow_depth = NULL;
        SDL_Log("B: reused cached sun beams");
    } else if (good) {
        good = beam_build(&beam_candidate, mesh, &tree, renderer->sun.direction);
    }

    if (good) bake_timing("sun visibility", started);

    if (good) SDL_Log("B: compressed sun beams into %u cells", beam_candidate.count);

    NriBuffer *volume_buffer = NULL;
    NriBuffer *beam_buffer = NULL;

    if (good) {
        volume_buffer = upload_probes(renderer, &volume_candidate);
        good = volume_buffer != NULL;
    }

    if (good) {
        beam_buffer = upload_beams(renderer, &beam_candidate);
        good = beam_buffer != NULL;
    }

    CACHED_LIGHTMAP candidate = {0};

    if (good) {
        bake_progress(renderer, "saving cache", 0u, 0u);
        started = SDL_GetPerformanceCounter();
        candidate.volume_probes = volume_candidate;
        candidate.beams = beam_candidate;

        if (reuse_lightmap) {
            candidate.pixels = previous.pixels;
            candidate.width = previous.width;
            candidate.height = previous.height;
            previous.pixels = NULL;
        } else {
            good = download_lightmap(renderer, &candidate);
        }

        if (good) {
            const Uint64 write_started = SDL_GetPerformanceCounter();
            good = cache_write(path, scene_hash, layout_hash, volume_hash, beam_hash, &candidate);

            if (good) bake_timing("cache serialization total", write_started);
        }

        candidate.volume_probes.probes = NULL;
        candidate.beams.cells = NULL;
        candidate.beams.shadow_depth = NULL;

        if (good) bake_timing("readback and cache write", started);
    }

    cache_free(&candidate);
    cache_free(&previous);
    release_bake_resources(renderer);
    bvh_free(&tree);

    if (good) {
        NriBuffer *old_volume = renderer->volume_probe_buffer;
        NriBuffer *old_beam = renderer->beam_buffer;

        renderer->volume_probe_buffer = volume_buffer;
        renderer->beam_buffer = beam_buffer;

        free_probe_grid(&renderer->volume_probes);
        renderer->volume_probes = volume_candidate;

        beam_free(&renderer->beams);
        renderer->beams = beam_candidate;

        renderer->has_bake = true;
        release_texture(renderer, old);
        release_buffer(renderer, old_volume);
        release_buffer(renderer, old_beam);
    } else {
        release_buffer(renderer, volume_buffer);
        release_buffer(renderer, beam_buffer);
        beam_free(&beam_candidate);
        free_probe_grid(&volume_candidate);
        release_texture(renderer, renderer->lightmap_texture);
        renderer->lightmap_texture = old;
        renderer->lightmap_width = old_width;
        renderer->lightmap_height = old_height;
        renderer->has_bake = had_bake;
    }

    return good;
}

static bool renderer_build_scene(RENDERER *renderer, const MESH *mesh, const GLTF_SCENE *visual, const LIGHTMAP *lightmap) {
    if (!renderer || !mesh || !visual || !lightmap || !visual->vertex_count || visual->vertex_count % 3u || visual->vertex_count / 3u != mesh->faces.count ||
        !visual->material_count || !lightmap->uvs) {
        SDL_Log("render/lightmap geometry mismatch");
        return false;
    }

    renderer->target = mesh->bounds.center;
    renderer->scene_radius = fmaxf(mesh->bounds.extents.x, fmaxf(mesh->bounds.extents.y, mesh->bounds.extents.z));

    if (renderer->scene_radius < 1.0f) renderer->scene_radius = 1.0f;

    renderer->distance = renderer->scene_radius * 2.15f;

    free(renderer->draws);
    renderer->draws = calloc(visual->material_count, sizeof(*renderer->draws));

    if (!renderer->draws) return false;

    renderer->draw_count = 0;
    renderer->vertex_count = 0;

    const size_t triangle_count = visual->vertex_count / 3u;

    for (uint32_t material = 0; material < visual->material_count; ++material) {
        const uint32_t first = renderer->vertex_count;

        for (size_t triangle = 0; triangle < triangle_count; ++triangle) {
            const GLTF_VERTEX *vertices = &visual->vertices[triangle * 3u];

            if (vertices[0].material != material) continue;

            const LMAP_UV *uv = &lightmap->uvs[triangle * 6u];

            if (!push_surface(renderer, &vertices[0], uv[0]) || !push_surface(renderer, &vertices[1], uv[1]) || !push_surface(renderer, &vertices[2], uv[2])) return false;
        }

        const uint32_t count = renderer->vertex_count - first;

        if (count) renderer->draws[renderer->draw_count++] = (DRAW_RANGE){first, count, material};
    }

    renderer->debug_vertex_start = renderer->vertex_count;

    const COLOR4 wire = {0.18f, 0.95f, 0.24f, 1.0f};

    for (size_t triangle = 0; triangle < triangle_count; ++triangle) {
        const GLTF_VERTEX *vertices = &visual->vertices[triangle * 3u];

        if (!add_wire_triangle(renderer, vertices[0].position, vertices[1].position, vertices[2].position, wire)) return false;
    }

    renderer->debug_vertex_count = renderer->vertex_count - renderer->debug_vertex_start;

    if (!upload_scene(renderer, visual)) return false;

    renderer->has_bake = false;
    SDL_Log("materials: %u | material draw ranges: %u | embedded images: %u", renderer->material_count, renderer->draw_count, renderer->image_texture_count);

    return true;
}

static void renderer_handle_event(RENDERER *renderer, const SDL_Event *event) {
    if (!renderer || !event) return;

    switch (event->type) {
        case SDL_EVENT_MOUSE_BUTTON_DOWN:
            if (event->button.button == SDL_BUTTON_LEFT) renderer->dragging = true;
            break;

        case SDL_EVENT_MOUSE_BUTTON_UP:
            if (event->button.button == SDL_BUTTON_LEFT) renderer->dragging = false;
            break;

        case SDL_EVENT_MOUSE_MOTION:
            if (renderer->dragging) {
                renderer->yaw += event->motion.xrel * 0.0075f;
                renderer->pitch += event->motion.yrel * 0.0075f;

                if (renderer->pitch > 1.45f) renderer->pitch = 1.45f;
                if (renderer->pitch < -1.45f) renderer->pitch = -1.45f;
            }
            break;

        case SDL_EVENT_MOUSE_WHEEL:
            renderer->distance -= event->wheel.y * (renderer->distance * 0.08f);

            if (renderer->distance < renderer->scene_radius * 0.05f) renderer->distance = renderer->scene_radius * 0.05f;
            if (renderer->distance > renderer->scene_radius * 20.0f) renderer->distance = renderer->scene_radius * 20.0f;
            break;

        case SDL_EVENT_KEY_DOWN:
            if (!event->key.repeat && event->key.key == SDLK_TAB) renderer->show_debug = !renderer->show_debug;
            if (!event->key.repeat && event->key.key == SDLK_F5) renderer->show_volume = !renderer->show_volume;

            if (!event->key.repeat && (event->key.key == SDLK_F1 || event->key.key == SDLK_F3 || event->key.key == SDLK_F4)) {
                const uint32_t view = (uint32_t)(event->key.key - SDLK_F1) + 1u;
                renderer->debug_view = renderer->debug_view == view ? 0u : view;
            }
            break;

        default:
            break;
    }
}

static bool renderer_draw(RENDERER *renderer, const struct LIGHT *light, const SKY *sky, const VOLUMETRICS_LIGHTING *volumetrics, const PERIPHERAL_VISION *vision) {
    if (!renderer || !renderer->window || !light || light->type != LIGHT_DIRECTIONAL || !sky || !volumetrics || !vision) return false;

    DIRECTIONAL_LIGHT sun = light->directional;
    sun.direction = v3_normalize(sun.direction);

    if (v3_len_sq(sun.direction) <= 0.0f) return false;

    renderer->sun = sun;
    renderer->sky = *sky;

    int width = 0;
    int height = 0;

    if (!SDL_GetWindowSizeInPixels(renderer->window, &width, &height)) return false;
    if (width <= 0 || height <= 0) return true;

    const float fov = 62.0f * 3.14159265358979323846f / 180.0f;
    const float cp = cosf(renderer->pitch);
    const VEC3 eye = v3(
        renderer->target.x + renderer->distance * cp * cosf(renderer->yaw),
        renderer->target.y + renderer->distance * sinf(renderer->pitch),
        renderer->target.z + renderer->distance * cp * sinf(renderer->yaw)
    );
    const VEC3 forward = v3_normalize(v3_sub(renderer->target, eye));
    const VEC3 right = v3_normalize(v3_cross(forward, v3(0.0f, 1.0f, 0.0f)));
    const VEC3 up = v3_cross(right, forward);
    const float aspect = (float)width / (float)height;
    const float tan_half = tanf(fov * 0.5f);
    const float znear = fmaxf(0.02f, renderer->scene_radius * 0.005f);
    const float zfar = fmaxf(100.0f, renderer->scene_radius * 10.0f);
    const MAT4 view = m4_look_at(eye, renderer->target, v3(0.0f, 1.0f, 0.0f));
    const MAT4 projection = m4_perspective(fov, aspect, znear, zfar);
    const MAT4 mvp = m4_mul(projection, view);

    RENDER_FRAME frame = {
        .eye = eye,
        .right = right,
        .up = up,
        .forward = forward,
        .sun = sun,
        .sky = *sky,
        .volumetrics = *volumetrics,
        .vision = *vision,
        .tan_half_fov = tan_half,
        .aspect = aspect
    };

    memcpy(frame.mvp, mvp.m, sizeof(frame.mvp));
    memcpy(frame.view, view.m, sizeof(frame.view));

    return draw_frame(renderer, &frame);
}

static struct MODEL *scene_model(const SCENE *scene) {
    if (!scene) return NULL;

    for (uint32_t i = 0; i < scene->object_count; ++i) {
        OBJECT *object = &scene->objects[i];

        if (object->type == MODEL && object->data) return object->data;
    }

    return NULL;
}

static struct LIGHT *scene_directional_light(const SCENE *scene) {
    if (!scene) return NULL;

    for (uint32_t i = 0; i < scene->object_count; ++i) {
        OBJECT *object = &scene->objects[i];

        if (object->type != LIGHT || !object->data) continue;

        struct LIGHT *light = object->data;

        if (light->type == LIGHT_DIRECTIONAL) return light;
    }

    return NULL;
}

bool gpu_init(GPU *gpu, const char *title, int width, int height) {
    if (!gpu || !title || width <= 0 || height <= 0) return false;

    *gpu = (GPU){
        .title = title,
        .width = width,
        .height = height
    };

    return true;
}

void gpu_deinit(GPU *gpu) {
    if (!gpu) return;
    memset(gpu, 0, sizeof(*gpu));
}

bool renderer_init(RENDERER *renderer, GPU *gpu) {
    if (!renderer || !gpu || !gpu->title || gpu->width <= 0 || gpu->height <= 0) return false;
    if (!r_init(renderer, gpu->title, gpu->width, gpu->height)) return false;

    renderer->gpu = gpu;
    return true;
}

bool renderer_set_scene(RENDERER *renderer, SCENE *scene) {
    if (!renderer || !scene || !scene->lightmap) return false;

    struct MODEL *model = scene_model(scene);

    if (!model || !model->geometry || !model->visual) return false;
    if (!renderer_build_scene(renderer, model->geometry, model->visual, scene->lightmap)) return false;

    renderer->scene = scene;
    scene->radius = renderer->scene_radius;

    return true;
}

void renderer_event(RENDERER *renderer, const SDL_Event *event) {
    renderer_handle_event(renderer, event);
}

bool renderer_frame(RENDERER *renderer) {
    if (!renderer || !renderer->scene) return false;

    struct LIGHT *light = scene_directional_light(renderer->scene);

    if (!light) return false;

    return renderer_draw(renderer, light, &renderer->scene->sky, &renderer->scene->volumetrics, &renderer->scene->vision);
}

void renderer_deinit(RENDERER *renderer) {
    if (!renderer) return;
    r_deinit(renderer);
}
