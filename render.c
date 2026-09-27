#include "game.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct mat4 {
    float m[16];
} mat4;

typedef struct color4 {
    float r, g, b, a;
} color4;

void bake_progress(renderer *r, const char *stage, Uint32 done, Uint32 total) {
    if (!r || !r->window) return;

    r->bake_stage = stage;

    char title[160];

    if (total) {
        snprintf(title, sizeof(title), "Pony - B baking %s: %u/%u", stage, done, total);

        const double valid_texels = (double)r->lightmap_sample_count;
        const double completed_work = valid_texels * (double)done / (double)total;

        printf("frame time: %.2f ms | bake: %.2e/%.2e (%s)\n", r->frame_time_ms, completed_work, valid_texels, stage);
        fflush(stdout);
    } else {
        snprintf(title, sizeof(title), "Pony - B baking %s...", stage);
    }

    SDL_SetWindowTitle(r->window, title);
    SDL_PumpEvents();
}

static void bake_timing(const char *stage, Uint64 started) {
    const double elapsed = (double)(SDL_GetPerformanceCounter() - started) * 1000.0 / (double)SDL_GetPerformanceFrequency();

    SDL_Log("B: %s took %.2f ms", stage, elapsed);
}

static vec3 scene_sun_direction(void) {
    return v3_normalize(v3(0.38f, 0.30f, 0.32f));
}

static mat4 m4_identity(void) {
    mat4 r = {0};
    r.m[0] = r.m[5] = r.m[10] = r.m[15] = 1.0f;

    return r;
}

static mat4 m4_mul(mat4 a, mat4 b) {
    mat4 r = {0};

    for (int c = 0; c < 4; ++c) {
        for (int row = 0; row < 4; ++row) {
            r.m[c * 4 + row] = a.m[row] * b.m[c * 4] + a.m[4 + row] * b.m[c * 4 + 1] + a.m[8 + row] * b.m[c * 4 + 2] + a.m[12 + row] * b.m[c * 4 + 3];
        }
    }

    return r;
}

static mat4 m4_perspective(float fov_y, float aspect, float znear, float zfar) {
    const float f = 1.0f / tanf(fov_y * 0.5f);
    mat4 r = {0};
    r.m[0] = f / aspect;
    r.m[5] = f;
    r.m[10] = zfar / (znear - zfar);
    r.m[11] = -1.0f;
    r.m[14] = (znear * zfar) / (znear - zfar);

    return r;
}

static mat4 m4_look_at(vec3 eye, vec3 target, vec3 up) {
    const vec3 f = v3_normalize(v3_sub(target, eye));
    const vec3 s = v3_normalize(v3_cross(f, up));
    const vec3 u = v3_cross(s, f);
    mat4 r = m4_identity();

    r.m[0] = s.x;
    r.m[1] = u.x;
    r.m[2] = -f.x;
    r.m[4] = s.y;
    r.m[5] = u.y;
    r.m[6] = -f.y;
    r.m[8] = s.z;
    r.m[9] = u.z;
    r.m[10] = -f.z;
    r.m[12] = -v3_dot(s, eye);
    r.m[13] = -v3_dot(u, eye);
    r.m[14] = v3_dot(f, eye);

    return r;
}

static bool reserve_vertices(renderer *r, uint32_t needed) {
    if (needed <= r->vertex_capacity) return true;

    uint32_t capacity = r->vertex_capacity ? r->vertex_capacity : 1024u;

    while (capacity < needed) {
        if (capacity > UINT32_MAX / 2u) return false;
        capacity *= 2u;
    }

    render_vertex *vertices = realloc(r->vertices, (size_t)capacity * sizeof(*vertices));

    if (!vertices) return false;
    r->vertices = vertices;
    r->vertex_capacity = capacity;

    return true;
}

static bool push_surface(renderer *r, const gltf_vertex *v, lmap_uv uv) {
    if (!reserve_vertices(r, r->vertex_count + 1u)) return false;
    r->vertices[r->vertex_count++] = (render_vertex){.x = v->position.x,
                                                     .y = v->position.y,
                                                     .z = v->position.z,
                                                     .nx = v->normal.x,
                                                     .ny = v->normal.y,
                                                     .nz = v->normal.z,
                                                     .u = v->u,
                                                     .v = v->v,
                                                     .lu = uv.u,
                                                     .lv = uv.v,
                                                     .r = 1,
                                                     .g = 1,
                                                     .b = 1,
                                                     .a = 1};

    return true;
}

static bool push_line_vertex(renderer *r, vec3 p, color4 c) {
    if (!reserve_vertices(r, r->vertex_count + 1u)) return false;
    r->vertices[r->vertex_count++] = (render_vertex){.x = p.x, .y = p.y, .z = p.z, .r = c.r, .g = c.g, .b = c.b, .a = c.a};

    return true;
}

static bool add_wire_triangle(renderer *r, vec3 a, vec3 b, vec3 c, color4 color) {
    return push_line_vertex(r, a, color) && push_line_vertex(r, b, color) && push_line_vertex(r, b, color) && push_line_vertex(r, c, color) && push_line_vertex(r, c, color) &&
           push_line_vertex(r, a, color);
}

static void free_probe_grid(probe_grid *grid) {
    if (!grid) return;
    free(grid->probes);
    memset(grid, 0, sizeof(*grid));
}

static bool make_probe_grid(const mesh *m, float spacing, probe_grid *grid) {
    if (!m || !grid || spacing <= 0.0f) return false;
    memset(grid, 0, sizeof(*grid));

    const vec3 extent = v3_sub(m->bounds.max, m->bounds.min);

    if (!isfinite(extent.x) || !isfinite(extent.y) || !isfinite(extent.z) || extent.x < 0.0f || extent.y < 0.0f || extent.z < 0.0f) return false;

    if (extent.x / spacing > 16384.0f || extent.y / spacing > 16384.0f || extent.z / spacing > 16384.0f) return false;

    grid->count_x = (uint32_t)ceilf(extent.x / spacing) + 1u;
    grid->count_y = (uint32_t)ceilf(extent.y / spacing) + 1u;
    grid->count_z = (uint32_t)ceilf(extent.z / spacing) + 1u;

    const uint64_t count = (uint64_t)grid->count_x * grid->count_y * grid->count_z;

    if (!count || count > 16384u) return false;

    grid->origin = m->bounds.min;
    grid->spacing = spacing;
    grid->probes = calloc((size_t)count, sizeof(*grid->probes));

    if (!grid->probes) return false;

    for (uint32_t z = 0; z < grid->count_z; ++z) {
        for (uint32_t y = 0; y < grid->count_y; ++y) {
            for (uint32_t x = 0; x < grid->count_x; ++x) {
                const size_t index = x + (size_t)grid->count_x * (y + (size_t)grid->count_y * z);

                probe *p = &grid->probes[index];
                p->position[0] = grid->origin.x + x * spacing;
                p->position[1] = grid->origin.y + y * spacing;
                p->position[2] = grid->origin.z + z * spacing;
                p->position[3] = 1.0f;
            }
        }
    }

    return true;
}

bool r_load_cached_lightmap(renderer *r, const char *path, uint64_t scene_hash, uint64_t layout_hash, uint64_t volume_hash, uint64_t beam_hash, const lightmap *lm) {
    if (!r || !r->device || !lm) return false;

    cached_lightmap cached = {0};

    if (!cache_read(path, scene_hash, layout_hash, volume_hash, beam_hash, &cached)) return false;

    NriTexture *replacement = NULL;
    NriBuffer *volume_buffer = NULL;
    NriBuffer *beam_buffer = NULL;

    bool good = cached.width == lm->width && cached.height == lm->height;

    if (good) {
        replacement = upload_lightmap(r, &cached);
        good = replacement != NULL;
    }

    if (good) {
        volume_buffer = upload_probes(r, &cached.volume_probes);
        good = volume_buffer != NULL;
    }

    if (good) {
        beam_buffer = upload_beams(r, &cached.beams);
        good = beam_buffer != NULL;
    }

    if (good) {
        NriTexture *old = r->lightmap_texture;
        NriBuffer *old_volume = r->volume_probe_buffer;
        NriBuffer *old_beam = r->beam_buffer;

        r->lightmap_texture = replacement;
        r->volume_probe_buffer = volume_buffer;
        r->beam_buffer = beam_buffer;
        r->lightmap_width = cached.width;
        r->lightmap_height = cached.height;

        free_probe_grid(&r->volume_probes);
        r->volume_probes = cached.volume_probes;
        cached.volume_probes.probes = NULL;

        beam_free(&r->beams);
        r->beams = cached.beams;
        cached.beams.cells = NULL;
        cached.beams.shadow_depth = NULL;

        r->has_bake = true;
        release_texture(r, old);
        release_buffer(r, old_volume);
        release_buffer(r, old_beam);
    } else {
        release_texture(r, replacement);
        release_buffer(r, volume_buffer);
        release_buffer(r, beam_buffer);
    }

    cache_free(&cached);

    return good;
}

bool r_rebake_current_scene(renderer *r, const mesh *m, const gltf_scene *visual, const lightmap *lm, const char *path, uint64_t scene_hash, uint64_t layout_hash,
                            uint64_t volume_hash, uint64_t beam_hash) {
    if (!r || !m || !lm || !r->device) return false;

    cached_lightmap previous = {0};
    bool reuse = cache_read_partial(path, scene_hash, &previous);

    if (reuse && previous.layout_hash == layout_hash && previous.volume_hash == volume_hash && previous.beam_hash == beam_hash) reuse = false;

    if (!reuse) cache_free(&previous);

    const bool reuse_lightmap = reuse && previous.layout_hash == layout_hash && previous.width == lm->width && previous.height == lm->height;

    const bool reuse_volume = reuse && previous.volume_hash == volume_hash;
    const bool reuse_beams = reuse && previous.beam_hash == beam_hash;

    bake_progress(r, "scene geometry", 0u, 0u);

    Uint64 started = SDL_GetPerformanceCounter();

    bvh tree = {0};

    if (!bvh_build(&tree, m, visual)) {
        cache_free(&previous);

        return false;
    }

    bake_timing("scene geometry", started);

    probe_grid volume_candidate = {0};
    beam_grid beam_candidate = {0};

    bake_progress(r, "volume probes", 0u, 0u);

    started = SDL_GetPerformanceCounter();

    if (reuse_volume) {
        volume_candidate = previous.volume_probes;

        previous.volume_probes.probes = NULL;
        SDL_Log("B: reused cached volume probes");
    } else {
        bool made = make_probe_grid(m, 4.0f, &volume_candidate) && bake_probe_grid(r, &volume_candidate, 1024u);

        if (!made) {
            free_probe_grid(&volume_candidate);
            bvh_free(&tree);
            cache_free(&previous);

            return false;
        }
    }

    bake_timing("volume probes", started);

    bake_progress(r, "lightmap shader", 0u, 0u);

    NriTexture *old = r->lightmap_texture;
    const Uint32 old_width = r->lightmap_width;
    const Uint32 old_height = r->lightmap_height;
    const bool had_bake = r->has_bake;

    r->lightmap_texture = NULL;

    started = SDL_GetPerformanceCounter();
    bool good = false;

    if (reuse_lightmap) {
        r->lightmap_texture = upload_lightmap(r, &previous);

        good = r->lightmap_texture && upload_bvh(r, &tree);

        if (good) {
            r->lightmap_width = previous.width;
            r->lightmap_height = previous.height;
            SDL_Log("B: reused cached surface lightmap");
        }
    } else {
        good = bake_lightmap(r, &tree, lm, &volume_candidate);
    }

    if (good) bake_timing(reuse_lightmap ? "cached lightmap upload submission" : "lightmap GPU submission", started);

    if (good) bake_progress(r, "sun visibility", 0u, 0u);

    started = SDL_GetPerformanceCounter();

    if (good && reuse_beams) {
        beam_candidate = previous.beams;

        previous.beams.cells = NULL;
        previous.beams.shadow_depth = NULL;
        SDL_Log("B: reused cached sun beams");
    } else if (good) {
        good = beam_build(&beam_candidate, m, &tree, scene_sun_direction());
    }

    if (good) bake_timing("sun visibility", started);

    if (good) SDL_Log("B: compressed sun beams into %u cells", beam_candidate.count);

    NriBuffer *volume_buffer = NULL;
    NriBuffer *beam_buffer = NULL;

    if (good) {
        volume_buffer = upload_probes(r, &volume_candidate);
        good = volume_buffer != NULL;
    }

    if (good) {
        beam_buffer = upload_beams(r, &beam_candidate);
        good = beam_buffer != NULL;
    }

    cached_lightmap candidate = {0};

    if (good) {
        bake_progress(r, "saving cache", 0u, 0u);

        started = SDL_GetPerformanceCounter();

        candidate.volume_probes = volume_candidate;
        candidate.beams = beam_candidate;

        if (reuse_lightmap) {
            candidate.pixels = previous.pixels;
            candidate.width = previous.width;
            candidate.height = previous.height;
            previous.pixels = NULL;
        } else {
            good = download_lightmap(r, &candidate);
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
    release_bake_resources(r);
    bvh_free(&tree);

    if (good) {
        NriBuffer *old_volume = r->volume_probe_buffer;
        NriBuffer *old_beam = r->beam_buffer;

        r->volume_probe_buffer = volume_buffer;
        r->beam_buffer = beam_buffer;

        free_probe_grid(&r->volume_probes);
        r->volume_probes = volume_candidate;

        beam_free(&r->beams);
        r->beams = beam_candidate;

        r->has_bake = true;
        release_texture(r, old);
        release_buffer(r, old_volume);
        release_buffer(r, old_beam);
    } else {
        release_buffer(r, volume_buffer);
        release_buffer(r, beam_buffer);
        beam_free(&beam_candidate);
        free_probe_grid(&volume_candidate);
        release_texture(r, r->lightmap_texture);
        r->lightmap_texture = old;
        r->lightmap_width = old_width;
        r->lightmap_height = old_height;
        r->has_bake = had_bake;
    }

    return good;
}

bool r_build_scene(renderer *r, const mesh *m, const gltf_scene *visual, const lightmap *lm) {
    if (!r || !m || !visual || !lm || !visual->vertex_count || visual->vertex_count % 3u || visual->vertex_count / 3u != m->faces.count || !visual->material_count || !lm->uvs) {
        SDL_Log("render/lightmap geometry mismatch");

        return false;
    }

    r->target = m->bounds.center;
    r->scene_radius = fmaxf(m->bounds.extents.x, fmaxf(m->bounds.extents.y, m->bounds.extents.z));

    if (r->scene_radius < 1.0f) r->scene_radius = 1.0f;
    r->distance = r->scene_radius * 2.15f;

    free(r->draws);
    r->draws = calloc(visual->material_count, sizeof(*r->draws));

    if (!r->draws) return false;
    r->draw_count = 0;
    r->vertex_count = 0;

    const size_t triangle_count = visual->vertex_count / 3u;

    for (uint32_t material = 0; material < visual->material_count; ++material) {
        const uint32_t first = r->vertex_count;

        for (size_t triangle = 0; triangle < triangle_count; ++triangle) {
            const gltf_vertex *v = &visual->vertices[triangle * 3u];

            if (v[0].material != material) continue;

            const lmap_uv *uv = &lm->uvs[triangle * 6u];

            if (!push_surface(r, &v[0], uv[0]) || !push_surface(r, &v[1], uv[1]) || !push_surface(r, &v[2], uv[2])) return false;
        }

        const uint32_t count = r->vertex_count - first;

        if (count) r->draws[r->draw_count++] = (draw_range){first, count, material};
    }

    r->debug_vertex_start = r->vertex_count;

    const color4 wire = {0.18f, 0.95f, 0.24f, 1.0f};

    for (size_t triangle = 0; triangle < triangle_count; ++triangle) {
        const gltf_vertex *v = &visual->vertices[triangle * 3u];

        if (!add_wire_triangle(r, v[0].position, v[1].position, v[2].position, wire)) return false;
    }

    r->debug_vertex_count = r->vertex_count - r->debug_vertex_start;

    if (!upload_scene(r, visual)) return false;

    r->has_bake = false;
    SDL_Log("materials: %u | material draw ranges: %u | embedded images: %u", r->material_count, r->draw_count, r->image_texture_count);

    return true;
}

void r_event(renderer *r, const SDL_Event *event) {
    if (!r || !event) return;

    switch (event->type) {
        case SDL_EVENT_MOUSE_BUTTON_DOWN:
            if (event->button.button == SDL_BUTTON_LEFT) r->dragging = true;

            break;

        case SDL_EVENT_MOUSE_BUTTON_UP:
            if (event->button.button == SDL_BUTTON_LEFT) r->dragging = false;

            break;

        case SDL_EVENT_MOUSE_MOTION:
            if (r->dragging) {
                r->yaw += event->motion.xrel * 0.0075f;
                r->pitch += event->motion.yrel * 0.0075f;

                if (r->pitch > 1.45f) r->pitch = 1.45f;

                if (r->pitch < -1.45f) r->pitch = -1.45f;
            }

            break;

        case SDL_EVENT_MOUSE_WHEEL:
            r->distance -= event->wheel.y * (r->distance * 0.08f);

            if (r->distance < r->scene_radius * 0.05f) r->distance = r->scene_radius * 0.05f;

            if (r->distance > r->scene_radius * 20.0f) r->distance = r->scene_radius * 20.0f;

            break;

        case SDL_EVENT_KEY_DOWN:
            if (!event->key.repeat && event->key.key == SDLK_TAB) r->show_debug = !r->show_debug;

            if (!event->key.repeat && event->key.key == SDLK_F5) r->show_volume = !r->show_volume;

            if (!event->key.repeat && (event->key.key == SDLK_F1 || event->key.key == SDLK_F3 || event->key.key == SDLK_F4)) {
                const uint32_t view = (uint32_t)(event->key.key - SDLK_F1) + 1u;

                r->debug_view = r->debug_view == view ? 0u : view;
            }

            break;

        default:
            break;
    }
}

bool r_draw(renderer *r) {
    if (!r || !r->window) return false;

    int width = 0;
    int height = 0;

    if (!SDL_GetWindowSizeInPixels(r->window, &width, &height)) return false;

    if (width <= 0 || height <= 0) return true;

    const float fov = 62.0f * 3.14159265358979323846f / 180.0f;
    const float cp = cosf(r->pitch);
    const vec3 eye = v3(r->target.x + r->distance * cp * cosf(r->yaw), r->target.y + r->distance * sinf(r->pitch), r->target.z + r->distance * cp * sinf(r->yaw));
    const vec3 forward = v3_normalize(v3_sub(r->target, eye));
    const vec3 right = v3_normalize(v3_cross(forward, v3(0, 1, 0)));
    const vec3 up = v3_cross(right, forward);
    const float aspect = (float)width / (float)height;
    const float tan_half = tanf(fov * 0.5f);
    const float znear = fmaxf(0.02f, r->scene_radius * 0.005f);
    const float zfar = fmaxf(100.0f, r->scene_radius * 10.0f);
    const mat4 view = m4_look_at(eye, r->target, v3(0, 1, 0));
    const mat4 proj = m4_perspective(fov, aspect, znear, zfar);
    const mat4 mvp = m4_mul(proj, view);

    render_frame frame = {.eye = eye, .right = right, .up = up, .forward = forward, .sun = scene_sun_direction(), .tan_half_fov = tan_half, .aspect = aspect};

    memcpy(frame.mvp, mvp.m, sizeof(frame.mvp));
    memcpy(frame.view, view.m, sizeof(frame.view));

    return draw_frame(r, &frame);
}
