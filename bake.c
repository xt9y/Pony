#include "game.h"

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef enum BAKE_PHASE { BAKE_PHASE_INIT = 0, BAKE_PHASE_SUN, BAKE_PHASE_PROBES, BAKE_PHASE_SEED, BAKE_PHASE_LIGHTMAP } BAKE_PHASE;

typedef struct BAKE_JOB {
    RENDERER *renderer;
    const MESH *scene;
    const GLTF_SCENE *visual;
    const LIGHTMAP *layout;
    const struct LIGHT *light;
    SKY sky;
    VOLUMETRICS_LIGHTING volumetrics;
    char *path;
    char *worker_path;
    uint64_t scene_hash;
    uint64_t layout_hash;
    uint64_t volume_hash;
    uint64_t beam_hash;
    SDL_Thread *thread;
    SDL_AtomicInt cancel;
    SDL_AtomicInt done;
    SDL_AtomicInt success;
    SDL_AtomicInt phase;
    SDL_AtomicInt phase_done;
    SDL_AtomicInt phase_total;
    SDL_AtomicInt phase_active;
    SDL_AtomicInt phase_started_ms;
    Uint64 started;
    uint64_t previous_cache_bytes;
    Uint64 save_seen_at;
    uint64_t save_seen_bytes;
    char error[256];
} BAKE_JOB;

static BAKE_JOB *g_bake;
static Uint64 g_title_tick;

static double bake_elapsed_ms(Uint64 started) {
    return (double)(SDL_GetPerformanceCounter() - started) * 1000.0 / (double)SDL_GetPerformanceFrequency();
}

static char *bake_path_suffix(const char *path, const char *suffix) {
    if (!path || !suffix) return NULL;

    const size_t a = strlen(path);
    const size_t b = strlen(suffix);

    if (a > SIZE_MAX - b - 1u) return NULL;

    char *result = malloc(a + b + 1u);

    if (!result) return NULL;
    memcpy(result, path, a);
    memcpy(result + a, suffix, b + 1u);

    return result;
}

static uint64_t bake_file_size(const char *path) {
    if (!path) return 0u;

    FILE *file = fopen(path, "rb");

    if (!file) return 0u;

    if (fseek(file, 0, SEEK_END) != 0) {
        fclose(file);

        return 0u;
    }

    const long size = ftell(file);

    fclose(file);

    return size > 0 ? (uint64_t)size : 0u;
}

static const char *bake_phase_name(int phase) {
    switch ((BAKE_PHASE)phase) {
        case BAKE_PHASE_SUN:
            return "SUN";
        case BAKE_PHASE_PROBES:
            return "PROBE";
        case BAKE_PHASE_SEED:
            return "CACHE";
        case BAKE_PHASE_LIGHTMAP:
            return "LIGHTMAP";
        default:
            return "BAKE";
    }
}

static void bake_set_phase(BAKE_PHASE phase, Uint32 done, Uint32 total, Uint32 active) {
    BAKE_JOB *job = g_bake;

    if (!job) return;

    const int previous = SDL_GetAtomicInt(&job->phase);

    if (previous != (int)phase) SDL_SetAtomicInt(&job->phase_started_ms, (int)(Uint32)SDL_GetTicks());
    SDL_SetAtomicInt(&job->phase_done, (int)done);
    SDL_SetAtomicInt(&job->phase_total, (int)total);
    SDL_SetAtomicInt(&job->phase_active, (int)active);
    SDL_SetAtomicInt(&job->phase, (int)phase);
}

static bool bake_cancelled(void) {
    return g_bake && SDL_GetAtomicInt(&g_bake->cancel) != 0;
}

static bool bake_probe_stats(const RENDERER *r, Uint32 *minimum, double *average, Uint32 *maximum, Uint32 *valid_count) {
    if (!r || !r->volume_probes.probes) return false;

    const uint64_t probe_count = (uint64_t)r->volume_probes.count_x * r->volume_probes.count_y * r->volume_probes.count_z;

    if (!probe_count) return false;

    Uint32 min_samples = UINT32_MAX;
    Uint32 max_samples = 0u;
    uint64_t sample_sum = 0u;
    Uint32 measured = 0u;

    for (uint64_t i = 0u; i < probe_count; ++i) {
        const PROBE *probe = &r->volume_probes.probes[i];

        if (probe->position[3] <= 0.0f) continue;

        const float encoded = probe->coefficients[2][3];

        if (!isfinite(encoded) || encoded < 1.0f || encoded > 65536.0f) continue;

        const Uint32 samples = (Uint32)(encoded + 0.5f);

        if (samples < min_samples) min_samples = samples;

        if (samples > max_samples) max_samples = samples;

        sample_sum += samples;
        measured++;
    }

    if (!measured) return false;

    if (minimum) *minimum = min_samples;

    if (average) *average = (double)sample_sum / (double)measured;

    if (maximum) *maximum = max_samples;

    if (valid_count) *valid_count = measured;

    return true;
}

static void bake_remove_worker_files(const char *path) {
    if (!path) return;
    remove(path);

    char *temporary = bake_path_suffix(path, ".tmp");

    if (temporary) {
        remove(temporary);
        free(temporary);
    }
}

static bool bake_copy_cache(const char *source_path, const char *destination_path) {
    FILE *source = fopen(source_path, "rb");

    if (!source) return false;

    FILE *destination = fopen(destination_path, "wb");

    if (!destination) {
        fclose(source);

        return false;
    }

    unsigned char *buffer = malloc(1024u * 1024u);
    bool good = buffer != NULL;

    while (good && !bake_cancelled()) {
        const size_t count = fread(buffer, 1, 1024u * 1024u, source);

        if (count && fwrite(buffer, 1, count, destination) != count) good = false;

        if (count < 1024u * 1024u) {
            if (ferror(source)) good = false;

            break;
        }
    }

    free(buffer);

    if (fclose(source) != 0) good = false;

    if (fclose(destination) != 0) good = false;

    if (bake_cancelled()) good = false;

    if (!good) remove(destination_path);

    return good;
}

static void bake_set_error(BAKE_JOB *job, const char *message) {
    if (!job) return;

    const char *text = message && *message ? message : "unknown bake error";
    snprintf(job->error, sizeof(job->error), "%s", text);
}

static bool bake_probe_progress(Uint32 done, Uint32 total, Uint32 active) {
    bake_set_phase(BAKE_PHASE_PROBES, done, total, active);

    return !bake_cancelled();
}

static bool bake_make_probe_grid(const MESH *m, float spacing, PROBE_GRID *grid) {
    if (!m || !grid || spacing <= 0.0f) return false;
    memset(grid, 0, sizeof(*grid));

    const VEC3 extent = v3_sub(m->bounds.max, m->bounds.min);

    if (!isfinite(extent.x) || !isfinite(extent.y) || !isfinite(extent.z) || extent.x < 0.0f || extent.y < 0.0f || extent.z < 0.0f) return false;

    float smallest_extent = INFINITY;
    if (extent.x > 0.0f) smallest_extent = fminf(smallest_extent, extent.x);
    if (extent.y > 0.0f) smallest_extent = fminf(smallest_extent, extent.y);
    if (extent.z > 0.0f) smallest_extent = fminf(smallest_extent, extent.z);
    if (isfinite(smallest_extent)) spacing = fminf(spacing, fmaxf(0.25f, smallest_extent * 0.5f));

    grid->count_x = (uint32_t)ceilf(extent.x / spacing) + 1u;
    grid->count_y = (uint32_t)ceilf(extent.y / spacing) + 1u;
    grid->count_z = (uint32_t)ceilf(extent.z / spacing) + 1u;

    const uint64_t count = (uint64_t)grid->count_x * grid->count_y * grid->count_z;

    if (!count || count > 16384u) return false;
    grid->origin = m->bounds.min;
    grid->spacing = spacing;
    grid->probes = calloc((size_t)count, sizeof(*grid->probes));

    if (!grid->probes) return false;

    for (uint32_t z = 0; z < grid->count_z; ++z)
        for (uint32_t y = 0; y < grid->count_y; ++y)
            for (uint32_t x = 0; x < grid->count_x; ++x) {
                const size_t index = x + (size_t)grid->count_x * (y + (size_t)grid->count_y * z);

                PROBE *p = &grid->probes[index];
                p->position[0] = grid->origin.x + x * spacing;
                p->position[1] = grid->origin.y + y * spacing;
                p->position[2] = grid->origin.z + z * spacing;
                p->position[3] = 1.0f;
            }

    return true;
}

static bool bake_write_fast_seed(BAKE_JOB *job, PROBE_GRID *probes, BEAM_GRID *beams) {
    unsigned char black_pixel[8] = {0, 0, 0, 0, 0, 0, 0x00, 0x3c};
    CACHED_LIGHTMAP seed = {0};

    seed.width = 1u;
    seed.height = 1u;
    seed.pixels = black_pixel;
    seed.direct_pixels = black_pixel;
    seed.volume_probes = *probes;
    seed.beams = *beams;

    const uint64_t stale_layout = job->layout_hash ^ UINT64_C(0x9e3779b97f4a7c15);

    bake_set_phase(BAKE_PHASE_SEED, 0u, 0u, 0u);

    return cache_write(job->worker_path, job->scene_hash, stale_layout, job->volume_hash, job->beam_hash, &seed);
}

static bool bake_prepare_fast_components(BAKE_JOB *job, RENDERER *worker) {
    Uint64 started = SDL_GetPerformanceCounter();
    BVH tree = {0};
    PROBE_GRID probes = {0};
    BEAM_GRID beams = {0};

    if (!job->light || job->light->type != LIGHT_DIRECTIONAL) return false;
    worker->sun = job->light->directional;
    worker->sun.direction = v3_normalize(worker->sun.direction);
    worker->sky = job->sky;
    worker->volumetrics = job->volumetrics;

    if (v3_len_sq(worker->sun.direction) <= 0.0f) return false;

    bool good = bvh_build(&tree, job->scene, job->visual);

    if (good) SDL_Log("B: fast probe BVH built in %.2f ms", bake_elapsed_ms(started));

    bake_set_phase(BAKE_PHASE_SUN, 0u, 0u, 0u);

    started = SDL_GetPerformanceCounter();

    if (good) good = beam_build(&beams, job->scene, &tree, worker->sun.direction);

    if (good) SDL_Log("B: fast sun field took %.2f ms", bake_elapsed_ms(started));

    started = SDL_GetPerformanceCounter();

    if (good) good = bake_make_probe_grid(job->scene, job->volumetrics.probe_spacing, &probes) && bake_probe_grid_fast(worker, &probes, &tree, &beams, bake_probe_progress);

    if (good) SDL_Log("B: wavefront volume probes took %.2f ms", bake_elapsed_ms(started));

    if (good) good = bake_write_fast_seed(job, &probes, &beams);

    bvh_free(&tree);
    free(probes.probes);
    beam_free(&beams);

    return good;
}

static int SDLCALL bake_thread_main(void *userdata) {
    BAKE_JOB *job = userdata;
    RENDERER worker = {0};

    if (!bake_worker_init(&worker)) {
        bake_set_error(job, SDL_GetError());
        SDL_SetAtomicInt(&job->done, 1);

        return 1;
    }

    SDL_Log("B: offscreen bake GPU device started");

    bool fast = !bake_cancelled() && bake_prepare_fast_components(job, &worker);

    if (!fast && !bake_cancelled()) {
        SDL_Log("B: fast probe path unavailable; falling back to original bake: %s", *SDL_GetError() ? SDL_GetError() : "unknown error");
        bake_remove_worker_files(job->worker_path);

        if (job->path && bake_copy_cache(job->path, job->worker_path)) SDL_Log("B: fallback seeded previous cache");
    }

    bake_set_phase(BAKE_PHASE_LIGHTMAP, 0u, 0u, 0u);

    bool good = !bake_cancelled() && r_rebake_current_scene(
                                         &worker,
                                         job->scene,
                                         job->visual,
                                         job->layout,
                                         job->light,
                                         &job->sky,
                                         &job->volumetrics,
                                         job->worker_path,
                                         job->scene_hash,
                                         job->layout_hash,
                                         job->volume_hash,
                                         job->beam_hash
                                     );

    if (!good) bake_set_error(job, SDL_GetError());
    bake_worker_deinit(&worker);

    if (bake_cancelled()) good = false;

    if (!good) bake_remove_worker_files(job->worker_path);

    SDL_SetAtomicInt(&job->success, good ? 1 : 0);
    SDL_SetAtomicInt(&job->done, 1);
    SDL_Log("B: offscreen bake %s after %.2f ms", good ? "finished" : "stopped", bake_elapsed_ms(job->started));

    return good ? 0 : 1;
}

static bool bake_publish_cache(const char *worker_path, const char *path) {
    char *backup = bake_path_suffix(path, ".previous");

    if (!backup) return false;
    remove(backup);

    bool had_previous = false;

    if (rename(path, backup) == 0) had_previous = true;
    else if (errno != ENOENT) {
        SDL_SetError("could not preserve previous bake cache: %s", strerror(errno));
        free(backup);

        return false;
    }

    if (rename(worker_path, path) != 0) {
        const int publish_error = errno;

        if (had_previous) rename(backup, path);
        SDL_SetError("could not publish bake cache: %s", strerror(publish_error));
        free(backup);

        return false;
    }

    if (had_previous) remove(backup);
    free(backup);

    return true;
}

static void bake_free_job(BAKE_JOB *job) {
    if (!job) return;
    free(job->worker_path);
    free(job->path);
    free(job);
}

bool bake_start(
    RENDERER *r,
    const MESH *scene,
    const GLTF_SCENE *visual,
    const LIGHTMAP *layout,
    const struct LIGHT *light,
    const SKY *sky,
    const VOLUMETRICS_LIGHTING *volumetrics,
    const char *path,
    uint64_t scene_hash,
    uint64_t layout_hash,
    uint64_t volume_hash,
    uint64_t beam_hash
) {
    if (!r || !r->device || !scene || !visual || !layout || !light || light->type != LIGHT_DIRECTIONAL || !sky || !volumetrics || !path) return false;

    if (g_bake) {
        SDL_SetError("a bake is already in progress");

        return false;
    }

    BAKE_JOB *job = calloc(1, sizeof(*job));

    if (!job) return false;
    job->renderer = r;
    job->scene = scene;
    job->visual = visual;
    job->layout = layout;
    job->light = light;
    job->sky = *sky;
    job->volumetrics = *volumetrics;
    job->scene_hash = scene_hash;
    job->layout_hash = layout_hash;
    job->volume_hash = volume_hash;
    job->beam_hash = beam_hash;
    job->started = SDL_GetPerformanceCounter();
    job->path = bake_path_suffix(path, "");
    job->worker_path = bake_path_suffix(path, ".worker");
    job->previous_cache_bytes = bake_file_size(path);

    if (!job->path || !job->worker_path) {
        bake_free_job(job);
        SDL_SetError("could not initialize offscreen bake job");

        return false;
    }

    bake_remove_worker_files(job->worker_path);

    g_bake = job;

    SDL_SetAtomicInt(&job->phase_started_ms, (int)(Uint32)SDL_GetTicks());
    job->thread = SDL_CreateThread(bake_thread_main, "pony-bake", job);

    if (!job->thread) {
        g_bake = NULL;

        bake_remove_worker_files(job->worker_path);
        bake_free_job(job);

        return false;
    }

    r->bake_stage = "offscreen GPU bake";

    if (r->window) SDL_SetWindowTitle(r->window, "BAKE | 0.0s");
    SDL_Log(
        "B: full-speed offscreen rebake started; render device remains "
        "independent"
    );
    return true;
}

bool bake_active(RENDERER *r) {
    return g_bake && (!r || g_bake->renderer == r);
}

void bake_update_title(RENDERER *r) {
    if (!r || !r->window) return;

    const Uint64 now = SDL_GetTicks();

    if (now - g_title_tick < 100u) return;

    g_title_tick = now;
    const double frame_ms = r->frame_time_ms;
    const double fps = frame_ms > 0.001 ? 1000.0 / frame_ms : 0.0;
    char title[256];

    BAKE_JOB *job = g_bake;

    if (job && job->renderer == r) {
        const int phase = SDL_GetAtomicInt(&job->phase);
        const Uint32 done = (Uint32)SDL_GetAtomicInt(&job->phase_done);
        const Uint32 total = (Uint32)SDL_GetAtomicInt(&job->phase_total);
        const Uint32 active = (Uint32)SDL_GetAtomicInt(&job->phase_active);
        const Uint32 phase_started = (Uint32)SDL_GetAtomicInt(&job->phase_started_ms);

        const double phase_seconds = (double)((Uint32)now - phase_started) / 1000.0;
        char *temporary = bake_path_suffix(job->worker_path, ".tmp");
        const uint64_t save_bytes = temporary ? bake_file_size(temporary) : 0u;
        free(temporary);

        if (phase == BAKE_PHASE_LIGHTMAP && save_bytes) {
            if (!job->save_seen_at) {
                job->save_seen_at = now;
                job->save_seen_bytes = save_bytes;
            }

            const double save_seconds = (double)(now - job->save_seen_at) / 1000.0;
            const double mib = (double)save_bytes / (1024.0 * 1024.0);
            const double expected = job->previous_cache_bytes ? (double)job->previous_cache_bytes / (1024.0 * 1024.0) : 0.0;

            const double rate = save_seconds > 0.05 ? ((double)(save_bytes - job->save_seen_bytes) / (1024.0 * 1024.0)) / save_seconds : 0.0;

            if (expected > 0.0) {
                const double percent = fmin(100.0, mib * 100.0 / expected);

                snprintf(title, sizeof(title), "%.1fms | %.1ffps | SAVE %.1f/%.1fMiB %.0f%% | %.0fMiB/s", frame_ms, fps, mib, expected, percent, rate);
            } else {
                snprintf(title, sizeof(title), "%.1fms | %.1ffps | SAVE %.1fMiB | %.0fMiB/s", frame_ms, fps, mib, rate);
            }
        } else if (phase == BAKE_PHASE_PROBES && total) {
            const double percent = 100.0 * (double)done / (double)total;
            const double eta = done && done < total ? phase_seconds * (double)(total - done) / (double)done : 0.0;

            if (eta > 0.0)
                snprintf(
                    title,
                    sizeof(title),
                    "%.1fms | %.1ffps | PROBE %u/%u %.0f%% | A%u | %.1fs ETA "
                    "%.1fs | F%u D%u",
                    frame_ms,
                    fps,
                    done,
                    total,
                    percent,
                    active,
                    phase_seconds,
                    eta,
                    r->show_volume ? 1u : 0u,
                    r->debug_view
                );
            else
                snprintf(
                    title,
                    sizeof(title),
                    "%.1fms | %.1ffps | PROBE %u/%u %.0f%% | A%u | %.1fs | F%u D%u",
                    frame_ms,
                    fps,
                    done,
                    total,
                    percent,
                    active,
                    phase_seconds,
                    r->show_volume ? 1u : 0u,
                    r->debug_view
                );
        } else {
            snprintf(title, sizeof(title), "%.1fms | %.1ffps | %s %.1fs | F%u D%u", frame_ms, fps, bake_phase_name(phase), phase_seconds, r->show_volume ? 1u : 0u, r->debug_view);
        }
    } else {
        Uint32 min_samples = 0u, max_samples = 0u, measured = 0u;
        double avg_samples = 0.0;

        if (r->has_bake && bake_probe_stats(r, &min_samples, &avg_samples, &max_samples, &measured)) {
            snprintf(
                title,
                sizeof(title),
                "%.1fms | %.1ffps | READY | LM %ux%u | P %u/%.0f/%u x%u | F%u D%u",
                frame_ms,
                fps,
                r->lightmap_width,
                r->lightmap_height,
                min_samples,
                avg_samples,
                max_samples,
                measured,
                r->show_volume ? 1u : 0u,
                r->debug_view
            );
        } else {
            snprintf(
                title,
                sizeof(title),
                "%.1fms | %.1ffps | %s | LM %ux%u | F%u D%u",
                frame_ms,
                fps,
                r->has_bake ? "READY" : "UNBAKED",
                r->lightmap_width,
                r->lightmap_height,
                r->show_volume ? 1u : 0u,
                r->debug_view
            );
        }
    }

    SDL_SetWindowTitle(r->window, title);
}

void bake_update(RENDERER *r) {
    BAKE_JOB *job = g_bake;

    if (!job || job->renderer != r || !SDL_GetAtomicInt(&job->done)) return;
    SDL_WaitThread(job->thread, NULL);
    job->thread = NULL;

    bool good = SDL_GetAtomicInt(&job->success) != 0 && SDL_GetAtomicInt(&job->cancel) == 0;

    if (good) {
        good = r_load_cached_lightmap(r, job->worker_path, job->scene_hash, job->layout_hash, job->volume_hash, job->beam_hash, job->layout);

        if (!good) bake_set_error(job, SDL_GetError());
    }

    if (good && !bake_publish_cache(job->worker_path, job->path)) SDL_Log("B: new lighting is active but cache publish failed: %s", SDL_GetError());

    if (good) {
        r->bake_stage = NULL;

        Uint32 min_samples = 0u, max_samples = 0u, measured = 0u;
        double avg_samples = 0.0;

        if (bake_probe_stats(r, &min_samples, &avg_samples, &max_samples, &measured))
            SDL_Log("B: probe samples %u/%.1f/%u across %u valid probes", min_samples, avg_samples, max_samples, measured);
        SDL_Log("B: bake ready on render device after %.2f ms", bake_elapsed_ms(job->started));
    } else {
        bake_remove_worker_files(job->worker_path);
        r->bake_stage = NULL;
        SDL_Log("B: offscreen bake failed; previous lighting retained: %s", job->error[0] ? job->error : "unknown bake error");
    }

    g_bake = NULL;

    bake_free_job(job);
}

void bake_cancel(RENDERER *r) {
    BAKE_JOB *job = g_bake;

    if (!job || (r && job->renderer != r)) return;
    SDL_SetAtomicInt(&job->cancel, 1);

    if (job->thread) {
        SDL_Log("B: waiting for offscreen GPU bake to stop");
        SDL_WaitThread(job->thread, NULL);
        job->thread = NULL;
    }

    bake_remove_worker_files(job->worker_path);

    g_bake = NULL;

    bake_free_job(job);
    SDL_Log("B: offscreen bake cancelled");
}
