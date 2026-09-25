#include "dustmite.h"

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct bake_job {
    renderer *renderer;
    const mesh *scene;
    const gltf_scene *visual;
    const lightmap *layout;
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
    SDL_AtomicInt cache_seeded;
    Uint64 started;
    uint64_t previous_cache_bytes;
    Uint64 save_seen_at;
    uint64_t save_seen_bytes;
    char error[256];
} bake_job;

static bake_job *g_bake;
static Uint64 g_title_tick;

static double bake_elapsed_ms(Uint64 started) {
    return (double)(SDL_GetPerformanceCounter() - started) * 1000.0 /
           (double)SDL_GetPerformanceFrequency();
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

static bool bake_probe_stats(const renderer *r, Uint32 *minimum,
                             double *average, Uint32 *maximum,
                             Uint32 *valid_count) {
    if (!r || !r->volume_probes.probes) return false;

    const uint64_t probe_count = (uint64_t)r->volume_probes.count_x *
                                 r->volume_probes.count_y *
                                 r->volume_probes.count_z;
    if (!probe_count) return false;

    Uint32 min_samples = UINT32_MAX;
    Uint32 max_samples = 0u;
    uint64_t sample_sum = 0u;
    Uint32 measured = 0u;

    for (uint64_t i = 0u; i < probe_count; ++i) {
        const dm_probe *probe = &r->volume_probes.probes[i];
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

static void bake_seed_worker_cache(bake_job *job) {
    if (!job || SDL_GetAtomicInt(&job->cancel)) return;

    FILE *source = fopen(job->path, "rb");
    if (!source) return;

    FILE *destination = fopen(job->worker_path, "wb");
    if (!destination) {
        fclose(source);
        SDL_Log("B: cache reuse seed unavailable: %s", strerror(errno));
        return;
    }

    unsigned char *buffer = malloc(1024u * 1024u);
    bool good = buffer != NULL;
    uint64_t copied = 0u;

    while (good && !SDL_GetAtomicInt(&job->cancel)) {
        const size_t count = fread(buffer, 1, 1024u * 1024u, source);
        if (count) {
            if (fwrite(buffer, 1, count, destination) != count) {
                good = false;
                break;
            }
            copied += count;
        }
        if (count < 1024u * 1024u) {
            if (ferror(source)) good = false;
            break;
        }
    }

    free(buffer);
    if (fclose(source) != 0) good = false;
    if (fclose(destination) != 0) good = false;
    if (SDL_GetAtomicInt(&job->cancel)) good = false;

    if (!good) {
        remove(job->worker_path);
        SDL_Log("B: cache reuse seed failed; continuing with full rebake");
        return;
    }

    SDL_SetAtomicInt(&job->cache_seeded, 1);
    SDL_Log("B: seeded %.2f MiB previous cache for component reuse",
            (double)copied / (1024.0 * 1024.0));
}

static void bake_set_error(bake_job *job, const char *message) {
    if (!job) return;
    const char *text = message && *message ? message : "unknown bake error";
    snprintf(job->error, sizeof(job->error), "%s", text);
}

static int SDLCALL bake_thread_main(void *userdata) {
    bake_job *job = userdata;
    renderer worker = {0};

    bake_seed_worker_cache(job);
    if (SDL_GetAtomicInt(&job->cancel)) {
        SDL_SetAtomicInt(&job->done, 1);
        return 1;
    }

    if (!bake_worker_init(&worker)) {
        bake_set_error(job, SDL_GetError());
        SDL_SetAtomicInt(&job->done, 1);
        return 1;
    }

    SDL_Log("B: offscreen bake GPU device started");
    bool good = r_rebake_current_scene(&worker, job->scene, job->visual,
                                       job->layout, job->worker_path,
                                       job->scene_hash, job->layout_hash,
                                       job->volume_hash, job->beam_hash);
    if (!good) bake_set_error(job, SDL_GetError());

    bake_worker_deinit(&worker);

    if (SDL_GetAtomicInt(&job->cancel)) good = false;
    if (!good) bake_remove_worker_files(job->worker_path);

    SDL_SetAtomicInt(&job->success, good ? 1 : 0);
    SDL_SetAtomicInt(&job->done, 1);
    SDL_Log("B: offscreen bake %s after %.2f ms",
            good ? "finished" : "stopped", bake_elapsed_ms(job->started));
    return good ? 0 : 1;
}

static bool bake_publish_cache(const char *worker_path, const char *path) {
    char *backup = bake_path_suffix(path, ".previous");
    if (!backup) return false;

    remove(backup);
    bool had_previous = false;
    if (rename(path, backup) == 0) {
        had_previous = true;
    } else if (errno != ENOENT) {
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

static void bake_free_job(bake_job *job) {
    if (!job) return;
    free(job->worker_path);
    free(job->path);
    free(job);
}

bool bake_start(renderer *r, const mesh *scene, const gltf_scene *visual,
                const lightmap *layout, const char *path,
                uint64_t scene_hash, uint64_t layout_hash,
                uint64_t volume_hash, uint64_t beam_hash) {
    if (!r || !r->device || !scene || !visual || !layout || !path) return false;
    if (g_bake) {
        SDL_SetError("a bake is already in progress");
        return false;
    }

    bake_job *job = calloc(1, sizeof(*job));
    if (!job) return false;
    job->renderer = r;
    job->scene = scene;
    job->visual = visual;
    job->layout = layout;
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
    job->thread = SDL_CreateThread(bake_thread_main, "dustmite-bake", job);
    if (!job->thread) {
        bake_remove_worker_files(job->worker_path);
        bake_free_job(job);
        return false;
    }

    g_bake = job;
    r->bake_stage = "offscreen GPU bake";
    if (r->window) SDL_SetWindowTitle(r->window, "BAKE | 0.0s");
    SDL_Log("B: full-speed offscreen rebake started; render device remains independent");
    return true;
}

bool bake_active(renderer *r) {
    return g_bake && (!r || g_bake->renderer == r);
}

void bake_update_title(renderer *r) {
    if (!r || !r->window) return;

    const Uint64 now = SDL_GetTicks();
    if (now - g_title_tick < 100u) return;
    g_title_tick = now;

    const double frame_ms = r->frame_time_ms;
    const double fps = frame_ms > 0.001 ? 1000.0 / frame_ms : 0.0;
    char title[256];

    bake_job *job = g_bake;
    if (job && job->renderer == r) {
        char *temporary = bake_path_suffix(job->worker_path, ".tmp");
        const uint64_t save_bytes = temporary ? bake_file_size(temporary) : 0u;
        free(temporary);

        if (save_bytes) {
            if (!job->save_seen_at) {
                job->save_seen_at = now;
                job->save_seen_bytes = save_bytes;
            }
            const double save_seconds = (double)(now - job->save_seen_at) / 1000.0;
            const double mib = (double)save_bytes / (1024.0 * 1024.0);
            const double rate = save_seconds > 0.05
                ? ((double)(save_bytes - job->save_seen_bytes) / (1024.0 * 1024.0)) /
                      save_seconds
                : 0.0;

            if (job->previous_cache_bytes) {
                const double expected = (double)job->previous_cache_bytes /
                                        (1024.0 * 1024.0);
                const double percent = fmin(100.0, 100.0 * (double)save_bytes /
                                                     (double)job->previous_cache_bytes);
                snprintf(title, sizeof(title),
                         "%.1fms | %.1ffps | SAVE %.1f/%.1fMiB ~%.0f%% | %.0fMiB/s",
                         frame_ms, fps, mib, expected, percent, rate);
            } else {
                snprintf(title, sizeof(title),
                         "%.1fms | %.1ffps | SAVE %.1fMiB | %.0fMiB/s",
                         frame_ms, fps, mib, rate);
            }
        } else {
            const double seconds = bake_elapsed_ms(job->started) / 1000.0;
            if (SDL_GetAtomicInt(&job->cache_seeded)) {
                snprintf(title, sizeof(title),
                         "%.1fms | %.1ffps | BAKE %.1fs | CACHE %.1fMiB | F%u D%u",
                         frame_ms, fps, seconds,
                         (double)job->previous_cache_bytes / (1024.0 * 1024.0),
                         r->show_volume ? 1u : 0u, r->debug_view);
            } else {
                snprintf(title, sizeof(title),
                         "%.1fms | %.1ffps | BAKE %.1fs | F%u D%u",
                         frame_ms, fps, seconds,
                         r->show_volume ? 1u : 0u, r->debug_view);
            }
        }
    } else {
        Uint32 min_samples = 0u, max_samples = 0u, measured = 0u;
        double avg_samples = 0.0;
        const bool have_probe_stats = r->has_bake &&
            bake_probe_stats(r, &min_samples, &avg_samples, &max_samples, &measured);
        if (have_probe_stats) {
            snprintf(title, sizeof(title),
                     "%.1fms | %.1ffps | READY | LM %ux%u | P %u/%.0f/%u x%u | F%u D%u",
                     frame_ms, fps, r->lightmap_width, r->lightmap_height,
                     min_samples, avg_samples, max_samples, measured,
                     r->show_volume ? 1u : 0u, r->debug_view);
        } else {
            snprintf(title, sizeof(title),
                     "%.1fms | %.1ffps | %s | LM %ux%u | F%u D%u",
                     frame_ms, fps, r->has_bake ? "READY" : "UNBAKED",
                     r->lightmap_width, r->lightmap_height,
                     r->show_volume ? 1u : 0u, r->debug_view);
        }
    }

    SDL_SetWindowTitle(r->window, title);
}

void bake_update(renderer *r) {
    bake_job *job = g_bake;
    if (!job || job->renderer != r || !SDL_GetAtomicInt(&job->done)) return;

    SDL_WaitThread(job->thread, NULL);
    job->thread = NULL;

    bool good = SDL_GetAtomicInt(&job->success) != 0 &&
                SDL_GetAtomicInt(&job->cancel) == 0;
    if (good) {
        good = r_load_cached_lightmap(r, job->worker_path,
                                      job->scene_hash, job->layout_hash,
                                      job->volume_hash, job->beam_hash,
                                      job->layout);
        if (!good) bake_set_error(job, SDL_GetError());
    }

    if (good && !bake_publish_cache(job->worker_path, job->path)) {
        SDL_Log("B: new lighting is active but cache publish failed: %s", SDL_GetError());
    }

    if (good) {
        r->bake_stage = NULL;
        if (r->window) SDL_SetWindowTitle(r->window, "READY");
        Uint32 min_samples = 0u, max_samples = 0u, measured = 0u;
        double avg_samples = 0.0;
        if (bake_probe_stats(r, &min_samples, &avg_samples, &max_samples, &measured)) {
            SDL_Log("B: probe samples %u/%.1f/%u across %u valid probes",
                    min_samples, avg_samples, max_samples, measured);
        }
        SDL_Log("B: bake ready on render device after %.2f ms",
                bake_elapsed_ms(job->started));
    } else {
        bake_remove_worker_files(job->worker_path);
        r->bake_stage = NULL;
        if (r->window) SDL_SetWindowTitle(r->window, "BAKE FAIL");
        SDL_Log("B: offscreen bake failed; previous lighting retained: %s",
                job->error[0] ? job->error : "unknown bake error");
    }

    g_bake = NULL;
    bake_free_job(job);
}

void bake_cancel(renderer *r) {
    bake_job *job = g_bake;
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
