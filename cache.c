#include "game.h"

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DM_CACHE_MAGIC 0x4b424d44u
#define DM_CACHE_VERSION 8u
#define DM_CACHE_MAX_DIMENSION 16384u
#define DM_CACHE_BEAM_WIDTH 64u
#define DM_CACHE_BEAM_HEIGHT 64u
#define DM_CACHE_MIN_BEAM_DEPTH 16u
#define DM_CACHE_MAX_BEAM_DEPTH 128u

typedef struct CACHE_HEADER {
    uint32_t magic;
    uint32_t version;
    uint64_t scene_hash;
    uint64_t layout_hash;
    uint32_t width;
    uint32_t height;
    uint64_t bytes;
    uint64_t payload_hash;
    uint32_t object_dims[3];
    uint32_t volume_dims[3];
    float object_origin[3];
    float volume_origin[3];
    float object_spacing;
    float volume_spacing;
    float beam_origin[3];
    float beam_step[3];
    uint32_t beam_dims[3];
    uint32_t beam_count;
    uint64_t volume_hash;
    uint64_t beam_hash;
} CACHE_HEADER;

uint64_t hash_bytes(uint64_t seed, const void *bytes, size_t size) {

    const unsigned char *p = bytes;
    uint64_t value = seed ? seed : UINT64_C(14695981039346656037);

    for (size_t i = 0; i < size; ++i) {
        value ^= p[i];
        value *= UINT64_C(1099511628211);
    }

    return value;
}

static bool valid_dimensions(uint32_t width, uint32_t height, uint64_t *bytes) {

    if (!width || !height || width > DM_CACHE_MAX_DIMENSION || height > DM_CACHE_MAX_DIMENSION) return false;

    *bytes = (uint64_t)width * height * 8u;
    return *bytes <= SIZE_MAX;
}

void cache_free(CACHED_LIGHTMAP *data) {

    if (!data) return;

    free(data->pixels);
    free(data->object_probes.probes);
    free(data->volume_probes.probes);
    beam_free(&data->beams);
    memset(data, 0, sizeof(*data));
}

static uint64_t grid_count(const PROBE_GRID *grid) {

    if (!grid->count_x || !grid->count_y || !grid->count_z || grid->count_x > 16384u || grid->count_y > 16384u || grid->count_z > 16384u || !grid->probes ||
        !(grid->spacing > 0.0f))
        return 0;

    const uint64_t count = (uint64_t)grid->count_x * grid->count_y * grid->count_z;

    return count <= 16384u ? count : 0;
}

static void unpack_grid(PROBE_GRID *grid, const uint32_t dims[3], const float origin[3], float spacing) {

    grid->origin = (VEC3){origin[0], origin[1], origin[2]};
    grid->spacing = spacing;
    grid->count_x = dims[0];
    grid->count_y = dims[1];
    grid->count_z = dims[2];
}

bool cache_read_partial(const char *path, uint64_t scene_hash, CACHED_LIGHTMAP *out) {

    if (!path || !out) return false;
    memset(out, 0, sizeof(*out));

    FILE *file = fopen(path, "rb");

    if (!file) return false;

    CACHE_HEADER header = {0};
    uint64_t expected = 0;
    bool good = fread(&header, sizeof(header), 1, file) == 1 && header.magic == DM_CACHE_MAGIC && header.version == DM_CACHE_VERSION && header.scene_hash == scene_hash &&
                valid_dimensions(header.width, header.height, &expected) && expected == header.bytes;

    if (good) {

        unpack_grid(&out->object_probes, header.object_dims, header.object_origin, header.object_spacing);
        unpack_grid(&out->volume_probes, header.volume_dims, header.volume_origin, header.volume_spacing);

        good = header.object_dims[0] <= 16384u && header.object_dims[1] <= 16384u && header.object_dims[2] <= 16384u && header.volume_dims[0] && header.volume_dims[0] <= 16384u &&
               header.volume_dims[1] && header.volume_dims[1] <= 16384u && header.volume_dims[2] && header.volume_dims[2] <= 16384u;

        const uint64_t object_count = good ? (uint64_t)header.object_dims[0] * header.object_dims[1] * header.object_dims[2] : 0;

        const uint64_t volume_count = good ? (uint64_t)header.volume_dims[0] * header.volume_dims[1] * header.volume_dims[2] : 0;

        const uint64_t beam_capacity = good ? (uint64_t)header.beam_dims[0] * header.beam_dims[1] * header.beam_dims[2] : 0;

        good = good && object_count <= 16384u && volume_count <= 16384u &&
               (!object_count || (header.object_spacing > 0.0f && isfinite(header.object_spacing) && isfinite(header.object_origin[0]) && isfinite(header.object_origin[1]) &&
                                  isfinite(header.object_origin[2]))) &&
               header.volume_spacing > 0.0f && isfinite(header.volume_spacing) && isfinite(header.volume_origin[0]) && isfinite(header.volume_origin[1]) &&
               isfinite(header.volume_origin[2]) && header.beam_dims[0] == DM_CACHE_BEAM_WIDTH && header.beam_dims[1] == DM_CACHE_BEAM_HEIGHT &&
               header.beam_dims[2] >= DM_CACHE_MIN_BEAM_DEPTH && header.beam_dims[2] <= DM_CACHE_MAX_BEAM_DEPTH && header.beam_count <= beam_capacity;

        for (uint32_t i = 0; i < 3u; ++i) {
            good = good && isfinite(header.beam_origin[i]) && isfinite(header.beam_step[i]) && header.beam_step[i] > 0.0f;
        }

        if (good) {
            if (object_count) out->object_probes.probes = malloc((size_t)object_count * sizeof(PROBE));
            out->volume_probes.probes = malloc((size_t)volume_count * sizeof(PROBE));

            good = (!object_count || out->object_probes.probes) && out->volume_probes.probes;

            out->beams.origin = v3(header.beam_origin[0], header.beam_origin[1], header.beam_origin[2]);
            out->beams.step = v3(header.beam_step[0], header.beam_step[1], header.beam_step[2]);
            out->beams.width = header.beam_dims[0];
            out->beams.height = header.beam_dims[1];
            out->beams.depth = header.beam_dims[2];
            out->beams.count = header.beam_count;

            if (good && header.beam_count) {
                out->beams.cells = malloc((size_t)header.beam_count * sizeof(BEAM_CELL));

                good = out->beams.cells != NULL;
            }

            if (good) {
                const size_t depth_count = (size_t)header.beam_dims[0] * header.beam_dims[1];

                out->beams.shadow_depth = malloc(depth_count * sizeof(float));

                good = out->beams.shadow_depth != NULL;
            }
        }
    }

    if (good) {

        out->pixels = malloc((size_t)expected);

        const uint64_t object_count = (uint64_t)header.object_dims[0] * header.object_dims[1] * header.object_dims[2];

        const uint64_t volume_count = (uint64_t)header.volume_dims[0] * header.volume_dims[1] * header.volume_dims[2];

        const size_t object_bytes = (size_t)object_count * sizeof(PROBE);
        const size_t volume_bytes = (size_t)volume_count * sizeof(PROBE);
        const size_t beam_bytes = (size_t)header.beam_count * sizeof(BEAM_CELL);
        const size_t depth_count = (size_t)header.beam_dims[0] * header.beam_dims[1];

        const size_t depth_bytes = depth_count * sizeof(float);

        good = out->pixels && fread(out->pixels, (size_t)expected, 1, file) == 1 && (!object_bytes || fread(out->object_probes.probes, object_bytes, 1, file) == 1) &&
               fread(out->volume_probes.probes, volume_bytes, 1, file) == 1 && (!beam_bytes || fread(out->beams.cells, beam_bytes, 1, file) == 1) &&
               fread(out->beams.shadow_depth, depth_bytes, 1, file) == 1 && fgetc(file) == EOF && !ferror(file);

        if (good) {
            uint64_t hash = hash_bytes(0, out->pixels, (size_t)expected);

            if (object_bytes) hash = hash_bytes(hash, out->object_probes.probes, object_bytes);

            hash = hash_bytes(hash, out->volume_probes.probes, volume_bytes);

            if (beam_bytes) hash = hash_bytes(hash, out->beams.cells, beam_bytes);

            hash = hash_bytes(hash, out->beams.shadow_depth, depth_bytes);
            good = hash == header.payload_hash;

            if (good) {
                for (size_t i = 0; i < depth_count; ++i) {
                    const float depth = out->beams.shadow_depth[i];

                    if (!(isfinite(depth) || depth == -INFINITY)) good = false;
                }

                float *expanded = beam_expand(&out->beams);
                good = good && expanded != NULL;
                free(expanded);
            }
        }
    }

    if (fclose(file) != 0) good = false;

    if (!good) {
        cache_free(out);

        return false;
    }

    out->width = header.width;
    out->height = header.height;
    out->layout_hash = header.layout_hash;
    out->volume_hash = header.volume_hash;
    out->beam_hash = header.beam_hash;

    return true;
}

bool cache_read(const char *path, uint64_t scene_hash, uint64_t layout_hash, uint64_t volume_hash, uint64_t beam_hash, CACHED_LIGHTMAP *out) {
    if (!cache_read_partial(path, scene_hash, out)) return false;

    if (out->layout_hash == layout_hash && out->volume_hash == volume_hash && out->beam_hash == beam_hash) return true;
    cache_free(out);

    return false;
}

bool cache_write(const char *path, uint64_t scene_hash, uint64_t layout_hash, uint64_t volume_hash, uint64_t beam_hash, const CACHED_LIGHTMAP *data) {

    uint64_t bytes = 0;

    if (!path || !data || !data->pixels || !valid_dimensions(data->width, data->height, &bytes)) return false;

    const uint64_t object_count = grid_count(&data->object_probes);
    const uint64_t volume_count = grid_count(&data->volume_probes);

    if (!volume_count) return false;

    const BEAM_GRID *beams = &data->beams;
    const uint64_t beam_capacity = (uint64_t)beams->width * beams->height * beams->depth;

    if (beams->width != DM_CACHE_BEAM_WIDTH || beams->height != DM_CACHE_BEAM_HEIGHT || beams->depth < DM_CACHE_MIN_BEAM_DEPTH || beams->depth > DM_CACHE_MAX_BEAM_DEPTH ||
        beams->count > beam_capacity || (beams->count && !beams->cells) || !beams->shadow_depth)
        return false;

    const size_t depth_count = (size_t)beams->width * beams->height;
    const size_t depth_bytes = depth_count * sizeof(float);

    for (size_t i = 0; i < depth_count; ++i) {
        const float depth = beams->shadow_depth[i];

        if (!(isfinite(depth) || depth == -INFINITY)) return false;
    }

    float *expanded = beam_expand(beams);

    if (!expanded) return false;
    free(expanded);

    const size_t object_bytes = (size_t)object_count * sizeof(PROBE);
    const size_t volume_bytes = (size_t)volume_count * sizeof(PROBE);
    const size_t beam_bytes = (size_t)beams->count * sizeof(BEAM_CELL);
    const Uint64 hash_started = SDL_GetPerformanceCounter();
    uint64_t payload_hash = hash_bytes(0, data->pixels, (size_t)bytes);

    if (object_bytes) payload_hash = hash_bytes(payload_hash, data->object_probes.probes, object_bytes);

    payload_hash = hash_bytes(payload_hash, data->volume_probes.probes, volume_bytes);

    if (beam_bytes) payload_hash = hash_bytes(payload_hash, beams->cells, beam_bytes);

    payload_hash = hash_bytes(payload_hash, beams->shadow_depth, depth_bytes);

    SDL_Log("B: cache hashing took %.2f ms", (double)(SDL_GetPerformanceCounter() - hash_started) * 1000.0 / (double)SDL_GetPerformanceFrequency());

    const size_t path_length = strlen(path);

    if (path_length > SIZE_MAX - 5) return false;

    char *temporary = malloc(path_length + 5);

    if (!temporary) return false;
    memcpy(temporary, path, path_length);
    memcpy(temporary + path_length, ".tmp", 5);

    FILE *file = fopen(temporary, "wb");

    if (!file) {
        SDL_SetError("could not write %s: %s", temporary, strerror(errno));
        free(temporary);

        return false;
    }

    const CACHE_HEADER header = {
        .magic = DM_CACHE_MAGIC,
        .version = DM_CACHE_VERSION,
        .scene_hash = scene_hash,
        .layout_hash = layout_hash,
        .volume_hash = volume_hash,
        .beam_hash = beam_hash,
        .width = data->width,
        .height = data->height,
        .bytes = bytes,
        .payload_hash = payload_hash,
        .object_dims = {data->object_probes.count_x, data->object_probes.count_y, data->object_probes.count_z},
        .volume_dims = {data->volume_probes.count_x, data->volume_probes.count_y, data->volume_probes.count_z},
        .object_origin = {data->object_probes.origin.x, data->object_probes.origin.y, data->object_probes.origin.z},
        .volume_origin = {data->volume_probes.origin.x, data->volume_probes.origin.y, data->volume_probes.origin.z},
        .object_spacing = data->object_probes.spacing,
        .volume_spacing = data->volume_probes.spacing,
        .beam_origin = {beams->origin.x, beams->origin.y, beams->origin.z},
        .beam_step = {beams->step.x, beams->step.y, beams->step.z},
        .beam_dims = {beams->width, beams->height, beams->depth},
        .beam_count = beams->count
    };

    const Uint64 write_started = SDL_GetPerformanceCounter();
    bool good = fwrite(&header, sizeof(header), 1, file) == 1 && fwrite(data->pixels, (size_t)bytes, 1, file) == 1 &&
                (!object_bytes || fwrite(data->object_probes.probes, object_bytes, 1, file) == 1) && fwrite(data->volume_probes.probes, volume_bytes, 1, file) == 1 &&
                (!beam_bytes || fwrite(beams->cells, beam_bytes, 1, file) == 1) && fwrite(beams->shadow_depth, depth_bytes, 1, file) == 1 && fflush(file) == 0;

    if (fclose(file) != 0) good = false;

    if (good) SDL_Log("B: cache file write took %.2f ms", (double)(SDL_GetPerformanceCounter() - write_started) * 1000.0 / (double)SDL_GetPerformanceFrequency());

    if (good && rename(temporary, path) != 0) {
        SDL_SetError("could not save %s: %s", path, strerror(errno));

        good = false;
    }

    if (!good) {
        if (!*SDL_GetError()) SDL_SetError("could not write bake data to %s", temporary);
        remove(temporary);
    }

    free(temporary);

    return good;
}
