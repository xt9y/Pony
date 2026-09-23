#ifndef DUSTMITE_CACHE_H
#define DUSTMITE_CACHE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include "dustmite.h"

typedef struct dm_cached_lightmap {
    uint32_t width;
    uint32_t height;
    unsigned char *pixels; /* tightly packed RGBA16F, width * height * 8 bytes */
    dm_probe_grid object_probes;
    dm_probe_grid volume_probes;
    dm_beam_grid beams;
} dm_cached_lightmap;

uint64_t dm_hash_bytes(uint64_t seed, const void *bytes, size_t size);
bool dm_cache_read(const char *path, uint64_t scene_hash, uint64_t layout_hash,
                   dm_cached_lightmap *out);
bool dm_cache_write(const char *path, uint64_t scene_hash, uint64_t layout_hash,
                    const dm_cached_lightmap *data);
void dm_cache_free(dm_cached_lightmap *data);

#endif
