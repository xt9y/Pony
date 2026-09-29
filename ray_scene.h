#ifndef RAY_SCENE_H
#define RAY_SCENE_H

#include <stdbool.h>
#include <stdint.h>

#include "gpu.h"

typedef enum TRACE_HIT_TYPE {
    TRACE_INACTIVE = 0,
    TRACE_MISS = 1,
    TRACE_SCREEN = 2,
    TRACE_SDF = 3
} TRACE_HIT_TYPE;

typedef struct TRACE_HIT {
    uint32_t type;
    uint32_t object_id;
    uint32_t hit_x;
    uint32_t hit_y;
    float distance;
    float confidence;
    uint32_t padding[2];
} TRACE_HIT;

typedef struct TRACE_RAY {
    float origin_tmin[4];
    float direction_tmax[4];
    uint32_t source_pixel[2];
    uint32_t padding[2];
} TRACE_RAY;

typedef struct GPU_SDF_MODEL {
    MAT4 world_to_local;
    float bounds_min[4];
    float bounds_max[4];
    uint32_t voxel_offset;
    uint32_t resolution;
    uint32_t object_id;
    uint32_t state;
} GPU_SDF_MODEL;

typedef struct GPU_LIGHT {
    float position_range[4];
    float direction_type[4];
    float color_intensity[4];
    float spot_angles[4];
} GPU_LIGHT;

typedef struct SURFACE_CACHE_ENTRY {
    float position[4];
    float radiance[4];
    uint32_t object_id;
    uint32_t padding[3];
} SURFACE_CACHE_ENTRY;

#endif
