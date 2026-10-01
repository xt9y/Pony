#!/usr/bin/env python3
from pathlib import Path
import re
import sys


def replace_once(text, old, new, label):
    count = text.count(old)
    if count != 1:
        raise SystemExit(f'{label}: expected 1 literal match, got {count}')
    return text.replace(old, new, 1)


def regex_once(text, pattern, replacement, label):
    out, count = re.subn(pattern, replacement, text, count=1, flags=re.S)
    if count != 1:
        raise SystemExit(f'{label}: expected 1 regex match, got {count}')
    return out


def task1():
    p = Path('game.h')
    s = p.read_text()
    marker = '''typedef struct GPU_GLOBAL_SDF_CLIPMAP {
    float center_extent[4];
    float voxel_brick[4];
    uint32_t grid[4];
    uint32_t data[4];
} GPU_GLOBAL_SDF_CLIPMAP;
'''
    addition = marker + '''
typedef struct GLOBAL_SDF_DATA {
    GPU_GLOBAL_SDF_CLIPMAP *cpu_clipmaps;
    uint32_t *cpu_page_table;
    float *cpu_bricks;
    uint32_t *cpu_surface_ids;

    NriBuffer *clipmaps;
    NriBuffer *page_table;
    NriBuffer *bricks;
    NriBuffer *surface_ids;

    NriDescriptor *clipmaps_srv;
    NriDescriptor *page_table_srv;
    NriDescriptor *bricks_srv;
    NriDescriptor *surface_ids_srv;

    NriAccessStage state;
    uint32_t clip_count;
    uint32_t page_table_count;
    uint32_t physical_brick_count;
    uint32_t voxel_count;
    float coarsest_voxel_size;
    bool valid;
} GLOBAL_SDF_DATA;

bool sdf_build_global_clipmaps(
    const SCENE *scene,
    const RADIANCE_SCENE_DATA *radiance_scene,
    const GPU_OBJECT *objects,
    uint32_t object_count,
    GLOBAL_SDF_DATA *out
);
void sdf_free_global_clipmaps(GLOBAL_SDF_DATA *data);
'''
    s = replace_once(s, marker, addition, 'task1 global owner')
    s = replace_once(
        s,
        '    RADIANCE_DYNAMIC_GRID dynamic_grid;\n    RADIANCE_SCENE_FALLBACKS radiance_fallbacks;\n',
        '    RADIANCE_DYNAMIC_GRID dynamic_grid;\n    GLOBAL_SDF_DATA global_sdf;\n    RADIANCE_SCENE_FALLBACKS radiance_fallbacks;\n',
        'task1 renderer owner'
    )
    s = replace_once(
        s,
        '    NriPipeline *wavefront_dynamic_pipeline;\n    NriPipeline *wavefront_local_pipeline;\n',
        '    NriPipeline *wavefront_dynamic_pipeline;\n    NriPipeline *wavefront_global_pipeline;\n    NriPipeline *wavefront_local_pipeline;\n',
        'task1 pipeline handle'
    )
    p.write_text(s)


def task2():
    p = Path('sdf.c')
    s = p.read_text()
    if 'bool sdf_build_global_clipmaps(' in s:
        raise SystemExit('task2 already applied')
    addition = r'''

static VEC3 sdf_transform_point(MAT4 matrix, VEC3 point) {
    return (VEC3){
        point.x * matrix.m[0] + point.y * matrix.m[4] + point.z * matrix.m[8] + matrix.m[12],
        point.x * matrix.m[1] + point.y * matrix.m[5] + point.z * matrix.m[9] + matrix.m[13],
        point.x * matrix.m[2] + point.y * matrix.m[6] + point.z * matrix.m[10] + matrix.m[14]
    };
}

static void global_bit_set(uint8_t *bits, uint32_t index) {
    bits[index >> 3u] |= (uint8_t)(1u << (index & 7u));
}

static bool global_bit_get(const uint8_t *bits, uint32_t index) {
    return (bits[index >> 3u] & (uint8_t)(1u << (index & 7u))) != 0u;
}

static uint32_t global_grid_coord(float value, float origin, float brick_world, uint32_t dimension) {
    int64_t coordinate = (int64_t)floorf((value - origin) / brick_world);
    if (coordinate < 0) coordinate = 0;
    if ((uint64_t)coordinate >= dimension) coordinate = (int64_t)dimension - 1;
    return (uint32_t)coordinate;
}

void sdf_free_global_clipmaps(GLOBAL_SDF_DATA *data) {
    if (!data) return;
    free(data->cpu_clipmaps);
    free(data->cpu_page_table);
    free(data->cpu_bricks);
    free(data->cpu_surface_ids);
    memset(data, 0, sizeof(*data));
}

bool sdf_build_global_clipmaps(
    const SCENE *scene,
    const RADIANCE_SCENE_DATA *radiance_scene,
    const GPU_OBJECT *objects,
    uint32_t object_count,
    GLOBAL_SDF_DATA *out
) {
    static const uint32_t grid_dimensions[3] = {32u, 16u, 8u};
    enum { LEVEL_COUNT = 3, BRICK_RESOLUTION = 4, BRICK_VOXELS = 64 };
    if (!scene || !radiance_scene || !objects || !out || !radiance_scene->cpu_triangles) return false;
    (void)scene;
    memset(out, 0, sizeof(*out));

    uint32_t static_count = 0u;
    for (uint32_t i = 0u; i < radiance_scene->triangle_count; ++i) {
        const GPU_SCENE_TRIANGLE *source = &radiance_scene->cpu_triangles[i];
        if (source->meta[0] >= object_count) return false;
        if (objects[source->meta[0]].state == (uint32_t)STATIC) ++static_count;
    }
    if (!static_count) return true;
    if ((uint64_t)static_count * 2u > SIZE_MAX / sizeof(SDF_BVH_NODE)) return false;

    SDF_BUILD build = {0};
    build.triangle_count = static_count;
    build.triangles = malloc((size_t)static_count * sizeof(*build.triangles));
    build.nodes = calloc((size_t)static_count * 2u, sizeof(*build.nodes));
    if (!build.triangles || !build.nodes) goto fail;

    VEC3 scene_min = {FLT_MAX, FLT_MAX, FLT_MAX};
    VEC3 scene_max = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
    uint32_t cursor = 0u;
    for (uint32_t i = 0u; i < radiance_scene->triangle_count; ++i) {
        const GPU_SCENE_TRIANGLE *source = &radiance_scene->cpu_triangles[i];
        const uint32_t object_index = source->meta[0];
        if (objects[object_index].state != (uint32_t)STATIC) continue;
        SDF_TRIANGLE *triangle = &build.triangles[cursor++];
        const MAT4 world = objects[object_index].world;
        triangle->a = sdf_transform_point(world, (VEC3){source->p0[0], source->p0[1], source->p0[2]});
        triangle->b = sdf_transform_point(world, (VEC3){source->p1[0], source->p1[1], source->p1[2]});
        triangle->c = sdf_transform_point(world, (VEC3){source->p2[0], source->p2[1], source->p2[2]});
        triangle->centroid = scale3(add3(add3(triangle->a, triangle->b), triangle->c), 1.0f / 3.0f);
        triangle->min = vmin3(triangle->a, vmin3(triangle->b, triangle->c));
        triangle->max = vmax3(triangle->a, vmax3(triangle->b, triangle->c));
        triangle->surface_id = i;
        scene_min = vmin3(scene_min, triangle->min);
        scene_max = vmax3(scene_max, triangle->max);
    }
    if (cursor != static_count) goto fail;
    build_node(&build, 0u, static_count);

    const VEC3 raw_size = sub3(scene_max, scene_min);
    const float raw_side = fmaxf(fmaxf(raw_size.x, raw_size.y), fmaxf(raw_size.z, 1.0e-3f));
    const float cube_side = raw_side * (32.0f / 30.0f);
    const float half_extent = cube_side * 0.5f;
    const VEC3 center = scale3(add3(scene_min, scene_max), 0.5f);
    const VEC3 cube_min = {center.x - half_extent, center.y - half_extent, center.z - half_extent};

    uint32_t page_bases[LEVEL_COUNT] = {0};
    uint32_t logical_counts[LEVEL_COUNT] = {0};
    uint32_t physical_counts[LEVEL_COUNT] = {0};
    uint8_t *requested[LEVEL_COUNT] = {0};
    uint64_t page_total = 0u;

    for (uint32_t level = 0u; level < LEVEL_COUNT; ++level) {
        const uint32_t dim = grid_dimensions[level];
        const uint64_t logical64 = (uint64_t)dim * dim * dim;
        if (!logical64 || logical64 > UINT32_MAX || page_total + logical64 > UINT32_MAX) goto fail_bits;
        page_bases[level] = (uint32_t)page_total;
        logical_counts[level] = (uint32_t)logical64;
        page_total += logical64;
        const size_t bytes = ((size_t)logical_counts[level] + 7u) / 8u;
        requested[level] = calloc(bytes, 1u);
        if (!requested[level]) goto fail_bits;

        if (level == 2u) {
            for (uint32_t i = 0u; i < logical_counts[level]; ++i) global_bit_set(requested[level], i);
        } else {
            const float voxel_size = cube_side / ((float)dim * (float)BRICK_RESOLUTION);
            const float brick_world = voxel_size * (float)BRICK_RESOLUTION;
            const float influence = 2.0f * brick_world;
            for (uint32_t i = 0u; i < build.triangle_count; ++i) {
                const SDF_TRIANGLE *triangle = &build.triangles[i];
                const uint32_t min_x = global_grid_coord(triangle->min.x - influence, cube_min.x, brick_world, dim);
                const uint32_t min_y = global_grid_coord(triangle->min.y - influence, cube_min.y, brick_world, dim);
                const uint32_t min_z = global_grid_coord(triangle->min.z - influence, cube_min.z, brick_world, dim);
                const uint32_t max_x = global_grid_coord(triangle->max.x + influence, cube_min.x, brick_world, dim);
                const uint32_t max_y = global_grid_coord(triangle->max.y + influence, cube_min.y, brick_world, dim);
                const uint32_t max_z = global_grid_coord(triangle->max.z + influence, cube_min.z, brick_world, dim);
                for (uint32_t z = min_z; z <= max_z; ++z)
                    for (uint32_t y = min_y; y <= max_y; ++y)
                        for (uint32_t x = min_x; x <= max_x; ++x)
                            global_bit_set(requested[level], x + dim * (y + dim * z));
            }
        }
        for (uint32_t i = 0u; i < logical_counts[level]; ++i)
            if (global_bit_get(requested[level], i)) ++physical_counts[level];
    }

    if (page_total != 37376u) goto fail_bits;
    const uint64_t physical_total = (uint64_t)physical_counts[0] + physical_counts[1] + physical_counts[2];
    const uint64_t voxel_total = physical_total * BRICK_VOXELS;
    if (!physical_total || physical_total > 37376u || voxel_total > 2392064u || voxel_total > UINT32_MAX ||
        voxel_total > SIZE_MAX / sizeof(float) || voxel_total > SIZE_MAX / sizeof(uint32_t))
        goto fail_bits;

    out->cpu_clipmaps = calloc(LEVEL_COUNT, sizeof(*out->cpu_clipmaps));
    out->cpu_page_table = malloc((size_t)page_total * sizeof(*out->cpu_page_table));
    out->cpu_bricks = malloc((size_t)voxel_total * sizeof(*out->cpu_bricks));
    out->cpu_surface_ids = malloc((size_t)voxel_total * sizeof(*out->cpu_surface_ids));
    if (!out->cpu_clipmaps || !out->cpu_page_table || !out->cpu_bricks || !out->cpu_surface_ids) goto fail_bits;
    for (uint32_t i = 0u; i < (uint32_t)page_total; ++i) out->cpu_page_table[i] = UINT32_MAX;

    uint32_t voxel_base = 0u;
    for (uint32_t level = 0u; level < LEVEL_COUNT; ++level) {
        const uint32_t dim = grid_dimensions[level];
        const float voxel_size = cube_side / ((float)dim * (float)BRICK_RESOLUTION);
        const float brick_world = voxel_size * (float)BRICK_RESOLUTION;
        GPU_GLOBAL_SDF_CLIPMAP *clip = &out->cpu_clipmaps[level];
        clip->center_extent[0] = center.x;
        clip->center_extent[1] = center.y;
        clip->center_extent[2] = center.z;
        clip->center_extent[3] = half_extent;
        clip->voxel_brick[0] = voxel_size;
        clip->voxel_brick[1] = brick_world;
        clip->grid[0] = dim;
        clip->grid[1] = dim;
        clip->grid[2] = dim;
        clip->grid[3] = page_bases[level];
        clip->data[0] = BRICK_RESOLUTION;
        clip->data[1] = BRICK_VOXELS;
        clip->data[2] = voxel_base;

        uint32_t physical = 0u;
        const float half_diagonal = 0.8660254037844386f * voxel_size;
        for (uint32_t logical = 0u; logical < logical_counts[level]; ++logical) {
            if (!global_bit_get(requested[level], logical)) continue;
            out->cpu_page_table[page_bases[level] + logical] = physical;
            const uint32_t bx = logical % dim;
            const uint32_t by = (logical / dim) % dim;
            const uint32_t bz = logical / (dim * dim);
            for (uint32_t vz = 0u; vz < BRICK_RESOLUTION; ++vz) {
                for (uint32_t vy = 0u; vy < BRICK_RESOLUTION; ++vy) {
                    for (uint32_t vx = 0u; vx < BRICK_RESOLUTION; ++vx) {
                        const uint32_t local = vx + BRICK_RESOLUTION * (vy + BRICK_RESOLUTION * vz);
                        const uint32_t index = voxel_base + physical * BRICK_VOXELS + local;
                        const VEC3 position = {
                            cube_min.x + (float)bx * brick_world + ((float)vx + 0.5f) * voxel_size,
                            cube_min.y + (float)by * brick_world + ((float)vy + 0.5f) * voxel_size,
                            cube_min.z + (float)bz * brick_world + ((float)vz + 0.5f) * voxel_size
                        };
                        uint32_t surface_id = UINT32_MAX;
                        const float distance_sq = nearest_distance_sq(&build, 0u, position, FLT_MAX, &surface_id);
                        if (surface_id == UINT32_MAX || surface_id >= radiance_scene->triangle_count || !isfinite(distance_sq)) goto fail_bits;
                        const float distance = sqrtf(fmaxf(distance_sq, 0.0f));
                        out->cpu_bricks[index] = fmaxf(0.0f, distance - half_diagonal);
                        out->cpu_surface_ids[index] = surface_id;
                    }
                }
            }
            ++physical;
        }
        if (physical != physical_counts[level]) goto fail_bits;
        voxel_base += physical * BRICK_VOXELS;
    }

    if (voxel_base != (uint32_t)voxel_total) goto fail_bits;
    out->clip_count = LEVEL_COUNT;
    out->page_table_count = (uint32_t)page_total;
    out->physical_brick_count = (uint32_t)physical_total;
    out->voxel_count = (uint32_t)voxel_total;
    out->coarsest_voxel_size = out->cpu_clipmaps[2].voxel_brick[0];
    out->valid = true;

    for (uint32_t level = 0u; level < LEVEL_COUNT; ++level) free(requested[level]);
    free(build.triangles);
    free(build.nodes);
    return true;

fail_bits:
    for (uint32_t level = 0u; level < LEVEL_COUNT; ++level) free(requested[level]);
fail:
    free(build.triangles);
    free(build.nodes);
    sdf_free_global_clipmaps(out);
    return false;
}
'''
    p.write_text(s + addition)


def task3():
    gp = Path('game.h')
    gs = gp.read_text()
    gs = regex_once(gs, r'\ntypedef struct RADIANCE_SCENE_FALLBACKS \{.*?\} RADIANCE_SCENE_FALLBACKS;\n', '\n', 'task3 remove fallback type')
    gs = replace_once(gs, '    RADIANCE_SCENE_FALLBACKS radiance_fallbacks;\n', '', 'task3 remove fallback owner')
    gp.write_text(gs)

    p = Path('render.c')
    s = p.read_text()
    replacement = r'''static void destroy_global_sdf_resources(RENDERER *renderer) {
    if (!renderer) return;
    GLOBAL_SDF_DATA *global = &renderer->global_sdf;
    if (renderer->gpu) {
        if (global->clipmaps_srv) renderer->gpu->core.DestroyDescriptor(global->clipmaps_srv);
        if (global->page_table_srv) renderer->gpu->core.DestroyDescriptor(global->page_table_srv);
        if (global->bricks_srv) renderer->gpu->core.DestroyDescriptor(global->bricks_srv);
        if (global->surface_ids_srv) renderer->gpu->core.DestroyDescriptor(global->surface_ids_srv);
        if (global->clipmaps) gpu_destroy_buffer(renderer->gpu, global->clipmaps);
        if (global->page_table) gpu_destroy_buffer(renderer->gpu, global->page_table);
        if (global->bricks) gpu_destroy_buffer(renderer->gpu, global->bricks);
        if (global->surface_ids) gpu_destroy_buffer(renderer->gpu, global->surface_ids);
    }
    sdf_free_global_clipmaps(global);
}

static bool create_global_sdf_resources(RENDERER *renderer) {
    if (!renderer || !renderer->scene || !renderer->cpu_objects) return false;
    destroy_global_sdf_resources(renderer);
    GLOBAL_SDF_DATA *global = &renderer->global_sdf;
    if (!sdf_build_global_clipmaps(renderer->scene, &renderer->radiance_scene, renderer->cpu_objects, renderer->gpu_object_count, global))
        return false;

    const GPU_GLOBAL_SDF_CLIPMAP zero_clip = {0};
    const uint32_t invalid = UINT32_MAX;
    const float zero_distance = 0.0f;
    const uint32_t clip_count = global->valid ? global->clip_count : 1u;
    const uint32_t page_count = global->valid ? global->page_table_count : 1u;
    const uint32_t voxel_count = global->valid ? global->voxel_count : 1u;
    const GPU_GLOBAL_SDF_CLIPMAP *clips = global->valid ? global->cpu_clipmaps : &zero_clip;
    const uint32_t *pages = global->valid ? global->cpu_page_table : &invalid;
    const float *bricks = global->valid ? global->cpu_bricks : &zero_distance;
    const uint32_t *surface_ids = global->valid ? global->cpu_surface_ids : &invalid;

    const NriBufferDesc clip_desc = {
        .size = (uint64_t)clip_count * sizeof(GPU_GLOBAL_SDF_CLIPMAP),
        .structureStride = sizeof(GPU_GLOBAL_SDF_CLIPMAP),
        .usage = NriBufferUsageBits_SHADER_RESOURCE
    };
    const NriBufferDesc page_desc = {
        .size = (uint64_t)page_count * sizeof(uint32_t),
        .structureStride = sizeof(uint32_t),
        .usage = NriBufferUsageBits_SHADER_RESOURCE
    };
    const NriBufferDesc brick_desc = {
        .size = (uint64_t)voxel_count * sizeof(float),
        .structureStride = sizeof(float),
        .usage = NriBufferUsageBits_SHADER_RESOURCE
    };
    const NriBufferDesc surface_desc = {
        .size = (uint64_t)voxel_count * sizeof(uint32_t),
        .structureStride = sizeof(uint32_t),
        .usage = NriBufferUsageBits_SHADER_RESOURCE
    };

    if (!gpu_create_buffer(renderer->gpu, &clip_desc, NriMemoryLocation_DEVICE, &global->clipmaps) ||
        !gpu_create_buffer(renderer->gpu, &page_desc, NriMemoryLocation_DEVICE, &global->page_table) ||
        !gpu_create_buffer(renderer->gpu, &brick_desc, NriMemoryLocation_DEVICE, &global->bricks) ||
        !gpu_create_buffer(renderer->gpu, &surface_desc, NriMemoryLocation_DEVICE, &global->surface_ids) ||
        !create_buffer_view(renderer, global->clipmaps, NriBufferView_STRUCTURED_BUFFER, clip_desc.size, sizeof(GPU_GLOBAL_SDF_CLIPMAP), &global->clipmaps_srv) ||
        !create_buffer_view(renderer, global->page_table, NriBufferView_STRUCTURED_BUFFER, page_desc.size, sizeof(uint32_t), &global->page_table_srv) ||
        !create_buffer_view(renderer, global->bricks, NriBufferView_STRUCTURED_BUFFER, brick_desc.size, sizeof(float), &global->bricks_srv) ||
        !create_buffer_view(renderer, global->surface_ids, NriBufferView_STRUCTURED_BUFFER, surface_desc.size, sizeof(uint32_t), &global->surface_ids_srv)) {
        destroy_global_sdf_resources(renderer);
        return false;
    }

    const NriAccessStage read = {
        .access = NriAccessBits_SHADER_RESOURCE,
        .stages = NriStageBits_COMPUTE_SHADER
    };
    if (!gpu_upload_buffer(renderer->gpu, global->clipmaps, clips, read) ||
        !gpu_upload_buffer(renderer->gpu, global->page_table, pages, read) ||
        !gpu_upload_buffer(renderer->gpu, global->bricks, bricks, read) ||
        !gpu_upload_buffer(renderer->gpu, global->surface_ids, surface_ids, read)) {
        destroy_global_sdf_resources(renderer);
        return false;
    }
    global->state = read;
    SDL_Log("Global SDF: %u clips, %u physical bricks, %u voxels", global->clip_count, global->physical_brick_count, global->voxel_count);
    return true;
}

'''
    s = regex_once(
        s,
        r'static void destroy_radiance_scene_fallbacks\(RENDERER \*renderer\) \{.*?(?=static void destroy_dynamic_grid\(RENDERER \*renderer\))',
        replacement,
        'task3 replace fallback resources'
    )
    s = replace_once(s, '        !create_radiance_scene_fallbacks(renderer) || !create_world_radiance_resources(renderer) || !create_pipelines(renderer) ||\n',
                     '        !create_world_radiance_resources(renderer) || !create_pipelines(renderer) ||\n', 'task3 init fallback')
    s = replace_once(s, '        destroy_radiance_scene_fallbacks(renderer);\n', '', 'task3 deinit fallback')
    s = replace_once(s, 'static void destroy_scene_resources(RENDERER *renderer) {\n    destroy_radiance_scene_data(renderer);\n',
                     'static void destroy_scene_resources(RENDERER *renderer) {\n    destroy_global_sdf_resources(renderer);\n    destroy_radiance_scene_data(renderer);\n', 'task3 scene destroy')
    s = replace_once(
        s,
        '    if (!create_dynamic_grid(renderer) || !rebuild_dynamic_grid(renderer) || !update_radiance_constants(renderer)) {\n',
        '    if (!create_dynamic_grid(renderer) || !rebuild_dynamic_grid(renderer) || !create_global_sdf_resources(renderer) || !update_radiance_constants(renderer)) {\n',
        'task3 scene setup'
    )
    s = replace_once(
        s,
        '        !renderer->sdf.surface_ids_srv || !renderer->dynamic_grid.cells_srv || !renderer->dynamic_grid.indices_srv ||\n        !renderer->radiance_fallbacks.global_sdf_clipmaps_srv || !renderer->radiance_fallbacks.global_sdf_page_table_srv || !renderer->radiance_fallbacks.global_sdf_bricks_srv ||\n        !renderer->radiance_fallbacks.global_sdf_surface_ids_srv || !renderer->light_srv || !renderer->material_id.srv || !renderer->primitive_id.srv)\n',
        '        !renderer->sdf.surface_ids_srv || !renderer->dynamic_grid.cells_srv || !renderer->dynamic_grid.indices_srv ||\n        !renderer->global_sdf.clipmaps_srv || !renderer->global_sdf.page_table_srv || !renderer->global_sdf.bricks_srv ||\n        !renderer->global_sdf.surface_ids_srv || !renderer->light_srv || !renderer->material_id.srv || !renderer->primitive_id.srv)\n',
        'task3 descriptor guard'
    )
    s = replace_once(
        s,
        '        renderer->sdf.surface_ids_srv, renderer->dynamic_grid.cells_srv,\n        renderer->dynamic_grid.indices_srv, renderer->radiance_fallbacks.global_sdf_clipmaps_srv,\n        renderer->radiance_fallbacks.global_sdf_page_table_srv, renderer->radiance_fallbacks.global_sdf_bricks_srv,\n        renderer->radiance_fallbacks.global_sdf_surface_ids_srv\n',
        '        renderer->sdf.surface_ids_srv, renderer->dynamic_grid.cells_srv,\n        renderer->dynamic_grid.indices_srv, renderer->global_sdf.clipmaps_srv,\n        renderer->global_sdf.page_table_srv, renderer->global_sdf.bricks_srv,\n        renderer->global_sdf.surface_ids_srv\n',
        'task3 descriptor data'
    )
    s = replace_once(s, '    constants.sdf_counts[2] = 0u;\n',
                     '    constants.sdf_counts[2] = renderer->global_sdf.valid ? renderer->global_sdf.clip_count : 0u;\n', 'task3 clip count')
    s = replace_once(
        s,
        '    apply_dynamic_grid_constants(renderer, &constants);\n    constants.feature_flags[1] = 1u;\n',
        '    apply_dynamic_grid_constants(renderer, &constants);\n    constants.feature_flags[0] &= ~RADIANCE_FEATURE_GLOBAL_SDF;\n    if (renderer->global_sdf.valid && renderer->global_sdf.clip_count) constants.feature_flags[0] |= RADIANCE_FEATURE_GLOBAL_SDF;\n    constants.feature_flags[1] = 1u;\n',
        'task3 feature flag'
    )
    s = replace_once(s, '    constants.global_sdf_params[2] = 1.0e-4f;\n',
                     '    constants.global_sdf_params[1] = renderer->global_sdf.valid ? renderer->global_sdf.coarsest_voxel_size : 0.25f;\n    constants.global_sdf_params[2] = 0.65f;\n', 'task3 global params')
    p.write_text(s)


def task4():
    p = Path('shader.hlsl')
    s = p.read_text()
    s = replace_once(s, '#define MAX_DYNAMIC_GRID_STEPS 256u\n',
                     '#define MAX_DYNAMIC_GRID_STEPS 256u\n#define RAY_FLAG_DYNAMIC_HIT 0x80000000u\n#define RAY_FLAG_GLOBAL_RESOLVED 0x40000000u\n', 'task4 ray flags')
    s = replace_once(s, '    float epsilon = max(cell_size * 0.65f, Radiance.global_sdf_params.z);\n',
                     '    float epsilon = max(cell_size * 0.65f, 1.0e-4f);\n', 'task4 local epsilon')

    pattern = r'float3 GlobalSDFNormal\(float3 p, float voxel_size\) \{.*?\n\}\n\nbool TraceGlobalSDF\(TraceRay ray, inout SurfaceHit best_hit\) \{.*?\n\}\n'
    replacement = r'''bool RefineGlobalSDFSurface(
    TraceRay ray,
    float3 position,
    float sdf_t,
    float epsilon,
    float voxel_size,
    float limit,
    out SurfaceHit hit
) {
    hit = MakeSurfaceHit(TRACE_MISS, limit);
    float window = max(max(epsilon * 3.0f, voxel_size * 2.0f), 1.0e-3f);
    float min_t = max(ray.origin_tmin.w, sdf_t - window);
    float max_t = min(limit, sdf_t + window);
    if (max_t < min_t) return false;

    bool found = false;
    float best_t = max_t;
    float3 best_normal = 0.0f;
    uint best_surface = INVALID_INDEX;
    float step_size = max(voxel_size, 1.0e-4f);
    [loop]
    for (int z = -1; z <= 1; ++z) {
        [loop]
        for (int y = -1; y <= 1; ++y) {
            [loop]
            for (int x = -1; x <= 1; ++x) {
                float3 sample_position = position + float3((float)x, (float)y, (float)z) * step_size;
                uint surface_id;
                float sampled_voxel_size;
                SampleGlobalSDF(sample_position, surface_id, sampled_voxel_size);
                if (surface_id == INVALID_INDEX) continue;
                float candidate_t;
                float3 candidate_normal;
                if (IntersectSceneTriangle(ray, surface_id, min_t, best_t, candidate_t, candidate_normal)) {
                    found = true;
                    best_t = candidate_t;
                    best_normal = candidate_normal;
                    best_surface = surface_id;
                }
            }
        }
    }
    if (!found || best_surface == INVALID_INDEX) return false;
    float3 world_position = ray.origin_tmin.xyz + ray.direction_tmax.xyz * best_t;
    hit = SurfaceFromTriangle(best_surface, world_position, best_normal, best_t, TRACE_GLOBAL_SDF);
    GPUSceneTriangle tri = SceneTriangles[best_surface];
    uint object_index = tri.meta.x;
    hit.meta.y = object_index < Radiance.scene_counts.x ? SceneObjects[object_index].meta.x : Radiance.feature_flags.y;
    return hit.identity.z != INVALID_INDEX;
}

bool TraceGlobalSDF(TraceRay ray, inout SurfaceHit best_hit) {
    if (!FeatureEnabled(RADIANCE_FEATURE_GLOBAL_SDF) || Radiance.sdf_counts.z == 0u) return false;
    float t = max(ray.origin_tmin.w, 0.0f);
    float limit = min(ray.direction_tmax.w, best_hit.position_distance.w);
    uint max_steps = max(Radiance.trace_limits.w, 1u);
    float epsilon_scale = max(Radiance.global_sdf_params.z, 0.5f);
    [loop]
    for (uint step = 0u; step < max_steps && t <= limit; ++step) {
        float3 p = ray.origin_tmin.xyz + ray.direction_tmax.xyz * t;
        uint surface_id;
        float voxel_size;
        float d = SampleGlobalSDF(p, surface_id, voxel_size);
        float epsilon = max(voxel_size * epsilon_scale, 1.0e-3f);
        if (d <= epsilon) {
            float near_limit = ray.origin_tmin.w + epsilon * 1.5f;
            if (t <= near_limit) {
                t += max(epsilon * 1.5f, 1.0e-4f);
                continue;
            }
            SurfaceHit refined;
            if (RefineGlobalSDFSurface(ray, p, t, epsilon, voxel_size, limit, refined)) {
                best_hit = refined;
                return true;
            }
            t += max(epsilon * 0.5f, 1.0e-4f);
            continue;
        }
        t += max(d, epsilon * 0.25f);
    }
    return false;
}
'''
    s = regex_once(s, pattern, replacement, 'task4 global refinement')
    s = replace_once(
        s,
        '    TraceDynamicGrid(ray, best);\n    TraceAllLocalSDFs(ray, best);\n    TraceGlobalSDF(ray, best);\n    return best;\n',
        '    TraceDynamicGrid(ray, best);\n    bool global_resolved = TraceGlobalSDF(ray, best);\n    if (!global_resolved) TraceAllLocalSDFs(ray, best);\n    return best;\n',
        'task4 unified order'
    )
    s = replace_once(s, '        RayFlags[original] |= 0x80000000u;\n', '        RayFlags[original] |= RAY_FLAG_DYNAMIC_HIT;\n', 'task4 dynamic flag')
    old_global = '''[numthreads(64, 1, 1)]
void CS_WavefrontGlobalTrace(uint3 dispatch_id : SV_DispatchThreadID) {
    uint index = dispatch_id.x;
    if (index >= RayCounters[1]) return;
    TraceRay ray = RayQueueB[index];
    uint original = ray.destination;
    SurfaceHit hit = RaySurfaceHits[original];
    if (hit.position_distance.w <= 0.0f) hit = MakeSurfaceHit(TRACE_MISS, ray.direction_tmax.w);
    TraceGlobalSDF(ray, hit);
    RaySurfaceHits[original] = hit;
}
'''
    new_global = '''[numthreads(64, 1, 1)]
void CS_WavefrontGlobalTrace(uint3 dispatch_id : SV_DispatchThreadID) {
    uint index = dispatch_id.x;
    if (index >= RayCounters[1]) return;
    TraceRay ray = RayQueueB[index];
    uint original = ray.destination;
    SurfaceHit hit = (RayFlags[original] & RAY_FLAG_DYNAMIC_HIT) != 0u
        ? RaySurfaceHits[original]
        : MakeSurfaceHit(TRACE_MISS, ray.direction_tmax.w);
    if (TraceGlobalSDF(ray, hit)) RayFlags[original] |= RAY_FLAG_GLOBAL_RESOLVED;
    RaySurfaceHits[original] = hit;
}
'''
    s = replace_once(s, old_global, new_global, 'task4 global wave pass')
    old_local = '''[numthreads(64, 1, 1)]
void CS_WavefrontLocalTrace(uint3 dispatch_id : SV_DispatchThreadID) {
    uint index = dispatch_id.x;
    if (index >= RayCounters[1]) return;
    TraceRay ray = RayQueueB[index];
    uint original = ray.destination;
    SurfaceHit hit = (RayFlags[original] & 0x80000000u) != 0u ? RaySurfaceHits[original] : MakeSurfaceHit(TRACE_MISS, ray.direction_tmax.w);
    TraceAllLocalSDFs(ray, hit);
    RaySurfaceHits[original] = hit;
}
'''
    new_local = '''[numthreads(64, 1, 1)]
void CS_WavefrontLocalTrace(uint3 dispatch_id : SV_DispatchThreadID) {
    uint index = dispatch_id.x;
    if (index >= RayCounters[1]) return;
    TraceRay ray = RayQueueB[index];
    uint original = ray.destination;
    SurfaceHit hit = (RayFlags[original] & RAY_FLAG_DYNAMIC_HIT) != 0u ? RaySurfaceHits[original] : MakeSurfaceHit(TRACE_MISS, ray.direction_tmax.w);
    if ((RayFlags[original] & RAY_FLAG_GLOBAL_RESOLVED) == 0u) TraceAllLocalSDFs(ray, hit);
    RaySurfaceHits[original] = hit;
}
'''
    s = replace_once(s, old_local, new_local, 'task4 local fallback')
    p.write_text(s)

    p = Path('render.c')
    s = p.read_text()
    s = replace_once(
        s,
        '           create_compute_pipeline(renderer, "build/shaders/radiance_dynamic.cs.spv", renderer->wavefront_layout, &renderer->wavefront_dynamic_pipeline) &&\n           create_compute_pipeline(renderer, "build/shaders/radiance_local.cs.spv", renderer->wavefront_layout, &renderer->wavefront_local_pipeline) &&\n',
        '           create_compute_pipeline(renderer, "build/shaders/radiance_dynamic.cs.spv", renderer->wavefront_layout, &renderer->wavefront_dynamic_pipeline) &&\n           create_compute_pipeline(renderer, "build/shaders/radiance_global.cs.spv", renderer->wavefront_layout, &renderer->wavefront_global_pipeline) &&\n           create_compute_pipeline(renderer, "build/shaders/radiance_local.cs.spv", renderer->wavefront_layout, &renderer->wavefront_local_pipeline) &&\n',
        'task4 pipeline creation'
    )
    s = replace_once(
        s,
        '            renderer->wavefront_dynamic_pipeline,\n            renderer->wavefront_local_pipeline,\n',
        '            renderer->wavefront_dynamic_pipeline,\n            renderer->wavefront_global_pipeline,\n            renderer->wavefront_local_pipeline,\n',
        'task4 pipeline destroy'
    )
    marker = '''    if (renderer->radiance_constants.feature_flags[0] & RADIANCE_FEATURE_DYNAMIC_GRID) {
        bind_wavefront(renderer, command_buffer, renderer->wavefront_dynamic_pipeline);
        renderer->gpu->core.CmdDispatch(command_buffer, &(NriDispatchDesc){.workGroupNumX = ray_groups, .workGroupNumY = 1, .workGroupNumZ = 1});
        barrier_wavefront_buffers(renderer, command_buffer, storage);
    }

    bind_wavefront(renderer, command_buffer, renderer->wavefront_local_pipeline);
'''
    replacement = '''    if (renderer->radiance_constants.feature_flags[0] & RADIANCE_FEATURE_DYNAMIC_GRID) {
        bind_wavefront(renderer, command_buffer, renderer->wavefront_dynamic_pipeline);
        renderer->gpu->core.CmdDispatch(command_buffer, &(NriDispatchDesc){.workGroupNumX = ray_groups, .workGroupNumY = 1, .workGroupNumZ = 1});
        barrier_wavefront_buffers(renderer, command_buffer, storage);
    }

    if (renderer->radiance_constants.feature_flags[0] & RADIANCE_FEATURE_GLOBAL_SDF) {
        bind_wavefront(renderer, command_buffer, renderer->wavefront_global_pipeline);
        renderer->gpu->core.CmdDispatch(command_buffer, &(NriDispatchDesc){.workGroupNumX = ray_groups, .workGroupNumY = 1, .workGroupNumZ = 1});
        barrier_wavefront_buffers(renderer, command_buffer, storage);
    }

    bind_wavefront(renderer, command_buffer, renderer->wavefront_local_pipeline);
'''
    s = replace_once(s, marker, replacement, 'task4 dispatch order')
    p.write_text(s)


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in {'1','2','3','4'}:
        raise SystemExit('usage: stage9_apply.py TASK')
    {'1': task1, '2': task2, '3': task3, '4': task4}[sys.argv[1]]()


if __name__ == '__main__':
    main()
