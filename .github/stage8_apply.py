from pathlib import Path
import re


def replace_once(text, old, new, label):
    count = text.count(old)
    if count != 1:
        raise SystemExit(f'{label}: expected 1 match, found {count}')
    return text.replace(old, new, 1)


def sub_once(text, pattern, replacement, label):
    out, count = re.subn(pattern, replacement, text, count=1, flags=re.S)
    if count != 1:
        raise SystemExit(f'{label}: expected 1 match, found {count}')
    return out

# -----------------------------------------------------------------------------
# game.h
# -----------------------------------------------------------------------------
p = Path('game.h')
h = p.read_text()

pattern = r'''typedef struct RADIANCE_SCENE_FALLBACKS \{.*?\} RADIANCE_SCENE_FALLBACKS;\n\ntypedef struct GPU_DYNAMIC_GRID_CELL \{\n    uint32_t range_flags\[4\];\n    float bounds_min\[4\];\n    float bounds_max\[4\];\n\} GPU_DYNAMIC_GRID_CELL;'''
replacement = '''typedef struct RADIANCE_SCENE_FALLBACKS {
    NriBuffer *global_sdf_clipmaps;
    NriBuffer *global_sdf_page_table;
    NriBuffer *global_sdf_bricks;
    NriBuffer *global_sdf_surface_ids;

    NriDescriptor *global_sdf_clipmaps_srv;
    NriDescriptor *global_sdf_page_table_srv;
    NriDescriptor *global_sdf_bricks_srv;
    NriDescriptor *global_sdf_surface_ids_srv;

    NriAccessStage state;
} RADIANCE_SCENE_FALLBACKS;

typedef struct GPU_DYNAMIC_GRID_CELL {
    uint32_t range_flags[4];
    float bounds_min[4];
    float bounds_max[4];
} GPU_DYNAMIC_GRID_CELL;

typedef struct RADIANCE_DYNAMIC_GRID {
    GPU_DYNAMIC_GRID_CELL *cpu_cells;
    uint32_t *cpu_indices;

    NriBuffer *cells;
    NriBuffer *indices;
    NriDescriptor *cells_srv;
    NriDescriptor *indices_srv;
    NriAccessStage state;

    uint32_t dimensions[3];
    uint32_t cell_count;
    uint32_t index_count;
    uint32_t model_count;
    uint32_t cell_capacity;
    uint32_t index_capacity;
    uint32_t dimension_limit;

    float origin[3];
    float cell_size;
    uint64_t signature;
    bool dirty;
} RADIANCE_DYNAMIC_GRID;'''
h = sub_once(h, pattern, replacement, 'dynamic grid resource types')

h = replace_once(
    h,
    '''    RADIANCE_SCENE_DATA radiance_scene;\n    RADIANCE_SCENE_FALLBACKS radiance_fallbacks;''',
    '''    RADIANCE_SCENE_DATA radiance_scene;\n    RADIANCE_DYNAMIC_GRID dynamic_grid;\n    RADIANCE_SCENE_FALLBACKS radiance_fallbacks;''',
    'renderer dynamic grid state'
)

h = replace_once(
    h,
    '''    NriPipeline *wavefront_screen_pipeline;\n    NriPipeline *wavefront_local_pipeline;''',
    '''    NriPipeline *wavefront_screen_pipeline;\n    NriPipeline *wavefront_dynamic_pipeline;\n    NriPipeline *wavefront_local_pipeline;''',
    'dynamic wavefront pipeline handle'
)
p.write_text(h)

# -----------------------------------------------------------------------------
# render.c
# -----------------------------------------------------------------------------
p = Path('render.c')
r = p.read_text()

r = replace_once(
    r,
    '#define NEAR_PLANE 0.05f\n',
    '#define NEAR_PLANE 0.05f\n#define DYNAMIC_GRID_MAX_DIM 16u\n#define DYNAMIC_GRID_MIN_DIM 4u\n#define DYNAMIC_GRID_INDEX_BUDGET 1048576u\n',
    'dynamic grid constants'
)

r = sub_once(
    r,
    r'''static void destroy_radiance_scene_fallbacks\(RENDERER \*renderer\) \{.*?\n\}\n\nstatic bool create_radiance_fallback_buffer''',
    '''static void destroy_radiance_scene_fallbacks(RENDERER *renderer) {
    if (!renderer || !renderer->gpu) return;

    RADIANCE_SCENE_FALLBACKS *fallbacks = &renderer->radiance_fallbacks;

    if (fallbacks->global_sdf_clipmaps_srv) renderer->gpu->core.DestroyDescriptor(fallbacks->global_sdf_clipmaps_srv);
    if (fallbacks->global_sdf_page_table_srv) renderer->gpu->core.DestroyDescriptor(fallbacks->global_sdf_page_table_srv);
    if (fallbacks->global_sdf_bricks_srv) renderer->gpu->core.DestroyDescriptor(fallbacks->global_sdf_bricks_srv);
    if (fallbacks->global_sdf_surface_ids_srv) renderer->gpu->core.DestroyDescriptor(fallbacks->global_sdf_surface_ids_srv);

    if (fallbacks->global_sdf_clipmaps) gpu_destroy_buffer(renderer->gpu, fallbacks->global_sdf_clipmaps);
    if (fallbacks->global_sdf_page_table) gpu_destroy_buffer(renderer->gpu, fallbacks->global_sdf_page_table);
    if (fallbacks->global_sdf_bricks) gpu_destroy_buffer(renderer->gpu, fallbacks->global_sdf_bricks);
    if (fallbacks->global_sdf_surface_ids) gpu_destroy_buffer(renderer->gpu, fallbacks->global_sdf_surface_ids);

    memset(fallbacks, 0, sizeof(*fallbacks));
}

static bool create_radiance_fallback_buffer''',
    'fallback destroy cleanup'
)

r = sub_once(
    r,
    r'''static bool create_radiance_scene_fallbacks\(RENDERER \*renderer\) \{.*?\n\}\n\nstatic bool refresh_emissive_sampling''',
    '''static bool create_radiance_scene_fallbacks(RENDERER *renderer) {
    destroy_radiance_scene_fallbacks(renderer);

    RADIANCE_SCENE_FALLBACKS *fallbacks = &renderer->radiance_fallbacks;

    if (!create_radiance_fallback_buffer(renderer, sizeof(GPU_GLOBAL_SDF_CLIPMAP), &fallbacks->global_sdf_clipmaps, &fallbacks->global_sdf_clipmaps_srv) ||
        !create_radiance_fallback_buffer(renderer, sizeof(uint32_t), &fallbacks->global_sdf_page_table, &fallbacks->global_sdf_page_table_srv) ||
        !create_radiance_fallback_buffer(renderer, sizeof(float), &fallbacks->global_sdf_bricks, &fallbacks->global_sdf_bricks_srv) ||
        !create_radiance_fallback_buffer(renderer, sizeof(uint32_t), &fallbacks->global_sdf_surface_ids, &fallbacks->global_sdf_surface_ids_srv)) {
        destroy_radiance_scene_fallbacks(renderer);
        return false;
    }

    fallbacks->state = (NriAccessStage){
        .access = NriAccessBits_SHADER_RESOURCE,
        .stages = NriStageBits_COMPUTE_SHADER
    };

    return true;
}

static void destroy_dynamic_grid(RENDERER *renderer) {
    if (!renderer || !renderer->gpu) return;
    RADIANCE_DYNAMIC_GRID *grid = &renderer->dynamic_grid;
    if (grid->cells_srv) renderer->gpu->core.DestroyDescriptor(grid->cells_srv);
    if (grid->indices_srv) renderer->gpu->core.DestroyDescriptor(grid->indices_srv);
    if (grid->cells) gpu_destroy_buffer(renderer->gpu, grid->cells);
    if (grid->indices) gpu_destroy_buffer(renderer->gpu, grid->indices);
    free(grid->cpu_cells);
    free(grid->cpu_indices);
    memset(grid, 0, sizeof(*grid));
}

static bool create_dynamic_grid(RENDERER *renderer) {
    destroy_dynamic_grid(renderer);
    RADIANCE_DYNAMIC_GRID *grid = &renderer->dynamic_grid;
    const uint32_t model_capacity = renderer->sdf.model_count ? renderer->sdf.model_count : 1u;
    uint32_t dim = DYNAMIC_GRID_MAX_DIM;
    while (dim > DYNAMIC_GRID_MIN_DIM && (uint64_t)dim * dim * dim * model_capacity > DYNAMIC_GRID_INDEX_BUDGET)
        dim >>= 1u;

    const uint64_t cell_capacity = (uint64_t)dim * dim * dim;
    const uint64_t index_capacity = cell_capacity * model_capacity;
    if (!cell_capacity || cell_capacity > UINT32_MAX || !index_capacity || index_capacity > UINT32_MAX) return false;

    grid->dimension_limit = dim;
    grid->cell_capacity = (uint32_t)cell_capacity;
    grid->index_capacity = (uint32_t)index_capacity;
    grid->cpu_cells = calloc(grid->cell_capacity, sizeof(*grid->cpu_cells));
    grid->cpu_indices = calloc(grid->index_capacity, sizeof(*grid->cpu_indices));
    if (!grid->cpu_cells || !grid->cpu_indices) {
        destroy_dynamic_grid(renderer);
        return false;
    }

    const NriBufferDesc cells_desc = {
        .size = (uint64_t)grid->cell_capacity * sizeof(GPU_DYNAMIC_GRID_CELL),
        .structureStride = sizeof(GPU_DYNAMIC_GRID_CELL),
        .usage = NriBufferUsageBits_SHADER_RESOURCE
    };
    const NriBufferDesc indices_desc = {
        .size = (uint64_t)grid->index_capacity * sizeof(uint32_t),
        .structureStride = sizeof(uint32_t),
        .usage = NriBufferUsageBits_SHADER_RESOURCE
    };
    if (!gpu_create_buffer(renderer->gpu, &cells_desc, NriMemoryLocation_DEVICE, &grid->cells) ||
        !gpu_create_buffer(renderer->gpu, &indices_desc, NriMemoryLocation_DEVICE, &grid->indices) ||
        !create_buffer_view(renderer, grid->cells, NriBufferView_STRUCTURED_BUFFER, cells_desc.size, sizeof(GPU_DYNAMIC_GRID_CELL), &grid->cells_srv) ||
        !create_buffer_view(renderer, grid->indices, NriBufferView_STRUCTURED_BUFFER, indices_desc.size, sizeof(uint32_t), &grid->indices_srv)) {
        destroy_dynamic_grid(renderer);
        return false;
    }

    grid->dirty = true;
    return true;
}

static uint64_t dynamic_grid_signature(const RENDERER *renderer) {
    uint64_t hash = 1469598103934665603ull;
    uint32_t dynamic_count = 0u;
    for (uint32_t i = 0; i < renderer->gpu_object_count; ++i) {
        const GPU_OBJECT *object = &renderer->cpu_objects[i];
        if (object->state != (uint32_t)DYNAMIC) continue;
        uint64_t token = ((uint64_t)(i + 1u) << 32u) ^ (uint64_t)object->revision;
        hash ^= token;
        hash *= 1099511628211ull;
        ++dynamic_count;
    }
    hash ^= dynamic_count;
    hash *= 1099511628211ull;
    return hash;
}

static void dynamic_model_world_bounds(const RENDERER *renderer, uint32_t model_index, VEC3 *out_min, VEC3 *out_max) {
    const GPU_SDF_MODEL *model = &renderer->sdf.cpu_models[model_index];
    const MAT4 world = renderer->cpu_objects[model_index].world;
    VEC3 minimum = v3(INFINITY, INFINITY, INFINITY);
    VEC3 maximum = v3(-INFINITY, -INFINITY, -INFINITY);
    for (uint32_t corner = 0u; corner < 8u; ++corner) {
        const VEC3 local = v3(
            (corner & 1u) ? model->bounds_max[0] : model->bounds_min[0],
            (corner & 2u) ? model->bounds_max[1] : model->bounds_min[1],
            (corner & 4u) ? model->bounds_max[2] : model->bounds_min[2]
        );
        const VEC3 p = mat4_point(world, local);
        minimum.x = fminf(minimum.x, p.x);
        minimum.y = fminf(minimum.y, p.y);
        minimum.z = fminf(minimum.z, p.z);
        maximum.x = fmaxf(maximum.x, p.x);
        maximum.y = fmaxf(maximum.y, p.y);
        maximum.z = fmaxf(maximum.z, p.z);
    }
    *out_min = minimum;
    *out_max = maximum;
}

static uint32_t dynamic_grid_coord(float value, float origin, float cell_size, uint32_t dim) {
    int32_t coordinate = (int32_t)floorf((value - origin) / cell_size);
    if (coordinate < 0) coordinate = 0;
    if ((uint32_t)coordinate >= dim) coordinate = (int32_t)dim - 1;
    return (uint32_t)coordinate;
}

static bool rebuild_dynamic_grid(RENDERER *renderer) {
    RADIANCE_DYNAMIC_GRID *grid = &renderer->dynamic_grid;
    if (!grid->cpu_cells || !grid->cpu_indices || !grid->cells || !grid->indices || renderer->sdf.model_count != renderer->gpu_object_count) return false;

    typedef struct DYNAMIC_MODEL_BOUNDS {
        uint32_t model_index;
        VEC3 min;
        VEC3 max;
    } DYNAMIC_MODEL_BOUNDS;

    uint32_t dynamic_count = 0u;
    for (uint32_t i = 0; i < renderer->gpu_object_count; ++i)
        if (renderer->cpu_objects[i].state == (uint32_t)DYNAMIC) ++dynamic_count;

    memset(grid->cpu_cells, 0, (size_t)grid->cell_capacity * sizeof(*grid->cpu_cells));
    grid->cpu_indices[0] = 0u;
    grid->dimensions[0] = grid->dimensions[1] = grid->dimensions[2] = 0u;
    grid->cell_count = 0u;
    grid->index_count = 0u;
    grid->model_count = dynamic_count;
    grid->origin[0] = grid->origin[1] = grid->origin[2] = 0.0f;
    grid->cell_size = 0.0f;
    grid->signature = dynamic_grid_signature(renderer);
    grid->dirty = true;
    if (!dynamic_count) return true;

    DYNAMIC_MODEL_BOUNDS *bounds = calloc(dynamic_count, sizeof(*bounds));
    if (!bounds) return false;

    VEC3 scene_min = v3(INFINITY, INFINITY, INFINITY);
    VEC3 scene_max = v3(-INFINITY, -INFINITY, -INFINITY);
    uint32_t dynamic_index = 0u;
    for (uint32_t model_index = 0; model_index < renderer->gpu_object_count; ++model_index) {
        if (renderer->cpu_objects[model_index].state != (uint32_t)DYNAMIC) continue;
        DYNAMIC_MODEL_BOUNDS *entry = &bounds[dynamic_index++];
        entry->model_index = model_index;
        dynamic_model_world_bounds(renderer, model_index, &entry->min, &entry->max);
        scene_min.x = fminf(scene_min.x, entry->min.x);
        scene_min.y = fminf(scene_min.y, entry->min.y);
        scene_min.z = fminf(scene_min.z, entry->min.z);
        scene_max.x = fmaxf(scene_max.x, entry->max.x);
        scene_max.y = fmaxf(scene_max.y, entry->max.y);
        scene_max.z = fmaxf(scene_max.z, entry->max.z);
    }

    const VEC3 extent = v3(scene_max.x - scene_min.x, scene_max.y - scene_min.y, scene_max.z - scene_min.z);
    const float longest = fmaxf(extent.x, fmaxf(extent.y, extent.z));
    const uint32_t interior_dim = grid->dimension_limit > 2u ? grid->dimension_limit - 2u : grid->dimension_limit;
    const float cell_size = fmaxf(longest / fmaxf((float)interior_dim, 1.0f), 0.25f);
    grid->cell_size = cell_size;
    grid->origin[0] = scene_min.x - cell_size;
    grid->origin[1] = scene_min.y - cell_size;
    grid->origin[2] = scene_min.z - cell_size;

    const float padded_max[3] = {scene_max.x + cell_size, scene_max.y + cell_size, scene_max.z + cell_size};
    for (uint32_t axis = 0u; axis < 3u; ++axis) {
        uint32_t dim = (uint32_t)ceilf((padded_max[axis] - grid->origin[axis]) / cell_size);
        if (!dim) dim = 1u;
        if (dim > grid->dimension_limit) dim = grid->dimension_limit;
        grid->dimensions[axis] = dim;
    }

    const uint64_t cell_count64 = (uint64_t)grid->dimensions[0] * grid->dimensions[1] * grid->dimensions[2];
    if (!cell_count64 || cell_count64 > grid->cell_capacity) {
        free(bounds);
        return false;
    }
    grid->cell_count = (uint32_t)cell_count64;

    uint32_t *counts = calloc(grid->cell_count, sizeof(*counts));
    uint32_t *cursor = calloc(grid->cell_count, sizeof(*cursor));
    if (!counts || !cursor) {
        free(counts);
        free(cursor);
        free(bounds);
        return false;
    }

    uint64_t total_indices = 0u;
    const uint32_t dim_x = grid->dimensions[0];
    const uint32_t dim_y = grid->dimensions[1];
    const uint32_t dim_z = grid->dimensions[2];
    for (uint32_t i = 0u; i < dynamic_count; ++i) {
        const DYNAMIC_MODEL_BOUNDS *entry = &bounds[i];
        const uint32_t min_x = dynamic_grid_coord(entry->min.x, grid->origin[0], cell_size, dim_x);
        const uint32_t min_y = dynamic_grid_coord(entry->min.y, grid->origin[1], cell_size, dim_y);
        const uint32_t min_z = dynamic_grid_coord(entry->min.z, grid->origin[2], cell_size, dim_z);
        const uint32_t max_x = dynamic_grid_coord(entry->max.x, grid->origin[0], cell_size, dim_x);
        const uint32_t max_y = dynamic_grid_coord(entry->max.y, grid->origin[1], cell_size, dim_y);
        const uint32_t max_z = dynamic_grid_coord(entry->max.z, grid->origin[2], cell_size, dim_z);
        for (uint32_t z = min_z; z <= max_z; ++z)
            for (uint32_t y = min_y; y <= max_y; ++y)
                for (uint32_t x = min_x; x <= max_x; ++x) {
                    const uint32_t flat = x + dim_x * (y + dim_y * z);
                    ++counts[flat];
                    ++total_indices;
                }
    }

    if (total_indices > grid->index_capacity) {
        free(counts);
        free(cursor);
        free(bounds);
        return false;
    }
    grid->index_count = (uint32_t)total_indices;

    uint32_t offset = 0u;
    for (uint32_t z = 0u; z < dim_z; ++z) {
        for (uint32_t y = 0u; y < dim_y; ++y) {
            for (uint32_t x = 0u; x < dim_x; ++x) {
                const uint32_t flat = x + dim_x * (y + dim_y * z);
                GPU_DYNAMIC_GRID_CELL *cell = &grid->cpu_cells[flat];
                cell->range_flags[0] = offset;
                cell->range_flags[1] = counts[flat];
                cursor[flat] = offset;
                offset += counts[flat];
                cell->bounds_min[0] = grid->origin[0] + (float)x * cell_size;
                cell->bounds_min[1] = grid->origin[1] + (float)y * cell_size;
                cell->bounds_min[2] = grid->origin[2] + (float)z * cell_size;
                cell->bounds_max[0] = cell->bounds_min[0] + cell_size;
                cell->bounds_max[1] = cell->bounds_min[1] + cell_size;
                cell->bounds_max[2] = cell->bounds_min[2] + cell_size;
            }
        }
    }

    for (uint32_t i = 0u; i < dynamic_count; ++i) {
        const DYNAMIC_MODEL_BOUNDS *entry = &bounds[i];
        const uint32_t min_x = dynamic_grid_coord(entry->min.x, grid->origin[0], cell_size, dim_x);
        const uint32_t min_y = dynamic_grid_coord(entry->min.y, grid->origin[1], cell_size, dim_y);
        const uint32_t min_z = dynamic_grid_coord(entry->min.z, grid->origin[2], cell_size, dim_z);
        const uint32_t max_x = dynamic_grid_coord(entry->max.x, grid->origin[0], cell_size, dim_x);
        const uint32_t max_y = dynamic_grid_coord(entry->max.y, grid->origin[1], cell_size, dim_y);
        const uint32_t max_z = dynamic_grid_coord(entry->max.z, grid->origin[2], cell_size, dim_z);
        for (uint32_t z = min_z; z <= max_z; ++z)
            for (uint32_t y = min_y; y <= max_y; ++y)
                for (uint32_t x = min_x; x <= max_x; ++x) {
                    const uint32_t flat = x + dim_x * (y + dim_y * z);
                    grid->cpu_indices[cursor[flat]++] = entry->model_index;
                }
    }

    free(counts);
    free(cursor);
    free(bounds);
    return true;
}

static void apply_dynamic_grid_constants(RENDERER *renderer, RADIANCE_CONSTANTS *constants) {
    const RADIANCE_DYNAMIC_GRID *grid = &renderer->dynamic_grid;
    constants->sdf_counts[1] = grid->cell_count;
    constants->dynamic_grid[0] = grid->dimensions[0];
    constants->dynamic_grid[1] = grid->dimensions[1];
    constants->dynamic_grid[2] = grid->dimensions[2];
    constants->dynamic_grid[3] = grid->index_count;
    constants->dynamic_grid_origin_cell[0] = grid->origin[0];
    constants->dynamic_grid_origin_cell[1] = grid->origin[1];
    constants->dynamic_grid_origin_cell[2] = grid->origin[2];
    constants->dynamic_grid_origin_cell[3] = grid->cell_size;
    constants->feature_flags[0] &= ~RADIANCE_FEATURE_DYNAMIC_GRID;
    if (grid->model_count && grid->cell_count && grid->index_count)
        constants->feature_flags[0] |= RADIANCE_FEATURE_DYNAMIC_GRID;
}

static bool refresh_dynamic_grid(RENDERER *renderer) {
    RADIANCE_DYNAMIC_GRID *grid = &renderer->dynamic_grid;
    if (!grid->cells || !grid->indices) return false;
    const uint64_t signature = dynamic_grid_signature(renderer);
    if (signature == grid->signature) return true;
    if (!rebuild_dynamic_grid(renderer)) return false;
    apply_dynamic_grid_constants(renderer, &renderer->radiance_constants);
    renderer->probe_history_valid = false;
    return true;
}

static bool refresh_emissive_sampling''',
    'dynamic grid implementation'
)

r = replace_once(
    r,
    '''    if (!ok || !create_sdf_scene(renderer, scene) || !create_radiance_scene_gpu_resources(renderer) || !update_radiance_constants(renderer) ||\n        !update_radiance_scene_descriptors(renderer)) {''',
    '''    if (!ok || !create_sdf_scene(renderer, scene) || !create_radiance_scene_gpu_resources(renderer)) {''',
    'defer scene descriptors until grid exists'
)

r = replace_once(
    r,
    '''static void destroy_scene_resources(RENDERER *renderer) {\n    destroy_radiance_scene_data(renderer);\n    destroy_sdf_scene(renderer);''',
    '''static void destroy_scene_resources(RENDERER *renderer) {\n    destroy_radiance_scene_data(renderer);\n    destroy_dynamic_grid(renderer);\n    destroy_sdf_scene(renderer);''',
    'destroy dynamic grid with scene'
)

r = replace_once(
    r,
    '''    constants.sdf_counts[0] = renderer->sdf.model_count;\n    constants.sdf_counts[1] = 0u;''',
    '''    constants.sdf_counts[0] = renderer->sdf.model_count;\n    constants.sdf_counts[1] = renderer->dynamic_grid.cell_count;''',
    'dynamic cell count constants'
)

r = replace_once(
    r,
    '''    if (renderer->radiance_scene.emissive_triangle_count) constants.feature_flags[0] |= RADIANCE_FEATURE_EMISSIVE;\n    constants.feature_flags[1] = 1u;''',
    '''    if (renderer->radiance_scene.emissive_triangle_count) constants.feature_flags[0] |= RADIANCE_FEATURE_EMISSIVE;\n    apply_dynamic_grid_constants(renderer, &constants);\n    constants.feature_flags[1] = 1u;''',
    'apply dynamic grid constants'
)

r = replace_once(
    r,
    '''           create_compute_pipeline(renderer, "build/shaders/radiance_screen.cs.spv", renderer->wavefront_layout, &renderer->wavefront_screen_pipeline) &&\n           create_compute_pipeline(renderer, "build/shaders/radiance_local.cs.spv", renderer->wavefront_layout, &renderer->wavefront_local_pipeline) &&''',
    '''           create_compute_pipeline(renderer, "build/shaders/radiance_screen.cs.spv", renderer->wavefront_layout, &renderer->wavefront_screen_pipeline) &&\n           create_compute_pipeline(renderer, "build/shaders/radiance_dynamic.cs.spv", renderer->wavefront_layout, &renderer->wavefront_dynamic_pipeline) &&\n           create_compute_pipeline(renderer, "build/shaders/radiance_local.cs.spv", renderer->wavefront_layout, &renderer->wavefront_local_pipeline) &&''',
    'dynamic pipeline creation'
)

r = replace_once(
    r,
    '''        !renderer->sdf.surface_ids_srv || !renderer->radiance_fallbacks.dynamic_grid_cells_srv || !renderer->radiance_fallbacks.dynamic_grid_indices_srv ||\n        !renderer->radiance_fallbacks.global_sdf_clipmaps_srv ||''',
    '''        !renderer->sdf.surface_ids_srv || !renderer->dynamic_grid.cells_srv || !renderer->dynamic_grid.indices_srv ||\n        !renderer->radiance_fallbacks.global_sdf_clipmaps_srv ||''',
    'dynamic descriptor validation'
)

r = replace_once(
    r,
    '''        renderer->sdf.surface_ids_srv, renderer->radiance_fallbacks.dynamic_grid_cells_srv,\n        renderer->radiance_fallbacks.dynamic_grid_indices_srv, renderer->radiance_fallbacks.global_sdf_clipmaps_srv,''',
    '''        renderer->sdf.surface_ids_srv, renderer->dynamic_grid.cells_srv,\n        renderer->dynamic_grid.indices_srv, renderer->radiance_fallbacks.global_sdf_clipmaps_srv,''',
    'dynamic descriptor binding'
)

# Replace the streamer function so dynamic-grid rebuilds and constants use the normal frame copy path.
r = sub_once(
    r,
    r'''static bool stream_dynamic_data\(RENDERER \*renderer, NriCommandBuffer \*command_buffer, const FRAME_CONSTANTS \*frame\) \{.*?\n\}\n\nstatic void transition_gbuffer_for_render''',
    '''static bool stream_dynamic_data(RENDERER *renderer, NriCommandBuffer *command_buffer, const FRAME_CONSTANTS *frame) {
    memset(&renderer->pass_constants, 0, sizeof(renderer->pass_constants));
    renderer->pass_constants.dispatch[0] = renderer->radiance_constants.feature_flags[1];
    renderer->pass_constants.dimensions[0] = renderer->screen_probes.width;
    renderer->pass_constants.dimensions[1] = renderer->screen_probes.height;
    renderer->pass_constants.flags[0] = renderer->probe_history_valid ? 1u : 0u;

    NriStreamerCopyBatch batch = renderer->gpu->streamer_api.BeginStreamerCopyBatch(renderer->gpu->streamer);
    if (!batch) return false;

    const NriDataSize object_data = {.data = renderer->cpu_objects, .size = (uint64_t)renderer->gpu_object_count * sizeof(GPU_OBJECT)};
    const NriDataSize light_data = {.data = renderer->cpu_lights, .size = (uint64_t)renderer->light_count * sizeof(GPU_LIGHT)};
    const NriDataSize sdf_data = {.data = renderer->sdf.cpu_models, .size = (uint64_t)renderer->sdf.model_count * sizeof(GPU_SDF_MODEL)};
    const NriDataSize frame_data = {.data = frame, .size = sizeof(*frame)};
    const NriDataSize pass_data = {.data = &renderer->pass_constants, .size = sizeof(renderer->pass_constants)};
    const NriDataSize radiance_data = {.data = &renderer->radiance_constants, .size = sizeof(renderer->radiance_constants)};
    const uint32_t cell_upload_count = renderer->dynamic_grid.cell_count ? renderer->dynamic_grid.cell_count : 1u;
    const uint32_t index_upload_count = renderer->dynamic_grid.index_count ? renderer->dynamic_grid.index_count : 1u;
    const NriDataSize grid_cell_data = {.data = renderer->dynamic_grid.cpu_cells, .size = (uint64_t)cell_upload_count * sizeof(GPU_DYNAMIC_GRID_CELL)};
    const NriDataSize grid_index_data = {.data = renderer->dynamic_grid.cpu_indices, .size = (uint64_t)index_upload_count * sizeof(uint32_t)};

    NriStreamBufferDataDesc uploads[8];
    uint32_t upload_count = 0u;
    uploads[upload_count++] = (NriStreamBufferDataDesc){.dataChunks = &object_data, .dataChunkNum = 1, .placementAlignment = 16, .copyBatch = batch, .dstBuffer = renderer->object_buffer};
    uploads[upload_count++] = (NriStreamBufferDataDesc){.dataChunks = &light_data, .dataChunkNum = 1, .placementAlignment = 16, .copyBatch = batch, .dstBuffer = renderer->light_buffer};
    uploads[upload_count++] = (NriStreamBufferDataDesc){.dataChunks = &sdf_data, .dataChunkNum = 1, .placementAlignment = 16, .copyBatch = batch, .dstBuffer = renderer->sdf.models};
    uploads[upload_count++] = (NriStreamBufferDataDesc){.dataChunks = &frame_data, .dataChunkNum = 1, .placementAlignment = 16, .copyBatch = batch, .dstBuffer = renderer->frame_buffer};
    uploads[upload_count++] = (NriStreamBufferDataDesc){.dataChunks = &pass_data, .dataChunkNum = 1, .placementAlignment = 16, .copyBatch = batch, .dstBuffer = renderer->pass_constants_buffer};
    uploads[upload_count++] = (NriStreamBufferDataDesc){.dataChunks = &radiance_data, .dataChunkNum = 1, .placementAlignment = 16, .copyBatch = batch, .dstBuffer = renderer->radiance_constants_buffer};
    const bool upload_grid = renderer->dynamic_grid.dirty;
    if (upload_grid) {
        uploads[upload_count++] = (NriStreamBufferDataDesc){.dataChunks = &grid_cell_data, .dataChunkNum = 1, .placementAlignment = 16, .copyBatch = batch, .dstBuffer = renderer->dynamic_grid.cells};
        uploads[upload_count++] = (NriStreamBufferDataDesc){.dataChunks = &grid_index_data, .dataChunkNum = 1, .placementAlignment = 16, .copyBatch = batch, .dstBuffer = renderer->dynamic_grid.indices};
    }

    for (uint32_t i = 0u; i < upload_count; ++i) {
        const NriBufferOffset streamed = renderer->gpu->streamer_api.StreamBufferData(renderer->gpu->streamer, &uploads[i]);
        if (!streamed.buffer) return false;
    }

    const NriAccessStage copy = {.access = NriAccessBits_COPY_DESTINATION, .stages = NriStageBits_COPY};
    const NriAccessStage object_read = {.access = NriAccessBits_SHADER_RESOURCE, .stages = NriStageBits_VERTEX_SHADER | NriStageBits_COMPUTE_SHADER};
    const NriAccessStage compute_read = {.access = NriAccessBits_SHADER_RESOURCE, .stages = NriStageBits_COMPUTE_SHADER};
    const NriAccessStage frame_read = {.access = NriAccessBits_CONSTANT_BUFFER, .stages = NriStageBits_VERTEX_SHADER | NriStageBits_COMPUTE_SHADER};
    const NriAccessStage pass_read = {.access = NriAccessBits_CONSTANT_BUFFER, .stages = NriStageBits_COMPUTE_SHADER};
    const NriAccessStage radiance_read = {.access = NriAccessBits_CONSTANT_BUFFER, .stages = NriStageBits_COMPUTE_SHADER | NriStageBits_FRAGMENT_SHADER};

    NriBufferBarrierDesc before[8];
    uint32_t barrier_count = 0u;
    before[barrier_count++] = (NriBufferBarrierDesc){.buffer = renderer->object_buffer, .before = renderer->object_state, .after = copy};
    before[barrier_count++] = (NriBufferBarrierDesc){.buffer = renderer->light_buffer, .before = renderer->light_state, .after = copy};
    before[barrier_count++] = (NriBufferBarrierDesc){.buffer = renderer->sdf.models, .before = renderer->sdf.models_state, .after = copy};
    before[barrier_count++] = (NriBufferBarrierDesc){.buffer = renderer->frame_buffer, .before = renderer->frame_state, .after = copy};
    before[barrier_count++] = (NriBufferBarrierDesc){.buffer = renderer->pass_constants_buffer, .before = renderer->pass_constants_state, .after = copy};
    before[barrier_count++] = (NriBufferBarrierDesc){.buffer = renderer->radiance_constants_buffer, .before = renderer->radiance_constants_state, .after = copy};
    if (upload_grid) {
        before[barrier_count++] = (NriBufferBarrierDesc){.buffer = renderer->dynamic_grid.cells, .before = renderer->dynamic_grid.state, .after = copy};
        before[barrier_count++] = (NriBufferBarrierDesc){.buffer = renderer->dynamic_grid.indices, .before = renderer->dynamic_grid.state, .after = copy};
    }
    renderer->gpu->core.CmdBarrier(command_buffer, &(NriBarrierDesc){.buffers = before, .bufferNum = barrier_count});
    renderer->gpu->streamer_api.CmdCopyStreamedData(command_buffer, renderer->gpu->streamer, batch);

    NriBufferBarrierDesc after[8];
    barrier_count = 0u;
    after[barrier_count++] = (NriBufferBarrierDesc){.buffer = renderer->object_buffer, .before = copy, .after = object_read};
    after[barrier_count++] = (NriBufferBarrierDesc){.buffer = renderer->light_buffer, .before = copy, .after = compute_read};
    after[barrier_count++] = (NriBufferBarrierDesc){.buffer = renderer->sdf.models, .before = copy, .after = compute_read};
    after[barrier_count++] = (NriBufferBarrierDesc){.buffer = renderer->frame_buffer, .before = copy, .after = frame_read};
    after[barrier_count++] = (NriBufferBarrierDesc){.buffer = renderer->pass_constants_buffer, .before = copy, .after = pass_read};
    after[barrier_count++] = (NriBufferBarrierDesc){.buffer = renderer->radiance_constants_buffer, .before = copy, .after = radiance_read};
    if (upload_grid) {
        after[barrier_count++] = (NriBufferBarrierDesc){.buffer = renderer->dynamic_grid.cells, .before = copy, .after = compute_read};
        after[barrier_count++] = (NriBufferBarrierDesc){.buffer = renderer->dynamic_grid.indices, .before = copy, .after = compute_read};
    }
    renderer->gpu->core.CmdBarrier(command_buffer, &(NriBarrierDesc){.buffers = after, .bufferNum = barrier_count});

    renderer->object_state = object_read;
    renderer->light_state = compute_read;
    renderer->sdf.models_state = compute_read;
    renderer->frame_state = frame_read;
    renderer->pass_constants_state = pass_read;
    renderer->radiance_constants_state = radiance_read;
    if (upload_grid) {
        renderer->dynamic_grid.state = compute_read;
        renderer->dynamic_grid.dirty = false;
    }
    return true;
}

static void transition_gbuffer_for_render''',
    'stream dynamic grid and radiance constants'
)

# Insert dynamic trace between screen tracing and static-local tracing.
r = replace_once(
    r,
    '''    bind_wavefront(renderer, command_buffer, renderer->wavefront_local_pipeline);''',
    '''    if (renderer->radiance_constants.feature_flags[0] & RADIANCE_FEATURE_DYNAMIC_GRID) {\n        bind_wavefront(renderer, command_buffer, renderer->wavefront_dynamic_pipeline);\n        renderer->gpu->core.CmdDispatch(command_buffer, &(NriDispatchDesc){.workGroupNumX = ray_groups, .workGroupNumY = 1, .workGroupNumZ = 1});\n        barrier_wavefront_buffers(renderer, command_buffer, storage);\n    }\n\n    bind_wavefront(renderer, command_buffer, renderer->wavefront_local_pipeline);''',
    'dynamic trace dispatch ordering'
)

r = replace_once(
    r,
    '''            renderer->wavefront_screen_pipeline,\n            renderer->wavefront_local_pipeline,''',
    '''            renderer->wavefront_screen_pipeline,\n            renderer->wavefront_dynamic_pipeline,\n            renderer->wavefront_local_pipeline,''',
    'destroy dynamic pipeline'
)

r = replace_once(
    r,
    '''    if (!create_scene_resources(renderer, scene)) return false;\n    renderer->scene = scene;''',
    '''    if (!create_scene_resources(renderer, scene)) return false;\n    renderer->scene = scene;\n    if (!create_dynamic_grid(renderer) || !rebuild_dynamic_grid(renderer) || !update_radiance_constants(renderer)) {\n        renderer->scene = NULL;\n        destroy_scene_resources(renderer);\n        return false;\n    }''',
    'build grid when scene is installed'
)

r = replace_once(
    r,
    '''    if (!renderer || !renderer->gpu || !renderer->gpu->device || !renderer->scene) return false;\n\n    if (!update_scene_objects(renderer) || !refresh_emissive_sampling(renderer)) return false;''',
    '''    if (!renderer || !renderer->gpu || !renderer->gpu->device || !renderer->scene) return false;\n\n    if (!update_scene_objects(renderer) || !refresh_dynamic_grid(renderer) || !refresh_emissive_sampling(renderer)) return false;''',
    'refresh dynamic grid per changed revision'
)

p.write_text(r)

# -----------------------------------------------------------------------------
# shader.hlsl - make grid traversal start at the grid AABB and walk cells exactly.
# -----------------------------------------------------------------------------
p = Path('shader.hlsl')
s = p.read_text()
s = sub_once(
    s,
    r'''bool TraceDynamicGrid\(TraceRay ray, inout SurfaceHit best_hit\) \{.*?\n\}\n\n// -----------------------------------------------------------------------------\n// Sparse global SDF clipmaps\.''',
    '''bool TraceDynamicGrid(TraceRay ray, inout SurfaceHit best_hit) {
    if (!FeatureEnabled(RADIANCE_FEATURE_DYNAMIC_GRID) || Radiance.sdf_counts.y == 0u) return false;
    uint3 dim = Radiance.dynamic_grid.xyz;
    float cell_size = Radiance.dynamic_grid_origin_cell.w;
    if (any(dim == 0u) || cell_size <= 0.0f) return false;

    float3 grid_min = Radiance.dynamic_grid_origin_cell.xyz;
    float3 grid_max = grid_min + float3(dim) * cell_size;
    float t_enter, t_exit;
    if (!IntersectAABB(ray.origin_tmin.xyz, ray.direction_tmax.xyz, grid_min, grid_max, t_enter, t_exit)) return false;

    float t = max(max(t_enter, ray.origin_tmin.w), 0.0f);
    float limit = min(min(t_exit, ray.direction_tmax.w), best_hit.position_distance.w);
    if (t > limit) return false;

    float sample_t = min(t + max(cell_size * 1.0e-5f, 1.0e-5f), limit);
    float3 position = ray.origin_tmin.xyz + ray.direction_tmax.xyz * sample_t;
    int3 cell = int3(floor((position - grid_min) / cell_size));
    cell = clamp(cell, int3(0, 0, 0), int3(dim) - int3(1, 1, 1));

    int3 step_dir = int3(
        ray.direction_tmax.x >= 0.0f ? 1 : -1,
        ray.direction_tmax.y >= 0.0f ? 1 : -1,
        ray.direction_tmax.z >= 0.0f ? 1 : -1
    );
    float3 cell_min = grid_min + float3(cell) * cell_size;
    float3 boundary = float3(
        step_dir.x > 0 ? cell_min.x + cell_size : cell_min.x,
        step_dir.y > 0 ? cell_min.y + cell_size : cell_min.y,
        step_dir.z > 0 ? cell_min.z + cell_size : cell_min.z
    );
    float3 t_axis = float3(1.0e30f, 1.0e30f, 1.0e30f);
    float3 t_delta = float3(1.0e30f, 1.0e30f, 1.0e30f);
    if (abs(ray.direction_tmax.x) > 1.0e-8f) {
        t_axis.x = (boundary.x - ray.origin_tmin.x) / ray.direction_tmax.x;
        t_delta.x = cell_size / abs(ray.direction_tmax.x);
    }
    if (abs(ray.direction_tmax.y) > 1.0e-8f) {
        t_axis.y = (boundary.y - ray.origin_tmin.y) / ray.direction_tmax.y;
        t_delta.y = cell_size / abs(ray.direction_tmax.y);
    }
    if (abs(ray.direction_tmax.z) > 1.0e-8f) {
        t_axis.z = (boundary.z - ray.origin_tmin.z) / ray.direction_tmax.z;
        t_delta.z = cell_size / abs(ray.direction_tmax.z);
    }

    bool found = false;
    uint max_steps = min(max(Radiance.trace_limits.w, 1u), MAX_DYNAMIC_GRID_STEPS);
    [loop]
    for (uint step = 0u; step < max_steps; ++step) {
        uint3 ucell = uint3(cell);
        uint flat = Flatten3D(ucell, dim);
        if (flat >= Radiance.sdf_counts.y) break;
        GPUDynamicGridCell grid_cell = DynamicGridCells[flat];
        uint offset = grid_cell.range_flags.x;
        uint count = grid_cell.range_flags.y;
        for (uint i = 0u; i < count; ++i) {
            uint index = offset + i;
            if (index >= Radiance.dynamic_grid.w) break;
            uint model_index = DynamicGridIndices[index];
            SurfaceHit candidate;
            if (TraceLocalSDFModel(ray, model_index, best_hit.position_distance.w, candidate) && candidate.position_distance.w < best_hit.position_distance.w) {
                best_hit = candidate;
                found = true;
            }
        }

        float next_t = min(t_axis.x, min(t_axis.y, t_axis.z));
        if (found && best_hit.position_distance.w <= next_t) break;
        if (next_t > limit) break;

        if (t_axis.x <= t_axis.y && t_axis.x <= t_axis.z) {
            cell.x += step_dir.x;
            if (cell.x < 0 || cell.x >= (int)dim.x) break;
            t_axis.x += t_delta.x;
        } else if (t_axis.y <= t_axis.z) {
            cell.y += step_dir.y;
            if (cell.y < 0 || cell.y >= (int)dim.y) break;
            t_axis.y += t_delta.y;
        } else {
            cell.z += step_dir.z;
            if (cell.z < 0 || cell.z >= (int)dim.z) break;
            t_axis.z += t_delta.z;
        }
    }
    return found;
}

// -----------------------------------------------------------------------------
// Sparse global SDF clipmaps.''',
    'exact dynamic grid traversal'
)
p.write_text(s)
