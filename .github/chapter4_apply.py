from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected 1 match, got {count}")
    return text.replace(old, new, 1)


def replace_function(text: str, signature: str, replacement: str) -> str:
    start = text.find(signature)
    if start < 0:
        raise SystemExit(f"function not found: {signature}")
    brace = text.find("{", start)
    if brace < 0:
        raise SystemExit(f"function has no body: {signature}")
    depth = 0
    end = None
    for i in range(brace, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end is None:
        raise SystemExit(f"unterminated function: {signature}")
    return text[:start] + replacement.rstrip() + text[end:]


# -----------------------------------------------------------------------------
# game.h
# -----------------------------------------------------------------------------
game = Path("game.h")
text = game.read_text()

text = replace_once(
    text,
    """typedef struct SDF_VOLUME {\n    float *distance;\n    uint32_t resolution;\n    AABB bounds;\n} SDF_VOLUME;\n""",
    """typedef struct SDF_VOLUME {\n    float *distance;\n    uint32_t *surface_id;\n    uint32_t resolution;\n    AABB bounds;\n} SDF_VOLUME;\n""",
    "SDF_VOLUME surface identity",
)

text = replace_once(
    text,
    """typedef struct SDF_GPU_SCENE {\n    NriBuffer *models;\n    NriBuffer *voxels;\n    NriDescriptor *models_srv;\n    NriDescriptor *voxels_srv;\n    NriAccessStage models_state;\n    NriAccessStage voxels_state;\n""",
    """typedef struct SDF_GPU_SCENE {\n    NriBuffer *models;\n    NriBuffer *voxels;\n    NriBuffer *surface_ids;\n    NriDescriptor *models_srv;\n    NriDescriptor *voxels_srv;\n    NriDescriptor *surface_ids_srv;\n    NriAccessStage models_state;\n    NriAccessStage voxels_state;\n    NriAccessStage surface_ids_state;\n""",
    "SDF GPU surface identity",
)

fallback_struct = """
typedef struct RADIANCE_SCENE_FALLBACKS {
    NriBuffer *dynamic_grid_cells;
    NriBuffer *dynamic_grid_indices;
    NriBuffer *global_sdf_clipmaps;
    NriBuffer *global_sdf_page_table;
    NriBuffer *global_sdf_bricks;
    NriBuffer *global_sdf_surface_ids;

    NriDescriptor *dynamic_grid_cells_srv;
    NriDescriptor *dynamic_grid_indices_srv;
    NriDescriptor *global_sdf_clipmaps_srv;
    NriDescriptor *global_sdf_page_table_srv;
    NriDescriptor *global_sdf_bricks_srv;
    NriDescriptor *global_sdf_surface_ids_srv;

    NriAccessStage state;
} RADIANCE_SCENE_FALLBACKS;
"""

text = replace_once(
    text,
    """typedef struct GPU_DYNAMIC_GRID_CELL {\n""",
    fallback_struct + "\ntypedef struct GPU_DYNAMIC_GRID_CELL {\n",
    "radiance fallbacks struct",
)

text = replace_once(
    text,
    """    SDF_GPU_SCENE sdf;\n    SURFACE_CACHE surface_cache;\n    RADIANCE_SCENE_DATA radiance_scene;\n\n    NriDescriptorPool *descriptor_pool;\n""",
    """    SDF_GPU_SCENE sdf;\n    SURFACE_CACHE surface_cache;\n    RADIANCE_SCENE_DATA radiance_scene;\n    RADIANCE_SCENE_FALLBACKS radiance_fallbacks;\n\n    NriDescriptorPool *descriptor_pool;\n""",
    "renderer radiance fallbacks",
)

text = replace_once(
    text,
    """    NriDescriptorSet *trace_set;\n    NriDescriptorSet *radiance_scene_set;\n\n    NriPipelineLayout *gbuffer_layout;\n""",
    """    NriDescriptorSet *trace_set;\n    NriDescriptorSet *radiance_scene_set;\n    NriDescriptorSet *emissive_trace_set;\n    NriDescriptorSet *emissive_scene_set;\n    NriDescriptorSet *emissive_probe_set;\n\n    NriPipelineLayout *gbuffer_layout;\n""",
    "renderer emissive descriptor sets",
)

text = replace_once(
    text,
    """    NriPipelineLayout *trace_layout;\n    NriPipelineLayout *radiance_scene_layout;\n\n    NriPipeline *gbuffer_pipeline;\n""",
    """    NriPipelineLayout *trace_layout;\n    NriPipelineLayout *radiance_scene_layout;\n    NriPipelineLayout *emissive_layout;\n\n    NriPipeline *gbuffer_pipeline;\n""",
    "renderer emissive layout",
)

text = replace_once(
    text,
    """    NriPipeline *surface_cache_pipeline;\n    NriPipeline *screen_probes_pipeline;\n\n    uint32_t width;\n""",
    """    NriPipeline *surface_cache_pipeline;\n    NriPipeline *screen_probes_pipeline;\n    NriPipeline *emissive_pipeline;\n\n    uint32_t width;\n""",
    "renderer emissive pipeline",
)

game.write_text(text)


# -----------------------------------------------------------------------------
# sdf.c
# -----------------------------------------------------------------------------
sdf = Path("sdf.c")
text = sdf.read_text()

text = replace_once(
    text,
    """typedef struct SDF_TRIANGLE {\n    VEC3 a;\n    VEC3 b;\n    VEC3 c;\n    VEC3 centroid;\n    VEC3 min;\n    VEC3 max;\n} SDF_TRIANGLE;\n""",
    """typedef struct SDF_TRIANGLE {\n    VEC3 a;\n    VEC3 b;\n    VEC3 c;\n    VEC3 centroid;\n    VEC3 min;\n    VEC3 max;\n    uint32_t surface_id;\n} SDF_TRIANGLE;\n""",
    "SDF triangle surface id",
)

nearest = r'''static float nearest_distance_sq(const SDF_BUILD *build, uint32_t node_index, VEC3 p, float best, uint32_t *surface_id) {
    const SDF_BVH_NODE *node = &build->nodes[node_index];

    if (point_aabb_distance_sq(p, node->min, node->max) >= best) return best;

    if (node->count) {
        for (uint32_t i = node->first; i < node->first + node->count; ++i) {
            const SDF_TRIANGLE *triangle = &build->triangles[i];
            const float d = point_triangle_distance_sq(p, triangle);

            if (d < best) {
                best = d;
                *surface_id = triangle->surface_id;
            }
        }

        return best;
    }

    const float dl = point_aabb_distance_sq(p, build->nodes[node->left].min, build->nodes[node->left].max);
    const float dr = point_aabb_distance_sq(p, build->nodes[node->right].min, build->nodes[node->right].max);

    if (dl < dr) {
        best = nearest_distance_sq(build, node->left, p, best, surface_id);
        best = nearest_distance_sq(build, node->right, p, best, surface_id);
    } else {
        best = nearest_distance_sq(build, node->right, p, best, surface_id);
        best = nearest_distance_sq(build, node->left, p, best, surface_id);
    }

    return best;
}'''
text = replace_function(text, "static float nearest_distance_sq(", nearest)

text = replace_once(
    text,
    """        triangle->max = vmax3(triangle->a, vmax3(triangle->b, triangle->c));\n""",
    """        triangle->max = vmax3(triangle->a, vmax3(triangle->b, triangle->c));\n        triangle->surface_id = i;\n""",
    "SDF source face identity",
)

text = replace_once(
    text,
    """    float *distance = malloc((size_t)voxel_count64 * sizeof(*distance));\n\n    if (!distance) {\n        free(build.triangles);\n        free(build.nodes);\n\n        return false;\n    }\n""",
    """    float *distance = malloc((size_t)voxel_count64 * sizeof(*distance));\n    uint32_t *surface_id = malloc((size_t)voxel_count64 * sizeof(*surface_id));\n\n    if (!distance || !surface_id) {\n        free(distance);\n        free(surface_id);\n        free(build.triangles);\n        free(build.nodes);\n\n        return false;\n    }\n""",
    "SDF surface id allocation",
)

text = replace_once(
    text,
    """                distance[index] = sqrtf(nearest_distance_sq(&build, 0, p, FLT_MAX));\n""",
    """                uint32_t nearest_surface = UINT32_MAX;\n                distance[index] = sqrtf(nearest_distance_sq(&build, 0, p, FLT_MAX, &nearest_surface));\n                surface_id[index] = nearest_surface;\n""",
    "SDF voxel identity",
)

text = replace_once(
    text,
    """    volume->distance = distance;\n    volume->resolution = resolution;\n""",
    """    volume->distance = distance;\n    volume->surface_id = surface_id;\n    volume->resolution = resolution;\n""",
    "SDF volume identity result",
)

text = replace_once(
    text,
    """    free(volume->distance);\n    memset(volume, 0, sizeof(*volume));\n""",
    """    free(volume->distance);\n    free(volume->surface_id);\n    memset(volume, 0, sizeof(*volume));\n""",
    "SDF volume identity teardown",
)

sdf.write_text(text)


# -----------------------------------------------------------------------------
# render.c
# -----------------------------------------------------------------------------
render = Path("render.c")
text = render.read_text()

fallback_functions = r'''
static void destroy_radiance_scene_fallbacks(RENDERER *renderer) {
    if (!renderer || !renderer->gpu) return;

    RADIANCE_SCENE_FALLBACKS *fallbacks = &renderer->radiance_fallbacks;

    if (fallbacks->dynamic_grid_cells_srv) renderer->gpu->core.DestroyDescriptor(fallbacks->dynamic_grid_cells_srv);
    if (fallbacks->dynamic_grid_indices_srv) renderer->gpu->core.DestroyDescriptor(fallbacks->dynamic_grid_indices_srv);
    if (fallbacks->global_sdf_clipmaps_srv) renderer->gpu->core.DestroyDescriptor(fallbacks->global_sdf_clipmaps_srv);
    if (fallbacks->global_sdf_page_table_srv) renderer->gpu->core.DestroyDescriptor(fallbacks->global_sdf_page_table_srv);
    if (fallbacks->global_sdf_bricks_srv) renderer->gpu->core.DestroyDescriptor(fallbacks->global_sdf_bricks_srv);
    if (fallbacks->global_sdf_surface_ids_srv) renderer->gpu->core.DestroyDescriptor(fallbacks->global_sdf_surface_ids_srv);

    if (fallbacks->dynamic_grid_cells) gpu_destroy_buffer(renderer->gpu, fallbacks->dynamic_grid_cells);
    if (fallbacks->dynamic_grid_indices) gpu_destroy_buffer(renderer->gpu, fallbacks->dynamic_grid_indices);
    if (fallbacks->global_sdf_clipmaps) gpu_destroy_buffer(renderer->gpu, fallbacks->global_sdf_clipmaps);
    if (fallbacks->global_sdf_page_table) gpu_destroy_buffer(renderer->gpu, fallbacks->global_sdf_page_table);
    if (fallbacks->global_sdf_bricks) gpu_destroy_buffer(renderer->gpu, fallbacks->global_sdf_bricks);
    if (fallbacks->global_sdf_surface_ids) gpu_destroy_buffer(renderer->gpu, fallbacks->global_sdf_surface_ids);

    memset(fallbacks, 0, sizeof(*fallbacks));
}

static bool create_radiance_fallback_buffer(RENDERER *renderer, uint32_t stride, NriBuffer **buffer, NriDescriptor **srv) {
    if (!stride || stride > sizeof(GPU_GLOBAL_SDF_CLIPMAP)) return false;

    const NriBufferDesc desc = {
        .size = stride,
        .structureStride = stride,
        .usage = NriBufferUsageBits_SHADER_RESOURCE
    };

    if (!gpu_create_buffer(renderer->gpu, &desc, NriMemoryLocation_DEVICE, buffer)) return false;
    if (!create_buffer_view(renderer, *buffer, NriBufferView_STRUCTURED_BUFFER, desc.size, stride, srv)) return false;

    const uint8_t zero[sizeof(GPU_GLOBAL_SDF_CLIPMAP)] = {0};
    const NriAccessStage read = {
        .access = NriAccessBits_SHADER_RESOURCE,
        .stages = NriStageBits_COMPUTE_SHADER
    };

    return gpu_upload_buffer(renderer->gpu, *buffer, zero, read);
}

static bool create_radiance_scene_fallbacks(RENDERER *renderer) {
    destroy_radiance_scene_fallbacks(renderer);

    RADIANCE_SCENE_FALLBACKS *fallbacks = &renderer->radiance_fallbacks;

    if (!create_radiance_fallback_buffer(renderer, sizeof(GPU_DYNAMIC_GRID_CELL), &fallbacks->dynamic_grid_cells, &fallbacks->dynamic_grid_cells_srv) ||
        !create_radiance_fallback_buffer(renderer, sizeof(uint32_t), &fallbacks->dynamic_grid_indices, &fallbacks->dynamic_grid_indices_srv) ||
        !create_radiance_fallback_buffer(renderer, sizeof(GPU_GLOBAL_SDF_CLIPMAP), &fallbacks->global_sdf_clipmaps, &fallbacks->global_sdf_clipmaps_srv) ||
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

static bool refresh_emissive_sampling(RENDERER *renderer) {
    RADIANCE_SCENE_DATA *scene = &renderer->radiance_scene;

    if (!scene->emissive_triangle_count) return true;
    if (!scene->cpu_emissive_triangles || !scene->cpu_triangles || !scene->emissive_triangles) return false;

    bool dirty = false;

    for (uint32_t i = 0; i < scene->emissive_triangle_count; ++i) {
        GPU_EMISSIVE_TRIANGLE *emitter = &scene->cpu_emissive_triangles[i];
        const uint32_t object_index = emitter->meta[2];

        if (object_index >= renderer->gpu_object_count) return false;

        if (emitter->meta[3] != renderer->cpu_objects[object_index].revision) {
            dirty = true;
            break;
        }
    }

    if (!dirty) return true;

    double total_weight = 0.0;

    for (uint32_t i = 0; i < scene->emissive_triangle_count; ++i) {
        GPU_EMISSIVE_TRIANGLE *emitter = &scene->cpu_emissive_triangles[i];
        const uint32_t triangle_id = emitter->meta[0];
        const uint32_t object_index = emitter->meta[2];

        if (triangle_id >= scene->triangle_count || object_index >= renderer->gpu_object_count) return false;

        const GPU_SCENE_TRIANGLE *triangle = &scene->cpu_triangles[triangle_id];
        const MAT4 world = renderer->cpu_objects[object_index].world;
        const VEC3 a = mat4_point(world, v3(triangle->p0[0], triangle->p0[1], triangle->p0[2]));
        const VEC3 b = mat4_point(world, v3(triangle->p1[0], triangle->p1[1], triangle->p1[2]));
        const VEC3 c = mat4_point(world, v3(triangle->p2[0], triangle->p2[1], triangle->p2[2]));
        const float area = triangle_area(a, b, c);
        const float luminance = emitter->radiance_area[0] * 0.2126f + emitter->radiance_area[1] * 0.7152f + emitter->radiance_area[2] * 0.0722f;
        const float weight = area * luminance;

        emitter->radiance_area[3] = area;
        emitter->sampling[2] = weight;
        emitter->meta[3] = renderer->cpu_objects[object_index].revision;
        total_weight += (double)weight;
    }

    if (total_weight <= 1.0e-12) return false;

    double cumulative = 0.0;

    for (uint32_t i = 0; i < scene->emissive_triangle_count; ++i) {
        GPU_EMISSIVE_TRIANGLE *emitter = &scene->cpu_emissive_triangles[i];
        const float probability = (float)((double)emitter->sampling[2] / total_weight);
        cumulative += probability;
        emitter->sampling[0] = i + 1u == scene->emissive_triangle_count ? 1.0f : (float)cumulative;
        emitter->sampling[1] = probability;
        emitter->sampling[3] = 0.0f;
    }

    const NriAccessStage read = {
        .access = NriAccessBits_SHADER_RESOURCE,
        .stages = NriStageBits_COMPUTE_SHADER
    };

    if (!gpu_upload_buffer(renderer->gpu, scene->emissive_triangles, scene->cpu_emissive_triangles, read)) return false;

    scene->emissive_triangles_state = read;
    return true;
}
'''

text = replace_once(
    text,
    """static void destroy_screen_trace(RENDERER *renderer) {\n""",
    fallback_functions + "\nstatic void destroy_screen_trace(RENDERER *renderer) {\n",
    "radiance fallbacks and emitter refresh",
)

new_destroy_sdf = r'''static void destroy_sdf_scene(RENDERER *renderer) {
    if (renderer->sdf.models_srv) renderer->gpu->core.DestroyDescriptor(renderer->sdf.models_srv);
    if (renderer->sdf.voxels_srv) renderer->gpu->core.DestroyDescriptor(renderer->sdf.voxels_srv);
    if (renderer->sdf.surface_ids_srv) renderer->gpu->core.DestroyDescriptor(renderer->sdf.surface_ids_srv);

    if (renderer->sdf.models) gpu_destroy_buffer(renderer->gpu, renderer->sdf.models);
    if (renderer->sdf.voxels) gpu_destroy_buffer(renderer->gpu, renderer->sdf.voxels);
    if (renderer->sdf.surface_ids) gpu_destroy_buffer(renderer->gpu, renderer->sdf.surface_ids);

    free(renderer->sdf.cpu_models);
    memset(&renderer->sdf, 0, sizeof(renderer->sdf));
}'''
text = replace_function(text, "static void destroy_sdf_scene(", new_destroy_sdf)

new_create_sdf = r'''static bool create_sdf_scene(RENDERER *renderer, SCENE *scene) {
    destroy_sdf_scene(renderer);

    uint32_t model_count = 0;

    for (uint32_t i = 0; i < scene->object_count; ++i)
        if (scene->objects[i].type == MODEL) ++model_count;

    if (!model_count) return false;

    SDF_VOLUME *volumes = calloc(model_count, sizeof(*volumes));
    GPU_SDF_MODEL *models = calloc(model_count, sizeof(*models));
    uint32_t *triangle_bases = calloc(model_count, sizeof(*triangle_bases));

    if (!volumes || !models || !triangle_bases) {
        free(volumes);
        free(models);
        free(triangle_bases);
        return false;
    }

    uint64_t total_voxels = 0;
    uint64_t triangle_cursor = 0;
    uint32_t model_index = 0;
    bool ok = true;

    for (uint32_t scene_index = 0; scene_index < scene->object_count; ++scene_index) {
        OBJECT *object = &scene->objects[scene_index];

        if (object->type != MODEL) continue;

        struct MODEL *model = object->data;

        if (!model || !model->geometry || !model->geometry->faces.buffer || model->geometry->faces.count > UINT32_MAX ||
            !sdf_build_volume(model->geometry, SDF_DEFAULT_RESOLUTION, &volumes[model_index])) {
            ok = false;
            break;
        }

        triangle_bases[model_index] = (uint32_t)triangle_cursor;
        triangle_cursor += model->geometry->faces.count;

        if (triangle_cursor > UINT32_MAX || triangle_cursor > renderer->radiance_scene.triangle_count) {
            ok = false;
            break;
        }

        const uint64_t volume_voxels =
            (uint64_t)volumes[model_index].resolution * volumes[model_index].resolution * volumes[model_index].resolution;

        if (total_voxels + volume_voxels > UINT32_MAX) {
            ok = false;
            break;
        }

        GPU_SDF_MODEL *gpu_model = &models[model_index];
        MAT4 world_to_local;

        if (!mat4_inverse(mat4_transform(object->transform), &world_to_local)) {
            ok = false;
            break;
        }

        gpu_model->world_to_local = world_to_local;
        gpu_model->bounds_min[0] = volumes[model_index].bounds.min.x;
        gpu_model->bounds_min[1] = volumes[model_index].bounds.min.y;
        gpu_model->bounds_min[2] = volumes[model_index].bounds.min.z;
        gpu_model->bounds_min[3] = 0.0f;
        gpu_model->bounds_max[0] = volumes[model_index].bounds.max.x;
        gpu_model->bounds_max[1] = volumes[model_index].bounds.max.y;
        gpu_model->bounds_max[2] = volumes[model_index].bounds.max.z;
        gpu_model->bounds_max[3] = 0.0f;
        gpu_model->voxel_offset = (uint32_t)total_voxels;
        gpu_model->resolution = volumes[model_index].resolution;
        gpu_model->object_id = scene_index + 1u;
        gpu_model->state = (uint32_t)object->state;
        gpu_model->revision = object->revision;
        gpu_model->padding[0] = 0u;
        gpu_model->padding[1] = 0u;
        gpu_model->padding[2] = 0u;

        total_voxels += volume_voxels;
        ++model_index;
    }

    if (ok && triangle_cursor != renderer->radiance_scene.triangle_count) ok = false;

    float *voxel_data = NULL;
    uint32_t *surface_id_data = NULL;

    if (ok) {
        voxel_data = malloc((size_t)total_voxels * sizeof(*voxel_data));
        surface_id_data = malloc((size_t)total_voxels * sizeof(*surface_id_data));

        if (!voxel_data || !surface_id_data) ok = false;
    }

    if (ok) {
        for (uint32_t i = 0; i < model_count && ok; ++i) {
            const uint64_t count = (uint64_t)volumes[i].resolution * volumes[i].resolution * volumes[i].resolution;
            const uint32_t offset = models[i].voxel_offset;

            memcpy(voxel_data + offset, volumes[i].distance, (size_t)count * sizeof(float));

            for (uint64_t j = 0; j < count; ++j) {
                const uint32_t local_surface = volumes[i].surface_id[j];

                if (local_surface == UINT32_MAX) {
                    surface_id_data[offset + j] = UINT32_MAX;
                    continue;
                }

                const uint64_t global_surface = (uint64_t)triangle_bases[i] + local_surface;

                if (global_surface >= renderer->radiance_scene.triangle_count) {
                    ok = false;
                    break;
                }

                surface_id_data[offset + j] = (uint32_t)global_surface;
            }
        }
    }

    if (ok) {
        const NriBufferDesc models_desc = {
            .size = (uint64_t)model_count * sizeof(GPU_SDF_MODEL),
            .structureStride = sizeof(GPU_SDF_MODEL),
            .usage = NriBufferUsageBits_SHADER_RESOURCE
        };

        const NriBufferDesc voxels_desc = {
            .size = total_voxels * sizeof(float),
            .structureStride = sizeof(float),
            .usage = NriBufferUsageBits_SHADER_RESOURCE
        };

        const NriBufferDesc surface_ids_desc = {
            .size = total_voxels * sizeof(uint32_t),
            .structureStride = sizeof(uint32_t),
            .usage = NriBufferUsageBits_SHADER_RESOURCE
        };

        ok = gpu_create_buffer(renderer->gpu, &models_desc, NriMemoryLocation_DEVICE, &renderer->sdf.models) &&
             gpu_create_buffer(renderer->gpu, &voxels_desc, NriMemoryLocation_DEVICE, &renderer->sdf.voxels) &&
             gpu_create_buffer(renderer->gpu, &surface_ids_desc, NriMemoryLocation_DEVICE, &renderer->sdf.surface_ids) &&
             create_buffer_view(renderer, renderer->sdf.models, NriBufferView_STRUCTURED_BUFFER, models_desc.size, sizeof(GPU_SDF_MODEL), &renderer->sdf.models_srv) &&
             create_buffer_view(renderer, renderer->sdf.voxels, NriBufferView_STRUCTURED_BUFFER, voxels_desc.size, sizeof(float), &renderer->sdf.voxels_srv) &&
             create_buffer_view(renderer, renderer->sdf.surface_ids, NriBufferView_STRUCTURED_BUFFER, surface_ids_desc.size, sizeof(uint32_t), &renderer->sdf.surface_ids_srv);

        if (ok) {
            const NriAccessStage read = {
                .access = NriAccessBits_SHADER_RESOURCE,
                .stages = NriStageBits_COMPUTE_SHADER
            };

            ok = gpu_upload_buffer(renderer->gpu, renderer->sdf.voxels, voxel_data, read) &&
                 gpu_upload_buffer(renderer->gpu, renderer->sdf.surface_ids, surface_id_data, read);

            renderer->sdf.voxels_state = read;
            renderer->sdf.surface_ids_state = read;
        }
    }

    for (uint32_t i = 0; i < model_count; ++i)
        sdf_free_volume(&volumes[i]);

    free(volumes);
    free(voxel_data);
    free(surface_id_data);
    free(triangle_bases);

    if (!ok) {
        free(models);
        destroy_sdf_scene(renderer);
        return false;
    }

    renderer->sdf.cpu_models = models;
    renderer->sdf.model_count = model_count;
    renderer->sdf.voxel_count = (uint32_t)total_voxels;
    renderer->sdf.clipmap_count = 0u;
    renderer->sdf.clipmap_resolution = 0u;
    renderer->sdf.clipmap_base_extent = 0.0f;

    SDL_Log("SDF scene: %u models, %u voxels with surface IDs", model_count, renderer->sdf.voxel_count);
    return true;
}'''
text = replace_function(text, "static bool create_sdf_scene(", new_create_sdf)

text = replace_once(
    text,
    """    constants.feature_flags[0] = 0u;\n""",
    """    constants.feature_flags[0] = renderer->radiance_scene.emissive_triangle_count ? RADIANCE_FEATURE_EMISSIVE : 0u;\n""",
    "enable emissive feature",
)

# Extend the permanent scene layout with the emissive dispatch layout.
text = replace_once(
    text,
    """    return renderer->gpu->core.CreatePipelineLayout(renderer->gpu->device, &radiance_scene_layout, &renderer->radiance_scene_layout) == NriResult_SUCCESS;\n}\n\nstatic bool create_descriptor_pool(RENDERER *renderer) {\n""",
    """    if (renderer->gpu->core.CreatePipelineLayout(renderer->gpu->device, &radiance_scene_layout, &renderer->radiance_scene_layout) != NriResult_SUCCESS) return false;\n\n    const NriDescriptorRangeDesc emissive_probe_range = {\n        .baseRegisterIndex = 7,\n        .descriptorNum = 1,\n        .descriptorType = NriDescriptorType_STORAGE_TEXTURE,\n        .shaderStages = NriStageBits_COMPUTE_SHADER\n    };\n\n    const NriDescriptorSetDesc emissive_probe_set = {\n        .registerSpace = 7,\n        .ranges = &emissive_probe_range,\n        .rangeNum = 1\n    };\n\n    const NriDescriptorSetDesc emissive_sets[] = {trace_set, radiance_scene_set, emissive_probe_set};\n\n    const NriPipelineLayoutDesc emissive_layout = {\n        .descriptorSets = emissive_sets,\n        .descriptorSetNum = 3,\n        .shaderStages = NriStageBits_COMPUTE_SHADER,\n        .flags = NriPipelineLayoutBits_IGNORE_GLOBAL_SPIRV_OFFSETS\n    };\n\n    return renderer->gpu->core.CreatePipelineLayout(renderer->gpu->device, &emissive_layout, &renderer->emissive_layout) == NriResult_SUCCESS;\n}\n\nstatic bool create_descriptor_pool(RENDERER *renderer) {\n""",
    "emissive pipeline layout",
)

new_pool = r'''static bool create_descriptor_pool(RENDERER *renderer) {
    const NriDescriptorPoolDesc desc = {
        .descriptorSetMaxNum = 7 + HZB_MAX_MIPS,
        .constantBufferMaxNum = 8,
        .textureMaxNum = 64,
        .storageTextureMaxNum = HZB_MAX_MIPS + 9,
        .structuredBufferMaxNum = 40,
        .storageStructuredBufferMaxNum = 16
    };

    if (renderer->gpu->core.CreateDescriptorPool(renderer->gpu->device, &desc, &renderer->descriptor_pool) != NriResult_SUCCESS) return false;

    return renderer->gpu->core.AllocateDescriptorSets(renderer->descriptor_pool, renderer->gbuffer_layout, 0, &renderer->gbuffer_set, 1, 0) == NriResult_SUCCESS &&
           renderer->gpu->core.AllocateDescriptorSets(renderer->descriptor_pool, renderer->present_layout, 0, &renderer->present_set, 1, 0) == NriResult_SUCCESS &&
           renderer->gpu->core.AllocateDescriptorSets(renderer->descriptor_pool, renderer->hzb_layout, 0, renderer->hzb_sets, HZB_MAX_MIPS, 0) == NriResult_SUCCESS &&
           renderer->gpu->core.AllocateDescriptorSets(renderer->descriptor_pool, renderer->trace_layout, 0, &renderer->trace_set, 1, 0) == NriResult_SUCCESS &&
           renderer->gpu->core.AllocateDescriptorSets(renderer->descriptor_pool, renderer->radiance_scene_layout, 0, &renderer->radiance_scene_set, 1, 0) == NriResult_SUCCESS &&
           renderer->gpu->core.AllocateDescriptorSets(renderer->descriptor_pool, renderer->emissive_layout, 0, &renderer->emissive_trace_set, 1, 0) == NriResult_SUCCESS &&
           renderer->gpu->core.AllocateDescriptorSets(renderer->descriptor_pool, renderer->emissive_layout, 1, &renderer->emissive_scene_set, 1, 0) == NriResult_SUCCESS &&
           renderer->gpu->core.AllocateDescriptorSets(renderer->descriptor_pool, renderer->emissive_layout, 2, &renderer->emissive_probe_set, 1, 0) == NriResult_SUCCESS;
}'''
text = replace_function(text, "static bool create_descriptor_pool(", new_pool)

text = replace_once(
    text,
    """           create_compute_pipeline(renderer, \"build/shaders/sdf_trace.cs.spv\", renderer->trace_layout, &renderer->sdf_trace_pipeline) &&\n           create_compute_pipeline(renderer, \"build/shaders/screen_probes.cs.spv\", renderer->trace_layout, &renderer->screen_probes_pipeline);\n""",
    """           create_compute_pipeline(renderer, \"build/shaders/sdf_trace.cs.spv\", renderer->trace_layout, &renderer->sdf_trace_pipeline) &&\n           create_compute_pipeline(renderer, \"build/shaders/screen_probes.cs.spv\", renderer->trace_layout, &renderer->screen_probes_pipeline) &&\n           create_compute_pipeline(renderer, \"build/shaders/radiance_emissive.cs.spv\", renderer->emissive_layout, &renderer->emissive_pipeline);\n""",
    "emissive compute pipeline",
)

new_trace_update = r'''static bool update_trace_descriptor_set(RENDERER *renderer, NriDescriptorSet *descriptor_set) {
    if (!descriptor_set || !renderer->depth.srv || !renderer->normal_roughness.srv || !renderer->albedo_metallic.srv || !renderer->emissive.srv || !renderer->hzb.srv ||
        !renderer->object_id.srv || !renderer->direct_radiance.srv || !renderer->frame_srv || !renderer->screen_trace.uav || !renderer->trace_hits.uav ||
        !renderer->miss_queue.rays_uav || !renderer->miss_queue.count_uav || !renderer->miss_queue.dispatch_args_uav || !renderer->sdf.models_srv || !renderer->sdf.voxels_srv ||
        !renderer->light_srv || !renderer->direct_radiance.uav || !renderer->surface_cache.keys_uav || !renderer->surface_cache.entries_uav || !renderer->screen_probes.uav ||
        !renderer->screen_probe_radiance.uav) {
        return false;
    }

    const NriDescriptor *textures[] = {
        renderer->depth.srv,
        renderer->normal_roughness.srv,
        renderer->albedo_metallic.srv,
        renderer->emissive.srv,
        renderer->hzb.srv,
        renderer->object_id.srv,
        renderer->direct_radiance.srv
    };

    const NriDescriptor *frame[] = {renderer->frame_srv};
    const NriDescriptor *screen_output[] = {renderer->screen_trace.uav};
    const NriDescriptor *trace_storage[] = {renderer->trace_hits.uav, renderer->miss_queue.rays_uav, renderer->miss_queue.count_uav, renderer->miss_queue.dispatch_args_uav};
    const NriDescriptor *scene[] = {renderer->sdf.models_srv, renderer->sdf.voxels_srv, renderer->light_srv};
    const NriDescriptor *radiance_output[] = {renderer->direct_radiance.uav};
    const NriDescriptor *cache[] = {renderer->surface_cache.keys_uav, renderer->surface_cache.entries_uav};
    const NriDescriptor *probes[] = {renderer->screen_probe_radiance.uav, renderer->screen_probes.uav};

    const NriUpdateDescriptorRangeDesc updates[] = {
        {.descriptorSet = descriptor_set, .rangeIndex = 0, .descriptors = textures, .descriptorNum = 7},
        {.descriptorSet = descriptor_set, .rangeIndex = 1, .descriptors = frame, .descriptorNum = 1},
        {.descriptorSet = descriptor_set, .rangeIndex = 2, .descriptors = screen_output, .descriptorNum = 1},
        {.descriptorSet = descriptor_set, .rangeIndex = 3, .descriptors = trace_storage, .descriptorNum = 4},
        {.descriptorSet = descriptor_set, .rangeIndex = 4, .descriptors = scene, .descriptorNum = 3},
        {.descriptorSet = descriptor_set, .rangeIndex = 5, .descriptors = radiance_output, .descriptorNum = 1},
        {.descriptorSet = descriptor_set, .rangeIndex = 6, .descriptors = cache, .descriptorNum = 2},
        {.descriptorSet = descriptor_set, .rangeIndex = 7, .descriptors = probes, .descriptorNum = 2}
    };

    renderer->gpu->core.UpdateDescriptorRanges(updates, sizeof(updates) / sizeof(updates[0]));
    return true;
}

static void update_trace_descriptors(RENDERER *renderer) {
    if (renderer->trace_set) update_trace_descriptor_set(renderer, renderer->trace_set);
    if (renderer->emissive_trace_set) update_trace_descriptor_set(renderer, renderer->emissive_trace_set);
}'''
text = replace_function(text, "static void update_trace_descriptors(", new_trace_update)

new_scene_update = r'''static bool update_radiance_scene_descriptor_set(RENDERER *renderer, NriDescriptorSet *descriptor_set) {
    if (!renderer || !descriptor_set || !renderer->radiance_constants_srv || !renderer->pass_constants_srv || !renderer->object_srv || !renderer->material_srv ||
        !renderer->radiance_scene.triangles_srv || !renderer->radiance_scene.emissive_triangles_srv || !renderer->sdf.models_srv || !renderer->sdf.voxels_srv ||
        !renderer->sdf.surface_ids_srv || !renderer->radiance_fallbacks.dynamic_grid_cells_srv || !renderer->radiance_fallbacks.dynamic_grid_indices_srv ||
        !renderer->radiance_fallbacks.global_sdf_clipmaps_srv || !renderer->radiance_fallbacks.global_sdf_page_table_srv || !renderer->radiance_fallbacks.global_sdf_bricks_srv ||
        !renderer->radiance_fallbacks.global_sdf_surface_ids_srv || !renderer->light_srv || !renderer->material_id.srv || !renderer->primitive_id.srv) {
        return false;
    }

    const NriDescriptor *constants[] = {renderer->radiance_constants_srv, renderer->pass_constants_srv};
    const NriDescriptor *scene_core[] = {
        renderer->object_srv,
        renderer->material_srv,
        renderer->radiance_scene.triangles_srv,
        renderer->radiance_scene.emissive_triangles_srv,
        renderer->sdf.models_srv,
        renderer->sdf.voxels_srv
    };
    const NriDescriptor *future_scene[] = {
        renderer->sdf.surface_ids_srv,
        renderer->radiance_fallbacks.dynamic_grid_cells_srv,
        renderer->radiance_fallbacks.dynamic_grid_indices_srv,
        renderer->radiance_fallbacks.global_sdf_clipmaps_srv,
        renderer->radiance_fallbacks.global_sdf_page_table_srv,
        renderer->radiance_fallbacks.global_sdf_bricks_srv,
        renderer->radiance_fallbacks.global_sdf_surface_ids_srv
    };
    const NriDescriptor *lights[] = {renderer->light_srv};
    const NriDescriptor *identity_textures[] = {renderer->material_id.srv, renderer->primitive_id.srv};

    const NriUpdateDescriptorRangeDesc updates[] = {
        {.descriptorSet = descriptor_set, .rangeIndex = 0, .descriptors = constants, .descriptorNum = 2},
        {.descriptorSet = descriptor_set, .rangeIndex = 1, .descriptors = scene_core, .descriptorNum = 6},
        {.descriptorSet = descriptor_set, .rangeIndex = 2, .descriptors = future_scene, .descriptorNum = 7},
        {.descriptorSet = descriptor_set, .rangeIndex = 3, .descriptors = lights, .descriptorNum = 1},
        {.descriptorSet = descriptor_set, .rangeIndex = 4, .descriptors = identity_textures, .descriptorNum = 2}
    };

    renderer->gpu->core.UpdateDescriptorRanges(updates, sizeof(updates) / sizeof(updates[0]));
    return true;
}

static bool update_radiance_scene_descriptors(RENDERER *renderer) {
    if (!update_radiance_scene_descriptor_set(renderer, renderer->radiance_scene_set)) return false;
    if (renderer->emissive_scene_set && !update_radiance_scene_descriptor_set(renderer, renderer->emissive_scene_set)) return false;
    return true;
}

static bool update_emissive_probe_descriptors(RENDERER *renderer) {
    if (!renderer || !renderer->emissive_probe_set || !renderer->screen_probes.uav) return false;

    const NriDescriptor *probe[] = {renderer->screen_probes.uav};
    const NriUpdateDescriptorRangeDesc update = {
        .descriptorSet = renderer->emissive_probe_set,
        .rangeIndex = 0,
        .descriptors = probe,
        .descriptorNum = 1
    };

    renderer->gpu->core.UpdateDescriptorRanges(&update, 1);
    return true;
}'''
text = replace_function(text, "static bool update_radiance_scene_descriptors(", new_scene_update)

text = replace_once(
    text,
    """    update_hzb_descriptors(renderer);\n    update_present_descriptors(renderer);\n    update_trace_descriptors(renderer);\n\n    if (renderer->radiance_scene.triangles_srv && !update_radiance_scene_descriptors(renderer)) return false;\n""",
    """    update_hzb_descriptors(renderer);\n    update_present_descriptors(renderer);\n    update_trace_descriptors(renderer);\n\n    if (!update_emissive_probe_descriptors(renderer)) return false;\n    if (renderer->radiance_scene.triangles_srv && !update_radiance_scene_descriptors(renderer)) return false;\n""",
    "emissive probe descriptors on resize",
)

# Stream PassConstants every frame together with existing dynamic data.
text = replace_once(
    text,
    """static bool stream_dynamic_data(RENDERER *renderer, NriCommandBuffer *command_buffer, const FRAME_CONSTANTS *frame) {\n    NriStreamerCopyBatch batch = renderer->gpu->streamer_api.BeginStreamerCopyBatch(renderer->gpu->streamer);\n""",
    """static bool stream_dynamic_data(RENDERER *renderer, NriCommandBuffer *command_buffer, const FRAME_CONSTANTS *frame) {\n    memset(&renderer->pass_constants, 0, sizeof(renderer->pass_constants));\n    renderer->pass_constants.dispatch[0] = renderer->radiance_constants.feature_flags[1];\n    renderer->pass_constants.dimensions[0] = renderer->screen_probes.width;\n    renderer->pass_constants.dimensions[1] = renderer->screen_probes.height;\n\n    NriStreamerCopyBatch batch = renderer->gpu->streamer_api.BeginStreamerCopyBatch(renderer->gpu->streamer);\n""",
    "pass constants CPU update",
)

text = replace_once(
    text,
    """    const NriDataSize frame_data = {\n        .data = frame,\n        .size = sizeof(*frame)\n    };\n\n    const NriStreamBufferDataDesc uploads[] = {\n""",
    """    const NriDataSize frame_data = {\n        .data = frame,\n        .size = sizeof(*frame)\n    };\n\n    const NriDataSize pass_data = {\n        .data = &renderer->pass_constants,\n        .size = sizeof(renderer->pass_constants)\n    };\n\n    const NriStreamBufferDataDesc uploads[] = {\n""",
    "pass constants stream data",
)

text = replace_once(
    text,
    """        {\n            .dataChunks = &frame_data,\n            .dataChunkNum = 1,\n            .placementAlignment = 16,\n            .copyBatch = batch,\n            .dstBuffer = renderer->frame_buffer\n        }\n    };\n\n    for (uint32_t i = 0; i < 4; ++i) {\n""",
    """        {\n            .dataChunks = &frame_data,\n            .dataChunkNum = 1,\n            .placementAlignment = 16,\n            .copyBatch = batch,\n            .dstBuffer = renderer->frame_buffer\n        },\n        {\n            .dataChunks = &pass_data,\n            .dataChunkNum = 1,\n            .placementAlignment = 16,\n            .copyBatch = batch,\n            .dstBuffer = renderer->pass_constants_buffer\n        }\n    };\n\n    for (uint32_t i = 0; i < 5; ++i) {\n""",
    "pass constants stream upload",
)

text = replace_once(
    text,
    """        {\n            .buffer = renderer->frame_buffer,\n            .before = renderer->frame_state,\n            .after = copy\n        }\n    };\n\n    renderer->gpu->core.CmdBarrier(command_buffer, &(NriBarrierDesc){\n        .buffers = before,\n        .bufferNum = 4\n""",
    """        {\n            .buffer = renderer->frame_buffer,\n            .before = renderer->frame_state,\n            .after = copy\n        },\n        {\n            .buffer = renderer->pass_constants_buffer,\n            .before = renderer->pass_constants_state,\n            .after = copy\n        }\n    };\n\n    renderer->gpu->core.CmdBarrier(command_buffer, &(NriBarrierDesc){\n        .buffers = before,\n        .bufferNum = 5\n""",
    "pass constants pre-copy barrier",
)

text = replace_once(
    text,
    """    const NriAccessStage frame_read = {\n        .access = NriAccessBits_CONSTANT_BUFFER,\n        .stages = NriStageBits_VERTEX_SHADER | NriStageBits_COMPUTE_SHADER\n    };\n\n    const NriBufferBarrierDesc after[] = {\n""",
    """    const NriAccessStage frame_read = {\n        .access = NriAccessBits_CONSTANT_BUFFER,\n        .stages = NriStageBits_VERTEX_SHADER | NriStageBits_COMPUTE_SHADER\n    };\n\n    const NriAccessStage pass_read = {\n        .access = NriAccessBits_CONSTANT_BUFFER,\n        .stages = NriStageBits_COMPUTE_SHADER\n    };\n\n    const NriBufferBarrierDesc after[] = {\n""",
    "pass constants read state",
)

text = replace_once(
    text,
    """        {\n            .buffer = renderer->frame_buffer,\n            .before = copy,\n            .after = frame_read\n        }\n    };\n\n    renderer->gpu->core.CmdBarrier(command_buffer, &(NriBarrierDesc){\n        .buffers = after,\n        .bufferNum = 4\n""",
    """        {\n            .buffer = renderer->frame_buffer,\n            .before = copy,\n            .after = frame_read\n        },\n        {\n            .buffer = renderer->pass_constants_buffer,\n            .before = copy,\n            .after = pass_read\n        }\n    };\n\n    renderer->gpu->core.CmdBarrier(command_buffer, &(NriBarrierDesc){\n        .buffers = after,\n        .bufferNum = 5\n""",
    "pass constants post-copy barrier",
)

text = replace_once(
    text,
    """    renderer->sdf.models_state = compute_read;\n    renderer->frame_state = frame_read;\n\n    return true;\n}\n""",
    """    renderer->sdf.models_state = compute_read;\n    renderer->frame_state = frame_read;\n    renderer->pass_constants_state = pass_read;\n\n    return true;\n}\n""",
    "pass constants state tracking",
)

emissive_build = r'''
static void bind_emissive(RENDERER *renderer, NriCommandBuffer *command_buffer) {
    renderer->gpu->core.CmdSetPipelineLayout(command_buffer, NriBindPoint_COMPUTE, renderer->emissive_layout);
    renderer->gpu->core.CmdSetPipeline(command_buffer, renderer->emissive_pipeline);

    const NriSetDescriptorSetDesc sets[] = {
        {.setIndex = 0, .descriptorSet = renderer->emissive_trace_set, .bindPoint = NriBindPoint_COMPUTE},
        {.setIndex = 1, .descriptorSet = renderer->emissive_scene_set, .bindPoint = NriBindPoint_COMPUTE},
        {.setIndex = 2, .descriptorSet = renderer->emissive_probe_set, .bindPoint = NriBindPoint_COMPUTE}
    };

    for (uint32_t i = 0; i < sizeof(sets) / sizeof(sets[0]); ++i)
        renderer->gpu->core.CmdSetDescriptorSet(command_buffer, &sets[i]);
}

static void build_emissive_gather(RENDERER *renderer, NriCommandBuffer *command_buffer) {
    if (!renderer->radiance_scene.emissive_triangle_count ||
        !(renderer->radiance_constants.feature_flags[0] & RADIANCE_FEATURE_EMISSIVE)) {
        return;
    }

    const NriAccessLayoutStage storage = {
        .access = NriAccessBits_SHADER_RESOURCE_STORAGE,
        .layout = NriLayout_SHADER_RESOURCE_STORAGE,
        .stages = NriStageBits_COMPUTE_SHADER
    };

    const NriTextureBarrierDesc to_storage = {
        .texture = renderer->screen_probes.texture,
        .before = renderer->screen_probes.state,
        .after = storage,
        .mipNum = 1,
        .layerNum = 1,
        .planes = NriPlaneBits_COLOR
    };

    renderer->gpu->core.CmdBarrier(command_buffer, &(NriBarrierDesc){
        .textures = &to_storage,
        .textureNum = 1
    });
    renderer->screen_probes.state = storage;

    bind_emissive(renderer, command_buffer);
    renderer->gpu->core.CmdDispatch(
        command_buffer,
        &(NriDispatchDesc){
            .workGroupNumX = (renderer->screen_probes.width + 7u) / 8u,
            .workGroupNumY = (renderer->screen_probes.height + 7u) / 8u,
            .workGroupNumZ = 1
        }
    );

    const NriAccessLayoutStage read = {
        .access = NriAccessBits_SHADER_RESOURCE,
        .layout = NriLayout_SHADER_RESOURCE,
        .stages = NriStageBits_FRAGMENT_SHADER
    };

    const NriTextureBarrierDesc to_read = {
        .texture = renderer->screen_probes.texture,
        .before = storage,
        .after = read,
        .mipNum = 1,
        .layerNum = 1,
        .planes = NriPlaneBits_COLOR
    };

    renderer->gpu->core.CmdBarrier(command_buffer, &(NriBarrierDesc){
        .textures = &to_read,
        .textureNum = 1
    });
    renderer->screen_probes.state = read;
}
'''
text = replace_once(
    text,
    """static void set_fullscreen_view(RENDERER *renderer, NriCommandBuffer *command_buffer) {\n""",
    emissive_build + "\nstatic void set_fullscreen_view(RENDERER *renderer, NriCommandBuffer *command_buffer) {\n",
    "emissive gather dispatch",
)

text = replace_once(
    text,
    """    if (!create_pipeline_layouts(renderer) || !create_descriptor_pool(renderer) || !create_frame_buffer(renderer) || !create_radiance_constant_buffers(renderer) ||\n        !create_pipelines(renderer) || !create_surface_cache(renderer) || !create_size_dependent_resources(renderer, gpu->swapchain_width, gpu->swapchain_height)) {\n""",
    """    if (!create_pipeline_layouts(renderer) || !create_descriptor_pool(renderer) || !create_frame_buffer(renderer) || !create_radiance_constant_buffers(renderer) ||\n        !create_radiance_scene_fallbacks(renderer) || !create_pipelines(renderer) || !create_surface_cache(renderer) ||\n        !create_size_dependent_resources(renderer, gpu->swapchain_width, gpu->swapchain_height)) {\n""",
    "create radiance fallbacks",
)

text = replace_once(
    text,
    """            renderer->surface_cache_pipeline,\n            renderer->screen_probes_pipeline\n        };\n""",
    """            renderer->surface_cache_pipeline,\n            renderer->screen_probes_pipeline,\n            renderer->emissive_pipeline\n        };\n""",
    "destroy emissive pipeline",
)

text = replace_once(
    text,
    """        destroy_scene_resources(renderer);\n        destroy_trace_queue(renderer);\n""",
    """        destroy_scene_resources(renderer);\n        destroy_radiance_scene_fallbacks(renderer);\n        destroy_trace_queue(renderer);\n""",
    "destroy radiance fallbacks",
)

text = replace_once(
    text,
    """        if (renderer->radiance_scene_layout) renderer->gpu->core.DestroyPipelineLayout(renderer->radiance_scene_layout);\n""",
    """        if (renderer->radiance_scene_layout) renderer->gpu->core.DestroyPipelineLayout(renderer->radiance_scene_layout);\n\n        if (renderer->emissive_layout) renderer->gpu->core.DestroyPipelineLayout(renderer->emissive_layout);\n""",
    "destroy emissive layout",
)

# Refresh emitter CDF before a frame command buffer is opened, then dispatch after legacy probe resolve.
text = replace_once(
    text,
    """bool renderer_frame(RENDERER *renderer) {\n    if (!renderer || !renderer->gpu || !renderer->gpu->device || !renderer->scene) return false;\n\n    NriCommandBuffer *command_buffer = NULL;\n""",
    """bool renderer_frame(RENDERER *renderer) {\n    if (!renderer || !renderer->gpu || !renderer->gpu->device || !renderer->scene) return false;\n\n    if (!update_scene_objects(renderer) || !refresh_emissive_sampling(renderer)) return false;\n    update_orbit_camera(renderer);\n\n    NriCommandBuffer *command_buffer = NULL;\n""",
    "refresh emissive sampling before frame",
)

text = replace_once(
    text,
    """    if (!update_scene_objects(renderer)) return false;\n    update_orbit_camera(renderer);\n\n    MAT4 view_projection;\n""",
    """    MAT4 view_projection;\n""",
    "remove duplicate dynamic scene update",
)

text = replace_once(
    text,
    """    finish_screen_trace(renderer, command_buffer);\n    build_screen_probes(renderer, command_buffer);\n    record_present_pass(renderer, command_buffer, swapchain_index);\n""",
    """    finish_screen_trace(renderer, command_buffer);\n    build_screen_probes(renderer, command_buffer);\n    build_emissive_gather(renderer, command_buffer);\n    record_present_pass(renderer, command_buffer, swapchain_index);\n""",
    "emissive gather frame order",
)

render.write_text(text)
