from pathlib import Path


def replace_once(text, old, new, label):
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected 1 match, got {count}")
    return text.replace(old, new, 1)


# -----------------------------------------------------------------------------
# game.h
# -----------------------------------------------------------------------------
game = Path("game.h")
text = game.read_text()

text = replace_once(
    text,
    """    RENDER_TEXTURE emissive;\n    RENDER_TEXTURE velocity;\n    RENDER_TEXTURE object_id;\n\n    HZB hzb;\n""",
    """    RENDER_TEXTURE emissive;\n    RENDER_TEXTURE velocity;\n    RENDER_TEXTURE object_id;\n    RENDER_TEXTURE material_id;\n    RENDER_TEXTURE primitive_id;\n\n    HZB hzb;\n""",
    "game.h gbuffer identity textures",
)

game.write_text(text)


# -----------------------------------------------------------------------------
# render.c
# -----------------------------------------------------------------------------
render = Path("render.c")
text = render.read_text()

text = replace_once(
    text,
    """static void destroy_gbuffer(RENDERER *renderer) {\n    destroy_render_texture(renderer, &renderer->depth);\n    destroy_render_texture(renderer, &renderer->normal_roughness);\n    destroy_render_texture(renderer, &renderer->albedo_metallic);\n    destroy_render_texture(renderer, &renderer->emissive);\n    destroy_render_texture(renderer, &renderer->velocity);\n    destroy_render_texture(renderer, &renderer->object_id);\n}\n""",
    """static void destroy_gbuffer(RENDERER *renderer) {\n    destroy_render_texture(renderer, &renderer->depth);\n    destroy_render_texture(renderer, &renderer->normal_roughness);\n    destroy_render_texture(renderer, &renderer->albedo_metallic);\n    destroy_render_texture(renderer, &renderer->emissive);\n    destroy_render_texture(renderer, &renderer->velocity);\n    destroy_render_texture(renderer, &renderer->object_id);\n    destroy_render_texture(renderer, &renderer->material_id);\n    destroy_render_texture(renderer, &renderer->primitive_id);\n}\n""",
    "destroy_gbuffer",
)

text = replace_once(
    text,
    """           create_render_texture(\n               renderer,\n               &renderer->object_id,\n               NriFormat_R32_UINT,\n               NriTextureUsageBits_COLOR_ATTACHMENT | NriTextureUsageBits_SHADER_RESOURCE,\n               NriTextureView_COLOR_ATTACHMENT,\n               NriPlaneBits_COLOR\n           );\n}\n""",
    """           create_render_texture(\n               renderer,\n               &renderer->object_id,\n               NriFormat_R32_UINT,\n               NriTextureUsageBits_COLOR_ATTACHMENT | NriTextureUsageBits_SHADER_RESOURCE,\n               NriTextureView_COLOR_ATTACHMENT,\n               NriPlaneBits_COLOR\n           ) &&\n           create_render_texture(\n               renderer,\n               &renderer->material_id,\n               NriFormat_R32_UINT,\n               NriTextureUsageBits_COLOR_ATTACHMENT | NriTextureUsageBits_SHADER_RESOURCE,\n               NriTextureView_COLOR_ATTACHMENT,\n               NriPlaneBits_COLOR\n           ) &&\n           create_render_texture(\n               renderer,\n               &renderer->primitive_id,\n               NriFormat_R32_UINT,\n               NriTextureUsageBits_COLOR_ATTACHMENT | NriTextureUsageBits_SHADER_RESOURCE,\n               NriTextureView_COLOR_ATTACHMENT,\n               NriPlaneBits_COLOR\n           );\n}\n""",
    "create_gbuffer identity textures",
)

text = replace_once(
    text,
    '    if (!load_shader("build/shaders/gbuffer.vs.spv", &vs, &vs_size) || !load_shader("build/shaders/gbuffer.ps.spv", &ps, &ps_size)) {',
    '    if (!load_shader("build/shaders/gbuffer.vs.spv", &vs, &vs_size) || !load_shader("build/shaders/gbuffer_full.ps.spv", &ps, &ps_size)) {',
    "gbuffer full fragment shader",
)

text = replace_once(
    text,
    """        {\n            .format = NriFormat_RGBA16_SFLOAT,\n            .colorWriteMask = NriColorWriteBits_RGBA\n        }\n    };\n\n    const NriGraphicsPipelineDesc gbuffer = {\n""",
    """        {\n            .format = NriFormat_RGBA16_SFLOAT,\n            .colorWriteMask = NriColorWriteBits_RGBA\n        },\n        {\n            .format = NriFormat_R32_UINT,\n            .colorWriteMask = NriColorWriteBits_RGBA\n        },\n        {\n            .format = NriFormat_R32_UINT,\n            .colorWriteMask = NriColorWriteBits_RGBA\n        }\n    };\n\n    const NriGraphicsPipelineDesc gbuffer = {\n""",
    "gbuffer pipeline identity formats",
)

text = replace_once(
    text,
    """        .outputMerger = {\n            .colors = colors,\n            .colorNum = 5,\n            .depth = {\n""",
    """        .outputMerger = {\n            .colors = colors,\n            .colorNum = 7,\n            .depth = {\n""",
    "gbuffer pipeline color count",
)

old_update = """static bool update_radiance_scene_descriptors(RENDERER *renderer) {\n    if (!renderer || !renderer->radiance_scene_set || !renderer->radiance_constants_srv || !renderer->pass_constants_srv || !renderer->object_srv ||\n        !renderer->material_srv || !renderer->radiance_scene.triangles_srv || !renderer->radiance_scene.emissive_triangles_srv || !renderer->sdf.models_srv ||\n        !renderer->sdf.voxels_srv || !renderer->light_srv) {\n        return false;\n    }\n\n    const NriDescriptor *constants[] = {renderer->radiance_constants_srv, renderer->pass_constants_srv};\n    const NriDescriptor *scene_core[] = {\n        renderer->object_srv,\n        renderer->material_srv,\n        renderer->radiance_scene.triangles_srv,\n        renderer->radiance_scene.emissive_triangles_srv,\n        renderer->sdf.models_srv,\n        renderer->sdf.voxels_srv\n    };\n    const NriDescriptor *lights[] = {renderer->light_srv};\n\n    const NriUpdateDescriptorRangeDesc updates[] = {\n        {.descriptorSet = renderer->radiance_scene_set, .rangeIndex = 0, .descriptors = constants, .descriptorNum = 2},\n        {.descriptorSet = renderer->radiance_scene_set, .rangeIndex = 1, .descriptors = scene_core, .descriptorNum = 6},\n        {.descriptorSet = renderer->radiance_scene_set, .rangeIndex = 3, .descriptors = lights, .descriptorNum = 1}\n    };\n\n    renderer->gpu->core.UpdateDescriptorRanges(updates, sizeof(updates) / sizeof(updates[0]));\n    return true;\n}\n"""

new_update = """static bool update_radiance_scene_descriptors(RENDERER *renderer) {\n    if (!renderer || !renderer->radiance_scene_set || !renderer->radiance_constants_srv || !renderer->pass_constants_srv || !renderer->object_srv ||\n        !renderer->material_srv || !renderer->radiance_scene.triangles_srv || !renderer->radiance_scene.emissive_triangles_srv || !renderer->sdf.models_srv ||\n        !renderer->sdf.voxels_srv || !renderer->light_srv || !renderer->material_id.srv || !renderer->primitive_id.srv) {\n        return false;\n    }\n\n    const NriDescriptor *constants[] = {renderer->radiance_constants_srv, renderer->pass_constants_srv};\n    const NriDescriptor *scene_core[] = {\n        renderer->object_srv,\n        renderer->material_srv,\n        renderer->radiance_scene.triangles_srv,\n        renderer->radiance_scene.emissive_triangles_srv,\n        renderer->sdf.models_srv,\n        renderer->sdf.voxels_srv\n    };\n    const NriDescriptor *lights[] = {renderer->light_srv};\n    const NriDescriptor *identity_textures[] = {renderer->material_id.srv, renderer->primitive_id.srv};\n\n    const NriUpdateDescriptorRangeDesc updates[] = {\n        {.descriptorSet = renderer->radiance_scene_set, .rangeIndex = 0, .descriptors = constants, .descriptorNum = 2},\n        {.descriptorSet = renderer->radiance_scene_set, .rangeIndex = 1, .descriptors = scene_core, .descriptorNum = 6},\n        {.descriptorSet = renderer->radiance_scene_set, .rangeIndex = 3, .descriptors = lights, .descriptorNum = 1},\n        {.descriptorSet = renderer->radiance_scene_set, .rangeIndex = 4, .descriptors = identity_textures, .descriptorNum = 2}\n    };\n\n    renderer->gpu->core.UpdateDescriptorRanges(updates, sizeof(updates) / sizeof(updates[0]));\n    return true;\n}\n"""

text = replace_once(text, old_update, new_update, "update_radiance_scene_descriptors")

text = replace_once(
    text,
    """    update_hzb_descriptors(renderer);\n    update_present_descriptors(renderer);\n    update_trace_descriptors(renderer);\n\n    return true;\n}\n""",
    """    update_hzb_descriptors(renderer);\n    update_present_descriptors(renderer);\n    update_trace_descriptors(renderer);\n\n    if (renderer->radiance_scene.triangles_srv && !update_radiance_scene_descriptors(renderer)) return false;\n\n    return true;\n}\n""",
    "refresh space4 after size-dependent resources",
)

# G-buffer render transitions: add both identity render targets.
text = replace_once(
    text,
    """        {\n            .texture = renderer->object_id.texture,\n            .before = renderer->object_id.state,\n            .after = color,\n            .mipNum = 1,\n            .layerNum = 1,\n            .planes = NriPlaneBits_COLOR\n        },\n        {\n            .texture = renderer->depth.texture,\n""",
    """        {\n            .texture = renderer->object_id.texture,\n            .before = renderer->object_id.state,\n            .after = color,\n            .mipNum = 1,\n            .layerNum = 1,\n            .planes = NriPlaneBits_COLOR\n        },\n        {\n            .texture = renderer->material_id.texture,\n            .before = renderer->material_id.state,\n            .after = color,\n            .mipNum = 1,\n            .layerNum = 1,\n            .planes = NriPlaneBits_COLOR\n        },\n        {\n            .texture = renderer->primitive_id.texture,\n            .before = renderer->primitive_id.state,\n            .after = color,\n            .mipNum = 1,\n            .layerNum = 1,\n            .planes = NriPlaneBits_COLOR\n        },\n        {\n            .texture = renderer->depth.texture,\n""",
    "transition gbuffer for render identity barriers",
)

text = replace_once(
    text,
    """    renderer->gpu->core.CmdBarrier(command_buffer, &(NriBarrierDesc){\n        .textures = barriers,\n        .textureNum = 6\n    });\n    renderer->normal_roughness.state = color;\n    renderer->albedo_metallic.state = color;\n    renderer->emissive.state = color;\n    renderer->velocity.state = color;\n    renderer->object_id.state = color;\n    renderer->depth.state = depth;\n}\n""",
    """    renderer->gpu->core.CmdBarrier(command_buffer, &(NriBarrierDesc){\n        .textures = barriers,\n        .textureNum = 8\n    });\n    renderer->normal_roughness.state = color;\n    renderer->albedo_metallic.state = color;\n    renderer->emissive.state = color;\n    renderer->velocity.state = color;\n    renderer->object_id.state = color;\n    renderer->material_id.state = color;\n    renderer->primitive_id.state = color;\n    renderer->depth.state = depth;\n}\n""",
    "transition gbuffer render identity states",
)

# G-buffer read transitions: this second object_id block occurs in transition_gbuffer_for_read.
needle = """        {\n            .texture = renderer->object_id.texture,\n            .before = renderer->object_id.state,\n            .after = read,\n            .mipNum = 1,\n            .layerNum = 1,\n            .planes = NriPlaneBits_COLOR\n        }\n    };\n\n    renderer->gpu->core.CmdBarrier(command_buffer, &(NriBarrierDesc){\n        .textures = barriers,\n        .textureNum = 5\n    });\n    renderer->normal_roughness.state = read;\n    renderer->albedo_metallic.state = read;\n    renderer->emissive.state = read;\n    renderer->velocity.state = read;\n    renderer->object_id.state = read;\n}\n"""
replacement = """        {\n            .texture = renderer->object_id.texture,\n            .before = renderer->object_id.state,\n            .after = read,\n            .mipNum = 1,\n            .layerNum = 1,\n            .planes = NriPlaneBits_COLOR\n        },\n        {\n            .texture = renderer->material_id.texture,\n            .before = renderer->material_id.state,\n            .after = read,\n            .mipNum = 1,\n            .layerNum = 1,\n            .planes = NriPlaneBits_COLOR\n        },\n        {\n            .texture = renderer->primitive_id.texture,\n            .before = renderer->primitive_id.state,\n            .after = read,\n            .mipNum = 1,\n            .layerNum = 1,\n            .planes = NriPlaneBits_COLOR\n        }\n    };\n\n    renderer->gpu->core.CmdBarrier(command_buffer, &(NriBarrierDesc){\n        .textures = barriers,\n        .textureNum = 7\n    });\n    renderer->normal_roughness.state = read;\n    renderer->albedo_metallic.state = read;\n    renderer->emissive.state = read;\n    renderer->velocity.state = read;\n    renderer->object_id.state = read;\n    renderer->material_id.state = read;\n    renderer->primitive_id.state = read;\n}\n"""
text = replace_once(text, needle, replacement, "transition gbuffer for read identity")

# Append the two integer identity attachments after emissive in record_gbuffer_pass.
text = replace_once(
    text,
    """        {\n            .descriptor = renderer->emissive.attachment,\n            .clearValue = {\n                .color = {\n                    .f = {\n                        .x = 0.0f,\n                        .y = 0.0f,\n                        .z = 0.0f,\n                        .w = 1.0f\n                    }\n                }\n            },\n            .loadOp = NriLoadOp_CLEAR,\n            .storeOp = NriStoreOp_STORE\n        }\n    };\n""",
    """        {\n            .descriptor = renderer->emissive.attachment,\n            .clearValue = {\n                .color = {\n                    .f = {\n                        .x = 0.0f,\n                        .y = 0.0f,\n                        .z = 0.0f,\n                        .w = 1.0f\n                    }\n                }\n            },\n            .loadOp = NriLoadOp_CLEAR,\n            .storeOp = NriStoreOp_STORE\n        },\n        {\n            .descriptor = renderer->material_id.attachment,\n            .clearValue = {\n                .color = {\n                    .ui = {\n                        .x = 0u,\n                        .y = 0u,\n                        .z = 0u,\n                        .w = 0u\n                    }\n                }\n            },\n            .loadOp = NriLoadOp_CLEAR,\n            .storeOp = NriStoreOp_STORE\n        },\n        {\n            .descriptor = renderer->primitive_id.attachment,\n            .clearValue = {\n                .color = {\n                    .ui = {\n                        .x = 0u,\n                        .y = 0u,\n                        .z = 0u,\n                        .w = 0u\n                    }\n                }\n            },\n            .loadOp = NriLoadOp_CLEAR,\n            .storeOp = NriStoreOp_STORE\n        }\n    };\n""",
    "record gbuffer identity attachments",
)

# This is the record_gbuffer_pass color count, after the pipeline count was already changed.
text = replace_once(
    text,
    """    const NriRenderingDesc rendering = {\n        .colors = colors,\n        .colorNum = 5,\n        .depth = depth\n    };\n""",
    """    const NriRenderingDesc rendering = {\n        .colors = colors,\n        .colorNum = 7,\n        .depth = depth\n    };\n""",
    "record gbuffer color count",
)

render.write_text(text)
