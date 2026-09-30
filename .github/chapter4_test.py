from pathlib import Path


def require(path: str, needle: str) -> None:
    text = Path(path).read_text()
    if needle not in text:
        raise AssertionError(f"{path}: missing {needle!r}")


require("game.h", "uint32_t *surface_id;")
require("game.h", "NriBuffer *surface_ids;")
require("game.h", "NriDescriptor *surface_ids_srv;")
require("game.h", "RADIANCE_SCENE_FALLBACKS radiance_fallbacks;")
require("game.h", "NriDescriptorSet *emissive_trace_set;")
require("game.h", "NriDescriptorSet *emissive_scene_set;")
require("game.h", "NriDescriptorSet *emissive_probe_set;")
require("game.h", "NriPipelineLayout *emissive_layout;")
require("game.h", "NriPipeline *emissive_pipeline;")

require("sdf.c", "triangle->surface_id = i;")
require("sdf.c", "volume->surface_id = surface_id;")
require("sdf.c", "free(volume->surface_id);")

require("render.c", "create_radiance_scene_fallbacks")
require("render.c", "renderer->sdf.surface_ids_srv")
require("render.c", "renderer->radiance_fallbacks.dynamic_grid_cells_srv")
require("render.c", "radiance_emissive.cs.spv")
require("render.c", "RADIANCE_FEATURE_EMISSIVE")
require("render.c", "build_emissive_gather(renderer, command_buffer);")
require("render.c", "update_emissive_probe_descriptors")
require("render.c", "refresh_emissive_sampling")
require("render.c", ".rangeIndex = 2")
