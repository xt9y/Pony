#!/usr/bin/env python3
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]

MOVED = (
    "window",
    "device",
    "core",
    "helper",
    "swapchain_api",
    "graphics_queue",
    "compute_queue",
    "copy_queue",
    "work_queue",
    "swapchain",
    "descriptor_pool",
    "frame_fence",
    "work_fence",
    "frame_contexts",
    "work_contexts",
    "active_frame",
    "active_work",
    "work_index",
    "work_next_fence",
    "upload",
    "swapchain_frames",
    "swapchain_textures",
    "swapchain_texture_count",
    "pipeline_cache",
    "timestamp_pool",
    "timestamp_readback",
    "timestamp_query_size",
    "timestamp_supported",
    "swapchain_width",
    "swapchain_height",
    "current_swap_index",
    "frame_index",
    "swapchain_format",
    "metal_view",
    "temporary_descriptors",
    "temporary_buffers",
    "temporary_descriptor_num",
    "temporary_descriptor_cap",
    "temporary_buffer_num",
    "temporary_buffer_cap",
    "texture_states",
    "texture_state_num",
    "texture_state_cap",
)


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def write(path: str, source: str) -> None:
    (ROOT / path).write_text(source, encoding="utf-8")


def redirected(source: str) -> str:
    fields = "|".join(re.escape(field) for field in MOVED)
    pattern = re.compile(rf"\b([A-Za-z_]\w*)->({fields})\b")

    def replace(match: re.Match[str]) -> str:
        owner, field = match.groups()
        if owner == "gpu":
            return match.group(0)
        return f"{owner}->gpu->{field}"

    return pattern.sub(replace, source)


def replace_once(source: str, pattern: str, replacement: str, label: str, flags: int = re.S) -> str:
    result, count = re.subn(pattern, replacement, source, count=1, flags=flags)
    if count != 1:
        raise RuntimeError(f"{label}: expected one replacement, got {count}")
    return result


def update_header() -> None:
    source = read("gpu.h")

    gpu = r'''typedef struct GPU {
    SDL_Window *window;

#if defined(__APPLE__)
    SDL_MetalView metal_view;
#endif

    NriDevice *device;
    NriCoreInterface core;
    NriHelperInterface helper;
    NriSwapChainInterface swapchain_api;

    NriQueue *graphics_queue;
    NriQueue *compute_queue;
    NriQueue *copy_queue;
    NriQueue *work_queue;

    NriSwapChain *swapchain;
    NriDescriptorPool *descriptor_pool;
    NriFence *frame_fence;
    NriFence *work_fence;

    FRAME_CONTEXT *frame_contexts;
    FRAME_CONTEXT *work_contexts;
    FRAME_CONTEXT *active_frame;
    FRAME_CONTEXT *active_work;
    uint64_t work_index;
    uint64_t work_next_fence;
    UPLOAD_CONTEXT *upload;

    SWAPCHAIN_TEXTURE *swapchain_frames;
    NriTexture **swapchain_textures;
    uint32_t swapchain_texture_count;

    NriPipelineCache *pipeline_cache;
    NriQueryPool *timestamp_pool;
    NriBuffer *timestamp_readback;
    uint32_t timestamp_query_size;
    bool timestamp_supported;

    uint32_t swapchain_width;
    uint32_t swapchain_height;
    uint32_t current_swap_index;
    uint64_t frame_index;
    NriFormat swapchain_format;

    NriDescriptor **temporary_descriptors;
    NriBuffer **temporary_buffers;
    uint32_t temporary_descriptor_num;
    uint32_t temporary_descriptor_cap;
    uint32_t temporary_buffer_num;
    uint32_t temporary_buffer_cap;

    TEXTURE_STATE *texture_states;
    uint32_t texture_state_num;
    uint32_t texture_state_cap;
} GPU;'''

    source = replace_once(source, r"typedef struct GPU \{.*?\} GPU;", gpu, "GPU struct")

    source = replace_once(
        source,
        r"(struct RENDERER \{\n    GPU \*gpu;\n    SCENE \*scene;\n).*?(    NriPipelineLayout \*surface_layout;)",
        r"\1\n    PROBE_WAVEFRONT_SCRATCH *probe_scratch;\n\n\2",
        "RENDERER low-level block",
    )

    write("gpu.h", source)

    internal = read("render_internal.h")
    internal = internal.replace(
        "bool r_init(RENDERER *renderer, const char *title, int width, int height);\nvoid r_deinit(RENDERER *renderer);",
        "bool renderer_gpu_resources_init(RENDERER *renderer);\nvoid renderer_gpu_resources_deinit(RENDERER *renderer);",
    )
    if "bool r_init(" in internal or "void r_deinit(" in internal:
        raise RuntimeError("legacy renderer lifecycle remained in render_internal.h")
    write("render_internal.h", internal)


def update_render() -> None:
    source = redirected(read("render.c"))

    source = replace_once(
        source,
        r"\nbool gpu_init\(GPU \*gpu, const char \*title, int width, int height\) \{.*?\n\}\n\nvoid gpu_deinit\(GPU \*gpu\) \{.*?\n\}\n",
        "\n",
        "transitional GPU lifecycle",
    )

    source = replace_once(
        source,
        r"bool renderer_init\(RENDERER \*renderer, GPU \*gpu\) \{.*?\n\}",
        '''bool renderer_init(RENDERER *renderer, GPU *gpu) {
    if (!renderer || !gpu || !gpu->device) return false;

    memset(renderer, 0, sizeof(*renderer));
    renderer->gpu = gpu;

    if (!renderer_gpu_resources_init(renderer)) {
        renderer->gpu = NULL;
        return false;
    }

    return true;
}''',
        "renderer_init",
    )

    source = replace_once(
        source,
        r"void renderer_deinit\(RENDERER \*renderer\) \{.*?\n\}",
        '''void renderer_deinit(RENDERER *renderer) {
    if (!renderer) return;
    renderer_gpu_resources_deinit(renderer);
}''',
        "renderer_deinit",
    )

    write("render.c", source)


def update_main_and_bake() -> None:
    main = read("main.c").replace("renderer.window", "gpu.window")
    write("main.c", main)
    write("bake.c", redirected(read("bake.c")))


def update_gpu_source() -> None:
    source = read("gpu.c")
    if '#include "render_internal.h"' not in source:
        source = source.replace('#include "game.h"', '#include "game.h"\n#include "render_internal.h"', 1)
    source = redirected(source)

    init_block = r'''bool gpu_init(GPU *gpu, const char *title, int width, int height) {
    if (!gpu || !title || width <= 0 || height <= 0) return false;

    memset(gpu, 0, sizeof(*gpu));
    gpu->window = SDL_CreateWindow(title, width, height,
                                   SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY
#if defined(__APPLE__)
            | SDL_WINDOW_METAL
#else
            | SDL_WINDOW_VULKAN
#endif
    );
    if (!gpu->window) return false;

    NriDeviceCreationDesc device_desc = {0};
    device_desc.graphicsAPI = NriGraphicsAPI_VK;
    device_desc.enableNRIValidation = false;
    device_desc.enableGraphicsAPIValidation = false;
    device_desc.vkBindingOffsets = (NriVKBindingOffsets){
        .sRegister = 0,
        .tRegister = 16,
        .bRegister = 32,
        .uRegister = 48
    };

    if (nriCreateDevice(&device_desc, &gpu->device) != NriResult_SUCCESS) {
        SDL_Log("NRI device creation failed");
        gpu_deinit(gpu);
        return false;
    }

    if (nriGetInterface(gpu->device, NRI_INTERFACE(NriCoreInterface), &gpu->core) != NriResult_SUCCESS ||
        nriGetInterface(gpu->device, NRI_INTERFACE(NriHelperInterface), &gpu->helper) != NriResult_SUCCESS ||
        nriGetInterface(gpu->device, NRI_INTERFACE(NriSwapChainInterface), &gpu->swapchain_api) != NriResult_SUCCESS) {
        SDL_Log("NRI interface acquisition failed");
        gpu_deinit(gpu);
        return false;
    }

    RENDERER shell = {.gpu = gpu};

    if (!acquire_queues(&shell) || !create_pipeline_cache(&shell) || !create_gpu_timestamps(&shell) ||
        !create_swapchain(&shell, width, height) || !create_descriptor_pool(&shell) ||
        !create_work_contexts(&shell) || !create_frame_contexts(&shell)) {
        gpu_deinit(gpu);
        return false;
    }

    return true;
}

bool renderer_gpu_resources_init(RENDERER *r) {
    if (!r || !r->gpu || !r->gpu->device) return false;

    r->show_volume = true;
    r->yaw = -0.78f;
    r->pitch = 0.34f;
    r->distance = 14.0f;
    r->target = v3(0.0f, 1.0f, 0.0f);

    if (!create_pipeline_layouts(r)) {
        renderer_gpu_resources_deinit(r);
        return false;
    }

    if (r->gpu->core.GetFormatSupport(r->gpu->device, NriFormat_D32_SFLOAT) & NriFormatSupportBits_DEPTH_STENCIL_ATTACHMENT) {
        r->depth_format = NriFormat_D32_SFLOAT;
    } else if (r->gpu->core.GetFormatSupport(r->gpu->device, NriFormat_D24_UNORM_S8_UINT) & NriFormatSupportBits_DEPTH_STENCIL_ATTACHMENT) {
        r->depth_format = NriFormat_D24_UNORM_S8_UINT;
    } else {
        r->depth_format = NriFormat_D16_UNORM;
    }

    NriShaderDesc surface_vs = compile_shader("shaders/vertex.hlsl", "surface_vs", "BUILD_SURFACE_VS", NriStageBits_VERTEX_SHADER);
    NriShaderDesc surface_ps = compile_shader("shaders/fragment.hlsl", "surface_fs", "BUILD_SURFACE_FS", NriStageBits_FRAGMENT_SHADER);
    NriShaderDesc line_vs = compile_shader("shaders/vertex.hlsl", "wireframe_vs", "BUILD_WIREFRAME_VS", NriStageBits_VERTEX_SHADER);
    NriShaderDesc line_ps = compile_shader("shaders/fragment.hlsl", "wireframe_fs", "BUILD_WIREFRAME_FS", NriStageBits_FRAGMENT_SHADER);
    NriShaderDesc sky_vs = compile_shader("shaders/vertex.hlsl", "fullscreen_vs", "BUILD_FULLSCREEN_VS", NriStageBits_VERTEX_SHADER);
    NriShaderDesc sky_ps = compile_shader("shaders/fragment.hlsl", "sky_fs", "BUILD_SKY_FS", NriStageBits_FRAGMENT_SHADER);

    if (!surface_vs.bytecode || !surface_ps.bytecode || !line_vs.bytecode || !line_ps.bytecode || !sky_vs.bytecode || !sky_ps.bytecode) {
        free_shader(&surface_vs);
        free_shader(&surface_ps);
        free_shader(&line_vs);
        free_shader(&line_ps);
        free_shader(&sky_vs);
        free_shader(&sky_ps);
        renderer_gpu_resources_deinit(r);
        return false;
    }

    r->solid_pipeline = make_surface_pipeline(r, &r->gpu->core, r->surface_layout, &surface_vs, &surface_ps);
    r->line_pipeline = make_line_pipeline(r, r->line_layout, &line_vs, &line_ps);
    r->sky_pipeline = make_sky_pipeline(r, r->sky_layout, &sky_vs, &sky_ps);

    free_shader(&surface_vs);
    free_shader(&surface_ps);
    free_shader(&line_vs);
    free_shader(&line_ps);
    free_shader(&sky_vs);
    free_shader(&sky_ps);

    if (!r->solid_pipeline || !r->line_pipeline || !r->sky_pipeline || !fx_init(&r->fx, r)) {
        renderer_gpu_resources_deinit(r);
        return false;
    }

    SDL_Log("GPU backend: NRI");
    SDL_Log("depth format: %s", r->depth_format == NriFormat_D32_SFLOAT ? "D32_FLOAT" : r->depth_format == NriFormat_D24_UNORM_S8_UINT ? "D24S8" : "D16_UNORM");
    SDL_Log("SDL_image: %d", IMG_Version());

    return true;
}
'''

    source = replace_once(
        source,
        r"bool r_init\(RENDERER \*r, const char \*title, int width, int height\) \{.*?\n\}\n\nbool draw_frame",
        init_block + "\nbool draw_frame",
        "renderer/GPU initialization",
    )

    worker_block = r'''bool bake_worker_init(RENDERER *r) {
    if (!r) return false;

    memset(r, 0, sizeof(*r));
    r->gpu = calloc(1, sizeof(*r->gpu));
    if (!r->gpu) return false;

    NriDeviceCreationDesc device_desc = {0};
    device_desc.graphicsAPI = NriGraphicsAPI_VK;
    device_desc.vkBindingOffsets = (NriVKBindingOffsets){
        .sRegister = 0,
        .tRegister = 16,
        .bRegister = 32,
        .uRegister = 48
    };

    if (nriCreateDevice(&device_desc, &r->gpu->device) != NriResult_SUCCESS) {
        bake_worker_deinit(r);
        return false;
    }

    if (nriGetInterface(r->gpu->device, NRI_INTERFACE(NriCoreInterface), &r->gpu->core) != NriResult_SUCCESS ||
        nriGetInterface(r->gpu->device, NRI_INTERFACE(NriHelperInterface), &r->gpu->helper) != NriResult_SUCCESS ||
        !acquire_queues(r)) {
        bake_worker_deinit(r);
        return false;
    }

    if (!create_pipeline_cache(r) || !create_gpu_timestamps(r) || !create_descriptor_pool(r) ||
        !create_work_contexts(r) || !create_pipeline_layouts(r)) {
        bake_worker_deinit(r);
        return false;
    }

    r->lightmap_sampler = sampler(r, NriFilter_LINEAR, NriFilter_LINEAR, NriAddressMode_CLAMP_TO_EDGE);
    r->lightmap_texture = create_texture(r, NriFormat_RGBA16_SFLOAT, NriTextureUsageBits_SHADER_RESOURCE, 1, 1);

    if (!r->lightmap_sampler || !r->lightmap_texture) {
        bake_worker_deinit(r);
        return false;
    }

    return true;
}

void bake_worker_deinit(RENDERER *r) {
    if (!r) return;

    GPU *gpu = r->gpu;
    free_probe_grid(&r->volume_probes);
    beam_free(&r->beams);

    if (gpu && gpu->device) {
        if (gpu->graphics_queue) gpu->core.QueueWaitIdle(gpu->graphics_queue);

        clear_temporary(r);
        probe_wavefront_scratch_destroy(r);
        release_bake_resources(r);
        release_buffer(r, r->volume_probe_buffer);
        release_buffer(r, r->beam_buffer);
        release_texture(r, r->lightmap_texture);

        if (r->lightmap_sampler) gpu->core.DestroyDescriptor(r->lightmap_sampler);
        destroy_pipeline_layouts(r);
    }

    if (gpu) {
        gpu_deinit(gpu);
        free(gpu);
    }

    memset(r, 0, sizeof(*r));
}

void renderer_gpu_resources_deinit(RENDERER *r) {
    if (!r) return;

    GPU *gpu = r->gpu;
    free(r->vertices);
    free(r->draws);
    free_probe_grid(&r->volume_probes);
    beam_free(&r->beams);

    if (gpu && gpu->device) {
        if (gpu->graphics_queue) gpu->core.QueueWaitIdle(gpu->graphics_queue);

        clear_temporary(r);
        probe_wavefront_scratch_destroy(r);
        fx_deinit(&r->fx);

        if (r->image_textures) {
            for (uint32_t i = 0; i < r->image_texture_count; ++i)
                release_texture(r, r->image_textures[i]);
        }

        free(r->image_textures);
        free(r->materials);
        release_texture(r, r->default_white);
        release_texture(r, r->default_normal);

        if (r->material_sampler) gpu->core.DestroyDescriptor(r->material_sampler);

        release_buffer(r, r->vertex_buffer);
        release_bake_resources(r);
        release_buffer(r, r->volume_probe_buffer);
        release_buffer(r, r->beam_buffer);
        release_texture(r, r->depth_texture);
        release_texture(r, r->lightmap_texture);

        if (r->lightmap_sampler) gpu->core.DestroyDescriptor(r->lightmap_sampler);
        if (r->sky_pipeline) gpu->core.DestroyPipeline(r->sky_pipeline);
        if (r->solid_pipeline) gpu->core.DestroyPipeline(r->solid_pipeline);
        if (r->line_pipeline) gpu->core.DestroyPipeline(r->line_pipeline);

        destroy_pipeline_layouts(r);
    }

    memset(r, 0, sizeof(*r));
}

void gpu_deinit(GPU *gpu) {
    if (!gpu) return;

    RENDERER shell = {.gpu = gpu};

    if (gpu->device) {
        if (gpu->graphics_queue) gpu->core.QueueWaitIdle(gpu->graphics_queue);

        destroy_upload_context(&shell);
        destroy_work_contexts(&shell);
        clear_temporary(&shell);
        destroy_frame_contexts(&shell);
        destroy_gpu_timestamps(&shell);
        destroy_swapchain(&shell);

        if (gpu->descriptor_pool) gpu->core.DestroyDescriptorPool(gpu->descriptor_pool);
        destroy_pipeline_cache(&shell);
        nriDestroyDevice(gpu->device);
    }

    free(gpu->temporary_descriptors);
    free(gpu->temporary_buffers);
    free(gpu->texture_states);

#if defined(__APPLE__)
    if (gpu->metal_view) SDL_Metal_DestroyView(gpu->metal_view);
#endif

    if (gpu->window) SDL_DestroyWindow(gpu->window);
    memset(gpu, 0, sizeof(*gpu));
}
'''

    source = replace_once(
        source,
        r"bool bake_worker_init\(RENDERER \*r\) \{.*?\n\}\n\nvoid bake_worker_deinit\(RENDERER \*r\) \{.*?\n\}\n\nvoid r_deinit\(RENDERER \*r\) \{.*?\n\}\s*$",
        worker_block,
        "worker and teardown lifecycle",
    )

    for field in MOVED:
        if re.search(rf"\br->{re.escape(field)}\b", source):
            raise RuntimeError(f"gpu.c still has direct renderer access to moved field {field}")

    write("gpu.c", source)


def validate() -> None:
    header = read("gpu.h")
    gpu = re.search(r"typedef struct GPU\s*\{(.*?)\}\s*GPU;", header, re.S)
    renderer = re.search(r"struct RENDERER\s*\{(.*?)\n\};", header, re.S)
    if not gpu or not renderer:
        raise RuntimeError("could not parse final GPU/RENDERER structs")

    required = ("NriDevice *device;", "NriCoreInterface core;", "NriQueue *graphics_queue;", "NriSwapChain *swapchain;")
    for field in required:
        if field not in gpu.group(1):
            raise RuntimeError(f"final GPU missing {field}")
        if field in renderer.group(1):
            raise RuntimeError(f"final RENDERER still contains {field}")

    if "r_init(" in read("render_internal.h") or "r_deinit(" in read("render_internal.h"):
        raise RuntimeError("legacy lifecycle declaration survived")


def main() -> None:
    update_header()
    update_render()
    update_main_and_bake()
    update_gpu_source()
    validate()


if __name__ == "__main__":
    main()
