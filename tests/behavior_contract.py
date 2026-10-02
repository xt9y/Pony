#!/usr/bin/env python3
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def text(path: str | Path) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def compact(source: str) -> str:
    return re.sub(r"\s+", "", source)


def shader_source(path: str, seen: set[Path] | None = None) -> str:
    source_path = (ROOT / path).resolve()
    if seen is None:
        seen = set()
    if source_path in seen:
        return ""
    seen.add(source_path)

    source = source_path.read_text(encoding="utf-8")
    expanded = [source]
    for include in re.findall(r'^\s*#\s*include\s+"([^"]+)"', source, re.M):
        candidate = source_path.parent / include
        if not candidate.is_file():
            candidate = ROOT / include
        if candidate.is_file():
            expanded.append(shader_source(str(candidate.relative_to(ROOT)), seen))
    return "\n".join(expanded)


def main() -> None:
    main_c = compact(text("main.c"))

    required_main = (
        "#defineLIGHTMAP_TEXELS_PER_UNIT24u",
        "#defineLIGHTMAP_MAX_SIZE4096u",
        "#defineBAKE_IDLE_GRACE_MS180u",
        "#defineBAKE_IDLE_RENDER_MS200u",
        "#defineBAKE_IDLE_SLEEP_MS2u",
        ".direction={0.38f,0.30f,0.32f}",
        ".color={1.00f,0.94f,0.84f}",
        ".intensity=0.0f",
        ".angular_radius=0.00465f",
        ".zenith={0.22f,0.42f,0.78f}",
        ".horizon={0.68f,0.76f,0.88f}",
        ".density=0.045f",
        ".anisotropy=0.55f",
        ".probe_intensity=0.15f",
        ".emissive_probe_intensity=1.0f",
        ".max_distance=10000.0f",
        ".probe_spacing=4.0f",
        ".probe_samples=1024u",
        ".emissive_samples=128u",
        ".center_radius=0.50f",
        ".middle_radius=0.82f",
        ".center_steps=4u",
        ".middle_steps=2u",
        ".peripheral_steps=1u",
        ".center_stride=1u",
        ".middle_stride=8u",
        ".peripheral_stride=24u",
    )
    missing = [needle for needle in required_main if needle not in main_c]
    if missing:
        raise AssertionError(f"Dustmite runtime defaults changed: {missing}")

    bake_array = re.search(r"constuint32_tbake_settings\[\]=\{(.*?)\};", main_c)
    if not bake_array:
        raise AssertionError("bake_settings array missing")
    if bake_array.group(1) != "LIGHTMAP_TEXELS_PER_UNIT,LIGHTMAP_MAX_SIZE,128u,3u,4u,32u,2u,32u,8u,50u,75u,1u,4u,16u,2u,1u,1u,4u,995u,25u,60u,100u,32u":
        raise AssertionError("bake_settings values/order changed")

    volume_array = re.search(r"constfloatvolume_bake_settings\[\]=\{(.*?)\};", main_c)
    if not volume_array or volume_array.group(1) != "scene.volumetrics.probe_spacing,(float)scene.volumetrics.probe_samples,(float)scene.volumetrics.emissive_samples,scene.volumetrics.emissive_probe_intensity,9.0f":
        raise AssertionError("volume_bake_settings values/order changed")

    if "constuint32_tbeam_settings[]={64u,16u,5u};" not in main_c:
        raise AssertionError("beam_settings values/order changed")

    build = text("build.c")
    jobs = re.findall(
        r'\{"([^"]+\.hlsl)",\s*"([^"]+)",\s*"([^"]+)",\s*(NULL|"[^"]+"),\s*"([^"]+)",\s*([01])\}',
        build,
    )
    if len(jobs) != 29:
        raise AssertionError(f"expected 29 shader jobs, found {len(jobs)}")

    for path, entry, define, fallback, stage, wave in jobs:
        source = shader_source(path)
        if not re.search(rf"\b{re.escape(entry)}\s*\(", source):
            raise AssertionError(f"{path}: shader entry {entry} missing from source/include graph")
        if define not in source:
            raise AssertionError(f"{path}: build define {define} missing from source/include graph")
        if fallback != "NULL" and fallback.strip('"') not in source:
            raise AssertionError(f"{path}: fallback define {fallback} missing from source/include graph")

    game = text("game.h")
    gltf = text("gltf.c")
    main = text("main.c")
    render = text("render.c")
    gpu = text("gpu.c")

    for needle in ("MAX_LAYOUT_RANGES = 32", "counts[set] > MAX_LAYOUT_RANGES"):
        if needle not in gpu:
            raise AssertionError(f"descriptor layout range capacity guard missing: {needle}")

    for needle in ("base_floor_anchor(", "extra_model_transform(", "load_extra_model(", "renderer_focus_object(", "camera_radius"):
        if needle in main + render + game:
            raise AssertionError(f"extra-model camera/placement convenience leaked: {needle}")

    fragment = text("shaders/fragment.hlsl")
    bvh = text("bvh.c")
    compute_base = text("shaders/compute_base.hlsl")
    probe_wavefront = text("shaders/probe_wavefront.hlsl")

    for needle in (
        "texture_triangle_channel_average",
        "transmission = fminf(fmaxf(mat->transmission_factor",
        ".normal = {n.x, n.y, n.z, transmission}",
        "bvh_triangle_transmissive",
        "transmissive geometry:",
    ):
        if needle not in bvh:
            raise AssertionError(f"transmissive bake visibility missing from BVH: {needle}")

    for needle in ("triangle_is_transmissive", "tri.normal.w >= 0.999f", "if (triangle_is_transmissive(tri)) continue;"):
        if needle not in compute_base:
            raise AssertionError(f"transmissive lightmap visibility missing: {needle}")

    for needle in ("probe_triangle_transmissive", "tri.normal.w >= 0.999f", "if (probe_triangle_transmissive(tri)) continue;"):
        if needle not in probe_wavefront:
            raise AssertionError(f"transmissive probe visibility missing: {needle}")

    if "abs(dot(normalize(tri.normal.xyz), -direction))" not in compute_base:
        raise AssertionError("direct emissive sampling must emit from both triangle sides")

    if probe_wavefront.count("abs(dot(normalize(tri.normal.xyz), -direction))") < 2:
        raise AssertionError("probe emissive sampling must emit from both triangle sides")


    bake = text("bake.c")

    for needle in (
        "LIGHTMAP_EMISSIVE_MAX_SAMPLES 32u",
        "lightmap emissive mesh samples: %u per direct texel",
        "(float)LIGHTMAP_EMISSIVE_MAX_SAMPLES",
    ):
        if needle not in bake:
            raise AssertionError(f"dedicated lightmap mesh-light budget missing: {needle}")

    for needle in (
        "direct_emissive_target",
        "direct_emissive_stratified",
        "emissive_data.x > 0.0f ? min((uint)emissive_data.w, 64u) : 0u",
        "direct += emissive_sum / (float)emissive_count",
    ):
        if needle not in compute_base:
            raise AssertionError(f"dedicated lightmap mesh-light sampler missing: {needle}")

    if "100u, 32u" not in main:
        raise AssertionError("emissive lightmap sample budget missing from bake cache hash")


    advanced_material_fields = (
        "ior",
        "transmission_factor",
        "thickness_factor",
        "attenuation_color",
        "attenuation_distance",
        "iridescence_factor",
        "iridescence_ior",
        "iridescence_thickness_min",
        "iridescence_thickness_max",
        "transmission_texture",
        "thickness_texture",
        "iridescence_texture",
        "iridescence_thickness_texture",
    )
    missing = [field for field in advanced_material_fields if field not in game]
    if missing:
        raise AssertionError(f"advanced glTF material state missing: {missing}")

    for extension in (
        "KHR_materials_ior",
        "KHR_materials_transmission",
        "KHR_materials_volume",
        "KHR_materials_iridescence",
    ):
        if extension not in gltf:
            raise AssertionError(f"glTF material extension parser missing: {extension}")

    for needle in (
        "transmission_pipeline",
        "snapshot_scene_color",
        "NriAccessBits_COPY_SOURCE",
        "NriAccessBits_COPY_DESTINATION",
        "material_transmissive",
        "iridescence_thickness",
    ):
        if needle not in render:
            raise AssertionError(f"advanced material renderer path missing: {needle}")

    for needle in (
        "Texture2D<float4> Transmission",
        "Texture2D<float4> Thickness",
        "Texture2D<float4> Iridescence",
        "Texture2D<float4> IridescenceThickness",
        "Texture2D<float4> SceneColor",
        "thin_film_fresnel",
        "volume_attenuation",
        "refracted_scene_uv",
        "environment_radiance",
        "transmission_weight",
    ):
        if needle not in fragment:
            raise AssertionError(f"advanced material shader path missing: {needle}")

    transport = text("shaders/transport.hlsl")

    for needle in (
        "transport_visible_emissive",
        "const float knee = 4.0f;",
        "const float white = 12.0f;",
        "transport_emitted_radiance",
        "transport_diffuse_albedo",
        "transport_diffuse_response",
        "transport_offset_surface",
    ):
        if needle not in transport:
            raise AssertionError(f"shared transport contract missing: {needle}")

    for path in ("shaders/fragment.hlsl", "shaders/compute_base.hlsl", "shaders/probe_wavefront.hlsl"):
        if '#include "transport.hlsl"' not in text(path):
            raise AssertionError(f"{path}: shared transport include missing")

    if "transport_visible_emissive(emissive)" not in fragment:
        raise AssertionError("visible emissive compression must stay presentation-only")

    for field in ("layout_hash", "volume_hash", "beam_hash", "object_probes", "volume_probes", "beams"):
        if field not in game:
            raise AssertionError(f"cache ABI field missing: {field}")

    init = text("init.c")

    for needle in (
        "bool model_load(MODEL_ASSET *asset, const char *path)",
        "bool scene_compile(SCENE *scene)",
        "transform_position(object->transform",
        "transform_normal(object->transform",
        "material_texture_offset(&material, texture_offset)",
        "texture.image += (int32_t)image_offset",
        "bool scene_build_lightmap(SCENE *scene",
        "uint64_t scene_content_hash(const SCENE *scene)",
    ):
        if needle not in init:
            raise AssertionError(f"multi-model scene compilation missing: {needle}")

    main = text("main.c")
    for needle in (
        '#define BASE_MODEL_PATH "hospital_hallway.glb"',
        'strcmp(argv[i], "--") == 0',
        "return argc > 1 ? 1 : argc;",
        "load_scene_model(&scene, &models[0], BASE_MODEL_PATH",
        "const char *path = argv[extra_start + i];",
        "load_scene_model(&scene, &models[i + 1], path,",
        "OBJECT_STATE state = DYNAMIC",
        'strncmp(path, "static:", 7u) == 0',
        "DYNAMIC_Z_AMPLITUDE 1.0f",
        "DYNAMIC_Z_PERIOD_SECONDS 4.0f",
        "animated_dynamic_base_z + sinf(phase) * DYNAMIC_Z_AMPLITUDE",
        "object_set_transform(animated, moved)",
        "scene_build_lightmap(&scene, LIGHTMAP_TEXELS_PER_UNIT, LIGHTMAP_MAX_SIZE)",
        "scene_content_hash(&scene)",
        "bake_start(&renderer, &scene",
    ):
        if needle not in main:
            raise AssertionError(f"multi-model main path missing: {needle}")

    render = text("render.c")
    for needle in (
        "renderer_build_scene(renderer, scene, scene->lightmap)",
        "model_surface_layout(",
        "renderer_allocate_dynamic_lighting(",
        "camera_uniforms_for_draw(",
        "dynamic_lighting_find(",
        "draw->object_id",
    ):
        if needle not in render:
            raise AssertionError(f"instance-aware renderer path missing: {needle}")

    for needle in ("MESH static_geometry;", "GLTF_SCENE static_visual;", "OBJECT_ID id;", "transform_revision", "lighting_revision", "LIGHTMAP *surface_layout;", "OBJECT_ID object_id;"):
        if needle not in game:
            raise AssertionError(f"dynamic-lighting scene contract missing: {needle}")

    for needle in ("scene_extract_static(", "lmap_build(scene->lightmap, &scene->static_geometry", "object->state == STATIC"):
        if needle not in init:
            raise AssertionError(f"static/dynamic bake separation missing: {needle}")

    bvh = text("bvh.c")
    for needle in ("BVH_SURFACE_REF", "BVH_SURFACE_REF *surfaces;", "float barycentric[3];", "source_triangle", "local_triangle", "OBJECT_ID object_id", "bvh_hit_surface_uv"):
        if needle not in game:
            raise AssertionError(f"stable BVH surface-hit contract missing: {needle}")

    for needle in ("build[i].surface", "surface_refs ? surface_refs[i]", "best.barycentric[0]", "tree->surfaces[triangle]", "bvh_hit_surface_uv(", "bvh_build_with_surfaces("):
        if needle not in bvh:
            raise AssertionError(f"BVH surface identity implementation missing: {needle}")

    lmap = text("lmap.c")
    for needle in ("lmap_surface_uv(", "local_triangle * 6u", "back_face ? 3u : 0u", "barycentric[0]"):
        if needle not in lmap:
            raise AssertionError(f"object-local lighting-chart lookup missing: {needle}")

    for needle in (
        "reference_static_texture",
        "reference_lighting_enabled",
        "bake_full_transport",
    ):
        if needle not in game:
            raise AssertionError(f"dynamic reference state missing: {needle}")

    for needle in (
        "scene->surface_refs",
        "bvh_build_with_surfaces(&tree, &scene->geometry, &scene->visual",
        "reference_world_layout(",
        "reference_bake_surface(",
        "renderer_update_reference_lighting(",
        "reference_texture",
        "reference_transform_revision",
        "reference_lighting_revision",
        "renderer->reference_lighting_enabled = !renderer->reference_lighting_enabled",
    ):
        if needle not in render:
            raise AssertionError(f"dynamic reference path missing: {needle}")

    if "bake_params.y >= 0.0f" not in compute_base:
        raise AssertionError("reference full-transport mode does not disable secondary probe reuse")

    for needle in (
        "direct_sun_visibility(",
        "encoded_visibility",
        "float4(indirect.rgb + direct.rgb, direct.a)",
    ):
        if needle not in compute_base:
            raise AssertionError(f"explicit baked sun visibility missing: {needle}")

    if "baked_luma * 0.55f" in fragment:
        raise AssertionError("brightness-derived sun visibility heuristic must not return")
    if "baked_sample.a - visibility_floor" not in fragment:
        raise AssertionError("surface shader does not consume explicit sun visibility")

    for needle in (
        "StructuredBuffer<float> SurfaceBeams",
        "StructuredBuffer<SurfaceProbe> SurfaceProbes",
        "surface_probe_irradiance(",
        "static_beam_visibility(",
        "dynamic_cache_valid",
        "surface_probe_irradiance(input.world_position, geometric_normal) / PI",
        "cached_sun_visibility = static_beam_visibility(input.world_position)",
        "reference_mode > 0.5f ? cached_sun_visibility : cached_sun_visibility * dynamic_visibility",
    ):
        if needle not in fragment:
            raise AssertionError(f"dynamic baked-context fallback missing: {needle}")

    for needle in (
        "dynamic_lighting_find_const(",
        "allocation->transform_revision == object->transform_revision",
        "allocation->lighting_revision == object->lighting_revision",
        "allocation->scene_lighting_revision == r->scene->lighting_revision",
        "pending_scene_lighting_revision",
    ):
        if needle not in render:
            raise AssertionError(f"stale dynamic cache rejection missing: {needle}")

    if "if (!scene_compile(renderer->scene)) return false;" in render:
        raise AssertionError("normal dynamic motion must not rebuild the full flattened CPU scene")
    if "if (!scene_compile(r->scene)) return false;" not in render:
        raise AssertionError("F2 reference mode must rebuild the full current scene before oracle baking")
    if "if (r->reference_lighting_enabled || !r->has_bake" in render:
        raise AssertionError("reference mode must not freeze runtime cache convergence")
    for needle in ("acceptance_transform_revision", "acceptance_scene_lighting_revision"):
        if needle not in render:
            raise AssertionError(f"one-shot acceptance logging state missing: {needle}")

    for needle in (
        "GPU_FRAME_QUEUE_DEPTH",
        "DYNAMIC_TIMESTAMP_BASE",
        'gpu_timestamp_log(r, timing_base, "dynamic surface cache")',
        'gpu_timestamp_log(r, timing_base + 2u, "dynamic shadow map")',
        "gpu_timestamp_begin(r, cmd, timing_base)",
        "gpu_timestamp_end(r, cmd, timing_base + 2u)",
    ):
        if needle not in render + text("render_internal.h"):
            raise AssertionError(f"dynamic GPU cost instrumentation missing: {needle}")

    if "bool lmap_build_density(" not in text("lmap.c") or "lmap_build_density" not in game:
        raise AssertionError("fractional lightmap density API missing for bounded dynamic atlases")

    dynamic_surface = text("shaders/dynamic_surface.hlsl")
    for needle in (
        "DYNAMIC_SURFACE_SAMPLES_PER_FRAME 2048u",
        "DYNAMIC_SURFACE_CONVERGENCE_PASSES 4u",
        "model_surface_layout(struct MODEL *model, uint32_t target_samples)",
        "lmap_build_density(",
        "DYNAMIC_LIGHTING_MIN_TEXELS_PER_UNIT",
        "model->surface_layout->sample_count <= target_samples",
        "DYNAMIC_SURFACE_SAMPLES_PER_FRAME / count",
        "dynamic_surface_target",
        "renderer_build_dynamic_static_transport(",
        "dynamic_static_node_buffer",
        "dynamic_static_surface_buffer",
        "dynamic_static_uv_buffer",
        "dynamic_object_node_buffer",
        "dynamic_object_triangle_buffer",
        "dynamic_trace_instance_write(",
        "dynamic_trace_instances(",
        "m4_inverse_transform(",
        "gpu_clear_texture_zero(",
        "update_dynamic_surface_caches(",
        "pending_transform_revision",
        "pending_lighting_revision",
        "pending_scene_lighting_revision",
        "allocation->sample_cursor = 0u",
        "allocation->sample_pass = 0u",
        "cache_needs_clear",
        "clear.sample_count = total",
        "clear.trace_params[1] = 1.0f",
        "gpu_texture_barrier(r, cmd, allocation->texture, storage, storage)",
        "allocation->transform_revision = object->transform_revision",
        "allocation->scene_lighting_revision = r->scene->lighting_revision",
        "allocation->sample_pass++",
        "allocation->sample_pass < DYNAMIC_SURFACE_CONVERGENCE_PASSES",
        "gpu_transition_texture(r, cmd, allocation->texture, NriAccessBits_SHADER_RESOURCE",
    ):
        if needle not in render:
            raise AssertionError(f"fixed-budget dynamic cache scheduling missing: {needle}")

    for needle in (
        "BUILD_DYNAMIC_SURFACE_CS",
        "DYNAMIC_RAYS_PER_SAMPLE = 8u",
        "StructuredBuffer<SurfaceSample> Samples",
        "StructuredBuffer<BvhNode> Nodes",
        "StructuredBuffer<BvhTriangle> Triangles",
        "StructuredBuffer<StaticSurfaceRef> SurfaceRefs",
        "StructuredBuffer<float2> StaticUVs",
        "StructuredBuffer<BvhNode> SelfNodes",
        "StructuredBuffer<BvhTriangle> SelfTriangles",
        "dynamic_instance_inverse",
        "dynamic_instance_normal",
        "dynamic_instance_meta",
        "Texture2D<float4> StaticLightmap",
        "static_closest(",
        "instance_closest(",
        "dynamic_closest(",
        "dynamic_any(",
        "self_closest(",
        "self_any(",
        "static_lightmap_uv(",
        "static_outgoing(",
        "trace_static_indirect(",
        "beam_visibility(",
        "encoded_visibility",
        "trace_params.y > 0.5f",
        "SurfaceSample clear_sample = Samples[sample_offset + id.x]",
        "Output[uint2(clear_pixel % texture_width, clear_pixel / texture_width)] = 0.0f",
        "pass_index = (uint)max(trace_params.w, 0.0f)",
        "float blend = saturate(trace_params.z)",
        "Output[output_pixel] = lerp(Output[output_pixel], current, blend)",
    ):
        if needle not in dynamic_surface:
            raise AssertionError(f"dynamic surface cache shader missing: {needle}")

    for needle in (
        "reference_mode",
        "dynamic_lightmap_sample(",
        "Lightmap.Load(int3(pixel, 0))",
        "sample.a < validity_floor",
        "baked_sample.a < visibility_floor * 0.5f",
        "reference_mode < 0.5f",
    ):
        if needle not in fragment:
            raise AssertionError(f"runtime/reference cache separation missing: {needle}")

    for needle in (
        "DYNAMIC_INFLUENCE_LIMIT",
        "model_lighting_summary(",
        "dynamic_influences(",
        "dynamic_influence_center_radius",
        "dynamic_influence_axis_x",
        "dynamic_influence_axis_y",
        "dynamic_influence_axis_z",
        "dynamic_influence_diffuse",
        "dynamic_influence_emissive",
        "uniforms->dynamic_influence_meta[1] = current",
        "dynamic_influences(r, &result, draw ? draw->object_id : 0u)",
    ):
        if needle not in render:
            raise AssertionError(f"dynamic near-field influence descriptor missing: {needle}")

    for needle in (
        "dynamic_proxy_reflection_hit(",
        "dynamic_reflection_radiance(",
        "dynamic_reflection_weight",
        "if (i == dynamic_influence_meta.y) continue",
        "reflected = lerp(reflected, dynamic_reflection, dynamic_reflection_weight)",
    ):
        if needle not in fragment:
            raise AssertionError(f"bounded dynamic reflection integration missing: {needle}")

    for needle in (
        "event->key.key == SDLK_F6",
        "event->key.key == SDLK_F7",
        "(float)r->debug_view",
    ):
        if needle not in render:
            raise AssertionError(f"dynamic lighting debug control missing: {needle}")

    for needle in (
        "dynamic_flags.w > 4.5f",
        "dynamic_cache_valid > 0.5f",
        "dynamic_flags.w > 5.5f",
        "float negative = length(max(-dynamic_correction, 0.0f))",
        "float positive = length(max(dynamic_correction, 0.0f))",
    ):
        if needle not in fragment:
            raise AssertionError(f"dynamic lighting debug visualization missing: {needle}")

    for needle in (
        "signed_dynamic_near_field(",
        "projected_area",
        "surface_probe_irradiance(closest, object_normal)",
        "outgoing - baseline",
        "dynamic_flags.z > 0.5f",
        "baked + signed_dynamic_near_field(input.world_position, geometric_normal)",
    ):
        if needle not in fragment:
            raise AssertionError(f"signed static receiver correction missing: {needle}")

    compute = text("shaders/compute.hlsl")
    for needle in (
        "Texture2D<float> DynamicShadow",
        "volume_dynamic_shadow_visibility(",
        "dynamic_visibility = volume_dynamic_shadow_visibility(world_midpoint)",
        "visibility_shape * dynamic_visibility",
    ):
        if needle not in compute:
            raise AssertionError(f"dynamic volumetric shadowing missing: {needle}")

    for needle in (
        "NriDescriptorType_STRUCTURED_BUFFER, NriDescriptorType_TEXTURE, NriDescriptorType_SAMPLER",
        "shadow_texel_enabled",
        "dynamic_shadow_projection(r, frame, u.shadow_u_min",
    ):
        if needle not in render:
            raise AssertionError(f"dynamic volumetric binding missing: {needle}")

    gpu = text("gpu.c")
    for needle in (
        "bool gpu_clear_texture_zero(",
        "memset(mapped, 0",
        "CmdUploadBufferToTexture",
    ):
        if needle not in gpu:
            raise AssertionError(f"dynamic atlas zero initialization missing: {needle}")

    cache = text("cache.c")
    for needle in ("DM_CACHE_VERSION 11u", "direct_pixels", "hash_bytes(hash, out->direct_pixels", "fwrite(data->direct_pixels"):
        if needle not in cache:
            raise AssertionError(f"versioned separated direct-light cache missing: {needle}")

    for needle in ("baked_direct_texture", "upload_direct_lightmap"):
        if needle not in game + render + text("bake.c"):
            raise AssertionError(f"persistent separated direct-light term missing: {needle}")

    if "exact_baked_direct * (dynamic_visibility - 1.0f)" in fragment or "Texture2D<float4> BakedDirect" in fragment:
        raise AssertionError("dynamic sun shadows must not erase emissive direct-light energy")
    if "static_direct * cached_sun_visibility * (dynamic_visibility - 1.0f)" not in fragment:
        raise AssertionError("static receiver sun correction must use explicit baked sun visibility only")

    for needle in (
        "cached_sun_visibility",
        "if (dynamic_any(shadow))",
        "lighting = max(lighting - baked_sun, 0.0f)",
    ):
        if needle not in dynamic_surface:
            raise AssertionError(f"dynamic cache does not feed self-shadowed static bounce: {needle}")

    for needle in (
        "download_rgba16f_texture(",
        "gpu_transition_texture(r, cmd, texture, NriAccessBits_SHADER_RESOURCE, NriLayout_SHADER_RESOURCE, NriStageBits_ALL)",
    ):
        if needle not in text("bake.c") + text("render_internal.h"):
            raise AssertionError(f"reference metric texture readback contract missing: {needle}")

    for needle in (
        "log_dynamic_matte_reference_error(",
        "DYNAMIC_ACCEPTANCE_MIN_COVERAGE_PERCENT 99.0",
        "DYNAMIC_ACCEPTANCE_MAX_RGB_MAE 0.03",
        "DYNAMIC_ACCEPTANCE_MAX_NRMSE 0.10",
        'accepted ? "PASS" : "FAIL"',
        "dynamic acceptance object %u: %s | matte cache coverage",
        "allocation->transform_revision != object->transform_revision",
        "allocation->scene_lighting_revision != r->scene->lighting_revision",
        "allocation->sample_pass < DYNAMIC_SURFACE_CONVERGENCE_PASSES",
        "allocation->lighting_revision != object->lighting_revision",
        "r->dynamic_lighting_count == 1u",
        "bvh_build(&self_tree, model_data->geometry, model_data->visual)",
        "m4_vector(inverse_model, sun)",
        "trace_any(&self_tree, local_shadow)",
        "RGB MAE",
        "NRMSE",
    ):
        if needle not in render:
            raise AssertionError(f"matte dynamic/reference acceptance metric missing: {needle}")

    bake = text("bake.c")
    render = text("render.c")

    for needle in (
        "seed.direct_pixels = black_pixel",
        "cache_write(job->worker_path",
    ):
        if needle not in bake:
            raise AssertionError(f"fast bake seed separated-direct regression missing: {needle}")

    for needle in (
        "if (!reuse_volume && !upload_bvh(renderer, &tree))",
        "fallback volume-probe bake failed",
    ):
        if needle not in render:
            raise AssertionError(f"legacy fallback probe prerequisites missing: {needle}")

    for needle in ("job->scene = &scene->static_geometry;", "job->visual = &scene->static_visual;", "job->layout = scene->lightmap;"):
        if needle not in bake:
            raise AssertionError(f"baker does not consume static-only scene: {needle}")

    if "!scene->compiled || !scene->lightmap_valid" in bake:
        raise AssertionError("dynamic motion must not block a static-only rebake")

    print("dustmite behavior contract: ok")


if __name__ == "__main__":
    main()
