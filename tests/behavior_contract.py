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
    if len(jobs) != 30:
        raise AssertionError(f"expected 30 shader jobs, found {len(jobs)}")

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
        "const float z = sinf(phase) * DYNAMIC_Z_AMPLITUDE",
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
        "reference_surface_layout(",
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
        "dynamic_reference_valid",
        "surface_probe_irradiance(input.world_position, geometric_normal) / PI",
        "cached_sun_visibility = static_beam_visibility(input.world_position)",
        "reference_mode > 0.5f ? cached_sun_visibility : cached_sun_visibility * dynamic_visibility",
    ):
        if needle not in fragment:
            raise AssertionError(f"dynamic baked-context fallback missing: {needle}")

    if "if (!scene_compile(renderer->scene)) return false;" in render:
        raise AssertionError("normal dynamic motion must not rebuild the full flattened CPU scene")
    if "if (!scene_compile(r->scene)) return false;" not in render:
        raise AssertionError("F2 reference mode must rebuild the full current scene before oracle baking")

    if "if (frame->sun.intensity <= 0.0f) return true;" not in render:
        raise AssertionError("disabled sun must skip the dynamic shadow-map pass")

    if "GPU_FRAME_QUEUE_DEPTH" not in text("render_internal.h"):
        raise AssertionError("frame queue depth contract missing")

    for forbidden in (
        "DYNAMIC_TIMESTAMP_BASE",
        "DYNAMIC_TIMESTAMP_STRIDE",
        "DYNAMIC_TIMING_LOG_INTERVAL",
        'gpu_timestamp_log(r, timing_base, "dynamic field visibility")',
        'gpu_timestamp_log(r, timing_base + 2u, "dynamic shadow map")',
    ):
        if forbidden in render:
            raise AssertionError(f"blocking realtime GPU timing instrumentation returned: {forbidden}")

    for forbidden in (
        'gpu_timestamp_log(r, timing_base, "dynamic object cache")',
        'gpu_timestamp_log(r, timing_base + 2u, "dynamic receiver cache")',
        "update_dynamic_surface_caches(",
        "update_dynamic_receiver_cache(",
        "renderer_build_dynamic_receiver_cache(",
        "renderer_build_dynamic_static_transport(",
        "dynamic_receiver_texture",
        "dynamic_receiver_sample_buffer",
        "dynamic_receiver_grid_offsets",
        "dynamic_static_surface_buffer",
        "dynamic_static_uv_buffer",
        "DYNAMIC_SURFACE_SAMPLES_PER_FRAME",
        "DYNAMIC_RECEIVER_SAMPLE_BUDGET",
    ):
        if forbidden in render + game:
            raise AssertionError(f"deleted runtime mini-baker returned: {forbidden}")

    if (ROOT / "shaders/dynamic_surface.hlsl").exists():
        raise AssertionError("obsolete dynamic surface mini-baker shader still exists")
    if (ROOT / "shaders/dynamic_receiver.hlsl").exists():
        raise AssertionError("obsolete dynamic receiver mini-baker shader still exists")

    if len(jobs) != 30:
        raise AssertionError(f"expected 30 shader jobs after mini-baker removal, found {len(jobs)}")

    dynamic_radiance = text("shaders/dynamic_radiance.hlsl")

    for needle in (
        "DYNAMIC_RADIANCE_FIELD_DIM 32u",
        "DYNAMIC_VISIBILITY_FIELD_DIM 8u",
        "DYNAMIC_RADIANCE_FIELD_SAMPLES 64u",
        "DYNAMIC_RADIANCE_IRRADIANCE_FLOOR 0.00075f",
        "DYNAMIC_RADIANCE_CACHE_MAGIC",
        "DYNAMIC_RADIANCE_CACHE_VERSION 2u",
        "dynamic_radiance_cache_read(",
        "dynamic_radiance_cache_write(",
        "read_dynamic_radiance_buffer(",
        ".pony-radiance-%016",
        "radiance_cache_hash",
        "hash_bytes(DYNAMIC_RADIANCE_CACHE_VERSION, tree.nodes",
        "hash_bytes(field_hash, tree.triangles",
        "NriAccessBits_COPY_SOURCE",
        "NriAccessBits_SHADER_RESOURCE",
        "DYNAMIC_SHADOW_SIZE 512u",
        "create_dynamic_radiance_layout(",
        "dynamic_radiance_pipeline",
        "dynamic_radiance_buffer",
        "dynamic_radiance_textures",
        "dynamic_radiance_views",
        "gpu_create_persistent_texture_view(",
        "dynamic radiance runtime: %u field(s) | max SH coefficient",
        "dynamic_radiance_visibility_buffer",
        "dynamic_radiance_fallback_texture",
        "dynamic_radiance_visibility_fallback_buffer",
        "dynamic_radiance_sampler",
        "gpu_create_texture_3d(",
        "gpu_upload_texture_3d_data(",
        "upload_dynamic_field_textures(",
        "radiance_visibility_node_buffer",
        "radiance_visibility_triangle_buffer",
        "renderer_build_radiance_visibility(",
        "generate_dynamic_radiance_fields(",
        "dynamic_emissive_bounds(",
        "radiance_probe_offset",
        "radiance_origin",
        "radiance_spacing",
        "allocation->emissive_weight",
        "gpu_submit_commands(renderer, allocator, cmd)",
        "dynamic radiance fields: %u generated + %u cached | %u probes",
    ):
        if needle not in render + game:
            raise AssertionError(f"object-local radiance field infrastructure missing: {needle}")

    for needle in (
        "BUILD_DYNAMIC_RADIANCE_CS",
        "StructuredBuffer<BvhNode> Nodes",
        "StructuredBuffer<BvhTriangle> Triangles",
        "RWStructuredBuffer<float4> Coefficients",
        "sh_basis(",
        "trace_any(",
        "((float)sample + random01(seed)) / (float)sample_count",
        "pdf_solid_angle",
        "contribution * basis[coefficient]",
        "Coefficients[coefficient_base + coefficient]",
    ):
        if needle not in dynamic_radiance:
            raise AssertionError(f"object-local radiance field generator missing: {needle}")

    for needle in (
        "Texture3D<float4> DynamicRadiance0",
        "Texture3D<float4> DynamicRadiance6",
        "StructuredBuffer<float> DynamicRadianceVisibility",
        "SamplerState DynamicRadianceSampler",
        "dynamic_radiance_origin_spacing",
        "dynamic_radiance_dims_offset",
        "dynamic_visibility_origin_spacing",
        "dynamic_visibility_dims_offset",
        "dynamic_radiance_inverse",
        "dynamic_radiance_model",
        "dynamic_radiance_filtered(",
        "transpose((float3x3)dynamic_radiance_model[index])",
        "dynamic_radiance_field(",
        "dynamic_radiance_lighting(",
        "dynamic_radiance_static_visibility(",
        "DynamicRadiance0.SampleLevel(DynamicRadianceSampler",
        "DynamicRadianceVisibility[probe]",
        "dynamic_influence_meta.w",
        "smoothstep(0.0f, 2.0f, edge)",
        "dynamic_correction = dynamic_radiance_lighting(input.world_position, geometric_normal)",
    ):
        if needle not in fragment:
            raise AssertionError(f"surface radiance-field sampling missing: {needle}")

    for needle in (
        "surface_probe_irradiance(input.world_position, geometric_normal) / PI",
        "cached_sun_visibility = static_beam_visibility(input.world_position)",
    ):
        if needle not in fragment:
            raise AssertionError(f"dynamic object static-probe receive path missing: {needle}")

    for needle in (
        "dynamic_influences(",
        "dynamic_radiance_origin_spacing",
        "dynamic_radiance_dims_offset",
        "dynamic_visibility_origin_spacing",
        "dynamic_visibility_dims_offset",
        "dynamic_radiance_inverse",
        "m4_inverse_transform(object->transform)",
        "result.dynamic_influence_meta[2] = r->dynamic_radiance_textures[0] ? 1u : 0u",
        "result.dynamic_influence_meta[3] = r->dynamic_radiance_field_count",
        "update_dynamic_radiance_visibility(",
        "visibility_transform_revision",
        "visibility_pending_revision",
        "finish_dynamic_visibility_updates(",
        "updated_fields < DYNAMIC_INFLUENCE_LIMIT",
        "allocation->visibility_dims[0] * allocation->visibility_dims[1] * allocation->visibility_dims[2]",
        "src, 27",
    ):
        if needle not in render:
            raise AssertionError(f"moving radiance-field transform metadata missing: {needle}")

    for needle in (
        "dynamic_proxy_reflection_hit(",
        "dynamic_reflection_radiance(",
        "dynamic_reflection_weight",
        "if (i == dynamic_influence_meta.y) continue",
        "reflected = lerp(reflected, dynamic_reflection, dynamic_reflection_weight)",
    ):
        if needle not in fragment:
            raise AssertionError(f"bounded dynamic reflection integration missing: {needle}")

    dynamic_visibility = text("shaders/dynamic_visibility.hlsl")
    for needle in (
        "BUILD_DYNAMIC_VISIBILITY_CS",
        "StructuredBuffer<BvhNode> Nodes",
        "StructuredBuffer<BvhTriangle> Triangles",
        "RWStructuredBuffer<float> Visibility",
        "trace_any(",
        "world_position = mul(model, float4(local_position, 1.0f)).xyz",
        "Visibility[field_meta.x + probe_index] = visibility",
    ):
        if needle not in dynamic_visibility:
            raise AssertionError(f"static-scene radiance visibility pass missing: {needle}")

    compute = text("shaders/compute.hlsl")
    for needle in (
        "Texture2D<float> DynamicShadow",
        "volume_dynamic_shadow_visibility(",
        "Texture3D<float4> DynamicRadiance0",
        "Texture3D<float4> DynamicRadiance6",
        "StructuredBuffer<float> DynamicRadianceVisibility",
        "SamplerState DynamicRadianceSampler",
        "dynamic_volume_filtered(",
        "dynamic_volume_static_visibility(",
        "dynamic_visibility_origin_spacing",
        "dynamic_visibility_dims_offset",
        "dynamic_volume_radiance(",
        "DynamicRadiance0.SampleLevel(DynamicRadianceSampler",
        "DynamicRadianceVisibility[probe]",
        "radiance += dynamic_volume_radiance(p, scattering_direction)",
    ):
        if needle not in compute:
            raise AssertionError(f"dynamic volumetric field integration missing: {needle}")

    for needle in (
        "dynamic_radiance_meta",
        "dynamic_shadow_projection(r, frame, u.shadow_u_min",
        "r->dynamic_radiance_views[i] ? r->dynamic_radiance_views[i] : r->dynamic_radiance_fallback_view",
        "r->dynamic_radiance_sampler ? r->dynamic_radiance_sampler : sampler_desc",
        "u.dynamic_radiance_meta[2] = r->dynamic_radiance_field_count",
    ):
        if needle not in render:
            raise AssertionError(f"dynamic volumetric field binding missing: {needle}")

    for forbidden in (
        "StructuredBuffer<float4> DynamicRadiance",
        "dynamic_radiance_probe_value(",
        "dynamic_volume_probe_value(",
    ):
        if forbidden in fragment + compute:
            raise AssertionError(f"manual 9-SH runtime radiance interpolation returned: {forbidden}")

    for needle in (
        "reference_mode",
        "reference_dynamic_lightmap_sample(",
        "renderer_update_reference_lighting(",
        "reference_texture",
    ):
        if needle not in fragment + render:
            raise AssertionError(f"F2 reference/oracle separation missing: {needle}")

    if "exact_baked_direct * (dynamic_visibility - 1.0f)" in fragment or "Texture2D<float4> BakedDirect" in fragment:
        raise AssertionError("dynamic sun shadows must not erase emissive direct-light energy")
    if "static_direct * cached_sun_visibility * (dynamic_visibility - 1.0f)" not in fragment:
        raise AssertionError("static receiver sun correction must use explicit baked sun visibility only")

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

    bake = text("bake.c")
    render = text("render.c")

    for needle in (
        "seed.direct_pixels = black_pixel",
        "release_texture(r, r->baked_direct_texture)",
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
