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
    if len(jobs) != 28:
        raise AssertionError(f"expected 28 shader jobs, found {len(jobs)}")

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
        "reference_world_layout(",
        "reference_bake_surface(",
        "renderer_update_reference_lighting(",
        "renderer->reference_lighting_enabled = !renderer->reference_lighting_enabled",
        "bvh_build(&tree, &scene->geometry, &scene->visual)",
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

    bake = text("bake.c")
    for needle in ("job->scene = &scene->static_geometry;", "job->visual = &scene->static_visual;", "job->layout = scene->lightmap;"):
        if needle not in bake:
            raise AssertionError(f"baker does not consume static-only scene: {needle}")

    print("dustmite behavior contract: ok")


if __name__ == "__main__":
    main()
