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
        ".intensity=1.0f",
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
    if bake_array.group(1) != "LIGHTMAP_TEXELS_PER_UNIT,LIGHTMAP_MAX_SIZE,128u,3u,4u,32u,2u,32u,8u,50u,75u,1u,4u,16u,2u,1u,1u,4u,995u,25u,60u,100u":
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
    if len(jobs) != 27:
        raise AssertionError(f"expected 27 frozen shader jobs, found {len(jobs)}")

    for path, entry, define, fallback, stage, wave in jobs:
        source = shader_source(path)
        if not re.search(rf"\b{re.escape(entry)}\s*\(", source):
            raise AssertionError(f"{path}: shader entry {entry} missing from source/include graph")
        if define not in source:
            raise AssertionError(f"{path}: build define {define} missing from source/include graph")
        if fallback != "NULL" and fallback.strip('"') not in source:
            raise AssertionError(f"{path}: fallback define {fallback} missing from source/include graph")

    game = text("game.h")
    for field in ("layout_hash", "volume_hash", "beam_hash", "object_probes", "volume_probes", "beams"):
        if field not in game:
            raise AssertionError(f"cache ABI field missing: {field}")

    print("dustmite behavior contract: ok")


if __name__ == "__main__":
    main()
