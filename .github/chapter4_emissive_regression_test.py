from pathlib import Path

shader = Path("shader.hlsl").read_text()

if "float3 ReflectedDirectAtPixel(uint2 pixel)" not in shader:
    raise AssertionError("missing reflected-direct helper")

if shader.count("ReflectedDirectAtPixel(") < 4:
    raise AssertionError("reflected-direct split is not used by all legacy/future hit paths")

legacy_screen_old = "float3 radiance = TraceDirectRadiance.Load(int3(hit.hit_pixel, 0)).rgb;"
if legacy_screen_old in shader:
    raise AssertionError("legacy screen hits still inject visible emissive radiance")

future_screen_old = "RayRadiance[index] = float4(TraceDirectRadiance.Load(int3(UnpackPixel(hit.meta.x), 0)).rgb, 1.0f);"
if future_screen_old in shader:
    raise AssertionError("future screen hits still inject visible emissive radiance")

if "entry.direct_radiance = float4(ReflectedDirectAtPixel(pixel), direct.w);" not in shader:
    raise AssertionError("legacy surface cache still stores emissive radiance in direct_radiance")

if "float3 contribution = EvaluateEmissiveSample(" not in shader:
    raise AssertionError("world-space emissive sampler was removed")
