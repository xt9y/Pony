from pathlib import Path

path = Path("shader.hlsl")
text = path.read_text()


def replace_once(old: str, new: str, label: str) -> None:
    global text
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected 1 match, got {count}")
    text = text.replace(old, new, 1)


helper = '''float3 ReflectedDirectAtPixel(uint2 pixel) {
    float3 direct = TraceDirectRadiance.Load(int3(pixel, 0)).rgb;
    float3 emissive = TraceEmissive.Load(int3(pixel, 0)).rgb;
    return max(direct - emissive, 0.0f);
}

'''

marker = '''[numthreads(8, 8, 1)]
void CS_DirectRadiance(uint3 dispatch_id : SV_DispatchThreadID) {
'''
replace_once(marker, helper + marker, "reflected direct helper insertion")

replace_once(
    '''        float3 radiance = TraceDirectRadiance.Load(int3(hit.hit_pixel, 0)).rgb;\n''',
    '''        float3 radiance = ReflectedDirectAtPixel(hit.hit_pixel);\n''',
    "legacy screen hit emission split",
)

replace_once(
    '''    entry.direct_radiance = direct;\n''',
    '''    entry.direct_radiance = float4(ReflectedDirectAtPixel(pixel), direct.w);\n''',
    "legacy surface cache emission split",
)

replace_once(
    '''        RayRadiance[index] = float4(TraceDirectRadiance.Load(int3(UnpackPixel(hit.meta.x), 0)).rgb, 1.0f);\n''',
    '''        RayRadiance[index] = float4(ReflectedDirectAtPixel(UnpackPixel(hit.meta.x)), 1.0f);\n''',
    "future screen hit emission split",
)

path.write_text(text)
