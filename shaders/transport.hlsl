#ifndef PONY_TRANSPORT_HLSL
#define PONY_TRANSPORT_HLSL

/*
 * Shared surface-transport contract.
 *
 * Bake and runtime shading include this file so STATIC/DYNAMIC changes only
 * how irradiance is produced/stored, never the material or transport formula.
 * Presentation-only functions are explicitly named as such and must never be
 * used by the bake/path transport.
 */

float3 transport_srgb_to_linear(float3 c) {
    float3 low = c / 12.92f;
    float3 high = pow((c + 0.055f) / 1.055f, 2.4f);
    return lerp(high, low, step(c, float3(0.04045f, 0.04045f, 0.04045f)));
}

float3 transport_diffuse_albedo(float3 base_color, float metallic) {
    return max(base_color, 0.0f) * (1.0f - saturate(metallic));
}

float3 transport_diffuse_response(float3 diffuse_albedo, float3 irradiance) {
    return max(diffuse_albedo, 0.0f) * max(irradiance, 0.0f);
}

float3 transport_bounce_albedo(float3 albedo) {
    return max(albedo, 0.0f);
}

float3 transport_emitted_radiance(float3 emissive) {
    return max(emissive, 0.0f);
}

float3 transport_offset_surface(float3 position, float3 normal, float epsilon) {
    return position + normalize(normal) * epsilon;
}

/* Presentation only. Never feed this value back into lighting transport. */
float3 transport_visible_emissive(float3 emissive) {
    const float knee = 4.0f;
    const float white = 12.0f;

    emissive = max(emissive, 0.0f);
    float luminance = dot(emissive, float3(0.2126f, 0.7152f, 0.0722f));
    if (luminance <= knee) return emissive;

    float mapped = knee + (white - knee) * (1.0f - exp(-(luminance - knee) / (white - knee)));
    return emissive * (mapped / max(luminance, 1.0e-4f));
}

#endif
