from pathlib import Path
import re

p = Path('shader.hlsl')
s = p.read_text()

def sub_once(text, pattern, replacement, label):
    out, count = re.subn(pattern, replacement, text, count=1, flags=re.S)
    if count != 1:
        raise SystemExit(f'{label}: expected 1 replacement, got {count}')
    return out

# 1. Screen hits become canonical geometry-addressed SurfaceHits.
pattern = r"SurfaceHit TraceScreenSurface\(TraceRay ray\) \{.*?\n\}\n\n\[numthreads\(8, 8, 1\)\]\nvoid CS_HZB"
replacement = r'''SurfaceHit SurfaceFromTriangle(uint triangle_id, float3 position, float3 normal, float distance, uint hit_type);
float3 SceneTriangleFacingNormal(uint triangle_id, float3 incoming_direction);
bool IntersectSceneTriangle(TraceRay ray, uint triangle_id, float min_t, float max_t, out float hit_t, out float3 hit_normal);

SurfaceHit TraceScreenSurface(TraceRay ray) {
    uint2 origin_pixel = UnpackPixel(ray.origin_pixel);
    float source_depth = TraceDepth.Load(int3(origin_pixel, 0));
    float3 source_position = ReconstructWorldPosition(origin_pixel, source_depth);
    float3 source_normal = normalize(TraceNormalRoughness.Load(int3(origin_pixel, 0)).xyz);
    float3 source_view_direction = normalize(source_position - TraceFrame.camera_position.xyz);
    if (dot(source_normal, source_view_direction) > 0.0f) source_normal = -source_normal;

    TraceHit legacy = TraceScreenRay(
        ray.origin_tmin.xyz,
        ray.direction_tmax.xyz,
        origin_pixel,
        ray.source_object_id,
        source_normal,
        ray.direction_tmax.w,
        Radiance.trace_params.z,
        Radiance.trace_params.w,
        Radiance.trace_limits.x,
        Radiance.trace_limits.y
    );

    if (legacy.type != TRACE_SCREEN) return MakeSurfaceHit(legacy.type, legacy.distance);

    uint triangle_id = TracePrimitiveId.Load(int3(legacy.hit_pixel, 0));
    if (triangle_id >= Radiance.scene_counts.z) return MakeSurfaceHit(TRACE_MISS, ray.direction_tmax.w);

    float tolerance = max(Radiance.trace_params.z * 4.0f, 0.05f);
    float exact_t;
    float3 exact_normal;
    SurfaceHit hit;

    if (IntersectSceneTriangle(
            ray,
            triangle_id,
            max(ray.origin_tmin.w, legacy.distance - tolerance),
            min(ray.direction_tmax.w, legacy.distance + tolerance),
            exact_t,
            exact_normal)) {
        float3 exact_position = ray.origin_tmin.xyz + ray.direction_tmax.xyz * exact_t;
        hit = SurfaceFromTriangle(triangle_id, exact_position, exact_normal, exact_t, TRACE_SCREEN);
    } else {
        float depth = TraceDepth.Load(int3(legacy.hit_pixel, 0));
        float3 position = ReconstructWorldPosition(legacy.hit_pixel, depth);
        float3 normal = SceneTriangleFacingNormal(triangle_id, ray.direction_tmax.xyz);
        if (dot(normal, normal) <= 1.0e-8f) {
            normal = normalize(TraceNormalRoughness.Load(int3(legacy.hit_pixel, 0)).xyz);
            if (dot(normal, ray.direction_tmax.xyz) > 0.0f) normal = -normal;
        }
        hit = SurfaceFromTriangle(triangle_id, position, normal, legacy.distance, TRACE_SCREEN);
    }

    if (hit.identity.z == INVALID_INDEX) return MakeSurfaceHit(TRACE_MISS, ray.direction_tmax.w);

    uint object_index = SceneTriangles[triangle_id].meta.x;
    hit.meta.y = object_index < Radiance.scene_counts.x
        ? SceneObjects[object_index].meta.x
        : Radiance.feature_flags.y;
    hit.meta.z = PackPixel(legacy.hit_pixel);
    return hit;
}

[numthreads(8, 8, 1)]
void CS_HZB'''
s = sub_once(s, pattern, replacement, 'TraceScreenSurface')

# 2. Exact triangle intersection and local-SDF refinement helpers.
marker = '\nfloat3 LocalSDFNormal(GPUSDFModel model, float3 p) {'
if s.count(marker) != 1:
    raise SystemExit(f'LocalSDFNormal marker count = {s.count(marker)}')
helpers = r'''
float3 SceneTriangleFacingNormal(uint triangle_id, float3 incoming_direction) {
    if (triangle_id >= Radiance.scene_counts.z) return 0.0f;
    GPUSceneTriangle tri = SceneTriangles[triangle_id];
    uint object_index = tri.meta.x;
    if (object_index >= Radiance.scene_counts.x) return 0.0f;

    row_major float4x4 world = SceneObjects[object_index].world;
    float3 a = TransformPoint(tri.p0.xyz, world);
    float3 b = TransformPoint(tri.p1.xyz, world);
    float3 c = TransformPoint(tri.p2.xyz, world);
    float3 n = cross(b - a, c - a);
    float n2 = dot(n, n);
    if (n2 <= 1.0e-12f) return 0.0f;
    n *= rsqrt(n2);
    if (dot(n, incoming_direction) > 0.0f) n = -n;
    return n;
}

bool IntersectSceneTriangle(
    TraceRay ray,
    uint triangle_id,
    float min_t,
    float max_t,
    out float hit_t,
    out float3 hit_normal
) {
    hit_t = max_t;
    hit_normal = 0.0f;
    if (triangle_id >= Radiance.scene_counts.z) return false;

    GPUSceneTriangle tri = SceneTriangles[triangle_id];
    uint object_index = tri.meta.x;
    if (object_index >= Radiance.scene_counts.x) return false;

    row_major float4x4 world = SceneObjects[object_index].world;
    float3 a = TransformPoint(tri.p0.xyz, world);
    float3 b = TransformPoint(tri.p1.xyz, world);
    float3 c = TransformPoint(tri.p2.xyz, world);
    float3 e1 = b - a;
    float3 e2 = c - a;
    float3 pvec = cross(ray.direction_tmax.xyz, e2);
    float det = dot(e1, pvec);
    if (abs(det) <= 1.0e-9f) return false;

    float inv_det = rcp(det);
    float3 tvec = ray.origin_tmin.xyz - a;
    float u = dot(tvec, pvec) * inv_det;
    if (u < -1.0e-4f || u > 1.0001f) return false;

    float3 qvec = cross(tvec, e1);
    float v = dot(ray.direction_tmax.xyz, qvec) * inv_det;
    if (v < -1.0e-4f || u + v > 1.0001f) return false;

    float t = dot(e2, qvec) * inv_det;
    if (t < min_t || t > max_t) return false;

    float3 n = cross(e1, e2);
    float n2 = dot(n, n);
    if (n2 <= 1.0e-12f) return false;
    n *= rsqrt(n2);
    if (dot(n, ray.direction_tmax.xyz) > 0.0f) n = -n;

    hit_t = t;
    hit_normal = n;
    return true;
}

bool RefineLocalSDFSurface(
    GPUSDFModel model,
    TraceRay ray,
    float3 local_position,
    float sdf_t,
    float epsilon,
    float direction_scale,
    float limit,
    out SurfaceHit hit
) {
    hit = MakeSurfaceHit(TRACE_MISS, limit);

    float3 size = max(model.bounds_max.xyz - model.bounds_min.xyz, float3(1.0e-6f, 1.0e-6f, 1.0e-6f));
    float3 uvw = (local_position - model.bounds_min.xyz) / size;
    if (any(uvw < 0.0f) || any(uvw > 1.0f)) return false;

    uint resolution = model.meta.y;
    float3 grid = saturate(uvw) * (float)(resolution - 1u);
    int3 center = int3(round(grid));
    int max_coord = (int)resolution - 1;

    float window = max(epsilon * 3.0f / max(direction_scale, 1.0e-8f), 1.0e-3f);
    float min_t = max(ray.origin_tmin.w, sdf_t - window);
    float max_t = min(limit, sdf_t + window);
    if (max_t < min_t) return false;

    bool found = false;
    float best_t = max_t;
    float3 best_normal = 0.0f;
    uint best_surface = INVALID_INDEX;

    [loop]
    for (int z = -1; z <= 1; ++z) {
        [loop]
        for (int y = -1; y <= 1; ++y) {
            [loop]
            for (int x = -1; x <= 1; ++x) {
                int3 q = clamp(center + int3(x, y, z), int3(0, 0, 0), int3(max_coord, max_coord, max_coord));
                uint surface_id = LocalSDFSurfaceIds[FutureSDFIndex(model, uint3(q))];
                if (surface_id == INVALID_INDEX) continue;

                float candidate_t;
                float3 candidate_normal;
                if (IntersectSceneTriangle(ray, surface_id, min_t, best_t, candidate_t, candidate_normal)) {
                    found = true;
                    best_t = candidate_t;
                    best_normal = candidate_normal;
                    best_surface = surface_id;
                }
            }
        }
    }

    if (!found || best_surface == INVALID_INDEX) return false;

    float3 world_position = ray.origin_tmin.xyz + ray.direction_tmax.xyz * best_t;
    hit = SurfaceFromTriangle(best_surface, world_position, best_normal, best_t, TRACE_SDF);
    if (hit.identity.x == INVALID_INDEX) hit.identity.x = model.meta.z;
    hit.meta.y = model.version.x;
    return hit.identity.z != INVALID_INDEX;
}
'''
s = s.replace(marker, '\n' + helpers + marker, 1)

# 3. Permanent local-SDF hits must be refined against exact triangles.
pattern = r"bool TraceLocalSDFModel\(TraceRay ray, uint model_index, float current_best, out SurfaceHit hit\) \{.*?\n\}\n\nbool TraceAllLocalSDFs"
replacement = r'''bool TraceLocalSDFModel(TraceRay ray, uint model_index, float current_best, out SurfaceHit hit) {
    hit = MakeSurfaceHit(TRACE_MISS, current_best);
    if (model_index >= Radiance.sdf_counts.x) return false;
    GPUSDFModel model = LocalSDFModels[model_index];
    float3 local_origin = TransformPoint(ray.origin_tmin.xyz, model.world_to_local);
    float3 local_direction = TransformVector(ray.direction_tmax.xyz, model.world_to_local);
    float direction_scale = length(local_direction);
    if (direction_scale <= 1.0e-8f) return false;
    float t_min, t_max;
    if (!IntersectAABB(local_origin, local_direction, model.bounds_min.xyz, model.bounds_max.xyz, t_min, t_max)) return false;
    float t = max(max(t_min, ray.origin_tmin.w), 0.0f);
    float limit = min(min(t_max, ray.direction_tmax.w), current_best);
    float3 extent = model.bounds_max.xyz - model.bounds_min.xyz;
    float cell_size = max(extent.x, max(extent.y, extent.z)) / max((float)model.meta.y, 1.0f);
    float epsilon = max(cell_size * 0.65f, Radiance.global_sdf_params.z);
    uint max_steps = max(Radiance.trace_limits.z, 1u);
    [loop]
    for (uint step = 0u; step < max_steps && t <= limit; ++step) {
        float3 lp = local_origin + local_direction * t;
        uint surface_id;
        float d = SampleLocalSDF(model, lp, surface_id);
        if (d <= epsilon) {
            float near_limit = ray.origin_tmin.w + epsilon * 1.5f;
            if (t <= near_limit) {
                t += max(epsilon * 1.5f / direction_scale, 1.0e-4f);
                continue;
            }

            SurfaceHit refined;
            if (RefineLocalSDFSurface(model, ray, lp, t, epsilon, direction_scale, limit, refined)) {
                hit = refined;
                return true;
            }

            // The coarse voxel identified a nearby surface, but not one that the
            // world-space ray actually intersects. Keep tracing instead of
            // shading the voxel approximation.
            t += max(epsilon * 0.5f / direction_scale, 1.0e-4f);
            continue;
        }
        t += max(d / direction_scale, epsilon * 0.25f / direction_scale);
    }
    return false;
}

bool TraceAllLocalSDFs'''
s = sub_once(s, pattern, replacement, 'TraceLocalSDFModel')

# 4. Visible screen hits use the exact same geometry-addressed radiance path.
pattern = r"\[numthreads\(8, 8, 1\)\]\nvoid CS_ScreenTrace\(uint3 dispatch_id : SV_DispatchThreadID\) \{.*?\n\}\n\n\[numthreads\(1, 1, 1\)\]\nvoid CS_ResetTraceQueue"
replacement = r'''[numthreads(8, 8, 1)]
void CS_ScreenTrace(uint3 dispatch_id : SV_DispatchThreadID) {
    uint ray_width, ray_height;
    ProbeRadianceOutput.GetDimensions(ray_width, ray_height);
    uint2 ray_pixel = dispatch_id.xy;
    if (ray_pixel.x >= ray_width || ray_pixel.y >= ray_height) return;

    TraceRay ray = BuildDiffuseProbeRay(ray_pixel);
    uint index = ray.destination;
    if (ray.type == TRACE_RAY_INACTIVE) {
        ScreenTraceHits[index] = MakeTraceHit(TRACE_INACTIVE, 0.0f);
        ScreenTraceOutput[ray_pixel] = 0.0f;
        ProbeRadianceOutput[ray_pixel] = 0.0f;
        return;
    }

    SurfaceHit surface = TraceScreenSurface(ray);
    if (surface.identity.w == TRACE_SCREEN) {
        TraceHit hit = MakeTraceHit(TRACE_SCREEN, surface.position_distance.w);
        hit.object_id = surface.identity.x;
        hit.hit_pixel = UnpackPixel(surface.meta.z);
        hit.confidence = surface.normal_confidence.w;
        ScreenTraceHits[index] = hit;

        float3 radiance = SurfaceReflectedRadiance(surface);
        ProbeRadianceOutput[ray_pixel] = float4(radiance, 1.0f);
        ScreenTraceOutput[ray_pixel] = float4(radiance, 1.0f);
        return;
    }

    ScreenTraceHits[index] = MakeTraceHit(TRACE_MISS, ray.direction_tmax.w);
    ProbeRadianceOutput[ray_pixel] = 0.0f;
    ScreenTraceOutput[ray_pixel] = 0.0f;
}

[numthreads(1, 1, 1)]
void CS_ResetTraceQueue'''
s = sub_once(s, pattern, replacement, 'CS_ScreenTrace')

# 5. Seed the permanent cache with the same canonical triangle normal used by
# screen/SDF hits instead of a camera-dependent raster normal.
old = '''    float3 position = ReconstructWorldPosition(pixel, depth);\n    float3 normal = normalize(TraceNormalRoughness.Load(int3(pixel, 0)).xyz);\n    float3 view = normalize(position - TraceFrame.camera_position.xyz);\n    if (dot(normal, view) > 0.0f) normal = -normal;\n\n    uint primitive_id = TracePrimitiveId.Load(int3(pixel, 0));\n    SurfaceHit hit = SurfaceFromTriangle(primitive_id, position, normal, 0.0f, TRACE_SCREEN);'''
new = '''    float3 position = ReconstructWorldPosition(pixel, depth);\n    float3 raster_normal = normalize(TraceNormalRoughness.Load(int3(pixel, 0)).xyz);\n    float3 view = normalize(position - TraceFrame.camera_position.xyz);\n    if (dot(raster_normal, view) > 0.0f) raster_normal = -raster_normal;\n\n    uint primitive_id = TracePrimitiveId.Load(int3(pixel, 0));\n    float3 normal = SceneTriangleFacingNormal(primitive_id, view);\n    if (dot(normal, normal) <= 1.0e-8f) normal = raster_normal;\n    SurfaceHit hit = SurfaceFromTriangle(primitive_id, position, normal, 0.0f, TRACE_SCREEN);'''
if s.count(old) != 1:
    raise SystemExit(f'CS_RadianceDirect normal block count = {s.count(old)}')
s = s.replace(old, new, 1)

p.write_text(s)
