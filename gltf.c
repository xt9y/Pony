#include "dustmite.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define GLTF_MAX_NODE_DEPTH 128u

typedef struct gm4 {
    float m[16];
} gm4;

static bool token_number(const glb_doc *d, int token, float *value) {

    double n;
    if (!glb_number(d, token, &n)) return false;

    *value = (float)n;
    return true;
}

static bool token_u32(const glb_doc *d, int token, uint32_t *value) {

    double n;
    if (!glb_number(d, token, &n) || n < 0.0 || n > 4294967295.0 || floor(n) != n) return false;

    *value = (uint32_t)n;
    return true;
}

static bool token_vec(const glb_doc *d, int token, float *value, size_t count) {

    if (token < 0 || glb_count(d, token) != count) return false;

    for (size_t i = 0; i < count; ++i)
        if (!token_number(d, glb_at(d, token, i), &value[i])) return false;

    return true;
}

static bool token_string_copy(const glb_doc *d, int token, char *dst, size_t capacity) {
    const char *text;
    size_t length;

    if (!dst || !capacity || !glb_string(d, token, &text, &length)) return false;
    if (length >= capacity) length = capacity - 1u;

    memcpy(dst, text, length);
    dst[length] = 0;
    return true;
}

static gm4 m_identity(void) {
    gm4 r = {{0}};
    r.m[0] = r.m[5] = r.m[10] = r.m[15] = 1.0f;
    return r;
}

static gm4 m_mul(gm4 a, gm4 b) {
    gm4 r = {{0}};

    for (int c = 0; c < 4; ++c) {

        for (int row = 0; row < 4; ++row) {

            r.m[c * 4 + row] = a.m[row] * b.m[c * 4] + a.m[4 + row] * b.m[c * 4 + 1] + a.m[8 + row] * b.m[c * 4 + 2] + a.m[12 + row] * b.m[c * 4 + 3];
        }

    }

    return r;
}

static gm4 m_trs(float tx, float ty, float tz, float qx, float qy, float qz, float qw, float sx, float sy, float sz) {
    const float xx = qx * qx, yy = qy * qy, zz = qz * qz;
    const float xy = qx * qy, xz = qx * qz, yz = qy * qz;
    const float wx = qw * qx, wy = qw * qy, wz = qw * qz;


    gm4 r = m_identity();
    
    r.m[0] = (1.0f - 2.0f * (yy + zz)) * sx;
    r.m[1] = (2.0f * (xy + wz)) * sx;
    r.m[2] = (2.0f * (xz - wy)) * sx;
    r.m[4] = (2.0f * (xy - wz)) * sy;
    r.m[5] = (1.0f - 2.0f * (xx + zz)) * sy;
    r.m[6] = (2.0f * (yz + wx)) * sy;
    r.m[8] = (2.0f * (xz + wy)) * sz;
    r.m[9] = (2.0f * (yz - wx)) * sz;
    
    r.m[10] = (1.0f - 2.0f * (xx + yy)) * sz;
    r.m[12] = tx;
    r.m[13] = ty;
    r.m[14] = tz;
    
    return r;
}

static gm4 node_local(const glb_doc *d, int node) {

    int t = glb_get(d, node, "matrix");
    if (t >= 0 && glb_count(d, t) == 16u) {

        gm4 m = m_identity();
        bool ok = true;
        for (int i = 0; i < 16; ++i)
            ok &= token_number(d, glb_at(d, t, (size_t)i), &m.m[i]);

        if (ok) return m;
    }

    float tr[3] = {0, 0, 0};
    float q[4] = {0, 0, 0, 1};
    float s[3] = {1, 1, 1};

    t = glb_get(d, node, "translation");
    if (t >= 0) (void)token_vec(d, t, tr, 3);

    t = glb_get(d, node, "rotation");
    if (t >= 0) (void)token_vec(d, t, q, 4);

    t = glb_get(d, node, "scale");
    if (t >= 0) (void)token_vec(d, t, s, 3);

    return m_trs(tr[0], tr[1], tr[2], q[0], q[1], q[2], q[3], s[0], s[1], s[2]);
}

static vec3 m_point(gm4 m, vec3 p) {

    float x = m.m[0] * p.x + m.m[4] * p.y + m.m[8] * p.z + m.m[12];
    float y = m.m[1] * p.x + m.m[5] * p.y + m.m[9] * p.z + m.m[13];
    float z = m.m[2] * p.x + m.m[6] * p.y + m.m[10] * p.z + m.m[14];
    float w = m.m[3] * p.x + m.m[7] * p.y + m.m[11] * p.z + m.m[15];
    
    if (w != 0.0f && w != 1.0f) {
        x /= w;
        y /= w;
        z /= w;
    }

    return v3(x, y, z);
}

static float m_det3(gm4 m) {

    return m.m[0] * (m.m[5] * m.m[10] - m.m[9] * m.m[6]) - m.m[4] * (m.m[1] * m.m[10] - m.m[9] * m.m[2]) + m.m[8] * (m.m[1] * m.m[6] - m.m[5] * m.m[2]);

}

static vec3 m_normal(gm4 m, vec3 n) {

    const float a00 = m.m[0], a01 = m.m[4], a02 = m.m[8];
    const float a10 = m.m[1], a11 = m.m[5], a12 = m.m[9];
    const float a20 = m.m[2], a21 = m.m[6], a22 = m.m[10];
    const float det = a00 * (a11 * a22 - a12 * a21) - a01 * (a10 * a22 - a12 * a20) + a02 * (a10 * a21 - a11 * a20);
    if (fabsf(det) < 1.0e-12f) return v3_normalize(v3(a00 * n.x + a01 * n.y + a02 * n.z, a10 * n.x + a11 * n.y + a12 * n.z, a20 * n.x + a21 * n.y + a22 * n.z));


    const float inv = 1.0f / det;
    const vec3 r = v3(((a11 * a22 - a12 * a21) * n.x + (a12 * a20 - a10 * a22) * n.y + (a10 * a21 - a11 * a20) * n.z) * inv,
                      ((a02 * a21 - a01 * a22) * n.x + (a00 * a22 - a02 * a20) * n.y + (a01 * a20 - a00 * a21) * n.z) * inv,
                      ((a01 * a12 - a02 * a11) * n.x + (a02 * a10 - a00 * a12) * n.y + (a00 * a11 - a01 * a10) * n.z) * inv);

    return v3_normalize(r);
}

static bool reserve_vertices(gltf_scene *s, size_t count) {

    if (count <= s->vertex_capacity) return true;
    size_t capacity = s->vertex_capacity ? s->vertex_capacity : 1024u;
    while (capacity < count) {

        if (capacity > SIZE_MAX / 2u) return false;
        capacity *= 2u;
    }

    gltf_vertex *p = realloc(s->vertices, capacity * sizeof(*p));
    if (!p) return false;

    s->vertices = p;
    s->vertex_capacity = capacity;

    return true;
}

static bool push_triangle(gltf_scene *s, gltf_vertex a, gltf_vertex b, gltf_vertex c) {

    if (!reserve_vertices(s, s->vertex_count + 3u)) return false;

    const vec3 face = v3_normalize(v3_cross(v3_sub(b.position, a.position), v3_sub(c.position, a.position)));

    if (v3_len_sq(a.normal) < 1.0e-8f) a.normal = face;
    if (v3_len_sq(b.normal) < 1.0e-8f) b.normal = face;
    if (v3_len_sq(c.normal) < 1.0e-8f) c.normal = face;
    
    s->vertices[s->vertex_count++] = a;
    s->vertices[s->vertex_count++] = b;
    s->vertices[s->vertex_count++] = c;
    return true;
}

static bool primitive_index(const glb_accessor *indices, size_t i, size_t vertex_count, uint32_t *out) {

    if (indices) {
        if (!glb_accessor_u32(indices, i, out)) return false;
        return *out < vertex_count;
    }

    if (i > UINT32_MAX || i >= vertex_count) return false;
    *out = (uint32_t)i;
    return true;
}

static bool open_attribute(const glb_doc *d, int attrs, const char *name, size_t count, uint32_t min_components, glb_accessor *out) {

    uint32_t index;
    const int token = glb_get(d, attrs, name);
    if (token < 0 || !token_u32(d, token, &index)) return false;
    if (!glb_accessor_open(d, index, out) || out->sparse_token >= 0 || out->count != count || out->components < min_components) return false;
    return true;
}

static bool read_vertex(const glb_accessor *pos, const glb_accessor *normal, const glb_accessor *uv, uint32_t index, gm4 world, uint32_t material, gltf_vertex *out) {
    float x, y, z;

    if (!glb_accessor_f32(pos, index, 0, &x) || !glb_accessor_f32(pos, index, 1, &y) || !glb_accessor_f32(pos, index, 2, &z)) return false;

    memset(out, 0, sizeof(*out));
    out->position = m_point(world, v3(x, y, z));
    out->material = material;

    if (normal) {
        float nx, ny, nz;
        if (!glb_accessor_f32(normal, index, 0, &nx) || !glb_accessor_f32(normal, index, 1, &ny) || !glb_accessor_f32(normal, index, 2, &nz)) return false;
        out->normal = m_normal(world, v3(nx, ny, nz));
    }

    if (uv) {
        if (!glb_accessor_f32(uv, index, 0, &out->u) || !glb_accessor_f32(uv, index, 1, &out->v)) return false;
    }

    return true;
}

static bool emit_triangle(gltf_scene *s, const glb_accessor *pos, const glb_accessor *normal, const glb_accessor *uv, gm4 world, uint32_t material, uint32_t a, uint32_t b,
                          uint32_t c) {

    if (a == b || b == c || c == a) return true;
    gltf_vertex va, vb, vc;


    if (!read_vertex(pos, normal, uv, a, world, material, &va) || !read_vertex(pos, normal, uv, b, world, material, &vb) || !read_vertex(pos, normal, uv, c, world, material, &vc))
        return false;

    return push_triangle(s, va, vb, vc);
}

static bool extract_primitive(const glb_doc *d, int prim, gm4 world, gltf_scene *s) {

    const int attrs = glb_get(d, prim, "attributes");
    uint32_t pos_index;

    if (attrs < 0 || !token_u32(d, glb_get(d, attrs, "POSITION"), &pos_index)) return true;

    glb_accessor pos;
    if (!glb_accessor_open(d, pos_index, &pos) || pos.components < 3 || pos.sparse_token >= 0) return false;


    glb_accessor normal_store, uv_store;
    glb_accessor *normal = open_attribute(d, attrs, "NORMAL", pos.count, 3u, &normal_store) ? &normal_store : NULL;
    glb_accessor *uv = open_attribute(d, attrs, "TEXCOORD_0", pos.count, 2u, &uv_store) ? &uv_store : NULL;


    uint32_t material = s->default_material;
    const int material_token = glb_get(d, prim, "material");
    uint32_t source_material;
    if (material_token >= 0 && token_u32(d, material_token, &source_material) && source_material < s->default_material) material = source_material;


    glb_accessor idx_store;
    glb_accessor *indices = NULL;

    uint32_t idx_index;
    const int index_token = glb_get(d, prim, "indices");
    if (index_token >= 0) {
        if (!token_u32(d, index_token, &idx_index) || !glb_accessor_open(d, idx_index, &idx_store) || idx_store.sparse_token >= 0) return false;
        indices = &idx_store;
    }

    const size_t count = indices ? indices->count : pos.count;
    uint32_t mode = 4u;
    const int mode_token = glb_get(d, prim, "mode");
    if (mode_token >= 0 && !token_u32(d, mode_token, &mode)) return false;
    const bool flip = m_det3(world) < 0.0f;

    if (mode == 4u) {

        for (size_t i = 0; i + 2u < count; i += 3u) {

            uint32_t a, b, c;

            if (!primitive_index(indices, i, pos.count, &a) || !primitive_index(indices, i + 1u, pos.count, &b) || !primitive_index(indices, i + 2u, pos.count, &c)) return false;
            if (flip) {
                uint32_t t = b;
                b = c;
                c = t;
            }

            if (!emit_triangle(s, &pos, normal, uv, world, material, a, b, c)) return false;

        }

    } else if (mode == 5u) {

        for (size_t i = 0; i + 2u < count; ++i) {

            uint32_t a, b, c;

            if (!primitive_index(indices, i, pos.count, &a) || !primitive_index(indices, i + 1u, pos.count, &b) || !primitive_index(indices, i + 2u, pos.count, &c)) return false;
            if (i & 1u) {
                uint32_t t = a;
                a = b;
                b = t;
            }
            if (flip) {
                uint32_t t = b;
                b = c;
                c = t;
            }

            if (!emit_triangle(s, &pos, normal, uv, world, material, a, b, c)) return false;

        }

    } else if (mode == 6u && count >= 3u) {

        uint32_t a;

        if (!primitive_index(indices, 0, pos.count, &a)) return false;
        for (size_t i = 1u; i + 1u < count; ++i) {

            uint32_t b, c;

            if (!primitive_index(indices, i, pos.count, &b) || !primitive_index(indices, i + 1u, pos.count, &c)) return false;
            if (flip) {
                uint32_t t = b;
                b = c;
                c = t;
            }

            if (!emit_triangle(s, &pos, normal, uv, world, material, a, b, c)) return false;
        
        }
    }

    return true;
}

static bool extract_mesh(const glb_doc *d, uint32_t index, gm4 world, gltf_scene *s) {

    const int meshes = glb_get(d, 0, "meshes");
    const int mesh = glb_at(d, meshes, index);

    if (mesh < 0) return false;


    const int primitives = glb_get(d, mesh, "primitives");
    for (size_t i = 0; i < glb_count(d, primitives); ++i)
        if (!extract_primitive(d, glb_at(d, primitives, i), world, s)) return false;

    return true;
}

static bool extract_node(const glb_doc *d, uint32_t index, gm4 parent, gltf_scene *s, unsigned depth) {

    if (depth > GLTF_MAX_NODE_DEPTH) return false;
    const int nodes = glb_get(d, 0, "nodes");
    const int node = glb_at(d, nodes, index);
    if (node < 0) return false;
    const gm4 world = m_mul(parent, node_local(d, node));


    uint32_t mesh_index;
    const int mesh_token = glb_get(d, node, "mesh");
    if (mesh_token >= 0 && (!token_u32(d, mesh_token, &mesh_index) || !extract_mesh(d, mesh_index, world, s))) return false;


    const int children = glb_get(d, node, "children");
    for (size_t i = 0; i < glb_count(d, children); ++i) {

        uint32_t child;
        if (!token_u32(d, glb_at(d, children, i), &child) || !extract_node(d, child, world, s, depth + 1u)) return false;
    }
    return true;
}

static int32_t texture_index(const glb_doc *d, int object, const char *name) {

    const int texture = glb_get(d, object, name);
    if (texture < 0) return -1;
    uint32_t index;

    return token_u32(d, glb_get(d, texture, "index"), &index) ? (int32_t)index : -1;
}

static void material_defaults(gltf_material *m) {

    memset(m, 0, sizeof(*m));
    
    m->base_color[0] = m->base_color[1] = m->base_color[2] = m->base_color[3] = 1.0f;
    m->metallic = 1.0f;
    m->roughness = 1.0f;
    m->normal_scale = 1.0f;
    m->occlusion_strength = 1.0f;
    m->base_color_texture = -1;
    m->metallic_roughness_texture = -1;
    m->normal_texture = -1;
    m->occlusion_texture = -1;
    m->emissive_texture = -1;
}

static bool extract_materials(const glb_doc *d, gltf_scene *s) {

    const int materials = glb_get(d, 0, "materials");
    const size_t source_count = glb_count(d, materials);
    
    if (source_count > UINT32_MAX - 1u) return false;
    s->material_count = (uint32_t)source_count + 1u;
    s->default_material = (uint32_t)source_count;
    s->materials = calloc(s->material_count, sizeof(*s->materials));
    if (!s->materials) return false;


    for (uint32_t i = 0; i < s->material_count; ++i)
        material_defaults(&s->materials[i]);
    
    for (uint32_t i = 0; i < (uint32_t)source_count; ++i) {

        gltf_material *m = &s->materials[i];
        const int material = glb_at(d, materials, i);
        const int pbr = glb_get(d, material, "pbrMetallicRoughness");
        if (pbr >= 0) {

            const int color = glb_get(d, pbr, "baseColorFactor");
            if (color >= 0) (void)token_vec(d, color, m->base_color, 4u);
            
            (void)token_number(d, glb_get(d, pbr, "metallicFactor"), &m->metallic);
            (void)token_number(d, glb_get(d, pbr, "roughnessFactor"), &m->roughness);
            
            m->base_color_texture = texture_index(d, pbr, "baseColorTexture");
            m->metallic_roughness_texture = texture_index(d, pbr, "metallicRoughnessTexture");
        }


        const int emissive = glb_get(d, material, "emissiveFactor");
        if (emissive >= 0) (void)token_vec(d, emissive, m->emissive, 3u);
        
        m->normal_texture = texture_index(d, material, "normalTexture");
        m->occlusion_texture = texture_index(d, material, "occlusionTexture");
        m->emissive_texture = texture_index(d, material, "emissiveTexture");


        const int normal = glb_get(d, material, "normalTexture");
        if (normal >= 0) (void)token_number(d, glb_get(d, normal, "scale"), &m->normal_scale);
        
        const int occlusion = glb_get(d, material, "occlusionTexture");
        if (occlusion >= 0) (void)token_number(d, glb_get(d, occlusion, "strength"), &m->occlusion_strength);


        const int extensions = glb_get(d, material, "extensions");
        const int strength = glb_get(d, extensions, "KHR_materials_emissive_strength");
        
        float emissive_strength = 1.0f;
        if (strength >= 0 && token_number(d, glb_get(d, strength, "emissiveStrength"), &emissive_strength)) {
            m->emissive[0] *= emissive_strength;
            m->emissive[1] *= emissive_strength;
            m->emissive[2] *= emissive_strength;
        }

    }

    return true;
}

static bool extract_textures(const glb_doc *d, gltf_scene *s) {

    const int textures = glb_get(d, 0, "textures");

    const size_t count = glb_count(d, textures);
    if (count > UINT32_MAX) return false;

    s->texture_count = (uint32_t)count;
    if (!count) return true;

    s->textures = malloc(count * sizeof(*s->textures));
    if (!s->textures) return false;
    for (uint32_t i = 0; i < s->texture_count; ++i) {

        s->textures[i].image = -1;
        const int texture = glb_at(d, textures, i);
        uint32_t image;
        if (token_u32(d, glb_get(d, texture, "source"), &image)) s->textures[i].image = (int32_t)image;
    }

    return true;

}

static bool extract_images(const glb_doc *d, gltf_scene *s) {

    const int images = glb_get(d, 0, "images");
    const size_t count = glb_count(d, images);
    if (count > UINT32_MAX) return false;

    s->image_count = (uint32_t)count;
    if (!count) return true;

    s->images = calloc(count, sizeof(*s->images));
    if (!s->images) return false;

    for (uint32_t i = 0; i < s->image_count; ++i) {

        const int image = glb_at(d, images, i);
        uint32_t view;
        if (token_u32(d, glb_get(d, image, "bufferView"), &view)) (void)glb_buffer_view(d, view, &s->images[i].bytes, NULL);
        
        (void)token_string_copy(d, glb_get(d, image, "mimeType"), s->images[i].mime, sizeof(s->images[i].mime));
    }

    return true;
}

void gltf_free(gltf_scene *s) {

    if (!s) return;
    
    free(s->vertices);
    free(s->materials);
    free(s->textures);
    free(s->images);
    
    memset(s, 0, sizeof(*s));
}

bool gltf_extract(const glb_doc *d, gltf_scene *s) {

    if (!d || !s || glb_root(d) < 0) return false;
    memset(s, 0, sizeof(*s));

    if (!extract_materials(d, s) || !extract_textures(d, s) || !extract_images(d, s)) goto fail;


    const gm4 identity = m_identity();
    const int scenes = glb_get(d, 0, "scenes");
    if (scenes >= 0 && glb_count(d, scenes)) {

        uint32_t scene_index = 0;
        const int selected = glb_get(d, 0, "scene");
        if (selected >= 0 && !token_u32(d, selected, &scene_index)) goto fail;
        
        const int scene = glb_at(d, scenes, scene_index);
        const int roots = glb_get(d, scene, "nodes");
        
        for (size_t i = 0; i < glb_count(d, roots); ++i) {

            uint32_t node;
            if (!token_u32(d, glb_at(d, roots, i), &node) || !extract_node(d, node, identity, s, 0u)) goto fail;
        }

    } else {

        const int nodes = glb_get(d, 0, "nodes");
        const size_t count = glb_count(d, nodes);
        bool *child = calloc(count, sizeof(*child));

        if (!child && count) goto fail;
        for (size_t i = 0; i < count; ++i) {

            const int node = glb_at(d, nodes, i);
            const int children = glb_get(d, node, "children");
            
            for (size_t j = 0; j < glb_count(d, children); ++j) {

                uint32_t index;
                if (token_u32(d, glb_at(d, children, j), &index) && index < count) child[index] = true;
            }

        }

        for (size_t i = 0; i < count; ++i) {

            if (!child[i] && !extract_node(d, (uint32_t)i, identity, s, 0u)) {
                free(child);
                goto fail;
            }

        }

        free(child);
    }

    if (!s->vertex_count) goto fail;
    return true;


fail:
    gltf_free(s);
    return false;
}
