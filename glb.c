#include "game.h"

#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#define GLB_MAGIC 0x46546c67u
#define GLB_JSON 0x4e4f534au
#define GLB_BIN 0x004e4942u
#define GLB_MAX_NODE_DEPTH 128u

typedef struct GM4 {
    float m[16];
} GM4;

static void set_error(GLB_DOC *d, const char *fmt, ...) {

    if (!d) return;

    va_list ap;

    va_start(ap, fmt);
    vsnprintf(d->error, sizeof(d->error), fmt, ap);
    va_end(ap);
}

static void release_file_data(GLB_DOC *d) {

    if (!d || !d->data) return;

#if defined(_WIN32)
    free(d->data);
#else
    (void)munmap(d->data, d->data_size);
#endif

    d->data = NULL;
    d->data_size = 0;
}

static void release_keep_error(GLB_DOC *d) {

    char error[sizeof(d->error)];

    memcpy(error, d->error, sizeof(error));

    free(d->tokens);
    release_file_data(d);

    memset(d, 0, sizeof(*d));
    memcpy(d->error, error, sizeof(error));
}

static uint32_t rd32(const unsigned char *p) {

    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static uint16_t rd16(const unsigned char *p) {

    return (uint16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8));
}

static bool tok_reserve(GLB_DOC *d, uint32_t n) {

    if (n <= d->token_capacity) return true;

    uint32_t cap = d->token_capacity ? d->token_capacity : 256u;

    while (cap < n) {
        if (cap > UINT32_MAX / 2u) return false;
        cap *= 2u;
    }

    GLB_TOKEN *p = realloc(d->tokens, (size_t)cap * sizeof(*p));

    if (!p) return false;

    d->tokens = p;
    d->token_capacity = cap;

    return true;
}

static int tok_add(GLB_DOC *d, GLB_TOKEN_TYPE type, uint32_t start, int parent) {

    if (!tok_reserve(d, d->token_count + 1u)) return -1;

    int idx = (int)d->token_count++;

    d->tokens[idx] = (GLB_TOKEN){.start = start, .end = start, .parent = parent, .children = 0, .type = type};

    if (parent >= 0) d->tokens[parent].children++;

    return idx;
}

static bool tokenize(GLB_DOC *d) {

    int parent = -1;
    const char *s = d->json;
    const size_t n = d->json_size;

    for (size_t i = 0; i < n;) {

        unsigned char c = (unsigned char)s[i];

        if (isspace(c) || c == '\0' || c == ',' || c == ':') {
            ++i;

            continue;
        }

        if (c == '{' || c == '[') {

            int t = tok_add(d, c == '{' ? GLB_TOKEN_OBJECT : GLB_TOKEN_ARRAY, (uint32_t)i, parent);

            if (t < 0) {
                set_error(d, "out of memory while tokenizing JSON");

                return false;
            }

            parent = t;
            ++i;

            continue;
        }

        if (c == '}' || c == ']') {

            if (parent < 0 || (c == '}' && d->tokens[parent].type != GLB_TOKEN_OBJECT) || (c == ']' && d->tokens[parent].type != GLB_TOKEN_ARRAY)) {
                set_error(d, "malformed JSON near byte %zu", i);

                return false;
            }

            d->tokens[parent].end = (uint32_t)(i + 1u);

            parent = d->tokens[parent].parent;
            ++i;

            continue;
        }

        if (c == '"') {

            const size_t begin = ++i;
            bool closed = false;

            while (i < n) {

                if (s[i] == '\\') {
                    if (++i >= n) break;
                    ++i;

                    continue;
                }

                if (s[i] == '"') {
                    closed = true;

                    break;
                }
                ++i;
            }

            if (!closed) {
                set_error(d, "unterminated JSON string");

                return false;
            }

            int t = tok_add(d, GLB_TOKEN_STRING, (uint32_t)begin, parent);

            if (t < 0) {
                set_error(d, "out of memory while tokenizing JSON");

                return false;
            }

            d->tokens[t].end = (uint32_t)i;
            ++i;

            continue;
        }

        const size_t begin = i;

        while (i < n) {

            c = (unsigned char)s[i];

            if (isspace(c) || c == '\0' || c == ',' || c == ']' || c == '}' || c == ':') {
                break;
            }

            ++i;
        }

        if (i == begin) {
            set_error(d, "malformed JSON near byte %zu", i);

            return false;
        }

        int t = tok_add(d, GLB_TOKEN_PRIMITIVE, (uint32_t)begin, parent);

        if (t < 0) {
            set_error(d, "out of memory while tokenizing JSON");

            return false;
        }

        d->tokens[t].end = (uint32_t)i;
    }

    if (parent != -1 || !d->token_count || d->tokens[0].type != GLB_TOKEN_OBJECT) {
        set_error(d, "incomplete or invalid glTF JSON root");

        return false;
    }

    return true;
}

bool glb_load(GLB_DOC *d, const char *path) {

    if (!d || !path) return false;
    memset(d, 0, sizeof(*d));

#if defined(_WIN32)
    FILE *f = fopen(path, "rb");

    if (!f) {
        set_error(d, "%s: %s", path, strerror(errno));

        return false;
    }

    if (fseek(f, 0, SEEK_END) != 0) {
        set_error(d, "failed to seek %s", path);
        fclose(f);

        return false;
    }

    const long end = ftell(f);

    if (end < 12 || fseek(f, 0, SEEK_SET) != 0) {
        set_error(d, "%s is not a valid GLB", path);
        fclose(f);

        return false;
    }

    d->data_size = (size_t)end;
    d->data = malloc(d->data_size);

    if (!d->data) {
        set_error(d, "out of memory reading %s", path);
        fclose(f);

        return false;
    }

    if (fread(d->data, 1, d->data_size, f) != d->data_size) {
        set_error(d, "failed to read %s", path);
        fclose(f);
        release_keep_error(d);

        return false;
    }

    fclose(f);
#else
    const int fd = open(path, O_RDONLY);

    if (fd < 0) {
        set_error(d, "%s: %s", path, strerror(errno));

        return false;
    }

    struct stat info;

    if (fstat(fd, &info) != 0) {
        const int error = errno;

        close(fd);
        set_error(d, "failed to stat %s: %s", path, strerror(error));

        return false;
    }

    if (info.st_size < 12 || (uint64_t)info.st_size > (uint64_t)SIZE_MAX) {
        close(fd);
        set_error(d, "%s is not a valid GLB", path);

        return false;
    }

    d->data_size = (size_t)info.st_size;

    void *mapping = mmap(NULL, d->data_size, PROT_READ, MAP_PRIVATE, fd, 0);
    const int map_error = errno;

    close(fd);

    if (mapping == MAP_FAILED) {
        d->data_size = 0;
        set_error(d, "failed to map %s: %s", path, strerror(map_error));

        return false;
    }

    d->data = mapping;
#endif

    if (rd32(d->data) != GLB_MAGIC || rd32(d->data + 4) != 2u) {
        set_error(d, "%s is not glTF 2.0 GLB", path);
        release_keep_error(d);

        return false;
    }

    const uint32_t declared = rd32(d->data + 8);

    if (declared > d->data_size || declared < 12u) {
        set_error(d, "invalid GLB length");
        release_keep_error(d);

        return false;
    }

    size_t off = 12;

    while (off + 8 <= declared) {

        const uint32_t len = rd32(d->data + off);
        const uint32_t type = rd32(d->data + off + 4);

        off += 8;

        if ((size_t)len > (size_t)declared - off) {
            set_error(d, "GLB chunk exceeds file length");
            release_keep_error(d);

            return false;
        }

        if (type == GLB_JSON && !d->json) {
            d->json = (const char *)(d->data + off);
            d->json_size = len;
        } else if (type == GLB_BIN && !d->bin) {
            d->bin = d->data + off;
            d->bin_size = len;
        }

        off += len;
    }

    if (!d->json) {
        set_error(d, "GLB has no JSON chunk");
        release_keep_error(d);

        return false;
    }

    if (!tokenize(d)) {
        release_keep_error(d);

        return false;
    }

    return true;
}

void glb_free(GLB_DOC *d) {

    if (!d) return;
    free(d->tokens);
    release_file_data(d);
    memset(d, 0, sizeof(*d));
}

const char *glb_error(const GLB_DOC *d) {

    return d ? d->error : "invalid glb_doc";
}

int glb_root(const GLB_DOC *d) {
    return d && d->token_count ? 0 : -1;
}

static bool token_eq(const GLB_DOC *d, int t, const char *text) {

    if (!d || !text || t < 0 || (uint32_t)t >= d->token_count || d->tokens[t].type != GLB_TOKEN_STRING) {
        return false;
    }

    const size_t len = d->tokens[t].end - d->tokens[t].start;

    return strlen(text) == len && memcmp(d->json + d->tokens[t].start, text, len) == 0;
}

int glb_get(const GLB_DOC *d, int object, const char *key) {

    if (!d || object < 0 || (uint32_t)object >= d->token_count || d->tokens[object].type != GLB_TOKEN_OBJECT || !key) {
        return -1;
    }

    int key_token = -1;

    for (uint32_t i = (uint32_t)object + 1u; i < d->token_count; ++i) {

        const GLB_TOKEN *t = &d->tokens[i];

        if (t->start >= d->tokens[object].end) break;

        if (t->parent != object) continue;

        if (key_token < 0) {
            key_token = (int)i;
        } else {
            if (token_eq(d, key_token, key)) return (int)i;

            key_token = -1;
        }
    }

    return -1;
}

int glb_at(const GLB_DOC *d, int array, size_t index) {

    if (!d || array < 0 || (uint32_t)array >= d->token_count || d->tokens[array].type != GLB_TOKEN_ARRAY) {
        return -1;
    }

    size_t at = 0;

    for (uint32_t i = (uint32_t)array + 1u; i < d->token_count; ++i) {

        const GLB_TOKEN *t = &d->tokens[i];

        if (t->start >= d->tokens[array].end) break;

        if (t->parent != array) continue;

        if (at++ == index) return (int)i;
    }

    return -1;
}

size_t glb_count(const GLB_DOC *d, int token) {

    if (!d || token < 0 || (uint32_t)token >= d->token_count) return 0;

    if (d->tokens[token].type == GLB_TOKEN_OBJECT) return d->tokens[token].children / 2u;

    if (d->tokens[token].type == GLB_TOKEN_ARRAY) return d->tokens[token].children;

    return 0;
}

bool glb_string(const GLB_DOC *d, int token, const char **data, size_t *length) {

    if (!d || token < 0 || (uint32_t)token >= d->token_count || d->tokens[token].type != GLB_TOKEN_STRING) {
        return false;
    }

    if (data) *data = d->json + d->tokens[token].start;

    if (length) *length = d->tokens[token].end - d->tokens[token].start;

    return true;
}

bool glb_number(const GLB_DOC *d, int token, double *value) {

    if (!d || !value || token < 0 || (uint32_t)token >= d->token_count || d->tokens[token].type != GLB_TOKEN_PRIMITIVE) {
        return false;
    }

    const size_t len = d->tokens[token].end - d->tokens[token].start;

    if (!len || len >= 64) return false;

    char tmp[64];

    memcpy(tmp, d->json + d->tokens[token].start, len);

    tmp[len] = 0;

    char *end = NULL;
    errno = 0;
    const double v = strtod(tmp, &end);

    if (errno || end != tmp + len) return false;
    *value = v;

    return true;
}

bool glb_boolean(const GLB_DOC *d, int token, bool *value) {

    if (!d || !value || token < 0 || (uint32_t)token >= d->token_count || d->tokens[token].type != GLB_TOKEN_PRIMITIVE) {
        return false;
    }

    const size_t len = d->tokens[token].end - d->tokens[token].start;
    const char *p = d->json + d->tokens[token].start;

    if (len == 4 && memcmp(p, "true", 4) == 0) {
        *value = true;
        return true;
    }

    if (len == 5 && memcmp(p, "false", 5) == 0) {
        *value = false;
        return true;
    }

    return false;
}

static bool tok_size(const GLB_DOC *d, int token, size_t *value) {

    double n;

    if (!glb_number(d, token, &n) || n < 0.0 || n > (double)SIZE_MAX || floor(n) != n) {
        return false;
    }

    *value = (size_t)n;

    return true;
}

static bool tok_u32(const GLB_DOC *d, int token, uint32_t *value) {
    size_t n;

    if (!tok_size(d, token, &n) || n > UINT32_MAX) return false;
    *value = (uint32_t)n;

    return true;
}

bool glb_buffer_view(const GLB_DOC *d, size_t index, GLB_SPAN *span, size_t *stride) {

    if (!d || !span || !d->bin) return false;

    const int views = glb_get(d, 0, "bufferViews");
    const int view = glb_at(d, views, index);

    if (view < 0) return false;

    uint32_t buffer = 0;
    int t = glb_get(d, view, "buffer");

    if (t >= 0 && !tok_u32(d, t, &buffer)) return false;

    if (buffer != 0) return false;

    size_t off = 0, len = 0, step = 0;

    t = glb_get(d, view, "byteOffset");

    if (t >= 0 && !tok_size(d, t, &off)) return false;

    t = glb_get(d, view, "byteLength");

    if (t < 0 || !tok_size(d, t, &len)) return false;

    t = glb_get(d, view, "byteStride");

    if (t >= 0 && !tok_size(d, t, &step)) return false;

    if (off > d->bin_size || len > d->bin_size - off) return false;

    span->data = d->bin + off;
    span->size = len;

    if (stride) *stride = step;

    return true;
}

static uint32_t component_count(const GLB_DOC *d, int type_token) {

    const char *p = NULL;
    size_t n = 0;

    if (!glb_string(d, type_token, &p, &n)) return 0;

#define TYPE_IS(s) (n == sizeof(s) - 1u && memcmp(p, s, sizeof(s) - 1u) == 0)
    if (TYPE_IS("SCALAR")) return 1;

    if (TYPE_IS("VEC2")) return 2;

    if (TYPE_IS("VEC3")) return 3;

    if (TYPE_IS("VEC4") || TYPE_IS("MAT2")) return 4;

    if (TYPE_IS("MAT3")) return 9;

    if (TYPE_IS("MAT4")) return 16;
#undef TYPE_IS

    return 0;
}

static size_t component_size(uint32_t t) {

    switch (t) {

        case 5120:

        case 5121:
            return 1;

        case 5122:

        case 5123:

            return 2;

        case 5125:

        case 5126:

            return 4;

        default:
            return 0;
    }
}

bool glb_accessor_open(const GLB_DOC *d, size_t index, GLB_ACCESSOR *out) {

    if (!d || !out) return false;
    memset(out, 0, sizeof(*out));

    const int accessors = glb_get(d, 0, "accessors");
    const int a = glb_at(d, accessors, index);

    if (a < 0) return false;

    uint32_t view_index = 0, ctype = 0;
    size_t count = 0, byte_offset = 0;

    int t = glb_get(d, a, "bufferView");

    if (t < 0 || !tok_u32(d, t, &view_index)) return false;

    t = glb_get(d, a, "componentType");

    if (t < 0 || !tok_u32(d, t, &ctype)) return false;

    t = glb_get(d, a, "count");

    if (t < 0 || !tok_size(d, t, &count)) return false;

    const uint32_t comps = component_count(d, glb_get(d, a, "type"));
    const size_t csize = component_size(ctype);

    if (!comps || !csize) return false;

    t = glb_get(d, a, "byteOffset");

    if (t >= 0 && !tok_size(d, t, &byte_offset)) return false;

    GLB_SPAN view;
    size_t stride = 0;

    if (!glb_buffer_view(d, view_index, &view, &stride)) return false;

    const size_t packed = csize * comps;

    if (!stride) stride = packed;

    if (stride < packed || byte_offset > view.size) return false;

    if (count && ((count - 1u) > (SIZE_MAX - packed - byte_offset) / stride || byte_offset + (count - 1u) * stride + packed > view.size)) {
        return false;
    }

    bool normalized = false;
    t = glb_get(d, a, "normalized");

    if (t >= 0 && !glb_boolean(d, t, &normalized)) return false;

    out->data = view.data + byte_offset;
    out->count = count;
    out->stride = stride;
    out->component_type = ctype;
    out->components = comps;
    out->normalized = normalized;
    out->token = a;
    out->sparse_token = glb_get(d, a, "sparse");

    return true;
}

static float read_component(const unsigned char *p, uint32_t type, bool normalized) {

    switch (type) {

        case 5120: {
            const int8_t v = (int8_t)p[0];

            if (!normalized) return (float)v;

            const float f = (float)v / 127.0f;

            return f < -1.0f ? -1.0f : f;
        }

        case 5121: {
            return normalized ? (float)p[0] / 255.0f : (float)p[0];
        }

        case 5122: {
            const int16_t v = (int16_t)rd16(p);

            if (!normalized) return (float)v;

            const float f = (float)v / 32767.0f;

            return f < -1.0f ? -1.0f : f;
        }

        case 5123: {
            const uint16_t v = rd16(p);

            return normalized ? (float)v / 65535.0f : (float)v;
        }

        case 5125: {
            const uint32_t v = rd32(p);

            return normalized ? (float)((double)v / 4294967295.0) : (float)v;
        }

        case 5126: {
            const uint32_t u = rd32(p);
            float v;
            memcpy(&v, &u, sizeof(v));

            return v;
        }

        default:
            return 0.0f;
    }
}

bool glb_accessor_f32(const GLB_ACCESSOR *a, size_t element, uint32_t component, float *value) {

    if (!a || !value || !a->data || element >= a->count || component >= a->components || a->sparse_token >= 0) {
        return false;
    }

    const size_t cs = component_size(a->component_type);

    if (!cs) return false;

    *value = read_component(a->data + element * a->stride + component * cs, a->component_type, a->normalized);
    return true;
}

bool glb_accessor_u32(const GLB_ACCESSOR *a, size_t element, uint32_t *value) {

    if (!a || !value || !a->data || element >= a->count || a->components != 1 || a->sparse_token >= 0) {
        return false;
    }

    const unsigned char *p = a->data + element * a->stride;

    switch (a->component_type) {

        case 5121:
            *value = p[0];
            return true;

        case 5123:
            *value = rd16(p);
            return true;

        case 5125:
            *value = rd32(p);
            return true;

        default:
            return false;
    }
}

static GM4 m_identity(void) {
    GM4 r = {{0}};
    r.m[0] = r.m[5] = r.m[10] = r.m[15] = 1.0f;

    return r;
}

static GM4 m_mul(GM4 a, GM4 b) {
    GM4 r = {{0}};

    for (int c = 0; c < 4; ++c) {
        for (int row = 0; row < 4; ++row) {
            r.m[c * 4 + row] = a.m[row] * b.m[c * 4] + a.m[4 + row] * b.m[c * 4 + 1] + a.m[8 + row] * b.m[c * 4 + 2] + a.m[12 + row] * b.m[c * 4 + 3];
        }
    }

    return r;
}

static GM4 m_trs(float tx, float ty, float tz, float qx, float qy, float qz, float qw, float sx, float sy, float sz) {

    const float xx = qx * qx, yy = qy * qy, zz = qz * qz;
    const float xy = qx * qy, xz = qx * qz, yz = qy * qz;
    const float wx = qw * qx, wy = qw * qy, wz = qw * qz;

    GM4 r = m_identity();

    r.m[0] = (1 - 2 * (yy + zz)) * sx;
    r.m[1] = (2 * (xy + wz)) * sx;
    r.m[2] = (2 * (xz - wy)) * sx;
    r.m[4] = (2 * (xy - wz)) * sy;
    r.m[5] = (1 - 2 * (xx + zz)) * sy;
    r.m[6] = (2 * (yz + wx)) * sy;
    r.m[8] = (2 * (xz + wy)) * sz;
    r.m[9] = (2 * (yz - wx)) * sz;

    r.m[10] = (1 - 2 * (xx + yy)) * sz;
    r.m[12] = tx;
    r.m[13] = ty;
    r.m[14] = tz;

    return r;
}

static VEC3 m_point(GM4 m, VEC3 p) {

    float x = m.m[0] * p.x + m.m[4] * p.y + m.m[8] * p.z + m.m[12];
    float y = m.m[1] * p.x + m.m[5] * p.y + m.m[9] * p.z + m.m[13];
    float z = m.m[2] * p.x + m.m[6] * p.y + m.m[10] * p.z + m.m[14];
    float w = m.m[3] * p.x + m.m[7] * p.y + m.m[11] * p.z + m.m[15];

    if (w != 0.0f && w != 1.0f) {
        x /= w;
        y /= w;
        z /= w;
    }

    return (VEC3){x, y, z};
}

static float m_det3(GM4 m) {

    return m.m[0] * (m.m[5] * m.m[10] - m.m[9] * m.m[6]) - m.m[4] * (m.m[1] * m.m[10] - m.m[9] * m.m[2]) + m.m[8] * (m.m[1] * m.m[6] - m.m[5] * m.m[2]);
}

static bool json_float(const GLB_DOC *d, int token, float *out) {
    double v;

    if (!glb_number(d, token, &v)) return false;
    *out = (float)v;
    return true;
}

static bool json_vec(const GLB_DOC *d, int token, float *v, size_t n) {

    if (token < 0 || glb_count(d, token) != n) return false;

    for (size_t i = 0; i < n; ++i) {
        if (!json_float(d, glb_at(d, token, i), &v[i])) return false;
    }

    return true;
}

static GM4 node_local(const GLB_DOC *d, int node) {

    int t = glb_get(d, node, "matrix");

    if (t >= 0 && glb_count(d, t) == 16) {
        GM4 m = m_identity();
        bool ok = true;

        for (int i = 0; i < 16; ++i)
            ok &= json_float(d, glb_at(d, t, (size_t)i), &m.m[i]);

        if (ok) return m;
    }

    float tr[3] = {0, 0, 0};
    float q[4] = {0, 0, 0, 1};
    float s[3] = {1, 1, 1};

    t = glb_get(d, node, "translation");

    if (t >= 0) (void)json_vec(d, t, tr, 3);

    t = glb_get(d, node, "rotation");

    if (t >= 0) (void)json_vec(d, t, q, 4);

    t = glb_get(d, node, "scale");

    if (t >= 0) (void)json_vec(d, t, s, 3);

    return m_trs(tr[0], tr[1], tr[2], q[0], q[1], q[2], q[3], s[0], s[1], s[2]);
}

static bool vec_reserve(VECTOR *v, size_t n) {

    if (n <= v->capacity) return true;

    size_t cap = v->capacity ? v->capacity : 256u;

    while (cap < n) {
        if (cap > SIZE_MAX / 2u) return false;
        cap *= 2u;
    }

    if (v->type_size && cap > SIZE_MAX / v->type_size) return false;

    void *p = realloc(v->buffer, cap * v->type_size);

    if (!p) return false;
    v->buffer = p;
    v->capacity = cap;

    return true;
}

static bool mesh_point_push(MESH *m, VEC3 p) {

    VECTOR *v = &m->vertices;

    if (!vec_reserve(v, v->count + 1u)) return false;
    ((POINT *)v->buffer)[v->count++] = (POINT){p};

    return true;
}

static VEC3 sub3(VEC3 a, VEC3 b) {

    return (VEC3){a.x - b.x, a.y - b.y, a.z - b.z};
}

static VEC3 cross3(VEC3 a, VEC3 b) {

    return (VEC3){a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

static VEC3 norm3(VEC3 a) {

    const float l = sqrtf(a.x * a.x + a.y * a.y + a.z * a.z);

    return l > 0.0f ? (VEC3){a.x / l, a.y / l, a.z / l} : (VEC3){0, 1, 0};
}

static bool mesh_face_push(MESH *m, uint32_t a, uint32_t b, uint32_t c) {

    if (a == b || b == c || c == a || a >= m->vertices.count || b >= m->vertices.count || c >= m->vertices.count) {
        return true;
    }

    VECTOR *v = &m->faces;

    if (!vec_reserve(v, v->count + 1u)) return false;

    POINT *p = m->vertices.buffer;
    const VEC3 n = norm3(cross3(sub3(p[b].p, p[a].p), sub3(p[c].p, p[a].p)));
    ((MESH_FACE *)v->buffer)[v->count++] = (MESH_FACE){{a, b, c}, n};

    return true;
}

static bool primitive_index(const GLB_ACCESSOR *idx, size_t i, size_t vertex_count, uint32_t *out) {

    if (idx) {
        if (!glb_accessor_u32(idx, i, out)) return false;

        return *out < vertex_count;
    }

    if (i > UINT32_MAX || i >= vertex_count) return false;
    *out = (uint32_t)i;
    return true;
}

static bool extract_primitive(const GLB_DOC *d, int prim, GM4 world, MESH *out) {

    const int attrs = glb_get(d, prim, "attributes");

    uint32_t pos_index;

    if (attrs < 0 || !tok_u32(d, glb_get(d, attrs, "POSITION"), &pos_index)) {
        return true;
    }

    GLB_ACCESSOR pos;

    if (!glb_accessor_open(d, pos_index, &pos) || pos.components < 3 || pos.sparse_token >= 0) {
        return false;
    }

    if (out->vertices.count > UINT32_MAX - pos.count) return false;

    const uint32_t base = (uint32_t)out->vertices.count;

    for (size_t i = 0; i < pos.count; ++i) {

        float x, y, z;

        if (!glb_accessor_f32(&pos, i, 0, &x) || !glb_accessor_f32(&pos, i, 1, &y) || !glb_accessor_f32(&pos, i, 2, &z)) {
            return false;
        }

        if (!mesh_point_push(out, m_point(world, (VEC3){x, y, z}))) return false;
    }

    GLB_ACCESSOR idx_store;
    GLB_ACCESSOR *indices = NULL;

    uint32_t idx_index;
    const int it = glb_get(d, prim, "indices");

    if (it >= 0) {
        if (!tok_u32(d, it, &idx_index) || !glb_accessor_open(d, idx_index, &idx_store) || idx_store.sparse_token >= 0) {
            return false;
        }

        indices = &idx_store;
    }

    const size_t icount = indices ? indices->count : pos.count;
    uint32_t mode = 4;
    const int mt = glb_get(d, prim, "mode");

    if (mt >= 0 && !tok_u32(d, mt, &mode)) return false;

    const bool flip = m_det3(world) < 0.0f;

    if (mode == 4) {

        for (size_t i = 0; i + 2 < icount; i += 3) {

            uint32_t a, b, c;

            if (!primitive_index(indices, i, pos.count, &a) || !primitive_index(indices, i + 1, pos.count, &b) || !primitive_index(indices, i + 2, pos.count, &c)) return false;

            if (flip) {
                uint32_t t = b;

                b = c;
                c = t;
            }

            if (!mesh_face_push(out, base + a, base + b, base + c)) return false;
        }

    } else if (mode == 5) {

        for (size_t i = 0; i + 2 < icount; ++i) {

            uint32_t a, b, c;

            if (!primitive_index(indices, i, pos.count, &a) || !primitive_index(indices, i + 1, pos.count, &b) || !primitive_index(indices, i + 2, pos.count, &c)) return false;

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

            if (!mesh_face_push(out, base + a, base + b, base + c)) return false;
        }

    } else if (mode == 6 && icount >= 3) {

        uint32_t a;

        if (!primitive_index(indices, 0, pos.count, &a)) return false;

        for (size_t i = 1; i + 1 < icount; ++i) {

            uint32_t b, c;

            if (!primitive_index(indices, i, pos.count, &b) || !primitive_index(indices, i + 1, pos.count, &c)) return false;

            if (flip) {
                uint32_t t = b;

                b = c;
                c = t;
            }

            if (!mesh_face_push(out, base + a, base + b, base + c)) return false;
        }
    }

    return true;
}

static bool extract_mesh_index(const GLB_DOC *d, uint32_t index, GM4 world, MESH *out) {

    const int meshes = glb_get(d, 0, "meshes");
    const int obj = glb_at(d, meshes, index);

    if (obj < 0) return false;

    const int prims = glb_get(d, obj, "primitives");

    if (prims < 0) return true;

    for (size_t i = 0; i < glb_count(d, prims); ++i) {
        if (!extract_primitive(d, glb_at(d, prims, i), world, out)) return false;
    }

    return true;
}

static bool extract_node(const GLB_DOC *d, uint32_t index, GM4 parent, MESH *out, unsigned depth) {

    if (depth > GLB_MAX_NODE_DEPTH) return false;

    const int nodes = glb_get(d, 0, "nodes");
    const int node = glb_at(d, nodes, index);

    if (node < 0) return false;

    const GM4 world = m_mul(parent, node_local(d, node));
    uint32_t mesh_index;
    const int m = glb_get(d, node, "mesh");

    if (m >= 0 && (!tok_u32(d, m, &mesh_index) || !extract_mesh_index(d, mesh_index, world, out))) {
        return false;
    }

    const int children = glb_get(d, node, "children");

    for (size_t i = 0; i < glb_count(d, children); ++i) {
        uint32_t child;

        if (!tok_u32(d, glb_at(d, children, i), &child) || !extract_node(d, child, world, out, depth + 1u)) {
            return false;
        }
    }

    return true;
}

static void mesh_bounds(MESH *m) {

    if (!m->vertices.count) return;

    POINT *p = m->vertices.buffer;
    VEC3 mn = p[0].p;
    VEC3 mx = p[0].p;

    for (size_t i = 1; i < m->vertices.count; ++i) {

        const VEC3 v = p[i].p;

        if (v.x < mn.x) mn.x = v.x;

        if (v.y < mn.y) mn.y = v.y;

        if (v.z < mn.z) mn.z = v.z;

        if (v.x > mx.x) mx.x = v.x;

        if (v.y > mx.y) mx.y = v.y;

        if (v.z > mx.z) mx.z = v.z;
    }

    m->bounds.min = mn;
    m->bounds.max = mx;
    m->bounds.center = (VEC3){(mn.x + mx.x) * 0.5f, (mn.y + mx.y) * 0.5f, (mn.z + mx.z) * 0.5f};
    m->bounds.extents = (VEC3){(mx.x - mn.x) * 0.5f, (mx.y - mn.y) * 0.5f, (mx.z - mn.z) * 0.5f};
}

static void mesh_partial_free(MESH *m) {

    free(m->vertices.buffer);
    free(m->faces.buffer);
    memset(m, 0, sizeof(*m));
}

bool glb_extract_mesh(const GLB_DOC *d, MESH *out) {

    if (!d || !out || !d->token_count) return false;
    memset(out, 0, sizeof(*out));
    out->vertices.type_size = sizeof(POINT);
    out->faces.type_size = sizeof(MESH_FACE);

    const GM4 identity = m_identity();
    const int scenes = glb_get(d, 0, "scenes");

    if (scenes >= 0 && glb_count(d, scenes)) {

        uint32_t scene_index = 0;
        const int selected = glb_get(d, 0, "scene");

        if (selected >= 0 && !tok_u32(d, selected, &scene_index)) {
            mesh_partial_free(out);

            return false;
        }

        const int scene = glb_at(d, scenes, scene_index);
        const int roots = glb_get(d, scene, "nodes");

        for (size_t i = 0; i < glb_count(d, roots); ++i) {

            uint32_t node;

            if (!tok_u32(d, glb_at(d, roots, i), &node) || !extract_node(d, node, identity, out, 0)) {
                mesh_partial_free(out);

                return false;
            }
        }

    } else {

        const int nodes = glb_get(d, 0, "nodes");
        const size_t n = glb_count(d, nodes);
        bool *child = calloc(n, sizeof(bool));

        if (!child && n) {
            mesh_partial_free(out);

            return false;
        }

        for (size_t i = 0; i < n; ++i) {

            const int node = glb_at(d, nodes, i);
            const int children = glb_get(d, node, "children");

            for (size_t j = 0; j < glb_count(d, children); ++j) {

                uint32_t ci;

                if (tok_u32(d, glb_at(d, children, j), &ci) && ci < n) child[ci] = true;
            }
        }

        for (size_t i = 0; i < n; ++i) {

            if (!child[i] && !extract_node(d, (uint32_t)i, identity, out, 0)) {
                free(child);
                mesh_partial_free(out);

                return false;
            }
        }

        free(child);
    }

    mesh_bounds(out);

    return out->vertices.count > 0 && out->faces.count > 0;
}
