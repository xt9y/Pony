#include "game.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

VEC3 v3(float x, float y, float z) {
    return (VEC3){x, y, z};
}

VEC3 v3_add(VEC3 a, VEC3 b) {
    return v3(a.x + b.x, a.y + b.y, a.z + b.z);
}

VEC3 v3_sub(VEC3 a, VEC3 b) {
    return v3(a.x - b.x, a.y - b.y, a.z - b.z);
}

VEC3 v3_scale(VEC3 v, float s) {
    return v3(v.x * s, v.y * s, v.z * s);
}

float v3_dot(VEC3 a, VEC3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

VEC3 v3_cross(VEC3 a, VEC3 b) {
    return v3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

float v3_len_sq(VEC3 v) {
    return v3_dot(v, v);
}

VEC3 v3_normalize(VEC3 v) {
    const float length = sqrtf(v3_len_sq(v));

    return length > FLT_EPSILON ? v3_scale(v, 1.0f / length) : v3(0, 0, 0);
}

void mesh_free(MESH *m) {
    if (!m) return;
    free(m->vertices.buffer);
    free(m->faces.buffer);
    memset(m, 0, sizeof(*m));
}
