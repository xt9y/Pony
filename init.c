#include "dustmite.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

vec3 v3(float x, float y, float z) {
    return (vec3){x, y, z};
}

vec3 v3_add(vec3 a, vec3 b) {
    return v3(a.x + b.x, a.y + b.y, a.z + b.z);
}

vec3 v3_sub(vec3 a, vec3 b) {
    return v3(a.x - b.x, a.y - b.y, a.z - b.z);
}

vec3 v3_scale(vec3 v, float s) {
    return v3(v.x * s, v.y * s, v.z * s);
}

float v3_dot(vec3 a, vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

vec3 v3_cross(vec3 a, vec3 b) {
    return v3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

float v3_len_sq(vec3 v) {
    return v3_dot(v, v);
}

vec3 v3_normalize(vec3 v) {
    const float length = sqrtf(v3_len_sq(v));
    return length > FLT_EPSILON ? v3_scale(v, 1.0f / length) : v3(0, 0, 0);
}

void mesh_free(mesh *m) {
    if (!m) return;
    free(m->vertices.buffer);
    free(m->faces.buffer);
    memset(m, 0, sizeof(*m));
}
