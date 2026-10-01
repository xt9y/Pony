#include "game.h"

_Static_assert(WORLD_PROBE_CAPACITY == 8192u, "world probe capacity");
_Static_assert(WORLD_PROBE_HASH_CAPACITY == 16384u, "world hash capacity");
_Static_assert(WORLD_PROBE_DIRECTION_SIZE == 4u, "world direction size");
_Static_assert(WORLD_PROBE_DIRECTION_COUNT == 16u, "world direction count");
_Static_assert(WORLD_PROBE_BANK_COUNT == 2u, "world bank count");
_Static_assert(WORLD_PROBE_UPDATES_PER_FRAME == 64u, "world updates/frame");
_Static_assert(WORLD_PROBE_HASH_PROBE_LIMIT == 8u, "world hash probe limit");
_Static_assert(sizeof(WORLD_PROBE_STATE) == 64u, "WORLD_PROBE_STATE ABI");
_Static_assert(sizeof(RADIANCE_CONSTANTS) == 256u, "RADIANCE_CONSTANTS ABI");

typedef bool (*WORLD_BUILDER)(
    const GLOBAL_SDF_DATA *, const RADIANCE_SCENE_DATA *, const GPU_OBJECT *, uint32_t,
    WORLD_PROBE_STATE *, uint32_t, uint32_t *, uint32_t *, uint32_t,
    float, float, float, float
);

static WORLD_BUILDER world_builder = sdf_build_world_probes;

int main(void) {
    return world_builder == 0;
}
