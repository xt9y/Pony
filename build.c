#include <cbuild.h>

void build(C_Build *b) {
    C_Target *app = c_executable(b, "dustmite");

    c_sources(app, "main.c");
    c_sources(app, "init.c");
    c_sources(app, "glb.c");
    c_sources(app, "gltf.c");
    c_sources(app, "bvh.c");
    c_sources(app, "lmap.c");
    c_sources(app, "fx_vision.c");
    c_sources(app, "gpu.c");
    c_sources(app, "render.c");
    c_sources(app, "cache.c");
    c_sources(app, "beam.c");

    c_standard(app, C_STANDARD_C11);
    c_warnings_strict(app);

    /* NRI is kept as a normal external checkout at deps/NRI. */
    c_include(app, "deps/NRI/Include");
    c_link_flag(app, "-Ldeps/NRI/_Bin");

#if defined(__APPLE__)
    c_include(app, "/opt/homebrew/include");
    c_link_flag(app, "-L/opt/homebrew/lib");
    c_link_flag(app, "-Wl,-rpath,/opt/homebrew/lib");
    c_link_flag(app, "-Wl,-rpath,@loader_path/../../deps/NRI/_Bin");
#endif

    c_include(app, "/usr/local/include");
    c_link_flag(app, "-L/usr/local/lib");
#if defined(__APPLE__)
    c_link_flag(app, "-Wl,-rpath,/usr/local/lib");
#endif

    c_link_system(app, "NRI");
    c_link_system(app, "SDL3_image");
    c_link_system(app, "SDL3_shadercross");
    c_link_system(app, "SDL3");
    c_link_system(app, "m");

    c_default_target(b, app);
}
