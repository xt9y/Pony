#include <cbuild.h>

void build(C_Build *b) {
    C_Target *app = c_executable(b, "game");

    c_sources(app, "main.c");
    c_sources(app, "init.c");
    c_sources(app, "glb.c");
    c_sources(app, "gltf.c");
    c_sources(app, "bvh.c");
    c_sources(app, "lmap.c");
    c_sources(app, "gpu.c");
    c_sources(app, "render.c");
    c_sources(app, "bake.c");
    c_sources(app, "cache.c");
    c_sources(app, "beam.c");

    C_Dependency *nri = c_git(b, "NRI", "https://github.com/NVIDIA-RTX/NRI.git", "main");
    c_dep_cmake(nri);
    c_dep_cmake_option(nri, "-DNRI_STATIC_LIBRARY=OFF");
    c_dep_cmake_option(nri, "-DNRI_ENABLE_VK_SUPPORT=ON");
    c_dep_cmake_option(nri, "-DNRI_ENABLE_VALIDATION_SUPPORT=ON");
    c_dep_cmake_option(nri, "-DNRI_ENABLE_NONE_SUPPORT=OFF");
    c_dep_cmake_option(nri, "-DNRI_ENABLE_NVTX_SUPPORT=OFF");
    c_dep_cmake_option(nri, "-DNRI_ENABLE_NIS_SDK=OFF");
    c_dep_cmake_option(nri, "-DNRI_ENABLE_IMGUI_EXTENSION=OFF");
    c_dep_cmake_option(nri, "-DNRI_ENABLE_WGPU_SUPPORT=OFF");
#if defined(__APPLE__)
    c_dep_cmake_option(nri, "-DCMAKE_PREFIX_PATH=/opt/homebrew");
#endif
    c_dep_include(nri, "Include");
    c_dep_link(nri, "NRI");
    c_use(app, nri);

    c_standard(app, C_STANDARD_C11);
    c_warnings_strict(app);

#if defined(__APPLE__)
    c_include(app, "/opt/homebrew/include");
    c_link_flag(app, "-L/opt/homebrew/lib");
    c_link_flag(app, "-Wl,-rpath,/opt/homebrew/lib");
#endif

    c_include(app, "/usr/local/include");
    c_link_flag(app, "-L/usr/local/lib");
#if defined(__APPLE__)
    c_link_flag(app, "-Wl,-rpath,/usr/local/lib");
#endif

    c_link_system(app, "SDL3_image");
    c_link_system(app, "SDL3_shadercross");
    c_link_system(app, "SDL3");
    c_link_system(app, "m");

    c_default_target(b, app);
}
