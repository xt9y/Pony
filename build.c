#include <cbuild.h>

#include <stdio.h>
#include <stdlib.h>

static void format_sources(void) {
#if defined(_WIN32)
    const char *command = "powershell -NoProfile -ExecutionPolicy Bypass -Command "
                          "\"py .\\felix-format -i "
                          "(Get-ChildItem -File *.c,*.h | ForEach-Object FullName)\"";
#else
    const char *command = "./felix-format -i *.c *.h";
#endif

    if (system(command) != 0) {
        fprintf(stderr, "felix-format failed\n");
        exit(1);
    }
}

static void compile_shaders(void) {
#if defined(_WIN32)

    if (system("if not exist build\\shaders mkdir build\\shaders") != 0) {
        fprintf(stderr, "Could not create shader directory\n");
        exit(1);
    }

    const char *commands[] = {
        "dxc -spirv -fspv-target-env=vulkan1.2 -fvk-use-dx-layout -Zpr -O3 "
        "-fvk-s-shift 0 0 -fvk-t-shift 16 0 -fvk-b-shift 32 0 -fvk-u-shift 48 0 "
        "-T vs_6_6 -E VS_GBuffer shader.hlsl "
        "-Fo build\\shaders\\gbuffer.vs.spv",

        "dxc -spirv -fspv-target-env=vulkan1.2 -fvk-use-dx-layout -Zpr -O3 "
        "-fvk-s-shift 0 0 -fvk-t-shift 16 0 -fvk-b-shift 32 0 -fvk-u-shift 48 0 "
        "-T ps_6_6 -E PS_GBuffer shader.hlsl "
        "-Fo build\\shaders\\gbuffer.ps.spv",

        "dxc -spirv -fspv-target-env=vulkan1.2 -fvk-use-dx-layout -Zpr -O3 "
        "-fvk-s-shift 0 0 -fvk-t-shift 16 0 -fvk-b-shift 32 0 -fvk-u-shift 48 0 "
        "-T vs_6_6 -E VS_Present shader.hlsl "
        "-Fo build\\shaders\\present.vs.spv",

        "dxc -spirv -fspv-target-env=vulkan1.2 -fvk-use-dx-layout -Zpr -O3 "
        "-fvk-s-shift 0 0 -fvk-t-shift 16 0 -fvk-b-shift 32 0 -fvk-u-shift 48 0 "
        "-T ps_6_6 -E PS_Present shader.hlsl "
        "-Fo build\\shaders\\present.ps.spv"
    };

#else

    if (system("mkdir -p build/shaders") != 0) {
        fprintf(stderr, "Could not create shader directory\n");
        exit(1);
    }

    const char *commands[] = {
        "dxc -spirv -fspv-target-env=vulkan1.2 -fvk-use-dx-layout -Zpr -O3 "
        "-fvk-s-shift 0 0 -fvk-t-shift 16 0 -fvk-b-shift 32 0 -fvk-u-shift 48 0 "
        "-T vs_6_6 -E VS_GBuffer shader.hlsl "
        "-Fo build/shaders/gbuffer.vs.spv",

        "dxc -spirv -fspv-target-env=vulkan1.2 -fvk-use-dx-layout -Zpr -O3 "
        "-fvk-s-shift 0 0 -fvk-t-shift 16 0 -fvk-b-shift 32 0 -fvk-u-shift 48 0 "
        "-T ps_6_6 -E PS_GBuffer shader.hlsl "
        "-Fo build/shaders/gbuffer.ps.spv",

        "dxc -spirv -fspv-target-env=vulkan1.2 -fvk-use-dx-layout -Zpr -O3 "
        "-fvk-s-shift 0 0 -fvk-t-shift 16 0 -fvk-b-shift 32 0 -fvk-u-shift 48 0 "
        "-T vs_6_6 -E VS_Present shader.hlsl "
        "-Fo build/shaders/present.vs.spv",

        "dxc -spirv -fspv-target-env=vulkan1.2 -fvk-use-dx-layout -Zpr -O3 "
        "-fvk-s-shift 0 0 -fvk-t-shift 16 0 -fvk-b-shift 32 0 -fvk-u-shift 48 0 "
        "-T ps_6_6 -E PS_Present shader.hlsl "
        "-Fo build/shaders/present.ps.spv"
    };

#endif

    for (size_t i = 0; i < sizeof(commands) / sizeof(commands[0]); ++i) {
        if (system(commands[i]) != 0) {
            fprintf(stderr, "SPIR-V shader compilation failed\n");
            exit(1);
        }
    }
}

void build(C_Build *b) {
    format_sources();
    compile_shaders();

    C_Target *app = c_executable(b, "game");

    c_sources(app, "main.c");
    c_sources(app, "init.c");
    c_sources(app, "glb.c");
    c_sources(app, "gltf.c");
    c_sources(app, "scene.c");
    c_sources(app, "gpu.c");
    c_sources(app, "render.c");

    C_Dependency *nri = c_git(b, "NRI", "https://github.com/NVIDIA-RTX/NRI.git", "main");

    c_dep_cmake(nri);
    c_dep_cmake_option(nri, "-DNRI_STATIC_LIBRARY=OFF");
    c_dep_cmake_option(nri, "-DNRI_ENABLE_VK_SUPPORT=ON");
    c_dep_cmake_option(nri, "-DNRI_ENABLE_D3D11_SUPPORT=OFF");
    c_dep_cmake_option(nri, "-DNRI_ENABLE_D3D12_SUPPORT=OFF");
    c_dep_cmake_option(nri, "-DNRI_ENABLE_WGPU_SUPPORT=OFF");
    c_dep_cmake_option(nri, "-DNRI_ENABLE_NONE_SUPPORT=OFF");
    c_dep_cmake_option(nri, "-DNRI_ENABLE_VALIDATION_SUPPORT=OFF");
    c_dep_cmake_option(nri, "-DNRI_ENABLE_NVTX_SUPPORT=OFF");
    c_dep_cmake_option(nri, "-DNRI_ENABLE_DEBUG_NAMES_AND_ANNOTATIONS=OFF");

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

    c_link_system(app, "SDL3");

#if !defined(_WIN32)
    c_link_system(app, "m");
#endif

    c_default_target(b, app);
}
