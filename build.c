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

/*
 * Build-time shader compiler: Slang (slangc); the runtime only loads SPIR-V.
 * Vulkan SDK 1.3.296+ bundles Slang. The full SDK is needed for development,
 * while a finished game only needs a Vulkan-capable driver/runtime.
 * Windows: winget install --id KhronosGroup.VulkanSDK -e
 * macOS: install the current LunarG SDK from https://vulkan.lunarg.com/sdk/home/
 *        (brew install vulkan-tools molten-vk installs the Vulkan stack, not slangc).
 * Arch/CachyOS: sudo pacman -S vulkan-devel, then install LunarG SDK or Slang for slangc.
 */
static void compile_shaders(void) {
#if defined(_WIN32)
    if (system("where slangc >nul 2>&1") != 0) {
        fprintf(stderr, "slangc was not found on PATH\n");
        exit(1);
    }

    const char *make_directory = "if not exist build\\shaders mkdir build\\shaders";
#else
    if (system("command -v slangc >/dev/null 2>&1") != 0) {
        fprintf(stderr, "slangc was not found on PATH\n");
        exit(1);
    }

    const char *make_directory = "mkdir -p build/shaders";
#endif

    if (system(make_directory) != 0) {
        fprintf(stderr, "Could not create shader directory\n");
        exit(1);
    }

    const char *commands[] = {
        "slangc shader.hlsl -entry VS_GBuffer -stage vertex -target spirv -profile sm_6_6 -capability spirv_1_5 -matrix-layout-row-major -fvk-use-dx-layout -O3 -o "
        "build/shaders/gbuffer.vs.spv",
        "slangc shader.hlsl -entry PS_GBuffer -stage fragment -target spirv -profile sm_6_6 -capability spirv_1_5 -matrix-layout-row-major -fvk-use-dx-layout -O3 -o "
        "build/shaders/gbuffer.ps.spv",
        "slangc shader.hlsl -entry VS_Present -stage vertex -target spirv -profile sm_6_6 -capability spirv_1_5 -matrix-layout-row-major -fvk-use-dx-layout -O3 -o "
        "build/shaders/present.vs.spv",
        "slangc shader.hlsl -entry PS_Present -stage fragment -target spirv -profile sm_6_6 -capability spirv_1_5 -matrix-layout-row-major -fvk-use-dx-layout -O3 "
        "-DFINAL_GI -o build/shaders/present.ps.spv",
        "slangc shader.hlsl -entry CS_HZB -stage compute -target spirv -profile sm_6_6 -capability spirv_1_5 -matrix-layout-row-major -fvk-use-dx-layout -O3 -o "
        "build/shaders/hzb.cs.spv",
        "slangc shader.hlsl -entry CS_DirectRadiance -stage compute -target spirv -profile sm_6_6 -capability spirv_1_5 -matrix-layout-row-major -fvk-use-dx-layout -O3 -o "
        "build/shaders/direct_radiance.cs.spv",
        "slangc shader.hlsl -entry CS_SurfaceCacheUpdate -stage compute -target spirv -profile sm_6_6 -capability spirv_1_5 -matrix-layout-row-major -fvk-use-dx-layout -O3 -o "
        "build/shaders/surface_cache.cs.spv",
        "slangc shader.hlsl -entry CS_ScreenTrace -stage compute -target spirv -profile sm_6_6 -capability spirv_1_5 -matrix-layout-row-major -fvk-use-dx-layout -O3 -o "
        "build/shaders/screen_trace.cs.spv",
        "slangc shader.hlsl -entry CS_ResetTraceQueue -stage compute -target spirv -profile sm_6_6 -capability spirv_1_5 -matrix-layout-row-major -fvk-use-dx-layout -O3 -o "
        "build/shaders/trace_reset.cs.spv",
        "slangc shader.hlsl -entry CS_CompactTraceMisses -stage compute -target spirv -profile sm_6_6 -capability spirv_1_5 -matrix-layout-row-major -fvk-use-dx-layout -O3 -o "
        "build/shaders/trace_compact.cs.spv",
        "slangc shader.hlsl -entry CS_BuildTraceDispatchArgs -stage compute -target spirv -profile sm_6_6 -capability spirv_1_5 -matrix-layout-row-major -fvk-use-dx-layout -O3 -o "
        "build/shaders/trace_args.cs.spv",
        "slangc shader.hlsl -entry CS_SDFTrace -stage compute -target spirv -profile sm_6_6 -capability spirv_1_5 -matrix-layout-row-major -fvk-use-dx-layout -O3 -o "
        "build/shaders/sdf_trace.cs.spv",
        "slangc shader.hlsl -entry CS_ScreenProbes -stage compute -target spirv -profile sm_6_6 -capability spirv_1_5 -matrix-layout-row-major -fvk-use-dx-layout -O3 -o "
        "build/shaders/screen_probes.cs.spv"
    };

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
    c_sources(app, "sdf.c");
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
