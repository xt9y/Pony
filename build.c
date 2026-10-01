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
typedef struct SHADER_JOB {
    const char *entry;
    const char *stage;
    const char *output;
    const char *define;
} SHADER_JOB;

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

    static const SHADER_JOB jobs[] = {
        {"VS_GBuffer", "vertex", "gbuffer.vs.spv", NULL},
        {"PS_GBuffer", "fragment", "gbuffer.ps.spv", NULL},
        {"PS_GBufferFull", "fragment", "gbuffer_full.ps.spv", NULL},
        {"VS_Present", "vertex", "present.vs.spv", NULL},
        {"PS_Present", "fragment", "present.ps.spv", "FINAL_GI"},
        {"PS_PresentRuntime", "fragment", "present_runtime.ps.spv", NULL},

        {"CS_HZB", "compute", "hzb.cs.spv", NULL},

        {"CS_RadianceDirect", "compute", "radiance_direct.cs.spv", NULL},
        {"CS_ResetWavefront", "compute", "radiance_wave_reset.cs.spv", NULL},
        {"CS_ClassifyRayBudgets", "compute", "radiance_budget.cs.spv", NULL},
        {"CS_GenerateProbeRays", "compute", "radiance_generate.cs.spv", NULL},
        {"CS_WavefrontScreenTrace", "compute", "radiance_screen.cs.spv", NULL},
        {"CS_WavefrontDynamicTrace", "compute", "radiance_dynamic.cs.spv", NULL},
        {"CS_WavefrontLocalTrace", "compute", "radiance_local.cs.spv", NULL},
        {"CS_WavefrontGlobalTrace", "compute", "radiance_global.cs.spv", NULL},
        {"CS_ShadeRayHits", "compute", "radiance_shade.cs.spv", NULL},
        {"CS_EmissiveGather", "compute", "radiance_emissive.cs.spv", NULL},
        {"CS_ResolveDirectionalProbes", "compute", "radiance_probe_resolve.cs.spv", NULL},
        {"CS_ReprojectScreenProbes", "compute", "radiance_probe_temporal.cs.spv", NULL},
        {"CS_SpatialReuseScreenProbes", "compute", "radiance_probe_spatial.cs.spv", NULL},
        {"CS_CommitScreenProbeHistory", "compute", "radiance_probe_history.cs.spv", NULL},
        {"CS_UpdateWorldRadianceCache", "compute", "radiance_world_cache.cs.spv", NULL},
        {"CS_InvalidateRadiance", "compute", "radiance_invalidate.cs.spv", NULL},
        {"CS_ReflectionTrace", "compute", "radiance_reflections.cs.spv", NULL}
    };

    for (size_t i = 0; i < sizeof(jobs) / sizeof(jobs[0]); ++i) {
        char command[1024];
        const SHADER_JOB *job = &jobs[i];
        const int written = snprintf(
            command,
            sizeof(command),
            "slangc shader.hlsl -entry %s -stage %s -target spirv -profile sm_6_6 -capability spirv_1_5 "
            "-matrix-layout-row-major -fvk-use-dx-layout -O3 %s%s%s -o build/shaders/%s",
            job->entry,
            job->stage,
            job->define ? "-D" : "",
            job->define ? job->define : "",
            job->define ? " " : "",
            job->output
        );

        if (written < 0 || (size_t)written >= sizeof(command) || system(command) != 0) {
            fprintf(stderr, "SPIR-V shader compilation failed: %s\n", job->entry);
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
#if defined(__clang__)
    c_flag(app, "-Wno-gnu-zero-variadic-macro-arguments");
    c_flag(app, "-Wno-variadic-macro-arguments-omitted");
#endif

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
