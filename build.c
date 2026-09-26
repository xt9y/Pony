#include <cbuild.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>


typedef struct SHADER_JOB {
    const char *path;
    const char *entry;
    const char *define;
    const char *fallback_define;
    const char *stage;
    int wave;
} SHADER_JOB;

#if !defined(_WIN32)
static const char *prepare_shadercross(void) {
    if (system("command -v shadercross >/dev/null 2>&1") == 0) return "shadercross";

    if (system("pkg-config --exists sdl3-shadercross sdl3 >/dev/null 2>&1") != 0) {
        fprintf(stderr, "SDL_shadercross is installed as a library but pkg-config cannot find sdl3-shadercross\n");
        exit(1);
    }

    static const char helper_source[] =
        "#include <SDL3/SDL.h>\n"
        "#include <SDL3_shadercross/SDL_shadercross.h>\n"
        "#include <stdio.h>\n"
        "#include <string.h>\n"
        "int main(int argc, char **argv) {\n"
        "    if (argc != 6) return 2;\n"
        "    SDL_ShaderCross_ShaderStage stage;\n"
        "    if (strcmp(argv[4], \"vertex\") == 0) stage = SDL_SHADERCROSS_SHADERSTAGE_VERTEX;\n"
        "    else if (strcmp(argv[4], \"fragment\") == 0) stage = SDL_SHADERCROSS_SHADERSTAGE_FRAGMENT;\n"
        "    else if (strcmp(argv[4], \"compute\") == 0) stage = SDL_SHADERCROSS_SHADERSTAGE_COMPUTE;\n"
        "    else return 2;\n"
        "    size_t source_size = 0;\n"
        "    char *source = SDL_LoadFile(argv[1], &source_size);\n"
        "    if (!source) { fprintf(stderr, \"%s\\n\", SDL_GetError()); return 1; }\n"
        "    if (!SDL_ShaderCross_Init()) { fprintf(stderr, \"%s\\n\", SDL_GetError()); SDL_free(source); return 1; }\n"
        "    SDL_ShaderCross_HLSL_Define defines[2] = {{.name = argv[3], .value = NULL}, {0}};\n"
        "    SDL_ShaderCross_HLSL_Info info = {.source = source, .entrypoint = argv[2], .include_dir = \"shaders\", .defines = defines, .shader_stage = stage, .props = 0};\n"
        "    size_t spirv_size = 0;\n"
        "    void *spirv = SDL_ShaderCross_CompileSPIRVFromHLSL(&info, &spirv_size);\n"
        "    if (!spirv) { fprintf(stderr, \"%s\\n\", SDL_GetError()); SDL_ShaderCross_Quit(); SDL_free(source); return 1; }\n"
        "    FILE *output = fopen(argv[5], \"wb\");\n"
        "    int ok = output && fwrite(spirv, 1, spirv_size, output) == spirv_size;\n"
        "    if (output) fclose(output);\n"
        "    SDL_free(spirv);\n"
        "    SDL_ShaderCross_Quit();\n"
        "    SDL_free(source);\n"
        "    return ok ? 0 : 1;\n"
        "}\n";

    FILE *file = fopen("/tmp/pony_shadercross.c", "wb");

    if (!file || fwrite(helper_source, 1, sizeof(helper_source) - 1u, file) != sizeof(helper_source) - 1u) {
        if (file) fclose(file);
        fprintf(stderr, "failed to write temporary SDL_shadercross helper\n");
        exit(1);
    }

    fclose(file);

    if (system("cc -std=c11 /tmp/pony_shadercross.c -o /tmp/pony_shadercross $(pkg-config --cflags --libs sdl3-shadercross sdl3)") != 0) {
        fprintf(stderr, "failed to build temporary SDL_shadercross helper\n");
        exit(1);
    }

    return "/tmp/pony_shadercross";
}
#endif

static void compile_nri_shaders(void) {
#if defined(_WIN32)
    if (system("if not exist build\\shaders mkdir build\\shaders") != 0) exit(1);

    const int dxc_available = system("where dxc >NUL 2>NUL") == 0;
    const char *shadercross = "shadercross";

    if (system("where shadercross >NUL 2>NUL") != 0) {
        fprintf(stderr, "shadercross CLI not found; install SDL_shadercross with its CLI enabled\n");
        exit(1);
    }
#else
    if (system("mkdir -p build/shaders") != 0) exit(1);

    const int dxc_available = system("command -v dxc >/dev/null 2>&1") == 0;
    const char *shadercross = prepare_shadercross();
#endif

    static const SHADER_JOB jobs[] = {
        {"shaders/vertex.hlsl", "surface_vs", "BUILD_SURFACE_VS", NULL, "vertex", 0},
        {"shaders/vertex.hlsl", "wireframe_vs", "BUILD_WIREFRAME_VS", NULL, "vertex", 0},
        {"shaders/vertex.hlsl", "fullscreen_vs", "BUILD_FULLSCREEN_VS", NULL, "vertex", 0},
        {"shaders/fragment.hlsl", "surface_fs", "BUILD_SURFACE_FS", NULL, "fragment", 0},
        {"shaders/fragment.hlsl", "wireframe_fs", "BUILD_WIREFRAME_FS", NULL, "fragment", 0},
        {"shaders/fragment.hlsl", "sky_fs", "BUILD_SKY_FS", NULL, "fragment", 0},
        {"shaders/fragment.hlsl", "compose_fs", "BUILD_COMPOSE_FS", NULL, "fragment", 0},
        {"shaders/compute.hlsl", "lightmap_cs", "BUILD_LIGHTMAP_CS", NULL, "compute", 0},
        {"shaders/compute.hlsl", "lightmap_cs", "BUILD_LIGHTMAP_WAVE_CS", "BUILD_LIGHTMAP_CS", "compute", 1},
        {"shaders/compute.hlsl", "probe_cs", "BUILD_PROBE_CS", NULL, "compute", 0},
        {"shaders/compute.hlsl", "ssao_cs", "BUILD_SSAO_CS", NULL, "compute", 0},
        {"shaders/compute.hlsl", "bloom_cs", "BUILD_BLOOM_CS", NULL, "compute", 0},
        {"shaders/compute.hlsl", "grade_cs", "BUILD_GRADE_CS", NULL, "compute", 0},
        {"shaders/vision_compute.hlsl", "volume_cs", "BUILD_VISION_VOLUME_CS", NULL, "compute", 0},
        {"shaders/vision_compute.hlsl", "volume_compose_cs", "BUILD_VISION_COMPOSE_CS", NULL, "compute", 0},
        {"shaders/dynamic.hlsl", "dynamic_shadow_vs", "BUILD_DYNAMIC_SHADOW_VS", NULL, "vertex", 0},
        {"shaders/dynamic.hlsl", "dynamic_trace_cs", "BUILD_DYNAMIC_TRACE_CS", NULL, "compute", 0},
        {"shaders/lightmap_queue.hlsl", "lightmap_queue_reset_cs", "BUILD_LIGHTMAP_QUEUE_RESET_CS", NULL, "compute", 0},
        {"shaders/lightmap_queue.hlsl", "lightmap_queue_args_cs", "BUILD_LIGHTMAP_QUEUE_ARGS_CS", NULL, "compute", 0},
        {"shaders/probe_wavefront.hlsl", "probe_prepare_cs", "BUILD_PROBE_PREP_CS", NULL, "compute", 0},
        {"shaders/probe_wavefront.hlsl", "probe_reset_cs", "BUILD_PROBE_RESET_CS", NULL, "compute", 0},
        {"shaders/probe_wavefront.hlsl", "probe_validate_cs", "BUILD_PROBE_VALIDATE_CS", NULL, "compute", 0},
        {"shaders/probe_wavefront.hlsl", "probe_primary_cs", "BUILD_PROBE_PRIMARY_CS", NULL, "compute", 0},
        {"shaders/probe_wavefront.hlsl", "probe_primary_cs", "BUILD_PROBE_PRIMARY_WAVE_CS", "BUILD_PROBE_PRIMARY_CS", "compute", 1},
        {"shaders/probe_wavefront.hlsl", "probe_args_cs", "BUILD_PROBE_ARGS_CS", NULL, "compute", 0},
        {"shaders/probe_wavefront.hlsl", "probe_bounce_cs", "BUILD_PROBE_BOUNCE_CS", NULL, "compute", 0},
        {"shaders/probe_wavefront.hlsl", "probe_bounce_cs", "BUILD_PROBE_BOUNCE_WAVE_CS", "BUILD_PROBE_BOUNCE_CS", "compute", 1},
        {"shaders/probe_wavefront.hlsl", "probe_reduce_cs", "BUILD_PROBE_REDUCE_CS", NULL, "compute", 0},
        {"shaders/probe_wavefront.hlsl", "probe_emissive_cs", "BUILD_PROBE_EMISSIVE_CS", NULL, "compute", 0},
    };

    for (size_t i = 0; i < sizeof(jobs) / sizeof(jobs[0]); ++i) {
        const SHADER_JOB *job = &jobs[i];
        char command[1200];
        int written;

        if (job->wave && dxc_available) {
            written = snprintf(command, sizeof(command), "dxc -spirv -fspv-target-env=vulkan1.2 -T cs_6_6 -E %s -I shaders -D%s %s -Fo build/shaders/%s.spv", job->entry,
                               job->define, job->path, job->define);
        } else {
            const char *define = job->wave ? job->fallback_define : job->define;
            written = snprintf(command, sizeof(command), "%s %s %s %s %s build/shaders/%s.spv", shadercross, job->path, job->entry, define, job->stage, job->define);
        }

        if (written < 0 || (size_t)written >= sizeof(command) || system(command) != 0) {
            fprintf(stderr, "shader compile failed: %s:%s (%s)\n", job->path, job->entry, job->define);
            exit(1);
        }
    }
}

void build(C_Build *b) {
    C_Target *app = c_executable(b, "game");

    compile_nri_shaders();

    c_sources(app, "main.c");
    c_sources(app, "init.c");
    c_sources(app, "glb.c");
    c_sources(app, "gltf.c");
    c_sources(app, "bvh.c");
    c_sources(app, "lmap.c");
    c_sources(app, "gpu.c");
    c_sources(app, "render.c");
    c_sources(app, "dynamic.c");
    c_sources(app, "bake.c");
    c_sources(app, "cache.c");
    c_sources(app, "beam.c");

    C_Dependency *nri = c_git(b, "NRI", "https://github.com/NVIDIA-RTX/NRI.git", "main");
    c_dep_cmake(nri);
    c_dep_cmake_option(nri, "-DNRI_STATIC_LIBRARY=OFF");
    c_dep_cmake_option(nri, "-DNRI_ENABLE_VK_SUPPORT=ON");
    c_dep_cmake_option(nri, "-DNRI_ENABLE_VALIDATION_SUPPORT=OFF");
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
    c_link_system(app, "SDL3");
    c_link_system(app, "m");

    c_default_target(b, app);
}
