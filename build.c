#include <cbuild.h>

void build(C_Build *b)
{
    C_Dependency *nk = c_git(
        b,
        "Nuklear",
        "https://github.com/Immediate-Mode-UI/Nuklear.git",
        "master"
    );
    c_dep_header_only(nk);

    C_Target *app = c_executable(b, "app");
    c_sources(app, "main.c");

    c_use(app, nk);

#ifdef __APPLE__
    c_include(app, "/opt/homebrew/include");
    c_link_flag(app, "-L/opt/homebrew/lib");
    c_link_flag(app, "-Wl,-rpath,/opt/homebrew/lib");
#endif

    c_link_system(app, "SDL3_shadercross");
    c_link_system(app, "SDL3");
    c_link_system(app, "m");

    C_Target *tests = c_executable(b, "tests");
    c_sources(tests, "tests.c");

#ifdef __APPLE__
    c_include(tests, "/opt/homebrew/include");
    c_link_flag(tests, "-L/opt/homebrew/lib");
    c_link_flag(tests, "-Wl,-rpath,/opt/homebrew/lib");
#endif

    c_link_system(tests, "SDL3_shadercross");
    c_link_system(tests, "SDL3");
    c_link_system(tests, "m");

    c_default_target(b, app);
}
