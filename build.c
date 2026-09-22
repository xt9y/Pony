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
    c_sources(app, "src/*.c");
    c_use(app, nk);
    c_link_system(app, "SDL3");
    c_default_target(b, app);
}
