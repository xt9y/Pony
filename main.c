#include "ren.c"
#include "inpt.c"

int main() {
    RENDERER renderer = {
        .w = 1270, .h = 750
    };

    int ret = 0;

    if (!r_init(&renderer)) {
        ret = -1;
        goto deinit;
    }

    INPUT input;

    bool running = true;

    while (i_poll(&input)) {

    }

deinit:
    r_deinit(&renderer); 
    return ret;
}
