#ifndef INPT_C
#define INPT_C

#include "sdl.h"

typedef struct Input {
    SDL_Event event;
} INPUT;

int i_poll(INPUT *i) {
    while (SDL_PollEvent(&i->event)) {
        if (i->event.type == SDL_EVENT_QUIT) {
            return 0;
        }
    }
    return 1;
}

#endif // INPT_C
