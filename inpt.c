#ifndef INPT_C
#define INPT_C

#include "sdl.h"

typedef struct Input {
    SDL_Event event;
    int fwd;
    int back;
    int left;
    int right;
    int run;
    int debug_mode;
    uint32_t debug_bvh_depth;
    float dx;
    float dy;
} INPUT;

int i_poll(INPUT *i, SDL_Window *win) {
    i->dx = 0.0f;
    i->dy = 0.0f;

    while (SDL_PollEvent(&i->event)) {
        if (i->event.type == SDL_EVENT_QUIT) {
            return 0;
        }

        if (i->event.type == SDL_EVENT_KEY_DOWN && !i->event.key.repeat) {
            switch (i->event.key.scancode) {
                case SDL_SCANCODE_W:
                    i->fwd = 1;
                    break;
                case SDL_SCANCODE_S:
                    i->back = 1;
                    break;
                case SDL_SCANCODE_A:
                    i->left = 1;
                    break;
                case SDL_SCANCODE_D:
                    i->right = 1;
                    break;
                case SDL_SCANCODE_LSHIFT:
                case SDL_SCANCODE_RSHIFT:
                    i->run = 1;
                    break;
                case SDL_SCANCODE_ESCAPE:
                    SDL_SetWindowRelativeMouseMode(win, false);
                    break;
                case SDL_SCANCODE_F1:
                    i->debug_mode = 0;
                    break;
                case SDL_SCANCODE_F2:
                    i->debug_mode = 1;
                    break;
                case SDL_SCANCODE_F3:
                    i->debug_mode = 2;
                    break;
                case SDL_SCANCODE_F4:
                    i->debug_mode = 3;
                    break;
                case SDL_SCANCODE_F5:
                    if (i->debug_mode == 4) {
                        i->debug_bvh_depth = (i->debug_bvh_depth + 1u) % 8u;
                    } else {
                        i->debug_mode = 4;
                        i->debug_bvh_depth = 0;
                    }
                    break;
                default:
                    break;
            }
        }

        if (i->event.type == SDL_EVENT_KEY_UP) {
            switch (i->event.key.scancode) {
                case SDL_SCANCODE_W:
                    i->fwd = 0;
                    break;
                case SDL_SCANCODE_S:
                    i->back = 0;
                    break;
                case SDL_SCANCODE_A:
                    i->left = 0;
                    break;
                case SDL_SCANCODE_D:
                    i->right = 0;
                    break;
                case SDL_SCANCODE_LSHIFT:
                case SDL_SCANCODE_RSHIFT:
                    i->run = 0;
                    break;
                default:
                    break;
            }
        }

        if (i->event.type == SDL_EVENT_MOUSE_BUTTON_DOWN) {
            if (i->event.button.button == SDL_BUTTON_LEFT) {
                SDL_SetWindowRelativeMouseMode(win, true);
            }
        }

        if (i->event.type == SDL_EVENT_MOUSE_MOTION) {
            if (SDL_GetWindowRelativeMouseMode(win) || (i->event.motion.state & SDL_BUTTON_LMASK)) {
                i->dx += i->event.motion.xrel;
                i->dy += i->event.motion.yrel;
            }
        }
    }
    return 1;
}

#endif // INPT_C
