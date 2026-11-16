"""One continuous take, three cameras with cuts.

1. Side view: the stick figure walks.
2. Front camera, fisheye: he comes almost right up to it, peers into the
   lens and turns his head slightly.
3. Camera on the ground, normal perspective: he jumps over it, and the
   camera follows him.

There is one action in world time; cameras are placed from where the
character will be (where), so changing the steps adjusts everything by itself.

This is the MVP shot file; the jump is now an intent (`jump`): the take-off
moment, distance and height — the push, flight and landing timings follow
from the poses and gravity, so the pelvis keeps its speed through the jump.
"""
from mavian import (STICKMAN, Actor, Cuts, Follow, Ground, Shake, Shot, Style, Watch,
                    jump, key, keys, walk, where)

CUT_FRONT = 3.0     # cut to the front camera
CUT_LOW = 8.6       # cut to the ground camera
LAUNCH = 9.30       # take-off; the rest of the jump follows from it
DISTANCE = 2.4      # from the take-off foot to the landing foot, m
HEIGHT = 0.5        # how high the pelvis rises, m

def make(model=STICKMAN):
    leap = jump(model, LAUNCH, DISTANCE, HEIGHT, plant="l")
    land = leap.end
    actor = Actor(
        model,
        path=(0.0, -3.4),                           # where he starts walking from
        poses=keys({
            **walk(0.0, steps=11, stop="stand"),    # 11 steps and a stop (by 5.85)
            6.40: "peer",                           # leans towards the lens
            6.95: "peer_l",                         # head to the left
            7.60: "peer_r",                         # head to the right
            8.15: "peer",
            8.60: "peer",
            9.05: key("crouch", plant="l"),         # wind-up
            **leap.keys,                            # push, flight, landing
            land + 0.11: key("land", plant="l"),
            land + 0.71: key("stand", plant="l"),
            land + 1.21: "look_back",               # looks back at the camera
        }),
        blink=keys({6.60: 0, 6.66: 1, 6.74: 0, 8.00: 0, 8.06: 1, 8.14: 0},
                   ease="linear"),
        squash=0.3,
    )

    stop = where(actor, 6.0)                        # where he stopped
    mid = (where(actor, 0.0)[2] + where(actor, CUT_FRONT)[2]) / 2.0

    side = Watch(pos=(-8.0, 1.3, mid), target=(0.0, 1.0, mid), fov=30)
    front = Watch(pos=(0.0, 1.52, stop[2] + 0.62), target=(0.0, 1.5, stop[2] - 2.0),
                  fov=140, fisheye=True)
    lens = (0.0, 0.12, stop[2] + 1.7)              # on the ground, in front of him
    # when he is right above the lens: from where he takes off and how fast
    over = LAUNCH + (lens[2] - where(actor, LAUNCH)[2]) / leap.speed[0]
    low = Watch(
        pos=lens,
        # follows the pelvis like an operator, over 0.2 s: the push is ~2 g,
        # and a camera locked to it would jerk
        target=Follow(0, "pelvis", offset=(0.0, 0.25, 0.0), smooth=0.2),
        fov=70,
        # while he flies over the camera, it smoothly turns after him
        turn=keys({LAUNCH: 180, leap.landing: 0}),
    )

    shot = Shot(
        fps=24, length=12.3, seed=5,
        camera=Cuts({0.0: side, CUT_FRONT: front, CUT_LOW: low}),
        actors=[actor],
        draw_on=keys({0.0: 2, CUT_LOW: (1, "hold")}),   # the jump is on ones
        # slow motion as he passes over the camera
        time_warp=keys({over - 0.17: 1.0, over - 0.09: 0.25,
                        over + 0.17: 0.25, over + 0.25: 1.0}, ease="linear"),
        shake=Shake(at=leap.landing, strength=0.05, decay=0.25),
        ground=Ground(size=8.0, center=(0.0, 1.8)),
        style=Style(boil=1.5),
    )
    return shot


shot = make()
