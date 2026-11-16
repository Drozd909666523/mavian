"""Demo shot: the stick figure crouches, jumps, lands with an impact and
waves. Meanwhile the camera circles him by 180°.

All times in poses and the path are world time. The camera follows the
shot's progress, so during the slow motion the world almost stands while the
camera keeps circling.

The MVP shot file. Changed: the import; `Patch` takes world time instead of
frame numbers (a patch by frames drifts when the timing changes).
"""
from mavian import STICKMAN, Actor, Ground, Orbit, Patch, Shake, Shot, Style, keys


def make(seed=7, draw_on=2, slowmo=True, face_trust=0.0, squash=0.3,
         shake=True, boil=1.5, fisheye=False, extras=False, model=STICKMAN):
    camera = Orbit(
        radius=5.3, height=1.5, target=(0.0, 1.2, 0.0), fov=32,
        angle=keys({0.0: -160, 1.0: 20}, ease="linear", clock="progress"),  # circling by 180°
    )
    if fisheye:
        camera = Orbit(
            radius=1.7, height=0.5, target=(0.0, 1.0, 0.0), fov=100, fisheye=True,
            angle=keys({0.0: -100, 1.0: 20}, ease="linear", clock="progress"),
        )
    return Shot(
        fps=24, length=4.1, seed=seed,
        draw_on=draw_on,                        # drawn on twos
        # slow motion at the top of the jump: the world runs five times slower
        time_warp=keys({1.20: 1.0, 1.32: 0.2, 1.56: 0.2, 1.68: 1.0},
                       ease="linear") if slowmo else 1.0,
        camera=camera,
        shake=Shake(at=1.84, strength=0.10, decay=0.28) if shake else None,
        ground=Ground(),
        style=Style(boil=boil),
        actors=[Actor(
            model,
            poses=keys({
                0.00: "stand",
                0.35: "stand",
                0.80: "crouch",                 # wind-up
                0.92: "crouch",
                1.04: ("launch", "in"),         # push
                1.44: ("jump", "out"),          # the top
                1.84: ("fall", "in"),           # touching the ground
                2.04: ("land", "out"),          # absorbing the impact
                2.14: "land",
                2.75: "stand",
                3.00: "wave_a",
                3.20: "wave_b",
                3.40: "wave_a",
                3.60: "wave_b",
                3.80: "wave_a",
            }),
            # a jump forward: does not slide on the ground, flies evenly
            path=keys({1.04: (0.0, -0.45), 1.84: ((0.0, 0.45), "linear")}),
            squash=squash,                      # squash and stretch
            face_trust=face_trust,              # 0 — the face by the rules of the view
            trails=0.09 if extras else 0.0,     # speed lines behind hands and feet
            perspective=4.0 if extras else 1.0, # own perspective: nearer is bigger
        )],
        # a correction by code: the left arm is raised sideways (the MVP's
        # frames 104–127, as world time)
        patches=[Patch(at=PATCH_AT, joint="arm_l", add=(0, 55, 0))] if extras else [],
    )


PATCH_AT = (3.13, 4.09)   # the world time of frames 104–127

shot = make()

# The same shot with one parameter changed.
variants = {
    "seed8": make(seed=8),                  # another seed: other boil and shake
    "ones": make(draw_on=1),                # no twos
    "noslow": make(slowmo=False),           # no slow motion
    "trust1": make(face_trust=1.0),         # the face is an honest projection
    "plain": make(draw_on=1, slowmo=False, squash=0.0, shake=False, boil=0.0),  # every trick off
    "fisheye": make(fisheye=True),          # a low wide-angle camera
    "extras": make(extras=True),            # speed lines, own perspective, a frame patch
}
