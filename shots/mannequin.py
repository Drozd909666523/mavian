"""Test 1 on the GPU renderer: the mannequin — the stick figure's skeleton
and poses with a volumetric body.

The base shot is the three-camera take (the view from below and the fisheye
were where the MVP's outlines from SVG pieces broke); the MVP's variants
follow: its jump, turning on the spot, everything off. Lines come from the
buffers (part ID and depth); parts of one skin meet without a seam.
"""
import math

import jump
import three_cams
from mavian import MANNEQUIN, Actor, Label, Orbit, Shot, Style, keys

shot = three_cams.make(MANNEQUIN)


def turnaround(poses=("stand", "crouch", "wave_b"), draw_on=2, boil=1.2):
    """Three mannequins in different poses turn on the spot: the form as it turns."""
    places, cam_r = (-1.0, 0.0, 1.0), 7.0

    def turn(x):
        # corrected for the place in the row: at one moment all face the camera alike
        fix = math.degrees(math.atan2(-x, cam_r))
        return keys({0.0: fix, 6.0: fix + 360}, ease="linear")

    return Shot(
        fps=24, length=6.0, seed=3, draw_on=draw_on,
        camera=Orbit(radius=cam_r, height=1.3, target=(0.0, 0.95, 0.0), fov=17.5),
        style=Style(boil=boil),
        actors=[Actor(MANNEQUIN, poses=p, path=(x, 0.0), facing=turn(x))
                for x, p in zip(places, poses)],
        labels=[Label(p, (x, -0.12, 0.0)) for x, p in zip(places, poses)],
    )


variants = {
    "jump": jump.make(model=MANNEQUIN),             # the MVP's base: the jump
    "turn": turnaround(),                           # turning on the spot
    "turn-ones": turnaround(draw_on=1, boil=0.0),   # the same smoothly, without boil
    "fisheye": jump.make(model=MANNEQUIN, fisheye=True),
    "plain": jump.make(model=MANNEQUIN, draw_on=1, slowmo=False, squash=0.0,
                       shake=False, boil=0.0),     # every trick off
}
