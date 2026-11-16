"""Testing the "trust dial": three identical stick figures turn on the spot.

The left one's face follows the rules of the view (trust = 0), the right
one's is an honest projection of a drawing on a ball (trust = 1), the middle
one is half and half. The question to the picture: which looks drawn, and
which looks like a 3D model?

The MVP shot file with only the import changed.
"""
import math

from mavian import Label, Orbit, Shot, Stickman, Style, keys

TRUST = (0.0, 0.5, 1.0)
PLACES = (-0.95, 0.0, 0.95)


CAM_R = 7.0     # the camera is far, with a narrow angle: all three seen almost alike


def make(draw_on=2, boil=1.2):
    def turn(x):
        # A full turn in 6 seconds. Corrected for the place in the row: at the
        # same moment all three face the camera alike, else the comparison is unfair.
        fix = math.degrees(math.atan2(-x, CAM_R))
        return keys({0.0: fix, 6.0: fix + 360}, ease="linear")

    return Shot(
        fps=24, length=6.0, seed=3, draw_on=draw_on,
        camera=Orbit(radius=CAM_R, height=1.45, target=(0.0, 1.35, 0.0), fov=14.8),
        style=Style(boil=boil),
        actors=[Stickman(poses="stand", path=(x, 0.0), facing=turn(x), face_trust=t)
                for x, t in zip(PLACES, TRUST)],
        labels=[Label(f"trust = {t:g}", (x, 0.60, 0.0)) for x, t in zip(PLACES, TRUST)],
    )


shot = make()

variants = {
    "ones": make(draw_on=1, boil=0.0),      # smooth and without boil — geometry only
}
