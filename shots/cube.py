"""Checking the basics: coordinate axes and a wire cube, the camera circles
them.

If the perspective looks right, the point, the camera and the frame output
work. Axes: X red, Y green (up), Z blue.

The MVP shot file with only the import changed.
"""
from mavian import Ground, Orbit, Shot, Style, axes, cube, keys

shot = Shot(
    fps=24, length=4.0, seed=1, draw_on=1,
    camera=Orbit(radius=5.0, height=2.4, target=(0.0, 0.5, 0.0), fov=32,
                 angle=keys({0.0: 0, 4.0: 360}, ease="linear")),
    ground=Ground(size=3.0, shadow=False),
    style=Style(boil=0.0),
    props=[cube(center=(0.0, 0.5, 0.0), size=1.0)] + axes(1.6),
)
