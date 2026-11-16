"""Effects test: fire.

A campfire at night: logs in a ring of stones, stylized flame (layers of
coloured shapes without an outline), flying-off tongues, sparks, a
flickering glow on the ground. The camera slowly pushes in. In the middle a
gust of wind — the flame and the sparks bend.

The architectural question: is a pure function of time (no state between
frames) enough for an effect? Here — yes: the flame's shape is noise of
time, tongues and sparks live in slots (the cycle number seeds the birth,
the age is the remainder), the wind is a key track.

The MVP shot file. Changed: the import (Fire comes from mavian).
"""
import math

from mavian import Box, Fire, Ground, Prop, Shot, Sphere, Style, Watch, keys

LENGTH = 6.0


def campfire():
    logs = [Prop(Box(0.95, 0.12, 0.13), at=(0.0, 0.0, 0.0), turn=a, name=f"log{i}",
                 fill="#6b4a33", shadow=False)
            for i, a in enumerate((25.0, -35.0, 95.0))]
    stones = [Prop(Sphere(0.07 + 0.02 * ((i * 7) % 3)), name=f"stone{i}", fill="#5d5866",
                   at=(0.62 * math.cos(math.tau * i / 12), 0.0, 0.62 * math.sin(math.tau * i / 12)))
              for i in range(12)]
    return logs + stones


def make(draw_on=2, wind=True):
    fire = Fire(at=(0.0, 0.13, 0.0), height=1.0, width=0.7, seed=4,   # stands on the logs
                wind=keys({0.0: 0.0, 2.6: 0.0, 3.0: (0.9, "out"), 3.8: 0.9, 4.6: 0.0})
                if wind else 0.0)
    return Shot(
        fps=24, length=LENGTH, seed=4, draw_on=draw_on,
        camera=Watch(pos=keys({0.0: (0.0, 1.15, 4.4), LENGTH: (1.0, 0.9, 3.1)}),
                     target=(0.0, 0.5, 0.0), fov=34),
        ground=Ground(size=4.0, boil=0.0, grid=False),
        style=Style(boil=0.0, pressure=0.0, paper="#1c1a25", ink="#0d0c12"),
        props=[*campfire(), fire],
    )


shot = make()

variants = {
    "ones": make(draw_on=1),        # on ones — to compare: in anime fire is on twos or threes
    "threes": make(draw_on=3),      # on threes
    "calm": make(wind=False),       # no wind
}
