"""Contact-gated acquisition: valid grip, overreach, pose lock and obstruction.

The release after a missed grab leaves the prop on its original support.
"""
from mavian import (
    MANNEQUIN, Actor, Box, Ground, Prop, Shot, Style, Watch,
    grab, key, keys, release,
)

MODEL = MANNEQUIN.with_poses({
    "carry": MANNEQUIN.pose("stand", arm_r=(65, 6), elbow_r=100),
})


def make(mode="valid"):
    actor = Actor(MODEL, squash=0.0, pose_lock=mode == "locked", poses=keys({
        0.0: key("stand", plant="l"),
        1.0: key("stand", plant="l"),
        1.45: key("carry", plant="l"),
        1.8: key("carry", plant="l"),
        2.75: key("stand", plant="l"),
    }))
    z = 1.6 if mode == "far" else 0.35
    cup = Prop(Box(0.13, 0.17, 0.13), at=(-0.30, 1.0, z),
               name="cup", fill="#d9bd88")
    pedestal = Prop(Box(0.4, 1.0, 0.4), at=(-0.30, 0.0, z),
                    name="pedestal", fill="#a6af9b")
    props = [pedestal, cup]
    if mode == "blocked":
        # A slab on the pedestal blocks the grasp envelope, not the actor.
        props.append(Prop(Box(0.13, 0.24, 0.04),
                          at=(-0.36, 1.0, 0.20),
                          name="barrier", fill="#b1a8a0"))
    return Shot(
        actors=[actor], props=props, fps=24, length=3.0, size=(800, 600), draw_on=1,
        camera=Watch(pos=(3.0, 2.0, 3.8), target=(-0.15, 1.0, 0.45), fov=37),
        ground=Ground(grid=False), style=Style(boil=0.0, pressure=0.0),
        actions=[
            grab(actor, "hand_r", cup, at=1.0),
            release(actor, "hand_r", cup, at=2.4, onto=pedestal),
        ],
    )


shot = make()
variants = {"far": make("far"), "locked": make("locked"), "blocked": make("blocked")}
