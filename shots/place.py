"""Support-gated placement: success, obstruction, high and narrow supports."""
from mavian import MANNEQUIN, Actor, Box, Ground, Prop, Shot, Style, Watch, grab, release, key, keys

MODEL = MANNEQUIN.with_poses({
    "carry": MANNEQUIN.pose("stand", arm_r=(65, 6), elbow_r=100),
})


def make(mode="valid"):
    poses = {
        0.0: key("stand", plant="l"),
        1.0: key("stand", plant="l"),
        1.45: key("carry", plant="l"),
        1.8: key("carry", plant="l"),
    }
    if mode == "valid":
        poses[2.75] = key("stand", plant="l")
    hero = Actor(MODEL, poses=keys(poses), squash=0.0)
    cup = Prop(Box(0.13, 0.17, 0.13), at=(-0.30, 1.0, 0.35), name="cup", fill="#d9bd88")
    pedestal = Prop(Box(0.4, 1.0, 0.4), at=(-0.30, 0.0, 0.35), name="pedestal", fill="#a6af9b")
    props = [pedestal, cup]
    support = pedestal
    if mode == "blocked":
        props.append(Prop(Box(0.10, 0.20, 0.04), at=(-0.16, 1.0, 0.31), name="barrier", fill="#b1a8a0"))
    elif mode == "high":
        support = Prop(Box(0.4, 0.1, 0.4), at=(-0.14, 1.9, 0.6), name="high", fill="#a6af9b")
        props.append(support)
    elif mode == "edge":
        support = Prop(Box(0.08, 0.15, 0.2), at=(-0.14, 1.0, 0.35), name="narrow", fill="#b1a8a0")
        props.append(support)
    return Shot(
        actors=[hero], props=props, fps=24, length=3.0, size=(800, 600), draw_on=1,
        camera=Watch(pos=(3.0, 2.0, 3.8), target=(-0.15, 1.0, 0.45), fov=37),
        ground=Ground(grid=False), style=Style(boil=0.0, pressure=0.0),
        actions=[grab(hero, "hand_r", cup, at=1.0), release(hero, "hand_r", cup, at=2.4, onto=support)],
    )


shot = make()
variants = {mode: make(mode) for mode in ("blocked", "high", "edge")}
