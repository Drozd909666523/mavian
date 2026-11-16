"""Bounded contact preparation: retreat, approach, blocked path, distance limit."""
from mavian import MANNEQUIN, Actor, Box, Ground, Label, Prop, Shot, Style, Watch, push


def make(enabled=True, locked=False):
    actors, props, actions, labels = [], [], [], []
    cases = [(-3.3, 0.40, 0.35, "Back"), (-1.1, 1.0, 0.35, "Forward"),
             (1.1, 1.0, 0.35, "Blocked"), (3.3, 1.3, 0.12, "Limited")]
    for x, z, limit, name in cases:
        actor = Actor(MANNEQUIN, poses="stand", path=(x, 0.0), pose_lock=locked)
        box = Prop(Box(0.6, 1.4, 0.4), at=(x, 0.0, z), name=name.lower(), fill="#d4c6a5")
        actors.append(actor)
        props.append(box)
        if name == "Blocked":
            props.append(Prop(Box(0.65, 0.18, 0.04), at=(x, 0.0, 0.213), name="barrier", fill="#b36b57"))
        actions.append(push(actor, box, start=1.8, end=2.6, reach=0.35, height=0.85,
                            auto_step=limit if enabled else 0.0, step_duration=0.9))
        labels.append(Label(name, at=(x, 2.05, 0.0), size=20))
    return Shot(actors=actors, props=props, actions=actions, labels=labels,
                camera=Watch(pos=(20.0, 7.0, 26.0), target=(0.0, 1.0, 0.4), fov=7),
                ground=Ground(grid=False), style=Style(boil=0.0, pressure=0.0),
                fps=24, length=3.2, draw_on=1, size=(1280, 720))


shot = make()
variants = {"off": make(enabled=False), "locked": make(locked=True)}
