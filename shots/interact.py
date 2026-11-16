"""Test 3: interacting with the surroundings.

The mannequin walks up to a table, takes a cup with the right hand, carries
it to a crate, puts the cup on the crate and pushes the crate forward — the
cup rides with it. A wall behind.

What we check: props in the same style as the character; the hand reaching
for a prop (inverse kinematics); a prop in the hand, on the table, on
another prop; pushing; a scene with a background. The props' places come
from where the character will be, not set by coordinates: change the number
of steps and the table and the crate move by themselves.

The MVP shot file. Changed: the import, and new poses are made with
`MANNEQUIN.pose(...)` / `.mirror(...)` / `.with_poses(...)`.
"""
from mavian import (
    MANNEQUIN,
    Actor,
    Box,
    Follow,
    Ground,
    Orbit,
    Prop,
    Shot,
    Style,
    Watch,
    grab,
    key,
    keys,
    push,
    release,
    walk,
    walk_end,
    where,
)

pose = MANNEQUIN.pose

POSES = {
    # carries the cup in front of him with the right hand
    "carry": pose("stand", arm_r=(40, 6), elbow_r=80),
    # Contacts add torso assistance only when the arms cannot reach on their own.
    "push_contact_l": pose("contact_l", arm=(60, 10), elbow=30),
    "push_pass_l": pose("pass_l", arm=(60, 10), elbow=30),
    "push_stand": pose("stand", arm=(60, 10), elbow=30),
}
POSES["push_contact_r"] = MANNEQUIN.mirror(POSES["push_contact_l"])
POSES["push_pass_r"] = MANNEQUIN.mirror(POSES["push_pass_l"])
# walking with the cup: legs and the left arm as in walking, the right holds the cup
for name in ("contact_l", "contact_r", "pass_l", "pass_r"):
    POSES["carry_" + name] = pose(name, arm_r=(40, 6), elbow_r=80)
HERO = MANNEQUIN.with_poses(POSES)

START = (0.0, -3.2)         # where he walks from (x, z); looks along +Z
STEP = 0.5                  # seconds per step

# --- choreography (world time) ------------------------------------------------
W1 = walk_end(0.0, 4, STEP, 0.35)               # reached the table
T_GRAB = W1 + 0.55                              # took the cup
T_GO = T_GRAB + 0.8                             # walked off with the cup (going into a
                                                # step ≥ 0.3 s, else the pelvis jerks half a step)
W2 = walk_end(T_GO, 2, STEP, 0.35)              # reached the crate
T_PUT = W2 + 0.75                               # put the cup on the crate
T_PUSH = T_PUT + 0.75                           # leaned on the crate — still standing
T_STEP = T_PUSH + 0.3                           # the first step — already with the crate
W3 = walk_end(T_STEP, 5, STEP, 0.35)            # pushed it there
END = W3 + 1.0


def make_hero():
    return Actor(
        HERO, path=START, squash=0.0,
        poses=keys({
            **walk(0.0, steps=4, step=STEP, stop="stand"),
            T_GRAB - 0.4: key("stand", plant="l"),
            T_GRAB + 0.45: key("carry", plant="l"),
            **walk(T_GO, steps=2, step=STEP, stop="carry", prefix="carry_"),
            W2 + 0.15: key("carry", plant="r"),
            T_PUT: key("stand", plant="r"),
            T_PUT + 0.4: key("stand", plant="r"),
            # Touch it standing first: an early step would carry him into the crate.
            T_PUSH: key("push_stand", plant="r"),
            **walk(T_STEP, steps=5, step=STEP, stop="push_stand", prefix="push_"),
            # Finish withdrawing the hands when the push contact settles.
            W3 + 0.35: key("stand", plant="r"),
            END - 0.4: key("stand", plant="r"),
        }),
    )


def table(x, z):
    """A table of five boxes: the top and the legs."""
    top_y, w, d = 0.72, 0.9, 0.6
    parts = [Prop(Box(w, 0.05, d), at=(x, top_y, z), name="table")]
    for sx in (-1, 1):
        for sz in (-1, 1):
            parts.append(Prop(Box(0.05, top_y, 0.05),
                              at=(x + sx * (w / 2 - 0.06), 0.0, z + sz * (d / 2 - 0.06)),
                              name=f"leg{sx}{sz}", shadow=False))
    return parts, top_y + 0.05


def make(camera="follow", boil=1.2, draw_on=None):
    hero = make_hero()
    stop1 = where(hero, W1)                     # where he stopped at the table
    stop2 = where(hero, W2)                     # where he stopped at the crate
    chest = where(hero, T_PUSH, "chest")        # authored chest before contact assistance
    # the table to his right (his right is −X), the cup at the near edge
    tab, top = table(stop1[0] - 0.78, stop1[2] + 0.12)
    cup = Prop(Box(0.13, 0.17, 0.13), at=(stop1[0] - 0.42, top, stop1[2] + 0.14), name="cup",
               fill="#e9dcc3")
    # the crate right in front of him: the near side 0.22 in front of the chest
    # The body stays upright unless the hand target needs extra reach.
    crate = Prop(Box(0.72, 0.9, 0.72), at=(stop2[0], 0.0, chest[2] + 0.22 + 0.36), name="crate",
                 fill="#efe6d4")
    wall = Prop(Box(0.15, 2.6, 14.0), at=(-2.1, 0.0, 0.0), name="wall", fill="#ebe3d2",
                shadow=False)

    # A camera following something moving needs it on ones. On twos the thing
    # stands while the camera moves, then jumps two frames: the motion jerks.
    if draw_on is None:
        draw_on = 1 if camera == "follow" else 2
    if camera == "follow":
        # a smooth fly-by alongside: follows the smoothed position, without step bob
        cam = Watch(pos=Follow(0, "pelvis", offset=(3.4, 0.55, 0.9), smooth=1.2),
                    target=Follow(0, "pelvis", offset=(-0.5, -0.05, 0.35), smooth=1.2), fov=40)
    else:   # a wide shot
        cam = Orbit(radius=9.0, height=3.0, angle=60, target=(-0.6, 0.6, 1.0), fov=46)

    return Shot(
        fps=24, length=END, seed=11, draw_on=draw_on,
        camera=cam,
        ground=Ground(size=6.0, center=(-0.5, 0.0), boil=0.0),   # the background does not boil
        style=Style(boil=boil),
        actors=[hero],
        props=[wall, *tab, cup, crate],
        actions=[
            grab(hero, "hand_r", cup, at=T_GRAB),
            release(hero, "hand_r", cup, at=T_PUT, onto=crate),
            push(hero, crate, start=T_PUSH, end=W3, height=0.85),
        ],
    )


shot = make()

variants = {
    "wide": make(camera="wide"),                    # a wide shot: the whole scene
    "ones": make(boil=0.0, draw_on=1),              # smooth and without boil
}
