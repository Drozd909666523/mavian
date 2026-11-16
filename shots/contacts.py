"""A permanent fixture: pin a foot, touch a prop, lean to reach, then overreach.

Green diagnostics mean satisfied contacts, blue means a ramp is active,
and red marks the remaining error of an unreachable or conflicting target.
"""
from mavian import MANNEQUIN, Actor, Box, Contact, Ground, Prop, Shot, Style, Watch, where


def make(body_reach=0.0, pose_lock=False):
    actor = Actor(MANNEQUIN, poses="stand", squash=0.15,
                  body_reach=body_reach, pose_lock=pose_lock)
    hand = where(actor, 0.0, "hand_r")
    shoulder = where(actor, 0.0, "arm_r")
    # The box stays near the resting hand, with a valid surface point on its side.
    box = Prop(Box(0.18, 0.18, 0.18), at=(hand[0], hand[1] + 0.01, hand[2] + 0.25),
               fill="#d9bd88", name="touch-box")
    near = (hand[0], hand[1] + 0.15, hand[2] + 0.10)
    far = (hand[0], hand[1] + 0.10, hand[2] + 1.50)
    assisted = (shoulder[0], shoulder[1], shoulder[2] + 0.63)
    return Shot(
        actors=[actor], props=[box, Prop(Box(0.35, hand[1] + 0.01, 0.35),
                                       at=(hand[0], 0.0, hand[2] + 0.25),
                                       fill="#a6af9b", name="pedestal")], size=(640, 480), fps=24, length=4.0, draw_on=1,
        camera=Watch(pos=(2.4, 1.7, 3.0), target=(0.0, 0.9, 0.1), fov=38),
        ground=Ground(grid=False), style=Style(boil=0.0, pressure=0.0),
        contacts=[
            Contact(actor, "foot_l", at=(0.0, 4.0), ramp=0.0, priority="support", name="left-foot"),
            Contact(actor, "hand_r", at=(0.0, 1.0), target=near, name="reach-point"),
            Contact(actor, "hand_r", at=(1.0, 2.0), target=box, local=(0.0, 0.09, -0.15), approach=(0.0, 0.10, -0.15), name="touch-prop"),
            # Lift past the pedestal before extending toward the distant point.
            Contact(actor, "hand_r", at=(2.0, 3.0), target=assisted,
                    approach=(hand[0] - assisted[0], 0.35, hand[2] - assisted[2]),
                    name="body-assisted"),
            Contact(actor, "hand_r", at=(3.0, 4.0), target=far, name="unreachable"),
        ],
    )


# Assistance is explicit. The variants show fixed-body IK and exact pose locking.
shot = make(body_reach=30.0)
variants = {
    "limbs": make(),
    "locked": make(pose_lock=True),
}
