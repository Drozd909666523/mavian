"""Test 2: a dynamic scene — two stick figures fight.

Silhouette stick figures, as in old flash fights: black A and red B. A comes
in with shuffle steps and throws a jab with a step — B blocks. A throws a
cross — B slips down and aside, a miss. Out of the slip B steps in and
lunges at the head — a hit. B kicks — A flies back and sits on the ground.
B raises his arms.

What we check: fast dynamics (strikes, knockback, hit-stops, impact frames,
slow motion, speed lines), contact of two characters (a strike goes into the
opponent's joint, a block, a miss) and the camera (a series of setups: a
wide one pushing in, a medium one with a tilted horizon, a low reverse, a
close-up with a zoom jerk in the hit-stop, a tracking one as he flies, a low
shot of the winner).

Strikes are intents: strike(who, with what, at whom, when, where). Reaction,
knockback, hit-stop, shake, impact frame follow by themselves. The fighting
distance comes from where the arm reaches (reach_gap), not picked by number.

The MVP shot file. Changed: the import (FIGHTER comes from mavian).
"""
from mavian import (
    FIGHTER,
    Actor,
    Between,
    Cuts,
    Follow,
    Ground,
    Shot,
    Style,
    Watch,
    key,
    keys,
    reach_gap,
    strike,
    where,
)

BLACK, RED = "#1c1b1f", "#c62f2f"
SHUFFLE = 0.42                                  # a shuffle step, seconds


def shuffle(t0, steps):
    """Shuffle steps, as a boxer moves: the front foot forward (weight on the
    rear), then the rear one follows (weight on the front). Feet stand while
    they carry the weight — nothing slides."""
    table = {}
    for i in range(steps):
        t = t0 + i * SHUFFLE
        table[t + SHUFFLE * 0.5] = key("guard_step", plant="r")
        table[t + SHUFFLE] = key("guard", plant="l")
    return table


# --- choreography (world time) ------------------------------------------------
A_IN = 0.3 + 2 * SHUFFLE                        # A came in with two shuffle steps
T_JAB = A_IN + 0.35                             # a jab with a step — B blocks
T_CROSS = T_JAB + 0.62                          # a right cross — B slips
T_UPPER = T_CROSS + 0.58                        # out of the slip — a step and a lunge at the head
T_KICK = T_UPPER + 0.85                         # a kick — A flies back
END = T_KICK + 2.6


def fighter_a():
    return Actor(
        FIGHTER, path=(-1.9, 0.0), facing=90, squash=0.3, trails=0.07, color=BLACK,
        poses=keys({
            0.0: "guard",
            **shuffle(0.3, 2),
            T_JAB - 0.1: key("guard", plant="l"),
            T_JAB: key("guard_step", plant="r"),               # the jab — with a step
            T_JAB + 0.32: key("guard", plant="r"),             # and back: the front foot
                                                               # returns, weight on the rear
            T_KICK + 0.05: key("guard", plant="l"),
            T_KICK + 0.45: key("sit", "out", plant="l"),       # flew back and sat down
        }),
    )


def fighter_b(x):
    return Actor(
        FIGHTER, path=(x, 0.0), facing=-90, squash=0.3, trails=0.07, color=RED,
        poses=keys({
            0.0: "guard",
            T_CROSS - 0.3: key("guard", plant="l"),
            T_CROSS - 0.06: key("slip", "out", plant="l"),     # a slip down and aside
            T_UPPER - 0.2: key("slip", plant="l"),
            T_UPPER: key("lunge", "in", plant="r"),            # a step and a lunge upward
            T_UPPER + 0.35: key("guard", plant="r"),           # and back
            T_KICK - 0.3: key("guard", plant="l"),
            T_KICK - 0.14: key("kick_r", plant="l"),           # the knee up
            T_KICK + 0.14: key("kick_r", plant="l"),
            T_KICK + 0.5: key("guard", plant="l"),
            T_KICK + 1.1: key("guard", plant="l"),
            T_KICK + 1.6: key("victory", plant="l"),
        }),
    )


def fighters():
    a = fighter_a()
    stop = where(a, A_IN)
    # B at the distance of a jab WITH A STEP: standing A does not reach, stepping he does
    trial = fighter_b(stop[0] + 1.3)
    gap = reach_gap(a, "hand_l", trial, T_JAB)
    return a, fighter_b(stop[0] + 1.3 - gap - 0.02)


def cameras(shot):
    """The fight's setups. The zoom in the hit-stop runs on shot time: the
    world froze, the camera did not."""
    tl = shot.timeline()
    up = tl.shot_time_of(T_UPPER)
    wide = Watch(pos=Between(offset=keys({0.0: (0.0, 0.9, 6.4), A_IN: (0.0, 0.5, 4.4)}),
                             smooth=0.8),
                 target=Between(offset=(0.0, 0.1, 0.0), smooth=0.8), fov=36)
    over_a = Watch(pos=Between(offset=(-1.6, 0.5, 3.1), smooth=0.4),          # over A's shoulder
                   target=Between(joint="chest", offset=(0.35, 0.1, 0.0), smooth=0.4),
                   fov=38, roll=-7)
    low_b = Watch(pos=Between(offset=(1.5, -0.5, 1.6), smooth=0.4),           # low, from B's side
                  target=Between(joint="chest", offset=(-0.2, 0.2, 0.0), smooth=0.4),
                  fov=44, roll=6)
    close = Watch(pos=Between(offset=(0.15, -0.6, 2.0), smooth=0.3),          # the uppercut close up
                  target=Between(joint="chest", offset=(0.0, 0.3, 0.0), smooth=0.3),
                  fov=keys({up - 0.25: 52.0, up: (37.0, "out"), up + 0.4: 46.0}, clock="shot"),
                  roll=keys({up: 0.0, up + 0.06: (10.0, "out"), up + 0.45: 4.0}, clock="shot"))
    track = Watch(pos=Between(offset=(0.0, 0.45, 4.8), smooth=0.5),           # A flies back
                  target=Between(offset=(0.0, 0.1, 0.0), smooth=0.5), fov=40)
    a_end = where(shot.actors[0], END)
    # the winner from below, A sits in the foreground with his back to us
    hero = Watch(pos=(a_end[0] - 1.3, 0.35, a_end[2] + 0.9),
                 target=Follow(1, "chest", offset=(0.0, 0.1, 0.0), smooth=0.6), fov=48)
    return Cuts({0.0: wide, A_IN - 0.25: over_a, T_CROSS - 0.2: low_b,
                 T_UPPER - 0.22: close, T_UPPER + 0.5: track, T_KICK + 1.0: hero})


def make(draw_on=1, plain=False):
    a, b = fighters()
    shot = Shot(
        fps=24, length=END, seed=21, draw_on=draw_on,
        # the kick — in slow motion
        time_warp=keys({T_KICK - 0.08: 1.0, T_KICK: 0.3, T_KICK + 0.3: 0.3,
                        T_KICK + 0.45: 1.0}, ease="linear"),
        ground=Ground(size=5.0, boil=0.0, grid=False),          # a clean background, shadows only
        style=Style(boil=0.0, pressure=0.0, zoom=1.0, paper="#f7f4ee"),
        actors=[a, b],
        actions=[
            strike(a, "hand_l", b, at=T_JAB, aim="head", blocked="hand_l"),
            strike(a, "hand_r", b, at=T_CROSS, aim="head", miss=True),
            strike(b, "hand_r", a, at=T_UPPER, aim="chin", force=0.2, windup=0.18,
                   flash=True),
            strike(b, "foot_r", a, at=T_KICK, aim="chest", force=0.9, windup=0.16,
                   stop=0.14, shake=0.1, recover=0.35, flash=True),
        ],
    )
    shot.camera = (Watch(pos=Between(offset=(0.0, 0.45, 4.6), smooth=0.8),
                         target=Between(offset=(0.0, 0.05, 0.0), smooth=0.8), fov=38)
                   if plain else cameras(shot))
    return shot


shot = make()

variants = {
    "side": make(plain=True),           # one side setup — to compare with the camera work
    "twos": make(draw_on=2),            # on twos
}
