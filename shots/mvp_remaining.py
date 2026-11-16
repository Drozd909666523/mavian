"""Short regression scene: drawing noise, held cup, attached wire and fire depth."""
from mavian import Actor, MANNEQUIN, Box, Prop, Fire, Follow, Wire, Shot, Style, Watch, Ground, keys, grab

def make():
    cup = Prop(Box(.22,.25,.22), at=(-.32,.85,.05), name="cup", boil=True, fill="#dfbd83")
    actor = Actor(MANNEQUIN, poses=keys({0.:"stand"}), squash=0.)
    return Shot(
        actors=[actor],
        camera=Watch(pos=(2.2,1.6,4.),target=(0.,.85,0.),fov=35),
        size=(480,360),fps=12,length=.5,draw_on=2,
        style=Style(boil=1.5,pressure=.25), ground=Ground(grid=False),
        props=[cup, Prop(Box(1.3,.25,.4),at=(.9,0.,-.5),turn=35,fill="#9da591"),
               Fire(at=(0.,.1,-1.4),height=1.4,width=.8,over=.2),
               Wire([((-.15,0.,0.),(.15,0.,0.)),((0.,-.15,0.),(0.,.15,0.))],at=Follow(0,"hand_r"),color="#347ab7")],
        actions=[grab(actor,"hand_r",cup,at=.2,reach=.2)],
    )
shot=make()
