"""Encoder/IMU-only conservative clearance estimates, shared with C++.

Ground estimate assumes flat ground and planted support wheels. The policy's
observations never use simulation-only contact/body-position information.
"""
import json
from pathlib import Path
import numpy as np

PARAMS = json.loads(Path(__file__).with_name('geometry.json').read_text())
RADIUS = .0505  # 48 mm nominal + 2.5 mm modeled manufacturing variation
HALF_WIDTH = .01675
BOUND_RADIUS = float(np.hypot(RADIUS, HALF_WIDTH))

def rotz(q, xp=np):
    c, s, z, o = xp.cos(q), xp.sin(q), xp.zeros_like(q), xp.ones_like(q)
    return xp.stack([c,-s,z,s,c,z,z,z,o], axis=-1).reshape((4,3,3))

def wheel_frames(q, xp=np):
    q=xp.asarray(q).reshape(4,3)
    r=xp.asarray(PARAMS['r1']) @ rotz(q[:,0],xp) @ xp.asarray(PARAMS['r2']) @ rotz(q[:,1],xp)
    centers=xp.asarray(PARAMS['p1'])+xp.einsum('bij,bj->bi',r,xp.asarray(PARAMS['p3']))
    r=r @ xp.asarray(PARAMS['r3'])
    axes=r[:,:,2]
    return centers + .03035*axes, axes

def margins(q, gravity, leg, xp=np):
    centers,axes=wheel_frames(q,xp)
    up=-xp.asarray(gravity)
    dot=axes @ up
    bottom=centers @ up - (RADIUS*xp.sqrt(xp.maximum(1-dot*dot,0)) + HALF_WIDTH*xp.abs(dot))
    floor=bottom[leg]-xp.max(xp.where(xp.arange(4)!=leg,bottom,-1e3))
    delta=centers[:,None,:]-centers[None,:,:]
    gaps=xp.sqrt(xp.sum(delta*delta,axis=-1)+xp.eye(4))-2*BOUND_RADIUS
    wheel=xp.min(gaps)
    local=(centers-xp.asarray(PARAMS['box_pos'])) @ xp.asarray(PARAMS['box_rot'])
    box=xp.min(xp.linalg.norm(xp.maximum(xp.abs(local)-xp.asarray(PARAMS['box_size']),0),axis=-1)-BOUND_RADIUS)
    return floor,wheel,box

def ready(q, angular_velocity, gravity, leg, xp=np):
    floor,wheel,box=margins(q,gravity,leg,xp)
    return (floor>.010)&(wheel>.010)&(box>.005)&(-gravity[2]>xp.cos(.12))&(xp.linalg.norm(angular_velocity)<.3)
