"""Task shaping, separate from posture rewards and the unchanged safety gates.

The angle potential is signed: rocking a wheel back and forth cannot repeatedly
collect positive progress. Events use supervisor transitions, not angle alone.
"""
import numpy as np
from . import contract as ct

def residual_cost(action,motion,dt,xp=np):
    # A constant saturated output must not evade the existing change penalty.
    return dt*.5*ct.up(motion)*(motion['residual_gain']>=1)*xp.sum(xp.square(action))

def task_terms(before, after, q_before, q_after, floor, gap, bodygap,
               tilt, angular_speed, unsafe, dt, xp=np):
    k=ct.leg(before,xp)
    # Targets sit slightly inside the strict runtime gate. Floor is the minimum
    # of actual simulated clearance and the encoder/IMU estimate used onboard.
    deficits=xp.stack([xp.maximum(.012-floor,0)/.012,
        xp.maximum(.012-gap,0)/.012, xp.maximum(.007-bodygap,0)/.007,
        xp.maximum(tilt-.08,0)/.12, xp.maximum(angular_speed-.20,0)/.3])
    quality=1/(1+xp.sum(deficits))
    apex=ct.up(before)&(before['progress']>=1)
    error_before=xp.abs(ct.wrap(before['target'][k]-q_before[3*k+2],xp))
    error_after=xp.abs(ct.wrap(before['target'][k]-q_after[3*k+2],xp))
    safe=(floor>.010)&(gap>.010)&(bodygap>.005)&(tilt<.12)&(angular_speed<.3)&~unsafe
    rotating=before['was_rotating']&safe
    progress=xp.where(rotating,error_before-error_after,0.)
    verified=(~before['verified'])&after['verified']
    completed=xp.sum(after['completed'].astype(int)-before['completed'].astype(int))
    # Dense quality remains useful before the first successful rotation. Once
    # at the apex, delay has a cost; holding forever is less valuable than finish.
    reward=dt*(-4*ct.up(before)*xp.sum(deficits)*xp.minimum(before["progress"]*2,1) -2*apex*error_after/xp.pi
        -ct.up(before).astype(float) -6*apex*~safe -10*unsafe) + 20*progress + 20*verified + 60*completed
    return dict(reward=reward, gate_quality=quality, angle_progress=progress,
                verified_event=verified.astype(float), completed_event=completed.astype(float),
                active_angle_error=error_after)
