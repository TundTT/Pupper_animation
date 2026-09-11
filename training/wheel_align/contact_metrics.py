"""Simulation-only support loads and per-contact peak approach costs.

The pinned model uses pyramidal condim=3 contacts: four constraint forces sum
to the contact normal force (MuJoCo MJX support._decode_pyramid). No force
measurement is added to the actor or the hardware controller.
"""
import numpy as np

def wheel_loads(contact,efc_force,wheels,floor,xp=np):
    address=contact.efc_address
    force=xp.sum(efc_force[xp.maximum(address[:,None]+xp.arange(4),0)],axis=1)
    valid=(address>=0)&(contact.dist<=0)&xp.any(contact.geom==floor,axis=1)
    match=xp.any(contact.geom[:,:,None]==xp.asarray(wheels)[None,None,:],axis=1)
    return xp.sum(xp.where(valid[:,None]&match,force[:,None],0),axis=0)

def approach_cost(previous_peak,clearance,velocity,xp=np):
    retained=xp.where(clearance>.012,0.,previous_peak)
    approaching=xp.where(clearance<.008,xp.maximum(velocity,0.),0.)
    peak=xp.maximum(retained,approaching)
    # Charge new peak severity once, rather than multiplying a millisecond
    # impact by control dt. Do not refund the cost when the wheel lifts again.
    severity=lambda x:xp.square(xp.maximum(x-.08,0)/.10)
    cost=20*xp.sum(severity(peak)-severity(retained))
    return peak,cost
