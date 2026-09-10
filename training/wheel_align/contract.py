"""NumPy/JAX motion equations mirrored by wheel_align_motion.hpp.

Use NumPy for cheap contract tests, JAX for training. All controller state exposed
to the actor is available from encoders, IMU, and this deterministic supervisor.
"""
import numpy as np
from . import configs as c, geometry

IDLE,LIFT,ROTATE,VERIFY,LOWER,HOLD=range(6)
COMMAND_LEG=np.array([-1,1,0,2,3])
POS=np.array(c.POSITION_ACTUATOR_ROWS)
WHEEL=np.array(c.WHEEL_ACTUATOR_ROWS)
NEUTRAL=c.DEFAULT_POSE[POS]
LOW=np.array([-1.12,-.32,-2.41,-3.04]*2)
HIGH=np.array([2.41,3.04,1.12,.32]*2)

def wrap(x,xp=np):return xp.arctan2(xp.sin(x),xp.cos(x))
def smooth(u,xp=np):
    u=xp.clip(u,0,1)
    return u*u*u*(10+u*(-15+6*u))
def up(s):return (s['phase']>=LIFT)&(s['phase']<=VERIFY)
def leg(s,xp=np):return xp.maximum(xp.asarray(COMMAND_LEG)[s['active_command']],0)

def reset(q,home,xp=np):
    p=q[POS]
    return dict(phase=xp.asarray(IDLE),command=xp.asarray(0),active_command=xp.asarray(0),
        phase_steps=xp.asarray(0),gate_steps=xp.asarray(0),settled_steps=xp.asarray(0),
        completed=xp.zeros(4,dtype=bool),verified=xp.asarray(False),was_rotating=xp.asarray(False),
        step_pending=xp.asarray(False),home=wrap(home,xp),target=wrap(home+xp.pi,xp),
        hold=wrap(q[WHEEL],xp),reference=wrap(q[WHEEL],xp),
        previous_phase=xp.asarray(IDLE),elapsed=xp.asarray(0.),progress=xp.asarray(0.),
        motion_reference=p,start=p,applied=p,velocity=xp.zeros(8),desired=p,last_action=xp.zeros(8))

def select(s,command,q,xp=np):
    s=dict(s); k=leg(s,xp)
    command=xp.where((command>=0)&(command<5),command,s['command'])
    interrupt=(command!=s['command'])&up(s)
    s['hold']=xp.where((xp.arange(4)==k)&interrupt,wrap(q[WHEEL],xp),s['hold'])
    s['phase']=xp.where(interrupt,LOWER,s['phase'])
    s['phase_steps']=xp.where(interrupt,0,s['phase_steps']);s['verified']=s['verified']&~interrupt
    s['command']=command
    requested=xp.maximum(xp.asarray(COMMAND_LEG)[command],0)
    start=((s['phase']==IDLE)|(s['phase']==HOLD))&(command!=0)&~s['completed'][requested]
    s['active_command']=xp.where(start,command,s['active_command'])
    s['phase']=xp.where(start,LIFT,s['phase']);s['phase_steps']=xp.where(start,0,s['phase_steps'])
    s['reference']=xp.where(start,s['hold'],s['reference']);s['verified']=s['verified']&~start
    return s

def prepare(s,dt=c.CONTROL_DT,xp=np):
    s=dict(s);k=leg(s,xp)
    previous_up=(s['previous_phase']>=LIFT)&(s['previous_phase']<=VERIFY)
    changed=(s['phase']!=s['previous_phase'])&~(up(s)&previous_up)
    s['start']=xp.where(changed,s['applied'],s['start'])
    s['elapsed']=xp.where(changed,0,s['elapsed'])+xp.clip(dt,0,.04)
    s['previous_phase']=s['phase']
    duration=xp.where(s['phase']==LOWER,c.LOWER_SECONDS,c.LIFT_SECONDS)
    s['progress']=xp.minimum(s['elapsed']/duration,1.)
    goal=xp.where((xp.arange(8)==2*k+1)&up(s),c.APEX_HIP*xp.where(k%2==0,1.,-1.),xp.asarray(NEUTRAL))
    s['motion_reference']=s['start']+smooth(s['progress'],xp)*(goal-s['start'])
    return s

def finish(s,q,qd,xp=np):
    s=dict(s);k=leg(s,xp);pending=s['step_pending'];s['step_pending']=xp.asarray(False)
    s['phase_steps']=s['phase_steps']+pending.astype(int)
    err=xp.abs(wrap(s['target'][k]-q[3*k+2],xp))
    settled=s['was_rotating']&(err<.035)&(xp.abs(qd[3*k+2])<.08)
    s['settled_steps']=xp.where(pending,xp.where(settled,s['settled_steps']+1,0),s['settled_steps'])
    to_rotate=pending&(s['phase']==LIFT)&(s['gate_steps']>=10)
    to_verify=pending&(s['phase']==ROTATE)&(err<.035)
    to_lower=pending&(s['phase']==VERIFY)&(s['settled_steps']>=25)
    motion_done=(s['progress']>=1)&(xp.abs(s['velocity'][2*k+1])<.03)&(xp.abs(s['applied'][2*k+1])<.03)
    to_hold=pending&(s['phase']==LOWER)&(s['phase_steps']>=50)&motion_done&(xp.abs(q[3*k+1])<.25)&(xp.abs(qd[3*k+1])<.2)
    s['hold']=xp.where((xp.arange(4)==k)&to_lower,wrap(q[WHEEL],xp),s['hold'])
    s['verified']=s['verified']|to_lower
    s['completed']=s['completed']|((xp.arange(4)==k)&to_hold&s['verified'])
    s['phase']=xp.where(to_rotate,ROTATE,xp.where(to_verify,VERIFY,xp.where(to_lower,LOWER,xp.where(to_hold,HOLD,s['phase']))))
    s['phase_steps']=xp.where(to_rotate|to_verify|to_lower|to_hold,0,s['phase_steps'])
    return s

def begin(s,q,qd,angular,gravity,action,dt=c.CONTROL_DT,xp=np):
    s=dict(s);k=leg(s,xp)
    gate=(s['progress']>=1)&geometry.ready(q,angular,gravity,k,xp)
    s['gate_steps']=xp.where(up(s)&gate,s['gate_steps']+1,0)
    rotating=((s['phase']==ROTATE)|(s['phase']==VERIFY))&gate
    pause=s['was_rotating']&~rotating&up(s)
    s['reference']=xp.where((xp.arange(4)==k)&pause,wrap(q[WHEEL],xp),s['reference'])
    s['reference']=wrap(s['reference']+xp.where((xp.arange(4)==k)&rotating,
        xp.clip(wrap(s['target']-s['reference'],xp),-.25*dt,.25*dt),0),xp)
    s['was_rotating']=rotating;s['step_pending']=xp.asarray(True)
    active=(xp.arange(8)//2==k)&(up(s)|(s['phase']==LOWER))
    scales=xp.where(active,xp.tile(xp.asarray(c.ACTIVE_RESIDUAL),4),xp.tile(xp.asarray(c.SUPPORT_RESIDUAL),4))
    scales*=xp.where(active&(s['phase']==LOWER),1-smooth(s['progress'],xp),1)
    s['desired']=xp.clip(s['motion_reference']+scales*action,xp.asarray(LOW),xp.asarray(HIGH))
    goal=xp.where((xp.arange(4)==k)&up(s),s['reference'],s['hold'])
    wheel=xp.clip(2*wrap(goal-q[WHEEL],xp)-.35*qd[WHEEL],-2,2)
    return s,wheel

def integrate(s,dt=c.PHYSICS_DT,xp=np):
    s=dict(s);k=leg(s,xp);dt=xp.clip(dt,0,.01)
    active=(xp.arange(8)//2==k)&(up(s)|(s['phase']==LOWER))
    speed=xp.where(active,xp.tile(xp.asarray(c.ACTIVE_SPEED),4),xp.tile(xp.asarray(c.SUPPORT_SPEED),4))
    accel=xp.where(active,xp.tile(xp.asarray(c.ACTIVE_ACCEL),4),xp.tile(xp.asarray(c.SUPPORT_ACCEL),4))
    bound=xp.maximum(speed,xp.abs(s['velocity'])-accel*dt)
    velocity=xp.clip(s['velocity']+xp.clip(36*(s['desired']-s['applied'])-12*s['velocity'],-accel,accel)*dt,-bound,bound)
    applied=s['applied']+velocity*dt
    s['velocity']=xp.where((applied<xp.asarray(LOW))|(applied>xp.asarray(HIGH)),0,velocity)
    s['applied']=xp.clip(applied,xp.asarray(LOW),xp.asarray(HIGH))
    return s

def observation(s,q,qd,angular,gravity,xp=np):
    effective=xp.where(up(s),s['active_command'],0)
    mixed=xp.where(xp.arange(12)%3==2,wrap(q,xp),q-xp.asarray(c.DEFAULT_POSE))
    error=wrap(s['target']-q[WHEEL],xp)
    return xp.concatenate([angular,gravity,(xp.arange(5)==effective).astype(float),mixed,.1*qd,s['last_action'],
        xp.sin(error),xp.cos(error),(xp.arange(6)==s['phase']).astype(float),xp.reshape(s['progress'],(1,)),
        s['motion_reference'],s['applied'],s['velocity']])
