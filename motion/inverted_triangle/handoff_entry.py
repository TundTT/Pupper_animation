"""Measured-state entry into the existing roll, with continuous command limits.

Simulation prototype. Source code is not permission to deploy or move hardware.
"""
import numpy as np
from .core import HUB,PROX,SPEED,ACCEL,smooth,duration

NEUTRAL=np.array([1.,0.,-1.,-1.,0.,1.,1.,0.,-1.,-1.,0.,1.])
ROLL_START=NEUTRAL.copy();ROLL_START[1::3]=[-.29,.29,-.29,.29]
SIGNS=np.array([-1.,1.,-1.,1.])
LOW=np.array([-1.12,-.32,-1000,-2.41,-3.04,-1000]*2)
HIGH=np.array([2.41,3.04,1000,1.12,.32,1000]*2)

class EntryRoll:
    def __init__(self,q,previous_command=None,entry_seconds=4.,roll_seconds=12.,settle_seconds=32.,entry_integral_gain=0.,entry_mode="measured",entry_splay=.29):
        q=np.asarray(q,dtype=float)
        if q.shape!=(12,) or not np.isfinite(q).all():raise ValueError('Finite measured 12-joint pose required')
        if np.any(q<LOW) or np.any(q>HIGH):raise ValueError('Measured start exceeds position limits')
        self.command=q.copy() if previous_command is None else np.array(previous_command,dtype=float)
        if self.command.shape!=(12,) or not np.isfinite(self.command).all() or np.max(abs(self.command-q))>.15:
            raise ValueError('Handoff command must agree with measured pose within 0.15 rad')
        if np.any(self.command<LOW) or np.any(self.command>HIGH):raise ValueError('Initial command exceeds position limits')
        if entry_seconds<=0 or roll_seconds<12 or settle_seconds<3:raise ValueError('Invalid stage durations')
        if entry_mode not in ('measured','direct'):raise ValueError('Unknown entry mode')
        if not np.isfinite(entry_integral_gain) or not 0<=entry_integral_gain<=1:raise ValueError('Invalid entry integral gain')
        if not np.isfinite(entry_splay) or not 0<=entry_splay<=.32:raise ValueError('Invalid entry splay')
        self.entry_integral_gain=entry_integral_gain;self.entry_integral=np.zeros(12);self.entry_mode=entry_mode
        self.start=self.command.copy();self.entry=ROLL_START.copy();self.entry[1::3]=entry_splay*SIGNS;self.entry[HUB]=q[HUB]
        # Fixed whole-turn choice for the FINAL TARGET only; never wrap a live state.
        self.goal=NEUTRAL.copy()
        self.goal[HUB]+=2*np.pi*np.round((q[HUB]+SIGNS*np.pi-self.goal[HUB])/(2*np.pi))
        if np.max(abs(self.goal[HUB]-q[HUB]-SIGNS*np.pi))>.35:
            raise ValueError('Expected aligned point-up hubs within 20 deg of model reference')
        self.entry_seconds=duration(self.start,self.entry,entry_seconds)
        self.roll_seconds=duration(self.entry,self.goal,roll_seconds)
        self.settle_seconds=settle_seconds;self.elapsed=0.;self.phase='entry';self.stable=0.
        if entry_mode=='direct':self.phase='roll'
        self.velocity=np.zeros(12);self.peak_speed_ratio=0.;self.peak_accel_ratio=0.;self.failure=None

    def step(self,q,qd,dt,correction=None):
        q=np.asarray(q);qd=np.asarray(qd)
        if q.shape!=(12,) or qd.shape!=(12,) or not np.isfinite(np.r_[q,qd]).all() or not 0<dt<=.01:
            raise ValueError('Invalid measurement/update period')
        if self.phase in ('done','failed'):return self.command.copy()
        self.elapsed+=dt
        if self.phase=='entry':
            reference=self.start+smooth(self.elapsed/self.entry_seconds)*(self.entry-self.start)
            self.entry_integral[PROX]=np.clip(self.entry_integral[PROX]+self.entry_integral_gain*(reference[PROX]-q[PROX])*dt,-.15,.15)
            target=reference+self.entry_integral
            good=np.max(abs(q-self.entry))<=.10 and np.max(abs(qd))<=.12
            self.stable=self.stable+dt if good and self.elapsed>=self.entry_seconds else 0.
            if self.stable>=.25:
                self.phase='roll';self.elapsed=0.;self.start=self.command.copy()
            elif self.elapsed>self.entry_seconds+8:
                self.failure='entry_did_not_settle';self.phase='failed'
        elif self.phase=='roll':
            target=self.start+smooth(self.elapsed/self.roll_seconds)*(self.goal-self.start)
            if self.elapsed>=self.roll_seconds:self.phase='settle';self.elapsed=0.
        else:
            delta=np.zeros(12) if correction is None else np.asarray(correction)
            if delta.shape!=(12,) or not np.isfinite(delta).all() or np.max(abs(delta))>.200001:
                raise ValueError('Invalid bounded support correction')
            target=self.goal+delta
            if self.elapsed>=self.settle_seconds:self.phase='done'
        target=np.clip(target,LOW,HIGH)
        # All phases share one acceleration-limited filter: no entry/roll jump.
        wanted=np.clip(5*(target-self.command),-SPEED,SPEED)
        next_velocity=self.velocity+np.clip(wanted-self.velocity,-ACCEL*dt,ACCEL*dt)
        next_command=self.command+dt*next_velocity
        if np.any(next_command<LOW) or np.any(next_command>HIGH):
            self.failure='filtered_command_limit';self.phase='failed';return self.command.copy()
        self.peak_speed_ratio=max(self.peak_speed_ratio,float(np.max(abs(next_velocity)/SPEED)))
        self.peak_accel_ratio=max(self.peak_accel_ratio,float(np.max(abs(next_velocity-self.velocity)/(ACCEL*dt))))
        self.command=next_command;self.velocity=next_velocity
        return self.command.copy()
