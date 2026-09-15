"""Button-driven experimental controller; all outputs are joint positions.

The caller supplies measured readiness on every tick. Never derive readiness
from a button count or the policy's requested pose. This module owns no hardware.
"""
from dataclasses import dataclass, field
import math
import numpy as np
from workspace import wheel_lift_config as c

@dataclass
class LiftAlignSequence:
    home: np.ndarray
    reference: np.ndarray
    stage: str = 'idle'
    leg: int = 0
    velocity: float = 0.
    settled: float = 0.
    elapsed: float = 0.
    verified: bool = False
    qualified: float = 0.
    rotating: bool = False
    max_speed: float = .15  # rad/s (~21 seconds for a half-turn)
    max_accel: float = .3  # rad/s^2, reference shaping, not velocity actuation
    goal: np.ndarray = field(init=False)

    def __post_init__(self):
        self.home = np.array(self.home, dtype=float, copy=True)
        self.reference = np.array(self.reference, dtype=float, copy=True)
        if self.home.shape != (4,) or self.reference.shape != (4,) or not np.isfinite([self.home,self.reference]).all():
            raise ValueError('Four finite calibrated homes and measured hub positions required')
        delta = (self.home + math.pi - self.reference + math.pi) % (2*math.pi) - math.pi
        delta[np.isclose(delta, -math.pi, atol=1e-10, rtol=0)] = math.pi
        self.goal = self.reference + delta

    @property
    def command(self):
        # Rotation is deliberately indistinguishable from lift to the actor.
        return [2,1,3,4][self.leg] if self.stage in ('lift','align','fault') else 0

    def press(self, *, supported=False, lift_ready=False):
        """Accept one rising edge. Rejected presses never advance or queue."""
        if self.stage == 'idle':
            self.stage = 'pose'
        elif self.stage == 'pose' and supported:
            self.stage = 'lift'
        elif self.stage == 'lift' and lift_ready:
            self.stage = 'align'; self.elapsed = self.settled = self.qualified = 0.
            self.verified = False
        elif self.stage == 'align' and self.verified:
            self.stage = 'lower'
        elif self.stage == 'lower' and supported:
            if self.leg == 3:
                self.stage = 'done'
            else:
                self.leg += 1; self.stage = 'lift'; self.verified = False
        else:
            return False
        return True

    def tick(self, dt, measured, measured_speed, *, lift_ready=False):
        q = np.asarray(measured); qd = np.asarray(measured_speed)
        if not 0 < dt <= .02 or q.shape != (4,) or qd.shape != (4,) or not np.isfinite([q,qd]).all():
            raise ValueError('Invalid controller sensors or interval')
        if self.stage != 'align':
            return self.reference.copy()
        k = self.leg
        if self.verified:
            if abs(q[k]-self.goal[k]) >= .035 or abs(qd[k]) >= .08:
                self.verified = False
                self.stage = 'fault'
            return self.reference.copy()
        self.elapsed += dt
        if self.elapsed > 60:
            self.reference[k] = q[k]; self.velocity = 0.; self.stage = 'fault'
            return self.reference.copy()
        self.qualified = self.qualified + dt if lift_ready else 0.
        if self.qualified < .2:
            if self.rotating:
                self.reference[k] = q[k]  # one-shot capture on gate loss
            self.rotating = False; self.velocity = self.settled = 0.
            return self.reference.copy()
        self.rotating = True
        error = self.goal[k] - self.reference[k]
        desired = float(np.clip(2*error, -self.max_speed, self.max_speed))
        self.velocity = float(np.clip(desired, self.velocity-self.max_accel*dt, self.velocity+self.max_accel*dt))
        advance = self.velocity*dt
        if abs(advance) >= abs(error) and advance*error >= 0:
            self.reference[k] = self.goal[k]; self.velocity = 0.
        else:
            self.reference[k] += advance
        at = abs(q[k]-self.goal[k]) < .025 and abs(qd[k]) < .08 and abs(error) < .001
        self.settled = self.settled+dt if at else 0.
        if self.settled >= .5:
            self.verified = True; self.velocity = 0.; self.reference[k] = self.goal[k]
        return self.reference.copy()

    def positions(self, action):
        """Expand eight actor actions to twelve position setpoints."""
        action = np.asarray(action)
        if action.shape != (8,) or not np.isfinite(action).all():
            raise ValueError('Expected eight finite proximal actions')
        target = c.DEFAULT_POSE.copy()
        target[c.PROXIMAL] += np.clip(action, -1, 1)*c.ACTION_SCALE[c.PROXIMAL]
        target = np.clip(target, c.JOINT_LOWER_LIMITS, c.JOINT_UPPER_LIMITS)
        target[c.HUBS] = self.reference
        return target
