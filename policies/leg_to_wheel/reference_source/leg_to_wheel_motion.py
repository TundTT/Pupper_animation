"""Ease the lowering hip's descent at the actuator boundary.

Every target comes from the policy. Abduction, knee/hub, support legs and upward
hip corrections stay responsive. The evaluator fades this constraint out before
the final approach and retains raw policy actions in the observation history.
"""
import numpy as np


def limit_lowering_targets(action, previous, foot, dt, action_scale, max_speed, strength=1.):
    """Blend a downward hip target limit in rad/s; foot=-1 disables it."""
    if max_speed < 0 or not np.isfinite(max_speed):
        raise ValueError('Lowering speed must be finite and nonnegative')
    if not 0 <= strength <= 1:
        raise ValueError('Lowering strength must be between zero and one')
    result = np.array(action, copy=True)
    if foot < 0 or max_speed == 0 or strength == 0:
        return result
    if foot > 3 or dt <= 0:
        raise ValueError('Expected foot 0..3 and a positive control timestep')
    joint = 3*foot+1  # lifting hip (_2), not abduction or knee/hub
    lift_sign = (1., -1., 1., -1.)[foot]
    delta = max_speed*dt/float(action_scale[joint])
    if (result[joint]-previous[joint])*lift_sign < -delta:
        limited = previous[joint]-lift_sign*delta
        result[joint] += strength*(limited-result[joint])
    return result
