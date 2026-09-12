"""Coordinate and command-envelope checks; no hardware or motion certification."""
import mujoco as mj
import numpy as np
import pytest
from .core import Robot, HUB, PROX, SPEED, ACCEL
from .roll_to_stand import initialize, commands

@pytest.mark.parametrize('direction', ['forward', 'backward'])
def test_winding_is_same_physical_start_and_finishes_at_policy_default(direction):
    original=Robot();r,home,goal=initialize(direction)
    np.testing.assert_allclose(r.d.xpos,original.d.xpos,atol=1e-10)
    np.testing.assert_allclose(r.d.xmat,original.d.xmat,atol=1e-10)
    delta=goal[HUB]-home[HUB]
    assert np.all(delta*np.array([-1,1,-1,1])*(1 if direction=='forward' else -1)>0)
    np.testing.assert_allclose(abs(delta),np.pi,atol=1e-12)
    trajectory=commands(home,goal,12.)
    np.testing.assert_array_equal(trajectory[-1],goal)
    velocity=np.diff(np.vstack([home,trajectory]),axis=0)*520
    acceleration=np.diff(np.vstack([np.zeros(12),velocity]),axis=0)*520
    assert np.max(abs(velocity)/SPEED)<=1.
    assert np.max(abs(acceleration)/ACCEL)<=1.
    limits=r.m.jnt_range[1:][PROX]
    assert np.all((trajectory[:,PROX]>=limits[:,0])&(trajectory[:,PROX]<=limits[:,1]))

def test_forward_hub_axes_produce_forward_rolling_tangent():
    r,home,goal=initialize()
    axes=r.d.xaxis[1+HUB]
    omega=axes*np.sign(goal[HUB]-home[HUB])[:,None]
    # With a floor contact below the hub, -omega cross r points along body +X.
    translation=-np.cross(omega,np.array([0.,0.,-1.]))
    assert np.all(translation[:,0]>.9)
