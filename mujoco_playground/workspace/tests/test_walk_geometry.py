import mujoco as mj
import numpy as np
from workspace import walk_geometry as g
from workspace.check_walk import check_model


def test_default_and_keyframe_hold_on_four_feet():
    check_model(seconds=20)


def test_side_rubbing_and_bottom_allowance_are_distinct():
    points = np.array([[[0., 0., .05], [0., 0., -.004]]] * 4)
    bottom = np.array([False, True])
    vel = np.zeros_like(points)
    c = g.ring_costs(points, vel, bottom)
    assert c['ring_side'] == c['ring_bottom'] == c['ring_rub'] == 0
    points[:, 0, 2] = -.003
    contact = g.ring_costs(points, vel, bottom)
    assert contact['ring_side'] > 0 and contact['ring_rub'] == 0
    vel[:, 0, 0] = .2
    assert g.ring_costs(points, vel, bottom)['ring_rub'] > 0
    points[:, 1, 2] = -.009
    assert g.ring_costs(points, vel, bottom)['ring_bottom'] > 0
    points[:, :, 2] = .1
    assert g.ring_costs(points, vel, bottom)['ring_rub'] == 0


def test_capsule_lowest_point_in_arbitrary_orientations():
    rng = np.random.default_rng(7)
    for _ in range(10):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        quat = ' '.join(map(str, q))
        m = mj.MjModel.from_xml_string(f'<mujoco><worldbody><geom type="capsule" pos="0 0 .04" size=".012 .018" quat="{quat}"/></worldbody></mujoco>')
        d = mj.MjData(m)
        mj.mj_forward(m, d)
        point = g.capsule_bottom(d.geom_xpos[:1], d.geom_xmat[:1], m.geom_size[:1])[0]
        axis = d.geom_xmat[0].reshape(3, 3)[:, 2]
        np.testing.assert_allclose(point[2], .04 - .018 * abs(axis[2]) - .012, atol=1e-12)


def test_probe_point_velocity_accounts_for_rotation():
    p = np.array([[[1., 0., 0.]]])
    v = np.array([[0., 0., 2., .5, 0., 0.]])
    np.testing.assert_allclose(g.point_velocities(p, np.zeros((1, 3)), v), [[[.5, 2., 0.]]])


def test_ring_penalty_detects_lateral_tilt():
    m = mj.MjModel.from_xml_path(str(g.MODEL_PATH))
    d = mj.MjData(m)
    bodies, _, points, mask = g.load_geometry(m)
    mj.mj_resetDataKeyframe(m, d, 0)
    d.qpos[3:7] = [np.cos(.25), np.sin(.25), 0., 0.]
    mj.mj_forward(m, d)
    world = g.world_points(d.xpos[bodies], d.xmat[bodies], points)
    assert g.ring_costs(world, np.zeros_like(world), mask)['ring_side'] > 0
