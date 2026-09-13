import mujoco
import numpy as np
from workspace import walk_geometry as g
from workspace.walk_env import PupperWalkEnv
from workspace.evaluate_walk_obstacles import add_obstacles


def test_obstacle_is_physical_and_foot_ids_are_resolved_after_compilation():
    env=PupperWalkEnv()
    m,centers=add_obstacles(env,.01)
    foot=m.geom('leg_front_r_3_foot_collision').id
    obstacle=m.geom('obstacle_0').id
    assert foot!=env.foot_ids[0]  # World geoms shift the compiled foot IDs.
    d=mujoco.MjData(m);d.qpos[:]=np.asarray(env.init_q);mujoco.mj_forward(m,d)
    point=g.capsule_bottom(d.geom_xpos[[foot]],d.geom_xmat[[foot]],m.geom_size[[foot]])[0]
    d.qpos[0]+=centers[0]-point[0]
    d.qpos[2]+=.0095-point[2]
    mujoco.mj_forward(m,d)
    hits=[x for x in d.contact if x.dist<0 and set(x.geom)=={foot,obstacle}]
    assert hits
    assert abs(hits[0].frame[2])>.7
    np.testing.assert_allclose(hits[0].friction[:2],2.)
    np.testing.assert_allclose(hits[0].solref,[.008,1.])
