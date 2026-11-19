"""Fast physics checks before training; no JAX required."""
import json
import mujoco
import numpy as np
from workspace.walk_geometry import MODEL_PATH, load_geometry, world_points, ring_costs


def check_model(seconds=20):
    m = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    bodies, feet, local, mask = load_geometry(m)
    results = {}
    for mode in ('default', 'home'):
        d = mujoco.MjData(m)
        if mode == 'home':
            mujoco.mj_resetDataKeyframe(m, d, m.key('home').id)
        max_tilt = 0.
        for _ in range(round(seconds / m.opt.timestep)):
            mujoco.mj_step(m, d)
            tilt = np.degrees(np.arccos(np.clip(d.xmat[1].reshape(3, 3)[2, 2], -1, 1)))
            max_tilt = max(max_tilt, float(tilt))
        floor = m.geom('floor').id
        hits = {int(c.geom2 if c.geom1 == floor else c.geom1) for c in d.contact
                if floor in (c.geom1, c.geom2) and c.dist <= 0}
        assert hits == set(feet), (mode, hits, set(feet))
        assert max_tilt < 2. and d.qpos[2] > .12 and np.linalg.norm(d.qvel) < .01
        assert all(w.number == 0 for w in d.warning)
        points = world_points(d.xpos[bodies], d.xmat[bodies], local)
        costs = ring_costs(points, np.zeros_like(points), mask)
        assert costs['ring_side'] == 0. and costs['ring_bottom'] == 0., costs
        results[mode] = dict(max_tilt_deg=max_tilt, height_m=float(d.qpos[2]),
                             foot_contacts=len(hits), ring_penetration_mm=float(costs['ring_penetration_m'] * 1000))
    return results


if __name__ == '__main__':
    print(json.dumps(check_model(), indent=2))
