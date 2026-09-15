"""Geometry shared by walking rewards, offline checks, and visualization.

The ring is a ground-plane clearance proxy, not a force sensor or deformable body.
Both rim edges are sampled so lateral tilt and rubbing are detected as well.
"""
from pathlib import Path
import hashlib
import json
import mujoco
import numpy as np

LEGS = ('front_r', 'front_l', 'back_r', 'back_l')
MODEL_PATH = Path(__file__).resolve().parents[2] / 'Stanford/training/pupper_v3_description/description/mujoco_xml/pupper_v3_complete.mjx.position.xml'

def load_geometry(model, model_path=MODEL_PATH):
    source = json.loads(Path(__file__).with_name('ring_outline.json').read_text())
    stl = Path(model_path).resolve().parent.parent / 'meshes/stl/CustomLegFoot.stl'
    if hashlib.sha256(stl.read_bytes()).hexdigest() != source['mesh_sha256']:
        raise ValueError('CustomLegFoot.stl changed: regenerate and review ring_outline.json before training.')
    mid = model.mesh('CustomLegFoot').id
    rotation = np.zeros(9)
    mujoco.mju_quat2Mat(rotation, model.mesh_quat[mid])
    native = (np.array(source['points']) - model.mesh_pos[mid]) @ rotation.reshape(3, 3)
    bodies, geoms, ring = [], [], []
    for leg in LEGS:
        body = model.body(f'leg_{leg}_3').id
        geom = model.geom(f'leg_{leg}_3_foot_collision').id
        if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_CAPSULE:
            raise ValueError(f'{leg}: expected a capsule foot')
        visual = np.flatnonzero((model.geom_bodyid == body) & (model.geom_dataid == mid) & (model.geom_group == 1))
        if len(visual) != 1:
            raise ValueError(f'{leg}: expected one CustomLegFoot visual')
        mujoco.mju_quat2Mat(rotation, model.geom_quat[visual[0]])
        ring.append(native @ rotation.reshape(3, 3).T + model.geom_pos[visual[0]])
        bodies.append(body); geoms.append(geom)
    return np.array(bodies), np.array(geoms), np.array(ring), np.array(source['bottom_mask'])

def world_points(body_pos, body_mat, local_points, xp=np):
    return body_pos[:, None, :] + xp.einsum('bij,bpj->bpi', body_mat.reshape(-1, 3, 3), local_points)

def point_velocities(points, com, cvel, xp=np):
    # MuJoCo cvel is COM-based [angular, linear], in world orientation.
    return cvel[:, None, 3:] + xp.cross(cvel[:, None, :3], points - com[:, None, :])

def capsule_bottom(centers, matrices, sizes, xp=np):
    axis = matrices.reshape(-1, 3, 3)[:, :, 2]
    # fromto orders can flip the axis, so choose the lower cap independently per leg.
    lower_center = centers - axis * xp.where(axis[:, 2:3] >= 0, 1., -1.) * sizes[:, 1:2]
    return lower_center - xp.array([0., 0., 1.]) * sizes[:, 0:1]

def ring_costs(points, velocities, bottom_mask, bottom_allowance=.006, side_clearance=.001, xp=np):
    height = points[..., 2]
    side = ~bottom_mask
    # Max per foot avoids reward dependence on the outline sampling density.
    side_depth = xp.max(xp.where(side, xp.maximum(side_clearance-height, 0.), 0.), axis=1)
    bottom_depth = xp.max(xp.where(bottom_mask, xp.maximum(-height-bottom_allowance, 0.), 0.), axis=1)
    touch = xp.clip((.001-height)/.002, 0., 1.)
    rub = xp.max(xp.where(side, touch * xp.sum(velocities[..., :2]**2, axis=-1), 0.), axis=1)
    return {
        # Capped at 2 (not 5): at reward_scales.ring_side=-1, a per-foot cap of 4
        # keeps the worst case below tracking_linear's own ceiling instead of
        # dwarfing it, so the ring behaves as a strong penalty rather than an
        # effectively-infinite one that swamps the task reward.
        'ring_side': xp.mean(xp.minimum(side_depth/.01, 2.)**2),
        'ring_bottom': xp.mean(xp.minimum(bottom_depth/.01, 2.)**2),
        'ring_rub': xp.mean(rub),
        'ring_side_fraction': xp.mean(xp.any((height < 0) & side, axis=1).astype(float)),
        'ring_penetration_m': xp.max(xp.maximum(-height, 0.)),
    }
