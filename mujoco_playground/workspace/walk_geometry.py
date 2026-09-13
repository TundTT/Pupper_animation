"""Geometry shared by walking rewards, offline checks, and visualization.

The ring is a ground-plane clearance proxy, not a force sensor or deformable body.
Both rim edges are sampled so lateral tilt and rubbing are detected as well.
"""
from pathlib import Path
import hashlib
import json
import xml.etree.ElementTree as ET
import mujoco
import numpy as np

LEGS = ('front_r', 'front_l', 'back_r', 'back_l')
MODEL_PATH = Path(__file__).resolve().parents[2] / 'Stanford/training/pupper_v3_description/description/mujoco_xml/pupper_v3_complete.mjx.position.xml'

def load_walk_model(model_path=MODEL_PATH, foot_model='rigid_flush'):
    """Compile a flush, stiff capsule; retain the original model for old runs.

    Extend only the distal cap center. Compilation updates all collision bounds,
    and the visual ring stays untouched. The ring outline supplies the axial
    outer edge, not a guessed mesh bounding-box dimension.
    """
    original = mujoco.MjModel.from_xml_path(str(model_path))
    if foot_model == 'legacy_soft':
        return original
    if foot_model != 'rigid_flush':
        raise ValueError(f'Unknown foot_model: {foot_model}')
    _, _, rings, _ = load_geometry(original, model_path)
    spec = mujoco.MjSpec.from_file(str(model_path))
    for leg, ring in zip(LEGS, rings):
        capsule = spec.geom(f'leg_{leg}_3_foot_collision')
        endpoints = np.array(capsule.fromto).reshape(2, 3)
        axis = endpoints[1] - endpoints[0]
        length = np.linalg.norm(axis)
        axis /= length
        extension = np.max((ring - endpoints[0]) @ axis) - length - capsule.size[0]
        if extension < 0.:
            raise ValueError('Flush calibration would shorten the capsule')
        endpoints[1] += extension * axis
        capsule.fromto = endpoints.ravel()
        spec.site(f'leg_{leg}_3_foot_site').pos = endpoints[1]
    # Both members of each floor/foot pair use the same stiff parameters. Raising
    # geom priority instead would also override the requested floor friction.
    for name in ['floor', *[f'leg_{leg}_3_foot_collision' for leg in LEGS]]:
        collision = spec.geom(name)
        collision.solref = [2 * original.opt.timestep, 1.]
        collision.solimp = [.99, .99, .001, .5, 2.]
    model = spec.compile()
    # Re-settle the home keyframe for the longer, stiff feet. The old home height
    # included soft-contact sinking and is no longer a consistent target/reset.
    data = mujoco.MjData(model)
    data.qpos[:] = model.key('home').qpos
    data.qpos[2] += .01
    for _ in range(round(4. / model.opt.timestep)):
        mujoco.mj_step(model, data)
    if not np.all(np.isfinite(data.qpos)) or np.linalg.norm(data.qvel) > .01:
        raise ValueError('Rigid capsule neutral pose did not settle')
    model.key('home').qpos[:] = data.qpos
    return model

def save_effective_model(model, path, source_path=MODEL_PATH):
    """Save the compiled overrides for direct inspection in the MuJoCo viewer."""
    mujoco.mj_saveLastXML(str(path), model)
    tree = ET.parse(path)
    compiler = tree.getroot().find('compiler')
    # A run folder is outside the source asset tree; resolve its mesh directory.
    meshdir = compiler.get('meshdir', '.')
    compiler.set('meshdir', str((Path(source_path).resolve().parent / meshdir).resolve()))
    tree.write(path, encoding='unicode')

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

def quat_axis_z(q, xp=np):
    """World-frame direction of the local +Z axis after rotating by quaternion q=(w,x,y,z).

    Used to turn a (possibly randomized/tilted) floor geom_quat into a ground-normal
    vector, so reward code can measure height-above-ground as a projection onto the
    true floor normal instead of assuming the floor is exactly the world XY-plane.
    """
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return xp.stack([2*(x*z+y*w), 2*(y*z-x*w), 1-2*(x*x+y*y)], axis=-1)

def touchdown_cost(previous_contact, contact, previous_normal_velocity, allowance=.15, xp=np):
    """Once per airborne-to-contact transition, tax excess approach speed.

    Use the preceding airborne sample: the contact solver may already have
    removed downward velocity in the current sample. Persistent contact and
    upward motion carry no cost. This is a velocity proxy, not impact force.
    """
    hit=(~previous_contact)&contact
    return xp.sum(hit*xp.maximum(-previous_normal_velocity-allowance,0.)**2)

def swing_duration_reward(air_time, touchdown, floor=-.08, xp=np):
    """Score only touchdown: short swings cost, longer swings earn credit."""
    return xp.sum(xp.clip(air_time-.08,floor,.25)*touchdown)

def linear_tracking_reward(velocity, command, variance_base=.0025, speed_gain=.55, xp=np, yaw_rate=0., yaw_gain=.27):
    """Tighten low-speed tracking without making fast tracking much sharper."""
    variance=xp.minimum(.07,variance_base+speed_gain*xp.sum(command**2)+yaw_gain*yaw_rate**2)
    return xp.exp(-xp.sum((velocity-command)**2)/variance)

def yaw_tracking_reward(velocity, command, xp=np):
    variance=xp.minimum(.25,.05+.8*command**2)
    return xp.exp(-(velocity-command)**2/variance)

def clearance_shortfall_cost(peak, previous_contact, contact, target=.004, xp=np):
    """Discourage replanting after a barely-airborne shuffle, once per event."""
    hit=(~previous_contact)&contact
    return xp.sum(hit*xp.maximum(1.-peak/target,0.)**2)

def tip_support_cost(alignment, contact, allowance_cos, scale_cos, xp=np):
    """Soft contact-only guard on alignment of the designated tip with ground.

    One at the scale angle (45 degrees by default), zero inside the allowance
    (15 degrees). Saturates at four per foot, then averages all four feet.
    Alignment is an orientation proxy, not a ring force measurement.
    """
    excess=xp.clip((allowance_cos-alignment)/(allowance_cos-scale_cos),0.,2.)
    return xp.mean(contact*excess**2)

def planned_swing_cost(elapsed, previous_contact, contact, height, dt, duration=.16, apex=.003, xp=np, bonus=0., weights=None):
    """A lift starts a full rise/fall window; early touchdown cannot erase it.

    Elapsed -1 means inactive. This is an endogenous, reward-only per-foot clock,
    not an externally prescribed diagonal phase or a new policy observation.
    Horizontal placement is free. Cap normalized errors to limit outlier costs.
    """
    start=previous_contact & ~contact & (elapsed<0.)
    time=xp.where(start,0.,xp.where(elapsed>=0.,elapsed+dt,-1.))
    active=(time>=0.) & (time<duration)
    phase=xp.clip(time/duration,0.,1.)
    reference=apex*xp.sin(xp.pi*phase)**2
    # Weights strengthen error correction for under-lifting feet, without raising
    # their swing bonus or making in-place stepping more profitable.
    weights=1. if weights is None else weights
    cost=xp.sum(active*(weights*xp.clip((height-reference)/apex,-2.,2.)**2-bonus))
    return xp.where(active,time,-1.),cost,reference,active

def ring_costs(points, velocities, bottom_mask, bottom_allowance=.006, side_clearance=.001, floor_normal=None, xp=np):
    # Defaults to world +Z (flat ground), which reduces exactly to points[...,2] --
    # existing flat-ground callers see no behavior change.
    normal = xp.array([0., 0., 1.]) if floor_normal is None else floor_normal
    height = points @ normal
    side = ~bottom_mask
    # Max per foot avoids reward dependence on the outline sampling density.
    side_depth = xp.max(xp.where(side, xp.maximum(side_clearance-height, 0.), 0.), axis=1)
    bottom_depth = xp.max(xp.where(bottom_mask, xp.maximum(-height-bottom_allowance, 0.), 0.), axis=1)
    touch = xp.clip((.001-height)/.002, 0., 1.)
    rub = xp.max(xp.where(side, touch * xp.sum(velocities[..., :2]**2, axis=-1), 0.), axis=1)
    return {
        # Depth ratio capped at 2 before squaring: each aggregate cost is at
        # most 4 (not below linear tracking's ceiling of 2). These are soft,
        # saturating penalties; no prohibition or physical contact force.
        'ring_side': xp.mean(xp.minimum(side_depth/.01, 2.)**2),
        'ring_bottom': xp.mean(xp.minimum(bottom_depth/.01, 2.)**2),
        'ring_rub': xp.mean(rub),
        'ring_side_fraction': xp.mean(xp.any((height < 0) & side, axis=1).astype(float)),
        'ring_penetration_m': xp.max(xp.maximum(-height, 0.)),
    }
