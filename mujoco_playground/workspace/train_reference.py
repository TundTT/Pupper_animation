"""Reference-pipeline walking trainer.

A local, non-Colab port of `Pupper_RL_PUBLIC.ipynb` (the original Stanford/Nathan
Kau notebook). Reuses `Stanford/training/pupperv3-mjx`'s environment, reward,
domain_randomization, export and utils modules UNCHANGED -- this is the original
proven pipeline (reward shaping, PPO hyperparameters, and domain randomization),
not the ring-clearance/tip-tilt/planned-swing/leg-length-DR additions built up in
this branch's walk_env.py/walk_config.py/walk_randomize.py. The only substantive
differences from the notebook:
  - Points at our vendored, already-customized model (backpack, 9 mm outward gap,
    CustomLegFoot ring) instead of a fresh `git clone` of the upstream description
    repo, and at the already-vendored pupperv3-mjx package instead of cloning it.
  - Colab-only plumbing (drive mount, userdata secrets, apt/pip installs, EGL ICD
    bootstrap, runtime.unassign()) is removed in favor of plain local setup.
  - See PupperReferenceEnv below for the one required environment-level deviation.

Run detached (per project convention, SSH sessions can drop mid-run):
    nohup .venv/bin/python -m workspace.train_reference > train_reference.log 2>&1 &
    disown
"""
import argparse
import functools
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

if subprocess.run(['nvidia-smi'], capture_output=True).returncode != 0:
    raise RuntimeError('No GPU visible to nvidia-smi; this trainer needs a CUDA GPU.')

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ['XLA_FLAGS'] = os.environ.get('XLA_FLAGS', '') + ' --xla_gpu_triton_gemm_any=True'

import matplotlib
matplotlib.use('Agg')  # Headless: no display when launched via nohup.

import jax
import jax.image
from jax import numpy as jp
import mujoco
import numpy as np
from brax import envs
from brax.io import model as brax_model
from brax.training.agents.ppo import train as ppo, networks as ppo_networks, losses as ppo_losses
from ml_collections import config_dict
import wandb

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PUPPERV3_MJX = _REPO_ROOT / 'Stanford/training/pupperv3-mjx'
sys.path.insert(0, str(_PUPPERV3_MJX))
from pupperv3_mjx import environment, domain_randomization, export, utils  # noqa: E402

MODEL_PATH = _REPO_ROOT / 'Stanford/training/pupper_v3_description/description/mujoco_xml/pupper_v3_complete.mjx.position.xml'
LEGS = ('front_r', 'front_l', 'back_r', 'back_l')


def build_terrain_model(base_path, output_path, grid_size=256, radius_x=10.0, radius_y=10.0,
                         elevation_z=0.02, base_z=0.2, seed=0):
    """Port of the notebook's height-field cell (Modify robot model > Add height
    field ground): replaces the flat floor with a procedurally-rough hfield
    terrain, written to a new model file -- the vendored source model is untouched.
    """
    tree = ET.parse(str(base_path))
    root = tree.getroot()

    noise = np.array(jax.random.uniform(jax.random.PRNGKey(seed), (grid_size, grid_size)))
    area_shape = (max(1, int(grid_size // radius_x)), max(1, int(grid_size // radius_y)))
    area_noise = jax.random.uniform(jax.random.PRNGKey(seed + 1), area_shape)
    upscaled_area_noise = np.array(jax.image.resize(area_noise, (grid_size, grid_size), method='nearest'))
    scaled_noise = noise * upscaled_area_noise

    asset = root.find('asset')
    ET.SubElement(asset, 'hfield', name='hfield_geom', nrow=str(grid_size), ncol=str(grid_size),
                  elevation=' '.join(scaled_noise.astype(str).flatten().tolist()),
                  size=f'{radius_x} {radius_y} {elevation_z} {base_z}')

    # Replace the flat floor plane with the hfield geom (same material, so the
    # rendered look stays close to the flat-ground videos) -- both floor geoms
    # are declared after every leg/body element in this file, so removing them
    # does not renumber any earlier geom/site id (including the feet, which
    # FOOT_GEOM_IDS/FOOT_SITE_IDS below assume stay put).
    worldbody = root.find('worldbody')
    for name in ('floor', 'floor_visual'):
        geom = worldbody.find(f"./geom[@name='{name}']")
        if geom is not None:
            worldbody.remove(geom)
    ET.SubElement(worldbody, 'geom', name='hfield_floor', type='hfield', hfield='hfield_geom',
                  material='grid', friction='0.8 0.02 0.01', condim='3', contype='1', conaffinity='1')

    # Non-flat terrain needs more contact points/geom pairs than the flat-floor
    # default -- the notebook recommends 20/20 for exactly this case.
    tree = utils.set_mjx_custom_options(tree, max_contact_points=20, max_geom_pairs=20)
    if tree is None:
        raise ValueError('Model has no <custom> block to set max_contact_points/max_geom_pairs on.')

    # The compiler's meshdir is relative to the source file's own directory;
    # written elsewhere (e.g. a training_runs/ output folder), it must be made
    # absolute (same fix as walk_geometry.save_effective_model).
    compiler = root.find('compiler')
    meshdir = compiler.get('meshdir', '.')
    compiler.set('meshdir', str((Path(base_path).resolve().parent / meshdir).resolve()))

    tree.write(str(output_path), encoding='unicode')


def foot_capsule_frame(mj_model):
    """Per-leg geom/site ids, local +Z axis, sign, and nominal radius for the
    foot capsules -- computed from whichever compiled model is actually being
    trained on (geom ids shift if the floor geoms are swapped for a heightfield,
    since MuJoCo orders geoms by body -- world-attached geoms like the floor
    come before any leg geoms -- not by XML document order; NEVER reuse ids
    computed from a different model variant). Used by randomize_leg_length to
    move each capsule's distal (tip) cap, and the foot site marking that same
    point, while keeping the proximal (hub) end fixed."""
    geom_ids = np.zeros(4, dtype=int)
    site_ids = np.zeros(4, dtype=int)
    axis = np.zeros((4, 3))
    sign = np.zeros(4)
    radius = np.zeros(4)
    for i, leg in enumerate(LEGS):
        gid = mj_model.geom(f'leg_{leg}_3_foot_collision').id
        sid = mj_model.site(f'leg_{leg}_3_foot_site').id
        geom_ids[i] = gid
        site_ids[i] = sid
        radius[i] = mj_model.geom_size[gid, 0]
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, mj_model.geom_quat[gid])
        local_z = rot.reshape(3, 3)[:, 2]
        # The foot site marks the capsule's distal (tip) cap center -- derive
        # which sign of the local +Z axis points that way rather than assume it.
        to_site = mj_model.site_pos[sid] - mj_model.geom_pos[gid]
        sign[i] = 1.0 if np.dot(to_site, local_z) >= 0 else -1.0
        axis[i] = local_z
    return geom_ids, site_ids, axis, sign, radius


def randomize_leg_length(sys, rng, length_range_m, foot_frame):
    """Independently resample each of the 4 feet's overall capsule length
    (hub-to-tip, including the rounded caps) within `length_range_m`, keeping
    the proximal (hub) end fixed. Moves the distal cap's geom_pos/geom_size and
    the foot site (which marks that same point) together, so ground-contact
    detection (foot_radius-based) and foot_slip stay consistent with the new
    length -- this is manufacturing/wear variance in the foot+ring assembly,
    independent of the thigh-length randomization this project deliberately
    does not use here. `foot_frame` is a foot_capsule_frame(...) result for
    the SAME compiled model as `sys`.
    """
    foot_geom_ids, foot_site_ids, axis_local, foot_sign, foot_radius = foot_frame
    axis = jp.asarray(axis_local)
    sign = jp.asarray(foot_sign)
    radius = jp.asarray(foot_radius)
    nominal_half = sys.geom_size[foot_geom_ids, 1]

    @jax.vmap
    def sample(key):
        key = jax.random.fold_in(key, 991827)  # decorrelate from domain_randomize's own splits
        new_length = jax.random.uniform(key, (4,), minval=length_range_m[0], maxval=length_range_m[1])
        new_half = new_length / 2. - radius
        delta_half = new_half - nominal_half
        geom_pos = sys.geom_pos.at[foot_geom_ids].add(sign[:, None] * delta_half[:, None] * axis)
        site_pos = sys.site_pos.at[foot_site_ids].add(2. * sign[:, None] * delta_half[:, None] * axis)
        geom_size = sys.geom_size.at[foot_geom_ids, 1].set(new_half)
        return geom_pos, site_pos, geom_size

    return sample(rng)


def combined_domain_randomize(sys, rng, leg_length_range_m, foot_frame, **domain_randomize_kwargs):
    """The unmodified reference domain_randomize plus independent per-leg
    foot-capsule-length DR (see randomize_leg_length), kept as a separate
    wrapper so pupperv3_mjx.domain_randomization stays unedited."""
    sys, in_axes = domain_randomization.domain_randomize(sys, rng, **domain_randomize_kwargs)
    geom_pos, site_pos, geom_size = randomize_leg_length(sys, rng, leg_length_range_m, foot_frame)
    sys = sys.tree_replace({'geom_pos': geom_pos, 'site_pos': site_pos, 'geom_size': geom_size})
    in_axes = in_axes.tree_replace({'geom_pos': 0, 'site_pos': 0, 'geom_size': 0})
    return sys, in_axes


class PupperReferenceEnv(environment.PupperV3Env):
    """PupperV3Env, adapted to this project's actuator convention.

    Our vendored MJCF bakes the home-pose offset directly into each actuator's
    `biasprm` -- ctrl=0 already holds home, and ctrl is a joint-angle OFFSET from
    home (see WALKING.md), matching the deployed neural_controller/exported-policy
    convention. Upstream PupperV3Env instead assumes ctrl IS the absolute joint
    angle: it adds `default_pose` on top of every action and overwrites the home
    keyframe with it. Forcing a zero default_pose fixes the ctrl math; this then
    restores the true standing pose for episode resets, which that zeroed-out
    keyframe overwrite would otherwise destroy. Reward calculation, observation
    construction and everything else is untouched upstream code.
    """

    def __init__(self, *args, full_home_qpos, **kwargs):
        kwargs['default_pose'] = jp.zeros(12)
        super().__init__(*args, **kwargs)
        self._init_q = jp.array(full_home_qpos)


class PupperStanceCapEnv(PupperReferenceEnv):
    """PupperReferenceEnv plus a penalty for continuous ground stance beyond
    `max_stance_s`, gated to active commands the same way reward_feet_air_time
    is (no penalty for legitimately standing still under a near-zero command).

    reward_feet_air_time is exactly zero for a foot that never leaves the
    ground -- first_contact can only fire after feet_air_time has already
    become positive, so it cannot create any gradient away from "never lift"
    (see the walk-quality investigation this followed from). This term is
    active from the very first step regardless of prior lifting, by directly
    tracking how long each foot has been continuously planted.
    """

    def __init__(self, *args, stance_penalty_weight=-5.0, max_stance_s=0.4, **kwargs):
        super().__init__(*args, **kwargs)
        self._stance_penalty_weight = stance_penalty_weight
        self._max_stance_s = max_stance_s

    def reset(self, rng):
        state = super().reset(rng)
        state.info['stance_time'] = jp.zeros(4)
        # metrics keys must match between reset() and step() output -- brax's
        # AutoReset/ActionRepeat wrappers scan over state and require an
        # identical pytree structure every iteration.
        state.metrics['stance_cap_cost'] = jp.zeros(())
        return state

    def step(self, state, action):
        stance_time = state.info['stance_time']
        state = super().step(state, action)
        contact = state.info['last_contact']
        new_stance_time = jp.where(contact, stance_time + self.dt, 0.0)
        moving = jp.linalg.norm(state.info['command'][:3]) > 0.05
        excess = jp.clip(new_stance_time - self._max_stance_s, 0., None)
        cost = self._stance_penalty_weight * jp.sum(excess ** 2) * moving * self.dt
        state.info['stance_time'] = new_stance_time
        state.metrics['stance_cap_cost'] = cost
        return state.replace(reward=state.reward + cost)


class PupperClearanceEnv(PupperReferenceEnv):
    """PupperReferenceEnv plus a dense per-step reward for foot height while
    airborne, capped at `target_clearance_m`.

    Unlike reward_feet_air_time (only evaluated once per touchdown, and only
    then if the foot was already airborne), this fires every single step a
    foot is off the ground, giving an immediate gradient toward lifting from a
    fully static start.
    """

    def __init__(self, *args, clearance_weight=2.0, target_clearance_m=0.02, **kwargs):
        super().__init__(*args, **kwargs)
        self._clearance_weight = clearance_weight
        self._target_clearance_m = target_clearance_m

    def reset(self, rng):
        state = super().reset(rng)
        # metrics keys must match between reset() and step() output -- see
        # PupperStanceCapEnv.reset for why.
        state.metrics['clearance_bonus'] = jp.zeros(())
        return state

    def step(self, state, action):
        state = super().step(state, action)
        foot_pos = state.pipeline_state.site_xpos[self._feet_site_id]
        foot_height = jp.clip(foot_pos[:, 2] - self._foot_radius, 0., None)
        airborne = ~state.info['last_contact']
        moving = jp.linalg.norm(state.info['command'][:3]) > 0.05
        bonus = self._clearance_weight * jp.sum(
            jp.minimum(foot_height, self._target_clearance_m) * airborne
        ) * moving * self.dt
        state.metrics['clearance_bonus'] = bonus
        return state.replace(reward=state.reward + bonus)


def build_config(args, model_path=MODEL_PATH):
    mj_model = mujoco.MjModel.from_xml_path(str(model_path))
    foot_frame = foot_capsule_frame(mj_model)
    full_home_qpos = jp.array(mj_model.keyframe('home').qpos.copy())
    home_joint_angles = jp.array(mj_model.keyframe('home').qpos[7:].copy())
    joint_lower_limits = mj_model.jnt_range[1:, 0].copy()
    joint_upper_limits = mj_model.jnt_range[1:, 1].copy()

    simulation_config = config_dict.ConfigDict()
    simulation_config.model_path = str(model_path)
    simulation_config.upper_leg_body_names = ['leg_front_r_2', 'leg_front_l_2', 'leg_back_r_2', 'leg_back_l_2']
    simulation_config.lower_leg_body_names = ['leg_front_r_3', 'leg_front_l_3', 'leg_back_r_3', 'leg_back_l_3']
    simulation_config.foot_site_names = [
        'leg_front_r_3_foot_site', 'leg_front_l_3_foot_site',
        'leg_back_r_3_foot_site', 'leg_back_l_3_foot_site',
    ]
    simulation_config.torso_name = 'base_link'
    # Our CustomLegFoot collision capsule's actual radius (see the model XML);
    # the notebook's foot_radius=0.02 was for the original, different foot mesh.
    simulation_config.foot_radius = 0.012
    simulation_config.joint_upper_limits = joint_upper_limits.tolist()
    simulation_config.joint_lower_limits = joint_lower_limits.tolist()
    simulation_config.physics_dt = 0.004

    training_config = config_dict.ConfigDict()
    training_config.checkpoint_run_number = None
    training_config.environment_dt = 0.02

    training_config.ppo = config_dict.ConfigDict()
    training_config.ppo.num_timesteps = args.num_timesteps
    training_config.ppo.episode_length = 500
    training_config.ppo.num_evals = 11
    training_config.ppo.reward_scaling = 1
    training_config.ppo.normalize_observations = True
    training_config.ppo.action_repeat = 1
    training_config.ppo.unroll_length = 20
    training_config.ppo.num_minibatches = 32
    training_config.ppo.num_updates_per_batch = 4
    training_config.ppo.discounting = 0.97
    training_config.ppo.learning_rate = args.learning_rate
    training_config.ppo.entropy_cost = 1e-2
    training_config.ppo.num_envs = args.num_envs
    training_config.ppo.batch_size = 256

    training_config.resample_velocity_step = training_config.ppo.episode_length // 2
    # Capped from +-0.75. Ring gap is set by stride geometry:
    #   gap ~= 150mm (same-side foot separation at home) - v * duty / f
    # measured 71mm on seq_c at v=0.32, duty=0.53, f=2.16 against a predicted 72mm.
    # Rings (r=48.8mm) touch below 98mm, so the clash-free speed at seq_c's 2.16Hz
    # is only ~0.21 m/s. Note this means raising duty makes ring clash WORSE at
    # constant speed (duty 0.75 at 0.32 m/s would give a 36mm gap), so duty and ring
    # clearance are in direct conflict unless step frequency rises. Capping the
    # command envelope keeps the achievable stride inside what the geometry allows
    # and lets w_ring_clash push toward higher frequency rather than longer strides.
    training_config.lin_vel_x_range = [-0.18, 0.18]
    # Strafing is dropped from the command set by decision, because it is
    # kinematically out of reach on this robot rather than merely untrained.
    # Sweeping each leg's three joints over hardware limits and keeping poses that
    # stay within +-5mm of the home stance height (z = 10.6mm) gives a load-bearing
    # lateral foot offset of ~11mm OUTWARD and ~74mm inward, measured symmetrically
    # on front_r (-11.1) and front_l (+10.1).
    # The binding constraint is NOT the abduction joint limit: outward headroom from
    # the settled home (abduction +0.027) is 0.347 rad, which is ample. It is that
    # abducting outward lifts the foot off the stance plane, so the pose stops
    # bearing load long before the joint runs out. Strafe is bound by that outward
    # reach, so the body translates ~11mm per gait cycle, i.e. ~0.014 m/s at 1.31Hz
    # -- about 35x below the +-0.5 range this used to request. With tracking_sigma
    # 0.25, going from 0.00 to the full achievable 0.014 m/s earns ~0.02 reward,
    # less than the abduction_angle penalty and torque cost of doing it, so PPO
    # correctly learned vy = 0 and every eval showed strafe at exactly 0.00 m/s.
    # Keeping the range open spent ~40% of command samples on an impossible task.
    # vy stays in the observation (all zeros) so the 36-per-frame layout and the
    # exported robot contract are unchanged; only command_high/low narrow.
    training_config.lin_vel_y_range = [0.0, 0.0]
    training_config.ang_vel_yaw_range = [-2.0, 2.0]
    training_config.zero_command_probability = 0.02
    training_config.stand_still_command_threshold = 0.05
    training_config.maximum_pitch_command = 0.0
    training_config.maximum_roll_command = 0.0
    training_config.desired_world_z_in_body_frame = (0.0, 0.0, 1.0)

    training_config.terminal_body_z = 0.05
    training_config.terminal_body_angle = 0.70
    training_config.early_termination_step_threshold = training_config.ppo.episode_length // 2

    training_config.dof_damping = 0.25
    training_config.position_control_kp = 5.0

    # True standing pose (absolute joint angles), read from the model's own home
    # keyframe. Used for episode resets and export metadata -- NOT as a ctrl bias
    # (see PupperReferenceEnv).
    training_config.default_pose = home_joint_angles
    training_config.full_home_qpos = full_home_qpos

    training_config.desired_abduction_angles = jp.array([0.0, 0.0, 0.0, 0.0])

    training_config.kick_probability = 0.04
    training_config.kick_vel = 0.10
    training_config.angular_velocity_noise = 0.1
    training_config.gravity_noise = 0.05
    training_config.motor_angle_noise = 0.05
    training_config.last_action_noise = 0.01

    training_config.position_control_kp_multiplier_range = (0.6, 1.1)
    training_config.position_control_kd_multiplier_range = (0.8, 1.5)

    training_config.start_position_config = domain_randomization.StartPositionRandomization(
        x_min=-2.0, x_max=2.0, y_min=-2.0, y_max=2.0, z_min=0.15, z_max=0.20
    )

    training_config.latency_distribution = jp.array([0.2, 0.8])
    training_config.imu_latency_distribution = jp.array([0.5, 0.5])

    training_config.body_com_x_shift_range = (-0.02, 0.03)
    training_config.body_com_y_shift_range = (-0.005, 0.005)
    training_config.body_com_z_shift_range = (-0.005, 0.005)
    training_config.body_mass_scale_range = (0.9, 1.3)
    training_config.body_inertia_scale_range = (0.9, 1.3)
    training_config.friction_range = (0.6, 1.4)

    policy_config = config_dict.ConfigDict()
    policy_config.use_imu = True
    policy_config.observation_history = 4
    # Upstream default: one scalar action_scale for all 12 joints. Our joints have
    # asymmetric reachable ranges around home (this project's own workspace pipeline
    # uses per-joint scales for exactly that reason) -- kept as the proven baseline
    # for this first reference-pipeline run; revisit if training struggles on the
    # narrow-range joints (abduction).
    policy_config.action_scale = 0.75
    policy_config.hidden_layer_sizes = (256, 128, 128, 128)
    policy_config.activation = 'elu'

    reward_config = config_dict.ConfigDict()
    reward_config.rewards = config_dict.ConfigDict()
    reward_config.rewards.scales = config_dict.ConfigDict()
    reward_config.rewards.scales.tracking_lin_vel = 1.5
    reward_config.rewards.scales.tracking_ang_vel = 0.8
    reward_config.rewards.scales.tracking_orientation = 0.5
    reward_config.rewards.scales.lin_vel_z = -0.1
    reward_config.rewards.scales.ang_vel_xy = -0.002
    reward_config.rewards.scales.orientation = -0.0
    reward_config.rewards.scales.torques = -0.025
    reward_config.rewards.scales.joint_acceleration = -1e-6
    reward_config.rewards.scales.mechanical_work = 0
    reward_config.rewards.scales.action_rate = -0.1
    # Back to the reference default. Raising this (and foot_slip below) to fix
    # dragging made the gait actively worse/unnatural when trained from scratch
    # -- feet_air_time structurally can't reward a first lift (it's exactly
    # zero for a foot that never leaves the ground; see the walk-quality
    # investigation), so cranking it just discouraged the dragging escape
    # valve without adding one for lifting. Current approach: warm-start from
    # lilac-snowflake-2 (this exact reference config, judged the best-looking
    # gait so far) and nudge lift height with a small explicit clearance bonus
    # (see PupperClearanceEnv / --warm_start_params_path) instead of retuning
    # the reference terms themselves.
    reward_config.rewards.scales.feet_air_time = 0.02
    reward_config.rewards.scales.stand_still = -0.00
    reward_config.rewards.scales.stand_still_joint_velocity = -0.2
    reward_config.rewards.scales.abduction_angle = -0.01
    reward_config.rewards.scales.termination = -100.0
    # Back to the reference default -- see feet_air_time comment above.
    reward_config.rewards.scales.foot_slip = -0.2
    reward_config.rewards.scales.knee_collision = -10.0
    reward_config.rewards.scales.body_collision = -0.5
    # MUST scale with lin_vel_x_range. The tracking reward is
    # 1.5 * exp(-|v-cmd|^2 / sigma), so sigma sets how much standing still costs.
    # At the original +-0.75 range, sigma 0.25 was fine. After capping speed to
    # +-0.18 for ring clearance, standing under a 0.18 command scored
    # 1.5*exp(-0.0324/0.25) = 1.32 against a perfect 1.50 -- a mere 0.18 reward for
    # walking at all, less than w_multi_swing or the drag/touchdown penalties. Both
    # seq_f and seq_g froze at vx = 0.00 for exactly this reason; it was the speed
    # cap flattening the velocity reward, NOT the move_gate exploit they were blamed
    # on. At sigma 0.04 standing scores 0.51 against 1.50, restoring a ~1.0 gradient.
    # getattr: build_reference_eval_env builds a minimal Namespace without this
    # field, and sigma only affects reward, never observations or the policy.
    reward_config.rewards.tracking_sigma = getattr(args, 'tracking_sigma', 0.25)

    config = config_dict.ConfigDict()
    config.simulation = simulation_config
    config.training = training_config
    config.policy = policy_config
    config.reward = reward_config
    return config_dict.FrozenConfigDict(config), foot_frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_timesteps', type=int, default=300_000_000)
    parser.add_argument('--num_envs', type=int, default=8192)
    parser.add_argument('--learning_rate', type=float, default=3.0e-4)
    parser.add_argument('--wandb_project', default='Walk')
    parser.add_argument('--wandb_entity', default='QuadMorph')
    parser.add_argument('--flat_ground', action='store_true',
                         help='Disable heightfield terrain randomization (default: randomized ground on).')
    parser.add_argument('--terrain_seed', type=int, default=0)
    parser.add_argument('--leg_length_min_m', type=float, default=0.056)
    # 60mm, not 61: the goal statement specifies 56-60mm. An earlier instruction
    # said 56-61mm, which is where the old default came from.
    parser.add_argument('--leg_length_max_m', type=float, default=0.060)
    parser.add_argument('--reward_shaping', choices=['none', 'stance_cap', 'clearance', 'shaped'], default='none',
                         help='Extra reward term to counter a dragging/never-lifts gait (see the '
                              'walk-quality investigation): "stance_cap" penalizes continuous ground '
                              'contact past a duration; "clearance" rewards foot height while airborne.')
    parser.add_argument('--w_foot_clearance', type=float, default=-2.0)
    parser.add_argument('--w_knee_clearance', type=float, default=-2.0)
    parser.add_argument('--w_tip_vertical', type=float, default=-1.0)
    parser.add_argument('--w_drag', type=float, default=-0.5)
    parser.add_argument('--w_touchdown_land', type=float, default=-1.0)
    parser.add_argument('--w_multi_swing', type=float, default=-1.5)
    parser.add_argument('--w_ring_clash', type=float, default=-3.0)
    parser.add_argument('--ring_clearance_m', type=float, default=0.105)
    parser.add_argument('--tracking_sigma', type=float, default=0.25,
                        help='Scale with lin_vel_x_range: ~0.04 for a +-0.18 cap, '
                             '0.25 for +-0.75. Too large and standing still is free.')
    parser.add_argument('--w_single_support', type=float, default=0.0)
    parser.add_argument('--w_over_lift', type=float, default=0.0)
    parser.add_argument('--over_lift_m', type=float, default=0.022)
    parser.add_argument('--target_clearance_m', type=float, default=0.022)
    parser.add_argument('--target_knee_height_m', type=float, default=0.068)
    parser.add_argument('--tip_allowance_deg', type=float, default=15.0)
    parser.add_argument('--tip_scale_deg', type=float, default=45.0)
    parser.add_argument('--warm_start_params_path', default=None,
                         help='Path to a brax_model.save_params(...) file (e.g. a prior run\'s '
                              'mjx_params_<timestamp>) to initialize training from, instead of random '
                              'init. NOT the vendored utils.save_checkpoint/orbax format -- that format '
                              'is incompatible with this installed brax\'s restore_checkpoint_path.')
    args = parser.parse_args()

    from workspace.shaped_env import PupperShapedEnv
    env_cls = {'none': PupperReferenceEnv, 'stance_cap': PupperStanceCapEnv,
               'clearance': PupperClearanceEnv, 'shaped': PupperShapedEnv}[args.reward_shaping]
    shaping_kwargs = {}
    if args.reward_shaping == 'shaped':
        shaping_kwargs = dict(
            w_foot_clearance=args.w_foot_clearance,
            w_knee_clearance=args.w_knee_clearance,
            w_tip_vertical=args.w_tip_vertical,
            w_drag=args.w_drag,
            w_touchdown=args.w_touchdown_land,
            w_multi_swing=args.w_multi_swing,
            w_ring_clash=args.w_ring_clash,
            ring_clearance_m=args.ring_clearance_m,
            w_single_support=args.w_single_support,
            w_over_lift=args.w_over_lift,
            over_lift_m=args.over_lift_m,
            target_clearance_m=args.target_clearance_m,
            target_knee_height_m=args.target_knee_height_m,
            tip_allowance_deg=args.tip_allowance_deg,
            tip_scale_deg=args.tip_scale_deg,
        )

    jax.config.update('jax_default_matmul_precision', 'high')

    train_datetime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_folder = _REPO_ROOT / 'training_runs' / f'reference_{train_datetime}'
    output_folder.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_PATH
    if not args.flat_ground:
        model_path = output_folder / 'model_with_terrain.xml'
        build_terrain_model(MODEL_PATH, model_path, seed=args.terrain_seed)

    CONFIG, foot_frame = build_config(args, model_path=model_path)

    envs.register_environment('pupper_reference', env_cls)
    env_kwargs = dict(
        path=CONFIG.simulation.model_path,
        action_scale=CONFIG.policy.action_scale,
        observation_history=CONFIG.policy.observation_history,
        joint_lower_limits=jp.array(CONFIG.simulation.joint_lower_limits),
        joint_upper_limits=jp.array(CONFIG.simulation.joint_upper_limits),
        dof_damping=CONFIG.training.dof_damping,
        position_control_kp=CONFIG.training.position_control_kp,
        foot_site_names=CONFIG.simulation.foot_site_names,
        torso_name=CONFIG.simulation.torso_name,
        upper_leg_body_names=CONFIG.simulation.upper_leg_body_names,
        lower_leg_body_names=CONFIG.simulation.lower_leg_body_names,
        resample_velocity_step=CONFIG.training.resample_velocity_step,
        linear_velocity_x_range=CONFIG.training.lin_vel_x_range,
        linear_velocity_y_range=CONFIG.training.lin_vel_y_range,
        angular_velocity_range=CONFIG.training.ang_vel_yaw_range,
        zero_command_probability=CONFIG.training.zero_command_probability,
        stand_still_command_threshold=CONFIG.training.stand_still_command_threshold,
        maximum_pitch_command=CONFIG.training.maximum_pitch_command,
        maximum_roll_command=CONFIG.training.maximum_roll_command,
        start_position_config=CONFIG.training.start_position_config,
        full_home_qpos=CONFIG.training.full_home_qpos,
        desired_abduction_angles=CONFIG.training.desired_abduction_angles,
        reward_config=CONFIG.reward,
        angular_velocity_noise=CONFIG.training.angular_velocity_noise,
        gravity_noise=CONFIG.training.gravity_noise,
        motor_angle_noise=CONFIG.training.motor_angle_noise,
        last_action_noise=CONFIG.training.last_action_noise,
        kick_vel=CONFIG.training.kick_vel,
        kick_probability=CONFIG.training.kick_probability,
        terminal_body_z=CONFIG.training.terminal_body_z,
        early_termination_step_threshold=CONFIG.training.early_termination_step_threshold,
        terminal_body_angle=CONFIG.training.terminal_body_angle,
        foot_radius=CONFIG.simulation.foot_radius,
        environment_timestep=CONFIG.training.environment_dt,
        physics_timestep=CONFIG.simulation.physics_dt,
        latency_distribution=CONFIG.training.latency_distribution,
        imu_latency_distribution=CONFIG.training.imu_latency_distribution,
        desired_world_z_in_body_frame=jp.array(CONFIG.training.desired_world_z_in_body_frame),
        use_imu=CONFIG.policy.use_imu,
        **shaping_kwargs,
    )

    wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        config={**CONFIG.to_dict(), 'reward_shaping': args.reward_shaping},
        save_code=True,
        settings={'_service_wait': 90, 'init_timeout': 90},
    )

    env = envs.get_environment('pupper_reference', **env_kwargs)
    eval_env = envs.get_environment('pupper_reference', **env_kwargs)
    jit_reset = jax.jit(eval_env.reset)
    jit_step = jax.jit(eval_env.step)

    make_networks_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=CONFIG.policy.hidden_layer_sizes,
        activation=utils.activation_fn_map(CONFIG.policy.activation),
    )
    train_fn = functools.partial(
        ppo.train,
        **CONFIG.training.ppo.to_dict(),
        network_factory=make_networks_factory,
        randomization_fn=functools.partial(
            combined_domain_randomize,
            leg_length_range_m=(args.leg_length_min_m, args.leg_length_max_m),
            foot_frame=foot_frame,
            friction_range=CONFIG.training.friction_range,
            kp_multiplier_range=CONFIG.training.position_control_kp_multiplier_range,
            kd_multiplier_range=CONFIG.training.position_control_kd_multiplier_range,
            body_com_x_shift_range=CONFIG.training.body_com_x_shift_range,
            body_com_y_shift_range=CONFIG.training.body_com_y_shift_range,
            body_com_z_shift_range=CONFIG.training.body_com_z_shift_range,
            body_mass_scale_range=CONFIG.training.body_mass_scale_range,
            body_inertia_scale_range=CONFIG.training.body_inertia_scale_range,
        ),
        seed=28,
    )

    x_data, y_data, ydataerr, times = [], [], [], [datetime.now()]

    def policy_params_fn(current_step, make_policy, params):
        # Our installed brax (0.14.2) hands policy_params_fn a flat
        # (normalizer, policy, value) tuple; pupperv3_mjx.utils was written
        # against an older brax where the last two were bundled into a single
        # PPONetworkParams object (params[1].policy). Re-bundle here rather than
        # edit the vendored reference module.
        normalizer_params, policy_params, value_params = params
        compat_params = (normalizer_params, ppo_losses.PPONetworkParams(policy=policy_params, value=value_params))
        # utils.visualize_policy renders a rollout, writes an mp4, and logs it as
        # wandb.Video under "eval/video" -- this is what puts it under the run's
        # Media tab. utils.save_checkpoint logs the orbax checkpoint as a model
        # artifact. Both are unchanged upstream code.
        utils.visualize_policy(
            current_step=current_step, make_policy=make_policy, params=compat_params,
            eval_env=eval_env, jit_step=jit_step, jit_reset=jit_reset,
            output_folder=str(output_folder),
        )
        utils.save_checkpoint(
            current_step=current_step, make_policy=make_policy, params=compat_params,
            checkpoint_path=output_folder,
        )

    warm_start_kwargs = {}
    if args.warm_start_params_path is not None:
        warm_start_kwargs['restore_params'] = brax_model.load_params(args.warm_start_params_path)
        print(f'Warm-starting from {args.warm_start_params_path}')

    make_inference_fn, params, _ = train_fn(
        environment=env,
        progress_fn=functools.partial(
            utils.progress, times=times, x_data=x_data, y_data=y_data, ydataerr=ydataerr,
            num_timesteps=CONFIG.training.ppo.num_timesteps, min_y=0, max_y=40,
        ),
        eval_env=eval_env,
        policy_params_fn=policy_params_fn,
        **warm_start_kwargs,
    )

    print(f'time to jit: {times[1] - times[0]}')
    print(f'time to train: {times[-1] - times[1]}')
    wandb.run.summary['time_to_jit'] = (times[1] - times[0]).total_seconds()
    wandb.run.summary['time_to_train'] = (times[-1] - times[1]).total_seconds()

    params_path = output_folder / f'mjx_params_{train_datetime}'
    brax_model.save_params(str(params_path), params)

    # Export the HARDWARE position limits, not the MJCF ones. The MJCF envelope is
    # 0.1 rad wider per joint, so exporting it lets the deployed controller (which
    # clips to these values) command abduction past the physical limit --
    # robot_info/validate_policy.py rejects exactly this:
    #   "joint 1 target envelope [-0.42, 0.777] exceeds hardware [-0.32, 3.04]".
    from workspace import gait_env as _ge
    params_rtneural = export.convert_params(
        jax.block_until_ready(params),
        activation=CONFIG.policy.activation,
        action_scale=CONFIG.policy.action_scale,
        kp=CONFIG.training.position_control_kp,
        kd=CONFIG.training.dof_damping,
        default_pose=CONFIG.training.default_pose,
        joint_upper_limits=_ge.HARDWARE_POSITION_MAX,
        joint_lower_limits=_ge.HARDWARE_POSITION_MIN,
        use_imu=CONFIG.policy.use_imu,
        observation_history=CONFIG.policy.observation_history,
        maximum_pitch_command=CONFIG.training.maximum_pitch_command,
        maximum_roll_command=CONFIG.training.maximum_roll_command,
        final_activation='tanh',
    )
    # Deployment metadata required by origin/robot-info robot_info/POLICY_INTERFACE.md.
    import hashlib as _hl
    params_rtneural['behavior'] = 'locomotion'
    params_rtneural['hardware_profile'] = 'leg_position'
    params_rtneural['robot_contract_schema_version'] = 1
    params_rtneural['joint_names'] = list(_ge.CANONICAL_JOINTS)
    params_rtneural['action_types'] = ['position'] * 12
    params_rtneural['action_scale'] = [float(CONFIG.policy.action_scale)] * 12
    params_rtneural['observation_layout'] = [
        'body_angular_velocity_xyz[3]', 'projected_gravity_xyz[3]',
        'command_vx_vy_yaw_rate[3]', 'desired_world_z_in_body_xyz[3]',
        'joint_position_minus_default[12]', 'previous_faded_normalized_action[12]']
    params_rtneural['command_low'] = [CONFIG.training.lin_vel_x_range[0],
                                      CONFIG.training.lin_vel_y_range[0],
                                      CONFIG.training.ang_vel_yaw_range[0]]
    params_rtneural['command_high'] = [CONFIG.training.lin_vel_x_range[1],
                                       CONFIG.training.lin_vel_y_range[1],
                                       CONFIG.training.ang_vel_yaw_range[1]]
    params_rtneural['orientation_command'] = list(CONFIG.training.desired_world_z_in_body_frame)
    params_rtneural['training_control_dt'] = CONFIG.training.environment_dt
    params_rtneural['training_model_sha256'] = _hl.sha256(
        Path(CONFIG.simulation.model_path).read_bytes()).hexdigest()
    try:
        params_rtneural['source_commit'] = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=str(_REPO_ROOT)).decode().strip()
    except Exception:
        params_rtneural['source_commit'] = 'unknown'
    policy_name = f'policy_{wandb.run.name}.json'
    policy_path = output_folder / policy_name
    with open(policy_path, 'w') as f:
        json.dump(params_rtneural, f)
    wandb.log_model(path=str(policy_path), name=policy_name)
    wandb.finish()
    print(f'Exported policy: {policy_path}')


if __name__ == '__main__':
    main()
