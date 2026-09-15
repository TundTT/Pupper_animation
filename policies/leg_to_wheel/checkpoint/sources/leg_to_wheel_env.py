"""MJX balance and lift policy with the deployment-compatible 35-value frame.

The policy controls all joints; an external sequencer owns the operator command.
Foot geometry and dynamics come from the current walking model unchanged.
"""
import jax
from jax import numpy as jp
import mujoco
from mujoco import mjx
import numpy as np
from brax.envs.base import Env, State
from workspace.leg_to_wheel_config import get_config, COMMAND_TO_FOOT
from workspace import walk_geometry as geom

class PupperLegToWheelEnv(Env):
    def __init__(self, config=None, model_path=None):
        self.config = config or get_config()
        self.model_path = str(model_path or geom.MODEL_PATH)
        self.mj_model = m = mujoco.MjModel.from_xml_path(self.model_path)
        self.sys = mjx.put_model(m, impl='jax')
        self.frames = int(round(self.config.ctrl_dt / m.opt.timestep))
        if not np.isclose(self.frames*m.opt.timestep, self.config.ctrl_dt):
            raise ValueError('Control timestep must be an integer multiple of simulation timestep')
        body, feet, ring, mask = geom.load_geometry(m, self.model_path)
        self.body_ids, self.foot_ids = body, feet
        self.ring_local, self.bottom_mask = jp.array(ring), jp.array(mask)
        self.torso = m.body('base_link').id
        self.floor = m.geom('floor').id
        if m.geom_type[self.floor] != mujoco.mjtGeom.mjGEOM_PLANE or not np.allclose(m.geom_pos[self.floor], 0):
            raise ValueError('Ring clearance currently requires the world-Z=0 ground plane')
        self.home = jp.array(m.qpos0[7:])
        self.init_q = jp.array(m.key('home').qpos)
        self.height = float(m.key('home').qpos[2])
        self.action_scale = jp.array(self.config.action_scale)
        self.touchdown_warmup_steps = round(self.config.touchdown_warmup_seconds/self.dt)
        self.limits = jp.array(m.jnt_range[1:])
        self.sizes = jp.array(m.geom_size[feet])
        self.root_ids = m.body_rootid[body]
        self.names = tuple(self.config.reward_scales.keys())
        np.testing.assert_allclose(m.actuator_biasprm[:,0], m.actuator_gainprm[:,0]*m.qpos0[7:])
        if m.nu != 12:
            raise ValueError('Expected the canonical 12 position actuators')

    @property
    def observation_size(self): return 35*self.config.observation_history
    @property
    def action_size(self): return 12
    @property
    def backend(self): return 'mjx'
    @property
    def dt(self): return self.config.ctrl_dt

    def sample_command(self, key):
        leg_key, stand_key = jax.random.split(key)
        index = jp.where(jax.random.bernoulli(stand_key, self.config.stand_probability),
                         0, jax.random.randint(leg_key, (), 1, 5))
        return jax.nn.one_hot(index, 5)

    def sample_hold(self, key):
        lo, hi = self.config.command_hold_seconds
        return jax.random.randint(key, (), max(1, round(lo/self.dt)), round(hi/self.dt)+1)

    def commanded_mask(self, command):
        foot = jp.array(COMMAND_TO_FOOT)[jp.argmax(command)]
        return jp.arange(4) == foot

    def observation(self, d, command, action, key):
        rot = d.xmat[self.torso].reshape(3, 3)
        frame = jp.concatenate([rot.T@d.cvel[self.torso, :3], rot.T@jp.array([0., 0., -1.]),
                                command, d.qpos[7:]-self.home, action])
        amplitude = jp.array([self.config.sensor_noise]*6 + [0.]*5 +
                             [self.config.sensor_noise]*12 + [0.]*12)
        return frame + amplitude*jax.random.uniform(key, (35,), minval=-1., maxval=1.)

    def set_command(self, state, index):
        """External command, held until changed; no episode timeout or resampling.

        Preserve real sensor/action history. Only the newest command slot changes.
        """
        command = jax.nn.one_hot(index, 5)
        info = dict(state.info, command=command, external_command=jp.array(True),
                    reset_next=jp.array(False))
        return state.replace(info=info, obs=state.obs.at[6:11].set(command))

    def foot_signals(self, d):
        points = geom.capsule_bottom(d.geom_xpos[self.foot_ids], d.geom_xmat[self.foot_ids], self.sizes, jp)
        velocity = geom.point_velocities(points[:, None, :], d.subtree_com[self.root_ids],
                                         d.cvel[self.body_ids], jp)[:, 0, :]
        return points, velocity

    def reset(self, rng):
        rng, ck, qk, xyk, yk, ok, hk = jax.random.split(rng, 7)
        q = self.init_q.at[7:].add(jax.random.uniform(qk,(12,), minval=-self.config.reset_joint_noise,maxval=self.config.reset_joint_noise))
        q = q.at[:2].add(jax.random.uniform(xyk,(2,),minval=-self.config.reset_xy_noise,maxval=self.config.reset_xy_noise))
        # Rotate the settled base attitude by a small random world yaw.
        yaw = jax.random.uniform(yk, (), minval=-self.config.reset_yaw_noise,maxval=self.config.reset_yaw_noise)
        from mujoco.mjx._src import math
        q = q.at[3:7].set(math.quat_mul(jp.array([jp.cos(yaw/2),0.,0.,jp.sin(yaw/2)]),q[3:7]))
        d = mjx.forward(self.sys, mjx.make_data(self.sys).replace(qpos=q))
        command = self.sample_command(ck)
        frame = self.observation(d,command,jp.zeros(12),ok)
        info = dict(rng=rng, command=command, initial_command=command, last_action=jp.zeros(12),
                    last_contact=self.contact_mask(d)[0], step=jp.array(0), reset_next=jp.array(False),
                    command_steps_left=self.sample_hold(hk), init_xy=q[:2], external_command=jp.array(False))
        metrics = {k: jp.array(0.) for k in (*self.names,'ring_side_fraction','ring_penetration_m','foot_contacts','tilt_deg','torso_z','lifted_clearance_m','body_drift_m','touchdown_speed_mps','stance_missing')}
        return State(d,jp.tile(frame,self.config.observation_history),jp.array(0.),jp.array(0.),metrics,info)

    def contact_mask(self,d):
        contact = d._impl.contact
        floor_pair = (contact.geom[:,0]==self.floor)|(contact.geom[:,1]==self.floor)
        active = floor_pair & (contact.dist <= 0.) & (contact.efc_address >= 0)
        hits = (contact.geom[:,None,:]==jp.array(self.foot_ids)[None,:,None]).any(axis=-1)
        feet = (active[:,None]&hits).any(axis=0)
        unwanted = (active & ~hits.any(axis=1)).any().astype(float)
        return feet, unwanted

    def ring_signals(self,d):
        points = geom.world_points(d.xpos[self.body_ids],d.xmat[self.body_ids],self.ring_local,jp)
        velocities = geom.point_velocities(points,d.subtree_com[self.root_ids],d.cvel[self.body_ids],jp)
        return geom.ring_costs(points,velocities,self.bottom_mask,self.config.bottom_allowance,self.config.side_clearance,jp)

    def step(self,state,action):
        c = self.config
        info = dict(state.info)
        fresh = info['reset_next']
        prev = jp.where(fresh,jp.zeros(12),info['last_action'])
        age = jp.where(fresh,0,info['step'])
        old_contact = jp.where(fresh,self.contact_mask(state.pipeline_state)[0],info['last_contact'])
        rng, ck, lk, ok, fk, pk1, pk2 = jax.random.split(info['rng'],7)
        # Brax restores reset observations before the next action. Score that
        # action against the command it actually saw, then expose a new command.
        command = jp.where(fresh,info['initial_command'],info['command'])
        initial_xy = jp.where(fresh,state.pipeline_state.qpos[:2],info['init_xy'])
        command_mask = self.commanded_mask(command)
        action = jp.clip(action,-1.,1.)
        applied = jp.where(jax.random.bernoulli(lk,c.latency_probability),prev,action)
        target = jp.clip(self.home + applied*self.action_scale,self.limits[:,0],self.limits[:,1])
        d = state.pipeline_state.replace(ctrl=target-self.home)
        # Occasional random horizontal velocity kick: the sim otherwise never
        # produces a disturbance, so the policy has no incentive to reject one.
        kick = jax.random.uniform(pk2,(2,),minval=-c.push_velocity,maxval=c.push_velocity)*jax.random.bernoulli(pk1,c.push_probability)
        d = d.replace(qvel=d.qvel.at[:2].add(kick))
        def substep(carry, _):
            d, previous_contact = carry
            _, before_velocity = self.foot_signals(d)
            d = mjx.forward(self.sys, mjx.step(self.sys,d))
            substep_contact, _ = self.contact_mask(d)
            impact = substep_contact & ~previous_contact
            impact_speed = jp.maximum(-before_velocity[:, 2], 0.) * impact
            return (d, substep_contact), (self.ring_signals(d), impact_speed)
        (d, _), (ring_history, impact_history) = jax.lax.scan(
            substep, (d, old_contact), None, length=self.frames)
        ring = jax.tree.map(lambda x: jp.mean(x,axis=0),ring_history)
        contact, unwanted = self.contact_mask(d)
        points, foot_velocity = self.foot_signals(d)
        rot = d.xmat[self.torso].reshape(3,3)
        torso_v = geom.point_velocities(d.xpos[self.torso][None,None,:],d.subtree_com[self.torso][None,:],d.cvel[self.torso][None,:],jp)[0,0]
        v = rot.T@torso_v
        omega = rot.T@d.cvel[self.torso,:3]
        tilt = jp.arccos(jp.clip(rot[2,2],-1.,1.))
        done = (tilt>c.terminal_tilt)|(d.qpos[2]<c.terminal_height)|(~jp.all(jp.isfinite(d.qpos)))
        lifted_height = jp.sum(points[:, 2]*command_mask)
        drift = jp.linalg.norm(d.qpos[:2]-initial_xy)
        stance_missing = jp.sum((~contact) & (~command_mask)).astype(float)
        low, high = self.limits[:,0]+c.joint_limit_margin,self.limits[:,1]-c.joint_limit_margin
        terms = dict(
            lift_height=jp.sum(command_mask)*jp.clip(lifted_height/c.lift_clearance_target,0.,1.),
            # A grounded capsule's height reward is flat. This small companion,
            # proven in leg_lift_env, pays for beginning the unloading motion.
            lift_progress=jp.sum(command_mask*jp.clip(
                (d.qpos[8::3]-self.home[1::3])*jp.array([1.,-1.,1.,-1.])/
                c.lift_progress_reference,0.,1.)),
            stance_contact=stance_missing,
            # Sum impact events across simulation substeps, using PRE-impact speed.
            soft_touchdown=jp.where(age>=self.touchdown_warmup_steps,
                jp.sum((impact_history/c.soft_touchdown_speed)**2),0.),
            body_drift=(jp.maximum(drift-c.allowed_body_drift,0.)/.04)**2,
            upright=rot[2,2], height=((d.qpos[2]-self.height)/.04)**2,
            vertical_velocity=v[2]**2, roll_pitch_velocity=jp.sum(omega[:2]**2),
            foot_slip=jp.sum(jp.sum(foot_velocity[:,:2]**2,axis=1)*contact),
            ring_side=ring['ring_side'],ring_bottom=ring['ring_bottom'],ring_rub=ring['ring_rub'],
            unwanted_contact=unwanted, action_rate=jp.mean((action-prev)**2),
            torques=jp.sum(d.actuator_force**2),
            joint_limits=jp.sum(jp.maximum(low-d.qpos[7:],0.)+jp.maximum(d.qpos[7:]-high,0.)),
            termination=done.astype(float),
        )
        scaled = {k: terms[k]*c.reward_scales[k] for k in self.names}
        reward = sum(scaled.values())*self.dt
        # Reward belongs to the old command; the returned observation exposes
        # the next command before the policy computes its next action.
        switch = (fresh | (info['command_steps_left'] <= 1)) & ~info['external_command']
        next_command = jp.where(switch,self.sample_command(ck),command)
        left = jp.where(switch,self.sample_hold(fk),info['command_steps_left']-1)
        obs = jp.roll(state.obs,35).at[:35].set(self.observation(d,next_command,action,ok))
        info.update(rng=rng,command=next_command,last_action=action,
                    last_contact=contact,step=age+1, command_steps_left=left, init_xy=initial_xy,
                    reset_next=(done|((age+1)>=c.episode_length)) & ~info['external_command'])
        metrics = {**state.metrics,**scaled,'ring_side_fraction':ring['ring_side_fraction'],
                   'ring_penetration_m':ring['ring_penetration_m'],
                   'foot_contacts':jp.sum(contact).astype(float),'tilt_deg':tilt*180/jp.pi,'torso_z':d.qpos[2],
                   'lifted_clearance_m':lifted_height,'body_drift_m':drift,
                   'touchdown_speed_mps':jp.max(impact_history),'stance_missing':stance_missing}
        return state.replace(pipeline_state=d,obs=obs,reward=reward,done=done.astype(float),metrics=metrics,info=info)
