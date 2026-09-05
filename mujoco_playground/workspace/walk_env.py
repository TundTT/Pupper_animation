"""Flat-ground capsule-foot walking with a training-only rubber-ring proxy.

Uses MJX directly inside the Brax Env interface: Brax's MJCF importer rejects
nonzero joint refs, which this model uses for a stable drag-and-drop default.
"""
import jax
from jax import numpy as jp
import mujoco
from mujoco import mjx
import numpy as np
from brax.envs.base import Env, State
from workspace.walk_config import get_config
from workspace import walk_geometry as geom

class PupperWalkEnv(Env):
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
        self.limits = jp.array(m.jnt_range[1:])
        self.sizes = jp.array(m.geom_size[feet])
        self.root_ids = m.body_rootid[body]
        self.names = tuple(self.config.reward_scales.keys())
        np.testing.assert_allclose(m.actuator_biasprm[:,0], m.actuator_gainprm[:,0]*m.qpos0[7:])
        if m.nu != 12:
            raise ValueError('Expected the canonical 12 position actuators')

    @property
    def observation_size(self): return 36*self.config.observation_history
    @property
    def action_size(self): return 12
    @property
    def backend(self): return 'mjx'
    @property
    def dt(self): return self.config.ctrl_dt

    def sample_command(self, key):
        key, stand = jax.random.split(key)
        cmd = jax.random.uniform(key, (3,), minval=jp.array(self.config.command_low), maxval=jp.array(self.config.command_high))
        return jp.where(jax.random.bernoulli(stand, self.config.stand_probability), jp.zeros(3), cmd)

    def observation(self, d, command, action, key):
        rot = d.xmat[self.torso].reshape(3,3)
        frame = jp.concatenate([rot.T@d.cvel[self.torso,:3], rot.T@jp.array([0.,0.,-1.]),
                                command, jp.array([0.,0.,1.]), d.qpos[7:]-self.home, action])
        amplitude = jp.array([self.config.sensor_noise]*6 + [0.]*6 + [self.config.sensor_noise]*12 + [0.]*12)
        return frame + amplitude*jax.random.uniform(key, (36,), minval=-1., maxval=1.)

    def reset(self, rng):
        rng, ck, qk, xyk, yk, ok = jax.random.split(rng, 6)
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
                    air_time=jp.zeros(4), last_contact=jp.ones(4,dtype=bool), step=jp.array(0), reset_next=jp.array(False))
        metrics = {k: jp.array(0.) for k in (*self.names,'ring_side_fraction','ring_penetration_m','foot_contacts','tilt_deg','torso_z','velocity_error')}
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
        air = jp.where(fresh,jp.zeros(4),info['air_time'])
        old_contact = jp.where(fresh,jp.ones(4,dtype=bool),info['last_contact'])
        command = jp.where(fresh,info['initial_command'],info['command'])
        rng, ck, lk, ok = jax.random.split(info['rng'],4)
        action = jp.clip(action,-1.,1.)
        applied = jp.where(jax.random.bernoulli(lk,c.latency_probability),prev,action)
        target = jp.clip(self.home + applied*self.action_scale,self.limits[:,0],self.limits[:,1])
        d = state.pipeline_state.replace(ctrl=target-self.home)
        def substep(d,_):
            d = mjx.step(self.sys,d)
            return d,self.ring_signals(d)
        d, ring_history = jax.lax.scan(substep,d,None,length=self.frames)
        # step() integrates qpos after forward; refresh geometry to the resulting pose.
        d = mjx.forward(self.sys,d)
        ring = jax.tree.map(lambda x: jp.mean(x,axis=0),ring_history)
        contact, unwanted = self.contact_mask(d)
        points = geom.capsule_bottom(d.geom_xpos[self.foot_ids],d.geom_xmat[self.foot_ids],self.sizes,jp)
        foot_velocity = geom.point_velocities(points[:,None,:],d.subtree_com[self.root_ids],d.cvel[self.body_ids],jp)[:,0,:]
        rot = d.xmat[self.torso].reshape(3,3)
        torso_v = geom.point_velocities(d.xpos[self.torso][None,None,:],d.subtree_com[self.torso][None,:],d.cvel[self.torso][None,:],jp)[0,0]
        v = rot.T@torso_v
        omega = rot.T@d.cvel[self.torso,:3]
        tilt = jp.arccos(jp.clip(rot[2,2],-1.,1.))
        done = (tilt>c.terminal_tilt)|(d.qpos[2]<c.terminal_height)|(~jp.all(jp.isfinite(d.qpos)))
        moving = jp.linalg.norm(command)> .05
        air += self.dt
        touchdown = contact & ~old_contact
        low, high = self.limits[:,0]+.05,self.limits[:,1]-.05
        terms = dict(
            tracking_linear=jp.exp(-jp.sum((v[:2]-command[:2])**2)/.1),
            tracking_yaw=jp.exp(-(omega[2]-command[2])**2/.25),
            upright=rot[2,2], height=((d.qpos[2]-self.height)/.04)**2,
            vertical_velocity=v[2]**2, roll_pitch_velocity=jp.sum(omega[:2]**2),
            foot_slip=jp.sum(jp.sum(foot_velocity[:,:2]**2,axis=1)*contact),
            air_time=jp.sum(jp.clip(air-.08,0.,.25)*touchdown)*moving,
            stand_pose=jp.mean((d.qpos[7:]-self.home)**2)*(~moving),
            ring_side=ring['ring_side'],ring_bottom=ring['ring_bottom'],ring_rub=ring['ring_rub'],
            unwanted_contact=unwanted, action_rate=jp.mean((action-prev)**2),
            torques=jp.sum(d.actuator_force**2),
            joint_limits=jp.sum(jp.maximum(low-d.qpos[7:],0.)+jp.maximum(d.qpos[7:]-high,0.)),
            termination=done.astype(float),
        )
        scaled = {k: terms[k]*c.reward_scales[k] for k in self.names}
        reward = sum(scaled.values())*self.dt
        # Current transition is scored against the command that generated its action.
        next_command = jp.where((age+1)%c.command_hold_steps==0,self.sample_command(ck),command)
        obs = jp.roll(state.obs,36).at[:36].set(self.observation(d,next_command,action,ok))
        info.update(rng=rng,command=next_command,last_action=action,air_time=jp.where(contact,0.,air),
                    last_contact=contact,step=age+1,reset_next=done|((age+1)>=c.episode_length))
        metrics = {**state.metrics,**scaled,'ring_side_fraction':ring['ring_side_fraction'],'ring_penetration_m':ring['ring_penetration_m'],
                   'foot_contacts':jp.sum(contact).astype(float),'tilt_deg':tilt*180/jp.pi,'torso_z':d.qpos[2],
                   'velocity_error':jp.linalg.norm(v[:2]-command[:2])}
        return state.replace(pipeline_state=d,obs=obs,reward=reward,done=done.astype(float),metrics=metrics,info=info)
