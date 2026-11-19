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
        self.mj_model = m = geom.load_walk_model(self.model_path, self.config.foot_model)
        if not np.isfinite(self.config.floor_friction) or self.config.floor_friction < 0.:
            raise ValueError('floor_friction must be nonnegative')
        if self.config.floor_friction > 0.:
            m.geom_friction[m.geom('floor').id,0] = self.config.floor_friction
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
        self.smoothness_scale = self.action_scale/jp.array(self.config.smoothness_reference_scale)
        self.limits = jp.array(m.jnt_range[1:])
        self.sizes = jp.array(m.geom_size[feet])
        nominal = mujoco.MjData(m)
        nominal.qpos[:] = np.asarray(self.init_q)
        mujoco.mj_forward(m, nominal)
        self.nominal_foot_centers = jp.array(nominal.geom_xpos[feet])
        self.nominal_foot_matrices = jp.array(nominal.geom_xmat[feet])
        tip_sites=np.array([m.site(f'leg_{leg}_3_foot_site').id for leg in geom.LEGS])
        axes=nominal.geom_xmat[feet].reshape(-1,3,3)[:,:,2]
        self.tip_sign=jp.array(np.sign(np.sum((nominal.site_xpos[tip_sites]-nominal.geom_xpos[feet])*axes,axis=1)))
        self.tip_allowance_cos=float(np.cos(np.deg2rad(self.config.tip_tilt_allowance_deg)))
        self.tip_scale_cos=float(np.cos(np.deg2rad(self.config.tip_tilt_scale_deg)))
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
        mode=jax.random.uniform(jax.random.fold_in(key,1))
        straight=mode<self.config.straight_probability
        single_axis=(mode>=self.config.straight_probability)&(mode<self.config.straight_probability+self.config.axis_probability)
        forward=cmd[0]>=0.
        maximum=jp.where(forward,self.config.command_high[0],-self.config.command_low[0])
        speed=jax.random.uniform(jax.random.fold_in(key,2),(),minval=self.config.straight_min_speed,maxval=maximum)
        cmd=jp.where(straight,jp.array([jp.where(forward,speed,-speed),0.,0.]),cmd)
        turn=jax.random.bernoulli(jax.random.fold_in(key,3))
        axis=jp.where(turn,2,1)
        positive=cmd[axis]>=0.
        maximum=jp.where(positive,jp.array(self.config.command_high)[axis],-jp.array(self.config.command_low)[axis])
        speed=jax.random.uniform(jax.random.fold_in(key,4),(),minval=jp.where(turn,.2,.05),maxval=maximum)
        axis_cmd=jp.zeros(3).at[axis].set(jp.where(positive,speed,-speed))
        cmd=jp.where(single_axis,axis_cmd,cmd)
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
                    last_action_delta=jp.zeros(12),
                    height_reference=self.neutral_height_reference(),
                    impact_contact=self.contact_mask(d)[0], impact_normal_velocity=jp.zeros(4),
                    swing_peak=jp.zeros(4),
                    swing_elapsed=-jp.ones(4),
                    air_time=jp.zeros(4), last_contact=jp.ones(4,dtype=bool), step=jp.array(0), reset_next=jp.array(False))
        metrics = {k: jp.array(0.) for k in (*self.names,'ring_side_fraction','ring_penetration_m','foot_contacts','tilt_deg','torso_z','torso_height','height_reference','action_delta_rms','velocity_error','yaw_error','tip_tilt_deg','tip_lean_fraction')}
        return State(d,jp.tile(frame,self.config.observation_history),jp.array(0.),jp.array(0.),metrics,info)

    def neutral_height_reference(self):
        """Adapt nominal height to this slot's length and mount geometry at home.

        Evaluate identical, noise-free root/joint poses. Only the reference nominal
        geometry is cached; self.sys is the actual randomized model inside Brax's
        vmap. The reference persists across auto-resets with the slot's geometry.
        """
        neutral = mjx.kinematics(self.sys, mjx.make_data(self.sys).replace(qpos=self.init_q))
        normal = geom.quat_axis_z(self.sys.geom_quat[self.floor], jp)
        # Exact capsule support projection, including endpoint selection on slopes.
        def bottom_height(centers, matrices):
            axes = matrices.reshape(-1, 3, 3)[:, :, 2]
            return centers@normal - self.sizes[:, 1]*jp.abs(axes@normal) - self.sizes[:, 0]
        delta = bottom_height(neutral.geom_xpos[self.foot_ids], neutral.geom_xmat[self.foot_ids]) - bottom_height(self.nominal_foot_centers, self.nominal_foot_matrices)
        return self.height - jp.mean(delta)

    def torso_height(self, d):
        normal = geom.quat_axis_z(self.sys.geom_quat[self.floor], jp)
        return d.xpos[self.torso]@normal

    def contact_mask(self,d):
        contact = d._impl.contact
        floor_pair = (contact.geom[:,0]==self.floor)|(contact.geom[:,1]==self.floor)
        active = floor_pair & (contact.dist <= 0.) & (contact.efc_address >= 0)
        hits = (contact.geom[:,None,:]==jp.array(self.foot_ids)[None,:,None]).any(axis=-1)
        feet = (active[:,None]&hits).any(axis=0)
        unwanted = (active & ~hits.any(axis=1)).any().astype(float)
        return feet, unwanted

    def ring_signals(self,d,floor_normal=None):
        points = geom.world_points(d.xpos[self.body_ids],d.xmat[self.body_ids],self.ring_local,jp)
        velocities = geom.point_velocities(points,d.subtree_com[self.root_ids],d.cvel[self.body_ids],jp)
        return geom.ring_costs(points,velocities,self.bottom_mask,self.config.bottom_allowance,self.config.side_clearance,floor_normal,jp)

    def step(self,state,action):
        return self.step_with_diagnostics(state,action)[0]

    def step_with_diagnostics(self,state,action):
        """Ordinary transition plus 250 Hz foot traces for matched evaluation.

        JAX prunes unused trace outputs from step() during training.
        """
        c = self.config
        info = dict(state.info)
        fresh = info['reset_next']
        prev = jp.where(fresh,jp.zeros(12),info['last_action'])
        previous_delta = jp.where(fresh,jp.zeros(12),info['last_action_delta'])
        age = jp.where(fresh,0,info['step'])
        air = jp.where(fresh,jp.zeros(4),info['air_time'])
        old_contact = jp.where(fresh,jp.ones(4,dtype=bool),info['last_contact'])
        rng, ck, lk, ok, fk, pk1, pk2 = jax.random.split(info['rng'],7)
        # Resample on every auto-reset instead of replaying the one command this env
        # slot happened to draw at its very first reset: with command_hold_steps=150
        # and episode_length=600, replaying `initial_command` meant the first 25% of
        # every episode always retraced one of only num_envs fixed commands.
        command = jp.where(fresh,self.sample_command(fk),info['command'])
        moving = jp.linalg.norm(command)> .05
        action = jp.clip(action,-1.,1.)
        applied = jp.where(jax.random.bernoulli(lk,c.latency_probability),prev,action)
        target = jp.clip(self.home + applied*self.action_scale,self.limits[:,0],self.limits[:,1])
        d = state.pipeline_state.replace(ctrl=target-self.home)
        # Occasional random horizontal velocity kick: the sim otherwise never
        # produces a disturbance, so the policy has no incentive to reject one.
        kick = jax.random.uniform(pk2,(2,),minval=-c.push_velocity,maxval=c.push_velocity)*jax.random.bernoulli(pk1,c.push_probability)
        d = d.replace(qvel=d.qvel.at[:2].add(kick))
        # Floor geom_quat may be randomized (a small terrain-tilt DR proxy -- see
        # walk_randomize.py); measure "height above ground" as a projection onto its
        # actual normal, not raw world Z, so ring/swing-clearance rewards stay valid
        # on a tilted floor. Constant for the whole rollout, so compute once here.
        floor_normal = geom.quat_axis_z(self.sys.geom_quat[self.floor],jp)
        impact_contact=jp.where(fresh,self.contact_mask(d)[0],info['impact_contact'])
        impact_velocity=jp.where(fresh,jp.zeros(4),info['impact_normal_velocity'])
        swing_peak=jp.where(fresh,jp.zeros(4),info['swing_peak'])
        swing_elapsed=jp.where(fresh|~moving,-jp.ones(4),info['swing_elapsed'])
        def substep(carry,_):
            d,previous_contact,previous_velocity,peak,elapsed=carry
            d = mjx.step(self.sys,d)
            substep_contact,_ = self.contact_mask(d)
            substep_points = geom.capsule_bottom(d.geom_xpos[self.foot_ids],d.geom_xmat[self.foot_ids],self.sizes,jp)
            foot_velocity=geom.point_velocities(substep_points[:,None,:],d.subtree_com[self.root_ids],d.cvel[self.body_ids],jp)[:,0,:]
            velocity=foot_velocity@floor_normal
            impact=geom.touchdown_cost(previous_contact,substep_contact,previous_velocity,c.touchdown_speed_allowance,jp)
            peak=jp.where(substep_contact,peak,jp.maximum(peak,substep_points@floor_normal))
            shortfall=geom.clearance_shortfall_cost(peak,previous_contact,substep_contact,c.minimum_swing_clearance,jp)
            peak=jp.where(substep_contact,0.,peak)
            elapsed,shape,reference,planned=geom.planned_swing_cost(elapsed,previous_contact,substep_contact,substep_points@floor_normal,self.dt/self.frames,c.planned_swing_duration,c.planned_swing_height,jp,bonus=c.planned_swing_bonus,weights=jp.array(c.planned_swing_foot_weights))
            alignment=-(d.geom_xmat[self.foot_ids].reshape(-1,3,3)[:,:,2]*self.tip_sign[:,None])@floor_normal
            tip=geom.tip_support_cost(alignment,substep_contact,self.tip_allowance_cos,self.tip_scale_cos,jp)
            swing = jp.sum(jp.clip(substep_points@floor_normal,0.,c.swing_clearance_target)*(~substep_contact))
            trace=dict(z=substep_points@floor_normal,contact=substep_contact,normal_velocity=velocity,tip_alignment=alignment,
                       tangential_speed=jp.linalg.norm(foot_velocity-velocity[:,None]*floor_normal,axis=-1),foot_position=substep_points,
                       joint_velocity=d.qvel[6:],angular_velocity=d.xmat[self.torso].reshape(3,3).T@d.cvel[self.torso,:3],torso_position=d.xpos[self.torso])
            trace.update(planned_height=reference,planned_swing=planned)
            return (d,substep_contact,velocity,peak,elapsed),(self.ring_signals(d,floor_normal),swing,impact,tip,shortfall,shape,trace)
        (d,impact_contact,impact_velocity,swing_peak,swing_elapsed), (ring_history,swing_history,impact_history,tip_history,shortfall_history,shape_history,trace) = jax.lax.scan(substep,(d,impact_contact,impact_velocity,swing_peak,swing_elapsed),None,length=self.frames)
        # step() integrates qpos after forward; refresh geometry to the resulting pose.
        d = mjx.forward(self.sys,d)
        ring = jax.tree.map(lambda x: jp.mean(x,axis=0),ring_history)
        # Sampled at the 250Hz substep rate (like the ring terms), not the 50Hz
        # control rate: a swing apex between control steps was otherwise undercounted.
        swing_clearance_raw = jp.mean(swing_history)
        contact, unwanted = self.contact_mask(d)
        points = geom.capsule_bottom(d.geom_xpos[self.foot_ids],d.geom_xmat[self.foot_ids],self.sizes,jp)
        foot_velocity = geom.point_velocities(points[:,None,:],d.subtree_com[self.root_ids],d.cvel[self.body_ids],jp)[:,0,:]
        rot = d.xmat[self.torso].reshape(3,3)
        torso_v = geom.point_velocities(d.xpos[self.torso][None,None,:],d.subtree_com[self.torso][None,:],d.cvel[self.torso][None,:],jp)[0,0]
        v = rot.T@torso_v
        omega = rot.T@d.cvel[self.torso,:3]
        tilt = jp.arccos(jp.clip(rot[2,2],-1.,1.))
        torso_height = self.torso_height(d)
        done = (tilt>c.terminal_tilt)|(torso_height<c.terminal_height)|(~jp.all(jp.isfinite(d.qpos)))
        air += self.dt
        touchdown = contact & ~old_contact
        low, high = self.limits[:,0]+c.joint_limit_margin,self.limits[:,1]-c.joint_limit_margin
        terms = dict(
            # Fixed .07 tolerance made standing attractive at .05--.10 m/s once
            # stepping costs were included. Preserve fast tolerance, tighten slow.
            tracking_linear=geom.linear_tracking_reward(v[:2],command[:2],c.tracking_variance_base,c.tracking_variance_speed_gain,jp,yaw_rate=command[2],yaw_gain=c.tracking_variance_yaw_gain),
            tracking_yaw=geom.yaw_tracking_reward(omega[2],command[2],jp),
            upright=rot[2,2], height=((torso_height-info['height_reference']-c.moving_height_offset*moving)/.04)**2,
            vertical_velocity=v[2]**2, roll_pitch_velocity=jp.sum(omega[:2]**2),
            foot_slip=jp.sum(jp.sum(foot_velocity[:,:2]**2,axis=1)*contact),
            air_time=geom.swing_duration_reward(air,touchdown,c.air_time_floor,jp)*moving,
            # Sum discrete touchdown events; averaging over substeps would dilute
            # their weight when simulation frequency changes. Outer dt remains.
            touchdown=jp.sum(impact_history),
            tip_support=jp.mean(tip_history),
            clearance_shortfall=jp.sum(shortfall_history)*moving,
            # Real swing-height reward. Weight derived to beat the ring's marginal
            # per-mm cost along the (otherwise favored) pure-knee lift direction --
            # see .notes; a weaker weight is arithmetically incapable of moving the
            # gait, which is why the first version of this term (air_time, and then
            # this term at weight .5) both realized as under 0.1% of total reward.
            swing_clearance=swing_clearance_raw*moving,
            stand_pose=jp.mean((d.qpos[7:]-self.home)**2)*(~moving),
            # stand_pose alone can't fix a lifted stance foot: it averages the
            # penalty over all 12 joints, so a single foot held 3mm up costs ~1/12th
            # as much as an equivalent whole-body deviation. Penalize missing foot
            # contacts directly, gated on genuinely being at rest (not just mid
            # deceleration right after a command drops to zero).
            stance_feet=(4.-jp.sum(contact.astype(float)))*(~moving)*(jp.linalg.norm(v[:2])<.05),
            ring_side=ring['ring_side'],ring_bottom=ring['ring_bottom'],ring_rub=ring['ring_rub'],
            ring_side_contact=ring['ring_side_fraction'],
            unwanted_contact=unwanted, action_rate=jp.mean(((action-prev)*self.smoothness_scale)**2),
            action_acceleration=jp.mean(((action-prev-previous_delta)*self.smoothness_scale)**2),
            planned_swing=jp.mean(shape_history)*moving,
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
                    last_action_delta=action-prev,
                    impact_contact=impact_contact,impact_normal_velocity=impact_velocity,
                    swing_peak=swing_peak,
                    swing_elapsed=jp.where(moving,swing_elapsed,-jp.ones(4)),
                    last_contact=contact,step=age+1,reset_next=done|((age+1)>=c.episode_length))
        metrics = {**state.metrics,**scaled,'ring_side_fraction':ring['ring_side_fraction'],'ring_penetration_m':ring['ring_penetration_m'],
                   'foot_contacts':jp.sum(contact).astype(float),'tilt_deg':tilt*180/jp.pi,'torso_z':d.qpos[2],
                   'torso_height':torso_height,'height_reference':info['height_reference']+c.moving_height_offset*moving,
                   'action_delta_rms':jp.sqrt(terms['action_rate']),
                   'tip_tilt_deg':jp.sum(jp.degrees(jp.arccos(jp.clip(trace['tip_alignment'],-1.,1.)))*trace['contact'])/jp.maximum(jp.sum(trace['contact']),1),
                   'tip_lean_fraction':jp.sum((trace['tip_alignment']<self.tip_allowance_cos)*trace['contact'])/jp.maximum(jp.sum(trace['contact']),1),
                   'velocity_error':jp.linalg.norm(v[:2]-command[:2])}
        metrics['yaw_error']=jp.abs(omega[2]-command[2])
        return state.replace(pipeline_state=d,obs=obs,reward=reward,done=done.astype(float),metrics=metrics,info=info),trace
