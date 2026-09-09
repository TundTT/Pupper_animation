"""Eight learned position targets; four encoder-based wheel PD controllers.

No alignment reward, teachers, inverse-kinematics action adapters or restored
policies. Commands are externally supplied through select_command(). All signals
used by the controller/actor are encoder, IMU, or persistent software state.
Simulation geometry is used only for training rewards and evaluation auditing.
"""
import jax
import mujoco
import numpy as np
from jax import numpy as jp
from brax import math
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from workspace import configs

POS = np.array(configs.POSITION_ACTUATOR_ROWS)
WHEEL = np.array(configs.WHEEL_ACTUATOR_ROWS)
COMMAND_LEG = jp.array([-1, 1, 0, 2, 3])
IDLE, LIFT, ROTATE, VERIFY, LOWER, HOLD = range(6)


def wrap(x):
    return jp.arctan2(jp.sin(x), jp.cos(x))


def wheel_pd(target, angle, velocity):
    return jp.clip(2.0 * wrap(target - angle) - 0.35 * velocity, -2., 2.)


class HybridAlignEnv(PipelineEnv):
    def __init__(self, slew_rate=0.25, noise=True):
        sys = mjcf.load(str(configs.resolve_model_path()))
        m = sys.mj_model
        assert m.nu == 12 and m.nq == 19
        for i, name in enumerate(configs.JOINT_NAMES):
            assert m.joint(name).id == m.actuator_trnid[i, 0]
        assert not np.any(m.jnt_limited[1 + WHEEL])
        assert np.all(m.jnt_limited[1 + POS])
        assert np.all(m.actuator_biasprm[WHEEL, 1] == 0)
        m.actuator_gainprm[POS, 0] = 5.
        m.actuator_biasprm[POS, 1] = -5.
        m.actuator_biasprm[POS, 2] = -.25
        m.opt.timestep = .004
        sys = sys.replace(actuator_gainprm=jp.array(m.actuator_gainprm),
                          actuator_biasprm=jp.array(m.actuator_biasprm))
        sys = sys.tree_replace({'opt.timestep': .004})
        super().__init__(sys, backend='mjx', n_frames=5)
        self.slew_rate, self.noise = slew_rate, noise
        self.default = jp.array(configs.DEFAULT_POSE[POS])
        self.scale = jp.array([.5, 1.6] * 4)
        self.lo = jp.array(m.jnt_range[1 + POS, 0])
        self.hi = jp.array(m.jnt_range[1 + POS, 1])
        self.geom_ids = np.array([m.geom(n).id for n in configs.WHEEL_COLLISION_GEOM_NAMES])
        d = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, d, 0)
        for _ in range(750):
            d.ctrl[POS] = np.array(self.default)
            d.ctrl[WHEEL] = 2 * np.arctan2(np.sin(-d.qpos[7+WHEEL]),np.cos(-d.qpos[7+WHEEL])) - .35*d.qvel[6+WHEEL]
            mujoco.mj_step(m, d)
        q = d.qpos.copy(); q[:2] = 0.
        self.init_q = jp.array(q)
        self.stand_z = float(q[2])

    @property
    def action_size(self):
        return 8

    def clearance(self, ps):
        """Actual tilted-cylinder lower surface; reward/audit only."""
        zaxis = ps.geom_xmat[self.geom_ids, 2, 2]
        sizes = self.sys.geom_size[self.geom_ids]
        extent = sizes[:, 0]*jp.sqrt(jp.maximum(1-zaxis*zaxis, 0)) + sizes[:, 1]*jp.abs(zaxis)
        return ps.geom_xpos[self.geom_ids, 2] - extent

    def imu(self, ps):
        inv = math.quat_inv(ps.q[3:7])
        return math.rotate(ps.xd.ang[0], inv), math.rotate(jp.array([0.,0.,-1.]), inv)

    def effective_command(self, info):
        return jp.where((info['phase'] >= LIFT) & (info['phase'] <= VERIFY), info['active_command'], 0)

    def observe(self, ps, info):
        ang, gravity = self.imu(ps)
        q = (ps.q[7:] - jp.array(configs.DEFAULT_POSE)).at[WHEEL].set(wrap(ps.q[7+WHEEL]))
        err = wrap(info['target'] - ps.q[7+WHEEL])
        obs = jp.concatenate([ang, gravity, jax.nn.one_hot(self.effective_command(info),5),
                              q, ps.qd[6:] * .1, info['last_action'], jp.sin(err), jp.cos(err)])
        if self.noise:
            noise_scale = jp.array([.03]*3+[.01]*3+[0.]*5+[.005]*12+[.002]*12+[0.]*16)
            obs += jax.random.uniform(info['rng'], obs.shape, minval=-1., maxval=1.)*noise_scale
        return obs

    def reset(self, rng):
        rng, h, p = jax.random.split(rng, 3)
        home = jax.random.uniform(h,(4,),minval=-jp.pi,maxval=jp.pi)
        angles = home + jax.random.uniform(p,(4,),minval=-jp.pi,maxval=jp.pi)
        ps = self.pipeline_init(self.init_q.at[7+WHEEL].set(angles),jp.zeros(18))
        info = dict(rng=rng, command=jp.int32(0), active_command=jp.int32(0), phase=jp.int32(IDLE),
                    phase_steps=jp.int32(0), gate_steps=jp.int32(0), settled_steps=jp.int32(0),
                    home=home, target=wrap(home+jp.pi), hold=wrap(angles), reference=wrap(angles),
                    completed=jp.zeros(4,dtype=bool), verified=jp.array(False), was_rotating=jp.array(False),
                    last_action=jp.zeros(8), applied=jp.zeros(8), buffer=jp.zeros((3,8)),
                    init_xy=ps.q[:2], step=jp.int32(0), rotation_unsafe=jp.int32(0))
        metrics={k:jp.float32(0) for k in ['lift','progress','stance','contact','upright','height','drift_reward','smooth','torque',
                                         'drift','tilt','clearance','completed','held_error','fall','unsafe_rotation']}
        return State(ps,self.observe(ps,info),jp.float32(0),jp.float32(0),metrics,info)

    def select_command(self, state, command):
        """External one-hot index; safely finish lowering before honoring a new leg."""
        info = dict(state.info)
        changed = command != info['command']
        interrupted = changed & (info['phase'] >= LIFT) & (info['phase'] <= VERIFY)
        leg = jp.maximum(COMMAND_LEG[info['active_command']], 0)
        angles = wrap(state.pipeline_state.q[7+WHEEL])
        info['hold'] = info['hold'].at[leg].set(jp.where(interrupted, angles[leg], info['hold'][leg]))
        info['phase'] = jp.where(interrupted, LOWER, info['phase'])
        info['phase_steps'] = jp.where(interrupted, 0, info['phase_steps'])
        info['verified'] = info['verified'] & ~interrupted
        info['command'] = jp.int32(command)
        ready = (info['phase'] == IDLE) | (info['phase'] == HOLD)
        requested = jp.maximum(COMMAND_LEG[command],0)
        start = ready & (command != 0) & ~info['completed'][requested]
        info['active_command'] = jp.where(start, command, info['active_command'])
        info['phase'] = jp.where(start,LIFT,info['phase'])
        info['phase_steps'] = jp.where(start,0,info['phase_steps'])
        info['reference'] = jp.where(start,info['hold'],info['reference'])
        info['verified'] = jp.where(start,False,info['verified'])
        return state.replace(info=info,obs=self.observe(state.pipeline_state,info))

    def step(self, state, action):
        state = self.select_command(state,state.info['command'])
        info = dict(state.info); ps = state.pipeline_state
        rng, lag = jax.random.split(info['rng']); info['rng'] = rng
        leg = jp.maximum(COMMAND_LEG[info['active_command']],0)
        up = (info['phase'] >= LIFT) & (info['phase'] <= VERIFY)
        ang, grav = self.imu(ps)
        signed_hip = ps.q[8+3*leg] * jp.where(leg%2==0,1.,-1.)
        # Conservative encoder/IMU guard; actual clearance is audited separately.
        gate = (signed_hip > 1.1) & (jp.abs(ps.q[7+3*leg]-self.default[2*leg]) < .25) & (-grav[2] > jp.cos(.12)) & (jp.linalg.norm(ang) < .3)
        info['gate_steps'] = jp.where(up & gate,info['gate_steps']+1,0)
        rotating = ((info['phase']==ROTATE)|(info['phase']==VERIFY)) & gate
        # On loss of the lift guard, latch once at the encoder angle. Continue
        # actively holding that snapshot; do not chase a stale moving reference.
        paused = info['was_rotating'] & ~rotating & up
        info['reference'] = info['reference'].at[leg].set(jp.where(paused,wrap(ps.q[7+WHEEL][leg]),info['reference'][leg]))
        info['was_rotating'] = rotating
        ref = info['reference'].at[leg].add(jp.where(rotating,jp.clip(wrap(info['target'][leg]-info['reference'][leg]),-self.slew_rate*self.dt,self.slew_rate*self.dt),0.))
        info['reference'] = wrap(ref)
        goal = info['hold'].at[leg].set(jp.where(up,info['reference'][leg],info['hold'][leg]))
        ctrl = jp.zeros(12).at[WHEEL].set(wheel_pd(goal,ps.q[7+WHEEL],ps.qd[6+WHEEL]))
        buf = jp.roll(info['buffer'],1,axis=0).at[0].set(action)
        delayed = buf[jax.random.choice(lag,3,p=jp.array([.8,.15,.05]))]
        hipmask = (jp.arange(8)==(2*leg+1)) & (up|(info['phase']==LOWER))
        applied = jp.where(hipmask,jp.clip(delayed,info['applied']-.05,info['applied']+.05),delayed)
        ctrl = ctrl.at[POS].set(jp.clip(self.default+self.scale*applied,self.lo,self.hi))
        ps = self.pipeline_step(ps,ctrl)
        clear = self.clearance(ps)
        _, grav = self.imu(ps)
        tilt = jp.arccos(jp.clip(-grav[2],-1.,1.))
        drift = jp.linalg.norm(ps.q[:2]-info['init_xy'])
        fall = (tilt > .5)|(ps.q[2] < .075)
        done = fall | (drift > .15) | ~jp.all(jp.isfinite(ps.q))
        pos = ps.q[7+POS]
        stance_mask = jp.repeat(jp.arange(4)!=leg,2) | ~up
        stance = jp.exp(-jp.sum(jp.where(stance_mask,jp.square(pos-self.default),0.))/.15)
        target_clearance = .03
        lift = jp.where(up,jp.clip(clear[leg]/target_clearance,0.,1.),jp.exp(-jp.sum(jp.square(jp.maximum(clear,0.)))/.0004))
        hip_progress = jp.where(up,jp.clip(signed_hip/1.2,0.,1.),0.)
        contact = jp.sum((clear<.006)*((jp.arange(4)!=leg)|~up))/jp.where(up,3.,4.)
        upright = jp.exp(-(1+grav[2])/.02)
        height = jp.exp(-jp.square(ps.q[2]-self.stand_z)/.0004)
        allowance = jp.where(up|(info['phase']==LOWER),.035,.020)
        drift_reward = jp.exp(-jp.square(jp.maximum(drift-allowance,0.))/.0025)
        smooth = jp.sum(jp.square(action-info['last_action']))
        torque = jp.sum(jp.clip((jp.abs(ps.qfrc_actuator[6+POS])-1.8)/1.2,0.,1.))
        reward = self.dt*(8*lift+3*hip_progress+3.5*stance+contact+2*upright+1.5*height+2.5*drift_reward-.1*smooth-2*torque)
        # Phase progress never contributes alignment/reaching reward to RL.
        info['phase_steps'] += 1
        err = jp.abs(wrap(info['target'][leg]-ps.q[7+WHEEL][leg]))
        settled = rotating & (err < .035) & (jp.abs(ps.qd[6+WHEEL][leg])<.08)
        info['settled_steps'] = jp.where(settled,info['settled_steps']+1,0)
        to_rotate = (info['phase']==LIFT)&(info['gate_steps']>=10)
        to_verify = (info['phase']==ROTATE)&(err<.035)
        to_lower = (info['phase']==VERIFY)&(info['settled_steps']>=25)
        lower_done = (info['phase']==LOWER)&(info['phase_steps']>=50)&(jp.abs(ps.q[8+3*leg])<.25)&(jp.abs(ps.qd[7+3*leg])<.2)
        info['hold'] = info['hold'].at[leg].set(jp.where(to_lower,wrap(ps.q[7+WHEEL][leg]),info['hold'][leg]))
        info['verified'] |= to_lower
        info['completed'] = info['completed'].at[leg].set(info['completed'][leg] | (lower_done & info['verified']))
        info['phase'] = jp.where(to_rotate,ROTATE,jp.where(to_verify,VERIFY,jp.where(to_lower,LOWER,jp.where(lower_done,HOLD,info['phase']))))
        info['phase_steps'] = jp.where(to_rotate|to_verify|to_lower|lower_done,0,info['phase_steps'])
        unsafe = rotating & (clear[leg]<.005)
        info['rotation_unsafe'] += unsafe.astype(jp.int32)
        info['last_action']=action; info['applied']=applied; info['buffer']=buf; info['step']+=1
        held = (jp.arange(4)!=leg)|~up
        held_error = jp.max(jp.where(held,jp.abs(wrap(info['hold']-ps.q[7+WHEEL])),0.))
        metrics=dict(state.metrics)
        metrics.update(lift=lift,progress=hip_progress,stance=stance,contact=contact,upright=upright,height=height,
                     drift_reward=drift_reward,smooth=smooth,torque=torque,drift=drift,tilt=tilt,clearance=clear[leg],
                     completed=jp.sum(info['completed']).astype(jp.float32),held_error=held_error,fall=fall.astype(jp.float32),unsafe_rotation=unsafe.astype(jp.float32))
        return state.replace(pipeline_state=ps,obs=self.observe(ps,info),reward=reward,done=done.astype(jp.float32),info=info,metrics=metrics)


class TrainingEnv(HybridAlignEnv):
    """Training-only external operator: random leg, 20 seconds, then stand."""
    def reset(self,rng):
        state=super().reset(rng)
        command=jax.random.randint(rng,(),1,5)
        return self.select_command(state,command)

    def step(self,state,action):
        # 20 s covers pi/.25 + settling + lift/lower; 4 s of explicit stand.
        command=jp.where(state.info['step']>=1000,0,state.info['command'])
        return super().step(self.select_command(state,command),action)
