"""Collision-aware wheel alignment; eight residual actions, four hub PD servos."""
import jax
from jax import numpy as jp
import mujoco
import numpy as np
from brax import math
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from . import configs as c, contract as ct, geometry

class AlignEnv(PipelineEnv):
    def __init__(self, noise=True, training=False):
        sys=mjcf.load(str(c.MODEL_PATH));m=sys.mj_model
        assert m.nu==12 and m.nq==19
        for i,name in enumerate(c.JOINT_NAMES):
            assert m.joint(name).id==m.actuator_trnid[i,0]
        self.wheels=np.array([m.geom(n).id for n in c.WHEEL_COLLISION_GEOM_NAMES])
        assert np.all(m.geom_contype[self.wheels]==2) and np.all(m.geom_conaffinity[self.wheels]==1)
        assert not np.any(m.jnt_limited[1+ct.WHEEL])
        m.actuator_gainprm[ct.POS,0]=5
        m.actuator_biasprm[ct.POS,1]=-5
        m.actuator_biasprm[ct.POS,2]=-.25
        sys=sys.replace(actuator_gainprm=jp.array(m.actuator_gainprm),actuator_biasprm=jp.array(m.actuator_biasprm))
        super().__init__(sys,backend='mjx',n_frames=1)
        # Short nominal stand settle, no learning. Produces the reset reference.
        d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,0)
        d.ctrl[ct.POS]=c.DEFAULT_POSE[ct.POS];d.ctrl[ct.WHEEL]=0
        for _ in range(1560):mujoco.mj_step(m,d)
        self.init_q=jp.asarray(d.qpos.copy());self.height=float(d.qpos[2])
        self.noise,self.training=noise,training
        self.default=jp.asarray(c.DEFAULT_POSE)

    @property
    def dt(self):return c.CONTROL_DT
    @property
    def action_size(self):return 8
    @property
    def observation_size(self):return c.OBSERVATION_SIZE

    def imu(self,ps):
        inv=math.quat_inv(ps.q[3:7])
        return math.rotate(ps.xd.ang[0],inv),math.rotate(jp.array([0.,0.,-1.]),inv)

    def observe(self,ps,info):
        ang,g=self.imu(ps)
        obs=ct.observation(info['motion'],ps.q[7:],ps.qd[6:],ang,g,jp)
        if self.noise:
            scale=jp.array([.03]*3+[.01]*3+[0.]*5+[.005]*12+[.002]*12+[0.]*47)
            obs+=scale*jax.random.uniform(info['rng'],(82,),minval=-1.,maxval=1.)
        return obs

    def clearance(self,ps):
        z=ps.geom_xmat[self.wheels,2,2];size=self.sys.geom_size[self.wheels]
        return ps.geom_xpos[self.wheels,2]-size[:,0]*jp.sqrt(jp.maximum(1-z*z,0))-size[:,1]*jp.abs(z)

    def reset(self,rng):
        rng,h,a,o=jax.random.split(rng,4)
        home=jax.random.uniform(h,(4,),minval=-jp.pi,maxval=jp.pi)
        angles=jax.random.uniform(a,(4,),minval=-jp.pi,maxval=jp.pi)
        ps=self.pipeline_init(self.init_q.at[7+ct.WHEEL].set(angles),jp.zeros(18))
        motion=ct.prepare(ct.reset(ps.q[7:],home,jp),xp=jp)
        info=dict(rng=rng,motion=motion,step=jp.asarray(0),origin=ps.q[:2],
            order=jax.random.permutation(o,jp.arange(1,5)),
            early_interrupt=jax.random.bernoulli(a,.2),
            action_buffer=jp.zeros((3,8)))
        metrics={k:jp.asarray(0.) for k in ['tracking','clearance','wheel_gap','body_gap','impact_speed','torque',
            'drift','tilt','fall','completed','unsafe_rotation','command_speed','command_accel']}
        return State(ps,self.observe(ps,info),jp.asarray(0.),jp.asarray(0.),metrics,info)

    def select_command(self,state,command):
        info=dict(state.info);old=info['motion'];motion=ct.select(old,command,state.pipeline_state.q[7:],jp)
        prepared=ct.prepare(motion,xp=jp)
        info['motion']=jax.tree.map(lambda a,b:jp.where(old['phase']!=motion['phase'],b,a),motion,prepared)
        return state.replace(info=info,obs=self.observe(state.pipeline_state,info))

    def step(self,state,action):
        ps=state.pipeline_state;info=dict(state.info);motion=info['motion']
        rng,lag=jax.random.split(info['rng']);info['rng']=rng
        ang,g=self.imu(ps);k=ct.leg(motion,jp)
        # Training delay represents real inference/transport uncertainty, not extra
        # random delay to be added to hardware. The last raw action remains raw.
        buffer=jp.roll(info['action_buffer'],1,axis=0).at[0].set(action)
        delay=jax.random.choice(lag,3,p=jp.array([.8,.15,.05])) if self.noise else 0
        motion,wheel=ct.begin(motion,ps.q[7:],ps.qd[6:],ang,g,buffer[delay],xp=jp)
        def substep(carry,_):
            ps,motion=carry;old_velocity=motion['velocity'];old_clear=self.clearance(ps)
            motion=ct.integrate(motion,xp=jp)
            control=jp.zeros(12).at[ct.POS].set(motion['applied']).at[ct.WHEEL].set(wheel)
            ps=self.pipeline_step(ps,control)
            clear=self.clearance(ps);_,gravity=self.imu(ps)
            _,gap,bodygap=geometry.margins(ps.q[7:],gravity,k,jp)
            impact=jp.where(clear[k]<.008,jp.maximum(-(clear[k]-old_clear[k])/c.PHYSICS_DT,0),0)
            unsafe=motion['was_rotating']&((clear[k]<.005)|(gap<.005)|(bodygap<0))
            return (ps,motion),(gap,bodygap,impact,unsafe,jp.max(jp.abs(motion['velocity'])),
                jp.max(jp.abs(motion['velocity']-old_velocity))/c.PHYSICS_DT)
        (new_ps,motion),audit=jax.lax.scan(substep,(ps,motion),None,length=10)
        gap,bodygap=jp.min(audit[0]),jp.min(audit[1])
        impact,unsafe=jp.max(audit[2]),jp.any(audit[3])
        clear=self.clearance(new_ps)
        _,g=self.imu(new_ps)
        tilt=jp.arccos(jp.clip(-g[2],-1,1));drift=jp.linalg.norm(new_ps.q[:2]-info['origin'])
        tracking=jp.exp(-jp.sum((new_ps.q[7+ct.POS]-motion['motion_reference'])**2)/.25)
        gap_cost=jp.square(jp.maximum(.015-gap,0)/.015)+jp.square(jp.maximum(.010-bodygap,0)/.010)
        torque=jp.sum(jp.square(jp.maximum(jp.abs(new_ps.qfrc_actuator[6+ct.POS])-1.8,0)))
        support=(jp.arange(4)!=k)|~ct.up(motion)
        contacts=jp.sum((clear<.006)&support)/jp.maximum(jp.sum(support),1)
        upright=jp.exp(-(1+g[2])/.02)
        active_target=motion['motion_reference'][2*k+1]
        motion_tracking=jp.exp(-jp.square(new_ps.q[8+3*k]-active_target)/.015)
        reward=self.dt*(4*motion_tracking+tracking+2*contacts+2*upright+
            jp.exp(-jp.square(new_ps.q[2]-self.height)/.0004)+
            2*jp.exp(-jp.square(jp.maximum(drift-.02,0))/.0025)-
            6*gap_cost-3*jp.square(impact/.10)-.1*jp.sum((action-motion['last_action'])**2)-.5*torque)
        fall=(tilt>.5)|(new_ps.q[2]<.075)
        done=fall|(drift>.15)|(gap<0)|(bodygap<-.005)|~jp.all(jp.isfinite(new_ps.q))
        motion['last_action']=action
        motion=ct.finish(motion,new_ps.q[7:],new_ps.qd[6:],jp)
        step=info['step']+1
        command=motion['command']
        if self.training:
            # Full four-leg sequences, all leg orders, and interrupted operations.
            slot=jp.minimum(step//1664,3);local=step%1664
            command=jp.where(local<104,0,info['order'][slot])
            command=jp.where(info['early_interrupt']&(local>=200)&(local<350),0,command)
        motion=ct.select(motion,command,new_ps.q[7:],jp)
        motion=ct.prepare(motion,xp=jp)
        info.update(motion=motion,step=step,action_buffer=buffer)
        metrics=dict(state.metrics)
        metrics.update(tracking=tracking,clearance=clear[k],wheel_gap=gap,body_gap=bodygap,
            impact_speed=impact,torque=torque,drift=drift,tilt=tilt,fall=fall.astype(float),
            completed=jp.sum(motion['completed']).astype(float),unsafe_rotation=unsafe.astype(float),
            command_speed=jp.max(audit[4]),command_accel=jp.max(audit[5]))
        return state.replace(pipeline_state=new_ps,info=info,obs=self.observe(new_ps,info),reward=reward,done=done.astype(float),metrics=metrics)
