"""Collision-aware wheel alignment; eight residual actions, four hub PD servos."""
import jax
from jax import numpy as jp
import mujoco
import numpy as np
from brax import math
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from . import configs as c, contract as ct, geometry, rewards, contact_metrics, schedule

class AlignEnv(PipelineEnv):
    def __init__(self, noise=True, training=False, stage="sequence", automatic=None):
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
        if stage not in ('foundation','single','sequence'):raise ValueError('Unknown training stage')
        self.stage=stage;self.automatic=training if automatic is None else automatic
        self.floor=int(np.flatnonzero(m.geom_type==mujoco.mjtGeom.mjGEOM_PLANE)[0])
        assert m.opt.cone==mujoco.mjtCone.mjCONE_PYRAMIDAL and np.all(m.geom_condim==3)
        self.weight=float(m.body_mass.sum()*9.81)
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
        if self.stage=='foundation':home=ct.wrap(angles+jax.random.uniform(h,(4,),minval=-.25,maxval=.25)-jp.pi,jp)
        motion=ct.prepare(ct.reset(ps.q[7:],home,jp),xp=jp)
        info=dict(rng=rng,motion=motion,step=jp.asarray(0),origin=ps.q[:2],
            order=jax.random.permutation(o,jp.arange(1,5)),
            early_interrupt=jax.random.bernoulli(a,.2),
            action_buffer=jp.zeros((3,8)),sequence=schedule.reset(jp),contact_peak=jp.zeros(4))
        metrics={k:jp.asarray(0.) for k in ['tracking','clearance','wheel_gap','body_gap','impact_speed','torque',
            'drift','tilt','fall','completed','unsafe_rotation','command_speed','command_accel',
            'floor_estimate','gate_quality','angle_progress','verified_event','completed_event','active_angle_error',
            'floor_gate_blocked','wheel_gate_blocked','body_gate_blocked','stability_gate_blocked',
            'rotation_enabled','support_fraction','active_load','contact_peak_cost','timeouts','phase_idle','phase_lift','phase_rotate','phase_verify','phase_lower','phase_hold']}
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
            ps,motion,contact_peak=carry;old_velocity=motion['velocity'];old_clear=self.clearance(ps)
            motion=ct.integrate(motion,xp=jp)
            control=jp.zeros(12).at[ct.POS].set(motion['applied']).at[ct.WHEEL].set(wheel)
            ps=self.pipeline_step(ps,control)
            clear=self.clearance(ps);_,gravity=self.imu(ps)
            _,gap,bodygap=geometry.margins(ps.q[7:],gravity,k,jp)
            # Support wheels can also slam down while the body shifts. Penalize
            # the worst approaching wheel, not just the requested lifted wheel.
            approach=jp.maximum(-(clear-old_clear)/c.PHYSICS_DT,0)
            impact=jp.max(jp.where(clear<.008,approach,0))
            contact_peak,impact_cost=contact_metrics.approach_cost(contact_peak,clear,approach,jp)
            unsafe=motion['was_rotating']&((clear[k]<.005)|(gap<.005)|(bodygap<0))
            return (ps,motion,contact_peak),(gap,bodygap,impact,unsafe,jp.max(jp.abs(motion['velocity'])),
                jp.max(jp.abs(motion['velocity']-old_velocity))/c.PHYSICS_DT,impact_cost)
        (new_ps,motion,contact_peak),audit=jax.lax.scan(substep,(ps,motion,info['contact_peak']),None,length=10)
        gap,bodygap=jp.min(audit[0]),jp.min(audit[1])
        impact,unsafe=jp.max(audit[2]),jp.any(audit[3])
        clear=self.clearance(new_ps)
        angular,g=self.imu(new_ps)
        tilt=jp.arccos(jp.clip(-g[2],-1,1));drift=jp.linalg.norm(new_ps.q[:2]-info['origin'])
        tracking=jp.exp(-jp.sum((new_ps.q[7+ct.POS]-motion['motion_reference'])**2)/.25)
        gap_cost=jp.square(jp.maximum(.015-gap,0)/.015)+jp.square(jp.maximum(.010-bodygap,0)/.010)
        torque=jp.sum(jp.square(jp.maximum(jp.abs(new_ps.qfrc_actuator[6+ct.POS])-1.8,0)))
        support=(jp.arange(4)!=k)|~ct.up(motion)
        loads=contact_metrics.wheel_loads(new_ps.contact,new_ps._impl.efc_force,self.wheels,self.floor,jp)
        contacts=jp.sum(jp.clip(loads/(.05*self.weight),0,1)*support)/jp.maximum(jp.sum(support),1)
        active_load=loads[k]/self.weight
        upright=jp.exp(-(1+g[2])/.02)
        # Tracking the nominal apex penalized the residual needed to clear the
        # other wheel. Track the actual bounded command instead.
        active_target=motion['applied'][2*k+1]
        motion_tracking=jp.exp(-jp.square(new_ps.q[8+3*k]-active_target)/.015)
        reward=self.dt*(motion_tracking+.5*tracking+3*contacts+upright+
            jp.exp(-jp.square(new_ps.q[2]-self.height)/.0004)+
            2*jp.exp(-jp.square(jp.maximum(drift-.02,0))/.0025)-
            12*gap_cost-6*jp.square(impact/.10)-.1*jp.sum((action-motion['last_action'])**2)-.5*torque)
        fall=(tilt>.5)|(new_ps.q[2]<.075)
        done=fall|(drift>.15)|(gap<0)|(bodygap<-.005)|~jp.all(jp.isfinite(new_ps.q))
        before_finish=motion
        motion=ct.finish(motion,new_ps.q[7:],new_ps.qd[6:],jp)
        floor,_,_=geometry.margins(new_ps.q[7:],g,k,jp)
        floor=jp.minimum(floor,clear[k]);angular_speed=jp.linalg.norm(angular)
        task=rewards.task_terms(before_finish,motion,ps.q[7:],new_ps.q[7:],
            floor,gap,bodygap,tilt,angular_speed,unsafe,self.dt,jp)
        reward+=task['reward']-100*done-jp.sum(audit[6])
        reward-=self.dt*5*ct.up(before_finish)*(before_finish['progress']>=1)*jp.clip(active_load/.1,0,1)
        motion['last_action']=action
        step=info['step']+1
        command=motion['command']
        if self.automatic:
            old_timeouts=jp.sum(info['sequence']['timeouts'])
            info['sequence'],command=schedule.advance(info['sequence'],motion,info['order'],
                info['early_interrupt'] if self.stage=='sequence' else False,
                4 if self.stage=='sequence' else 1,jp)
            reward-=80*(jp.sum(info['sequence']['timeouts'])-old_timeouts)
            if self.training:done=done|(info['sequence']['index']>=(4 if self.stage=='sequence' else 1))
        motion=ct.select(motion,command,new_ps.q[7:],jp)
        motion=ct.prepare(motion,xp=jp)
        info.update(motion=motion,step=step,action_buffer=buffer,contact_peak=contact_peak)
        metrics=dict(state.metrics)
        metrics.update(tracking=tracking,clearance=clear[k],wheel_gap=gap,body_gap=bodygap,
            impact_speed=impact,torque=torque,drift=drift,tilt=tilt,fall=fall.astype(float),
            completed=jp.sum(motion['completed']).astype(float),unsafe_rotation=unsafe.astype(float),
            command_speed=jp.max(audit[4]),command_accel=jp.max(audit[5]))
        metrics.update(floor_estimate=geometry.margins(new_ps.q[7:],g,k,jp)[0],support_fraction=contacts,active_load=active_load,contact_peak_cost=jp.sum(audit[6]),timeouts=jp.sum(info['sequence']['timeouts']).astype(float))
        metrics.update({key:value for key,value in task.items() if key!='reward'})
        apex=ct.up(before_finish)&(before_finish['progress']>=1)
        metrics.update(floor_gate_blocked=(apex&(floor<=.010)).astype(float),
            wheel_gate_blocked=(apex&(gap<=.010)).astype(float),
            body_gate_blocked=(apex&(bodygap<=.005)).astype(float),
            stability_gate_blocked=(apex&((tilt>=.12)|(angular_speed>=.3))).astype(float),
            rotation_enabled=before_finish['was_rotating'].astype(float))
        for phase,name in enumerate(('idle','lift','rotate','verify','lower','hold')):
            metrics['phase_'+name]=(before_finish['phase']==phase).astype(float)
        return state.replace(pipeline_state=new_ps,info=info,obs=self.observe(new_ps,info),reward=reward,done=done.astype(float),metrics=metrics)
