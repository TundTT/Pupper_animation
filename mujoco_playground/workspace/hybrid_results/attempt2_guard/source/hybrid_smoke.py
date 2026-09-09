"""Executable pre-training invariants and physical wheel slew experiment."""
import json
from pathlib import Path
import jax
from jax import numpy as jp
import mujoco
import numpy as np
from brax.envs import training
from workspace.hybrid_env import HybridAlignEnv, TrainingEnv, POS, WHEEL, wrap, wheel_pd, LOWER, LIFT
from workspace.hybrid_wrappers import wrap_for_training
from workspace.randomize import domain_randomize_wheeled


def main():
    env=HybridAlignEnv(noise=False)
    state=jax.jit(env.reset)(jax.random.PRNGKey(13))
    assert state.obs.shape==(51,) and env.action_size==8
    # Widen only abduction: the other readiness constraints still reject.
    gravity=jp.array([0.,0.,-1.]);ang=jp.zeros(3)
    for leg in range(4):
        sign=1. if leg%2==0 else -1.
        angles=jp.array([1.,0.,0.,-1.,0.,0.,1.,0.,0.,-1.,0.,0.])
        angles=angles.at[3*leg].set(sign*1.30).at[3*leg+1].set(sign*1.2)
        assert env.lift_ready(angles,ang,gravity,leg)
        assert not env.lift_ready(angles.at[3*leg].set(sign*1.36),ang,gravity,leg)
        assert not env.lift_ready(angles.at[3*leg+1].set(sign*1.05),ang,gravity,leg)
        assert not env.lift_ready(angles,jp.array([.31,0.,0.]),gravity,leg)
        assert not env.lift_ready(angles,ang,jp.array([jp.sin(.13),0.,-jp.cos(.13)]),leg)
    np.testing.assert_allclose(jp.cos(state.info['target']-state.info['home']),-1,atol=1e-6)
    # Arbitrary accumulated revolutions leave the PD and target error unchanged.
    angle=jp.array([.2,-.4,2.,-2.]);target=jp.array([-.3,.8,-2.,2.]);vel=jp.array([.1,-.2,.3,-.4])
    np.testing.assert_allclose(wheel_pd(target,angle,vel),wheel_pd(target+2*jp.pi,angle-6*jp.pi,vel),atol=1e-5)
    np.testing.assert_allclose(wheel_pd(angle,angle,vel),-.35*vel,atol=1e-6)
    selected=env.select_command(state,1)
    assert selected.info['phase']==LIFT
    interrupted=env.select_command(selected,2)
    assert interrupted.info['phase']==LOWER
    np.testing.assert_array_equal(interrupted.info['target'],state.info['target'])
    np.testing.assert_array_equal(interrupted.info['home'],state.info['home'])
    # Actor cannot observe world translation or contacts.
    translated=state.pipeline_state.replace(q=state.pipeline_state.q.at[:2].add(jp.array([9.,-3.])))
    np.testing.assert_allclose(env.observe(translated,state.info),env.observe(state.pipeline_state,state.info))
    randomized,_=domain_randomize_wheeled(env.sys,jax.random.split(jax.random.PRNGKey(8),8))
    assert np.all(np.asarray(randomized.actuator_biasprm)[:,WHEEL,1]==0)
    np.testing.assert_allclose(np.asarray(randomized.actuator_biasprm)[:,WHEEL,2],-np.asarray(randomized.actuator_gainprm)[:,WHEEL,0])
    step=jax.jit(env.step)
    stepped=step(selected,jp.zeros(8))
    assert np.all(np.isfinite(stepped.obs)) and np.isfinite(stepped.reward)
    np.testing.assert_array_equal(stepped.info['hold'],selected.info['hold'])
    wrapped=wrap_for_training(TrainingEnv(noise=False),episode_length=3,action_repeat=1)
    ws=jax.jit(wrapped.reset)(jax.random.split(jax.random.PRNGKey(99),2))
    first=ws
    advance=jax.jit(wrapped.step)
    for i in range(8):
        ws=advance(ws,jp.zeros((2,8)))
        assert np.all(np.isfinite(ws.reward))
        if (i+1)%3==0:
            assert np.all(ws.done==1) and np.all(ws.info['step']==0)
            assert np.all(ws.info['truncation']==1) and np.all(ws.info['episode_done']==1)
            assert np.all(ws.info['command']>0)
            for key in ['command','phase','target','hold','completed','reference']:
                np.testing.assert_array_equal(ws.info[key],first.info[key])
            np.testing.assert_array_equal(ws.pipeline_state.q,first.pipeline_state.q)
    print('PASS: 8 actions, 51 hardware observations, persistent calibration, wrap invariance, interruption, active holds, mixed DR, JIT physics',flush=True)
    # Isolate wheel reaction with chassis/leg configuration reset every 4 ms.
    # This measures actuator reaction, not learned-policy stability.
    experiments=[]
    for speed in [.25,.5,1.,10.]:
        m=env.sys.mj_model;d=mujoco.MjData(m)
        q=np.asarray(env.init_q).copy();q[8]=1.2
        d.qpos[:]=q;d.qvel[:]=0.;ref=float(q[9]);goal=ref+np.pi
        peak_tau=0.;peak_vel=0.;completed=None
        for i in range(int(20/.004)):
            # Hold torso and eight position coordinates to isolate spin response.
            d.qpos[:7]=q[:7];d.qvel[:6]=0.;d.qpos[7+POS]=q[7+POS];d.qvel[6+POS]=0.
            ref+=np.clip(goal-ref,-speed*.004,speed*.004)
            d.ctrl[POS]=q[7+POS]
            error=np.arctan2(np.sin(ref-d.qpos[9]),np.cos(ref-d.qpos[9]))
            d.ctrl[WHEEL]=0.;d.ctrl[2]=np.clip(2*error-.35*d.qvel[8],-2,2)
            mujoco.mj_step(m,d)
            peak_tau=max(peak_tau,abs(float(d.qfrc_actuator[8])))
            peak_vel=max(peak_vel,abs(float(d.qvel[8])))
            e=abs(np.arctan2(np.sin(goal-d.qpos[9]),np.cos(goal-d.qpos[9])))
            if completed is None and e<.035 and abs(d.qvel[8])<.08:completed=(i+1)*.004
        experiments.append(dict(slew_rad_s=speed,peak_reaction_torque_nm=peak_tau,peak_wheel_speed_rad_s=peak_vel,settled_seconds=completed))
    result=dict(invariants='passed',slew_experiment=experiments,chosen_slew=.25,scope='isolated actuator reaction; balance evaluated after training')
    out=Path(__file__).parent/'hybrid_results'/'smoke.json';out.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
