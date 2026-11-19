"""Reproducible stage/geometry diagnosis of the frozen 12M pilot."""
import functools
import importlib.util
import json
from pathlib import Path

import jax
import mujoco
import numpy as np
from brax.envs.wrappers.training import DomainRandomizationVmapWrapper
from brax.io import model
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks
from jax import numpy as jp
from ml_collections import ConfigDict
from workspace import configs

RUN = Path('workspace/output/wheel-align_2026-09-07_18-02-52')
OUT = Path('workspace/output/align_diagnosis')
spec = importlib.util.spec_from_file_location('pilot_env', RUN / 'training_source/wheel_align_env.py')
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)

class ProbeEnv(pilot.PupperWheelAlignEnv):
    def reset(self, rng):
        s = super().reset(rng)
        s.info['probe'] = jp.zeros((4, 6))
        return s

    def step(self, state, action):
        s = super().step(state, action)
        s.info['probe'] = jp.stack((self._wheel_clearance(s.pipeline_state),
            self._wheel_loads(s.pipeline_state), jp.abs(self._errors(s.pipeline_state, s.info)),
            jp.abs(s.pipeline_state.qd[self._wheel_dof]),
            self._hip_sign * (s.pipeline_state.q[self._hip_qpos] - self._init_q[self._hip_qpos]),
            self._hip_sign * s.info['last_applied_action'][self._hip_rows] * self._action_scale[self._hip_rows]), axis=-1)
        return s


def main():
    cfg = ConfigDict(json.loads((RUN / 'config.json').read_text()))
    cfg.align.initial_completed_max = 0
    env = ProbeEnv(cfg)
    net = networks.make_ppo_networks(356, 8, policy_hidden_layer_sizes=(128,128),
        value_hidden_layer_sizes=(128,128), activation=jax.nn.elu,
        preprocess_observations_fn=running_statistics.normalize)
    inference = networks.make_inference_fn(net)(model.load_params(RUN/'mjx_params'), deterministic=True)
    wrapped = DomainRandomizationVmapWrapper(env, functools.partial(pilot.domain_randomize_wheel_align,
        rng=jax.random.split(jax.random.PRNGKey(901), 16)))
    @jax.jit
    def rollout():
        state = wrapped.reset(jax.random.split(jax.random.PRNGKey(902),16))
        def step(carry,_):
            state, rng, alive = carry
            rng,key=jax.random.split(rng)
            action,_=inference(state.obs,key)
            oldcmd,oldphase=state.info['command'],state.info['phase']
            nxt=wrapped.step(state,action)
            def choose(new,old):
                return jp.where(alive.reshape(alive.shape+(1,)*(new.ndim-alive.ndim)),new,old)
            nxt=jax.tree.map(choose,nxt,state)
            trace=(oldcmd,oldphase,nxt.info['probe'],alive,nxt.info['phase'],nxt.info['rotation_verified'])
            return (nxt,rng,alive & ~nxt.done.astype(bool)), trace
        return jax.lax.scan(step,(state,jax.random.PRNGKey(903),jp.ones(16,dtype=bool)),None,length=1800)[1]
    cmd,phase,probe,alive,after,verified=jax.device_get(rollout())
    np.savez_compressed(OUT/'pilot_trace.npz',command=cmd,phase=phase,probe=probe,alive=alive,after=after,verified=verified)
    report={}
    for leg,row in configs.FOOT_ROW_BY_LEG.items():
        selected=(cmd==configs.COMMAND_STATES.index(leg)) & alive
        active=selected & ((phase[:,:,row]==1)|(phase[:,:,row]==2))
        x=probe[:,:,row]
        report[leg]={
            'selected_samples':int(selected.sum()),
            'selected_phase_fractions':{str(p):float(np.mean(phase[:,:,row][selected]==p)) for p in range(5)},
            'episodes_reached_rotate':int(np.any((after[:,:,row]==2)&alive,axis=0).sum()),
            'episodes_verified_rotation':int(np.any(verified[:,:,row]&alive,axis=0).sum()),
            'episodes_completed':int(np.any((after[:,:,row]==4)&alive,axis=0).sum()),
            'clearance_mm_p50_p95_max':(np.percentile(x[:,:,0][active],[50,95,100])*1000).tolist(),
            'load_N_p50_p95':np.percentile(x[:,:,1][active],[50,95]).tolist(),
            'actual_hip_rad_p50_p95':np.percentile(x[:,:,4][active],[50,95]).tolist(),
            'requested_hip_rad_p50_p95':np.percentile(x[:,:,5][active],[50,95]).tolist(),
            'unloaded_fraction':float(np.mean((x[:,:,0][active]>=.003)&(x[:,:,1][active]<.5))),
        }
    (OUT/'pilot_stages.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)

    # Free-body native physics: home stance, then raise only one hip. Compare
    # real supported clearance against the previous fixed-torso FK calculation.
    m=env.sys.mj_model
    def clearance(d):
        ids=env._wheel_geom_id
        az=np.abs(d.geom_xmat[ids].reshape(4,3,3)[:,2,2])
        return d.geom_xpos[ids,2]-m.geom_size[ids,0]*np.sqrt(1-az**2)-m.geom_size[ids,1]*az
    def margin(d, leg):
        pts=d.geom_xpos[env._wheel_geom_id,:2][np.arange(4)!=leg]
        center=pts.mean(0)
        pts=pts[np.argsort(np.arctan2(pts[:,1]-center[1],pts[:,0]-center[0]))]
        edges=np.roll(pts,-1,axis=0)-pts
        v=d.subtree_com[env._torso_idx,:2]-pts
        return np.min((edges[:,0]*v[:,1]-edges[:,1]*v[:,0])/np.linalg.norm(edges,axis=1))
    mechanical=[]
    for leg,row in configs.FOOT_ROW_BY_LEG.items():
        for delta in (0.,.4,.6,.8,1.0):
            d=mujoco.MjData(m);d.qpos[:]=np.asarray(env._init_q);d.ctrl[:]=np.asarray(env._default_pose)
            for t in range(750):
                d.ctrl[env._hip_rows[row]]=float(env._default_pose[env._hip_rows[row]])+float(env._hip_sign[row])*delta*min(t/125,1.)
                mujoco.mj_step(m,d)
            mechanical.append({'leg':leg,'hip_ctrl_delta':delta,'actual_delta':float(env._hip_sign[row]*(d.qpos[env._hip_qpos[row]]-env._init_q[env._hip_qpos[row]])),
                'clearance_mm':float(clearance(d)[row]*1000),'support_margin_mm':float(margin(d,row)*1000),
                'body_xy_mm':(d.qpos[:2]*1000).tolist(),'body_z':float(d.qpos[2])})
    (OUT/'mechanical.json').write_text(json.dumps(mechanical,indent=2))
    print(json.dumps(mechanical,indent=2),flush=True)

if __name__=='__main__':main()
