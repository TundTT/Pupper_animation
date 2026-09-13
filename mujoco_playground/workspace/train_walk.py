"""PPO walking entry point. Use the venv Python directly, not uv run."""
import argparse
from datetime import datetime
from pathlib import Path
import functools
import hashlib
import json
import platform
import importlib.metadata
import jax
from jax import numpy as jp
import mediapy as media
import mujoco
import numpy as np
from brax.io import model
from brax.training.agents.ppo import networks, train as ppo
from workspace.walk_env import PupperWalkEnv
from workspace.walk_config import get_config
from workspace.walk_randomize import domain_randomize

# (vx, vy, yaw, label). Fixed showcase so eval videos are comparable frame-for-frame.
_SHOWCASE=[(0.,0.,0.),(.2,0.,0.),(.35,0.,0.),(.2,0.,.4),(.2,0.,-.4),(0.,0.,.5),(-.15,0.,0.),(0.,0.,0.)]
# Preserve the original first 16 seconds, then expose reverse-speed and lateral
# weaknesses that a single brief -0.15 m/s segment can hide.
_SHOWCASE += [(-.2,0.,0.),(-.35,0.,0.),(-.2,0.,.4),(0.,.1,0.),(0.,-.1,0.),(0.,0.,0.)]
# Include both spin directions; the original showcase only turned left in place.
_SHOWCASE += [(0.,0.,-.5),(0.,0.,0.)]

def render_showcase(env,inference_fn,rng,observation_history,steps_per_command=100,render_every=1,commands=None):
    """Roll the policy through a fixed command sequence and render it. Returns (frames, fps)."""
    jit_reset=jax.jit(env.reset);jit_step=jax.jit(env.step);jit_policy=jax.jit(inference_fn)
    state=jit_reset(rng)
    renderer=mujoco.Renderer(env.mj_model,height=480,width=640)
    data=mujoco.MjData(env.mj_model)
    # `tracking_cam` in the MJCF only aims at the torso from a fixed world position
    # (mode="targetbody"), so by the reverse segment the robot has walked away and
    # occupies a few percent of the frame. A free camera whose lookat we update to
    # the torso position every rendered frame keeps it centered and legible.
    cam=mujoco.MjvCamera()
    cam.type=mujoco.mjtCamera.mjCAMERA_FREE
    # Closer and more side-on than the first pass: at distance=1.0/elevation=-25 the
    # robot filled ~15% of the frame and near-side legs occluded the far side,
    # making single-digit-mm foot clearance impossible to judge from the video.
    cam.distance=0.55; cam.azimuth=90; cam.elevation=-8
    frames=[]
    try:
        for vx,vy,wz in (_SHOWCASE if commands is None else commands):
            cmd=jp.array([vx,vy,wz])
            state.info['command']=cmd;state.info['initial_command']=cmd;state.info['step']=jp.array(0)
            obs=state.obs.reshape(observation_history,36).at[:,6:9].set(cmd)
            state=state.replace(obs=obs.reshape(-1))
            done=False
            for t in range(steps_per_command):
                rng,pk=jax.random.split(rng)
                action,_=jit_policy(state.obs,pk)
                state=jit_step(state,action)
                if t%render_every==0:
                    data.qpos[:]=np.asarray(state.pipeline_state.qpos);mujoco.mj_forward(env.mj_model,data)
                    cam.lookat[:]=data.xpos[env.torso]
                    renderer.update_scene(data,camera=cam);frames.append(renderer.render().copy())
                if bool(state.done):done=True;break
            if done:break
    finally:
        renderer.close()
    fps=max(int(round(1/env.dt/render_every)),1)
    return frames,fps


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num_timesteps',type=int)
    parser.add_argument('--num_envs',type=int)
    parser.add_argument('--num_evals',type=int)
    parser.add_argument('--learning_rate',type=float)
    parser.add_argument('--air_time',type=float,help='Override swing-duration reward for controlled trials')
    parser.add_argument('--touchdown_weight',type=float,help='Override touchdown penalty (zero disables it)')
    parser.add_argument('--action_rate',type=float,help='Override command smoothness weight')
    parser.add_argument('--action_acceleration',type=float,help='Second action-difference penalty for abrupt target reversals')
    parser.add_argument('--swing_shape',type=float,help='Weight of lift-triggered, full-duration swing shape')
    parser.add_argument('--swing_duration',type=float,help='Planned swing duration in seconds')
    parser.add_argument('--swing_clearance',type=float,help='Weight of the original height-area reward')
    parser.add_argument('--ring_side_weight',type=float)
    parser.add_argument('--tracking_linear_weight',type=float)
    parser.add_argument('--tracking_yaw_weight',type=float)
    parser.add_argument('--action_scale',type=float,nargs=3,metavar=('HIP','ABD','KNEE'))
    parser.add_argument('--moving_height_offset',type=float)
    parser.add_argument('--swing_height',type=float)
    parser.add_argument('--swing_foot_weights',type=float,nargs=4,metavar=('FR','FL','BR','BL'),help='Per-foot trajectory error multipliers; bonus stays unchanged')
    parser.add_argument('--minimum_swing_clearance',type=float,help='Minimum swing peak scored at touchdown')
    parser.add_argument('--clearance_shortfall_weight',type=float,help='Penalty for replanting after a shallow swing')
    parser.add_argument('--swing_bonus',type=float)
    parser.add_argument('--height_weight',type=float)
    parser.add_argument('--ring_contact_weight',type=float)
    parser.add_argument('--floor_friction',type=float,help='Nominal/evaluation floor sliding friction')
    parser.add_argument('--friction_range',type=float,nargs=2,metavar=('MIN','MAX'),help='Training sliding-friction randomization range')
    parser.add_argument('--leg_length_common_range',type=float,nargs=2,metavar=('MIN','MAX'))
    parser.add_argument('--leg_length_per_leg_range',type=float,nargs=2,metavar=('MIN','MAX'))
    parser.add_argument('--straight_probability',type=float,help='Fraction of moving commands that are straight')
    parser.add_argument('--axis_probability',type=float,help='Fraction of moving commands that are pure lateral or yaw')
    parser.add_argument('--air_time_floor',type=float,help='Lower clip for signed swing-duration reward')
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--model_path')
    parser.add_argument('--output_dir',default=str(Path(__file__).with_name('output')))
    parser.add_argument('--smoke',action='store_true',help='Tiny CPU-compatible PPO integration check; not a trained policy')
    parser.add_argument('--init_params',help='Warm-start weights, not optimizer-state resume')
    parser.add_argument('--use_wandb',action='store_true')
    parser.add_argument('--wandb_project',default='pupper-leg')
    parser.add_argument('--wandb_entity',default='QuadMorph')
    videos=parser.add_mutually_exclusive_group()
    videos.add_argument('--no_eval_videos',action='store_true',help='Disable ALL videos, including final and best checkpoints')
    videos.add_argument('--final_videos_only',action='store_true',help='Skip intermediate videos; still save and upload final and best checkpoints')
    parser.add_argument('--no_randomization',action='store_true')
    args=parser.parse_args()
    video_mode='disabled' if args.smoke or args.no_eval_videos else ('final_and_best' if args.final_videos_only else 'all')
    print(f'Evaluation videos: {video_mode}',flush=True)
    if args.use_wandb and video_mode=='disabled':
        print('WARNING: this run will upload no W&B videos (including final/best). Use --final_videos_only to retain end-of-run videos.',flush=True)
    c=get_config(); c.ppo.seed=args.seed
    # Mature warm starts proved susceptible to late KL spikes at 3e-4.
    if args.init_params:c.ppo.learning_rate=5e-5
    if args.smoke:
        c.ppo.update(num_timesteps=64,num_envs=2,num_evals=2,num_eval_envs=2,
                     unroll_length=4,batch_size=2,num_minibatches=1,num_updates_per_batch=1)
        c.episode_length=16; c.hidden_layer_sizes=(32,32)
        c.sensor_noise=0.; c.latency_probability=0.
    if args.num_timesteps is not None:c.ppo.num_timesteps=args.num_timesteps
    if args.num_envs is not None:c.ppo.num_envs=args.num_envs
    if args.num_evals is not None:c.ppo.num_evals=args.num_evals
    if args.learning_rate is not None:c.ppo.learning_rate=args.learning_rate
    if args.air_time is not None:c.reward_scales.air_time=args.air_time
    if args.touchdown_weight is not None:c.reward_scales.touchdown=args.touchdown_weight
    if args.action_rate is not None:c.reward_scales.action_rate=args.action_rate
    if args.action_acceleration is not None:c.reward_scales.action_acceleration=args.action_acceleration
    if args.swing_shape is not None:c.reward_scales.planned_swing=args.swing_shape
    if args.swing_duration is not None:c.planned_swing_duration=args.swing_duration
    if args.swing_clearance is not None:c.reward_scales.swing_clearance=args.swing_clearance
    if args.ring_side_weight is not None:c.reward_scales.ring_side=args.ring_side_weight
    if args.tracking_linear_weight is not None:c.reward_scales.tracking_linear=args.tracking_linear_weight
    if args.tracking_yaw_weight is not None:c.reward_scales.tracking_yaw=args.tracking_yaw_weight
    if args.action_scale is not None:c.action_scale=tuple(args.action_scale)*4
    if args.moving_height_offset is not None:c.moving_height_offset=args.moving_height_offset
    if args.swing_height is not None:c.planned_swing_height=args.swing_height
    if args.swing_foot_weights is not None:c.planned_swing_foot_weights=tuple(args.swing_foot_weights)
    if args.minimum_swing_clearance is not None:c.minimum_swing_clearance=args.minimum_swing_clearance
    if args.clearance_shortfall_weight is not None:c.reward_scales.clearance_shortfall=args.clearance_shortfall_weight
    if args.swing_bonus is not None:c.planned_swing_bonus=args.swing_bonus
    if args.height_weight is not None:c.reward_scales.height=args.height_weight
    if args.ring_contact_weight is not None:c.reward_scales.ring_side_contact=args.ring_contact_weight
    if args.floor_friction is not None:c.floor_friction=args.floor_friction
    if args.friction_range is not None:c.friction_range=tuple(args.friction_range)
    if args.leg_length_common_range is not None:c.leg_length_common_range=tuple(args.leg_length_common_range)
    if args.leg_length_per_leg_range is not None:c.leg_length_per_leg_range=tuple(args.leg_length_per_leg_range)
    if not 0.<c.friction_range[0]<=c.friction_range[1]:raise ValueError('friction_range must be positive and ordered')
    if min(c.action_scale)<=0.:raise ValueError('Action scales must be positive')
    if c.planned_swing_duration<=0. or c.planned_swing_height<=0.:raise ValueError('Planned swing duration and height must be positive')
    if c.minimum_swing_clearance<=0.:raise ValueError('Minimum swing clearance must be positive')
    if not np.all(np.isfinite(c.planned_swing_foot_weights)) or min(c.planned_swing_foot_weights)<=0.:raise ValueError('Swing foot weights must be finite and positive')
    if not 0.<=c.planned_swing_bonus<.375:raise ValueError('swing_bonus must be in [0,.375) so floor-level hovering cannot earn net swing credit')
    if args.straight_probability is not None:c.straight_probability=args.straight_probability
    if args.axis_probability is not None:c.axis_probability=args.axis_probability
    if args.air_time_floor is not None:c.air_time_floor=args.air_time_floor
    if not 0.<=c.straight_probability<=1.:raise ValueError('straight_probability must be in [0,1]')
    if not 0.<=c.axis_probability<=1.-c.straight_probability:raise ValueError('straight and axis probabilities must sum to at most one')
    devices=jax.devices()
    print('JAX devices:',devices,flush=True)
    if not args.smoke and not any(d.platform=='gpu' for d in devices):
        raise RuntimeError('Full training requires JAX CUDA on the remote Linux host; use --smoke for a CPU check.')
    count=len(devices)
    if c.ppo.num_envs % count or c.ppo.batch_size % count:
        raise ValueError('num_envs and batch_size must be divisible by visible device count')
    if (c.ppo.batch_size*c.ppo.num_minibatches) % c.ppo.num_envs:
        raise ValueError('batch_size * num_minibatches must be divisible by num_envs')
    out=Path(args.output_dir).resolve()/f'walk_{datetime.now():%Y-%m-%d_%H-%M-%S}'
    out.mkdir(parents=True)
    env=PupperWalkEnv(c,args.model_path)
    restore_params=model.load_params(args.init_params) if args.init_params else None
    old_scale=None
    if args.init_params:
        from workspace.walk_warm_start import rescale_action_head
        old_scale=json.loads((Path(args.init_params).parent/'run.json').read_text())['config']['action_scale']
        if not np.allclose(old_scale,c.action_scale):
            restore_params=rescale_action_head(restore_params,old_scale,c.action_scale,c.observation_history)
            print('Adjusted warm-start action head/history normalization for new command scales (approximate tanh remapping).',flush=True)
    eval_c=get_config(); eval_c.update(c.to_dict()); eval_c.sensor_noise=0.; eval_c.latency_probability=0.; eval_c.reset_joint_noise=0.
    eval_c.push_probability=0.
    eval_env=PupperWalkEnv(eval_c,args.model_path)
    metadata={'config':c.to_dict(),'model_path':env.model_path,
              'eval_config':eval_c.to_dict(),'eval_video_fps':round(1/env.dt),'eval_video_mode':video_mode,
              'init_params':str(Path(args.init_params).resolve()) if args.init_params else None,
              'init_params_sha256':hashlib.sha256(Path(args.init_params).read_bytes()).hexdigest() if args.init_params else None,
              'warm_start_action_scale':old_scale,
              'height_reference':'randomized neutral capsule projection; floor-normal torso height',
              'randomization_lifetime':'environment slot',
              'randomization_enabled':not (args.no_randomization or args.smoke),'smoke_test':args.smoke,
              'model_sha256':hashlib.sha256(Path(env.model_path).read_bytes()).hexdigest(),
              'ring_sha256':hashlib.sha256(Path(__file__).with_name('ring_outline.json').read_bytes()).hexdigest(),
              'home_joint_pos':env.mj_model.qpos0[7:].tolist(),
              'effective_home_qpos':np.asarray(env.init_q).tolist(),
              'effective_foot_geometry':dict(size=env.mj_model.geom_size[env.foot_ids].tolist(),
                                           pos=env.mj_model.geom_pos[env.foot_ids].tolist(),
                                           solref=env.mj_model.geom_solref[env.foot_ids].tolist(),
                                           solimp=env.mj_model.geom_solimp[env.foot_ids].tolist()),
              'joint_lower_limits':env.mj_model.jnt_range[1:,0].tolist(),
              'joint_upper_limits':env.mj_model.jnt_range[1:,1].tolist(),
              'kp':env.mj_model.actuator_gainprm[:,0].tolist(),
              'kd':(-env.mj_model.actuator_biasprm[:,2]).tolist(),'platform':platform.platform(),
              'devices':[str(d) for d in devices],
              'versions':{p:importlib.metadata.version(p) for p in ('jax','jaxlib','mujoco','mujoco-mjx','brax','flax','orbax-checkpoint')}}
    (out/'run.json').write_text(json.dumps(metadata,indent=2))
    source_dir=out/'source'; source_dir.mkdir()
    for name in ('walk_config.py','walk_env.py','walk_geometry.py','walk_randomize.py','walk_warm_start.py','train_walk.py','evaluate_walk.py','evaluate_gait_suite.py','render_gait_review.py','upload_walk_video.py'):
        (source_dir/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    Path(out/'model.xml').write_bytes(Path(env.model_path).read_bytes())
    from workspace.walk_geometry import save_effective_model
    save_effective_model(env.mj_model,out/'effective_model.xml',env.model_path)
    run=None
    if args.use_wandb:
        import wandb
        run=wandb.init(project=args.wandb_project,entity=args.wandb_entity,name=out.name,config=metadata)
        metadata['wandb_run_path']=run.path
        (out/'run.json').write_text(json.dumps(metadata,indent=2))
    # PPO has no built-in checkpoint selection -- it just runs to num_timesteps and
    # stops, even mid-collapse. Parameter and metric callbacks arrive separately
    # (in opposite order at step zero). Match by step so the warm start and every
    # later evaluated policy are eligible, regardless of where training ends.
    pending={'step':None,'params':None}
    best={'step':None,'reward':-float('inf'),'kl':float('inf'),'params':None}
    evaluated={'step':None,'reward':-float('inf'),'kl':0.}
    def select_best():
        if pending['step']==evaluated['step'] and pending['params'] is not None and np.isfinite(evaluated['reward']) and evaluated['reward']>best['reward']:
            best.update(**evaluated,params=pending['params'])
            model.save_params(str(out/'best_params'),best['params'])
            (out/'best_checkpoint.json').write_text(json.dumps(evaluated,indent=2))
    def progress(step,metrics):
        with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps({'step':step,**{k:float(v) for k,v in metrics.items()}})+'\n')
        n=max(float(metrics.get('eval/avg_episode_length',1)),1)
        mean=lambda k:float(metrics.get('eval/episode_'+k,float('nan')))/n
        reward=float(metrics.get('eval/episode_reward',float('nan')))
        kl=float(metrics.get('training/kl_mean',0.))
        print(f"{step:,} steps | reward {reward:.3f} | length {n:.0f} | tilt {mean('tilt_deg'):.2f} deg | feet {mean('foot_contacts'):.2f} | ring side {mean('ring_side_fraction'):.3f} | velocity error {mean('velocity_error'):.3f} | kl {kl:.3f}",flush=True)
        if kl>0.15:print(f'WARNING: elevated KL ({kl:.3f}) at step {step:,} -- possible PPO collapse in progress',flush=True)
        evaluated.update(step=step,reward=reward,kl=kl)
        select_best()
        # Video and metric callbacks arrive in either order at the same step.
        # Keep that row open so W&B does not discard the second callback.
        if run is not None:run.log(metrics,step=step,commit=False)
    def save(step,make_policy,params):
        # Independent parameter snapshots survive interrupted SSH sessions/runs.
        model.save_params(str(out/f'params_{step:012d}'),params)
        pending.update(step=step,params=params)
        # Brax reports the initial evaluation BEFORE its initial parameter callback.
        # Include the warm start in selection even if subsequent updates regress.
        select_best()
        if args.smoke or args.no_eval_videos or args.final_videos_only:return
        try:
            inference_fn=make_policy(params,deterministic=True)
            frames,fps=render_showcase(eval_env,inference_fn,jax.random.PRNGKey(0),c.observation_history)
            path=out/f'rollout_step_{step:012d}.mp4'
            media.write_video(str(path),frames,fps=fps)
            if run is not None:run.log({'eval/video':wandb.Video(str(path),fps=fps,format='mp4')},step=step,commit=False)
        except Exception as e:  # noqa: BLE001 -- a video hiccup must not kill a multi-hour run
            print(f'WARNING: eval video render failed at step {step}: {e!r}',flush=True)
    factory=functools.partial(networks.make_ppo_networks,policy_hidden_layer_sizes=tuple(c.hidden_layer_sizes),activation=jax.nn.elu)
    kwargs=c.ppo.to_dict()
    make,params,_=ppo.train(environment=env,eval_env=eval_env,episode_length=c.episode_length,
                           network_factory=factory,progress_fn=progress,policy_params_fn=save,
                           randomization_fn=None if args.no_randomization or args.smoke else functools.partial(
                               domain_randomize,friction_range=c.friction_range,foot_model=c.foot_model,
                               leg_length_common_range=c.leg_length_common_range,
                               leg_length_per_leg_range=c.leg_length_per_leg_range),
                           restore_params=restore_params,
                           **kwargs)
    model.save_params(str(out/'mjx_params'),params)
    if run is not None:run.log({},commit=True)
    print(f'Saved {out}/mjx_params',flush=True)
    if not args.smoke and not args.no_eval_videos:
        try:
            inference_fn=make(params,deterministic=True)
            frames,fps=render_showcase(eval_env,inference_fn,jax.random.PRNGKey(1),c.observation_history)
            path=out/'rollout_final.mp4'
            media.write_video(str(path),frames,fps=fps)
            print(f'Final video -> {path}',flush=True)
            if run is not None:run.log({'eval/video_final':wandb.Video(str(path),fps=fps,format='mp4')})
        except Exception as e:  # noqa: BLE001
            print(f'WARNING: final video render failed: {e!r}',flush=True)
    if not args.smoke and best['params'] is not None:
        model.save_params(str(out/'best_params'),best['params'])
        print(f"Best eval checkpoint: step {best['step']:,} (reward {best['reward']:.3f}, kl {best['kl']:.3f}) -> {out}/best_params",flush=True)
        print('Use best_params for evaluation/export unless the final step matched it.',flush=True)
        if not args.no_eval_videos:
            try:
                inference_fn=make(best['params'],deterministic=True)
                frames,fps=render_showcase(eval_env,inference_fn,jax.random.PRNGKey(2),c.observation_history)
                path=out/'rollout_best.mp4'
                media.write_video(str(path),frames,fps=fps)
                print(f'Best video -> {path}',flush=True)
                if run is not None:run.log({'eval/video_best':wandb.Video(str(path),fps=fps,format='mp4')})
            except Exception as e:  # noqa: BLE001
                print(f'WARNING: best-checkpoint video render failed: {e!r}',flush=True)
    if run is not None:run.finish()

if __name__=='__main__':main()
