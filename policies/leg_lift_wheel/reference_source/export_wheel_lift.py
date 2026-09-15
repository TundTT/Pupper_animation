"""Export only the new eight-action wheel-lift contract; never label it legacy leg_lift."""
import argparse,json,hashlib
from pathlib import Path
from brax.io import model
from workspace.export_policy import convert_params
from workspace import wheel_lift_config as c

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--params',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    params=model.load_params(str(a.params))
    net=convert_params(params,activation='elu')
    if net['in_shape'] != [None,27] or net['layers'][-1]['shape'] != [None,8]:
        raise ValueError('Checkpoint does not match the 27-input / 8-action wheel-lift contract')
    config=c.get_config()
    net.update(behavior='leg_lift_wheel_position',contract='leg-lift-wheel-position-v1',
        policy_action_size=8,motor_command_count=12,action_types=['position']*12,
        proximal_indices=c.PROXIMAL.tolist(),hub_indices=c.HUBS.tolist(),
        default_joint_pos=c.DEFAULT_POSE.tolist(),action_scale=c.ACTION_SCALE.tolist(),
        kps=list(config.position_control_kp),kds=list(config.dof_damping),
        joint_lower_limits=c.JOINT_LOWER_LIMITS.tolist(),joint_upper_limits=c.JOINT_UPPER_LIMITS.tolist(),
        observation_history=1,observation_layout=['angular[3]','gravity[3]','command[5]','proximal_position_minus_default[8]','previous_proximal_action[8]'],
        command_states=c.COMMAND_STATES,hub_observations=False,rotation_command_observed=False,
        checkpoint_sha256=hashlib.sha256(a.params.read_bytes()).hexdigest(),
        model_sha256=hashlib.sha256(c.resolve_model_path().read_bytes()).hexdigest(),
        runtime_status='EXPERIMENTAL_REQUIRES_NEW_RUNTIME_ADAPTER')
    # Compare the exported graph with actual Brax deterministic inference.
    import jax
    import numpy as np
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks
    from workspace.evaluate_wheel_lift import actor
    network=networks.make_ppo_networks(27,8,preprocess_observations_fn=running_statistics.normalize,
        policy_hidden_layer_sizes=(128,128,128),value_hidden_layer_sizes=(128,128,128),activation=jax.nn.elu)
    infer=networks.make_inference_fn(network)(params,deterministic=True)
    samples=np.random.default_rng(812).normal(size=(128,27)).astype(np.float32)
    expected=np.asarray(jax.vmap(lambda obs:infer(obs,jax.random.PRNGKey(0))[0])(samples))
    actual=np.array([actor(net,x) for x in samples])
    error=float(np.max(np.abs(expected-actual)))
    if not np.isfinite(error) or error>2e-4:raise ValueError('Export inference mismatch: '+str(error))
    net['export_parity_max_abs_error']=error
    net['export_parity_samples']=len(samples)
    a.out.write_text(json.dumps(net,indent=2)+'\n')
if __name__=='__main__':main()
