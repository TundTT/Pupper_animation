"""Held-out teacher-state enable errors, separate from closed-loop task success."""
import argparse,json
from pathlib import Path
import jax,numpy as np
from brax.io import model
from brax.training.acme import running_statistics as rs
from brax.training.agents.ppo import networks
from workspace import configs
p=argparse.ArgumentParser();p.add_argument('params',type=Path);a=p.parse_args()
c=json.loads((a.params.parent/'config.json').read_text()); norm,actor,_=model.load_params(a.params)
with np.load(c['direct']['supervision_path']) as d:
 obs,labels,episode,group=[d[k] for k in ('obs','actions','episode','group')]
valid=np.isin(episode,np.unique(episode)[::5]);r=np.random.default_rng(44)
ids=np.concatenate([r.choice(np.flatnonzero(valid&(group==g)),128) for g in np.unique(group)])
net=networks.make_ppo_networks(224,12,policy_hidden_layer_sizes=tuple(c['policy']['hidden_layer_sizes']),value_hidden_layer_sizes=tuple(c['policy']['hidden_layer_sizes']),activation=jax.nn.elu,preprocess_observations_fn=rs.normalize)
logits=np.asarray(jax.jit(net.policy_network.apply)(norm,actor,obs[ids]))[:,:12]
gates=np.tanh(logits[:,configs.WHEEL_ACTUATOR_ROWS]);group=group[ids]
report={}
for leg,row in configs.FOOT_ROW_BY_LEG.items():
 cmd=configs.COMMAND_STATES.index(leg);lift=group==cmd*5+1;rotate=group==cmd*5+2
 report[leg]={'lift_false_enable_fraction':float((gates[lift,row]>0).mean()),'rotate_recall':float((gates[rotate,row]>0).mean()),'lift_output_quantiles':np.percentile(gates[lift,row],[5,50,95]).tolist(),'rotate_output_quantiles':np.percentile(gates[rotate,row],[5,50,95]).tolist()}
out=a.params.with_name(a.params.name+'_gate_classifier.json');out.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
