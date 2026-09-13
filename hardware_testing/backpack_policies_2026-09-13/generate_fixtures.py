import argparse,json
from pathlib import Path
import numpy as np,jax
from jax import numpy as jp
from brax.io import model
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks
p=argparse.ArgumentParser();p.add_argument('kind');p.add_argument('params');p.add_argument('output');a=p.parse_args()
params=model.load_params(a.params);width=int(params[0].mean.shape[0]);rng=np.random.default_rng(20260913)
mean=np.asarray(params[0].mean);std=np.asarray(params[0].std)
obs=(mean+std*rng.normal(0,.75,(128,width))).astype(np.float32);obs[0]=mean;obs[1]=0
if a.kind=='leg':
 for i in range(4):obs[:,i*36+9:i*36+12]=[0,0,1]
net=networks.make_ppo_networks(width,12,preprocess_observations_fn=running_statistics.normalize,policy_hidden_layer_sizes=(128,128,128),activation=jax.nn.elu)
f=networks.make_inference_fn(net)(params,deterministic=True)
actions,_=jax.jit(jax.vmap(f,in_axes=(0,None)))(jp.array(obs),jax.random.PRNGKey(0));actions=np.asarray(actions)
assert actions.shape==(128,12) and np.isfinite(actions).all()
np.savetxt(a.output,np.concatenate([obs,actions],axis=1),delimiter=',',fmt='%.10g')
print(a.kind,'128 independent JAX checkpoint reference actions saved:',a.output)
