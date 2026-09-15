"""Distill the actual RL policy's tested left/right symmetry into its original MLP."""
import argparse,json,hashlib
from pathlib import Path
import jax
import jax.numpy as jp
import numpy as np
import optax
from brax.io import model
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks
import workspace.audit_wheel_lift as audit

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--params',type=Path,required=True);p.add_argument('--policy',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--steps',type=int,default=8000);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);params=model.load_params(str(a.params));swap=np.array([2,3,0,1,6,7,4,5]);observations=[]
    original=audit.actor
    def teacher_native(net,obs):
        observations.append(obs.copy())
        if np.argmax(obs[6:11])!=1:return original(net,obs)
        x=obs.copy();x[[0,2,4]]*=-1;x[6:11]=np.eye(5)[2];x[11:19]=-obs[11:19][swap];x[19:27]=-obs[19:27][swap]
        return -original(net,x)[swap]
    audit.actor=teacher_native
    result=audit.run(a.policy,a.out/'teacher_audit');result['scope']='Symmetry-transformed teacher diagnostic'
    (a.out/'teacher_audit/result.json').write_text(json.dumps(result,indent=2)+'\n')
    data=jp.asarray(np.asarray(observations,dtype=np.float32));np.save(a.out/'teacher_observations.npy',np.asarray(data))
    net=networks.make_ppo_networks(27,8,preprocess_observations_fn=running_statistics.normalize,policy_hidden_layer_sizes=(128,128,128),value_hidden_layer_sizes=(128,128,128),activation=jax.nn.elu)
    def predict(weights,obs):return jp.tanh(net.policy_network.apply(params[0],weights,obs)[...,:8])
    def teacher(obs):
        x=obs.at[:,jp.array([0,2,4])].multiply(-1)
        x=x.at[:,6:11].set(jax.nn.one_hot(jp.full((obs.shape[0],),2),5))
        x=x.at[:,11:19].set(-obs[:,11:19][:,swap]);x=x.at[:,19:27].set(-obs[:,19:27][:,swap])
        return jp.where((obs[:,7]>.5)[:,None],-predict(params[1],x)[:,swap],predict(params[1],obs))
    optimizer=optax.adam(optax.cosine_decay_schedule(3e-4,a.steps,alpha=.03));weights=params[1];opt_state=optimizer.init(weights)
    scale=jp.array([.06]*3+[.015]*3+[0.]*5+[.025]*8+[.04]*8)
    @jax.jit
    def step(weights,opt_state,key):
        key,ik,nk=jax.random.split(key,3);index=jax.random.randint(ik,(512,),0,data.shape[0]);obs=data[index]+jax.random.normal(nk,(512,27))*scale
        expected=teacher(obs)
        loss,grad=jax.value_and_grad(lambda w:jp.mean(jp.square(predict(w,obs)-expected)))(weights)
        updates,opt_state=optimizer.update(grad,opt_state,weights)
        return optax.apply_updates(weights,updates),opt_state,key,loss
    key=jax.random.PRNGKey(17);metrics=[]
    for i in range(a.steps):
        weights,opt_state,key,loss=step(weights,opt_state,key)
        if (i+1)%500==0:
            value=float(loss);metrics.append(dict(step=i+1,mse=value));print(i+1,value,flush=True)
            model.save_params(str(a.out/f'params_{i+1}'),(params[0],weights,params[2]))
    model.save_params(str(a.out/'mjx_params'),(params[0],weights,params[2]))
    record=dict(method='RL-policy symmetry distillation into unchanged MLP',teacher_sha256=hashlib.sha256(a.params.read_bytes()).hexdigest(),teacher_export_sha256=hashlib.sha256(a.policy.read_bytes()).hexdigest(),samples=int(data.shape[0]),gradient_steps=a.steps,seed=17,metrics=metrics,hub_observations=False,external_uploads=False)
    (a.out/'training.json').write_text(json.dumps(record,indent=2)+'\n')

if __name__=='__main__':main()
