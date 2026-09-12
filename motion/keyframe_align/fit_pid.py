"""Offline pose fitting through the position-PID core and native dynamics. No training."""
import argparse,json,math,copy,time
from pathlib import Path
import numpy as np,mujoco
from scipy.optimize import minimize
from .native import Controller,CONFIG
from training.wheel_align.configs import MODEL_PATH
POS=np.array([0,1,3,4,6,7,9,10]);WHEEL=np.array([2,5,8,11]);DT=1/520

def fit(leg,output,maxfev=280,config=CONFIG):
 cfg=json.loads(Path(config).read_text());base=np.array(cfg['poses'][leg]);model=mujoco.MjModel.from_xml_path(str(MODEL_PATH));d=mujoco.MjData(model)
 kp=np.full(12,5.);kd=np.full(12,.25);kp[WHEEL]=cfg['wheel_position_kp'];kd[WHEEL]=cfg['wheel_position_kd']
 def drive(target,effort=None):
  tau=kp*(target-d.qpos[7:])-kd*d.qvel[6:]
  if effort is not None:tau[WHEEL]+=effort
  d.ctrl[:]=np.clip(tau,-3,3);mujoco.mj_step(model,d)
 mujoco.mj_resetDataKeyframe(model,d,0);initial=d.qpos[7:].copy()
 for _ in range(1560):drive(initial)
 state=np.zeros(mujoco.mj_stateSize(model,mujoco.mjtState.mjSTATE_FULLPHYSICS));mujoco.mj_getState(model,d,state,mujoco.mjtState.mjSTATE_FULLPHYSICS)
 count=0;best=[float('inf'),None];start=time.monotonic()
 def evaluate(x):
  nonlocal count
  count+=1;mujoco.mj_setState(model,d,state,mujoco.mjtState.mjSTATE_FULLPHYSICS);mujoco.mj_forward(model,d)
  candidate=copy.deepcopy(cfg);candidate['poses'][leg]=x.tolist();c=Controller(config=candidate);c.reset(d.qpos[7:],d.qpos[7+WHEEL]);values=[];peak=0.;last=None
  command=(2,1,3,4)[leg]
  for n in range(5200):
   r=d.xmat[1].reshape(3,3);g=r.T@np.array([0.,0.,-1.]);v=np.empty(6);mujoco.mj_objectVelocity(model,d,mujoco.mjtObj.mjOBJ_XBODY,1,v,1)
   o=c.step(DT,0 if n<2080 else command,d.qpos[7:],d.qvel[6:],v[:3],g)
   tilt=math.acos(np.clip(-g[2],-1,1));peak=max(peak,tilt)
   if not o[23]:break
   target=np.zeros(12);target[POS]=o[:8];target[WHEEL]=o[25:29];drive(target,o[29:33])
   if n>4680:values.append([*o[17:20],tilt,*v[:3]])
   last=o
  c.close()
  if not values:cost=100+peak;tilt=None
  else:
   v=np.array(values);margins=v[:,:3].min(axis=0);tilt=v[:,3].max();ang=np.linalg.norm(v[:,4:],axis=1).max()
   cost=float(30000*np.sum(np.maximum(np.array([.018,.020,.012])-margins,0)**2)+250*max(tilt-.05,0)**2+10*max(ang-.08,0)**2+400*max(peak-.11,0)**2+.008*np.sum((x-base)**2))
  if cost<best[0]:
   best[:]=[cost,x.tolist()];record=dict(leg=leg,cost=cost,pose=x.tolist(),evaluations=count,seconds=time.monotonic()-start,peak_tilt=peak,tail_margins=margins.tolist() if values else None,tail_tilt=tilt,source_config=cfg,hardware_validated=False)
   Path(output).write_text(json.dumps(record,indent=2));print(leg,count,round(cost,5),record['tail_margins'],round(peak,3),flush=True)
  return cost
 evaluate(base)
 lo=np.maximum(base-.5,[-1.12,-.32,-2.41,-3.04,-1.12,-.32,-2.41,-3.04]);hi=np.minimum(base+.5,[2.41,3.04,1.12,.32,2.41,3.04,1.12,.32])
 minimize(evaluate,base,method='Powell',bounds=list(zip(lo,hi)),options={'maxfev':maxfev,'maxiter':8,'xtol':.006,'ftol':.001})
 print('DONE',leg,count,best[0],flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--leg',type=int,choices=range(4),required=True);p.add_argument('--output',required=True);p.add_argument('--config',type=Path,default=CONFIG);p.add_argument('--maxfev',type=int,default=280);a=p.parse_args();fit(a.leg,a.output,a.maxfev,a.config)
