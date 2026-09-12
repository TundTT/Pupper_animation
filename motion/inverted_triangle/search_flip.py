"""Optimize complete dynamic flips; never accept an objective as an audit pass."""
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize,differential_evolution
from .core import Robot,LEGS,PROX,HUB,smooth,duration,provenance,versions
from .simulate import trajectory


class FlipSearch:
    def __init__(self,leg,output,start=None,direction=-1):
        self.r=Robot();self.leg=leg;self.output=Path(output);self.output.mkdir(parents=True,exist_ok=False)
        self.home=self.r.restore(start) if start else self.r.initial[7:].copy()
        self.start=start or self.r.snapshot(self.home);self.continuation=start is not None
        self.direction=direction;self.best=None;self.n=0;self.env_steps=0
        (self.output/'source_hashes.json').write_text(json.dumps(provenance(),indent=2))
        (self.output/'environment.json').write_text(json.dumps(versions(),indent=2))

    def evaluate(self,x):
        r=self.r;r.restore(self.start)
        lift=self.home.copy();lift[PROX]=x[:8]
        phases=trajectory(self.home,lift,self.leg,float(x[8]),self.direction)
        limits=r.m.jnt_range[1:][PROX]
        if any(np.any((q[PROX]<limits[:,0])|(q[PROX]>limits[:,1])) for _,q,_ in phases):return 1e6
        previous=self.home.copy();minimum_gap=1.;minimum_support=1e6;max_contact=0.;other_floor=0.;tilt=0.;torque=0.;terminated=False
        for phase,target,seconds in phases:
            steps=int(np.ceil(duration(previous,target,seconds)/r.m.opt.timestep))
            for k in range(steps):
                tau=r.tick(previous+smooth((k+1)/steps)*(target-previous));self.env_steps+=1
                torque=max(torque,float(np.abs(tau).max()))
                if k%26:continue
                tilt=max(tilt,r.tilt());other_floor=max(other_floor,r.unintended_floor_force())
                if phase in ['rotate','angle_hold']:
                    minimum_gap=min(minimum_gap,float(r.bottoms()[self.leg]))
                    f=r.contacts_precise();minimum_support=min(minimum_support,float(np.delete(f,self.leg).min()));max_contact=max(max_contact,float(f[self.leg]))
                if tilt>.6 or r.d.qpos[2]<.045 or not np.isfinite(r.d.qpos).all():terminated=True;break
            if terminated:break
            previous=target.copy()
        f=r.contacts_precise();tracking=abs(r.d.qpos[7+HUB[self.leg]]-(self.home[HUB[self.leg]]+self.direction*np.pi))
        cost=1000.*terminated
        cost+=max(.008-minimum_gap,0)**2*40000+max(2.-minimum_support,0)**2*4
        cost+=max(max_contact-.1,0)**2*5+max(other_floor-.05,0)**2*10
        cost+=np.sum(np.maximum(2.-f,0)**2)*4+max(f[self.leg]-12.,0)**2
        cost+=max(tilt-np.deg2rad(4),0)**2*400+max(torque-2.5,0)**2*4
        cost+=max(tracking-.025,0)**2*10+float(np.sum((x[:8]-self.home[PROX])**2))*.02
        self.n+=1
        item=dict(status='UNACCEPTED_FULL_FLIP_CANDIDATE',leg=LEGS[self.leg],full_pose=lift.tolist(),proximal_pose=x[:8].tolist(),landing_delta=float(x[8]),direction=self.direction,cost=float(cost),evaluation=self.n,environment_steps=self.env_steps,model_sha256=r.manifest['model_sha256'],minimum_rotation_floor_m=minimum_gap,minimum_support_force_N=minimum_support,maximum_unintended_floor_force_N=other_floor,final_normal_force_N=f.tolist(),max_tilt_deg=float(np.rad2deg(tilt)),early_termination=terminated)
        if self.continuation:item['start_state']=self.start
        if self.best is None or cost<self.best['cost']:
            self.best=item
            (self.output/'best_flip.json').write_text(json.dumps(item,indent=2))
            with (self.output/'improvements.jsonl').open('a') as out:out.write(json.dumps(item)+'\n')
            print(json.dumps(item),flush=True)
        return float(cost)


def optimize(output,leg='back_r',candidate=None,start=None,maxiter=12,method='powell',radius=.3,seed=0,direction=-1):
    s=FlipSearch(LEGS.index(leg),output,start,direction)
    x=np.r_[s.home[PROX],.3]
    # Generic seed only; optimize all support joints for the actual mixed stance.
    x[2*s.leg+1]=(1.2 if s.leg%2==0 else -1.2)
    if candidate:
        c=json.loads(Path(candidate).read_text())
        if c['model_sha256']!=s.r.manifest['model_sha256'] or c['leg']!=leg:raise ValueError('Candidate model/leg mismatch')
        x=np.r_[c['proximal_pose'],c.get('landing_delta',.3)]
    if not np.isfinite(radius) or radius<=0:raise ValueError('radius must be positive')
    limits=s.r.m.jnt_range[1:][PROX]
    bounds=list(zip(np.maximum(limits[:,0],x[:8]-radius),np.minimum(limits[:,1],x[:8]+radius)))+[(.03,.8)]
    s.evaluate(x);begin=time.monotonic()
    if method=='powell':result=minimize(s.evaluate,x,method='Powell',bounds=bounds,options=dict(maxiter=maxiter,maxfev=maxiter*120,xtol=.005,ftol=.001))
    else:result=differential_evolution(s.evaluate,bounds,x0=x,maxiter=maxiter,popsize=6,seed=seed,polish=False)
    report=dict(method=method,seed=seed,direction=direction,bounds=bounds,seconds=time.monotonic()-begin,evaluations=s.n,environment_steps=s.env_steps,solver_success=bool(result.success),message=str(result.message),best=s.best)
    (Path(output)/'search.json').write_text(json.dumps(report,indent=2))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--leg',choices=LEGS,default='back_r');p.add_argument('--candidate',type=Path);p.add_argument('--start-state',type=Path)
    p.add_argument('--maxiter',type=int,default=12);p.add_argument('--method',choices=['powell','de'],default='powell');p.add_argument('--radius',type=float,default=.3);p.add_argument('--seed',type=int,default=0);p.add_argument('--direction',type=int,choices=[-1,1],default=-1)
    a=p.parse_args();start=json.loads(a.start_state.read_text()) if a.start_state else None
    optimize(a.output,a.leg,a.candidate,start,a.maxiter,a.method,a.radius,a.seed,a.direction)

if __name__=='__main__':main()
