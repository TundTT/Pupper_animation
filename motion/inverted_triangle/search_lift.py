"""Bounded offline physics search for a three-support clearance pose.

Output is an UNACCEPTED candidate, never hardware motion. This searches the
post-cooled model directly, not an RL policy. Every objective is a PD replay.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
import mujoco as mj
from scipy.optimize import differential_evolution, minimize
from .core import Robot,LEGS,PROX,HUB,smooth,HERE,duration,provenance

def envelope(robot,leg):
    """Conservative minimum z over any axial rotation, actual CAD hull vertices."""
    g=robot.geoms[leg];bid=robot.bodies[leg]
    # Express vertices in the joint-body frame, before its world transform.
    R=np.empty(9);mj.mju_quat2Mat(R,robot.m.geom_quat[g])
    v=robot.vertices[leg]@R.reshape(3,3).T+robot.m.geom_pos[g]
    row=robot.d.xmat[bid].reshape(3,3)[2]
    z=robot.d.xpos[bid,2]+v[:,2]*row[2]-np.linalg.norm(v[:,:2],axis=1)*np.linalg.norm(row[:2])
    return float(z.min())

class LiftSearch:
    def __init__(self,leg=0,start=None):
        self.leg=leg;self.r=Robot();self.n=0;self.best=None
        self.home=self.r.initial[7:].copy()
        self.start=start
        if start:self.home=self.r.restore(start)
        self.limits=self.r.m.jnt_range[1:][PROX].copy()
        self.cache={}
        self.settled=self.r.d.qpos.copy()
        for _ in range(1040):self.r.tick(self.home)
        self.settled=self.r.d.qpos.copy();self.settled_v=self.r.d.qvel.copy()
        self.settled_state=self.r.snapshot(self.home)
        self.env_steps=1040
    def evaluate(self,x):
        r=self.r;r.restore(self.settled_state)
        end=self.home.copy();end[PROX]=x
        shift=end.copy();shift[3*self.leg:3*self.leg+2]=self.home[3*self.leg:3*self.leg+2]
        worst_tilt=0.;worst_torque=0.;other_floor_force=0.
        for start,target,seconds in [(self.home,shift,1.5),(shift,end,2.),(end,end,.8)]:
            steps=int(np.ceil(duration(start,target,seconds)/r.m.opt.timestep))
            for k in range(steps):
                tau=r.tick(start+smooth((k+1)/steps)*(target-start))
                self.env_steps+=1
                if k%26==0:
                    worst_tilt=max(worst_tilt,r.tilt());worst_torque=max(worst_torque,float(np.max(np.abs(tau))))
                    other_floor_force=max(other_floor_force,r.unintended_floor_force())
                    if r.tilt()>.65 or r.d.qpos[2]<.045:
                        self.n+=1
                        return 1000+worst_tilt*100
        forces=r.contacts_precise();bottoms=r.bottoms();env=envelope(r,self.leg)
        support=np.delete(forces,self.leg)
        tracking=float(np.max(np.abs(r.d.qpos[7:][PROX]-x)))
        cost=(max(.008-env,0)*180)**2+(max(worst_tilt-.08,0)*20)**2
        cost+=float(np.sum(np.maximum(2.-support,0)**2))*2
        cost+=max(forces[self.leg]-.2,0)**2
        cost+=max(worst_torque-2.5,0)**2*3
        cost+=float(np.sum((x-self.home[PROX])**2))*.04
        cost+=max(tracking-.15,0)**2*10
        cost+=max(other_floor_force-.1,0)**2*10
        self.n+=1
        item=dict(cost=float(cost),evaluation=self.n,leg=LEGS[self.leg],proximal_pose=x.tolist(),full_pose=end.tolist(),sweep_envelope_m=env,tilt_deg=float(np.rad2deg(worst_tilt)),normal_force_N=forces.tolist(),peak_requested_torque_Nm=worst_torque,maximum_unintended_floor_force_N=other_floor_force,tracking_rad=tracking,model_sha256=r.manifest['model_sha256'],status='UNACCEPTED_LIFT_CANDIDATE')
        if self.start:item['start_state']=self.start
        if self.best is None or cost<self.best['cost']:
            self.best=item
            if hasattr(self,'output'):
                (self.output/'best_lift.json').write_text(json.dumps(item,indent=2))
                with (self.output/'improvements.jsonl').open('a') as f:f.write(json.dumps(item)+'\n')
            print(json.dumps(item),flush=True)
        return cost

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--leg',choices=LEGS,default='front_r');p.add_argument('--output',type=Path,required=True);p.add_argument('--maxiter',type=int,default=12);p.add_argument('--seed',type=int,default=0);p.add_argument('--method',choices=['powell','de'],default='powell');p.add_argument('--candidate',type=Path);p.add_argument('--radius',type=float,default=.35)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'source_hashes.json').write_text(json.dumps(provenance(),indent=2))
    s=LiftSearch(LEGS.index(a.leg));s.output=a.output
    poses=json.loads((HERE.parents[1]/'ros2_ws/src/neural_controller/launch/keyframe_config.json').read_text())['poses']
    x=np.array(poses[s.leg])
    for leg in range(4):
        if leg!=s.leg:x[2*leg+1]=s.home[3*leg+1]
    if a.candidate:
        candidate=json.loads(a.candidate.read_text())
        if candidate['model_sha256']!=s.r.manifest['model_sha256'] or candidate['leg']!=a.leg:
            raise ValueError('Candidate model or leg differs; regenerate it explicitly')
        x=np.asarray(candidate['proximal_pose'])
    s.evaluate(x)
    start=time.monotonic()
    # Allow the planner to find materially higher lifts within actual soft limits.
    if not np.isfinite(a.radius) or a.radius<=0:raise ValueError('radius must be positive')
    bounds=list(zip(np.maximum(s.limits[:,0],x-a.radius),np.minimum(s.limits[:,1],x+a.radius)))
    if a.method=='powell':result=minimize(s.evaluate,x,method='Powell',bounds=bounds,options=dict(maxiter=a.maxiter,maxfev=a.maxiter*100,xtol=.008,ftol=.001))
    else:result=differential_evolution(s.evaluate,bounds,x0=x,maxiter=a.maxiter,popsize=5,seed=a.seed,polish=False)
    (a.output/'search.json').write_text(json.dumps(dict(method=a.method,seed=a.seed,bounds=bounds,evaluations=s.n,seconds=time.monotonic()-start,solver_success=bool(result.success),message=str(result.message),best=s.best),indent=2))

if __name__=='__main__':main()
