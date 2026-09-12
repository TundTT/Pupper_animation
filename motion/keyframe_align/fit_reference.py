"""Bounded native-physics keyframe fitting. No network, training, or robot access."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
import mujoco
from scipy.optimize import minimize
from training.wheel_align import configs as c, geometry

def fit(output):
    m=mujoco.MjModel.from_xml_path(str(c.MODEL_PATH));d=mujoco.MjData(m);pos=c.POSITION_ACTUATOR_ROWS
    m.actuator_gainprm[pos,0]=5;m.actuator_biasprm[pos,1]=-5;m.actuator_biasprm[pos,2]=-.25
    base=c.APEX_POSES[0].copy();neutral=c.DEFAULT_POSE[pos];best=[float('inf'),None,None]
    def evaluate(x):
        mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_step(m,d,nstep=1560)
        shift=x.copy();shift[1]=.5
        for end,start in [(shift,neutral),(x,shift)]:
            for n in range(65):
                u=(n+1)/65;u=u*u*u*(10+u*(-15+6*u));d.ctrl[pos]=start+u*(end-start)
                mujoco.mj_step(m,d,nstep=20)
        mujoco.mj_step(m,d,nstep=1560)
        r=np.empty(9);mujoco.mju_quat2Mat(r,d.qpos[3:7]);g=r.reshape(3,3).T@np.array([0.,0.,-1.])
        tilt=math.acos(np.clip(-g[2],-1,1));q=d.qpos[7:]
        gravities=[]
        for a in [-.04,0,.04]:
            for b in [-.04,0,.04]:
                ca,sa,cb,sb=np.cos(a),np.sin(a),np.cos(b),np.sin(b)
                gravities.append(np.array([[cb,sb*sa,sb*ca],[0,ca,-sa],[-sb,cb*sa,cb*ca]]).T@g)
        margins=np.min([geometry.margins(q,v,0) for v in gravities],axis=0)
        ideal=c.DEFAULT_POSE.copy();ideal[pos]=x
        static_g=[np.array([a,b,-np.sqrt(1-a*a-b*b)]) for a in [-.05,0,.05] for b in [-.05,0,.05]]
        static_margins=np.min([geometry.margins(ideal,v,0) for v in static_g],axis=0)
        deficit=np.maximum(np.array([.016,.016,.010])-margins,0)
        static_deficit=np.maximum(np.array([.014,.014,.008])-static_margins,0)
        cost=np.sum((x-base)**2)*.02+(np.sum(deficit**2)+np.sum(static_deficit**2))*30000+max(tilt-.055,0)**2*100
        if cost<best[0]:
            best[:]=[float(cost),x.tolist(),dict(margins=margins.tolist(),static_margins=static_margins.tolist(),tilt_rad=tilt)]
            Path(output).write_text(json.dumps(dict(cost=best[0],pose=best[1],diagnostic=best[2]),indent=2))
        return cost
    lo=np.maximum(base-.7,np.array([-1.12,-.32,-2.41,-3.04,-1.12,-.32,-2.41,-3.04]))
    hi=np.minimum(base+.7,np.array([2.41,3.04,1.12,.32,2.41,3.04,1.12,.32]))
    initial=np.array([.8818665917,1.1671579454,-1.2625993809,.2875457219,.8387629874,-.1964845862,-.7262780738,-.0873463978])
    result=minimize(evaluate,initial,method='Powell',bounds=list(zip(lo,hi)),options=dict(maxiter=20,maxfev=700,xtol=.002,ftol=.0001))
    print(json.dumps(dict(best_cost=best[0],pose=best[1],diagnostic=best[2],evaluations=result.nfev),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);fit(p.parse_args().output)
