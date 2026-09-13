"""Explicit local physics search; independent experimental controller source."""
import json,concurrent.futures,hashlib
from pathlib import Path
import numpy as np,mujoco as mj
from motion.inverted_triangle.core import HERE,provenance,versions,KP,KD
from motion.inverted_triangle.adaptive_support import AdaptiveSupport
from motion.inverted_triangle.roll_to_stand import probe
root=Path('runs/roll_to_stand/support-screen-v2')
class Support(AdaptiveSupport):
 def update(self,quat,q,qd,previous,dt):
  r=self.nominal;r.d.qpos[:3]=[0,0,1];r.d.qpos[3:7]=quat;r.d.qpos[7:]=q;r.d.qvel[:]=0;mj.mj_forward(r.m,r.d)
  tau=KP*(previous-q)-KD*qd
  for i in range(4):
   p=r.points(i)[r.tip_masks[i]];point=p[np.argmin(p[:,2])];jac=np.zeros((3,r.m.nv));rot=np.zeros_like(jac);mj.mj_jac(r.m,r.d,jac,rot,point,r.bodies[i]);j=np.arange(3*i,3*i+3);J=jac[:,6+j];external=r.d.qfrc_bias[6+j]-tau[j]
   if self.config['estimator']=='vertical':force=J[2]@external/(J[2]@J[2]+1e-8)
   else:force=np.linalg.solve(J@J.T+np.eye(3)*1e-9,J@external)[2]
   self.estimated_load_N[i]+=(1-np.exp(-dt/.4))*(force-self.estimated_load_N[i])
   error=np.clip(self.config['force_target_N']-self.estimated_load_N[i],-6,6)
   self.reference[j]-=self.config['force_gain']*error*J[2]/max(np.linalg.norm(J[2]),.01)*dt
  cap=self.config['reference_cap'];self.reference=np.clip(self.reference,self.goal-cap,self.goal+cap)
  self.integral=np.clip(self.integral+self.config['integral_gain']*(self.reference-q)*dt,-.15,.15)
  offset=np.clip(self.reference+self.integral-self.goal,-.2,.2);self.max_offset_rad=max(self.max_offset_rad,float(abs(offset).max()));return offset

def run(c):
 import motion.inverted_triangle.adaptive_support as module
 module.AdaptiveSupport=Support
 return probe(c,root/c['name'],cad=False,video=False)
if __name__=='__main__':
 root.mkdir(exist_ok=False);cases=json.loads((HERE/'formation_cases.json').read_text());configs=[]
 variants=[dict(estimator=e,force_gain=f,force_target_N=9.,integral_gain=.5,reference_cap=cap) for e in ['vertical','full'] for f in [.003,.008] for cap in [.06,.085]]
 for k,v in enumerate(variants):configs.extend([dict(c,name=f'v{k}-'+c['name'],hold_seconds=18.,adaptive_support=v) for c in cases])
 (root/'provenance.json').write_text(json.dumps(dict(wandb_mode='disabled-local-optimization-will-archive',source_hashes=provenance(),experimental_driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),environment=versions(),configs=configs),indent=2))
 with concurrent.futures.ProcessPoolExecutor(12) as pool:reports=list(pool.map(run,configs))
 (root/'summary.json').write_text(json.dumps(reports,indent=2))
 for k in range(len(variants)):
  batch=reports[k*7:k*7+7];print('VARIANT',k,'PASS',sum(all(v for key,v in r['gates'].items() if 'cad' not in key) for r in batch),[r['config']['name']+':'+','.join(key for key,v in r['gates'].items() if v is False) for r in batch],flush=True)
