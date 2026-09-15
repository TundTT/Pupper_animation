"""Closed-loop native validation of the shipped C++ core and Eigen RTNeural actor."""
import argparse,ctypes,json,math
from pathlib import Path
import mujoco
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--library',required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--hz',type=int,default=520);a=p.parse_args()
r=Path(__file__).resolve().parents[2];policy=r/'ros2_ws/src/neural_controller/launch/policy_leg_lift_wheel.json'
lib=ctypes.CDLL(a.library);D=ctypes.POINTER(ctypes.c_double);F=ctypes.POINTER(ctypes.c_float)
lib.wheel_create.argtypes=[ctypes.c_char_p];lib.wheel_create.restype=ctypes.c_void_p
lib.wheel_step.argtypes=[ctypes.c_void_p,ctypes.c_double,D,ctypes.c_int,D];lib.wheel_step.restype=ctypes.c_int
lib.wheel_error.argtypes=[ctypes.c_void_p];lib.wheel_error.restype=ctypes.c_char_p
lib.wheel_destroy.argtypes=[ctypes.c_void_p];lib.wheel_actor.argtypes=[ctypes.c_void_p,F,F]
b=lib.wheel_create(str(policy).encode());assert b
net=json.loads(policy.read_text());rng=np.random.default_rng(913);parity=0
for _ in range(128):
 obs=rng.normal(0,.5,27).astype(np.float32);expected=obs.astype(float)
 for layer in net['layers']:
  w,bias=layer['weights'];expected=expected@np.asarray(w)+np.asarray(bias)
  if layer['activation']=='elu':expected=np.where(expected>0,expected,np.expm1(np.minimum(expected,0)))
  elif layer['activation']=='tanh':expected=np.tanh(expected)
 output=np.zeros(8,np.float32);lib.wheel_actor(b,obs.ctypes.data_as(F),output.ctypes.data_as(F));parity=max(parity,float(np.max(np.abs(expected-output))))
assert parity<1e-5,parity
m=mujoco.MjModel.from_xml_path(str(r/'policies/leg_lift_wheel/model/model.xml'));m.opt.timestep=1/a.hz;np.testing.assert_allclose(m.actuator_gainprm[:,0],[5,5,4]*4);np.testing.assert_allclose(-m.actuator_biasprm[:,2],[.25,.25,.15]*4);d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,0)
nominal=np.array([1,0,0,-1,0,0]*2);d.ctrl[:]=nominal
for _ in range(2*a.hz):mujoco.mj_step(m,d)
root=m.body('base_link').id;wheels=[m.geom('leg_'+k+'_3_wheel_collision').id for k in ('front_r','front_l','back_r','back_l')]
start=d.qpos[:2].copy();result=np.zeros(20);dwell=0;events=[];peak_tilt=peak_drift=peak_pd=0;reason='timeout';previous=None
for tick in range(180*a.hz):
 mujoco.mj_forward(m,d)
 gravity=d.xmat[root].reshape(3,3).T@np.array([0,0,-1.]);angular=d.sensor('body_gyro').data.copy()
 sensor=np.r_[d.qpos[7:],d.qvel[6:],angular,gravity].astype(np.float64)
 event=2 if tick==0 else 1 if dwell>=2 else 0
 if event:dwell=0
 if lib.wheel_step(b,1/a.hz,sensor.ctypes.data_as(D),event,result.ctypes.data_as(D)):
  reason=lib.wheel_error(b).decode();break
 stage,leg=int(result[12]),int(result[13]);state=(stage,leg)
 if state!=previous:events.append({'time':tick/a.hz,'stage':stage,'leg':leg});previous=state
 condition=result[14] if stage in (1,4) else result[15] if stage==2 else result[16] if stage==3 else False
 dwell=dwell+1/a.hz if condition else 0
 tilt=math.acos(float(np.clip(-gravity[2],-1,1)));drift=float(np.linalg.norm(d.qpos[:2]-start));peak_tilt=max(tilt,peak_tilt);peak_drift=max(drift,peak_drift)
 kp=np.array([5,5,4]*4);kd=np.array([.25,.25,.15]*4);peak_pd=max(peak_pd,float(np.max(np.abs(kp*(result[:12]-sensor[:12])-kd*sensor[12:24]))))
 # Compare compiled kinematics against independent native world transforms.
 axis=d.geom_xmat[wheels].reshape(4,3,3)[:,2,2];z=d.geom_xpos[wheels,2]
 lower=z-.0505*np.sqrt(np.maximum(1-axis**2,0))-.01675*np.abs(axis)
 upper=z-.0455*np.sqrt(np.maximum(1-axis**2,0))-.01675*np.abs(axis)
 assert abs(result[17]-(lower[leg]-np.min(np.delete(upper,leg))))<1e-9, (tick,leg,result[17],lower[leg]-np.min(np.delete(upper,leg)))
 centers=d.geom_xpos[wheels];gap=min(np.linalg.norm(centers[i]-centers[j]) for i in range(4) for j in range(i))-2*np.hypot(.0505,.01675)
 assert abs(result[18]-gap)<1e-9
 collision=any(c.dist<0 and m.geom_bodyid[c.geom1]!=0 and m.geom_bodyid[c.geom2]!=0 for c in d.contact)
 if collision:reason='self collision';break
 if drift>.09 or tilt>.4 or d.qpos[2]<.08:reason='body instability';break
 if stage==6:reason='alignment fault';break
 if stage==5:reason='completed';break
 d.ctrl[:]=result[:12];mujoco.mj_step(m,d)
lib.wheel_destroy(b)
report=dict(passed=reason=='completed',reason=reason,command_hz=a.hz,actor_hz=50,rtneural_parity_max_abs_error=parity,events=events,peak_tilt_rad=peak_tilt,peak_drift_m=peak_drift,peak_estimated_pd_nm=peak_pd,final_hubs=d.qpos[9::3].tolist())
a.out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));raise SystemExit(0 if report['passed'] else 1)
