"""Conservative rigid-floor coplanarity bound, distinct from compliant dynamics.

Enclose every foot's reachable lowest height over the complete joint cube using
sampled FK plus a second-derivative interpolation error. Independently enlarge
allowed orientations to a square enclosing the eight-degree normal cone.
"""
import json,itertools
from pathlib import Path
import numpy as np,mujoco as mj
from scipy.spatial import ConvexHull
from motion.inverted_triangle.core import Robot,HERE,provenance
from motion.inverted_triangle.walking_policy import WalkingPolicy

def run(case,cap=.1):
 r=Robot(formation=case['formation']);home=WalkingPolicy(np.zeros(12)).home
 r.d.qpos[:3]=0;r.d.qpos[3:7]=[1,0,0,0];r.d.qpos[7:]=home;mj.mj_forward(r.m,r.d)
 # All hinges have zero local joint position and unit axis. A point's second
 # derivative w.r.t. one hinge is bounded by the downstream chain's total
 # translation length plus the point's terminal-body radius, uniformly in q.
 assert np.max(abs(r.m.jnt_pos[1:]))==0
 radii=[];lows=[];ups=[];grid=np.linspace(-cap,cap,9)
 for i in range(4):
  g=r.point_geoms[i];R=r.d.xmat[r.bodies[i]].reshape(3,3);local=(r.points(i)-r.d.xpos[r.bodies[i]])@R
  radius=np.linalg.norm(local,axis=1).max()+sum(np.linalg.norm(r.m.body_pos[b]) for b in [r.bodies[i],r.m.body_parentid[r.bodies[i]]])
  assert radius<.25;radii.append(float(radius))
  chosen=np.argmin(r.points(i)[:,2]);cloud=[];point=[]
  for offsets in itertools.product(grid,repeat=3):
   r.d.qpos[7:]=home;r.d.qpos[7+3*i:10+3*i]+=offsets;mj.mj_forward(r.m,r.d)
   p=r.points(i);cloud.append(p);point.append(p[chosen])
  cloud=np.concatenate(cloud);point=np.array(point)
  lows.append(cloud[ConvexHull(cloud).vertices]);ups.append(point[ConvexHull(point).vertices])
 # Scalar height interpolation on a 3-D box: each coordinate contributes
 # max|d²height/dq_j²| * cell_width²/8. Triangle inequality adds three errors.
 joint_error=.25*3*(2*cap/8)**2/8
 s=np.sin(np.deg2rad(8));angles=np.linspace(-s,s,257);xy=np.array(list(itertools.product(angles,repeat=2)))
 normals=np.c_[xy,np.sqrt(1-np.sum(xy**2,axis=1))]
 lower=np.stack([(p@normals.T).min(axis=0)-joint_error for p in lows])
 upper=np.stack([(p@normals.T).max(axis=0)+joint_error for p in ups])
 gap=lower.max(axis=0)-upper.min(axis=0);k=gap.argmin()
 # Minimum/maximum of linear heights has Lipschitz constant <= maximum
 # point norm. Normal-map Jacobian norm <=1/sqrt(1-2 sin(8°)^2).
 # Uniform radius about the base origin, including the fixed hip offset.
 assert max(np.linalg.norm(r.m.body_pos[r.m.body_parentid[r.m.body_parentid[r.bodies[i]]]])+radii[i] for i in range(4))<.3
 max_radius=.3
 angular_error=2*max_radius*(2*s/256)/np.sqrt(2)/np.sqrt(1-2*s*s)
 result=dict(case=case,pose_cap_rad=cap,grid_min_interval_gap_m=float(gap[k]),joint_interpolation_error_m=joint_error,orientation_lipschitz_error_m=float(angular_error),conservative_rigid_coplanarity_gap_lower_bound_m=float(gap[k]-angular_error),worst_grid_normal=normals[k].tolist(),downstream_point_radius_bounds_m=radii,uniform_second_derivative_bound_m=.25,normal_square_contains_full_8_degree_cone=True,normal_grid=257,joint_grid_per_axis=9,roundoff_allowance_m=1e-8,rigid_four_contact_infeasible=bool(gap[k]-angular_error>1e-8),limitations='Numerical conservative FK bound with 10 nm roundoff allowance, not formal interval arithmetic. Proves separation of rigid-floor contact height intervals; does not independently prove infeasibility of soft-penetration MuJoCo equilibria or load stability. No dynamic gate is waived.')
 print(json.dumps(result),flush=True);return result
if __name__=='__main__':
 cases=json.loads((HERE/'formation_cases.json').read_text());results=[run(cases[i]) for i in [3,6]]
 Path('runs/roll_to_stand/geometric-bound-refined.json').write_text(json.dumps(dict(source_hashes=provenance(),results=results),indent=2))
