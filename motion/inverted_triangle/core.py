from pathlib import Path
import hashlib, json
import importlib.metadata, platform
import mujoco as mj
import numpy as np

HERE=Path(__file__).resolve().parent
LEGS=('front_r','front_l','back_r','back_l')
HUB=np.array([2,5,8,11]);PROX=np.array([i for i in range(12) if i not in HUB])
KP=np.tile([5.,5.,4.],4);KD=np.tile([.25,.25,.15],4)
SPEED=np.tile([.45,.65,.5],4);ACCEL=np.tile([2.,2.,1.2],4)

def duration(start,end,minimum):
    delta=np.abs(np.asarray(end)-np.asarray(start))
    return max(float(minimum),float(np.max(1.875*delta/SPEED)),float(np.sqrt(np.max((10/np.sqrt(3))*delta/ACCEL))))

def provenance():
    files=list(HERE.glob('*.py'))+[HERE/'model.xml',HERE/'source_manifest.json',HERE/'requirements.txt']
    if (HERE/'requirements.lock.txt').exists():files.append(HERE/'requirements.lock.txt')
    return {str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}

def versions():
    return dict(python=platform.python_version(),platform=platform.platform(),packages={name:importlib.metadata.version(name) for name in ['numpy','scipy','mujoco','trimesh','python-fcl','wandb']})

def smooth(u):
    u=np.clip(u,0.,1.)
    return u*u*u*(10+u*(-15+6*u))

def verify_sources():
    manifest=json.loads((HERE/'source_manifest.json').read_text())
    files={HERE/'model.xml':manifest['model_sha256'],HERE/'source.xml':manifest['source_xml_sha256']}
    files.update({HERE/'assets'/name:digest for name,digest in manifest['asset_sha256'].items()})
    for path,digest in files.items():
        if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise ValueError(f'Source hash mismatch: {path}; rebuild/review manifest, do not bypass')
    return manifest

class Robot:
    def __init__(self,friction=.8,dynamics=None,formation=None):
        if not np.isfinite(friction) or friction<=0:raise ValueError('Friction must be positive')
        self.friction=float(friction)
        self.dynamics=dict(mass_scale=1.,base_com_offset_m=[0.,0.,0.],kp_scale=1.,kd_scale=1.,torque_limit_Nm=3.,delay_steps=0)
        if dynamics:
            if set(dynamics)-set(self.dynamics):raise ValueError('Unknown dynamics parameter')
            self.dynamics.update(dynamics)
        for key in ['mass_scale','kp_scale','kd_scale','torque_limit_Nm']:
            if not np.isfinite(self.dynamics[key]) or self.dynamics[key]<=0:raise ValueError('Invalid dynamics factor')
        if self.dynamics['torque_limit_Nm']>3:raise ValueError('Cannot increase available torque beyond 3 Nm')
        delay=self.dynamics['delay_steps']
        if isinstance(delay,bool) or not isinstance(delay,int) or delay<0:raise ValueError('Delay must be nonnegative integer steps')
        offset=np.asarray(self.dynamics['base_com_offset_m'],dtype=float)
        if offset.shape!=(3,) or not np.isfinite(offset).all():raise ValueError('Invalid COM offset')
        self.command_history=[]
        self.manifest=verify_sources()
        self.model_xml=(HERE/'model.xml').read_bytes();self.model_assets=None
        if formation is None:
            self.m=mj.MjModel.from_xml_path(str(HERE/'model.xml'))
        else:
            from .formation import build
            self.model_xml,self.model_assets,self.manifest=build(formation,self.manifest)
            self.m=mj.MjModel.from_xml_string(self.model_xml.decode(),assets=self.model_assets)
        self.d=mj.MjData(self.m)
        self.geoms=np.array([self.m.geom(f'{leg}_floor_contact').id for leg in LEGS])
        self.bodies=np.array([self.m.body('leg_'+leg+'_3').id for leg in LEGS])
        self.floor=self.m.geom('floor').id;self.base=self.m.body('base_link').id
        self.m.body_mass[1:]*=self.dynamics['mass_scale']
        self.m.body_inertia[1:]*=self.dynamics['mass_scale']
        self.m.body_ipos[self.base]+=offset
        self.m.actuator_ctrlrange[:]=[-self.dynamics['torque_limit_Nm'],self.dynamics['torque_limit_Nm']]
        mj.mj_setConst(self.m,self.d)
        self.vertices=[];self.tip_masks=[]
        for leg,g in enumerate(self.geoms):
            mid=self.m.geom_dataid[g]
            self.vertices.append(self.m.mesh_vert[self.m.mesh_vertadr[mid]:self.m.mesh_vertadr[mid]+self.m.mesh_vertnum[mid]].copy())
            R=np.empty(9);mj.mju_quat2Mat(R,self.m.geom_quat[g])
            body_vertices=self.vertices[-1]@R.reshape(3,3).T+self.m.geom_pos[g]
            self.tip_masks.append(body_vertices[:,1]*(1 if leg%2 else -1)>.050)
        # MuJoCo combines equal-priority friction using the larger value.
        # Set both surfaces, otherwise the default floor silently defeats sweeps.
        self.m.geom_friction[:,0]=friction
        self.reset()
    def reset(self):
        mj.mj_resetDataKeyframe(self.m,self.d,0);mj.mj_forward(self.m,self.d)
        self.initial=self.d.qpos.copy()
        self.command_history=[self.initial[7:].copy() for _ in range(self.dynamics['delay_steps'])]
        return self.initial.copy()
    def points(self,leg):
        g=self.geoms[leg]
        return self.vertices[leg]@self.d.geom_xmat[g].reshape(3,3).T+self.d.geom_xpos[g]
    def snapshot(self,command):
        spec=mj.mjtState.mjSTATE_INTEGRATION
        state=np.empty(mj.mj_stateSize(self.m,spec));mj.mj_getState(self.m,self.d,state,spec)
        return dict(model_sha256=self.manifest['model_sha256'],generated_asset_sha256=self.manifest.get('generated_asset_sha256'),friction=self.friction,state_spec=int(spec),state=state.tolist(),command=np.asarray(command).tolist(),dynamics=self.dynamics,command_history=[q.tolist() for q in self.command_history])
    def restore(self,snapshot):
        if snapshot['model_sha256']!=self.manifest['model_sha256'] or snapshot['friction']!=self.friction:
            raise ValueError('Continuation model/friction mismatch')
        if snapshot.get('generated_asset_sha256')!=self.manifest.get('generated_asset_sha256'):
            raise ValueError('Continuation generated geometry mismatch')
        expected=dict(mass_scale=1.,base_com_offset_m=[0.,0.,0.],kp_scale=1.,kd_scale=1.,torque_limit_Nm=3.,delay_steps=0)
        if snapshot.get('dynamics',expected)!=self.dynamics:raise ValueError('Continuation dynamics mismatch')
        history=np.asarray(snapshot.get('command_history',[]),dtype=float)
        if history.size==0:history=history.reshape(0,12)
        if history.shape!=(self.dynamics['delay_steps'],12) or not np.isfinite(history).all():raise ValueError('Invalid delayed command history')
        self.command_history=[q.copy() for q in history]
        spec=mj.mjtState.mjSTATE_INTEGRATION
        state=np.asarray(snapshot['state'],dtype=float)
        if snapshot['state_spec']!=int(spec) or state.shape!=(mj.mj_stateSize(self.m,spec),) or not np.all(np.isfinite(state)):
            raise ValueError('Invalid continuation state')
        mj.mj_setState(self.m,self.d,state,spec);mj.mj_forward(self.m,self.d)
        return np.asarray(snapshot['command'],dtype=float)
    def bottoms(self):return np.array([self.points(i)[:,2].min() for i in range(4)])
    def tip_bottoms(self):return np.array([self.points(i)[self.tip_masks[i],2].min() for i in range(4)])
    def contacts_precise(self):
        forces=np.zeros(4)
        for k in range(self.d.ncon):
            c=self.d.contact[k]
            if self.floor not in (c.geom1,c.geom2):continue
            g=c.geom2 if c.geom1==self.floor else c.geom1
            ids=np.flatnonzero(self.geoms==g)
            if ids.size:
                force=np.zeros(6);mj.mj_contactForce(self.m,self.d,k,force);forces[ids[0]]+=max(0,force[0])
        return forces
    def unintended_floor_force(self):
        total=0.
        for k in range(self.d.ncon):
            c=self.d.contact[k]
            if self.floor not in (c.geom1,c.geom2):continue
            g=c.geom2 if c.geom1==self.floor else c.geom1
            if g in self.geoms:continue
            force=np.zeros(6);mj.mj_contactForce(self.m,self.d,k,force);total+=max(0,force[0])
        return float(total)
    def tilt(self):return float(np.arccos(np.clip(self.d.xmat[self.base].reshape(3,3)[2,2],-1,1)))
    def tick(self,target,kp=KP,kd=KD,feedforward=None):
        ff=np.zeros(12) if feedforward is None else np.asarray(feedforward)
        target=np.asarray(target)
        if self.dynamics['delay_steps']:
            self.command_history.append(target.copy());target=self.command_history.pop(0)
        tau=self.dynamics['kp_scale']*kp*(target-self.d.qpos[7:])-self.dynamics['kd_scale']*kd*self.d.qvel[6:]+ff
        limit=self.dynamics['torque_limit_Nm'];self.d.ctrl[:]=np.clip(tau,-limit,limit);mj.mj_step(self.m,self.d)
        return tau

if __name__=='__main__':
    r=Robot();target=r.initial[7:].copy()
    for _ in range(520*8):r.tick(target)
    print(json.dumps(dict(q=r.d.qpos.tolist(),tilt_deg=np.rad2deg(r.tilt()),bottom_mm=(r.bottoms()*1000).tolist(),normal_force_N=r.contacts_precise().tolist(),max_speed=float(np.abs(r.d.qvel).max())),indent=2))
