"""Replay an unchanged keyframe checkout and export its actual terminal state.

Run as a SCRIPT (not -m), with --alignment-root pointing at the selected source.
This adapter does not claim that keyframes are the installed robot's v5 policy.
It never starts ROS or accesses hardware. Failed alignment records are retained.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import mujoco as mj
import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(root, *args):
    try:
        return subprocess.check_output(['git', '-C', str(root), *args], text=True, stderr=subprocess.DEVNULL).strip()
    except subprocess.CalledProcessError:
        # Windows-created linked worktrees inspected through WSL.
        return subprocess.check_output(['git.exe', '-C', str(root), *args], text=True).strip()


def verify_frames(source, target):
    """Compare joint anchor/axis and limb frames with identical physical q.

    Wheel versus triangle hub phase is separately calibrated at the boundary.
    Joint kinematics must match; mesh shape and inertial changes are not certified.
    """
    names=['leg_'+leg+'_'+str(j) for leg in ('front_r','front_l','back_r','back_l') for j in (1,2,3)]
    records=[]
    for m in (source,target):
        if m.nq!=19 or m.nv!=18:raise ValueError('Expected floating base plus 12 joints')
        d=mj.MjData(m);d.qpos[:]=0;d.qpos[3]=1
        for n,q in zip(names,[1,0,0,-1,0,0]*2):
            j=m.joint(n).id;d.qpos[m.jnt_qposadr[j]]=q
        mj.mj_forward(m,d)
        records.append(np.concatenate([np.r_[d.xanchor[m.joint(n).id],d.xaxis[m.joint(n).id],
            d.xmat[m.jnt_bodyid[m.joint(n).id]]] for n in names]))
    error=float(np.max(abs(records[0]-records[1])))
    return dict(verified=error<1e-8,max_anchor_axis_frame_error=error,
                scope='12 named joint anchors, axes and body rotations at common neutral; not geometry/inertia validation')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--alignment-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--friction',type=float,default=0.)
    p.add_argument('--video',action='store_true')
    p.add_argument('--source-record',type=Path,help='Git provenance exported on host when WSL cannot read a Windows worktree')
    args=p.parse_args();root=args.alignment_root.resolve()
    args.output.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(root))
    from motion.keyframe_align import simulate
    cfg_path=root/'motion/keyframe_align/config.json'
    cfg=json.loads(cfg_path.read_text())
    target_path=Path(__file__).with_name('model.xml')
    source_path=simulate.model_config.MODEL_PATH
    if args.source_record:
        provenance=json.loads(args.source_record.read_text())
        required={'motion/keyframe_align/simulate.py','motion/keyframe_align/native.py',
            'motion/keyframe_align/api.cpp','motion/keyframe_align/config.json',
            'motion/keyframe_align/controller.hpp','training/wheel_align/model.xml',
            'training/wheel_align/configs.py','training/wheel_align/geometry.py'}
        if not required.issubset(provenance['files']):raise ValueError('Incomplete source record')
        for name,expected in provenance['files'].items():
            if digest(root/name)!=expected:raise ValueError('Source record mismatch: '+name)
        source_commit=provenance['source_commit'];source_dirty=provenance['source_dirty']
    else:
        source_commit=git(root,'rev-parse','HEAD');source_dirty=bool(git(root,'status','--porcelain','--untracked-files=no'))
        provenance=None
    frames=verify_frames(mj.MjModel.from_xml_path(str(source_path)),mj.MjModel.from_xml_path(str(target_path)))
    captured={}
    def collect(frame,event,arg):
        if event=='return' and frame.f_code is simulate.run.__code__:
            for name in ('model','data','home','o'):captured[name]=frame.f_locals[name]
    previous=sys.getprofile()
    try:
        sys.setprofile(collect)
        report,trace=simulate.run(cfg,seed=args.seed,friction=args.friction,
            video=str(args.output/'alignment.mp4') if args.video else None)
    finally:sys.setprofile(previous)
    d=captured['data'];m=captured['model'];o=captured['o']
    terminal=dict(schema_version=1,source_kind='replayed_deterministic_keyframe_alignment',
        source_commit=source_commit,source_dirty=source_dirty,external_source_record=provenance,
        model_sha256=digest(source_path),target_model_sha256=digest(target_path),
        config=cfg,config_sha256=digest(cfg_path),controller_binary_sha256=digest(os.environ['KEYFRAME_ALIGN_LIBRARY']),
        simulation_source_sha256=digest(simulate.__file__),exporter_sha256=digest(__file__),
        joint_names=[mj.mj_id2name(m,mj.mjtObj.mjOBJ_JOINT,j) for j in range(1,m.njnt)],
        completed_mask=report['completed_mask'],failed_mask=report['failed_mask'],alignment_passed=report['passed'],
        lowering_and_settling_finished=bool(int(o[12])==6 and all(x['returned_to_hold'] for x in report['legs'])),
        proximal_frame_verified=frames['verified'],frame_verification=frames,
        qpos=d.qpos.tolist(),qvel=d.qvel.tolist(),wheel_home=captured['home'].tolist(),
        last_alignment_output=o.tolist(),last_proximal_position_target=o[:8].tolist(),simulation_time_s=float(d.time),
        versions=dict(python=sys.version,mujoco=mj.__version__,numpy=np.__version__),
        source_command_semantics='8 proximal position targets followed by 4 hub control outputs; not 12 position commands',
        boundary_note='End of alignment only. Subsequent manual cooling is not simulated.',seed=args.seed)
    (args.output/'terminal.json').write_text(json.dumps(terminal,indent=2)+'\n')
    (args.output/'alignment_report.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savetxt(args.output/'alignment_trace.csv',trace,delimiter=',')
    print(json.dumps(dict(alignment_passed=report['passed'],frames=frames,terminal=str(args.output/'terminal.json'))))
    raise SystemExit(0 if report['passed'] and frames['verified'] else 1)


if __name__=='__main__':main()
