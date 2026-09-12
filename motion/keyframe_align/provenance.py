from pathlib import Path
import hashlib,subprocess,os

def provenance():
 root=Path(__file__).resolve().parents[2]
 git=['git']; marker=root/'.git'
 if marker.is_file():
  directory=marker.read_text().strip().removeprefix('gitdir: ')
  if os.name!='nt' and len(directory)>2 and directory[1]==':':directory='/mnt/'+directory[0].lower()+directory[2:].replace('\\','/')
  git+=['--git-dir='+directory,'--work-tree='+str(root)]
 paths=list((root/'motion/keyframe_align').glob('*'))
 paths+=list((root/'ros2_ws/src/neural_controller/include/neural_controller').glob('wheel_align*.hpp'))
 paths+=list((root/'training/wheel_align').glob('*.xml'))+list((root/'training/wheel_align').glob('*.py'))
 paths+=list((root/'training/wheel_align/meshes').glob('*'))+list((root/'models/heating_module').glob('*'))
 files={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}
 return dict(source_commit=subprocess.check_output(git+['rev-parse','HEAD'],text=True).strip(),source_dirty=bool(subprocess.check_output(git+['status','--porcelain'],text=True).strip()),source_sha256=files,controller_binary_sha256=hashlib.sha256(Path(os.environ['KEYFRAME_ALIGN_LIBRARY']).read_bytes()).hexdigest())
