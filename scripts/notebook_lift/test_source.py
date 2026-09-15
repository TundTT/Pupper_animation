"""Source/launch structure tests without ROS nodes; lifecycle tests are separate."""
import importlib.util,json,runpy,sys,types,unittest,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
class Value:
 def __init__(self,*args,**kwargs):self.args=args;self.kwargs=kwargs
class Description:
 def __init__(self,entities):self.entities=entities
class SourceTests(unittest.TestCase):
 def test_launch_is_explicit_and_inactive(self):
  modules={}
  for name in ['launch','launch.substitutions','launch_ros','launch_ros.actions','launch_ros.parameter_descriptions','launch_ros.substitutions']:
   modules[name]=types.ModuleType(name)
  modules['launch'].LaunchDescription=Description
  for key in ['Command','FindExecutable','PathJoinSubstitution']:setattr(modules['launch.substitutions'],key,Value)
  modules['launch_ros.actions'].Node=Value;modules['launch_ros.parameter_descriptions'].ParameterFile=Value;modules['launch_ros.substitutions'].FindPackageShare=Value
  saved={k:sys.modules.get(k) for k in modules};sys.modules.update(modules)
  try:
   launch=runpy.run_path(str(ROOT/'ros2_ws/src/neural_controller/launch/notebook_lift_trial.launch.py'))['generate_launch_description']()
   align=runpy.run_path(str(ROOT/'ros2_ws/src/neural_controller/launch/notebook_lift_align_trial.launch.py'))['generate_launch_description']()
   combined=runpy.run_path(str(ROOT/'ros2_ws/src/neural_controller/launch/combined_motion.launch.py'))['generate_launch_description']()
  finally:
   for key,value in saved.items():
    if value is None:sys.modules.pop(key,None)
    else:sys.modules[key]=value
  spawners=[x.kwargs for x in launch.entities if x.kwargs['executable']=='spawner']
  self.assertEqual(len(spawners),3)
  motion=[x for x in spawners if x['arguments'][0].startswith('neural_controller')]
  self.assertEqual(len(motion),1);self.assertIn('--inactive',motion[0]['arguments']);self.assertEqual(motion[0]['arguments'][0],'neural_controller_notebook_lift')
  buttons=next(x.kwargs['parameters'][1] for x in launch.entities if x.kwargs['executable']=='estop_controller')
  self.assertTrue(buttons['calibration_required']);self.assertEqual(buttons['controller_names'],['neural_controller_notebook_lift'])
  self.assertEqual(buttons['wheel_align_hybrid_cycle_states'],['front_r','stand','front_l','stand','back_r','stand','back_l','stand'])
  self.assertEqual(buttons['switch_button_indices'],[-1]);self.assertEqual(buttons['leg_lift_button_index'],-1)
  alignment=next(x.kwargs['parameters'][1] for x in align.entities if x.kwargs['executable']=='estop_controller')
  self.assertEqual(alignment['wheel_align_hybrid_command_states'],['stand','front_l','front_r','back_r','back_l','rotate'])
  self.assertEqual(alignment['wheel_align_hybrid_cycle_states'],['front_r','rotate','stand','front_l','rotate','stand','back_r','rotate','stand','back_l','rotate','stand'])
  self.assertIn('--inactive',next(x.kwargs['arguments'] for x in align.entities if x.kwargs['executable']=='spawner' and x.kwargs['arguments'][0]=='neural_controller_notebook_lift'))
  combined_motion=[x.kwargs['arguments'] for x in combined.entities if x.kwargs['executable']=='spawner' and x.kwargs['arguments'][0].startswith('neural_controller')]
  self.assertEqual([x[0] for x in combined_motion],['neural_controller_triangle_roll','neural_controller_walk_v2','neural_controller_wheel','neural_controller_notebook_lift','neural_controller_wheel_to_walk_ready'])
  self.assertTrue(all('--inactive' in x for x in combined_motion))
  self.assertEqual(sum(x.kwargs['executable']=='motion_buttons.py' for x in combined.entities),1)
 def test_yaml_gains_and_abi(self):
  path=ROOT/'ros2_ws/src/neural_controller/launch/notebook_lift_config.yaml'
  block=path.read_text().split('\nneural_controller_notebook_lift:\n',1)[1];params={}
  for line in block.splitlines():
   if line.startswith('    '):key,value=line.strip().split(': ',1);params[key]=json.loads(value)
  self.assertEqual(params['kps'],[5,5,8]*4);self.assertEqual(params['kds'],[.25,.25,1]*4)
  self.assertTrue(params['calibration_required']);self.assertEqual(params['repeat_action'],10);self.assertEqual(params['observation_history'],4)
  self.assertEqual(params['action_types'],['position']*12)
  self.assertEqual(params['init_duration'],0);self.assertEqual(params['fade_in_duration'],0)
  align=(ROOT/'ros2_ws/src/neural_controller/launch/notebook_lift_align_config.yaml').read_text()
  self.assertEqual(align.replace('    alignment_mode: \"quadmorph-notebook-lift-align-v1\"\n',''),path.read_text())
 def test_startup_guard_and_help_do_not_start_ros(self):
  for name in ['prepare_notebook_lift.sh','run_notebook_lift.sh']:
   r=subprocess.run(['bash',str(ROOT/'scripts'/name),'--help'],capture_output=True,text=True);self.assertEqual(r.returncode,0);self.assertIn('Usage:',r.stdout)
  r=subprocess.run(['bash',str(ROOT/'scripts/run_notebook_lift.sh')],capture_output=True,text=True);self.assertEqual(r.returncode,2);self.assertIn('support',r.stderr)
 def test_existing_launch_defaults_do_not_reference_candidate(self):
  for name in ['config.yaml','launch.py','locomotion_trial.launch.py']:
   self.assertNotIn('notebook_lift',(ROOT/'ros2_ws/src/neural_controller/launch'/name).read_text())
if __name__=='__main__':unittest.main()
