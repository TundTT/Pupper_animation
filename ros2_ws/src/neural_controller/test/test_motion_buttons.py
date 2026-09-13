import importlib.util
import math
from pathlib import Path
import unittest
import os
import subprocess
import sys
import tempfile

spec=importlib.util.spec_from_file_location('buttons',Path(__file__).parents[1]/'scripts/motion_buttons.py')
b=importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)

class ButtonsTest(unittest.TestCase):
    @unittest.skipUnless(os.name == 'posix', 'Pi/Linux process lock')
    def test_second_process_cannot_create_another_dispatcher(self):
        with tempfile.TemporaryDirectory() as folder:
            owner=b.acquire_instance(Path(folder))
            code=(f"import runpy; from pathlib import Path; "
                  f"module=runpy.run_path({str(Path(b.__file__).resolve())!r}); "
                  f"lock=module['acquire_instance'](Path({folder!r}))")
            duplicate=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
            self.assertNotEqual(duplicate.returncode,0)
            self.assertIn('already running',duplicate.stderr)
            owner.close()
            replacement=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
            self.assertEqual(replacement.returncode,0,replacement.stderr)

    def setUp(self):
        self.names=[f'leg_{leg}_{j}' for leg in ('front_r','front_l','back_r','back_l') for j in (1,2,3)]
        self.home=[1,0,-1,-1,0,1]*2
        self.cal={'calibration_id':'live','joint_names':self.names,'wheel_home':self.home[2::3]}
        self.q={n:(v,0) for n,v in zip(self.names,self.home)}
        up=self.home.copy()
        for i in (2,5,8,11):up[i]+= math.pi if i in (2,8) else -math.pi
        self.mapping={'calibration_id':'live','captured_q':up,'model_to_encoder_offset':[0]*12}

    def test_buttons_and_edges(self):
        for index,mode in ((0,b.ROLL),(2,b.WALK),(1,b.WHEEL)):
            pad=[0]*13;pad[index]=1
            self.assertEqual(b.choose_button([0]*13,pad,False),mode)
            self.assertIsNone(b.choose_button(pad,pad,False))
            self.assertIsNone(b.choose_button([0]*13,pad,True))
            pad[10]=1;self.assertIsNone(b.choose_button([0]*13,pad,False))

    def test_ambiguous_and_short_inputs(self):
        self.assertIsNone(b.choose_button([0]*13,[],False))
        self.assertIsNone(b.choose_button([0]*13,[1,1]+[0]*11,False))

    def test_walking_full_turn_and_up_rejection(self):
        self.q[self.names[11]]=(1+2*math.pi,0)
        b.validate_entry(b.WALK,self.q,self.cal,self.mapping)
        self.q[self.names[2]]=(-1+math.pi,0)
        with self.assertRaises(ValueError):b.validate_entry(b.WALK,self.q,self.cal,self.mapping)

    def test_x_requires_up(self):
        with self.assertRaises(ValueError):b.validate_entry(b.ROLL,self.q,self.cal,self.mapping)
        self.q={n:(v,0) for n,v in zip(self.names,self.mapping['captured_q'])}
        b.validate_entry(b.ROLL,self.q,self.cal,self.mapping)

    def test_running_and_failed_roll_cannot_switch(self):
        for state in (0,1,2,5):
            status=[state,0,0,0,0]+[0]*11
            with self.assertRaises(ValueError):b.validate_entry(b.WALK,self.q,self.cal,self.mapping,status)
        b.validate_entry(b.WALK,self.q,self.cal,self.mapping,[4,1,3,46,0]+[0]*11)

    def test_stale_mapping_rejected_for_x(self):
        self.mapping['calibration_id']='old'
        with self.assertRaises(ValueError):b.validate_entry(b.ROLL,self.q,self.cal,self.mapping)

if __name__=='__main__':unittest.main()
