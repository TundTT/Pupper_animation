import ast
from types import SimpleNamespace
from unittest.mock import Mock
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
        for index,mode in ((0,b.ROLL),(2,b.WALK),(1,b.WHEEL),(7,b.LIFT)):
            pad=[0]*13;pad[index]=1
            self.assertEqual(b.choose_button([0]*13,pad,False),mode)
            self.assertIsNone(b.choose_button(pad,pad,False))
            self.assertIsNone(b.choose_button([0]*13,pad,True))
            pad[10]=1;self.assertIsNone(b.choose_button([0]*13,pad,False))

    def test_ambiguous_and_short_inputs(self):
        self.assertIsNone(b.choose_button([0]*13,[],False))
        self.assertIsNone(b.choose_button([0]*13,[1,1]+[0]*11,False))

    def test_r2_only_and_previous_input_recovery(self):
        pad=[0]*13;pad[3]=1
        self.assertIsNone(b.choose_button([0]*13,pad,False))  # Square unbound
        pad[7]=1;pad[3]=0
        self.assertIsNone(b.choose_button([],pad,False))
        pad[0]=1
        self.assertIsNone(b.choose_button([0]*13,pad,False))  # X+R2 ambiguous

    def test_real_joy_callback_r2_cycle_and_edge_consumption(self):
        # Compile the actual nested class against a dummy Node; exercise its joy
        # callback with mocked ROS endpoints, without starting ROS or hardware.
        tree=ast.parse(Path(b.__file__).read_text())
        main=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='main')
        cls=next(x for x in main.body if isinstance(x,ast.ClassDef) and x.name=='Buttons')
        scope=dict(vars(b),Node=object,Int32=lambda **kw:SimpleNamespace(**kw))
        exec(compile(ast.Module(body=[cls],type_ignores=[]),b.__file__,'exec'),scope)
        handler=scope['Buttons'].__new__(scope['Buttons'])
        handler.previous=[1]*13;handler.busy=False;handler.active_mode=None
        handler.lift_index=0;handler.axes=[];handler.lift_pub=Mock()
        handler.lift_pub.get_subscription_count.return_value=1
        handler.request=Mock();handler.stop=Mock();handler.get_logger=Mock()
        handler.pending_mode=None;handler.pending_deadline=0.;handler.pending_ok_since=None
        handler.bridging=False;handler.pending_start=False
        def send(*pressed,axis=-1.):
            pad=[0]*13
            for i in pressed:pad[i]=1
            handler.joy(SimpleNamespace(buttons=pad,axes=[0,0,1,0,0,axis]))
        def settle():
            # Sticks read centered (axes 0,1,3 are always 0 in these fixtures);
            # fast-forward past the settle window instead of really sleeping.
            handler.watch()
            handler.pending_ok_since-=b.SWITCH_SETTLE_S+.01
            handler.watch()
        send(7);handler.request.assert_not_called()  # Held at process startup
        send();send(7)
        handler.request.assert_not_called()  # armed, not switched until sticks settle
        settle();handler.request.assert_called_once_with(b.LIFT)
        handler.active_mode=b.LIFT
        send(7);handler.lift_pub.publish.assert_not_called()  # Activation pull not reused
        for expected in [1]*12:
            send();send(7)
            self.assertEqual(handler.lift_pub.publish.call_args.args[0].data,expected)
            count=handler.lift_pub.publish.call_count
            send(7,axis=1.);self.assertEqual(handler.lift_pub.publish.call_count,count)
        count=handler.lift_pub.publish.call_count
        send();send(3);send();send(0,7)  # Square and ambiguous mode edge never advance
        self.assertEqual(handler.lift_pub.publish.call_count,count)
        send();handler.busy=True;send(7);handler.busy=False;send(7)
        self.assertEqual(handler.lift_pub.publish.call_count,count)  # No queue after busy
        send();send(7,10);handler.stop.assert_called_once()
        self.assertEqual(handler.lift_pub.publish.call_count,count)

    def test_switch_waits_for_sticks_to_settle_then_abandons_on_timeout(self):
        # A stray touch at the instant of pressing must not immediately switch
        # (no bleed-over) nor abort the request outright; it should only fire
        # once the sticks read centered continuously for SWITCH_SETTLE_S, and
        # give up (without ever calling request or stop) if they never do.
        tree=ast.parse(Path(b.__file__).read_text())
        main=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='main')
        cls=next(x for x in main.body if isinstance(x,ast.ClassDef) and x.name=='Buttons')
        scope=dict(vars(b),Node=object,Int32=lambda **kw:SimpleNamespace(**kw))
        exec(compile(ast.Module(body=[cls],type_ignores=[]),b.__file__,'exec'),scope)
        handler=scope['Buttons'].__new__(scope['Buttons'])
        handler.previous=[1]*13;handler.busy=False;handler.active_mode=None
        handler.axes=[];handler.request=Mock();handler.stop=Mock();handler.get_logger=Mock()
        handler.pending_mode=None;handler.pending_deadline=0.;handler.pending_ok_since=None
        handler.bridging=False;handler.pending_start=False
        def send(*pressed,stick=0.):
            pad=[0]*13
            for i in pressed:pad[i]=1
            handler.joy(SimpleNamespace(buttons=pad,axes=[stick,0,0,0]))
        send();send(2,stick=.3)  # Triangle pressed while stick deflected
        self.assertEqual(handler.pending_mode,b.WALK)
        handler.request.assert_not_called();handler.stop.assert_not_called()
        for _ in range(5):
            handler.watch()  # still off-center: never starts the settle clock
        self.assertIsNone(handler.pending_ok_since)
        handler.request.assert_not_called()
        send(2,stick=0.)  # stick released
        handler.watch()
        self.assertIsNotNone(handler.pending_ok_since)
        handler.pending_ok_since-=b.SWITCH_SETTLE_S+.01
        handler.watch()
        handler.request.assert_called_once_with(b.WALK)
        handler.stop.assert_not_called()
        # A second attempt whose sticks never settle is abandoned quietly, not estopped.
        handler.request.reset_mock()
        send();send(2,stick=.3)
        handler.pending_deadline=0.  # force the timeout to have already elapsed
        handler.watch()
        self.assertIsNone(handler.pending_mode)
        handler.request.assert_not_called();handler.stop.assert_not_called()

    def test_wheel_bridge_uses_actual_owner_after_dispatcher_restart(self):
        tree=ast.parse(Path(b.__file__).read_text())
        main=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='main')
        cls=next(x for x in main.body if isinstance(x,ast.ClassDef) and x.name=='Buttons')
        class Request:
            STRICT=2
            def __init__(self): self.timeout=SimpleNamespace(sec=0)
        with tempfile.TemporaryDirectory() as folder:
            scope=dict(vars(b),Node=object,load_current=lambda:self.cal,
                       load_roll_mapping=lambda mode:None,directory=lambda:Path(folder),
                       SwitchController=SimpleNamespace(Request=Request))
            exec(compile(ast.Module(body=[cls],type_ignores=[]),b.__file__,'exec'),scope)
            for target in (b.WALK,b.LIFT):
                handler=scope['Buttons'].__new__(scope['Buttons'])
                handler.epoch=0;handler.active_mode=None;handler.bridge_target=None
                handler.fresh=Mock();handler.get_logger=Mock();handler.switch_client=Mock()
                handler.joints=dict(self.q)
                for i in (0,3,6,9): handler.joints[self.names[i]]=(.65 if i in (0,6) else -.65,0)
                states=[SimpleNamespace(name=b.WHEEL,state='active',claimed_interfaces=['motor']),
                        SimpleNamespace(name=b.READY,state='inactive',claimed_interfaces=[]),
                        SimpleNamespace(name=target,state='inactive',claimed_interfaces=[])]
                future=Mock();future.result.return_value=SimpleNamespace(controller=states)
                handler.listed(future,target,0)
                handler.switch_client.call_async.assert_called_once()
                request=handler.switch_client.call_async.call_args.args[0]
                self.assertEqual(request.activate_controllers,[b.READY])
                self.assertEqual(request.deactivate_controllers,[b.WHEEL])
                self.assertEqual(handler.bridge_target,target)

    def test_lift_exit_requires_supported_lower_and_fresh_status(self):
        status=[0.]*24
        status[0]=4;status[3]=1
        b.validate_lift_exit(status,.1)
        for stage in (0,2,3,6):
            status[0]=stage
            with self.assertRaises(ValueError):b.validate_lift_exit(status,.1)
        status[0]=4
        with self.assertRaises(ValueError):b.validate_lift_exit(status,.3)
        status[3]=0
        with self.assertRaises(ValueError):b.validate_lift_exit(status,.1)

    def test_walking_full_turn_and_up_rejection(self):
        self.q[self.names[11]]=(1+2*math.pi,0)
        b.validate_entry(b.WALK,self.q,self.cal,self.mapping)
        self.q[self.names[2]]=(-1+math.pi,0)
        with self.assertRaises(ValueError):b.validate_entry(b.WALK,self.q,self.cal,self.mapping)

    def test_x_requires_up(self):
        with self.assertRaises(ValueError):b.validate_entry(b.ROLL,self.q,self.cal,self.mapping)
        self.q={n:(v,0) for n,v in zip(self.names,self.mapping['captured_q'])}
        b.validate_entry(b.ROLL,self.q,self.cal,self.mapping)

    def test_walking_can_interrupt_running_roll_near_standing(self):
        b.validate_entry(b.WALK,self.q,self.cal,self.mapping,[2,0,2,14,0]+[0]*11)
        self.q[self.names[2]]=(-1+math.pi,0)
        with self.assertRaises(ValueError):
            b.validate_entry(b.WALK,self.q,self.cal,self.mapping,[2,0,1,6,0]+[0]*11)

    def test_unready_and_failed_roll_cannot_switch(self):
        for state in (0,1,5):
            status=[state,0,0,0,0]+[0]*11
            with self.assertRaises(ValueError):b.validate_entry(b.WALK,self.q,self.cal,self.mapping,status)
        b.validate_entry(b.WALK,self.q,self.cal,self.mapping,[4,1,3,46,0]+[0]*11)

    def test_running_roll_cannot_switch_to_wheels(self):
        with self.assertRaises(ValueError):
            b.validate_entry(b.WHEEL,self.q,self.cal,self.mapping,[2,0,2,14,0]+[0]*11)

    def test_stale_mapping_rejected_for_x(self):
        self.mapping['calibration_id']='old'
        with self.assertRaises(ValueError):b.validate_entry(b.ROLL,self.q,self.cal,self.mapping)

if __name__=='__main__':unittest.main()
