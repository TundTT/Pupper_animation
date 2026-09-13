"""Focused fake-input checks; does not import ROS or send motor commands."""
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"scripts"))
from capture_triangle_hold_reference import (CONTROLLER, FIELDS, JOINT_NAMES,
                                            check_owners, check_hold, make_map)


class HoldReferenceTest(unittest.TestCase):
    def setUp(self):
        self.q=[1.,0.,19.,-1.,0.,-18.,1.,0.,13.,-1.,0.,-12.]
        self.commands=[]
        for q in self.q:
            self.commands.extend([q,0.,0.,5.,.25])
        self.status=[0.]*28
        self.status[:8]=[1.,0.,-1.,0.,1.,0.,-1.,0.]
        for i,v in ((12,6),(16,-1),(23,1),(26,1/520),(27,.01)):
            self.status[i]=v
        self.owner=SimpleNamespace(name=CONTROLLER,type="neural_controller/KeyframeController",
            state="active",claimed_interfaces=[f"{j}/{f}" for j in JOINT_NAMES for f in FIELDS])

    def check(self,status=None,commands=None,command=0,receipts=None,q=None):
        return check_hold(self.status if status is None else status,
                          self.commands if commands is None else commands,command,
                          [1.,1.,1.] if receipts is None else receipts,1.1,self.q if q is None else q)

    def test_exact_owner_only(self):
        check_owners([self.owner])
        for owners in ([],[self.owner,self.owner]):
            with self.assertRaises(ValueError): check_owners(owners)
        for field,value in (("name","another"),("type","unknown"),("state","inactive"),
                            ("claimed_interfaces",self.owner.claimed_interfaces[:-1])):
            owner=deepcopy(self.owner);setattr(owner,field,value)
            with self.assertRaises(ValueError): check_owners([owner])

    def test_fresh_stationary_hold_required(self):
        self.assertEqual(self.check(),self.q)
        for field,value in ((12,0),(13,1),(14,1),(16,1),(23,0),(24,1),(25,1),(26,0),(27,.2)):
            status=self.status.copy();status[field]=value
            with self.assertRaises(ValueError): self.check(status=status)
        for command in (1,4,-1):
            with self.assertRaises(ValueError): self.check(command=command)
        for receipts in ([.1,1.,1.],[1.,.1,1.],[1.,1.,.1],[2.,1.,1.]):
            with self.assertRaises(ValueError): self.check(receipts=receipts)
        for index,value in ((1,.1),(2,.1),(3,0.),(4,2.),(0,1.2)):
            commands=self.commands.copy();commands[index]=value
            with self.assertRaises(ValueError): self.check(commands=commands)
        q=self.q.copy();q[2]+=.031
        with self.assertRaises(ValueError): self.check(q=q)

    def test_map_keeps_calibration_and_absolute_hub_winding(self):
        plan=dict(schema_version=3,joint_names=JOINT_NAMES,axial_gap_m=.009,
                  initial=[1.,0.,2.14,-1.,0.,-2.14]*2,plan_sha256="fixture")
        calibration=dict(calibration_id="unchanged",encoder_session_id="live",wheel_home=[-1,1,-1,1])
        before=deepcopy(calibration)
        record=make_map(plan,calibration,self.q,{"test":True})
        self.assertEqual(calibration,before)
        self.assertEqual(record["schema_version"],2)
        self.assertEqual(record["captured_q"],self.q)
        for i in range(12):
            expected=self.q[i]-plan["initial"][i] if i%3==2 else 0.
            self.assertEqual(record["model_to_encoder_offset"][i],expected)
        q=self.q.copy();q[3]=-.79
        with self.assertRaises(ValueError): make_map(plan,calibration,q,{})
        plan["schema_version"]=2
        with self.assertRaises(ValueError): make_map(plan,calibration,self.q,{})


if __name__=="__main__": unittest.main()
