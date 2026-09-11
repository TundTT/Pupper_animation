"""Generate a ZERO-output 83-input fixture for lifecycle tests, never deployment."""
import json
import sys
from pathlib import Path

data=json.loads(Path(sys.argv[1]).read_text())
data.update(in_shape=[None,83],single_observation_size=83,motion_contract_version=5,
    motion_contract_id='quadmorph-align-motion-v5',ctrl_dt=10/520,
    status='UNTRAINED ZERO-OUTPUT UNIT TEST FIXTURE')
offset=51
for name,size in [('phase',6),('progress',1),('motion_reference',8),('applied_position',8),('applied_velocity',8),('residual_gain',1)]:
    data['observation_layout'].append(dict(name=name,offset=offset,size=size));offset+=size
data['layers']=[dict(type='dense',activation='tanh',shape=[None,8],weights=[[[0.]*8 for _ in range(83)],[0.]*8])]
data['joint_lower_limits']=[-1.12,-.32,-2.,-2.41,-3.04,-2.]*2
data['joint_upper_limits']=[2.41,3.04,2.,1.12,.32,2.]*2
Path(sys.argv[2]).write_text(json.dumps(data))
