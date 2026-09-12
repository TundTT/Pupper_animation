"""Apply the user-selected 9 mm gap to the two active training models."""
from pathlib import Path
from decimal import Decimal
import re

REL='Stanford/training/pupper_v3_description/description/mujoco_xml/pupper_v3_complete.mjx.position.xml'
for branch in ['leg','wheel']:
    root=Path(f'C:/Users/tundt/Desktop/Pupper_gap_{branch}')
    path=root/REL
    text=path.read_bytes().decode()
    changed=[]
    def shift_body(match):
        block=match[0]
        def shift_tag(m):
            tag=m[0]
            def shift_attribute(a):
                vals=a[2].split()
                for i in ([2,5] if a[1]=='fromto' else [2]):
                    vals[i]=format(Decimal(vals[i])+Decimal('0.009'),'f')
                return a[1]+'="'+' '.join(vals)+'"'
            return re.sub(r'\b(pos|fromto)="([^"]+)"',shift_attribute,tag)
        block,n=re.subn(r'<(?:inertial|geom|site)\b[^>]*>',shift_tag,block)
        assert n==4,(branch,n)
        changed.append(match[1])
        return block
    text=re.sub(r'<body name="(leg_(?:front|back)_[rl]_3)".*?</body>',shift_body,text,flags=re.S)
    assert len(changed)==4,changed
    if branch=='leg':
        text=text.replace('CustomLegFoot mounted at the wheel\'s Z offset (pos 0 0 0.0136),',
            'CustomLegFoot uses a 22.6 mm local-Z mount offset: the original\n                             13.6 mm plus the user-selected 9 mm outward gap. Visual,\n                             capsule, foot site and CoM move together; hub joint stays fixed.\n                             Existing mass and inertia about the CoM are retained; no added mass.\n                             Mounted at pos 0 0 0.0226,')
        text=text.replace('fromto includes the 13.6 mm mesh mount offset.','fromto includes the 22.6 mm mesh mount offset.')
    else:
        text=text.replace('0.0136m by eye against the physical assembly to sit flush; folded into\n                        the numbers below (offsets are relative to this body\'s own origin, since',
            '0.0136m by eye against the physical assembly to sit flush (legacy baseline).\n                        The user-selected 9 mm outward gap makes the mesh offset 0.0226m.\n                        Visual, collision cylinder, foot site and CoM move together; the hub\n                        joint stays fixed. Existing mass and inertia about the CoM are retained;\n                        no added mass. These offsets are relative to this body\'s own origin, since')
        text=text.replace('the wheel is no longer a separate child body). Orientation',
                          'the wheel is no longer a separate child body. Orientation')
    path.write_bytes(text.encode())
    if branch=='leg':
        p=root/'mujoco_playground/workspace/tools/prepare_walk_model.py'
        t=p.read_bytes().decode();assert t.count('0.029884')==1
        t=t.replace('0.029884','0.038884')
        t=t.replace('# Mark the centers of the lower hemispheres; radius is taken from the actual capsule.',
            '# Mark the lower cap centers, including the 9 mm outward mounting gap.')
    else:
        p=root/'mujoco_playground/workspace/configs.py'
        t=p.read_bytes().decode();assert t.count('WHEEL_CENTER_LOCAL_Z = 0.03035')==1
        t=t.replace('WHEEL_CENTER_LOCAL_Z = 0.03035','WHEEL_CENTER_LOCAL_Z = 0.03935')
        t=t.replace('(0.0136 mount standoff + 0.01675 half-width).',
                    '(0.0136 original mount + 0.009 outward gap + 0.01675 half-width).')
    p.write_bytes(t.encode())
    print(branch,changed)
