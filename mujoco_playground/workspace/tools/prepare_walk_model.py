"""Rebase XML reference frames to a stable stance without changing joint kinematics.

Run from mujoco_playground: python -m workspace.tools.prepare_walk_model
Actuator controls become position offsets from this stance (zero holds home).
"""
from pathlib import Path
import re
import mujoco as mj
import numpy as np

MODEL = Path(__file__).resolve().parents[3] / 'Stanford/training/pupper_v3_description/description/mujoco_xml/pupper_v3_complete.mjx.position.xml'
LEGS = ('front_r', 'front_l', 'back_r', 'back_l')
TARGET = np.array([1., 0., -1., -1., 0., 1.] * 2)

def attr(tag, key, value):
    pattern = rf'\b{key}="[^"]*"'
    return re.sub(pattern, f'{key}="{value}"', tag) if re.search(pattern, tag) else tag[:-2].rstrip() + f' {key}="{value}" />' if tag.endswith('/>') else tag[:-1] + f' {key}="{value}">'

def numbers(values):
    return ' '.join(f'{v:.12g}' for v in values)

def main():
    text = MODEL.read_text()
    original = mj.MjModel.from_xml_path(str(MODEL))
    for k, leg in enumerate(LEGS):
        for j in range(3):
            name = f'leg_{leg}_{j+1}'
            joint = original.joint(name)
            b = original.body(name).id
            reference = TARGET[3*k+j]
            delta = reference - original.qpos0[joint.qposadr[0]]
            rotation = np.array([np.cos(delta/2), 0., 0., np.sin(delta/2)])
            quat = np.empty(4)
            mj.mju_mulQuat(quat, original.body_quat[b], rotation)
            text = re.sub(rf'<body\b[^>]*\bname="{name}"[^>]*>', lambda m: attr(m[0], 'quat', numbers(quat)), text)
            text = re.sub(rf'<joint\b[^>]*\bname="{name}"[^>]*>', lambda m: attr(m[0], 'ref', numbers([reference])), text)
            # Position actuator length is the joint angle, so bias adds the home target.
            text = re.sub(rf'<general\b[^>]*\bname="{name}"[^>]*/>', lambda m: m[0][:-2].rstrip() + f' biasprm="{5*reference:g} -5 -0.25" />' if 'biasprm=' not in m[0] else re.sub(r'biasprm="[^"]*"', f'biasprm="{5*reference:g} -5 -0.25"', m[0]), text)
    text = re.sub(r'<body name="base_link"[^>]*>', lambda m: attr(attr(m[0], 'pos', '0 0 0.1448'), 'quat', '1 0 0 0'), text)
    # Mark the centers of the lower hemispheres; radius is taken from the actual capsule.
    for leg in LEGS:
        x, y = (.00063, -.048) if leg.endswith('r') else (-.00063, .048)
        text = re.sub(rf'<site name="leg_{leg}_3_foot_site"[^>]*/>', f'<site name="leg_{leg}_3_foot_site" pos="{x} {y} 0.029884" />', text)
    header = ('    <!-- Walking reference stance: body frames and joint refs are rebased together,\n'
              '         preserving the physical pose for every absolute joint angle. Zero ctrl\n'
              '         holds the home targets; ctrl is a joint-angle OFFSET from home in radians.\n'
              '         The home keyframe also includes the settled base pose and joint droop. -->\n')
    if 'Walking reference stance:' not in text:
        text = text.replace('    <keyframe>', header + '    <keyframe>')
    # More than one solver iteration is needed for repeatable foot contacts in training.
    text = text.replace('iterations="1" ls_iterations="5"', 'iterations="8" ls_iterations="8"')
    compiled_text = text.replace('../meshes/stl/', (MODEL.parent.parent / 'meshes/stl').as_posix() + '/')
    m = mj.MjModel.from_xml_string(compiled_text)
    # Confirm rebasing did not change FK for the same absolute coordinates.
    a, b = mj.MjData(original), mj.MjData(m)
    rng = np.random.default_rng(4)
    for _ in range(8):
        q = original.qpos0.copy(); q[2] = .3; q[7:] = TARGET + rng.uniform(-.1, .1, 12)
        a.qpos[:] = q; b.qpos[:] = q
        mj.mj_forward(original, a); mj.mj_forward(m, b)
        np.testing.assert_allclose(a.xpos, b.xpos, atol=1e-10)
        np.testing.assert_allclose(a.xmat, b.xmat, atol=1e-10)
    d = mj.MjData(m)
    for _ in range(round(12/m.opt.timestep)):
        mj.mj_step(m,d)
    tilt = np.degrees(np.arccos(np.clip(d.xmat[1].reshape(3,3)[2,2],-1,1)))
    assert tilt < 2 and d.qpos[2] > .12 and np.linalg.norm(d.qvel) < .01, (tilt,d.qpos,np.linalg.norm(d.qvel))
    q = d.qpos.copy(); q[:2] = 0
    text = re.sub(r'<key name="home".*?/>', f'<key name="home"\n            qpos="{numbers(q)}"\n            ctrl="{numbers(np.zeros(12))}" />', text, flags=re.S)
    MODEL.write_text(text)
    print(f'Home settled: z={q[2]:.6f} m; tilt={tilt:.4f} deg; max speed={np.max(abs(d.qvel)):.6f}')

if __name__ == '__main__':
    main()
