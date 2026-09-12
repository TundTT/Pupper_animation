"""Keep the user-approved additional hub-axis gap at 9 mm, without moving joints.

The baseline is the original mesh mounting position, not a 9 mm total standoff.
AdaptiveWheel/CustomLegFoot: 13.6 mm original + 9 mm = 22.6 mm.
Legacy ROS limbs keep their existing shape and receive the same axial translation.
No spacer mass has been supplied; centroidal inertias and masses stay unchanged.
"""
from decimal import Decimal
from pathlib import Path
import re
import xml.etree.ElementTree as ET

NAME = 'quadmorph_additional_hub_gap_m'
GAP = Decimal('0.009')
LIMB = r'leg_(?:front|back)_[rl]_3'


def limb_bodies(root):
    bodies = [b for b in root.iter('body') if re.fullmatch(LIMB, b.get('name', ''))]
    if len(bodies) != 4 or any(b.findall('body') for b in bodies):
        raise ValueError('Expected four terminal hub bodies')
    return bodies


def current_gap(root):
    marker = root.find(f"./custom/numeric[@name='{NAME}']")
    if marker is not None:
        return Decimal(marker.get('data'))
    values = []
    for body in limb_bodies(root):
        mesh = next((g for g in body.findall('geom')
                     if g.get('mesh') in ('AdaptiveWheel', 'CustomLegFoot')), None)
        if mesh is not None:
            values.append(Decimal(mesh.get('pos').split()[2]) - Decimal('0.0136'))
        else:
            # The legacy ROS geometry has no QuadMorph mesh-origin convention.
            # Recognize its unchanged source COM rather than blindly adding twice.
            z = Decimal(body.find('inertial').get('pos').split()[2])
            if z != Decimal('0.01833'):
                raise ValueError(f'Unrecognized unmarked legacy geometry: {body.get("name")}, COM z={z}')
            values.append(Decimal(0))
    if len(set(values)) != 1 or values[0] not in (Decimal(0), GAP):
        raise ValueError(f'Inconsistent/unrecognized hub mounting offsets: {values}')
    return values[0]


def shifted(value, delta, fromto=False):
    v = value.split()
    for i in ([2, 5] if fromto else [2]):
        v[i] = format(Decimal(v[i]) + delta, 'f')
    return ' '.join(v)


def apply_to_tree(root):
    delta = GAP - current_gap(root)
    if delta:
        for body in limb_bodies(root):
            for element in body:
                if element.tag not in ('inertial', 'geom', 'site'):
                    continue
                key = 'fromto' if 'fromto' in element.attrib else 'pos'
                element.set(key, shifted(element.get(key, '0 0 0'), delta, key == 'fromto'))
    marker = root.find(f"./custom/numeric[@name='{NAME}']")
    if marker is None:
        custom = root.find('custom')
        if custom is None:
            custom = ET.SubElement(root, 'custom')
        marker = ET.SubElement(custom, 'numeric', name=NAME)
    marker.set('data', str(GAP))
    return float(delta)


def patch_file(path):
    path = Path(path)
    text = path.read_bytes().decode()
    root = ET.fromstring(re.sub(r'<!--.*?-->', '', text, flags=re.S))
    delta = GAP - current_gap(root)
    limb_bodies(root)
    if delta:
        def edit_body(match):
            def edit_tag(tag_match):
                tag = tag_match[0]
                key = 'fromto' if re.search(r'\bfromto=', tag) else 'pos'
                pattern = rf'\b{key}="([^"]+)"'
                if re.search(pattern, tag):
                    return re.sub(pattern, lambda m: f'{key}="{shifted(m[1], delta, key == "fromto")}"', tag)
                return tag.replace('/>', f' pos="0 0 {delta}" />')
            return re.sub(r'<(?:inertial|geom|site)\b[^>]*>', edit_tag, match[0])
        text, count = re.subn(rf'<body\b[^>]*\bname="{LIMB}".*?</body>', edit_body, text, flags=re.S)
        assert count == 4
    marker = f'<numeric name="{NAME}" data="{GAP}" />'
    if root.find(f"./custom/numeric[@name='{NAME}']") is not None:
        text = re.sub(rf'<numeric\b[^>]*name="{NAME}"[^>]*/>', marker, text)
    elif '</custom>' in text:
        text = text.replace('</custom>', marker + '\n    </custom>', 1)
    else:
        text = text.replace('</mujoco>', '    <custom>\n        ' + marker + '\n    </custom>\n</mujoco>')
    path.write_bytes(text.encode())
    return float(delta)
