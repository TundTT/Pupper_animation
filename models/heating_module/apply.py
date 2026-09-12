"""Shared, fixed heating backpack approved in the September 12 placement review.

Uses the supplied explicit COM/inertia. No mass is inferred from the mesh.
The same local body pose applies to every base_link configuration.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re
import runpy
import shutil
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def numbers(values):
    return ' '.join(format(float(v), '.14g') for v in values)


def elements(floor_only=False):
    spec = json.loads((HERE / 'spec.json').read_text())
    mesh = ET.Element('mesh', name='heating_module_mesh', file='Heating_module.stl',
                      scale=numbers([spec['stl_to_m']] * 3))
    body = ET.Element('body', name='heating_module', pos=numbers(spec['body_pos_m']),
                      quat=numbers(spec['body_quat_wxyz']))
    factor = .45359237 * .0254 ** 2
    tensor = spec['inertia_lb_in2']
    ET.SubElement(body, 'inertial', mass=numbers([spec['mass_lb'] * .45359237]),
                  pos=numbers([v * .0254 for v in spec['com_in']]),
                  fullinertia=numbers([tensor[i][j] * factor for i, j in
                                      [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]]))
    ET.SubElement(body, 'geom', name='heating_module_visual', type='mesh',
                  mesh='heating_module_mesh', group='1', density='0',
                  contype='0', conaffinity='0', rgba='0.96 0.49 0.08 1')
    for proxy in spec['collision_boxes_cad_mm']:
        lo, hi = proxy['bounds']
        ET.SubElement(body, 'geom', name='heating_module_' + proxy['name'], type='box',
                      pos=numbers([(a + b) * .0005 for a, b in zip(lo, hi)]),
                      size=numbers([(b - a) * .0005 for a, b in zip(lo, hi)]),
                      density='0', group='3', contype='0' if floor_only else '1',
                      conaffinity='1', rgba='0.9 0.35 0.08 0.25', friction='0.8 0.02 0.01')
    return mesh, body


def install_mesh(meshdir):
    meshdir = Path(meshdir)
    meshdir.mkdir(parents=True, exist_ok=True)
    target = meshdir / 'Heating_module.stl'
    source = HERE / target.name
    if target.resolve() != source.resolve():
        if not target.exists() or digest(target) != digest(source):
            shutil.copy2(source, target)
    return target


def add_to_tree(root, meshdir, floor_only=False):
    """Idempotently add the approved assembly to an MJCF generator's tree."""
    gap = runpy.run_path(str(HERE / 'gap.py'))
    gap['apply_to_tree'](root)
    base = root.find(".//body[@name='base_link']")
    if base is None:
        raise ValueError('Expected base_link body')
    for existing in base.findall("body[@name='heating_module']"):
        base.remove(existing)
    for asset in root.findall('asset'):
        for existing in asset.findall("mesh[@name='heating_module_mesh']"):
            asset.remove(existing)
    mesh, body = elements(floor_only)
    asset = root.find('asset')
    if asset is None:
        asset = ET.SubElement(root, 'asset')
    asset.append(mesh)
    base.append(body)
    install_mesh(meshdir)
    return dict(spec_sha256=digest(HERE / 'spec.json'),
                mesh_sha256=digest(HERE / 'Heating_module.stl'),
                generator_sha256=digest(Path(__file__)), gap_generator_sha256=digest(HERE / 'gap.py'),
                additional_hub_gap_m=.009, placement_revision=3,
                placement_approved=True, collision_mode='floor-only proxies' if floor_only else 'box proxies',
                added_mass_kg=.60191707499)


def patch_file(path, floor_only=False):
    """Keep existing source text/comments intact; update only our marked blocks."""
    path = Path(path)
    gap = runpy.run_path(str(HERE / 'gap.py'))
    gap['patch_file'](path)
    text = path.read_bytes().decode('utf-8')
    text = re.sub(r'\r?\n[ \t]*<!-- HEATING_MODULE_BEGIN -->\r?\n.*?\r?\n[ \t]*<!-- HEATING_MODULE_END -->[ \t]*\r?\n', '', text, flags=re.S)
    root = ET.fromstring(re.sub(r'<!--.*?-->', '', text, flags=re.S))
    if root.find(".//body[@name='heating_module']") is not None:
        raise ValueError('Unmarked heating module already exists; use add_to_tree for generated XML')
    compiler = root.find('compiler')
    meshdir = path.parent / (compiler.get('meshdir', '.') if compiler is not None else '.')
    mesh, body = elements(floor_only)
    def block(element):
        ET.indent(element, space='  ')
        return '\n<!-- HEATING_MODULE_BEGIN -->\n' + ET.tostring(element, encoding='unicode') + '\n<!-- HEATING_MODULE_END -->\n'
    if '</asset>' not in text:
        raise ValueError('Expected inline asset section')
    text = re.sub(r'([ \t]*)</asset>', lambda m: block(mesh) + m[1] + '</asset>', text, count=1)
    text, count = re.subn(r'<body\b[^>]*\bname="base_link"[^>]*>',
                          lambda m: m[0] + block(body), text, count=1)
    if count != 1:
        raise ValueError('Expected one base_link')
    install_mesh(meshdir)
    path.write_bytes(text.encode('utf-8'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('xml', type=Path, nargs='+')
    parser.add_argument('--floor-only', action='store_true')
    args = parser.parse_args()
    for path in args.xml:
        patch_file(path, args.floor_only)
