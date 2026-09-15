"""Render the actual candidate XML, without running a policy or physical rollout."""
from pathlib import Path
import json
import mujoco as mj
import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent


def main():
    revision = json.loads((OUT / 'module_spec.json').read_text())['placement_revision']
    model = mj.MjModel.from_xml_path(str(OUT / 'wheel_gap9_backpack_preview.xml'))
    data = mj.MjData(model)
    mj.mj_resetDataKeyframe(model, data, 0)
    mj.mj_forward(model, data)
    # Put the wheels on a flat floor for a static placement illustration.
    # Do not rewrite the source home keyframe or imply a balanced policy rollout.
    bottoms = []
    for leg in ['front_r', 'front_l', 'back_r', 'back_l']:
        gid = model.geom('leg_' + leg + '_3_wheel_collision').id
        nz = data.geom_xmat[gid].reshape(3, 3)[2, 2]
        extent = model.geom_size[gid, 0] * np.sqrt(max(0, 1 - nz * nz))
        extent += model.geom_size[gid, 1] * abs(nz)
        bottoms.append(data.geom_xpos[gid, 2] - extent)
    data.qpos[2] -= min(bottoms)
    mj.mj_forward(model, data)
    model.vis.global_.offwidth = 900
    model.vis.global_.offheight = 630
    opt = mj.MjvOption()
    opt.geomgroup[3] = 0
    opt.sitegroup[:] = 0
    # Source visual color is mostly gray. Keep the CAD shape and isolate the
    # added assembly with orange coloring, matching the review caption.
    renderer = mj.Renderer(model, height=630, width=900)
    cam = mj.MjvCamera()
    mj.mjv_defaultCamera(cam)
    cam.lookat[:] = [.02, 0, .17]
    canvas = Image.new('RGB', (1800, 1490), '#f2f5f7')
    draw = ImageDraw.Draw(canvas)
    title = ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf', 34)
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 25)
    small = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 23)
    draw.text((24, 14), f'BACKPACK PLACEMENT | REVISION {revision}', font=title, fill='#173044')
    draw.text((24, 57), 'Orange = supplied heating assembly | Wheel model retains the 9 mm hub gap', font=font, fill='#314d60')
    views = [('Oblique', 135, -22, .72), ('Side', 90, -3, .67),
             ('Top', 90, -89.5, .67), ('End view', 0, -10, .67)]
    for index, (name, az, el, dist) in enumerate(views):
        col, row = index % 2, index // 2
        x, y = col * 900, 110 + row * 670
        cam.azimuth, cam.elevation, cam.distance = az, el, dist
        renderer.update_scene(data, camera=cam, scene_option=opt)
        im = Image.fromarray(renderer.render())
        canvas.paste(im, (x, y + 32))
        draw.text((x + 20, y), name, font=font, fill='#173044')
        im.save(OUT / ('view_' + name.lower().replace(' ', '_') + '.png'))
    renderer.close()
    draw.text((24, 1457), 'Static XML views, not a rollout | 0.602 kg added | Position and orientation awaiting your review', font=small, fill='#314d60')
    canvas.save(OUT / 'placement-review.png')
    model.vis.global_.offwidth = 1400
    model.vis.global_.offheight = 1100
    with mj.Renderer(model, height=1100, width=1400) as detail_renderer:
        cam.lookat[:] = [0, 0, .13]
        for name, elevation, distance in [('side', 0, .60), ('top', -90, .53)]:
            cam.azimuth, cam.elevation, cam.distance = 90, elevation, distance
            detail_renderer.update_scene(data, camera=cam, scene_option=opt)
            detail_renderer.scene.flags[mj.mjtRndFlag.mjRND_SHADOW] = False
            Image.fromarray(detail_renderer.render()).save(OUT / f'revision{revision}_{name}.png')
    print(OUT / 'placement-review.png')


if __name__ == '__main__':
    main()
