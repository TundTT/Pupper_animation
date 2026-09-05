"""Regenerate the ring probe data from the unchanged STL; review after mesh changes."""
import hashlib
import json
from pathlib import Path
import mujoco as mj
import numpy as np
from scipy.spatial import ConvexHull
from workspace.walk_geometry import MODEL_PATH


def main():
    m = mj.MjModel.from_xml_path(str(MODEL_PATH))
    mid = m.mesh('CustomLegFoot').id
    vertices = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid]+m.mesh_vertnum[mid]]
    rotation = np.zeros(9)
    mj.mju_quat2Mat(rotation,m.mesh_quat[mid])
    raw = vertices @ rotation.reshape(3,3).T + m.mesh_pos[mid]
    outline = raw[ConvexHull(raw[:,:2]).vertices,:2]
    closed = np.vstack([outline,outline[:1]])
    length = np.r_[0,np.cumsum(np.linalg.norm(np.diff(closed,axis=0),axis=1))]
    distance = np.linspace(0,length[-1],48,endpoint=False)
    xy = np.stack([np.interp(distance,length,closed[:,j]) for j in range(2)],axis=1)
    # These are the measured rim edges, not the hub's full depth bounds.
    points = np.concatenate([np.column_stack([xy,np.full(48,z)]) for z in [.00484706,.02884707]])
    stl = MODEL_PATH.parent.parent/'meshes/stl/CustomLegFoot.stl'
    data = dict(description='48 STL outer XY perimeter samples on each axial rim edge. Clearance proxy only. Bottom arc: authored Y < -54 mm.',
                mesh_sha256=hashlib.sha256(stl.read_bytes()).hexdigest(),points=np.round(points,9).tolist(),bottom_mask=(points[:,1]<-.054).tolist())
    Path(__file__).resolve().parents[1].joinpath('ring_outline.json').write_text(json.dumps(data,indent=2)+'\n')

if __name__=='__main__':main()
