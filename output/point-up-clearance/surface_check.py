"""Triangle-surface intersection checks independent of collision proxies."""
import time
import numpy as np
import trimesh
from geometry_review import Review

TIP = 0.06269415283203125
CUTOFF = .026

def shortened(mesh, length):
    result=mesh.copy()
    y=result.vertices[:,1]
    mask=y < -CUTOFF
    result.vertices[mask,1]=-CUTOFF+(y[mask]+CUTOFF)*(length-CUTOFF)/(TIP-CUTOFF)
    return result

def edge_hits(a,b):
    edges=a.vertices[a.edges_unique]
    lo,hi=b.bounds
    selected=np.all(edges.max(axis=1)>=lo-1e-8,axis=1)&np.all(edges.min(axis=1)<=hi+1e-8,axis=1)
    edges=edges[selected]
    if not len(edges):return np.empty((0,3))
    delta=edges[:,1]-edges[:,0];length=np.linalg.norm(delta,axis=1)
    valid=length>1e-9;edges=edges[valid];delta=delta[valid];length=length[valid]
    locations,ray_index,_=b.ray.intersects_location(edges[:,0],delta/length[:,None],multiple_hits=True)
    t=np.einsum('ij,ij->i',locations-edges[ray_index,0],delta[ray_index]/length[ray_index,None])
    return locations[(t>=-1e-8)&(t<=length[ray_index]+1e-8)]

def intersection(a,b):
    h=edge_hits(a,b)
    if len(h):return h
    return edge_hits(b,a)

def run():
    r=Review()
    for length,angle in [(TIP,180),(.06156,180),(.060,180),(.059,180),(.055,180),(.054,180),(.055,134),(.054,132),(.052,128),(.050,126),(.054,198),(.055,198),(.058,198)]:
        r.set_pose(spin=np.deg2rad(angle))
        a=r.world_mesh('leg_front_r_3',custom=shortened(r.meshes['CustomLegFoot'],length))
        b=r.world_mesh('leg_front_r_1')
        t=time.time();h=intersection(a,b)
        print('length_mm',length*1000,'spin_deg',angle,'intersections',len(h),'sec',round(time.time()-t,3),flush=True)

if __name__=='__main__':run()
