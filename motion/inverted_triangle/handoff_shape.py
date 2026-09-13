"""Synthetic coupled under-compression; rigid geometry, never a material model.

Mesh coordinates are millimetres. The nominal CustomLegFoot tip points along
negative Y. Lower lateral supports occupy positive Y; axial mounting is Z.
A nonzero compression error extends the two lower supports, leaves the central
underside/mount unchanged, and shortens the tip. The resulting central gap and
tip height must be measured in the posed CAD, not equated to the input parameter.
"""
import numpy as np

def smooth(x):
    x=np.clip(x,0.,1.)
    return x*x*x*(10+x*(-15+6*x))

def deform(vertices, support_extension_mm=0., tip_shortening_mm=0., bend_deg=0.):
    v=np.asarray(vertices,dtype=float)
    if v.ndim!=2 or v.shape[1]!=3 or not np.isfinite(v).all():
        raise ValueError("Expected finite Nx3 CAD vertices in millimetres")
    values=np.array([support_extension_mm,tip_shortening_mm,bend_deg],float)
    if not np.isfinite(values).all() or not 0<=values[0]<=6 or not 0<=values[1]<=12 or abs(values[2])>5:
        raise ValueError("Prototype range: support 0..6 mm, shortening 0..12 mm, bend +/-5 deg")
    out=v.copy()
    if np.all(values==0):return out
    # Central attachment is protected as a rigid region, including its axial Z.
    radial=smooth((np.linalg.norm(v[:,:2],axis=1)-18.)/8.)
    lateral=smooth((np.abs(v[:,0])-8.)/12.)
    lower=smooth((v[:,1]-3.)/12.)
    out[:,1]+=support_extension_mm*radial*lateral*lower
    extent=max(float(-v[:,1].min()),40.0001)
    distal=smooth((-v[:,1]-40.)/(extent-40.))
    out[:,1]+=tip_shortening_mm*distal
    a=np.deg2rad(bend_deg)*distal
    x=out[:,0].copy();y=out[:,1]+40.
    out[:,0]=np.cos(a)*x-np.sin(a)*y
    out[:,1]=np.sin(a)*x+np.cos(a)*y-40.
    return out

def sample(seed, count=1):
    """Trial distributions, not measured compression tolerances.

    Common incomplete formation plus per-limb variation. The effective tip
    shortening range within each robot is bounded to the user's 10 mm spread.
    """
    rng=np.random.default_rng(seed)
    cases=[]
    for _ in range(count):
        common=rng.uniform(0.,4.)
        support=np.clip(common+rng.uniform(-1.,1.,4),0.,5.)
        shortening=np.clip(2*support+rng.uniform(-.5,.5,4),0.,10.)
        cases.append(dict(support_extension_mm=support.tolist(),
                          tip_shortening_mm=shortening.tolist(),
                          bend_deg=rng.uniform(-3.,3.,4).tolist()))
    return cases

def geometry_metrics(vertices, deformed):
    """Local-shape sanity metrics; floor clearance still needs FK/contact pose."""
    v=np.asarray(vertices);w=np.asarray(deformed)
    central=(np.abs(v[:,0])<=8)&(v[:,1]>3)
    lateral=(np.abs(v[:,0])>=20)&(v[:,1]>3)
    if not central.any() or not lateral.any():raise ValueError("CAD regions missing")
    return dict(central_attachment_max_change_mm=float(np.max(np.abs(
        w[np.linalg.norm(v[:,:2],axis=1)<=18]-v[np.linalg.norm(v[:,:2],axis=1)<=18]))),
        central_underside_y_mm=float(w[central,1].max()),
        lateral_support_y_mm=float(w[lateral,1].max()),
        local_central_recess_mm=float(w[lateral,1].max()-w[central,1].max()),
        tip_extent_mm=float(-w[:,1].min()),
        axial_change_mm=float(np.max(np.abs(w[:,2]-v[:,2]))))
