"""Rigid profile fit to operator-measured hub-to-top/base extents.

This is a provisional CAD fit, not a scan or a polymer deformation model.
Preserve mounting region, width and axial coordinates. Nominal inertia is retained.
"""
import numpy as np
from .handoff_shape import smooth

NOMINAL_TOP_MM=62.69415283203125
NOMINAL_BOTTOM_MM=25.447282791137695
PROTECTED_Y_MM=18.


def deform(vertices,top_mm,bottom_mm,bend_deg=0.):
    v=np.asarray(vertices,dtype=float)
    if v.ndim!=2 or v.shape[1]!=3 or not np.isfinite(v).all():raise ValueError('Finite Nx3 vertices required')
    if not np.isfinite([top_mm,bottom_mm,bend_deg]).all() or not 50<=top_mm<=70 or not 25<=bottom_mm<=45:
        raise ValueError('Provisional extent fit supports top 50..70 mm and bottom 25..45 mm')
    if bend_deg!=0:raise ValueError('Measured profile has no measured angle fit; nonzero bend unsupported')
    w=v.copy()
    upper=smooth((-v[:,1]-PROTECTED_Y_MM)/(NOMINAL_TOP_MM-PROTECTED_Y_MM))
    lower=smooth((v[:,1]-PROTECTED_Y_MM)/(NOMINAL_BOTTOM_MM-PROTECTED_Y_MM))
    w[:,1]-=(top_mm-NOMINAL_TOP_MM)*upper
    w[:,1]+=(bottom_mm-NOMINAL_BOTTOM_MM)*lower
    return w
