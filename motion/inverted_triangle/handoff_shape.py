"""Synthetic coupled under-compression; rigid geometry, never a material model.

Mesh coordinates are millimetres. The nominal CustomLegFoot tip points along
negative Y. Lower lateral supports occupy positive Y; axial mounting is Z.
A nonzero compression error extends the two lower supports, leaves the central
underside/mount unchanged, and shortens the tip. The resulting central gap and
tip height must be measured in the posed CAD, not equated to the input parameter.
"""
import numpy as np
TIP_EXTENT_MM=62.69415283203125  # Pinned nominal CustomLegFoot.stl, negative-Y extent.

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
    distal=smooth((-v[:,1]-40.)/(TIP_EXTENT_MM-40.))
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

def build(spec, nominal_manifest):
    """Use the same deformed CAD for rendering, clearance and floor contacts.

    Two half-mesh convex hulls preserve the central base recess better than one
    whole-limb hull. This remains a contact approximation requiring convergence
    checks before acceptance. Nominal inertial properties are retained.
    """
    import hashlib,io,xml.etree.ElementTree as ET
    import trimesh
    from .core import HERE,LEGS
    keys=('support_extension_mm','tip_shortening_mm','bend_deg')
    if set(spec)-{'contact_parts'}!=set(keys):raise ValueError('Unexpected coupled formation fields')
    parts=spec.get('contact_parts',2)
    if type(parts) is not int or parts not in (2,4,8):raise ValueError('Contact parts must be 2, 4 or 8')
    fields={key:np.asarray(spec[key],dtype=float) for key in keys}
    if any(x.shape!=(4,) for x in fields.values()):raise ValueError('Four values per field required')
    root=ET.parse(HERE/'model.xml').getroot()
    assets={m.get('file'):(HERE/'assets'/m.get('file')).read_bytes() for m in root.iter('mesh')}
    original=trimesh.load(io.BytesIO(assets['CustomLegFoot.stl']),file_type='stl',process=True)
    generated={};metrics=[]
    for i,leg in enumerate(LEGS):
        values={key:float(fields[key][i]) for key in keys}
        vertices=deform(original.vertices,**values)
        metrics.append(geometry_metrics(original.vertices,vertices))
        mesh=trimesh.Trimesh(vertices=vertices,faces=original.faces,process=False)
        body=root.find(f'.//body[@name="leg_{leg}_3"]')
        visual=body.find(f'geom[@name="{leg}_shin_visual"]')
        floor=body.find(f'geom[@name="{leg}_floor_contact"]')
        if parts==2:
            right=ET.fromstring(ET.tostring(floor));right.set('name',leg+'_floor_contact_right');body.append(right)
            edges=mesh.vertices[mesh.edges_unique];cross=edges[:,0,0]*edges[:,1,0]<0
            a,b=edges[cross,0],edges[cross,1]
            cut=a+(-a[:,0]/(b[:,0]-a[:,0]))[:,None]*(b-a)
            shapes=[mesh]+[trimesh.convex.convex_hull(np.vstack((vertices[side*vertices[:,0]>=0],cut))) for side in (1,-1)]
            suffixes=['visual','half_a','half_b'];geoms=[visual,floor,right]
        else:
            floors=[floor]
            for part in range(1,parts):
                extra=ET.fromstring(ET.tostring(floor));extra.set('name',leg+'_floor_contact_'+str(part));body.append(extra);floors.append(extra)
            # Clip the SAME triangle mesh; exact edge-plane intersections, no inflation.
            bounds=np.linspace(float(vertices[:,0].min())-1e-6,float(vertices[:,0].max())+1e-6,parts+1)
            shapes=[mesh]
            for lo,hi in zip(bounds[:-1],bounds[1:]):
                points=[vertices[(vertices[:,0]>=lo)&(vertices[:,0]<=hi)]]
                edges=vertices[mesh.edges_unique]
                for plane in (lo,hi):
                    cross=(edges[:,0,0]-plane)*(edges[:,1,0]-plane)<0
                    a,b=edges[cross,0],edges[cross,1]
                    points.append(a+((plane-a[:,0])/(b[:,0]-a[:,0]))[:,None]*(b-a))
                shapes.append(trimesh.convex.convex_hull(np.vstack(points)))
            suffixes=['visual']+['part_'+str(k) for k in range(parts)];geoms=[visual]+floors
        for suffix,shape,geom in zip(suffixes,shapes,geoms):
            name=f'underformed_{leg}_{suffix}';file=name+'.stl';assets[file]=shape.export(file_type='stl')
            generated[file]=hashlib.sha256(assets[file]).hexdigest()
            ET.SubElement(root.findall('asset')[-1],'mesh',name=name,file=file,scale='.001 .001 .001')
            geom.set('mesh',name)
        for site in body.findall('site'):
            pos=np.fromstring(site.get('pos'),sep=' ')*1000
            if i%2:pos[:2]*=-1
            pos=deform(pos[None,:],**values)[0]
            if i%2:pos[:2]*=-1
            site.set('pos',' '.join(str(x/1000) for x in pos))
    xml=ET.tostring(root,encoding='utf-8');manifest=dict(nominal_manifest)
    manifest.update(nominal_model_sha256=nominal_manifest['model_sha256'],
        model_sha256=hashlib.sha256(xml).hexdigest(),formation=dict({k:v.tolist() for k,v in fields.items()},contact_parts=parts),
        generated_asset_sha256=generated,local_shape_metrics=metrics,
        formation_status='Synthetic coupled rigid under-compression; trial ranges, not measured polymer behavior',
        floor_contact_model=f'{parts} convex CAD slabs; contact convergence requires comparison',
        inertia_status='Nominal mass, COM and inertia retained; shape redistribution unvalidated',deformable_material=False)
    return xml,assets,manifest
