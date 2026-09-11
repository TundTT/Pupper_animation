"""Staged task profiles and provenance-checked weight transfer (not resumption)."""
import functools
import hashlib
import json
from pathlib import Path
from . import configs as c
from .randomize import domain_randomize_wheeled

STEPS={'foundation':5_000_000,'single':10_000_000,'sequence':35_000_000}

def randomization(stage):
    if stage=='foundation':return None
    if stage=='sequence':return domain_randomize_wheeled
    if stage!='single':raise ValueError('Unknown curriculum stage')
    return functools.partial(domain_randomize_wheeled,friction_range=(.6,1.2),
        kp_multiplier_range=(.8,1.2),kd_multiplier_range=(.8,1.4),kv_multiplier_range=(.8,1.2),
        wheel_diameter_jitter=.002,body_com_x_shift_range=(-.005,.005),
        body_com_y_shift_range=(-.005,.005),body_com_z_shift_range=(-.005,.005),
        body_mass_scale_range=(.95,1.1),body_inertia_scale_range=(.95,1.1))

def initialization(path,stage,hashes):
    if path is None:return None,None
    from brax.io import model
    path=Path(path);config=json.loads(path.parent.joinpath('config.json').read_text())
    if config.get('motion_contract_version')!=c.MOTION_VERSION or config.get('source_hashes')!=hashes:
        raise ValueError('Initial checkpoint must use this exact v5 source and contract; v2/v3/v4 are incompatible.')
    previous=config.get('curriculum_stage')
    if (previous,stage) not in {('foundation','single'),('single','sequence'),(stage,stage)}:
        raise ValueError('Expected foundation -> single -> sequence stage order.')
    return model.load_params(str(path)),dict(path=str(path.resolve()),stage=previous,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        semantics='Transfer actor and normalization only; fresh optimizer, critic and environment-step count')
