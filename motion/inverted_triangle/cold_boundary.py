"""Detailed CAD floor feasibility at a fixed, measured cold-start posture.

Geometry-only diagnosis. Root-height search cannot cure motor-below-foot order:
any common vertical translation preserves the signed difference. Changing the
supported joint posture before manual formation is a separate prerequisite,
never a measured endpoint or a dynamic rollout.
"""
import numpy as np
from .clearance import CADClearance


def inspect(robot):
    checker=CADClearance(robot)
    bottoms={}
    for name,vertices in checker.local_vertices.items():
        bid=checker.body_ids[name]
        points=vertices@robot.d.xmat[bid].reshape(3,3).T+robot.d.xpos[bid]
        bottoms[name]=float(points[:,2].min())
    feet={name:z for name,z in bottoms.items() if name.endswith('_3')}
    other={name:z for name,z in bottoms.items() if not name.endswith('_3')}
    nearest=min(other,key=other.get);foot=min(feet.values())
    margin=other[nearest]-foot
    return dict(detailed_bottom_m=bottoms,lowest_unintended_part=nearest,
                unintended_minus_lowest_foot_m=margin,
                contact_free_at_fixed_measured_posture=bool(margin>=0),
                root_translation_can_fix=bool(margin>=0),
                interpretation=('geometrically_possible_not_dynamic_support_proof' if margin>=0 else
                    'infeasible_at_fixed_measured_posture; establish_supported_posture_before_manual_formation_or_verify_physical_geometry'),
                user_reported_approximate_clearance_m=.0127,
                physical_measurement_context='Operator ruler estimate at photographed current-angle hold pose; not verified identical to saved simulated endpoint',
                earlier_user_estimate_m=.005,
                measurement_status='pinned_CAD_prediction_not_physical_measurement')
