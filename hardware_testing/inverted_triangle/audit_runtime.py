"""Replay actual C++ controller targets/guards through the pinned MuJoCo audit.

Starts after externally supported arming. No robot, training or W&B upload.
The supplied .so is built from test/inverted_triangle_test.cpp with TRIANGLE_BRIDGE.
"""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--library', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--pause-seconds', type=float, default=5.)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(a.source.resolve()))
    from motion.inverted_triangle import core, simulate
    from motion.inverted_triangle.clearance import CADClearance
    import numpy as np
    root = Path(__file__).resolve().parents[2]
    library = ctypes.CDLL(str(a.library.resolve()))
    ptr = ctypes.POINTER(ctypes.c_double)
    library.triangle_create.argtypes = [ctypes.c_char_p]
    library.triangle_create.restype = ctypes.c_void_p
    library.triangle_tick.argtypes = [ctypes.c_void_p, ptr, ptr, ctypes.c_double, ctypes.c_int, ctypes.c_double, ptr]
    library.triangle_destroy.argtypes = [ctypes.c_void_p]
    handle = library.triangle_create(str(root/'ros2_ws/src/neural_controller/launch/inverted_triangle_plan.json').encode())
    if not handle:
        raise ValueError('C++ plan load failed')
    output = np.zeros(16)
    original_tick = core.Robot.tick
    request, robot, steps, parity = 1, None, 0, 0.
    paused = False
    report = dict(status='FAILED', software_only=True, supported_arming_not_simulated=True,
                  library_sha256=hashlib.sha256(a.library.read_bytes()).hexdigest(), pause_seconds=a.pause_seconds,
                  source_hashes={str(path.relative_to(root)): hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
                    for path in [root/'ros2_ws/src/neural_controller/include/neural_controller/inverted_triangle.hpp',
                                 root/'ros2_ws/src/neural_controller/test/inverted_triangle_test.cpp']})

    def tick(self, target, *args, **kwargs):
        nonlocal request, robot, steps, parity
        robot = self
        q = np.ascontiguousarray(self.d.qpos[7:])
        velocity = np.ascontiguousarray(self.d.qvel[6:])
        library.triangle_tick(handle, q.ctypes.data_as(ptr), velocity.ctypes.data_as(ptr), self.tilt(),
                              request, 1/520, output.ctypes.data_as(ptr))
        request = 0
        if int(output[12]) == 5:
            raise ValueError(f'C++ runtime fault {int(output[14])} at step {steps}, segment {int(output[15])}')
        error = float(np.max(np.abs(output[:12]-target)))
        parity = max(parity, error)
        if error > 1e-9:
            report['mismatch_feedback'] = dict(q=q.tolist(), velocity=velocity.tolist(), command=output[:12].tolist(),
                                               tracking_error=(output[:12]-q).tolist(), controller_state=output[12:].tolist())
            raise ValueError(f'C++/original target mismatch {error} at step {steps}, segment {int(output[15])}')
        if not paused:
            steps += 1
        return original_tick(self, output[:12].copy(), *args, **kwargs)

    core.Robot.tick = tick
    plan_dir = a.source/'motion/inverted_triangle/results/robustness_20260912/plan'
    plan = json.loads((plan_dir/'plan.json').read_text())
    state, audits, pause_audits = None, [], []
    try:
        for stage, item in enumerate(plan['stages']):
            request = stage+1
            audit = simulate.run(plan_dir/item['candidate'], a.output/f'{stage+1:02d}-{item["leg"]}',
                                 landing_delta=item['landing_delta'], direction=item['direction'], start_override=state)
            audits.append(audit['status'])
            if audit['status'] != 'PASS_NOMINAL_SINGLE_FLIP' or int(output[13]) != stage+1:
                raise ValueError('Physics audit or controller stage completion failed')
            hold = output[:12].copy()
            paused = True
            minimum_support, maximum_tilt, maximum_floor, minimum_cad = 1e6, 0., 0., 1.
            cad = CADClearance(robot)
            for step in range(round(a.pause_seconds*520)):
                robot.tick(hold)
                if step % 13 == 0:
                    minimum_support = min(minimum_support, float(robot.contacts_precise().min()))
                    maximum_tilt = max(maximum_tilt, float(np.rad2deg(robot.tilt())))
                    maximum_floor = max(maximum_floor, robot.unintended_floor_force())
                if step % 52 == 0:
                    minimum_cad = min(minimum_cad, cad.measure()['minimum_m'])
            passed = minimum_support >= 1. and maximum_tilt < 8 and maximum_floor < .2 and minimum_cad >= .001
            pause_audits.append(dict(stage=stage, passed=passed, support_N=minimum_support,
                                     tilt_deg=maximum_tilt, motor_floor_N=maximum_floor, cad_m=minimum_cad))
            if not passed:
                raise ValueError('Planted pause audit failed')
            paused = False
            state = robot.snapshot(hold)
        report.update(status='PASS', trajectory_steps=steps, maximum_target_error_rad=parity,
                      stage_audits=audits, pauses=pause_audits)
    except Exception as e:
        report.update(error=str(e), trajectory_steps=steps, maximum_target_error_rad=parity, stage_audits=audits, pauses=pause_audits)
        raise
    finally:
        library.triangle_destroy(handle)
        (a.output/'runtime_audit.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
