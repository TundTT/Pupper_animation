"""Call the exact C++ controller from native MuJoCo; no duplicated control math."""
import ctypes as C
import json
import os
from pathlib import Path
import numpy as np

FIELDS = ('entry_seconds', 'shift_seconds', 'lift_seconds', 'land_seconds',
          'recenter_seconds', 'attempt_timeout_seconds', None, None,
          None, None, 'wheel_speed_limit',
          'wheel_acceleration_limit', 'abduction_speed_limit', 'hip_speed_limit',
          'joint_acceleration_limit')
PHASES = ('ENTRY', 'SHIFT', 'LIFT', 'ROTATE', 'LOWER', 'RECENTER', 'HOLD', 'STOPPED')
CONFIG = Path(__file__).with_name('config.json')
D = C.POINTER(C.c_double)

class Controller:
    def __init__(self, library=None, config=None):
        self.config = json.loads(CONFIG.read_text()) if config is None else config
        library = library or os.environ.get('KEYFRAME_ALIGN_LIBRARY')
        if not library:
            raise ValueError('Set KEYFRAME_ALIGN_LIBRARY to the CMake-built library')
        self.lib = C.CDLL(str(library))
        self.lib.kf_abi_version.restype = C.c_int
        if self.lib.kf_abi_version()!=3: raise ValueError('Rebuild the keyframe library: ABI version mismatch')
        self.lib.kf_create.argtypes = [D, D, D]; self.lib.kf_create.restype = C.c_void_p
        self.lib.kf_destroy.argtypes = [C.c_void_p]
        self.lib.kf_reset.argtypes = [C.c_void_p, D, D]; self.lib.kf_reset.restype = C.c_int
        self.lib.kf_step.argtypes = [C.c_void_p, C.c_double, C.c_int, D, D, D, D, C.c_int, D]
        self.lib.kf_step.restype = None
        values = np.array([self.config[k] if k else 0. for k in FIELDS], dtype=np.float64)
        poses = np.ascontiguousarray(self.config['poses'], dtype=np.float64)
        if poses.shape != (4, 8): raise ValueError('Expected four 8-joint poses in FR/FL/BR/BL order')
        settings=np.array([self.config[k] for k in (
            'rotation_floor_clearance_m','wheel_position_kp','wheel_position_kd',
            'alignment_angle_tolerance_rad','landing_angle_tolerance_rad','alignment_speed_tolerance_rad_s',
            'alignment_settle_seconds','hold_error_limit_rad','wheel_integral_ki','wheel_integral_limit_nm',
            'wheel_integral_window_rad')],dtype=np.float64)
        if self.config['wheel_control_mode']!='position_pid': raise ValueError('Position PID configuration required')
        self.handle = self.lib.kf_create(values.ctypes.data_as(D), poses.ctypes.data_as(D),settings.ctypes.data_as(D))
        if not self.handle: raise ValueError('C++ controller rejected configuration')

    @staticmethod
    def array(value, size):
        result = np.ascontiguousarray(value, dtype=np.float64)
        if result.shape != (size,): raise ValueError(f'Expected {size} values')
        return result

    def reset(self, q, calibrated_home):
        if not self.handle: raise ValueError('Controller is closed')
        q = self.array(q, 12); home = self.array(calibrated_home, 4)
        if self.lib.kf_reset(self.handle, q.ctypes.data_as(D), home.ctypes.data_as(D)):
            raise ValueError('Invalid state/calibration')

    def step(self, dt, command, q, qd, angular, gravity, stop=False):
        if not self.handle: raise ValueError('Controller is closed')
        arrays = [self.array(v, n) for v,n in zip((q,qd,angular,gravity), (12,12,3,3))]
        out = np.zeros(33, dtype=np.float64)
        self.lib.kf_step(self.handle, dt, command, *(a.ctypes.data_as(D) for a in arrays),
                         int(stop), out.ctypes.data_as(D))
        return out

    def close(self):
        if getattr(self, 'handle', None):
            self.lib.kf_destroy(self.handle); self.handle = None

    def __del__(self):
        self.close()
