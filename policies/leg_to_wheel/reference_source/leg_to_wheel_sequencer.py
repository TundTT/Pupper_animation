"""Operator-facing sequencing only; the learned policy owns all joint motion."""
from dataclasses import dataclass
from workspace.leg_to_wheel_config import COMMAND_TO_FOOT


@dataclass
class PolicySequencer:
    """A command stays lifted indefinitely until heating is explicitly confirmed.

    Clearance is actual capsule-bottom world height in metres. Completion requires
    a stable contact after commanding stand; no fixed converted-leg angle is held.
    """
    clearance_gate: float = .015
    settle_seconds: float = .3
    command: int = 0
    active: int = 0
    converted: int = 0  # FR, FL, BR, BL bits, matching the existing Controller
    phase: str = 'stand'
    stable_seconds: float = 0.

    def request(self, command):
        if command not in (1,2,3,4):raise ValueError('Expected FL/FR/BR/BL command 1..4')
        if self.phase!='stand' or self.converted & (1<<COMMAND_TO_FOOT[command]):return False
        self.command=self.active=command
        self.phase='lifting';self.stable_seconds=0.
        return True

    def update(self, dt, clearance, contact, heating_confirmed=False):
        if dt<0:raise ValueError('dt must be nonnegative')
        if self.phase in ('lifting','heating'):
            unloaded=clearance>=self.clearance_gate and not contact
            self.stable_seconds=self.stable_seconds+dt if unloaded else 0.
            self.phase='heating' if self.stable_seconds>=self.settle_seconds else 'lifting'
            if self.phase=='heating' and heating_confirmed:
                self.command=0;self.phase='lowering';self.stable_seconds=0.
        elif self.phase=='lowering':
            self.stable_seconds=self.stable_seconds+dt if contact else 0.
            if self.stable_seconds>=self.settle_seconds:
                self.converted |= 1<<COMMAND_TO_FOOT[self.active]
                self.active=0;self.phase='stand';self.stable_seconds=0.
        return self.command
