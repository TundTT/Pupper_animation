# Combined roll, walking and wheel controls

Use `ros2 launch neural_controller combined_motion.launch.py` for this workflow.
It loads the latest policies selected by `policies/latest.json` (backpack and 9 mm
spacing). All three controllers start inactive. The legacy full launch retains
its alignment controls; do not run two button handlers or hardware managers.

September 13 lab result: the operator verified the physical tips-up to tips-down
floor roll and subsequently reported successful walking-policy testing. The
combined stack logged successful walking activation with the mapped hub offsets.
These are supervised lab observations, not a broad hardware robustness result.
See `hardware_testing/combined_motion_2026-09-13/hardware_session_summary.md`.

Once initial calibration and the tips-up starting pose are confirmed, the
operator's button presses authorize the motions. Do not require another chat
confirmation between X and Triangle. The roll takes approximately 46 seconds,
including settling; early Triangle presses are rejected and are not queued.
Release and press Triangle again after completion, with the sticks centered.
Startup confirmation after power loss remains required.

| Button | Action |
| --- | --- |
| X (0) | From the prepared, calibrated tips-up pose: activate the roll controller, wait for READY, send START once, then hold after completion. |
| Triangle (2) | Activate walking at zero initial command, using the current-session hub reference and a fixed full-turn offset. From roll, wait until it completes successfully. |
| Circle (1) | Select the wheel policy. This changes control mode; it does not reshape the appendages. Use it with the appropriate physical configuration. |
| PS (10) | Stop. Release alone does not resume motion. |

Release the drive sticks before changing modes. Only one controller owns motor
commands at a time. Holding X cannot repeat the roll; face-button presses during
a pending switch are consumed rather than queued. Pressing a mode's button while
it is already active does not restart it. A running or faulted roll cannot be
interrupted by a drive-mode button. The e-stop remains available.

Heating and shape selection are manual. X does not run wheel alignment, heat,
calibrate, or automatically start walking. Restore tips-up on the supported stand
using the tested joint-pose workflow before another X trial.

## Walking coordinates and transfer

`calibrated_walk_frame` is enabled only for walking in the combined launch. It
uses a valid roll map when present, or the current startup tips-down hub reference
when no roll map exists. A stale roll map is rejected, not silently reused.
The nearest full-turn equivalent is selected once at activation and kept fixed:

    model_position = encoder_position - offset
    encoder_target = model_target + offset

The trained defaults, action scales, network weights and model-space limits are
unchanged. Upper joints retain their established calibration. Activation rejects
a pose more than 0.30 rad from mapped standing home; it does not infer geometry
or try to flip tips-up appendages by activating walking.

For a completed-roll handoff, the dispatcher saves fresh final command positions
and gains in `~/.local/state/quadmorph/walking-handoff.json`, bound to the current
calibration. Walking starts from those targets and gains, then smoothly blends
to its home/gains over the configured two-second initialization. The existing
two-second action fade and observation-history reset follow. A stale snapshot
or inconsistent current pose rejects activation. This removes the full-turn
unwind and commanded target/gain step. Supervised physical handoffs have now been
reported successful; broader balance robustness remains unvalidated.

The successful September 13 floor roll ended with the rear-left encoder near
7.33 rad. Direct ordinary walking would target 1 rad, about a 363-degree change.
The fixed-frame implementation preserves that winding instead. Evidence:
`hardware_testing/triangle_roll/lab/floor_1789304250787618321/`.

## Installation and physical test

`scripts/prepare_combined_motion.sh` builds into a separate `install-combined`
overlay and runs focused software tests. It never starts hardware. Do not replace
a library mapped by a running process; the helper checks its destination.

For a later physical launch, first support the robot and follow
`STARTUP_CALIBRATION.md`. Starting a new stack homes hardware and requires the
operator's actual confirmation and a fresh shared calibration capture. Source
the base `install`, `install-roll` dependencies and the new `install-combined`
overlay before launch. The tested Pi-specific wrapper is preserved at
`hardware_testing/combined_motion_2026-09-13/start_combined.sh`; it checks for a
leftover manager or button handler and resolves dependencies before launching.
Complete the existing current-session tips-up reference capture before using X.

First verify X on the stand, then the floor roll, then Triangle at zero stick
command. Validate the new walking policy's stationary response before sending a
short walking command. Successful old-policy roll trials are not physical
validation of these new weights or this controller handoff.

Hardware sources: `robot-info:robot_info/QUADMORPH.md` records the user-confirmed
continuous hubs, 9 mm spacing, manual heating, and calibration meaning;
`pupper_v3_description/description/components.xacro` specifies the selected joint
interfaces and bounds. `LATEST_POLICIES.md` and `policies/latest.json` identify
the new exports. No physical motor actions are authorized by this document.
