# Front-left alignment investigation, September 14

Operator observed all legs lift but front-left does not rotate. Analysis of the recorded alignment_status confirms both FL attempts remained in LIFT with zero rotation-enabled samples. After the 6.5-second reference trajectory, estimated floor clearance was -4.12 to -3.50 mm on attempt 1 and -4.58 to -0.59 mm on attempt 2. Rotation requires clearance strictly above 10 mm. Wheel spacing and body spacing passed throughout those mature FL samples. This clearance failure alone explains the blocked rotation; additional attitude/motion gate failures were not evaluated.

FR rotation clearance: 10.29–10.86 mm. BR: 31.81–33.62 mm. BL: 19.64–20.42 mm. FL residual gain faded to zero, without achieving clearance. The second FL attempt timed out at 48 seconds, lowered, and reached HOLD. Controller remained active in HOLD at inspection. No control changes or additional motion commands were issued during investigation. The recorder was stopped with SIGINT to finalize the evidence; hardware was not stopped.

Clearance is calculated from joint geometry and projected gravity relative to the highest modeled bottom of the other three wheels. It is not a measured floor distance. Physical clearance, joint tracking, calibration/model mapping, and support/tilt need comparison before selecting a correction. Do not infer a failed hub motor or bypass the clearance gate from these data.

Other three wheels entered ROTATE, but were switched to the next leg before VERIFY/completion. Their final target errors were approximately 5.0, 5.5, and 6.5 degrees; completed bitmask stayed zero. Their full automatic sequences are not yet verified.

Evidence: status_analysis.json; analyze_status.py; Pi /home/pi/robot-code-leglift/hardware_testing/align_v5_2026-09-14/trial_bag and stack.log. Calibration d947187f179143f3adf1e20bfebebb53.
