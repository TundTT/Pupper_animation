# Hardware test status — September 15, 2026

## R2 lift and align: NOT WORKING

Operator explicitly reported this test as not working and requested stopping.
Do not describe the operator-verified R2 implementation as hardware validated.
Exact observed failure and root cause have not been established. Software test
passes do not override this hardware result. Do not resume motion without a new
operator request and the required startup/calibration confirmation.

Last test used the manual lift/rotate/lower implementation documented in
LIFT_ALIGN_MANUAL.md, with calibration `be9b9928be414b6d931867827d002e1a`.
Remote evidence directory: `/tmp/quadmorph-r2-manual` (temporary).

## Earlier operator observations

- Walking: looked more like walking after the observation-frame fix; stand only.
- X roll: operator reported it looked good and verified it on the stand.
- Square: operator reported motor movement, but motion looked unusual; no floor
  performance or successful physical conversion established.

No floor validation is claimed for these tests.
