# Selected-export reconciliation, September 15, 2026

This archive preserves the previous release records and inference fixtures while
reconciling robot-code at 04c0de9 with the JSON exports actually selected by the
combined launcher. It is not a training-checkpoint parity certification.

Walking uses the seq_d export introduced at 63820b0, source run
`reference_2026-09-14_06-51-00`. Its command contract is forward +/-0.35, lateral
zero, yaw +/-2, fixed upright orientation, and action scale 0.75. Previous
checkpoint-step claims were removed because they describe the older export.

The original seq_d JSON is preserved as `seq_d_original_export.json`. Its large
first-layer weights on fixed orientation inputs caused float32 cancellation.
The selected JSON folds those fixed inputs into the first-layer bias using
float64, then zeros their weights. This is equivalent only for the supported
fixed upright command; it is not retraining. On 128 reference observations,
original and transformed float64 actions differ by at most 1.224e-11. The
Windows RTNeural/Eigen float32 comparison passes the unchanged 3e-5 tolerance
with maximum error 7.004e-7.

New reference outputs are independent NumPy float64 evaluations of the shipped
exports. Walking expected outputs use the original seq_d weights. Inputs retain
the historical fixtures with lateral commands set to the current zero contract.
The previous reference CSVs and manifest entries remain archived here.

Notebook release hash changes cover combined integration files only; they do
not establish a new trained actor or physical validation. Pi build/test logs
are in `hardware_testing/setup_20260915` on the robot. No motors were started
during software preparation. Physical calibration and validation remain pending.
