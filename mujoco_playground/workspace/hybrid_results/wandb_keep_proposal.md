# Approved W&B cleanup

Inventory: 41 existing runs before this new attempt.

| Run | ID | Reason |
|---|---|---|
| [wheel-align_2026-09-07_20-41-57](https://wandb.ai/QuadMorph/wheel-leg%20lift%20and%20align%20triangle%20base/runs/olqtuca3) | olqtuca3 | Named favorite; unseen 64: 45/64, survival 100%, drift p95 4.80 cm, held p95 0.0248 rad. |
| [wheel-align_2026-09-07_20-25-49](https://wandb.ai/QuadMorph/wheel-leg%20lift%20and%20align%20triangle%20base/runs/1wpnxywj) | 1wpnxywj | Parent reference; unseen 64 validation 46/64 with 100% survival. |
| [wheel-align_2026-09-07_19-04-12](https://wandb.ai/QuadMorph/wheel-leg%20lift%20and%20align%20triangle%20base/runs/9xcraanb) | 9xcraanb | Earlier working milestone; audit 37.5% success with 100% survival (audit size not inferred). |

Also keep this new hybrid attempt and its evaluation. The user approved deleting the remaining 38 original runs. All 38 deletions completed and absence was verified against a fresh project listing; see wandb_deletion_log.jsonl and wandb_after_cleanup.json.

Inventory stores scalar summaries and identifiers only; no old policy code or checkpoints were used.
