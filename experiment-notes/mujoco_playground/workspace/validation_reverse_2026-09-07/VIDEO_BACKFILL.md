# W&B video backfill

All seven recent trials were launched with `--no_eval_videos`, which disabled intermediate, final, and best videos. Best-checkpoint review videos have now been uploaded to their original runs as `eval/video_best`. Videos use the expanded push-free 50 fps showcase; no retraining was performed.

Remote file sizes and SHA-256 hashes match local MP4s, and final reward/episode-length metrics were preserved. Exact verification: [JSON](video_backfill_verified.json).

| Trial | W&B run |
|---|---|
| soft_landing | [Open video panel](https://wandb.ai/QuadMorph/pupper-leg/runs/7mjjsvco) |
| airtime | [Open video panel](https://wandb.ai/QuadMorph/pupper-leg/runs/jbkpxue0) |
| balanced_soft_landing | [Open video panel](https://wandb.ai/QuadMorph/pupper-leg/runs/6wnmmn1t) |
| balanced_duration | [Open video panel](https://wandb.ai/QuadMorph/pupper-leg/runs/ps18o6yr) |
| precision | [Open video panel](https://wandb.ai/QuadMorph/pupper-leg/runs/eadtr3rb) |
| precision_smooth | [Open video panel](https://wandb.ai/QuadMorph/pupper-leg/runs/xdgf5hzk) |
| tip_clearance | [Open video panel](https://wandb.ai/QuadMorph/pupper-leg/runs/emap44a2) |

Future training keeps videos enabled by default. `--final_videos_only` skips intermediate rendering while preserving final/best uploads. `--no_eval_videos` remains an explicit all-video opt-out with a startup warning. Video/metric logging now leaves each training-step row open until both callbacks can populate it; an offline W&B check verified both callback orders.
