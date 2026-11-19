# Obstacle-test data

Native MuJoCo at 250 Hz, 50 Hz policy; nominal rigid-flush geometry, friction 2, forward command 0.20 m/s, eight seconds per trial. Three 30 mm-wide transverse bars are centered at X=0.32,0.57,0.82 m, plus a 35 mm offset per seed. No terrain observations, pushes, observation noise or action latency. Seeds change reset pose and bar placement.

Completion means no fall and final torso X greater than the last bar center plus 150 mm. It is a traversal criterion, not clean clearance. Tracking error is mean XY velocity error after the first second; contact/overlap counts include the full trial.

| Policy | Height, mm | Completed | Falls | Mean XY error, m/s | Side-contact records | Mean ring-side overlap |
|---|---:|---:|---:|---:|---:|---:|
| preferred_15-29-41 | 10 | 1/3 | 0 | 0.1058 | 2116 | 7.496% |
| selected_long21 | 0 | 3/3 | 0 | 0.0402 | 0 | 0.000% |
| selected_long21 | 5 | 3/3 | 0 | 0.0426 | 189 | 0.000% |
| selected_long21 | 10 | 3/3 | 0 | 0.0594 | 860 | 4.737% |

Side-contact records sum penetrating capsule/obstacle contact records whose normal has |Z| <= 0.7; top contacts have |Z| > 0.7. They are sampled every 4 ms and are not counts of distinct strikes or measurements of impact force. A persistent contact contributes repeatedly. Ring-side overlap is the fraction of physics samples with any side-outline probe inside a bar, averaged equally over four feet and three trials. The ring is a geometric proxy, not a force-bearing collision body.

[Trial CSV](obstacle_trials.csv), [per-foot CSV](obstacle_feet.csv), [summary JSON](obstacle_summary.json). Original reports: [preferred](preferred_obstacles/report.json), [selected](selected_obstacles/report.json). [Selected 10 mm video](selected_obstacles/obstacles_10mm.mp4).

Only the selected policy has 0/5 mm tests here; the preferred-policy comparison is available at 10 mm. Three seeds are a small simulation sample. No force traces, distinct-impact event counts or obstacle-relative clearance trajectories were saved in these reports.
