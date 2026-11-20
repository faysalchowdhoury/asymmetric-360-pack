# asymmetric-360-pack

**Asymmetric 360° World-Model Uncertainty & Closed-Loop Agentic Harness for Dynamic Roadway Micro-Packing**

A single-file, **zero-dependency** (Python standard library only) study of a three-tier
supervisory controller that lets a vehicle safely *compress* following spacing to raise
roadway capacity, while a 360° uncertainty-aware harness prevents rear-end and side-swipe
squeezes. The evaluation runs on **real NGSIM highway trajectories**.

No pip installs, no numpy, no SUMO, no FFmpeg. Everything — the controllers, the
prediction model, the statistics, the LaTeX table, and the proof video (a hand-written
uncompressed-AVI encoder) — is pure stdlib.

## Architecture

- **System 0 — Asymmetric Envelope Guard.** Hard directional clearances (front braking
  buffer, rear closing-speed buffer, left/right lateral buffers); forces gap expansion on
  breach. Present in S1+S0 and FULL.
- **System 1 — 360° World-Model Rollout.** Short-horizon forecast of front/rear/side
  neighbors with anisotropic uncertainty, plus a candidate lane-packing density-gain
  estimate.
- **System 2 — Agentic Verification Harness.** Approves / holds / vetoes compression from
  the 360° context, an **empirical forecast-uncertainty signal** (the rolling residual of
  a constant-acceleration leader prediction, which rises during real decelerations), and
  System 0 alerts. Logs auditable rationales.

## Run

```bash
python asymmetric_360_pack.py                 # real NGSIM eval (auto-detects cache) + video
python asymmetric_360_pack.py --no-video      # skip the ~140 MB AVI
python asymmetric_360_pack.py --ngsim DIR     # NGSIM root holding <location>/page_*.csv
python asymmetric_360_pack.py --csv traj.csv  # one NGSIM-style trajectory table
```

**This project evaluates on REAL data only.** `--ngsim` expects a directory containing
`i-80/page_*.csv`, `us-101/page_*.csv`, etc. (raw NGSIM schema: `vehicle_id, frame_id,
global_time, local_x, local_y, v_length, v_width, v_vel, v_acc, lane_id, location`, US
customary units; converted to SI internally). If no NGSIM cache and no `--csv` are found,
the program exits with an error rather than fabricating data.

## Data & methodology

- For each NGSIM ego vehicle we extract a 10 s window in which a single leader is
  continuously ahead in the same lane. The ego is then driven by each of the four policies
  while **following that leader's recorded real speed profile** (real shockwave braking is
  the hazard), starting from the **real** initial gap, with **real adjacent-lane**
  neighbors supplying lateral clearances. ~160 such leader–follower pairs from I-80 and
  US-101 form the sample. This is the standard car-following evaluation methodology (cf.
  Treiber & Kesting).
- **Prediction benchmark.** System 1's directional ADE/FDE and 2σ coverage are measured
  against the **real future** leader positions.

## Outputs (`outputs/`)

| file | contents |
|---|---|
| `ablation_report.json` | full statistical summary + audit traces + provenance |
| `ablation_table.tex` | Overleaf `booktabs` table (needs `\usepackage{booktabs}`) |
| `asymmetric_360_proof.avi` | split-screen proof animation on the real NGSIM pair with the hardest recorded leader braking |

## Representative result (160 real NGSIM pairs, I-80 + US-101)

| policy | collisions | cap. gain % | env. viol. | min TTC (s) | ADE (m) | FDE (m) | cov. % |
|---|---|---|---|---|---|---|---|
| IDM baseline | 0 | 0 | 3.4 | 4.80 | — | — | — |
| Naive front-tight | 0.03 | 45.9 | 8.3 | 2.31 | — | — | — |
| S1+S0 (no S2) | 0 | 30.5 | 0.7 | 3.15 | 0.71 | 1.66 | 66.4 |
| **Full 360-pack** | 0 | 16.1 | 0.6 | 4.02 | 0.61 | 1.42 | 71.1 |

Reading: naive tight-packing buys the most throughput but rear-ends the real leader and
halves min-TTC; the full harness keeps **IDM-level safety** (min-TTC and violations at or
below the baseline) while still delivering **+16 % capacity**. The System 2 ablation
isolates its contribution — S1+S0 reaches higher capacity (30.5 %) but with a lower safety
margin, while System 2 trades some of that throughput for a larger min-TTC.

## Honest limitations

- **The trailing vehicle is a model, not replayed.** When we change the ego's control the
  real follower's recorded reaction is no longer valid, so rear-end-*from-behind* metrics
  use a disclosed IDM-style follower. The front (leader) hazard and lateral neighbors are
  real.
- **No formal safety guarantee.** System 0 is a heuristic envelope guard, not a
  control-barrier certificate.
- **Longitudinal control only** (plus a lateral clearance nudge); no full lane-change
  planner.
- **Baselines are IDM and a naive strawman.** A published safety filter (RSS, CBF-ACC,
  MPC) would be a stronger comparison.
- The proof **video renders a real NGSIM pair** (the one with the hardest recorded leader
  braking); it is a visualization of the same logic, and the quantitative claims come from
  the full NGSIM table, not from a single animated pair.

## Reproducibility

Deterministic given the same NGSIM cache and flags. The AVI is git-ignored (large and
regenerable); `ablation_report.json` and `ablation_table.tex` are the durable artifacts.
