# BeamNG benchmark harness

## Single run (with metrics)

```bash
python scripts/07_run_beamng_sim.py --controller sqp --dynamics sindy --target-speed 3.5
python scripts/07_run_beamng_sim.py --controller mppi --dynamics sindy --target-speed 3.5
```

Key flags:

| Flag | Purpose |
|------|---------|
| `--target-speed MPS` | MPC/MPPI cruise setpoint (auto-scales `--max-steps`) |
| `--actuator-profile responsive\|smooth` | BeamNG throttle/brake mapping |
| `--profile` | Log per-tick solve time |
| `--log-path PATH` | JSONL telemetry (default `results/beamng_live.jsonl`) |

Startup prints resolved config; end-of-run prints mean speed, cross-track error, throttle, and MPC acceleration.

## Offline open-loop (speed bins)

No BeamNG required — one-step RK4 RMSE vs logged telemetry, binned by |v|:

```bash
python scripts/08_benchmark_beamng.py --offline-only
```

Output: `results/benchmark/<timestamp>/offline_speed_bins.json` and `results/benchmark/summary.csv`.

## Live closed-loop sweep

Requires BeamNG.tech:

```bash
python scripts/08_benchmark_beamng.py --live \
  --target-speeds 2,3.5,5 \
  --controllers sqp,mppi \
  --dynamics sindy,bicycle \
  --no-launch
```

Each cell runs `07_run_beamng_sim.py` and writes:

- `results/benchmark/<run_id>/<ctrl>_<dyn>_<speed>mps/beamng_live.jsonl`
- `results/benchmark/<run_id>/<ctrl>_<dyn>_<speed>mps/summary.json`

Aggregate:

- `results/benchmark/summary.csv` — one row per run
- `results/benchmark/speed_vs_accuracy.png` — poster plot

## Fixed-path speed sweep (1–10 m/s, SINDy)

Plans RRT **once**, saves `rrt_path.json`, and reuses it for every cell so comparisons are fair.
Runs SQP-MPC and MPPI with **SINDy** dynamics at 1, 2, …, 10 m/s (20 BeamNG runs total).

```bash
# BeamNG already running:
python scripts/08_benchmark_beamng.py --live --speed-sweep --no-launch

# Or launch BeamNG once for the full matrix:
python scripts/08_benchmark_beamng.py --live --speed-sweep
```

Outputs under `results/benchmark/<timestamp>_speed_sweep_sindy/`:

| File | Description |
|------|-------------|
| `rrt_path.json` | Shared reference waypoints |
| `sqp_sindy_<N>mps/` | Per-speed logs + `summary.json` |
| `mppi_sindy_<N>mps/` | Per-speed logs + `summary.json` |
| `summary.csv` | All runs |
| `speed_vs_accuracy.png` | Target vs error / mean speed |
| `sqp_sindy_speed_sweep_trajectories.png` | All 10 SQP trajectories on one plot |
| `mppi_sindy_speed_sweep_trajectories.png` | All 10 MPPI trajectories on one plot |

Override speeds or controllers if needed:

```bash
python scripts/08_benchmark_beamng.py --live --speed-sweep \
  --target-speeds 2,4,6,8,10 --controllers sqp --no-launch
```

Single runs can reuse a saved path:

```bash
python scripts/07_run_beamng_sim.py --controller sqp --dynamics sindy \
  --target-speed 5 --rrt-path results/benchmark/<run>/rrt_path.json
```

## Interpreting speed vs accuracy

- **x-axis (target speed):** commanded cruise from `--target-speed`
- **Final goal error:** distance to goal at end (lower is better)
- **Mean speed:** achieved |v| over the run — compare to target to see under-speed (SQP slowness)
- **Mean cross-track error:** lateral deviation from reference path
- **Success:** final error &lt; goal threshold (~5 m)

For the poster, compare **bicycle vs SINDy** at the same target speed and controller — the trade-off is accuracy vs completion time, not “SINDy always wins.”

Default cruise is **5.0 m/s** (`BEAMNG_TARGET_SPEED_MPS`). Use `--target-speed 3.5` for slower sweeps.

**Obstacles:** edit ``sindympc/config/scenario_obstacles.py`` — the same ``(x, y, radius)`` disks are used for RRT, matplotlib plots, and BeamNG procedural cylinders (disable with ``SINDYMPC_OBSTACLES=0``).

**Tuning notes (2026-06):**
- SQP slowness was largely a **v_ref low-speed trap** (reference blended from current v); fixed in `sqp_mpc._waypoints_to_reference`.
- MPPI approach braking is **overspeed-gated** (brake only when faster than ~85% of target near goal).
- Actuator layer: SQP uses responsive throttle assist; MPPI caps brake and uses gentler decel feedforward.

## Environment overrides

- `SINDYMPC_SQP_TARGET_SPEED` / `SINDYMPC_MPPI_TARGET_SPEED` — defaults when `--target-speed` omitted
- `SINDYMPC_SINDY_AUTO_THETA_GAIN=0` — disable runtime θ̇ correction on weak SINDy models
- `SINDYMPC_BNG_ACTUATOR_PROFILE=smooth` — gentler actuation
