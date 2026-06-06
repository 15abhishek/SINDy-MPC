"""Generate LaTeX report figures from existing benchmark artifacts.

Produces (by default) five PNGs under ``docs/figs/``:

1. ``offline_speed_bins_rmse.png`` — open-loop bicycle vs SINDy RMSE by speed bin
2. ``sindy_coefficients.png`` — learned vs nominal kinematic coefficients
3. ``cross_track_error_3.5.png`` — SQP bicycle vs SINDy CTE time series
4. ``sqp_solve_time_hist.png`` — SQP solve-time histogram (bicycle vs SINDy)
5. ``rrt_reference_path.png`` — fixed RRT reference path with scenario obstacles

Usage
-----
# Defaults (auto-find latest offline bins + known benchmark run):
python scripts/plot_report_figures.py

# Custom paths:
python scripts/plot_report_figures.py \\
    --offline-bins results/benchmark/20260603_000857/offline_speed_bins.json \\
    --benchmark-run results/benchmark/20260603_013925 \\
    --rrt-path results/benchmark/20260604_163838_speed_sweep_sindy/rrt_path.json \\
    --target-speed 3.5 \\
    --out-dir docs/figs

# Subset of figures:
python scripts/plot_report_figures.py --figures offline,coefficients
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from sindympc.config.beamng_defaults import BEAMNG_GOAL_XY, BEAMNG_START_XY  # noqa: E402
from sindympc.utils.obstacles import get_scenario_obstacles  # noqa: E402
from sindympc.utils.plot_utils import apply_scenario_axes  # noqa: E402

from sindympc.system_id.sindy_features import KINEMATIC_FEATURE_NAMES  # noqa: E402
from sindympc.system_id.sindy_training import (  # noqa: E402
    DEFAULT_WHEEL_BASE,
    STATE_NAMES,
    kinematic_bicycle_targets,
)
from sindympc.utils.benchmark_metrics import cross_track_error, load_jsonl_log  # noqa: E402

DEFAULT_OUT_DIR = PROJECT_ROOT / "docs" / "figs"
DEFAULT_MODEL = PROJECT_ROOT / "data" / "sindy_model.npz"
DEFAULT_BENCHMARK_RUN = PROJECT_ROOT / "results" / "benchmark" / "20260603_013925"
DEFAULT_RRT_PATH = (
    PROJECT_ROOT
    / "results"
    / "benchmark"
    / "20260604_163838_speed_sweep_sindy"
    / "rrt_path.json"
)
DEFAULT_TARGET_SPEED = 3.5

FIGURE_CHOICES = ("offline", "coefficients", "cte", "solve_time", "rrt", "all")


def _latest_file(pattern: str, root: Path = PROJECT_ROOT / "results") -> Path | None:
    hits = sorted(root.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return hits[0] if hits else None


def load_rrt_path(path: Path) -> list[tuple[float, float]]:
    """Load reference waypoints from ``[[x, y], ...]`` JSON."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return [(float(p[0]), float(p[1])) for p in data]


def _speed_bin_label(speed_bin: str) -> str:
    mapping = {
        "low_v_lt2": r"$v < 2$",
        "med_v_2_4": r"$2 \leq v < 4$",
        "high_v_ge4": r"$v \geq 4$",
    }
    return mapping.get(speed_bin, speed_bin)


def plot_offline_speed_bins(
    offline_path: Path,
    out_path: Path,
) -> None:
    """Grouped bar chart of open-loop RMSE by speed bin."""
    rows = json.loads(offline_path.read_text(encoding="utf-8"))
    if not rows:
        raise ValueError(f"No rows in {offline_path}")

    labels = [_speed_bin_label(r["speed_bin"]) for r in rows]
    b_xy = [r["bicycle_rmse_xy"] for r in rows]
    s_xy = [r["sindy_rmse_xy"] for r in rows]
    b_v = [r["bicycle_rmse_v"] for r in rows]
    s_v = [r["sindy_rmse_v"] for r in rows]

    x = np.arange(len(rows))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.5))
    axes[0].bar(x - width / 2, b_xy, width, label="Bicycle", color="#4c72b0")
    axes[0].bar(x + width / 2, s_xy, width, label="SINDy", color="#dd8452")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("RMSE $(x,y)$ [m]")
    axes[0].set_title("Position error")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].bar(x - width / 2, b_v, width, label="Bicycle", color="#4c72b0")
    axes[1].bar(x + width / 2, s_v, width, label="SINDy", color="#dd8452")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("RMSE $v$ [m/s]")
    axes[1].set_title("Speed error")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Offline one-step RK4 RMSE by speed bin", fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[Plot] Saved {out_path}")


def _kinematic_learned_coefficients(
    archive: np.lib.npyio.NpzFile,
    wheel_base: float,
) -> tuple[list[float], list[float], list[str]]:
    """Extract learned and nominal coeffs for the kinematic bicycle basis."""
    xi = np.asarray(archive["coefficients"], dtype=np.float64)
    version = int(np.asarray(archive.get("model_version", [1])).reshape(-1)[0])
    n_kin = int(np.asarray(archive.get("n_kinematic_features", [4])).reshape(-1)[0])

    if version not in (4, 6) or xi.shape[1] < n_kin:
        # Fallback: largest-magnitude entry per state row
        learned = [float(np.max(np.abs(xi[i]))) for i in range(min(4, xi.shape[0]))]
    else:
        learned = []
        for i in range(4):
            row = xi[i, :n_kin]
            idx = int(np.argmax(np.abs(row)))
            learned.append(float(row[idx]))

    targets = kinematic_bicycle_targets(wheel_base)
    nominal: list[float] = []
    for state, feat_name in zip(
        STATE_NAMES,
        KINEMATIC_FEATURE_NAMES,
        strict=True,
    ):
        nominal.append(float(targets[state][feat_name]))

    labels = [
        r"$\dot{x}$: $v\cos\theta$",
        r"$\dot{y}$: $v\sin\theta$",
        r"$\dot{\theta}$: $v\tan\delta/L$",
        r"$\dot{v}$: $a$",
    ]
    return learned, nominal, labels


def plot_sindy_coefficients(
    model_path: Path,
    out_path: Path,
    *,
    wheel_base: float = DEFAULT_WHEEL_BASE,
) -> None:
    """Bar chart of learned vs nominal kinematic SINDy coefficients."""
    if not model_path.is_file():
        raise FileNotFoundError(
            f"SINDy model not found: {model_path}\n"
            "Run: python scripts/02_train_sindy.py"
        )

    with np.load(model_path) as archive:
        wb = float(np.asarray(archive.get("wheel_base", [wheel_base])).reshape(-1)[0])
        learned, nominal, labels = _kinematic_learned_coefficients(archive, wb)

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7.5, 3.5))
    ax.bar(x - width / 2, nominal, width, label="Nominal", color="#4c72b0")
    ax.bar(x + width / 2, learned, width, label="SINDy", color="#dd8452")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_ylabel("Coefficient")
    ax.set_title(f"Kinematic SINDy coefficients ($L={wb:.1f}$ m)")
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[Plot] Saved {out_path}")


def _cell_log_path(benchmark_run: Path, controller: str, dynamics: str, speed: float) -> Path:
    tag = f"{controller}_{dynamics}_{speed:.1f}mps"
    return benchmark_run / tag / "beamng_live.jsonl"


def plot_cross_track_error(
    *,
    benchmark_run: Path,
    rrt_path: Path,
    target_speed: float,
    out_path: Path,
) -> None:
    """Time series of cross-track error for SQP bicycle vs SINDy."""
    if not rrt_path.is_file():
        raise FileNotFoundError(f"RRT path not found: {rrt_path}")

    reference = load_rrt_path(rrt_path)
    runs = {
        "SQP + bicycle": _cell_log_path(benchmark_run, "sqp", "bicycle", target_speed),
        "SQP + SINDy": _cell_log_path(benchmark_run, "sqp", "sindy", target_speed),
    }

    fig, ax = plt.subplots(figsize=(8.5, 3.2))
    plotted = False
    for label, log_path in runs.items():
        if not log_path.is_file():
            print(f"[Warn] Missing log for CTE plot: {log_path}")
            continue
        records = load_jsonl_log(log_path)
        xy = np.array([rec["state"][:2] for rec in records], dtype=np.float64)
        cte = cross_track_error(xy, reference)
        ticks = np.arange(cte.shape[0])
        ax.plot(ticks, cte, linewidth=1.2, label=label)
        plotted = True

    if not plotted:
        raise FileNotFoundError(
            f"No beamng_live.jsonl logs found under {benchmark_run} "
            f"for target speed {target_speed} m/s"
        )

    ax.set_xlabel("Control tick (10 Hz)")
    ax.set_ylabel("Cross-track error [m]")
    ax.set_title(f"SQP cross-track error @ {target_speed:.1f} m/s target")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[Plot] Saved {out_path}")


def plot_solve_time_histogram(
    *,
    benchmark_run: Path,
    target_speed: float,
    out_path: Path,
) -> None:
    """Histogram of per-tick SQP solve times for bicycle vs SINDy."""
    runs = {
        "Bicycle": _cell_log_path(benchmark_run, "sqp", "bicycle", target_speed),
        "SINDy": _cell_log_path(benchmark_run, "sqp", "sindy", target_speed),
    }

    fig, ax = plt.subplots(figsize=(7.5, 3.5))
    plotted = False
    for label, log_path in runs.items():
        if not log_path.is_file():
            print(f"[Warn] Missing log for solve-time plot: {log_path}")
            continue
        records = load_jsonl_log(log_path)
        if not records or "solve_time_s" not in records[0]:
            print(f"[Warn] No solve_time_s field in {log_path}")
            continue
        times_ms = 1e3 * np.array([r["solve_time_s"] for r in records], dtype=np.float64)
        ax.hist(times_ms, bins=40, alpha=0.55, label=label, density=True)
        plotted = True

    if not plotted:
        raise FileNotFoundError(
            f"No SQP logs with solve_time_s under {benchmark_run} "
            f"for target speed {target_speed} m/s"
        )

    ax.set_xlabel("Solve time [ms]")
    ax.set_ylabel("Density")
    ax.set_title(f"SQP solve time @ {target_speed:.1f} m/s target")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[Plot] Saved {out_path}")


def plot_rrt_reference_path(
    rrt_path: Path,
    out_path: Path,
    *,
    obstacle_list: Sequence[tuple[float, float, float]] | None = None,
) -> None:
    """Top-down map of the fixed RRT reference path used in closed-loop runs."""
    if not rrt_path.is_file():
        raise FileNotFoundError(f"RRT path not found: {rrt_path}")

    reference = load_rrt_path(rrt_path)
    obstacles = list(obstacle_list or get_scenario_obstacles())

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    for ox, oy, radius in obstacles:
        ax.add_patch(Circle((ox, oy), radius, color="dimgray", alpha=0.55))

    xs, ys = zip(*reference)
    ax.plot(xs, ys, "g--", linewidth=2.2, label="Reference path", zorder=2)
    ax.plot(
        BEAMNG_START_XY[0],
        BEAMNG_START_XY[1],
        "go",
        markersize=10,
        label="Start",
        zorder=5,
    )
    ax.plot(
        BEAMNG_GOAL_XY[0],
        BEAMNG_GOAL_XY[1],
        "r*",
        markersize=14,
        label="Goal",
        zorder=5,
    )

    apply_scenario_axes(
        ax,
        extra_xy=[BEAMNG_GOAL_XY, BEAMNG_START_XY, *reference],
    )
    ax.set_title(
        "RRT reference path (smoothed, 2 m spacing)\n"
        f"{len(reference)} waypoints, GridMap v2 scenario"
    )
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[Plot] Saved {out_path}")


def parse_figures(raw: str) -> set[str]:
    parts = {p.strip().lower() for p in raw.split(",") if p.strip()}
    if "all" in parts:
        return {"offline", "coefficients", "cte", "solve_time", "rrt"}
    unknown = parts - set(FIGURE_CHOICES)
    if unknown:
        raise ValueError(f"Unknown figures: {unknown}. Choose from {FIGURE_CHOICES}")
    return parts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory for output PNGs (default: docs/figs)",
    )
    parser.add_argument(
        "--offline-bins",
        type=Path,
        default=None,
        help="offline_speed_bins.json (default: latest under results/)",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="Trained SINDy NPZ (default: data/sindy_model.npz)",
    )
    parser.add_argument(
        "--benchmark-run",
        type=Path,
        default=DEFAULT_BENCHMARK_RUN,
        help="Closed-loop benchmark folder with sqp_*_*mps cells",
    )
    parser.add_argument(
        "--rrt-path",
        type=Path,
        default=DEFAULT_RRT_PATH,
        help="Fixed RRT reference path JSON for CTE computation",
    )
    parser.add_argument(
        "--target-speed",
        type=float,
        default=DEFAULT_TARGET_SPEED,
        help="Target speed [m/s] for CTE and solve-time plots",
    )
    parser.add_argument(
        "--figures",
        type=str,
        default="all",
        help="Comma-separated: offline,coefficients,cte,solve_time,rrt,all",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip figures that fail instead of exiting with error",
    )
    return parser.parse_args()


def _run_safe(fn, *, skip_missing: bool) -> None:
    try:
        fn()
    except FileNotFoundError as exc:
        if skip_missing:
            print(f"[Skip] {exc}")
        else:
            raise


def main() -> None:
    args = parse_args()
    figures = parse_figures(args.figures)

    offline_bins = args.offline_bins or _latest_file(
        "benchmark/*/offline_speed_bins.json"
    )
    if offline_bins is None:
        offline_bins = PROJECT_ROOT / "results" / "benchmark" / "20260603_000857" / "offline_speed_bins.json"

    out_dir: Path = args.out_dir
    speed_tag = f"{args.target_speed:.1f}".replace(".", "_")

    if "offline" in figures:
        _run_safe(
            lambda: plot_offline_speed_bins(
                offline_bins,
                out_dir / "offline_speed_bins_rmse.png",
            ),
            skip_missing=args.skip_missing,
        )

    if "coefficients" in figures:
        _run_safe(
            lambda: plot_sindy_coefficients(
                args.model,
                out_dir / "sindy_coefficients.png",
            ),
            skip_missing=args.skip_missing,
        )

    if "cte" in figures:
        _run_safe(
            lambda: plot_cross_track_error(
                benchmark_run=args.benchmark_run,
                rrt_path=args.rrt_path,
                target_speed=args.target_speed,
                out_path=out_dir / f"cross_track_error_{speed_tag}.png",
            ),
            skip_missing=args.skip_missing,
        )

    if "solve_time" in figures:
        _run_safe(
            lambda: plot_solve_time_histogram(
                benchmark_run=args.benchmark_run,
                target_speed=args.target_speed,
                out_path=out_dir / f"sqp_solve_time_hist_{speed_tag}.png",
            ),
            skip_missing=args.skip_missing,
        )

    if "rrt" in figures:
        _run_safe(
            lambda: plot_rrt_reference_path(
                args.rrt_path,
                out_dir / "rrt_reference_path.png",
            ),
            skip_missing=args.skip_missing,
        )

    print(f"\n[Done] Figures written to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
