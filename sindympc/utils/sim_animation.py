"""MPC / RRT visualization helpers (static axes + optional GIF export).

Interactive toolbar play/pause is unreliable outside Jupyter, so replay is
offered as an exported GIF (``save_path``) rather than an on-screen animation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Polygon

from sindympc.planners.rrt import Node
from sindympc.utils.plot_utils import apply_scenario_axes

ObstacleDisk = Tuple[float, float, float]  # (x, y, radius)
ObstaclesAtStep = Callable[[int], Sequence[ObstacleDisk]]

TRAIL_COLOR = "blue"
TRAIL_LINEWIDTH = 2.5
TRAIL_LABEL = "Traversed path"
RRT_PATH_STYLE = {"color": "green", "linestyle": "--", "linewidth": 2}


@dataclass
class AnimationConfig:
    """Timing and display options for simulation animations."""

    dt: float = 0.1
    interval_ms: int = 50
    vehicle_length_m: float = 4.0
    vehicle_width_m: float = 2.0
    repeat: bool = True


def draw_obstacle_disks(
    ax: plt.Axes,
    obstacle_list: Sequence[ObstacleDisk],
    *,
    color: str = "dimgray",
    alpha: float = 0.6,
    zorder: float = 1,
) -> List[Circle]:
    """Add circular obstacle patches; returns handles for later updates."""
    patches: List[Circle] = []
    for ox, oy, radius in obstacle_list:
        patch = Circle(
            (ox, oy),
            radius,
            color=color,
            alpha=alpha,
            zorder=zorder,
        )
        ax.add_patch(patch)
        patches.append(patch)
    return patches


def _vehicle_polygon_xy(
    x: float,
    y: float,
    theta: float,
    length: float,
    width: float,
) -> np.ndarray:
    """Rectangle centered at rear axle, heading along +x body axis."""
    half_l = 0.5 * length
    half_w = 0.5 * width
    corners = np.array(
        [
            [half_l, half_w],
            [half_l, -half_w],
            [-half_l, -half_w],
            [-half_l, half_w],
        ],
        dtype=np.float64,
    )
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s], [s, c]])
    return corners @ rot.T + np.array([x, y])


def _setup_map_axes(
    ax: plt.Axes,
    *,
    trajectory_xy: Optional[np.ndarray] = None,
    extra_xy: Optional[Sequence[Tuple[float, float]]] = None,
) -> None:
    apply_scenario_axes(
        ax,
        trajectory_xy=trajectory_xy,
        extra_xy=extra_xy,
    )


def animate_mpc_simulation(
    executed_trajectory: np.ndarray,
    *,
    obstacle_list: Sequence[ObstacleDisk] = (),
    static_obstacle_list: Sequence[ObstacleDisk] = (),
    obstacles_at_step: Optional[ObstaclesAtStep] = None,
    rrt_path: Optional[Sequence[Tuple[float, float]]] = None,
    goal_position: Tuple[float, float],
    start_state: np.ndarray,
    title: str = "MPC simulation",
    config: Optional[AnimationConfig] = None,
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[float, float] = (8, 8),
) -> FuncAnimation:
    """Build a frame-by-frame replay and optionally write a GIF/MP4.

    Set ``save_path`` to export; ``show=True`` without ``save_path`` is discouraged
    (toolbar controls are often missing outside notebooks).
    """
    cfg = config or AnimationConfig()
    traj = np.asarray(executed_trajectory, dtype=np.float64)
    if traj.ndim != 2 or traj.shape[0] < 1:
        raise ValueError("executed_trajectory must be shape (N, state_dim) with N >= 1")

    n_frames = traj.shape[0]
    time_varying = obstacles_at_step is not None
    fixed_static = list(static_obstacle_list) or (
        list(obstacle_list) if not time_varying else []
    )

    extra = [goal_position, (float(start_state[0]), float(start_state[1]))]
    if rrt_path:
        extra.extend(rrt_path)
    fig, ax = plt.subplots(figsize=figsize)
    _setup_map_axes(ax, trajectory_xy=traj[:, :2], extra_xy=extra)

    if fixed_static:
        draw_obstacle_disks(ax, fixed_static)

    obstacle_patches: List[Circle] = []
    if not time_varying and obstacle_list and not fixed_static:
        obstacle_patches = draw_obstacle_disks(ax, obstacle_list)

    if rrt_path is not None and len(rrt_path) > 0:
        rx, ry = zip(*rrt_path)
        ax.plot(rx, ry, label="RRT path", zorder=2, **RRT_PATH_STYLE)

    ax.plot(
        float(start_state[0]),
        float(start_state[1]),
        "go",
        markersize=10,
        label="Start",
        zorder=6,
    )
    ax.plot(
        goal_position[0],
        goal_position[1],
        "r*",
        markersize=14,
        label="Goal",
        zorder=6,
    )

    (trail_line,) = ax.plot(
        [],
        [],
        color=TRAIL_COLOR,
        linestyle="-",
        linewidth=TRAIL_LINEWIDTH,
        label=TRAIL_LABEL,
        zorder=4,
        solid_capstyle="round",
    )
    vehicle_body = Polygon(
        np.zeros((4, 2)),
        closed=True,
        facecolor="tomato",
        edgecolor="darkred",
        linewidth=1.5,
        alpha=0.9,
        zorder=7,
    )
    ax.add_patch(vehicle_body)

    time_text = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    ax.set_title(title)
    ax.legend(loc="upper right")

    def _update_dynamic_obstacles(step: int) -> None:
        nonlocal obstacle_patches
        if obstacles_at_step is None:
            return
        disks = list(obstacles_at_step(step))
        for p in obstacle_patches:
            p.remove()
        obstacle_patches = draw_obstacle_disks(
            ax, disks, color="darkorange", alpha=0.65
        )

    def init():
        trail_line.set_data([], [])
        vehicle_body.set_xy(np.zeros((4, 2)))
        time_text.set_text("")
        return trail_line, vehicle_body, time_text

    def update(frame: int):
        if time_varying:
            _update_dynamic_obstacles(frame)

        # Cumulative blue trace from start through current frame
        trail_line.set_data(traj[: frame + 1, 0], traj[: frame + 1, 1])
        trail_line.set_color(TRAIL_COLOR)

        x, y, theta = float(traj[frame, 0]), float(traj[frame, 1]), float(traj[frame, 2])
        vehicle_body.set_xy(
            _vehicle_polygon_xy(
                x,
                y,
                theta,
                cfg.vehicle_length_m,
                cfg.vehicle_width_m,
            )
        )

        t_s = frame * cfg.dt
        time_text.set_text(f"step {frame}  |  t = {t_s:.1f} s")
        ax.set_title(f"{title}  (step {frame}/{n_frames - 1})")
        artists: list = [trail_line, vehicle_body, time_text]
        if time_varying:
            artists.extend(obstacle_patches)
        return artists

    anim = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=n_frames,
        interval=cfg.interval_ms,
        repeat=cfg.repeat,
        blit=False,
    )

    if save_path:
        _save_animation(anim, save_path)
        plt.close(fig)
    elif show:
        plt.tight_layout()
        plt.show()
    else:
        plt.close(fig)

    return anim


def animate_rrt_growth(
    nodes: Sequence[Node],
    *,
    path: Optional[Sequence[Tuple[float, float]]] = None,
    obstacle_list: Sequence[ObstacleDisk] = (),
    start: Tuple[float, float],
    goal: Tuple[float, float],
    title: str = "RRT growth",
    config: Optional[AnimationConfig] = None,
    nodes_per_frame: int = 8,
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[float, float] = (8, 8),
) -> FuncAnimation:
    """Animate RRT tree growth with standard ``FuncAnimation`` controls."""
    cfg = config or AnimationConfig()
    if len(nodes) < 1:
        raise ValueError("nodes must be non-empty")

    extra = [start, goal]
    if path:
        extra.extend(path)
    fig, ax = plt.subplots(figsize=figsize)
    _setup_map_axes(ax, extra_xy=extra)
    draw_obstacle_disks(ax, obstacle_list)

    (tree_line,) = ax.plot([], [], color="lightgray", linewidth=1, zorder=2)
    (path_line,) = ax.plot(
        [],
        [],
        color=TRAIL_COLOR,
        linewidth=TRAIL_LINEWIDTH,
        label=TRAIL_LABEL,
        zorder=3,
    )
    ax.plot(start[0], start[1], "go", markersize=10, label="Start", zorder=5)
    ax.plot(goal[0], goal[1], "r*", markersize=14, label="Goal", zorder=5)

    n_nodes = len(nodes)
    frame_indices = list(range(1, n_nodes, max(1, nodes_per_frame)))
    if frame_indices[-1] != n_nodes:
        frame_indices.append(n_nodes)

    def _edges_up_to(count: int) -> Tuple[List[float], List[float]]:
        xs: List[float] = []
        ys: List[float] = []
        for node in nodes[:count]:
            if node.parent is not None:
                xs.extend([node.parent.x, node.x, np.nan])
                ys.extend([node.parent.y, node.y, np.nan])
        return xs, ys

    def init():
        tree_line.set_data([], [])
        path_line.set_data([], [])
        return tree_line, path_line

    def update(frame_idx: int):
        count = frame_indices[frame_idx]
        xs, ys = _edges_up_to(count)
        tree_line.set_data(xs, ys)
        if path is not None and count >= n_nodes:
            px, py = zip(*path)
            path_line.set_data(px, py)
        else:
            path_line.set_data([], [])
        ax.set_title(f"{title}  ({count}/{n_nodes} nodes)")
        return tree_line, path_line

    anim = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=len(frame_indices),
        interval=cfg.interval_ms,
        repeat=cfg.repeat,
        blit=False,
    )
    ax.legend(loc="upper right")

    if save_path:
        _save_animation(anim, save_path)
        plt.close(fig)
    elif show:
        plt.tight_layout()
        plt.show()
    else:
        plt.close(fig)
    return anim


def _save_animation(anim: FuncAnimation, path: str) -> None:
    """Write GIF (pillow) or MP4 (ffmpeg) depending on file extension."""
    lower = path.lower()
    if lower.endswith(".gif"):
        anim.save(path, writer="pillow", dpi=120)
    elif lower.endswith(".mp4"):
        anim.save(path, writer="ffmpeg", dpi=120)
    else:
        raise ValueError(f"Unsupported animation extension: {path} (use .gif or .mp4)")
    print(f"Saved animation to {path}")
