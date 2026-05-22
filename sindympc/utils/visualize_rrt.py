"""Visualize RRT planning results with matplotlib.

Runs the planner from ``sindympc.planners.rrt`` on a user-defined scenario and
plots the search tree, final path, and circular obstacles.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from sindympc.planners.rrt import Node, RRT


# -----------------------------------------------------------------------------
# Scenario configuration — edit these values for your test case
# -----------------------------------------------------------------------------

START: Tuple[float, float] = (0.0, 0.0)
GOAL: Tuple[float, float] = (10.0, 10.0)
PADDING_FRACTION = 0.25

OBSTACLE_LIST: List[Tuple[float, float, float]] = [
    # (x, y, radius)
    (5, 5, 1),
    (2, 2, 1),
    (2, 8, 1),
    (8, 2, 1),
    (8, 8, 1),
]

STEP_SIZE: float = 0.1
MAX_ITER: int = 5000


def run_rrt_planner(
    start: Tuple[float, float],
    goal: Tuple[float, float],
    padding_fraction: float,
    obstacle_list: List[Tuple[float, float, float]],
    step_size: float,
    max_iter: int,
) -> Tuple[RRT, Optional[List[Tuple[float, float]]]]:
    """Instantiate the RRT planner and run ``plan()``.

    Args:
        start: (x, y) start configuration.
        goal: (x, y) goal configuration.
        padding_fraction: Padding fraction for the workspace bounds.
        obstacle_list: Circular obstacles as (x, y, radius) tuples.
        step_size: RRT extension step size.
        max_iter: Maximum RRT iterations.

    Returns:
        Tuple of (planner instance, path waypoints or None if planning failed).
    """
    planner = RRT(
        start=start,
        goal=goal,
        obstacle_list=obstacle_list,
        step_size=step_size,
        max_iter=max_iter,
        padding_fraction=padding_fraction,
    )
    path = planner.plan()
    return planner, path


def draw_obstacles(
    ax: plt.Axes,
    obstacle_list: List[Tuple[float, float, float]],
) -> None:
    """Draw circular obstacles on the axes.

    Args:
        ax: Matplotlib axes to draw on.
        obstacle_list: Obstacles as (x, y, radius) tuples.
    """
    for ox, oy, radius in obstacle_list:
        patch = Circle((ox, oy), radius, color="dimgray", alpha=0.6)
        ax.add_patch(patch)


def draw_rrt_tree(ax: plt.Axes, nodes: List[Node]) -> None:
    """Draw all RRT tree edges as light grey lines.

    Iterate over ``nodes`` and, for each node with a parent, draw a line segment
    from ``(parent.x, parent.y)`` to ``(node.x, node.y)``.

    Args:
        ax: Matplotlib axes to draw on.
        nodes: All nodes in the planner tree after ``plan()`` completes.
    """
    for node in nodes:
        if node.parent is not None:
            ax.plot([node.parent.x, node.x], [node.parent.y, node.y], "lightgray", linewidth=1)


def draw_path(ax: plt.Axes, path: List[Tuple[float, float]]) -> None:
    """Draw the final start-to-goal path as a bold, distinct line.

    Args:
        ax: Matplotlib axes to draw on.
        path: Ordered list of (x, y) waypoints from start to goal.
    """
    xs, ys = zip(*path)
    ax.plot(xs, ys, "b-", linewidth=2, label="Path")


def visualize_rrt(
    planner: RRT,
    path: Optional[List[Tuple[float, float]]],
    start: Tuple[float, float],
    goal: Tuple[float, float],
    obstacle_list: List[Tuple[float, float, float]],
) -> None:
    """Assemble and display the RRT visualization figure.

    Args:
        planner: RRT instance after planning (provides ``planner.nodes``).
        path: Solution path from ``plan()``, or None if no path was found.
        start: (x, y) start configuration for marker plotting.
        goal: (x, y) goal configuration for marker plotting.
        obstacle_list: Obstacles passed to the planner.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    draw_obstacles(ax, obstacle_list)
    draw_rrt_tree(ax, planner.nodes)

    if path is not None:
        draw_path(ax, path)

    ax.plot(start[0], start[1], "go", markersize=10, label="Start", zorder=5)
    ax.plot(goal[0], goal[1], "r*", markersize=14, label="Goal", zorder=5)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("RRT Planner")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def main() -> None:
    """Run RRT on the configured scenario and show the plot."""
    planner, path = run_rrt_planner(
        start=START,
        goal=GOAL,
        padding_fraction=PADDING_FRACTION,
        obstacle_list=OBSTACLE_LIST,
        step_size=STEP_SIZE,
        max_iter=MAX_ITER,
    )

    if path is None:
        print("RRT did not find a path. The tree will still be plotted.")

    visualize_rrt(
        planner=planner,
        path=path,
        start=START,
        goal=GOAL,
        obstacle_list=OBSTACLE_LIST,
    )


if __name__ == "__main__":
    main()
