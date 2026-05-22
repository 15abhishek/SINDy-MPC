"""Rapidly-exploring Random Tree (RRT) path planner."""

from typing import List, Optional, Tuple

import numpy as np


class Node:
    """A vertex in the RRT search tree.

    Attributes:
        x: x-coordinate of the node in the workspace.
        y: y-coordinate of the node in the workspace.
        parent: Reference to the parent node in the tree, or None for the root.
    """

    def __init__(self, x: float, y: float, parent: Optional["Node"] = None) -> None:
        self.x = x
        self.y = y
        self.parent = parent


class RRT:
    """Rapidly-exploring Random Tree planner for 2D holonomic navigation.

    Builds a collision-free tree from start toward goal by repeatedly sampling
    random configurations, extending toward them, and checking obstacle clearance.

    Args:
        start: (x, y) start configuration.
        goal: (x, y) goal configuration.
        obstacle_list: List of circular obstacles as (x, y, radius) tuples.
        step_size: Maximum extension distance per steering step.
        max_iter: Maximum number of RRT iterations before giving up.
        goal_sample_rate: Probability of sampling the goal instead of a random point.
        padding_fraction: Fraction of start–goal span used as workspace padding
            when ``bounds`` is None (e.g. 0.1 → 10% padding on each side).
        bounds: Optional workspace limits as ((x_min, x_max), (y_min, y_max)).
            If None, bounds are inferred from start and goal with ``padding_fraction``.
    """

    def __init__(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float],
        obstacle_list: List[Tuple[float, float, float]],
        step_size: float,
        max_iter: int,
        goal_sample_rate: float = 0.1,
        padding_fraction: float = 0.1,
        bounds: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
    ) -> None:
        self.start = start
        self.goal = goal
        self.padding_fraction = padding_fraction
        self.obstacle_list = obstacle_list
        self.step_size = step_size
        self.max_iter = max_iter
        self.goal_sample_rate = goal_sample_rate

        if bounds is not None:
            self.x_bounds, self.y_bounds = bounds
        else:
            self.x_bounds, self.y_bounds = self._default_bounds(
                start, goal, padding_fraction
            )

        self.nodes: List[Node] = [Node(start[0], start[1], parent=None)]

    @staticmethod
    def _default_bounds(
        start: Tuple[float, float],
        goal: Tuple[float, float],
        padding_fraction: float = 0.1,
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Compute workspace bounds from start/goal with fractional padding."""
        x_min = min(start[0], goal[0])
        x_max = max(start[0], goal[0])
        y_min = min(start[1], goal[1])
        y_max = max(start[1], goal[1])

        x_span = x_max - x_min
        y_span = y_max - y_min
        if x_span < 1e-9:
            x_span = 1.0
        if y_span < 1e-9:
            y_span = 1.0

        pad_x = padding_fraction * x_span
        pad_y = padding_fraction * y_span

        x_bounds = (x_min - pad_x, x_max + pad_x)
        y_bounds = (y_min - pad_y, y_max + pad_y)
        return x_bounds, y_bounds

    def _get_random_point(self) -> Tuple[float, float]:
        """Sample a random point uniformly inside the workspace bounds.

        With probability ``goal_sample_rate``, returns the goal instead.
        Otherwise draws x ~ Uniform(x_bounds) and y ~ Uniform(y_bounds).

        Returns:
            (x, y) sampled configuration.
        """
        if np.random.random() < self.goal_sample_rate:
            return self.goal

        x = np.random.uniform(self.x_bounds[0], self.x_bounds[1])
        y = np.random.uniform(self.y_bounds[0], self.y_bounds[1])
        return (float(x), float(y))

    def _get_nearest_node(self, random_point: Tuple[float, float]) -> Node:
        """Find the tree node closest to the given point.

        Args:
            random_point: (x, y) query configuration.

        Returns:
            Nearest node in the current tree.
        """
        return min(
            self.nodes,
            key=lambda node: np.hypot(node.x - random_point[0], node.y - random_point[1]),
        )

    def _steer(self, from_node: Node, to_point: Tuple[float, float]) -> Node:
        """Extend from a tree node toward a target point by at most step_size.

        Args:
            from_node: Tree node to extend from.
            to_point: (x, y) target configuration (e.g. the random sample).

        Returns:
            New node placed along the segment from from_node to to_point.
        """
        dx = to_point[0] - from_node.x
        dy = to_point[1] - from_node.y
        dist = np.hypot(dx, dy)

        if dist <= self.step_size:
            new_x, new_y = to_point[0], to_point[1]
        else:
            ratio = self.step_size / dist
            new_x = from_node.x + ratio * dx
            new_y = from_node.y + ratio * dy

        return Node(new_x, new_y, parent=from_node)

    @staticmethod
    def _point_to_segment_distance(
        px: float,
        py: float,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
    ) -> float:
        """Minimum distance from point (px, py) to segment (x1, y1)-(x2, y2)."""
        dx = x2 - x1
        dy = y2 - y1
        seg_len_sq = dx * dx + dy * dy

        if seg_len_sq == 0.0:
            return np.hypot(px - x1, py - y1)

        t = ((px - x1) * dx + (py - y1) * dy) / seg_len_sq
        t = max(0.0, min(1.0, t))
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy
        return np.hypot(px - proj_x, py - proj_y)

    def _is_collision_free(self, node: Node) -> bool:
        """Check whether the edge from node.parent to node is collision-free.

        Args:
            node: Candidate node whose incoming edge should be validated.

        Returns:
            True if the segment does not intersect any obstacle; False otherwise.
        """
        if node.parent is None:
            return True

        x1, y1 = node.parent.x, node.parent.y
        x2, y2 = node.x, node.y

        for ox, oy, radius in self.obstacle_list:
            if self._point_to_segment_distance(ox, oy, x1, y1, x2, y2) <= radius:
                return False

        return True

    def _extract_path(self, goal_node: Node) -> List[Tuple[float, float]]:
        """Backtrack parent pointers from goal_node to the root."""
        path: List[Tuple[float, float]] = []
        current: Optional[Node] = goal_node
        while current is not None:
            path.append((current.x, current.y))
            current = current.parent
        return path[::-1]

    def plan(self) -> Optional[List[Tuple[float, float]]]:
        """Run the main RRT loop and return a path from start to goal.

        Returns:
            List of (x, y) waypoints from start to goal if found, else None.
        """
        for _ in range(self.max_iter):
            random_point = self._get_random_point()
            nearest_node = self._get_nearest_node(random_point)
            new_node = self._steer(nearest_node, random_point)

            if not self._is_collision_free(new_node):
                continue

            self.nodes.append(new_node)

            goal_node = self._steer(new_node, self.goal)
            if (
                np.hypot(goal_node.x - self.goal[0], goal_node.y - self.goal[1])
                < 1e-9
                and self._is_collision_free(goal_node)
            ):
                self.nodes.append(goal_node)
                return self._extract_path(goal_node)

        return None
