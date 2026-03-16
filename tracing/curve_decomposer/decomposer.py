"""
decomposer.py - Functions to decompose a polyline/curve into n linear segments
of specified lengths by arc-length reparameterisation.
"""

import math
from typing import List, Tuple, Optional, Callable


Point = Tuple[float, float]
Segment = Tuple[Point, Point]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dist(a: Point, b: Point) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def _total_length(points: List[Point]) -> float:
    return sum(_dist(points[i], points[i + 1]) for i in range(len(points) - 1))


def _point_at_distance(points: List[Point], target_dist: float) -> Point:
    """
    Walk along the polyline and return the point that is exactly
    `target_dist` from the start (by arc length).
    """
    accumulated = 0.0
    for i in range(len(points) - 1):
        seg_len = _dist(points[i], points[i + 1])
        if accumulated + seg_len >= target_dist - 1e-9:
            remainder = target_dist - accumulated
            if seg_len < 1e-12:
                return points[i]
            t = remainder / seg_len
            x = points[i][0] + t * (points[i + 1][0] - points[i][0])
            y = points[i][1] + t * (points[i + 1][1] - points[i][1])
            return (x, y)
        accumulated += seg_len
    return points[-1]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def decompose_curve(
    points: List[Point],
    n: int,
    segment_lengths: Optional[List[float]] = None,
    max_angle: Optional[float] = None,
) -> List[Segment]:
    """
    Decompose a polyline into *n* linear segments.

    Parameters
    ----------
    points : list of (x, y)
        The raw drawn points (dense polyline).
    n : int
        Number of segments to produce.
    segment_lengths : list of float, optional
        Desired arc-length for each segment.
        - If *None*   → all segments are equal (total length / n).
        - If a single-element list → that length is used for every segment.
        - If a list of length n → each segment gets its own target length.
          length of a segment will be list[n]/sum(list) * length of curve drawn

    Returns
    -------
    list of ((x0,y0), (x1,y1))
        The *n* segments as pairs of endpoints.

    Raises
    ------
    ValueError
        If `points` has fewer than 2 entries, `n` < 1, or `segment_lengths`
        has a length other than 1 or n.
    """
    if len(points) < 2:
        raise ValueError("Need at least 2 points to form a curve.")
    if n < 1:
        raise ValueError("n must be >= 1.")

    total = _total_length(points)
    if total < 1e-9:
        raise ValueError("Curve has zero length.")

    # Build the per-segment length list
    if segment_lengths is None:
        lengths = [1] * n
    elif len(segment_lengths) == 1:
        lengths = [segment_lengths[0]] * n
    elif len(segment_lengths) == n:
        lengths = list(segment_lengths)
    else:
        raise ValueError(
            f"segment_lengths must have length 1 or {n}, got {len(segment_lengths)}."
        )
    sum_len = sum(lengths)
    lengths = [length * total / sum_len for length in lengths]
    segments = []

    # add logic
    # should follow the drawn lines, but each segment should have the length in lengths
    # and each segment should be at most max_angle away from parallel

    return segments


def segments_to_points(segments: List[Segment]) -> List[Point]:
    """Flatten a segment list back into an ordered list of unique points."""
    if not segments:
        return []
    pts = [segments[0][0]]
    for _, end in segments:
        pts.append(end)
    return pts
