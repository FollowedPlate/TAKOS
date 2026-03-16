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

    # Cumulative arc-length targets on the original curve.
    # target_dists[i] is where the end of segment i *should* land if there
    # were no angle constraint.  Even when we drift off-curve we keep using
    # these as "aim points" so we always know where to head back to.
    cumulative = 0.0
    target_dists = []
    for l in lengths:
        cumulative += l
        target_dists.append(min(cumulative, total))

    start_pt: Point = points[0]
    prev_dx: Optional[float] = None   # unit direction of the last segment
    prev_dy: Optional[float] = None

    for i in range(n):
        seg_len = lengths[i]

        # ── Step 1: desired direction ──────────────────────────────────────
        # Aim from current start_pt toward the expected arc-length position on
        # the original curve.
        aim_pt = _point_at_distance(points, target_dists[i])
        dx = aim_pt[0] - start_pt[0]
        dy = aim_pt[1] - start_pt[1]
        mag = math.hypot(dx, dy)

        if mag < 1e-12:
            # start_pt is already on top of the aim point; keep previous
            # direction or fall back to the local curve tangent.
            if prev_dx is not None:
                dx, dy = prev_dx, prev_dy
            else:
                look_ahead = _point_at_distance(points, min(target_dists[i] + 1.0, total))
                look_back  = _point_at_distance(points, max(target_dists[i] - 1.0, 0.0))
                dx = look_ahead[0] - look_back[0]
                dy = look_ahead[1] - look_back[1]
                mag = math.hypot(dx, dy)
                dx, dy = (dx / mag, dy / mag) if mag > 1e-12 else (1.0, 0.0)
        else:
            dx, dy = dx / mag, dy / mag

        # ── Step 2: clamp direction to max_angle cone ──────────────────────
        if max_angle is not None and prev_dx is not None:
            cos_a = max(-1.0, min(1.0, prev_dx * dx + prev_dy * dy))
            turn  = math.acos(cos_a)

            if turn > max_angle + 1e-9:
                # Rotate prev_direction by ±max_angle toward the desired
                # direction. The sign of the cross product tells us which way.
                cross = prev_dx * dy - prev_dy * dx   # z of (prev × desired)
                angle = math.copysign(max_angle, cross)
                cos_r, sin_r = math.cos(angle), math.sin(angle)
                dx = cos_r * prev_dx - sin_r * prev_dy
                dy = sin_r * prev_dx + cos_r * prev_dy

        # ── Step 3: emit segment of exact length in clamped direction ──────
        end_pt: Point = (start_pt[0] + dx * seg_len,
                         start_pt[1] + dy * seg_len)
        segments.append((start_pt, end_pt))

        prev_dx, prev_dy = dx, dy
        start_pt = end_pt

    return segments


def segments_to_points(segments: List[Segment]) -> List[Point]:
    """Flatten a segment list back into an ordered list of unique points."""
    if not segments:
        return []
    pts = [segments[0][0]]
    for _, end in segments:
        pts.append(end)
    return pts
