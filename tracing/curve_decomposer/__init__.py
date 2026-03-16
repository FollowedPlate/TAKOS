"""
curve_decomposer - Draw a curve and decompose it into linear segments.
"""

from tracing.curve_decomposer.app import CurveDrawer
from tracing.curve_decomposer.decomposer import decompose_curve, Point, Segment

__all__ = ["CurveDrawer", "decompose_curve", "Point", "Segment"]
