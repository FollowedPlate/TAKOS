
import math
from math import exp

from curve_decomposer import CurveDrawer
from curve_decomposer import decompose_curve, Point, Segment
from typing import List, Optional, Callable

n = 20
max_angle = math.radians(30)  # 30 deg in either CW or CCW
b = exp(.13 * max_angle)  # scale factor = exp(.13 * del_theta)

def on_decompose(segments) -> None:
    ret = []

    for i in range(1, len(segments)):
        (x0, y0), (x1, y1) = segments[i - 1]
        (x2, y2), (x3, y3) = segments[i]

        dx0, dy0 = x1 - x0, y1 - y0
        dx1, dy1 = x3 - x2, y3 - y2

        mag0 = math.hypot(dx0, dy0)
        mag1 = math.hypot(dx1, dy1)

        if mag0 < 1e-12 or mag1 < 1e-12:
            ret.append(0.0)
            continue

        cos_a = (dx0 * dx1 + dy0 * dy1) / (mag0 * mag1)
        cos_a = max(-1.0, min(1.0, cos_a))

        # signed angle: positive = CCW turn, negative = CW turn
        cross = dx0 * dy1 - dy0 * dx1
        angle = math.copysign(math.acos(cos_a), cross)
        angle = math.degrees(angle)
        angle = int(angle*100)/100
        ret.append(angle)

    print(ret)

def generate_line_relative_lengths(b_val:float = .13, num_segments: int = 20) -> List[float]:
    return [b_val ** i for i in range(num_segments)][::-1]

if __name__ == "__main__":
    def decompose(points: List[Point]) -> List[Segment]:
        return decompose_curve(points,n, generate_line_relative_lengths(b, n), max_angle)

    drawer = CurveDrawer(on_decompose=on_decompose, decomposer=decompose)
    drawer.run()

    # After window closes, the segments are accessible programmatically:
    segs = drawer.segments
    if segs:
        print(f"\nFinal segments available: {len(segs)}")
