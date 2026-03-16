
import math

from curve_decomposer import CurveDrawer
from curve_decomposer import decompose_curve, Point, Segment
from typing import List, Optional, Callable


def on_decompose(segments):
    print(f"\n=== {len(segments)} segments ===")
    for i, (start, end) in enumerate(segments):
        length = math.hypot(end[0] - start[0], end[1] - start[1])
        print(f"  [{i:>2}] ({start[0]:7.2f}, {start[1]:7.2f}) → "
              f"({end[0]:7.2f}, {end[1]:7.2f})  len={length:.2f}px")

def generate_line_relative_lengths(b_val:float = .13, num_segments: int = 20) -> List[float]:
    return [(1+b_val)**i for i in range(num_segments)]

if __name__ == "__main__":
    n = 20
    b = .13 # .13 is the b value r=a*e^(b*theta)
    max_angle = math.radians(30) # 30 deg in either CW or CCW
    def decompose(points: List[Point]) -> List[Segment]:
        return decompose_curve(points,n, generate_line_relative_lengths(b, n), max_angle)

    drawer = CurveDrawer(on_decompose=on_decompose, decomposer=decompose)
    drawer.run()

    # After window closes, the segments are accessible programmatically:
    segs = drawer.segments
    if segs:
        print(f"\nFinal segments available: {len(segs)}")
