"""
app.py - Tkinter drawing window for CurveDrawer.
"""

import tkinter as tk
from tkinter import messagebox
from typing import List, Optional, Callable

from .decomposer import decompose_curve, segments_to_points, Segment, Point

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
BG          = "#fdf6ee"
CANVAS_BG   = "#fffdf8"
CURVE_COL   = "#3ffffa"
SEG_COL     = "#e8734a"
NODE_COL    = "#f5a623"
GRID_COL    = "#e8dfd2"

class CurveDrawer:
    """
    Opens a Tk window for freehand curve drawing and linear-segment
    decomposition.

    Parameters
    ----------
    width, height : int
        Canvas dimensions in pixels.
    on_decompose : callable, optional
        Called with the list of segments every time the curve is decomposed.
        Signature: ``on_decompose(segments: list[((x0,y0),(x1,y1))])``
    """

    def __init__(
        self,
        decomposer: Callable[[List[Point]], List[Segment]],
        width: int = 900,
        height: int = 650,
        on_decompose: Optional[Callable[[List[Segment]], None]] = None,
    ):
        self.width = width
        self.height = height
        self.on_decompose = on_decompose
        self.decomposer = decomposer

        self._points: List[Point] = []
        self._segments: List[Segment] = []
        self._drawing = False

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        self.root = tk.Tk()
        self.root.title("TAKOS")
        self.root.configure(bg=BG)
        self.root.resizable(False, False)

        # ── Top toolbar ──────────────────────────────────────────────
        self._n_var = tk.IntVar(value=20)

        # ── Canvas ───────────────────────────────────────────────────
        canvas_frame = tk.Frame(self.root, bg=BG, padx=10, pady=4)
        canvas_frame.pack()

        self.canvas = tk.Canvas(
            canvas_frame,
            width=self.width, height=self.height,
            bg=CANVAS_BG, highlightthickness=1, cursor="crosshair",
        )
        self.canvas.pack()
        self._draw_grid()

        self.canvas.bind("<ButtonPress-1>",   self._on_press)
        self.canvas.bind("<B1-Motion>",       self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

    def _draw_grid(self, spacing: int = 50):
        r = 2
        for x in range(0, self.width, spacing):
            for y in range(0, self.height, spacing):
                self.canvas.create_oval(
                    x - r, y - r, x + r, y + r,
                    fill=GRID_COL, outline="",
                )


    # ------------------------------------------------------------------
    # Mouse events
    # ------------------------------------------------------------------

    def _on_press(self, event):
        self._clear_canvas_drawings()
        self._points = [(event.x, event.y)]
        self._segments = []
        self._drawing = True

    def _on_drag(self, event):
        if not self._drawing:
            return
        pt = (event.x, event.y)
        prev = self._points[-1]
        if abs(pt[0] - prev[0]) > 1 or abs(pt[1] - prev[1]) > 1:
            self._points.append(pt)
            self.canvas.create_line(
                prev[0], prev[1], pt[0], pt[1],
                fill=CURVE_COL, width=2, smooth=True, tags="curve",
            )

    def _on_release(self, event):
        self._drawing = False
        if len(self._points) >= 2:
            self._do_decompose()

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _do_decompose(self):
        if len(self._points) < 2:
            messagebox.showwarning("No curve", "Please draw a curve first.")
            return

        try:
            self._segments = self.decomposer(self._points)

        except ValueError as exc:
            messagebox.showerror("Decomposition error", str(exc))
            return

        self._redraw_segments()

        if self.on_decompose:
            self.on_decompose(self._segments)

    def _clear(self):
        self._points = []
        self._segments = []
        self._clear_canvas_drawings()

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def _clear_canvas_drawings(self):
        """Remove all drawn content and redraw the background grid."""
        self.canvas.delete("all")
        self._draw_grid()

    def _redraw_segments(self):
        self.canvas.delete("segment")
        self.canvas.delete("node")

        pts = segments_to_points(self._segments)
        for i, ((x0, y0), (x1, y1)) in enumerate(self._segments):
            # Segment line
            self.canvas.create_line(
                x0, y0, x1, y1,
                fill=SEG_COL, width=2.5, tags="segment",
            )

        # Nodes
        r = 4
        for i, (x, y) in enumerate(pts):
            self.canvas.create_oval(
                x - r, y - r, x + r, y + r,
                fill=NODE_COL, outline="", tags="node",
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def points(self) -> List[Point]:
        """Raw drawn points (dense polyline)."""
        return list(self._points)

    @property
    def segments(self) -> List[Segment]:
        """Last computed decomposition segments."""
        return list(self._segments)

    def run(self):
        """Start the Tk main loop (blocking)."""
        self.root.mainloop()