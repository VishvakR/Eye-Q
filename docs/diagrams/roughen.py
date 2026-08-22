"""Hand-drawn (Excalidraw-style) SVG primitives.

Every shape is emitted as real path data with per-point jitter and a doubled
outline, which is what gives Excalidraw its look. Deliberately *not* an SVG
``feTurbulence``/``feDisplacementMap`` filter: filters rasterise when Chrome
prints, so a filtered diagram goes soft in the PDF while jittered paths stay
vector and crisp at any zoom.

The line algorithm follows Rough.js — a cubic Bezier whose two control points
are displaced along the stroke, plus a "bowing" term perpendicular to it. Each
stroke is drawn twice with independent noise so the two passes cross, the way a
pen does.

Seeded from a fixed value, so a rebuild produces byte-identical SVG and the
diagrams do not churn in git.

Usage:
    >>> c = Canvas(900, 520)
    >>> c.rect(40, 40, 200, 90, label="YOLOv8n watchdog")
    >>> c.arrow(240, 85, 400, 85, label="ZMQ :5557")
    >>> open("fig.svg", "w").write(c.render())
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

# ── Visual constants ─────────────────────────────────────────────

INK = "#1b1b1f"          # near-black, the only structural colour
ACCENT = "#c2410c"       # warm orange, reserved for the trigger/verdict path
MUTED = "#6b7280"        # secondary annotation text
WASH = "#00000008"       # 3% grey fill for grouping containers
PAPER = "#ffffff"

HAND = "'Chalkboard SE','Bradley Hand','Comic Sans MS',cursive"
SANS = "'Helvetica Neue',Helvetica,Arial,sans-serif"


def esc(text: str) -> str:
    """XML-escape a label."""
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )


# ── Rough path generation ────────────────────────────────────────


class _Pen:
    """Seeded noise source shared by every shape on one canvas."""

    def __init__(self, seed: int = 20260821) -> None:
        self._rng = random.Random(seed)

    def offset(self, amount: float) -> float:
        """A random displacement in [-amount, +amount]."""
        return self._rng.uniform(-amount, amount)


def _rough_line(
    pen: _Pen,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    roughness: float = 1.0,
    bowing: float = 1.0,
    move: bool = True,
) -> str:
    """One hand-drawn stroke from (x1,y1) to (x2,y2), as path commands.

    Args:
        pen: Noise source.
        roughness: Multiplier on the random displacement.
        bowing: How much the stroke bows away from true.
        move: Emit a leading ``M``. False continues an open subpath.
    """
    length = math.hypot(x2 - x1, y2 - y1)
    # Long strokes wander less per unit length than short ones, or a long
    # container edge reads as a wobbly mess rather than a drawn line.
    roughness = roughness * (0.6 if length > 320 else 1.0)
    offset = min(2.0, length / 10.0) * roughness

    diverge = 0.2 + pen.offset(0.1)
    mid_x = bowing * offset * (y2 - y1) / 220.0
    mid_y = bowing * offset * (x2 - x1) / -220.0

    def jitter() -> float:
        return pen.offset(offset)

    c1x = mid_x + x1 + (x2 - x1) * diverge + jitter()
    c1y = mid_y + y1 + (y2 - y1) * diverge + jitter()
    c2x = mid_x + x1 + 2 * (x2 - x1) * diverge + jitter()
    c2y = mid_y + y1 + 2 * (y2 - y1) * diverge + jitter()
    ex = x2 + jitter() * 0.6
    ey = y2 + jitter() * 0.6

    head = f"M{x1 + jitter() * 0.5:.1f},{y1 + jitter() * 0.5:.1f}" if move else ""
    return f"{head}C{c1x:.1f},{c1y:.1f} {c2x:.1f},{c2y:.1f} {ex:.1f},{ey:.1f}"


def _rounded_corners(
    x: float, y: float, w: float, h: float, r: float
) -> list[tuple[float, float]]:
    """Corner waypoints for a rounded rectangle, walked clockwise."""
    r = min(r, w / 2, h / 2)
    return [
        (x + r, y), (x + w - r, y),
        (x + w, y + r), (x + w, y + h - r),
        (x + w - r, y + h), (x + r, y + h),
        (x, y + h - r), (x, y + r),
    ]


# ── Canvas ───────────────────────────────────────────────────────


@dataclass
class Canvas:
    """Accumulates SVG fragments and renders a complete document.

    Args:
        width: viewBox width in design units.
        height: viewBox height in design units.
        seed: Noise seed. Change it to reroll the hand-drawn wobble.
    """

    width: float
    height: float
    seed: int = 20260821
    _parts: list[str] = field(default_factory=list)
    _pen: _Pen = field(init=False)

    def __post_init__(self) -> None:
        self._pen = _Pen(self.seed)

    # ── Shapes ───────────────────────────────────────────────────

    def rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        *,
        radius: float = 8.0,
        stroke: str = INK,
        width: float = 1.6,
        fill: str | None = None,
        dashed: bool = False,
        passes: int = 2,
    ) -> "Canvas":
        """A hand-drawn rounded rectangle."""
        pts = _rounded_corners(x, y, w, h, radius)

        if fill:
            self._parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
                f'rx="{radius:.1f}" fill="{fill}"/>'
            )

        dash = ' stroke-dasharray="7 5"' if dashed else ""
        # Two passes at slightly different pressure reads as a pen going round
        # twice; one pass reads as a shaky computer line.
        for pass_no in range(passes):
            d = []
            for i in range(0, len(pts), 2):
                ax, ay = pts[i]
                bx, by = pts[(i + 1) % len(pts)]
                d.append(_rough_line(self._pen, ax, ay, bx, by, roughness=0.9))
                # Corner arc, drawn as a short straight hop — at this radius the
                # difference is invisible and it keeps the path light.
                cx, cy = pts[(i + 2) % len(pts)]
                d.append(_rough_line(self._pen, bx, by, cx, cy, roughness=0.7, bowing=2.4))
            self._parts.append(
                f'<path d="{"".join(d)}" fill="none" stroke="{stroke}" '
                f'stroke-width="{width * (1.0 if pass_no == 0 else 0.72):.2f}" '
                f'stroke-linecap="round"{dash}/>'
            )
        return self

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        stroke: str = INK,
        width: float = 1.5,
        dashed: bool = False,
        passes: int = 2,
    ) -> "Canvas":
        """A hand-drawn straight stroke."""
        dash = ' stroke-dasharray="6 5"' if dashed else ""
        for pass_no in range(passes):
            d = _rough_line(self._pen, x1, y1, x2, y2)
            self._parts.append(
                f'<path d="{d}" fill="none" stroke="{stroke}" '
                f'stroke-width="{width * (1.0 if pass_no == 0 else 0.7):.2f}" '
                f'stroke-linecap="round"{dash}/>'
            )
        return self

    def arrow(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        stroke: str = INK,
        width: float = 1.5,
        head: float = 9.0,
        dashed: bool = False,
    ) -> "Canvas":
        """A hand-drawn arrow, shaft plus a two-stroke open head."""
        self.line(x1, y1, x2, y2, stroke=stroke, width=width, dashed=dashed)

        angle = math.atan2(y2 - y1, x2 - x1)
        for sign in (1, -1):
            spread = angle + sign * 2.6
            hx = x2 + head * math.cos(spread)
            hy = y2 + head * math.sin(spread)
            self.line(x2, y2, hx, hy, stroke=stroke, width=width, passes=1)
        return self

    def elbow(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        stroke: str = INK,
        width: float = 1.5,
        vertical_first: bool = False,
        arrow: bool = True,
    ) -> "Canvas":
        """An L-shaped connector, for routing around boxes."""
        if vertical_first:
            self.line(x1, y1, x1, y2, stroke=stroke, width=width)
            if arrow:
                self.arrow(x1, y2, x2, y2, stroke=stroke, width=width)
            else:
                self.line(x1, y2, x2, y2, stroke=stroke, width=width)
        else:
            self.line(x1, y1, x2, y1, stroke=stroke, width=width)
            if arrow:
                self.arrow(x2, y1, x2, y2, stroke=stroke, width=width)
            else:
                self.line(x2, y1, x2, y2, stroke=stroke, width=width)
        return self

    # ── Text ─────────────────────────────────────────────────────

    def text(
        self,
        x: float,
        y: float,
        content: str,
        *,
        size: float = 14.0,
        anchor: str = "middle",
        fill: str = INK,
        family: str = HAND,
        weight: str = "normal",
        italic: bool = False,
        leading: float | None = None,
    ) -> "Canvas":
        """Draw text. ``\\n`` in ``content`` starts a new line."""
        lines = content.split("\n")
        step = leading if leading is not None else size * 1.32
        style = ' font-style="italic"' if italic else ""
        spans = []
        for i, line in enumerate(lines):
            dy = 0 if i == 0 else step
            spans.append(f'<tspan x="{x:.1f}" dy="{dy:.1f}">{esc(line)}</tspan>')
        self._parts.append(
            f'<text x="{x:.1f}" y="{y:.1f}" font-family="{family}" '
            f'font-size="{size:.1f}" font-weight="{weight}" fill="{fill}" '
            f'text-anchor="{anchor}"{style}>{"".join(spans)}</text>'
        )
        return self

    def node(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        label: str,
        *,
        sub: str = "",
        size: float = 14.0,
        fill: str | None = PAPER,
        stroke: str = INK,
        radius: float = 8.0,
        dashed: bool = False,
    ) -> "Canvas":
        """A labelled box — the workhorse. ``sub`` prints smaller underneath."""
        self.rect(x, y, w, h, radius=radius, fill=fill, stroke=stroke, dashed=dashed)
        lines = label.split("\n")
        block = len(lines) * size * 1.28 + (size * 1.05 if sub else 0)
        top = y + (h - block) / 2 + size * 0.92
        self.text(x + w / 2, top, label, size=size, fill=stroke)
        if sub:
            self.text(
                x + w / 2,
                top + (len(lines) - 1) * size * 1.32 + size * 1.18,
                sub,
                size=size * 0.78,
                fill=MUTED,
            )
        return self

    def callout(
        self,
        x: float,
        y: float,
        w: float,
        text: str,
        *,
        point_to: tuple[float, float] | None = None,
        size: float = 12.5,
        padding: float = 9.0,
        align: str = "left",
    ) -> "Canvas":
        """A dashed annotation box with an optional leader line.

        This is the grammar from the reference sketch: the note lives outside
        the diagram and reaches in with a thin line, so the structure stays
        readable on its own.
        """
        lines = text.split("\n")
        h = len(lines) * size * 1.3 + padding * 2 - size * 0.3

        if point_to is not None:
            # Leave from whichever edge faces the target.
            tx, ty = point_to
            ax = x if tx < x else (x + w if tx > x + w else x + w / 2)
            ay = y + h / 2 if x <= tx <= x + w else max(y, min(y + h, ty))
            if x <= tx <= x + w:
                ay = y if ty < y else y + h
            self.line(ax, ay, tx, ty, stroke=MUTED, width=1.0, dashed=True, passes=1)

        self.rect(x, y, w, h, radius=5, fill=PAPER, stroke=MUTED, width=1.1, dashed=True)
        tx = x + padding if align == "left" else x + w / 2
        self.text(
            tx,
            y + padding + size * 0.85,
            text,
            size=size,
            anchor="start" if align == "left" else "middle",
            fill=MUTED,
            leading=size * 1.3,
        )
        return self

    def slot_strip(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        labels: list[str],
        *,
        gap: float = 7.0,
        tail: str = "",
    ) -> "Canvas":
        """A row of small boxes — the ring-buffer frame slots from the sketch."""
        count = len(labels)
        slot_w = (w - gap * (count - 1)) / count
        for i, label in enumerate(labels):
            sx = x + i * (slot_w + gap)
            self.rect(sx, y, slot_w, h, radius=4, fill=PAPER)
            self.text(sx + slot_w / 2, y + h / 2 + 4.5, label, size=12.5)
        if tail:
            self.text(x + w / 2, y + h + 17, tail, size=11.5, fill=MUTED)
        return self

    def camera_icon(self, x: float, y: float, scale: float = 1.0) -> "Canvas":
        """A small hand-drawn CCTV camera glyph, as in the reference."""
        s = scale
        self.rect(x, y, 30 * s, 15 * s, radius=3, fill=PAPER)
        self.line(x + 30 * s, y + 4 * s, x + 40 * s, y + 1 * s)
        self.line(x + 40 * s, y + 1 * s, x + 40 * s, y + 14 * s)
        self.line(x + 40 * s, y + 14 * s, x + 30 * s, y + 11 * s)
        self.line(x + 12 * s, y + 15 * s, x + 12 * s, y + 23 * s)
        self.line(x + 4 * s, y + 23 * s, x + 20 * s, y + 23 * s)
        return self

    def raw(self, fragment: str) -> "Canvas":
        """Append a literal SVG fragment."""
        self._parts.append(fragment)
        return self

    # ── Output ───────────────────────────────────────────────────

    def render(self) -> str:
        """The complete SVG document."""
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'viewBox="0 0 {self.width:g} {self.height:g}" '
            f'width="{self.width:g}" height="{self.height:g}" '
            f'role="img">\n'
            f'<rect width="{self.width:g}" height="{self.height:g}" fill="{PAPER}"/>\n'
            + "\n".join(self._parts)
            + "\n</svg>\n"
        )
