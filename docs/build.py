"""Assemble the Eye-Q technical documentation HTML.

Substitutes each ``{{FIG_n}}`` placeholder in ``document.html.in`` with the
corresponding SVG *inlined*. Inlining rather than <img> linking is deliberate:
it keeps the rendered PDF vector (Chrome has been known to rasterise linked
SVGs) and leaves the HTML self-contained with no external requests at all.
"""

from __future__ import annotations

import pathlib
import re

HERE = pathlib.Path(__file__).parent
FIGURES = {
    "FIG_1": "01-system-architecture.svg",
    "FIG_2": "02-workflow-dataflow.svg",
    "FIG_3": "03-agent-decision-path.svg",
    "FIG_4": "04-innovation-map.svg",
}


def inline(svg: str) -> str:
    """Strip the fixed width/height so the SVG scales to its container."""
    return re.sub(r'\s(width|height)="[\d.]+"', "", svg, count=2)


def main() -> None:
    """Write the assembled document."""
    html = (HERE / "document.html.in").read_text()

    for key, filename in FIGURES.items():
        svg = (HERE / "diagrams" / filename).read_text()
        html = html.replace("{{" + key + "}}", inline(svg))

    missing = re.findall(r"\{\{(\w+)\}\}", html)
    if missing:
        raise SystemExit(f"unsubstituted placeholders: {sorted(set(missing))}")

    out = HERE / "eye-q-technical-documentation.html"
    out.write_text(html)
    print(f"wrote {out.name} ({len(html) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
