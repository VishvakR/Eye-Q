"""Region of Interest (ROI) geometry for spatial filtering.

Provides a polygon-based ROI that determines whether a detected object
falls within the area of interest in a camera's field of view. All
coordinates use **normalized** values (0.0–1.0) for resolution independence.

The core algorithm is ray-casting (point-in-polygon), which handles
arbitrary convex and concave polygons.

Example:
    >>> roi = RegionOfInterest.from_rect(0.1, 0.1, 0.9, 0.9)  # Exclude edges
    >>> roi.contains_point(0.5, 0.5)
    True
    >>> roi.contains_point(0.05, 0.05)
    False
    >>> from app.schemas.events import BoundingBox
    >>> bbox = BoundingBox(x1=0.2, y1=0.3, x2=0.4, y2=0.6)
    >>> roi.contains_bbox_center(bbox)
    True
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from app.schemas.events import BoundingBox

logger = logging.getLogger(__name__)


@dataclass
class RegionOfInterest:
    """A polygon defining the area of interest within a camera's field of view.

    Coordinates are normalized [0.0, 1.0] relative to frame dimensions.
    The polygon is defined by an ordered list of ``(x, y)`` vertices.

    Attributes:
        vertices: Ordered list of ``(x, y)`` tuples defining the polygon.
                  Must have at least 3 vertices to form a valid polygon.
        name: Optional human-readable name for the ROI.
    """

    vertices: list[tuple[float, float]]
    name: str = field(default="default")

    def __post_init__(self) -> None:
        """Validate the polygon after initialization."""
        if len(self.vertices) < 3:
            raise ValueError(
                f"ROI requires at least 3 vertices, got {len(self.vertices)}."
            )

        # Validate all coordinates are in [0, 1]
        for i, (x, y) in enumerate(self.vertices):
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                raise ValueError(
                    f"Vertex {i} ({x}, {y}) is out of normalized range [0.0, 1.0]."
                )

    def contains_point(self, x: float, y: float) -> bool:
        """Test if a point is inside the polygon using the ray-casting algorithm.

        Casts a horizontal ray from the test point to the right and counts
        how many polygon edges it crosses. An odd count means the point
        is inside; an even count means it's outside.

        Args:
            x: Normalized x-coordinate of the point.
            y: Normalized y-coordinate of the point.

        Returns:
            True if the point is inside the polygon.
        """
        n = len(self.vertices)
        inside = False

        j = n - 1
        for i in range(n):
            xi, yi = self.vertices[i]
            xj, yj = self.vertices[j]

            # Check if the ray crosses this edge
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside

            j = i

        return inside

    def contains_bbox_center(self, bbox: BoundingBox) -> bool:
        """Check if the center of a bounding box falls within the ROI.

        This is the primary method used by the YOLO watchdog to determine
        whether a detection is "inside" the area of interest. Using the
        center point (rather than full overlap) is fast and works well
        for typical surveillance scenarios.

        Args:
            bbox: Normalized bounding box from a detection event.

        Returns:
            True if the bbox center is inside the ROI polygon.
        """
        cx, cy = bbox.center
        return self.contains_point(cx, cy)

    def intersects_bbox(self, bbox: BoundingBox, min_overlap: float = 0.1) -> bool:
        """Check if a bounding box overlaps the ROI above a minimum threshold.

        Uses a sampling-based approximation: tests a grid of points inside
        the bounding box and checks what fraction fall within the ROI.

        This is more accurate than ``contains_bbox_center`` for objects
        that straddle the ROI boundary but is slightly more expensive.

        Args:
            bbox: Normalized bounding box from a detection event.
            min_overlap: Minimum fraction of sample points that must be
                         inside the ROI (0.0–1.0). Default 0.1 (10%).

        Returns:
            True if the overlap fraction exceeds ``min_overlap``.
        """
        # Sample a 5×5 grid within the bounding box
        steps = 5
        total_points = steps * steps
        inside_count = 0

        for ix in range(steps):
            for iy in range(steps):
                px = bbox.x1 + (bbox.x2 - bbox.x1) * (ix + 0.5) / steps
                py = bbox.y1 + (bbox.y2 - bbox.y1) * (iy + 0.5) / steps
                if self.contains_point(px, py):
                    inside_count += 1

        overlap_ratio = inside_count / total_points
        return overlap_ratio >= min_overlap

    @classmethod
    def full_frame(cls) -> RegionOfInterest:
        """Create a ROI covering the entire frame.

        Returns:
            ROI with vertices at the four corners of the normalized frame.
        """
        return cls(
            vertices=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
            name="full_frame",
        )

    @classmethod
    def from_rect(
        cls,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        name: str = "rect",
    ) -> RegionOfInterest:
        """Create a rectangular ROI from corner coordinates.

        Args:
            x1: Left edge (normalized).
            y1: Top edge (normalized).
            x2: Right edge (normalized).
            y2: Bottom edge (normalized).
            name: Optional name for the ROI.

        Returns:
            Rectangular ROI polygon.
        """
        return cls(
            vertices=[(x1, y1), (x2, y1), (x2, y2), (x1, y2)],
            name=name,
        )

    @classmethod
    def from_json(cls, data: list[list[float]], name: str = "json") -> RegionOfInterest:
        """Create a ROI from a JSON-compatible list of [x, y] pairs.

        Args:
            data: List of ``[x, y]`` coordinate pairs.
            name: Optional name for the ROI.

        Returns:
            ROI polygon from the parsed coordinates.
        """
        vertices = [(float(pt[0]), float(pt[1])) for pt in data]
        return cls(vertices=vertices, name=name)

    def to_json(self) -> list[list[float]]:
        """Serialize the ROI vertices to a JSON-compatible list.

        Returns:
            List of ``[x, y]`` coordinate pairs.
        """
        return [[x, y] for x, y in self.vertices]

    def __repr__(self) -> str:
        return (
            f"RegionOfInterest(name='{self.name}', "
            f"vertices={len(self.vertices)} points)"
        )
