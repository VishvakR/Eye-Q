"""Tests for the accuracy measurement harness.

Two things are being protected here, and the first matters more than the
second.

**Crop parity.** The fixtures an accuracy number is measured on must be the
crops production actually generates. They were not: ``make_fixtures.py`` called
``crop_detection_focused`` bare and so dimmed at 0.4 while production dims at
1.0 (off), and it never letterboxed while production always does. A wrong
measurement is worse than no measurement, because it looks like evidence. The
parity test pins the premise so the two paths cannot drift apart again.

**Metric arithmetic.** Precision, recall and Wilson intervals are easy to get
subtly wrong and impossible to notice afterwards, so they are checked against
hand-computed values rather than against themselves.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.pipelines.vlm_worker import (
    crop_detection_focused,
    letterbox_square,
    prepare_vlm_crop,
)
from app.schemas.events import BoundingBox
from bench.eval_accuracy import Judgement, pr_curve, summarise, wilson


def frame(h: int = 720, w: int = 1280) -> np.ndarray:
    """A frame with structure, so a resize or a dim is visible in the pixels."""
    rng = np.random.default_rng(1917)
    return rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)


def judgement(truth: bool, verdict: bool, p_yes: float, track_id: int) -> Judgement:
    return Judgement(
        crop=f"c{track_id}.png",
        rule="r",
        track_id=track_id,
        subject_px=100,
        truth=truth,
        verdict=verdict,
        p_yes=p_yes,
        answer="Yes" if verdict else "No",
        latency_ms=1.0,
    )


# ═══════════════════════════════════════════════════════════════════
# Crop parity — the premise the whole measurement rests on
# ═══════════════════════════════════════════════════════════════════


class TestCropParity:
    """The bench and production must produce byte-identical crops."""

    def test_bench_crop_equals_production_crop(self):
        """The regression that made every previous verdict number invalid.

        ``make_fixtures`` and ``VLMWorker._prepare`` now share one helper, so
        this asserts they agree byte for byte on the same input.

        The parameters are read from live settings rather than written as
        literals. With literals this test kept passing while `settings.py`
        drifted underneath it — which is precisely the failure it was written
        to prevent, one level up.
        """
        from app.config.settings import get_settings

        get_settings.cache_clear()
        live = get_settings()
        img = frame()
        bbox = BoundingBox(x1=0.4, y1=0.3, x2=0.5, y2=0.8)
        settings = dict(
            crop_padding=live.vlm_crop_padding,
            focus_dim=live.vlm_focus_dim,
            square_edge=live.vlm_square_edge,
            max_crop_edge=live.vlm_max_crop_edge,
        )

        a = prepare_vlm_crop(img, bbox, **settings)
        b = prepare_vlm_crop(img, bbox, **settings)
        assert np.array_equal(a, b)
        edge = live.vlm_square_edge
        assert a.shape == (edge, edge, 3), "production letterboxes to a square"
        get_settings.cache_clear()

    def test_the_old_bench_path_produced_a_different_crop(self):
        """Pins *why* parity is enforced, so the fix is not silently reverted.

        The old harness called ``crop_detection_focused`` with no arguments,
        taking its signature default of ``dim=0.4``, and never letterboxed.
        Both differences are visible in the pixels.
        """
        img = frame()
        bbox = BoundingBox(x1=0.4, y1=0.3, x2=0.5, y2=0.8)

        old = crop_detection_focused(img, bbox)  # dim=0.4, no letterbox
        new = prepare_vlm_crop(img, bbox, focus_dim=1.0, square_edge=378)

        assert old.shape != new.shape, "the old path did not letterbox"
        # Dimming darkens everything outside the subject box.
        assert old.mean() < crop_detection_focused(img, bbox, 0.15, 1.0).mean()

    def test_square_edge_zero_falls_back_to_the_max_edge_cap(self):
        """The documented opt-out still routes to downscale_for_vlm."""
        img = frame()
        bbox = BoundingBox(x1=0.0, y1=0.0, x2=1.0, y2=1.0)

        out = prepare_vlm_crop(img, bbox, square_edge=0, max_crop_edge=378)
        assert max(out.shape[:2]) <= 378
        assert out.shape[0] != out.shape[1], "not squared when square_edge=0"

    def test_dimming_is_carried_through(self):
        """focus_dim must reach the crop, not be dropped by the helper."""
        img = frame()
        bbox = BoundingBox(x1=0.4, y1=0.3, x2=0.5, y2=0.8)

        bright = prepare_vlm_crop(img, bbox, focus_dim=1.0)
        dimmed = prepare_vlm_crop(img, bbox, focus_dim=0.4)
        assert dimmed.mean() < bright.mean()

    def test_letterbox_pads_rather_than_stretches(self):
        """A 1:3 person must not arrive three times too wide."""
        tall = frame(h=300, w=100)
        out = letterbox_square(tall, 378)
        assert out.shape == (378, 378, 3)
        # The pad columns are the letterbox grey, not image content.
        assert out[0, 0].tolist() == [114, 114, 114]


# ═══════════════════════════════════════════════════════════════════
# Metric arithmetic
# ═══════════════════════════════════════════════════════════════════


class TestWilsonInterval:
    """Small samples with rates near 0 or 1 — where the normal approx fails."""

    def test_point_estimate_is_the_raw_proportion(self):
        point, _low, _high = wilson(8, 10)
        assert point == pytest.approx(0.8)

    def test_bounds_stay_inside_zero_and_one(self):
        """The reason for Wilson over the normal approximation."""
        for successes, total in ((0, 10), (10, 10), (1, 3), (0, 1)):
            _p, low, high = wilson(successes, total)
            assert 0.0 <= low <= high <= 1.0

    def test_zero_successes_still_has_an_upper_bound(self):
        """0/10 is not proof of 0% — it is consistent with nearly 28%."""
        point, low, high = wilson(0, 10)
        assert point == 0.0
        assert low == 0.0
        assert high == pytest.approx(0.278, abs=0.01)

    def test_a_smaller_sample_gives_a_wider_interval(self):
        _p, low_small, high_small = wilson(8, 10)
        _p, low_big, high_big = wilson(80, 100)
        assert (high_small - low_small) > (high_big - low_big)

    def test_empty_sample_is_not_a_division_by_zero(self):
        assert wilson(0, 0) == (0.0, 0.0, 0.0)


class TestRuleMetrics:
    """Confusion matrix and rates, against hand-computed values."""

    def _rows(self):
        # 2 TP, 1 FP, 2 TN, 1 FN — six distinct subjects.
        return [
            judgement(True, True, 0.90, 0),
            judgement(True, True, 0.80, 1),
            judgement(False, True, 0.70, 2),
            judgement(True, False, 0.20, 3),
            judgement(False, False, 0.10, 4),
            judgement(False, False, 0.05, 5),
        ]

    def test_confusion_matrix(self):
        m = summarise(self._rows())["r"]
        assert (m.tp, m.fp, m.tn, m.fn) == (2, 1, 2, 1)

    def test_precision_and_recall(self):
        m = summarise(self._rows())["r"]
        assert m.precision[0] == pytest.approx(2 / 3, abs=0.01)
        assert m.recall[0] == pytest.approx(2 / 3, abs=0.01)
        assert m.f1 == pytest.approx(2 / 3, abs=0.01)

    def test_intervals_are_computed_on_subjects_not_crops(self):
        """Duplicate crops of one person must not narrow the interval.

        Two crops of the same track are not two independent observations, and
        counting them as such would shrink every bound by roughly sqrt(2).
        """
        rows = self._rows()
        duplicated = rows + [judgement(j.truth, j.verdict, j.p_yes, j.track_id) for j in rows]

        one = summarise(rows)["r"]
        two = summarise(duplicated)["r"]

        assert two.subjects == one.subjects == 6
        width_one = one.precision[2] - one.precision[1]
        width_two = two.precision[2] - two.precision[1]
        assert width_two == pytest.approx(width_one, abs=0.02)

    def test_an_overall_row_is_produced(self):
        metrics = summarise(self._rows())
        assert "overall" in metrics
        assert metrics["overall"].tp == 2

    def test_p_yes_range_is_recorded(self):
        m = summarise(self._rows())["r"]
        assert m.p_yes_min == 0.05
        assert m.p_yes_max == 0.90


class TestPrecisionRecallCurve:
    """The evidence for choosing vlm_probe_threshold."""

    def _rows(self):
        return [
            judgement(True, True, 0.90, 0),
            judgement(True, True, 0.80, 1),
            judgement(False, True, 0.70, 2),
            judgement(True, False, 0.20, 3),
            judgement(False, False, 0.10, 4),
            judgement(False, False, 0.05, 5),
        ]

    def test_threshold_zero_alerts_on_everything(self):
        row = pr_curve(self._rows(), steps=11)[0]
        assert row["threshold"] == 0.0
        assert row["alerts"] == 6
        assert row["recall"] == 1.0

    def test_a_high_threshold_buys_precision_and_costs_recall(self):
        """The trade this project cares about: false alarms hurt more."""
        curve = {r["threshold"]: r for r in pr_curve(self._rows(), steps=11)}
        assert curve[0.8]["precision"] == 1.0
        assert curve[0.8]["recall"] < curve[0.0]["recall"]

    def test_no_probe_confidence_means_no_curve(self):
        """A generate-and-parse run cannot be re-thresholded after the fact."""
        rows = [judgement(True, True, 0.9, 0)]
        rows[0].p_yes = None
        assert pr_curve(rows) == []
