"""Tests for the FastAPI control plane.

Runs the real lifespan with the video pipeline disabled, so the REST surface
and rule-synchronisation logic are exercised without loading YOLO or the VLM.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import httpx
import pytest

from app.api.server import _parse_labels, _parse_roi, create_app, lifespan, sync_rules
from app.config.settings import get_settings


@pytest.fixture
async def client(tmp_path, monkeypatch):
    """An HTTP client bound to an app whose lifespan has actually run.

    ``httpx.ASGITransport`` does not trigger lifespan events, so the context
    manager is driven explicitly — otherwise ``app.state.db`` would not exist.
    """
    monkeypatch.setenv("EYEQ_DATABASE_PATH", str(tmp_path / "api_test.db"))
    monkeypatch.setenv("EYEQ_ENABLE_PIPELINE", "false")
    get_settings.cache_clear()

    app = create_app()
    async with lifespan(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            c.app = app
            yield c

    get_settings.cache_clear()


SAMPLE_RULE = {
    "camera_id": "default",
    "name": "Fire exit check",
    "prompt": "Is a pallet blocking the fire exit?",
    "target_labels": ["person", "forklift"],
    "roi_json": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]],
    "confidence": 0.4,
    "cooldown_s": 3.0,
    "enabled": True,
}


class TestRulesCRUD:
    """Rules are the user's primary artifact — the full cycle must round-trip."""

    async def test_create_and_read_back(self, client):
        response = await client.post("/api/rules", json=SAMPLE_RULE)
        assert response.status_code == 201

        created = response.json()
        assert created["name"] == "Fire exit check"
        assert json.loads(created["target_labels"]) == ["person", "forklift"]
        assert created["confidence"] == pytest.approx(0.4)

        fetched = await client.get(f"/api/rules/{created['id']}")
        assert fetched.status_code == 200
        assert fetched.json()["prompt"] == SAMPLE_RULE["prompt"]

    async def test_list_and_filter_by_camera(self, client):
        await client.post("/api/rules", json=SAMPLE_RULE)
        await client.post(
            "/api/rules", json={**SAMPLE_RULE, "camera_id": "other", "name": "Other"}
        )

        assert len((await client.get("/api/rules")).json()) == 2

        filtered = (await client.get("/api/rules", params={"camera_id": "other"})).json()
        assert len(filtered) == 1
        assert filtered[0]["name"] == "Other"

    async def test_partial_update(self, client):
        rule_id = (await client.post("/api/rules", json=SAMPLE_RULE)).json()["id"]

        response = await client.put(f"/api/rules/{rule_id}", json={"enabled": False})
        assert response.status_code == 200
        assert response.json()["enabled"] == 0
        # Untouched fields survive the partial update.
        assert response.json()["prompt"] == SAMPLE_RULE["prompt"]

    async def test_delete(self, client):
        rule_id = (await client.post("/api/rules", json=SAMPLE_RULE)).json()["id"]

        assert (await client.delete(f"/api/rules/{rule_id}")).status_code == 204
        assert (await client.get(f"/api/rules/{rule_id}")).status_code == 404

    async def test_missing_rule_returns_404(self, client):
        assert (await client.get("/api/rules/4242")).status_code == 404
        assert (await client.put("/api/rules/4242", json={"name": "x"})).status_code == 404
        assert (await client.delete("/api/rules/4242")).status_code == 404


class TestAlertsEndpoint:
    """Alert history drives the dashboard feed and must paginate correctly."""

    async def test_empty_history(self, client):
        body = (await client.get("/api/alerts")).json()
        assert body["alerts"] == []
        assert body["total"] == 0

    async def test_pagination_envelope(self, client):
        rule_id = (await client.post("/api/rules", json=SAMPLE_RULE)).json()["id"]
        db = client.app.state.db

        for i in range(5):
            await db.create_alert(
                rule_id=rule_id,
                camera_id="default",
                timestamp=100.0 + i,
                verdict=i % 2 == 0,
                raw_response=f"Response {i}",
                inference_ms=500.0,
                bbox_json={"x1": 0.1, "y1": 0.1, "x2": 0.2, "y2": 0.2},
            )

        body = (await client.get("/api/alerts", params={"limit": 2})).json()
        assert body["total"] == 5
        assert len(body["alerts"]) == 2
        assert body["limit"] == 2

        page_two = (await client.get("/api/alerts", params={"limit": 2, "offset": 2})).json()
        assert len(page_two["alerts"]) == 2
        assert page_two["alerts"][0]["id"] != body["alerts"][0]["id"]

    async def test_missing_alert_returns_404(self, client):
        assert (await client.get("/api/alerts/999")).status_code == 404


class TestStatusEndpoint:
    """The dashboard polls this every 3s; its shape is a contract."""

    async def test_shape(self, client):
        body = (await client.get("/api/status")).json()

        for key in ("uptime_seconds", "buffer", "watchdog", "vlm", "database"):
            assert key in body

        assert body["pipeline_enabled"] is False
        assert body["watchdog"]["is_running"] is False
        assert body["database"]["active_rules"] == 0

    async def test_active_rules_tracks_creation(self, client):
        await client.post("/api/rules", json=SAMPLE_RULE)
        body = (await client.get("/api/status")).json()
        assert body["database"]["active_rules"] == 1


class TestStreamEndpoint:
    """Both live-view endpoints must fail cleanly when no pipeline is running."""

    async def test_mjpeg_returns_503_without_pipeline(self, client):
        response = await client.get("/api/stream/default.mjpg")
        assert response.status_code == 503

    async def test_snapshot_returns_503_without_pipeline(self, client):
        """The fallback the dashboard polls when MJPEG fails to render."""
        response = await client.get("/api/stream/default/frame.jpg")
        assert response.status_code == 503


class TestRuleParsing:
    """Column decoding guards the sync path against malformed stored JSON."""

    def test_parse_labels(self):
        assert _parse_labels({"target_labels": '["person"]'}) == ["person"]
        assert _parse_labels({"target_labels": "[]"}) == []
        assert _parse_labels({"target_labels": "not json"}) == []
        assert _parse_labels({"target_labels": '{"a": 1}'}) == []

    def test_parse_roi(self):
        roi = _parse_roi({"roi_json": "[[0,0],[1,0],[1,1]]", "name": "zone"})
        assert roi is not None
        assert len(roi.vertices) == 3
        assert roi.contains_point(0.9, 0.5)

        # Fewer than 3 points cannot form a polygon — treat as full frame.
        assert _parse_roi({"roi_json": "[[0,0],[1,1]]", "name": "bad"}) is None
        assert _parse_roi({"roi_json": "garbage", "name": "bad"}) is None


class TestRuleSync:
    """Rules must actually reach both stages — the bug that silenced prompts."""

    async def test_sync_pushes_prompts_and_roi_to_the_vlm(self, client):
        vlm_worker = MagicMock()
        client.app.state.vlm_worker = vlm_worker

        await client.post("/api/rules", json=SAMPLE_RULE)

        vlm_worker.clear_rules.assert_called()
        assert vlm_worker.add_rule.call_count == 2  # one per target label

        labels = {c.kwargs["label"] for c in vlm_worker.add_rule.call_args_list}
        assert labels == {"person", "forklift"}

        call = vlm_worker.add_rule.call_args_list[0]
        assert call.kwargs["prompt"] == SAMPLE_RULE["prompt"]
        assert call.kwargs["roi"] is not None

    async def test_sync_narrows_watchdog_to_the_label_union(self, client):
        watchdog = MagicMock()
        client.app.state.watchdog = watchdog

        await client.post("/api/rules", json=SAMPLE_RULE)

        watchdog.update_target_labels.assert_called_with({"person", "forklift"})
        # Coarsest thresholds across rules, so Stage 1 never filters out
        # something a rule still wants to see.
        watchdog.update_confidence.assert_called_with(0.4)
        watchdog.update_cooldown.assert_called_with(3.0)

    async def test_rule_without_labels_opens_the_watchdog_to_all(self, client):
        watchdog = MagicMock()
        client.app.state.watchdog = watchdog

        await client.post("/api/rules", json={**SAMPLE_RULE, "target_labels": []})

        watchdog.update_target_labels.assert_called_with(None)

    async def test_no_enabled_rules_triggers_on_nothing(self, client):
        """No rule means no question — Stage 2 must stay asleep.

        An empty set filters every label, unlike None which passes them all.
        Getting this backwards made the appliance run a generic prompt against
        every detection the moment a user disabled their only rule.
        """
        watchdog = MagicMock()
        client.app.state.watchdog = watchdog

        await sync_rules(client.app)

        watchdog.update_target_labels.assert_called_with(set())

    async def test_disabling_the_only_rule_stops_triggering(self, client):
        rule_id = (await client.post("/api/rules", json=SAMPLE_RULE)).json()["id"]

        watchdog = MagicMock()
        client.app.state.watchdog = watchdog
        await client.put(f"/api/rules/{rule_id}", json={"enabled": False})

        watchdog.update_target_labels.assert_called_with(set())

    async def test_disabled_rules_are_not_synced(self, client):
        vlm_worker = MagicMock()
        client.app.state.vlm_worker = vlm_worker

        await client.post("/api/rules", json={**SAMPLE_RULE, "enabled": False})

        vlm_worker.add_rule.assert_not_called()

    async def test_deleting_the_last_rule_restores_defaults(self, client):
        rule_id = (await client.post("/api/rules", json=SAMPLE_RULE)).json()["id"]

        watchdog = MagicMock()
        client.app.state.watchdog = watchdog
        await client.delete(f"/api/rules/{rule_id}")

        settings = client.app.state.settings
        watchdog.update_target_labels.assert_called_with(set())
        watchdog.update_confidence.assert_called_with(settings.yolo_confidence)

    async def test_sync_is_safe_without_components(self, client):
        """A pipeline-less deployment must not crash on rule changes."""
        client.app.state.watchdog = None
        client.app.state.vlm_worker = None
        await sync_rules(client.app)  # Must not raise


class TestClearAllRules:
    """Bulk delete must remove rules without destroying the evidence log."""

    async def test_deletes_every_user_rule(self, client):
        for i in range(3):
            await client.post("/api/rules", json={**SAMPLE_RULE, "name": f"Rule {i}"})

        response = await client.delete("/api/rules")
        assert response.status_code == 200
        assert response.json()["deleted"] == 3

        remaining = (await client.get("/api/rules")).json()
        # Only the sentinel survives, and it stays disabled.
        assert all(r["id"] == 0 and r["enabled"] == 0 for r in remaining)

    async def test_alert_history_survives(self, client):
        """`alerts.rule_id` cascades on delete — history must be reassigned."""
        rule_id = (await client.post("/api/rules", json=SAMPLE_RULE)).json()["id"]
        db = client.app.state.db

        await db.create_alert(
            rule_id=rule_id,
            camera_id="default",
            timestamp=1.0,
            verdict=True,
            raw_response="Yes",
            inference_ms=500.0,
            bbox_json={"x1": 0.1, "y1": 0.1, "x2": 0.2, "y2": 0.2},
        )

        await client.delete("/api/rules")

        alerts = (await client.get("/api/alerts")).json()
        assert alerts["total"] == 1, "Clearing rules destroyed the alert history"
        assert alerts["alerts"][0]["rule_id"] == 0

    async def test_resets_the_watchdog_filters(self, client):
        """Clearing rules must widen Stage 1 back to its defaults."""
        await client.post("/api/rules", json=SAMPLE_RULE)

        watchdog = MagicMock()
        client.app.state.watchdog = watchdog
        await client.delete("/api/rules")

        watchdog.update_target_labels.assert_called_with(set())
        watchdog.update_confidence.assert_called_with(
            client.app.state.settings.yolo_confidence
        )

    async def test_is_safe_when_no_rules_exist(self, client):
        response = await client.delete("/api/rules")
        assert response.status_code == 200
        assert response.json()["deleted"] == 0
