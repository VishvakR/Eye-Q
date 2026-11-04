"""Tests for the FastAPI control plane.

Runs the real lifespan with the video pipeline disabled, so the REST surface
and rule-synchronisation logic are exercised without loading YOLO or the VLM.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock

import cv2
import httpx
import numpy as np
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

        # One atomic swap, never clear-then-add: a batch running in between
        # would see no rules and drop its subjects silently.
        vlm_worker.clear_rules.assert_not_called()
        vlm_worker.add_rule.assert_not_called()
        vlm_worker.replace_rules.assert_called_once()

        registrations = vlm_worker.replace_rules.call_args.args[0]
        assert len(registrations) == 2  # one per target label
        assert {r["label"] for r in registrations} == {"person", "forklift"}
        assert registrations[0]["prompt"] == SAMPLE_RULE["prompt"]
        assert registrations[0]["roi"] is not None

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

        vlm_worker.replace_rules.assert_called_once_with([])

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


class TestPairRequirementSync:
    """Co-occurrence requirements compose under the same coarsest-wins rule.

    A pair requirement is a *narrowing*, and there is no coarsest narrowing —
    so a single unpaired rule on the same label drops the gate. That is the
    correct outcome (enforcing it would silently blind the unpaired rule) but
    it is invisible, so it must be logged.
    """

    PHONE_RULE = {
        **SAMPLE_RULE,
        "name": "Phone use",
        "target_labels": ["person"],
        "pair_label": "cell phone",
    }

    async def test_a_pair_label_reaches_the_watchdog(self, client):
        watchdog = MagicMock()
        client.app.state.watchdog = watchdog

        await client.post("/api/rules", json=self.PHONE_RULE)

        watchdog.update_pair_requirements.assert_called_with(
            {"person": {"cell phone"}}
        )

    async def test_a_pair_label_round_trips(self, client):
        created = (await client.post("/api/rules", json=self.PHONE_RULE)).json()
        assert created["pair_label"] == "cell phone"

    async def test_a_rule_without_a_pair_label_requires_nothing(self, client):
        watchdog = MagicMock()
        client.app.state.watchdog = watchdog

        await client.post("/api/rules", json=SAMPLE_RULE)

        watchdog.update_pair_requirements.assert_called_with(None)

    async def test_an_unpaired_rule_on_the_same_label_drops_the_gate(
        self, client, caplog
    ):
        """The load-bearing degradation, and the warning that makes it visible."""
        await client.post("/api/rules", json=self.PHONE_RULE)

        watchdog = MagicMock()
        client.app.state.watchdog = watchdog

        with caplog.at_level("WARNING"):
            await client.post(
                "/api/rules",
                json={
                    **SAMPLE_RULE,
                    "name": "Forklift watch",
                    "target_labels": ["person"],
                },
            )

        watchdog.update_pair_requirements.assert_called_with(None)
        assert "Forklift watch" in caplog.text
        assert "Phone use" in caplog.text
        assert "cell phone" in caplog.text

    async def test_an_unpaired_rule_on_a_different_label_keeps_the_gate(
        self, client
    ):
        """"person + cell phone" says nothing about a forklift."""
        await client.post("/api/rules", json=self.PHONE_RULE)

        watchdog = MagicMock()
        client.app.state.watchdog = watchdog
        await client.post(
            "/api/rules",
            json={
                **SAMPLE_RULE,
                "name": "Forklift watch",
                "target_labels": ["forklift"],
            },
        )

        watchdog.update_pair_requirements.assert_called_with(
            {"person": {"cell phone"}}
        )

    async def test_a_wildcard_rule_suppresses_every_gate(self, client, caplog):
        """A rule with no label filter must see everything, gates included."""
        await client.post("/api/rules", json=self.PHONE_RULE)

        watchdog = MagicMock()
        client.app.state.watchdog = watchdog

        with caplog.at_level("WARNING"):
            await client.post(
                "/api/rules",
                json={**SAMPLE_RULE, "name": "Scene check", "target_labels": []},
            )

        watchdog.update_pair_requirements.assert_called_with(None)
        assert "Scene check" in caplog.text

    async def test_two_paired_rules_on_one_label_accept_either_partner(
        self, client
    ):
        await client.post("/api/rules", json=self.PHONE_RULE)

        watchdog = MagicMock()
        client.app.state.watchdog = watchdog
        await client.post(
            "/api/rules",
            json={
                **SAMPLE_RULE,
                "name": "Laptop use",
                "target_labels": ["person"],
                "pair_label": "laptop",
            },
        )

        watchdog.update_pair_requirements.assert_called_with(
            {"person": {"cell phone", "laptop"}}
        )

    async def test_a_disabled_paired_rule_stops_narrowing(self, client):
        rule_id = (await client.post("/api/rules", json=self.PHONE_RULE)).json()["id"]

        watchdog = MagicMock()
        client.app.state.watchdog = watchdog
        await client.put(f"/api/rules/{rule_id}", json={"enabled": False})

        watchdog.update_pair_requirements.assert_called_with(None)


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


class TestSnapshotMount:
    """Evidence JPEGs must be served from wherever the worker writes them.

    The mount used to hardcode ``data/snapshots`` while the VLM worker honoured
    ``snapshot_dir``, so every alert thumbnail 404'd the moment the setting
    pointed anywhere else — the dashboard rendered a broken-image icon with no
    error to go on.
    """

    async def _client_with_snapshot_dir(self, tmp_path, monkeypatch, snapshot_dir):
        monkeypatch.setenv("EYEQ_DATABASE_PATH", str(tmp_path / "snap_test.db"))
        monkeypatch.setenv("EYEQ_ENABLE_PIPELINE", "false")
        monkeypatch.setenv("EYEQ_SNAPSHOT_DIR", str(snapshot_dir))
        get_settings.cache_clear()
        return create_app()

    async def test_serves_from_the_configured_directory(self, tmp_path, monkeypatch):
        snapshot_dir = tmp_path / "custom_snapshots"
        snapshot_dir.mkdir()
        # Content only has to round-trip; it never gets decoded.
        (snapshot_dir / "evidence.jpg").write_bytes(b"\xff\xd8\xff\xdb-not-a-real-jpeg")

        app = await self._client_with_snapshot_dir(tmp_path, monkeypatch, snapshot_dir)
        try:
            async with lifespan(app):
                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(
                    transport=transport, base_url="http://test"
                ) as c:
                    response = await c.get("/snapshots/evidence.jpg")

            assert response.status_code == 200, (
                "Configured snapshot directory is not mounted at /snapshots"
            )
            assert response.content == b"\xff\xd8\xff\xdb-not-a-real-jpeg"
        finally:
            get_settings.cache_clear()

    async def test_directory_is_created_when_missing(self, tmp_path, monkeypatch):
        """A fresh checkout has produced no alerts, so the mount must not 500."""
        snapshot_dir = tmp_path / "not_yet_created"

        app = await self._client_with_snapshot_dir(tmp_path, monkeypatch, snapshot_dir)
        try:
            assert snapshot_dir.is_dir(), "Snapshot directory was not created eagerly"

            async with lifespan(app):
                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(
                    transport=transport, base_url="http://test"
                ) as c:
                    response = await c.get("/snapshots/absent.jpg")

            assert response.status_code == 404
        finally:
            get_settings.cache_clear()


class TestRuleScopeWarning:
    """An unscoped rule about a specific object must not fail silently.

    Leaving target_labels empty sends every detected object to the VLM. That is
    legitimate for a scene-level question, so the API informs rather than
    refuses — but it must inform.
    """

    async def test_creating_an_unscoped_rule_returns_a_warning(self, client):
        response = await client.post(
            "/api/rules",
            json={
                "camera_id": "default",
                "name": "pep",
                "prompt": "alert is person did not wear helmet",
                "target_labels": [],
                "confidence": 0.25,
                "cooldown_s": 5.0,
            },
        )
        assert response.status_code == 201
        body = response.json()
        assert "warning" in body
        assert "person" in body["warning"]

    async def test_a_scoped_rule_carries_no_warning(self, client):
        response = await client.post(
            "/api/rules",
            json={
                "camera_id": "default",
                "name": "scoped",
                "prompt": "alert is person did not wear helmet",
                "target_labels": ["person"],
                "confidence": 0.25,
                "cooldown_s": 5.0,
            },
        )
        assert response.status_code == 201
        assert "warning" not in response.json()

    async def test_a_scene_level_rule_carries_no_warning(self, client):
        response = await client.post(
            "/api/rules",
            json={
                "camera_id": "default",
                "name": "exit",
                "prompt": "Is anything blocking the fire exit?",
                "target_labels": [],
                "confidence": 0.25,
                "cooldown_s": 5.0,
            },
        )
        assert response.status_code == 201
        assert "warning" not in response.json()

    async def test_the_warning_does_not_block_the_rule(self, client):
        """The rule must still be created, and still be active."""
        created = await client.post(
            "/api/rules",
            json={
                "camera_id": "default",
                "name": "pep",
                "prompt": "Is this worker wearing a hard hat?",
                "target_labels": [],
                "confidence": 0.25,
                "cooldown_s": 5.0,
            },
        )
        rule_id = created.json()["id"]
        fetched = await client.get(f"/api/rules/{rule_id}")
        assert fetched.status_code == 200
        assert fetched.json()["enabled"] == 1

    async def test_narrowing_a_rule_clears_the_warning(self, client):
        created = await client.post(
            "/api/rules",
            json={
                "camera_id": "default",
                "name": "pep",
                "prompt": "alert is person did not wear helmet",
                "target_labels": [],
                "confidence": 0.25,
                "cooldown_s": 5.0,
            },
        )
        rule_id = created.json()["id"]

        updated = await client.put(
            f"/api/rules/{rule_id}", json={"target_labels": ["person"]}
        )
        assert updated.status_code == 200
        assert "warning" not in updated.json()

    async def test_labels_endpoint_suggests_from_a_prompt(self, client):
        response = await client.get(
            "/api/labels", params={"prompt": "Is this worker wearing a hard hat?"}
        )
        assert response.status_code == 200
        body = response.json()
        assert body["suggested"] == ["person"]
        assert "person" in body["labels"]

    async def test_labels_endpoint_without_a_prompt_suggests_nothing(self, client):
        body = (await client.get("/api/labels")).json()
        assert body["suggested"] == []
        assert len(body["labels"]) == 80


class TestPreviewStreamResilience:
    """The live feed must never go quietly blank.

    A stream carrying no parts is not an error: the response is open and valid,
    so the browser's <img> never fires onerror and the snapshot fallback never
    engages. The operator sees an empty panel with nothing to explain it, which
    is the worst possible failure for a monitoring appliance.
    """

    async def _first_part(self, app, timeout=5.0):
        """Pull one multipart chunk out of the generator, or time out."""
        from app.api.server import _mjpeg_generator

        agen = _mjpeg_generator(app)
        try:
            return await asyncio.wait_for(agen.__anext__(), timeout=timeout)
        finally:
            await agen.aclose()

    async def test_a_part_is_sent_even_with_no_frames_yet(self, client):
        """Before the first capture there is nothing to show — say so."""
        app = client.app
        app.state.buffer = None

        part = await self._first_part(app)

        assert b"--frame" in part
        assert b"Content-Type: image/jpeg" in part
        assert len(part) > 500, "Placeholder carried no image data"

    async def test_a_part_is_sent_when_the_buffer_is_empty(self, client):
        """A camera that has dropped must not silence the stream."""
        app = client.app
        buffer = MagicMock()
        buffer.read_latest.side_effect = RuntimeError("Buffer is empty")
        app.state.buffer = buffer

        part = await self._first_part(app)
        assert b"--frame" in part

    async def test_a_rendering_failure_does_not_end_the_stream(self, client):
        """One bad frame must not cost the operator the feed until they reload."""
        from app.api.server import _mjpeg_generator

        app = client.app
        buffer = MagicMock()
        buffer.read_latest.side_effect = ValueError("corrupt frame")
        app.state.buffer = buffer

        agen = _mjpeg_generator(app)
        try:
            parts = [
                await asyncio.wait_for(agen.__anext__(), timeout=5.0) for _ in range(3)
            ]
        finally:
            await agen.aclose()

        assert len(parts) == 3, "The stream died on a rendering error"
        assert all(b"--frame" in part for part in parts)

    async def test_the_stream_recovers_once_frames_return(self, client):
        """A reconnecting camera must bring the real picture back by itself."""
        from app.api.server import _mjpeg_generator

        app = client.app
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        buffer = MagicMock()
        buffer.read_latest.side_effect = [
            RuntimeError("Buffer is empty"),
            (frame, 1.0, 0),
        ]
        app.state.buffer = buffer
        app.state.watchdog = None

        agen = _mjpeg_generator(app)
        try:
            placeholder = await asyncio.wait_for(agen.__anext__(), timeout=5.0)
            recovered = await asyncio.wait_for(agen.__anext__(), timeout=5.0)
        finally:
            await agen.aclose()

        assert b"--frame" in placeholder
        assert b"--frame" in recovered
        assert placeholder != recovered, "Never switched back to the live frame"

    async def test_the_placeholder_is_a_decodable_jpeg(self, client):
        """A malformed placeholder would leave the panel as blank as no part."""
        from app.api.server import _placeholder_jpeg

        data = _placeholder_jpeg(640, 70, "Waiting for camera...")
        decoded = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)

        assert decoded is not None
        assert decoded.shape[1] == 640


class TestControlPlaneAuditRegressions:
    """Control-plane defects found by the audit, pinned so they stay fixed."""

    async def test_the_sentinel_rule_cannot_be_edited_or_deleted(self, client):
        assert (await client.put("/api/rules/0", json={"enabled": True})).status_code == 403
        assert (await client.delete("/api/rules/0")).status_code == 403

    async def test_deleting_one_rule_keeps_its_alert_history(self, client):
        db = client.app.state.db
        rule_id = (await client.post("/api/rules", json=SAMPLE_RULE)).json()["id"]
        await db.create_alert(
            rule_id=rule_id,
            camera_id="default",
            timestamp=1.0,
            verdict=True,
            raw_response="Yes",
            inference_ms=1.0,
            bbox_json={},
        )

        assert (await client.delete(f"/api/rules/{rule_id}")).status_code == 204

        alerts = (await client.get("/api/alerts")).json()["alerts"]
        assert len(alerts) == 1
        assert alerts[0]["rule_id"] == 0

    @pytest.mark.parametrize(
        "patch",
        [
            {"confidence": 1.5},
            {"cooldown_s": -1.0},
            {"roi_json": [[0.1, 0.1], [0.9, 0.9]]},
            {"roi_json": [[0.1, 0.1], [1.9, 0.1], [0.5, 0.9]]},
            {"roi_json": [[0.1, 0.1, 0.2], [0.9, 0.1], [0.5, 0.9]]},
        ],
    )
    async def test_invalid_rules_are_rejected_not_misapplied(self, client, patch):
        created = await client.post("/api/rules", json={**SAMPLE_RULE, **patch})
        assert created.status_code == 422

        rule_id = (await client.post("/api/rules", json=SAMPLE_RULE)).json()["id"]
        updated = await client.put(f"/api/rules/{rule_id}", json=patch)
        assert updated.status_code == 422

    async def test_an_explicit_null_is_a_422_not_a_500(self, client):
        rule_id = (await client.post("/api/rules", json=SAMPLE_RULE)).json()["id"]

        response = await client.put(f"/api/rules/{rule_id}", json={"name": None})

        assert response.status_code == 422

    async def test_an_unknown_label_is_warned_about(self, client):
        response = await client.post(
            "/api/rules", json={**SAMPLE_RULE, "target_labels": ["persons"]}
        )

        assert response.status_code == 201
        assert "'persons'" in response.json()["warning"]

    async def test_alert_history_is_ordered_by_insertion_and_named(self, client):
        db = client.app.state.db
        rule_id = (await client.post("/api/rules", json=SAMPLE_RULE)).json()["id"]
        # Same created_at second for all of them, which is what made the old
        # created_at ordering undefined across pages.
        for i in range(5):
            await db.create_alert(
                rule_id=rule_id,
                camera_id="default",
                timestamp=float(i),
                verdict=False,
                raw_response=str(i),
                inference_ms=1.0,
                bbox_json={},
            )

        first = (await client.get("/api/alerts?limit=2&offset=0")).json()["alerts"]
        second = (await client.get("/api/alerts?limit=2&offset=2")).json()["alerts"]

        assert [a["raw_response"] for a in first + second] == ["4", "3", "2", "1"]
        assert first[0]["rule_name"] == SAMPLE_RULE["name"]

    async def test_dashboard_assets_are_uncached_and_the_api_is_untouched(self, client):
        dashboard = await client.get("/")
        assert dashboard.headers["cache-control"] == "no-store, no-cache, must-revalidate"

        api = await client.get("/api/rules")
        assert "no-store" not in api.headers.get("cache-control", "")

    async def test_status_reports_webhook_backlog(self, client):
        status = (await client.get("/api/status")).json()
        assert status["webhooks"] == {"in_flight": 0, "dropped": 0}

    async def test_viewers_share_one_render(self, client):
        """Two streams, one encoder: rendering must not scale with viewers."""
        from app.api import server
        from app.api.server import _mjpeg_generator

        app = client.app
        buffer = MagicMock()
        buffer.latest_timestamp.return_value = 1.0
        buffer.read_latest.return_value = (np.zeros((120, 160, 3), dtype=np.uint8), 1.0, 0)
        app.state.buffer = buffer
        app.state.watchdog = None

        renders = []
        real_render = server._render_preview_frame

        def counting_render(*args):
            renders.append(1)
            return real_render(*args)

        server._render_preview_frame = counting_render
        streams = [_mjpeg_generator(app), _mjpeg_generator(app)]
        try:
            parts = await asyncio.wait_for(
                asyncio.gather(*(s.__anext__() for s in streams)), timeout=5.0
            )
            await asyncio.sleep(0.5)  # several stream_fps ticks, same frame
        finally:
            server._render_preview_frame = real_render
            for stream in streams:
                await stream.aclose()

        assert parts[0] == parts[1]
        assert len(renders) == 1, f"rendered {len(renders)} times for one unchanged frame"


class TestStartupFailure:
    async def test_a_failed_startup_tears_down_what_started(self, tmp_path, monkeypatch):
        from app.api import server

        monkeypatch.setenv("EYEQ_DATABASE_PATH", str(tmp_path / "boot.db"))
        monkeypatch.setenv("EYEQ_ENABLE_PIPELINE", "true")
        get_settings.cache_clear()

        capture = MagicMock()

        async def half_started(app, settings, db):
            app.state.captures = [capture]
            raise RuntimeError("VLM failed to load")

        monkeypatch.setattr(server, "_start_pipeline", half_started)
        app = create_app()
        try:
            with pytest.raises(RuntimeError, match="VLM failed to load"):
                async with lifespan(app):
                    pass
        finally:
            get_settings.cache_clear()

        capture.stop.assert_called_once()
        with pytest.raises(ValueError):
            await app.state.db.list_rules()  # connection closed


class TestBroadcast:
    async def test_one_stalled_client_does_not_hold_up_the_rest(self, monkeypatch):
        from app.api import routes

        monkeypatch.setattr(routes, "_BROADCAST_SEND_TIMEOUT_S", 0.2)
        manager = routes.ConnectionManager()

        class Client:
            def __init__(self, stall: bool) -> None:
                self.stall = stall
                self.received = []

            async def send_json(self, message) -> None:
                if self.stall:
                    await asyncio.sleep(60)
                self.received.append(message)

        stalled, healthy = Client(stall=True), Client(stall=False)
        manager._connections = [stalled, healthy]

        started = asyncio.get_running_loop().time()
        await manager.broadcast({"type": "alert"})
        elapsed = asyncio.get_running_loop().time() - started

        assert healthy.received == [{"type": "alert"}]
        assert elapsed < 1.0
        assert manager.active_count == 1, "the stalled client was not dropped"
