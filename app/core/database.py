"""Async SQLite database manager using aiosqlite.

Provides CRUD operations for rules, alerts, and cameras using
WAL mode for concurrent reads. Tables are auto-created on startup.

Example:
    >>> db = await Database.create("data/eyeq.db")
    >>> await db.initialize_tables()
    >>> rule_id = await db.create_rule(
    ...     camera_id="cam_1",
    ...     name="Pallet check",
    ...     prompt="Is a pallet blocking the fire exit?",
    ...     target_labels=["person", "forklift"],
    ... )
    >>> rules = await db.list_rules()
    >>> await db.close()
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone

import aiosqlite

logger = logging.getLogger(__name__)


SCHEMA_SQL = """
-- Rules: User-defined alert conditions
CREATE TABLE IF NOT EXISTS rules (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id   TEXT    NOT NULL,
    name        TEXT    NOT NULL,
    prompt      TEXT    NOT NULL,
    target_labels TEXT  NOT NULL DEFAULT '[]',
    roi_json    TEXT    NOT NULL DEFAULT '[]',
    confidence  REAL    NOT NULL DEFAULT 0.25,
    cooldown_s  REAL    NOT NULL DEFAULT 5.0,
    enabled     INTEGER NOT NULL DEFAULT 1,
    created_at  TEXT    NOT NULL DEFAULT (datetime('now')),
    updated_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- Alerts: Historical log of VLM verdicts
CREATE TABLE IF NOT EXISTS alerts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    rule_id         INTEGER NOT NULL REFERENCES rules(id) ON DELETE CASCADE,
    camera_id       TEXT    NOT NULL,
    timestamp       REAL    NOT NULL,
    verdict         INTEGER NOT NULL,
    raw_response    TEXT    NOT NULL,
    inference_ms    REAL    NOT NULL,
    bbox_json       TEXT    NOT NULL,
    frame_snapshot  TEXT,
    created_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- Cameras: Registered video sources
CREATE TABLE IF NOT EXISTS cameras (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    source      TEXT NOT NULL,
    resolution  TEXT NOT NULL DEFAULT '1920x1080',
    fps         INTEGER NOT NULL DEFAULT 30,
    enabled     INTEGER NOT NULL DEFAULT 1,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_alerts_rule_id ON alerts(rule_id);
CREATE INDEX IF NOT EXISTS idx_alerts_created_at ON alerts(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_rules_camera_id ON rules(camera_id);
"""


class Database:
    """Async SQLite database manager for rules, alerts, and cameras.

    Uses WAL mode for concurrent reads from multiple async tasks
    and aiosqlite for non-blocking I/O.

    Args:
        db_path: Path to the SQLite database file.
    """

    def __init__(self, db: aiosqlite.Connection, db_path: str) -> None:
        self._db = db
        self._db_path = db_path

    @classmethod
    async def create(cls, db_path: str) -> Database:
        """Create and connect to the database.

        Args:
            db_path: Path to the SQLite database file. Parent directories
                     will be created if they don't exist.

        Returns:
            Connected Database instance.
        """
        # Ensure data directory exists
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)

        db = await aiosqlite.connect(db_path)
        db.row_factory = aiosqlite.Row

        # Enable WAL mode for concurrent reads
        await db.execute("PRAGMA journal_mode=WAL")
        await db.execute("PRAGMA foreign_keys=ON")

        logger.info(f"Database connected: {db_path}")
        return cls(db, db_path)

    async def initialize_tables(self) -> None:
        """Create tables and indexes if they don't exist."""
        await self._db.executescript(SCHEMA_SQL)
        await self._db.commit()
        logger.info("Database tables initialized.")

    async def close(self) -> None:
        """Close the database connection."""
        await self._db.close()
        logger.info("Database connection closed.")

    # ── Rules CRUD ───────────────────────────────────────────────

    async def create_rule(
        self,
        camera_id: str,
        name: str,
        prompt: str,
        target_labels: list[str] | None = None,
        roi_json: list[list[float]] | None = None,
        confidence: float = 0.25,
        cooldown_s: float = 5.0,
        enabled: bool = True,
    ) -> int:
        """Create a new alert rule.

        Returns:
            The auto-generated rule ID.
        """
        cursor = await self._db.execute(
            """INSERT INTO rules (camera_id, name, prompt, target_labels, roi_json,
                                  confidence, cooldown_s, enabled)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                camera_id,
                name,
                prompt,
                json.dumps(target_labels or []),
                json.dumps(roi_json or []),
                confidence,
                cooldown_s,
                1 if enabled else 0,
            ),
        )
        await self._db.commit()
        rule_id = cursor.lastrowid
        logger.info(f"Rule created: id={rule_id}, name='{name}'")
        return rule_id

    async def get_rule(self, rule_id: int) -> dict | None:
        """Get a single rule by ID."""
        cursor = await self._db.execute("SELECT * FROM rules WHERE id = ?", (rule_id,))
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def list_rules(self, camera_id: str | None = None) -> list[dict]:
        """List all rules, optionally filtered by camera."""
        if camera_id:
            cursor = await self._db.execute(
                "SELECT * FROM rules WHERE camera_id = ? ORDER BY created_at DESC",
                (camera_id,),
            )
        else:
            cursor = await self._db.execute(
                "SELECT * FROM rules ORDER BY created_at DESC"
            )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def update_rule(self, rule_id: int, **fields) -> dict | None:
        """Update a rule's fields. Only provided fields are updated."""
        if not fields:
            return await self.get_rule(rule_id)

        # Handle special JSON fields
        if "target_labels" in fields and isinstance(fields["target_labels"], list):
            fields["target_labels"] = json.dumps(fields["target_labels"])
        if "roi_json" in fields and isinstance(fields["roi_json"], list):
            fields["roi_json"] = json.dumps(fields["roi_json"])
        if "enabled" in fields and isinstance(fields["enabled"], bool):
            fields["enabled"] = 1 if fields["enabled"] else 0

        fields["updated_at"] = datetime.now(timezone.utc).isoformat()

        set_clause = ", ".join(f"{k} = ?" for k in fields)
        values = list(fields.values()) + [rule_id]

        await self._db.execute(
            f"UPDATE rules SET {set_clause} WHERE id = ?", values
        )
        await self._db.commit()

        return await self.get_rule(rule_id)

    async def delete_rule(self, rule_id: int) -> bool:
        """Delete a rule by ID. Returns True if deleted."""
        cursor = await self._db.execute("DELETE FROM rules WHERE id = ?", (rule_id,))
        await self._db.commit()
        deleted = cursor.rowcount > 0
        if deleted:
            logger.info(f"Rule deleted: id={rule_id}")
        return deleted

    # ── Alerts CRUD ──────────────────────────────────────────────

    async def create_alert(
        self,
        rule_id: int,
        camera_id: str,
        timestamp: float,
        verdict: bool,
        raw_response: str,
        inference_ms: float,
        bbox_json: dict,
        frame_snapshot: str | None = None,
    ) -> int:
        """Record a VLM verdict as an alert."""
        cursor = await self._db.execute(
            """INSERT INTO alerts (rule_id, camera_id, timestamp, verdict,
                                   raw_response, inference_ms, bbox_json, frame_snapshot)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                rule_id,
                camera_id,
                timestamp,
                1 if verdict else 0,
                raw_response,
                inference_ms,
                json.dumps(bbox_json),
                frame_snapshot,
            ),
        )
        await self._db.commit()
        return cursor.lastrowid

    async def get_alert(self, alert_id: int) -> dict | None:
        """Get a single alert by ID."""
        cursor = await self._db.execute("SELECT * FROM alerts WHERE id = ?", (alert_id,))
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def list_alerts(
        self,
        limit: int = 50,
        offset: int = 0,
        rule_id: int | None = None,
        verdict: bool | None = None,
    ) -> list[dict]:
        """List alerts with pagination and optional filters."""
        conditions = []
        params: list = []

        if rule_id is not None:
            conditions.append("rule_id = ?")
            params.append(rule_id)
        if verdict is not None:
            conditions.append("verdict = ?")
            params.append(1 if verdict else 0)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.extend([limit, offset])

        cursor = await self._db.execute(
            f"SELECT * FROM alerts {where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
            params,
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def count_alerts(self, rule_id: int | None = None) -> int:
        """Count total alerts, optionally filtered by rule."""
        if rule_id is not None:
            cursor = await self._db.execute(
                "SELECT COUNT(*) FROM alerts WHERE rule_id = ?", (rule_id,)
            )
        else:
            cursor = await self._db.execute("SELECT COUNT(*) FROM alerts")
        row = await cursor.fetchone()
        return row[0] if row else 0

    # ── Cameras CRUD ─────────────────────────────────────────────

    async def create_camera(
        self,
        camera_id: str,
        name: str,
        source: str,
        resolution: str = "1920x1080",
        fps: int = 30,
        enabled: bool = True,
    ) -> str:
        """Register a camera. Returns the camera ID."""
        await self._db.execute(
            """INSERT OR REPLACE INTO cameras (id, name, source, resolution, fps, enabled)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (camera_id, name, source, resolution, fps, 1 if enabled else 0),
        )
        await self._db.commit()
        logger.info(f"Camera registered: id={camera_id}, name='{name}'")
        return camera_id

    async def get_camera(self, camera_id: str) -> dict | None:
        """Get a camera by ID."""
        cursor = await self._db.execute("SELECT * FROM cameras WHERE id = ?", (camera_id,))
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def list_cameras(self) -> list[dict]:
        """List all registered cameras."""
        cursor = await self._db.execute("SELECT * FROM cameras ORDER BY created_at DESC")
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def delete_camera(self, camera_id: str) -> bool:
        """Delete a camera by ID."""
        cursor = await self._db.execute("DELETE FROM cameras WHERE id = ?", (camera_id,))
        await self._db.commit()
        return cursor.rowcount > 0

    # ── Stats ────────────────────────────────────────────────────

    async def get_stats(self) -> dict:
        """Get aggregate statistics for the dashboard."""
        rules_count = (await (await self._db.execute("SELECT COUNT(*) FROM rules WHERE enabled = 1")).fetchone())[0]
        alerts_count = (await (await self._db.execute("SELECT COUNT(*) FROM alerts")).fetchone())[0]
        violations_count = (await (await self._db.execute("SELECT COUNT(*) FROM alerts WHERE verdict = 1")).fetchone())[0]
        cameras_count = (await (await self._db.execute("SELECT COUNT(*) FROM cameras WHERE enabled = 1")).fetchone())[0]

        return {
            "active_rules": rules_count,
            "total_alerts": alerts_count,
            "total_violations": violations_count,
            "active_cameras": cameras_count,
        }
