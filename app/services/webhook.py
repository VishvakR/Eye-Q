"""Outbound webhook dispatch for confirmed violations.

Fire-and-forget by design: a webhook POST must never delay the inference
pipeline, and a dead endpoint must never build a retry backlog that outlives
the event it describes (PLAN.md §8). Failures are logged and dropped.

Example:
    >>> dispatcher = WebhookDispatcher("https://hooks.example.com/xyz")
    >>> await dispatcher.dispatch({"rule_name": "Fire exit", "verdict": True})
    >>> await dispatcher.close()
"""

from __future__ import annotations

import logging

import httpx

logger = logging.getLogger(__name__)


class WebhookDispatcher:
    """POSTs alert payloads to a user-configured HTTP endpoint.

    Args:
        url: Destination URL. If falsy, dispatch becomes a no-op.
        timeout_s: Per-request timeout.
    """

    def __init__(self, url: str | None, timeout_s: float = 5.0) -> None:
        self._url = url
        self._timeout_s = timeout_s
        self._client: httpx.AsyncClient | None = None

        if url:
            self._client = httpx.AsyncClient(timeout=timeout_s)
            logger.info(f"Webhook dispatcher enabled → {url}")

    @property
    def enabled(self) -> bool:
        """Whether a destination URL is configured."""
        return bool(self._url and self._client)

    async def dispatch(self, payload: dict) -> bool:
        """POST a payload to the configured webhook.

        Never raises — a broken webhook must not break alerting.

        Args:
            payload: JSON-serializable alert body.

        Returns:
            True if the endpoint accepted the request, False otherwise.
        """
        if not self.enabled:
            return False

        try:
            response = await self._client.post(self._url, json=payload)
            response.raise_for_status()
            logger.info(f"Webhook delivered ({response.status_code}).")
            return True
        except httpx.HTTPStatusError as exc:
            logger.warning(f"Webhook rejected: HTTP {exc.response.status_code}")
        except Exception as exc:
            logger.warning(f"Webhook delivery failed: {exc}")

        return False

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
