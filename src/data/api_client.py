"""Tausi API (Kenya Law) client with adaptive rate limiting and resilience.

Implements:
- Token-based authentication
- Adaptive rate limiting (starts conservative, increases if no errors)
- Exponential backoff on failures
- Circuit breaker (pauses after consecutive failures)
- Paginated result fetching
- Response caching to disk (resume-safe downloads)
"""

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from configs.settings import settings

logger = logging.getLogger(__name__)


class CircuitBreakerOpen(Exception):
    """Raised when the circuit breaker is open (too many consecutive failures)."""


class TausiClient:
    """Async HTTP client for the Tausi Decision Database API.

    Usage:
        async with TausiClient() as client:
            async for decision in client.iter_decisions(court="kehc", year=2020):
                process(decision)
    """

    def __init__(
        self,
        token: str | None = None,
        base_url: str | None = None,
        cache_dir: Path | None = None,
    ):
        cfg = settings.tausi
        self.token = token or cfg.token
        self.base_url = (base_url or cfg.base_url).rstrip("/")
        self.cache_dir = cache_dir or settings.data.raw_dir
        self.timeout = cfg.timeout_seconds

        # Adaptive rate limiting
        self._rate_limit = cfg.initial_rate_limit
        self._max_rate = cfg.max_rate_limit
        self._last_request_time = 0.0

        # Circuit breaker
        self._consecutive_failures = 0
        self._cb_threshold = cfg.circuit_breaker_threshold
        self._cb_pause = cfg.circuit_breaker_pause_seconds

        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self):
        self._client = httpx.AsyncClient(
            headers={
                "Authorization": f"Token {self.token}",
                "Accept": "application/json",
            },
            timeout=httpx.Timeout(self.timeout),
            follow_redirects=True,
        )
        return self

    async def __aexit__(self, *args):
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _throttle(self):
        """Enforce rate limiting between requests."""
        elapsed = time.monotonic() - self._last_request_time
        min_interval = 1.0 / self._rate_limit
        if elapsed < min_interval:
            await asyncio.sleep(min_interval - elapsed)
        self._last_request_time = time.monotonic()

    def _on_success(self):
        """Adjust rate limit upward on success."""
        self._consecutive_failures = 0
        # Gradually increase rate limit (10% per success, up to max)
        self._rate_limit = min(self._rate_limit * 1.1, self._max_rate)

    def _on_failure(self):
        """Track failures for circuit breaker."""
        self._consecutive_failures += 1
        # Reduce rate limit on failure
        self._rate_limit = max(self._rate_limit * 0.5, 0.5)

    def _check_circuit_breaker(self):
        """Raise if too many consecutive failures."""
        if self._consecutive_failures >= self._cb_threshold:
            raise CircuitBreakerOpen(
                f"{self._consecutive_failures} consecutive failures. Pausing for {self._cb_pause}s."
            )

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.TransportError)),
    )
    async def _request(self, method: str, url: str, **kwargs) -> httpx.Response:
        """Make a rate-limited, retried HTTP request."""
        self._check_circuit_breaker()
        await self._throttle()

        try:
            response = await self._client.request(method, url, **kwargs)
            response.raise_for_status()
            self._on_success()
            return response
        except (httpx.HTTPStatusError, httpx.TransportError) as e:
            self._on_failure()
            logger.warning(
                "Request failed: %s %s -> %s (failures: %d, rate: %.1f req/s)",
                method,
                url,
                e,
                self._consecutive_failures,
                self._rate_limit,
            )
            raise

    async def get_json(self, endpoint: str, params: dict | None = None) -> dict:
        """GET a JSON endpoint, return parsed response."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        response = await self._request("GET", url, params=params)
        return response.json()

    async def download_file(self, url: str, dest: Path) -> Path:
        """Download a file (PDF, etc.) to disk. Skips if already exists."""
        if dest.exists() and dest.stat().st_size > 0:
            logger.debug("Skipping download (cached): %s", dest.name)
            return dest

        dest.parent.mkdir(parents=True, exist_ok=True)
        response = await self._request("GET", url)

        dest.write_bytes(response.content)
        logger.debug("Downloaded: %s (%d bytes)", dest.name, len(response.content))
        return dest

    # --- High-level API methods ---

    async def get_courts(self) -> list[dict]:
        """Fetch all courts from /courts."""
        data = await self.get_json("courts/.json")
        return data.get("results", data) if isinstance(data, dict) else data

    async def get_judges(self) -> list[dict]:
        """Fetch all judges from /judges."""
        data = await self.get_json("judges/.json")
        return data.get("results", data) if isinstance(data, dict) else data

    async def get_decision(self, frbr_uri: str) -> dict:
        """Fetch a single decision by its FRBR URI.

        Args:
            frbr_uri: e.g., "/akn/ke/judgment/kehc/2020/123"
        """
        endpoint = f"decisions{frbr_uri}.json"
        return await self.get_json(endpoint)

    async def iter_decisions(
        self,
        court: str | None = None,
        year: int | None = None,
        **extra_params: Any,
    ) -> AsyncIterator[dict]:
        """Iterate through paginated decisions with optional filters.

        Args:
            court: Court code (e.g., "kehc")
            year: Filing year filter
            **extra_params: Additional query parameters

        Yields:
            Individual decision dicts from the API.
        """
        params: dict[str, Any] = {}
        if court:
            params["court"] = court
        if year:
            params["year"] = year
        params.update(extra_params)

        endpoint = "decisions/.json"
        url = f"{self.base_url}/{endpoint}"

        page = 0
        while url:
            page += 1
            try:
                response = await self._request("GET", url, params=params if page == 1 else None)
                data = response.json()
            except CircuitBreakerOpen:
                logger.error("Circuit breaker open. Pausing %ds...", self._cb_pause)
                await asyncio.sleep(self._cb_pause)
                self._consecutive_failures = 0
                continue
            except Exception:
                logger.exception("Failed to fetch page %d", page)
                break

            results = data.get("results", [])
            for item in results:
                yield item

            url = data.get("next")
            if url:
                logger.info(
                    "Page %d: fetched %d decisions (total: %s)",
                    page,
                    len(results),
                    data.get("count", "?"),
                )

    async def count_decisions(self, court: str | None = None, year: int | None = None) -> int:
        """Get the total count of decisions matching filters."""
        params: dict[str, Any] = {}
        if court:
            params["court"] = court
        if year:
            params["year"] = year

        data = await self.get_json("decisions/.json", params=params)
        return data.get("count", 0)

    async def search_decisions(self, query: str, **filters) -> AsyncIterator[dict]:
        """Full-text search via the Tausi ElasticSearch endpoint.

        Args:
            query: Search string (e.g., "Data Protection Act")
            **filters: Additional filters (court, year, etc.)
        """
        params = {"q": query, **filters}
        url = f"{self.base_url}/search/.json"

        while url:
            try:
                response = await self._request("GET", url, params=params)
                data = response.json()
            except Exception:
                logger.exception("Search failed for query: %s", query)
                break

            for item in data.get("results", []):
                yield item

            url = data.get("next")
            params = None  # next URL includes params
