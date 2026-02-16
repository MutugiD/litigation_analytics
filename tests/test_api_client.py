"""Tests for Tausi API client."""

import pytest

from src.data.api_client import CircuitBreakerOpen, TausiClient


class TestTausiClient:
    """Tests for TausiClient without hitting the real API."""

    def test_init_defaults(self):
        """Client initializes with settings from config."""
        client = TausiClient()
        assert client.base_url.startswith("http")
        assert client.token  # Should have a token from .env

    def test_circuit_breaker_threshold(self):
        """Circuit breaker raises after consecutive failures."""
        client = TausiClient()
        client._consecutive_failures = client._cb_threshold
        with pytest.raises(CircuitBreakerOpen):
            client._check_circuit_breaker()

    def test_adaptive_rate_on_success(self):
        """Rate limit increases on success."""
        client = TausiClient()
        initial_rate = client._rate_limit
        client._on_success()
        assert client._rate_limit > initial_rate
        assert client._consecutive_failures == 0

    def test_adaptive_rate_on_failure(self):
        """Rate limit decreases on failure."""
        client = TausiClient()
        client._rate_limit = 2.0
        client._on_failure()
        assert client._rate_limit < 2.0
        assert client._consecutive_failures == 1

    def test_rate_limit_bounds(self):
        """Rate limit stays within bounds."""
        client = TausiClient()
        # Max bound
        for _ in range(100):
            client._on_success()
        assert client._rate_limit <= client._max_rate
        # Min bound
        for _ in range(100):
            client._on_failure()
        assert client._rate_limit >= 0.5
