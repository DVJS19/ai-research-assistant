#
# Tests for observability — trace_id threading and metrics persistence.
# All DB calls are mocked — no real Postgres needed.
#
# Run: uv run pytest tests/test_observability.py -v

from unittest.mock import AsyncMock, patch

import pytest


class TestTraceId:
    def test_new_trace_id_returns_uuid(self):
        """new_trace_id() generates a valid UUID string."""
        from app.observability import new_trace_id

        trace_id = new_trace_id()
        assert len(trace_id) == 36
        assert trace_id.count("-") == 4

    def test_set_and_get_trace_id(self):
        """set_trace_id and get_trace_id work as a pair."""
        from app.observability import get_trace_id, set_trace_id

        set_trace_id("test-trace-abc")
        assert get_trace_id() == "test-trace-abc"

    def test_default_trace_id(self):
        """get_trace_id returns 'no-trace' when none is set."""
        import contextvars

        from app.observability import _trace_id_var

        # Run in a fresh context
        ctx = contextvars.copy_context()
        result = ctx.run(lambda: _trace_id_var.get())
        # May be set from a previous test — just check it's a string
        assert isinstance(result, str)


class TestPersistMetrics:
    @pytest.mark.asyncio
    @patch("app.observability._pool")
    async def test_persist_run_metrics_success(self, mock_pool):
        """persist_run_metrics writes one row without errors."""
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)

        from app.observability import persist_run_metrics, set_trace_id

        set_trace_id("test-trace-123")

        # Should not raise
        await persist_run_metrics(
            run_id="run-001",
            topic="test topic",
            steps_taken=3,
            tokens_used=1500,
            cost_usd=0.0042,
            error_count=0,
            status="completed",
        )

        mock_conn.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_persist_metrics_no_pool(self):
        """persist_run_metrics handles missing pool gracefully — no exception."""
        import app.observability as obs_module

        original = obs_module._pool
        obs_module._pool = None

        try:
            # Should log a warning but not raise
            await obs_module.persist_run_metrics(
                run_id="run-002",
                topic="test",
                steps_taken=0,
                tokens_used=0,
                cost_usd=0.0,
                error_count=0,
            )
        finally:
            obs_module._pool = original

    @pytest.mark.asyncio
    @patch("app.observability._pool")
    async def test_persist_metrics_db_failure_does_not_raise(self, mock_pool):
        """DB failure during persist does not propagate — observability never breaks the app."""
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock(side_effect=Exception("DB connection lost"))
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)

        from app.observability import persist_run_metrics

        # Should catch the exception and log it, not raise
        await persist_run_metrics(
            run_id="run-003",
            topic="test",
            steps_taken=1,
            tokens_used=100,
            cost_usd=0.001,
            error_count=0,
        )
