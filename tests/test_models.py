"""Tests for RateLimiter and provider clients."""
import os
import time
from unittest.mock import MagicMock, patch, call

import pytest

from src.models.rate_limiter import RateLimiter
from src.models.groq_client import GroqClient
from src.models.nim_client import NIMClient
from src.models.openrouter_client import OpenRouterClient


# ---------------------------------------------------------------------------
# RateLimiter tests
# ---------------------------------------------------------------------------

def test_rate_limiter_no_sleep_on_first_call():
    """First acquire() should not sleep regardless of rpm."""
    limiter = RateLimiter(rpm=30)
    with patch("time.sleep") as mock_sleep:
        limiter.acquire()
    mock_sleep.assert_not_called()


def test_rate_limiter_sleeps_when_called_too_fast():
    """Second acquire() called immediately should sleep for ~interval seconds."""
    limiter = RateLimiter(rpm=60)  # interval = 1.0 s
    t0 = time.monotonic()

    # acquire() calls time.monotonic() twice: once for 'now', once to update _last_time.
    # Two acquire() calls = 4 total time.monotonic() calls.
    with patch("time.sleep") as mock_sleep, \
         patch("time.monotonic", side_effect=[t0, t0, t0 + 0.1, t0 + 1.0]):
        limiter.acquire()   # first call — _last_time set to t0, no sleep
        limiter.acquire()   # second call — 0.1 s elapsed, needs ~0.9 s sleep

    assert mock_sleep.called
    slept = mock_sleep.call_args[0][0]
    assert slept == pytest.approx(0.9, abs=0.05)


def test_rate_limiter_no_sleep_when_interval_elapsed():
    """acquire() should not sleep if enough time has already passed."""
    limiter = RateLimiter(rpm=60)  # interval = 1.0 s
    t0 = time.monotonic()

    with patch("time.sleep") as mock_sleep, \
         patch("time.monotonic", side_effect=[t0, t0, t0 + 1.5, t0 + 1.5]):
        limiter.acquire()  # sets _last_time = t0
        limiter.acquire()  # 1.5 s elapsed — no sleep needed

    mock_sleep.assert_not_called()


# ---------------------------------------------------------------------------
# Client configuration tests
# ---------------------------------------------------------------------------

def test_groq_client_model_and_base_url():
    """GroqClient must use llama-3.3-70b-versatile and Groq's OpenAI-compatible base URL."""
    with patch("src.models.groq_client.OpenAI"):
        client = GroqClient(api_key="test-key")
    assert client.model == "llama-3.3-70b-versatile"
    assert "groq.com" in client.base_url


def test_nim_client_model_and_base_url():
    """NIMClient must use llama-3.3-70b-instruct and NVIDIA NIM base URL."""
    with patch("src.models.nim_client.OpenAI"):
        client = NIMClient(api_key="test-key")
    assert client.model == "meta/llama-3.3-70b-instruct"
    assert "nvidia.com" in client.base_url


def test_openrouter_client_model_and_base_url():
    """OpenRouterClient must use OpenRouter OpenAI-compatible endpoint."""
    with patch("src.models.openrouter_client.OpenAI"):
        client = OpenRouterClient(api_key="test-key")
    assert client.model == "qwen/qwen3-32b"
    assert "openrouter.ai" in client.base_url
