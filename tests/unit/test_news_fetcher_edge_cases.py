"""Edge case tests for NewsFetcher."""
import json
import pytest
import requests

from src import article_fetcher
from src.article_fetcher import NewsFetcher
from src.rate_limiter import RateLimiter


class DummyResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(
                f"{self.status_code}", response=self
            )

    def json(self):
        return self._payload


def test_rate_limit_exceeded_returns_empty(monkeypatch, tmp_path):
    """When daily limit is reached, fetch_by_query returns [] without calling the API."""
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(article_fetcher.config, "news_cache_dir", str(cache_dir))
    monkeypatch.setattr(article_fetcher.config, "newsapi_daily_limit", 900)

    api_called = {"count": 0}

    def fake_get(*args, **kwargs):
        api_called["count"] += 1
        return DummyResponse({"articles": []})

    monkeypatch.setattr(article_fetcher.requests, "get", fake_get)
    # Simulate limit exhausted
    monkeypatch.setattr(RateLimiter, "can_call", lambda self: False)

    fetcher = NewsFetcher()
    result = fetcher.fetch_by_query("some query")

    assert result == []
    assert api_called["count"] == 0  # API must NOT be called


def test_request_timeout_returns_empty(monkeypatch, tmp_path):
    """A timeout during requests.get is caught and returns []."""
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(article_fetcher.config, "news_cache_dir", str(cache_dir))
    monkeypatch.setattr(article_fetcher.config, "newsapi_daily_limit", 900)
    monkeypatch.setattr(RateLimiter, "can_call", lambda self: True)
    monkeypatch.setattr(RateLimiter, "record_call", lambda self: None)

    def fake_get(*args, **kwargs):
        raise requests.exceptions.Timeout("timed out")

    monkeypatch.setattr(article_fetcher.requests, "get", fake_get)

    fetcher = NewsFetcher()
    result = fetcher.fetch_by_query("timeout query")

    assert result == []


def test_http_error_returns_empty(monkeypatch, tmp_path):
    """An HTTP 4xx/5xx error is caught and returns []."""
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(article_fetcher.config, "news_cache_dir", str(cache_dir))
    monkeypatch.setattr(article_fetcher.config, "newsapi_daily_limit", 900)
    monkeypatch.setattr(RateLimiter, "can_call", lambda self: True)
    monkeypatch.setattr(RateLimiter, "record_call", lambda self: None)

    def fake_get(*args, **kwargs):
        return DummyResponse({"message": "apiKeyInvalid"}, status_code=401)

    monkeypatch.setattr(article_fetcher.requests, "get", fake_get)

    fetcher = NewsFetcher()
    result = fetcher.fetch_by_query("error query")

    assert result == []


def test_empty_api_response_returns_empty(monkeypatch, tmp_path):
    """API returns 200 but articles list is empty — result is []."""
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(article_fetcher.config, "news_cache_dir", str(cache_dir))
    monkeypatch.setattr(article_fetcher.config, "newsapi_daily_limit", 900)
    monkeypatch.setattr(RateLimiter, "can_call", lambda self: True)
    monkeypatch.setattr(RateLimiter, "record_call", lambda self: None)

    def fake_get(*args, **kwargs):
        return DummyResponse({"status": "ok", "articles": []})

    monkeypatch.setattr(article_fetcher.requests, "get", fake_get)

    fetcher = NewsFetcher()
    result = fetcher.fetch_by_query("no results query")

    assert result == []


def test_articles_without_content_filtered_out(monkeypatch, tmp_path):
    """Articles missing both content and url are excluded from results."""
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(article_fetcher.config, "news_cache_dir", str(cache_dir))
    monkeypatch.setattr(article_fetcher.config, "newsapi_daily_limit", 900)
    monkeypatch.setattr(RateLimiter, "can_call", lambda self: True)
    monkeypatch.setattr(RateLimiter, "record_call", lambda self: None)

    def fake_get(*args, **kwargs):
        return DummyResponse({
            "articles": [
                {   # valid article
                    "title": "Good article",
                    "content": "Has content",
                    "source": {"name": "Reuters"},
                    "url": "https://example.com/good",
                    "publishedAt": "2026-07-01T10:00:00Z",
                },
                {   # missing content and url — should be filtered
                    "title": "Bad article",
                    "content": None,
                    "source": {"name": "Unknown"},
                    "url": None,
                    "publishedAt": "2026-07-01T10:00:00Z",
                },
            ]
        })

    monkeypatch.setattr(article_fetcher.requests, "get", fake_get)

    fetcher = NewsFetcher()
    result = fetcher.fetch_by_query("filter test")

    assert len(result) == 1
    assert result[0]["title"] == "Good article"


def test_bulk_fetch_stops_at_rate_limit(monkeypatch, tmp_path):
    """fetch_bulk stops processing topics once the rate limit is reached."""
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(article_fetcher.config, "news_cache_dir", str(cache_dir))
    monkeypatch.setattr(article_fetcher.config, "newsapi_daily_limit", 900)

    call_count = {"count": 0}

    # Allow only first call, block subsequent ones
    def can_call(self):
        return call_count["count"] == 0

    def fake_get(*args, **kwargs):
        call_count["count"] += 1
        return DummyResponse({
            "articles": [{
                "title": "Article",
                "content": "Content",
                "source": {"name": "Reuters"},
                "url": f"https://example.com/{call_count['count']}",
                "publishedAt": "2026-07-01T10:00:00Z",
            }]
        })

    monkeypatch.setattr(article_fetcher.requests, "get", fake_get)
    monkeypatch.setattr(RateLimiter, "can_call", can_call)
    monkeypatch.setattr(RateLimiter, "record_call", lambda self: None)

    fetcher = NewsFetcher()
    topics = ["topic1", "topic2", "topic3"]
    result = fetcher.fetch_bulk(topics, articles_per_topic=1)

    # Only first topic should be fetched; remaining stopped by rate limit
    assert call_count["count"] == 1
