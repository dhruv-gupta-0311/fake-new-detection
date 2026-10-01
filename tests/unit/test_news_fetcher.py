import json

import pytest

from src import article_fetcher
from src.article_fetcher import NewsFetcher


class DummyResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_fetch_by_query_uses_cache_and_parses_articles(monkeypatch, tmp_path):
    cache_dir = tmp_path / "newsapi_cache"
    monkeypatch.setattr(article_fetcher.config, "news_cache_dir", str(cache_dir))
    monkeypatch.setattr(article_fetcher.config, "newsapi_daily_limit", 900)

    call_count = {"count": 0}

    def fake_get(*args, **kwargs):
        call_count["count"] += 1
        return DummyResponse(
            {
                "articles": [
                    {
                        "title": "Federal Reserve keeps interest rates steady",
                        "content": "The Federal Reserve announced it will keep interest rates unchanged.",
                        "source": {"name": "Reuters"},
                        "url": "https://example.com/fed-rates",
                        "publishedAt": "2026-07-01T10:00:00Z",
                    }
                ]
            }
        )

    monkeypatch.setattr(article_fetcher.requests, "get", fake_get)
    monkeypatch.setattr(article_fetcher.RateLimiter, "can_call", lambda self: True)
    monkeypatch.setattr(article_fetcher.RateLimiter, "record_call", lambda self: None)

    fetcher = NewsFetcher()
    articles = fetcher.fetch_by_query("Federal Reserve interest rates", max_articles=1)

    assert call_count["count"] == 1
    assert len(articles) == 1
    assert articles[0]["source"] == "Reuters"
    assert articles[0]["url"] == "https://example.com/fed-rates"

    # second call should use cache and not call requests.get again
    articles_cached = fetcher.fetch_by_query("Federal Reserve interest rates", max_articles=1)
    assert call_count["count"] == 1
    assert articles_cached == articles
