import requests
import hashlib
import json
import os
from datetime import datetime, timedelta
from src.config import config
from src.logger import get_logger
from src.rate_limiter import RateLimiter

logger = get_logger(__name__)
class NewsFetcher:
    def __init__(self):
        self.api_key = config.newsapi_key
        self.base_url = "https://newsapi.org/v2"
        self.rate_limiter = RateLimiter(config.newsapi_daily_limit)
        os.makedirs(config.news_cache_dir, exist_ok=True)

        if not self.api_key:
            logger.warning("NEWSAPI_KEY not set in .env")
    def _cache_key(self, query: str) -> str:
        today = datetime.now().strftime('%Y-%m-%d')
        return hashlib.md5(f"{query}_{today}".encode()).hexdigest()
    def _get_cached(self, query: str):
        path = f"{config.news_cache_dir}/{self._cache_key(query)}.json"
        if os.path.exists(path):
            logger.info(f"Cache hit: '{query}'")
            with open(path) as f:
                return json.load(f)
        return None
    def _set_cached(self, query: str, articles: list):
        path = f"{config.news_cache_dir}/{self._cache_key(query)}.json"
        with open(path, 'w') as f:
            json.dump(articles, f, indent=2)
        logger.info(f"Cached {len(articles)} articles for: '{query}'")

    def fetch_by_query(self, query: str, days_back=7, max_articles=20) -> list:
        """
        Fetch articles for a query.
        Returns cached result if same query was made today.
        """
        cached = self._get_cached(query)
        if cached is not None:
            return cached
        if not self.rate_limiter.can_call():
            logger.error("Daily API limit reached. Returning empty.")
            return []

        try:
            from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
            response = requests.get(
                f"{self.base_url}/everything",
                params={
                    'q': query,
                    'from': from_date,
                    'sortBy': 'relevancy',
                    'language': 'en',
                    'pageSize': min(max_articles, 100),
                    'apiKey': self.api_key
                },
                timeout=10
            )
            response.raise_for_status()
            self.rate_limiter.record_call()

            raw = response.json().get('articles', [])
            articles = [
                self._parse(a) for a in raw
                if a.get('content') and a.get('url')
            ]

            logger.info(f"Fetched {len(articles)} articles for: '{query}'")
            self._set_cached(query, articles)
            return articles

        except requests.exceptions.Timeout:
            logger.warning(f"Timeout for query: '{query}'")
            return []
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return []
    def fetch_bulk(self, topics: list, articles_per_topic=50) -> list:
        """Fetch across multiple topics for training pipeline."""
        all_articles = []
        for topic in topics:
            if not self.rate_limiter.can_call():
                logger.warning("Rate limit reached. Stopping bulk fetch.")
                break
            articles = self.fetch_by_query(topic, max_articles=articles_per_topic)
            all_articles.extend(articles)
            logger.info(f"Bulk fetch total so far: {len(all_articles)}")
        return all_articles

    def _parse(self, raw: dict) -> dict:
        return {
            'title': raw.get('title', ''),
            'content': raw.get('content', '') or raw.get('description', ''),
            'source': raw.get('source', {}).get('name', 'Unknown'),
            'url': raw.get('url', ''),
            'published_at': raw.get('publishedAt', ''),
            'fetched_at': datetime.now().isoformat()
        }

    @property
    def calls_remaining(self) -> int:
        return self.rate_limiter.remaining
